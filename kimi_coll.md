# NCCL Collective 任务处理机制深度分析

本文档详细分析NCCL（NVIDIA Collective Communications Library）中collective任务的拆分、调度、入队以及launch的完整流程，同时涵盖通信拓扑创建、tuning机制以及kernel线程配置等核心机制。

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [Collective 任务入队流程](#2-collective-任务入队流程)
3. [任务调度与拆分机制](#3-任务调度与拆分机制)
4. [Kernel Launch 流程](#4-kernel-launch-流程)
5. [通信拓扑创建与Tuning](#5-通信拓扑创建与tuning)
6. [Kernel 线程与Warp配置](#6-kernel-线程与warp配置)
7. [关键数据结构](#7-关键数据结构)
8. [总结与流程图](#8-总结与流程图)

---

## 1. 整体架构概览

NCCL的collective操作处理流程分为以下几个主要阶段：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User API Layer                                │
│  ncclAllReduce / ncclAllGather / ncclBroadcast / ...                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Task Enqueue (ncclEnqueueCheck)                  │
│  - 参数检查、任务创建、入队到planner                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Group End Processing                              │
│  - ncclPrepareTasks: 算法选择、协议选择、线程数确定                    │
│  - 任务排序、聚合、拆分到channels                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Kernel Plan Generation                             │
│  - scheduleCollTasksToPlan: 创建work batches                          │
│  - finishPlan: 构建kernel参数                                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Kernel Launch                                     │
│  - ncclLaunchKernel: CUDA kernel启动                                  │
│  - Proxy op处理（网络通信）                                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Collective 任务入队流程

### 2.1 API入口

所有collective操作都遵循相同的入口模式（以`ncclAllReduce`为例）：

```cpp
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream,
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
```

### 2.2 任务入队核心流程 (ncclEnqueueCheck)

```cpp
// 简化流程
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  // 1. 参数检查
  NCCLCHECK(ArgsCheck(info));
  
  // 2. 创建ncclTaskColl结构
  struct ncclTaskColl* task = ncclMemoryPoolAlloc<struct ncclTaskColl>(...);
  task->func = info->coll;
  task->sendbuff = info->sendbuff;
  task->recvbuff = info->recvbuff;
  task->count = info->count;
  task->datatype = info->datatype;
  task->opHost = info->op;
  task->chunkSteps = info->chunkSteps;
  task->sliceSteps = info->sliceSteps;
  // ... 初始化其他字段
  
  // 3. 计算trafficBytes用于排序
  task->trafficBytes = task->count * ncclFuncTrafficPerByte(task->func, comm->nRanks);
  
  // 4. 插入到collSorter（按大小降序排列）
  ncclTaskCollSorterInsert(&planner->collSorter, task, task->trafficBytes);
  
  return ncclSuccess;
}
```

### 2.3 任务排序机制

NCCL使用**ncclTaskCollSorter**对collective任务进行排序：

```cpp
struct ncclTaskCollSorter {
  static constexpr int UnitLog2 = 10;      // 1KB
  static constexpr int MaxLog2 = 30;       // 1GB
  static constexpr int BitsPerPow2 = 2;    // 每幂次4个bin
  static constexpr int BinCount = 1 + (MaxLog2-UnitLog2)*BinsPerPow2;
  
  struct ncclTaskColl* head;
  struct ncclTaskColl* tail;
  int binEdge;
  struct ncclTaskColl** bins[BinCount];  // 分桶指针数组
};
```

**排序原理**：
- 任务按`trafficBytes`降序排列
- 使用桶排序，将任务分到不同大小区间
- 目的是让大任务优先获得channel资源，提高整体吞吐量

---

## 3. 任务调度与拆分机制

### 3.1 Group End 处理流程

当用户调用`ncclGroupEnd()`时，触发完整的调度流程：

```cpp
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t* simInfo) {
  // 1. 对每个comm执行PrepareTasks（确定算法、协议、线程数）
  NCCLCHECK(ncclPrepareTasks(comm, algoNeedConnect, &needConnect, simInfo));
  
  // 2. 执行连接（如需要）
  NCCLCHECK(ncclCollPreconnect(comm, algoNeedConnect));
  
  // 3. 注册buffer
  NCCLCHECK(ncclTasksRegAndEnqueue(comm));
  
  // 4. 创建kernel plan并launch
  NCCLCHECK(doLaunches(commHead));
}
```

### 3.2 算法与协议选择 (ncclPrepareTasks)

```cpp
ncclResult_t ncclPrepareTasks(struct ncclComm* comm, ...) {
  // 1. 从sorter获取大小降序排列的任务
  struct ncclTaskColl* task = ncclTaskCollSorterDequeueAll(&planner->collSorter);
  
  // 2. 对称任务处理（Symmetric Collective）
  if (comm->symmetricSupport) {
    NCCLCHECK(ncclMakeSymmetricTaskList(comm, task, &planner->collSymTaskQueue, &task));
  }
  
  // 3. 按(func, op, type)分bin聚合
  // 将相同类型的任务聚合，计算总大小
  
  // 4. 调用ncclGetAlgoInfo选择算法和协议
  NCCLCHECK(ncclGetAlgoInfo(comm, &agg, collNetSupport, nvlsSupport, nTasksPerChannel, simInfo));
  
  // 5. 根据算法分类到不同bin: [isCollnet][isNvls]
  // - Collnet任务：使用CollNet网络
  // - NVLS任务：使用NVLink SHARP
  // - 标准任务：使用Ring/Tree算法
}
```

### 3.3 算法选择核心逻辑 (ncclGetAlgoInfo)

```cpp
ncclResult_t ncclGetAlgoInfo(struct ncclComm* comm, struct ncclTaskColl* info, ...) {
  // 1. 计算数据大小
  size_t nBytes = elementSize * ncclFuncMaxSendRecvCount(info->func, comm->nRanks, info->count);
  
  // 2. 初始化cost表
  initCollCostTable((float **)collCostTable);
  updateCollCostTable(comm, info, nBytes, collNetSupport, nvlsSupport, numPipeOps, (float **)collCostTable);
  
  // 3. 如果有tuner插件，调用插件
  if (comm->tuner != NULL) {
    NCCLCHECK(comm->tuner->getCollInfo(...));
  }
  
  // 4. 默认使用topology-based tuner
  NCCLCHECK(topoGetAlgoInfo(comm, info, nBytes, (float **)collCostTable, simInfo));
  
  // 5. 根据CTA Policy调整（如NCCL_CTA_POLICY_EFFICIENCY）
  // 优先选择NVLS算法当buffer已注册时
}
```

### 3.4 线程数和Channel数计算 (topoGetAlgoInfo)

```cpp
static ncclResult_t topoGetAlgoInfo(struct ncclComm* comm, struct ncclTaskColl* info, 
                                     size_t nBytes, float** collCostTable, ncclSimInfo_t* simInfo) {
  // 1. 根据cost表选择最优的(algo, proto)组合
  // cost = latency * latCount + nBytes / (1000 * bw)
  
  // 2. 确定channel数
  int nc = info->nMaxChannels;  // 默认使用最大channel数
  int nt = comm->maxThreads[algo][proto] / WARP_SIZE;  // warp数
  
  // 3. 根据数据量调整channel数和线程数
  // 小数据量时减少channel数和线程数
  ssize_t threadThreshold = comm->threadThresholds[algo][proto];
  while (nBytes < nc * nt * threadThreshold) {
    if (nc >= 2) nc--;
    else if (nt % 128 == 0) nt /= 2;
    else break;
  }
  
  // 4. 特殊算法调整
  if (algo == NCCL_ALGO_RING && proto == NCCL_PROTO_SIMPLE) nt += 1; // 额外warp用于同步
  if (algo == NCCL_ALGO_TREE) nt = NCCL_MAX_NTHREADS / WARP_SIZE;    // Tree使用全部线程
  if (algo == NCCL_ALGO_PAT) nt = NCCL_MAX_NTHREADS / WARP_SIZE;
  
  info->nMaxChannels = nc;
  info->nWarps = nt;
}
```

### 3.5 任务拆分到Channels (scheduleCollTasksToPlan)

```cpp
static ncclResult_t scheduleCollTasksToPlan(struct ncclComm* comm, 
    struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget) {
  
  while (!ncclIntruQueueEmpty(&planner->collTaskQueue)) {
    struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collTaskQueue);
    
    if (task->isCollnet) {
      // CollNet调度：所有channel均匀分配
      devWork->channelLo = 0;
      devWork->channelHi = nChannels-1;
      devWork->collnet.count = task->count;
      devWork->collnet.chunkCount = chunkSize/elemSize;
    } else {
      // 标准调度：基于数据量的continuous-byte-distribution (CBD)
      // 计算每个channel处理的数据量
      size_t cellSize = ...;
      size_t cells = divUp(task->count*elementSize, cellSize);
      
      // 分配cells到channels
      size_t cellsLo = ...;  // 第一个channel
      size_t cellsMid = ...; // 中间channels
      size_t cellsHi = ...;  // 最后一个channel
      
      devWork->cbd.countLo = cellsLo * elementsPerCell;
      devWork->cbd.countMid = cellsMid * elementsPerCell;
      devWork->cbd.countHi = cellsHi * elementsPerCell;
      devWork->cbd.chunkGrainsLo = ...;
      devWork->cbd.chunkGrainsMid = ...;
      devWork->cbd.chunkGrainsHi = ...;
    }
    
    // 为每个channel创建proxy op
    for (int c = devWork->channelLo; c <= devWork->channelHi; c++) {
      ncclAddWorkBatchToPlan(comm, plan, c, workType, task->devFuncId, plan->workBytes);
      ncclAddProxyOpIfNeeded(comm, plan, proxyOp);
    }
  }
}
```

---

## 4. Kernel Launch 流程

### 4.1 Launch 准备 (ncclLaunchPrepare)

```cpp
ncclResult_t ncclLaunchPrepare(struct ncclComm* comm) {
  // 1. 创建新的kernel plan
  struct ncclKernelPlan* plan = ncclMemoryPoolAlloc<struct ncclKernelPlan>(...);
  
  // 2. 调度collective任务到plan
  NCCLCHECK(scheduleCollTasksToPlan(comm, plan, &budget));
  
  // 3. 调度P2P任务到plan
  NCCLCHECK(scheduleP2pTasksToPlan(comm, plan, &budget));
  
  // 4. 完成plan（构建kernel参数）
  finishPlan(comm, plan);
  
  // 5. 添加到plan queue
  ncclIntruQueueEnqueue(&planner->planQueue, plan);
}
```

### 4.2 Kernel Launch 执行

```cpp
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // 1. 计算grid/block配置
  int nChannels = countOneBits(plan->channelMask);
  dim3 grid = {(unsigned)nChannels, 1, 1};
  dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
  int smem = plan->isSymColl ? plan->kernelDynSmem : ncclShmemDynamicSize(comm->cudaArch);
  
  // 2. 获取kernel函数指针
  void* sym = plan->kernelFn;  // 如ncclDevKernel_Generic
  CUfunction fn;
  cudaGetFuncBySymbol(&fn, sym);
  
  // 3. 构建kernel参数
  void* extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs,
    CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
    CU_LAUNCH_PARAM_END
  };
  
  // 4. Launch kernel（支持CUDA 12.0+的扩展属性）
  #if CUDART_VERSION >= 11080
    CUlaunchConfig launchConfig = {...};
    // 配置Cluster、Mem Sync Domain等属性
    cuLaunchKernelEx(&launchConfig, fn, nullptr, extra);
  #else
    cuLaunchKernel(fn, grid.x, grid.y, grid.z, 
                   block.x, block.y, block.z, 
                   smem, launchStream, nullptr, extra);
  #endif
}
```

### 4.3 Launch 流程图

```
ncclGroupEnd
    │
    ▼
doLaunches
    │
    ├──► ncclLaunchPrepare(comm) ──► 创建plan, 调度任务
    │
    ├──► ncclCommIntraBarrierIn/Out ──► 进程内同步
    │
    ├──► ncclLaunchKernelBefore_NoUncapturedCuda ──► uploadWork
    │
    ├──► ncclLaunchKernel ──► CUDA Kernel Launch
    │
    ├──► ncclLaunchKernelAfter_NoCuda ──► hostStreamPlanTask
    │
    └──► ncclLaunchFinish ──► 流同步、事件处理
```

---

## 5. 通信拓扑创建与Tuning

### 5.1 拓扑发现流程

```
ncclCommInitRank
    │
    ▼
initTransportsRank
    │
    ├──► ncclTopoGetSystem ──► 从XML创建拓扑系统
    │
    ├──► ncclTopoComputePaths ──► 计算GPU间最短路径
    │
    ├──► ncclTopoSearchInit ──► 初始化搜索状态
    │
    └──► ncclTopoCompute ──► 计算Ring/Tree/Collnet图
```

### 5.2 拓扑数据结构

```cpp
// 拓扑系统
struct ncclTopoSystem {
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];  // GPU/PCI/CPU/NIC等节点
  float maxBw;      // 最大带宽
  float totalBw;    // 总带宽
  int inter;        // 是否需要跨节点通信
};

// 拓扑节点
struct ncclTopoNode {
  int type;  // GPU, PCI, CPU, NIC, NET, NVS等
  uint64_t id;
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];  // 出边连接
  int nlinks;
  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];  // 到各类型节点的路径
  
  union {
    struct { int dev; int rank; int cudaCompCap; int gdrSupport; } gpu;
    struct { int arch; int vendor; int model; } cpu;
    struct { uint64_t asic; float bw; float latency; int port; int gdrSupport; int collSupport; } net;
  };
};

// 连接链接
struct ncclTopoLink {
  int type;      // LINK_NVL, LINK_PCI, LINK_NET等
  float bw;      // 带宽
  struct ncclTopoNode* remNode;  // 对端节点
};
```

### 5.3 通信图搜索 (ncclTopoCompute)

```cpp
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  // 根据pattern类型选择搜索策略
  switch (graph->pattern) {
    case NCCL_TOPO_PATTERN_RING:
      // 使用贪心算法构建ring
      // 优先选择NVLink连接，其次PCIe
      break;
      
    case NCCL_TOPO_PATTERN_TREE:
    case NCCL_TOPO_PATTERN_BALANCED_TREE:
    case NCCL_TOPO_PATTERN_SPLIT_TREE:
      // 构建树形拓扑
      // 使用启发式搜索，考虑NIC位置
      break;
      
    case NCCL_TOPO_PATTERN_NVLS:
      // NVLink SHARP拓扑
      // 检查NVSwitches连接
      break;
      
    case NCCL_TOPO_PATTERN_COLLNET_DIRECT:
      // CollNet直接连接
      break;
  }
}
```

### 5.4 Tuning模型 (ncclTopoTuneModel)

```cpp
ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, 
                               struct ncclTopoGraph** graphs) {
  // 1. 设置各算法+协议组合的最大线程数
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, 256);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_MAX_NTHREADS;
  comm->maxThreads[NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] = NCCL_MAX_NTHREADS;
  // ... 其他组合
  
  // 2. 基于拓扑图计算各组合的带宽
  for (int coll = 0; coll < NCCL_NUM_FUNCTIONS; coll++) {
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
        // 计算busBw = nChannels * bw
        float bw = (nNodes <= 2 || collnet) ? graphs[a]->bwIntra : graphs[a]->bwInter;
        float busBw = graphs[a]->nChannels * bw;
        
        // 应用各种校正因子
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL) busBw = min(llMaxBw, busBw * 0.5);
        if (a == NCCL_ALGO_TREE) busBw = min(busBw * 0.92, graphs[a]->nChannels * perChMaxTreeBw);
        // ... 其他校正
        
        comm->bandwidths[coll][a][p] = busBw;
      }
    }
  }
  
  // 3. 计算延迟模型
  // latency = baseLatency + nSteps * hwLatency
  
  // 4. 设置线程阈值（决定何时减少线程数）
  comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL] = NCCL_LL_THREAD_THRESHOLD * nRanks;
  comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
  // ...
}
```

### 5.5 算法选择Cost模型

```cpp
ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, 
                                 int protocol, size_t nBytes, int numPipeOps, float* time) {
  float bw = comm->bandwidths[coll][algorithm][protocol];
  float lat = comm->latencies[coll][algorithm][protocol];
  
  if (bw == 0) {
    *time = -1.0;  // 不支持该组合
    return ncclSuccess;
  }
  
  // 应用Tree校正因子（中等大小优化）
  int logSize = log2i(nBytes >> 6);
  if (algorithm == NCCL_ALGO_TREE && coll == ncclFuncAllReduce && logSize >= 0 && logSize < 23) {
    bw *= treeCorrectionFactor[protocol][logSize];
  }
  
  // Pipeline节省延迟
  int latCount = (algorithm == NCCL_ALGO_RING) ? numPipeOps 
                                                : DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS);
  
  // 总时间 = 延迟 * 管道数 + 数据量 / 带宽
  *time = lat * latCount + nBytes / (1000 * bw);
  return ncclSuccess;
}
```

---

## 6. Kernel 线程与Warp配置

### 6.1 线程数配置常量

```cpp
#define WARP_SIZE 32
#define MAXCHANNELS 64
#define NCCL_MAX_NTHREADS 640       // 最大线程数
#define NCCL_MIN_NTHREADS (4*WARP_SIZE)  // 128，最小线程数
#define NCCL_SIMPLE_MAX_NTHREADS 512     // Simple协议最大线程数
#define NCCL_LL_MAX_NTHREADS 512         // LL协议最大线程数
#define NCCL_LL128_MAX_NTHREADS 640      // LL128协议最大线程数

// 各算法+协议的默认线程数（在tuning中可覆盖）
// - Ring + Simple: 256或512（取决于PCIe带宽）
// - Tree + Simple: 512
// - NVLS/NVLS_Tree: 640
// - Ring/Tree + LL: 512
// - Ring/Tree + LL128: 640
```

### 6.2 Warp分配策略

```cpp
// 线程数计算流程（topoGetAlgoInfo）
int nt = comm->maxThreads[algo][proto] / WARP_SIZE;  // 初始warp数

// 1. 小数据量时减少线程数
while (nBytes < nc * nt * threadThreshold) {
  if (nc >= 2) nc--;              // 优先减少channel数
  else if (nt % 4 == 0) nt /= 2;  // 然后减少warp数
  else break;
}

// 2. 特殊调整
if (proto == NCCL_PROTO_SIMPLE && algo == NCCL_ALGO_RING) {
  nt += 1;  // Ring Simple增加1个warp用于同步
}
if (algo == NCCL_ALGO_TREE) {
  nt = NCCL_MAX_NTHREADS / WARP_SIZE;  // Tree固定使用20个warp
}
if (algo == NCCL_ALGO_PAT) {
  nt = NCCL_MAX_NTHREADS / WARP_SIZE;  // PAT也使用全部线程
}

// 3. 确保最少warp数
nt = (nt < 3) ? 3 : nt;  // 最少3个warp（96线程）
```

### 6.3 线程阈值配置

```cpp
// 每个线程处理的数据量阈值
#define NCCL_LL_THREAD_THRESHOLD 8        // LL: 8 bytes/thread
#define NCCL_LL128_THREAD_THRESHOLD 8     // LL128: 8 bytes/thread  
#define NCCL_SIMPLE_THREAD_THRESHOLD 64   // Simple: 64 bytes/thread

// 实际阈值在tuning时计算（考虑rank数）
comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL] = NCCL_LL_THREAD_THRESHOLD * nRanks;
comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
```

### 6.4 Kernel Grid配置

```cpp
// Kernel Launch时的配置
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  int nChannels = countOneBits(plan->channelMask);  // 实际使用的channel数
  
  dim3 grid = {(unsigned)nChannels, 1, 1};          // gridDim.x = channel数
  dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};  // blockDim.x = 线程数
  int smem = ncclShmemDynamicSize(comm->cudaArch);  // 动态共享内存大小
  
  // CUDA 12.0+支持额外属性
  #if CUDART_VERSION >= 12000
    // - Cluster配置（SM90+）
    // - Mem Sync Domain
    // - Launch Completion Event
    // - NVLink Util Centric Scheduling（SM100+）
  #endif
}
```

### 6.5 共享内存计算

```cpp
// 每个warp的scratch内存大小
__host__ __device__ constexpr int ncclShmemScratchWarpSize(int cudaArch = NCCL_CUDA_ARCH) {
  return (max_constexpr<int>(
    /*LL    */ 0,
    /*LL128 */ (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE)*sizeof(uint64_t),
    /*SIMPLE*/ (ncclCollUnroll(cudaArch)*WARP_SIZE + 1)*16,
    /*NVLS  */ WARP_SIZE*(cudaArch >= 900 ? ncclNvlsUnrollBytes(cudaArch) : 0) + 16
  ) + 15) & -16;
}

// 每block的动态共享内存 = 每warp大小 * warp数
__host__ __device__ constexpr int ncclShmemDynamicSize(int cudaArch = NCCL_CUDA_ARCH) {
  return cudaArch < 700 ? 0 : ncclShmemScratchWarpSize(cudaArch)*(NCCL_MAX_NTHREADS/WARP_SIZE);
}
```

---

## 7. 关键数据结构

### 7.1 任务结构 (ncclTaskColl)

```cpp
struct ncclTaskColl {
  struct ncclTaskColl* next;
  ncclFunc_t func;              // collective类型（AllReduce等）
  void const* sendbuff;
  void* recvbuff;
  size_t count;
  int root;
  ncclDataType_t datatype;
  ncclRedOp_t opHost;
  struct ncclDevRedOpFull opDev;
  int chunkSteps, sliceSteps;
  
  // 计算结果
  size_t trafficBytes;          // 用于排序
  int32_t nMaxChannels:8;       // 最大channel数
  int32_t nWarps:8;             // warp数
  int32_t algorithm:8;          // 算法选择
  int32_t protocol:8;           // 协议选择
  uint32_t devFuncId:29;        // kernel函数ID
  
  // 注册buffer相关
  int regBufType;
  void* sendMhandle;
  void* recvMhandle;
  
  // Profiler
  int eActivationMask;
  void* eventHandle;
  uint8_t nChannels;
};
```

### 7.2 Kernel Plan结构

```cpp
struct ncclKernelPlan {
  struct ncclCommCallback reclaimer;
  struct ncclComm* comm;
  struct ncclKernelPlan* next;
  
  bool persistent;              // 是否被CUDA Graph捕获
  bool isSymColl;               // 是否对称collective
  bool isCeColl;                // 是否Copy Engine collective
  int kernelDynSmem;            // 动态共享内存大小
  void* kernelFn;               // kernel函数指针
  union {
    struct ncclDevKernelArgs* kernelArgs;
    struct ncclSymkDevWorkArgs* kernelSymArgs;
  };
  size_t kernelArgsSize;
  uint64_t channelMask;         // 使用的channel位掩码
  int threadPerBlock;           // 每block线程数
  
  int collOpCount;              // collective操作数
  int nWorkBatches;             // work batch数
  size_t workBytes;             // 总work大小
  
  // 任务队列
  struct ncclIntruQueue<struct ncclWorkList> workQueue;
  struct ncclIntruQueue<struct ncclProxyOp> proxyOpQueue;
  struct ncclIntruQueue<struct ncclTaskColl> collTaskQueue;
};
```

### 7.3 Device Work结构

```cpp
struct alignas(16) ncclDevWorkColl {
  uint32_t channelLo:8, channelHi:8;  // 负责的channel范围
  uint32_t nWarps:8;
  uint32_t redOpArgIsPtr:1, regUsed:1, netRegUsed:1, oneNode:1, direct:2;
  uint32_t root;
  void* recvbuff;
  void* sendbuff;
  
  // Continuous-byte-distribution参数
  struct {
    size_t countLo, countMid, countHi;
    uint64_t chunkGrainsLo:21, chunkGrainsMid:21, chunkGrainsHi:21;
  } cbd;
  
  uint64_t redOpArg;
};

struct ncclDevKernelArgs {
  struct ncclKernelComm* comm;
  uint64_t channelMask;
  enum ncclDevWorkStorageType workStorageType;
  uint32_t workMask;
  void* workBuf;
  // struct ncclDevWorkBatch batches[];  // 变长数组
};
```

---

## 8. 总结与流程图

### 8.1 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            用户调用阶段                                       │
│  ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            任务入队阶段                                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │  参数检查        │───▶│ 创建ncclTaskColl │───▶│ 插入collSorter（按大小排序）│ │
│  │ (ArgsCheck)     │    │ (填充实参信息)   │    │ （桶排序，降序）          │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ncclGroupEnd 处理阶段                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ncclPrepareTasks                                                   │   │
│  │  ├─ 从sorter获取排序后的任务                                         │   │
│  │  ├─ 对称任务分离（Symmetric）                                        │   │
│  │  ├─ 按(func,op,type)聚合任务                                         │   │
│  │  ├─ 调用ncclGetAlgoInfo选择算法/协议/线程数                           │   │
│  │  │   ├─ 查询tuner插件（如果有）                                      │   │
│  │  │   └─ 使用默认topology-based tuner                                 │   │
│  │  │       ├─ 计算cost = latency + nBytes/bw                          │   │
│  │  │       └─ 选择cost最小的(algo,proto)组合                          │   │
│  │  └─ 分类到collBins[isCollnet][isNvls]                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ncclTasksRegAndEnqueue                                             │   │
│  │  ├─ 注册buffer（NVLS/CollNet/IPC/NET）                              │   │
│  │  └─ 创建ncclDevWorkColl结构                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Kernel Plan生成阶段                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  scheduleCollTasksToPlan                                            │   │
│  │  ├─ 估算channel需求                                                  │   │
│  │  ├─ 为每个任务分配channel范围 [channelLo..channelHi]                 │   │
│  │  ├─ 计算CBD参数(countLo/Mid/Hi, chunkGrainsLo/Mid/Hi)               │   │
│  │  ├─ 调用calcCollChunking计算chunkSize、sliceSize等                  │   │
│  │  ├─ 为每个channel创建work batch                                     │   │
│  │  └─ 创建proxyOp（网络通信需要）                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  finishPlan                                                         │   │
│  │  ├─ 确定workStorageType（Args/Fifo/Persistent）                     │   │
│  │  ├─ 分配kernelArgs内存                                              │   │
│  │  ├─ 填充kernelArgs（comm, channelMask, workStorageType, batches）   │   │
│  │  └─ 合并各channel的proxyOp到plan的proxyOpQueue                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Kernel Launch阶段                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ncclLaunchPrepare                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ncclCommIntraBarrierIn/Out（进程内多rank同步）                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ncclLaunchKernelBefore_NoUncapturedCuda                            │   │
│  │  └─ uploadWork: 将work数据上传到GPU                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ncclLaunchKernel                                                   │   │
│  │  ├─ grid = (nChannels, 1, 1)                                        │   │
│  │  ├─ block = (threadPerBlock, 1, 1)                                  │   │
│  │  ├─ smem = ncclShmemDynamicSize()                                   │   │
│  │  ├─ 支持CUDA 12.0+扩展属性（Cluster/MemSyncDomain等）               │   │
│  │  └─ cuLaunchKernel/cuLaunchKernelEx                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ncclLaunchKernelAfter_NoCuda                                       │   │
│  │  └─ hostStreamPlanTask: 提交proxy ops到proxy线程                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ncclLaunchFinish                                                   │   │
│  │  ├─ 记录finishedEvent                                               │   │
│  │  ├─ deviceStream等待launchStream                                    │   │
│  │  └─ 各user stream等待finishedEvent                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 核心设计要点

1. **任务排序优化**：通过ncclTaskCollSorter将任务按大小降序排列，确保大任务优先获得channel资源

2. **算法自适应选择**：基于topology信息和cost模型动态选择最优算法（Ring/Tree/NVLS/CollNet）和协议（LL/LL128/Simple）

3. **Channel动态分配**：使用Continuous-Byte-Distribution (CBD)机制，根据数据量动态分配各channel的工作量

4. **线程数动态调整**：根据数据量大小、算法类型和硬件能力动态调整线程数（128~640线程）

5. **Proxy机制**：网络通信通过独立的proxy线程处理，实现计算与通信重叠

6. **拓扑感知**：基于实际硬件拓扑（NVLink/PCIe/NIC）构建通信图，优化路由

7. **CUDA Graph支持**：支持将collective操作捕获到CUDA Graph中，降低launch开销

---

## 附录：关键参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| NCCL_NTHREADS | 根据架构 | Simple协议线程数 |
| NCCL_LL128_NTHREADS | 640 | LL128协议线程数 |
| NCCL_ALGO | - | 强制指定算法 |
| NCCL_PROTO | - | 强制指定协议 |
| NCCL_MAX_NCHANNELS | 64 | 最大channel数 |
| NCCL_CTA_POLICY | DEFAULT | CTA调度策略 |
| NCCL_PAT_ENABLE | 2 | PAT算法启用 |

---

*文档生成日期：2026-03-30*
*基于NCCL源代码分析*
