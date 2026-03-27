# NCCL Channel 与 Pipeline 机制技术文档

## 目录
1. [概述](#概述)
2. [Channel 拆分机制](#channel-拆分机制)
   - 2.1 [Channel 概念与作用](#channel-概念与作用)
   - 2.2 [Channel 数量确定](#channel-数量确定)
   - 2.3 [Channel 结构体定义](#channel-结构体定义)
   - 2.4 [Channel 初始化流程](#channel-初始化流程)
   - 2.5 [Channel 与 GPU/网卡映射](#channel-与-gpu网卡映射)
3. [Pipeline 机制](#pipeline-机制)
   - 3.1 [Pipeline 基本原理](#pipeline-基本原理)
   - 3.2 [计算与通信重叠](#计算与通信重叠)
   - 3.3 [Pipeline Stage 划分](#pipeline-stage-划分)
   - 3.4 [Chunk 与 Slice 机制](#chunk-与-slice-机制)
4. [性能优化策略](#性能优化策略)
   - 4.1 [Channel 与 Pipeline 协同](#channel-与-pipeline-协同)
   - 4.2 [不同 Collective 的优化](#不同-collective-的优化)
   - 4.3 [协议选择 (LL/LL128/SIMPLE)](#协议选择-llll128simple)
5. [代码实现细节](#代码实现细节)
   - 5.1 [Channel 分配代码](#channel-分配代码)
   - 5.2 [任务调度代码](#任务调度代码)
   - 5.3 [Proxy 进度机制](#proxy-进度机制)

---

## 概述

NCCL (NVIDIA Collective Communications Library) 通过 **Channel** 和 **Pipeline** 两个核心机制实现高效的分布式通信：

- **Channel**: 将通信资源（如连接、缓冲区）划分为多个独立通道，实现并行通信
- **Pipeline**: 将大数据传输划分为多个阶段（stages），实现计算与通信重叠，隐藏延迟

这两个机制协同工作，使 NCCL 能够充分利用多 GPU 和多节点的带宽资源。

---

## Channel 拆分机制

### Channel 概念与作用

Channel 是 NCCL 中通信的基本执行单元。每个 Channel 代表一个独立的通信路径，包含：
- 发送/接收连接器 (send/recv connectors)
- 通信缓冲区 (buffers for different protocols)
- 连接状态信息
- 环形/树形拓扑信息

**核心作用**:
1. **并行通信**: 多个 Channel 同时传输不同数据段，提高带宽利用率
2. **负载均衡**: 数据均匀分布到多个 Channel
3. **资源隔离**: 不同 Collective 操作可使用不同 Channel 集合

### Channel 数量确定

Channel 数量由以下因素决定：

```c
// 关键参数 (src/graph/connect.cc)
NCCL_PARAM(MinNchannels, "MIN_NCHANNELS", -2);  // 最小通道数
NCCL_PARAM(MaxNchannels, "MAX_NCHANNELS", -2);  // 最大通道数

#define MAXCHANNELS 64  // 硬编码上限 (src/include/device.h)
```

**确定流程** (`ncclTopoPostset` in `src/graph/connect.cc`):

1. **初始通道数**: 由拓扑搜索算法根据 NVLink/PCIe/网络拓扑决定
2. **CollNet 扩展**: 如果启用 CollNet，通道数可能增加
3. **NVLS 通道**: 独立的 NVLS 通道数 (`nvlsChannels`)
4. **P2P 通道**: 独立的点对点通信通道 (`p2pnChannels`)
5. **用户覆盖**: 环境变量 `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS`

**典型配置**:
- 单节点 NVLink: 通常 12-16 个通道
- 多节点: 根据网络带宽和 GPU 数量动态调整
- NVLS (Hopper+): 独立的通道用于 NVLink SHARP

### Channel 结构体定义

**核心结构体** (`src/include/comm.h`):

```c
struct ncclChannel {
  struct ncclChannelPeer** peers;           // 指向所有 rank 的 peer 信息
  struct ncclDevChannelPeer** devPeers;     // 设备端 peer 信息
  struct ncclDevChannelPeer** devPeersHostPtr;  // 主机访问用的设备指针
  
  // 拓扑结构
  struct ncclRing ring;                     // 环形拓扑
  struct ncclTree tree;                     // 树形拓扑 (Tree/AllReduce)
  struct ncclTree collnetChain;             // CollNet Chain 拓扑
  struct ncclDirect collnetDirect;          // CollNet Direct 拓扑
  struct ncclNvls nvls;                     // NVLS 拓扑
  
  int id;                                   // Channel 索引
  uint32_t workFifoProduced;                // Work FIFO 生产者指针
  
  // CollNet/NVLS 专用 peers
  struct ncclChannelPeer* collnetPeers;
  struct ncclDevChannelPeer* collnetDevPeers;
  struct ncclChannelPeer* nvlsPeers;
  struct ncclDevChannelPeer* nvlsDevPeers;
};
```

**Peer 结构体**:
```c
#define NCCL_MAX_CONNS 2
struct ncclChannelPeer {
  struct ncclConnector send[NCCL_MAX_CONNS];  // 发送连接器 (2个连接)
  struct ncclConnector recv[NCCL_MAX_CONNS];  // 接收连接器 (2个连接)
  int refCount;
};
```

**Connector 结构体**:
```c
struct ncclConnector {
  int connected;                            // 连接状态
  int hasSeen;                              // 是否已见过
  int p2pOnly;                              // 是否仅用于 P2P
  struct ncclProxyConnector proxyConn;      // Proxy 连接
  struct ncclTransportComm* transportComm;  // 传输通信层
  void* transportResources;                 // 传输资源
  struct ncclConnInfo conn;                 // 连接信息
};
```

### Channel 初始化流程

**初始化入口** (`src/channel.cc`):

```c
ncclResult_t initChannel(struct ncclComm* comm, int channelId) {
  struct ncclChannel* channel = &comm->channels[channelId];
  if (channel->id != -1) return ncclSuccess;  // 已初始化

  int nRanks = comm->nRanks;
  int nvlsRanks = comm->localRanks;
  int nPeers = nRanks + 1 /* Collnet */ + nvlsRanks /* NVLS */;
  channel->id = channelId;
  channel->workFifoProduced = 0;

  // 1. 分配 peers 数组
  if (channel->peers == NULL) {
    if (sharedRes->peers[channelId] == NULL) {
      NCCLCHECK(ncclCalloc(sharedRes->peers + channelId, sharedRes->tpNRanks));
    }
    channel->peers = ncclMemoryStackAlloc<struct ncclChannelPeer*>(
        &comm->memPermanent, nPeers);
    for (int r = 0; r < nRanks; r++) {
      channel->peers[r] = comm->sharedRes->peers[channelId] + comm->topParentRanks[r];
      ncclAtomicRefCountIncrement(&channel->peers[r]->refCount);
    }
  }

  // 2. 分配设备端 peers
  if (channel->devPeers == NULL) {
    if (sharedRes->devPeers[channelId] == NULL) {
      NCCLCHECK(ncclCudaCallocAsync(sharedRes->devPeers + channelId, 
          sharedRes->tpNRanks, deviceStream, comm->memManager));
    }
    NCCLCHECK(ncclCudaCallocAsync(&channel->devPeers, nPeers, 
        deviceStream, comm->memManager));
    // 设置设备端 peer 指针
    for (int r = 0; r < nRanks; r++) {
      uintptr_t addr = (uintptr_t)(comm->sharedRes->devPeers[channelId] 
          + comm->topParentRanks[r]);
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + r), 
          (uintptr_t*)&addr, 1, deviceStream));
    }
  }

  // 3. 初始化环形拓扑
  channel->ring.userRanks = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
  channel->ring.rankToIndex = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
  NCCLCHECK(ncclCudaCallocAsync(&channel->devRingUserRanks, nRanks, 
      deviceStream, comm->memManager));
  
  return ncclSuccess;
}
```

### Channel 与 GPU/网卡映射

**映射关系**:

```
+-----------------------------------------------------------+
|                      GPU 0                                |
|  +-------------------+  +-------------------+             |
|  |   Channel 0       |  |   Channel 1       |  ...        |
|  |  - Ring Prev/Next |  |  - Ring Prev/Next |             |
|  |  - Tree Up/Down   |  |  - Tree Up/Down   |             |
|  |  - NVLS Head      |  |  - NVLS Head      |             |
|  +--------+----------+  +--------+----------+             |
|           |                      |                        |
|           v                      v                        |
|  +--------+----------+  +--------+----------+             |
|  |  Connector 0      |  |  Connector 1      |             |
|  |  (to GPU 1)       |  |  (to GPU 2)       |             |
|  +-------------------+  +-------------------+             |
+-----------------------------------------------------------+
```

**拓扑连接** (`src/graph/connect.cc`):

```c
// 环形连接
static ncclResult_t connectRings(struct ncclComm* comm, 
    int* ringRecv, int* ringSend, int* ringPrev, int* ringNext) {
  int nChannels = comm->nChannels;
  int nNodes = comm->nNodes;
  for (int c=0; c<nChannels; c++) {
    int* recv = ringRecv+c*comm->nNodes;
    int* send = ringSend+c*comm->nNodes;
    int* prev = ringPrev+c*comm->nRanks;
    int* next = ringNext+c*comm->nRanks;
    for (int n=0; n<nNodes; n++) {
      int recvRank = recv[n];
      int prevSendRank = send[(n-1+nNodes)%nNodes];
      prev[recvRank] = prevSendRank;
      int sendRank = send[n];
      int nextRecvRank = recv[(n+1)%nNodes];
      next[sendRank] = nextRecvRank;
    }
  }
  return ncclSuccess;
}
```

---

## Pipeline 机制

### Pipeline 基本原理

Pipeline 机制将大数据传输划分为多个小块（chunks），通过流水线方式处理，实现：
1. **隐藏延迟**: 当前 chunk 传输时，准备下一个 chunk
2. **计算通信重叠**: GPU 计算与网络传输同时进行
3. **流量控制**: 通过 step 机制防止缓冲区溢出

**核心概念**:
- **Chunk**: 数据传输的基本单元
- **Slice**: Chunk 的子划分，用于细粒度流水线
- **Step**: 流水线阶段，每个 step 处理一个 slice
- **Loop**: 完整的数据传输周期

### 计算与通信重叠

NCCL 通过以下方式实现重叠：

```
时间轴 ->
GPU 0: [Compute Chunk 0] [Compute Chunk 1] [Compute Chunk 2]
          |                    |                    |
          v                    v                    v
NIC  :    [Send Chunk 0]       [Send Chunk 1]       [Send Chunk 2]
          ^                    ^                    ^
          |                    |                    |
GPU 1:    [Recv Chunk 0]       [Recv Chunk 1]       [Recv Chunk 2]
          [Compute Chunk 0]    [Compute Chunk 1]    [Compute Chunk 2]
```

**实现机制**:
1. **异步传输**: CUDA 异步拷贝与计算 kernel 重叠
2. **双缓冲 (Double Buffering)**: 使用 NCCL_STEPS (默认 8) 个缓冲区轮转
3. **Proxy 线程**: 主机端线程处理网络 I/O，不阻塞 GPU

### Pipeline Stage 划分

**Step 定义** (`src/include/device.h`):
```c
#define NCCL_STEPS 8  // 缓冲区步数
```

**Chunk/Slice 配置** (`src/include/collectives.h`):
```c
// AllReduce 使用 2-step pipeline
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)   // 2
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)   // 4

// AllGather/ReduceScatter
#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)   // 2
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)   // 4

// P2P/Simple collectives
#define ALLTOALL_SLICESTEPS 1
#define ALLTOALL_CHUNKSTEPS 1
```

**Step 状态机**:
```
+-------+      +-------+      +-------+      +-------+
|  Idle | ---> | Ready | ---> |Posted | ---> | Trans | ---> Done
+-------+      +-------+      +-------+      +-------+
   ^                                              |
   +----------------------------------------------+
```

### Chunk 与 Slice 机制

**数据结构** (`src/include/proxy.h`):
```c
struct ncclProxySubArgs {
  // ...
  uint64_t base;       // 基础 step
  uint64_t posted;     // 已提交 step
  uint64_t received;   // 已接收 step
  uint64_t flushed;    // 已刷新 step
  uint64_t transmitted;// 已传输 step
  uint64_t done;       // 已完成 step
  uint64_t end;        // 结束 step
  // ...
};
```

**Chunk 计算** (`src/enqueue.cc`):
```c
static ncclResult_t calcCollChunking(
    struct ncclComm* comm, struct ncclTaskColl* info, 
    int nChannels, size_t nBytes,
    uint32_t* outChunkSize, uint32_t* outDirectFlags, 
    struct ncclProxyOp* proxyOp) {
  
  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && 
                   info->algorithm == NCCL_ALGO_RING) ? 
                   info->chunkSteps : 1;
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && 
                   info->algorithm == NCCL_ALGO_RING) ? 
                   info->sliceSteps : 1;
  
  int stepSize = comm->buffSizes[info->protocol]/NCCL_STEPS;
  int chunkSize = stepSize*chunkSteps;
  
  // 根据数据量动态调整 chunk size
  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads 
           * chunkSize) < comm->channels[0].collnetDirect.depth * 64 
           && chunkSize > 131072) {
      chunkSize /= 2;
    }
  }
  // ...
  
  // 计算 step 数
  size_t loopSize = size_t(nChannels)*nchunksPerLoop*chunkSize;
  int nLoops = (int)DIVUP(nBytes, loopSize);
  proxyOp->nsteps = nstepsPerLoop * nLoops * chunkSteps;
  proxyOp->sliceSteps = sliceSteps;
  proxyOp->chunkSteps = chunkSteps;
  proxyOp->chunkSize = chunkSize;
  proxyOp->sliceSize = chunkSize / chunkSteps * sliceSteps;
  
  return ncclSuccess;
}
```

---

## 性能优化策略

### Channel 与 Pipeline 协同

**协同工作机制**:

```
数据分布策略 (Continuous Byte Distribution - CBD):

总数据: 128MB
Channel 0 (Lo):  48MB  (chunkSize=16MB)
Channel 1..n-1:  每个 16MB
Channel n (Hi):  剩余数据

每个 Channel 内部使用 Pipeline:
Channel 0: [Slice 0] [Slice 1] [Slice 2] ... (流水线处理 48MB)
Channel 1: [Slice 0] [Slice 1] ...          (流水线处理 16MB)
...
```

**代码实现** (`src/enqueue.cc`):
```c
// 计算每个 channel 的数据分布
size_t cellSize = divUp(divUp(MinTrafficPerChannel, (size_t)trafficPerByte), 16) * 16;
int elementsPerCell = cellSize/elementSize;
size_t cells = divUp(task->count*elementSize, cellSize);
size_t trafficPerCell = cellSize*trafficPerByte;
size_t cellsPerChannel = std::min(cells, divUp(trafficPerChannel, trafficPerCell));

// Lo/Mid/Hi 分布
size_t cellsLo = std::min(cells, divUp((trafficPerChannel-currentTraffic),trafficPerCell));
int nMidChannels = (cells-cellsLo)/cellsPerChannel;
size_t cellsHi = (cells-cellsLo)%cellsPerChannel;

// 设置到 work struct
devWork->channelLo = channelId;
devWork->channelHi = channelId + nChannels-1;
devWork->cbd.countLo = countLo;
devWork->cbd.countMid = countMid;
devWork->cbd.countHi = countHi;
```

### 不同 Collective 的优化

**AllReduce**:
- **Ring**: 2*(nRanks-1) steps，每个 rank 接收、reduce、发送
- **Tree**: 2*log(nRanks) steps，reduce-scatter + allgather
- **NVLS**: 1 step intra-node + 1 step inter-node

**AllGather/ReduceScatter**:
- **Ring**: (nRanks-1) steps，数据环状传递
- **NVLS**: NVLink SHARP 加速

**代码片段** (`src/enqueue.cc`):
```c
switch (info->func) {
case ncclFuncAllReduce:
  pattern =
    info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
    info->algorithm == NCCL_ALGO_NVLS_TREE ? ncclPatternNvlsTree :
    info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
    info->algorithm == NCCL_ALGO_COLLNET_CHAIN ? ncclPatternCollnetChain :
    info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUpDown :
    ncclPatternRingTwice;  // Ring 需要 2 轮
  break;
case ncclFuncAllGather:
  pattern =
    info->algorithm == NCCL_ALGO_PAT ? ncclPatternPatDown :
    info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
    info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
    ncclPatternRing;
  break;
}

// 计算 steps per loop
switch (pattern) {
case ncclPatternRing:
  nstepsPerLoop = comm->nRanks-1; 
  nchunksPerLoop = comm->nRanks;
  break;
case ncclPatternRingTwice:
  nstepsPerLoop = 2*(comm->nRanks-1);
  nchunksPerLoop = comm->nRanks;
  break;
case ncclPatternNvls:
  nstepsPerLoop = 1; 
  nchunksPerLoop = comm->channels[0].nvls.nHeads;
  break;
}
```

### 协议选择 (LL/LL128/SIMPLE)

**三种协议对比**:

| 协议 | 延迟 | 带宽 | 适用场景 |
|------|------|------|----------|
| LL (Low Latency) | 最低 | 较低 | 小数据 (<256KB) |
| LL128 | 低 | 中 | 中小数据 (256KB-1MB) |
| SIMPLE | 稍高 | 最高 | 大数据 (>1MB) |

**选择逻辑** (`src/graph/tuning.cc`):
```c
// LL128 启用条件
if (pEnable == 2 && p == NCCL_PROTO_LL128) {
  pEnable = 1;
  // Hopper/Blackwell 上启用
  if (ncclParamLl128C2c() && minCompCap >= 90) {
    pEnable &= (graphs[a]->typeInter <= PATH_PXN);
  } else {
    pEnable &= (graphs[a]->typeInter <= PATH_PXB);
  }
  pEnable &= (graphs[a]->typeIntra <= PATH_NVB);
  pEnable &= (minCompCap == maxCompCap);
}
```

---

## 代码实现细节

### Channel 分配代码

**任务到 Channel 映射** (`src/enqueue.cc`):
```c
// 确定 channel 范围
int nChannels = task->nMaxChannels;
int channelId = 0;

// CollNet 使用所有通道
if (task->isCollnet) {
  devWork->channelLo = 0;
  devWork->channelHi = nChannels-1;
} else {
  // 标准算法: CBD 分布
  // 计算每个 channel 的数据量
  size_t countPerChannel = task->count / nChannels;
  devWork->channelLo = startChannel;
  devWork->channelHi = startChannel + nChannels - 1;
  devWork->cbd.countLo = countLo;
  devWork->cbd.countMid = countMid;
  devWork->cbd.countHi = countHi;
}

// 更新 channel mask
plan->channelMask |= (2ull<<devWork->channelHi) - (1ull<<devWork->channelLo);
```

### 任务调度代码

**WipPlan (Work-in-Progress Plan)** (`src/include/comm.h`):
```c
struct ncclKernelPlanner {
  struct WipPlan {
    struct Channel {
      struct {
        int workBytes;
        int nP2ps;
        int nBcasts;
        int p2pEpoch;
        int p2pRounds[NCCL_MAX_DEV_WORK_P2P_PER_BATCH];
      } wipBatch;
      int nWorkBatchesP2p;
      int nWorkBatchesBcast;
      struct ncclIntruQueue<struct ncclWorkBatchList, ...> workBatchQueue;
      struct ncclIntruQueue<struct ncclProxyOp, ...> proxyOpQueue;
    } channels[MAXCHANNELS];
  } wipPlan;
};
```

**添加 Work Batch** (`src/enqueue.cc`):
```c
void ncclAddWorkBatchToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, 
    int channelId, enum ncclDevWorkType workType, 
    int devFuncId, uint32_t workOffset,
    int p2pEpoch, int p2pRound, bool newBatch) {
  
  ncclKernelPlanner::WipPlan::Channel* chan = 
      &comm->planner.wipPlan.channels[channelId];
  
  // 检查是否需要新 batch
  newBatch = (chan->workBatchQueue.tail == nullptr);
  if (!newBatch) {
    batch = &chan->workBatchQueue.tail->batch;
    newBatch |= batch->workType != (uint8_t)workType;
    newBatch |= batch->funcId != devFuncId;
    // P2P 特定检查
    if (workType == ncclDevWorkTypeP2p) {
      newBatch |= chan->wipBatch.nP2ps == NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
    }
  }
  
  // 创建或扩展 batch
  if (newBatch || extendBatch) {
    // 创建新的 batch node
    struct ncclWorkBatchList* batchNode = 
        ncclMemoryStackAlloc<ncclWorkBatchList>(&comm->memScoped);
    ncclIntruQueueEnqueue(&chan->workBatchQueue, batchNode);
    batch = &batchNode->batch;
    batch->workType = (uint32_t)workType;
    batch->funcId = devFuncId;
    batch->offsetBase = workOffset;
    batch->offsetBitset = 0;
  }
  
  // 设置 work offset
  batch->offsetBitset |= 1ull<<(offset/workSize);
}
```

### Proxy 进度机制

**Proxy 状态机** (`src/proxy.cc`):
```c
// Proxy 操作状态
enum ncclProxyOpState { 
  ncclProxyOpNone,      // 无操作
  ncclProxyOpReady,     // 准备就绪
  ncclProxyOpProgress   // 进行中
};

// 进度函数
static ncclResult_t ProxyAppend(struct ncclProxyProgressState* state, 
    struct ncclProxyOp* op) {
  struct ncclProxyConnection* connection = op->connection;
  struct ncclProxyArgs* args = *connection->proxyAppendPtr;

  if (args && shared && args->opCount == op->opCount) {
    // 合并到现有操作组
    NCCLCHECK(ncclProxyOpToArgs(op, args, args->nsubs));
  } else {
    // 分配新操作
    NCCLCHECK(allocateArgs(state, &args));
    NCCLCHECK(ncclProxyOpToArgs(op, args, 0));
    // 添加到活动列表
    if (state->active == NULL) {
      state->active = args;
    } else {
      // 追加到链表尾部
      struct ncclProxyArgs* last = state->active;
      while (last->next) last = last->next;
      last->next = args;
    }
  }
  return ncclSuccess;
}
```

**Step 进度追踪**:
```c
// 接收状态机
if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) {
  // 提交接收请求
}
if (sub->received < sub->posted) {
  // 处理接收完成
}
if (sub->transmitted < sub->received) {
  // 传输到下一节点 (对于 relay)
}
if (sub->done < sub->transmitted) {
  // 等待 GPU 完成
}
```

---

## 总结

NCCL 的 Channel 和 Pipeline 机制是其高性能的关键：

1. **Channel 机制**:
   - 通过多通道并行通信，充分利用硬件带宽
   - 支持多种拓扑结构 (Ring/Tree/CollNet/NVLS)
   - 动态分配和数据分布 (CBD) 策略

2. **Pipeline 机制**:
   - 细粒度的 chunk/slice 划分
   - 8-step 双缓冲隐藏延迟
   - 计算与通信重叠

3. **协同优化**:
   - 根据数据大小自动选择最佳算法和协议
   - Proxy 线程异步处理网络 I/O
   - 针对不同的 Collective 操作定制化优化

这些机制共同确保了 NCCL 在多 GPU 和多节点环境中能够实现接近硬件极限的通信性能。
