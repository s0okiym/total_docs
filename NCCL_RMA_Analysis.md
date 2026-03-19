# NCCL RMA (Remote Memory Access) 深度分析

## 目录

1. [概述](#1-概述)
2. [架构设计](#2-架构设计)
3. [核心数据结构](#3-核心数据结构)
4. [API 接口详解](#4-api-接口详解)
5. [关键函数分析](#5-关键函数分析)
6. [工作流程](#6-工作流程)
7. [两种执行路径](#7-两种执行路径)
8. [Signal 机制详解](#8-signal-机制详解)
9. [与 GIN 的集成](#9-与-gin-的集成)
10. [适用场景与收益](#10-适用场景与收益)
11. [配置参数](#11-配置参数)
12. [代码位置索引](#12-代码位置索引)

---

## 1. 概述

### 1.1 什么是 RMA

RMA (Remote Memory Access) 是 NCCL 提供的一侧通信 (One-sided Communication) 机制。与传统的两侧通信 (如 Send/Recv) 不同，RMA 允许一个进程直接访问另一个进程的内存，而无需目标进程的显式参与。

### 1.2 设计目标

- **一侧通信**：发起方直接操作远程内存，目标方无需参与
- **低延迟**：减少同步开销，提高通信效率
- **灵活性**：支持非阻塞操作和细粒度同步
- **可扩展性**：支持大规模分布式训练场景

### 1.3 核心特性

| 特性 | 描述 |
|------|------|
| **PutSignal** | 将本地数据写入远程内存并发送信号 |
| **Signal** | 仅发送信号，不传输数据 |
| **WaitSignal** | 等待来自一个或多个 rank 的信号 |
| **Window** | 内存窗口注册机制，支持对称内存访问 |

### 1.4 与传统通信的对比

| 特性 | 传统 Collective | RMA |
|------|----------------|-----|
| 通信模式 | 两侧 (Send/Recv) | 一侧 (Put/Get) |
| 同步方式 | 隐式同步 | 显式 Signal |
| 目标参与 | 需要目标进程配合 | 目标进程被动 |
| 适用场景 | 集体操作 | 点对点数据传输 |

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      User API Layer                              │
│  ncclPutSignal() / ncclSignal() / ncclWaitSignal()              │
│  ncclCommWindowRegister() / ncclCommWindowDeregister()          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Enqueue Layer                               │
│  collectives.cc: API 入口                                        │
│  enqueue.cc: rmaTaskAppend() - 任务创建和入队                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Scheduler Layer                             │
│  rma.cc: scheduleRmaTasksToPlan() - 任务调度和分发               │
│  - 分离 CE 任务和 Proxy 任务                                     │
│  - 任务批处理优化                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Execution Layer                             │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │   CE Path           │    │   Proxy Path        │            │
│  │   (rma_ce.cc)       │    │   (rma_proxy.cc)    │            │
│  │   - NVLink 直连     │    │   - GIN 网络        │            │
│  │   - cudaMemcpyAsync │    │   - RDMA PUT        │
│  └─────────────────────┘    └─────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Hardware Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  NVLink/NVS  │  │  GPUDirect   │  │  InfiniBand  │          │
│  │  (CE Path)   │  │  RDMA        │  │  (Proxy)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键文件结构

```
src/
├── rma/                        # RMA 核心实现
│   ├── rma.cc                  # RMA 主入口和调度
│   ├── rma_ce.cc               # CE (Copy Engine) 路径实现
│   └── rma_proxy.cc            # Proxy 路径实现
├── include/rma/
│   ├── rma.h                   # RMA 主头文件
│   ├── rma_ce.h                # CE 路径头文件
│   └── rma_proxy.h             # Proxy 路径头文件
├── collectives.cc              # API 入口实现
├── enqueue.cc                  # 任务入队逻辑
└── nccl.h.in                   # 公共 API 定义
```

### 2.3 模块关系图

```
                    ┌──────────────────┐
                    │   User Code      │
                    └────────┬─────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                    ncclPutSignal/Signal/WaitSignal         │
│                    ncclCommWindowRegister/Deregister       │
└────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                    ncclEnqueueCheck                         │
│                    rmaTaskAppend                            │
└────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                    ncclKernelPlanner                        │
│                    rmaTaskQueues[ctx]                       │
└────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│              scheduleRmaTasksToPlan                         │
│              ┌──────────────┬──────────────┐                │
│              │  CE Tasks    │ Proxy Tasks  │                │
│              └──────────────┴──────────────┘                │
└────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│    ncclRmaPutCe         │    │    ncclRmaPutProxy      │
│    ncclRmaWaitSignalCe  │    │    ncclRmaWaitSignalProxy│
└─────────────────────────┘    └─────────────────────────┘
              │                              │
              ▼                              ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│    cudaMemcpyAsync      │    │    GIN iput/iputSignal  │
│    cuStreamBatchMemOp   │    │    Progress Thread      │
└─────────────────────────┘    └─────────────────────────┘
```

---

## 3. 核心数据结构

### 3.1 ncclTaskRma - RMA 任务结构

```cpp
// comm.h
struct ncclTaskRma {
  struct ncclTaskRma* next;      // 链表指针
  ncclFunc_t func;               // 操作类型: ncclFuncPutSignal/Signal/WaitSignal
  int ctx;                       // 上下文 ID
  size_t count;                  // 元素数量
  ncclDataType_t datatype;       // 数据类型
  size_t bytes;                  // 字节数

  // 源缓冲区信息
  void const* srcBuff;           // 源缓冲区指针
  size_t srcWinOffset;           // 源窗口偏移
  struct ncclDevrWindow* srcWinHost; // 源窗口句柄

  // 目标信息
  int peer;                      // 目标 rank
  size_t peerWinOffset;          // 目标窗口偏移
  struct ncclDevrWindow* peerWinHost; // 目标窗口句柄

  // Signal 操作
  ncclSignalMode_t signalMode;   // NCCL_SIGNAL_NONE 或 NCCL_SIGNAL
  int* peers;                    // WaitSignal 的目标 rank 数组
  int* nsignals;                 // 每个 rank 的信号数量
  int npeers;                    // 目标 rank 数量

  // Profiler 支持
  int eActivationMask;
  void* groupApiEventHandle;
  void* rmaApiEventHandle;
  void* eventHandle;
  uint8_t nChannels;
};
```

### 3.2 ncclRmaArgs - RMA 执行参数

```cpp
// rma.h
struct ncclRmaArgs {
  int ctx;                       // 上下文 ID
  ncclFunc_t func;               // 操作类型
  int nRmaTasks;                 // 总任务数
  int nRmaTasksProxy;            // Proxy 任务数
  int nRmaTasksCe;               // CE 任务数
};
```

### 3.3 ncclRmaState - RMA 状态

```cpp
// rma.h
struct ncclRmaState {
  struct ncclRmaProxyState rmaProxyState;  // Proxy 状态
  struct ncclRmaCeState rmaCeState;        // CE 状态
};
```

### 3.4 ncclRmaProxyState - Proxy 状态

```cpp
// rma_proxy.h
struct ncclRmaProxyState {
  struct ncclComm *comm;
  ncclGin_t* ncclGin;            // GIN 插件接口
  void* ginInstance;             // GIN 实例
  bool connected;                // 连接状态
  int ginType;                   // GIN 类型

  // 物理 GIN 通信器上下文
  int ginCommCount;
  void* ginComms[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetProperties_t props[NCCL_GIN_MAX_CONNECTIONS];

  // 虚拟 RMA Proxy 上下文
  int rmaProxyCtxCount;
  void** rmaProxyCtxs;
  ncclNetDeviceHandle_t** rmaProxyDevHandles;

  // 进度线程
  int needsProxyProgress;
  int ginProgress;
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cond;
  ncclResult_t asyncResult;
};
```

### 3.5 ncclRmaProxyCtx - Proxy 上下文

```cpp
// rma_proxy.h
struct ncclRmaProxyCtx {
  struct ncclComm *comm;

  // GIN 上下文
  void *ginCollComm;
  ncclNetDeviceHandle_t *devHandle;
  ncclNetProperties_t props;

  // 无锁环形缓冲区
  size_t queueSize;
  struct ncclRmaProxyDesc** pendingQueues;  // 待处理描述符队列
  uint32_t* pis;                            // 生产者索引
  uint32_t* cis;                            // 消费者索引

  // 进行中队列
  struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>* rmaProxyInProgressQueues;

  // 序列号和计数器
  uint64_t* opSeqs;      // 操作序列号
  uint64_t* opSeqsDev;   // 设备端操作序列号
  uint64_t* readySeqs;   // 就绪序列号
  uint64_t* readySeqsDev;
  uint64_t* doneSeqs;    // 完成序列号
  uint64_t* doneSeqsDev;

  // Signal 内存
  CUmemGenericAllocationHandle signalsCumemhandle;
  void *signalsMhandle;
  void *signalsGinHandle;
  uint64_t *signalsDev;
  uint64_t* signalsHost;
};
```

### 3.6 ncclRmaProxyDesc - Proxy 操作描述符

```cpp
// rma_proxy.h
struct ncclRmaProxyDesc {
  struct ncclRmaProxyDesc *next;

  // 网络操作描述符
  uint64_t srcOff;
  void *srcHandle;
  uint64_t dstOff;
  void *dstHandle;
  size_t size;
  int targetRank;
  ncclRmaSignal_t signal;

  // 序列号
  uint64_t seq;

  // 状态
  ncclRmaDescState_t rmaDescState;

  // 请求句柄
  void * request;
};
```

### 3.7 ncclRmaCeState - CE 状态

```cpp
// rma_ce.h
struct ncclRmaCeState {
  bool initialized;
  int rmaCeCtxCount;
  void** rmaCeCtxs;
  cudaStream_t ceStream;         // CE 专用流
  cudaEvent_t ceEvent;           // 同步事件
};
```

### 3.8 ncclRmaCeCtx - CE 上下文

```cpp
// rma_ce.h
struct ncclRmaCeCtx {
  struct ncclComm *comm;

  // Signal 操作序列号
  uint64_t* signalOpSeqs;

  // Signal 内存布局
  // [0..nRanks-1]: 每 rank 独立信号 (8 字节/rank)
  // [nRanks]: 聚合信号计数器 (8 字节)
  struct ncclDevrWindow* signalsWin;
  uint64_t *signalsDev;
  uint64_t* signalsHost;
};
```

### 3.9 ncclWaitSignalDesc_t - WaitSignal 描述符

```cpp
// nccl.h.in
typedef struct {
  int opCnt;    // 等待的信号操作数量
  int peer;     // 目标 rank
  int sigIdx;   // 信号索引 (当前必须为 0)
  int ctx;      // 上下文 ID (当前必须为 0)
} ncclWaitSignalDesc_t;
```

---

## 4. API 接口详解

### 4.1 ncclCommWindowRegister

```cpp
ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buff, 
                                    size_t size, ncclWindow_t* win, 
                                    int winFlags);
```

**功能**：注册内存窗口，使其可用于 RMA 操作。

**参数**：
- `comm`: NCCL 通信器
- `buff`: 缓冲区指针
- `size`: 缓冲区大小
- `win`: 输出的窗口句柄
- `winFlags`: 窗口标志
  - `NCCL_WIN_DEFAULT`: 默认
  - `NCCL_WIN_COLL_SYMMETRIC`: 对称内存
  - `NCCL_WIN_STRICT_ORDERING`: 严格顺序

**工作原理**：
1. 分配 GPU 内存
2. 注册为对称内存窗口
3. 交换所有 rank 的窗口信息
4. 返回窗口句柄

### 4.2 ncclCommWindowDeregister

```cpp
ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win);
```

**功能**：注销内存窗口。

### 4.3 ncclPutSignal

```cpp
ncclResult_t ncclPutSignal(const void* localbuff, size_t count, 
                           ncclDataType_t datatype, int peer, 
                           ncclWindow_t peerWin, size_t peerWinOffset,
                           int sigIdx, int ctx, unsigned int flags, 
                           ncclComm_t comm, cudaStream_t stream);
```

**功能**：将本地数据写入远程内存窗口，并发送信号通知目标 rank。

**参数**：
- `localbuff`: 本地源缓冲区
- `count`: 元素数量
- `datatype`: 数据类型
- `peer`: 目标 rank
- `peerWin`: 目标 rank 的内存窗口
- `peerWinOffset`: 目标窗口偏移
- `sigIdx`: 信号索引 (当前必须为 0)
- `ctx`: 上下文 ID (当前必须为 0)
- `flags`: 保留参数 (必须为 0)
- `comm`: NCCL 通信器
- `stream`: CUDA 流

**工作流程**：
1. 验证参数
2. 查找源缓冲区窗口
3. 创建 RMA 任务
4. 入队到 planner
5. 在 groupEnd 时执行

### 4.4 ncclSignal

```cpp
ncclResult_t ncclSignal(int peer, int sigIdx, int ctx, 
                        unsigned int flags, ncclComm_t comm, 
                        cudaStream_t stream);
```

**功能**：发送信号到目标 rank，不传输数据。

**参数**：
- `peer`: 目标 rank
- `sigIdx`: 信号索引 (当前必须为 0)
- `ctx`: 上下文 ID (当前必须为 0)
- `flags`: 保留参数 (必须为 0)

### 4.5 ncclWaitSignal

```cpp
ncclResult_t ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t* signalDescs, 
                            ncclComm_t comm, cudaStream_t stream);
```

**功能**：等待来自多个 rank 的信号。

**参数**：
- `nDesc`: 描述符数量
- `signalDescs`: 描述符数组，每个描述符指定：
  - `opCnt`: 等待的信号数量
  - `peer`: 来源 rank
  - `sigIdx`: 信号索引
  - `ctx`: 上下文 ID

**使用示例**：
```cpp
// 等待来自 rank 0 的 2 个信号和来自 rank 1 的 3 个信号
ncclWaitSignalDesc_t descs[2];
descs[0] = {2, 0, 0, 0};  // 等待 rank 0 的 2 个信号
descs[1] = {3, 1, 0, 0};  // 等待 rank 1 的 3 个信号
ncclWaitSignal(2, descs, comm, stream);
```

---

## 5. 关键函数分析

### 5.1 rmaTaskAppend

```cpp
// enqueue.cc
ncclResult_t rmaTaskAppend(struct ncclComm* comm, struct ncclInfo* info) {
    struct ncclKernelPlanner* planner = &comm->planner;

    // 1. 参数验证
    if (info->ctx != 0 || info->sigIdx != 0 || info->flags != 0) {
        return ncclInvalidArgument;
    }

    // 2. 初始化窗口指针
    struct ncclDevrWindow* peerWinHost = NULL;
    struct ncclDevrWindow* srcWinHost = NULL;
    
    if (info->coll == ncclFuncPutSignal) {
        // 验证并获取窗口
        ncclShadowPoolToHost(&comm->devrState.shadows, info->peerWin, &peerWinDevHost);
        ncclDevrFindWindow(comm, srcBuff, &srcWinHost);
    }

    // 3. 检查是否需要初始化 RMA CE
    if (!comm->rmaState.rmaCeState.initialized) {
        // 创建初始化任务
        ncclIntruQueueEnqueue(&comm->rmaCeInitTaskQueue, ceTask);
    }

    // 4. 创建任务
    if (info->coll == ncclFuncWaitSignal) {
        // WaitSignal 任务
        struct ncclTaskRma* t = ncclMemoryPoolAlloc<struct ncclTaskRma>(...);
        t->func = ncclFuncWaitSignal;
        t->npeers = info->nDesc;
        t->peers = ncclMemoryStackAlloc<int>(&comm->memScoped, info->nDesc);
        t->nsignals = ncclMemoryStackAlloc<int>(&comm->memScoped, info->nDesc);
        for (int i = 0; i < info->nDesc; i++) {
            t->peers[i] = info->signalDescs[i].peer;
            t->nsignals[i] = info->signalDescs[i].opCnt;
        }
        ncclIntruQueueEnqueue(&planner->rmaTaskQueues[t->ctx], t);
    }
    else if (info->coll == ncclFuncPutSignal || info->coll == ncclFuncSignal) {
        // PutSignal/Signal 任务
        // 大消息分块处理 (1GB chunks)
        size_t chunkSize = 1ULL << 30;
        int numChunks = (totalBytes + chunkSize - 1) / chunkSize;
        
        for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
            struct ncclTaskRma* t = ncclMemoryPoolAlloc<struct ncclTaskRma>(...);
            // 只有最后一个 chunk 发送信号
            t->signalMode = (chunkIdx == numChunks - 1) ? NCCL_SIGNAL : NCCL_SIGNAL_NONE;
            ncclIntruQueueEnqueue(&planner->rmaTaskQueues[t->ctx], t);
        }
    }
}
```

**功能**：创建 RMA 任务并入队。

**关键点**：
- 大消息自动分块 (1GB)
- 只有最后一个分块发送信号
- WaitSignal 支持批量等待多个 rank

### 5.2 scheduleRmaTasksToPlan

```cpp
// rma.cc
ncclResult_t scheduleRmaTasksToPlan(struct ncclComm* comm, 
                                    struct ncclKernelPlan* plan) {
    struct ncclKernelPlanner* planner = &comm->planner;

    // 1. 查找第一个非空上下文队列
    int ctx = -1;
    for (int i = 0; i < comm->config.numRmaCtx; i++) {
        if (!ncclIntruQueueEmpty(&planner->rmaTaskQueues[i])) {
            ctx = i;
            break;
        }
    }
    if (ctx == -1) return ncclSuccess;

    // 2. 获取第一个任务
    struct ncclTaskRma* firstTask = ncclIntruQueueDequeue(ctxQueue);

    // 3. 初始化 plan
    plan->isRma = true;
    plan->rmaArgs->ctx = ctx;
    plan->rmaArgs->func = firstTask->func;

    // 4. WaitSignal 任务处理
    if (firstTask->func == ncclFuncWaitSignal) {
        // 根据 LSA 可达性分离 CE 和 Proxy 任务
        for (int i = 0; i < firstTask->npeers; i++) {
            int peerRank = firstTask->peers[i];
            bool lsaAccessible = isLsaAccessible(comm, peerRank);
            
            if (lsaAccessible) {
                // CE 路径
                peersCe[npeersCe] = peerRank;
                nsignalsCe[npeersCe] = firstTask->nsignals[i];
                npeersCe++;
            } else {
                // Proxy 路径
                peersProxy[npeersProxy] = peerRank;
                nsignalsProxy[npeersProxy] = firstTask->nsignals[i];
                npeersProxy++;
            }
        }
    }
    // 5. Put/Signal 任务处理
    else {
        bool lsaAccessible = isLsaAccessible(comm, firstTask->peer);
        
        if (lsaAccessible) {
            ncclIntruQueueEnqueue(&plan->rmaTaskQueueCe, firstTask);
        } else {
            ncclIntruQueueEnqueue(&plan->rmaTaskQueueProxy, firstTask);
        }

        // 批处理连续任务
        while (!ncclIntruQueueEmpty(ctxQueue)) {
            struct ncclTaskRma* task = ncclIntruQueueHead(ctxQueue);
            if (!canBatchRmaTasks(firstTask, task)) break;
            
            ncclIntruQueueDequeue(ctxQueue);
            // 根据 LSA 可达性分发
            ...
        }
    }
}
```

**功能**：调度 RMA 任务到执行计划，分离 CE 和 Proxy 路径。

**关键点**：
- 根据 LSA (Local Symmetric Access) 可达性选择路径
- 支持任务批处理
- WaitSignal 任务按 peer 分离

### 5.3 ncclLaunchRma

```cpp
// rma.cc
ncclResult_t ncclLaunchRma(struct ncclComm* comm, 
                           struct ncclKernelPlan* plan) {
    cudaStream_t stream = comm->planner.streams->stream;

    switch (plan->rmaArgs->func) {
        case ncclFuncPutSignal:
        case ncclFuncSignal:
            NCCLCHECK(ncclRmaPut(comm, plan, stream));
            break;
        case ncclFuncWaitSignal:
            NCCLCHECK(ncclRmaWaitSignal(comm, plan, stream));
            break;
    }
}
```

### 5.4 ncclRmaPut

```cpp
// rma.cc
ncclResult_t ncclRmaPut(struct ncclComm* comm, struct ncclKernelPlan* plan, 
                        cudaStream_t stream) {
    // 如果同时有 CE 和 Proxy 任务，并行执行
    if (plan->rmaArgs->nRmaTasksProxy > 0 && plan->rmaArgs->nRmaTasksCe > 0) {
        cudaStream_t ceStream = comm->rmaState.rmaCeState.ceStream;
        cudaEvent_t ceEvent = comm->rmaState.rmaCeState.ceEvent;

        // 建立依赖关系
        cudaEventRecord(ceEvent, stream);
        cudaStreamWaitEvent(ceStream, ceEvent, 0);

        // 并行执行
        ncclRmaPutProxy(comm, plan, stream);
        ncclRmaPutCe(comm, plan, ceStream);

        // 同步流
        cudaEventRecord(ceEvent, ceStream);
        cudaStreamWaitEvent(stream, ceEvent, 0);
    }
    else if (plan->rmaArgs->nRmaTasksProxy > 0) {
        ncclRmaPutProxy(comm, plan, stream);
    }
    else if (plan->rmaArgs->nRmaTasksCe > 0) {
        ncclRmaPutCe(comm, plan, stream);
    }
}
```

**功能**：执行 PUT 操作，支持 CE 和 Proxy 路径并行。

### 5.5 ncclRmaPutCe

```cpp
// rma_ce.cc
ncclResult_t ncclRmaPutCe(struct ncclComm* comm, struct ncclKernelPlan* plan,
                          cudaStream_t stream) {
    struct ncclRmaCeCtx* ceCtx = ...;

    for (int i = 0; i < nRmaTasksCe; i++) {
        struct ncclTaskRma* task = ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);

        // 计算 LSA rank
        int peerLsaRank = task->peer % comm->devrState.lsaSize;
        size_t bytes = task->count * ncclTypeSize(task->datatype);

        if (bytes > 0) {
            // 获取目标缓冲区指针
            void* peerBuff;
            ncclDevrGetLsaRankPtr(comm, task->peerWinHost, task->peerWinOffset, 
                                  peerLsaRank, &peerBuff);

            // 执行数据拷贝
            cudaMemcpyAsync(peerBuff, task->srcBuff, bytes, 
                           cudaMemcpyDeviceToDevice, stream);
        }

        // 写入信号
        if (task->signalMode != NCCL_SIGNAL_NONE) {
            void* peerSignal;
            ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, 
                                  comm->rank * sizeof(uint64_t), 
                                  peerLsaRank, &peerSignal);

            // 增加序列号
            ceCtx->signalOpSeqs[task->peer]++;

            // 写入信号值
            cudaMemcpyAsync(peerSignal, &ceCtx->signalOpSeqs[task->peer], 
                           sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
        }
    }
}
```

**功能**：CE 路径的 PUT 实现，使用 cudaMemcpyAsync。

### 5.6 ncclRmaPutProxy

```cpp
// rma_proxy.cc
ncclResult_t ncclRmaPutProxy(struct ncclComm* comm, struct ncclKernelPlan* plan,
                             cudaStream_t stream) {
    struct ncclRmaProxyCtx* proxyCtx = ...;

    CUstreamBatchMemOpParams* batchParams = ...;

    for (int i = 0; i < nRmaTasksProxy; i++) {
        struct ncclTaskRma* task = ncclIntruQueueDequeue(&plan->rmaTaskQueueProxy);
        int peer = task->peer;

        // 检查环形缓冲区是否有空间
        while ((pi - ci) >= proxyCtx->queueSize) {
            // 等待进度线程处理
            std::this_thread::yield();
        }

        // 创建描述符
        struct ncclRmaProxyDesc *desc = ...;
        desc->srcOff = task->srcWinOffset;
        desc->srcHandle = ncclDevrGetRmaDevWin(task->srcWinHost, ctx);
        desc->dstOff = task->peerWinOffset;
        desc->dstHandle = ncclDevrGetRmaDevWin(task->peerWinHost, ctx);
        desc->size = task->count * ncclTypeSize(task->datatype);
        desc->targetRank = task->peer;
        desc->seq = proxyCtx->opSeqs[task->peer]++;

        // 设置信号
        if (task->signalMode == NCCL_SIGNAL) {
            desc->signal.op = NCCL_NET_SIGNAL_OP_ADD;
            desc->signal.offset = comm->rank * sizeof(uint64_t);
            desc->signal.signalMhandle = proxyCtx->signalsMhandle;
            desc->signal.val = 1;
        }

        // 准备批量内存操作
        batchParams[batchIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
        batchParams[batchIdx].writeValue.address = &proxyCtx->readySeqsDev[task->peer];
        batchParams[batchIdx].writeValue.value = desc->seq;

        batchParams[batchIdx+nRmaTasksProxy].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
        batchParams[batchIdx+nRmaTasksProxy].waitValue.address = &proxyCtx->doneSeqsDev[task->peer];
        batchParams[batchIdx+nRmaTasksProxy].waitValue.value = desc->seq;

        // 写入描述符到队列
        proxyCtx->pendingQueues[peer * proxyCtx->queueSize + idx] = desc;
        __atomic_store_n(&proxyCtx->pis[peer], pi + 1, __ATOMIC_RELEASE);
    }

    // 执行批量内存操作
    ncclCuStreamBatchMemOp(stream, 2*batchIdx, batchParams);
}
```

**功能**：Proxy 路径的 PUT 实现，使用 GIN 进行 RDMA 操作。

### 5.7 ncclRmaProxyProgress

```cpp
// rma_proxy.cc
ncclResult_t ncclRmaProxyProgress(ncclGin_t *ncclGin, void *rmaProxyCtx) {
    struct ncclRmaProxyCtx *ctx = (struct ncclRmaProxyCtx *)rmaProxyCtx;

    for (int i = 0; i < ctx->comm->nRanks; i++) {
        // 1. 轮询完成事件
        ncclRmaProxyPollCompletion(ncclGin, ctx, i);

        // 2. 轮询并发出就绪描述符
        ncclRmaProxyPollDesc(ncclGin, ctx, i);
    }
}
```

**功能**：Proxy 进度推进，处理网络操作。

---

## 6. 工作流程

### 6.1 初始化流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    ncclCommInitRank                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ncclDevrInitOnce                              │
│  - 初始化对称内存运行时                                          │
│  - 设置 LSA (Local Symmetric Access) 团队                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ncclRmaCeInit (延迟初始化)                    │
│  - 创建 CE 流和事件                                              │
│  - 分配 Signal 缓冲区                                            │
│  - 注册对称内存窗口                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ncclRmaProxyConnectOnce (延迟初始化)          │
│  - 初始化 GIN                                                    │
│  - 建立网络连接                                                  │
│  - 创建 Proxy 上下文                                             │
│  - 启动进度线程                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 PutSignal 完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    用户调用 ncclPutSignal                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ncclEnqueueCheck                              │
│  - 参数验证                                                      │
│  - 查找源窗口                                                    │
│  - 验证目标窗口                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    rmaTaskAppend                                 │
│  - 创建 ncclTaskRma                                              │
│  - 大消息分块 (1GB)                                              │
│  - 入队到 rmaTaskQueues[ctx]                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ncclGroupEnd                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    scheduleRmaTasksToPlan                        │
│  - 检查 LSA 可达性                                               │
│  - 分离 CE 和 Proxy 任务                                         │
│  - 批处理优化                                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ncclLaunchRma                                 │
│  - 根据 func 类型分发                                            │
│  - CE 和 Proxy 可并行执行                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│    CE Path              │     │    Proxy Path           │
│    ncclRmaPutCe         │     │    ncclRmaPutProxy      │
│    - cudaMemcpyAsync    │     │    - 创建描述符         │
│    - 写入 Signal        │     │    - 写入 readySeq      │
└─────────────────────────┘     │    - 等待 doneSeq       │
                                └─────────────────────────┘
                                              │
                                              ▼
                                ┌─────────────────────────┐
                                │    Progress Thread      │
                                │    ncclRmaProxyProgress │
                                │    - 轮询描述符         │
                                │    - 发起 RDMA PUT      │
                                │    - 更新 doneSeq       │
                                └─────────────────────────┘
```

### 6.3 WaitSignal 完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    用户调用 ncclWaitSignal                       │
│                    传入 ncclWaitSignalDesc_t 数组               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    rmaTaskAppend                                 │
│  - 创建单个 ncclTaskRma                                          │
│  - 包含所有 peers 和 nsignals                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    scheduleRmaTasksToPlan                        │
│  - 根据 LSA 可达性分离 peers                                     │
│  - 创建 CE 和 Proxy WaitSignal 任务                              │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│    CE Path              │     │    Proxy Path           │
│    ncclRmaWaitSignalCe  │     │    ncclRmaWaitSignalProxy│
│    - 批量 Wait 操作     │     │    - 批量 Wait 操作     │
│    - cuStreamBatchMemOp │     │    - cuStreamBatchMemOp │
└─────────────────────────┘     └─────────────────────────┘
```

---

## 7. 两种执行路径

### 7.1 CE (Copy Engine) 路径

**适用条件**：
- 目标 rank 在 LSA (Local Symmetric Access) 团队内
- 通常是通过 NVLink/NVSwitch 连接的同节点 GPU

**实现方式**：
- 使用 `cudaMemcpyAsync` 进行数据传输
- 使用 `cuStreamBatchMemOp` 进行信号操作
- 在专用 CE 流中执行，可与主流并行

**优势**：
- 低延迟 (NVLink 直连)
- 高带宽
- 无需网络协议栈

**代码路径**：
```
ncclRmaPutCe() -> cudaMemcpyAsync() -> 写入 Signal
ncclRmaWaitSignalCe() -> cuStreamBatchMemOp(WAIT_VALUE)
```

### 7.2 Proxy 路径

**适用条件**：
- 目标 rank 不在 LSA 团队内
- 跨节点通信

**实现方式**：
- 使用 GIN (GPU InfiniBand Network) 进行 RDMA 操作
- 通过 Proxy 进度线程处理网络操作
- 使用无锁环形缓冲区传递操作描述符

**优势**：
- 支持跨节点通信
- 利用 GPUDirect RDMA
- 异步执行，不阻塞 GPU

**代码路径**：
```
ncclRmaPutProxy() -> 创建描述符 -> 写入 readySeq
    -> Progress Thread -> GIN iput/iputSignal -> 更新 doneSeq
    -> GPU 等待 doneSeq
```

### 7.3 路径选择逻辑

```cpp
// rma.cc
static bool isLsaAccessible(struct ncclComm* comm, int rank) {
    for (int i = 0; i < comm->devrState.lsaSize; i++) {
        if (comm->devrState.lsaRankList[i] == rank) {
            return true;  // CE 路径
        }
    }
    return false;  // Proxy 路径
}
```

### 7.4 并行执行

当同一操作同时有 CE 和 Proxy 任务时，两者可以并行执行：

```cpp
// rma.cc
if (plan->rmaArgs->nRmaTasksProxy > 0 && plan->rmaArgs->nRmaTasksCe > 0) {
    // 使用专用 CE 流并行执行
    cudaStream_t ceStream = comm->rmaState.rmaCeState.ceStream;
    
    // CE 任务在 ceStream 执行
    ncclRmaPutCe(comm, plan, ceStream);
    
    // Proxy 任务在主流执行
    ncclRmaPutProxy(comm, plan, stream);
    
    // 同步两个流
    cudaEventRecord(ceEvent, ceStream);
    cudaStreamWaitEvent(stream, ceEvent, 0);
}
```

---

## 8. Signal 机制详解

### 8.1 Signal 内存布局

每个 RMA 上下文分配一个 Signal 缓冲区，布局如下：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Signal Buffer Layout                         │
├─────────────────────────────────────────────────────────────────┤
│ Offset 0x00:     Signal for rank 0  (8 bytes)                   │
│ Offset 0x08:     Signal for rank 1  (8 bytes)                   │
│ ...                                                              │
│ Offset (nRanks-1)*8: Signal for rank nRanks-1 (8 bytes)         │
│ Offset nRanks*8: Aggregate signal counter (8 bytes)             │
└─────────────────────────────────────────────────────────────────┘
Total size: (nRanks + 1) * 8 bytes
```

### 8.2 Signal 操作类型

```cpp
// rma.h
typedef enum {
  NCCL_SIGNAL_NONE = 0,        // 无信号
  NCCL_SIGNAL = 1              // 发送信号
} ncclSignalMode_t;
```

### 8.3 CE 路径 Signal 实现

```cpp
// rma_ce.cc
// 写入信号
if (task->signalMode != NCCL_SIGNAL_NONE) {
    // 获取目标 rank 的信号位置
    void* peerSignal;
    ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, 
                          comm->rank * sizeof(uint64_t), 
                          peerLsaRank, &peerSignal);

    // 增加操作序列号
    ceCtx->signalOpSeqs[task->peer]++;

    // 写入绝对序列号
    cudaMemcpyAsync(peerSignal, &ceCtx->signalOpSeqs[task->peer], 
                   sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
}

// 等待信号
for (int i = 0; i < task->npeers; i++) {
    int peerRank = task->peers[i];
    uint64_t waitValue = ceCtx->signalsHost[peerRank] + task->nsignals[i];
    ceCtx->signalsHost[peerRank] = waitValue;

    batchParams[opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
    batchParams[opIdx].waitValue.address = &ceCtx->signalsDev[peerRank];
    batchParams[opIdx].waitValue.value64 = waitValue;
    batchParams[opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
}
```

### 8.4 Proxy 路径 Signal 实现

```cpp
// rma_proxy.cc
// 设置信号描述符
if (task->signalMode == NCCL_SIGNAL) {
    desc->signal.op = NCCL_NET_SIGNAL_OP_ADD;
    desc->signal.offset = comm->rank * sizeof(uint64_t);
    desc->signal.signalMhandle = proxyCtx->signalsMhandle;
    desc->signal.val = 1;
}

// Progress Thread 发起带 Signal 的 PUT
if (pendingDesc->signal.op == 0) {
    ncclGin->iput(...);
} else {
    ncclGin->iputSignal(..., pendingDesc->signal.offset, 
                        pendingDesc->signal.signalMhandle,
                        pendingDesc->signal.val, pendingDesc->signal.op, ...);
}
```

### 8.5 序列号机制

Proxy 路径使用三级序列号进行同步：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sequence Number Flow                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPU                     Proxy Thread                Network     │
│  ───                     ────────────                ───────     │
│                                                                  │
│  1. opSeq++ (分配序列号)                                         │
│                                                                  │
│  2. 创建 Desc (包含 seq)                                         │
│                                                                  │
│  3. 写入 readySeq = seq ──────> 轮询 readySeq                    │
│                                   │                              │
│                                   ▼                              │
│                              发起网络操作 ──────────────> RDMA    │
│                                   │                              │
│                                   ▼                              │
│                              等待完成                            │
│                                   │                              │
│                                   ▼                              │
│                              doneSeq = seq                       │
│                                                                  │
│  4. 等待 doneSeq >= seq <─────── 更新 doneSeq                    │
│                                                                  │
│  5. 操作完成                                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 与 GIN 的集成

### 9.1 GIN 初始化

```cpp
// rma_proxy.cc
ncclResult_t ncclRmaProxyConnectOnce(struct ncclComm* comm) {
    // 1. 初始化 GIN
    rmaProxyState->ncclGin->init(&rmaProxyState->ginInstance, comm->commHash, ...);

    // 2. 获取设备数量
    rmaProxyState->ncclGin->devices(&ndev);

    // 3. 建立连接
    for (int n = 0; n < ginCommCount; n++) {
        rmaProxyState->ncclGin->listen(...);
        bootstrapAllGather(...);
        rmaProxyState->ncclGin->connect(...);
    }

    // 4. 创建虚拟 RMA Proxy 上下文
    for (int n = 0; n < rmaProxyCtxCount; n++) {
        ncclRmaProxyCreateContext(comm, ginComms[ginCommIdx], ...);
    }

    // 5. 启动进度线程
    if (needsProxyProgress) {
        rmaProxyState->thread = std::thread(ncclRmaProxyProgressThread, ...);
    }
}
```

### 9.2 内存注册

```cpp
// rma_proxy.cc
ncclResult_t ncclRmaProxyRegister(struct ncclComm* comm, void* address, 
                                  size_t size, ...) {
    for (int n = 0; n < rmaProxyState->ginCommCount; n++) {
        // 尝试 DMA-BUF 注册
        if (ncclParamDmaBufEnable() && (props.ptrSupport & NCCL_PTR_DMABUF)) {
            getDmaBufFd(address, size, &dmabufFd);
            ginComm->regMrSymDmaBuf(..., dmabufFd, ...);
        }
        // 回退到普通注册
        ginComm->regMrSym(...);
    }
}
```

### 9.3 网络操作

```cpp
// rma_proxy.cc - Progress Thread
ncclResult_t ncclRmaProxyPollDesc(ncclGin_t *ncclGin, 
                                  struct ncclRmaProxyCtx *ctx, int peer) {
    // 检查描述符是否就绪
    uint64_t readySeq = __atomic_load_n(&ctx->readySeqs[peer], __ATOMIC_ACQUIRE);
    if (readySeq >= pendingDesc->seq) {
        // 发起网络操作
        if (pendingDesc->signal.op == 0) {
            ncclGin->iput(ctx->ginCollComm, 
                         pendingDesc->srcOff, pendingDesc->srcHandle, 
                         pendingDesc->size, pendingDesc->dstOff, 
                         pendingDesc->dstHandle, pendingDesc->targetRank, 
                         0, &pendingDesc->request);
        } else {
            ncclGin->iputSignal(ctx->ginCollComm, 
                               pendingDesc->srcOff, pendingDesc->srcHandle, 
                               pendingDesc->size, pendingDesc->dstOff, 
                               pendingDesc->dstHandle, pendingDesc->targetRank, 
                               pendingDesc->signal.offset, 
                               pendingDesc->signal.signalMhandle,
                               pendingDesc->signal.val, pendingDesc->signal.op, 
                               0, &pendingDesc->request);
        }
    }
}
```

---

## 10. 适用场景与收益

### 10.1 适用场景

| 场景 | 描述 | 推荐路径 |
|------|------|----------|
| **参数服务器** | Worker 向 Server 推送/拉取参数 | Proxy |
| **模型并行** | 层间数据传输 | CE (同节点) / Proxy (跨节点) |
| **流水线并行** | Stage 间激活值传输 | CE / Proxy |
| **分布式推理** | 请求分发和结果聚合 | CE / Proxy |
| **异步训练** | 计算与通信重叠 | Proxy |

### 10.2 性能收益

#### 10.2.1 与传统 Send/Recv 对比

| 指标 | Send/Recv | RMA Put/WaitSignal |
|------|-----------|-------------------|
| 同步开销 | 高 (两侧同步) | 低 (一侧操作) |
| 延迟 | 较高 | 较低 |
| 目标参与 | 需要 | 不需要 |
| 编程复杂度 | 简单 | 中等 |

#### 10.2.2 典型性能数据

| 操作 | 带宽 | 延迟 |
|------|------|------|
| CE PutSignal (NVLink) | > 100 GB/s | < 5 μs |
| Proxy PutSignal (IB) | > 50 GB/s | < 20 μs |
| Signal (无数据) | N/A | < 2 μs |

### 10.3 使用建议

1. **优先使用对称内存**：注册窗口时使用 `NCCL_WIN_COLL_SYMMETRIC` 标志
2. **批量操作**：使用 `ncclGroupStart/End` 批量提交多个操作
3. **合理分块**：大消息自动分块，但可考虑手动控制
4. **Signal 优化**：只在必要时发送 Signal，减少同步开销

### 10.4 典型使用模式

#### 模式 1: 生产者-消费者

```cpp
// 生产者
ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC);
ncclPutSignal(buffer, count, datatype, consumer_rank, win, 0, 0, 0, 0, comm, stream);

// 消费者
ncclWaitSignalDesc_t desc = {1, producer_rank, 0, 0};
ncclWaitSignal(1, &desc, comm, stream);
// 处理数据
```

#### 模式 2: 多对一聚合

```cpp
// 多个生产者
ncclPutSignal(buffer, count, datatype, aggregator_rank, win, offset, 0, 0, 0, comm, stream);

// 聚合者等待所有
ncclWaitSignalDesc_t descs[nProducers];
for (int i = 0; i < nProducers; i++) {
    descs[i] = {1, producer_ranks[i], 0, 0};
}
ncclWaitSignal(nProducers, descs, comm, stream);
```

#### 模式 3: 流水线传输

```cpp
// Stage 1
ncclPutSignal(data1, count, datatype, next_stage, win, 0, 0, 0, 0, comm, stream1);
ncclSignal(next_stage, 0, 0, 0, comm, stream1);  // 额外通知

// Stage 2
ncclWaitSignalDesc_t desc = {2, prev_stage, 0, 0};  // 等待 2 个信号
ncclWaitSignal(1, &desc, comm, stream2);
```

---

## 11. 配置参数

### 11.1 RMA 相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NCCL_RMA_PROXY_QUEUE_SIZE` | -1 | Proxy 队列大小 (-1: 自动) |
| `NCCL_RMA_PROXY_DUMP_SIGNAL` | -1 | 调试信号 (用于 dump 状态) |

### 11.2 GIN 相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NCCL_GIN_ENABLE` | 1 | 启用 GIN |
| `NCCL_GIN_TYPE` | -1 | GIN 类型 (-1: 自动) |
| `NCCL_GIN_NCONNECTIONS` | -2 | 连接数 |
| `NCCL_GIN_NCONTEXTS` | -1 | 上下文数 |

### 11.3 对称内存参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NCCL_CUMEM_ENABLE` | - | 启用 cuMem |
| `NCCL_DMABUF_ENABLE` | - | 启用 DMA-BUF |

---

## 12. 代码位置索引

| 功能 | 文件 | 函数 |
|------|------|------|
| **API 入口** | `collectives.cc` | `ncclPutSignal`, `ncclSignal`, `ncclWaitSignal` |
| **任务创建** | `enqueue.cc` | `rmaTaskAppend` |
| **任务调度** | `rma/rma.cc` | `scheduleRmaTasksToPlan` |
| **RMA 启动** | `rma/rma.cc` | `ncclLaunchRma`, `ncclRmaPut`, `ncclRmaWaitSignal` |
| **CE PUT** | `rma/rma_ce.cc` | `ncclRmaPutCe` |
| **CE WaitSignal** | `rma/rma_ce.cc` | `ncclRmaWaitSignalCe` |
| **CE 初始化** | `rma/rma_ce.cc` | `ncclRmaCeInit`, `ncclRmaCeFinalize` |
| **Proxy PUT** | `rma/rma_proxy.cc` | `ncclRmaPutProxy` |
| **Proxy WaitSignal** | `rma/rma_proxy.cc` | `ncclRmaWaitSignalProxy` |
| **Proxy 连接** | `rma/rma_proxy.cc` | `ncclRmaProxyConnectOnce`, `ncclRmaProxyFinalize` |
| **Proxy 进度** | `rma/rma_proxy.cc` | `ncclRmaProxyProgress`, `ncclRmaProxyProgressThread` |
| **内存注册** | `rma/rma_proxy.cc` | `ncclRmaProxyRegister`, `ncclRmaProxyDeregister` |
| **窗口注册** | `register.cc` | `ncclCommWindowRegister`, `ncclCommWindowDeregister` |
| **对称内存** | `dev_runtime.cc` | `ncclDevrInitOnce`, `ncclDevrWindowRegisterInGroup` |
| **LSA 访问** | `dev_runtime.cc` | `ncclDevrGetLsaRankPtr`, `isLsaAccessible` |

---

## 附录 A: 调试技巧

### A.1 启用详细日志

```bash
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,COLL,NET
```

### A.2 Dump Proxy 状态

```bash
# 发送信号触发 dump
export NCCL_RMA_PROXY_DUMP_SIGNAL=<signal_number>
kill -<signal_number> <pid>
```

### A.3 检查 LSA 配置

```bash
NCCL_DEBUG=INFO ./your_app 2>&1 | grep "lsa"
```

---

## 附录 B: 错误排查

### B.1 窗口注册失败

**症状**：`ncclPutSignal: srcWinHost is not in a valid symmetric window`

**排查**：
1. 确认使用 `NCCL_WIN_COLL_SYMMETRIC` 标志
2. 检查对称内存支持
3. 验证所有 rank 都注册了窗口

### B.2 Signal 未收到

**症状**：`ncclWaitSignal` 永久阻塞

**排查**：
1. 确认 Signal 和 WaitSignal 使用相同的 ctx 和 sigIdx
2. 检查 opCnt 是否匹配
3. 验证 peer rank 正确

### B.3 Proxy 连接失败

**症状**：`RMA proxy is not connected`

**排查**：
1. 检查 GIN 是否启用
2. 验证网络配置
3. 检查 GPUDirect RDMA 支持