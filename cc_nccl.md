# NCCL 源码深度分析文档

## 目录

1. [概述](#1-概述)
2. [核心架构](#2-核心架构)
3. [初始化流程](#3-初始化流程)
4. [通信器管理](#4-通信器管理)
5. [Channel机制](#5-channel机制)
6. [集合通信实现](#6-集合通信实现)
7. [传输层](#7-传输层)
8. [代理机制](#8-代理机制)
9. [拓扑发现与图计算](#9-拓扑发现与图计算)
10. [设备端CUDA内核](#10-设备端cuda内核)
11. [协议实现](#11-协议实现)
12. [算法选择与调优](#12-算法选择与调优)
13. [插件系统](#13-插件系统)
14. [内存管理](#14-内存管理)
15. [缓冲区注册机制](#15-缓冲区注册机制)
16. [性能分析与Profiling](#16-性能分析与profiling)
17. [Bootstrap机制](#17-bootstrap机制)
18. [组操作](#18-组操作)
19. [错误处理与容错](#19-错误处理与容错)
20. [环境变量配置](#20-环境变量配置)
21. [MNNVL多节点NVLink](#21-mnnvl多节点nvlink)
22. [GIN全局互联网络](#22-gin全局互联网络)
23. [CollNet集合网络](#23-collnet集合网络)
24. [对称内存运行时](#24-对称内存运行时)
25. [RMA远程内存访问](#25-rma远程内存访问)
26. [CE Copy Engine](#26-ce-copy-engine)

---

## 1. 概述

### 1.1 NCCL简介

NCCL (NVIDIA Collective Communications Library) 是NVIDIA开发的高性能GPU集合通信库，实现了以下核心功能：

- **集合通信操作**: AllReduce, AllGather, ReduceScatter, Broadcast, Reduce, Send/Recv
- **多传输支持**: NVLink, PCIe P2P, Shared Memory, Network (InfiniBand/RoCE/TCP)
- **多协议支持**: LL (Low Latency), LL128, SIMPLE
- **多算法支持**: Tree, Ring, CollNet, NVLS, PAT

### 1.2 版本信息

当前版本: NCCL 2.29.7-1 (定义于 `src/makefiles/version.mk`)

### 1.3 核心设计理念

1. **零拷贝优化**: 尽可能使用直接内存访问
2. **流水线并行**: 多Channel并行处理数据
3. **拓扑感知**: 根据硬件拓扑选择最优通信路径
4. **可扩展性**: 支持单节点到多节点的透明扩展

---

## 2. 核心架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Application                            │
├─────────────────────────────────────────────────────────────────┤
│                      NCCL API Layer                              │
│   ncclAllReduce / ncclSend / ncclRecv / ncclGroupStart/End      │
├─────────────────────────────────────────────────────────────────┤
│                    Collective Layer                              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │   Enqueue   │  │   Planner   │  │   Launcher  │            │
│   └─────────────┘  └─────────────┘  └─────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│                    Transport Layer                               │
│   ┌──────┐  ┌──────┐  ┌──────┐  ┌─────────┐                    │
│   │  P2P │  │  SHM │  │  NET │  │ CollNet │                    │
│   └──────┘  └──────┘  └──────┘  └─────────┘                    │
├─────────────────────────────────────────────────────────────────┤
│                    Device Kernel Layer                           │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐     │
│   │ Prims_LL      │  │ Prims_LL128   │  │ Prims_Simple  │     │
│   └───────────────┘  └───────────────┘  └───────────────┘     │
├─────────────────────────────────────────────────────────────────┤
│                    Hardware Layer                                │
│   NVLink / PCIe / InfiniBand / RoCE / TCP                       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心数据结构

#### 2.2.1 ncclComm (通信器)

定义于 `src/include/comm.h`，是NCCL最核心的数据结构：

```cpp
struct ncclComm {
  // 魔数校验
  uint64_t startMagic;
  uint64_t endMagic;

  // 内存管理
  struct ncclMemoryStack memPermanent, memScoped;

  // 基本属性
  int rank;      // 当前rank
  int nRanks;    // 总rank数
  int cudaDev;   // CUDA设备索引
  int node;      // 节点索引
  int nNodes;    // 总节点数
  int localRank; // 本地rank
  int localRanks;// 本节点rank数

  // Channel数组
  struct ncclChannel channels[MAXCHANNELS];

  // 网络相关
  ncclNet_t* ncclNet;
  void* netContext;

  // 拓扑信息
  struct ncclTopoSystem* topo;
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS];

  // 代理状态
  struct ncclProxyState* proxyState;

  // 设备端通信器
  struct ncclKernelComm* devComm;

  // 内核规划器
  struct ncclKernelPlanner planner;
};
```

关键成员说明：
- `startMagic/endMagic`: 用于检测内存损坏
- `channels[]`: 通信通道数组，每个Channel负责一部分数据传输
- `topo`: 系统拓扑结构，用于路径选择
- `planner`: 负责规划和执行内核

#### 2.2.2 ncclChannel (通道)

```cpp
struct ncclChannel {
  int id;  // Channel索引

  // Peer连接
  struct ncclChannelPeer** peers;
  struct ncclDevChannelPeer** devPeers;

  // Ring拓扑
  struct ncclRing ring;
  int* devRingUserRanks;

  // Tree拓扑
  struct ncclTree tree;

  // CollNet拓扑
  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;

  // NVLS拓扑
  struct ncclNvls nvls;

  // 工作队列
  uint32_t workFifoProduced;
};
```

#### 2.2.3 ncclChannelPeer (对等连接)

```cpp
struct ncclChannelPeer {
  struct ncclConnector send[NCCL_MAX_CONNS];  // 发送连接器
  struct ncclConnector recv[NCCL_MAX_CONNS];  // 接收连接器
  int refCount;
};

struct ncclConnector {
  int connected;
  struct ncclProxyConnector proxyConn;
  struct ncclTransportComm* transportComm;
  void* transportResources;
  struct ncclConnInfo conn;
};

struct ncclConnInfo {
  char *buffs[NCCL_NUM_PROTOCOLS];  // 各协议缓冲区
  void* mhandles[NCCL_NUM_PROTOCOLS]; // 内存句柄
  uint64_t *tail;  // 尾指针
  uint64_t *head;  // 头指针
  int flags;       // 标志位
  int stepSize;    // 步进大小
  void **ptrExchange;  // 指针交换
  struct ncclConnFifo* connFifo;  // 连接FIFO
  uint64_t step;   // 当前步进
};
```

### 2.3 文件组织结构

```
nccl/
├── src/
│   ├── init.cc           # 初始化入口
│   ├── bootstrap.cc      # 引导机制
│   ├── group.cc          # 组操作
│   ├── collectives.cc    # 集合通信API
│   ├── enqueue.cc        # 操作入队
│   ├── channel.cc        # Channel管理
│   ├── proxy.cc          # 代理线程
│   ├── transport.cc      # 传输抽象层
│   ├── transport/
│   │   ├── p2p.cc       # P2P传输
│   │   ├── shm.cc       # 共享内存传输
│   │   ├── net.cc       # 网络传输
│   │   └── nvls.cc      # NVLink SHARP
│   ├── graph/
│   │   ├── topo.cc      # 拓扑发现
│   │   ├── search.cc    # 路径搜索
│   │   └── tuning.cc    # 算法调优
│   ├── device/
│   │   ├── generate.py  # 内核生成脚本
│   │   ├── primitives.h # 原语定义
│   │   ├── prims_ll.h   # LL协议
│   │   ├── prims_ll128.h# LL128协议
│   │   └── prims_simple.h # SIMPLE协议
│   ├── include/
│   │   ├── comm.h       # 通信器定义
│   │   ├── device.h     # 设备端定义
│   │   ├── nccl_net.h   # 网络插件接口
│   │   └── plugin/      # 插件接口
│   └── plugin/
│       ├── net/         # 网络插件
│       ├── tuner/       # 调优插件
│       └── profiler/    # 性能分析插件
├── bindings/
│   └── nccl4py/         # Python绑定
└── docs/examples/       # 示例代码
```

---

## 3. 初始化流程

### 3.1 初始化入口

`ncclCommInitRank` 是主要的初始化入口：

```cpp
ncclResult_t ncclCommInitRank(ncclComm_t* commptr, int nranks,
                              ncclUniqueId commId, int myRank) {
  // 1. 创建通信器对象
  ncclComm_t comm;
  NCCLCHECK(ncclCommCreate(&comm));

  // 2. 设置基本属性
  comm->nRanks = nranks;
  comm->rank = myRank;

  // 3. Bootstrap初始化
  NCCLCHECK(bootstrapInit(...));

  // 4. 拓扑发现
  NCCLCHECK(ncclTopoGetSystem(comm, &comm->topo, ...));

  // 5. 传输层初始化
  NCCLCHECK(ncclTransportInit(comm));

  // 6. Channel初始化
  NCCLCHECK(initChannels(comm));

  // 7. 连接建立
  NCCLCHECK(ncclConnectInit(comm));
}
```

### 3.2 Bootstrap机制

Bootstrap负责在通信器初始化期间协调各rank之间的信息交换：

```cpp
struct ncclBootstrapHandle {
  uint64_t magic;
  union ncclSocketAddress addr;
  int nRanks;
};
```

关键函数：
- `bootstrapCreateRoot()`: 创建根节点
- `bootstrapInit()`: 初始化bootstrap连接
- `bootstrapAllGather()`: 全局收集操作
- `bootstrapSend()/bootstrapRecv()`: 点对点通信

Bootstrap流程：
1. 根rank创建监听socket
2. 其他rank连接到根rank
3. 交换peer信息
4. 建立全连接mesh

### 3.3 拓扑发现

拓扑发现是NCCL优化的关键步骤：

```cpp
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm,
                               struct ncclTopoSystem** system) {
  // 1. 解析XML拓扑文件
  ncclTopoLoadSystem(xmlTopoFile, system);

  // 2. 检测GPU拓扑
  detectGpuTopology(system);

  // 3. 检测网络拓扑
  detectNetTopology(system);

  // 4. 计算路径
  ncclTopoComputePaths(system, comm);

  // 5. 排序系统
  ncclTopoSortSystem(system);
}
```

拓扑节点类型：
```cpp
#define GPU 0    // GPU节点
#define PCI 1    // PCI设备
#define NVS 2    // NVSwitch
#define CPU 3    // NUMA域
#define NIC 4    // 网卡
#define NET 5    // 网络节点
#define GIN 6    // Global Interconnect Network
```

路径类型：
```cpp
#define PATH_LOC 0   // 本地
#define PATH_NVL 1   // NVLink
#define PATH_NVB 2   // NVLink through bridge
#define PATH_C2C 3   // C2C
#define PATH_PIX 4   // PCIe单桥
#define PATH_PXB 5   // PCIe多桥
#define PATH_P2C 6   // GPU到NIC via C2C
#define PATH_PXN 7   // GPU to NIC via intermediate GPU
#define PATH_PHB 8   // PCIe Host Bridge
#define PATH_SYS 9   // 系统互联
#define PATH_NET 10  // 网络
```

---

## 4. 通信器管理

### 4.1 通信器生命周期

```
创建 → 初始化 → 活跃使用 → Finalize → 销毁
```

### 4.2 通信器创建

```cpp
ncclResult_t ncclCommCreate(ncclComm_t* commptr) {
  ncclComm_t comm;
  NCCLCHECK(ncclCalloc(&comm, 1));

  // 设置魔数
  comm->startMagic = NCCL_MAGIC;
  comm->endMagic = NCCL_MAGIC;

  // 初始化内存栈
  ncclMemoryStackInit(&comm->memPermanent);
  ncclMemoryStackInit(&comm->memScoped);

  // 初始化规划器
  plannerInit(&comm->planner);

  *commptr = comm;
}
```

### 4.3 共享资源

```cpp
struct ncclSharedResources {
  int refCount;
  struct ncclComm* owner;

  // Channel Peers
  struct ncclChannelPeer* peers[MAXCHANNELS];
  struct ncclDevChannelPeer* devPeers[MAXCHANNELS];

  // 操作计数器
  uint64_t p2pOpCount[MAXCHANNELS];
  uint64_t collOpCount;

  // 代理状态
  struct ncclProxyState* proxyState;

  // GIN状态
  struct ncclGinState ginState;
};
```

共享资源用于`ncclCommSplit`操作，子通信器可以共享父通信器的资源。

### 4.4 通信器销毁

```cpp
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == NULL) return ncclSuccess;

  // 1. 设置销毁标志
  comm->destroyFlag = 1;

  // 2. 停止代理
  ncclProxyStop(comm);

  // 3. 释放资源
  while (comm->destructorHead) {
    ncclDestructor* d = comm->destructorHead;
    comm->destructorHead = d->next;
    d->fn(d);
  }

  // 4. 释放内存
  ncclMemoryStackFree(&comm->memScoped);
  ncclMemoryStackFree(&comm->memPermanent);

  free(comm);
}
```

---

## 5. Channel机制

### 5.1 Channel概述

Channel是NCCL并行处理的核心机制。每个Channel:
- 维护独立的发送/接收连接
- 拥有独立的缓冲区
- 可以并行执行通信任务

```cpp
#define MAXCHANNELS 64  // 最大Channel数
```

### 5.2 Channel初始化

```cpp
ncclResult_t initChannel(struct ncclComm* comm, int channelId) {
  struct ncclChannel* channel = comm->channels + channelId;
  channel->id = channelId;

  // 分配peers数组
  NCCLCHECK(ncclCalloc(&channel->peers, comm->nRanks));
  NCCLCHECK(ncclCalloc(&channel->devPeers, comm->nRanks));

  // 初始化Ring
  channel->ring.prev = (comm->rank - 1 + comm->nRanks) % comm->nRanks;
  channel->ring.next = (comm->rank + 1) % comm->nRanks;
  channel->ring.index = comm->rank;

  return ncclSuccess;
}
```

### 5.3 Ring拓扑

```cpp
struct ncclRing {
  int prev;      // 前驱rank
  int next;      // 后继rank
  int* userRanks; // 用户rank映射
  int* rankToIndex; // rank到索引的映射
  int index;     // 当前rank在ring中的索引
};
```

Ring是集合通信的基础拓扑：
- AllReduce: 数据沿ring传递，每个节点执行reduce
- AllGather: 每个节点沿ring传递自己的数据块
- ReduceScatter: 类似AllReduce，但结果分散

### 5.4 Tree拓扑

```cpp
struct ncclTree {
  int depth;     // 树深度
  int up;        // 父节点
  int down[NCCL_MAX_TREE_ARITY]; // 子节点数组
};

#define NCCL_MAX_TREE_ARITY 3  // 最大分支因子
```

Tree拓扑用于：
- Reduce/Broadcast: 树形reduce/broadcast
- AllReduce Tree算法: 双树结构

### 5.5 NVLS拓扑

```cpp
struct ncclNvls {
  int out;           // 输出方向
  int nHeads;        // Head数量
  int headRank;      // Head rank索引
  int up[NCCL_MAX_NVLS_ARITY];   // 上行连接
  int down;          // 下行连接
  int treeUp;        // 树上行
  int treeDown[NCCL_MAX_NVLS_TREE_ARITY]; // 树下行
};

#define NCCL_MAX_NVLS_ARITY 32
#define NCCL_MAX_NVLS_TREE_ARITY 3
```

NVLS (NVLink SHARP) 利用NVSwitch进行硬件加速的reduce操作。

---

## 6. 集合通信实现

### 6.1 API层

集合通信API定义在 `src/collectives.cc`:

```cpp
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff,
                           size_t count, ncclDataType_t datatype,
                           ncclRedOp_t op, ncclComm_t comm,
                           cudaStream_t stream) {
  // 参数检查
  NCCLCHECK(ncclGroupStart());

  // 创建信息结构
  struct ncclInfo info = { ncclFuncAllReduce, ... };

  // 入队操作
  NCCLCHECK(ncclEnqueueKernel(&info));

  NCCLCHECK(ncclGroupEnd());
}
```

### 6.2 任务队列

```cpp
struct ncclTaskColl {
  struct ncclTaskColl* next;
  ncclFunc_t func;
  void const* sendbuff;
  void* recvbuff;
  size_t count;
  int root;
  ncclDataType_t datatype;
  ncclRedOp_t opHost;
  struct ncclDevRedOpFull opDev;

  // 调度参数
  int chunkSteps, sliceSteps;

  // 算法/协议选择
  int32_t algorithm:8, protocol:8;
  int32_t nMaxChannels:8;
  int32_t nWarps:8;

  // 性能分析
  uint32_t isCollnet:1, isNvls:1, isSymLast:1;
};
```

### 6.3 内核规划器

```cpp
struct ncclKernelPlanner {
  // 任务排序器
  struct ncclTaskCollSorter collSorter;

  // 各类任务队列
  struct ncclIntruQueue<ncclTaskP2p> sendQueue, recvQueue;
  struct ncclIntruQueue<ncclTaskColl> collTaskQueue;

  // WIP计划
  struct WipPlan {
    struct Channel {
      int workBytes;
      int nP2ps, nBcasts;
      struct ncclIntruQueue<ncclWorkBatchList> workBatchQueue;
      struct ncclIntruQueue<ncclProxyOp> proxyOpQueue;
    } channels[MAXCHANNELS];
  } wipPlan;

  // 已构建的计划队列
  struct ncclIntruQueue<ncclKernelPlan> planQueue;
};
```

### 6.4 内核计划

```cpp
struct ncclKernelPlan {
  struct ncclComm* comm;

  // 内核配置
  void* kernelFn;
  struct ncclDevKernelArgs* kernelArgs;
  int threadPerBlock;

  // Channel掩码
  uint64_t channelMask;

  // 工作队列
  struct ncclIntruQueue<ncclWorkList> workQueue;

  // 代理操作队列
  struct ncclIntruQueue<ncclProxyOp> proxyOpQueue;

  // 任务队列
  struct ncclIntruQueue<ncclTaskP2p> p2pTaskQueue;
  struct ncclIntruQueue<ncclTaskColl> collTaskQueue;
};
```

### 6.5 工作批处理

```cpp
struct alignas(16) ncclDevWorkBatch {
  uint32_t nextJump:14;    // 下一个批次的跳转
  uint32_t nextExtends:1;  // 是否扩展
  uint32_t workType:2;     // 工作类型
  uint32_t funcId:15;      // 函数ID

  uint32_t offsetBase;     // FIFO偏移基址
  uint64_t offsetBitset;   // 工作偏移位集
};

enum ncclDevWorkType: uint8_t {
  ncclDevWorkTypeP2p,      // 点对点
  ncclDevWorkTypeColl,     // 集合通信
  ncclDevWorkTypeCollReg,  // 带注册的集合通信
  ncclDevWorkTypeBcast,    // 广播
};
```

---

## 7. 传输层

### 7.1 传输抽象

```cpp
#define TRANSPORT_P2P 0      // P2P传输
#define TRANSPORT_SHM 1      // 共享内存
#define TRANSPORT_NET 2      // 网络
#define TRANSPORT_COLLNET 3  // 集合网络
#define TRANSPORT_PROFILER 4 // 性能分析

struct ncclTransport {
  const char name[8];
  ncclResult_t (*canConnect)(int*, struct ncclComm*,
                             struct ncclTopoGraph*,
                             struct ncclPeerInfo*,
                             struct ncclPeerInfo*);
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

struct ncclTransportComm {
  ncclResult_t (*setup)(...);
  ncclResult_t (*connect)(...);
  ncclResult_t (*free)(...);
  ncclResult_t (*proxySharedInit)(...);
  ncclResult_t (*proxySetup)(...);
  ncclResult_t (*proxyConnect)(...);
  ncclResult_t (*proxyFree)(...);
  ncclResult_t (*proxyProgress)(...);
};
```

### 7.2 P2P传输

P2P传输用于GPU之间的直接通信：

```cpp
// src/transport/p2p.cc

ncclResult_t p2pSendSetup(struct ncclComm* comm, ...) {
  // 1. 检查P2P能力
  cudaDeviceCanAccessPeer(&canAccess, dev1, dev2);

  // 2. 分享CUDA IPC handle
  cudaIpcGetMemHandle(&handle, buffer);

  // 3. 打开peer memory
  cudaIpcOpenMemHandle(&peerPtr, handle, ...);
}

ncclResult_t p2pSendProxyProgress(struct ncclProxyState* state,
                                  struct ncclProxyArgs* args) {
  // 直接内存拷贝
  cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
}
```

### 7.3 共享内存传输

用于同一节点上不同进程间的通信：

```cpp
// src/transport/shm.cc

ncclResult_t shmSendSetup(struct ncclComm* comm, ...) {
  // 1. 创建共享内存段
  shm_open(name, O_CREAT|O_RDWR, 0666);

  // 2. 映射到进程地址空间
  mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
}
```

### 7.4 网络传输

```cpp
// src/transport/net.cc

ncclResult_t netSendSetup(struct ncclComm* comm, ...) {
  // 使用网络插件
  comm->ncclNet->listen(...);
  comm->ncclNet->connect(...);
  comm->ncclNet->accept(...);
}

ncclResult_t netSendProxyProgress(struct ncclProxyState* state,
                                  struct ncclProxyArgs* args) {
  // RDMA发送
  comm->ncclNet->isend(...);
  comm->ncclNet->test(...);
}
```

### 7.5 NVLS传输

NVLink SHARP利用NVSwitch进行硬件加速：

```cpp
// src/transport/nvls.cc

ncclResult_t ncclNvlsInit(struct ncclComm* comm) {
  // 创建multicast对象
  CUmulticastObjectProp prop = {};
  prop.size = bufferSize;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

  cuMulticastCreate(&handle, &prop);

  // 添加参与者
  cuMulticastAddDevice(handle, dev);
}
```

---

## 8. 代理机制

### 8.1 代理概述

Proxy是NCCL的核心组件，负责：
- 管理网络连接
- 执行RDMA操作
- 处理GPU-Kernel与网络的协调

### 8.2 代理状态

```cpp
struct ncclProxyState {
  int refCount;
  struct ncclComm* comm;

  // 线程
  std::thread thread;       // 主进度线程
  std::thread threadUDS;    // Unix域socket线程

  // 网络资源
  ncclNet_t* ncclNet;
  ncclCollNet_t* ncclCollNet;
  void* netContext;
  void* collNetContext;

  // 通信资源
  struct ncclSocket* listenSock;
  struct ncclSocket* peerSocks;
  struct ncclProxyOps* proxyOps;

  // 进度状态
  struct ncclProxyProgressState progressState;
};
```

### 8.3 代理操作

```cpp
struct ncclProxyOp {
  struct ncclProxyConnection* connection;
  ssize_t nbytes;
  uint64_t opCount;
  int root;

  // 调度参数
  int nsteps;
  size_t chunkSize;
  size_t sliceSize;
  uint8_t sliceSteps;
  uint8_t chunkSteps;
  uint8_t channelId;

  // 类型和协议
  uint8_t dtype;
  uint8_t redOp;
  uint8_t coll;
  uint8_t pattern;
  uint8_t protocol;
  uint8_t algorithm;

  // 内存句柄
  void* sendMhandle;
  void* recvMhandle;
};
```

### 8.4 代理进度模式

```cpp
enum ncclPattern_t {
  ncclPatternRing,          // Ring模式
  ncclPatternRingTwice,     // 双Ring
  ncclPatternPipelineFrom,  // 流水线发送
  ncclPatternPipelineTo,    // 流水线接收
  ncclPatternTreeUp,        // 树上行
  ncclPatternTreeDown,      // 树下行
  ncclPatternTreeUpDown,    // 树双向
  ncclPatternCollnetChain,  // CollNet链式
  ncclPatternCollnetDirect, // CollNet直连
  ncclPatternNvls,          // NVLS
  ncclPatternNvlsTree,      // NVLS树
  ncclPatternPatUp,         // PAT上行
  ncclPatternPatDown,       // PAT下行
  ncclPatternSend,          // 发送
  ncclPatternRecv,          // 接收
};
```

### 8.5 代理线程流程

```cpp
void* proxyThread(void* arg) {
  struct ncclProxyState* state = (ncclProxyState*)arg;

  while (!state->stop) {
    // 1. 处理新的操作请求
    processNewOps(state);

    // 2. 进度现有操作
    for (struct ncclProxyArgs* args = state->active; args; args = args->next) {
      if (args->progress) {
        args->progress(state, args);
      }
    }

    // 3. 清理完成的操作
    cleanupCompletedOps(state);

    // 4. 等待新事件
    waitForEvents(state);
  }
}
```

---

## 9. 拓扑发现与图计算

### 9.1 拓扑系统

```cpp
struct ncclTopoSystem {
  int systemId;
  uint64_t hostHashes[NCCL_TOPO_MAX_NODES];
  int nHosts;
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];
  float maxBw;
  float totalBw;
};

struct ncclTopoNode {
  int type;
  int64_t id;

  // 类型特定数据
  union {
    struct { int dev; int rank; int cudaCompCap; int gdrSupport; } gpu;
    struct { int dev; uint64_t pciId; float bw; float latency; } net;
    struct { int arch; int vendor; int model; ncclAffinity affinity; } cpu;
    struct { uint64_t device; } pci;
  };

  int nlinks;
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];

  // 预计算的路径
  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];
};
```

### 9.2 路径计算

```cpp
ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system,
                                  struct ncclComm* comm) {
  // BFS计算所有节点间的最短路径
  for (int i = 0; i < system->nodes[GPU].count; i++) {
    struct ncclTopoNode* gpuNode = &system->nodes[GPU].nodes[i];

    // 从GPU到所有其他节点的路径
    bfsFromNode(gpuNode);

    // 存储路径到目标节点
    for (int j = 0; j < system->nodes[GPU].count; j++) {
      gpuNode->paths[GPU][j] = findPathTo(&system->nodes[GPU].nodes[j]);
    }

    // GPU到NIC的路径
    for (int j = 0; j < system->nodes[NET].count; j++) {
      gpuNode->paths[NET][j] = findPathTo(&system->nodes[NET].nodes[j]);
    }
  }
}
```

### 9.3 图计算

```cpp
struct ncclTopoGraph {
  int id;          // 图ID (ring=0, tree=1, collnet=2, nvls=3)
  int pattern;     // 拓扑模式

  int minChannels;
  int maxChannels;
  int nChannels;   // 输出: 使用的Channel数

  float bwIntra;   // 节点内带宽
  float bwInter;   // 节点间带宽
  float latencyInter;

  int typeIntra;   // 节点内路径类型
  int typeInter;   // 节点间路径类型

  int intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES]; // 节点内路径
  int64_t inter[MAXCHANNELS*2];               // 节点间路径
};

ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system,
                             struct ncclTopoGraph* graph) {
  // 根据pattern选择算法
  switch (graph->pattern) {
    case NCCL_TOPO_PATTERN_RING:
      computeRing(system, graph);
      break;
    case NCCL_TOPO_PATTERN_TREE:
    case NCCL_TOPO_PATTERN_SPLIT_TREE:
    case NCCL_TOPO_PATTERN_BALANCED_TREE:
      computeTree(system, graph);
      break;
    case NCCL_TOPO_PATTERN_NVLS:
      computeNvls(system, graph);
      break;
    case NCCL_TOPO_PATTERN_COLLNET_DIRECT:
      computeCollnetDirect(system, graph);
      break;
  }
}
```

---

## 10. 设备端CUDA内核

### 10.1 内核生成

NCCL使用Python脚本(`src/device/generate.py`)生成专用内核：

```python
all_colls = ["Broadcast","Reduce","AllGather","AllGatherV",
             "ReduceScatter","AllReduce","SendRecv"]
all_redops = ["Sum","Prod","MinMax","PreMulSum","SumPostDiv"]
all_tys = ["i8","u8","i32","u32","i64","u64","f16","f32","f64",
           "bf16","f8e4m3","f8e5m2"]
all_protos = ["LL","LL128","SIMPLE"]
all_algos = ["TREE","RING","COLLNET_DIRECT","COLLNET_CHAIN","NVLS","NVLS_TREE","PAT"]
```

### 10.2 Primitives类

```cpp
template<typename T, typename RedOp, typename Fan, int Direct,
         typename Proto, int P2p, bool isNetOffload = false>
class Primitives;
```

核心操作：
- `send()`: 发送数据
- `recv()`: 接收数据
- `recvReduceSend()`: 接收、reduce、发送
- `recvReduceCopySend()`: 接收、reduce、复制、发送
- `directSend()`: 直接发送
- `directRecv()`: 直接接收

### 10.3 共享内存结构

```cpp
struct ncclShmem {
  struct ncclKernelComm comm;

  // 输入输出缓冲区
  const void* userInput;
  void* userOutput;

  // Channel相关
  int channelId;

  // Groups
  struct Group {
    void* srcs[MaxRecv];
    void* dsts[MaxSend];
  } groups[NCCL_MAX_GROUPS];
};
```

### 10.4 内核参数

```cpp
struct ncclDevKernelArgs {
  struct ncclKernelComm* comm;
  uint64_t channelMask;
  enum ncclDevWorkStorageType workStorageType;
  uint32_t workMask;
  void* workBuf;
  // struct ncclDevWorkBatch batches[];
};
```

---

## 11. 协议实现

### 11.1 LL协议 (Low Latency)

LL协议用于小消息低延迟场景：

```cpp
struct ProtoLL {
  static constexpr int Id = NCCL_PROTO_LL;

  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_LL]/NCCL_STEPS/2;
  }

  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t);
  }
};

union ncclLLFifoLine {
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
  int4 i4;
};
```

特点：
- 数据与标志交织存储
- 每个cache line一半是数据，一半是标志
- 适合小消息低延迟场景

### 11.2 LL128协议

LL128协议使用128字节的cache line：

```cpp
struct ProtoLL128 {
  static constexpr int Id = NCCL_PROTO_LL128;

  __device__ static int calcBytePerStep() {
    return (ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS) *
           NCCL_LL128_DATAELEMS/NCCL_LL128_LINEELEMS;
  }
};

#define NCCL_LL128_LINESIZE 128
#define NCCL_LL128_LINEELEMS (NCCL_LL128_LINESIZE/sizeof(uint64_t))
#define NCCL_LL128_DATAELEMS (NCCL_LL128_LINEELEMS-1)  // 最后一项是flag
```

特点：
- 128字节对齐
- 一个flag控制整个line
- 更高的数据利用率

### 11.3 SIMPLE协议

SIMPLE协议用于大消息高带宽场景：

```cpp
template<int SlicePerChunk_1, int StepPerSlice_1,
         int Unroll_1 = COLL_UNROLL,
         int MultimemSrcs_1 = 0, int MultimemDsts_1 = 0>
struct ProtoSimple {
  static constexpr int Id = NCCL_PROTO_SIMPLE;
  static constexpr int SlicePerChunk = SlicePerChunk_1;
  static constexpr int StepPerSlice = StepPerSlice_1;
  static constexpr int Unroll = Unroll_1;

  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  }
};
```

特点：
- 纯数据缓冲区
- 使用head/tail指针同步
- 适合大消息高带宽场景

### 11.4 协议选择

```cpp
// 数据量阈值
#define NCCL_LL_THREAD_THRESHOLD 8
#define NCCL_LL128_THREAD_THRESHOLD 8
#define NCCL_SIMPLE_THREAD_THRESHOLD 64

// 根据数据量选择协议
int selectProtocol(size_t nbytes) {
  if (nbytes < NCCL_LL_THRESHOLD) return NCCL_PROTO_LL;
  if (nbytes < NCCL_LL128_THRESHOLD) return NCCL_PROTO_LL128;
  return NCCL_PROTO_SIMPLE;
}
```

---

## 12. 算法选择与调优

### 12.1 支持的算法

```cpp
#define NCCL_ALGO_TREE 0          // 树算法
#define NCCL_ALGO_RING 1          // Ring算法
#define NCCL_ALGO_COLLNET_DIRECT 2 // CollNet直连
#define NCCL_ALGO_COLLNET_CHAIN 3  // CollNet链式
#define NCCL_ALGO_NVLS 4          // NVLS
#define NCCL_ALGO_NVLS_TREE 5     // NVLS树
#define NCCL_ALGO_PAT 6           // PAT算法
#define NCCL_NUM_ALGORITHMS 7
```

### 12.2 Tuner接口

```cpp
typedef ncclTuner_v5_t ncclTuner_t;

struct ncclTuner_v5_t {
  const char* name;
  ncclResult_t (*init)(size_t nRanks, size_t nNodes,
                       ncclDebugLogger_t logFunction, void** context);
  ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType,
                              size_t nBytes, int numPipeOps,
                              float** collCostTable, int numAlgo,
                              int numProto, int regBuff, int* nChannels);
  ncclResult_t (*destroy)(void* context);
};
```

### 12.3 成本模型

```cpp
struct ncclComm {
  // 调优常量
  ncclTunerConstants_t tunerConstants;

  // 线程阈值
  ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

  // 延迟和带宽模型
  float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

  // 最大线程数
  int maxThreads[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
};
```

### 12.4 Ring算法实现

AllReduce Ring算法：
1. 数据分成nChunks块
2. 每个节点执行nRanks-1次reduce前向传播
3. 每个节点执行nRanks-1次结果后向传播

```cpp
class RingARAlgorithm : public RingAlgorithm {
  void getNextSendAddr(int curStep, uint8_t **sendbuffOut,
                       size_t *sizeOut, void **mhandleOut);
  void getNextRecvAddr(int curStep, uint8_t **recvbuffOut,
                       size_t *sizeOut, void **mhandleOut);
};
```

### 12.5 Tree算法实现

```cpp
// 树形AllReduce: 先reduce到root，再broadcast
void runTreeAllReduce(ncclComm_t comm, void* sendbuff,
                      void* recvbuff, size_t count) {
  // 上行阶段：reduce到root
  if (tree.up >= 0) {
    recvReduceSend(up, ...);  // 从子节点接收并reduce后发送给父节点
  } else {
    // root节点
    for (int i = 0; i < tree.nChildren; i++) {
      recvReduceCopySend(down[i], ...);
    }
  }

  // 下行阶段：broadcast结果
  if (tree.up >= 0) {
    recvCopySend(up, ...);  // 从父节点接收并复制给子节点
  }
  for (int i = 0; i < tree.nChildren; i++) {
    send(down[i], ...);
  }
}
```

### 12.6 PAT算法

PAT (Parallel All-to-all Tree) 算法用于大规模AllGather和ReduceScatter：

```cpp
template<typename T>
class PatRSAlgorithm {
  size_t offset, end, count;
  int chunkCount, nelem;
  int rank, nranks, nrPow2;
  int aggFactor, as, a;
  int parallelFactor;
  int phase;

  __device__ void getNextOp(struct ncclPatStep* ps);
};
```

PAT算法特点：
- 并行化树操作
- 聚合多个步骤减少同步
- 适合大规模集群

---

## 13. 插件系统

### 13.1 网络插件

```cpp
// 插件必须导出的符号
extern ncclNet_t ncclNetPlugin;

// 网络接口
typedef struct {
  const char* name;
  ncclResult_t (*init)(void** ctx, uint64_t commId, ...);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_t* props);
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, ...);
  ncclResult_t (*connect)(void* ctx, int dev, void* handle, ...);
  ncclResult_t (*accept)(void* listenComm, ...);
  ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, ...);
  ncclResult_t (*isend)(void* sendComm, void* data, size_t size, ...);
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, ...);
  ncclResult_t (*test)(void* request, int* done, int* sizes);
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclNet_t;
```

插件加载：
```cpp
// 加载插件
void* handle = dlopen("libnccl-net.so", RTLD_NOW);
ncclNet_t* net = dlsym(handle, "ncclNetPlugin");
```

### 13.2 Tuner插件

```cpp
typedef struct {
  const char* name;
  ncclResult_t (*init)(size_t nRanks, size_t nNodes,
                       ncclDebugLogger_t logFn, void** ctx);
  ncclResult_t (*getCollInfo)(void* ctx, ncclFunc_t coll, size_t nBytes,
                              int numPipeOps, float** costTable,
                              int numAlgo, int numProto,
                              int regBuff, int* nChannels);
  ncclResult_t (*destroy)(void* ctx);
} ncclTuner_v5_t;
```

### 13.3 Profiler插件

```cpp
typedef struct ncclProfiler_v2_t {
  ncclResult_t (*init)(ncclProfilerDesc_v2_t* desc);
  ncclResult_t (*startEvent)(void* eHandle);
  ncclResult_t (*stopEvent)(void* eHandle);
  ncclResult_t (*recordEvent)(void* eHandle,
                              ncclProfilerEventState_t eState);
  ncclResult_t (*finalize)(void);
} ncclProfiler_v2_t;
```

---

## 14. 内存管理

### 14.1 内存管理器

```cpp
typedef struct ncclMemManager {
  ncclDynMemEntry* entries;  // 内存条目链表
  int numEntries;
  std::mutex lock;
  int released;
  int initialized;
  int refCount;

  // 统计
  size_t totalPersist;
  size_t totalScratch;
  size_t totalOffload;
  size_t cpuBackupUsage;
} ncclMemManager;

typedef enum {
  ncclMemPersist  = 0,  // 持久内存
  ncclMemScratch  = 1,  // 临时内存
  ncclMemOffload  = 2   // 可卸载内存
} ncclMemType_t;
```

### 14.2 CUDA内存分配

```cpp
// 使用CUDA虚拟内存管理
ncclResult_t ncclCuMemAlloc(void **ptr,
                            CUmemGenericAllocationHandle *handlep,
                            CUmemAllocationHandleType type,
                            size_t size,
                            struct ncclMemManager* manager,
                            ncclMemType_t memType) {
  // 1. 设置分配属性
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = type;

  // 2. 获取粒度
  cuMemGetAllocationGranularity(&granularity, &prop, ...);

  // 3. 创建物理内存
  cuMemCreate(&handle, size, &prop, 0);

  // 4. 预留虚拟地址
  cuMemAddressReserve((CUdeviceptr *)ptr, size, ...);

  // 5. 映射虚拟地址到物理内存
  cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0);

  // 6. 设置访问权限
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1);
}
```

### 14.3 内存池

```cpp
struct ncclMemoryPool {
  void** freeList;
  int freeCount;
  int capacity;
  size_t elementSize;
};

void* poolAlloc(ncclMemoryPool* pool) {
  if (pool->freeCount > 0) {
    return pool->freeList[--pool->freeCount];
  }
  return malloc(pool->elementSize);
}

void poolFree(ncclMemoryPool* pool, void* ptr) {
  if (pool->freeCount < pool->capacity) {
    pool->freeList[pool->freeCount++] = ptr;
  } else {
    free(ptr);
  }
}
```

---

## 15. 缓冲区注册机制

### 15.1 注册缓存

```cpp
struct ncclRegCache {
  struct ncclReg** slots;
  int capacity, population;
  uintptr_t pageSize;
};

struct ncclReg {
  uintptr_t begAddr, endAddr;
  int localRefs;
  int graphRefs;
  uint32_t state;

  // 网络注册
  struct ncclRegNetHandles* netHandleHead;

  // NVLS注册
  CUdeviceptr regAddr;
  size_t regUCSize, regMCSize;
  CUmemGenericAllocationHandle mcHandle;

  // CollNet注册
  void* collnetHandle;

  // IPC注册
  struct ncclPeerRegIpcAddr regIpcAddrs;
  struct ncclIpcRegInfo* ipcInfos[NCCL_MAX_LOCAL_RANKS];
};
```

### 15.2 注册流程

```cpp
ncclResult_t ncclCommRegister(ncclComm_t comm, void* buff,
                              size_t size, void** handle) {
  // 1. 查找缓存
  struct ncclReg* reg = findRegCache(comm, buff, size);
  if (reg) {
    reg->localRefs++;
    *handle = reg;
    return ncclSuccess;
  }

  // 2. 创建新条目
  reg = createRegEntry(buff, size);

  // 3. 网络注册
  if (comm->ncclNet) {
    comm->ncclNet->regMr(comm->netContext, buff, size, ...);
  }

  // 4. NVLS注册
  if (comm->nvlsSupport) {
    cuMemAddressReserve(&reg->regAddr, ...);
    cuMulticastBindMemhandle(...);
  }

  // 5. 插入缓存
  insertRegCache(comm, reg);
}
```

### 15.3 IPC注册

```cpp
ncclResult_t ncclIpcLocalRegisterBuffer(ncclComm* comm,
                                        const void* userbuff,
                                        size_t buffSize,
                                        int* peerRanks,
                                        int nPeers, ...) {
  // 1. 获取CUDA IPC handle
  cudaIpcMemHandle_t ipcHandle;
  cudaIpcGetMemHandle(&ipcHandle, (void*)userbuff);

  // 2. 交换handle
  bootstrapAllGather(&ipcHandle, ...);

  // 3. 打开peer buffers
  for (int i = 0; i < nPeers; i++) {
    cudaIpcOpenMemHandle(&peerAddrs[i], peerHandles[i], ...);
  }
}
```

---

## 16. 性能分析与Profiling

### 16.1 Profiler接口

```cpp
struct ncclProfilerProxy {
  bool initialized;
  struct ncclDevProfiler* workStarted[MAXCHANNELS];
  struct ncclDevProfiler* workCompleted[MAXCHANNELS];
  uint64_t workCounter[MAXCHANNELS];
};

struct ncclDevProfiler {
  struct {
    uint64_t counter;
    uint64_t timestamp;
  } data[MAX_PROFILER_EVENTS_PER_CHANNEL];
};

#define MAX_PROFILER_EVENTS_PER_CHANNEL 64
```

### 16.2 事件类型

```cpp
enum ncclProfilerEvent {
  NCCL_PROFILER_EVENT_GROUP_API,
  NCCL_PROFILER_EVENT_COLL_API,
  NCCL_PROFILER_EVENT_P2P_API,
  NCCL_PROFILER_EVENT_KERNEL_LAUNCH,
  NCCL_PROFILER_EVENT_PROXY_OP,
  NCCL_PROFILER_EVENT_PROXY_STEP,
};
```

### 16.3 Profiler回调

```cpp
ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo* info,
                                            bool isGraphCaptured);
ncclResult_t ncclProfilerStopGroupApiEvent();
ncclResult_t ncclProfilerStartKernelLaunchEvent(struct ncclKernelPlan* plan,
                                                cudaStream_t stream);
ncclResult_t ncclProfilerStopKernelLaunchEvent(struct ncclKernelPlan* plan);
ncclResult_t ncclProfilerStartProxyOpEvent(int sub, struct ncclProxyArgs* args);
ncclResult_t ncclProfilerStopProxyOpEvent(int sub, struct ncclProxyArgs* args);
```

### 16.4 NVTX集成

```cpp
// NVTX标记
#define NVTX_RANGE_START(name) \
  nvtxRangePush(name)
#define NVTX_RANGE_END() \
  nvtxRangePop()

// 内核启动
nvtxRangePush("NCCL Kernel");
launchKernel(plan, stream);
nvtxRangePop();
```

---

## 17. Bootstrap机制

### 17.1 Bootstrap Handle

```cpp
struct ncclBootstrapHandle {
  uint64_t magic;
  union ncclSocketAddress addr;
  int nRanks;
};
```

### 17.2 Bootstrap初始化流程

```cpp
ncclResult_t bootstrapInit(int nHandles, void* handle,
                           struct ncclComm* comm,
                           struct ncclComm* parent) {
  // 1. 创建根节点
  if (comm->rank == 0) {
    createRootBootstrap(comm);
  }

  // 2. 其他rank连接到根
  for (int r = 1; r < comm->nRanks; r++) {
    connectToRoot(comm, r);
  }

  // 3. 建立全连接mesh
  establishMesh(comm);

  // 4. 交换peer信息
  exchangePeerInfo(comm);
}
```

### 17.3 Bootstrap通信

```cpp
// 点对点通信
ncclResult_t bootstrapSend(void* commState, int peer, int tag,
                           void* data, int size);
ncclResult_t bootstrapRecv(void* commState, int peer, int tag,
                           void* data, int size);

// 集合通信
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);
ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag);
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks,
                                int root, void* bcastData, int size);
```

---

## 18. 组操作

### 18.1 组机制

```cpp
extern thread_local int ncclGroupNum;
extern thread_local ncclComm_t ncclGroupCommList;

ncclResult_t ncclGroupStart() {
  ncclGroupNum++;
  if (ncclGroupNum == 1) {
    ncclGroupCommList = NULL;
  }
  return ncclSuccess;
}

ncclResult_t ncclGroupEnd() {
  ncclGroupNum--;
  if (ncclGroupNum == 0) {
    // 执行所有挂起的操作
    return executeGroupOperations();
  }
  return ncclSuccess;
}
```

### 18.2 组内操作合并

```cpp
struct ncclComm* groupNext[ncclGroupTaskTypeNum];

// 在组内，操作被收集而不是立即执行
ncclResult_t enqueueCollective(ncclComm_t comm, ...) {
  if (ncclGroupNum > 0) {
    // 添加到组的任务列表
    addToGroupTaskList(comm, task);

    // 标记通信器在组中
    if (comm->groupNext[ncclGroupTaskTypeCollective] == (ncclComm*)0x1) {
      comm->groupNext[ncclGroupTaskTypeCollective] = ncclGroupCommList;
      ncclGroupCommList = comm;
    }
  } else {
    // 立即执行
    executeCollective(comm, task);
  }
}
```

---

## 19. 错误处理与容错

### 19.1 错误码

```cpp
typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6,
  ncclInProgress = 7,
  ncclNumResults = 8
} ncclResult_t;
```

### 19.2 错误检查宏

```cpp
#define NCCLCHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    WARN("NCCL error: %s", ncclGetErrorString(res)); \
    return res; \
  } \
} while(0)

#define NCCLCHECKGOTO(call, res, label) do { \
  res = call; \
  if (res != ncclSuccess) { \
    WARN("NCCL error: %s", ncclGetErrorString(res)); \
    goto label; \
  } \
} while(0)

#define CUDACHECK(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    WARN("CUDA error: %s", cudaGetErrorString(err)); \
    return ncclUnhandledCudaError; \
  } \
} while(0)
```

### 19.3 Abort机制

```cpp
struct ncclComm {
  uint32_t* abortFlag;      // 主机端abort标志
  uint32_t* abortFlagDev;   // 设备端abort标志
  uint32_t destroyFlag;     // 销毁标志
  uint32_t revokedFlag;     // 撤销标志
};

// 设备端检查abort
__device__ inline int checkAbort(int& abortCache, const int abortValue,
                                 int& spins) {
  if (abortCache & abortValue) return 1;
  if (++spins < NCCL_SPINS_BEFORE_CHECK_ABORT) return 0;
  spins = 0;
  int abort = *ncclShmem.comm.abortFlag;
  if (abort) {
    ncclShmem.aborted = abort;
    abortCache |= abortValue;
  }
  return abort;
}
```

### 19.4 异步错误处理

```cpp
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t state) {
  comm->asyncResult = state;
  // 通知所有线程
  if (comm->abortFlag) {
    *comm->abortFlag = 1;
  }
  return ncclSuccess;
}

ncclResult_t ncclGetAsyncResult(ncclResult_t* result) {
  *result = comm->asyncResult;
  return ncclSuccess;
}
```

---

## 20. 环境变量配置

### 20.1 调试选项

| 环境变量 | 说明 |
|---------|------|
| `NCCL_DEBUG` | 调试级别: VERSION, WARN, INFO, TRACE |
| `NCCL_DEBUG_SUBSYS` | 调试子系统: ALL, INIT, COLL, P2P, NET, ... |
| `NCCL_DEBUG_FILE` | 调试输出文件 |

### 20.2 网络配置

| 环境变量 | 说明 |
|---------|------|
| `NCCL_SOCKET_IFNAME` | Socket接口名 |
| `NCCL_IB_DISABLE` | 禁用InfiniBand |
| `NCCL_IB_HCA` | IB HCA列表 |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA级别 |
| `NCCL_NET_PLUGIN` | 网络插件名称 |

### 20.3 性能调优

| 环境变量 | 说明 |
|---------|------|
| `NCCL_NSOCKS_PERTHREAD` | 每线程socket数 |
| `NCCL_SOCKET_NTHREADS` | Socket线程数 |
| `NCCL_NTHREADS` | NCCL线程数 |
| `NCCL_MAX_NRINGS` | 最大ring数 |
| `NCCL_MIN_NCHANNELS` | 最小Channel数 |
| `NCCL_MAX_NCHANNELS` | 最大Channel数 |
| `NCCL_P2P_LEVEL` | P2P级别: SYS, NODE, PHB, PXN, PIX, NVL |
| `NCCL_ALGO` | 强制算法: Tree, Ring, ... |
| `NCCL_PROTO` | 强制协议: LL, LL128, Simple |

### 20.4 拓扑配置

| 环境变量 | 说明 |
|---------|------|
| `NCCL_TOPO_FILE` | 拓扑文件路径 |
| `NCCL_TOPO_DUMP_FILE` | 拓扑dump文件 |
| `NCCL_IGNORE_DISABLED_P2P` | 忽略禁用的P2P |

### 20.5 插件配置

| 环境变量 | 说明 |
|---------|------|
| `NCCL_TUNER_PLUGIN` | Tuner插件路径 |
| `NCCL_PROFILER_PLUGIN` | Profiler插件路径 |
| `NCCL_PROFILER_ENABLE` | 启用Profiler |

---

## 21. MNNVL多节点NVLink

### 21.1 概述

MNNVL (Multi-Node NVLink) 允许跨节点的GPU通过NVLink直接通信，利用NVSwitch架构实现多节点GPU互联。

### 21.2 数据结构

```cpp
struct ncclComm {
  // MNNVL支持标志
  int MNNVL;

  // Clique信息
  struct cliqueInfo {
    int id;          // Clique ID
    int size;        // Clique大小
    int* ranks;      // Clique中的rank列表
  } clique;

  int cliqueRank;    // 在clique中的rank
};

struct ncclPeerInfo {
  // MNNVL支持
  nvmlGpuFabricInfoV_t fabricInfo;
  int cuMemSupport;
};
```

### 21.3 MNNVL检测

```cpp
ncclResult_t ncclMnnvlCheck(struct ncclComm* comm) {
  // 1. 检查cuMem支持
  if (!ncclCuMemEnable()) return ncclSuccess;

  // 2. 检查FABRIC handle支持
  cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev);
  if (!flag) return ncclSuccess;

  // 3. 检查所有rank的fabric状态
  for (int i = 0; i < comm->nRanks; i++) {
    if (comm->peerInfo[i].fabricInfo.state != NVML_GPU_FABRIC_STATE_COMPLETED)
      return ncclSuccess;
  }

  // 4. 确定MNNVL domain/clique
  for (int i = 0; i < comm->nRanks; i++) {
    if (clusterUuid匹配 && cliqueId匹配) {
      comm->clique.ranks[comm->clique.size++] = i;
    }
  }

  // 5. 验证FABRIC handles可以导入导出
  cuMemExportToShareableHandle(&cuDesc, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
  cuMemImportFromShareableHandle(&handle, &cuDesc, CU_MEM_HANDLE_TYPE_FABRIC);

  comm->MNNVL = 1;
}
```

### 21.4 IMEX通道

MNNVL依赖IMEX (Inter-Process Memory Export) 通道进行跨进程的内存共享：

- 设备文件: `/dev/nvidia-caps-imex-channels`
- 配置工具: `nvidia-imex-ctl`

---

## 22. GIN全局互联网络

### 22.1 概述

GIN (Global Interconnect Network) 是NCCL的多节点网络互联抽象层，支持多种网络后端。

### 22.2 GIN类型

```cpp
typedef enum {
  ncclGinTypeNone = 0,
  ncclGinTypeNet = 1,      // 标准网络
  ncclGinTypeRailed = 2,   // 多轨道网络
} ncclGinType_t;

typedef enum {
  ncclGinConnectionDefault = 0,
  ncclGinConnectionRdma = 1,
} ncclGinConnectionType_t;
```

### 22.3 GIN状态

```cpp
struct ncclGinState {
  ncclGin_t* ncclGin;
  void* ginInstance;
  bool connected;
  ncclGinType_t ginType;

  int ginCommCount;
  int ginContextCount;
  void* ginComms[NCCL_GIN_MAX_CONNECTIONS];
  void* ginCtx[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetDeviceHandle_t* ginDevHandles[NCCL_GIN_MAX_CONNECTIONS];

  int needsProxyProgress;
  int ginProgress;

  std::thread thread;
  std::mutex mutex;
  std::condition_variable cond;

  // 信号和计数器空间
  int signalSpaceSize;
  int counterSpaceSize;
  ncclSpace signalSpace;
  ncclSpace counterSpace;

  int ginQueueDepth;
  ncclGinConnectionType_t ginConnectionType;
};
```

### 22.4 GIN操作

```cpp
// 连接GIN
ncclResult_t ncclGinConnectOnce(struct ncclComm* comm,
                                ncclGinConnectionType_t type,
                                int reqGinContextCount,
                                int reqGinQueueDepth);

// 注册内存
ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address,
                             size_t size, void* ginHostWins[],
                             ncclGinWindow_t ginDevWins[], int winFlags);

// 分配信号和计数器
ncclResult_t ncclGinAllocSignalsCounters(struct ncclComm* comm,
                                         int nSignals, uint32_t* outSignal0,
                                         int nCounters, uint32_t* outCounter0);
```

### 22.5 GIN插件接口

```cpp
typedef struct ncclGin_v12_t {
  const char* name;
  ncclResult_t (*init)(void** context, ncclDebugLogger_t logFn);
  ncclResult_t (*connect)(void* context, ncclGinConnectionType_t type, ...);
  ncclResult_t (*registerMem)(void* context, void* address, size_t size, ...);
  ncclResult_t (*deregisterMem)(void* context, void* handle);
  ncclResult_t (*allocSignals)(void* context, int nSignals, ...);
  ncclResult_t (*freeSignals)(void* context, void* handle);
  ncclResult_t (*progress)(void* context);
  ncclResult_t (*finalize)(void* context);
} ncclGin_v12_t;
```

---

## 23. CollNet集合网络

### 23.1 概述

CollNet是网络级集合通信加速技术，利用交换机或SmartNIC进行硬件加速的集合操作。

### 23.2 CollNet接口

```cpp
typedef struct {
  const char* name;
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_t* props);
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(void* handles[], int nranks, int rank,
                          void* listenComm, void** collComm);
  ncclResult_t (*reduceSupport)(ncclDataType_t dataType, ncclRedOp_t redOp,
                                int* supported);
  ncclResult_t (*regMr)(void* collComm, void* data, size_t size,
                        int type, void** mhandle);
  ncclResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData,
                             int count, ncclDataType_t dataType,
                             ncclRedOp_t redOp, void* sendMhandle,
                             void* recvMhandle, void** request);
  ncclResult_t (*test)(void* request, int* done, int* size);
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclCollNet_t;
```

### 23.3 CollNet模式

```cpp
// CollNet Direct模式
#define NCCL_ALGO_COLLNET_DIRECT 2
// GPU -> NIC -> Switch(SHARP) -> NIC -> GPU

// CollNet Chain模式
#define NCCL_ALGO_COLLNET_CHAIN 3
// GPU -> GPU -> NIC -> Switch(SHARP) -> NIC -> GPU -> GPU
```

### 23.4 CollNet拓扑

```cpp
struct ncclDirect {
  int depth;
  int out;
  int nHeads;     // 并行head数量
  int headRank;   // 当前rank的head索引
  int shift;      // scatter/gather的偏移
  int heads[NCCL_MAX_DIRECT_ARITY+1];  // head rank列表
  int up[NCCL_MAX_DIRECT_ARITY];
  int down[NCCL_MAX_DIRECT_ARITY];
};
```

### 23.5 支持矩阵

```cpp
struct ncclComm {
  uint8_t collNetSupportMatrix[4/*sum,prod,max,min*/][ncclNumTypes];
  int* collNetHeads;
  int collNetHeadsNum;
};
```

---

## 24. 对称内存运行时

### 24.1 概述

对称内存运行时(Devr)支持跨rank的对称内存窗口，用于高性能的对称集合操作。

### 24.2 窗口结构

```cpp
struct ncclDevrWindow {
  struct ncclDevrMemory* memory;
  void* userPtr;
  size_t size;
  size_t bigOffset;       // 在大VA空间中的偏移
  int winFlags;
  void* localRegHandle;
  struct ncclWindow_vidmem* vidmem;
  struct ncclDevrWindow* next;
  struct ncclComm* comm;
};
```

### 24.3 Devr状态

```cpp
struct ncclDevrState {
  // LSA (Locally Symmetric Allocation) team
  int lsaSelf;
  int lsaSize;
  int* lsaRankList;
  int nLsaTeams;

  size_t granularity;
  bool ginEnabled;
  bool rmaProxyEnabled;

  struct ncclDevrMemory* memHead;
  struct ncclDevrWindowSorted* winSorted;

  // 大虚拟地址空间
  size_t bigSize;
  struct ncclSpace bigSpace;
  void* lsaFlatBase;  // 所有lsa rank的big VA拼接基址

  struct ncclShadowPool shadows;
  struct ncclDevCommWindowTable* windowTable;
};
```

### 24.4 窗口注册

```cpp
ncclResult_t ncclDevrWindowRegisterInGroup(
  struct ncclComm* comm, void* ptr, size_t size,
  int winFlags, ncclWindow_t* outWinDev
);

// 获取其他lsa rank的对称指针
ncclResult_t ncclDevrGetLsaRankPtr(struct ncclComm* comm,
                                   struct ncclDevrWindow* winHost,
                                   size_t offset, int lsaRank, void** outPtr);
```

---

## 25. RMA远程内存访问

### 25.1 概述

RMA (Remote Memory Access) 提供单边通信能力，允许直接读写远程内存。

### 25.2 RMA任务结构

```cpp
struct ncclTaskRma {
  struct ncclTaskRma* next;
  ncclFunc_t func;
  int ctx;
  size_t count;
  ncclDataType_t datatype;
  size_t bytes;

  void const* srcBuff;
  size_t srcWinOffset;
  struct ncclDevrWindow* srcWinHost;

  int peer;
  size_t peerWinOffset;
  struct ncclDevrWindow* peerWinHost;

  // 信号操作
  ncclSignalMode_t signalMode;
  int* peers;
  int* nsignals;
  int npeers;
};
```

### 25.3 信号模式

```cpp
typedef enum {
  ncclSignalModeNone = 0,
  ncclSignalModeSend = 1,
  ncclSignalModeRecv = 2,
} ncclSignalMode_t;
```

### 25.4 RMA操作

```cpp
// Put操作：将本地数据写入远程窗口
ncclResult_t ncclRmaPut(ncclComm_t comm, void* srcBuff,
                        size_t count, int peer,
                        ncclWindow_t srcWin, ncclWindow_t peerWin,
                        ncclDataType_t datatype);

// Get操作：从远程窗口读取数据到本地
ncclResult_t ncclRmaGet(ncclComm_t comm, void* dstBuff,
                        size_t count, int peer,
                        ncclWindow_t dstWin, ncclWindow_t peerWin,
                        ncclDataType_t datatype);
```

---

## 26. CE Copy Engine

### 26.1 概述

CE (Copy Engine) 集合利用GPU的Copy Engine进行异步数据传输，释放计算资源。

### 26.2 CE状态

```cpp
struct ncclCeColl {
  bool initialized;
  cudaStream_t stream;
  void* ceHandle;
  struct ncclIntruQueue<ncclCeInitTask> ceInitTaskQueue;
};

struct ncclComm {
  struct ncclCeColl ceColl;
  struct ncclIntruQueue<ncclCeInitTask> ceInitTaskQueue;
};
```

### 26.3 CE参数

```cpp
struct ncclCeCollArgs {
  struct ncclKernelComm* comm;
  int channelId;
  int nChannels;
  ncclFunc_t func;
  int algorithm;
  int protocol;
  void* sendbuff;
  void* recvbuff;
  size_t count;
  int root;
  ncclDataType_t datatype;
  ncclDevRedOpFull redOp;
};
```

### 26.4 CE内核

```cpp
// CE内核在Copy Engine上运行
__global__ void ncclCeCollKernel(struct ncclCeCollArgs* args) {
  // 使用Copy Engine进行数据传输
  // 不占用SM资源
}
```

### 26.5 CE Profiling

```cpp
ncclResult_t ncclProfilerStartCeCollEvent(struct ncclComm* comm,
                                          struct ncclCeCollArgs* args,
                                          cudaStream_t stream);
ncclResult_t ncclProfilerStopCeCollEvent(struct ncclComm* comm,
                                         struct ncclCeCollArgs* args,
                                         cudaStream_t stream);
```

---

## 附录A: 关键常量定义

```cpp
#define NCCL_MAX_OPS 2048           // 最大操作数
#define NCCL_STEPS 8                 // FIFO深度
#define WARP_SIZE 32                 // Warp大小
#define MAXCHANNELS 64               // 最大Channel数
#define NCCL_MAX_LOCAL_RANKS 72      // 最大本地rank数
#define NCCL_MAX_NTHREADS 640        // 最大线程数
#define NCCL_MIN_NTHREADS 128        // 最小线程数
#define CACHE_LINE_SIZE 128          // Cache line大小
#define MEM_ALIGN 4096               // 内存对齐

// LL协议参数
#define NCCL_LL_LINES_PER_THREAD 8
#define NCCL_LL_CLEAN_MASK 0x7ffffff8
#define NCCL_LL_MAX_NTHREADS 512

// LL128协议参数
#define NCCL_LL128_LINESIZE 128
#define NCCL_LL128_ELEMS_PER_THREAD 120
#define NCCL_LL128_MAX_NTHREADS 640

// 最大网络设备数
#define NCCL_MAX_NETDEVS 128
```

## 附录B: 带宽常量

```cpp
#define SM60_NVLINK_BW 18.0   // Pascal NVLink
#define SM70_NVLINK_BW 20.0   // Volta NVLink
#define SM80_NVLINK_BW 20.0   // Ampere NVLink
#define SM90_NVLINK_BW 20.6   // Hopper NVLink
#define SM100_NVLINK_BW 40.1  // Blackwell NVLink
#define PCI_BW 12.0           // PCIe Gen3 x16
#define NET_BW 12.0           // 100Gbit网络
#define LOC_BW 5000.0         // 本地带宽
```

## 附录C: 数据类型大小

```cpp
inline int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
  case ncclInt8: case ncclUint8:
  case ncclFloat8e4m3: case ncclFloat8e5m2:
    return 1;
  case ncclFloat16: case ncclBfloat16:
    return 2;
  case ncclInt32: case ncclUint32: case ncclFloat32:
    return 4;
  case ncclInt64: case ncclUint64: case ncclFloat64:
    return 8;
  default:
    return -1;
  }
}
```

---

*文档版本: 2.0*
*基于NCCL版本: 2.29.7-1*
*生成日期: 2026-03-27*

## 修订历史

- v1.0: 初始版本，包含核心架构分析
- v2.0: 补充MNNVL、GIN、CollNet、对称内存运行时、RMA、CE模块
