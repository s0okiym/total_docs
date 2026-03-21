# NCCL 通信机制与数据路径详解

本文档基于 NCCL 源代码分析，详细阐述 NCCL 的通信链路类型、数据流转机制以及 AllReduce 等集合通信操作的实现原理。

---

## 目录

1. [NCCL 架构概览](#1-nccl-架构概览)
2. [通信链路类型](#2-通信链路类型)
3. [AllReduce 实现机制](#3-allreduce-实现机制)
4. [通信协议详解](#4-通信协议详解)
5. [总结](#5-总结)

---

## 1. NCCL 架构概览

NCCL (NVIDIA Collective Communications Library) 是一个为 GPU 设计的高性能集合通信库。

### 核心架构层次

```
User Application
       |
   NCCL API Layer (ncclAllReduce, ncclBroadcast, etc.)
       |
   Enqueue Layer (任务调度与批处理)
       |
   Kernel Launch (CUDA Kernel 执行)
       |
   Device Kernel (Ring/Tree/NVLS 算法实现)
       |
   Primitives Layer (Send/Recv/Reduce 原语)
       |
   Transport Layer (P2P/SHM/NET/CollNet/NVLS 传输)
       |
   Proxy Thread (异步网络通信处理)
```

### 核心组件

1. **Transport Layer**: 负责实际的跨 GPU/跨节点数据传输
2. **Primitives**: 提供基础的 send/recv/reduce 操作原语
3. **Device Kernel**: 在 GPU 上执行的通信算法
4. **Proxy Thread**: 处理需要 CPU 协助的异步操作（主要是网络通信）

---

## 2. 通信链路类型

NCCL 支持多种通信链路类型，根据硬件拓扑和节点位置自动选择最优路径。

### 传输类型枚举

```cpp
#define NTRANSPORTS 4
#define TRANSPORT_UNDEFINED -1
#define TRANSPORT_P2P 0      // GPU 直接 P2P 访问
#define TRANSPORT_SHM 1      // 共享内存
#define TRANSPORT_NET 2      // 网络 (IB/RoCE/Eth)
#define TRANSPORT_COLLNET 3  // 集合网络 (SHARP)
```

---

### 2.1 P2P (Peer-to-Peer)

**适用场景**: 同一节点内、支持 P2P 访问的 GPU 之间

#### 2.1.1 连接类型

P2P 传输根据进程和 GPU 关系分为四种类型：

```cpp
enum p2pType { 
    P2P_DIRECT,       // 同进程内 GPU 直接指针访问
    P2P_INTERMEDIATE, // 通过中间 GPU 中转
    P2P_IPC,          // 跨进程 IPC (Legacy CUDA IPC)
    P2P_CUMEM         // 跨进程 cuMem API
};
```

#### 2.1.2 实现机制

| 类型 | 同进程 | 同 GPU | 实现方式 | 内存访问 |
|------|--------|--------|----------|----------|
| P2P_DIRECT | 是 | 否 | 直接指针 | cudaDeviceEnablePeerAccess |
| P2P_CUMEM | 是/否 | 否 | cuMem API | cuMemMap + cuMemSetAccess |
| P2P_IPC | 否 | 否 | cudaIpcMemHandle | cudaIpcOpenMemHandle |

#### 2.1.3 数据传输流程

发送端 GPU:
1. 写入本地 buffer
2. 更新 head (sendMem)

接收端 GPU:
3. 读取数据
4. 更新 tail (recvMem)

#### 2.1.4 P2P Read vs P2P Write

```cpp
// P2P Read: 数据存储在接收端，发送端主动读取
// P2P Write: 数据存储在发送端，发送端主动写入

// 默认行为：NVLink 连接时优先使用 Read (性能更好)
// 配置参数：NCCL_P2P_READ_ENABLE
```

**P2P Read 优势**:
- NVLink 上读性能优于写
- 减少接收端的内存占用
- 适合 Ring 算法中的 Scatter-Reduce 阶段

---

### 2.2 SHM (Shared Memory)

**适用场景**: 同一节点内、不同进程、无法使用 P2P 时（如 GPU 跨 NUMA 域）

#### 2.2.1 架构

SHM 使用 Host Memory 作为中间缓冲区：
- ncclSendMem (head) - 发送端状态
- ncclRecvMem (tail + connFifo) - 接收端状态
- Data Buffer - 实际数据

#### 2.2.2 两种实现方式

1. **Legacy SHM**: 使用 `/dev/shm/` 文件映射
2. **cuMem SHM**: CUDA 12.2+，使用 `cuMemHostAlloc` 分配可共享内存

#### 2.2.3 CE (Copy Engine) memcpy 模式

当启用 `NCCL_SHM_USE_CUDA_MEMCPY=1` 时：

```
GPU --> Proxy Thread --> SHM --> Proxy Thread --> GPU
      (cudaMemcpyAsync)      (cudaMemcpyAsync)
```

优点：
- 不占用 GPU SM 资源
- 使用专用拷贝引擎

---

### 2.3 NET (Network)

**适用场景**: 跨节点通信

#### 2.3.1 架构

节点 A: GPU -> Proxy Thread -> NIC -> Network -> NIC -> Proxy Thread -> GPU :节点 B

#### 2.3.2 GDR (GPU Direct RDMA)

```cpp
enum ncclTopoGdrMode {
    ncclTopoGdrModeNone = 0,    // 不使用 GDR
    ncclTopoGdrModeC2C = 1,     // C2C 模式
    ncclTopoGdrModePci = 2      // PCIe 模式
};
```

**GDR 数据传输路径**:
```
GPU Memory --> NIC --> Network --> NIC --> GPU Memory
      |                              |
      +------- GDR (绕过 CPU) --------+
```

#### 2.3.3 Proxy Thread 工作流程

```cpp
// 发送端 Proxy
sendProxyProgress() {
    1. 检查 GPU 是否有新数据 (tail > posted)
    2. 调用 ncclNet->isend() 发送数据
    3. 等待发送完成 (test)
    4. 更新 head 通知 GPU
}

// 接收端 Proxy  
recvProxyProgress() {
    1. 调用 ncclNet->irecv() 接收数据
    2. 等待接收完成 (test)
    3. 可选：GDR Flush 确保数据可见
    4. 更新 tail 通知 GPU
}
```

---

### 2.4 NVLink-Specific (NVLS)

**适用场景**: NVLink 4.0+ 支持的 GPU，利用 NVLink 多播功能

#### 2.4.1 NVLS 架构

```
                    NVLS Multicast Buffer
                           |
        +------------------+------------------+
        |                  |                  |
    GPU 0 (Head 0)  <--> GPU 1 (Head 1) <--> GPU 2 (Head 2)
                               |
                              NIC
```

#### 2.4.2 NVLS 多播机制

**工作流程**:
1. **Scatter**: 各 GPU 将数据写入 NVLS 多播缓冲区
2. **Reduce**: NVLink 硬件自动执行归约
3. **Gather**: 各 GPU 从多播缓冲区读取结果

#### 2.4.3 单节点 vs 多节点

**单节点**: GPU 0,1,2 --> NVLS Buffer --> 归约结果 --> 各 GPU

**多节点**: 本地 NVLS 归约 --> Head GPU --> 网络 --> 跨节点 AllReduce --> Head GPU --> NVLS 广播

---

### 2.5 CollNet (Collective Network)

**适用场景**: 支持 SHARP 的 InfiniBand 网络

#### 2.5.1 架构

SHARP Switch 包含 In-Network Reduction Engine，可在交换机层面执行归约。

#### 2.5.2 两种模式

1. **CollNet Direct**: 每个节点的 Head GPU 直接连接网络
2. **CollNet Chain**: 节点内 GPU 形成链，由 Root 节点统一收发

---

## 3. AllReduce 实现机制

### 3.1 数据切分策略

AllReduce 的数据切分采用多层次策略：

```
总数据量 N 字节
    |
    +--> 按 Channel 切分 (nChannels)
    |     每个 channel 处理: N / nChannels
    |
    +--> Channel 内部按 Loop 切分
          每个 loop: nRanks * chunkSize
```

#### 3.1.1 Chunk 切分公式

```cpp
// 核心函数: ncclCollCbdPart
// 步骤 1: 确定每个 channel 的数据范围
if (channelId == channelLo) {
    partCount = countLo;
    chunkCount = chunkGrainsLo * eltPerGrain;
} else if (channelId == channelHi) {
    partOffset = countLo + nMidChannels * countMid;
    partCount = countHi;
    chunkCount = chunkGrainsHi * eltPerGrain;
} else {
    mid = channelId - channelLo - 1;
    partOffset = countLo + mid * countMid;
    partCount = countMid;
    chunkCount = chunkGrainsMid * eltPerGrain;
}

// 步骤 2: Ring 中的 chunk 分配
loopCount = nranks * chunkCount;

// 每个 rank 在一轮循环中处理一个 chunk
for (elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
    // 2*(nRanks-1) 个步骤完成一个 loop
    // 步骤 0..nRanks-2: Scatter-Reduce
    // 步骤 nRanks-1..2*nRanks-3: AllGather
}
```

#### 3.1.2 Slice vs Chunk

```cpp
#define NCCL_STEPS 8
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)  // = 2
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)  // = 4

// Chunk: 一次完整操作的数据单元
// Slice: Chunk 的子单元，用于流水线

// 关系: 1 Chunk = CHUNKSTEPS/SLICESTEPS = 2 Slices
```

---

### 3.2 Ring 算法

#### 3.2.1 Ring 拓扑

```
Rank 0 --> Rank 1 --> Rank 2 --> ... --> Rank N-1 --> Rank 0
  ^                                                    |
  +----------------------------------------------------+
```

#### 3.2.2 2N-2 步骤流程

```cpp
// 核心循环
for (elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
    
    // 步骤 0: 初始发送
    chunk = modRanks(ringIx + nranks - 1);
    directSend(offset, offset, nelem);
    
    // 步骤 1..nRanks-2: Scatter-Reduce
    for (j = 2; j < nranks; ++j) {
        chunk = modRanks(ringIx + nranks - j);
        directRecvReduceDirectSend(offset, offset, nelem);
    }
    
    // 步骤 nRanks-1: 最终归约
    chunk = ringIx;
    directRecvReduceCopyDirectSend(offset, offset, nelem, postOp=true);
    
    // 步骤 nRanks..2*nRanks-3: AllGather
    for (j = 1; j < nranks - 1; ++j) {
        chunk = modRanks(ringIx + nranks - j);
        directRecvCopyDirectSend(offset, offset, nelem);
    }
    
    // 最后: 接收最终结果
    chunk = modRanks(ringIx + 1);
    directRecv(offset, nelem);
}
```

#### 3.2.3 可视化流程 (4 Ranks)

```
Rank 0: [发送C3]<-[收C2归约发]<-[收C1归约发]<-[收C0归约存发]<-[转发C1]<-[转发C2]<-[收C3]
Rank 1: [发送C0]<-[收C3归约发]<-[收C2归约发]<-[收C1归约存发]<-[转发C2]<-[转发C3]<-[收C0]
Rank 2: [发送C1]<-[收C0归约发]<-[收C3归约发]<-[收C2归约存发]<-[转发C3]<-[转发C0]<-[收C1]
Rank 3: [发送C2]<-[收C1归约发]<-[收C0归约发]<-[收C3归约存发]<-[转发C0]<-[转发C1]<-[收C2]

C0-C3: Chunk 0-3 (数据块)
归约: 接收数据与本地数据归约后发送
存: 将归约结果存入最终缓冲区
```

---

### 3.3 Tree 算法

#### 3.3.1 Tree 拓扑

```
               Root (Rank 0)
              /             \
         Node 1            Node 2
        /      \          /      \
    Node 3    Node 4   Node 5    Node 6
```

#### 3.3.2 Tree Split 模式

```cpp
// Tree Split 将线程分为两组
if (tree->up == -1) {
    // 我是 Root：接收子节点数据，归约后发送给自己
    directRecvReduceCopyDirectSend(...);
}
else if (tid < nthreadsSplit) {
    // Reduce Up 组 (约 70% 线程)
    directRecvReduceDirectSend(...);
}
else {
    // Broadcast Down 组 (约 30% 线程)
    directRecvCopyDirectSend(...);
}
```

---

### 3.4 NVLS 算法

#### 3.4.1 单节点 NVLS

```cpp
// 线程分工
nThreadsScatter  = scatterWarps  * WARP_SIZE;  // Scatter 阶段
nThreadsGather   = gatherWarps   * WARP_SIZE;  // Gather 阶段
nThreadsReduce   = reduceWarps   * WARP_SIZE;  // Reduce + NVLS 操作
nThreadsBcast    = bcastWarps    * WARP_SIZE;  // Broadcast 阶段

// 单节点流程
if (tid < tidEndScatter) {
    // Scatter: 将数据分散到 NVLS buffer
    prims.scatter(offset, nelem, chunkSize);
}
else if (tid < tidEndGather) {
    // Gather: 从 NVLS buffer 收集结果
    prims.gather(offset, nelem, chunkSize);
}
else if (tid < tidEndReduce && nvls->headRank != -1) {
    // Reduce: 通过 NVLink 多播执行归约
    prims.directRecvDirectSend(offset, offset, nelem);
}
```

#### 3.4.2 多节点 NVLS Tree

本地 NVLS 归约 --> Head GPU --> 网络 --> 跨节点 AllReduce --> Head GPU --> NVLS 广播

---

### 3.5 CollNet 算法

```cpp
// 线程分工
nThreadsScatter = WARP_SIZE + (hasUp ? COLLNET_COPY_THREADS : 0);
nThreadsGather  =             (hasUp ? COLLNET_COPY_THREADS : 0);
nThreadsBcast   = WARP_SIZE + (hasUp ? 0 : COLLNET_COPY_THREADS);
nThreadsReduce  = 剩余线程;

if (tid >= tidStartScatter && tid < tidStartReduce && hasUp) {
    // Scatter: 发送数据到网络
    prims.scatter(offset, nelem, chunkSize, peerOffset, headRank, shift);
}
else if (tid >= tidStartReduce && direct->out != -1) {
    if (hasDn) {
        // Reduce + 发送到网络
        prims.recvReduceDirectSend(offset, offset, nelem);
    }
}
else if (tid < tidStartBcast && hasUp) {
    // Gather: 从网络收集
    prims.directGather(offset, nelem, chunkSize, peerOffset, headRank, shift);
}
else if (tid >= tidStartBcast && tid < tidStartScatter && direct->out != -1) {
    if (hasDn) {
        // 从网络接收 + 广播
        prims.directRecvCopyDirectSend(offset, offset, nelem, postOp=true);
    }
}
```

---

## 4. 通信协议详解

### 4.1 Simple 协议

**适用场景**: 大数据量传输 (通常 > 32KB)

#### 4.1.1 数据布局

```
ncclRecvMem:
+--------------+----------------------------------------+
|   tail       |         buff[]                         |
|  (8 bytes)   |   (buffSize = stepSize * NCCL_STEPS)   |
+--------------+----------------------------------------+
               ^
               |
        +------+------+
        v             v
   +---------+   +---------+
   | Step 0  |   | Step 1  |  ...
   | (slot)  |   | (slot)  |
   +---------+   +---------+
```

#### 4.1.2 通信流程

```cpp
// 发送端
void send() {
    // 1. 等待接收端就绪 (head 检查)
    while (sendMem->head < nextStep - NCCL_STEPS);
    
    // 2. 写入数据
    copyData(buff[slot], src, size);
    
    // 3. 更新 tail
    __threadfence_system();
    recvMem->tail = nextStep;
}

// 接收端
void recv() {
    // 1. 等待数据就绪 (tail 检查)
    while (recvMem->tail < nextStep);
    
    // 2. 读取数据
    copyData(dst, buff[slot], size);
    
    // 3. 更新 head
    __threadfence_system();
    sendMem->head = nextStep;
}
```

---

### 4.2 LL (Low Latency) 协议

**适用场景**: 小数据量传输 (通常 < 4KB)

#### 4.2.1 数据格式

```cpp
union ncclLLFifoLine {
    struct {
        uint32_t data1;  // 数据
        uint32_t flag1;  // 标志
        uint32_t data2;  // 数据
        uint32_t flag2;  // 标志
    };
    uint64_t v[2];
};

// 每个 line: 8 bytes 数据 + 8 bytes flag
```

#### 4.2.2 握手协议

```cpp
// Flag 编码
#define NCCL_LL_FLAG(a) ((uint32_t)(a))

// 发送
line->data1 = payload1;
line->flag1 = NCCL_LL_FLAG(step + 1);

// 接收
while (line->flag1 != NCCL_LL_FLAG(expectedStep));
result = line->data1;
```

---

### 4.3 LL128 协议

**适用场景**: 中等数据量 (4KB - 32KB)，需要平衡延迟和带宽

#### 4.3.1 数据格式

```cpp
#define NCCL_LL128_LINESIZE 128
#define NCCL_LL128_LINEELEMS (NCCL_LL128_LINESIZE/sizeof(uint64_t))  // = 16
#define NCCL_LL128_DATAELEMS (NCCL_LL128_LINEELEMS-1)  // = 15

// 每行: 15 个数据元素 + 1 个标志元素
// 总线宽: 128 字节
```

#### 4.3.2 布局

```
Line 0: [data0] [data1] ... [data13] [data14] [flag]
Line 1: [data0] [data1] ... [data13] [data14] [flag]
...
```

---

## 5. 总结

### 5.1 传输方式选择优先级

```
1. 同 GPU: 直接指针拷贝
2. 同节点 P2P 可用: P2P Direct/IPC/cuMem
3. 同节点 P2P 不可用: SHM
4. 跨节点: NET (GDR 优先)
5. NVLS 可用: NVLS (NVLink 4.0+)
6. SHARP 可用: CollNet
```

### 5.2 算法选择策略

| 场景 | 推荐算法 | 原因 |
|------|----------|------|
| 小数据量 (< 256KB) | Ring LL/LL128 | 低延迟 |
| 大数据量, 单机 | NVLS | 高带宽, 硬件归约 |
| 大数据量, 多机 | Tree / Ring | 平衡带宽和延迟 |
| SHARP 网络 | CollNet | 网内计算 |

### 5.3 核心设计原则

1. **流水线化**: 通过 Steps/SliceSteps 实现通信计算重叠
2. **分层切分**: Channel -> Loop -> Chunk -> Slice 多级并行
3. **零拷贝**: 支持用户缓冲区注册，减少数据拷贝
4. **自适应**: 根据拓扑自动选择最优传输方式和算法
5. **硬件加速**: 利用 GDR、NVLink 多播、SHARP 等硬件特性

---

*本文档基于 NCCL 源代码分析，描述了 NCCL 的核心通信机制和数据路径。实际实现可能因版本不同而有所差异。*