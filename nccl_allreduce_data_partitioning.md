# NCCL AllReduce 数据切分与通信Pipeline深度解析

## 一、总体架构概览

NCCL AllReduce的实现采用**分层数据切分策略**，从宏观到微观分为三个层次：

```
┌─────────────────────────────────────────────────────────────────┐
│  Level 1: Host-Level Channel 切分                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │  Channel 0  │ │  Channel 1  │ │  Channel N  │               │
│  │  countLo    │ │  countMid   │ │  countHi    │               │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘               │
└─────────┼───────────────┼───────────────┼───────────────────────┘
          │               │               │
          ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Level 2: Device-Level Chunk 切分 (per Channel)                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  Channel Count = chunkCount × nranks (Ring算法)      │       │
│  │  LoopSize = nranks × chunkCount                      │       │
│  └─────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Level 3: Protocol-Level Step/Slice 切分 (per Chunk)             │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  ChunkSteps = 4 (ALLREDUCE_CHUNKSTEPS)               │       │
│  │  SliceSteps = 2 (ALLREDUCE_SLICESTEPS)               │       │
│  │  SlicePerChunk = ChunkSteps / SliceSteps = 2         │       │
│  └─────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、核心概念定义

### 2.1 关键参数定义

| 参数名 | 默认值 | 计算公式 | 含义 |
|--------|--------|----------|------|
| `NCCL_STEPS` | 8 | - | FIFO队列深度 |
| `ALLREDUCE_CHUNKSTEPS` | 4 | `NCCL_STEPS/2` | 每个chunk包含的step数 |
| `ALLREDUCE_SLICESTEPS` | 2 | `NCCL_STEPS/4` | 每个slice包含的step数 |
| `SlicePerChunk` | 2 | `CHUNKSTEPS/SLICESTEPS` | 每个chunk中的slice数 |

### 2.2 核心数据结构

```c
// Device端工作结构体 (device.h)
struct alignas(16) ncclDevWorkColl {
    void* sendbuff;
    void* recvbuff;
    uint32_t channelLo;      // 起始channel ID
    uint32_t channelHi;      // 结束channel ID
    
    // Channel数据切分参数 (CBD = Channel Block Distribution)
    struct {
        uint32_t countLo;       // 第一个channel处理的数据量
        uint32_t countMid;      // 中间channels每个处理的数据量
        uint32_t countHi;       // 最后一个channel处理的数据量
        uint32_t chunkGrainsLo; // 第一个channel的chunk粒度
        uint32_t chunkGrainsMid;// 中间channels的chunk粒度
        uint32_t chunkGrainsHi; // 最后一个channel的chunk粒度
    } cbd;
    
    // CollNet专用参数
    struct {
        uint64_t count;         // 总数据量
        uint32_t chunkCount;    // 每个chunk的元素数
    } collnet;
};
```

---

## 三、Level 1: Host-Level Channel 切分详解

### 3.1 切分流程

在`enqueue.cc`的`scheduleCollTasksToPlan`函数中完成channel分配：

```c
// 1. 计算每个cell的大小和channel数量
size_t cellSize = divUp(divUp(MinTrafficPerChannel, trafficPerByte), 16) * 16;
int elementsPerCell = cellSize / elementSize;
size_t cells = divUp(task->count * elementSize, cellSize);

// 2. 计算每个channel应处理的cells数
size_t cellsPerChannel = min(cells, divUp(trafficPerChannel, trafficPerCell));

// 3. 分配cells到channels（三种情况）
size_t cellsLo;  // 第一个channel
size_t cellsHi;  // 最后一个channel  
int nMidChannels; // 中间channels数量
```

### 3.2 切分策略图解

```
总数据量: 1000 elements, 使用 4 channels

均匀分配情况:
┌────────────────────────────────────────────────────────┐
│  Channel 0  │  Channel 1  │  Channel 2  │  Channel 3   │
│  250 elems  │  250 elems  │  250 elems  │  250 elems   │
│  countLo    │  countMid   │  countMid   │  countHi     │
└────────────────────────────────────────────────────────┘

非均匀分配情况（边界优化）:
┌────────────────────────────────────────────────────────┐
│  Channel 0  │  Channel 1  │  Channel 2  │  Channel 3   │
│  200 elems  │  250 elems  │  250 elems  │  300 elems   │
│  countLo    │  countMid   │  countMid   │  countHi     │
└────────────────────────────────────────────────────────┘
```

### 3.3 关键代码解析

```c
// enqueue.cc: scheduleCollTasksToPlan 函数
// 确定三个数据区的大小
countLo = cellsLo * elementsPerCell;
countMid = nMidChannels != 0 ? cellsPerChannel * elementsPerCell : 0;
countHi = cellsHi * elementsPerCell;

// 调整最后一个channel的数据量以补偿舍入误差
(countHi != 0 ? countHi : countLo) -= cells * elementsPerCell - task->count;

// 设置到work结构体
devWork->cbd.countLo = countLo;
devWork->cbd.countMid = countMid;
devWork->cbd.countHi = countHi;
```

---

## 四、Level 2: Device-Level Chunk 切分详解

### 4.1 ncclCollCbdPart 函数解析

这是**最核心的设备端切分函数**，位于`device.h`：

```c
template<typename Int>
__host__ __device__ inline void ncclCollCbdPart(
    struct ncclDevWorkColl* work, 
    uint32_t channelId,    // 当前channel ID
    int proto,             // 协议类型 (SIMPLE/LL/LL128)
    int eltSize,           // 元素大小
    Int* count,            // 输出: 总元素数
    Int* partOffset,       // 输出: 本channel的数据偏移
    Int* partCount,        // 输出: 本channel负责的数据量
    Int* chunkCount        // 输出: 每个chunk的元素数
) {
    int eltPerGrain = ncclProtoGrainSize(proto) / eltSize;
    int nMidChannels = work->channelHi - work->channelLo - 1;
    
    // 计算总数据量
    if (count != nullptr) {
        *count = work->cbd.countLo + work->cbd.countMid * nMidChannels + work->cbd.countHi;
    }
    
    // 根据channel位置分配数据
    if (channelId == work->channelLo) {
        // 第一个channel
        *partOffset = 0;
        *partCount = work->cbd.countLo;
        *chunkCount = work->cbd.chunkGrainsLo * eltPerGrain;
    } else if (channelId == work->channelHi) {
        // 最后一个channel
        *partOffset = work->cbd.countLo + nMidChannels * work->cbd.countMid;
        *partCount = work->cbd.countHi;
        *chunkCount = work->cbd.chunkGrainsHi * eltPerGrain;
    } else {
        // 中间channel
        int mid = channelId - work->channelLo - 1;
        *partOffset = work->cbd.countLo + mid * work->cbd.countMid;
        *partCount = work->cbd.countMid;
        *chunkCount = work->cbd.chunkGrainsMid * eltPerGrain;
    }
}
```

### 4.2 Ring算法的Loop结构

```c
// all_reduce.h: runRing 函数
void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    int ringIx = ring->index;           // 本rank在ring中的索引
    const int nranks = ncclShmem.comm.nRanks;
    
    ssize_t gridOffset;      // 本channel的全局偏移
    ssize_t channelCount;    // 本channel负责的总数据量
    ssize_t chunkCount;      // 每个chunk的大小
    
    // 获取本channel的数据分配
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), 
                    (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    
    // Ring的关键参数
    const ssize_t loopCount = nranks * chunkCount;  // 一个完整loop的大小
    
    // 外层循环：处理所有loop
    for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        ssize_t remCount = channelCount - elemOffset;
        
        // 最后一个loop可能需要调整chunkCount
        if (remCount < loopCount) {
            chunkCount = alignUp(divUp(remCount, nranks), 16/sizeof(T));
        }
        
        // 内层循环：Ring的2*(nranks-1)个steps
        // ... (详见下文)
    }
}
```

### 4.3 Chunk分配可视化

```
假设: 4个ranks (0,1,2,3), channelCount=1024, chunkCount=256

一个Loop的数据分布 (loopCount = 4 * 256 = 1024):
┌────────────────────────────────────────────────────────────┐
│  Chunk 0    │  Chunk 1    │  Chunk 2    │  Chunk 3         │
│  [0..255]   │  [256..511] │  [512..767] │  [768..1023]     │
│  Rank 0发送  │  Rank 1发送  │  Rank 2发送  │  Rank 3发送       │
└────────────────────────────────────────────────────────────┘

每个Rank按索引处理对应的chunk:
- Rank 0: 从chunk 3开始 (ringIx + nranks - 1 = 0 + 4 - 1 = 3 % 4 = 3)
- Rank 1: 从chunk 2开始
- Rank 2: 从chunk 1开始
- Rank 3: 从chunk 0开始
```

---

## 五、Level 3: Protocol-Level Step/Slice 切分详解

### 5.1 Step、Slice与Chunk的关系

```
┌────────────────────────────────────────────────────────────────┐
│                        Channel Count                           │
│  ┌────────────────────────────────────────────────────────┐   │
│  │                      Loop 0                            │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │   │
│  │  │   Chunk 0    │ │   Chunk 1    │ │   Chunk N    │   │   │
│  │  │  (256 elems) │ │  (256 elems) │ │  (256 elems) │   │   │
│  │  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘   │   │
│  │         │                │                │            │   │
│  │    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐      │   │
│  │    ▼         ▼      ▼         ▼      ▼         ▼      │   │
│  │  Slice 0  Slice 1  Slice 2  Slice 3  Slice 4  Slice 5  │   │
│  │  (128B)   (128B)   (128B)   (128B)   (128B)   (128B)  │   │
│  │     │        │        │        │        │        │    │   │
│  │     ▼        ▼        ▼        ▼        ▼        ▼    │   │
│  │   Step 0   Step 1   Step 2   Step 3   Step 4   Step 5 │   │
│  └────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘

关系总结:
- 1 Channel = N Loops
- 1 Loop = nranks Chunks
- 1 Chunk = ChunkSteps(4) Steps = SlicePerChunk(2) Slices
- 1 Step = 异步通信原语的一次操作
- 1 Slice = min(nelem/SlicePerChunk, stepSize/32) 的实际数据
```

### 5.2 Slice大小计算

在`prims_simple.h`的`genericOp`函数中：

```c
// stepSize是FIFO队列中一个step的字节数 (buffSizes[proto]/NCCL_STEPS)
int stepSize = ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;

// sliceSize是实际每次传输的数据量
// 它取以下两者的较大值：
// 1. nelem / (16 * SlicePerChunk) * 16  - 将数据均匀分配到各slices
// 2. stepSize / 32                      - 最小传输粒度
int sliceSize = stepSize * StepPerSlice;
sliceSize = max(divUp(nelem, 16 * SlicePerChunk) * 16, sliceSize / 32);
```

### 5.3 Step与Slice的动态关系

```c
// prims_simple.h
#pragma unroll SlicePerChunk  // SlicePerChunk = 2
do {
    // 调整最后一个slice的大小
    sliceSize = min(sliceSize, nelem - offset);
    
    // 1. 等待peer就绪 (waitPeer)
    waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(srcIx, dstIx, offset, sliceSize);
    
    // 2. 线程内同步
    subBarrier();
    
    // 3. 执行实际的reduce/copy操作
    reduceCopy<...>(tid, nworkers, redOpArgs, postOp, 
                    nrecv, srcs, nsend, dsts, workSize);
    
    // 4. 跨线程同步
    barrier();
    
    // 5. 通知peer完成 (postPeer)
    postPeer<Recv, Send>(0 < workSize);
    
    offset += sliceSize;
    slice += 1;
} while (slice < SlicePerChunk && offset < nelem);
```

---

## 六、Ring AllReduce 通信Pipeline详解

### 6.1 Ring算法的数学模型

对于N个ranks的Ring AllReduce：
- **总steps**: `2 * (N - 1)`
- **Reduce阶段**: steps 0 到 N-2 (共N-1步)
- **Broadcast阶段**: steps N-1 到 2N-3 (共N-1步)

### 6.2 Pipeline执行流程

```
Ring AllReduce Pipeline (4 ranks示例)

初始状态:
Rank 0: [A0] [A1] [A2] [A3]
Rank 1: [B0] [B1] [B2] [B3]
Rank 2: [C0] [C1] [C2] [C3]
Rank 3: [D0] [D1] [D2] [D3]

Reduce-Scatter阶段 (Step 0~2):

Step 0 (push to next):
  Rank 0发送A3 -> Rank 1
  Rank 1发送B2 -> Rank 2
  Rank 2发送C1 -> Rank 3
  Rank 3发送D0 -> Rank 0

Step 1 (recvReduceSend):
  Rank 0: 接收D0, reduce(A0+D0), 发送给Rank 1
  Rank 1: 接收A3, reduce(B1+A3), 发送给Rank 2
  Rank 2: 接收B2, reduce(C2+B2), 发送给Rank 3
  Rank 3: 接收C1, reduce(D3+C1), 发送给Rank 0

Step 2 (recvReduceSend):
  Rank 0: 接收C1, reduce(A0+D0+C1), 发送给Rank 1
  Rank 1: 接收D0, reduce(B1+A3+D0), 发送给Rank 2
  Rank 2: 接收A3, reduce(C2+B2+A3), 发送给Rank 3
  Rank 3: 接收B2, reduce(D3+C1+B2), 发送给Rank 0

Step 3 (recvReduceCopySend - 完成reduce):
  Rank 0: 接收B2, reduce(A0+D0+C1+B2=A0+B0+C0+D0), 复制到输出, 发送
  ... (其他rank类似)

此时每个rank持有部分reduce结果:
Rank 0: [A0+B0+C0+D0] [    ] [    ] [    ]
Rank 1: [    ] [A1+B1+C1+D1] [    ] [    ]
Rank 2: [    ] [    ] [A2+B2+C2+D2] [    ]
Rank 3: [    ] [    ] [    ] [A3+B3+C3+D3]

AllGather阶段 (Step 4~6):

Step 4~5 (recvCopySend):
  每个rank接收并转发其他rank的reduce结果

Step 6 (recv - 最后一步):
  所有rank收集到完整的reduce结果

最终状态 (所有rank都有相同结果):
Rank 0~3: [A0+B0+C0+D0] [A1+B1+C1+D1] [A2+B2+C2+D2] [A3+B3+C3+D3]
```

### 6.3 代码实现详解

```c
// all_reduce.h: runRing函数核心逻辑
template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    int ringIx = ring->index;
    const int nranks = ncclShmem.comm.nRanks;
    
    ssize_t gridOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), 
                    (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    const ssize_t loopCount = nranks * chunkCount;
    
    // 创建primitives实例，用于实际的通信操作
    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims(
        tid, nthreads, &ring->prev, &ring->next, 
        work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work
    );
    
    // 外层循环：处理所有数据loop
    for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        ssize_t remCount = channelCount - elemOffset;
        if (remCount < loopCount) {
            chunkCount = alignUp(divUp(remCount, nranks), 16/sizeof(T));
        }
        
        auto modRanks = [&]__device__(int r)->int {
            return r - (r >= nranks ? nranks : 0);
        };
        
        // ====== Reduce-Scatter阶段 ======
        
        // Step 0: 发送初始数据到下一个rank
        chunk = modRanks(ringIx + nranks - 1);
        chunkOffset = chunk * chunkCount;
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = min(chunkCount, remCount - chunkOffset);
        prims.directSend(offset, offset, nelem);
        
        // Steps 1 ~ nranks-2: 接收、reduce、转发
        for (int j = 2; j < nranks; ++j) {
            chunk = modRanks(ringIx + nranks - j);
            chunkOffset = chunk * chunkCount;
            offset = gridOffset + elemOffset + chunkOffset;
            nelem = min(chunkCount, remCount - chunkOffset);
            prims.directRecvReduceDirectSend(offset, offset, nelem);
        }
        
        // Step nranks-1: 最终reduce，生成结果并转发
        chunk = ringIx + 0;
        chunkOffset = chunk * chunkCount;
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = min(chunkCount, remCount - chunkOffset);
        prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);
        
        // ====== AllGather阶段 ======
        
        // Steps nranks ~ 2*nranks-3: 接收并转发结果
        for (int j = 1; j < nranks - 1; ++j) {
            chunk = modRanks(ringIx + nranks - j);
            chunkOffset = chunk * chunkCount;
            offset = gridOffset + elemOffset + chunkOffset;
            nelem = min(chunkCount, remCount - chunkOffset);
            prims.directRecvCopyDirectSend(offset, offset, nelem);
        }
        
        // Final Step: 接收最后的数据
        chunk = modRanks(ringIx + 1);
        chunkOffset = chunk * chunkCount;
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = min(chunkCount, remCount - chunkOffset);
        prims.directRecv(offset, nelem);
    }
}
```

---

## 七、Tree AllReduce 数据流详解

### 7.1 TreeUpDown算法

```c
// all_reduce.h: runTreeUpDown函数
template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runTreeUpDown(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclTree *tree = &ncclShmem.channel.tree;
    size_t gridOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), 
                    (size_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    
    // ====== Reduce Up阶段 ======
    {
        // 使用FanAsymmetric<3,1>: 最多3个接收，1个发送
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, 1, Proto, 0> prims(
            tid, nthreads, tree->down, &tree->up, 
            work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work
        );
        
        if (tree->up == -1) {
            // Root节点: 从子节点接收并reduce到输出
            for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
                offset = gridOffset + elemOffset;
                nelem = min(chunkCount, channelCount - elemOffset);
                prims.directRecvReduceCopy(offset, offset, nelem, /*postOp=*/true);
            }
        } else if (tree->down[0] == -1) {
            // Leaf节点: 发送数据到父节点
            for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
                offset = gridOffset + elemOffset;
                nelem = min(chunkCount, channelCount - elemOffset);
                prims.directSend(offset, offset, nelem);
            }
        } else {
            // Internal节点: 接收子节点数据，reduce后转发给父节点
            for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
                offset = gridOffset + elemOffset;
                nelem = min(chunkCount, channelCount - elemOffset);
                prims.directRecvReduceDirectSend(offset, offset, nelem);
            }
        }
    }
    
    // ====== Broadcast Down阶段 ======
    {
        // 使用FanAsymmetric<1,3>: 1个接收，最多3个发送
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, 1, Proto, 0> prims(
            tid, nthreads, &tree->up, tree->down, 
            work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work
        );
        
        if (tree->up == -1) {
            // Root节点: 发送结果到子节点
            for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
                offset = gridOffset + elemOffset;
                nelem = min(chunkCount, channelCount - elemOffset);
                prims.directSendFromOutput(offset, nelem);
            }
        } else if (tree->down[0] == -1) {
            // Leaf节点: 从父节点接收结果
            for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
                offset = gridOffset + elemOffset;
                nelem = min(chunkCount, channelCount - elemOffset);
                prims.directRecv(offset, nelem);
            }
        } else {
            // Internal节点: 接收父节点结果并转发给子节点
            for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
                offset = gridOffset + elemOffset;
                nelem = min(chunkCount, channelCount - elemOffset);
                prims.directRecvCopyDirectSend(offset, offset, nelem);
            }
        }
    }
}
```

### 7.2 TreeSplit算法

TreeSplit通过**线程分割**实现并行化的Reduce和Broadcast：

```
Thread分配 (假设nthreads=512):
┌─────────────────────────────────────────────────────────┐
│  Threads 0~255 (70%): Reduce Up组                        │
│  - 从子节点接收并reduce                                   │
│  - 发送给父节点                                          │
├─────────────────────────────────────────────────────────┤
│  Threads 256~511 (30%): Broadcast Down组                 │
│  - 从父节点接收                                          │
│  - 转发给子节点                                          │
└─────────────────────────────────────────────────────────┘

并行执行:
┌──────────────┐     ┌──────────────┐
│ Reduce Up    │     │ Broadcast    │
│              │     │ Down         │
│  [========]  │ <-- │  [========]  │
│              │     │              │
└──────────────┘     └──────────────┘
      │                      │
      ▼                      ▼
  向上发送              向下广播
```

---

## 八、CollNet/NVLS 数据切分详解

### 8.1 CollNet Direct 架构

```
数据流: Scatter -> Reduce -> (Network) -> Gather/Broadcast

┌────────────────────────────────────────────────────────────────────┐
│                        CollNet Direct                              │
├────────────────────────────────────────────────────────────────────┤
│  Node 0                   Node 1                   Node 2          │
│  ┌──────────┐             ┌──────────┐             ┌──────────┐   │
│  │ GPU 0    │             │ GPU 0    │             │ GPU 0    │   │
│  │ ┌──────┐ │             │ ┌──────┐ │             │ ┌──────┐ │   │
│  │ │Head 0│◄├──────┬──────►│Head 0│◄├──────┬──────►│Head 0│◄├─┐ │   │
│  │ └──┬───┘ │      │      └──┬───┘ │      │      └──┬───┘ │ │ │   │
│  │    │     │      │         │     │      │         │     │ │ │   │
│  │ ┌──┴───┐ │      │      ┌──┴───┐ │      │      ┌──┴───┐ │ │ │   │
│  │ │GPU 1 │ │      │      │GPU 1 │ │      │      │GPU 1 │ │ │ │   │
│  │ │GPU 2 │ │      │      │GPU 2 │ │      │      │GPU 2 │ │ │ │   │
│  │ │GPU 3 │─┘      │      │GPU 3 │─┘      │      │GPU 3 │─┘ │ │   │
│  │ └──────┘        │      └──────┘        │      └──────┘   │ │   │
│  └─────────────────┘                      │                 │ │   │
│                    │                      │                 │ │   │
│                    └──────────────────────┴─────────────────┘ │   │
│                                                               │   │
│                          Network Switch                       │   │
└───────────────────────────────────────────────────────────────┼───┘
                                                                │
                                                                ▼
```

### 8.2 NVLS (NVLink SHARP) 数据流

```c
// all_reduce.h: NVLS算法核心逻辑
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
    __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
        struct ncclNvls* nvls = &ncclShmem.channel.nvls;
        
        // 线程分组：Scatter + Reduce + Gather/Broadcast
        const int nThreadsScatter = scatterWarps * WARP_SIZE;
        const int nThreadsGather  = gatherWarps * WARP_SIZE;
        const int nThreadsReduce  = reduceWarps * WARP_SIZE;
        const int nThreadsBcast   = bcastWarps * WARP_SIZE;
        
        // 1. Scatter阶段: 将数据分散到NVLS域
        if (tid < nThreadsScatter) {
            Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, 0, Proto, 0> prims(...);
            for (...) {
                prims.scatter(offset, nelem, chunkSize, chunkSize, -1, 0);
            }
        }
        // 2. Gather阶段: 从NVLS域收集数据
        else if (tid < nThreadsScatter + nThreadsGather) {
            Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, 0, Proto, 0> prims(...);
            for (...) {
                prims.gather(offset, nelem, chunkSize, chunkSize, -1, 0);
            }
        }
        // 3. Reduce阶段: NVLS硬件reduce
        else if (tid < nThreadsScatter + nThreadsGather + nThreadsReduce && nvls->headRank != -1) {
            Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims(...);
            for (...) {
                prims.directRecvDirectSend(offset, offset, nelem);
            }
        }
        // 4. Broadcast阶段 (多节点时)
        else if (nvls->headRank != -1) {
            // 接收网络数据并广播
        }
    }
};
```

---

## 九、Chunk Size 计算策略

### 9.1 calcCollChunking 函数详解

```c
// enqueue.cc: 计算chunk size的核心函数
static ncclResult_t calcCollChunking(
    struct ncclComm* comm, 
    struct ncclTaskColl* info, 
    int nChannels, 
    size_t nBytes,
    uint32_t* outChunkSize, 
    uint32_t* outDirectFlags, 
    struct ncclProxyOp* proxyOp
) {
    // 基础参数
    int stepSize = comm->buffSizes[info->protocol] / NCCL_STEPS;
    int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) 
                     ? info->chunkSteps : 1;
    int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) 
                     ? info->sliceSteps : 1;
    int chunkSize = stepSize * chunkSteps;
    
    // 根据算法优化chunk size
    if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
        // 根据数据量和并发度动态调整
        while (nBytes / (nChannels * collnetDirect.nHeads * chunkSize) < depth * 64 
               && chunkSize > 131072) {
            chunkSize /= 2;
        }
    } else if (info->algorithm == NCCL_ALGO_NVLS) {
        // NVLS有最大chunk size限制
        int maxChunkSize = comm->nvlsChunkSize;
        if (chunkSize > maxChunkSize) chunkSize = maxChunkSize;
        
        // 小数据量时使用更小的chunk
        uint64_t concurrentOps = nChannels * nvls.nHeads;
        if (nBytes < 64 * concurrentOps * chunkSize && chunkSize > 65536) {
            chunkSize = 65536;
        }
    }
    
    // 对齐到grain size
    size_t grainSize = ncclProtoGrainSize(info->protocol);
    chunkSize = chunkSize / grainSize * grainSize;
    
    // 计算pattern相关的steps
    switch (pattern) {
    case ncclPatternRing:
        nstepsPerLoop = nranks - 1;
        nchunksPerLoop = nranks;
        break;
    case ncclPatternRingTwice:  // AllReduce
        nstepsPerLoop = 2 * (nranks - 1);
        nchunksPerLoop = nranks;
        break;
    case ncclPatternTreeUpDown:
        nstepsPerLoop = nchunksPerLoop = 1;
        break;
    case ncclPatternNvls:
        nstepsPerLoop = 1;
        nchunksPerLoop = nvls.nHeads;
        break;
    }
    
    // 计算总steps
    size_t loopSize = nChannels * nchunksPerLoop * chunkSize;
    int nLoops = DIVUP(nBytes, loopSize);
    proxyOp->nsteps = nstepsPerLoop * nLoops * chunkSteps;
    proxyOp->sliceSteps = sliceSteps;
    proxyOp->chunkSteps = chunkSteps;
    proxyOp->chunkSize = chunkSize;
    proxyOp->sliceSize = chunkSize / chunkSteps * sliceSteps;
    
    *outChunkSize = chunkSize;
}
```

### 9.2 不同算法的Chunk Size策略

| 算法 | 默认Chunk Size | 调整策略 |
|------|---------------|----------|
| Ring | stepSize × 4 | 固定，不动态调整 |
| Tree | stepSize × 4 | 固定，不动态调整 |
| CollNet Direct | 动态 | 根据`nBytes/(nChannels×nHeads×chunkSize)`与`depth×64`的关系调整 |
| NVLS | min(nvlsChunkSize, 动态) | 小数据量时减小到64KB/32KB/16KB |
| NVLS Tree | 动态 | 根据数据量使用64KB~256KB |

---

## 十、总结与关键洞察

### 10.1 数据切分层次总结

```
┌─────────────────────────────────────────────────────────────────┐
│                    NCCL AllReduce 数据切分架构                    │
├─────────────────────────────────────────────────────────────────┤
│  Host侧 (CPU)                                                   │
│  ├── 1. 任务收集与聚合 (Task Aggregation)                        │
│  ├── 2. 算法选择 (Ring/Tree/CollNet/NVLS)                        │
│  └── 3. Channel分配 (countLo/Mid/Hi)                            │
│                    ↓                                              │
│  设备侧 (GPU) 每个Channel独立执行                                │
│  ├── 4. 数据分区计算 (ncclCollCbdPart)                           │
│  │   └── partOffset, partCount, chunkCount                      │
│  ├── 5. Loop迭代 (channelCount / loopCount)                     │
│  │   └── 每个Loop处理 nranks × chunkCount 数据                  │
│  └── 6. Step/Slice执行 (Primitives)                              │
│      └── ChunkSteps(4) × SlicePerChunk(2) 次操作                │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Pipeline优化关键点

1. **双缓冲机制**: `NCCL_STEPS=8` 实现异步流水，允许同时有多个in-flight的通信操作

2. **ChunkSteps vs SliceSteps**: 
   - `ChunkSteps=4` 控制FIFO队列的占用
   - `SliceSteps=2` 提供更细的粒度用于负载均衡
   - `SlicePerChunk=2` 在每个chunk内实现小步快跑

3. **线程分工**:
   - **Worker threads**: 执行实际的reduce/copy操作
   - **Wait threads**: 等待peer就绪
   - **Post threads**: 通知peer完成

4. **Direct Memory Access**:
   - 支持直接读写peer的GPU内存（通过P2P或NVLink）
   - 避免通过FIFO缓冲区的中转，减少内存拷贝

### 10.3 性能调优建议

```
1. 大数据量 ( > 1GB ):
   - 使用Ring算法
   - 启用所有channels
   - chunkSize使用默认值

2. 中等数据量 ( 100MB ~ 1GB ):
   - 考虑NVLS (如果硬件支持)
   - 调整chunkSize匹配网络带宽延迟积

3. 小数据量 ( < 100MB ):
   - 使用Tree或NVLS算法
   - 减少channel数量以避免overhead
   - 考虑LL/LL128协议降低延迟

4. 多节点场景:
   - 使用CollNet Direct或NVLS+网络
   - 调整chunkSize匹配网络带宽
   - 启用GDR (GPUDirect RDMA)
```

---

**文档版本**: 1.0  
**分析源码**: NCCL 2.x (src/device/all_reduce.h, src/enqueue.cc, src/include/device.h)  
**核心文件**: 
- `src/device/all_reduce.h` - AllReduce设备端实现
- `src/device/prims_simple.h` - Simple协议原语实现
- `src/enqueue.cc` - Host端任务调度和数据切分
- `src/include/device.h` - 核心数据结构定义
- `src/include/collectives.h` - 算法参数定义
