# NCCL AllReduce 数据全链路追踪

## 1. 总览：一次 AllReduce 的完整旅程

用户调用 `ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)`，数据经历以下阶段：

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. API 入口 & 参数校验                                               │
│ 2. 任务入队（Task Append）                                           │
│ 3. Group 提交（算法选择 + 通道分配）                                    │
│ 4. 内存注册 & 连接建立                                                │
│ 5. GPU Kernel 启动（数据拆分 + 计算）                                  │
│ 6. Primitives 层（收发同步 + 直接/间接数据搬运）                         │
│ 7. 代理服务（Proxy Thread → 网络传输）                                 │
│ 8. net_ib 数据路径（CTS → RDMA Write → 完成）                         │
│ 9. 结果产出                                                          │
└─────────────────────────────────────────────────────────────────────┘
```

下面逐层展开。

---

## 2. 第一层：API 入口

**文件**: `src/collectives.cc`

```c
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct ncclInfo info = {
    ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream,
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS  // 默认 chunk/slice 配置
  };
  return ncclEnqueueCheck(&info);
}
```

`ncclInfo` 封装了这次调用的所有参数。关键常量：
- `ALLREDUCE_CHUNKSTEPS = 1`（每个 chunk 包含的 step 数）
- `ALLREDUCE_SLICESTEPS = 1`（每个 slice 包含的 step 数）

---

## 3. 第二层：Enqueue — 任务入队

**文件**: `src/enqueue.cc`

### 3.1 `ncclEnqueueCheck(info)`

1. **Comm 校验**: 检查 communicator 是否有效、是否被 revoke
2. **隐式 Group**: 自动包裹 `ncclGroupStartInternal()` / `ncclGroupEndInternal()`
3. **参数校验**: `ArgsCheck(info)` — 类型、大小、buffer 地址合法性
4. **任务入队**: `taskAppend(comm, info)` — 将此次调用转化为内部任务

### 3.2 `taskAppend(comm, info)`

对于 AllReduce（非 Send/Recv/PutSignal），走 `collTaskAppend` 路径：

1. **单 rank 优化**: 如果 `nRanks == 1`，直接做本地 memcpy + reduce，不走网络
2. **CE（Copy Engine）检查**: 检查是否可用 CE 做集合通信（Blackwell+ 平台，大消息 AllGather）
3. **集合任务创建**: 走 `collTaskAppend(comm, info, opDev)`

### 3.3 `collTaskAppend(comm, info, opDev)`

创建 `ncclTaskColl` 结构：
- 记录 `func=ncclFuncAllReduce`, `sendbuff`, `recvbuff`, `count`, `datatype`, `opDev`
- 计算 `trafficBytes = count × elementSize × 2`（AllReduce 每字节产生 2 倍流量）
- 按流量大小插入 `collSorter`（降序排列，大任务优先调度）

---

## 4. 第三层：Group 提交 — 算法选择与通道分配

**文件**: `src/group.cc` + `src/enqueue.cc`

当 `ncclGroupEndInternal()` 被调用（depth 归零时）：

### 4.1 整体流程

```
ncclGroupEndInternal()
  → groupLaunch()
    → ncclPrepareTasksAndCollPreconnect(comm)
      → ncclPrepareTasks(comm)          // 算法选择 + 通道分配
    → ncclTasksRegAndEnqueue(comm)      // 内存注册 + Work 构建 + Kernel 启动
```

### 4.2 `ncclPrepareTasks` — 核心调度

#### 4.2.1 任务排序与聚合

1. 从 `collSorter` 取出按流量降序排列的任务列表
2. 按 `(func, op, datatype)` 三元组分桶（`tasksByFnOpTy`）
3. 同桶内相邻任务如果流量比 < 4x，则**聚合**（合并 count，统一调度）

#### 4.2.2 算法选择 (`ncclGetAlgoInfo`)

对每个聚合后的任务，选择 **算法（Algorithm）** 和 **协议（Protocol）**：

| 算法 | 说明 | 适用场景 |
|------|------|---------|
| **RING** | 环形 AllReduce | 通用场景，默认首选 |
| **TREE** | 二叉树 AllReduce | 节点数多、小消息场景 |
| **NVLS** | NVLink + SHARP（NVLink 直连） | 单节点 NVLink + 多节点网络 |
| **NVLS_TREE** | NVLink + 树形 | 单节点内 NVLS + 跨节点树 |
| **COLLNET_DIRECT** | SHARP 网络内归约 | InfiniBand SHARP 支持 |
| **COLLNET_CHAIN** | SHARP 链式 | SHARP 支持，备选 |

| 协议 | 说明 | 适用场景 |
|------|------|---------|
| **SIMPLE** | 直接缓冲区传输 | 大消息（默认） |
| **LL** | Low-Latency，64 字节行 | 极小消息（<4KB） |
| **LL128** | 128 字节行 | 小消息（4KB~32KB） |

选择依据（`tuning.cc` + `ncclGetAlgoInfo`）：
- **数据量**: 小数据走 Tree/LL，大数据走 Ring/Simple
- **节点数**: 单节点优先 NVLS，多节点优先 Ring + 网络
- **硬件能力**: NVLink 拓扑、SHARP 是否可用、GPU Direct 支持
- **通道数**: 影响并行度，由 `NCCL_MAX_NCHANNELS` 和 `NCCL_MIN_NCHANNELS` 控制

#### 4.2.3 通道分配

每个任务分配 `[channelLo, channelHi]` 范围的通道：
- 通道数 = `min(nMaxChannels, 可用通道数)`
- `nMaxChannels` 由算法+数据量决定
- 多个通道并行传输同一段数据的不同分片

### 4.3 `ncclTasksRegAndEnqueue` — Work 构建与 Kernel 启动

1. **内存注册**: 对 sendbuff/recvbuff 调用 `ncclRegFind` → `ncclNet.regMr`
2. **构建 `ncclDevWorkColl`**: 将任务转化为设备端 Work 结构
   - 包含 `sendbuffOffset`, `recvbuffOffset`, `count`, `channelLo/Hi`
   - 计算 **CBD（Channel-Based Distribution）** 参数：`countLo/Mid/Hi`, `chunkGrainsLo/Mid/Hi`
3. **添加 Proxy 操作**: `ncclAddProxyOpIfNeeded` — 为需要网络传输的通道注册 proxy op
4. **构建 Work Batch**: `ncclAddWorkBatchToPlan` — 将 work 添加到 kernel plan
5. **启动 CUDA Kernel**: `ncclLaunchKernel` — 最终触发 GPU kernel

---

## 5. 第四层：GPU Kernel 执行

**文件**: `src/device/all_reduce.h` + `src/device/common_kernel.h`

### 5.1 Kernel 入口

NCCL 的 collective kernel 是一个统一的 CUDA kernel（`ncclDevKernel`），每个 channel 一个 block。kernel 从 work FIFO 中取出 `ncclDevWorkColl`，根据 `devFuncId` 路由到对应的模板实例化：

```c++
// 编译时生成的路由表
template<int Algo, int Proto>
RunWorkColl<ncclFuncAllReduce, T, RedOp, Algo, Proto>::run(tid, nthreads, work);
```

### 5.2 Ring AllReduce 的数据流转（最常见路径）

**核心函数**: `runRing<T, RedOp, Proto>(tid, nthreads, work)`

假设 N 个 rank，环形排列为 `rank[0] → rank[1] → ... → rank[N-1] → rank[0]`。

数据被分为 N 个 chunk，经过 2×(N-1) 步完成：

```
数据布局：| chunk_0 | chunk_1 | ... | chunk_{N-1} |
          ↑ 每个 chunk 包含 count/N 个元素

步骤（以 rank[i] 视角，ringIx = i）：

第 0 步（Push）：发送自己的 chunk_{(i-1)%N} 给下一个 rank
第 1~N-2 步（Reduce+Forward）：接收上家的数据，与本地 chunk 做 reduce，发给下家
第 N-1 步（Final Reduce+Send）：接收并 reduce，得到最终结果，发给下家
第 N~2N-3 步（Copy Forward）：接收最终结果并转发给下家
最后：接收属于自己的最终结果
```

具体代码逻辑（以 RING + SIMPLE 协议为例）：

```c++
// 1. 计算本 channel 负责的数据范围
ncclCollCbdPart(work, channelId, Proto, sizeof(T),
    nullptr, &gridOffset, &channelCount, &chunkCount);
// gridOffset: 本 channel 在全局数据中的起始偏移
// channelCount: 本 channel 负责的总元素数
// chunkCount: 每个 chunk 的元素数 = channelCount / nranks

// 2. 创建 Primitives 对象，连接到 ring 的 prev/next
Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
  (tid, nthreads, &ring->prev, &ring->next,
   sendbuff, recvbuff, redOpArg, 0, 0, 0, work);

// 3. 分步执行
for (elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
  // Step 0: 发送 chunk[(ringIx+N-1)%N]（自己的前一个 chunk）到 next
  chunk = modRanks(ringIx + N - 1);
  prims.directSend(offset, offset, nelem);

  // Steps 2..N-1: 接收 + reduce + 发送
  for (j = 2; j < N; j++) {
    chunk = modRanks(ringIx + N - j);
    prims.directRecvReduceDirectSend(offset, offset, nelem);
  }

  // Step N-1: 接收 + reduce + copy 到 output + 发送（最终结果）
  chunk = ringIx;
  prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);

  // Steps N..2N-3: 接收最终结果 + 转发
  for (j = 1; j < N-1; j++) {
    chunk = modRanks(ringIx + N - j);
    prims.directRecvCopyDirectSend(offset, offset, nelem);
  }

  // Final: 接收属于自己的最终结果
  chunk = modRanks(ringIx + 1);
  prims.directRecv(offset, nelem);
}
```

**Ring AllReduce 总步骤 = 2×(N-1)**，每步传输 `chunkCount` 个元素。

### 5.3 Tree AllReduce 的数据流转

**核心函数**: `runTreeSplit<T, RedOp, Proto>(tid, nthreads, work)`

二叉树结构，每个节点最多 3 个子节点（`NCCL_MAX_TREE_ARITY=3`）。

分两个阶段，线程分为两半：

**阶段 1 — Reduce Up（前半线程）**：
```
叶子节点 → 发送本地数据给父节点
内部节点 → 接收所有子节点数据 + 本地数据 → reduce → 发给父节点
根节点 → 接收所有子节点数据 + 本地数据 → reduce → 得到最终结果
```

**阶段 2 — Broadcast Down（后半线程）**：
```
根节点 → 发送最终结果给所有子节点
内部节点 → 接收父节点数据 → 发给所有子节点 → 写入 recvbuff
叶子节点 → 接收父节点数据 → 写入 recvbuff
```

总步骤 = 2 × ⌈log₃(N)⌉，延迟显著低于 Ring。

### 5.4 NVLS AllReduce 的数据流转

**核心函数**: `RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE>`

NVLS 利用了 NVLink 的 **multimem（多播内存操作）** 能力，线程分为 4 组：

```
┌──────────────────────────────────────────┐
│ Scatter 线程组: 将 sendbuff 分散到各 NVLink peer │
│ Gather 线程组:  从各 NVLink peer 汇聚到 recvbuff  │
│ Reduce 线程组:  接收 NVLS multicast → reduce     │
│ Bcast 线程组:   接收网络数据 → NVLS multicast    │
└──────────────────────────────────────────┘
```

单节点场景 (`oneNode=true`)：
1. Scatter: 数据通过 NVLink 发送到各 peer 的 scratch buffer
2. Gather: 从各 peer 的 scratch buffer 收集到 recvbuff
3. Reduce: head rank 做 multimem reduce（NVLS multicast reduce）

多节点场景：
- Reduce 组：NVLS reduce 后通过网路发给远端
- Bcast 组：从网络接收后通过 NVLS multicast 分发

---

## 6. 第五层：Primitives — 数据搬运的原子操作

**文件**: `src/device/primitives.h` + `src/device/prims_simple.h`

`Primitives` 模板类是 NCCL 数据路径的核心抽象，封装了所有数据搬运操作。

### 6.1 角色分工

每个线程被分配一个**角色**：

| 角色 | 职责 | 线程范围 |
|------|------|---------|
| **WaitRecv** | 等待远端数据到达（poll `tail` 指针） | `tid < nrecv` |
| **WaitSend** | 等待发送缓冲区空闲（poll `head` 指针） | `nrecv ≤ tid < nrecv+nsend` |
| **PostSend** | 通知远端数据已写入（更新 `tail` 指针） | `nthreads-nsend ≤ tid` |
| **PostRecv** | 通知本地已消费数据（更新 `head` 指针） | `nthreads-nrecv-nsend ≤ tid` |
| **Worker** | 执行实际的 memcpy / reduce 计算 | 剩余线程 |

### 6.2 直接传输（Direct）vs 间接传输

**间接传输**（非 Direct）：
```
GPU kernel → 写入中间缓冲区（ncclShmem buff）→ Proxy 线程读取 → 网络发送
远端 → Proxy 线程接收 → 写入中间缓冲区 → GPU kernel 读取
```

**直接传输**（Direct，GDR/NVLink）：
```
GPU kernel → RDMA Write 直接到远端 GPU 内存（或 NVLink 对端内存）
远端 → 直接从本地 GPU 内存读取
```

### 6.3 `genericOp` — 所有操作的统一模板

以 `directRecvReduceDirectSend` 为例，展开 `genericOp<1, 1, 1, 1, Input, -1>` 的执行流程：

```
每个 Slice（默认 1 个 Slice per Chunk）：

1. Worker 线程：
   ├─ WaitRecv: poll conn->tail 直到 step + StepPerSlice <= tail（数据就绪）
   │   └─ 设置 srcs[0] = 远端 buffer 地址（Direct 时为远端 GPU 地址）
   ├─ WaitSend: poll conn->head 直到 step + NCCL_STEPS <= head（缓冲区空闲）
   │   └─ 设置 dsts[0] = 中间缓冲区 / 远端地址
   ├─ subBarrier(): 确保所有 worker 就绪
   ├─ reduceCopy(): 执行 reduce + copy
   │   ├─ srcs[0]: 从远端接收的数据（或 user input）
   │   ├─ srcs[1]: 本地 user input（如果有 reduce）
   │   └─ dsts[0]: 输出到中间缓冲区或远端
   ├─ barrier(): 同步所有角色
   └─ PostSend/PostRecv: 更新 step 指针，通知对端

2. 非 Worker 线程（纯 Wait/Post 角色）：
   └─ 只做指针 poll 和 step 更新
```

### 6.4 同步机制

数据传输通过 **step 计数器** 同步：

```
发送端维护:
  conn->tail = 已写入的最大 step（PostSend 更新，远端 WaitRecv poll）
  conn->head = 对端已消费的最大 step（WaitSend poll，对端 PostRecv 更新）

接收端维护:
  conn->head = 已消费的最大 step（PostRecv 更新，远端 WaitSend poll）
  conn->tail = 对端已写入的最大 step（WaitRecv poll，对端 PostSend 更新）

流控: 发送端检查 head + NCCL_STEPS > step 才能写入（环形缓冲区有 NCCL_STEPS=8 个槽位）
```

---

## 7. 第六层：代理服务（Proxy Thread）

**文件**: `src/proxy.cc`

### 7.1 为什么需要 Proxy

对于**网络传输**（IB/RoCE），GPU 不能直接发起 RDMA 操作（除非 GDAKI）。Proxy 线程在 CPU 上运行，充当 GPU kernel 和网络之间的桥梁。

### 7.2 Proxy Op 注册

`ncclProxySaveOp(comm, op, &needed)` 根据算法的 pattern 注册需要 proxy 的连接：

| Pattern | 注册的 Proxy 连接 |
|---------|-----------------|
| Ring | `ring.prev`（recv）+ `ring.next`（send） |
| TreeUp | `tree.down[0..2]`（recv）+ `tree.up`（send） |
| TreeDown | `tree.up`（recv）+ `tree.down[0..2]`（send） |
| Nvls | `nvls.out`（send+recv） |
| CollnetChain | `collnetChain.up`（send+recv） |

### 7.3 Proxy Progress 循环

每个 Proxy Op 包含：
- `nSteps`: 需要传输的步数
- `sliceSize`: 每步传输的大小
- `send/recv`: 连接信息

Proxy 线程循环：
```
while (有未完成的 op) {
  for each op:
    if (send needed):
      ncclNet.isend(sendComm, buff, size, ...)   // 发起网络发送
    if (recv needed):
      ncclNet.irecv(recvComm, n, buff, ...)      // 发起网络接收
    test all pending requests:
      ncclNet.test(request, &done, sizes)          // 轮询完成
}
```

### 7.4 间接路径完整数据流

```
GPU Kernel (Worker)
  │
  ├─ 写入 ncclConn.buffs[PROTO_SIMPLE][step % NCCL_STEPS]
  │   (位于 GPU 内存的中间缓冲区)
  │
  ├─ PostSend: st_relaxed_sys_global(conn->tail, step)
  │   (通过 GPU→CPU 机制更新 tail 指针)
  │
  ▼
Proxy Thread (CPU)
  │
  ├─ 检测到 conn->tail 更新
  ├─ ncclNet.isend(sendComm, buff+step*stepSize, sliceSize, tag, mhandle)
  │   (调用 net_ib 的发送接口)
  │
  ▼
net_ib (`ncclIbIsend`)
  │
  ├─ 等待 CTS（接收端通知）
  ├─ RDMA Write 数据到远端 GPU 内存
  ├─ RDMA Write With Imm 通知接收端
  │
  ▼
远端 net_ib (`ncclIbIrecv`)
  │
  ├─ 收到 IBV_WC_RECV_RDMA_WITH_IMM
  ├─ 数据已直接写入远端中间缓冲区
  │
  ▼
远端 Proxy Thread
  │
  ├─ ncclNet.test() 返回 done
  ├─ 更新 conn->head（通知 GPU kernel 数据就绪）
  │
  ▼
远端 GPU Kernel (Worker)
  │
  ├─ WaitRecv: 检测到 conn->tail 更新
  ├─ 从 ncclConn.buffs 中读取数据
  └─ 执行 reduce/copy 操作
```

### 7.5 直接路径（GDR/NVLink P2P）

当 GPU Direct RDMA 或 NVLink P2P 可用时，**绕过 Proxy**：

```
GPU Kernel (Direct Write)
  │
  ├─ 直接写入远端 GPU 内存（通过 CUDA IPC / NVLink / RDMA）
  ├─ 更新 conn->tail
  │
  ▼
远端 GPU Kernel
  │
  ├─ poll conn->tail
  ├─ 直接从本地 GPU 内存读取（数据已被远端 RDMA Write 写入）
  └─ Proxy 只做 control path（转发 step 指针）
```

---

## 8. 第七层：net_ib 网络传输

**详细见 `net_ib.md`**，此处从 AllReduce 视角追踪。

### 8.1 数据拆分到网络路径

一次 AllReduce 的数据可能经过多条路径：
- **多 Channel**: 不同的 CUDA kernel block 处理不同的数据范围
- **每 Channel 一对 Send/Recv Comm**: 通过 `ncclIbConnect` 建立
- **多 QP per Comm**: `NCCL_IB_QPS_PER_CONNECTION` 控制每连接 QP 数

数据拆分层级：
```
全局数据 (count 个元素)
  ├─ 按 Channel 拆分: gridOffset = channelLo * chunkSize, ...
  │   ├─ Channel 0: [0, count/nChannels)
  │   ├─ Channel 1: [count/nChannels, 2*count/nChannels)
  │   └─ ...
  │
  ├─ 每 Channel 内按 Ring Step 拆分:
  │   ├─ loopCount = nranks * chunkCount
  │   └─ 每步处理 chunkCount 个元素
  │
  └─ 每 Step 内按 Slice 拆分:
      ├─ sliceSize = stepSize * StepPerSlice
      └─ Primitives 的 genericOp 循环处理每个 slice
```

### 8.2 CTS 协议追踪

以 Ring AllReduce 中 rank[1] 发送给 rank[2] 的一个 step 为例：

```
时间线：
t0  rank[2] GPU kernel: irecv() 被调用
t1  rank[2] Proxy: ncclNet.irecv → ncclIbIrecv
t2  rank[2] net_ib: 准备 CTS {addr=recvbuff+offset, rkeys[], tag, idx}
t3  rank[2] net_ib: RDMA Write CTS → rank[1] 的 CTS FIFO

t4  rank[1] GPU kernel: isend() 被调用
t5  rank[1] Proxy: ncclNet.isend → ncclIbIsend
t6  rank[1] net_ib: 轮询 CTS FIFO，匹配 idx 和 tag
t7  rank[1] net_ib: RDMA Write 数据 → rank[2] 的 recvbuff+offset
    （使用 rank[2] 提供的 addr 和 rkeys）
t8  rank[1] net_ib: RDMA Write With Imm (imm_data = 数据大小)

t9  rank[2] net_ib: 收到 IBV_WC_RECV_RDMA_WITH_IMM
t10 rank[2] net_ib: 从 imm_data 提取数据大小
t11 rank[2] Proxy: ncclNet.test() → done=1
t12 rank[2] Proxy: 更新 conn->head，通知 GPU kernel
t13 rank[2] GPU kernel: 数据就绪，开始 reduce
```

### 8.3 GDR Flush 路径

如果使用 GPU Direct RDMA，数据被 RDMA Write 直接写入远端 GPU 内存。但 GPU 可能看不到最新数据，需要 flush：

```
rank[2] net_ib: ncclIbIflush()
  ├─ 通过 flush QP 发起 IBV_WR_RDMA_READ（loopback，读自身 host 内存）
  ├─ RDMA Read 完成 → 保证之前的 RDMA Write 已被 GPU 看到
  └─ ncclNet.test(flushReq) → done=1
```

---

## 9. CBD（Channel-Based Distribution）详解

### 9.1 数据如何分给多个 Channel

`ncclCollCbdPart` 函数将全局数据分配给各个 channel：

```
全局数据: [0, totalCount)

channelLo, channelHi: 本任务使用的 channel 范围
nMidChannels = channelHi - channelLo - 1

countLo   = cbd.countLo              // 第一个 channel 的元素数
countMid  = cbd.countMid             // 中间 channel 的元素数
countHi   = cbd.countHi              // 最后一个 channel 的元素数

totalCount = countLo + nMidChannels * countMid + countHi
```

每个 channel 的 `partOffset`：
- channelLo: `offset = 0`, count = countLo
- 中间 channel: `offset = countLo + (mid * countMid)`, count = countMid
- channelHi: `offset = countLo + nMidChannels * countMid`, count = countHi

### 9.2 Chunk 计算

每个 channel 将自己负责的数据再按 `chunkCount` 分块：
```
chunkCount = cbd.chunkGrains * (protoGrainSize / elementSize)

loopCount = nranks * chunkCount  // Ring 算法每轮处理这么多元素
```

---

## 10. 完整数据流追踪示例

### 场景：4 节点 × 1 GPU，Ring AllReduce，Simple 协议，2 Channel

**输入**: 每个节点 1M 个 float32（4MB），`ncclSum`

#### 步骤 1: API 调用

```
每个节点:
ncclAllReduce(sendbuff=gpu_ptr_0, recvbuff=gpu_ptr_1, count=1048576,
              datatype=ncclFloat32, op=ncclSum, comm, stream)
```

#### 步骤 2: Enqueue

```
trafficBytes = 1048576 × 4 × 2 = 8MB (每节点)
算法选择: RING (8MB, 4 节点)
协议选择: SIMPLE
通道分配: channel[0..1]
CBD 计算:
  channelCount = 1048576 / 2 = 524288 per channel
  chunkCount = 524288 / 4 = 131072 per chunk
```

#### 步骤 3: Kernel 启动

```
2 个 CUDA block (每个 channel 一个)
每个 block 512 线程
Block 0: 处理数据 [0, 524288)
Block 1: 处理数据 [524288, 1048576)
```

#### 步骤 4: Ring 执行（以 Block 0, Rank 0 视角）

```
ring[0].prev = rank 3, ring[0].next = rank 1
chunkCount = 131072 floats (512KB)
loopCount = 4 × 131072 = 524288

Iteration 1 (elemOffset = 0):
  Step 0: directSend chunk[3] (offset=393216, 131072 floats) → rank 1
  Step 1: directRecvReduceDirectSend chunk[2] (offset=262144) from rank 3 → reduce → rank 1
  Step 2: directRecvReduceDirectSend chunk[1] (offset=131072) from rank 3 → reduce → rank 1
  Step 3: directRecvReduceCopyDirectSend chunk[0] (offset=0) from rank 3 → reduce+copy → rank 1
          ↑ 此时 chunk[0] 包含所有 4 个节点的归约结果
  Step 4: directRecvCopyDirectSend chunk[3] (offset=393216) from rank 3 → copy → rank 1
  Step 5: directRecvCopyDirectSend chunk[2] (offset=262144) from rank 3 → copy → rank 1
  Step 6: directRecv chunk[1] (offset=131072) from rank 3 → 写入 recvbuff
```

#### 步骤 5: 每步的实际数据传输

以 Step 0 为例（rank 0 → rank 1, 131072 floats = 512KB）：

```
GPU Kernel Block 0, Worker 线程:
  ├─ 将 sendbuff[393216..524288] 写入中间缓冲区 buff[step % 8]
  ├─ PostSend: 更新 conn→rank1.send.tail = step+1

Proxy Thread (发送端, rank 0):
  ├─ 检测到 tail 更新
  ├─ ncclNet.isend(sendComm_to_rank1, buff, 512KB, tag, mrHandle)
  │   └─ net_ib: ncclIbIsend()
  │       ├─ 等待 CTS from rank 1
  │       ├─ RDMA Write 512KB → rank 1 的 recv buff
  │       └─ RDMA Write With Imm

net_ib (接收端, rank 1):
  ├─ ncclIbIrecv() 已提前调用
  ├─ 已发送 CTS {addr, rkeys, tag, idx} to rank 0
  ├─ 收到 IBV_WC_RECV_RDMA_WITH_IMM
  └─ 数据已写入 rank 1 的中间缓冲区

Proxy Thread (接收端, rank 1):
  ├─ ncclNet.test() → done
  └─ 更新 conn→rank0.recv.head = step+1

GPU Kernel Block 0 (接收端, rank 1):
  ├─ WaitRecv: 检测到 tail 更新
  └─ 从中间缓冲区读取数据，执行后续 reduce 操作
```

#### 步骤 6: 完成后

```
所有 step 完成后，每个 rank 的 recvbuff 包含完整的归约结果：
recvbuff[0..1048576] = Σ(rank_0.sendbuff + rank_1.sendbuff + rank_2.sendbuff + rank_3.sendbuff)
```

---

## 11. 多路径并行

### 11.1 多 Channel 并行

不同 channel 的数据范围不重叠，完全并行：
```
Channel 0: Ring AllReduce on data [0, count/2)
Channel 1: Ring AllReduce on data [count/2, count)
→ 带宽翻倍，延迟不变
```

### 11.2 多 QP 并行

单 channel 内，数据可进一步拆分到多个 QP：
```
QP 0: 传输数据的前半部分
QP 1: 传输数据的后半部分
→ 单连接带宽提升，适用于大消息
（需 NCCL_IB_QPS_PER_CONNECTION > 1 且 NCCL_IB_SPLIT_DATA_ON_QPS=1）
```

### 11.3 NVLink + IB 双路径（NVLS）

```
节点内: NVLink 直接传输（低延迟、高带宽）
节点间: IB/RoCE 网络传输
两种路径并行工作:
  Scatter/Gather: NVLink
  Reduce/Bcast: NVLink(节点内) + IB(节点间)
```

---

## 12. 关键数据结构汇总

| 结构 | 位置 | 说明 |
|------|------|------|
| `ncclInfo` | CPU, collectives.cc | API 调用参数 |
| `ncclTaskColl` | CPU, enqueue.cc | 调度器中的集合任务 |
| `ncclDevWorkColl` | GPU, device.h | GPU kernel 读取的 work 描述 |
| `ncclDevWorkBatch` | GPU, device.h | 批量 work 的容器 |
| `ncclChannel` | CPU+GPU, comm.h | 通道：包含 ring/tree/nvls 拓扑 |
| `ncclDevChannelPeer` | GPU, device.h | 对端连接信息（send/recv conn） |
| `ncclConnInfo` | GPU, device.h | 单向连接：buffs, head, tail, stepSize |
| `ncclRing` | GPU, device.h | 环形拓扑：prev, next, userRanks |
| `ncclTree` | GPU, device.h | 树形拓扑：up, down[0..2] |
| `ncclNvls` | GPU, device.h | NVLS 拓扑：up[], down, out, headRank |
| `Primitives<...>` | GPU, prims_simple.h | 数据搬运原子操作封装 |
| `ncclProxyOp` | CPU, proxy.cc | 代理操作描述 |
| `ncclIbSendComm/RecvComm` | CPU, net_ib | IB 连接状态 |

---

## 13. 流程图：完整 AllReduce 数据流

```
用户空间                    GPU Kernel Space                CPU Proxy Space           网络 (IB/RoCE)
───────                    ────────────────                ───────────────           ─────────────

ncclAllReduce()
  │
  ├─ ncclEnqueueCheck()
  │   ├─ collTaskAppend()
  │   │   └─ 排入 collSorter
  │   └─ ncclGroupEndInternal()
  │
  ├─ ncclPrepareTasks()
  │   ├─ 按 (fn,op,ty) 分桶
  │   ├─ ncclGetAlgoInfo() → 选择 RING/TREE/NVLS
  │   ├─ 分配 channels [lo, hi]
  │   └─ 构建 ncclDevWorkColl
  │
  ├─ ncclTasksRegAndEnqueue()
  │   ├─ ncclRegFind() → ncclNet.regMr()
  │   ├─ ncclAddProxyOpIfNeeded()
  │   └─ ncclLaunchKernel()
  │
  ▼                         ▼                              ▼
                       ┌──────────┐                  ┌───────────┐
                       │ GPU Block │                  │Proxy Thread│
                       │ (channel) │                  │            │
                       └─────┬─────┘                  └─────┬─────┘
                             │                              │
                    ┌────────┴────────┐             ┌───────┴───────┐
                    │ 读取 WorkColl    │             │ 读取 ProxyOp   │
                    │ 选择 Algo/Proto  │             │                │
                    └────────┬────────┘             └───────┬───────┘
                             │                              │
                    ┌────────┴────────┐                     │
                    │ Ring: 2(N-1)步   │                     │
                    │ Tree: ReduceUp   │                     │
                    │       + BcastDn  │                     │
                    └────────┬────────┘                     │
                             │                              │
              ┌──────────────┼──────────────┐               │
              │              │              │               │
         ┌────┴────┐   ┌────┴────┐   ┌────┴────┐     ┌─────┴─────┐
         │WaitRecv │   │ Worker  │   │WaitSend │     │ncclNet.   │
         │poll tail│   │reduce   │   │poll head│     │isend/irecv│
         └────┬────┘   │+copy    │   └────┬────┘     │+test      │
              │        └────┬────┘        │          └─────┬─────┘
              │             │             │                │
         ┌────┴────┐   ┌────┴────┐   ┌────┴────┐          │
         │PostRecv │   │PostSend │   │         │          │
         │st head  │   │st tail  │   │         │          │
         └────┬────┘   └────┬────┘   │         │          │
              │             │        │         │          │
              └──────┬──────┘        │         │          │
                     │               │         │          │
                     ▼               ▼         │          │
               直接/间接传输          │         │          │
                     │               │         │          │
          ┌──────────┴──────────────────────────┴──────────┐
          │                                                  │
          │          间接路径: 通过 Proxy                     │
          │   GPU buff ←→ Proxy Thread ←→ net_ib ←→ 网络     │
          │                                                  │
          │          直接路径: 绕过 Proxy                      │
          │   GPU buff ─── RDMA Write ──────→ 远端 GPU buff  │
          │   (GDR/NVLink)                                   │
          │                                                  │
          └──────────────────────────────────────────────────┘
                     │
                     ▼
              recvbuff 包含最终归约结果
              ncclAllReduce() 返回 ncclSuccess
```

---

## 14. 关键环境变量对数据路径的影响

| 环境变量 | 默认值 | 对数据路径的影响 |
|---------|--------|----------------|
| `NCCL_ALGO` | -1(自动) | 强制选择算法: 0=Tree, 1=Ring, 4=NVLS |
| `NCCL_PROTO` | -1(自动) | 强制协议: 0=LL, 1=LL128, 2=Simple |
| `NCCL_MAX_NCHANNELS` | 自动 | 最大 channel 数，直接影响并行度 |
| `NCCL_MIN_NCHANNELS` | 1 | 最小 channel 数 |
| `NCCL_NET_GDR_LEVEL` | 自动 | GPU Direct RDMA 启用级别 |
| `NCCL_IB_QPS_PER_CONNECTION` | 1 | 每连接 QP 数 |
| `NCCL_IB_SPLIT_DATA_ON_QPS` | 0 | 是否在多 QP 间分片数据 |
| `NCCL_SHARP_SUPPORT` | 1 | 是否启用 SHARP（CollNet） |
| `NCCL_NVLS_ENABLE` | 自动 | 是否启用 NVLS |
| `NCCL_BUFFSIZE` | 自动 | 中间缓冲区大小（间接路径） |
| `NCCL_NSOCKS_PERTHREAD` | 1 | proxy 每线程 socket 数 |
| `NCCL_PROXY_MAX_CONCURRENCY` | 2 | proxy 最大并发操作数 |

---

## 15. 各算法的适用场景总结

| 算法 | 延迟 | 带宽利用率 | 适用数据量 | 适用拓扑 |
|------|------|----------|----------|---------|
| **Ring** | O(N) | 极高（100%链路利用率）| >1MB | 任意，尤其大消息 |
| **Tree** | O(logN) | 较低（~33%链路利用率）| <1MB | 任意，尤其小消息 |
| **NVLS** | 低 | 极高（NVLink多播）| 任意 | 单节点 NVLink 或多节点 NVLink+IB |
| **CollNet Direct** | 极低 | 中（网络内reduce）| 任意 | SHARP IB 交换机 |
| **CollNet Chain** | 低 | 中 | 任意 | SHARP IB 交换机 |
| **PAT** | O(logN) | 高 | 任意 | 大规模（>64 节点）|
