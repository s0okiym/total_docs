# NCCL net_ib 传输层详解

## 1. 概述

`net_ib` 是 NCCL 的 **InfiniBand/RoCE 网络传输插件**，是 NCCL 最核心、最高性能的网络后端。它基于 libibverbs 和 libmlx5 实现，支持 InfiniBand 和 RoCE（RDMA over Converged Ethernet）两种链路层，利用 RC（Reliable Connection）QP 进行零拷贝 RDMA 数据传输。

### 文件结构

| 文件 | 行数 | 职责 |
|------|------|------|
| `common.h` | 536 | 全局数据结构、通信器、请求、FIFO 定义 |
| `common.cc` | 171 | 基础初始化、异步事件线程、全局 `ncclNetIb` 导出 |
| `init.cc` | 456 | 设备发现、PCI 排序、虚拟设备（vDevice）创建 |
| `connect.h/cc` | 1330 | 连接建立：Listen/Connect/Accept、QP 创建与状态转换 |
| `p2p.h/cc` | 845 | 数据路径：isend/irecv/iflush/test、CTS 协议 |
| `p2p_resiliency.h/cc` | 1048 | 弹性容错：QP/设备故障检测与恢复 |
| `reg.cc` | 123 | 内存注册与 MR 缓存 |
| `gdr.cc` | 84 | GPU Direct RDMA / DMA-BUF 支持检测 |
| `gin.h/cc` | 677 | GIN（GPU In-Network）框架：GDAKI 和 Proxy 两种后端 |
| `gdaki/` | ~46文件 | DOCA GPUNetIO 集成（GDAKI 后端的具体实现） |

---

## 2. 核心数据结构

### 2.1 设备层

```
ncclIbDev         — 物理IB设备（mlx5_0, mlx5_1, ...）
ncclIbMergedDev   — 虚拟设备（vDevice），可合并多个物理设备（如双口NIC的rails）
```

**`ncclIbDev`** — 单个物理 IB 设备端口：
- `context`: ibv_context，libibverbs 设备句柄
- `pd`: Protection Domain（引用计数共享）
- `portNum`, `lid`, `speed`, `link` (IB/RoCE)
- `mrCache`: 内存注册缓存（避免重复 reg/dereg）
- `ar`: 是否启用自适应路由（Adaptive Routing）
- `ibProvider`: IB_PROVIDER_MLX5 或 NONE
- `capsProvider.mlx5.dataDirect`: 是否支持 Data Direct DMA

**`ncclIbMergedDev`** — 虚拟设备，由 `NCCL_IB_MERGE_NICS=1` 控制：
- 可将同一物理 NIC 的多个端口（rails）合并为一个 vDevice
- `vProps.devs[]` 存物理设备索引，`vProps.ndevs` 为物理设备数
- 速度为所有子设备之和
- 例如：mlx5_0+mlx5_1 作为一个 vDevice，两个子设备各自承担部分 QP

### 2.2 通信器层

```
ncclIbListenComm  — 监听器（listen阶段）
ncclIbSendComm    — 发送端通信器
ncclIbRecvComm    — 接收端通信器
ncclIbNetCommBase — Send/Recv 通信器的公共基类
```

**`ncclIbNetCommBase`** — 公共基类：
- `qps[NCCL_IB_MAX_QPS]`: 所有 QP（最多 128）
- `activeQps[]`: 指向实际使用的 QP（弹性容错时可替换）
- `nqps`: QP 总数 = `NCCL_IB_QPS_PER_CONNECTION × ndevs`
- `nDataQps`: 数据 QP 数 = max(local_ndevs, remote_ndevs)
- `splitDataOnQps`: 是否将数据在多个 QP 间分片
- `reqs[NET_IB_MAX_REQUESTS]`: 请求池
- `remDevs[]`: 远端设备信息（GID, LID, rkey 等）
- `resiliency`: 弹性容错上下文

**`ncclIbSendComm`** — 发送端：
- `ctsFifo[slot][nreqs]`: CTS FIFO，接收端写入后发送端读取
- `devs[ndevs]`: 每个物理设备的资源（PD, CQ, MR）
- `sendReqs[slot][nreqs]`: 按 slot 索引的发送请求
- `remCmplsRecords`: 远端完成记录的影子副本

**`ncclIbRecvComm`** — 接收端：
- `remCtsFifo`: 发送给发送端的 CTS 消息的本地副本
- `cmplsRecords[slot]`: 完成记录数组，发送端通过 RDMA 写入
- `devs[ndevs]`: 每设备的资源（含 flush QP）
- `gpuFlushHostMem` + flush QP: 用于 GDR flush 操作
- `prepostReceiveWorkRequests`: 是否预投递 receive WQE

### 2.3 请求层

**`ncclIbRequest`** — 单个异步请求：
- `type`: SEND / RECV / FLUSH / GIN_IPUT
- `events[4]`: 每设备待完成事件计数，归零表示完成
- `devBases[4]`: 每设备的 CQ/PD 基指针，用于 poll
- `send.sendData[128]`: 弹性容错中跟踪每个 QP 是否已发送

---

## 3. 初始化流程

### 3.1 设备发现 (`init.cc: ncclIbInitDevices`)

```
ncclIbInit() → ncclIbInitDevices()
```

1. **加载 libibverbs / libmlx5 符号** (`wrap_ibv_symbols()`, `wrap_mlx5dv_symbols()`)
2. **查找 IP 接口** — 用作 OOB（带外）通信的 TCP 地址
3. **枚举 IB 设备** — `ibv_get_device_list()`
4. **过滤设备** — 根据 `NCCL_IB_HCA` 环境变量
5. **逐端口处理**：
   - 查询端口属性（speed, MTU, link layer）
   - 检测 mlx5 Data Direct DMA 支持 (`ncclMlx5dvDmaBufCapable`)
   - 创建 `ncclIbDev` 条目
   - 启动异步事件监控线程 (`ncclIbAsyncThreadMain`)
6. **PCI 排序** — `NCCL_IB_DEVICE_PCI_ORDER=1`（默认）按 PCI 地址排序保证跨节点一致性
7. **创建 vDevice** — 每个物理设备自动创建单设备 vDevice；多设备 vDevice 由 `NCCL_IB_MERGE_NICS` 触发

**关键环境变量**：
- `NCCL_IB_DISABLE`: 禁用 IB 传输
- `NCCL_IB_HCA`: 指定使用的 HCA 列表（支持 `^` 排除、`=` 精确匹配）
- `NCCL_IB_MERGE_NICS`: 合并多端口 NIC（默认 1）
- `NCCL_IB_MERGE_VFS`: 合并虚拟功能（默认 1）
- `NCCL_IB_PCI_RELAXED_ORDERING`: PCIe Relaxed Ordering（默认 2=自动检测）
- `NCCL_IB_ADAPTIVE_ROUTING`: 自适应路由（IB 网络默认开启）

### 3.2 虚拟设备 (`ncclIbMakeVDevice`)

多设备 vDevice 将多个物理设备捆绑为一个逻辑设备。连接建立时，QP 以 **round-robin（轮询）** 方式分布在各物理设备上。例如 2 设备 4 QP：
```
Dev0: QP0, QP2
Dev1: QP1, QP3
```

---

## 4. 连接建立流程

### 4.1 总体流程

```
发送端 (Connect)                          接收端 (Accept)
─────────────────                         ───────────────
1. ncclIbConnect()                        1. ncclIbListen()
   ├─ 创建 SendComm                          └─ 创建 socket 监听
   ├─ socket 连接                       2. ncclIbAccept()
   ├─ 交换 vProps (设备列表)                ├─ socket accept
   ├─ 创建 QPs (INIT→RTR→RTS)               ├─ 交换 vProps
   ├─ 注册 CTS FIFO MR                      ├─ 创建 QPs (INIT→RTR→RTS)
   ├─ 发送 metadata 给接收端                 ├─ 注册 completion records MR
   ├─ 接收接收端的 metadata                  ├─ 发送 metadata 给发送端
   └─ 发送 ready 信号                        └─ 接收 ready 信号
```

### 4.2 状态机

连接建立采用**非阻塞状态机**，通过 `ncclIbCommStage` 跟踪进度：

```
Start → Connect → SendDevList → RecvDevList → Send → Connecting → Connected → 完成
                     ↑                                    ↑
                  发送本地 vProps                   接收远端 metadata
```

Accept 端类似：
```
Start → Accept → RecvDevList → SendDevList → Recv → Send → PendingReady → 完成
```

### 4.3 QP 创建与状态转换

QP 遵循标准 IB RC 状态机：

```
Create → RESET → INIT → RTR → RTS
```

**`ncclIbCreateQp`**: 创建 QP 并修改到 INIT
- 设置 PKey、端口号、访问权限
- 发送端: `IBV_ACCESS_REMOTE_WRITE`
- 接收端: `IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC`（Atomic 用于 GIN）

**`ncclIbRtrQp`**: 修改到 RTR（Ready to Receive）
- 设置路径 MTU、对端 QPN、GID/LID 路由信息
- RoCE: 使用 GID + Traffic Class
- IB: 同子网用 LID，跨子网用 FLID（Forwarding LID）

**`ncclIbRtsQp`**: 修改到 RTS（Ready to Send）
- 设置 timeout、retry_cnt、rnr_retry

**ECE（Enhanced Connection Establishment）**: 如果两端都支持，在 RTR 前设置 ECE 参数。

### 4.4 GID 索引选择 (`ncclIbGetGidIndex`)

- **IB 网络**: 优先使用 routable FLID GID（index 由 `NCCL_IB_ROUTABLE_FLID_GID_INDEX` 指定）
- **RoCE 网络**: 自动选择匹配 `NCCL_IB_ADDR_FAMILY` + `NCCL_IB_ADDR_RANGE` + `NCCL_IB_ROCE_VERSION_NUM` 的 GID

---

## 5. 数据传输协议

### 5.1 CTS（Clear-to-Send）协议

net_ib 采用 **Receiver-driven** 的 CTS 协议，核心流程：

```
接收端                              发送端
──────                              ──────
1. irecv() 被调用
2. 准备 CTS 消息:
   {addr, size, rkeys, tag, idx}
3. RDMA Write CTS → 发送端 FIFO
                                    4. ncclIbIsend() 轮询 CTS FIFO
                                    5. 检查 idx 和 tag 匹配
                                    6. RDMA Write 数据 → 接收端 buffer
                                    7. RDMA Write With Imm (通知接收端)
8. 收到 IBV_WC_RECV_RDMA_WITH_IMM
9. 从 imm_data 获取大小（或从 completion records 读取）
```

**CTS FIFO 结构** (`ncclIbSendFifo`)：
- 32 字节对齐（Relaxed Ordering 安全）
- 包含：`addr`（远端 buffer 地址）、`rkeys[]`（每设备 rkey）、`size`、`tag`、`idx`（序列号）

**完成记录** (`ncclIbRequestCompletionRecord`)：
- 多 recv 场景下，发送端通过 RDMA Write 将各请求的 size 写入接收端的完成记录
- `completions[]` 数组用于弹性容错：接收端标记哪些 QP 已完成

### 5.2 发送流程 (`ncclIbIsend` → `ncclIbMultiSend`)

1. **等待 CTS**: 轮询 `ctsFifo[slot]` 检查 `idx` 是否匹配
2. **匹配 tag**: 在 multi-recv 场景中匹配多个 tag
3. **分配请求**: 从请求池获取 `ncclIbRequest`
4. **等待所有子请求匹配**: `sendReqsCnt[slot]` 计数
5. **构造 Work Request 链**:
   - 多个 `IBV_WR_RDMA_WRITE`（数据分片到多个 QP）
   - 最后一个 `IBV_WR_RDMA_WRITE_WITH_IMM`（通知接收端）
6. **投递到 QP**: 按 QP 分发，每个 QP 发送数据的分片

**Adaptive Routing 优化**：
- 当 AR 启用且数据 > `NCCL_IB_AR_THRESHOLD`（默认 8192），先发 RDMA Write（数据），再发 Write With Imm（通知）
- 小消息直接用 Write With Imm

**多 QP 分片**：
- 当 `NCCL_IB_SPLIT_DATA_ON_QPS=1` 时，数据被均匀分片到所有 QP
- 每个 QP 的分片大小按 128 字节对齐（`IB_WRITE_CHUNK_ALIGNMENT`）

### 5.3 接收流程 (`ncclIbIrecv`)

1. **分配请求**
2. **Post receive WQE** 到所有 QP（如非预投递模式）
3. **准备 CTS 消息**: 填充 `remCtsFifo.elems[slot]` 中的 addr/rkeys/tag/idx
4. **RDMA Write CTS**: 通过 `ncclIbPostFifo()` 发送 CTS 到发送端
5. CTS QP 选择: `req->id % ndevs` 对应设备的第一个 QP

### 5.4 完成测试 (`ncclIbTest`)

1. 检查 `events[]` 是否全归零 → 直接完成
2. 遍历所有设备的 CQ，`ibv_poll_cq` 批量拉取完成事件
3. 处理每个 CQE：
   - **发送端**: `IBV_WC_RDMA_WRITE` → 递减对应请求的 `events[devIndex]`
   - **接收端**: `IBV_WC_RECV_RDMA_WITH_IMM` → 从 imm_data 获取 size，递减 events
   - **Flush**: `IBV_WC_RDMA_READ` → 递减 events
   - **CTS**: `IBV_WC_RDMA_WRITE` → CTS 发送完成（接收端）
4. 错误处理：弹性容错启用时调用 `ncclIbResiliencyHandleCompletionError`

### 5.5 请求匹配机制

两种接收端匹配方案（`NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME`）：
- **BY_INDEX (0, 默认)**: 用 `wr_id`（slot 索引）匹配请求，`imm_data` 携带数据大小
- **BY_ID (1)**: `imm_data` 携带请求 ID，通过 hash 表查找请求

---

## 6. GPU Direct RDMA

### 6.1 GDR 检测 (`gdr.cc`)

三种 GPU Direct RDMA 路径：

| 路径 | 检测方法 | 说明 |
|------|---------|------|
| **nv_peer_mem** | 检查 `/sys/module/nvidia_peermem/version` | 传统内核模块 |
| **nv_mem / nv_mem_nc** | 检查 `/sys/kernel/mm/memory_peers/` | 旧版内核模块 |
| **DMA-BUF** | `ibv_reg_dmabuf_mr` 测试调用 | 现代 Linux 内核（5.12+） |

### 6.2 GDR Flush

当使用 GDR 时，InfiniBand NIC 的 RDMA Write 操作直接将数据写入 GPU 内存。
由于 PCIe 与 IB 之间的顺序性差异，**GPU 可能在 RDMA Write 完成后仍看不到最新数据**（
IB HCA 通过 PCIe 写入，但 GPU 通过 PCIe/C2C 读取，两者之间无显式一致性保证）。

**Flush 原理**：
1. 接收端为每设备创建专用的 **flush QP**（类型 RC， 访问权限为 `LOCAL_WRITE | REMOTE_READ`）
2. Flush QP 的目的端指向**自身**（`dest_qp_num = 本 QP 的 qp_num`），形成 loopback RDMA Read
3. Read 操作的目标是一个 **host 端 dummy 变量** (`gpuFlushHostMem`)，4. 当 RDMA Read 完成 → 说明数据已经穿过 PCIe 到达 host → GPU 已经能看到之前的 RDMA Write

本质上是利用 IB 硬件的 **read-after-write 顺序性**保证：
RDMA Read 必须从 NIC → PCIe → GPU 方向读数据，这与 RDMA Write 的数据方向（NIC → PCIe → GPU）一致
因此 Read 完成意味着 Write 的数据已被 GPU 看到到。

- `NCCL_GDR_FLUSH_DISABLE=1` 可禁用 flush（某些平台不需要）
- Data Direct 设备 (`dataDirect=1`) 强制启用 flush（Data Direct 有独立的数据路径，不经过 PCIe）

---

## 7. 内存注册 (`reg.cc`)

### 7.1 MR 缓存机制

net_ib 实现了一个 **按页对齐的 MR 缓存**（`ncclIbMrCache`）：

- 按 `addr`（页对齐）排序的数组
- **命中**: 如果新注册的范围完全包含在已有 MR 中，引用计数 +1
- **未命中**: 分配新 MR 并插入排序位置
- **释放**: 引用计数 -1，归零时 deregister

### 7.2 注册流程 (`ncclIbRegMr` / `ncclIbRegMrDmaBuf`)

1. 对每个物理设备调用 `ncclIbRegMrDmaBufInternal2`
2. 页对齐地址和大小
3. 查找缓存
4. 注册标志：
   - `IBV_ACCESS_LOCAL_WRITE | REMOTE_WRITE | REMOTE_READ | REMOTE_ATOMIC`
   - Relaxed Ordering: `IBV_ACCESS_RELAXED_ORDERING`（如果支持且未被 `NCCL_NET_MR_FLAG_FORCE_SO` 强制禁用）
   - DMA-BUF: 使用 `ibv_reg_dmabuf_mr` 或 `mlx5dv_reg_dmabuf_mr`（Data Direct 场景）
   - Relaxed Ordering + 非 DMABUF: 使用 `ibv_reg_mr_iova2`（IBVERBS_1.8 API）

### 7.3 多设备 MR 包装

`ncclIbMrHandle` 为每个物理设备维护独立的 `ibv_mr*`，支持 merged vDevice 场景下每设备独立的 lkey/rkey。

---

## 8. 弹性容错 (`p2p_resiliency`)

### 8.1 概述

弹性容错机制在 QP 或设备故障时自动恢复数据传输，避免整个通信崩溃。

### 8.2 设备状态机

```
ncclIbResiliencyDevStateOk → ncclIbResiliencyDevStateError → ncclIbResiliencyDevStateErrorPermanent
                                  ↑                                    |
                                  |___(可恢复)________________________|
                                  |
                            (不可恢复) → 返回 ncclRemoteError
```

### 8.3 错误检测与恢复流程

1. **CQE 错误**: `ncclIbTest` 中检测到 `wc->status != IBV_WC_SUCCESS`
2. **调用 `ncclIbResiliencyHandleCompletionError`**：
   - 检查是否所有设备都故障 → 是则致命错误
   - 检查是否只是 transient error → 可能继续
   - 标记故障设备状态
3. **QP 替换**: `ncclIbResiliencyReplaceQps` — 将故障 QP 替换为健康设备上的 QP
4. **探测（Probe）**: 通过专用的 probing QP 发起 RDMA Read 到接收端的 completion records
5. **判断完成**: 如果接收端已标记 `completions[qpIndex]=true`，则请求已完成
6. **重传**: 如果未完成，在新的 QP 上重传数据

### 8.4 Probing QP

- 每设备一个专用的 probing QP
- 使用独立的 CQ 避免干扰正常数据路径
- Probing 通过 RDMA Read 检查远端的 `ncclIbRequestCompletionRecord.completions[]`

### 8.5 适用场景

- **Silent data corruption / link error**: QP 级故障可恢复
- **设备故障**: 如果有冗余设备（multi-rail），可自动 failover
- **全部设备故障**: 不可恢复，返回 `ncclRemoteError`

**前提条件**：需要 `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1`（弹性容错会自动启用）

---

## 9. GIN（GPU In-Network Computing）

### 9.1 概述

GIN 允许 GPU 通过网络直接进行集合通信（AllGather, AllToAll）和点对点操作（IPut, IPutSignal），绕过 CPU 参与。

### 9.2 两种后端

| 后端 | 说明 | 适用场景 |
|------|------|---------|
| **GDAKI** | 基于 DOCA GPUNetIO，GPU 直接发起 RDMA 操作 | NVIDIA BlueField DPU + CX-8 等支持 DOCA 的硬件 |
| **Proxy** | CPU 代理模式，通过标准 IB Verbs 实现 | 通用 IB/RoCE 硬件 |

选择逻辑（由 `NCCL_GIN_TYPE` 控制）：
- 默认 `-1`: 先尝试 GDAKI，失败回退到 Proxy
- `NCCL_GIN_TYPE=0`: 强制 GDAKI
- `NCCL_GIN_TYPE=1`: 强制 Proxy

### 9.3 GIN Proxy 后端

**连接建立** (`ncclGinIbProxyConnect`)：
- 建立全连接拓扑（每个 rank 与所有其他 rank 建立连接）
- 通过 barrier 同步确保所有连接就绪

**IPut 操作** (`ncclGinIbProxyIPut`):
- 使用 `IBV_WR_RDMA_WRITE` 直接写远端内存
- 支持 offset + MR handle 方式寻址

**IPutSignal 操作** (`ncclGinIbProxyIPutSignal`):
- 先 `IBV_WR_RDMA_WRITE` 写数据
- 然后 `IBV_WR_ATOMIC_FETCH_AND_ADD` 发信号（支持 INC/ADD）
- 两个操作通过 WR 链一次性提交
- 信号值写入 `putSignalScratchpad`（发送端本地 MR），通过 RDMA Atomic 操作递增远端信号地址

### 9.4 GIN 集合操作

**AllGather** (`ncclGinIbAllGather`):
- Ring 算法：每 rank 发送自己的数据段，接收前一个 rank 的段，迭代 nranks-1 轮
- 先本地 memcpy 自己的数据到 recvBuf

**AllToAll** (`ncclGinIbAllToAll`):
- 基于 AllGather 实现：先 AllGather 所有数据，再本地 scatter

### 9.5 GDAKI 后端

GDAKI（GPU Direct Async Kernel Interface）利用 NVIDIA DOCA GPUNetIO 库实现 GPU 内核直接发起网络操作：
- `ncclGinGdakiCreateContext`: 创建 GPUNetIO 上下文
- `ncclGinGdakiRegMrSym` / `ncclGinGdakiDeregMrSym`: 对称内存注册
- 设备类型为 `NCCL_NET_DEVICE_GIN_GDAKI`

---

## 10. 异步事件处理

### 10.1 异步事件线程 (`ncclIbAsyncThreadMain`)

每个物理 IB 设备启动一个独立的异步事件监控线程：

| 事件 | 处理 |
|------|------|
| `IBV_EVENT_DEVICE_FATAL` | 标记设备致命错误 (`ncclIbDevFatalError`) |
| `IBV_EVENT_CQ_ERR` | 标记 CQ 致命错误 |
| `IBV_EVENT_QP_FATAL / QP_REQ_ERR / QP_ACCESS_ERR` | 标记 QP 致命错误 |
| `IBV_EVENT_GID_CHANGE` | 警告日志 |
| 其他非致命事件 | 警告日志 |

当致命事件被记录后，后续的 `ncclIbTest` 通过 `ncclIbStatsCheckFatalCount` 检测到错误计数 > 0，返回 `ncclSystemError`。

---

## 11. 关键环境变量汇总

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `NCCL_IB_DISABLE` | 0 | 禁用 IB 传输 |
| `NCCL_IB_HCA` | 全部 | 指定使用的 IB 设备 |
| `NCCL_IB_MERGE_NICS` | 1 | 合并多端口 NIC |
| `NCCL_IB_MERGE_VFS` | 1 | 合并 VF |
| `NCCL_IB_DEVICE_PCI_ORDER` | 1 | 按 PCI 地址排序 |
| `NCCL_IB_QPS_PER_CONNECTION` | 1 | 每连接每设备 QP 数 |
| `NCCL_IB_SPLIT_DATA_ON_QPS` | 0 | 在多个 QP 间分片数据 |
| `NCCL_IB_ADAPTIVE_ROUTING` | IB=1, RoCE=0 | 自适应路由 |
| `NCCL_IB_AR_THRESHOLD` | 8192 | AR 启用阈值 |
| `NCCL_IB_PCI_RELAXED_ORDERING` | 2 | PCIe Relaxed Ordering |
| `NCCL_IB_USE_INLINE` | 0 | 使用 inline send |
| `NCCL_IB_TIMEOUT` | 20 | QP ACK 超时 |
| `NCCL_IB_RETRY_CNT` | 7 | QP 重试计数 |
| `NCCL_IB_SL` | -1 | Service Level |
| `NCCL_IB_TC` | -1 | Traffic Class |
| `NCCL_IB_FIFO_TC` | -1 | FIFO 专用 Traffic Class |
| `NCCL_IB_GID_INDEX` | -1 | 强制 GID 索引 |
| `NCCL_IB_ROCE_VERSION_NUM` | 2 | RoCE 版本 |
| `NCCL_IB_ADDR_FAMILY` | AF_INET | GID 地址族 |
| `NCCL_IB_RETURN_ASYNC_EVENTS` | 1 | 检查异步致命事件 |
| `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS` | 0 | 预投递接收 WQE |
| `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME` | 0 | 接收端匹配方案 |
| `NCCL_IB_ECE_ENABLE` | 1 | 启用 ECE |
| `NCCL_IB_DATA_DIRECT` | 1 | Data Direct DMA |
| `NCCL_GDR_FLUSH_DISABLE` | 0 | 禁用 GDR flush |
| `NCCL_GIN_TYPE` | -1 | GIN 后端类型 |
| `NCCL_IB_WARN_RAIL_LOCAL` | 0 | 警告 rail-local 连接 |

---

## 12. NCCL 核心如何使用 net_ib

### 12.1 插件注册

`common.cc` 中定义并导出 `ncclNetIb`（类型 `ncclNet_t`），包含所有传输 API 的函数指针：

```c
ncclNet_t ncclNetIb = {
  .name = "IB",
  .init = ncclIbInit,
  .devices = ncclIbDevices,
  .getProperties = ncclIbGetProperties,
  .listen = ncclIbListen,
  .connect = ncclIbConnect,
  .accept = ncclIbAccept,
  .regMr = ncclIbRegMr,
  .regMrDmaBuf = ncclIbRegMrDmaBuf,
  .deregMr = ncclIbDeregMr,
  .isend = ncclIbIsend,
  .irecv = ncclIbIrecv,
  .iflush = ncclIbIflush,
  .test = ncclIbTest,
  ...
};
```

NCCL 核心通过 `ncclNet_t` 接口（定义在 `nccl_net.h`）统一调用不同传输后端。`net_ib` 作为内置传输在 NCCL 初始化时被自动加载。

### 12.2 调用路径

```
ncclCommInitRank()
  → ncclTransports[NET].setup()         // 建立 P2P 通道
    → ncclNet.listen() / connect() / accept()   // 连接建立
    → ncclNet.regMr() / deregMr()              // 内存注册
    → ncclNet.isend() / irecv() / test()       // 数据传输
    → ncclNet.iflush()                         // GDR flush
    → ncclNet.closeSend() / closeRecv()        // 关闭
```

### 12.3 vDevice 与 QP 映射

```
NCCL core 选择 vDevice (逻辑 NIC)
  → ncclNet.connect(dev=vDeviceIndex, ...)
    → ncclIbConnect 查找 ncclIbMergedDevs[vDeviceIndex]
    → 获取 vProps.devs[] (物理设备列表)
    → 为每个物理设备创建 CQ/PD
    → 创建 nqps = QPS_PER_CONN × ndevs 个 RC QP
    → QP 以 round-robin 分配到各物理设备
```

### 12.4 数据路径中的 NCCL 调用

NCCL channel 层（`src/channel.cc`）维护 `ncclSendMem` / `ncclRecvMem`：
- `ncclSendMem.buffPtrs[i]` → 发送 buffer
- `ncclRecvMem.buffPtrs[i]` → 接收 buffer

每次 `channel.cfi` / `channel.cbi` 调用会触发：
```
ncclNet.isend(sendComm, data, size, tag, mhandle, ...)   // 数据发送
ncclNet.irecv(recvComm, n, data, sizes, tags, mhandles, ...)  // 数据接收
ncclNet.test(request, &done, sizes)                         // 轮询完成
```

---

## 13. 流程图

### 13.1 初始化与设备发现

```
┌─────────────────────┐
│  ncclIbInit()       │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ ncclIbInitDevices() │
└────────┬────────────┘
         ▼
  加载 ibverbs/mlx5dv 符号
         │
         ▼
  查找 IP 接口 (OOB)
         │
         ▼
  枚举 IB 设备 ──────────► NCCL_IB_HCA 过滤
         │
         ▼
  逐端口: 查询属性, 检测 GDR/DMA-BUF
         │
         ▼
  PCI 排序 (NCCL_IB_DEVICE_PCI_ORDER)
         │
         ▼
  创建 vDevice (ncclIbMakeVDevice)
         │
         ▼
  启动异步事件线程
```

### 13.2 连接建立

```
发送端 (Connect)                     接收端 (Accept)
══════════════                       ══════════════
ncclIbConnect()                      ncclIbListen()
  │                                    └─ socket listen
  ├─ 创建 SendComm
  ├─ socket 连接                      ncclIbAccept()
  │                                    ├─ socket accept
  ├─ SendDevList ───────────────────►  ├─ RecvDevList ◄─────────┐
  │                                    │                         │
  ├─ RecvDevList ◄───────────────────► ├─ SendDevList ──────────►│
  │                                    │                         │
  ├─ 创建 PD/CQ per 设备                ├─ 创建 PD/CQ per 设备    │
  ├─ 创建 QPs (INIT)                    ├─ 创建 QPs (INIT)       │
  ├─ 注册 CTS FIFO MR                   ├─ 注册 completion MR    │
  ├─ 填充 metadata (QP info, GID, ...)  ├─ 填充 metadata          │
  │                                    │                         │
  ├─ Send metadata ─────────────────►  ├─ Recv metadata ◄───────┘
  │                                    │
  ├─ Recv metadata ◄─────────────────── ├─ Send metadata ──────────►
  │                                    │
  ├─ QP → RTR → RTS                    ├─ QP → RTR → RTS
  ├─ Send ready ───────────────────►   ├─ Recv ready ◄────────────┐
  │                                    │                            │
  └─ 返回 SendComm                      └─ 返回 RecvComm            │
                                                                    │
                                     ──────────────────────────────┘
```

### 13.3 数据传输（CTS 协议）

```
接收端 (irecv)                       发送端 (isend)
═══════════════                      ═══════════════
ncclIbIrecv()
  │
  ├─ 分配 Request
  ├─ Post Recv WQE (每 QP)
  ├─ 准备 CTS:
  │  {addr, rkeys[], tag, idx}
  │
  ├─ RDMA Write CTS ─────────────►  ncclIbIsend()
  │                                    │
  │                                    ├─ 轮询 CTS FIFO
  │                                    ├─ 匹配 idx + tag
  │                                    ├─ 分配 Request
  │                                    ├─ 设置 lkeys from mhandle
  │                                    │
  │                                    ├─ 多 QP 分片:
  │                                    │  每个 QP 发送 chunk
  │                                    │  IBV_WR_RDMA_WRITE (数据)
  │                                    │  最后: IBV_WR_RDMA_WRITE_WITH_IMM
  │                                    │
  │  ◄───── IBV_WC_RECV_RDMA_WITH_IMM ─┘
  │
  ├─ 从 imm_data 获取 size
  └─ Request 返回给调用者

ncclIbTest()
  │
  ├─ 检查 events[] 是否归零
  ├─ 遍历每设备的 CQ poll
  ├─ 处理每个 CQE → 递减 events[]
  └─ events 全归零 → done=1
```

### 13.4 弹性容错恢复

```
ncclIbTest() 检测到 CQE error
  │
  ▼
ncclIbResiliencyHandleCompletionError()
  │
  ├─ 所有设备都故障? ──是──► 返回 ncclRemoteError
  │
  ├─ 仅 transient error? ──是──► 忽略，继续
  │
  ├─ 标记设备 state = Error
  │
  ▼
ncclIbResiliencyReplaceQps()
  │
  ├─ 找到故障 QP (devIndex == failedDevIndex)
  ├─ 在健康设备上找替换 QP
  └─ 更新 activeQps[] 指向新 QP
  │
  ▼
ncclIbResiliencyProgress() [每次 Test 调用]
  │
  ├─ 对每个 failedRequest:
  │   ├─ 发 RDMA Read probe 到远端 completion records
  │   ├─ 检查 completions[qpIndex] == true?
  │   │   ├─ 是 → 请求已完成，释放
  │   │   └─ 否 → 在新 QP 上重传数据
  │   └─ 重传时跳过已成功发送的 QP (sentData[qpIndex] == true)
  │
  └─ 继续正常 poll CQ
```