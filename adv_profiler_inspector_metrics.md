# NCCL Profiler 与 Inspector 监控指标完整参考手册

本文档详细列出 NCCL Profiler（示例插件）和 Inspector 工具能提供的所有监控和性能数据指标。

---

## 一、概述

| 工具 | 定位 | 输出格式 | 主要用途 |
|------|------|----------|----------|
| **Example Profiler** | 示例/参考实现 | Chrome Trace JSON (`.json`) | 详细事件时间线分析、性能瓶颈定位 |
| **Inspector** | 生产级诊断工具 | JSON (`.log`) / Prometheus (`.prom`) | 实时监控、集群级指标聚合、长期趋势分析 |

---

## 二、Example Profiler 监控指标详表

### 2.1 事件类型总览

Profiler 支持 **15 种核心事件类型**，每种类型都有对应的监控指标：

| 事件类型 | 标识符 | 说明 |
|----------|--------|------|
| Group API | `ncclProfileGroupApi` | ncclGroupStart/End API 调用 |
| Coll API | `ncclProfileCollApi` | 集合操作 API 调用 (AllReduce等) |
| P2P API | `ncclProfileP2pApi` | 点对点 API 调用 (Send/Recv) |
| Kernel Launch | `ncclProfileKernelLaunch` | CUDA Kernel 启动事件 |
| Group | `ncclProfileGroup` | Group 内部任务组 |
| Collective | `ncclProfileColl` | 集合操作执行 |
| P2P | `ncclProfileP2p` | 点对点操作执行 |
| Proxy Op | `ncclProfileProxyOp` | Proxy 层操作 |
| Proxy Step | `ncclProfileProxyStep` | Proxy 网络传输步骤 |
| Proxy Ctrl | `ncclProfileProxyCtrl` | Proxy 控制线程状态 |
| Kernel Channel | `ncclProfileKernelCh` | Kernel 通道执行 |
| Net Plugin | `ncclProfileNetPlugin` | 网络插件事件 (IB/Socket) |
| CE Coll | `ncclProfileCeColl` | Copy Engine 集合操作 (v6) |
| CE Sync | `ncclProfileCeSync` | Copy Engine 同步 (v6) |
| CE Batch | `ncclProfileCeBatch` | Copy Engine 批处理 (v6) |

---

### 2.2 Group API 事件指标

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `groupApiId` | int | Group API 事件唯一 ID | 42 |
| `groupDepth` | int | Group 嵌套深度 | 1 |
| `graphCaptured` | bool | 是否被 CUDA Graph 捕获 | true/false |
| `startTs` | double | 开始时间戳 (微秒) | 123456.78 |
| `stopTs` | double | 结束时间戳 (微秒) | 123789.01 |
| `endOfncclGroupStartTs` | double | GroupStart 阶段结束时间 | 123500.00 |
| `startOfncclGroupEndTs` | double | GroupEnd 阶段开始时间 | 123600.00 |
| `refCount` | int | 引用计数（子事件数量） | 5 |

**功能说明**：
- 追踪 ncclGroupStart() 到 ncclGroupEnd() 的完整生命周期
- `groupDepth` 支持嵌套 Group 调用分析
- `endOfncclGroupStartTs` 和 `startOfncclGroupEndTs` 用于区分 Group 的两个阶段
- Chrome Trace 中以 `"cat": "GROUP_API"` 标识

---

### 2.3 Coll API 事件指标

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `collApiId` | int | Coll API 事件唯一 ID | 100 |
| `func` | string | 集合操作名称 | "ncclAllReduce" |
| `count` | size_t | 元素数量 | 1048576 |
| `datatype` | string | 数据类型 | "ncclFloat32" |
| `root` | int | 根节点 rank | 0 |
| `graphCaptured` | bool | CUDA Graph 捕获标志 | false |
| `stream` | pointer | CUDA Stream 指针 | 0x7f8b2c000000 |
| `startTs` | double | API 调用开始时间 | 123456.78 |
| `stopTs` | double | API 调用结束时间 | 123457.50 |
| `refCount` | int | 关联任务事件数 | 1 |

**功能说明**：
- 记录用户调用 ncclAllReduce/ncclBroadcast 等 API 的入队时间
- `stopTs` 表示操作入队完成，而非实际执行完成
- 支持 AllReduce、AllGather、ReduceScatter、Broadcast、Reduce 等全部集合操作

---

### 2.4 P2P API 事件指标

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `p2pApiId` | int | P2P API 事件唯一 ID | 200 |
| `func` | string | 操作名称 | "ncclSend"/"ncclRecv" |
| `count` | size_t | 元素数量 | 1024 |
| `datatype` | string | 数据类型 | "ncclFloat32" |
| `graphCaptured` | bool | CUDA Graph 捕获标志 | false |
| `stream` | pointer | CUDA Stream 指针 | 0x7f8b2c000000 |
| `startTs` | double | API 调用开始时间 | 123456.78 |
| `stopTs` | double | API 调用结束时间 | 123456.80 |

**功能说明**：
- 追踪 ncclSend/ncclRecv API 调用
- SendRecv 操作会生成一对事件（Send + Recv）

---

### 2.5 Kernel Launch 事件指标

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `kernelLaunchId` | int | Kernel Launch 事件 ID | 50 |
| `stream` | pointer | CUDA Stream 指针 | 0x7f8b2c000000 |
| `groupId` | int | 所属 Group ID | 42 |
| `startTs` | double | Kernel 启动开始时间 | 123500.00 |
| `stopTs` | double | Kernel 启动完成时间 | 123500.10 |

**功能说明**：
- 记录 CUDA Kernel 启动的延迟
- 与实际的 Kernel 执行（KernelCh 事件）区分开

---

### 2.6 Group 事件指标

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `groupId` | int | Group 事件唯一 ID | 10 |
| `startTs` | double | Group 开始时间 | 123500.00 |
| `stopTs` | double | Group 结束时间 | 123800.00 |
| `refCount` | int | 子任务事件引用计数 | 3 |

**功能说明**：
- Group 事件是 Coll/P2P 任务事件的父容器
- 当所有子任务完成时，Group 事件才标记为完成

---

### 2.7 Collective 任务事件指标

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `seqNumber` | uint64_t | 集合操作序列号 | 1000 |
| `func` | string | 操作名称 | "AllReduce" |
| `sendBuff` | pointer | 发送缓冲区地址 | 0x7f8b20000000 |
| `recvBuff` | pointer | 接收缓冲区地址 | 0x7f8b30000000 |
| `count` | size_t | 元素数量 | 1048576 |
| `root` | int | 根节点 rank | 0 |
| `datatype` | string | 数据类型 | "ncclFloat32" |
| `nChannels` | uint8_t | 使用通道数 | 4 |
| `nWarps` | uint8_t | Kernel warp 数 | 12 |
| `algo` | string | 算法类型 | "RING"/"TREE"/"NVLS" |
| `proto` | string | 协议类型 | "SIMPLE"/"LL"/"LL128" |
| `rank` | int | 当前 rank | 2 |
| `startTs` | double | 任务开始时间 | 123550.00 |
| `stopTs` | double | 任务完成时间 | 123600.00 |
| `refCount` | int | 引用计数 | 5 |

**功能说明**：
- 记录实际的集合操作执行（区别于 API 入队）
- `seqNumber` 用于关联同一集合操作的不同 rank
- `algo` 和 `proto` 帮助分析算法/协议选择对性能的影响
- 包含 `kernel[MAX_CHANNELS]` 子事件和 `op[MAX_CHANNELS][MAX_OPS]` Proxy 子事件

---

### 2.8 P2P 任务事件指标

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `func` | string | 操作类型 | "Send"/"Recv" |
| `buff` | pointer | 数据缓冲区 | 0x7f8b20000000 |
| `count` | size_t | 元素数量 | 1024 |
| `datatype` | string | 数据类型 | "ncclFloat32" |
| `peer` | int | 对端 rank | 3 |
| `nChannels` | uint8_t | 使用通道数 | 1 |
| `rank` | int | 当前 rank | 2 |
| `startTs` | double | 任务开始时间 | 123550.00 |
| `stopTs` | double | 任务完成时间 | 123551.00 |
| `refCount` | int | 引用计数 | 3 |

**功能说明**：
- 记录点对点操作的实际执行
- `peer` 标识通信对端

---

### 2.9 Proxy Op 事件指标

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `channelId` | uint8_t | 通道 ID | 0 |
| `pid` | pid_t | 进程 ID | 1234 |
| `rank` | int | 当前 rank | 2 |
| `peer` | int | 对端 rank | 3 |
| `nSteps` | int | 传输步骤数 | 8 |
| `chunkSize` | int | 数据块大小 | 1048576 |
| `isSend` | bool | 是否发送操作 | true/false |
| `transSize` | size_t | 实际传输数据量 | 8388608 |
| `startTs` | double | 操作开始时间 | 123550.00 |
| `progrTs` | double | 进入 Progress 阶段时间 | 123560.00 |
| `stopTs` | double | 操作完成时间 | 123580.00 |
| `stepCount` | int | 已处理步骤数 | 8 |
| `step[MAX_STEPS]` | array | Proxy Step 子事件数组 | - |

**功能说明**：
- Proxy Op 表示一个完整的网络操作（Send/Recv）
- `transSize` 记录实际传输字节数，可能与 `nSteps * chunkSize` 不同
- 细分为 Schedule 阶段（start→progr）和 Progress 阶段（progr→stop）
- Chrome Trace 中显示为 `"ScheduleSend"/"ProgressSend"` 或 `"ScheduleRecv"/"ProgressRecv"`

---

### 2.10 Proxy Step 事件指标（网络传输详情）

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `step` | int | 步骤序号 | 0-7 |
| `isSend` | bool | 是否发送 | true/false |
| `state` | int | 当前状态 | - |
| `timestamp[0]` | double | Send: GPUWait / Recv: RecvWait 时间 | 123550.00 |
| `timestamp[1]` | double | Send: PeerWait / Recv: FlushWait 时间 | 123555.00 |
| `timestamp[2]` | double | Send: SendWait / Recv: GPUWait 时间 | 123560.00 |
| `startTs` | double | 步骤开始时间 | 123550.00 |
| `stopTs` | double | 步骤完成时间 | 123565.00 |
| `nNetEvents` | int | 关联网络插件事件数 | 1 |
| `net[MAX_EVENTS_PER_REQ]` | array | Net Plugin 子事件 | - |

**发送操作状态时间戳**：
- `SendGpuWait`: 等待 GPU 数据就绪
- `SendPeerWait`: 等待对端就绪
- `SendWait`: 等待发送完成（网络传输）

**接收操作状态时间戳**：
- `RecvWait`: 等待接收数据（网络传输）
- `RecvFlushWait`: 等待 Flush 完成（GDR 相关）
- `RecvGpuWait`: 等待 GPU 消费数据

**功能说明**：
- Proxy Step 是网络传输的最细粒度追踪
- 状态时间戳帮助定位瓶颈（GPU vs 网络）

---

### 2.11 Proxy Ctrl 事件指标

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `state` | enum | 控制线程状态 | - |
| `appended` | int | 追加的 Proxy Op 数量 | 5 |
| `startTs` | double | 状态开始时间 | 123400.00 |
| `stopTs` | double | 状态结束时间 | 123450.00 |

**状态类型**：
| 状态值 | 说明 |
|--------|------|
| `ncclProfilerProxyCtrlIdle` | 空闲状态 |
| `ncclProfilerProxyCtrlActive` | 活跃状态 |
| `ncclProfilerProxyCtrlSleep` | 睡眠状态 |
| `ncclProfilerProxyCtrlWakeup` | 唤醒状态 |
| `ncclProfilerProxyCtrlAppend` | 追加操作状态 |
| `ncclProfilerProxyCtrlAppendEnd` | 追加完成状态 |

**功能说明**：
- 追踪 Proxy 控制线程的生命周期
- `appended` 在 `AppendEnd` 状态下记录本次追加的操作数

---

### 2.12 Kernel Channel 事件指标

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `channelId` | uint8_t | 通道 ID | 0 |
| `startGpuClk` | uint64_t | GPU 开始时钟周期 | 1234567890 |
| `stopGpuClk` | uint64_t | GPU 结束时钟周期 | 1234567990 |
| `startTs` | double | Host 记录开始时间 | 123550.00 |
| `stopTs` | double | Host 记录结束时间 | 123600.00 |

**功能说明**：
- 记录每个通道的 Kernel 执行时间
- `startGpuClk/stopGpuClk` 使用 GPU 全局定时器（纳秒精度）
- 与 Host 时间戳对比可分析调度延迟

---

### 2.13 Net Plugin 事件指标

#### IB (InfiniBand) 插件

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `pluginType` | int | 网络类型 | `NCCL_PROFILER_NET_TYPE_IB` (0x01000000) |
| `pluginVer` | int | 版本 | 1 |
| `pluginEvent` | int | 事件类型 | `ncclProfileQp` |
| `device` | int | IB 设备 ID | 0 |
| `qpNum` | uint32_t | Queue Pair 编号 | 12345 |
| `opcode` | int | IB 操作码 | 0 (Send) / 1 (Recv) |
| `wr_id` | uint64_t | Work Request ID | 67890 |
| `length` | size_t | 数据长度 | 1048576 |
| `startTs` | double | 事件开始时间 | 123560.00 |
| `stopTs` | double | 事件结束时间 | 123561.00 |

#### Socket 插件

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `pluginType` | int | 网络类型 | `NCCL_PROFILER_NET_TYPE_SOCK` (0x02000000) |
| `pluginVer` | int | 版本 | 1 |
| `pluginEvent` | int | 事件类型 | `ncclProfileSocket` |
| `fd` | int | Socket 文件描述符 | 15 |
| `op` | int | 操作类型 | 0 (Send) / 1 (Recv) |
| `length` | size_t | 数据长度 | 1048576 |
| `startTs` | double | 事件开始时间 | 123560.00 |
| `stopTs` | double | 事件结束时间 | 123561.00 |

**功能说明**：
- 网络插件可报告底层网络事件
- 支持 IB Verbs 和 Socket 两种传输层
- 对于 IB，可追踪每个 QP 的操作详情

---

### 2.14 CE (Copy Engine) 事件指标（v6）

#### CE Coll 事件

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `func` | string | CE 操作名称 | "CeAllGather" |
| `seqNumber` | uint64_t | 序列号 | 1000 |
| `count` | size_t | 元素数量 | 1048576 |
| `datatype` | string | 数据类型 | "ncclFloat32" |
| `root` | int | 根节点 | 0 |
| `syncStrategy` | string | 同步策略 | "MC"/"UC" |
| `stream` | pointer | CUDA Stream | 0x7f8b2c000000 |
| `eventId` | uint64_t | 事件 ID | 5000 |
| `timingMode` | enum | 计时模式 | `CE_TIMING_GPU` / `CE_TIMING_CPU` |
| `cpuStartTime` | double | CPU 开始时间 | 123456.78 |
| `cpuStopTime` | double | CPU 结束时间 | 123478.90 |
| `cpuDuration` | double | CPU 测量持续时间 | 22.12 |
| `elapsedTime` | uint64_t | GPU/CPU 测量时间 (微秒) | 22 |
| `startCompleted` | bool | 开始事件是否完成 | true |
| `stopCompleted` | bool | 停止事件是否完成 | true |

#### CE Sync 事件

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `isComplete` | bool | 是否为 Complete 同步 | true/false |
| `seqNumber` | uint32_t | 同步序列号 | 100 |
| `nRanks` | int | 参与 rank 数 | 8 |
| `timingMode` | enum | 计时模式 | `CE_TIMING_GPU` |

#### CE Batch 事件

| 指标名 | 数据类型 | 说明 | 示例值 |
|--------|----------|------|--------|
| `numOps` | int | 批处理操作数 | 10 |
| `totalBytes` | size_t | 总字节数 | 10485760 |
| `useIntraSync` | bool | 是否使用批内同步 | true |

**功能说明**：
- CE 事件用于追踪 Copy Engine 操作（非 SM Kernel）
- 支持 GPU 计时（cudaEvent）和 CPU 计时两种模式
- Poller 线程异步收集 CE 事件完成状态

---

## 三、Inspector 监控指标详表

### 3.1 通信器 (Communicator) 元数据

| 指标名 | 数据类型 | 说明 | 数据来源 |
|--------|----------|------|----------|
| `commHash` | uint64_t | 通信器唯一哈希 | NCCL 内部生成 |
| `commHashStr` | string | 哈希字符串表示 | 16进制编码 |
| `commName` | string | 通信器名称 | 用户指定或自动生成 |
| `rank` | int | 当前进程 rank | NCCL 初始化参数 |
| `nranks` | int | 通信器总 rank 数 | NCCL 初始化参数 |
| `nnodes` | int | 节点数 | NCCL 拓扑检测 |
| `cudaDeviceId` | int | CUDA 设备 ID | 当前 GPU 设备 |
| `deviceUuidStr` | string | GPU UUID 字符串 | `nvmlDeviceGetUUID` |
| `hostname` | string | 主机名 | `gethostname()` |
| `pid` | int | 进程 ID | `getpid()` |

---

### 3.2 集合操作性能指标

| 指标名 | 数据类型 | 说明 | 单位 | 计算公式 |
|--------|----------|------|------|----------|
| `coll` | string | 集合操作类型 | - | AllReduce/AllGather/... |
| `coll_sn` | uint64_t | 序列号 | - | 单调递增 |
| `coll_msg_size_bytes` | uint64_t | 消息大小 | bytes | `count * typeSize` |
| `coll_exec_time_us` | uint64_t | 执行时间 | 微秒 | `tsCompleted - tsStart` |
| `coll_timing_source` | string | 时间源 | - | kernel_gpu/kernel_cpu/collective_cpu |
| `coll_algobw_gbs` | double | 算法带宽 | GB/s | `msgSize / execTime` |
| `coll_busbw_gbs` | double | 总线带宽 | GB/s | `algoBw * factor` |

**带宽计算公式**：
- **算法带宽** (`algoBw`): 实际数据传输速率
- **总线带宽** (`busBw`): 考虑算法因子后的等效带宽

| 操作类型 | 带宽因子 |
|----------|----------|
| AllReduce | 2 * (n-1)/n |
| AllGather | (n-1)/n |
| ReduceScatter | (n-1)/n |
| Broadcast | 1 |

---

### 3.3 事件追踪详细指标 (Verbose Mode)

#### 集合操作事件追踪

| 指标名 | 数据类型 | 说明 |
|--------|----------|------|
| `event_trace_sn.coll_start_sn` | uint64_t | Coll 开始序列号 |
| `event_trace_sn.coll_stop_sn` | uint64_t | Coll 停止序列号 |
| `event_trace_ts.coll_start_ts` | uint64_t | Coll 开始时间戳 (微秒) |
| `event_trace_ts.coll_stop_ts` | uint64_t | Coll 停止时间戳 (微秒) |

#### Kernel 通道事件追踪

| 指标名 | 数据类型 | 说明 |
|--------|----------|------|
| `channel_id` | int | 通道 ID |
| `kernel_start_sn` | uint64_t | Kernel 开始序列号 |
| `kernel_stop_sn` | uint64_t | Kernel 停止序列号 |
| `kernel_record_sn` | uint64_t | Kernel 记录序列号 |
| `kernel_start_ts` | uint64_t | Kernel 开始时间戳 |
| `kernel_stop_ts` | uint64_t | Kernel 停止时间戳 |
| `kernel_record_ts` | uint64_t | Kernel 记录时间戳 |

**序列号说明**：
- `sn` (Sequence Number): 用于排序和关联事件的单调递增编号
- 不同事件类型有独立的序列号空间

---

### 3.4 Prometheus 格式指标

#### 基础指标

| 指标名 | 类型 | 说明 | 标签 |
|--------|------|------|------|
| `nccl_algorithm_bandwidth_gbs` | Gauge | 算法带宽 | comm_id, hostname, rank, collective, message_size, ... |
| `nccl_bus_bandwidth_gbs` | Gauge | 总线带宽 | 同上 |
| `nccl_collective_exec_time_microseconds` | Gauge | 执行时间 | 同上 |

#### Prometheus 标签 (Labels)

| 标签名 | 说明 | 示例值 |
|--------|------|--------|
| `comm_id` | 通信器哈希 | "abc123def456" |
| `hostname` | 主机名 | "node01" |
| `rank` | 当前 rank | "2" |
| `slurm_job` | SLURM 作业名 | "my_job" |
| `slurm_job_id` | SLURM 作业 ID | "12345" |
| `nranks` | 总 rank 数 | "8" |
| `n_nodes` | 节点数 | "2" |
| `gpu_device_id` | GPU 设备 | "GPU0" |
| `collective` | 集合操作类型 | "AllReduce" |
| `coll_sn` | 序列号 | "100" |
| `timestamp` | ISO 8601 时间戳 | "2026-03-26T10:00:00Z" |
| `message_size` | 消息大小 | "4.00MB" |

**功能说明**：
- Prometheus 格式适合与 Prometheus + Grafana 集成
- 每个 GPU 设备有独立的 `.prom` 文件
- 支持自动文件轮转（flush）

---

### 3.5 JSON 格式输出结构

```json
{
  "header": {
    "id": "comm_hash_str",
    "rank": 0,
    "n_ranks": 8,
    "nnodes": 2
  },
  "metadata": {
    "inspector_output_format_version": "v4.0",
    "git_rev": "abc123",
    "rec_mechanism": "nccl_profiler_interface",
    "dump_timestamp_us": 1234567890,
    "hostname": "node01",
    "pid": 1234
  },
  "coll_perf": {
    "coll": "AllReduce",
    "coll_sn": 100,
    "coll_msg_size_bytes": 4194304,
    "coll_exec_time_us": 1234,
    "coll_timing_source": "kernel_gpu",
    "coll_algobw_gbs": 3.40,
    "coll_busbw_gbs": 5.95,
    "event_trace_sn": { ... },
    "event_trace_ts": { ... }
  }
}
```

---

## 四、指标对比与选择指南

### 4.1 Profiler vs Inspector 指标对比

| 指标类别 | Example Profiler | Inspector | 说明 |
|----------|------------------|-----------|------|
| **时间粒度** | 微秒级 | 微秒级 | 两者都使用微秒级时间戳 |
| **事件类型** | 15+ 种 | 聚焦集合操作 | Profiler 更详细 |
| **网络细节** | Proxy Step/Net Plugin | 无 | Profiler 提供网络层细节 |
| **带宽计算** | 无 | 自动计算 | Inspector 提供 algo/bus BW |
| **实时输出** | 否（finalize 时） | 是（dump 线程） | Inspector 支持实时监控 |
| **聚合视图** | 时间线 | 统计指标 | Inspector 更适合趋势分析 |
| **CUDA Graph** | 支持 | 支持 | 都支持 Graph 追踪 |
| **CE 事件** | 完整支持 | 不支持 | Profiler v6 特有 |
| **Prometheus** | 不支持 | 支持 | Inspector 特有 |

### 4.2 使用场景推荐

| 场景 | 推荐工具 | 关键指标 |
|------|----------|----------|
| **性能瓶颈定位** | Profiler | Proxy Step 状态时间戳、Kernel Channel 时间 |
| **算法/协议分析** | Profiler | Collective 的 algo/proto 字段 |
| **集群监控** | Inspector | Prometheus 指标、algo/bus BW |
| **长期趋势分析** | Inspector | JSON/Prometheus 历史数据 |
| **网络问题诊断** | Profiler | Net Plugin 事件、Proxy Step |
| **CUDA Graph 分析** | Profiler | graphCaptured 标志、Group API |
| **CE 操作分析** | Profiler | CE Coll/Sync/Batch 事件 |

---

## 五、配置参考

### 5.1 Profiler 环境变量

```bash
# 启用 Profiler
export NCCL_PROFILE_EVENT_MASK=0x1FFF  # 启用所有事件

# 内存池配置
export NCCL_PROFILE_COLL_POOL_SIZE=128
export NCCL_PROFILE_PROXY_STEP_POOL_SIZE=256
export NCCL_PROFILE_CE_COLL_POOL_SIZE=64

# 输出文件
export NCCL_PROFILE_DUMP_FILE=/path/to/trace
```

### 5.2 Inspector 环境变量

```bash
# 启用 Inspector
export NCCL_INSPECTOR_ENABLE=1

# 输出目录
export NCCL_INSPECTOR_DUMP_DIR=/path/to/output

# 输出格式
export NCCL_INSPECTOR_PROMETHEUS=1  # Prometheus 格式
export NCCL_INSPECTOR_DUMP_VERBOSE=1  # 包含 event_trace

# Dump 线程间隔（微秒）
export NCCL_INSPECTOR_DUMP_INTERVAL_USECS=1000000  # 1秒

# SLURM 集成（自动读取）
export SLURM_JOBID=12345
export SLURM_JOB_NAME=my_job
```

---

## 六、指标准确性说明

### 6.1 时间戳精度

| 时间源 | 精度 | 说明 |
|--------|------|------|
| CPU (`gettime()`) | ~1 微秒 | `std::chrono::steady_clock` |
| GPU (`%globaltimer`) | ~1 纳秒 | PTX 指令读取 GPU 时钟 |
| CUDA Event | ~0.5 微秒 | `cudaEventElapsedTime` |

### 6.2 开销影响

| 事件类型 | 典型开销 | 建议 |
|----------|----------|------|
| API 事件 | <1% | 可安全启用 |
| Kernel 事件 | ~2% | 性能分析时启用 |
| Proxy Op | ~3% | 网络分析时启用 |
| Proxy Step | ~8% | 详细诊断时启用 |
| Net Plugin | ~5% | 网络问题诊断时启用 |
| CE 事件 | ~3% | CE 分析时启用 |

---

*文档基于 NCCL Profiler 和 Inspector 源代码分析生成*
