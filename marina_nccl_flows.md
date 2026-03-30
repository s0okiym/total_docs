# NCCL 关键流程图集

本文档以纯 Mermaid 流程图形式，系统梳理 NCCL 所有核心机制与流程。

---

## 目录

1. [通信器初始化总流程](#1-通信器初始化总流程)
2. [拓扑发现与构建](#2-拓扑发现与构建)
3. [图搜索与 Channel 分配](#3-图搜索与-channel-分配)
4. [Transport 连接建立](#4-transport-连接建立)
5. [集合通信 API 调用入口](#5-集合通信-api-调用入口)
6. [Group 操作机制](#6-group-操作机制)
7. [线程本地状态与 Comm 链表管理](#7-线程本地状态与-comm-链表管理)
8. [任务累积与排序](#8-任务累积与排序)
9. [算法与协议选择](#9-算法与协议选择)
10. [Channel-Based Distribution (CBD)](#10-channel-based-distribution-cbd)
11. [Kernel Plan 构建](#11-kernel-plan-构建)
12. [groupLaunch 五阶段](#12-grouplaunch-五阶段)
13. [doLaunches 核心调度循环](#13-dolaunches-核心调度循环)
14. [ncclLaunchPrepare 详解](#14-nccllaunchprepare-详解)
15. [ncclLaunchKernel 与 GPU 启动配置](#15-nccllaunchkernel-与-gpu-启动配置)
16. [Proxy 操作流程](#16-proxy-操作流程)
17. [Work 数据上传与存储](#17-work-数据上传与存储)
18. [异步任务生命周期](#18-异步任务生命周期)
19. [阻塞与非阻塞模式对比](#19-阻塞与非阻塞模式对比)
20. [Ring AllReduce Device Kernel](#20-ring-allreduce-device-kernel)
21. [Tree AllReduce Device Kernel](#21-tree-allreduce-device-kernel)
22. [NVLS AllReduce Device Kernel](#22-nvls-allreduce-device-kernel)
23. [Device Kernel 线程角色分配](#23-device-kernel-线程角色分配)
24. [Simple 协议原语](#24-simple-协议原语)
25. [LL 协议原语](#25-ll-协议原语)
26. [LL128 协议原语](#26-ll128-协议原语)
27. [内存管理栈机制](#27-内存管理栈机制)
28. [错误处理与 Abort 机制](#28-错误处理与-abort-机制)

---

## 1. 通信器初始化总流程

```mermaid
flowchart TD
    A["ncclCommInitRank(newcomm, nranks, commId, myrank)"] --> B[ncclInit: 全局初始化]
    B --> C[cudaFree NULL: 初始化 CUDA Runtime]
    C --> D[分配 ncclComm 结构体]
    D --> E[分配 abortFlag / abortFlagDev]
    E --> F[parseCommConfig: 解析配置]
    F --> G["创建 ncclCommInitRankAsyncJob"]
    G --> H["ncclAsyncLaunch → ncclCommInitRankFunc"]

    H --> I["cudaSetDevice(cudaDev)"]
    I --> J["ncclInitKernelsForDevice: 查询 kernel 属性, 设置 smem/carveout"]
    J --> K["commAlloc: 分配 comm 内存"]
    K --> L["bootstrapInit: 建立 bootstrap 通信"]
    L --> M["initTransportsRank: 核心 Transport 初始化"]

    M --> M1["AllGather1: 交换 peerInfo"]
    M1 --> M2["ncclTopoGetSystem: 拓扑发现"]
    M2 --> M3["ncclTopoComputePaths: 计算路径"]
    M3 --> M4["ncclTopoTrimSystem: 裁剪无用节点"]
    M4 --> M5["ncclTopoCompute: 计算 Ring/Tree/CollNet/NVLS 图"]
    M5 --> M6["AllGather3: 交换 graphInfo + topoRanks"]
    M6 --> M7["ncclTopoPostset: 构建 ring/tree 拓扑"]
    M7 --> M8["setupChannel: 设置 channel 结构"]
    M8 --> M9["ncclTransportRingConnect: Ring 连接"]
    M9 --> M10["ncclTransportTreeConnect: Tree 连接"]
    M10 --> M11["ncclNvlsSetup + BufferSetup"]
    M11 --> M12["ncclProxyCreate: 启动 Proxy 线程"]
    M12 --> M13["ncclProxyConnect: 连接 Proxy"]
    M13 --> M14["初始化完成 comm->initState = ncclSuccess"]
```

## 2. 拓扑发现与构建

```mermaid
flowchart TD
    A["ncclTopoGetSystem(comm, &topo)"] --> B[探测本地 GPU: NVML]
    B --> C[探测本地 NIC: net plugin]
    C --> D[探测 CPU 拓扑: PCI / NUMA]
    D --> E{有共享 topo?}
    E -->|是| F["从共享内存加载 topo XML"]
    E -->|否| G["ncclTopoGetSystemFromXml: 构建 system graph"]
    G --> H["构建 XML 节点: GPU/NIC/CPU/PCI"]

    H --> I["ncclTopoComputePaths(comm->topo, comm)"]
    I --> I1["遍历所有 GPU-NIC 对"]
    I1 --> I2["BFS 搜索最短路径"]
    I2 --> I3["计算路径带宽和类型<br/>LOC/NVL/NVB/PIX/PXB/PHB/SYS/NET"]
    I3 --> I4["存储 path[type, bandwidth]"]

    I4 --> J["ncclTopoTrimSystem: 移除不可达 GPU 和无用 NIC"]
    J --> K["重新计算 ncclTopoComputePaths"]

    K --> L["ncclTopoSearchInit: 初始化搜索"]
    L --> M["ncclTopoPrint: 打印最终拓扑"]
```

```mermaid
flowchart LR
    subgraph "路径类型层级"
        A["LOC (本地)"] --> B["NVL (NVLink 直连)"]
        B --> C["NVB (NVLink 桥接)"]
        C --> D["PIX (同 PCI 交换机)"]
        D --> E["PXB (跨 PCI 交换机, 同 CPU)"]
        E --> F["PHB (同 CPU, 跨 PCIe)"]
        F --> G["SYS (跨 CPU/NUMA)"]
        G --> H["NET (网络)"]
    end
```

## 3. 图搜索与 Channel 分配

```mermaid
flowchart TD
    A["对每种算法计算 ncclTopoCompute(topo, graph)"] --> B{算法类型?}

    B -->|Ring| C1["pattern = NCCL_TOPO_PATTERN_RING<br/>minChannels=1, maxChannels=MAXCHANNELS/2"]
    B -->|Tree| C2["pattern = NCCL_TOPO_PATTERN_BALANCED_TREE<br/>minChannels=ring.nChannels"]
    B -->|CollNet Chain| C3["pattern = NCCL_TOPO_PATTERN_TREE<br/>collNet=1"]
    B -->|CollNet Direct| C4["pattern = NCCL_TOPO_PATTERN_COLLNET_DIRECT<br/>collNet=1"]
    B -->|NVLS| C5["pattern = NCCL_TOPO_PATTERN_NVLS"]

    C1 --> D["ncclTopoSearch: 搜索最优拓扑"]
    D --> E["贪心搜索: 最大化 channel 数量"]
    E --> F["计算 bwIntra / bwInter"]
    F --> G["确定 nChannels / sameChannels"]

    G --> H["ncclTopoPreset: 预设 channel 结构"]
    H --> I["AllGather3: 交换所有 rank 的 graphInfo"]
    I --> J["取所有 rank 的最小 nChannels / 最小带宽"]
    J --> K["ncclTopoPostset: 构建最终 ring/tree 拓扑"]
    K --> L["为每个 channel 分配 ring.prev/next 和 tree.up/down"]
```

```mermaid
flowchart TD
    subgraph "Channel 结构 (per channel)"
        CH["ncclChannel"]
        CH --> RING["Ring: prev, next, userRanks[nRanks]"]
        CH --> TREE["Tree: up, down[0..2]"]
        CH --> COLLNET["CollNet: chain/direct"]
        CH --> NVLS_CH["NVLS: scatter/gather"]
        CH --> BUF["Buffers:LL/LL128/Simple<br/>sendBuff/recvBuff"]
        CH --> CONN["Connections: send/recv connectors<br/>对每个 rank"]
    end
```

## 4. Transport 连接建立

```mermaid
flowchart TD
    A["initTransportsRank → 连接阶段"] --> B["setupChannel(c): 初始化 channel 基本结构"]
    B --> C["ncclTransportRingConnect(comm)"]
    C --> C1["对每个 channel 的每个 rank 对"]
    C1 --> C2["建立 send/recv transport 连接"]
    C2 --> C3["P2P transport (NVLink/PCIe) 或<br/>NET transport (IB/Socket)"]
    C3 --> C4["交换连接信息 via Bootstrap"]
    C4 --> C5["分配 send/recv buffer"]

    C5 --> D["ncclTransportTreeConnect(comm)"]
    D --> D1["对每个 channel 的 tree 父子关系"]
    D1 --> D2["建立 tree up/down 连接"]
    D2 --> D3["分配 tree 专用 buffer"]

    D3 --> E["ncclNvlsSetup: NVLS 设置"]
    E --> F["ncclNvlsBufferSetup: NVLS 缓冲区"]
    F --> G["ncclNvlsTreeConnect: NVLS Tree 连接"]

    G --> H["ncclCollNetSetup: CollNet 设置 (可选)"]
    H --> I["ncclCollNetChainBufferSetup"]
    I --> J["ncclCollNetDirectBufferSetup"]

    J --> K["ncclProxyConnect: 连接到 Proxy 线程"]
    K --> L["ncclP2PConnect: P2P 连接"]
    L --> M["ncclProxyConnect: 所有连接注册到 Proxy"]

    style C3 fill:#f9f,stroke:#333
    style K fill:#bbf,stroke:#333
```

## 5. 集合通信 API 调用入口

```mermaid
flowchart TD
    A["ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)"] --> B["构造 ncclInfo 结构"]
    B --> C["ncclEnqueueCheck(&info)"]
    C --> D["CommCheck: 检查 comm 有效性"]
    D --> E["检查 comm->revokedFlag"]
    E --> F["ncclGroupStartInternal: depth++"]

    F --> G["ncclCommEnsureReady: 确保通信器初始化完成"]
    G --> H["ArgsCheck: 参数校验"]
    H --> I["taskAppend(comm, info)"]

    I --> J{任务类型判断}
    J -->|"单 Rank"| K["ncclLaunchOneRank: 直接 D2D 拷贝"]
    J -->|"P2P"| L["p2pTaskAppend: 加入 peers[rank].sendQueue/recvQueue"]
    J -->|"Collective"| M["collTaskAppend"]

    M --> M1["ncclGroupCommJoin: 加入 ncclGroupCommHead 链表"]
    M1 --> M2["ncclMemoryStackPush: 推入内存栈"]
    M2 --> M3["初始化 planner (如果是首次加入)"]
    M3 --> M4["分配 ncclTaskColl"]
    M4 --> M5["计算 trafficBytes = count × elemSize × trafficPerByte"]
    M5 --> M6["ncclTaskCollSorterInsert: 按大小降序插入"]
    M6 --> M7["planner.nTasksColl++"]

    K --> N["ncclGroupEndInternal: depth--"]
    L --> N
    M7 --> N
    N --> O{"depth == 0?"}
    O -->|否| P["返回,等待更多操作"]
    O -->|是| Q["触发 Group 执行 → 见第6节"]
```

## 6. Group 操作机制

```mermaid
flowchart TD
    A["ncclGroupEndInternal()"] --> B["--ncclGroupDepth → 检查是否为0"]
    B --> C{depth > 0?}
    C -->|是| D["直接返回 (嵌套 Group)"]
    C -->|否| E["检查 ncclGroupError"]

    E --> F["创建 ncclGroupJob"]
    F --> F1["memcpy(groupJob->groupCommHead, ncclGroupCommHead)"]
    F1 --> F2["groupJob->groupCommPreconnectHead = ncclGroupCommPreconnectHead"]
    F2 --> F3["ncclIntruQueueTransfer(asyncJobs)"]
    F3 --> F4["groupJob->abortFlag = false"]

    F4 --> G{ncclGroupBlocking?}
    G -->|"阻塞 (1)"| H["groupLaunch(&groupJob->base, simInfo)<br/>在当前线程同步执行"]
    H --> H1["delete groupJob"]

    G -->|"非阻塞 (0)"| I["设置 comm->groupJob 引用"]
    I --> I1["计算 groupRefCount"]
    I1 --> I2["groupJob->base.func = groupLaunchNonBlocking"]
    I2 --> I3["STDTHREADCREATE: 创建后台线程"]
    I3 --> I4["返回 ncclInProgress"]

    H1 --> J["groupLocalResetJobState<br/>清空所有 thread_local 变量"]
    I4 --> J
```

## 7. 线程本地状态与 Comm 链表管理

```mermaid
flowchart TD
    subgraph "Thread-Local 变量"
        V1["ncclGroupDepth: Group 嵌套深度"]
        V2["ncclGroupError: 累积错误"]
        V3["ncclGroupCommHead[type]: 通信器链表头<br/>[Collective, P2P, SymRegister, ...]"]
        V4["ncclGroupCommPreconnectHead: 需预连接链表"]
        V5["ncclAsyncJobs: 异步任务队列"]
        V6["ncclGroupBlocking: 阻塞模式 (-1/0/1)"]
    end

    subgraph "ncclGroupCommJoin 排列策略"
        S1["comm->groupNext[type] == 0x1?"] -->|首次| S2["按 intraComm0 分组<br/>同进程 comm 相邻"]
        S2 --> S3["同组内按 commHash 升序"]
        S3 --> S4["插入到 ncclGroupCommHead[type] 链表"]
        S4 --> S5["ncclMemoryStackPush(comm->memScoped)"]
        S5 --> S6["memset(&comm->planner, 0)<br/>初始化 planner"]
    end
```

```mermaid
flowchart LR
    subgraph "ncclGroupCommHead[Collective] 链表示例"
        H["HEAD"] --> C1["Comm A<br/>intraComm0=X"]
        C1 --> C2["Comm B<br/>intraComm0=X"]
        C2 --> C3["Comm C<br/>intraComm0=Y"]
        C3 --> C4["Comm D<br/>intraComm0=Y"]
    end
```

## 8. 任务累积与排序

```mermaid
flowchart TD
    A["多次 ncclAllReduce 调用"] --> B["collTaskAppend → 创建 ncclTaskColl"]
    B --> C["ncclTaskCollSorterInsert"]

    subgraph "ncclTaskCollSorter 排序"
        C --> D["将 trafficBytes 映射到 64 个 bin<br/>对数分桶: bin = 63 - clz(size)"]
        D --> E["插入到 bin 头部 (LIFO)"]
    end

    subgraph "ncclTaskCollSorterDequeueAll"
        E --> F["按 bin 63→0 顺序链接<br/>实现近似降序排列"]
        F --> G["返回按 trafficBytes 降序的任务链表"]
    end

    G --> H["ncclPrepareTasks"]
    H --> I["按 (func, op, datatype) 分桶<br/>tasksByFnOpTy[func×ops×types]"]
    I --> J["聚合大小在 4X 以内的任务"]
    J --> K["对每组调用 ncclGetAlgoInfo<br/>选择算法和协议"]
    K --> L["按 (collnet, nvls) 分到 4 个 bin"]
    L --> M["拼接为 collTaskQueue<br/>顺序: std→nvls→collnet→collnet+nvls"]
```

```mermaid
flowchart LR
    subgraph "任务排序示例"
        A["AllReduce 1MB"] --> B["AllReduce 512KB"]
        B --> C["ReduceScatter 256KB"]
        C --> D["AllGather 128KB"]
        D --> E["AllReduce 64KB"]
    end
```

## 9. 算法与协议选择

```mermaid
flowchart TD
    A["ncclGetAlgoInfo(comm, task, ...)"] --> B["计算 nvlsSupport"]
    B --> C["计算 collNetSupport"]
    C --> D["估算 nTasksPerChannel"]

    D --> E["遍历所有 (algo, proto) 组合"]
    E --> F["对每个组合计算代价"]

    subgraph "代价计算"
        F --> G["latency = latencies[func][algo][proto]"]
        G --> H["bandwidth = bandwidths[func][algo][proto]"]
        H --> I["cost = latency + trafficBytes / bandwidth"]
    end

    I --> J["选择最小 cost 的 (algo, proto)"]
    J --> K["确定 nMaxChannels"]
    K --> L["确定 nWarps"]

    L --> M["设置 task->algorithm, protocol"]
    M --> N["设置 task->devFuncId = ncclDevFuncId(func, op, dtype, algo, proto)"]

    subgraph "算法选项"
        A1["NCCL_ALGO_RING"]
        A2["NCCL_ALGO_TREE"]
        A3["NCCL_ALGO_NVLS"]
        A4["NCCL_ALGO_NVLS_TREE"]
        A5["NCCL_ALGO_COLLNET_CHAIN"]
        A6["NCCL_ALGO_COLLNET_DIRECT"]
    end

    subgraph "协议选项"
        P1["NCCL_PROTO_SIMPLE: 大消息, ~100%效率"]
        P2["NCCL_PROTO_LL128: 中等消息, ~94%效率"]
        P3["NCCL_PROTO_LL: 小消息, ~50%效率, 低延迟"]
    end
```

## 10. Channel-Based Distribution (CBD)

```mermaid
flowchart TD
    A["scheduleCollTasksToPlan: 为任务分配 channel 范围"] --> B["计算 trafficPerChannel"]
    B --> C["将 task->count 按元素切分为 cells"]

    subgraph "CBD 切分策略"
        C --> D["cellsPerChannel = ⌈trafficPerChannel / trafficPerCell⌉"]
        D --> E["cellsLo: 当前 channel 剩余容量"]
        E --> F["cellsMid: 完整的中间 channel"]
        F --> G["cellsHi: 最后一个 channel 的余量"]
    end

    G --> H["countLo = cellsLo × elementsPerCell"]
    H --> I["countMid = cellsPerChannel × elementsPerCell"]
    I --> J["countHi = cellsHi × elementsPerCell<br/>(修正: countLo+nMid×countMid+countHi == task->count)"]

    J --> K["devWork->channelLo = channelId"]
    K --> L["devWork->channelHi = channelId + nChannels - 1"]
    L --> M["devWork->cbd.countLo/Mid/Hi"]
    M --> N["devWork->cbd.chunkGrainsLo/Mid/Hi"]

    N --> O["更新 channelId 和 currentTraffic<br/>移向下一个 channel"]
```

```mermaid
flowchart LR
    subgraph "CBD 示例: task 分配到 4 个 channel"
        direction LR
        CH0["Channel 0<br/>countLo"] --> CH1["Channel 1<br/>countMid"]
        CH1 --> CH2["Channel 2<br/>countMid"]
        CH2 --> CH3["Channel 3<br/>countHi"]
    end
```

## 11. Kernel Plan 构建

```mermaid
flowchart TD
    A["ncclLaunchPrepare: 循环构建 Plan"] --> B["分配 ncclKernelPlan"]
    B --> C["计算 budget<br/>inArgsBytes = workArgsBytes - sizeof(DevKernelArgs)<br/>outArgsBytes = fifoBytes/2 或 1GB"]

    C --> D{有 RMA 任务?}
    D -->|是| E["scheduleRmaTasksToPlan"]
    D -->|否| F{有 CE 任务?}
    F -->|是| G["构建 CE Plan: isCeColl=true"]
    F -->|否| H["scheduleCollTasksToPlan (消耗集合任务)"]

    H --> I{"还有 Bcast?"}
    I -->|是| J["ncclScheduleBcastTasksToPlan"]
    I -->|否| K{"还有 P2P?"}
    K -->|是| L["scheduleP2pTasksToPlan"]
    K -->|否| M["finishPlan"]

    M --> M1["确定 workStorageType: Args/Fifo/Persistent"]
    M1 --> M2["分配 kernelArgs (ncclMemoryStackAlloc)"]
    M2 --> M3["设置 kernelArgs->comm, channelMask"]
    M3 --> M4["按 channel 轮转排布 WorkBatch<br/>确保 blockIdx.x 直接索引"]
    M4 --> M5["归并排序各 channel 的 ProxyOp<br/>按 opCount 合并到 plan->proxyOpQueue"]
    M5 --> M6["设置 kernelFn, threadPerBlock"]

    M6 --> N{"所有任务消耗完?"}
    N -->|否| O["继续循环,构建下一个 Plan"]
    N -->|是| P["设置 unlaunchedPlansHead"]

    P --> Q["流同步: launchStream 等待所有 userStream"]
    Q --> R["注册 host callback (如有 proxyOps)"]
```

## 12. groupLaunch 五阶段

```mermaid
flowchart TD
    A["groupLaunch(groupJob)"] --> B

    subgraph B["阶段1: P2P 预连接"]
        B1["遍历 groupCommPreconnectHead"]
        B1 --> B2["为每个 comm 创建 PreconnectJob<br/>func = ncclP2PPreconnectFunc"]
        B2 --> B3["asyncJobLaunch: 并行执行<br/>ncclTransportP2pSetup"]
    end

    B3 --> C

    subgraph C["阶段2: 对称注册"]
        C1["遍历 SymRegister 类型 comm"]
        C1 --> C2["创建 SymmetricJob<br/>func = ncclCommGroupRegisterSymmetric"]
        C2 --> C3["asyncJobLaunch: 并行注册<br/>处理 devr/collReg/ceInit 任务"]
    end

    C3 --> D

    subgraph D["阶段3: 集合任务准备与连接"]
        D1["按 clique 分批处理 comm"]
        D1 --> D2["ncclPrepareTasksAndCollPreconnect<br/>对每个 comm"]
        D2 --> D3["ncclPrepareTasks: 算法选择 + Work 构建"]
        D3 --> D4["asyncJobLaunch: 并行执行"]
        D4 --> D5["所有 clique 完成后<br/>ncclTasksRegAndEnqueue: 缓冲区注册"]
    end

    D5 --> E

    subgraph E["阶段4: 实际 Launch"]
        E1["doLaunches(groupCommHead[Collective])<br/>见第13节详细流程"]
    end

    E1 --> F

    subgraph F["阶段5: 清理"]
        F1["释放异步任务 (destructor)"]
        F1 --> F2["ncclGroupCommLeave: 内存栈 Pop"]
        F2 --> F3["重置 planner"]
        F3 --> F4["ncclCommPollCallbacks: 轮询回调"]
    end
```

## 13. doLaunches 核心调度循环

```mermaid
flowchart TD
    A["doLaunches(head)"] --> B["cliqueHead = head"]

    subgraph "外层循环: 遍历 clique"
        B --> C["comm = cliqueHead"]
        C --> D["遍历同 clique 的 comm"]
        D --> D1["cudaSetDevice(comm->cudaDev)"]
        D1 --> D2["ncclLaunchPrepare(comm)<br/>构建所有 Plan"]
        D2 --> D3["ncclCommIntraBarrierIn(comm, 1)<br/>进程内屏障同步"]
        D3 --> D4["下一个 comm"]
        D4 --> D5{"同 clique?"}
        D5 -->|是| D1
        D5 -->|否| E["cliqueNextHead = comm"]
    end

    E --> F

    subgraph "内层循环: 逐轮 launch"
        F --> G["comm = cliqueHead"]
        G --> H["检查 moreRounds<br/>(barrier 结果或 unlaunchedPlansHead)"]

        H --> I{moreRounds?}
        I -->|是| J["取出 plan = unlaunchedPlansHead"]
        J --> K["ncclLaunchKernelBefore: uploadWork"]
        K --> L["ncclLaunchKernel: cuLaunchKernelEx"]
        L --> M["ncclLaunchKernelAfter: proxyOps"]
        M --> N["下一个 comm"]

        I -->|否| O["ncclLaunchFinish(comm)"]
        O --> N

        N --> P{"同 clique?"}
        P -->|是| H
        P -->|否| Q{"moreRounds?"}
        Q -->|是| F
        Q -->|否| R["cliqueHead = cliqueNextHead"]
    end

    R --> S{"cliqueHead != NULL?"}
    S -->|是| C
    S -->|否| T["完成"]
```

## 14. ncclLaunchPrepare 详解

```mermaid
flowchart TD
    A["ncclLaunchPrepare(comm)"] --> B["循环: memset wipPlan, 分配 plan"]

    B --> C["计算 budget"]
    C --> D["消耗任务到 plan 中"]

    D --> E["所有 plan 构建完成"]
    E --> F["设置 unlaunchedPlansHead"]

    F --> G["流同步准备"]
    G --> G1["ncclStrongStreamAcquire(deviceStream)"]
    G1 --> G2["userStream[0] 等待所有 userStream[i]<br/>via cudaEventRecord + cudaStreamWaitEvent"]
    G2 --> G3["userStream[0] 等待 deviceStream"]

    G3 --> H{需要 host callback?}
    H -->|"persistent 或 launchBlocking"| I["cudaLaunchHostFunc(hostStream, callback)"]
    I --> J["launchStream 等待 hostStream"]
    H -->|否| K["跳过"]

    I --> L["ncclStrongStreamRelease"]
    K --> L

    subgraph "uploadWork 方式"
        W1{"workStorageType?"}
        W1 -->|Args| W2["数据已在 kernelArgs 中"]
        W1 -->|Fifo| W3["写入环形缓冲区 workFifoBuf<br/>更新 workFifoProduced"]
        W1 -->|Persistent| W4["数据在持久化缓冲区"]
    end
```

## 15. ncclLaunchKernel 与 GPU 启动配置

```mermaid
flowchart TD
    A["ncclLaunchKernel(comm, plan)"] --> B["grid = {nChannels, 1, 1}<br/>block = {threadPerBlock, 1, 1}<br/>smem = ncclShmemDynamicSize(cudaArch)"]

    B --> C["准备 kernel args<br/>extra = {BUFFER_PTR, BUFFER_SIZE, END}"]

    C --> D["cudaGetFuncBySymbol → CUfunction"]

    D --> E["构建 CUlaunchConfig + CUlaunchAttribute"]
    E --> F{compCap >= 90?}

    F -->|是| G["Cluster Dimension (CGA)<br/>clusterSize = config.cgaClusterSize<br/>grid.x 必须整除 clusterSize"]
    G --> H["Cluster Scheduling Policy: SPREAD"]
    F -->|否| I["跳过 CGA"]

    H --> J{CUDA >= 12.0 且 compCap >= 90?}
    J -->|是| K["Mem Sync Domain: Remote<br/>减少跨 SM 同步开销"]
    J -->|否| L["跳过"]

    K --> M{CUDA >= 12.3?}
    M -->|是| N["Launch Completion Event<br/>kernel 开始执行时触发<br/>而非完成时"]
    M -->|否| O["跳过"]

    N --> P{CUDA >= 13.0 且 compCap >= 100?}
    P -->|是| Q["NVLink Util Centric Scheduling<br/>Blackwell 优化"]
    P -->|否| R["跳过"]

    Q --> S["cuLaunchKernelEx(&launchConfig, fn, nullptr, extra)"]
    R --> S
    L --> S
    O --> S
    I --> S

    style S fill:#f96,stroke:#333,stroke-width:3px
```

```mermaid
flowchart LR
    subgraph "Kernel 执行模型"
        direction TB
        G["Grid<br/>blockIdx.x = channel ID"] --> B0["Block 0<br/>Channel 0<br/>threadPerBlock threads"]
        G --> B1["Block 1<br/>Channel 1"]
        G --> B2["Block 2<br/>Channel 2"]
        G --> BN["Block N<br/>Channel N"]

        B0 --> W0["Warp 0: RecvWait"]
        B0 --> W1["Warp 1: Reduce"]
        B0 --> W2["Warp 2: SendWait"]
        B0 --> W3["Warp 3: PostSend"]
    end
```

## 16. Proxy 操作流程

```mermaid
flowchart TD
    A["hostStreamPlanCallback / hostStreamPlanTask"] --> B["遍历 plan->proxyOpQueue"]
    B --> C["ncclProxySaveOp: 优化/合并 proxy op"]
    C --> D["发送到 proxy 线程: ncclProxySendMessage"]

    D --> E["Proxy 线程接收"]
    E --> F{操作类型}

    F -->|"Ring Send"| G["通过 transport 发送数据<br/>IB verbs / Socket / P2P"]
    F -->|"Ring Recv"| H["通过 transport 接收数据<br/>写入 recv buffer"]
    F -->|"Tree Up"| I["向上发送规约结果"]
    F -->|"Tree Down"| J["向下广播结果"]
    F -->|"CollNet"| K["与 CollNet 交换数据"]
    F -->|"NVLS"| L["NVLS scatter/gather 操作"]

    G --> M["完成: ncclProxyCompleted"]
    H --> M
    I --> M
    J --> M
    K --> M
    L --> M

    M --> N["更新 device 端 step 计数器<br/>通知 kernel 数据就绪"]
```

## 17. Work 数据上传与存储

```mermaid
flowchart TD
    A["uploadWork(comm, plan)"] --> B{"workStorageType?"}

    B -->|Args| C["Work 已在 kernelArgs 中<br/>kernelArgs + batches + works<br/>一起传给 cuLaunchKernelEx"]

    B -->|Fifo| D["写入环形缓冲区"]
    D --> D1["fifoWritePtr = workFifoBuf + (workFifoProduced % workFifoBytes)"]
    D1 --> D2["memcpy(fifoWritePtr, works, workBytes)"]
    D2 --> D3["workFifoProduced += workBytes"]
    D3 --> D4["kernelArgs 中存储 fifo offset"]

    B -->|Persistent| E["Work 在持久化缓冲区<br/>(CUDA Graph 场景)"]
    E --> E1["kernelArgs 中存储持久化偏移"]

    subgraph "Kernel Args 内存布局"
        L1["ncclDevKernelArgs<br/>  comm (设备端通信器)<br/>  channelMask<br/>  workStorageType"]
        L1 --> L2["ncclDevWorkBatch[0]<br/>  for channel 0"]
        L2 --> L3["ncclDevWorkBatch[1]<br/>  for channel 0 (extends)"]
        L3 --> L4["ncclDevWorkBatch[2]<br/>  for channel 1"]
        L4 --> L5["ncclDevWorkColl<br/>  work 数据 (Args 模式)"]
    end
```

## 18. 异步任务生命周期

```mermaid
stateDiagram-v2
    state "ncclAsyncJob 状态" as states {
        [*] --> Created: ncclAsyncLaunch
        Created --> InQueue: 入队 ncclAsyncJobs
        InQueue --> Running: asyncJobLaunch 创建线程
        Running --> Done: ncclAsyncJobMain 完成<br/>ATOMIC_STORE(state, Done)
        Done --> Joined: 主线程 ncclThreadJoin
        Joined --> Freed: destructor 调用

        state "单任务优化" as opt {
            InQueue --> Done: isThreadMain=true<br/>直接在主线程执行
        }
    }
```

```mermaid
flowchart TD
    A["asyncJobLaunch(asyncJobs, abortFlag)"] --> B{"队列中只有1个任务?"}
    B -->|是| C["isThreadMain = true<br/>直接调用 ncclAsyncJobMain<br/>返回 result"]
    B -->|否| D["为每个任务创建 std::thread"]

    D --> E["轮询等待"]
    E --> F["检查每个 job 的 state<br/>ATOMIC_LOAD(state)"]
    F --> G{state?}
    G -->|Running| H["jobsDone = false<br/>usleep(1) 避免忙等"]
    G -->|Done| I["ncclThreadJoin<br/>state = Joined<br/>检查 result"]
    G -->|Joined| J["已处理,跳过"]

    H --> E
    I --> K{"有错误?"}
    K -->|是| L["设置所有 job 的 abortFlag<br/>ATOMIC_STORE(abortFlag, 1)"]
    K -->|否| M["继续"]
    M --> E
```

## 19. 阻塞与非阻塞模式对比

```mermaid
flowchart TD
    subgraph "阻塞模式"
        direction TB
        BA["ncclGroupEnd"] --> BB["groupLaunch 同步执行"]
        BB --> BC["所有阶段在用户线程完成"]
        BC --> BD["返回 ncclSuccess"]
    end

    subgraph "非阻塞模式"
        direction TB
        NB1["ncclGroupEnd"] --> NB2["创建后台线程"]
        NB2 --> NB3["groupLaunchNonBlocking"]
        NB3 --> NB4["返回 ncclInProgress"]

        NB4 --> NB5["后续操作检查完成状态"]
        NB5 --> NB6["ncclGroupJobComplete<br/>ATOMIC_EXCHANGE(joined)"]
        NB6 --> NB7["ncclAsyncJobComplete: join 线程"]
        NB7 --> NB8["refCount-- → 0? → delete"]
    end
```

## 20. Ring AllReduce Device Kernel

```mermaid
sequenceDiagram
    participant K as Kernel (每个 Block=1 Channel)
    participant Prev as 前一个 GPU
    participant Self as 当前 GPU
    participant Next as 下一个 GPU

    Note over K: Primitives<T,RedOp,FanSymmetric<1>,1,Proto,0>
    Note over K: ncclCollCbdPart → gridOffset, channelCount, chunkCount

    Note over K,Next: === Reduce-Scatter 阶段 (nRanks-1 步) ===

    loop "j = 0; j < nRanks-1; j++"
        K->>K: offset = (ringIx + nRanks - j) % nRanks × chunkSize
        alt j == 0
            K->>Next: prims.directSend(offset, nelem)
        else j < nRanks-1
            Prev->>K: prims.directRecvReduceDirectSend(offset, nelem)
        else j == nRanks-1
            Prev->>K: prims.directRecvReduceCopyDirectSend(offset, nelem, postOp=true)
        end
    end

    Note over K,Next: === AllGather 阶段 (nRanks-1 步) ===

    loop "j = 1; j < nRanks; j++"
        K->>K: offset = (ringIx + nRanks - j) % nRanks × chunkSize
        alt j < nRanks-1
            Prev->>K: prims.directRecvCopyDirectSend(offset, nelem)
        else j == nRanks-1
            Prev->>K: prims.directRecv(offset, nelem)
        end
    end
```

## 21. Tree AllReduce Device Kernel

```mermaid
flowchart TD
    A["runTreeUpDown(tid, nthreads, work)"] --> B{node 角色?}

    subgraph "Reduce 阶段 (上行)"
        B -->|"根节点 (up==-1)"| C["directRecvReduceCopy<br/>接收所有子节点, 规约到 recvbuff<br/>postOp=true: 应用规约操作"]
        B -->|"叶子节点 (down[0]==-1)"| D["directSend<br/>发送 sendbuff 到父节点"]
        B -->|"中间节点"| E["directRecvReduceDirectSend<br/>接收子节点, 规约, 转发到父节点"]
    end

    C --> F
    D --> F
    E --> F

    subgraph "Broadcast 阶段 (下行)"
        F --> F1{node 角色?}
        F1 -->|"根节点"| G["directSendFromOutput<br/>从 recvbuff 发送到子节点"]
        F1 -->|"叶子节点"| H["directRecv<br/>接收结果到 recvbuff"]
        F1 -->|"中间节点"| I["directRecvCopyDirectSend<br/>接收并转发到子节点"]
    end
```

```mermaid
flowchart TD
    A["runTreeSplit(tid, nthreads, work)"] --> B["nthreadsSplit = nthreads/2"]

    subgraph "线程分割"
        B --> C["tid < nthreadsSplit: Reduce 线程"]
        B --> D["tid >= nthreadsSplit: Broadcast 线程"]
    end

    C --> E["同时执行 Reduce 上行<br/>和 Broadcast 下行"]
    D --> E

    E --> F["Simple: 50/50 分割<br/>LL/LL128: 70/30 分割"]
```

## 22. NVLS AllReduce Device Kernel

```mermaid
flowchart TD
    A["runNvls(tid, nthreads, work)"] --> B["计算 warp 分配:<br/>totalWarps = NCCL_MAX_NTHREADS/WARP_SIZE"]
    B --> C["scatterWarps, gatherWarps, reduceWarps, bcastWarps"]

    subgraph "NVLS Scatter 阶段"
        C --> D["将数据 scatter 到所有 rank 的 NVLS buffer"]
        D --> E["使用 NVLink Switch 直接写入远程内存"]
    end

    subgraph "NVLS Gather + Reduce 阶段"
        E --> F["从所有 rank 的 scatter buffer gather 数据"]
        F --> G["本地 reduce 所有 gathered 数据"]
    end

    subgraph "NVLS Broadcast 阶段"
        G --> H["将 reduce 结果写入 NVLS multicast buffer"]
        H --> I["所有 rank 通过 NVLS 读取结果"]
    end
```

## 23. Device Kernel 线程角色分配

```mermaid
flowchart TD
    A["tid = threadIdx.x<br/>nthreads = blockDim.x"] --> B["nrecv, nsend = 连接数"]

    B --> C{"tid < nrecv?"}
    C -->|是| D["flags |= RoleWaitRecv<br/>等待接收数据就绪<br/>index = tid"]

    C -->|否| E{"tid < nrecv+nsend?"}
    E -->|是| F["flags |= RoleWaitSend<br/>等待发送槽位可用<br/>index = tid-nrecv"]

    E -->|否| G{"tid >= nthreads-nsend?"}
    G -->|是| H["flags |= RolePostSend<br/>通知发送完成<br/>index = tid-(nthreads-nsend)"]

    G -->|否| I{"tid >= nthreads-nrecv-nsend?"}
    I -->|是| J["flags |= RolePostRecv<br/>通知接收完成<br/>index = tid-(nthreads-nrecv-nsend)"]

    I -->|否| K["flags |= RoleInput | RoleOutput<br/>计算线程: 执行 reduce/copy<br/>从 recvBuff 读取, 规约, 写入 sendBuff/outputBuff"]
```

```mermaid
flowchart LR
    subgraph "线程布局 (示例: 512 threads, 1 recv, 1 send)"
        direction LR
        T0["T0<br/>WaitRecv"] --> T1["T1..T494<br/>Compute<br/>(Reduce/Copy)"]
        T1 --> T2["T495..T509<br/>空闲"]
        T2 --> T3["T510<br/>PostRecv"]
        T3 --> T4["T511<br/>PostSend"]
    end
```

## 24. Simple 协议原语

```mermaid
flowchart TD
    A["ProtoSimple"] --> B["MaxGroupWidth = 2<br/>支持一次处理 2 个 slice"]

    B --> C["同步机制: Step 计数器"]
    C --> D["发送方: 等待 connStepCache + NCCL_STEPS < step + StepPerSlice"]
    D --> E["接收方: 等待 connStepCache >= step"]

    subgraph "数据流动"
        F["directSend:<br/>直接从 sendbuff 写入远程 recvBuff"]
        F --> G["directRecvReduceDirectSend:<br/>从远程 recvBuff 读取<br/>本地 reduce<br/>写入下一个远程 sendBuff"]
        G --> H["directRecvReduceCopyDirectSend:<br/>同上 + 拷贝到本地 output"]
        H --> I["directRecvCopyDirectSend:<br/>从远程读取, 拷贝到本地, 转发"]
        I --> J["directRecv:<br/>从远程读取到本地"]
    end

    subgraph "缓冲区布局"
        K["buffSize / NCCL_STEPS<br/>每个 step 等分缓冲区"]
        K --> L["Step 0<br/>sliceSize bytes"]
        L --> M["Step 1<br/>sliceSize bytes"]
        M --> N["..."]
        N --> O["Step NCCL_STEPS-1"]
    end
```

## 25. LL 协议原语

```mermaid
flowchart TD
    A["ProtoLL (Low Latency)"] --> B["MaxGroupWidth = 1<br/>数据效率 ~50%"]

    subgraph "数据格式: 16 字节/line"
        C["data1 (4B) | flag1 (4B) | data2 (4B) | flag2 (4B)"]
    end

    B --> D["同步: Flag 匹配"]
    D --> E["readLL: volatile load v4.u32<br/>等待 flag1 == recvFlag && flag2 == recvFlag"]
    E --> F["writeLL: store v4.u32<br/>设置 data + flag"]

    subgraph "Flag 递增机制"
        G["每次完整传输后<br/>recvFlag += 1"]
        G --> H["发送方用对应 flag 写入数据"]
        H --> I["接收方 volatile load 直到 flag 匹配"]
    end

    style A fill:#fbb,stroke:#333
```

## 26. LL128 协议原语

```mermaid
flowchart TD
    A["ProtoLL128"] --> B["数据效率 ~94%<br/>7/8 数据 + 1/8 flag"]

    subgraph "128 字节行格式 (32 threads × 4B)"
        C["T0: data(4B) | T1: data(4B) | ... | T6: data(4B) | T7: flag(4B)"]
        C --> C2["每 8 个线程中 7 个传数据, 1 个传 flag"]
        C2 --> C3["16 个这样的组 × 8B = 128B<br/>数据: 112B, Flag: 16B"]
    end

    B --> D["同步: 128-bit flag 检查"]
    D --> E["load128: 一次加载 128 字节"]
    E --> F["检查最后 16 字节的 flag 是否匹配"]

    subgraph "与 Simple/LL 对比"
        G["Simple: 高吞吐, 高延迟<br/>~100% 数据效率"]
        G --> H["LL128: 中等吞吐, 中等延迟<br/>~94% 数据效率"]
        H --> I["LL: 低吞吐, 最低延迟<br/>~50% 数据效率"]
    end

    style A fill:#bfb,stroke:#333
```

## 27. 内存管理栈机制

```mermaid
flowchart TD
    subgraph "ncclMemoryStack 作用域机制"
        A["comm 加入 Group:<br/>ncclMemoryStackPush(&comm->memScoped)"] --> B["记录当前栈顶位置 savepoint"]
        B --> C["任务分配使用 memScoped<br/>ncclMemoryStackAlloc 用于 TaskColl/WorkList 等"]

        C --> D["Group 完成: ncclGroupCommLeave"]
        D --> E["ncclMemoryStackPop(&comm->memScoped)"]
        E --> F["栈顶回到 savepoint<br/>所有本 Group 的临时分配一次性释放"]
    end

    subgraph "内存池 (Memory Pool)"
        G["memPool_ncclTaskColl<br/>ncclTaskColl 对象池"]
        G --> H["memPool_ncclKernelPlan<br/>ncclKernelPlan 对象池"]
        H --> I["memPool_ncclProxyOp<br/>ncclProxyOp 对象池"]
        I --> J["memPool_ncclTaskP2p<br/>ncclTaskP2p 对象池"]
    end

    subgraph "永久 vs 临时分配"
        K["memPermanent: 通信器生命周期<br/>如 peers[], channels[]"]
        K --> L["memScoped: Group 生命周期<br/>如 ncclDevWorkColl, WorkBatch"]
    end
```

## 28. 错误处理与 Abort 机制

```mermaid
flowchart TD
    A["错误发生"] --> B{"在 Group 中?"}

    B -->|是| C["ncclGroupErrCheck(ret)<br/>设置 ncclGroupError"]
    C --> D["继续累积错误<br/>GroupEnd 时统一处理"]

    B -->|否| E["直接返回错误码"]

    D --> F["ncclGroupEndInternal 检测到错误"]
    F --> G["groupCleanup:<br/>遍历所有 comm 和 asyncJob"]

    subgraph "清理流程"
        G --> H["ncclGroupCommLeave<br/>重置 comm->groupNext"]
        H --> I["清空 planner.planQueue<br/>释放 ncclKernelPlan + ncclProxyOp"]
        I --> J["重置 planner (memset)"]
        J --> K["调用 job->undo (回滚)"]
        K --> L["调用 job->destructor (释放)"]
    end

    subgraph "Abort 传播"
        M["asyncJobLaunch 检测到错误"] --> N["设置 groupAbortFlag"]
        N --> O["遍历所有 job<br/>设置 job->abortFlag = 1<br/>设置 job->abortFlagDev = 1"]
        O --> P["Device Kernel 检查 abortFlagDev<br/>发现为 1 → 提前退出"]
    end

    subgraph "非阻塞 Abort"
        Q["ncclGroupJobAbort(groupJob)"] --> R["ATOMIC_EXCHANGE(joined, true)"]
        R --> S["设置 abortFlag"]
        S --> T["ncclAsyncJobComplete: join 线程"]
        T --> U["refCount-- → 释放"]
    end
```

---

*文档生成时间: 2026-03-30*
*基于 NCCL 源码: /root/source/nccl*

---

## 附录 A: Ring AllReduce offset 计算详解

```mermaid
flowchart TD
    A["ncclCollCbdPart(work, channelId, protoId, elemSize, nullptr, &gridOffset, &channelCount, &chunkCount)"] --> B["gridOffset: 当前 channel 在全局数据中的起始偏移"]
    B --> C["channelCount: 当前 channel 处理的总元素数<br/>(来自 CBD 的 countLo/Mid/Hi)"]
    C --> D["chunkCount: 每个 chunk 的元素数<br/>= channelCount / nRanks (对齐到16)"]

    D --> E["loopCount = nRanks × chunkCount"]
    E --> F["外层循环: elemOffset = 0; < channelCount; += loopCount"]
    F --> G["remCount = channelCount - elemOffset"]
    G --> H["if remCount < loopCount:<br/>  chunkCount = alignUp(remCount/nRanks, 16)"]

    H --> I["Reduce-Scatter: j=0..nRanks-2"]
    I --> J["chunk = (ringIx + nRanks - j) % nRanks<br/>offset = gridOffset + elemOffset + chunk × chunkCount<br/>nelem = min(chunkCount, remCount - chunk × chunkCount)"]

    J --> K["j==0: directSend(offset, nelem)"]
    K --> L["j==1..nRanks-2: directRecvReduceDirectSend(offset, nelem)"]
    L --> M["j==nRanks-1: directRecvReduceCopyDirectSend(offset, nelem, postOp=true)"]

    M --> N["AllGather: j=1..nRanks-1"]
    N --> O["chunk = (ringIx + nRanks - j) % nRanks<br/>offset = gridOffset + elemOffset + chunk × chunkCount"]

    O --> P["j==1..nRanks-2: directRecvCopyDirectSend(offset, nelem)"]
    P --> Q["j==nRanks-1: directRecv(offset, nelem)"]
```

## 附录 B: Proxy 线程架构

```mermaid
flowchart TD
    A["ncclProxyCreate(comm)"] --> B["创建 proxy 线程 (std::thread)"]
    B --> C["proxy 线程主循环: ncclProxyThread"]

    C --> D["epoll_wait: 等待事件"]
    D --> E{"事件类型"}
    E -->|"网络可读"| F["ncclNetRecv: 接收网络数据"]
    E -->|"网络可写"| G["ncclNetSend: 发送网络数据"]
    E -->|"新操作到达"| H["处理 proxy operation"]

    H --> I["proxyOp->func(proxyOp)<br/>执行具体传输操作"]
    I --> J["Ring: 在 rank 间转发数据<br/>Tree: 向上/向下传输"]
    J --> K["完成后更新 device step counter"]

    K --> L["kernel 检测到 step 更新<br/>继续处理下一个 slice"]
```

## 附录 C: Work Fifo 环形缓冲区机制

```mermaid
flowchart TD
    A["workFifoBuf (GPU 内存)"] --> B["大小: workFifoBytes (2的幂次)"]
    B --> C["writePtr = workFifoProduced % workFifoBytes"]

    subgraph "环形缓冲区"
        direction LR
        S0["已消费区域"] --> S1["最新写入<br/>offset=P%F"] --> S2["空闲区域"]
    end

    subgraph "生产者 (Host)"
        P1["uploadWork:<br/>memcpy(fifoBuf + offset, workData, workBytes)"]
        P2["workFifoProduced += workBytes"]
    end

    subgraph "消费者 (Device)"
        C1["kernel 读取 workFifoBufDev[consumed % fifoBytes]"]
        C2["通过 ncclDevWorkBatch.offsetBase 定位 work"]
    end

    subgraph "同步"
        SYN1["Host: workFifoProduced 记录写入量"]
        SYN2["Device: 通过完成事件回调更新 workFifoConsumed"]
        SYN3["防止: produced - consumed > fifoBytes (溢出)"]
    end
```

