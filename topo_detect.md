# NCCL Graph 创建与 Tuning 技术文档

## 目录
1. [概述](#概述)
2. [拓扑检测整体流程](#拓扑检测整体流程)
3. [Graph 创建流程](#graph-创建流程)
4. [通信链路选择策略](#通信链路选择策略)
5. [Tuning 机制](#tuning-机制)
6. [关键代码实现分析](#关键代码实现分析)

---

## 概述

NCCL (NVIDIA Collective Communications Library) 的 Graph 系统负责：
1. **拓扑检测**：自动发现 GPU、NIC、CPU、PCIe 交换机、NVLink 等硬件连接关系
2. **路径计算**：计算任意两个设备间的最优通信路径
3. **Graph 搜索**：基于拓扑寻找最优的 Ring/Tree/NVLS 通信模式
4. **Tuning**：根据拓扑特征选择最佳算法、协议和线程配置

---

## 拓扑检测整体流程

### 1. 拓扑数据结构设计

```c
// 核心数据结构定义于 src/graph/topo.h

// 节点类型（7种）
#define GPU 0    // GPU 设备
#define PCI 1    // PCI/PCIe 设备
#define NVS 2    // NVSwitch
#define CPU 3    // CPU/NUMA 节点
#define NIC 4    // 网卡设备
#define NET 5    // 网络端点（逻辑NIC）
#define GIN 6    // Generic Interface Network

// 路径类型（12种，按优先级排序）
#define PATH_LOC 0   // 本地（自己）
#define PATH_NVL 1   // NVLink 直连
#define PATH_NVB 2   // NVLink 通过中间GPU（1跳）
#define PATH_C2C 3   // Chip-to-Chip (Grace Hopper)
#define PATH_PIX 4   // 单个PCIe交换机
#define PATH_PXB 5   // 多个PCIe交换机
#define PATH_P2C 6   // GPU-NIC通过C2C+PCIe
#define PATH_PXN 7   // GPU-NIC通过中间GPU
#define PATH_PHB 8   // 经过PCIe Host Bridge
#define PATH_SYS 9   // 经过系统互联（QPI/UPI）
#define PATH_NET 10  // 通过网络
#define PATH_DIS 11  // 断开
```

### 2. 拓扑发现流程

拓扑发现由 `ncclTopoGetSystem()` 函数发起（位于 `src/graph/topo.cc`）：

```
ncclTopoGetSystem()
    ├── 1. 加载 XML 拓扑文件（如果存在 NCCL_TOPO_FILE）
    ├── 2. 自动检测本地 GPU 拓扑
    │      └── ncclTopoFillGpu() - 通过 NVML 获取 GPU 信息
    ├── 3. 检测网络设备（NIC）
    │      ├── GIN 设备检测
    │      ├── CollNet 检测
    │      └── 普通 Net 检测
    │          └── ncclTopoProcessNet()
    │              ├── 物理设备枚举
    │              ├── NIC 融合（vNic 创建）
    │              └── XML 节点填充
    ├── 4. XML 拓扑融合（多节点/MNNVL）
    │      └── bootstrapIntraNodeAllGather() - 收集所有 rank 的拓扑
    ├── 5. 构建拓扑系统
    │      └── ncclTopoGetSystemFromXml()
    │          ├── ncclTopoAddCpu() - 添加 CPU 节点
    │          ├── ncclTopoAddNvLinks() - 添加 NVLink 连接
    │          ├── ncclTopoAddC2c() - 添加 C2C 连接
    │          ├── ncclTopoFlattenBcmSwitches() - 展平 BCM 交换机
    │          └── ncclTopoConnectCpus() - 连接 CPU 节点
    └── 6. 计算路径
           └── ncclTopoComputePaths()
```

### 3. 关键拓扑检测函数

#### 3.1 GPU 信息填充 (`ncclTopoFillGpu`)

```c
// 从 NVML 获取 GPU 信息
ncclResult_t ncclTopoFillGpu(ncclXml* xml, const char* busId, ncclXmlNode** gpuNode) {
    // 1. 获取 GPU 计算能力 (sm)
    // 2. 获取 NVLink 信息（通过 nvmlDeviceGetNvLinkState）
    // 3. 获取 PCIe 信息（链路宽度、速度）
    // 4. 创建 XML 节点
}
```

#### 3.2 NVLink 检测 (`ncclTopoAddNvLinks`)

```c
ncclResult_t ncclTopoAddNvLinks(ncclXmlNode* node, ncclTopoSystem* system, ...) {
    // 遍历 XML 中的 nvlink 节点
    // 根据目标类型区分：
    // - GPU: NVLink P2P 连接
    // - CPU: NVLink 到 CPU（IBM Power）
    // - NVS: NVSwitch 连接
    
    // NVLink 带宽计算：
    float nvlBw = ncclTopoNVLinkBw(gpu->gpu.cudaCompCap);
    // SM60: 18 GB/s
    // SM70: 20 GB/s
    // SM80: 20 GB/s
    // SM90: 20.6 GB/s
    // SM100: 40.1 GB/s
}
```

#### 3.3 NIC 融合 (`ncclTopoMakeVNics`)

多端口 NIC 融合为虚拟 NIC，提高带宽利用率：

```c
ncclResult_t ncclTopoMakeVNics(ncclXml* xml, ncclTopoNetInfo* netInfo, int physicalDevs) {
    // 1. 计算 NIC 间路径类型（PATH_LOC/PIX/PXB/PHB/SYS）
    // 2. 根据 mergeLevel 策略自动分组
    // 3. 创建虚拟设备（makeVDevice）
    // 4. 更新 XML 拓扑
}
```

---

## Graph 创建流程

### 1. Graph 结构定义

```c
// src/include/graph.h
struct ncclTopoGraph {
    int id;                    // 0=Ring, 1=Tree, 2=CollNet, 3=NVLS, 4=CollNetDirect
    int pattern;               // 通信模式（见下文）
    int crossNic;              // 是否跨 NIC 通信
    int collNet;               // 是否使用 CollNet
    int minChannels;           // 最小通道数
    int maxChannels;           // 最大通道数
    
    // 输出参数
    int nChannels;             // 实际通道数
    float bwIntra;             // 节点内带宽
    float bwInter;             // 节点间带宽
    float latencyInter;        // 节点间延迟
    int typeIntra;             // 节点内路径类型
    int typeInter;             // 节点间路径类型
    int sameChannels;          // 是否使用相同通道
    int nHops;                 // 跳数
    int intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES];  // 节点内 rank 顺序
    int64_t inter[MAXCHANNELS*2];                 // 节点间 NIC 配对
};
```

### 2. 通信模式 (Pattern)

```c
#define NCCL_TOPO_PATTERN_BALANCED_TREE 1   // NIC 流量分散到两个 GPU
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2      // 分离树模式
#define NCCL_TOPO_PATTERN_TREE 3            // 标准树（所有 NIC 流量到同一 GPU）
#define NCCL_TOPO_PATTERN_RING 4            // 环形
#define NCCL_TOPO_PATTERN_NVLS 5            // NVLink Switch (NVLS+SHARP/Tree)
#define NCCL_TOPO_PATTERN_COLLNET_DIRECT 6  // CollNet Direct 模式
```

### 3. Graph 搜索流程

Graph 搜索由 `ncclTopoCompute()` 函数执行：

```
ncclTopoCompute()
    ├── 1. 初始化搜索参数
    │      ├── 确定 crossNic 策略
    │      ├── 计算 min/max 路径类型
    │      └── 选择速度数组（基于计算能力）
    ├── 2. 多轮搜索策略
    │      └── 搜索循环 (search label)
    │          ├── Pass 1: 寻找可行解
    │          │      ├── 尝试 sameChannels
    │          │      ├── 尝试 simpler tree
    │          │      ├── 放宽 typeIntra
    │          │      ├── 放宽 typeInter
    │          │      ├── 尝试 crossNic
    │          │      └── 降低 bw 要求
    │          └── Pass 2: 优化解
    │                  ├── 通道复制（DupChannels）
    │                  └── 增加 bw
    ├── 3. 搜索结果评估
    │      └── ncclTopoCompareGraphs() - 比较图的质量
    └── 4. 输出最优 Graph
```

### 4. 递归搜索算法 (`ncclTopoSearchRec`)

```c
ncclResult_t ncclTopoSearchRec(ncclTopoSystem* system, 
                               ncclTopoGraph* graph, 
                               ncclTopoGraph* saveGraph, 
                               int* time) {
    // 1. 确定搜索参数
    int backToNet, backToFirstRank;
    ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank);
    
    // 2. 根据是否有网络设备选择搜索起点
    if (system->inter) {
        // 从 NIC 开始搜索（节点间通信）
        ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time);
    } else {
        // 从 GPU 开始搜索（纯节点内通信）
        // 尝试 NVLS、PCI 顺序、重放等多种策略
    }
}
```

### 5. GPU 选择评分机制

搜索过程中，NCCL 使用评分机制选择下一个 GPU：

```c
struct ncclGpuScore {
    int g;              // GPU 索引
    int startIndex;     // 起始索引（最不重要）
    int intraNhops;     // 节点内跳数
    int intraBw;        // 节点内带宽
    int interNhops;     // 节点间跳数
    int interPciBw;     // PCIe 带宽
    int interBw;        // 节点间带宽（最重要）
};

// 评分比较函数（优先级从高到低）
static int cmpScore(const void* g1, const void* g2) {
    // 1. 更高 interBw 优先
    // 2. 更高 interPciBw 优先
    // 3. 更少 interNhops 优先
    // 4. 更高 intraBw 优先
    // 5. 更少 intraNhops 优先
    // 6. 更小 startIndex 优先
}
```

---

## 通信链路选择策略

### 1. 路径计算算法

路径计算采用 **BFS (广度优先搜索)**：

```c
ncclResult_t ncclTopoSetPaths(ncclTopoNode* baseNode, ncclTopoSystem* system) {
    // 1. 初始化起点路径
    basePath->count = 0;
    basePath->bw = LOC_BW;  // 5000 GB/s (本地带宽)
    basePath->type = PATH_LOC;
    
    // 2. BFS 遍历所有节点
    while (nodeList.count) {
        for (每个节点) {
            for (每个链路) {
                // 计算带宽 = min(当前路径带宽, 链路带宽)
                bw = min(path->bw, link->bw);
                
                // 确定路径类型
                int type = link->type;
                if (node->type == PCI && remNode->type == PCI) type = PATH_PXB;
                if (经过 CPU) type = PATH_PHB;
                if (NVLink 1跳) type = PATH_NVB;
                
                // 更新路径如果更优
                if (remPath->bw < bw || (remPath->bw == bw && remPath->count > path->count + 1)) {
                    // 更新路径
                }
            }
        }
    }
}
```

### 2. P2P 路径选择 (`ncclTopoCheckP2p`)

```c
ncclResult_t ncclTopoCheckP2p(ncclComm* comm, ncclTopoSystem* system, 
                              int rank1, int rank2,
                              int* p2p, int* read, int* intermediateRank, int* cudaP2p) {
    // 1. 检查是否同节点
    if (不同节点) {
        // 检查 MNNVL 支持
        ncclTopoCheckMNNVL(system, info1, info2, &mnnvl);
        if (!mnnvl) return;  // 不支持 P2P
    }
    
    // 2. 确定 P2P 级别阈值
    int p2pLevel = PATH_PXB;  // 默认不超过 PCIe 交换机
    
    // AMD CPU 且 GPU <= 2 时允许 SYS 级别 P2P
    if (AMD_CPU && gpu_count <= 2) p2pLevel = PATH_SYS;
    
    // 用户覆盖（NCCL_P2P_LEVEL）
    ncclGetUserP2pLevel(&p2pLevel);
    
    // 3. 比较路径类型与阈值
    if (path->type <= p2pLevel) *p2p = 1;
    
    // 4. 检查 NVML P2P 状态
    if (checkNvml) {
        // 验证 NVML P2P 是否可用
        // 如果 P2P 被禁用但路径是 NVLink，发出警告
    }
}
```

### 3. GDR (GPU Direct RDMA) 检查 (`ncclTopoCheckGdr`)

```c
ncclResult_t ncclTopoCheckGdr(ncclTopoSystem* system, int rank, int64_t netId, 
                              int read, ncclTopoGdrMode* gdrMode) {
    // 1. 基础检查
    if (!net->net.gdrSupport || !gpu->gpu.gdrSupport) return;
    
    // 2. 读取模式额外检查
    if (read) {
        // 禁用 Pre-Ampere 的 GDR Read（当存在其他 PCI 流量时）
        if (gdrReadParam < 0 && gpu->gpu.cudaCompCap < 80) {
            // 检查是否存在 NVLink
            if (!nvlink) return;  // 禁用 GDR Read
        }
    }
    
    // 3. 距离检查
    int netGdrLevel = ncclParamNetGdrC2c() ? PATH_P2C : PATH_PXB;
    
    // PXN 情况：使用中间 GPU 的距离
    if (path->type == PATH_PXN) {
        ncclTopoGetIntermediateRank(system, rank, netId, &proxyRank);
        // 使用 proxy GPU 重新计算距离
    }
    
    // 4. 模式选择
    if (distance <= netGdrLevel) {
        if (C2C 系统 && distance != PATH_P2C) *gdrMode = ncclTopoGdrModePci;
        else *gdrMode = ncclTopoGdrModeDefault;
    }
}
```

### 4. NIC 选择策略 (`ncclTopoSelectNets`)

```c
ncclResult_t ncclTopoSelectNets(ncclTopoSystem* system, int typeInter, int gpu,
                                int nets[], int* netCountRet) {
    // 1. 首先添加首选 NIC（基于 NETDEVS_POLICY）
    if (MNNVL 系统 && scatterEnable) {
        // GPU 优先排序
        ncclTopoPrefNetsGpuFirst(system, gpu, nets, &netCount);
    } else {
        // 通道优先排序
        ncclTopoPrefNetsChannelFirst(system, gpu, nets, &netCount);
    }
    
    // 2. 根据策略限制 NIC 数量
    // - AUTO: 自动计算（netsPerGpu = DIVUP(localNetCount, localGpuCount)）
    // - ALL: 使用所有 NIC
    // - MAX:n: 最多使用 n 个 NIC
    
    // 3. 添加满足 typeInter 要求的其他 NIC
    for (int t = 0; t <= typeInter; t++) {
        // 添加路径类型为 t 的 NIC
    }
}
```

### 5. PXN (Proxy Xfer Network) 策略

PXN 允许 GPU 通过 NVLink 连接的相邻 GPU 访问 NIC：

```c
// 在 ncclTopoComputePaths 中
if (ncclPxnDisable(comm) != 1) {
    int localGpuIndex;
    ncclTopoGetLocalGpu(system, netNode->id, &localGpuIndex);
    
    if (localGpuIndex != g && localGpuIndex != -1) {
        // 检查 PXN 条件：
        // 1. peer GPU 到 NIC 的路径类型 <= PATH_PXB (或 PATH_P2C 如果启用 C2C)
        // 2. peer GPU 到当前 GPU 的路径类型 <= PATH_NVL
        // 3. 同节点
        // 4. peer 到 NIC 的带宽更高，或当前 GPU 到 NIC 路径类型 > PATH_PXN
        
        if (条件满足) {
            // 添加 PXN 路径：GPU -> peer GPU -> NIC
            addInterStep(system, GPU, localGpuIndex, GPU, g, NET, n);
        }
    }
}
```

---

## Tuning 机制

### 1. Tuner 常量定义

```c
// src/graph/tuning.cc
static const ncclTunerConstants_t ncclTunerConstantsDefaults = {
    // 基础延迟 (μs)
    .baseLatencies = {
        { 6.8, 14.0, 8.4 },   // Tree: LL/LL128/Simple
        { 6.6, 14.0, 8.4 },   // Ring: LL/LL128/Simple
        // ...
    },
    
    // 硬件延迟 (μs) - 按路径类型
    .hwLatencies = {
        // NVLink 路径
        { { 0.6, 1.25, 4.0 },   // Tree
          { 0.6, 1.9, 3.4 },    // Ring
          // ...
        },
        // PCI 路径
        { { 1.0, 1.9, 4.0 },    // Tree
          { 1.0, 2.5, 5.7 },    // Ring
          // ...
        },
        // NET 路径
        { { 5.0, 8.5, 14.0 },   // Tree
          { 2.7, 4.0, 14.0 },   // Ring
          // ...
        },
    },
    
    // LL 协议最大带宽 (GB/s)
    .llMaxBws = {
        { 39.0, 39.0, 20.4 },   // Volta
        { 87.7, 22.5, 19.0 },   // Ampere
        { 141.0, 45.0, 35.0 },  // Hopper
        { 2*141.0, 2*45.0, 2*35.0 },  // Blackwell
    },
    
    // 各代 GPU 的 Tree/Ring 带宽限制
    .perChMaxTreeBws = { ... },
    .perChMaxRingLL128Bws = { ... },
};
```

### 2. 线程数配置

```c
ncclResult_t ncclTopoTuneModel(ncclComm* comm, int minCompCap, int maxCompCap, 
                               ncclTopoGraph** graphs) {
    // Ring Simple 协议线程数
    int simpleDefaultThreads = (graphs[NCCL_ALGO_RING]->bwIntra * graphs[NCCL_ALGO_RING]->nChannels <= PCI_BW) 
                               ? 256 : NCCL_SIMPLE_MAX_NTHREADS;
    comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 
        getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 
                    2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
    
    // Tree Simple 协议线程数
    comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 
        getNthreads("NCCL_NTHREADS", ncclParamNthreads(),
                    2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
    
    // NVLS/CollNet 使用最大线程数
    comm->maxThreads[NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] = NCCL_MAX_NTHREADS;
}
```

### 3. 带宽模型计算

```c
ncclResult_t ncclTopoTuneModel(...) {
    for (每个集合操作 coll) {
        for (每个算法 algo) {
            for (每个协议 proto) {
                // 基础带宽
                float bw = (nNodes <= 2 || collnet) ? graphs[a]->bwIntra : graphs[a]->bwInter;
                
                // NVLS 带宽计算
                if (algo == NCCL_ALGO_NVLS_TREE || algo == NCCL_ALGO_NVLS) {
                    // 转换为 NVLS busBW/channel
                    float intraBw = graphs[a]->bwIntra * nvlsEfficiency[compCapIndex] 
                                    * (graphs[a]->nChannels - 1) / graphs[a]->nChannels;
                    if (coll == ncclFuncAllReduce) intraBw *= 2.0f;
                    
                    bw = min({intraBw, interBw, perChMaxNVLSTreeBw});
                }
                
                // 各种协议调整
                if (algo == NCCL_ALGO_RING && proto == NCCL_PROTO_LL) {
                    busBw = min(llMaxBw, busBw * 0.5);  // LL 只有 50% 效率
                }
                if (algo == NCCL_ALGO_RING && proto == NCCL_PROTO_LL128) {
                    busBw = min(busBw * (120.0/128.0), nChannels * perChMaxRingLL128Bw);
                }
                if (algo == NCCL_ALGO_TREE) {
                    busBw = min(busBw * 0.92, nChannels * perChMaxTreeBw);
                }
                
                // 转换为算法带宽
                float ratio = (algo == NCCL_ALGO_RING) ? (1.0 * nRanks) / nsteps : 0.5;
                comm->bandwidths[coll][algo][proto] = busBw * ratio;
                
                // 延迟计算
                computeLatency(...);
            }
        }
    }
}
```

### 4. 算法时间估计

```c
ncclResult_t ncclTopoGetAlgoTime(ncclComm* comm, int coll, int algorithm, 
                                 int protocol, size_t nBytes, int numPipeOps, 
                                 float* time) {
    float bw = comm->bandwidths[coll][algorithm][protocol];
    float lat = comm->latencies[coll][algorithm][protocol];
    
    if (bw == 0) { *time = -1.0; return; }  // 禁用
    
    // Tree 静态修正因子（中尺寸优化）
    int logSize = log2i(nBytes >> 6);
    if (algorithm == NCCL_ALGO_TREE && coll == ncclFuncAllReduce) {
        bw *= treeCorrectionFactor[protocol][logSize];
    }
    
    // Ring 平台效应修正
    if (algorithm == NCCL_ALGO_RING && protocol == NCCL_PROTO_SIMPLE 
        && comm->nNodes > 1 && coll == ncclFuncAllReduce 
        && nBytes/(comm->nChannels*comm->nRanks) >= 64) {
        lat *= (comm->minCompCap < 80) ? 1.9 : 1.4;
    }
    
    // 计算总时间
    int latCount = (algorithm == NCCL_ALGO_RING) 
                   ? numPipeOps 
                   : DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS);
    *time = lat * latCount + nBytes / (1000 * bw);
}
```

### 5. 协议/算法启用控制

```c
// 环境变量控制
NCCL_PROTO="LL,Simple;allreduce:^LL"      // AllReduce 禁用 LL
NCCL_ALGO="ring,collnetdirect;allreduce:tree,collnetdirect"

// LL128 自动启用条件（pEnable == 2 时）
if (pEnable == 2 && proto == NCCL_ALGO_LL128) {
    pEnable = 1;
    
    // Hopper/Blackwell 默认启用条件
    if (ncclParamLl128C2c() && minCompCap >= 90) {
        // 启用条件：
        // 1. typeInter <= PATH_PXN (包含 P2C)
        pEnable &= (graphs[a]->typeInter <= PATH_PXN);
    } else {
        // 其他架构：
        // 1. typeInter <= PATH_PXB
        pEnable &= (graphs[a]->typeInter <= PATH_PXB);
    }
    
    // 2. typeIntra <= PATH_NVB
    pEnable &= (graphs[a]->typeIntra <= PATH_NVB);
    
    // 3. 同构 GPU（minCompCap == maxCompCap）
    pEnable &= (minCompCap == maxCompCap);
    
    // 4. 排除特定情况（Volta、CUDA 11.8 特定配置等）
}
```

---

## 关键代码实现分析

### 1. 路径带宽追踪机制

NCCL 在搜索过程中动态追踪路径带宽：

```c
// followPath - 检查并消耗路径带宽
static ncclResult_t followPath(ncclTopoLinkList* path, ncclTopoNode* start, 
                               int maxSteps, float bw, int* steps) {
    float pciBw = bw;
    
    // Intel CPU P2P 开销调整
    for (int step = 0; step < path->count; step++) {
        if (node->type == CPU && path->type == PATH_PHB 
            && node->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 
            && node->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
            pciBw = INTEL_P2P_OVERHEAD(bw);  // bw * 6/5
        }
    }
    
    // 检查每跳的带宽
    for (int step = 0; step < maxSteps; step++) {
        struct ncclTopoLink* link = path->list[step];
        float fwBw = (link->type == LINK_PCI) ? pciBw : bw;
        
        // 特殊处理：Pre-Ampere GPU 的反向带宽限制
        if (link->remNode->type == GPU && link->remNode->gpu.cudaCompCap < 80) {
            revBw += fwBw / 8;  // 反向带宽限制
        }
        
        // 检查带宽是否足够
        if (link->bw < fwBw || (revBw && revLink->bw < revBw)) {
            *steps = step;  // 带宽不足，返回实际步数
            return ncclSuccess;
        }
        
        // 消耗带宽
        SUB_ROUND(link->bw, fwBw);
        if (revBw) SUB_ROUND(revLink->bw, revBw);
    }
    
    *steps = maxSteps;
    return ncclSuccess;
}
```

### 2. NVSwitch 优化

```c
// 在 ncclTopoSearchNextGpuSort 中
if (system->nodes[NVS].count) {
    // NVSwitch 偏好与有限的对等方通信
    // 优先尝试邻居 GPU
    int prevGpu = (index - 1 + ngpus) % ngpus;
    int nextGpu = (index + 1) % ngpus;
    
    if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
        firstGpus[0] = nextGpu; firstGpus[1] = prevGpu;
    } else if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ||
               graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
        firstGpus[0] = prevGpu; firstGpus[1] = nextGpu;
    }
    
    // 将邻居 GPU 排到搜索列表最前面
    for (...) {
        for (i = 0; i < count && next[i] != firstGpus[g]; i++);
        if (i < count) {
            // 移动到列表头部
            for (; i > 0; i--) next[i] = next[i-1];
            next[0] = firstGpus[g];
        }
    }
}
```

### 3. 通道复制策略

```c
ncclResult_t ncclTopoDupChannels(ncclTopoGraph* graph, int ccMin, int ngpus) {
    // 复制条件：
    // 1. 已有通道
    // 2. 非 NVLS 模式
    // 3. 节点内带宽 >= 25 GB/s
    // 4. 不是 Ampere+ 且 bwIntra < 50 且 nChannels > 4
    if (graph->nChannels == 0) return ncclSuccess;
    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) return ncclSuccess;
    if (graph->bwIntra < 25.0) return ncclSuccess;
    if (ccMin > 80 && graph->bwIntra < 50.0 && graph->nChannels > 4) return ncclSuccess;
    
    // 复制通道（最多翻倍）
    int dupChannels = min(graph->nChannels * 2, graph->maxChannels);
    memcpy(graph->intra + graph->nChannels * ngpus, graph->intra, 
           (dupChannels - graph->nChannels) * ngpus * sizeof(int));
    
    // 带宽均摊
    graph->bwIntra /= DIVUP(dupChannels, graph->nChannels);
    graph->bwInter /= DIVUP(dupChannels, graph->nChannels);
    graph->nChannels = dupChannels;
}
```

### 4. 树结构连接 (`connectTrees`)

```c
static ncclResult_t connectTrees(ncclComm* comm, int* treeToParent, 
                                 int* treeToChild0, int* treeToChild1, 
                                 int* treePatterns) {
    // 计算树深度近似值
    int depth = comm->nRanks / nNodes - 1 + log2i(nNodes);
    
    // 获取双树结构（类似蝴蝶网络）
    int t0u, t0d0, t0d1, t0ChildType;
    int t1u, t1d1, t1d1, t1ChildType;
    ncclGetDtree(nNodes, node, &t0u, &t0d0, &t0d1, &t0ChildType,
                 &t1u, &t1d0, &t1d1, &t1ChildType);
    
    // 为每个通道设置树连接
    for (int c = 0; c < nChannels; c++) {
        struct ncclChannel* channel0 = comm->channels + c;
        struct ncclChannel* channel1 = channel0 + nChannels;  // 双树
        
        // 根据当前 rank 在树中的位置设置 up/down
        if (comm->rank == ttp[node]) {  // 当前节点是 parent
            setTreeUp(&channel0->tree, t0ChildType == 0 ? ttc0 : ttc1, t0u);
            setTreeUp(&channel1->tree, t1ChildType == 0 ? ttc0 : ttc1, t1u);
        }
        if (comm->rank == ttc0[node]) {  // child0
            setTreeDown(&channel0->tree, ttp, t0d0);
            setTreeDown(&channel1->tree, ttp, t1d0);
        }
        if (comm->rank == ttc1[node]) {  // child1
            setTreeDown(&channel0->tree, ttp, t0d1);
            setTreeDown(&channel1->tree, ttp, t1d1);
        }
        
        channel0->tree.depth = channel1->tree.depth = depth;
    }
}
```

### 5. 速度数组定义

NCCL 针对不同架构使用预定义的速度数组：

```c
// 通用速度数组
float speedArrayIntra[] = { 40.0, 30.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 
                            7.0, 6.0, 5.0, 4.0, 3.0 };
float speedArrayInter[] = { 48.0, 30.0, 28.0, 24.0, 20.0, 18.0, 15.0, 12.0, 
                            10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.4, 1.2, 
                            0.24, 0.12 };

// Hopper (SM90) 专用
float sm90SpeedArrayIntra[] = { 60.0, 50.0, 40.0, 30.0, 24.0, 20.0, 15.0, 
                                12.0, 11.0, 6.0, 3.0 };
float sm90SpeedArrayInter[] = { 48.0, 45.0, 42.0, 40.0, 30.0, 24.0, 22.0, 
                                20.0, 17.5, 15.0, 12.0, 6.0, 3.0, 2.4, 1.2, 
                                0.24, 0.12 };

// Blackwell (SM100) 专用
float sm100SpeedArrayIntra[] = { 90.0, 80.0, 70.0, 60.0, 50.0, 45.0, 40.0, 
                                 30.0, 24.0, 20.0, 19.0, 18.0 };
float sm100SpeedArrayInter[] = { 96.0, 80.0, 48.0, 45.1, 42.0, 40.0, 30.0, 
                                 24.0, 22.0, 20.0, 17.5, 15.0, 12.0, 6.0, 3.0, 
                                 2.4, 1.2, 0.24, 0.12 };
```

---

## 环境变量总结

| 环境变量 | 功能 | 示例值 |
|---------|------|--------|
| `NCCL_TOPO_FILE` | 指定拓扑 XML 文件 | `/path/to/topo.xml` |
| `NCCL_GRAPH_FILE` | 指定 Graph XML 文件 | `/path/to/graph.xml` |
| `NCCL_GRAPH_DUMP_FILE` | 导出检测到的 Graph | `/path/to/dump.xml` |
| `NCCL_P2P_LEVEL` | 设置 P2P 最大路径级别 | `NVL`, `PIX`, `PXB`, `PHB`, `SYS` |
| `NCCL_P2P_DISABLE` | 禁用 P2P | `1` |
| `NCCL_NET_GDR_LEVEL` | 设置 GDR 启用级别 | `PIX`, `PXB`, `PHB` |
| `NCCL_NET_GDR_READ` | 启用 GDR Read | `1` |
| `NCCL_CROSS_NIC` | 允许跨 NIC 通信 | `0`, `1`, `2` |
| `NCCL_NETDEVS_POLICY` | NIC 选择策略 | `AUTO`, `ALL`, `MAX:4` |
| `NCCL_NTHREADS` | 设置线程数 | `256`, `512` |
| `NCCL_LL128_NTHREADS` | 设置 LL128 线程数 | `256`, `512`, `640` |
| `NCCL_PROTO` | 协议选择 | `LL,Simple`, `^LL128` |
| `NCCL_ALGO` | 算法选择 | `ring,tree`, `collnetdirect` |
| `NCCL_MIN_NCHANNELS` | 最小通道数 | `4`, `8` |
| `NCCL_MAX_NCHANNELS` | 最大通道数 | `16`, `32` |
| `NCCL_MNNVL_SCATTER_NETS_ENABLE` | MNNVL 分散 NIC 启用 | `1`, `0` |
| `NCCL_PXN_DISABLE` | 禁用 PXN | `1` |
| `NCCL_P2P_PXN_LEVEL` | PXN 使用级别 | `0`, `1`, `2` |

---

## 总结

NCCL 的 Graph 创建和 Tuning 系统是一个复杂的多阶段优化过程：

1. **拓扑检测阶段**：通过 NVML、sysfs、网络插件等自动发现硬件拓扑，构建 XML 表示
2. **路径计算阶段**：使用 BFS 算法计算所有设备间的最优路径，考虑带宽、延迟和路径类型
3. **Graph 搜索阶段**：采用启发式搜索算法寻找最优的 Ring/Tree/NVLS 拓扑，支持多轮优化
4. **Tuning 阶段**：基于拓扑特征选择最佳算法、协议和线程配置，使用性能模型预测执行时间

整个系统高度可配置，通过环境变量可以精细控制各种优化策略，以适应不同的硬件配置和应用场景。
