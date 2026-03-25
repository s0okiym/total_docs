# NCCL 通信拓扑检测与链路选择机制分析

## 目录
1. [概述](#概述)
2. [拓扑结构数据模型](#拓扑结构数据模型)
3. [拓扑检测流程](#拓扑检测流程)
4. [通信路径类型](#通信路径类型)
5. [链路选择机制](#链路选择机制)
6. [Fallback 机制](#fallback-机制)
7. [关键配置参数](#关键配置参数)

---

## 概述

NCCL (NVIDIA Collective Communications Library) 的通信拓扑检测是其核心功能之一，负责：
- 发现和构建系统硬件拓扑
- 计算设备间的最优通信路径
- 选择合适的通信链路类型
- 在无法直接通信时提供回退机制

核心代码位于：
- `src/graph/topo.cc` - 拓扑构建和管理
- `src/graph/topo.h` - 拓扑数据结构定义
- `src/graph/paths.cc` - 路径计算
- `src/graph/search.cc` - 图搜索算法

---

## 拓扑结构数据模型

### 节点类型 (Node Types)

NCCL 定义了 7 种拓扑节点类型：

```c
#define NCCL_TOPO_NODE_TYPES 7
#define GPU 0    // GPU 设备
#define PCI 1    // PCI/PCIe 交换机
#define NVS 2    // NVLink 交换机
#define CPU 3    // NUMA 域 (实际上是 CPU 域)
#define NIC 4    // 网络接口卡
#define NET 5    // 网络设备 (逻辑概念)
#define GIN 6    // GIN 设备 (用于特定网络)
```

### 核心数据结构

```c
// 拓扑节点
struct ncclTopoNode {
  int type;                          // 节点类型
  int64_t id;                        // 节点 ID
  int nlinks;                        // 链接数量
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];  // 链接列表
  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES]; // 到各类型节点的路径
  // 类型特定数据
  union {
    struct { int dev; int rank; int cudaCompCap; int gdrSupport; } gpu;
    struct { int dev; uint64_t pciId; float bw; int gdrSupport; ... } net;
    struct { int arch; int vendor; int model; ncclAffinity affinity; } cpu;
    struct { uint64_t device; } pci;
  };
};

// 拓扑系统
struct ncclTopoSystem {
  int systemId;
  uint64_t hostHashes[NCCL_TOPO_MAX_NODES];
  int nHosts;
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];
  float maxBw;     // 最大带宽
  float totalBw;   // 总带宽
  int inter;       // 是否跨节点
};
```

---

## 拓扑检测流程

### 1. 入口函数：`ncclTopoGetSystem`

```c
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm,
                                struct ncclTopoSystem** system,
                                const char* dumpXmlFile)
```

**流程步骤：**

```
┌─────────────────────────────────────────────────────────────┐
│                    ncclTopoGetSystem                         │
├─────────────────────────────────────────────────────────────┤
│ 1. 加载 XML 拓扑文件 (如果存在)                               │
│    - NCCL_TOPO_FILE 环境变量指定                             │
│    - /var/run/nvidia-topologyd/virtualTopology.xml          │
├─────────────────────────────────────────────────────────────┤
│ 2. 检测本地 GPU                                              │
│    - ncclTopoFillGpu(): 填充 GPU 信息                        │
├─────────────────────────────────────────────────────────────┤
│ 3. 导入网络插件                                              │
│    - GIN (GDR Inter-Network)                                │
│    - CollNet (集合网络)                                      │
│    - NET (普通网络)                                          │
├─────────────────────────────────────────────────────────────┤
│ 4. XML 拓扑融合                                              │
│    - 单节点: bootstrapIntraNodeAllGather                    │
│    - MNNVL: 跨节点融合                                       │
├─────────────────────────────────────────────────────────────┤
│ 5. 从 XML 构建拓扑系统                                        │
│    - ncclTopoGetSystemFromXml()                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. XML 解析与拓扑构建

```c
ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml,
                                       struct ncclTopoSystem** topoSystem,
                                       uint64_t localHostHash)
```

**构建顺序：**
1. **添加 CPU 节点** - `ncclTopoAddCpu()`
   - 解析 NUMA 域信息
   - 设置 CPU 架构、厂商、型号
   - 递归添加 PCI 设备

2. **添加 PCI 设备** - `ncclTopoAddPci()`
   - 识别设备类型 (GPU/NIC/PCI Switch)
   - 计算链路带宽
   - 建立父子连接

3. **添加 NVLink 连接** - `ncclTopoAddNvLinks()`
   - 解析 NVLink 连接信息
   - 设置 NVLink 带宽

4. **添加 C2C 连接** - `ncclTopoAddC2c()`
   - 处理 CPU-GPU 直连 (Grace-Hopper)

5. **连接 CPU 节点** - `ncclTopoConnectCpus()`
   - 设置 NUMA 节点间的互连带宽

### 3. 路径计算

```c
ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system,
                                   struct ncclComm* comm)
```

**计算步骤：**
```c
// 1. 移除旧路径
ncclTopoRemovePaths(system);

// 2. 设置到各类型节点的路径 (BFS 搜索)
for (int c=0; c<system->nodes[CPU].count; c++)
  ncclTopoSetPaths(system->nodes[CPU].nodes+c, system);
for (int g=0; g<system->nodes[GPU].count; g++)
  ncclTopoSetPaths(system->nodes[GPU].nodes+g, system);
for (int n=0; n<system->nodes[NET].count; n++)
  ncclTopoSetPaths(system->nodes[NET].nodes+n, system);

// 3. 处理 P2P 不通的路径 (绕行 CPU)
for (int g=0; g<system->nodes[GPU].count; g++) {
  for (int p=0; p<system->nodes[GPU].count; p++) {
    if (p2p == 0) {
      // 通过 CPU 中转
      addInterStep(system, CPU, cpu, GPU, p, GPU, g);
    }
  }
}

// 4. 处理 PXN (PCI + NVLink) 路径
for (int n=0; n<system->nodes[NET].count; n++) {
  // 使用 NVLink 连接的 GPU 作为中转访问 NIC
}
```

---

## 通信路径类型

### 路径类型定义

```c
#define PATH_LOC  0   // 本地 (同一设备)
#define PATH_NVL  1   // 直连 NVLink
#define PATH_NVB  2   // 通过中间 GPU 的 NVLink
#define PATH_C2C  3   // CPU-GPU 直连 (Grace-Hopper)
#define PATH_PIX  4   // 同一 PCIe 交换机
#define PATH_PXB  5   // 多级 PCIe 交换机 (不经过 CPU)
#define PATH_P2C  6   // GPU 通过 C2C 到 CPU 再到 NIC
#define PATH_PXN  7   // 通过中间 GPU 访问 NIC
#define PATH_PHB  8   // 经过 PCIe Host Bridge (CPU)
#define PATH_SYS  9   // 跨 NUMA 域 (经过 QPI/UPI)
#define PATH_NET  10  // 通过网络
#define PATH_DIS  11  // 断开
```

### 路径类型优先级

```
PATH_LOC  > PATH_NVL > PATH_NVB > PATH_C2C > PATH_PIX > PATH_PXB > PATH_P2C > PATH_PXN > PATH_PHB > PATH_SYS > PATH_NET
   ↑                                                                                                                ↓
  最优                                                                                                            最差
```

### 路径选择示意图

```
┌──────────────────────────────────────────────────────────────────┐
│                        GPU 通信路径选择                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPU A ──────┐                                                   │
│              │                                                   │
│              ├─── NVLink ───────────────────→ GPU B (PATH_NVL)   │
│              │                                                   │
│              ├─── NVLink ──→ GPU C ── NVLink → GPU D (PATH_NVB)  │
│              │                                                   │
│              ├─── PCIe Switch ───────────────→ GPU E (PATH_PIX)  │
│              │                                                   │
│              ├─── PCIe Switch ──→ Switch ─────→ GPU F (PATH_PXB) │
│              │                                                   │
│              ├─── CPU (PHB) ──────────────────→ GPU G (PATH_PHB) │
│              │                                                   │
│              └─── NUMA (QPI) ─── CPU ──────────→ GPU H (PATH_SYS)│
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 链路选择机制

### 1. P2P 通信选择

```c
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm,
                               struct ncclTopoSystem* system,
                               int rank1, int rank2,
│              │         中间 GPU                        │
│              │                                                   │
│              ├─── PCIe ───→ PCIe Switch ── PCIe → GPU E (PATH_PIX)│
│              │
│              ├─── PCIe ──→ CPU ── PCIe ─────→ GPU E (PATH_PHB)   │
│              │                                                   │
│              └─── QPI/UPI ───────────────────→ GPU F (PATH_SYS)  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 链路选择机制

### 1. P2P 通信检查

```c
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm,
                               struct ncclTopoSystem* system,
                               int rank1, int rank2,
                               int* p2p, int *read,
                               int* intermediateRank, int* cudaP2p)
```

**检查流程：**
```
1. 检查是否在同一节点/容器
   ├── 不同 hostHash → 检查 MNNVL
   └── 不同 shmDev → 返回不可通信

2. 获取拓扑中的 GPU 节点

3. 计算路径类型
   ├── PATH_NVL  → 直接 NVLink
   ├── PATH_NVB  → 间接 NVLink (通过中间 GPU)
   └── 其他路径  → 根据路径类型判断

4. 确定 P2P Level
   ├── 默认: PATH_PXB (PCIe 交换机内)
   ├── AMD 系统: PATH_SYS (允许跨 CPU)
   └── 用户可通过 NCCL_P2P_LEVEL 覆盖

5. 检查 NVML P2P 状态
   └── 验证硬件层面的 P2P 可用性
```

### 2. 图搜索算法

```c
ncclResult_t ncclTopoCompute(ncclTopoSystem* system,
                              struct ncclTopoGraph* graph)
```

**搜索策略：**

```
Pass 1: 寻找初始解
├── 1. 尝试 crossNic 模式
├── 2. 尝试相同通道 (sameChannels)
├── 3. 增加路径类型容忍度
├── 4. 尝试更简单的树模式
└── 5. 降低带宽要求

Pass 2: 优化解
├── 从初始解出发
└── 尝试提高带宽
```

### 3. NIC 选择策略

```c
ncclResult_t ncclTopoSelectNets(struct ncclTopoSystem* system,
                                 int typeInter, int gpu,
                                 int nets[NCCL_TOPO_MAX_NODES],
                                 int* netCountRet)
```

**选择策略：**
```
策略 1: MNNVL 系统 (GPU 优先)
├── 按 GPU 分组
└── 每组内按通道排序

策略 2: 普通系统 (通道优先)
├── 按通道分组
└── 每组内按 GPU 排序

策略 3: NETDEVS_POLICY
├── AUTO: 自动分配 NIC 数量
├── ALL: 使用所有可用 NIC
└── MAX:N: 最多使用 N 个 NIC
```

### 4. GPU 排序评分

```c
struct ncclGpuScore {
  int g;             // GPU 索引
  int startIndex;    // 起始索引
  int intraNhops;    // 节点内跳数
  int intraBw;       // 节点内带宽
  int interNhops;    // 跨节点跳数
  int interPciBw;    // 跨节点 PCI 带宽
  int interBw;       // 跨节点带宽 (最重要)
};

// 排序优先级: interBw > interPciBw > interNhops > intraBw > intraNhops > startIndex
```

---

## Fallback 机制

### 1. P2P Fallback

当 P2P 不可用时：

```c
// 路径: paths.cc
if (p2p == 0) {
  // 1. 尝试通过 CPU 中转
  int cpu;
  ncclGetLocalCpu(system, g, &cpu);
  addInterStep(system, CPU, cpu, GPU, p, GPU, g);

  // 2. 检查是否可通过网络通信更快
  int useNet = 0;
  ncclTopoCheckNet(system, rank1, rank2, &useNet);
  if (useNet) {
    // 使用网络通信代替 P2P
  }
}
```

### 2. GPU Direct RDMA Fallback

```c
// 检查 GDR 是否可用
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* system,
                               int rank, int64_t netId,
                               int read, enum ncclTopoGdrMode* gdrMode)
{
  // 1. 检查 NIC 和 GPU 是否都支持 GDR
  if (net->net.gdrSupport == 0 || gpu->gpu.gdrSupport == 0)
    return ncclSuccess;  // GDR 禁用

  // 2. 检查距离是否足够近
  int netGdrLevel = ncclParamNetGdrC2c() ? PATH_P2C : PATH_PXB;
  if (distance > netGdrLevel) {
    // 距离太远，不启用 GDR
    return ncclSuccess;
  }

  // 3. 设置 GDR 模式
  *gdrMode = (distance == PATH_P2C) ? ncclTopoGdrModePci
                                    : ncclTopoGdrModeDefault;
}
```

### 3. 拓扑搜索 Fallback

```c
// search.cc: ncclTopoCompute()
if (graph->nChannels == 0) {
  // 找不到有效路径时的最终回退
  INFO(NCCL_GRAPH, "Could not find a path, falling back to simple order");

  // 使用简单的顺序排列
  for (int i=0; i<ngpus; i++)
    graph->intra[i] = system->nodes[GPU].nodes[i].gpu.rank;

  graph->bwIntra = 0.1;
  graph->typeIntra = PATH_SYS;

  // 选择任意可用的 NIC
  graph->inter[0] = system->nodes[NET].nodes[nets[0]].id;
  graph->nChannels = 1;
}
```

### 4. PXN (PCI + NVLink) 回退

当 GPU 无法直接访问 NIC 时：

```c
// 检查是否可以使用 PXN
if (ncclPxnDisable(comm) != 1) {
  int localGpuIndex;
  ncclTopoGetLocalGpu(system, netNode->id, &localGpuIndex);

  if (localGpuIndex != g && localGpuIndex != -1) {
    // 条件检查:
    // 1. 中间 GPU 到 NIC 的路径 <= PATH_PXB/P2C
    // 2. 本 GPU 到中间 GPU 通过 NVLink 连接
    // 3. 两个 GPU 在同一节点
    // 4. 中间 GPU 有更高带宽或路径更优

    if (peerNode->paths[NET][n].type <= pxnType &&
        peerNode->paths[GPU][g].type <= PATH_NVL &&
        sameNode && betterPath) {
      // 使用 PXN 路径
      addInterStep(system, GPU, localGpuIndex, GPU, g, NET, n);
    }
  }
}
```

### 5. Fallback 层级图

```
┌─────────────────────────────────────────────────────────────┐
│                     通信路径选择层级                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 1: NVLink 直连 (PATH_NVL)                           │
│     ↓ 不可用                                                 │
│                                                             │
│  Level 2: NVLink 间接 (PATH_NVB) - 通过中间 GPU            │
│     ↓ 不可用                                                 │
│                                                             │
│  Level 3: PCIe 直连 (PATH_PIX/PXB)                         │
│     ↓ 不可用                                                 │
│                                                             │
│  Level 4: GPU Direct RDMA (PATH_PXB/P2C)                   │
│     ↓ 不可用                                                 │
│                                                             │
│  Level 5: PXN (PCI + NVLink) - 通过 NVLink GPU 访问 NIC    │
│     ↓ 不可用                                                 │
│                                                             │
│  Level 6: 通过 CPU 中转 (PATH_PHB/SYS)                     │
│     ↓ 不可用                                                 │
│                                                             │
│  Level 7: 网络通信 (PATH_NET)                              │
│     ↓ 不可用                                                 │
│                                                             │
│  Level 8: 简单顺序 Fallback (最低带宽)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 关键配置参数

### P2P 相关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NCCL_P2P_LEVEL` | PATH_PXB | P2P 通信的最大路径距离 |
| `NCCL_P2P_DISABLE` | 0 | 禁用 P2P 通信 |
| `NCCL_IGNORE_DISABLED_P2P` | 0 | 忽略硬件禁用的 P2P |
| `NCCL_P2P_PER_CHANNEL_NET_BW` | 14 GB/s | 每通道网络带宽 |

### 网络相关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NCCL_NET_GDR_LEVEL` | PATH_PXB | 启用 GDR 的最大距离 |
| `NCCL_NET_GDR_READ` | -2 | 是否启用 GDR 读 |
| `NCCL_NET_GDR_C2C` | 1 | C2C 平台是否使用 GDR |
| `NCCL_NETDEVS_POLICY` | AUTO | NIC 使用策略 |
| `NCCL_NET_MERGE_LEVEL` | PATH_PORT | NIC 融合级别 |
| `NCCL_NET_FORCE_MERGE` | - | 强制融合的 NIC 列表 |

### 拓扑搜索相关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NCCL_CROSS_NIC` | 2 | 是否允许跨 NIC 通信 |
| `NCCL_PXN_DISABLE` | 0 | 禁用 PXN 路径 |
| `NCCL_PXN_C2C` | 1 | C2C 平台是否使用 PXN |
| `NCCL_MIN_NCHANNELS` | 0 | 最小通道数 |
| `NCCL_MAX_NCHANNELS` | MAXCHANNELS | 最大通道数 |

### 调试相关

| 参数 | 说明 |
|------|------|
| `NCCL_TOPO_FILE` | 指定拓扑 XML 文件路径 |
| `NCCL_TOPO_DUMP_FILE` | 导出拓扑到文件 |
| `NCCL_GRAPH_FILE` | 从文件加载图配置 |
| `NCCL_GRAPH_DUMP_FILE` | 导出图配置到文件 |

---

## 总结

NCCL 的通信拓扑检测是一个多层次、自适应的过程：

1. **拓扑发现**：通过 XML 文件和运行时检测构建完整的硬件拓扑
2. **路径计算**：使用 BFS 算法计算所有节点间的最优路径
3. **链路选择**：根据带宽、延迟和路径类型选择最佳通信链路
4. **Fallback 机制**：提供多达 8 层的回退机制确保通信可达

关键设计原则：
- **最优优先**：总是尝试使用最高带宽、最低延迟的路径
- **渐进回退**：当最优路径不可用时，逐步降级到次优方案
- **用户可控**：提供丰富的环境变量让用户调整策略
- **容错设计**：即使在极端情况下也能保证通信功能

这种设计使得 NCCL 能够在各种复杂的硬件拓扑上高效运行，同时保持良好的兼容性和可靠性。
