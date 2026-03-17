# NCCL Graph 模块深度解析

## 一、概述

NCCL的graph模块是整个通信库的核心组成部分，负责拓扑发现、路径计算、通信图构建和算法调优。本文档详细分析`/root/source/nccl/src/graph`目录下所有代码文件的功能、原理和实现细节。

### 1.1 目录结构

```
src/graph/
├── topo.h         # 拓扑系统头文件 - 定义核心数据结构
├── topo.cc        # 拓扑系统实现 - 节点、链路、路径管理
├── paths.cc       # 路径计算实现 - GPU间通信路径搜索
├── search.cc      # 搜索算法 - 最优通信图搜索
├── connect.cc     # 连接管理 - Channel和Graph连接建立
├── rings.cc       # Ring算法实现 - 构建环形通信拓扑
├── rings.h        # Ring算法头文件
├── trees.cc       # Tree算法实现 - 构建树形通信拓扑
├── tuning.cc      # 调优参数 - 带宽、延迟计算
├── xml.cc         # XML解析/序列化 - 拓扑配置
├── xml.h          # XML头文件 - XML解析工具函数
└── CMakeLists.txt # 构建配置
```

---

## 二、核心数据结构 (topo.h)

### 2.1 节点类型定义

```c
#define NCCL_TOPO_NODE_TYPES 7
#define GPU 0    // GPU设备
#define PCI 1    // PCIe交换机
#define NVS 2    // NVSwitch
#define CPU 3    // CPU (NUMA域)
#define NIC 4    // 网卡(NIC)
#define NET 5    // 网络设备
#define GIN 6    // 聚合网卡
```

### 2.2 链路类型定义

```c
#define LINK_LOC  0   // 本地连接
#define LINK_NVL  1   // NVLink
#define LINK_C2C  3   // CPU-CPU (C2C)
#define LINK_PCI  4   // PCIe
#define LINK_SYS  9   // 系统级(QPI/UPI)
#define LINK_NET  10  // 网络
```

### 2.3 路径类型定义

```c
#define PATH_LOC  0    // 本地
#define PATH_NVL  1    // NVLink直连
#define PATH_NVB  2    // 经过GPU的NVLink(1跳)
#define PATH_PIX  4    // 单PCIe桥
#define PATH_PXB  5    // 多PCIe桥
#define PATH_P2C  6    // GPU-NIC经过CPU
#define PATH_PXN  7    // GPU经过另一个GPU到NIC
#define PATH_PHB  8    // 经过PCI Host Bridge
#define PATH_SYS  9    // 经过CPU互联
#define PATH_NET  10   // 经过网络
#define PATH_DIS  11   // 断连
```

### 2.4 核心结构体

```c
// 拓扑节点
struct ncclTopoNode {
    int type;                    // 节点类型(GPU/PCI/NVS/CPU/NIC/NET/GIN)
    int64_t id;                  // 节点唯一ID
    
    // 类型特定数据 - 联合体
    union {
        struct {
            int dev;              // NVML设备号
            int rank;             // NCCL rank
            int cudaCompCap;      // CUDA计算能力
            int gdrSupport;       // GPU Direct RDMA支持
        } gpu;
        struct {
            int dev;              // 网卡设备号
            uint64_t pciId;       // PCIe ID
            uint64_t asic;        // ASIC ID
            int port;             // 端口号
            float bw;             // 带宽
            float latency;        // 延迟
            int gdrSupport;       // GDR支持
            int collSupport;      // 集合通信支持
            int maxChannels;      // 最大channel数
            int localGpu;         // 本地GPU
        } net;
        // ... 其他类型
    };
    
    int nlinks;                  // 链路数量
    struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];  // 出向链路
    
    // 预计算的到GPU和NIC的路径
    struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];
    uint64_t used;               // 搜索过程中使用标记
};

// 拓扑链路
struct ncclTopoLink {
    int type;                    // 链路类型
    float bw;                    // 带宽(GB/s)
    struct ncclTopoNode* remNode;  // 目标节点
};

// 拓扑系统
struct ncclTopoSystem {
    int systemId;                // 系统ID
    uint64_t hostHashes[NCCL_TOPO_MAX_NODES];  // 主机哈希
    int nHosts;                  // 主机数量
    struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];  // 各类节点集
    float maxBw;                 // 最大带宽
    float totalBw;               // 总带宽
    int inter;                   // 是否跨节点
};
```

---

## 三、拓扑系统实现 (topo.cc)

### 3.1 函数清单

| 函数名 | 功能描述 |
|--------|----------|
| `ncclTopoGetNode` | 根据类型和ID获取拓扑节点 |
| `ncclTopoCreateNode` | 创建新的拓扑节点 |
| `ncclTopoRemoveNode` | 删除拓扑节点 |
| `ncclTopoConnectNodes` | 连接两个拓扑节点 |
| `ncclTopoFlattenBcmSwitches` | 扁平化BCM Gen4交换机 |
| `ncclTopoConnectCpus` | 连接所有CPU节点 |
| `ncclTopoSortSystem` | 排序系统以便加速遍历 |
| `ncclTopoPrint` | 打印拓扑信息 |
| `ncclTopoAddNet` | 添加网卡到拓扑 |
| `ncclTopoAddGin` | 添加GIN到拓扑 |
| `ncclTopoAddNic` | 添加NIC到拓扑 |
| `ncclTopoAddGpu` | 添加GPU到拓扑 |
| `ncclTopoAddPci` | 添加PCI设备到拓扑 |

### 3.2 ncclTopoConnectNodes 函数详解

**功能**: 连接两个拓扑节点，聚合带宽

**使用场景**: 构建拓扑图时，将设备节点连接起来

```c
ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw) {
    // 查找现有链路，聚合带宽
    struct ncclLink* link;
    for (link = node->links; link - node->links != NCCL_TOPO_MAX_LINKS && link->remNode; link++) {
        if (link->remNode == remNode && link->type == type) break;
    }
    
    // 新增或更新链路
    if (link->remNode == NULL) node->nlinks++;
    link->type = type;
    link->remNode = remNode;
    link->bw += bw;  // 带宽聚合

    // 按带宽降序排序
    // ...
}
```

**关键点**:
- 同一对节点间的多条同类链路会**聚合带宽**
- 链路按带宽**降序排列**，优先使用高带宽路径

### 3.3 findLocalCpu 函数详解

**功能**: 向上遍历PCI树找到最近的CPU

```c
static ncclResult_t findLocalCpu(struct ncclTopoNode* node, struct ncclTopoNode** cpu, struct ncclTopoNode* from) {
    if (node->type == CPU) {
        *cpu = node;
        return ncclSuccess;
    }
    // 递归遍历PCI链路
    for (int l=0; l<node->nlinks; l++) {
        if (node->links[l].type == LINK_PCI
            && node->links[l].remNode != from
            && (node->links[l].remNode->type == PCI || node->links[l].remNode->type == CPU)) {
            NCCLCHECK(findLocalCpu(node->links[l].remNode, cpu, node));
        }
        if (*cpu != NULL) return ncclSuccess;
    }
    return ncclSuccess;
}
```

---

## 四、路径计算实现 (paths.cc)

### 4.1 核心函数

| 函数名 | 功能描述 |
|--------|----------|
| `ncclTopoSetPaths` | BFS计算所有节点间的路径 |
| `ncclTopoPrintPaths` | 打印路径信息 |
| `ncclGetLocalCpu` | 获取最近的CPU |
| `ncclTopoCheckP2p` | 检查P2P连通性 |
| `ncclTopoCheckGdr` | 检查GPU Direct RDMA支持 |
| `ncclTopoCheckMNNVL` | 检查MNNVL连接 |

### 4.2 ncclTopoSetPaths 函数详解

**功能**: 使用BFS算法预计算所有GPU到其他GPU、CPU、NET的路径

**使用场景**: NCCL初始化时，计算所有可能的通信路径及其带宽

```c
static ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode, struct ncclTopoSystem* system) {
    // 1. 初始化路径数组
    if (baseNode->paths[baseNode->type] == NULL) {
        NCCLCHECK(ncclCalloc(baseNode->paths+baseNode->type, system->nodes[baseNode->type].count));
    }

    // 2. BFS遍历
    struct ncclTopoNodeList nodeList, nextNodeList;
    nodeList.count = 1; nodeList.list[0] = baseNode;
    
    while (nodeList.count) {
        nextNodeList.count = 0;
        for (int n=0; n<nodeList.count; n++) {
            struct ncclTopoNode* node = nodeList.list[n];
            
            // 遍历当前节点的所有链路
            for (int l=0; l<node->nlinks; l++) {
                struct ncclTopoLink* link = node->links+l;
                struct ncclTopoNode* remNode = link->remNode;
                
                // 计算路径带宽(最小链路带宽)
                float bw = std::min(path->bw, link->bw);
                
                // GPU转发限制: 只允许1跳NVLink转发
                if (node != baseNode && node->type == GPU &&
                    (ncclParamNvbDisable() || link->type != LINK_NVL || 
                     remNode->type != GPU || path->count > 1)) continue;

                // 更新更优路径
                if ((remPath->bw == 0 || remPath->count > path->count) && remPath->bw < bw) {
                    // 复制路径信息
                    // ...
                    remPath->count = path->count + 1;
                    remPath->bw = bw;
                    // 添加到下一层
                }
            }
        }
        memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
    }
    return ncclSuccess;
}
```

**关键算法**:
- 使用**广度优先搜索(BFS)**遍历拓扑
- 路径带宽 = 路径上所有链路的**最小带宽**
- GPU转发**最多允许1跳**NVLink

### 4.3 ncclTopoCheckP2p 函数详解

**功能**: 检查两个rank之间的P2P连通性

```c
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm, struct ncclTopoSystem* system, 
                               int rank1, int rank2, int* p2p, int* read, 
                               int* intermediateRank, int* cudaP2p) {
    // 1. 检查是否同节点/同容器
    if (info1->hostHash != info2->hostHash) {
        if (comm && comm->MNNVL) {
            NCCLCHECK(ncclTopoCheckMNNVL(comm->topo, info1, info2, &mnnvl));
            if (!mnnvl) return ncclSuccess;
        } else {
            return ncclSuccess;  // 不同节点
        }
    }

    // 2. 获取GPU索引和路径
    struct ncclTopoLinkList* path = gpu1->paths[GPU]+g2;

    // 3. 检查是否需要中间GPU转发
    if (path->count == 2) {  // 1跳经过中间GPU
        intermediateNode = path->list[0]->remNode;
        *intermediateRank = intermediateNode->gpu.rank;
    }

    // 4. 比较路径类型与用户指定的p2pLevel
    if (path->type <= p2pLevel) *p2p = 1;

    // 5. 使用NVML验证P2P支持
    if (*p2p == 1 && checkNvml) {
        // 调用NVML检查P2P状态
        nvmlGpuP2PStatus_t status;
        status = ncclNvmlDevicePairs[indexes[i-1]][indexes[i-0]].p2pStatusRead;
        // ...
    }
}
```

---

## 五、搜索算法实现 (search.cc)

### 5.1 核心函数

| 函数名 | 功能描述 |
|--------|----------|
| `ncclTopoSearchInit` | 初始化搜索参数 |
| `ncclTopoComputeCommCPU` | 计算通信CPU |
| `ncclTopoFollowPath` | 跟随路径并检查带宽 |
| `ncclTopoSearchNextGpuSort` | 对GPU进行评分排序 |
| `ncclTopoSearchRec` | 递归搜索最优图 |
| `ncclTopoSearchRecGpu` | GPU级别递归搜索 |
| `ncclTopoSearchRecNet` | NET级别递归搜索 |
| `ncclTopoCompareGraphs` | 比较两个图的优劣 |

### 5.2 ncclTopoSearchInit 函数详解

**功能**: 初始化系统带宽参数

```c
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) {
    system->maxBw = 0.0;
    system->totalBw = 0.0;
    int inter = system->inter;

    // 单GPU情况
    if (inter == 0 && system->nodes[GPU].count == 1) {
        system->maxBw = LOC_BW;
        system->totalBw = LOC_BW;
        return ncclSuccess;
    }

    // 多GPU情况
    for (int g=0; g<system->nodes[GPU].count; g++) {
        struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
        // maxBw: 到其他GPU/NIC的最大带宽
        system->maxBw = std::max(system->maxBw, getMaxBw(system, gpu, inter ? NET : GPU));
        // totalBw: NVLink + PCIe 总带宽
        system->totalBw = std::max(system->totalBw, getTotalBw(system, gpu));
    }
    return ncclSuccess;
}
```

### 5.3 ncclTopoSearchNextGpuSort 函数详解

**功能**: 对候选GPU进行评分和排序，决定搜索顺序

**评分标准** (优先级从高到低):
1. **interBw** (跨节点带宽) - 越高越好
2. **interPciBw** (PCI带宽) - 越高越好
3. **interNhops** (跳数) - 越少越好
4. **intraBw** (节点内带宽) - 越高越好
5. **intraNhops** (节点内跳数) - 越少越好
6. **startIndex** (起始索引) - 保持公平性

```c
static int cmpScore(const void * g1, const void * g2) {
   struct ncclGpuScore *s1 = (struct ncclGpuScore*)g1;
   struct ncclGpuScore *s2 = (struct ncclGpuScore*)g2;
   int d;
   // 逐项比较
   if ((d = (s2->interBw - s1->interBw))) return d;
   if ((d = (s2->interPciBw - s1->interPciBw))) return d;
   if ((d = (s1->interNhops - s2->interNhops))) return d;
   if ((d = (s2->intraBw - s1->intraBw))) return d;
   if ((d = (s1->intraNhops - s2->intraNhops))) return d;
   return s1->startIndex - s2->startIndex;
}
```

### 5.4 搜索模式

```c
// search.cc 定义的通信模式
// Ring: GPU a -> GPU b -> .. -> GPU x -> GPU a
// Tree: GPU a -> GPU b -> .. -> GPU x (无回环)

// 跨节点Ring: NET n -> GPU a -> GPU b -> .. -> GPU x -> NET n (或m如果crossNic)
// 跨节点Tree: NET n -> GPU a -> GPU b -> .. -> GPU x -> NET n
```

### 5.5 ncclTopoCompareGraphs 函数详解

**功能**: 比较两个通信图的优劣，决定是否保存

```c
ncclResult_t ncclTopoCompareGraphs(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, 
                                   struct ncclTopoGraph* refGraph, int* copy) {
    // 1. 检查最小channel数
    if (graph->nChannels < graph->minChannels) return ncclSuccess;

    // 2. NVLS特殊处理
    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
        if (graph->nChannels > refGraph->nChannels && graph->nChannels <= system->nodes[GPU].count) *copy = 1;
        if (graph->nChannels*graph->bwInter > refGraph->nChannels*refGraph->bwInter) *copy = 1;
        return ncclSuccess;
    }

    // 3. 比较带宽 (主要标准)
    if (graph->nChannels*graph->bwIntra > refGraph->nChannels*refGraph->bwIntra) {
        *copy = 1;
        return ncclSuccess;
    }
    if (graph->nChannels*graph->bwIntra < refGraph->nChannels*refGraph->bwIntra) return ncclSuccess;

    // 4. 比较跳数 (次要标准)
    if (graph->pattern == refGraph->pattern && 
        graph->crossNic == refGraph->crossNic && 
        graph->nHops < refGraph->nHops) *copy = 1;
    return ncclSuccess;
}
```

---

## 六、连接管理 (connect.cc)

### 6.1 核心函数

| 函数名 | 功能描述 |
|--------|----------|
| `ncclTopoPreset` | 预设拓扑结构 |
| `ncclTopoPostset` | 后处理和连接建立 |
| `connectRings` | 建立Ring连接 |
| `connectTrees` | 建立Tree连接 |
| `connectCollNet` | 建立CollNet连接 |
| `connectNvls` | 建立NVLS连接 |

### 6.2 ncclTopoPreset 函数详解

**功能**: 预设channel的基础拓扑结构

```c
ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph** graphs, struct ncclTopoRanks* topoRanks) {
    int rank = comm->rank;
    int localRanks = comm->topo->nodes[GPU].count;
    int nChannels = comm->nChannels;

    // 初始化所有channel的ring/tree结构
    for (int c=0; c<nChannels; c++) {
        struct ncclChannel* channel = comm->channels+c;
        
        // 初始化Ring参数
        channel->ring.prev = channel->ring.next = -1;
        
        // 初始化Tree参数
        channel->tree.up = -1;
        for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->tree.down[i] = -1;
        
        // 初始化CollNet参数
        channel->collnetChain.up = -1;
        for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->collnetChain.down[i] = -1;
        channel->collnetDirect.out = -1;
        channel->collnetDirect.headRank = -1;
    }

    // 为每个channel设置ring/tree索引
    for (int i=0; i<localRanks; i++) {
        if (ringIntra[i] == rank) {
            topoRanks->ringRecv[c] = ringIntra[0];
            topoRanks->ringSend[c] = ringIntra[localRanks-1];
            topoRanks->ringPrev[c] = (i == 0) ? -1 : ringIntra[i-1];
            topoRanks->ringNext[c] = (i == localRanks-1) ? -1 : ringIntra[i+1];
        }
        // Tree设置类似...
    }
}
```

### 6.3 connectRings 函数详解

**功能**: 建立跨节点Ring连接

```c
static ncclResult_t connectRings(struct ncclComm* comm, int* ringRecv, int* ringSend, 
                                  int* ringPrev, int* ringNext) {
    int nChannels = comm->nChannels;
    int nNodes = comm->nNodes;
    
    for (int c=0; c<nChannels; c++) {
        int* recv = ringRecv + c*comm->nNodes;
        int* send = ringSend + c*comm->nNodes;
        int* prev = ringPrev + c*comm->nRanks;
        int* next = ringNext + c*comm->nRanks;
        
        // 遍历每个节点，构建ring
        for (int n=0; n<nNodes; n++) {
            int recvRank = recv[n];  // 当前节点接收来自该rank
            int prevSendRank = send[(n-1+nNodes)%nNodes];  // 前一个节点发送到的rank
            prev[recvRank] = prevSendRank;
            
            int sendRank = send[n];  // 当前节点发送给该rank
            int nextRecvRank = recv[(n+1)%nNodes];  // 下一个节点从该rank接收
            next[sendRank] = nextRecvRank;
        }
    }
    return ncclSuccess;
}
```

**Ring连接示意**:
```
节点0 ringRecv[0]=R0, ringSend[0]=R0
节点1 ringRecv[1]=R1, ringSend[1]=R1
节点2 ringRecv[2]=R2, ringSend[2]=R2

Ring连接: R0 -> R1 -> R2 -> R0
- prev[R0] = 发送给R0的节点 = R2
- next[R0] = 从R0接收的节点 = R1
```

---

## 七、Ring算法 (rings.cc / rings.h)

### 7.1 ncclBuildRings 函数详解

**功能**: 构建从`prev`和`next`数组生成完整的ring顺序

```c
ncclResult_t ncclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next) {
    uint64_t* rankFound;
    NCCLCHECK(ncclCalloc(&rankFound, DIVUP(nranks, 64)));

    for (int r=0; r<nrings; r++) {
        // 从当前rank开始，沿next指针遍历
        int current = rank;
        for (int i=0; i<nranks; i++) {
            rankFound[current/64] |= (1ULL<<(current%64));
            rings[r*nranks+i] = current;
            current = next[r*nranks+current];
        }
        
        // 检查ring是否闭合
        if (current != rank) {
            WARN("Error : ring %d does not loop back to start (%d != %d)", r, current, rank);
            return ncclInternalError;
        }
        
        // 检查是否包含所有rank
        for (int i=0; i<nranks; i++) {
            if ((rankFound[i/64] & (1ULL<<(i%64))) == 0) {
                WARN("Error : ring %d does not contain rank %d", r, i);
                return ncclInternalError;
            }
        }
        memset(rankFound, 0, DIVUP(nranks, 64)*sizeof(uint64_t));
    }
    free(rankFound);
    return ncclSuccess;
}
```

### 7.2 ncclGetDtree 函数详解

**功能**: 构建双二叉树，用于Tree算法

```c
// 双二叉树结构示意:
// Tree 0:                   Tree 1 (镜像):
// 0---------------8         3----------------11
//       ______/ \__                 / \______
//      4         \                /         7
//    /   \        \          /   \        /   \
//   2     6       10       1      5     9
//  / \   / \     /  \     / \    / \   / \
// 1   3  5   7  9   11   0   2  4   6 8   10

ncclResult_t ncclGetDtree(int nranks, int rank, int* s0, int* d0_0, int* d0_1, int* parentChildType0,
                          int* s1, int* d1_0, int* d1_1, int* parentChildType1) {
    // 第一棵树: 使用二叉树
    ncclGetBtree(nranks, rank, s0, d0_0, d0_1, parentChildType0);
    
    // 第二棵树: 镜像或偏移
    if (nranks % 2 == 1) {
        // 奇数: 偏移
        int shiftrank = (rank-1+nranks) % nranks;
        ncclGetBtree(nranks, shiftrank, &u, &d0, &d1, parentChildType1);
        *s1 = (u == -1) ? -1 : (u+1) % nranks;
        // ...
    } else {
        // 偶数: 镜像
        ncclGetBtree(nranks, nranks-1-rank, &u, &d0, &d1, parentChildType1);
        *s1 = (u == -1) ? -1 : nranks-1-u;
        // ...
    }
    return ncclSuccess;
}
```

---

## 八、Tree算法 (trees.cc)

### 8.1 ncclGetBtree 函数详解

**功能**: 构建二叉树结构

```c
// 二叉树结构 (nranks=15):
//        0---------------8
//              ______/ \______
//             4               12
//           /   \            /  \
//         2       6       10     \
//        / \     / \     /  \     \
//       1   3   5   7   9   11    13

ncclResult_t ncclGetBtree(int nranks, int rank, int* u, int* d0, int* d1, int* parentChildType) {
    // 找到第一个置位比特
    int bit;
    for (bit=1; bit<nranks; bit<<=1) {
        if (bit & rank) break;
    }

    if (rank == 0) {
        // 根节点
        *u = -1;
        *d0 = -1;
        *d1 = nranks > 1 ? bit >> 1 : -1;
        return ncclSuccess;
    }

    // 父节点计算
    up = (rank ^ bit) | (bit << 1);
    if (up >= nranks) up = (rank ^ bit);
    *parentChildType = (rank < up) ? 0 : 1;
    *u = up;

    // 子节点计算
    int lowbit = bit >> 1;
    down0 = lowbit == 0 ? -1 : rank - lowbit;
    
    down1 = lowbit == 0 ? -1 : rank + lowbit;
    while (down1 >= nranks) {
        down1 = lowbit == 0 ? -1 : rank + lowbit;
        lowbit >>= 1;
    }
    *d0 = down0; *d1 = down1;

    return ncclSuccess;
}
```

---

## 九、调优参数 (tuning.cc)

### 9.1 核心函数

| 函数名 | 功能描述 |
|--------|----------|
| `ncclTopoInitTunerConstants` | 初始化调优常量 |
| `ncclTopoTuneModel` | 计算带宽和延迟模型 |

### 9.2 调优常量定义

```c
// 基础延迟 (微秒)
static const ncclTunerConstants_t ncclTunerConstantsDefaults = {
    .baseLatencies = {
        {  6.8, 14.0,  8.4 },  // Tree: LL, LL128, Simple
        {  6.6, 14.0,  8.4 },  // Ring: LL, LL128, Simple
        // ... 其他算法
    },
    .hwLatencies = {
        // NVLINK
        { { 0.6, 1.25, 4.0 }, { 0.6, 1.9, 3.4 }, /* Tree, Ring */ },
        // PCI
        { { 1.0, 1.9, 4.0 }, { 1.0, 2.5, 5.7 }, /* Tree, Ring */ },
        // NET
        { { 5.0, 8.5, 14 }, { 2.7, 4.0, 14.0 }, /* Tree, Ring */ },
    },
    // LL最大带宽
    .llMaxBws = {
        {39.0, 39.0, 20.4},   // Volta
        {87.7, 22.5, 19.0},   // Ampere
        {141.0, 45.0, 35.0},  // Hopper
        {2*141.0, 2*45.0, 2*35.0},  // Blackwell
    },
};
```

### 9.3 ncclTopoTuneModel 函数详解

**功能**: 计算每个算法/协议组合的带宽和延迟

```c
ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, 
                               struct ncclTopoGraph** graphs) {
    // 1. 计算线程数
    comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 
        getNthreads("NCCL_NTHREADS", ...);
    
    // 2. 计算带宽
    for (int coll=0; coll<NCCL_NUM_FUNCTIONS; coll++) {
        for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
            for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
                float bw = nNodes <= 2 ? graphs[a]->bwIntra : graphs[a]->bwInter;
                
                // Ring算法带宽 = nChannels * bw * (nranks / nsteps)
                if (a == NCCL_ALGO_RING) {
                    float ratio = (1.0 * nRanks) / nsteps;
                    busBw *= ratio;
                }
                
                // Tree算法带宽
                if (a == NCCL_ALGO_TREE) {
                    busBw *= .5;
                }
                
                comm->bandwidths[coll][a][p] = busBw;
            }
        }
    }
    
    // 3. 计算延迟
    // Ring: (nsteps-nInterSteps)*intraLat + nInterSteps*interLat
    // Tree: 2 * ((nRanks/nNodes-1) * intraLat + log2(nNodes) * interLat)
}
```

---

## 十、XML解析 (xml.cc / xml.h)

### 10.1 核心函数

| 函数名 | 功能描述 |
|--------|----------|
| `xmlGetNode` | 解析XML节点 |
| `xmlGetToken` | 获取XML token |
| `xmlLoadSub` | 递归加载子节点 |
| `ncclTopoDumpXmlToFile` | 导出拓扑到XML |
| `ncclTopoGetXmlFromFile` | 从文件加载XML |
| `ncclTopoGetSystemFromXml` | 从XML构建拓扑系统 |

### 10.2 XML结构示例

```xml
<system>
  <cpu>
    <pci busid="0000:00:02.0">
      <gpu dev="0" sm="90" rank="0" gdr="1">
        <nvlink id="0" target="1" type="NVL"/>
        <nvlink id="1" target="2" type="NVL"/>
      </gpu>
      <nic>
        <net dev="0" guid="0x123456" speed="100000" port="0"/>
      </nic>
    </pci>
  </cpu>
</system>
```

### 10.3 ncclTopoGetSystemFromXml 函数使用场景

```c
// 从XML构建拓扑系统的典型流程:
ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem, 
                                       uint64_t localHostHash) {
    // 1. 创建拓扑系统
    NCCLCHECK(ncclCalloc(topoSystem, 1));
    
    // 2. 解析XML，填充节点
    for each xmlNode in xml->nodes {
        if (strcmp(xmlNode.name, "cpu") == 0) {
            // 添加CPU节点
        } else if (strcmp(xmlNode.name, "gpu") == 0) {
            // 添加GPU节点并解析NVLink
        } else if (strcmp(xmlNode.name, "net") == 0) {
            // 添加网卡节点
        }
    }
    
    // 3. 计算路径
    for each gpu in system->nodes[GPU] {
        ncclTopoSetPaths(gpu, system);
    }
}
```

---

## 十一、完整通信图构建流程

### 11.1 初始化流程

```
1. ncclTopoGetSystemFromXml()
   ├── 解析XML拓扑文件
   ├── 创建GPU/CPU/NIC/NET节点
   ├── 连接节点构建链路
   └── ncclTopoSetPaths() 预计算路径

2. ncclTopoSearchInit()
   ├── 计算maxBw (最大带宽)
   └── 计算totalBw (总带宽)

3. ncclTopoTuneModel()
   ├── 初始化线程数
   ├── 计算各算法/协议的带宽
   └── 计算各算法/协议的延迟

4. ncclTopoComputePaths()
   ├── Ring算法搜索
   ├── Tree算法搜索
   ├── CollNet搜索
   └── NVLS搜索
```

### 11.2 连接建立流程

```
1. ncclTopoPreset()
   ├── 初始化channel结构
   └── 设置基础rank映射

2. ncclTopoPostset()
   ├── connectRings()  - 建立Ring连接
   ├── connectTrees()  - 建立Tree连接
   ├── connectCollNet()- 建立CollNet连接
   └── connectNvls()   - 建立NVLS连接

3. ncclBuildRings()
   └── 生成完整ring顺序
```

---

## 十二、环境变量参考

| 环境变量 | 功能 | 默认值 |
|----------|------|--------|
| `NCCL_TOPO_FILE` | 指定拓扑XML文件 | 自动检测 |
| `NCCL_GRAPH_FILE` | 指定通信图XML | 自动检测 |
| `NCCL_ALGO` | 指定算法列表 | ring,tree,collnet,nvls |
| `NCCL_PROTO` | 指定协议列表 | LL,LL128,Simple |
| `NCCL_NTHREADS` | 线程数 | 自动 |
| `NCCL_MIN_NRINGS` | 最小channel数 | 1 |
| `NCCL_MAX_NRINGS` | 最大channel数 | MAXCHANNELS |
| `NCCL_P2P_LEVEL` | P2P连接级别 | PATH_PXB |
| `NCCL_NET_GDR_LEVEL` | GDR级别 | PATH_PXB |

---

**文档版本**: 1.0  
**分析源码版本**: NCCL 2.x  
**核心分析文件**:
- `src/graph/topo.h` - 91行
- `src/graph/topo.cc` - 1800+行
- `src/graph/paths.cc` - 1000+行
- `src/graph/search.cc` - 1500+行
- `src/graph/connect.cc` - 600+行
- `src/graph/rings.cc` - 200+行
- `src/graph/trees.cc` - 200+行
- `src/graph/tuning.cc` - 600+行
- `src/graph/xml.cc` - 1000+行
- `src/graph/xml.h` - 400+行
