# NCCL search.cc 搜索算法深度解析

## 一、概述

`search.cc` 是NCCL通信图构建的核心模块，负责在拓扑系统中搜索**最优的通信路径组合**。其核心目标是为Ring、Tree、CollNet、NVLS等不同通信算法找到满足带宽约束的最优channel配置。

### 1.1 核心功能

```
┌────────────────────────────────────────────────────────────────┐
│                    NCCL 通信图搜索流程                          │
├────────────────────────────────────────────────────────────────┤
│  1. 初始化搜索参数 (ncclTopoSearchInit)                         │
│     └── 计算 maxBw, totalBw                                     │
│                                                                 │
│  2. 设置搜索约束 (ncclTopoCompute)                              │
│     └── 带宽目标、路径类型约束、crossNic设置                    │
│                                                                 │
│  3. 递归搜索 (ncclTopoSearchRec)                                │
│     ├── 跨节点: ncclTopoSearchRecNet                            │
│     │   └── 从NET开始搜索                                       │
│     └── 单节点: ncclTopoSearchRecGpu                            │
│         └── 从GPU开始搜索                                       │
│                                                                 │
│  4. 比较保存最优解 (ncclTopoCompareGraphs)                      │
│     └── 比较带宽、channel数、跳数                               │
│                                                                 │
│  5. 输出结果 (ncclTopoPrintGraph)                               │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 关键常量定义

```c
// 搜索超时控制
#define NCCL_SEARCH_GLOBAL_TIMEOUT (1ULL<<19)  // 全局超时: 524288次迭代
#define NCCL_SEARCH_TIMEOUT        (1<<14)      // 单次搜索: 16384次迭代
#define NCCL_SEARCH_TIMEOUT_TREE   (1<<14)      // Tree搜索: 16384次迭代
#define NCCL_SEARCH_TIMEOUT_SAMECHANNELS (1<<8) // 同Channel: 256次迭代

// 强制顺序标志
#define FORCED_ORDER_PCI    1   // 按PCI顺序尝试
#define FORCED_ORDER_REPLAY 2   // 重放上一次channel顺序
```

---

## 二、核心数据结构

### 2.1 ncclTopoGraph 结构体

```c
// graph.h 中定义的图结构
struct ncclTopoGraph {
    int id;                    // 图ID (算法类型)
    int pattern;               // 通信模式
    int crossNic;              // 是否允许跨NIC (0/1/2)
    int nChannels;             // 当前channel数
    int minChannels;           // 最小channel数
    int maxChannels;           // 最大channel数
    
    float bwIntra;             // 节点内带宽
    float bwInter;             // 跨节点带宽
    float latencyInter;        // 跨节点延迟
    
    int typeIntra;             // 节点内路径类型 (PATH_NVL/PATH_PIX等)
    int typeInter;             // 跨节点路径类型
    
    int sameChannels;          // 是否使用相同channel配置
    int nHops;                 // 总跳数
    
    int* intra;                // GPU rank顺序 [nChannels * ngpus]
    int64_t* inter;            // NIC ID数组 [nChannels * 2] (入口/出口)
};
```

### 2.2 ncclGpuScore 结构体

```c
// GPU评分结构，用于决定搜索顺序
struct ncclGpuScore {
    int g;             // GPU索引
    int startIndex;    // 起始索引（公平性）
    int intraNhops;    // 节点内跳数
    int intraBw;       // 节点内带宽
    int interNhops;    // 跨节点跳数
    int interPciBw;    // PCI带宽
    int interBw;       // 跨节点带宽（最重要）
};
```

### 2.3 通信模式定义

```c
// 搜索模式定义（代码注释）
// 
// 节点内模式:
// Ring            : GPU a -> GPU b -> .. -> GPU x -> GPU a
// Tree            : GPU a -> GPU b -> .. -> GPU x
//
// 跨节点模式:
// Ring            : NET n -> GPU a -> GPU b -> .. -> GPU x -> NET n (or m if crossNic)
// Tree            : NET n -> GPU a -> GPU b -> .. -> GPU x
//                              `--> NET n (or m if crossNic)
// Split Tree      : NET n -> GPU a -> GPU b -> .. -> GPU x
//                                       `--> NET n (or m if crossNic)
// Split Tree Loop : NET n -> GPU a -> GPU b -> .. -> GPU x -> GPU a
//                                       `--> NET n (or m if crossNic)
```

---

## 三、搜索初始化函数

### 3.1 ncclTopoSearchInit 函数详解

**功能**: 初始化系统的带宽参数，为搜索提供基准

**使用场景**: NCCL初始化时，在搜索前调用

```c
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) {
    system->maxBw = 0.0;
    system->totalBw = 0.0;
    int inter = system->inter;  // 是否跨节点

    // 特殊情况: 单GPU且不跨节点
    if (inter == 0 && system->nodes[GPU].count == 1) {
        system->maxBw = LOC_BW;    // 5000 GB/s (本地带宽)
        system->totalBw = LOC_BW;
        return ncclSuccess;
    }

    // 遍历所有GPU，计算最大带宽
    for (int g=0; g<system->nodes[GPU].count; g++) {
        struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
        
        // maxBw: GPU到其他GPU或NET的最大路径带宽
        system->maxBw = std::max(system->maxBw, 
                                 getMaxBw(system, gpu, inter ? NET : GPU));
        
        // totalBw: GPU的总出向带宽 (NVLink或PCI)
        system->totalBw = std::max(system->totalBw, getTotalBw(system, gpu));
    }
    return ncclSuccess;
}
```

### 3.2 getMaxBw 辅助函数

```c
// 获取GPU到特定类型节点的最大带宽
static float getMaxBw(struct ncclTopoSystem* system, struct ncclTopoNode* gpu, int type) {
    float maxBw = 0.0;
    for (int i=0; i<system->nodes[type].count; i++) {
        struct ncclTopoLinkList* path = gpu->paths[type]+i;
        float bw = path->bw;
        if (path->count == 0) continue;  // 无路径
        maxBw = std::max(maxBw, bw);
    }
    return maxBw;
}
```

### 3.3 getTotalBw 辅助函数

```c
// 计算GPU的总出向带宽
static float getTotalBw(struct ncclTopoSystem* system, struct ncclTopoNode* gpu) {
    float nvlinkBw = 0.0, pciBw = 0.0;
    for (int l=0; l<gpu->nlinks; l++) {
        struct ncclTopoLink* link = gpu->links+l;
        if (link->type == LINK_NVL) nvlinkBw += link->bw;  // NVLink带宽累加
        if (link->type == LINK_PCI) pciBw = link->bw;      // PCI带宽取单个
    }
    return std::max(pciBw, nvlinkBw);  // 取较大值
}
```

**关键点**:
- `maxBw`: 单channel的理论最大带宽
- `totalBw`: GPU所有链路的聚合带宽

---

## 四、路径跟随与带宽检查

### 4.1 ncclTopoFollowPath 函数详解

**功能**: 尝试沿路径前进，检查带宽是否足够

**参数**:
- `mult`: 1表示占用带宽，-1表示释放带宽

```c
static ncclResult_t ncclTopoFollowPath(
    struct ncclTopoSystem* system, 
    struct ncclTopoGraph* graph,
    int type1, int index1,        // 起点 (类型, 索引)
    int type2, int index2,        // 终点 (类型, 索引)
    float mult,                    // 1: 占用, -1: 释放
    struct ncclTopoNode** node     // 输出: 到达的节点
) {
    *node = system->nodes[type2].nodes+index2;
    if (type1 == -1) return ncclSuccess;  // 起点-1表示直接返回终点

    struct ncclTopoNode* node1 = system->nodes[type1].nodes+index1;
    struct ncclTopoLinkList* path = node1->paths[type2]+index2;
    struct ncclTopoNode* node2 = system->nodes[type2].nodes+index2;
    struct ncclTopoLinkList* revPath = node2->paths[type1]+index1;

    // 确定带宽和路径类型
    int intra = (type1 == GPU || type1 == NVS) && (type2 == GPU || type2 == NVS);
    float bw = intra ? graph->bwIntra : graph->bwInter;
    int type = intra ? graph->typeIntra : graph->typeInter;

    // 检查路径有效性
    if (path->type >= PATH_DIS) return ncclSuccess;  // 断连
    if (mult == 1 && (path->type > type)) return ncclSuccess;  // 超出类型限制
    
    // Tree需要双向检查
    if (mult == 1 && (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
                      graph->pattern == NCCL_TOPO_PATTERN_TREE ||
                      graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) &&
        (revPath->type > type)) return ncclSuccess;

    bw *= mult;  // 占用或释放

    // 尝试跟随路径，检查带宽
    int step = 0;
    NCCLCHECK(followPath(path, node1, path->count, bw, &step));
    
    if (step < path->count) goto rewind;  // 带宽不足

    // 成功: 更新跳数，返回目标节点
    graph->nHops += mult*path->count;
    *node = system->nodes[type2].nodes+index2;
    return ncclSuccess;

rewind:
    // 回滚已占用的带宽
    NCCLCHECK(followPath(path, node1, step, -bw, &step));
    return ncclSuccess;
}
```

### 4.2 followPath 辅助函数

```c
// 实际执行带宽占用/释放
static ncclResult_t followPath(
    struct ncclTopoLinkList* path, 
    struct ncclTopoNode* start, 
    int maxSteps,    // 要走的步数
    float bw,        // 带宽变化量
    int* steps       // 输出: 实际走过的步数
) {
    float pciBw = bw;
    
    // 预处理: Intel CPU P2P开销
    for (int step=0; step<path->count; step++) {
        struct ncclTopoNode* node = path->list[step]->remNode;
        if (node->type == CPU) {
            if (path->type == PATH_PHB && start->type == GPU &&
                node->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 &&
                node->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
                // Intel P2P开销: 带宽 * 6/5
                pciBw = INTEL_P2P_OVERHEAD(bw);
            }
        }
    }

    struct ncclTopoNode* node = start;
    for (int step=0; step<maxSteps; step++) {
        struct ncclTopoLink* link = path->list[step];
        float fwBw = link->type == LINK_PCI ? pciBw : bw;
        float revBw = 0;
        
        // 检查反向带宽需求 (某些架构需要)
        if (link->remNode->type == GPU && 
            link->remNode->gpu.cudaCompCap < 80 && 
            start->type != GPU) {
            revBw += fwBw/8;  // Volta需要额外带宽
        }
        
        // 带宽检查
        if (link->bw < fwBw || (revBw && revLink->bw < revBw)) {
            *steps = step;
            return ncclSuccess;  // 带宽不足
        }
        
        // 占用带宽 (使用舍入避免浮点误差)
        SUB_ROUND(link->bw, fwBw);
        if (revBw) SUB_ROUND(revLink->bw, revBw);
        
        node = link->remNode;
    }
    *steps = maxSteps;
    return ncclSuccess;
}

// 舍入宏定义
#define SUB_ROUND(a, b) (a = roundf((a-b)*1000)/1000)
```

**关键点**:
- 带宽是**累减**的，每次搜索尝试都会占用链路带宽
- Intel CPU有**额外的P2P开销**（20%）
- Volta GPU需要**额外的反向带宽**

---

## 五、GPU评分排序

### 5.1 ncclTopoSearchNextGpuSort 函数详解

**功能**: 对候选GPU进行评分排序，决定搜索优先级

**使用场景**: 在搜索过程中选择下一个要访问的GPU

```c
ncclResult_t ncclTopoSearchNextGpuSort(
    struct ncclTopoSystem* system, 
    struct ncclTopoGraph* graph, 
    struct ncclTopoNode* gpu,      // 当前GPU
    int* next,                      // 输出: 排序后的GPU索引数组
    int* countPtr,                  // 输出: 候选GPU数量
    int sortNet                     // 是否按网络排序
) {
    const uint64_t flag = 1ULL<<(graph->nChannels);
    int ngpus = system->nodes[GPU].count;
    struct ncclTopoLinkList* paths = gpu->paths[GPU];
    struct ncclTopoLinkList* netPaths = NULL;
    
    if (sortNet) {
        NCCLCHECK(getNetPaths(system, graph, &netPaths));
    }

    struct ncclGpuScore scores[NCCL_TOPO_MAX_NODES];
    int start = gpu - system->nodes[GPU].nodes;
    int count = 0;
    
    // 收集所有候选GPU
    for (int i=1; i<ngpus; i++) {
        int g = (start+i)%ngpus;  // 环形遍历
        
        // 跳过无路径或已使用的GPU
        if (paths[g].count == 0) continue;
        if (system->nodes[GPU].nodes[g].used & flag) continue;
        
        scores[count].g = g;
        scores[count].startIndex = i;
        scores[count].intraNhops = paths[g].count;
        scores[count].intraBw = paths[g].bw;
        
        if (netPaths) {
            scores[count].interNhops = netPaths[g].count;
            scores[count].interPciBw = gpuPciBw(system->nodes[GPU].nodes+g);
            scores[count].interBw = netPaths[g].bw;
        }
        count++;
    }

    // 按评分排序
    qsort(scores, count, sizeof(struct ncclGpuScore), cmpScore);

    // 处理NVSwitch特殊情况
    if (system->nodes[NVS].count) {
        // NVSwitch倾向于与相邻GPU通信
        // ... (见下文)
    }
    
    // 输出结果
    for (int i=0; i<count; i++) {
        next[i] = scores[i].g;
    }
    *countPtr = count;
    return ncclSuccess;
}
```

### 5.2 cmpScore 比较函数

**功能**: 定义GPU评分的优先级顺序

```c
static int cmpScore(const void * g1, const void * g2) {
    struct ncclGpuScore *s1 = (struct ncclGpuScore*)g1;
    struct ncclGpuScore *s2 = (struct ncclGpuScore*)g2;
    int d;
    
    // 优先级从高到低:
    if ((d = (s2->interBw - s1->interBw))) return d;        // 1. 跨节点带宽
    if ((d = (s2->interPciBw - s1->interPciBw))) return d;  // 2. PCI带宽
    if ((d = (s1->interNhops - s2->interNhops))) return d;  // 3. 跨节点跳数(少好)
    if ((d = (s2->intraBw - s1->intraBw))) return d;        // 4. 节点内带宽
    if ((d = (s1->intraNhops - s2->intraNhops))) return d;  // 5. 节点内跳数(少好)
    return s1->startIndex - s2->startIndex;                // 6. 公平性
}
```

**评分优先级表**:

| 优先级 | 评分项 | 说明 |
|--------|--------|------|
| 1 | interBw | 跨节点带宽，越高越好 |
| 2 | interPciBw | PCI带宽，越高越好 |
| 3 | interNhops | 跨节点跳数，越少越好 |
| 4 | intraBw | 节点内带宽，越高越好 |
| 5 | intraNhops | 节点内跳数，越少越好 |
| 6 | startIndex | 公平性，避免饥饿 |

### 5.3 NVSwitch特殊处理

```c
// 对于NVSwitch系统，优先与相邻GPU通信
if (system->nodes[NVS].count) {
    int index = gpu - system->nodes[GPU].nodes;
    int prevGpu = (index-1+ngpus)%ngpus;
    int nextGpu = (index+1)%ngpus;
    
    int firstGpus[2];
    int firstGpuCount = 0;
    
    // 根据模式选择优先方向
    if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
        firstGpus[0] = nextGpu; 
        firstGpus[1] = prevGpu; 
        firstGpuCount = 2;
    } else if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ||
               graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
        firstGpus[0] = prevGpu; 
        firstGpus[1] = nextGpu; 
        firstGpuCount = 2;
    } else {
        firstGpus[0] = nextGpu; 
        firstGpuCount = 1;
    }
    
    // 将优先GPU移到数组前面
    for (int g=0; g<firstGpuCount; g++) {
        for (i=0; i<count && next[i] != firstGpus[g]; i++);
        if (i<count) {
            for (; i>0; i--) next[i] = next[i-1];
            next[0] = firstGpus[g];
        }
    }
}
```

---

## 六、核心搜索算法

### 6.1 ncclTopoSearchRec 函数详解

**功能**: 递归搜索入口，根据是否跨节点选择搜索策略

```c
ncclResult_t ncclTopoSearchRec(
    struct ncclTopoSystem* system, 
    struct ncclTopoGraph* graph,      // 当前搜索状态
    struct ncclTopoGraph* saveGraph,  // 保存最优解
    int* time                          // 剩余搜索时间
) {
    int backToNet, backToFirstRank;
    NCCLCHECK(ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank));
    
    if (system->inter) {
        // 跨节点: 从NET开始搜索
        ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time);
    } else {
        // 单节点: 从GPU开始搜索
        if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
            // NVLS特殊处理
            NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, graph->nChannels));
            return ncclSuccess;
        } else if (graph->nChannels == 0) {
            // 第一次搜索: 尝试PCI顺序
            NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time, -1, -1, 0));
        } else {
            // 后续搜索: 尝试重放上一次channel顺序
            int g;
            NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
            NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, -1, -1, g));
        }
        
        if (graph->sameChannels == 0 || graph->nChannels == 0) {
            // 尝试所有GPU
            for (int g=0; g<system->nodes[GPU].count; g++) {
                NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, g));
            }
        }
    }
    return ncclSuccess;
}
```

### 6.2 ncclTopoSearchParams 函数

**功能**: 根据模式和是否跨节点，确定搜索参数

```c
ncclResult_t ncclTopoSearchParams(
    struct ncclTopoSystem* system, 
    int pattern, 
    int* backToNet,       // 在哪个step返回NET
    int* backToFirstRank  // 在哪个step返回第一个rank
) {
    if (system->inter) {
        // 跨节点模式
        if (pattern == NCCL_TOPO_PATTERN_RING) 
            *backToNet = system->nodes[GPU].count-1;  // 最后一个GPU返回NET
        else if (pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) 
            *backToNet = 1;  // 第二个GPU返回NET
        else 
            *backToNet = 0;  // 第一个GPU返回NET
        *backToFirstRank = -1;
    } else {
        // 单节点模式
        *backToNet = -1;
        if (pattern == NCCL_TOPO_PATTERN_RING) 
            *backToFirstRank = system->nodes[GPU].count-1;  // 返回第一个rank
        else 
            *backToFirstRank = -1;
    }
    return ncclSuccess;
}
```

**参数表**:

| 模式 | 跨节点 | backToNet | backToFirstRank |
|------|--------|-----------|-----------------|
| Ring | 否 | -1 | ngpus-1 |
| Ring | 是 | ngpus-1 | -1 |
| Tree | 否 | -1 | -1 |
| Tree | 是 | 0 | -1 |
| Split Tree | 是 | 1 | -1 |

### 6.3 ncclTopoSearchRecGpu 函数详解

**功能**: GPU级别的递归搜索

```c
ncclResult_t ncclTopoSearchRecGpu(
    struct ncclTopoSystem* system, 
    struct ncclTopoGraph* graph, 
    struct ncclTopoGraph* saveGraph, 
    struct ncclTopoNode* gpu,      // 当前GPU
    int step,                       // 当前step (0~ngpus-1)
    int backToNet,                  // 返回NET的step
    int backToFirstRank,            // 返回第一个rank的step
    int forcedOrder,                // 强制顺序标志
    int *time                       // 剩余时间
) {
    // 超时检查
    if ((*time) <= 0) return ncclSuccess;
    (*time)--;

    int ngpus = system->nodes[GPU].count;
    
    // ===== 搜索完成检查 =====
    if (step == ngpus) {
        int copy = 0;
        graph->nChannels++;
        NCCLCHECK(ncclTopoCompareGraphs(system, graph, saveGraph, &copy));
        
        if (copy) {
            memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));
            if (graph->nChannels == graph->maxChannels) 
                *time = -1;  // 达到最大channel，立即停止
        }
        
        // 继续搜索更多channel
        if (graph->nChannels < graph->maxChannels) {
            NCCLCHECK(ncclTopoSearchRec(system, graph, saveGraph, time));
        }
        graph->nChannels--;
        return ncclSuccess;
    }
    
    // 记录当前GPU rank
    graph->intra[graph->nChannels*ngpus+step] = gpu->gpu.rank;
    int g = gpu - system->nodes[GPU].nodes;

    // ===== 分支1: 返回NET =====
    if (step == backToNet) {
        if (system->inter) {
            int startNetIndex;
            NCCLCHECK(getNetIndex(system, graph->inter[graph->nChannels*2], &startNetIndex));
            struct ncclTopoNode* startNet = system->nodes[NET].nodes+startNetIndex;
            
            int nets[NCCL_TOPO_MAX_NODES];
            int netCount;
            NCCLCHECK(ncclTopoSelectNets(system, graph->typeInter, g, nets, &netCount));
            
            for (int i=0; i<netCount; i++) {
                int n = nets[i];
                if (!ncclTopoSearchCheckNet(system, graph, startNet, n, step)) continue;
                
                struct ncclTopoNode* net;
                NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, 1, &net));
                if (net) {
                    graph->inter[graph->nChannels*2+1] = net->id;
                    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, ...));
                    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, -1, &net));
                }
            }
        }
    }
    // ===== 分支2: NVLS模式 =====
    else if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
        NCCLCHECK(ncclTopoSearchTryNvls(system, graph, saveGraph, g, ngpus, time));
    }
    // ===== 分支3: CollNet Direct模式 =====
    else if (graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
        NCCLCHECK(ncclTopoSearchTryCollnetDirect(system, graph, saveGraph, g, ngpus, time));
    }
    // ===== 分支4: 继续到下一个GPU =====
    else if (step < ngpus-1) {
        int next[NCCL_TOPO_MAX_NODES];
        int count;
        
        if (forcedOrder == FORCED_ORDER_PCI) {
            next[0] = step+1;
            count = 1;
        } else if (forcedOrder == FORCED_ORDER_REPLAY) {
            NCCLCHECK(ncclTopoReplayGetGpu(system, graph, step, next));
            count = 1;
        } else {
            NCCLCHECK(ncclTopoSearchNextGpuSort(system, graph, gpu, next, &count, ...));
        }
        
        for (int i=0; i<count; i++) {
            NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, step+1, backToNet, backToFirstRank, forcedOrder, time, GPU, g, next[i]));
        }
    }
    // ===== 分支5: 返回第一个GPU (Ring闭合) =====
    else if (step == backToFirstRank) {
        int p;
        NCCLCHECK(getGpuIndex(system, graph->intra[graph->nChannels*ngpus], &p));
        struct ncclTopoNode* firstGpu;
        NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, 1, &firstGpu));
        if (firstGpu) {
            NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, firstGpu, step+1, backToNet, -1, forcedOrder, time));
            NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, -1, &firstGpu));
        }
    }
    // ===== 分支6: 完成当前channel =====
    else {
        NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, ngpus, -1, -1, forcedOrder, time));
    }
    return ncclSuccess;
}
```

### 6.4 ncclTopoSearchTryGpu 函数

**功能**: 尝试移动到下一个GPU，处理带宽占用和回滚

```c
ncclResult_t ncclTopoSearchTryGpu(
    struct ncclTopoSystem* system, 
    struct ncclTopoGraph* graph, 
    struct ncclTopoGraph* saveGraph, 
    int step, int backToNet, int backToFirstRank, int forcedOrder, 
    int *time, 
    int type, int index,    // 起点
    int g                    // 目标GPU索引
) {
    const uint64_t flag = 1ULL<<(graph->nChannels);
    struct ncclTopoNode* gpu;
    
    // 尝试移动到目标GPU
    NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, 1, &gpu));
    
    if (gpu) {
        // 标记GPU为已使用
        gpu->used ^= flag;
        
        // 递归搜索
        NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, backToNet, backToFirstRank, forcedOrder, time));
        
        // 回滚标记
        gpu->used ^= flag;
        
        // 释放带宽
        NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, -1, &gpu));
    }
    return ncclSuccess;
}
```

---

## 七、网络搜索

### 7.1 ncclTopoSearchRecNet 函数详解

**功能**: 网卡级别的搜索，选择合适的入口NIC

```c
ncclResult_t ncclTopoSearchRecNet(
    struct ncclTopoSystem* system, 
    struct ncclTopoGraph* graph, 
    struct ncclTopoGraph* saveGraph, 
    int backToNet, int backToFirstRank, 
    int* time
) {
    const int bw = graph->bwInter;
    int nets[NCCL_TOPO_MAX_NODES];
    int netCount;
    int graphFound = 0;
    
    // 选择候选NIC列表
    NCCLCHECK(ncclTopoSelectNets(system, graph->typeInter, -1, nets, &netCount));
    
    for (int i=0; i<netCount; i++) {
        // 使用轮转方式选择NIC（增加channel数时轮转）
        int n = nets[(graph->nChannels+i)%netCount];
        struct ncclTopoNode* net = system->nodes[NET].nodes+n;
        
        // 检查NIC是否满足条件
        if (graph->collNet && net->net.collSupport == 0) continue;  // 需要集合通信支持
        if (net->net.bw < bw) continue;  // 带宽不足
        
        // 记录入口NIC
        graph->inter[graph->nChannels*2] = net->id;
        graph->latencyInter = net->net.latency;
        
        // 占用NIC带宽（同一ASIC的所有端口）
        for (int i=0; i<system->nodes[NET].count; i++) {
            if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
                (system->nodes[NET].nodes[i].net.port == net->net.port)) {
                system->nodes[NET].nodes[i].net.bw -= bw;
            }
        }
        
        // 根据模式选择GPU搜索策略
        if (graph->pattern == NCCL_TOPO_PATTERN_NVLS || 
            graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
            // NVLS/CollNet: 只找NIC对应的本地GPU作为head
            int gpu = net->net.localGpu;
            if (gpu != -1) {
                // 检查重复head
                int duplicate = 0;
                for (int gc = 0; gc < graph->nChannels; gc++) {
                    if (graph->intra[gc * system->nodes[GPU].count] == 
                        system->nodes[GPU].nodes[gpu].gpu.rank) {
                        duplicate = 1;
                        break;
                    }
                }
                if (!duplicate) {
                    NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, gpu));
                    graphFound = 1;
                }
            }
        } else {
            // Ring/Tree: 正常搜索
            if (graph->nChannels > 0 && graph->sameChannels == 1) {
                // 重放上一次channel的GPU顺序
                int g;
                NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
                NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, NET, n, g));
            } else {
                // 首次搜索: 先尝试PCI顺序
                if (graph->nChannels == 0 && system->nodes[NVS].count == 0) {
                    int t = 1 << 10;  // 短超时
                    NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, &t, NET, n, 0));
                    if (t == -1) *time = -1;
                }
                
                // 尝试NIC的本地GPU
                int localGpu = net->net.localGpu;
                if (localGpu != -1) {
                    NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, localGpu));
                }
                
                // 尝试其他本地GPU
                int localGpus[NCCL_TOPO_MAX_NODES], localGpuCount, pathType;
                NCCLCHECK(ncclTopoGetLocal(system, NET, n, GPU, localGpus, &localGpuCount, &pathType));
                if (pathType == PATH_DIS) continue;  // 无连接GPU
                
                for (int g = 0; g < localGpuCount; ++g) {
                    if (localGpus[g] == localGpu) continue;
                    NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, localGpus[g]));
                }
            }
        }
        
        // 释放NIC带宽
        for (int i=0; i<system->nodes[NET].count; i++) {
            if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
                (system->nodes[NET].nodes[i].net.port == net->net.port)) {
                system->nodes[NET].nodes[i].net.bw += bw;
            }
        }
    }
    return ncclSuccess;
}
```

### 7.2 ncclTopoSelectNets 函数

**功能**: 选择候选NIC列表

```c
ncclResult_t ncclTopoSelectNets(
    struct ncclTopoSystem* system, 
    int typeInter,    // 路径类型限制
    int gpu,          // 指定GPU索引 (-1表示所有GPU)
    int nets[NCCL_TOPO_MAX_NODES], 
    int* netCountRet
) {
    int netCount = 0;

    // 步骤1: 添加优先NIC
    if (system->nHosts > 1 && ncclParamScatterEnable()) {
        // MNNVL系统: 按GPU优先排序
        NCCLCHECK(ncclTopoPrefNetsGpuFirst(system, gpu, nets, &netCount));
    } else {
        // 普通系统: 按Channel优先排序
        NCCLCHECK(ncclTopoPrefNetsChannelFirst(system, gpu, nets, &netCount));
    }

    // 步骤2: 添加其他满足条件的NIC
    // ...
    
    *netCountRet = netCount;
    return ncclSuccess;
}
```

---

## 八、图比较函数

### 8.1 ncclTopoCompareGraphs 函数详解

**功能**: 比较两个通信图，决定是否保存新结果

```c
ncclResult_t ncclTopoCompareGraphs(
    struct ncclTopoSystem* system, 
    struct ncclTopoGraph* graph,    // 当前搜索结果
    struct ncclTopoGraph* refGraph, // 当前最优解
    int* copy                        // 输出: 是否复制
) {
    *copy = 0;
    
    // 检查: 必须满足最小channel数
    if (graph->nChannels < graph->minChannels) return ncclSuccess;

    // NVLS特殊处理: channel越多越好
    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
        if (graph->nChannels > refGraph->nChannels && 
            graph->nChannels <= system->nodes[GPU].count) *copy = 1;
        if (graph->nChannels*graph->bwInter > refGraph->nChannels*refGraph->bwInter) *copy = 1;
        return ncclSuccess;
    }
    
    // 标准1: 比较总带宽 (channel数 * 带宽)
    if (graph->nChannels*graph->bwIntra > refGraph->nChannels*refGraph->bwIntra) {
        *copy = 1;
        return ncclSuccess;
    }
    if (graph->nChannels*graph->bwIntra < refGraph->nChannels*refGraph->bwIntra) {
        return ncclSuccess;
    }
    
    // 标准2: 相同带宽下，比较跳数
    if (graph->pattern == refGraph->pattern && 
        graph->crossNic == refGraph->crossNic && 
        graph->nHops < refGraph->nHops) {
        *copy = 1;
    }
    return ncclSuccess;
}
```

**比较优先级**:

1. **总带宽** = `nChannels × bwIntra`
2. **跳数** = `nHops` (带宽相同时)

---

## 九、NVLS和CollNet搜索

### 9.1 ncclTopoSearchTryNvls 函数

**功能**: NVLS模式的特殊搜索逻辑

```c
ncclResult_t ncclTopoSearchTryNvls(
    struct ncclTopoSystem* system, 
    struct ncclTopoGraph* graph, 
    struct ncclTopoGraph* saveGraph, 
    int g, int ngpus, int *time
) {
    struct ncclTopoNode* nvs;
    struct ncclTopoNode* gpu;
    
    // 检查NVSwitch->GPU方向带宽
    int d0=0;
    do {
        // head GPU需要2倍带宽
        NCCLCHECK(ncclTopoFollowPath(system, graph, NVS, 0, GPU, d0, d0 == g ? 2 : 1, &gpu));
        d0++;
    } while (gpu && d0 < system->nodes[GPU].count);
    
    if (gpu == NULL) {
        d0--;
    } else {
        // 检查GPU->NVSwitch方向带宽
        int d1=0;
        do {
            NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, d1, NVS, 0, d1 == g ? 2 : 1, &nvs));
            d1++;
        } while (nvs && d1 < system->nodes[GPU].count);
        
        if (nvs == NULL) {
            d1--;
        } else {
            // 双向都满足，完成搜索
            NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, NULL, ngpus, -1, -1, 0, time));
        }
        
        // 回滚GPU->NVS带宽
        while (d1) {
            d1--;
            NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, d1, NVS, 0, d1 == g ? -2 : -1, &nvs));
        }
    }
    
    // 回滚NVS->GPU带宽
    while (d0) {
        d0--;
        NCCLCHECK(ncclTopoFollowPath(system, graph, NVS, 0, GPU, d0, d0 == g ? -2 : -1, &gpu));
    }
    return ncclSuccess;
}
```

### 9.2 ncclTopoSearchTryCollnetDirect 函数

**功能**: CollNet Direct模式的搜索逻辑

```c
ncclResult_t ncclTopoSearchTryCollnetDirect(
    struct ncclTopoSystem* system, 
    struct ncclTopoGraph* graph, 
    struct ncclTopoGraph* saveGraph, 
    int g, int ngpus, int *time
) {
    int fwdg = 0;
    int bwdg = 0;
    struct ncclTopoNode* gpu = NULL;
    
    // 每个GPU只占用 1/(ngpus-1) 的带宽
    float mul = 1.0 / (float)(system->nodes[GPU].count - 1);
    
    // 检查从head到所有其他GPU的带宽
    do {
        NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, fwdg, mul, &gpu));
    } while (gpu && ++fwdg < system->nodes[GPU].count);

    if (gpu != NULL) {
        // 检查从所有其他GPU到head的带宽
        do {
            NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, bwdg, GPU, g, mul, &gpu));
        } while (gpu && ++bwdg < system->nodes[GPU].count);
        
        if (gpu != NULL) {
            // 双向都满足，填充intra数组
            int step = 1;
            for (int index = 0; index < ngpus; ++index) {
                if (index != g) {
                    graph->intra[graph->nChannels * ngpus + step] = 
                        system->nodes[GPU].nodes[index].gpu.rank;
                    step++;
                }
            }
            NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, NULL, ngpus, -1, -1, 0, time));
        }
        
        // 回滚...
    }
    // 回滚...
    return ncclSuccess;
}
```

---

## 十、完整搜索流程

### 10.1 ncclTopoCompute 函数详解

**功能**: 主搜索函数，协调整个搜索过程

```c
ncclResult_t ncclTopoCompute(ncclTopoSystem* system, struct ncclTopoGraph* graph) {
    int ccMin;
    NCCLCHECK(ncclTopoGetCompCap(system, &ccMin, NULL));
    int ngpus = system->nodes[GPU].count;

    // 1. 设置crossNic参数
    int crossNic = (system->nodes[NET].count > 1) &&
        (graph->pattern == NCCL_TOPO_PATTERN_RING ||
         graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
         graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) ? ncclParamCrossNic() : 0;
    graph->crossNic = crossNic == 1 ? 1 : 0;

    // 2. 确定路径类型范围
    int minTypeIntra = PATH_LOC, minTypeInter = PATH_PIX;
    int maxTypeIntra = PATH_SYS, maxTypeInter = PATH_SYS;
    
    if (ngpus > 1) {
        NCCLCHECK(ncclTopoGetGpuMinPath(system, GPU, &minTypeIntra));
        NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxTypeIntra));
    }
    if (system->inter) {
        NCCLCHECK(ncclTopoGetGpuMinPath(system, NET, &minTypeInter));
        NCCLCHECK(ncclTopoGetGpuMaxPath(system, NET, &maxTypeInter));
    }

    // 3. 初始化搜索参数
    graph->typeIntra = minTypeIntra;
    graph->typeInter = minTypeInter;
    graph->nChannels = 0;
    graph->sameChannels = 1;  // 默认使用相同channel

    // 4. 选择带宽目标数组
    int nspeeds;
    float* speedArray;
    if (system->inter == 0) {
        nspeeds = ccMin >= 100 ? NSPEEDSINTRA_SM100 : 
                  (ccMin >= 90 ? NSPEEDSINTRA_SM90 : NSPEEDSINTRA);
        speedArray = ccMin >= 100 ? sm100SpeedArrayIntra : 
                     (ccMin >= 90 ? sm90SpeedArrayIntra : speedArrayIntra);
    } else {
        nspeeds = ccMin >= 100 ? NSPEEDSINTER_SM100 : 
                  (ccMin >= 90 ? NSPEEDSINTER_SM90 : NSPEEDSINTER);
        speedArray = ccMin >= 100 ? sm100SpeedArrayInter : 
                     (ccMin >= 90 ? sm90SpeedArrayInter : speedArrayInter);
    }

    // 5. 找到合适的起始带宽
    int speedIndex = 0;
    while ((speedArray[speedIndex] > system->maxBw || 
            speedArray[speedIndex]*graph->minChannels > system->totalBw) && 
           speedIndex < nspeeds-1) {
        speedIndex++;
    }

    struct ncclTopoGraph tmpGraph;
    memcpy(&tmpGraph, graph, sizeof(struct ncclTopoGraph));
    tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
    int64_t globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;

search:
    // 6. 执行搜索
    int time = tmpGraph.sameChannels ? NCCL_SEARCH_TIMEOUT_SAMECHANNELS :
               tmpGraph.pattern == NCCL_TOPO_PATTERN_TREE ? 
               NCCL_SEARCH_TIMEOUT_TREE : NCCL_SEARCH_TIMEOUT;
    tmpGraph.nChannels = 0;
    globalTimeout -= time;

    NCCLCHECK(ncclTopoSearchRec(system, &tmpGraph, graph, &time));

    // 7. 检查是否找到完美解
    if (time == -1) goto done;
    if (graph->nChannels*graph->bwInter >= system->totalBw) goto done;

    // 8. Pass 1: 尝试不同参数组合
    if (pass == 1) {
        // 尝试不同channel配置
        if (tmpGraph.sameChannels == 1 && ...) {
            tmpGraph.sameChannels = 0;
            goto search;
        }
        
        // 尝试更宽松的路径类型
        if (tmpGraph.typeIntra < maxIntra) {
            tmpGraph.typeIntra += 1;
            goto search;
        }
        
        // 尝试crossNic
        if (crossNic == 2 && tmpGraph.crossNic == 0) {
            tmpGraph.crossNic = 2;
            goto search;
        }
        
        // 降低带宽目标
        if (speedIndex < nspeeds-1) {
            tmpGraph.bwInter = tmpGraph.bwIntra = speedArray[++speedIndex];
            goto search;
        }
    }

done:
    // 9. Pass 2: 尝试提高带宽
    if (pass == 1) {
        NCCLCHECK(ncclTopoDupChannels(graph, ccMin, ngpus));
        memcpy(&tmpGraph, graph, sizeof(struct ncclTopoGraph));
        pass = 2;
        // 尝试更高带宽...
    }

    // 10. 回退方案
    if (graph->nChannels == 0) {
        INFO(NCCL_GRAPH, "Could not find a path, falling back to simple order");
        for (int i=0; i<ngpus; i++) 
            graph->intra[i] = system->nodes[GPU].nodes[i].gpu.rank;
        graph->bwIntra = 0.1;
        graph->typeIntra = PATH_SYS;
        graph->nChannels = 1;
    }
    return ncclSuccess;
}
```

### 10.2 带宽目标数组

```c
// 不同GPU架构的带宽目标 (GB/s)
float speedArrayIntra[] = { 40.0, 30.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0 };
float speedArrayInter[] = { 48.0, 30.0, 28.0, 24.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.4, 1.2, 0.24, 0.12 };

// SM90 (Hopper) 带宽目标
float sm90SpeedArrayIntra[] = { 60.0, 50.0, 40.0, 30.0, 24.0, 20.0, 15.0, 12.0, 11.0, 6.0, 3.0 };
float sm90SpeedArrayInter[] = { 48.0, 45.0, 42.0, 40.0, 30.0, 24.0, 22.0, 20.0, 17.5, 15.0, 12.0, 6.0, 3.0, 2.4, 1.2, 0.24, 0.12 };

// SM100 (Blackwell) 带宽目标
float sm100SpeedArrayIntra[] = { 90.0, 80.0, 70.0, 60.0, 50.0, 45.0, 40.0, 30.0, 24.0, 20.0, 19.0, 18.0 };
float sm100SpeedArrayInter[] = { 96.0, 80.0, 48.0, 45.1, 42.0, 40.0, 30.0, 24.0, 22.0, 20.0, 17.5, 15.0, 12.0, 6.0, 3.0, 2.4, 1.2, 0.24, 0.12 };
```

---

## 十一、搜索流程图

```
ncclTopoCompute
├── 1. 设置搜索约束
│   ├── crossNic = ncclParamCrossNic()
│   ├── typeIntra = minTypeIntra
│   ├── typeInter = minTypeInter
│   └── bwIntra = bwInter = speedArray[0]
│
├── 2. Pass 1: 寻找可行解
│   ├── search:
│   │   ├── ncclTopoSearchRec
│   │   │   ├── 跨节点: ncclTopoSearchRecNet
│   │   │   │   ├── 选择NIC (ncclTopoSelectNets)
│   │   │   │   └── 对每个NIC:
│   │   │   │       └── ncclTopoSearchTryGpu
│   │   │   │           └── ncclTopoSearchRecGpu
│   │   │   │               ├── step == backToNet → 返回NET
│   │   │   │               ├── step == ngpus → 比较+保存
│   │   │   │               └── 继续下一个GPU
│   │   │   └── 单节点: ncclTopoSearchRecGpu
│   │   │       └── 类似流程
│   │   │
│   │   └── 检查结果:
│   │       ├── time == -1 → 完美解，结束
│   │       ├── 找到解 → 继续搜索更多channel
│   │       └── 未找到 → 调整参数goto search
│   │
│   └── 参数调整顺序:
│       ├── sameChannels = 0
│       ├── pattern = TREE (如果BALANCED_TREE失败)
│       ├── typeIntra += 1
│       ├── typeInter += 1
│       ├── crossNic = 2
│       └── bw = speedArray[++speedIndex]
│
├── 3. Pass 2: 优化带宽
│   ├── ncclTopoDupChannels (复制channel)
│   └── 尝试更高带宽
│
└── 4. 回退方案
    └── 使用简单PCI顺序
```

---

## 十二、环境变量

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `NCCL_CROSS_NIC` | 2 | 跨NIC策略 (0=禁用, 1=启用, 2=自动) |
| `NCCL_GRAPH_FILE` | - | 从XML加载图配置 |
| `NCCL_GRAPH_DUMP_FILE` | - | 导出图到XML |
| `NCCL_MNNVL_SCATTER_NETS_ENABLE` | 1 | MNNVL scatter网络启用 |
| `NCCL_MNNVL_RAIL_PER_HOST` | 0 | 每主机rail模式 |
| `NCCL_P2P_PXN_LEVEL` | 2 | PXN级别 |
| `NCCL_NETDEVS_POLICY` | - | 网卡选择策略 |

---

**文档版本**: 1.0  
**源码文件**: `/root/source/nccl/src/graph/search.cc`  
**总行数**: ~1400行  
**核心函数数**: 25个
