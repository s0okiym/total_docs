# NCCL paths.cc 源码深度解析

## 一、ncclTopoComputePaths 函数详解

### 1.1 函数整体架构

```c
ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm)
```

**函数作用**：这是NCCL拓扑系统的核心函数，负责预计算系统中所有GPU、NIC、CPU、NVS（NVSwitch）之间的最优通信路径。

**执行流程总览**：

```
ncclTopoComputePaths
├── 1. 清理旧路径 (ncclTopoRemovePaths)
├── 2. 计算基础路径 (ncclTopoSetPaths BFS遍历)
│   ├── CPU → 所有节点
│   ├── GPU → 所有节点
│   ├── NET → 所有节点
│   ├── GIN → 所有节点
│   └── NVS → 所有节点
├── 3. 更新GPU路径 (处理P2P、SHM、NET等约束)
├── 4. 更新NIC路径 (处理GDR、PXN等)
└── 5. 预计算NET本地GPU (加速后续搜索)
```

---

### 1.2 第一阶段：基础路径计算

#### 1.2.1 路径清理

```c
// Remove everything in case we're re-computing
ncclTopoRemovePaths(system);
```

**作用**：清除之前计算的所有路径，确保重新计算时不会受旧数据影响。

#### 1.2.2 BFS路径搜索核心 - ncclTopoSetPaths

这是整个拓扑计算的**核心算法**，使用**广度优先搜索（BFS）**计算从基准节点到所有其他节点的最优路径。

```c
static ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode, struct ncclTopoSystem* system)
```

**算法流程**：

```
初始化：
  - 为baseNode分配路径数组
  - baseNode到自身的路径：count=0, bw=LOC_BW(5000), type=PATH_LOC

BFS循环：
  nodeList = [baseNode]  // 当前层节点
  
  while nodeList不为空:
    nextNodeList = []    // 下一层节点
    
    for node in nodeList:
      for link in node的所有链接:
        remNode = link.remNode  // 邻居节点
        
        // 计算通过当前node到remNode的路径带宽
        bw = min(当前路径带宽, link带宽)
        
        // 检查是否经过GPU转发（限制1跳）
        if node是GPU且不是baseNode:
          if link不是NVLink 或 路径超过1跳: continue
        
        // 如果找到更优路径，更新
        if (remNode还没有路径 或 新路径更短 或 新路径带宽更高):
          更新remNode的路径
          记录路径类型（NVL/PCI/PHB等）
          将remNode加入nextNodeList
    
    nodeList = nextNodeList  // 进入下一层
```

**关键代码段解析**：

```c
// 带宽计算：取路径最小值（木桶效应）
float bw = std::min(path->bw, link->bw);

// GPU转发限制：只允许1跳NVLink转发
if (node != baseNode && node->type == GPU &&
    (ncclParamNvbDisable() || link->type != LINK_NVL || 
     remNode->type != GPU || path->count > 1)) 
    continue;

// 路径类型判断逻辑
int type = link->type == LINK_NET ? LINK_LOC : link->type;
// PCI桥接识别
if (node->type == PCI && remNode->type == PCI) type = PATH_PXB;
// 经过CPU识别
if (link->type == LINK_PCI && (node->type == CPU || link->remNode->type == CPU)) 
    type = PATH_PHB;
// NVLink桥接识别（经1个中间GPU）
if (node->type == GPU && path->type == PATH_NVL && type == PATH_NVL && remPath->count > 1) 
    type = PATH_NVB;
```

**ncclTopoComputePaths 中调用顺序**：

```c
// 1. CPU路径（作为基础）
for (int c=0; c<system->nodes[CPU].count; c++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[CPU].nodes+c, system));
}

// 2. GPU路径
for (int g=0; g<system->nodes[GPU].count; g++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[GPU].nodes+g, system));
}

// 3. NIC路径
for (int n=0; n<system->nodes[NET].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[NET].nodes+n, system));
}

// 4. GIN设备路径
for (int n=0; n<system->nodes[GIN].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[GIN].nodes+n, system));
}

// 5. NVSwitch路径
for (int n=0; n<system->nodes[NVS].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[NVS].nodes+n, system));
}
```

---

### 1.3 第二阶段：GPU路径更新

#### 1.3.1 P2P路径调整

```c
for (int g=0; g<system->nodes[GPU].count; g++) {
    for (int p=0; p<system->nodes[GPU].count; p++) {
        int p2p;
        // 检查GPU p到GPU g是否支持P2P
        NCCLCHECK(ncclTopoCheckP2p(comm, system, 
            system->nodes[GPU].nodes[p].gpu.rank,
            system->nodes[GPU].nodes[g].gpu.rank, &p2p, NULL, NULL, NULL));
        
        if (p2p == 0) {
            // 如果不支持P2P，改走CPU路径
            int cpu;
            NCCLCHECK(ncclGetLocalCpu(system, g, &cpu));
            NCCLCHECK(addInterStep(system, CPU, cpu, GPU, p, GPU, g));
        }
    }
}
```

**作用**：对于不支持P2P的GPU对，将路径改为通过CPU转发。

#### 1.3.2 不可达GPU标记

```c
// Remove GPUs we can't communicate with through P2P or SHM
for (int p=0; p<system->nodes[GPU].count; p++) {
    if (p == g) continue;
    int p2p, shm;
    NCCLCHECK(ncclTransports[TRANSPORT_P2P]->canConnect(&p2p, comm, NULL, srcInfo, dstInfo));
    if (p2p == 0) {
        NCCLCHECK(ncclTransports[TRANSPORT_SHM]->canConnect(&shm, comm, NULL, srcInfo, dstInfo));
        if (shm == 0) {
            // 标记为只能通过NET通信
            system->nodes[GPU].nodes[p].paths[GPU][g].type = PATH_NET;
        }
    }
}
```

---

### 1.4 第三阶段：NIC路径更新

#### 1.4.1 C2C + PHB 路径优化

```c
// 处理C2C平台（如Grace Hopper）上GPU到NIC的特殊路径
for (int g = 0; g < system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpuNode = system->nodes[GPU].nodes + g;
    int c = 1, localNetCount = 0, localNet[NCCL_TOPO_MAX_NODES];
    NCCLCHECK(ncclGetLocalCpu(system, g, &c));
    NCCLCHECK(ncclTopoGetLocal(system, GPU, g, NET, localNet, &localNetCount, /*pathType=*/NULL));
    
    for (int l = 0; l < localNetCount; l++) {
        int n = localNet[l];
        struct ncclTopoNode* netNode = system->nodes[NET].nodes + n;
        // 如果是P2C路径，更新类型
        if (mergePathType(gpuNode->paths[CPU][c].type, netNode->paths[CPU][c].type) == PATH_P2C) {
            gpuNode->paths[NET][n].type = std::min(PATH_P2C, gpuNode->paths[NET][n].type);
            netNode->paths[GPU][g].type = std::min(PATH_P2C, netNode->paths[GPU][g].type);
        }
    }
}
```

#### 1.4.2 PXN（Proxy by NVLink）优化

```c
// 检查是否可以通过另一GPU（有NVLink到NIC）作为代理
if (ncclPxnDisable(comm) != 1) {
    int localGpuIndex;
    NCCLCHECK(ncclTopoGetLocalGpu(system, netNode->id, &localGpuIndex));
    
    if (localGpuIndex != g && localGpuIndex != -1) {
        struct ncclTopoNode* peerNode = system->nodes[GPU].nodes+localGpuIndex;
        int pxnType = ncclParamPxnC2c() ? PATH_P2C : PATH_PXB;
        
        // PXN使用条件：
        // 1. 代理GPU到NIC的路径足够好（<= pxnType）
        // 2. 代理GPU到目标GPU有NVLink连接
        // 3. 代理GPU和目标GPU在同一节点
        // 4. 代理GPU到NIC的带宽更高或原路径需要经过CPU
        if (peerNode->paths[NET][n].type <= pxnType &&
            peerNode->paths[GPU][g].type <= PATH_NVL &&
            NCCL_TOPO_ID_SYSTEM_ID(peerNode->id) == NCCL_TOPO_ID_SYSTEM_ID(gpu->id) &&
            (peerNode->paths[NET][n].bw > gpu->paths[NET][n].bw || 
             gpu->paths[NET][n].type > PATH_PXN)) {
            
            // 添加中转路径：GPU g -> GPU localGpuIndex -> NIC n
            NCCLCHECK(addInterStep(system, GPU, localGpuIndex, GPU, g, NET, n));
        }
    }
}
```

#### 1.4.3 GDR（GPU Direct RDMA）路径调整

```c
if (gpu->paths[NET][n].type < PATH_PHB) {
    enum ncclTopoGdrMode gdr;
    NCCLCHECK(ncclTopoCheckGdr(system, rank, netNode->id, 0, &gdr));
    
    if (gdr == 0) {
        // 不支持GDR，改走CPU路径
        int localCpu;
        NCCLCHECK(ncclGetLocalCpu(system, g, &localCpu));
        NCCLCHECK(addInterStep(system, CPU, localCpu, NET, n, GPU, g));
        NCCLCHECK(addInterStep(system, CPU, localCpu, GPU, g, NET, n));
    }
}
```

---

### 1.5 辅助函数解析

#### 1.5.1 addInterStep - 添加中转路径

```c
static ncclResult_t addInterStep(struct ncclTopoSystem* system, 
    int tx, int ix,  // 中转节点类型和索引 (如 CPU, cpuIndex)
    int t1, int i1,  // 源节点类型和索引 (如 GPU, srcGpu)
    int t2, int i2)  // 目标节点类型和索引 (如 NET, nicIndex)
{
    struct ncclTopoNode* cpuNode = system->nodes[tx].nodes+ix;
    struct ncclTopoNode* srcNode = system->nodes[t1].nodes+i1;

    int l=0;
    // 构建新路径：src -> 中转 -> dst
    for (int i=0; i<srcNode->paths[tx][ix].count; i++) 
        srcNode->paths[t2][i2].list[l++] = srcNode->paths[tx][ix].list[i];
    for (int i=0; i<cpuNode->paths[t2][i2].count; i++) 
        srcNode->paths[t2][i2].list[l++] = cpuNode->paths[t2][i2].list[i];

    // 更新路径属性
    srcNode->paths[t2][i2].count = l;
    srcNode->paths[t2][i2].type = mergePathType(srcNode->paths[tx][ix].type, 
                                                 cpuNode->paths[t2][i2].type);
    // 如果是GPU作为中转，标记为PXN类型
    if (tx == GPU) srcNode->paths[t2][i2].type = PATH_PXN;
    srcNode->paths[t2][i2].bw = std::min(srcNode->paths[tx][ix].bw, 
                                          cpuNode->paths[t2][i2].bw);
}
```

---

## 二、其他关键函数详解

### 2.1 ncclTopoCheckP2p - P2P连通性检查

```c
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm, struct ncclTopoSystem* system, 
    int rank1, int rank2, int* p2p, int *read, int* intermediateRank, int* cudaP2p)
```

**检查流程**：

```
1. 排除不同节点/隔离容器
   ├── hostHash不同 → 检查MNNVL
   └── shmDev不同 → 不支持P2P

2. 获取GPU索引
   └── 如果找不到rank对应的GPU → 不支持P2P

3. 检查是否需要中间GPU转发
   └── 如果路径是2跳且中间是GPU → 记录intermediateRank

4. 检查拓扑距离
   ├── 默认p2pLevel = PATH_PXB
   ├── AMD系统特殊处理：允许到PATH_SYS
   └── 用户可覆盖：NCCL_P2P_LEVEL

5. NVML验证（可选）
   └── 检查NVML P2P状态是否OK

6. P2P Read优化（Ampere+NVLink）
   └── 如果是NVLink直连且都是Ampere → 启用read
```

### 2.2 ncclTopoCheckGdr - GDR支持检查

```c
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* system, int rank, 
    int64_t netId, int read, enum ncclTopoGdrMode* gdrMode)
```

**决策逻辑**：

```
1. 基本检查
   ├── NIC支持GDR？
   └── GPU支持GDR？

2. Read检查（如果是发送操作）
   ├── 用户禁用？→ 不支持
   └── Pre-Ampere无NVLink？→ 不支持（避免PCIe拥塞）

3. 距离检查
   ├── 默认netGdrLevel = PATH_PXB (或PATH_P2C if C2C)
   ├── 用户可覆盖：NCCL_NET_GDR_LEVEL
   └── PXN路径：使用中间GPU的距离

4. 距离判断
   └── 如果distance <= netGdrLevel → 启用GDR
```

### 2.3 ncclTopoCheckNet - 网络vs P2P选择

```c
ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* system, int rank1, int rank2, int* net)
```

**逻辑**：

```
如果NCCL_NET_DISABLE_INTRA=1:
    net = 0 (禁用内部网络)
否则：
    检查GPU到GPU的P2P带宽
    检查各自到NIC的网络带宽
    如果两者都能通过PXB或更好路径访问NIC且NIC带宽 > P2P带宽:
        net = 1 (使用网络更快)
    否则:
        net = 0 (P2P更快)
```

---

## 三、路径类型层级详解

### 3.1 路径类型定义（topo.h）

| 类型值 | 名称 | 含义 | 说明 |
|-------|------|------|------|
| 0 | PATH_LOC | 本地 | 自身到自身 |
| 1 | PATH_NVL | NVLink直连 | 最佳GPU-GPU连接 |
| 2 | PATH_NVB | NVLink桥接 | 经1中间GPU的NVLink |
| 3 | PATH_C2C | Chip-to-Chip | CPU-GPU直连（如Grace）|
| 4 | PATH_PIX | PCIe单桥 | 经过单个PCIe交换机 |
| 5 | PATH_PXB | PCIe多桥 | 经过多个PCIe交换机 |
| 6 | PATH_P2C | P2C路径 | GPU经C2C到NIC |
| 7 | PATH_PXN | Proxy NVLink | GPU经另一GPU到NIC |
| 8 | PATH_PHB | PCIe Host Bridge | 经过CPU Host Bridge |
| 9 | PATH_SYS | 系统互连 | 跨NUMA节点（QPI/UPI）|
| 10 | PATH_NET | 网络 | 跨节点 |
| 11 | PATH_DIS | 断开 | 不可达 |

### 3.2 路径类型使用场景

```
P2P传输决策：
  if path.type <= PATH_PXB → 使用P2P
  else → 使用SHM或NET

GDR传输决策：
  if path.type <= PATH_PXB (或PATH_P2C) → 启用GDR
  else → 走CPU拷贝

网络代理决策（PXN）：
  if 直接NIC路径 > PATH_PXN → 考虑使用代理GPU
```

---

## 四、ncclTopoSetPaths 详细算法图解

### 4.1 BFS示例

假设系统拓扑：
```
GPU0 --NVLink(20GB/s)-- GPU1 --PCIe(12GB/s)-- NIC0
   |                       |
   |--PCIe(12GB/s)---------|
```

**从GPU0开始的BFS**：

```
初始化：
  baseNode = GPU0
  basePath[GPU0] = {count=0, bw=5000, type=PATH_LOC}
  nodeList = [GPU0]

第1层：
  GPU0的邻居：
    - GPU1 (NVLink, 20GB/s)
    - GPU1 (PCIe, 12GB/s)  // 被忽略，NVLink更优
  
  更新GPU1路径：
    path[GPU1] = {
      count=1, 
      bw=min(5000,20)=20, 
      type=PATH_NVL,
      list=[NVLink链接]
    }
  nextNodeList = [GPU1]

第2层：
  GPU1的邻居：
    - GPU0 (跳过，已有路径)
    - NIC0 (PCIe, 12GB/s)
  
  更新NIC0路径：
    path[NIC0] = {
      count=2,
      bw=min(20,12)=12,
      type=PATH_PIX,
      list=[NVLink链接, PCIe链接]
    }
  nextNodeList = [NIC0]

第3层：
  NIC0的邻居都被访问过，结束
```

---

## 五、完整文件结构总结

### 5.1 函数分类

| 类别 | 函数 | 功能 |
|------|------|------|
| **核心计算** | ncclTopoComputePaths | 主入口，计算所有路径 |
| | ncclTopoSetPaths | BFS路径搜索算法 |
| **连通性检查** | ncclTopoCheckP2p | GPU-GPU P2P检查 |
| | ncclTopoCheckGdr | GPU-NIC GDR检查 |
| | ncclTopoCheckNet | 网络vs P2P选择 |
| | ncclTopoCheckMNNVL | 多节点NVLink检查 |
| **路径修改** | addInterStep | 添加中转路径 |
| | ncclTopoRemovePaths | 清理路径 |
| **辅助查询** | ncclGetLocalCpu | 获取最近CPU |
| | ncclTopoGetIntermediateRank | 获取PXN中间GPU |
| | ncclTopoGetPxnRanks | 获取所有PXN代理GPU |
| | ncclTopoNeedFlush | GDR flush需求检查 |
| **通道计算** | ncclTopoGetNchannels | 计算P2P通道数 |
| | ncclTopoComputeP2pChannels | 计算P2P总通道 |
| | ncclTopoComputeP2pChannelsPerPeer | 每peer通道数 |
| **拓扑分析** | ncclTopoPathAllNVLink | 全NVLink检查 |
| | ncclTopoPathAllDirectNVLink | 全直连NVLink检查 |
| | ncclTopoSplitNvLink | 分割NVLink域检查 |
| | ncclTopoGetNvbGpus | 获取NVB连接GPU |
| **系统修剪** | ncclTopoTrimSystem | 移除非连通GPU |
| **打印调试** | ncclTopoPrintPaths | 打印所有路径 |
| | printNodePaths | 打印节点路径 |
| **环境解析** | ncclGetLevel | 解析环境变量 |
| | ncclGetUserP2pLevel | 获取P2P级别设置 |

### 5.2 关键数据结构关系

```
ncclTopoSystem
├── nodes[NCCL_TOPO_NODE_TYPES]
│   ├── GPU: ncclTopoNode[]
│   │   ├── gpu.rank, gpu.dev
│   │   ├── links[] → 物理连接
│   │   └── paths[NCCL_TOPO_NODE_TYPES][] → 预计算路径
│   ├── NET: ncclTopoNode[]
│   ├── CPU: ncclTopoNode[]
│   ├── NVS: ncclTopoNode[]
│   └── GIN: ncclTopoNode[]
├── maxBw, totalBw
└── inter (是否跨节点)

ncclTopoLink
├── type (LINK_NVL/LINK_PCI/LINK_C2C)
├── bw (带宽)
└── remNode (指向邻居节点)

ncclTopoLinkList (路径)
├── list[] (链接数组)
├── count (跳数)
├── bw (瓶颈带宽)
└── type (路径类型: PATH_NVL/PATH_PXB等)
```

---

## 六、性能优化策略

### 6.1 路径计算优化

1. **缓存机制**：路径计算一次，多次使用
2. **延迟计算**：只在需要时计算特定路径
3. **BFS剪枝**：限制GPU转发只1跳

### 6.2 传输选择优化

```c
// 优先级从高到低：
1. P2P Direct (NVLink/PCIe) - 最低延迟
2. P2P NVB (经中间GPU) - 次低延迟
3. SHM (共享内存) - 无PCIe P2P时
4. PXN (代理到NIC) - 优化网络访问
5. NET (网络) - 跨节点
```

### 6.3 环境变量调优

| 环境变量 | 作用 | 典型值 |
|---------|------|--------|
| NCCL_P2P_LEVEL | P2P最大距离 | PATH_PXB(5) |
| NCCL_P2P_DISABLE | 禁用P2P | 0/1 |
| NCCL_NET_GDR_LEVEL | GDR最大距离 | PATH_PXB(5) |
| NCCL_PXN_DISABLE | 禁用PXN | 0/1 |
| NCCL_NET_DISABLE_INTRA | 禁用内部网络 | 0/1 |

---

**文档生成时间**: 2026-03-16  
**基于NCCL源码**: src/graph/paths.cc
