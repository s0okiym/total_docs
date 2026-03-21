# NCCL 数据通信路径详解

本文档基于NCCL源代码分析，详细描述了NCCL通信中数据从一端GPU到另一端GPU的完整传输路径和方式。

## 一、NCCL传输层概览

NCCL（NVIDIA Collective Communication Library）支持以下几种主要传输方式，按优先级排序：

| 传输类型 | 优先级 | 适用场景 | 延迟/带宽 |
|---------|--------|---------|----------|
| P2P (Peer-to-Peer) | 最高 | 同一节点内GPU直连 | 最低延迟，最高带宽 |
| SHM (Shared Memory) | 次高 | 同一节点内进程间 | 低延迟 |
| NET (Network) | 中等 | 跨节点通信 | 依赖网络设备 |
| NVLS (NVLink SHARP) | 特殊 | NVLink集合通信加速 | 高带宽 |
| GIN (Generic Inter-Node) | 特殊 | 新型网络设备 | 设备相关 |
| CollNet | 集合 | 集合通信网络加速 | 依赖设备 |

**传输选择逻辑**（`transport.cc:selectTransport`）：
```c
for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports[t];
    // 依次尝试: P2P -> SHM -> NET -> CollNet
    NCCLCHECK(transport->canConnect(&ret, comm, graph, myInfo, peerInfo));
    if (ret) {
        // 找到可用传输方式
        connector->transportComm = transportComm;
        ...
    }
}
```

---

## 二、详细数据传输路径

### 2.1 同一节点内GPU到GPU通信

#### 路径A: P2P Direct（最理想情况）
```
GPU A显存 → NVLink/PCIe → GPU B显存
```

**触发条件**：
- 两个GPU位于同一节点
- GPU支持CUDA P2P访问（通过`cudaDeviceCanAccessPeer`检测）
- 拓扑距离不超过`NCCL_P2P_LEVEL`设定（默认PXB级别）
- 同一进程内或启用IPC

**实现代码**（`transport/p2p.cc`）：
```c
if (P2P_SAME_PID(myInfo, peerInfo) && ncclParamP2pDirectDisable() == 0) {
    resources->type = P2P_DIRECT;
    // 直接使用指针传递，无需拷贝
}
```

#### 路径B: P2P IPC（跨进程）
```
GPU A显存 → cudaIpcGetMemHandle → IPC Handle传递 → 
cudaIpcOpenMemHandle → GPU B显存映射 → 直接访问
```

**或cuMem API方式**：
```
GPU A显存 → cuMemExportToShareableHandle → 共享Handle →
cuMemImportFromShareableHandle → GPU B显存映射
```

**关键代码**（`transport/p2p.cc:p2pMap`）：
```c
if (P2P_SAME_PID(myInfo, peerInfo)) {
    // 同进程，直接启用P2P访问
    cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
    *devMem = p2pBuff->directPtr;
} else {
    // 跨进程，使用IPC导入
    ncclP2pImportShareableBuffer(comm, peerInfo->rank, ...);
}
```

#### 路径C: P2P + Intermediate GPU（经由中间GPU转发）
```
GPU A显存 → NVLink → GPU C显存 → NVLink → GPU B显存
```

**触发条件**：当两个GPU没有直接NVLink连接，但通过某个中间GPU可以连通时。

**代码逻辑**（`graph/paths.cc:ncclTopoCheckP2p`）：
```c
// Set intermediate GPU rank, if routing through an intermediate GPU.
struct ncclTopoLinkList* path = gpu1->paths[GPU]+g2;
if (path->count == 2) {
    struct ncclTopoNode* intermediateNode = path->list[0]->remNode;
    if (intermediateNode->type == GPU) {
        *intermediateRank = intermediateNode->gpu.rank;
    }
}
```

#### 路径D: 共享内存传输（SHM）
```
GPU A显存 → cudaMemcpy (D2H) → 共享内存 → 
cudaMemcpy (H2D) → GPU B显存
```

**触发条件**：
- P2P不可用（如GPU不支持P2P）
- 拓扑距离超过阈值
- 使用`NCCL_SHM_USE_CUDA_MEMCPY`启用显式拷贝

**两种实现方式**：
1. **传统MMAP**：使用`/dev/shm/nccl-xxx`共享文件
2. **cuMem Host**：使用CUDA的cuMem API分配主机共享内存

---

### 2.2 跨节点GPU到GPU通信

#### 路径E: 标准网络传输（NET Transport）
```
源GPU显存 → [GDRDMA/PCIe] → 本地网卡 → 网络 → 
对端网卡 → [GDRDMA/PCIe] → 对端GPU显存
```

**详细步骤**：

1. **本地GPU到网卡**（`transport/net.cc`）：
   - 如果启用GDRDMA（GPU Direct RDMA）：直接DMA从GPU显存到网卡
   - 如果走PCIe：先拷贝到主机内存，再传给网卡

2. **网络传输**：
   - 通过NCCL网络插件（IB Verbs、NCCL-Tuner等）发送

3. **对端网卡到GPU**：
   - GDRDMA：直接DMA到对端GPU显存
   - 否则：先到主机内存，再拷贝到GPU

**关键代码**（`transport/net.cc:sendProxyProgress`）：
```c
// 发送侧
if (ready) {
    NCCLCHECK(proxyState->ncclNet->isend(resources->netSendComm, buff, size, ...));
}

// 接收侧  
NCCLCHECK(proxyState->ncclNet->irecv(resources->netRecvComm, subCount, ptrs, sizes, ...));
```

#### 路径F: PXN（Proxy by NVLink）优化
```
源GPU显存 → NVLink → 本地代理GPU → PCIe → 本地网卡 → 
网络 → 对端网卡 → PCIe → 对端目标GPU
```

**优化原理**：选择通过NVLink连接到NIC的GPU作为代理，减少PCIe瓶颈。

**代码逻辑**（`graph/paths.cc:ncclTopoGetPxnRanks`）：
```c
// 当直接连接的GPU到NIC带宽不如通过另一个GPU时
if (peerNode->paths[NET][n].type <= pxnType &&
    peerNode->paths[GPU][g].type <= PATH_NVL) {
    // 使用该GPU作为NIC的代理
    NCCLCHECK(addInterStep(system, GPU, localGpuIndex, GPU, g, NET, n));
}
```

#### 路径G: NVLink SHARP（NVLS）集合通信
```
多个GPU通过NVLink Switch同时Reduce → 结果广播回各GPU
```

**NVLS工作原理**：
1. 创建多播组（Multicast Group）
2. 各GPU绑定物理内存到多播组
3. 写操作自动广播到组内所有成员
4. 读操作可从多播地址获取聚合结果

**关键代码**（`transport/nvls.cc`）：
```c
// 创建多播组
CUCHECK(cuMulticastCreate(mcHandle, prop));

// 各rank绑定物理内存
err = CUPFN(cuMulticastBindMem(*mcHandle, 0, *ucHandle, 0, ucsize, 0));

// Map多播地址
CUCHECK(cuMemMap((CUdeviceptr)*mcptr, mcsize, 0, *mcHandle, 0));
```

**NVLS支持的通信模式**：
- **ReduceScatter**: UC写入 → MC聚合 → 各GPU读自己的部分
- **AllGather**: 各GPU写入部分 → MC广播 → 各GPU读取全部
- **AllReduce**: UC写入+原子操作 → MC广播 → 各GPU读取结果

#### 路径H: GIN（Generic Inter-Node）传输
```
GPU显存 → GIN设备 → 网络 → GIN设备 → 对端GPU显存
```

**GIN特点**：
- 新一代网络设备抽象
- 支持细粒度信号/计数器机制
- 设备端可直接编程

**代码逻辑**（`gin/gin_host.cc`）：
```c
// 创建GIN连接
NCCLCHECK(ginState->ncclGin->connect(comm->ginContext, handles, nGinRanks, 
                                     myGinRank, nContextsPerComm, 
                                     ginState->ginQueueDepth, listenComm, 
                                     ginState->ginComms + n));

// 注册内存窗口
NCCLCHECK(ncclGinRegister(comm, address, size, ginHostWins, ginDevWins, winFlags));
```

---

## 三、拓扑路径计算

NCCL通过`graph/paths.cc`预先计算所有设备间的最优路径：

### 3.1 路径类型层级
```
PATH_LOC  (0) - 本地（自己）
PATH_NVL  (1) - NVLink直连
PATH_NVB  (2) - NVLink经由中间GPU
PATH_C2C  (3) - Chip-to-Chip（CPU-GPU之间）
PATH_PIX  (4) - 单PCIe桥
PATH_PXB  (5) - 多PCIe桥
PATH_P2C  (6) - GPU通过C2C到NIC
PATH_PXN  (7) - GPU通过另一GPU到NIC（Proxy）
PATH_PHB  (8) - 经过CPU Host Bridge
PATH_SYS  (9) - 跨NUMA节点
PATH_NET (10) - 经过网络
PATH_DIS (11) - 断开
```

### 3.2 路径搜索算法
```c
// 广度优先搜索（BFS）设置路径
ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode, struct ncclTopoSystem* system) {
    // 初始化起始节点
    basePath->count = 0;
    basePath->bw = LOC_BW;
    basePath->type = PATH_LOC;
    
    // BFS遍历所有可达节点
    while (nodeList.count) {
        for (每个节点的每个链接) {
            // 计算带宽和跳数
            float bw = std::min(path->bw, link->bw);
            // 更新最优路径
            if (更好的路径条件) {
                remPath->count = path->count + 1;
                remPath->bw = bw;
                remPath->type = pathType;
            }
        }
    }
}
```

---

## 四、缓冲区管理和注册

### 4.1 用户缓冲区注册流程

```
用户缓冲区
    ↓
nccIpcLocalRegisterBuffer / ncclNetLocalRegisterBuffer
    ↓
检查缓冲区是否已注册（ncclRegFind）
    ↓
分配注册记录（ncclReg）
    ↓
对P2P传输：
    - 获取物理段信息（cuMemGetAddressRange）
    - 获取IPC Handle（cudaIpcGetMemHandle / cuMemExportToShareableHandle）
    - 传递给对端代理（ncclProxyCallBlocking）
    - 对端导入（cudaIpcOpenMemHandle / cuMemImportFromShareableHandle）
    
对NET传输：
    - 注册内存到网卡（ncclNet->regMr / regMrDmaBuf）
    - DMA-BUF方式更优（零拷贝）
    
对NVLS传输：
    - 创建多播组
    - 绑定用户缓冲区物理地址（cuMulticastBindAddr）
    - 映射到多播地址空间
```

### 4.2 缓冲区注册类型

| 注册类型 | 用途 | 适用范围 |
|---------|------|---------|
| NCCL_IPC_REG_BUFFER | P2P直接访问 | 同一节点内 |
| NCCL_NET_REG_BUFFER | 网络传输 | 跨节点 |
| NCCL_NVLS_REG_BUFFER | NVLS聚合 | NVLink集合通信 |

---

## 五、Proxy（代理）线程机制

NCCL使用Proxy线程处理异步网络操作和复杂传输：

### 5.1 Proxy工作流程
```
主线程提交ProxyOp → Proxy队列 → Proxy线程轮询处理 → 完成回调
```

### 5.2 各传输的Proxy使用

| 传输方式 | Proxy用途 |
|---------|----------|
| P2P Direct | 无需Proxy（设备端直接完成） |
| P2P IPC | 需要Proxy处理IPC Handle交换和内存映射 |
| SHM | 需要Proxy处理共享内存同步 |
| NET | 必须Proxy处理网络IO（isend/irecv/test） |
| NVLS | Proxy处理多播组创建和内存绑定 |
| GIN | Proxy处理GIN设备操作 |

### 5.3 Proxy进度函数
```c
// P2P发送进度
static ncclResult_t p2pSendProxyProgress(...) {
    // 处理CE memcpy（如启用）
    if (useMemcpy) {
        cudaMemcpyAsync(...);
        cudaEventRecord(...);
        cudaEventQuery(...); // 检查完成
    }
}

// 网络发送进度
static ncclResult_t sendProxyProgress(...) {
    // 检查GPU就绪，提交网络发送
    NCCLCHECK(proxyState->ncclNet->isend(...));
    // 检查网络完成
    NCCLCHECK(proxyState->ncclNet->test(...));
}
```

---

## 六、完整通信示例

### 6.1 单节点AllReduce（NVLS路径）
```
阶段1: ReduceScatter
    GPU 0-7各自写入UC内存 → NVSwitch自动Reduce → MC内存
    
阶段2: AllGather  
    各GPU从MC内存读取完整结果
```

### 6.2 跨节点Send/Recv（标准网络路径）
```
发送端Rank 0 (Node 0):
    ncclSend(sendbuff, ...)
    → 创建send task
    → Proxy线程检查head/tail
    → 数据就绪后调用isend
    → 网络传输
    
接收端Rank 1 (Node 1):
    ncclRecv(recvbuff, ...)
    → 创建recv task  
    → Proxy线程预先post irecv
    → 数据到达后写入recvbuff
    → 更新tail通知GPU
```

### 6.3 同一进程内P2P
```
Rank 0 → Rank 1:
    ncclSend(sendbuff, ...)
    → 检测到P2P Direct可用
    → 设备端直接写对端显存（通过NVLink/PCIe P2P）
    → 无需Proxy参与
```

---

## 七、性能优化建议

### 7.1 拓扑感知优化
1. **P2P优化**：确保同节点GPU使用NVLink连接，避免走PCIe
2. **PXN优化**：让有NVLink到NIC的GPU做网络代理
3. **SHM优化**：确保`/dev/shm`在同一文件系统上

### 7.2 缓冲区注册优化
1. 提前注册常用缓冲区（`ncclCommRegister`）
2. 确保注册缓冲区对齐到页边界
3. 使用DMA-BUF支持（`ncclCuMemEnable`）

### 7.3 NVLS优化
1. 确保Fabric Manager正常运行（NVSwitch配置）
2. 使用CUDA 12.1+获得完整NVLS支持
3. 大数据量时NVLS收益最明显

---

## 八、关键源代码文件

| 文件路径 | 功能说明 |
|---------|---------|
| `src/transport.cc` | 传输层主框架，传输方式选择 |
| `src/transport/p2p.cc` | P2P传输实现（Direct/IPC/cuMem） |
| `src/transport/shm.cc` | 共享内存传输实现 |
| `src/transport/net.cc` | 网络传输实现（GDRDMA、Proxy） |
| `src/transport/nvls.cc` | NVLink SHARP实现 |
| `src/gin/gin_host.cc` | GIN传输实现 |
| `src/graph/paths.cc` | 拓扑路径计算 |
| `src/graph/topo.h` | 拓扑类型定义 |
| `src/enqueue.cc` | 任务入队和调度 |
| `src/proxy.cc` | Proxy线程实现 |

---

**文档生成时间**: 2026-03-16  
**基于NCCL版本**: 2.x (Git Commit: 源码分析)
