# NCCL RMA 机制文档清单

## 文档列表

### 1. nccl_rma_mechanism.md (28KB)
**RMA机制详解**
- RMA概述与核心特点
- 应用场景与启用条件
- 架构设计（双路径：CE路径和Proxy路径）
- 核心概念（LSA、Context、Symmetric Window、Signal机制）
- 代码实现分析
- CE路径详解（初始化、Put操作、WaitSignal操作）
- Proxy路径详解（初始化、Progress线程、完成轮询、描述符发出）
- 信号机制详解
- 详细的流程图
- 配置与调优建议

### 2. nccl_rma_gin_integration.md (16KB)
**RMA与GIN插件关系详解**
- GIN概述与架构
- GIN插件接口定义
- RMA与GIN集成架构
- 连接建立流程
- RMA Proxy Context创建流程
- 拓扑发现（LSA发现）
- 路径选择决策
- 代码分析（初始化、内存注册、Put操作、信号处理）

## 核心概念速查

### RMA API
```c
ncclPutSignal()   // 将数据写入远程内存并发送信号
ncclSignal()      // 仅发送信号（无数据传输）
ncclWaitSignal()  // 等待来自多个peer的信号
```

### 双路径设计
| 路径 | 适用场景 | 实现方式 |
|-----|---------|---------|
| CE路径 | LSA内（本地） | CUDA Memcpy D2D |
| Proxy路径 | LSA外（网络） | GIN iput/iputSignal |

### LSA (Local System Area)
- 可通过NVLink/PCIe直接访问的GPU集合
- 同一物理节点内
- 对称内存窗口已建立

### 关键环境变量
```bash
NCCL_GIN_ENABLE=1                    # 启用GIN插件
NCCL_RMA_PROXY_QUEUE_SIZE=<size>     # Proxy队列大小
```

## 文件结构

```
/root/workspace/nccl/src/rma/
├── rma.cc           # RMA主逻辑，任务调度
├── rma_ce.cc        # CE路径实现
└── rma_proxy.cc     # Proxy路径实现

/root/workspace/nccl/src/include/rma/
├── rma.h            # RMA主头文件
├── rma_ce.h         # CE路径头文件
└── rma_proxy.h      # Proxy路径头文件
```

## 关键数据结构

```c
// RMA任务描述符
struct ncclTaskRma {
  ncclFunc_t func;              // PutSignal/Signal/WaitSignal
  int ctx;                      // 上下文ID
  int peer;                     // 目标rank
  ncclSignalMode_t signalMode;  // NONE/SIGNAL
  // ...
};

// Proxy描述符
struct ncclRmaProxyDesc {
  uint64_t srcOff, dstOff;      // 源/目标偏移
  void *srcHandle, *dstHandle;  // 内存句柄
  size_t size;                  // 传输大小
  ncclRmaSignal_t signal;       // 信号信息
  uint64_t seq;                 // 序列号
  void *request;                // 网络请求
};

// Proxy上下文
struct ncclRmaProxyCtx {
  void *ginCollComm;            // GIN通信器
  struct ncclRmaProxyDesc** pendingQueues;  // 无锁缓冲区
  uint32_t *pis, *cis;          // 生产者/消费者索引
  uint64_t *readySeqs, *doneSeqs;  // 序列号
  uint64_t *signalsDev, *signalsHost;  // 信号缓冲区
};
```

## 使用示例

```c
// 1. 创建对称内存窗口
ncclWinCreate(buff, size, comm, &win);

// 2. 获取远程peer的窗口信息
ncclWinGetAttrs(peerWin, &attrs);

// 3. Put数据到远程窗口
ncclPutSignal(localBuff, count, ncclFloat32,
              peerRank, peerWindow, offset,
              sigIdx, ctx, flags, comm, stream);

// 4. 等待信号
ncclWaitSignalDesc_t descs[] = {
  {peerRank, opCnt, sigIdx, ctx}
};
ncclWaitSignal(1, descs, comm, stream);

// 5. 读取远程数据（现在安全）
// ...
```

## 调试技巧

```bash
# 启用RMA调试
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=RMA,PROXY

# 关键日志关键字
# - "scheduleRmaTasksToPlan"  - 任务调度
# - "ncclRmaPutCe"            - CE路径操作
# - "ncclRmaPutProxy"         - Proxy路径操作
# - "ncclRmaProxyPollDesc"    - Proxy进度
```

## 性能考虑

1. **批处理**：连续相同类型的操作自动批处理
2. **并行执行**：CE和Proxy路径可并行执行
3. **大传输切分**：超过1GB的传输自动切分
4. **无锁设计**：Proxy使用无锁循环缓冲区

## 与其他文档的关系

- 与超时检测文档的关系：RMA操作同样适用超时检测机制
- 与Proxy线程文档的关系：RMA Proxy使用独立的Progress线程
- 与GIN文档的关系：RMA依赖GIN插件实现网络传输
