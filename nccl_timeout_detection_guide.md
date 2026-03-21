# NCCL 通信超时检测机制实现指南

## 目录
1. [概述](#概述)
2. [NCCL 架构分析](#nccl-架构分析)
3. [Proxy Thread 机制详解](#proxy-thread-机制详解)
4. [通信路径分析](#通信路径分析)
5. [超时检测实现方案](#超时检测实现方案)
6. [代码修改详解](#代码修改详解)
7. [环境变量配置](#环境变量配置)
8. [测试与验证](#测试与验证)

---

## 概述

### 目标
在NCCL中实现一个统一的通信超时检测机制，支持：
- 监测各种通信路径（Network、RDMA、TCP、Shared Memory、P2P）
- 通过环境变量配置超时时间
- 超时发生时打印详细的日志信息
- 为0时禁用监测

### 关键设计原则
1. **非侵入式**: 尽量不影响现有代码结构
2. **可配置**: 通过环境变量控制超时时间和行为
3. **详细日志**: 提供足够的信息用于故障诊断
4. **低 overhead**: 避免对正常通信性能造成显著影响

---

## NCCL 架构分析

### 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                     NCCL Communication                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Main Thread │  │ Proxy Thread │  │   Progress Thread   │  │
│  │   (User)     │  │  (Service)   │  │   (Network I/O)     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                    │              │
│         ▼                ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Transport Layer                         │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐  │   │
│  │  │  P2P   │ │  SHM   │ │  NET   │ │  COLLNET     │  │   │
│  │  │Transport│ │Transport│ │Transport│ │  (NVLink SHARP)│  │   │
│  │  └────────┘ └────────┘ └────────┘ └──────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 关键文件结构

```
/root/workspace/nccl/src/
├── proxy.cc              # Proxy thread 主实现
├── include/
│   ├── proxy.h           # Proxy 相关数据结构定义
│   └── transport.h       # Transport 接口定义
└── transport/
    ├── net.cc            # Network (RDMA/TCP) transport
    ├── p2p.cc            # P2P (GPU Direct) transport
    ├── shm.cc            # Shared Memory transport
    └── coll_net.cc       # Collective Network transport
```

---

## Proxy Thread 机制详解

### Proxy Thread 主循环

**文件**: `src/proxy.cc` (约1000行)

```cpp
void* ncclProxyProgress(void *proxyState_) {
  struct ncclProxyState* proxyState = (struct ncclProxyState*)proxyState_;
  struct ncclProxyProgressState* state = &proxyState->progressState;
  
  do {
    int idle = 1;
    // 1. 处理活跃的操作
    ncclResult_t ret = progressOps(proxyState, state, state->active, &idle);
    
    // 2. 获取新的操作
    if (idle || !state->active || (++proxyOpAppendCounter == ncclParamProgressAppendOpFreq())) {
      int added = 0;
      ret = ncclProxyGetPostedOps(proxyState, &added);
    }
    
    lastIdle = idle;
  } while ((state->stop == 0 || (state->stop == 1 && state->active)) 
           && COMPILER_ATOMIC_LOAD(proxyState->abortFlag, std::memory_order_acquire) == 0);
  
  return NULL;
}
```

### 操作处理流程

**函数**: `progressOps()` (src/proxy.cc:400)

```cpp
static ncclResult_t progressOps(struct ncclProxyState* proxyState, 
                                struct ncclProxyProgressState* state, 
                                struct ncclProxyArgs* opStart, 
                                int* idle) {
  struct ncclProxyArgs* op = opStart;
  while (op) {
    // 调用 transport 特定的 progress 函数
    ncclResult_t ret = op->progress(proxyState, op);
    
    if (op->state == ncclProxyOpNone || ret != ncclSuccess) {
      NCCLCHECK(removeOp(state, &op, &prevOp));
    } else {
      prevOp = op;
      op = op->next;
    }
  }
  return ncclSuccess;
}
```

### 关键数据结构

**结构体**: `ncclProxyArgs` (src/include/proxy.h:160)

```cpp
struct ncclProxyArgs {
  struct ncclProxySubArgs subs[NCCL_PROXY_MAX_SUBS];  // 子操作数组
  proxyProgressFunc_t progress;                        // Progress 函数指针
  int nsubs;                                           // 子操作数量
  int done;                                            // 完成的子操作数
  uint64_t opCount;                                    // 操作计数
  int state;                                           // 操作状态
  int idle;                                            // 是否空闲
  // ... 其他字段
};
```

**结构体**: `ncclProxySubArgs` (src/include/proxy.h:130)

```cpp
struct ncclProxySubArgs {
  struct ncclProxyConnection* connection;
  uint64_t base;          // 起始步骤
  uint64_t posted;        // 已提交步骤
  uint64_t received;      // 已接收步骤
  uint64_t transmitted;   // 已传输步骤
  uint64_t done;          // 已完成步骤
  uint64_t end;           // 结束步骤
  int peer;               // 对端rank
  int channelId;          // 通道ID
  // ... 其他字段
};
```

---

## 通信路径分析

### 1. Network Transport (net.cc)

**支持的协议**: RDMA (IB/RoCE), TCP, GDRDMA

**关键函数**:
- `sendProxyProgress()` - 发送端progress (约1300行)
- `recvProxyProgress()` - 接收端progress (约1400行)

**发送流程**:
```cpp
static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    // 初始化操作
    for (int s=0; s<args->nsubs; s++) {
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  
  // 处理每个子操作
  for (int s=0; s<args->nsubs; s++) {
    // 1. Post buffers to GPU
    if (sub->posted < sub->nsteps && sub->posted < sub->done + maxDepth) {
      // ... 提交缓冲区
    }
    
    // 2. Check GPU data and send to network
    if (sub->transmitted < sub->posted && sub->transmitted < sub->done + NCCL_STEPS) {
      // ... 检查数据就绪，发送到网络
      NCCLCHECK(proxyState->ncclNet->isend(resources->netSendComm, buff, size, ...));
    }
    
    // 3. Check network completion
    if (sub->done < sub->transmitted) {
      NCCLCHECK(proxyState->ncclNet->test(sub->requests[buffSlot], &done, &size));
    }
  }
}
```

### 2. P2P Transport (p2p.cc)

**类型**: P2P_DIRECT, P2P_INTERMEDIATE, P2P_IPC, P2P_CUMEM

**关键函数**:
- `p2pSendProxyProgress()` - 发送端progress (CE memcpy)
- `p2pSendProxyProgress()` 在useMemcpy时启用

### 3. Shared Memory Transport (shm.cc)

**使用场景**: 同一主机上的跨进程通信

**关键函数**:
- `shmSendProxyProgress()`
- `shmRecvProxyProgress()`

### 4. CollNet Transport (coll_net.cc)

**使用场景**: NVLink SHARP 等集合网络

---

## 超时检测实现方案

### 设计方案

```
┌────────────────────────────────────────────────────────────┐
│                    超时检测架构                              │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │            Proxy Progress Thread                      │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │   Args[0]   │  │   Args[1]   │  │   Args[N]   │   │ │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │   │ │
│  │  │ │startTime│ │  │ │startTime│ │  │ │startTime│ │   │ │
│  │  │ │lastActive│ │  │ │lastActive│ │  │ │lastActive│ │   │ │
│  │  │ │ timeout │ │  │ │ timeout │ │  │ │ timeout │ │   │ │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  │           ▲                                           │ │
│  │           │ 周期性检查                                  │ │
│  │           │ (每次progress循环)                          │ │
│  └───────────┼───────────────────────────────────────────┘ │
│              │                                              │
│              ▼                                              │
│  ┌──────────────────────────────────────────────────────┐ │
│  │              超时检测逻辑                               │ │
│  │  1. 检查当前时间 - lastActive > timeout               │ │
│  │  2. 如是，打印超时日志                                 │ │
│  │  3. 可选择：标记失败或继续等待                          │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 超时检测点

| 通信路径 | Progress函数 | 检测位置 |
|---------|-------------|---------|
| Network Send | `sendProxyProgress()` | GPU→Network, Network completion |
| Network Recv | `recvProxyProgress()` | Network→GPU, GPU ack |
| P2P CE memcpy | `p2pSendProxyProgress()` | cudaMemcpy, event completion |
| SHM CE memcpy | `shmSendProxyProgress()` | cudaMemcpy, event completion |
| CollNet | `sendProxyProgress()` / `recvProxyProgress()` | 同 Network |

---

## 代码修改详解

### 第一步：添加超时检测头文件和数据结构

**文件**: `src/include/proxy.h`

在 `ncclProxySubArgs` 结构体中添加超时相关字段：

```cpp
// 在 ncclProxySubArgs 结构体中添加 (约130行后)
struct ncclProxySubArgs {
  // ... 现有字段 ...
  
  // 超时检测字段
  uint64_t startTime;        // 操作开始时间 (微秒)
  uint64_t lastActiveTime;   // 最后活跃时间 (微秒)
  int timeoutDetected;       // 是否已检测到过超时
};
```

在 `ncclProxyArgs` 结构体中添加：

```cpp
struct ncclProxyArgs {
  // ... 现有字段 ...
  
  // 超时检测字段
  uint64_t opStartTime;      // 整个操作的开始时间
  int timeoutLogged;         // 是否已记录超时日志
};
```

### 第二步：添加超时配置环境变量

**文件**: `src/proxy.cc` (在文件头部添加)

```cpp
// 超时检测环境变量 (在现有NCCL_PARAM之后添加)
// NCCL_PROXY_TIMEOUT_MS: 超时时间(毫秒)，0表示禁用
NCCL_PARAM(ProxyTimeoutMs, "PROXY_TIMEOUT_MS", 0);

// NCCL_PROXY_TIMEOUT_LOG_ONLY: 1=仅打印日志，0=超时后报错
NCCL_PARAM(ProxyTimeoutLogOnly, "PROXY_TIMEOUT_LOG_ONLY", 1);

// NCCL_PROXY_TIMEOUT_CHECK_INTERVAL: 检查间隔(毫秒)
NCCL_PARAM(ProxyTimeoutCheckInterval, "PROXY_TIMEOUT_CHECK_INTERVAL", 1000);
```

### 第三步：添加时间获取工具函数

**文件**: `src/include/proxy.h`

```cpp
// 在文件末尾添加时间工具函数声明

// 获取当前时间(微秒)
static inline uint64_t ncclProxyGetTimeUs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

// 超时配置结构体
struct ncclProxyTimeoutConfig {
  int enabled;               // 是否启用
  uint64_t timeoutUs;        // 超时时间(微秒)
  int logOnly;               // 是否仅打印日志
  uint64_t checkIntervalUs;  // 检查间隔(微秒)
};

// 获取超时配置
static inline void ncclProxyGetTimeoutConfig(struct ncclProxyTimeoutConfig* config) {
  config->enabled = ncclParamProxyTimeoutMs() > 0;
  config->timeoutUs = (uint64_t)ncclParamProxyTimeoutMs() * 1000;
  config->logOnly = ncclParamProxyTimeoutLogOnly();
  config->checkIntervalUs = (uint64_t)ncclParamProxyTimeoutCheckInterval() * 1000;
}
```

### 第四步：修改 Proxy 操作初始化

**文件**: `src/proxy.cc`

修改 `ProxyAppend()` 函数，在添加操作时初始化超时检测字段：

```cpp
static ncclResult_t ProxyAppend(struct ncclProxyProgressState* state, struct ncclProxyOp* op) {
  // ... 现有代码 ...
  
  // 在操作初始化时设置超时检测字段
  if (args->state == ncclProxyOpReady) {
    args->opStartTime = ncclProxyGetTimeUs();
    args->timeoutLogged = 0;
    
    // 初始化所有子操作的超时字段
    for (int i = 0; i < args->nsubs; i++) {
      args->subs[i].startTime = args->opStartTime;
      args->subs[i].lastActiveTime = args->opStartTime;
      args->subs[i].timeoutDetected = 0;
    }
  }
  
  // ... 剩余代码 ...
}
```

### 第五步：添加超时检测函数

**文件**: `src/proxy.cc`

添加超时检测核心函数：

```cpp
// 超时检测函数
static void ncclProxyCheckTimeout(struct ncclProxyArgs* args, struct ncclProxySubArgs* sub,
                                   int subIndex, const char* transportName, 
                                   const char* stage) {
  struct ncclProxyTimeoutConfig config;
  ncclProxyGetTimeoutConfig(&config);
  
  if (!config.enabled) return;
  
  uint64_t currentTime = ncclProxyGetTimeUs();
  uint64_t elapsed = currentTime - sub->lastActiveTime;
  
  if (elapsed > config.timeoutUs) {
    // 检测到过超时
    if (!sub->timeoutDetected) {
      sub->timeoutDetected = 1;
      
      // 计算整体操作已运行时间
      uint64_t totalElapsed = currentTime - sub->startTime;
      
      WARN("NCCL Proxy Timeout Detected! "
           "Transport: %s, Stage: %s, "
           "Sub: %d/%d, Peer: %d, Channel: %d, "
           "Progress: posted=%lu transmitted=%lu received=%lu done=%lu nsteps=%lu, "
           "Elapsed: %lu ms (stage), %lu ms (total), "
           "Timeout: %lu ms",
           transportName, stage,
           subIndex, args->nsubs, sub->peer, sub->channelId,
           sub->posted, sub->transmitted, sub->received, sub->done, sub->nsteps,
           elapsed / 1000, totalElapsed / 1000,
           config.timeoutUs / 1000);
    }
    
    // 更新最后活跃时间，避免重复打印
    sub->lastActiveTime = currentTime;
  }
}

// 更新活跃时间
static inline void ncclProxyUpdateActiveTime(struct ncclProxySubArgs* sub) {
  sub->lastActiveTime = ncclProxyGetTimeUs();
  sub->timeoutDetected = 0;  // 重置超时标志
}
```

### 第六步：修改 progressOps 函数添加超时检查

**文件**: `src/proxy.cc`

修改 `progressOps()` 函数：

```cpp
static ncclResult_t progressOps(struct ncclProxyState* proxyState, 
                                struct ncclProxyProgressState* state, 
                                struct ncclProxyArgs* opStart, 
                                int* idle) {
  struct ncclProxyArgs* prevOp = NULL;
  struct ncclProxyArgs* op = opStart;
  ncclResult_t status = ncclSuccess;
  
  // 获取超时配置
  struct ncclProxyTimeoutConfig timeoutConfig;
  ncclProxyGetTimeoutConfig(&timeoutConfig);
  uint64_t currentTime = ncclProxyGetTimeUs();
  static uint64_t lastCheckTime = 0;
  int shouldCheckTimeout = timeoutConfig.enabled && 
                           (currentTime - lastCheckTime > timeoutConfig.checkIntervalUs);
  
  if (shouldCheckTimeout) {
    lastCheckTime = currentTime;
  }
  
  while (op) {
    if (op->state == ncclProxyOpNone) return ncclInternalError;
    
    TIME_START(0); TIME_START(1);
    ncclResult_t ret = op->progress(proxyState, op);
    if (op->idle) { TIME_STOP(1); TIME_CANCEL(0); } 
    else { TIME_CANCEL(1); TIME_STOP(0); }
    
    *idle &= op->idle;
    
    // 超时检测
    if (shouldCheckTimeout && timeoutConfig.enabled && op->state == ncclProxyOpProgress) {
      // 获取transport名称
      const char* transportName = "UNKNOWN";
      if (op->nsubs > 0 && op->subs[0].connection) {
        int transport = op->subs[0].connection->transport;
        switch(transport) {
          case TRANSPORT_P2P: transportName = "P2P"; break;
          case TRANSPORT_SHM: transportName = "SHM"; break;
          case TRANSPORT_NET: transportName = "NET"; break;
          case TRANSPORT_COLLNET: transportName = "COLLNET"; break;
          default: transportName = "UNKNOWN"; break;
        }
      }
      
      // 检查每个子操作
      for (int i = 0; i < op->nsubs; i++) {
        struct ncclProxySubArgs* sub = &op->subs[i];
        
        // 确定当前阶段
        const char* stage = "UNKNOWN";
        if (sub->done == sub->nsteps) {
          continue;  // 已完成，不需要检查
        } else if (sub->posted < sub->nsteps) {
          stage = "POSTING";
        } else if (sub->transmitted < sub->posted) {
          stage = "TRANSMITTING";
        } else if (sub->received < sub->transmitted) {
          stage = "RECEIVING";
        } else if (sub->done < sub->received) {
          stage = "WAITING_ACK";
        }
        
        ncclProxyCheckTimeout(op, sub, i, transportName, stage);
      }
    }
    
    if (op->state == ncclProxyOpNone || ret != ncclSuccess) {
      if (ret != ncclSuccess && status == ncclSuccess) status = ret;
      TIME_START(2);
      NCCLCHECK(removeOp(state, &op, &prevOp));
      TIME_STOP(2);
    } else {
      prevOp = op;
      op = op->next;
    }
  }
  return status;
}
```

### 第七步：修改 Network Transport

**文件**: `src/transport/net.cc`

修改 `sendProxyProgress()` 函数，在关键点更新活跃时间：

```cpp
static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  // ... 初始化代码 ...
  
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int maxDepth = std::min(NCCL_STEPS, NCCL_SHARED_STEPS/args->nsubs);
    
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      // ...
      
      // Post buffers to GPU
      if (sub->posted < sub->nsteps && sub->posted < sub->done + maxDepth) {
        // ... 执行post操作 ...
        ncclProxyUpdateActiveTime(sub);  // 更新活跃时间
        args->idle = 0;
        continue;
      }
      
      // Check GPU data and send
      if (sub->transmitted < sub->posted && sub->transmitted < sub->done + NCCL_STEPS) {
        // ... 执行发送 ...
        if (sub->requests[buffSlot] != NULL) {
          ncclProxyUpdateActiveTime(sub);  // 更新活跃时间
          sub->transmitted += args->sliceSteps;
          args->idle = 0;
          continue;
        }
      }
      
      // Check network completion
      if (sub->done < sub->transmitted) {
        int done;
        NCCLCHECK(proxyState->ncclNet->test(sub->requests[buffSlot], &done, &size));
        if (done) {
          ncclProxyUpdateActiveTime(sub);  // 更新活跃时间
          sub->done += args->sliceSteps;
          args->idle = 0;
          // ...
        }
      }
    }
  }
  return ncclSuccess;
}
```

同样修改 `recvProxyProgress()`：

```cpp
static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  // ... 初始化代码 ...
  
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    // ...
    
    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      // Post receive
      if (subGroup->posted < subGroup->nsteps) {
        // ...
        if (*requestPtr) {
          ncclProxyUpdateActiveTime(subGroup);  // 更新活跃时间
          // ...
        }
      }
      
      // Check receive completion
      if (subGroup->posted > subGroup->received) {
        // ...
        if (done) {
          ncclProxyUpdateActiveTime(subGroup);  // 更新活跃时间
          // ...
        }
      }
      
      // Check flush completion
      if (subGroup->received > subGroup->transmitted) {
        // ...
        if (done) {
          ncclProxyUpdateActiveTime(subGroup);  // 更新活跃时间
          // ...
        }
      }
      
      // Check GPU ack
      if (sub->transmitted > sub->done) {
        // ...
        while (done > sub->base + sub->done) {
          ncclProxyUpdateActiveTime(sub);  // 更新活跃时间
          // ...
        }
      }
    }
  }
  return ncclSuccess;
}
```

### 第八步：修改 P2P Transport (CE memcpy)

**文件**: `src/transport/p2p.cc`

修改 `p2pSendProxyProgress()` 函数：

```cpp
static ncclResult_t p2pSendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  // ... 初始化代码 ...
  
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    // ...
    
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*) (sub->connection->transportResources);
      
      if (sub->transmitted < sub->done + NCCL_STEPS && sub->transmitted < sub->nsteps) {
        // ...
        if ((*recvTail > sub->base+sub->transmitted)) {
          // 执行cudaMemcpy
          CUDACHECK(cudaMemcpyAsync(...));
          CUDACHECK(cudaEventRecord(resources->events[buffSlot], resources->stream));
          ncclProxyUpdateActiveTime(sub);  // 更新活跃时间
          sub->transmitted += args->sliceSteps;
        }
      }
      
      if (sub->done < sub->transmitted) {
        // ...
        if (res == cudaSuccess) {
          ncclProxyUpdateActiveTime(sub);  // 更新活跃时间
          sub->done += args->sliceSteps;
          // ...
        }
      }
    }
    // ...
  }
  return ncclSuccess;
}
```

### 第九步：修改 SHM Transport

**文件**: `src/transport/shm.cc`

类似P2P的修改，在 `shmSendProxyProgress()` 和 `shmRecvProxyProgress()` 中添加 `ncclProxyUpdateActiveTime()` 调用。

### 第十步：修改 CollNet Transport

**文件**: `src/transport/coll_net.cc`

CollNet使用类似的progress模式，需要找到其progress函数并添加相同的超时检测逻辑。

---

## 环境变量配置

### 新增环境变量

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `NCCL_PROXY_TIMEOUT_MS` | 0 | 超时时间（毫秒），0表示禁用超时检测 |
| `NCCL_PROXY_TIMEOUT_LOG_ONLY` | 1 | 1=仅打印日志，0=超时后返回错误 |
| `NCCL_PROXY_TIMEOUT_CHECK_INTERVAL` | 1000 | 超时检查间隔（毫秒） |

### 使用示例

```bash
# 启用30秒超时检测，仅打印日志
export NCCL_PROXY_TIMEOUT_MS=30000
export NCCL_PROXY_TIMEOUT_LOG_ONLY=1

# 运行NCCL程序
./my_nccl_program
```

### 日志输出示例

```
NCCL WARN NCCL Proxy Timeout Detected! Transport: NET, Stage: TRANSMITTING, 
Sub: 0/4, Peer: 1, Channel: 0, 
Progress: posted=4 transmitted=2 received=0 done=0 nsteps=8, 
Elapsed: 30012 ms (stage), 60045 ms (total), Timeout: 30000 ms
```

---

## 测试与验证

### 测试场景

1. **正常通信**: 确保超时检测不影响正常性能
2. **网络延迟**: 模拟网络延迟，验证超时检测触发
3. **对端故障**: 模拟对端进程崩溃，验证超时检测

### 测试脚本

```bash
#!/bin/bash
# test_timeout.sh

# 编译测试程序
mpicc -o test_timeout test_timeout.c -lnccl -lcudart

# 测试1: 禁用超时
echo "Test 1: Timeout disabled"
NCCL_PROXY_TIMEOUT_MS=0 mpirun -np 2 ./test_timeout

# 测试2: 启用5秒超时
echo "Test 2: 5 second timeout"
NCCL_PROXY_TIMEOUT_MS=5000 mpirun -np 2 ./test_timeout

# 测试3: 启用超时并模拟延迟
echo "Test 3: Timeout with simulated delay"
NCCL_PROXY_TIMEOUT_MS=3000 NCCL_DEBUG=INFO mpirun -np 2 ./test_timeout --delay
```

### 性能影响评估

| 场景 | 预期开销 |
|-----|---------|
| 超时禁用 (0) | 无开销 |
| 超时启用 | 每次progress循环一次时间获取和比较 |
| 检查间隔 | 默认1秒检查一次，开销可忽略 |

---

## 总结

### 修改文件列表

1. `src/include/proxy.h` - 添加超时检测数据结构
2. `src/proxy.cc` - 添加超时检测逻辑
3. `src/transport/net.cc` - 更新Network Transport活跃时间
4. `src/transport/p2p.cc` - 更新P2P Transport活跃时间
5. `src/transport/shm.cc` - 更新SHM Transport活跃时间
6. `src/transport/coll_net.cc` - 更新CollNet Transport活跃时间

### 关键设计点

1. **统一入口**: 在 `progressOps()` 中统一进行超时检测
2. **分层检测**: 全局检查间隔 + 每个子操作的独立超时计数
3. **阶段识别**: 根据posted/transmitted/received/done判断当前阶段
4. **可配置**: 通过环境变量控制超时时间和行为
5. **详细日志**: 超时日志包含transport类型、阶段、进度等信息

### 注意事项

1. 超时检测默认禁用（NCCL_PROXY_TIMEOUT_MS=0）
2. 时间获取使用CLOCK_MONOTONIC，不受系统时间调整影响
3. 检查间隔避免过于频繁的系统调用
4. 日志打印频率控制，避免日志风暴
