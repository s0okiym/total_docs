# PyTorch DDP 自动故障治愈开发指南

> 本文档基于 PyTorch ProcessGroupNCCL 源代码分析，专注于 Torch Process Group 封装的 NCCL Communicator 故障自愈 API
> 
> 分析文件：`torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`
> 
> 生成日期：2026-03-27

---

## 目录

1. [概述](#1-概述)
2. [核心故障自愈 API](#2-核心故障自愈-api)
3. [Communicator 生命周期管理](#3-communicator-生命周期管理)
4. [故障检测机制](#4-故障检测机制)
5. [自愈流程与机制](#5-自愈流程与机制)
6. [开发实践指南](#6-开发实践指南)
7. [Q&A](#7-qa)

---

## 1. 概述

### 1.1 PyTorch NCCL 故障自愈架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Application (DDP/FSDP)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ProcessGroupNCCL (Python/C++)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  - split()       - shrink()       - grow()                         │   │
│  │  - abort()       - finalize()     - destroy()                      │   │
│  │  - 故障检测      - Watchdog线程    - Heartbeat                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NCCLComm (C++ Wrapper)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  - ncclCommSplit()   - ncclCommShrink()   - ncclCommGrow()         │   │
│  │  - ncclCommAbort()   - ncclCommFinalize() - ncclCommDestroy()      │   │
│  │  - ncclCommRevoke()  - 异步错误处理                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NCCL Library                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 支持的 NCCL 版本要求

| 功能 | NCCL 版本 | 宏定义 |
|------|----------|--------|
| Comm Split | 2.18.0+ | `NCCL_HAS_COMM_SPLIT` |
| Comm Shrink | 2.27.0+ | `NCCL_HAS_COMM_SHRINK` |
| Comm Grow | 2.27.0+ | `NCCL_HAS_COMM_GROW` |
| Non-blocking | 2.14.0+ | `NCCL_HAS_COMM_NONBLOCKING` |
| Comm Register | 2.19.0+ | `NCCL_HAS_COMM_REGISTER` |

---

## 2. 核心故障自愈 API

### 2.1 Communicator Split（通信器分割）

#### API 定义

```cpp
// ProcessGroupNCCL::split - 分割通信器创建子组
class TORCH_API ProcessGroupNCCL : public Backend {
public:
    // 从当前进程组分出一个子组
    c10::intrusive_ptr<Backend> split(
        const c10::intrusive_ptr<Store>& store,
        const std::vector<int>& ranks,  // 子组包含的ranks
        const c10::intrusive_ptr<Backend::Options>& opts
    );
};
```

#### 使用场景

- **动态子组创建**：从现有进程组创建更小的子组
- **Pipeline Parallelism**：不同阶段使用不同子组
- **MoE 专家分组**：专家并行时分组通信

#### 实现机制

```cpp
c10::intrusive_ptr<Backend> ProcessGroupNCCL::split(
    const c10::intrusive_ptr<Store>& store,
    const std::vector<int>& ranks,
    const c10::intrusive_pointer_cast<Backend::Options>& opts) {
    
    // 1. 检查当前rank是否在子组中
    auto it = std::find(ranks.begin(), ranks.end(), rank_);
    if (it == ranks.end()) {
        // 不在子组中，执行nocolor split
        performNocolorSplit(device);
        return nullptr;
    }
    
    // 2. 计算在子组中的新rank
    int groupRank = std::distance(ranks.begin(), it);
    
    // 3. 设置split参数
    auto ncclOpts = c10::dynamic_intrusive_pointer_cast<Options>(opts);
    ncclOpts->split_from = c10::intrusive_ptr<ProcessGroupNCCL>::unsafe_reclaim_from_nonowning(this);
    ncclOpts->split_color = *std::min_element(ranks.cbegin(), ranks.cend());
    ncclOpts->global_ranks_in_group = globalRanksInGroup;
    
    // 4. 创建新的ProcessGroupNCCL
    auto pg = c10::make_intrusive<ProcessGroupNCCL>(
        store->clone(), groupRank, ranks.size(), ncclOpts);
    
    // 5. eager初始化
    pg->eagerConnectSingleDevice(device);
    return pg;
}
```

#### 底层 NCCL 实现

```cpp
// NCCLComm::split - 底层NCCL通信器分割
std::shared_ptr<NCCLComm> NCCLComm::split(
    NCCLComm* source,
    int color_id,      // 组标识，相同color的rank在同一个新组
    int rank,          // 在新组中的rank
    ncclConfig_t& config) {
    
    auto comm = std::make_shared<NCCLComm>();
    auto sourceComm = source->getNcclComm();
    
    // 调用NCCL API
    C10D_NCCL_CHECK_TIMEOUT_SLEEP(
        ncclCommSplit(sourceComm, color_id, rank, &(comm->ncclComm_), &config),
        source,
        std::nullopt);
    
    // 等待子通信器就绪
    if (color_id >= 0) {
        while (!comm->ncclComm_) {
            C10D_SCHED_SLEEP();
        }
    }
    
    comm->rank_ = rank;
    comm->deviceIndex_ = source->deviceIndex_;
    comm->nonBlocking_ = config.blocking == 0;
    
    return comm;
}
```

### 2.2 Communicator Shrink（通信器收缩）

#### API 定义

```cpp
// ProcessGroupNCCL::shrink - 移除故障rank
class TORCH_API ProcessGroupNCCL : public Backend {
public:
    c10::intrusive_ptr<Backend> shrink(
        const std::vector<int64_t>& ranks_to_exclude,  // 要排除的ranks
        int shrink_flags = 0,                          // shrink标志
        const c10::intrusive_ptr<Backend::Options>& opts_override = nullptr
    );
};
```

#### 使用场景

- **故障rank移除**：训练过程中检测到某些rank故障时移除
- **动态缩容**：根据负载动态减少参与训练的节点
- **弹性训练**：云环境中根据资源可用性调整规模

#### 实现机制

```cpp
c10::intrusive_ptr<Backend> ProcessGroupNCCL::shrink(
    const std::vector<int64_t>& ranks_to_exclude,
    int shrink_flags,
    const c10::intrusive_ptr<Backend::Options>& opts_override) {
    
    // 1. 版本检查 (NCCL 2.27.0+)
    auto runtime_version = torch::cuda::nccl::version();
    TORCH_CHECK(runtime_version >= NCCL_VERSION(2, 27, 0),
        "ProcessGroupNCCL::shrink requires NCCL version 2.27.0 or later");
    
    // 2. 参数验证
    TORCH_CHECK_VALUE(!ranks_to_exclude.empty(), "ranks_to_exclude cannot be empty");
    TORCH_CHECK_VALUE(static_cast<int>(ranks_to_exclude.size()) < size_,
        "Cannot exclude all ranks");
    
    // 3. 获取主通信器
    auto primary_device_index = guessDeviceId();
    auto primary_key = getKeyFromDevice(primary_device);
    std::shared_ptr<NCCLComm> primary_comm = getNCCLComm(primary_key);
    
    // 4. 调用NCCL shrink
    std::shared_ptr<NCCLComm> shrunk_comm = NCCLComm::shrink(
        primary_comm.get(),
        int_ranks_to_exclude,
        (config != nullptr ? config : &options_->config),
        shrink_flags);
    
    // 5. 计算新参数
    int new_size = size_ - static_cast<int>(ranks_to_exclude.size());
    int new_rank = shrunk_comm->rank_;  // NCCL自动分配新rank
    
    // 6. 创建新的ProcessGroupNCCL
    auto new_pg = c10::make_intrusive<ProcessGroupNCCL>(
        new_store, new_rank, new_size, new_opts);
    
    // 7. 初始化设备状态
    new_pg->initializeDeviceStateForComm(primary_device, shrunk_comm);
    
    return new_pg;
}
```

#### 底层 NCCL 实现

```cpp
// NCCLComm::shrink - 底层NCCL通信器收缩
std::shared_ptr<NCCLComm> NCCLComm::shrink(
    NCCLComm* source,
    std::vector<int>& ranks_to_exclude,
    ncclConfig_t* config,
    int shrinkFlags) {
    
    auto comm = std::make_shared<NCCLComm>();
    auto sourceComm = source->getNcclComm();
    
    // 调用NCCL API
    C10D_NCCL_CHECK_NONBLOCKING(
        ncclCommShrink(
            sourceComm,
            ranks_to_exclude.data(),
            ranks_to_exclude.size(),
            reinterpret_cast<ncclComm_t*>(&(comm->ncclComm_)),
            config,
            shrinkFlags),
        source->getNcclCommFailureReason());
    
    // 等待子通信器就绪
    source->waitReady(true);
    comm->initialized_ = true;
    
    // 查询NCCL分配的新rank
    int assigned_rank;
    C10D_NCCL_CHECK(ncclCommUserRank(comm->ncclComm_, &assigned_rank), std::nullopt);
    comm->rank_ = assigned_rank;
    
    comm->deviceIndex_ = source->deviceIndex_;
    comm->nonBlocking_ = config ? config->blocking == 0 : source->nonBlocking_;
    
    return comm;
}
```

#### Shrink Flags

```cpp
// ncclCommShrink flags
#define NCCL_SHRINK_DEFAULT 0x00        // 默认：收缩父通信器
#define NCCL_SHRINK_ABORT 0x01          // 首先终止父通信器上的操作，然后收缩
```

### 2.3 Communicator Abort（通信器中止）

#### API 定义

```cpp
class WorkNCCL : public Work {
public:
    void abort();  // 中止当前work的通信器
};

class NCCLComm {
public:
    void abort(std::optional<std::string> commFailureReason = std::nullopt);
    bool isAborted() const { return aborted_; }
};
```

#### 实现机制

```cpp
void NCCLComm::abort(std::optional<std::string> commFailureReason) {
    LockType lock(mutex_);
    
    // 避免重复abort
    if (aborted_ && !initialized_) {
        return;
    }
    
    // 1. Deregister所有注册的内存段
#ifdef NCCL_HAS_COMM_REGISTER
    for (auto& it : registeredSegmentHandles_) {
        void* handle = it.second;
        C10D_NCCL_CHECK(::ncclCommDeregister(ncclComm_, handle), ...);
    }
    registeredSegmentHandles_.clear();
#endif
    
    // 2. 设置失败原因
    commFailureReason_ = commFailureReason;
    LOG(INFO) << "Aborting ncclComm_ " << ncclComm_ << " with reason: "
              << (commFailureReason ? *commFailureReason : "No abort reason");
    
    // 3. 调用NCCL abort
    C10D_NCCL_CHECK(::ncclCommAbort(ncclComm_), ...);
    
    // 4. 标记状态
    aborted_ = true;
    ncclComm_ = nullptr;
}
```

### 2.4 Communicator Finalize & Destroy

```cpp
// 优雅关闭通信器
void NCCLComm::finalize() {
    LockType lock(mutex_);
    if (aborted_) {
        LOG(INFO) << "NCCL communicator already aborted. Skip finalize.";
        return;
    }
    
    at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex_);
    auto comm = getNcclComm();
    C10D_NCCL_CHECK_NONBLOCKING(ncclCommFinalize(comm), std::nullopt);
}

// 销毁通信器
void NCCLComm::destroy() {
    LockType lock(mutex_);
    if (aborted_) {
        LOG(INFO) << "NCCL communicator already aborted. Skip destroy.";
        return;
    }
    
    at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex_);
    auto comm = getNcclComm();
    C10D_NCCL_CHECK(ncclCommDestroy(comm), std::nullopt);
    aborted_ = true;  // Poison future getNcclComm
}
```

---

## 3. Communicator 生命周期管理

### 3.1 状态机

```
┌─────────────┐     create()      ┌─────────────┐
│   Initial   │ ─────────────────►│   Created   │
└─────────────┘                   └──────┬──────┘
                                         │
                    getNcclComm()        │
                    waitReady()          ▼
                              ┌─────────────────────┐
                              │    Initialized      │
                              │  (ready for ops)    │
                              └──────────┬──────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
        ▼                                ▼                                ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│    Split      │              │    Shrink     │              │    Abort      │
│  (子通信器)    │              │  (收缩通信器)  │              │  (紧急中止)    │
└───────┬───────┘              └───────┬───────┘              └───────┬───────┘
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│ Child Comm    │              │ Shrunken Comm │              │   Aborted     │
│   Created     │              │   Created     │              │    State      │
└───────────────┘              └───────────────┘              └───────────────┘
                                                                         │
                                    finalize()/destroy()                  │
                                         │                               │
                                         ▼                               ▼
                              ┌─────────────────────┐         ┌─────────────────────┐
                              │     Finalized       │         │     Destroyed       │
                              │    / Destroyed      │         │   (资源已释放)       │
                              └─────────────────────┘         └─────────────────────┘
```

### 3.2 Options 配置

```cpp
struct Options : Backend::Options {
    // 父通信器（用于split）
    c10::intrusive_ptr<ProcessGroupNCCL> split_from;
    
    // Split color标识
    int split_color{NCCL_SPLIT_NOCOLOR - 1};
    
    // 全局rank映射
    std::vector<uint64_t> global_ranks_in_group;
    
    // NCCL配置
    ncclConfig_t config;
    
    // 高优先级stream
    bool is_high_priority_stream{false};
    
    // 创建Options的工厂方法
    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false) {
        auto opts = c10::make_intrusive<Options>();
        opts->is_high_priority_stream = is_high_priority_stream;
        return opts;
    }
};
```

---

## 4. 故障检测机制

### 4.1 Watchdog 线程

```cpp
// Watchdog线程主循环
void ProcessGroupNCCL::watchdogLoop() {
    while (!terminateProcessGroup_.load()) {
        // 1. 检查work队列
        std::unique_lock<std::mutex> lock(workMetaListMutex_);
        
        for (auto& work : workMetaList_) {
            // 2. 检查是否完成
            if (work->isCompleted()) {
                handleCompletedWork(work);
                continue;
            }
            
            // 3. 检查超时
            if (work->checkTimeout()) {
                handleTimeout(work);
                continue;
            }
            
            // 4. 检查NCCL错误
            auto exception_ptr = work->checkForNCCLErrors();
            if (exception_ptr) {
                work->setException(exception_ptr);
                handleException(work);
            }
        }
        
        // 5. 休眠等待
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kWatchdogThreadSleepMillis));
    }
}
```

### 4.2 异步错误检测

```cpp
std::exception_ptr WorkNCCL::checkForNCCLErrors() {
    // 获取NCCL异步错误
    ncclResult_t ncclAsyncErr = ncclSuccess;
    ncclComm_->getAsyncError(&ncclAsyncErr);
    
    if (ncclAsyncErr != ncclSuccess) {
        // 构建错误信息
        auto exceptionMsg = c10::str(
            "NCCL error: ", ncclGetErrorWithVersion(ncclAsyncErr),
            "\n", getNcclErrorDetailStr(ncclAsyncErr, failureReason));
        
        return std::make_exception_ptr(
            C10_BUILD_ERROR(DistBackendError, exceptionMsg));
    }
    
    return nullptr;
}
```

### 4.3 Heartbeat 监控

```cpp
// Heartbeat机制
void ProcessGroupNCCL::heartbeatMonitor() {
    auto lastTime = std::chrono::steady_clock::now();
    
    while (!terminateProcessGroup_.load()) {
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            currentTime - lastTime).count();
        
        // 检查watchdog是否卡住
        if (elapsed > heartbeatTimeoutSec_) {
            LOG(ERROR) << "Watchdog heartbeat timeout! Aborting process.";
            // 触发dump
            dumpDebuggingInfo();
            // 终止进程
            std::terminate();
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
```

### 4.4 错误传播

```cpp
// 通过Store传播错误
void ProcessGroupNCCL::broadcastErrorToRanks(const std::string& errorMsg) {
    if (globalStore_) {
        globalStore_->set(kStoreErrorSignalKey, errorMsg);
    }
}

// 检查远程错误
bool ProcessGroupNCCL::checkForRemoteErrors() {
    try {
        auto remoteError = globalStore_->get(kStoreErrorSignalKey);
        if (!remoteError.empty()) {
            LOG(ERROR) << "Remote error detected: " << remoteError;
            return true;
        }
    } catch (...) {
        // Store may not be available
    }
    return false;
}
```

---

## 5. 自愈流程与机制

### 5.1 故障自愈完整流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           故障检测阶段                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│  Watchdog检测  │           │  异步错误检测  │           │  Heartbeat超时 │
│  (超时/异常)   │           │  (NCCL错误)   │           │  (线程卡住)   │
└───────┬───────┘           └───────┬───────┘           └───────┬───────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           故障定位阶段                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. 确定故障rank列表                                                         │
│  2. 收集Flight Recorder日志                                                  │
│  3. 分析NCCL comm状态                                                        │
│  4. 判断故障类型 (网络/GPU/软件)                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           自愈决策阶段                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
    │   可恢复故障   │      │   需缩容恢复   │      │   不可恢复    │
    │ (临时网络抖动) │      │ (rank永久失效) │      │ (系统性故障)  │
    └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
            │                       │                       │
            ▼                       ▼                       ▼
    ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
    │ 重试/回滚操作  │      │ 调用shrink()  │      │ 保存checkpoint│
    │               │      │ 创建新通信器   │      │ 终止训练      │
    └───────────────┘      └───────┬───────┘      └───────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           恢复执行阶段                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. 使用shrink()创建新的ProcessGroupNCCL                                    │
│  2. 更新DDP/FSDP的process_group引用                                          │
│  3. 重新初始化optimizer状态 (可选)                                           │
│  4. 从checkpoint恢复 (如果需要)                                              │
│  5. 继续训练                                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Shrink 恢复示例代码

```cpp
// 故障自愈Manager类
class DDPFaultToleranceManager {
public:
    // 检测并处理故障
    void handleFailure(
        c10::intrusive_ptr<ProcessGroupNCCL> old_pg,
        const std::vector<int64_t>& failed_ranks) {
        
        LOG(INFO) << "Detected failed ranks: " << failed_ranks;
        
        // 1. 中止旧通信器
        abortProcessGroup(old_pg);
        
        // 2. 创建新的shrink后的通信器
        auto new_pg = createShrunkenProcessGroup(old_pg, failed_ranks);
        
        // 3. 更新DDP的process group
        updateDDPProcessGroup(new_pg);
        
        // 4. 调整学习率 (因为world_size变化)
        adjustLearningRate(new_pg->getSize());
        
        // 5. 恢复训练
        resumeTraining();
    }
    
private:
    c10::intrusive_ptr<ProcessGroupNCCL> createShrunkenProcessGroup(
        c10::intrusive_ptr<ProcessGroupNCCL> old_pg,
        const std::vector<int64_t>& failed_ranks) {
        
        // 使用shrink创建新通信器
        auto opts = ProcessGroupNCCL::Options::create();
        opts->timeout = old_pg->getOptions()->timeout;
        
        auto new_pg = old_pg->shrink(
            failed_ranks,           // 要排除的ranks
            NCCL_SHRINK_DEFAULT,    // shrink标志
            opts                    // 选项
        );
        
        return c10::dynamic_intrusive_pointer_cast<ProcessGroupNCCL>(new_pg);
    }
    
    void abortProcessGroup(c10::intrusive_ptr<ProcessGroupNCCL> pg) {
        // 中止所有work
        pg->abort();
    }
    
    void updateDDPProcessGroup(
        c10::intrusive_ptr<ProcessGroupNCCL> new_pg) {
        // 更新DDP module的process_group
        // 注意：这需要DDP支持动态更换process group
        for (auto& ddp_module : ddp_modules_) {
            ddp_module->update_process_group(new_pg);
        }
    }
    
    void adjustLearningRate(int new_world_size) {
        // 线性缩放学习率
        float scale = static_cast<float>(new_world_size) / original_world_size_;
        for (auto& param_group : optimizer_->param_groups()) {
            param_group.options().set_lr(param_group.options().lr() * scale);
        }
    }
};
```

### 5.3 Python API 封装

```python
# torch/distributed/fault_tolerance.py

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Optional

class FaultTolerantDDP:
    """支持故障自愈的DDP包装器"""
    
    def __init__(self, module, device_ids=None, **kwargs):
        self.module = module
        self.device_ids = device_ids
        self.ddp_kwargs = kwargs
        
        # 创建初始DDP
        self.ddp_module = DDP(module, device_ids=device_ids, **kwargs)
        self.process_group = self.ddp_module.process_group
        
        # 故障检测配置
        self.enable_fault_tolerance = kwargs.get('enable_fault_tolerance', True)
        self.heartbeat_timeout = kwargs.get('heartbeat_timeout', 60)
        
    def detect_failed_ranks(self) -> List[int]:
        """检测故障rank列表"""
        failed_ranks = []
        
        # 使用heartbeat检测
        for rank in range(self.process_group.size()):
            if rank == dist.get_rank():
                continue
            if not self._check_rank_healthy(rank):
                failed_ranks.append(rank)
        
        return failed_ranks
    
    def recover_from_failure(self, failed_ranks: List[int]):
        """从故障中恢复"""
        if not failed_ranks:
            return
        
        print(f"Recovering from failure. Failed ranks: {failed_ranks}")
        
        # 1. 中止当前process group
        self.process_group.abort()
        
        # 2. 使用shrink创建新process group
        new_pg = self.process_group.shrink(
            ranks_to_exclude=failed_ranks,
            shrink_flags=0
        )
        
        # 3. 更新DDP的process group
        self.ddp_module.process_group = new_pg
        self.process_group = new_pg
        
        # 4. 重新初始化必要的组件
        self._reinitialize_reducers()
        
        print(f"Recovery complete. New world size: {new_pg.size()}")
    
    def _check_rank_healthy(self, rank: int) -> bool:
        """检查rank是否健康"""
        try:
            # 发送heartbeat
            dist.send(torch.tensor([1]), dst=rank)
            return True
        except Exception:
            return False
    
    def _reinitialize_reducers(self):
        """重新初始化DDP的reducers"""
        # 调用DDP内部方法重新初始化
        self.ddp_module._init_streams()
        self.ddp_module._init_reducers()


def create_fault_tolerant_process_group(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    **kwargs
):
    """
    创建支持故障自愈的process group
    
    Args:
        backend: 后端类型 (nccl/gloo)
        init_method: 初始化方法
        timeout: 超时时间
        **kwargs: 其他参数
    
    Returns:
        ProcessGroup: 配置的process group
    """
    # 设置故障检测环境变量
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '3'  # SkipCleanUp
    os.environ['TORCH_NCCL_ENABLE_MONITORING'] = '1'
    os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '120'
    os.environ['TORCH_NCCL_TRACE_BUFFER_SIZE'] = '2000'
    
    # 创建process group
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        timeout=timeout,
        **kwargs
    )
    
    return dist.group.WORLD
```

---

## 6. 开发实践指南

### 6.1 环境变量配置

```bash
# 启用异步错误处理
export TORCH_NCCL_ASYNC_ERROR_HANDLING=3  # 0=NoHandling, 1=TearDown, 2=CleanUpOnly, 3=SkipCleanUp

# 启用监控线程
export TORCH_NCCL_ENABLE_MONITORING=1

# 设置heartbeat超时（秒）
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120

# 启用Flight Recorder
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

# 超时dump
export TORCH_NCCL_DUMP_ON_TIMEOUT=1

# 错误传播
export TORCH_NCCL_PROPAGATE_ERROR=1

# 启用Timing
export TORCH_NCCL_ENABLE_TIMING=1
```

### 6.2 故障自愈最佳实践

```python
# 1. 使用try-except捕获并处理故障
import torch.distributed as dist
from torch.distributed.fault_tolerance import FaultTolerantDDP

def train_with_fault_tolerance():
    # 初始化
    dist.init_process_group("nccl")
    
    # 创建故障容错DDP
    model = MyModel()
    ft_ddp = FaultTolerantDDP(model)
    
    max_retries = 3
    for epoch in range(num_epochs):
        try:
            for batch in dataloader:
                loss = ft_ddp(batch)
                loss.backward()
                optimizer.step()
                
        except dist.DistBackendError as e:
            # 检测到故障
            print(f"Detected failure: {e}")
            
            # 检测故障ranks
            failed_ranks = ft_ddp.detect_failed_ranks()
            
            if failed_ranks and max_retries > 0:
                # 尝试恢复
                ft_ddp.recover_from_failure(failed_ranks)
                max_retries -= 1
                continue
            else:
                # 无法恢复，抛出异常
                raise


# 2. Checkpoint保存与恢复
def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'world_size': dist.get_world_size(),
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


# 3. 动态调整batch size
class AdaptiveBatchSize:
    """根据world size动态调整batch size"""
    
    def __init__(self, base_batch_size, base_world_size):
        self.base_batch_size = base_batch_size
        self.base_world_size = base_world_size
    
    def get_current_batch_size(self):
        current_world_size = dist.get_world_size()
        # 保持global batch size恒定
        return self.base_batch_size * self.base_world_size // current_world_size
```

### 6.3 监控与诊断

```python
# 使用Flight Recorder诊断故障
from torch.distributed import dump_nccl_trace

# 在故障时dump trace
def on_failure():
    trace = dump_nccl_trace(
        includeCollectives=True,
        includeStackTraces=True,
        onlyActive=False
    )
    with open('nccl_trace.json', 'w') as f:
        f.write(trace)


# 自定义Profiler
class NCCLProfiler:
    def __init__(self):
        self.events = []
    
    def record_start(self, op_name):
        self.events.append({
            'op': op_name,
            'start': time.time(),
            'rank': dist.get_rank()
        })
    
    def record_end(self, op_name):
        for event in reversed(self.events):
            if event['op'] == op_name and 'end' not in event:
                event['end'] = time.time()
                event['duration'] = event['end'] - event['start']
                break
```

---

## 7. Q&A

### Q1: shrink() 和 split() 的主要区别是什么？

**A:** 
- **split()**: 基于color将通信器分割成多个不相交的子组，原通信器保持不变。用于创建独立工作的子组。
- **shrink()**: 从通信器中移除指定rank，创建一个新的更小的通信器。原通信器应该被中止。用于故障恢复。

### Q2: 如何判断一个rank是否故障？

**A:** 多维度检测：
1. **Watchdog超时**: 检查work是否在规定时间内完成
2. **NCCL异步错误**: 调用`ncclCommGetAsyncError()`检查
3. **Heartbeat**: 定期检测rank响应
4. **Store错误传播**: 通过TCPStore传播错误信息

### Q3: shrink后的新rank如何确定？

**A:** NCCL自动分配新rank，可以通过`ncclCommUserRank()`查询。通常新rank会重新从0开始编号，保持连续。

### Q4: 故障自愈后学习率如何调整？

**A:** 推荐线性缩放：
```python
new_lr = old_lr * (new_world_size / old_world_size)
```
保持global batch size不变，确保收敛行为一致。

### Q5: shrink操作对性能有什么影响？

**A:** 
- **初始化开销**: shrink需要创建新的NCCL通信器，有初始化开销
- **带宽变化**: 减少rank数可能改变集合通信的带宽特性
- **拓扑变化**: 可能需要重新选择算法（如从Ring切换到Tree）

### Q6: 是否可以多次shrink？

**A:** 可以。可以基于已经shrink的通信器继续shrink，但每次都需要创建新的ProcessGroupNCCL实例。

### Q7: 如何处理shrink过程中的新故障？

**A:** 建议：
1. 设置shrink超时
2. 如果shrink失败，回退到checkpoint恢复
3. 考虑使用多级容错策略

### Q8: abort和destroy的区别？

**A:** 
- **abort()**: 立即中止通信器，不等待未完成操作，用于紧急故障处理
- **destroy()**: 优雅关闭，等待操作完成，正常退出时使用

### Q9: Flight Recorder如何帮助故障诊断？

**A:** Flight Recorder记录：
- 每个collective的开始和结束事件
- 调用栈信息
- 通信器状态
- 可用于分析hang的原因和位置

### Q10: NCCL版本要求是什么？

**A:** 
- **split**: NCCL 2.18.0+
- **shrink**: NCCL 2.27.0+
- **grow**: NCCL 2.27.0+

### Q11: shrink_flags有哪些选项？

**A:** 
- `NCCL_SHRINK_DEFAULT (0x00)`: 默认行为
- `NCCL_SHRINK_ABORT (0x01)`: 先中止父通信器操作再shrink

### Q12: 如何测试故障自愈功能？

**A:** 测试方法：
```python
# 模拟kill一个rank
import os
import signal

if dist.get_rank() == target_rank:
    os.kill(os.getpid(), signal.SIGKILL)
```

### Q13: shrink后checkpoint如何处理？

**A:** 
1. 保存checkpoint时记录world_size
2. 恢复时检查world_size是否匹配
3. 如果不匹配，调整模型状态（如BatchNorm统计量）

### Q14: 是否所有NCCL操作都支持故障自愈？

**A:** 是的，shrink后的新通信器完全支持所有NCCL操作（AllReduce, AllGather, Send/Recv等）。

### Q15: 如何处理多节点环境下的网络分区？

**A:** 
1. 检测网络分区（部分rank不可达）
2. 确定最大可达子集
3. 使用shrink创建新通信器
4. 通知其他子集停止或合并

### Q16: shrink对梯度同步有什么影响？

**A:** 
- 被移除rank的梯度不再参与同步
- 需要确保被移除rank的数据在其他rank上被正确处理
- 可能需要重新平衡数据加载

### Q17: 如何在PyTorch Lightning中使用故障自愈？

**A:** 
```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

class FaultToleranceCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 定期检查故障
        if self.should_check_failure():
            failed_ranks = self.detect_failed_ranks()
            if failed_ranks:
                self.recover(trainer, failed_ranks)

trainer = Trainer(callbacks=[FaultToleranceCallback()])
```

### Q18: 故障自愈的监控指标有哪些？

**A:** 
- **MTTR** (Mean Time To Recovery): 平均恢复时间
- **故障频率**: 单位时间内故障次数
- **恢复成功率**: 成功恢复次数/总故障次数
- **性能损失**: 恢复后的训练吞吐量损失

### Q19: 是否支持自动扩缩容（shrink + grow）？

**A:** NCCL支持shrink，grow API也存在但需要协调：
1. 新rank需要加入
2. 需要全局一致的新通信器创建
3. 需要重新分配数据

### Q20: 生产环境中使用故障自愈的最佳实践？

**A:** 
1. **配置监控**: 启用TORCH_NCCL_ENABLE_MONITORING
2. **定期Checkpoint**: 配置合理的checkpoint间隔
3. **测试恢复流程**: 定期演练故障恢复
4. **日志收集**: 收集Flight Recorder日志用于分析
5. **优雅降级**: 准备好shrink后继续训练的策略
6. **资源预留**: 预留一定计算资源用于故障恢复

---

## 附录

### A. 参考代码位置

| 文件 | 功能 |
|------|------|
| `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp` | ProcessGroupNCCL实现 |
| `torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp` | 头文件定义 |
| `torch/csrc/distributed/c10d/NCCLUtils.cpp` | NCCLComm工具类 |
| `torch/csrc/distributed/c10d/NCCLUtils.hpp` | 宏定义和工具函数 |

### B. 相关NCCL API

```c
// Split
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, 
                           ncclComm_t* newcomm, ncclConfig_t* config);

// Shrink
ncclResult_t ncclCommShrink(ncclComm_t comm, int* excludeRanksList, 
                            int excludeRanksCount, ncclComm_t* newcomm,
                            ncclConfig_t* config, int shrinkFlags);

// Grow
ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, 
                          const ncclUniqueId* uniqueId, int rank,
                          ncclComm_t* newcomm, ncclConfig_t* config);

// Abort
ncclResult_t ncclCommAbort(ncclComm_t comm);

// Get Async Error
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);
```

### C. 故障自愈开发Checklist

- [ ] 确认NCCL版本 >= 2.27.0
- [ ] 启用异步错误处理
- [ ] 配置Heartbeat监控
- [ ] 实现故障检测逻辑
- [ ] 实现shrink恢复流程
- [ ] 配置checkpoint保存
- [ ] 测试故障恢复流程
- [ ] 配置监控和告警
- [ ] 编写运维文档

---

*文档结束*
