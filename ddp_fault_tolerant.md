# PyTorch ProcessGroupNCCL Fault Tolerance API Documentation

## Overview

PyTorch's `ProcessGroupNCCL` provides a comprehensive set of APIs for building fault-tolerant distributed training systems. This document focuses on the **communicator splitting**, **shrinking**, and **fault recovery mechanisms** that enable automatic fault healing in DDP (Distributed Data Parallel) training.

---

## Table of Contents

1. [NCCL Communicator Lifecycle Management](#1-nccl-communicator-lifecycle-management)
2. [Communicator Split API (ncclCommSplit)](#2-communicator-split-api-ncclcommsplit)
3. [Communicator Shrink API (ncclCommShrink)](#3-communicator-shrink-api-ncclcommshrink)
4. [Error Handling and Fault Detection](#4-error-handling-and-fault-detection)
5. [Watchdog and Heartbeat Monitoring](#5-watchdog-and-heartbeat-monitoring)
6. [Practical Fault Recovery Patterns](#6-practical-fault-recovery-patterns)
7. [Environment Variables for Fault Tolerance](#7-environment-variables-for-fault-tolerance)

---

## 1. NCCL Communicator Lifecycle Management

### 1.1 Core NCCL Communicator Operations

```cpp
// NCCLUtils.hpp - NCCLComm class
class NCCLComm {
public:
    // Create a new NCCL communicator
    static std::shared_ptr<NCCLComm> create(
        int numRanks, int rank, ncclUniqueId commId, 
        at::DeviceIndex deviceIndex, ncclConfig_t& config);
    
    // Split from parent communicator
    static std::shared_ptr<NCCLComm> split(
        NCCLComm* source, int color_id, int rank, ncclConfig_t& config);
    
    // Shrink communicator (exclude failed ranks)
    static std::shared_ptr<NCCLComm> shrink(
        NCCLComm* source, std::vector<int>& ranks_to_exclude,
        ncclConfig_t* config, int shrinkFlags = 0);
    
    // Lifecycle operations
    void abort(std::optional<std::string> commFailureReason = std::nullopt);
    void finalize();  // Graceful shutdown
    void destroy();   // Resource cleanup
    
    // Status checking
    bool isAborted() const;
    bool isInitialized() const;
    ncclResult_t checkForNcclError();
};
```

### 1.2 ProcessGroupNCCL Lifecycle Methods

```cpp
class ProcessGroupNCCL : public Backend {
public:
    // Graceful shutdown - wait for operations to complete
    void shutdown() override;
    
    // Immediate abort - terminate all NCCL operations
    void abort() override;
    
    // Check if communicator is fully initialized
    bool isInitialized();
    
    // Get current error state
    ErrorType getError() override;
    
    // Eager initialization for single device
    void eagerConnectSingleDevice(at::Device device) override;
};
```

**Key Differences:**
- `shutdown()`: Waits for pending operations to complete, then destroys communicators
- `abort()`: Immediately terminates NCCL kernels and aborts communicators

---

## 2. Communicator Split API (ncclCommSplit)

### 2.1 Overview

The `split` API enables creating sub-groups from an existing ProcessGroupNCCL. This is essential for:
- Creating hierarchical communication patterns
- Isolating failed ranks into separate groups
- Dynamic group reconfiguration

**NCCL Version Required:** 2.18.0+

### 2.2 ProcessGroupNCCL::split Implementation

```cpp
c10::intrusive_ptr<Backend> ProcessGroupNCCL::split(
    const c10::intrusive_ptr<Store>& store,
    const std::vector<int>& ranks,           // Ranks to include in new group
    const c10::intrusive_ptr<Backend::Options>& opts) {
    
    // Determine if this rank participates in the split
    auto it = std::find(ranks.begin(), ranks.end(), rank_);
    if (it == ranks.end()) {
        // This rank is NOT in the new group - perform nocolor split
        performNocolorSplit(device);
        return nullptr;
    }
    
    // Calculate new rank within subgroup
    int groupRank = std::distance(ranks.begin(), it);
    
    // Set up split configuration
    auto ncclOpts = c10::dynamic_intrusive_pointer_cast<Options>(opts);
    ncclOpts->split_from = c10::intrusive_ptr<ProcessGroupNCCL>::unsafe_reclaim_from_nonowning(this);
    ncclOpts->split_color = *std::min_element(ranks.cbegin(), ranks.cend());
    
    // Create new ProcessGroupNCCL with split configuration
    auto pg = c10::make_intrusive<ProcessGroupNCCL>(
        store->clone(), groupRank, ranks.size(), ncclOpts);
    
    // Eagerly initialize the communicator
    pg->eagerConnectSingleDevice(device);
    return pg;
}
```

### 2.3 NCCLComm::split Implementation

```cpp
std::shared_ptr<NCCLComm> NCCLComm::split(
    NCCLComm* source,
    int color_id,       // Non-negative: in group; NCCL_SPLIT_NOCOLOR (-1): excluded
    int rank,           // Rank in new communicator
    ncclConfig_t& config) {
    
    auto comm = std::make_shared<NCCLComm>();
    auto sourceComm = source->getNcclComm();  // Blocks until parent ready
    
    // Perform the split
    C10D_NCCL_CHECK_TIMEOUT_SLEEP(
        ncclCommSplit(sourceComm, color_id, rank, &(comm->ncclComm_), &config),
        source, std::nullopt);
    
    ++source->ncclCommSplitCounter_;
    comm->rank_ = rank;
    comm->deviceIndex_ = source->deviceIndex_;
    comm->nonBlocking_ = config.blocking == 0;
    
    return comm;
}
```

### 2.4 Python API Usage

```python
import torch.distributed as dist

# Create parent process group
dist.init_process_group("nccl", ...)
parent_pg = dist.group.WORLD

# Split into subgroups (e.g., for pipeline parallelism)
ranks_group_a = [0, 1, 2, 3]
ranks_group_b = [4, 5, 6, 7]

# On ranks 0-3, returns new PG; on ranks 4-7, returns None
sub_pg = parent_pg.split(ranks_group_a)

# Use NCCL_SPLIT_NOCOLOR for ranks not participating
# This is handled automatically by performNocolorSplit()
```

### 2.5 Split Configuration Options

```cpp
struct Options : Backend::Options {
    // Parent communicator for splitting
    c10::intrusive_ptr<ProcessGroupNCCL> split_from;
    
    // Color assignment:
    // - Non-negative: in group, same color = same sub-group
    // - NCCL_SPLIT_NOCOLOR (-1): excluded from new communicator
    int split_color{NCCL_SPLIT_NOCOLOR - 1};
    
    // NCCL configuration
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
};
```

---

## 3. Communicator Shrink API (ncclCommShrink)

### 3.1 Overview

The `shrink` API enables removing failed ranks from an existing communicator without recreating the entire ProcessGroup. This is critical for fault-tolerant training where individual ranks may fail.

**NCCL Version Required:** 2.27.0+

### 3.2 ProcessGroupNCCL::shrink Implementation

```cpp
c10::intrusive_ptr<Backend> ProcessGroupNCCL::shrink(
    const std::vector<int64_t>& ranks_to_exclude,  // Ranks to remove
    int shrink_flags = 0,                          // NCCL shrink flags
    const c10::intrusive_ptr<Backend::Options>& opts_override = nullptr) {
    
    // Version check
    auto runtime_version = torch::cuda::nccl::version();
    TORCH_CHECK(runtime_version >= NCCL_VERSION(2, 27, 0),
                "shrink requires NCCL 2.27.0+");
    
    // Validation
    TORCH_CHECK_VALUE(!ranks_to_exclude.empty(), 
                      "ranks_to_exclude cannot be empty");
    TORCH_CHECK_VALUE(static_cast<int>(ranks_to_exclude.size()) < size_,
                      "Cannot exclude all ranks");
    
    // Get primary communicator
    auto primary_device_index = guessDeviceId();
    auto primary_key = getKeyFromDevice(at::Device(at::kCUDA, primary_device_index));
    std::shared_ptr<NCCLComm> primary_comm = getNCCLComm(primary_key);
    
    // Convert ranks to exclude
    std::vector<int> int_ranks_to_exclude;
    for (int64_t rank : ranks_to_exclude) {
        TORCH_CHECK_VALUE(rank >= 0 && rank < size_, "Invalid rank");
        int_ranks_to_exclude.push_back(static_cast<int>(rank));
    }
    
    // Perform shrink operation
    ncclConfig_t* config = opts_override ? 
        &c10::static_intrusive_pointer_cast<Options>(opts_override)->config : 
        &options_->config;
    
    std::shared_ptr<NCCLComm> shrunk_comm = NCCLComm::shrink(
        primary_comm.get(), int_ranks_to_exclude, config, shrink_flags);
    
    // Calculate new dimensions
    int new_size = size_ - static_cast<int>(ranks_to_exclude.size());
    int new_rank = shrunk_comm->rank_;  // NCCL assigns new rank
    
    // Create new ProcessGroupNCCL
    auto new_store = store_->clone();
    auto new_opts = ProcessGroupNCCL::Options::create(options_->is_high_priority_stream);
    new_opts->timeout = options_->timeout;
    new_opts->config = *config;
    
    auto new_pg = c10::make_intrusive<ProcessGroupNCCL>(
        new_store, new_rank, new_size, new_opts);
    
    // Initialize with shrunk communicator
    new_pg->initializeDeviceStateForComm(
        at::Device(at::kCUDA, primary_device_index), shrunk_comm);
    
    return new_pg;
}
```

### 3.3 NCCLComm::shrink Implementation

```cpp
std::shared_ptr<NCCLComm> NCCLComm::shrink(
    NCCLComm* source,
    std::vector<int>& ranks_to_exclude,
    ncclConfig_t* config,
    int shrinkFlags) {
    
    at::cuda::OptionalCUDAGuard gpuGuard(source->deviceIndex_);
    auto comm = std::make_shared<NCCLComm>();
    
    // Ensure source communicator is ready
    auto sourceComm = source->getNcclComm();
    
    // Call NCCL shrink API
    C10D_NCCL_CHECK_NONBLOCKING(
        ncclCommShrink(sourceComm,
                       ranks_to_exclude.data(),
                       ranks_to_exclude.size(),
                       reinterpret_cast<ncclComm_t*>(&(comm->ncclComm_)),
                       config,
                       shrinkFlags),
        source->getNcclCommFailureReason());
    
    // Wait for child communicator to be ready
    source->waitReady(true);
    comm->initialized_ = true;
    
    // Query NCCL-assigned rank
    int assigned_rank;
    C10D_NCCL_CHECK(ncclCommUserRank(comm->ncclComm_, &assigned_rank), std::nullopt);
    comm->rank_ = assigned_rank;
    
    // Inherit device and blocking mode
    comm->deviceIndex_ = source->deviceIndex_;
    comm->nonBlocking_ = config ? config->blocking == 0 : source->nonBlocking_;
    
    return comm;
}
```

### 3.4 Python API Usage

```python
import torch.distributed as dist

# Initialize process group
dist.init_process_group("nccl", ...)
pg = dist.group.WORLD

# Detect failed ranks (e.g., via heartbeat timeout)
failed_ranks = detect_failed_ranks()  # e.g., [2, 5]

# Shrink the process group to exclude failed ranks
if pg.supports_shrinking():
    new_pg = pg.shrink(ranks_to_exclude=failed_ranks)
    
    # Update references
    pg = new_pg
    
    # Continue training with reduced world size
    # New rank assignment is handled by NCCL
```

### 3.5 supportsShrinking Check

```cpp
bool ProcessGroupNCCL::supportsShrinking() const override {
#ifdef NCCL_HAS_COMM_SHRINK
    return true;
#else
    return false;
#endif
}
```

---

## 4. Error Handling and Fault Detection

### 4.1 Error Handling Modes

```cpp
enum ErrorHandlingMode {
    NoHandling = 0,      // Do not handle asynchronous NCCL errors
    TearDown = 1,        // Tear down process upon error
    CleanUpOnly = 2,     // Clean up collectives and abort communicators
    SkipCleanUp = 3      // Tear down without cleaning up (last resort)
};

#define SHOULD_CLEAN_UP(a) (a != NoHandling && a != SkipCleanUp)
#define SHOULD_TEAR_DOWN(a) (a != NoHandling && a != CleanUpOnly)
```

### 4.2 Error Detection Mechanisms

```cpp
// Check for NCCL errors on communicator
std::exception_ptr NCCLComm::checkForNcclError() {
    LockType lock(mutex_);
    if (ncclAsyncErr_ != ncclSuccess) {
        return ncclAsyncErr_;
    }
    // Query async error from NCCL
    C10D_NCCL_CHECK(ncclCommGetAsyncError(ncclComm_, &ncclAsyncErr_), 
                    commFailureReason_);
    return ncclAsyncErr_;
}

// ProcessGroupNCCL error checking
std::exception_ptr ProcessGroupNCCL::checkForNCCLErrorsInternal(
    std::shared_ptr<NCCLComm>& ncclComm) {
    
    // Check for failure reason set by ProcessGroupNCCL
    auto commFailureReason = ncclComm->getNcclCommFailureReason();
    if (commFailureReason != std::nullopt) {
        return std::make_exception_ptr(C10_BUILD_ERROR(
            DistBackendError,
            c10::str("NCCL communicator encountered error: ", 
                     *commFailureReason)));
    }
    
    // Check NCCL async errors
    ncclResult_t ncclAsyncErr = ncclComm->checkForNcclError();
#ifdef NCCL_HAS_COMM_NONBLOCKING
    if (ncclAsyncErr != ncclSuccess && ncclAsyncErr != ncclInProgress) {
#else
    if (ncclAsyncErr != ncclSuccess) {
#endif
        return std::make_exception_ptr(C10_BUILD_ERROR(
            DistBackendError,
            "NCCL error: " + ncclGetErrorWithVersion(ncclAsyncErr)));
    }
    return nullptr;
}
```

### 4.3 Error Propagation via TCPStore

```cpp
// Signal remote errors to other ranks
void ProcessGroupNCCL::Watchdog::checkAndSetRemoteError() {
    if (pg_->getError() != ErrorType::SUCCESS) {
        return;  // Error already set
    }
    
    // Check for error signal from other ranks
    int remoteErrorRank = getSignalSrcRank(
        pg_->store_, 
        std::string(kStoreErrorSignalKey) + ':' + pg_->pg_uid_);
    
    if (remoteErrorRank != -1) {
        std::lock_guard<std::mutex> lock(pg_->errorMutex_);
        pg_->error_ = ErrorType::REMOTE_ERROR;
        LOG(ERROR) << "Remote error detected from rank: " << remoteErrorRank;
    }
}

// Broadcast error signal
void ProcessGroupNCCL::broadcastSignal(
    c10::intrusive_ptr<Store>& store,
    const std::string& signal,
    int srcRank) {
    
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(&srcRank),
        reinterpret_cast<uint8_t*>(&srcRank) + sizeof(srcRank));
    store->set(signal, vec);
}
```

---

## 5. Watchdog and Heartbeat Monitoring

### 5.1 Watchdog Thread

The Watchdog thread monitors NCCL operations for timeouts and errors:

```cpp
class ProcessGroupNCCL::Watchdog {
public:
    void runLoop() {
        while (!done || !pg_->terminateProcessGroup_.load()) {
            std::unique_lock<std::mutex> lock(pg_->workMetaListMutex_);
            
            // Poll every kWatchdogThreadSleepMillis (100ms)
            workMetaListCV_.wait_for(lock, 
                std::chrono::milliseconds(kWatchdogThreadSleepMillis));
            
            // Increment heartbeat
            heartbeat_++;
            
            for (auto it = pg_->workMetaList_.begin(); 
                 it != pg_->workMetaList_.end(); ) {
                auto& work = *it;
                
                // Check for NCCL errors
                if (!pg_->terminateProcessGroup_.load()) {
                    work.checkAndSetException();
                }
                
                // Check for timeout
                bool timedout = !work.exception() && work.checkTimeout();
                
                if (timedout) {
                    // Set error state
                    std::lock_guard<std::mutex> lock(pg_->errorMutex_);
                    if (pg_->error_ == ErrorType::SUCCESS) {
                        pg_->error_ = ErrorType::TIMEOUT;
                    }
                    // Run desync debugger
                    desyncDebugger_.run();
                }
                
                if (work.exception()) {
                    // Handle failure
                    handleWorkFailure(work);
                    break;
                }
                
                // Update progress
                updateWorkProgress(work);
                
                // Clean up completed work
                if (work.isCompleted()) {
                    it = pg_->workMetaList_.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
    
private:
    void handleWorkFailure(WorkNCCL& work) {
        LOG(ERROR) << "Failure detected at work sequence id: " << work.seq_;
        
        // Print traceback
        work.printTraceback();
        
        // Broadcast error signal
        if (propagatePgError_) {
            pg_->broadcastSignal(pg_->store_, 
                std::string(kStoreErrorSignalKey) + ':' + pg_->pg_uid_,
                pg_->rank_);
        }
        
        // Signal dump
        pg_->broadcastDumpSignal();
        
        // Wait for dump coordination
        std::this_thread::sleep_for(std::chrono::milliseconds(
            pg_->heartbeatMonitor_->getDumpTimeout() * 4));
        
        if (SHOULD_CLEAN_UP(pg_->asyncErrorHandling_)) {
            work.abort();
            pg_->abortComms();
        }
        
        // Throw exception
        work.handleException(pg_->asyncErrorHandling_);
    }
    
    std::atomic_uint64_t heartbeat_;
    bool propagatePgError_;
    bool desyncDebug_;
};
```

### 5.2 Heartbeat Monitor Thread

Monitors the Watchdog thread for hangs:

```cpp
class ProcessGroupNCCL::HeartbeatMonitor {
public:
    void runLoop() {
        uint64_t heartBeatCounter = 0ULL;
        
        while (true) {
            std::unique_lock<std::mutex> lock(monitorMutex_);
            
            // Wait for heartbeat timeout or termination
            if (monitorWakeUpCV_.wait_for(lock, 
                std::chrono::milliseconds(monitorPollInterval), 
                [&] { return terminateHeartbeatMonitorThread_.load(); })) {
                return;  // Normal termination
            }
            
            // Check watchdog heartbeat
            if (computeDeltaMS(lastTimeHeartBeatCheck, currentTime) >= 
                heartbeatTimeoutInSec_ * 1000l) {
                
                auto heartbeat = pg_->getWatchdogHeartbt();
                if (heartbeat != heartBeatCounter) {
                    heartBeatCounter = heartbeat;
                } else {
                    // Watchdog hang detected
                    shouldDump_.store(true);
                    errorMsg = "Watchdog got stuck for " + 
                              std::to_string(heartbeatTimeoutInSec_) + 
                              " seconds";
                    break;
                }
            }
        }
        
        // Dump debug info and terminate
        dumpDebugInfo();
        pg_->terminateProcess(errorMsg);
    }
    
private:
    int heartbeatTimeoutInSec_;      // Default: 8 minutes
    int waitTimeoutDumpInMilSec_;    // Default: 15 seconds
    int coordCheckIntervalMilSec_;   // Default: 1 second
    bool watchdogHeartbeatMonitorEnabled_;
    bool dumpOnTimeoutOrEx_;
};
```

### 5.3 Work Timeout Mechanism

```cpp
bool ProcessGroupNCCL::WorkNCCL::checkTimeout(
    std::optional<std::chrono::milliseconds> timeout) {
    
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        currentTimepoint - workStartTime_);
    auto workTimeout = timeout ? *timeout : opTimeout_;
    
    if (timeElapsed < workTimeout) {
        return false;  // Not timed out yet
    }
    
    // Timeout occurred
    std::string exceptionMsg = c10::str(
        "Watchdog caught collective operation timeout: ",
        "ran for ", timeElapsed.count(), " milliseconds");
    
    auto exception_ptr = std::make_exception_ptr(
        C10_BUILD_ERROR(DistBackendError, exceptionMsg));
    setException(exception_ptr);
    
    // Mark future result as TIMEOUT
    if (futureWorkResult_ && !futureWorkResult_->completed()) {
        futureWorkResult_->markCompleted(
            at::IValue(static_cast<uint8_t>(WorkResult::TIMEOUT)));
    }
    return true;
}
```

---

## 6. Practical Fault Recovery Patterns

### 6.1 Automatic Fault Recovery Flow

```
1. Training Loop
   ↓
2. Collective Operation (e.g., all_reduce)
   ↓
3. Watchdog detects timeout/error
   ↓
4. Error signal broadcast to all ranks
   ↓
5. Flight Recorder dumps debug info
   ↓
6. Abort current communicators
   ↓
7. Identify failed ranks
   ↓
8. Create new ProcessGroup via shrink()
   ↓
9. Resume training with reduced size
```

### 6.2 Example Fault Recovery Implementation

```python
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroupNCCL

def fault_tolerant_all_reduce(tensor, pg, max_retries=3):
    """Perform all_reduce with automatic fault recovery."""
    for attempt in range(max_retries):
        try:
            work = dist.all_reduce(tensor, group=pg, async_op=True)
            work.wait()
            return True
        except Exception as e:
            print(f"All_reduce failed: {e}")
            
            # Check if we can recover
            if not pg.supports_shrinking():
                raise RuntimeError("Shrinking not supported, cannot recover")
            
            # Identify failed ranks (simplified)
            failed_ranks = detect_failed_ranks(pg)
            
            if len(failed_ranks) >= pg.size():
                raise RuntimeError("Too many failures, cannot recover")
            
            # Shrink process group
            new_pg = pg.shrink(ranks_to_exclude=failed_ranks)
            pg = new_pg
            
            # Retry with new process group
            continue
    
    return False

def detect_failed_ranks(pg):
    """Detect which ranks have failed."""
    failed_ranks = []
    
    # Use store-based detection or heartbeat monitoring
    store = dist.get_default_store()
    
    # Each rank tries to communicate with others
    for rank in range(pg.size()):
        if rank == pg.rank():
            continue
        
        # Try to ping the rank
        try:
            # Use send/recv or store-based check
            tensor = torch.zeros(1).cuda()
            work = dist.send(tensor, rank, group=pg, async_op=True)
            work.wait(timeout=timedelta(seconds=5))
        except:
            failed_ranks.append(rank)
    
    return failed_ranks
```

### 6.3 Process Group Recreation After Failure

```cpp
// Example: Recovery after detecting rank failures
void recoverFromFailure(
    std::shared_ptr<ProcessGroupNCCL>& pg,
    const std::vector<int64_t>& failed_ranks) {
    
    // 1. Abort current process group
    pg->abort();
    
    // 2. Check if shrink is supported
    if (!pg->supportsShrinking()) {
        throw std::runtime_error("Shrinking not supported by NCCL version");
    }
    
    // 3. Create new process group without failed ranks
    auto new_pg = pg->shrink(failed_ranks);
    
    // 4. Update reference
    pg = std::dynamic_pointer_cast<ProcessGroupNCCL>(new_pg);
    
    // 5. Log recovery
    LOG(INFO) << "Recovered from failure. New world size: " << pg->getSize()
              << ", New rank: " << pg->getRank();
}
```

---

## 7. Environment Variables for Fault Tolerance

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | 3 | Error handling mode: 0=None, 1=TearDown, 2=CleanUpOnly, 3=SkipCleanUp |
| `TORCH_NCCL_BLOCKING_WAIT` | false | Block CPU until NCCL operations complete |
| `TORCH_NCCL_ENABLE_MONITORING` | true | Enable heartbeat monitor thread |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 480 | Timeout before considering watchdog stuck |
| `TORCH_NCCL_DUMP_ON_TIMEOUT` | true | Enable Flight Recorder dump on timeout |
| `TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC` | 15000 | Wait time for dump before throwing exception |
| `TORCH_NCCL_DESYNC_DEBUG` | false | Enable desync root cause analysis |
| `TORCH_NCCL_PROPAGATE_ERROR` | false | Propagate NCCL errors to all ranks via TCPStore |
| `TORCH_NCCL_USE_COMM_NONBLOCKING` | false | Use non-blocking NCCL communicator creation |
| `TORCH_NCCL_NONBLOCKING_TIMEOUT` | 1800 | Timeout for non-blocking NCCL operations |
| `TORCH_NCCL_RETHROW_CUDA_ERRORS` | true | Rethrow CUDA errors in watchdog |
| `TORCH_NCCL_TRACE_BUFFER_SIZE` | 2000 | Flight Recorder trace buffer size |
| `TORCH_NCCL_EXTRA_DUMP_ON_EXEC` | false | Extra dump on exception |

---

## 8. Key APIs Summary

### ProcessGroupNCCL Fault Tolerance APIs

| API | Purpose | NCCL Version |
|-----|---------|--------------|
| `split(ranks, opts)` | Create sub-group from parent | 2.18+ |
| `shrink(ranks_to_exclude, flags, opts)` | Remove failed ranks | 2.27+ |
| `supportsShrinking()` | Check shrink availability | - |
| `supportsSplitting()` | Check split availability | - |
| `abort()` | Immediate abort of communicators | - |
| `shutdown()` | Graceful shutdown | - |
| `isInitialized()` | Check communicator state | - |
| `getError()` | Get current error type | - |
| `eagerConnectSingleDevice(device)` | Pre-initialize communicator | - |

### NCCLComm Lifecycle APIs

| API | Purpose |
|-----|---------|
| `create(numRanks, rank, commId, device, config)` | Create new communicator |
| `split(source, color_id, rank, config)` | Split from parent |
| `shrink(source, ranks_to_exclude, config, flags)` | Shrink communicator |
| `abort(reason)` | Abort communicator |
| `finalize()` | Graceful finalize |
| `destroy()` | Resource cleanup |
| `checkForNcclError()` | Query async errors |
| `isAborted()` | Check abort state |
| `isInitialized()` | Check init state |

---

## 9. References

- [NCCL User Guide - Communicator Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)
- [NCCL API Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)
- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
- ProcessGroupNCCL Source: `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`
- NCCLUtils Source: `torch/csrc/distributed/c10d/NCCLUtils.cpp`
