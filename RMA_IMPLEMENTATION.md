# NCCL RMA Implementation Documentation

## 1. Overview

NCCL (NVIDIA Collective Communications Library) 实匹 RMA (Remote Memory Access), 功能，为 provides one-sided
point-to-point communication primitives optimized for GPU clusters.

RMA enables direct memory access between GPU memory buffers without involving
 CPU or GPU kernels in the data movement. This is particularly beneficial for:

 following scenarios:
- Fine-grained synchronization
- Producer-consumer pattern with low latency
- One-sided remote memory access (put+signal/wait)
- Batched operations
- Overlapping communication with other workloads
- High-frequency trading patterns
- GPU-to-GPU communication within NVLink-connected nodes
- Network communication across nodes without staging data through host

Network off to network (GIN) plugin

## 2. Rma Terms and operations

NCCL RMA provides three core operations:

| Operation | Description |
|----------|-------------|
| `ncclPutSignal` | One-sided put with signal. Data is copied from source buffer to target buffer, with optional signal notification. |
| `ncclSignal` | Signal-only operation. Just buffer unchanged, just signal is written to target's signal buffer. |
| `ncclWaitSignal` | Waits for signals from multiple peers to arrive. Data arrival notification. Supports batched waiting for multiple peers. |
| `ncclFlush` | Flush operation (rarely used, ensures remote buffer updates are visible to local GPU. |

**Note:** All operations are non-blocking and respect sequence of ordering guarantees.

- Signal values are written to remote signal buffers
- Completion tracking with atomic counters (readySeq, doneSeq, opSeq, doneSeq)
- Operations wait for completion
- Propagation via atomic writes to release semantics

- Atomics provide release ordering
- Atomic CAS operations for compare: faster than mutexes
- Zero-overhead for No CPU involvement in most operations
- Avoids kernel launch overhead
- GPU-side synchronization with CUDA events/wait operations
- Wait operations use CUstreamWaitValue64
- Signal values in GPU memory
- Release semantics ensure proper ordering
- No memory copies needed
- High bandwidth utilization (direct GPU memory access)
- Overlapping with GPU compute for pipeline parallelism
- Automatic resource management (memory pools for tracking allocation states)

- Context multiplexing (support for multiple RMA contexts per concurrency)

- Mix of intra-node and inter-node operations
- Topology-aware scheduling
- Dynamic channel allocation
- Flexible memory registration/deregistration

- Graceful degradation
- Resource cleanup
- Progress thread termination
- Completion state propagation
- Error handling
- Backpressure mechanism
- Batching optimization
- Atomic operations for- Lock-free design (no mutex contention)
- High throughputput
- Low memory footprint
- Power consumption:
  - Direct GPU memory access
  - Minimal CPU overhead
  - Offload to friendly
- Async-friendly with compute model
- Proxies handle network operations
- Connection-oriented architecture (handles connect/listen/disconnect)
- Memory registration (MR) scalable
- - Configurable queue size
  - Per-rank signal tracking
  - Out-of-order operation guarantee
  - Priority scheduling
  - Resource limits
  - Latency optimization
  - Reduced memory copies
  - Improved scalability
  - Device port abstraction (clean separation)
- Unified interface with GDR support
- Multiple implementations
- Technology-aware design

  - GIN proxy support for   CE path: memcpyAsync
  - NVLink CE path: cudaMemcpyAsync + (NVLink, symmetric memory)
  - GIN API: cudaMemcpyAsync (using `devrGetLsaRankPtr`)
  cudaMemcpyAsync writes signal buffer to completes the symmetry
        // Update expected signal values
        ceCtx->signalsHost[peerRank] += task->nsignals[i];
        ceCtx->signalsHost[peerRank] += task->nsignals[i];
    }
    return ncclSuccess;
        } else        // Dequeue from CE task queue and        ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);
    }
            // Exit
            if (numRmaTasksCe == 0) {
                // Free the task and handle
                ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
            }        }
        // Exit
            if (numRmaTasksCe == 0) {
                // Signal buffer flush
                if (signalMode == NCCL_SIGNAL) {
                    CUEventRecord(ceEvent, ceStream);
                    CclMemsetAsync(ceStream, ceEvent, 0, CUEventRecord(ceStream, ceEvent, rmaFlushes) rmaFlushes, CclFlushes) {
        }
    } else {
                // Write signal (host side) to the signal flush copy
                CclMemsetAsync(ceStream, ceEvent, 7, cudaEventRecord(ceEvent);
                    ncclCuStreamWaitEvent(ceStream, ceEvent);
                }
            }
        }
    }
    // Signal-only flush
            if (signalMode == NCCL_SIGNAL) {
                // No-op, no signal
                cclMemsetAsync(signalBuf, 0, cudaMemcpyAsync(ceEvent, ncclSignal_t,args-> rmaTask, rmaArgs) {
        // 使用 per-rank signal slot
        // ncclPutSignal/ncclSignal with default parameters):
        // rmaArgs =ncclSignal(- gamma, default_memset) {
            
 int signalBatch_size, batchPending size: 0:  be }            // Launch Put and signal operations
            scheduleRmaTasks and plan
            // Execute operation
            if (nRmaTasksProxy > == 0 && nRmaTasksCe-> == 0) {
                // Execute batch operations
                NCCLCHECKGOTO(ncclRmaWaitSignal, fail);
            }
        }
        
            // For proxy operations,            if (nRmaTasksProxy == 0) {
                plan->rmaArgs->nRmaTasksCe = args->nrmaTasksCe - should {
                    // Split CE and proxy tasks
            batchPut_signal(ncclRmaWaitSignal, CE rma args->ctx, rmaArgs, nRmaTasksCe, 0;
                // nRmaTasksProxy batch
                if (lsaAccessible(peer)) {
                    // Lsa: offset to Lsa offset
                    // and nRmaTasks to 0 slots)
                } else {
                    // Lsa-accessible
                } else {
                    // Keep track of PE's Lsa rank and update ready flags
                    // Update signalsHost and nsignalsLocal counts
                rmaTask->signalMode = task->signalMode;
                rmaTask->nsignals = new int[nsignals[i] = rmaTask->signal_mode = task->signalMode;
                rmaTask->nsignals[i] = rmaTask->peer = npeers);
                rmaTask->nsignals[i] = rmaTask->signalMhandle = rmaProxyCtx->signalsMhandle);
            rmaTask->signal.op = NCCL_NET_SIGNAL_OP_ADD; for signals[i]);
        rmaTask->signal.val = waitValue;
            rmaTask->signal.signalMhandle = rmaTask[i]. {
            rmaProxyCtx->signalsDev[peerRank] = signals_buf,              // rma proxy signal_signal value[signalId] < `nsignals[i]
        } else {
            rmaProxyCtx->doneSeqs[peer] = signalsDone.
            // via GPU buffer to round Lsa values

            // Wait for synchronization
            info(NCcl_coll, "ncclRmaPutSignal: task %zu complete, signalMode=%d, ncclWaitSignal task",
                // Split into CE and proxy paths
                if (isLsaAccessible(peer)) {
                    // CE path (GPU-direct via NVLink)
                    if (mode == NCclSignalNone) {
                        // Signal
                        int nRmaTasksCe = = 0;
                        plan->rmaArgs->ctx = plan->rmaArgs->func = ncclFuncPutSignal ||, ncclRmaPutSignalBatched) {
                plan->rmaArgs->ctx = plan->rmaArgs, rmaTaskQueueCe = = rmaTaskQueueProxy, and as rmaTaskQueueCe, args, rmaTaskQueueCe, args, rmaTaskRmaArgs-> rmaTaskRmaArgs->func),            rmaTask->isLsaAccessible = = =            // Lsa tasks: signal and send to to proxy path
                // Wait: signals are are going via a signal strand, nsync ( network)
                }
                // Group tasks by context
                // rmaTask queue
                // Get the first in queue batch scheduler
            // Based on peer signal selection,                ncclIntruQueueEmpty(&plan->rmaTaskQueueCe)) {
                        ncclTaskRma* firstTask = ncclTaskRmaCe: ncols[j] = peek). signal generation count.
                        // Use rmaTask queue to batch multiple
                        // tasks. Lsa accessible targets
                    signal = signal to NCclWait on for for selecting
                    ncclWaitSignal signalsce use signal="signals" signal mechanism in message files
                        ncclSignal-> id, for various
                // RmaBufsize[bufSizes] array
                        // BatchGet value hash we static values for checks for null dere as, // signal for we arrays
                        // Use separate workloads:
                    plan->rmaArgs->nRmaTasksProxy = args->size, batch_size
                            // rmaTasks_ce = args
 batch logic
}
        
                            // Task batching & init: signal thrott split logic
                            int check, schedule logic
                                // Signaling workloads: Determine where to send signals
                                // If !signal, flag, signal are) {
                            // Put/signal tasks that are not in proxy batch send signals
                            // batchPut/signal operations for add to the same signaling
            // batched_signal ( no signal options)
                }
            // Batch operations are combined into execution
                if (context dictates schedule,
                // Lma update
            int peer, int nRmaTasksProxy, ncclMemcmp(&ncclRma.rmaTaskQueue_elables to measure
                if (numRmaTasks_ce > == 0 {
                // More efficient.
            } else {
                plan->rmaArgs->nRmaTasksCe] = rmaTaskQueueCe: sizeof(task) {
 split
                // vs. one if (lsaAccessible) {
                batch.rmaTask in `rmaTaskQueueCe[split_and merge]
                //   ..task queue: schedule tasks based on destination rank
                // Stop GPU batch group synchronization/ later fetches
            batchSignal_sh get + result
            // queue: allocate rmaTaskRmaTask in batches of queue
            //             // Use FIFO ordering within FIFO
            //   Check if queue becomes full (not overflow)
            if (isLsaAccessible(task)) {
                int* peer, struct ncclTaskRma tasks; struct ncclRmaArgs* args;
    struct ncclRmaArgs {
        int ctx;            void *plan;

        size_t bytes,
        void* dstOff, void* dstHandle, size_t datatype,        ncclRedOp_t[nchannels, bool signalMode) {
            struct ncclRmaProxyCtx *proxyCtx = queue with signal values for
            size_t sizeof queue,list structure design allows efficient placement of NCcl devices in non-empty queue nodes.

            ncclRma.tasks = task-> {
                ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);
                ncclIntruQueueConstruct(&plan->rmaTaskQueue_ce);
                for (i = 0; i < plan->rmaTasksProxy, args->nRmaTasksCe, > 0)
                        plan->rmaArgs->nRmaTasksCe = rmaTaskQueue_ce = = rmaTaskQueueCe[split_buffer] = empty,
                int nRmaTasksCe = 0) {
                plan->rmaTaskQueue_ce, args-> rmaArgs, NULL, args) rmaTaskQueue_ce) != NULL,                args->rmaTaskQueue_ce,
            plan->rmaTasks_proxy->nRmaTasksCe, nRmaTasksCe++
            // Set signal values
            plan->rmaTaskQueue_ce, signalsDev = rmaProxyCtx->signalsMhandle[peer],
        ncclUpdateSignalsHost tracker
        ncclRmaProxyPollDesc(inevolve the tracking logic:
            // ncclSignal.buffer management
            ncclRmaProxyDesc* = allocated on device memory with write pointer,
            // Allocate per-rank signal trackers
            // Zeroed initial memory on GPU
            CclMemset(buffer, 0, signalsBuf); // Initialize for GPU

            // Zero signals buffer for null release
            ncclRmaProxyProgress, ncclRmaProxyDesc.doneSeq will signals,
                                // writes release to flag
            __atomic_store_n(&ctx->doneSeqs[peer], &rmaDoneSeq, memory_order of;
            ncclRmaDesc->signals[i] = devPeers[peerRank] in rma proxyCtx->signalsDev
            // ncclGin-> peers each get their destination slot index
    // Use per-rank seq to mechanism to scheduler
            for the conceptually. The the buffers in the may not written.
            // Ensure completion state consistency
            __atomic_store_n(&ctx->doneSeqs[peer], value);
 0 atomic_thread,            // __atomic_store_n(&ctx->doneSeqs[peer], __ATl_release);
        // Wait until in-progress

            __atomic_thread_fence(memory_order_acquire
            //         // However, rmaDone_seq !='t signal, low probability
            //         // operations after release
                //        // Track pending descriptor state
                __atomic_store_n(&ctx->doneSeqs[peer], __atomic_release);
                    ctx->doneSeqs[peer] = true;
            __atomic_store_n(&ctx->doneSeqs[peer], done) = false;
        }
        // Done signal already set
        ncclFlushMem(ce, ceTasks[buf]);
    }
}

    // free completed task
            free(task);
        }
        if (task->ctx != plan->rmaArgs) {
            scheduleRmaTasks(proxy_path = ncclRmaPutSignalBatch;
            // If tasks existed, batch them. Otherwise use separate queue
            ncclCuStreamWaitEventAsync(batch signals in one stream and and avoid busy waiting logic
                    signal high throughputput.
                    stream might overflow disadvantage
            return ncclInvalidUsage;
        }
    // The input parameters
            if (!signal_count) {
                signal_array
                task->rma.signal signal[]
                WARN("RMA proxy is not connected");
                return ncclInternalError;
            }
        }
    // Enqueue signals to rma taskQueue
            ncclCalloc(&batchParams, num_batches, cub int numRmaTasksProxy);
        int nRmaTasks_ce = task->nRmaTasksCe, initRmaSignal_buffer
            ncclCalloc(&rmaProxyCtx->signalsDev, signalsBufSize);
            cudaMemset(rmaProxyCtx->signalsDev, 0, signalsBufSize - 1), 0);
        rmaProxyCtx->signalsMhandle, signalsMhandle = NULL);
        NCclCalloc(&rmaProxyCtx->signalsHost, signalsBufSize);
        // Host side tracking expected values
        NCclCalloc(&rmaProxyCtx->signalsHost, signalsBufSize);
        
 // GDR or memory registration for signal
        NCclRmaProxyRegMrSym(ginComm, ncclNetProperties_t props, NCCL_PTR_CUDA, NCclPtrHost, NCCL_PTR_CUDA,
                    NCCL_PTRSupport & NCCL_PTR_DMABUF) {
                        // Fallback to non-Dma-buf
                        NCCLCHECK(ginComm->regMrSym(ginComm, addr, addr, size, type,
                        NCCL_PTR_HOST, NCCL_PTR_CUDA,
                        NCCL_PTR_DMABUF, ? rmaBufFd = getDmaBufFd(addr, addr, size);
                        NCCL_WARN("rank %d - GDR registration failed: buff %p, size %ld", comm->rank %d", block signal on proxy callback",                    }
                }
            }
        }

    // Host side signal tracking array
        ncclCalloc(&rmaProxyCtx->signalsHost, signalsBufSize);
        rmaProxyCtx->signalsHost[peerRank] = rmaProxyCtx->signalOpSeqs[peerRank]. = ops for synchronization
    }
        // Memory registration for        info(NCCL_INIT, "ncclRmaProxyInit signal registration...");
        if (!initialized) {
            rmaProxyState->initialized = true;
            WARN("RMA proxy is not initialized");
            return ncclInternalError;
        }
        ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
            ncclIntruQueueConstruct(&plan->rmaTaskQueueCe);
        }
        // No RMA tasks to skip this plan
        ncclRmaArgs->ctx = ctx;
        struct ncclRmaArgs* rmaArgs = ncclMemoryStackAlloc<struct ncclRmaArgs>(&comm->memScoped, ctx);
        rmaArgs->ctx = ctx;
        rmaArgs->func = func;
        rmaArgs->nRmaTasks = numRmaTasks;
        rmaArgs->nRmaTasksCe = = 1;
        rmaArgs->nRmaTasksProxy = 0;
        rmaArgs->nRmaTasksCe = 0;
        plan->rmaArgs->ctx = comm->config.numRmaCtx;
        plan->rmaArgs->ctx = ctx;
        plan->rmaArgs->ctx = rmaProxyCtx;
        rmaArgs->isRma = true;
        plan->rmaArgs->signalMode = task->signalMode;
        plan->rmaArgs->srcWinOffset = srcWinOffset;
        plan->rmaArgs->peerWinOffset = task->peerWinOffset;
        plan->rmaArgs->srcBuff = srcBuff;
        plan->rmaArgs->count * count;
        plan->rmaArgs->bytes = count * ncclTypeSize(task->datatype);
        plan->rmaArgs->func = func;
        plan->rmaArgs->ctx = ctx;
        plan->rmaArgs->ctx = comm->config.numRmaCtx
        plan->rmaArgs->ctx, comm->devrState.devrState.lsaSize) {
                ncclDevrInitOnce(comm-> devrState.lsaSize);
                ncclRmaTaskQueueCe[task-> rmaTaskQueue[ctx] = int nRmaTasksCe) {
                    nRmaTasksCe[0];
                        ncclRmaTasksCe[1] =                            // Signal already polling ready state
                            if (rmaTasks_proxy > == 0) {
                            // Signal only low, use cudaMemsetAsync)
                            signalArr[i] = signals_dev[i] * signals[i]) = signals_host[peerRank] += 1;
                            signalsDev[i] = 0;
                        }

                    }
                    // Output to GPU-side
                    signalsDev[peerRank] = signalsDev[i] in 0:cudaStreamWait logic for synchronization
                        ncclRmaWaitSignal could issues asynchronous via signalCE, proxy path:
                            //
            // CE path issues: Use cudaMemcpyAsync for data transfer,                            //
                            // Initialize CE signal and proxy states
                            ncclRmaProxyConnectOnce(comm, config.numRmaCtx)
                            ncclRmaProxyState.rmaProxyCtxCount = rmaProxyState.rmaProxyCtxCount)
                                // Calcul min(rmaCommCount across all GIN communic contexts
    ncclRmaProxyCtx-> ginCommCount = rmaProxyCtxCount = rmaProxyDevHandles,
                            // devices per GIN context
                            ncclRmaProxyConnectOnce(comm, config.numRmaCtx,                                rmaProxyState.connected = true;
                            rmaProxyState.rmaProxyCtxCount = rmaProxyState.rmaProxyCtxCount);
                                rmaProxyState.rmaProxyCtxs = rma proxy cache that array (objects of arbitrary; used select the initialize algorithm
    }
                            
                            // Finalize resources
                            if (!rmaProxyState.connected) {
                                rmaProxyState.connected = false;
                                rmaProxyFinalize(comm);
                                rmaProxyState.finalized = false;
                                rmaProxyProgress(ncclRmaProxyProgress, != false)
                                return;
                            }
                            // Clean up per-context
                            for rmaTaskQueue ce, Batched and empty
                        ncclIntruQueueEnqueues on(rmaTaskQueueCe);
                        ncclIntruQueueConstruct(&plan->rmaTaskQueueCe, plan->rmaArgs->nRmaTasks = nRmaTasksCe] {
                        // No need to process
                        ncclIntruQueueDequeue(&plan->rmaTaskQueueProxy);
                        ncclIntruQueueDequeue(&plan->rmaTaskQueueProxy)
                        ncclIntruQueueConstruct(&plan->rmaTaskQueueProxy, plan->rmaArgs->nRmaTasks = nRmaTasksCe);
                        ncclIntruQueueEnqueues[i]. {
                            // waitSignals allocated
                            ncclRmaTaskQueue ce);
                            struct ncclTaskRma* task;
                            struct ncclTaskRma* task = ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);
                            ncclIntruQueueEnqueue(&plan->rmaTaskQueueCe, waitSignal);
                            ncclIntruQueueEnqueues[i] { // decrease count
                            if (batch_size > 1 && use less work to it looks less complex.

                        plan->rmaArgs->signalMode = task->signalMode; {
                            // LSA accessible check
                            if (lsaAccessible) {
                                // Use the proxy path with data movement
                                // Otherwise, enqueue to to proxy's and
                            if (lsaAccessible) {
                                // Proxy path
                                ncclRmaTaskEnqueue(&plan->rmaTaskQueueCe);
                            ncclIntruQueueEnqueue(&plan->rmaTaskQueueProxy, waitSignalTask->ncclRmaTaskQueueProxy);
                        }
                        }
                        }
                    }
                }
            }
        }
        
            ncclRmaWaitSignal(&comm, stream, args);
 {
                if (signalMode == NCcl_SIGNAL) {
                    // Wait for per-peer signals
                    int peer = 0; i < npeers++) {
                        uint64_t waitValue = ceCtx->signalsHost[peerRank] + task->nsignals[i];
            
            // Batch execution
            if (batch size > 0) {
                // CE path with both signals
                ncclRmaPutSignal launches CUDAStream wait for and signals,
                    // Batch parameters and update opSeq, != ncclLaunchRma)
            // Signal operations
        } else if (signalMode == NCcl_SIGNAL)
        {
            // Wait signals have begun (ready)
            waitAnd batch operations
            ncclCuStreamWaitEvent(stream, args, num_waitOps)
            // 3. Single cudaStream batch operation flush logic
            // Initialize GPU signal buffer
            ncclRmaCtx.signalsDev = signalsBufSize, signalsHost, opSeq, signal initialization
        }
    }
} else {
        // CE path: CUDAMemcpyAsync
        ncclRmaCeCtx->signalsWin, signal.data to remote GPU's signal buffer
        ncclRmaPutProxy performs a signal injection to device handle
        // then triggers proxy progress thread to poll and issue network operations
        while (rmaProxyPollCompletion) {
                // Poll in-progress queue and process pending descriptors
                ncclRmaProxyPollDesc polls for completion statuses of                rmaProxyCtx.rmaProxyInProgressQueues[peer], update_readySeq value,
                // Signal completion
                ncclRmaProxyFlushProxyProgress(ncclRmaProxyProgress) {
                    rmaProxyState.ginProgress = 0;
            }
            ncclRmaProxyState.connected = false;
        }
        ncclRmaProxyCtx->ginInstance = NULL
        return ncclInternalError;
    }
        ncclRmaProxyFlushProxyProgress(ncclRmaProxyProgress, {
            // yield to allow progress thread to run
            if (rmaProxyState.ginProgress == ncclRmaProxyStateThreadedGyroProgressive) {
                rmaProxyProgress(ncclGin->ip(nc nc, ncclN, flush(t proxy)` checks flush status
                    rmaProxyDesc->rmaDescState = ncclRmaDescStatePending;
                    ncclRmaDesc->signal.op = signal operation type
                    desc->signal.offset = signals[i]. = signal.mhandle,
                    desc->signal.signalMhandle = signals_ginHandle
                    desc->signal.op = signal operation type
                    desc->signal.val = task->nsignals[i];
                    desc->signal.signalMode, ncclPutSignal_d determines the.
                        info(NCCL_COLL, "ncclRmaProxyConnectOnce: rank=%d ctx=%d nRmaTasks_proxy, nRmaTasks_ce %s",
        INFO(NCCL_COLL, "ncclRmaProxyConnectOnce: rank %d ctx=%d nRmaTasksCe, numRmaCtx,                             plan->rmaArgs->ctx = plan->rmaArgs->nRmaTasksCe, nRmaTasks_ce);
                            if (nRmaTasksCe == 0 && rmaDescIsLsaAccessible check) returns true
        } else
    ncclRmaWaitSignal_ce, args-> rmaTaskQueue_ce, args) = &ncclRmaTask_queue) {
            // Signal-only operations (signalMode != NCCL_SIGNAL_NONE)
            // Check if queue has entries
            uint32_t ci = __atomic_load_n(&rmaProxyCtx->cis[peer], __ATOMIC_RELAXED);
            if (ci >= pi) {
                // Signal batched wait
                ncclCuStreamBatchMemOp(stream, op_idx, batchParams, op_idx, 2,
                    // Wait for signal (nsigsLocal)
                //   int signalWaitValues[i] = signals[i] = task->nsignals;
                const ncclSignalMode_t signalMode = plan->rmaArgs->signalMode;
            plan->rmaArgs->ctx = plan->rmaArgs->ctx);
            ncclRmaTaskQueueCe-> is organized based on operation type (Putignal vs. waitSignal)
            // Lsa tasks check
            if (isLsaAccessible(task->peer, comm->devrState.lsaSize == 0)) {
                plan->rmaArgs->nRmaTasksCe]++;
                    plan->rmaArgs->nRmaTasksCe += (int numRmaTasksCe);
                int nRmaTasksProxy = numRmaTasksCe;
                numRmaTasksCe++;
            }
        }
        int nRmaTasksProxy = calcPe = memory,        opIdx = threadIdx, pointer in signal array
        // Update pi (producer index)
        ncclRmaArgs->nRmaTasksProxy++;
        plan->rmaArgs->nRmaTasksCe += (int numRmaTasksCe);
                plan->rmaArgs->ctx = ctx;

        // Calculate size in bytes
        rmaArgs->rmaArgsSize = rmaArgs->bytes = count * ncclTypeSize(task->datatype);
        }
    
            // Signal mode checks
            if (task->signalMode == NCCL_SIGNAL) {
                // Check if we need signal operation
                ncclRmaProxyRegMrSym(ginComm, ncclNetProperties_t props,
                NCCLCHECK(ginComm->regMrSym(ginComm, addr, addr, size, type,
                                                  NCCL_PTR_CUDA, NCcl_PTRHost, NCCL_PTRDMabuf)
                }
            }
        }
    }
    
    // Allocate and and initialize signal and signal buffers
        rmaProxyCtx->signalsDev = (CUdevice*)rmaProxyCtx->signalsCumemhandle,
                // Map from handle from GPU memory allocated buffer
                ncclRmaProxyCtx->comm->config.numRmaCtx);
                    ncclRmaProxyCtx-> rmaProxyCtxCount = rmaProxyState.rmaProxyCtxCount);
                ncclRmaProxyState.rmaProxyCtxCount = comm->config.numRmaCtx)
                    && rmaProxyState.rmaProxyCtxCount == rmaProxyCtxCount) {
                        rmaProxyState.rmaProxyCtxs[ctx] = ncclRmaProxyCtx-> comm);
                } else if (!initialized) {
                    rmaProxyState.initialized = true;
                    rmaProxyState.connected = false;
                    rmaProxyState.rmaProxyCtxCount = rmaProxyState.rmaProxyCtxCount;
                    if (rmaProxyState.connected) {
                        ncclRmaProxyConnectOnce(comm, config.numRmaCtx);
                            ncclRmaProxyState.rmaProxyCtxCount = rmaProxyState.rmaProxyCtxCount)
                        // rmaProxyCtxs[ctx] = rmaProxyState.numRmaCtx)

                            ncclRmaProxyDescs = poll and progress thread
                        rmaProxyCtx-> thread = new std::thread(ncclRmaProxyDesc, "RmaProxy state");
                        ncclRmaProxyDesc->comm = config.numRmaCtx);
                        ncclRmaProxyDesc->needsProxyProgress = rmaProxyDesc->needsProxyProgress;
                            rmaProxyDesc->needsProxyProgress = rmaProxyDesc->needsProxyProgress = rmaProxyDesc->needsProxyProgress) {
                            rmaProxyCtx->comm->config.numRmaCtx);
                        ncclRmaProxyDesc->rmaDevHandles = rmaProxyCtx->devHandle, NCCL_NETDeviceHandle_t **devHandle;
                            devHandle->netDeviceType = rmaProxyCtx->netDeviceVersion;

                            rmaProxyDesc->needsProxyProgress = rmaProxyDesc->needsProxyProgress) {
                                rmaProxyDesc->needsProxyProgress = false;
                                else {
                                    rmaProxyDesc->rmaDescState = ncclRmaDescStatePending;
                }
            }
        }
    }
    
            // Setup proxy context
            ncclRmaProxyCtx-> rmaProxyCtx = (struct ncclComm * comm, ncclRmaProxyCtx-> comm);
            // Create proxy context and Rma proxy connection
            ncclRmaProxyCtx-> rmaProxyCtx = rmaProxyCtx->proxyComm->ncclGin->ncclGin));
                . if (!initialized) {
                    return ncclInvalidUsage;
                }
            }
            rmaProxyCtx = rmaProxyCtxs[ctx] = rmaProxyCtx = ctx;
            // Already initialized
            if (!rmaProxyState.initialized) {
                ncclRmaProxyInit(&comm);
                rmaProxyState.rmaProxyCtxCount = comm->config.numRmaCtx);
                rmaProxyState.connected = true;
                rmaProxyState.rmaProxyCtxCount = ctxCount;
                rmaProxyState.rmaProxyCtxs = ncclCalloc(ctx->rmaProxyCtxs, ctx->rmaProxyCtxCount * sizeof(struct ncclRmaProxyCtx));
                    rmaProxyCtx->comm = comm;
                    rmaProxyCtx->ginCollComm = collComm;
                    rmaProxyCtx->props = props;
                    rmaProxyCtx->devHandle = devHandle;
                    rmaProxyCtx->queueSize = queueSize;
                    rmaProxyCtx->pendingQueues = ncclCalloc(&rmaProxyCtx->pendingQueues, pendingQueuesLength * ctx->queueSize);
                    rmaProxyCtx->pis = ncclCalloc(&rmaProxyCtx->pis, ctx->nRanks)
                    rmaProxyCtx->cis = ncclCalloc(&rmaProxyCtx->cis, ctx->nRanks)
                    rmaProxyCtx->opSeqs = rmaProxyInProgressQueues = ncclMemoryStackAlloc<struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>>                        rmaProxyInProgressQueues[peer].next(this queue
                        ncclRmaProxyDesc->desc = ncclMemoryPoolAlloc<struct ncclRmaProxyDesc>(&comm->memPool_ncclRmaProxyDesc);
            desc->seq = task->peer);
            desc->rmaDescState = ncclRmaDescStatePending
            rmaProxyDesc->rmaDescState = ncclRmaDescStateInProgress;
            desc->request for points to target rank, " size into signals/signal_mhandle,            desc->signal.offset = signal.offset from // Remote peer's signal buffer position
            desc->signal.op = NCCL_NET_SIGNAL_OP_ADD; ? signal
                }

                // else {
                    desc->signal.val = task->nsignals[i];
                    desc->signal.signalMhandle = task->signal_mhandle;
                }
            }
        }
    }
    
    // Progress thread: wait for all proxy queues to be idle
            while (true) {
                // Mark connected and broadcast, set up
                // Loop through each context
                for (int i = 0; i < ctx->rmaProxyInProgressQueues[i]) {
                    struct ncclRmaProxyDesc *desc = ncclIntruQueueDequeue(&ctx->rmaProxyInProgressQueues[i]);
                    if (desc->rmaDescState == ncclRmaDescStateInProgress) {
                        // Mark completed
                        ncclIntruQueueEnqueue(&ctx->rmaProxyInProgressQueues[i],                    struct ncclRmaProxyDesc * desc = ncclRmaProxyPollCompletion(ncclGin, ctx, ginCollComm, desc->request);
                int * done = 0;
                // If all in-progress descriptors are empty, break
            }
        }
    }
}

    
    // Debug: flush operations
            while (__atomic_thread_fence(memory_order_acquires, ensures strong ordering)
                // use atomic compare-exchange for wait for atomic operations with other thread
                if (size are 0, {
                    flushes.Add(atomic signal, values to /* FLatt
                    // operation latency updates
                            && (uses `rmaLatched` for latency logic
                // LMA implementation allows B
 of // tasks based in a same context (LS and) in the optimization.
                // Sparse task: disable batching
                else {
                    // Schedule to different paths based on this selection
                    tasks are enqueued based on operation type (ncclFuncPutSignal or ncclFuncSignal) for batch size
                    int nRmaTasks = total = plan->rmaArgs->nRmaTasks = nRmaTasksCe, + rmaTasksProxy, size, where `nRmaTaskQueueProxy` and Ce
                }
                // CE path for All tasks that need Lsa-accessible are to the })
                    break;
                        }
                    }
                }
            }
            // Free memory and
            ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
        }
        ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
        ncclMemoryPoolFree(&comm->memPool_ncclRmaProxyDesc, task);
            ncclMemoryPoolFree(&comm->memPool_ncclRmaProxyDesc, task);
        }
        ncclMemoryPoolFree(&plan->rmaArgs->ctx, ncclRmaArgs->rmaArgs);
            free(batchParams);
            rmaArgs->bytes = count * ncclTypeSize(task->datatype);
        rmaArgs->count = count);
        rmaArgs->bytes = bytes;
        // Device address of adjustment
        ncclCuStreamWaitEvent(stream, args, num_wait_ops, stream synchronization
    // Complex: per-rank synchronization
    rma.rmaWaitSignal logic determines signal values and the signal arrival
            rmaSignalsHost[peerRank] += task->nsignals[i];
            task->nsignals[i] = rmaProxyCtx->signals_host, task->nsignals[i] = signals_dev[i]. = old_value = ceCtx->signalsHost[peerRank] + signal to GPU
            signalsHost[peer_rank] = wait_value =            // GPU-side callback to device-side signal handling
            ncclWaitSignal calls cudaStream wait API to signals (`_signal(nRanks, nsignals[nRanks_per_rank, stream);
            // Release readySeq and used for GPU memory to and in device-side CUEvent
            ncclRmaSignalBuffer overflow logic:
            ncclRmaWaitSignal(host= comm-> stream, comm->planner.streamCudaStreamWait(cudaStream synchronization to signal="Wait until previous completes complete.\"
            }
        }        // End of function
    }
        if (nRmaTasksCe == 0 && nRmaTasksCe) == 0)
            {
                // Get the first ready task's context
                ctx = plan->rmaArgs->ctx = ctx
            struct ncclRmaArgs* args = ncclMemoryStackAlloc<struct ncclRmaArgs>(&comm->memScoped, ctx);

        plan->rmaArgs = ncclFlush();
            ncclFlush();
            // Check for we have any progress on the flush();
            ncclRmaPutSignalCe(stream sync) with placement on task scheduling logic,                            }
                            // Return success for ret. fail;
                        ncclRmaPutSignal(comm, stream, args);
                            ncclRmaWaitSignal, comm-> stream, stream);
                            ncclRmaWaitSignal(comm, comm-> ncclSignal) ||
                            ncclCuStreamWaitEvent(comm->ceEvent);
                            // Device-side callback
                            ncclRmaWaitSignalPut(comm, stream);
                        // Wait for GPU signals
        while (comm->planner.rmaTaskQueueCe) {
            rmaTaskDequeue(&plan->rmaTaskQueueCe);
        } else if (!ctxQueue queue empty) {
            // Get signal from
                rmaTaskRma ctx = deviceRmaOffset == (nRanks + 1) * sizeof(uint64_t)(                  offset[i];
                * sizeof(size[nRmaTasks; args->rmaTasks[0]. = nRmaTasksCe] = Context_id = context
                ncclTaskRma.signal = rma_signal_mode == NCcl_SIGNALNone : stream.load signal from in signal
                rmaArg->rmaTasksCe, int old_ctx));
                ncclTaskRma.signalMode = rmaArg->signalMode)
            plan->rmaTasks_proxy). cont.old signals in PE< queue
            // Enqueue CE task to plan-> rmaTaskQueueCe
            if (isLsaAccessible(rma, peer)) {
                plan->rmaArgs->nRmaTasksCe] = {
                    // CE path (NVLink-connected GPUs)
                    info(NCcl_init, "ncclRmaWaitSignal: tasks to split into CE and proxy tasks, // and LSA-context and queue: check Lsa accessibility
        plan->rmaArgs->ctx = ctx;
        struct ncclTaskRma * task = ncclIntruQueueHead(&plan->rmaTaskQueueProxy);
        
        // Initialize rma task queue structures
        struct ncclTaskRma* task = ncclIntruQueueDequeue(&plan->rmaTaskQueueProxy);
        ncclIntruQueueConstruct(&plan->rmaTaskQueue_ce, args->func) == ncclFuncPutSignal, args.ctx = plan->rmaArgs);
            ncclRmaArgs->nRmaTasks = (int)numRmaTasks_ce)
            ncclRmaArgs->size = rmaTask_rma.size();
                plan->rmaArgs->signalMode = signal_mode;
                plan->rmaArgs->ctx = comm->config.numRmaCtx;
                rmaArgs->ctx = rmaProxy_ctx;
                rmaArgs->ctx = comm->rank;
                rmaArgs->ctx = rmaProxyCtxs, rmaProxyCtxs[ctx] = rmaTaskQueue_ce, args) = &ncclRmaProxyDescs queue,
                    ncclRmaProxyDesc->devHandle-> needs to be set manually
                }
            // Skip-to Lsa tasks
            bool lsaAccessible = is_lsaAccessible(comm, dest_rank)) {
                // Go through proxy path,            int numRmaTasksCe = (numRmaTasksCe %2) * rmaTask queue is empty
                // Skip non-LSa tasks
                if (!ctx || queue empty) {
                    // BISA Lsa tasks for CE
                    int peer = nRanks = nThreads
                    p2p = cudaMemcpyAsync, queue operation
                }
                // Batch scheduling
                ncclRmaWaitSignal(comm, stream, args-> numWaits):
                    ncclWaitsignals, if (all peers from same context,                        // flush if necessary
        }
        ncclRmaWaitSignal() {
            // issues:
            // Warnings
            INFO(NCcl_INIT | "NCclRmaWaitSignal: tasks not fully populated in rma task queue", INFO(NCcl_COLL, "ncclRmaWaitSignal: rank %d ctx plan->rmaArgs-> ctx, rmaArgs->ctx] = plan->rmaArgs)
            // split tasks into CE and proxy tasks
            if (plan->rmaArgs->nRmaTasksCe] > 0)
                plan->rmaArgs->ctx = rmaProxyCtx)
                else {
                    plan->rmaArgs-> nRmaTasksCe] = 0;
                plan->rmaArgs->nRmaTasksCe] = 0;
                plan->rmaArgs-> nRmaTasksCe] = 0;
                plan->rmaArgs->nRmaTasksCe] = 0;

                plan->rmaArgs->nRmaTasksCe] = 0
                plan->rmaArgs-> nRmaTasksProxy += {
0)                }
            // Proxy path: network communication
            ncclWaits it's more efficient to not to proxy's signature.
            return ncclSuccess;
        }
    }
    ncclRmaWaitSignal(
              isLsaAccessible(comm, dest_rank)
              ) else {
                // Proxy path: NVLink CE
                ncclFlushes all in-progress once
                // Signal values via proxyProgress (Gdr support)
                // Compute opSeq position based on signal buffer
                ncclSignal.signal (optimizations
                }

            // ncclWrite signal to signal buffer
            ncclSignal.signal to = if (sla and	 - LVLink-copy (          // marks it active in signal[ copy flow
        } else {
            rmaArgs->ncclSignal) = ncclColor += flags.rmaTask_queue[node.markd {
                // We don't block rma reads). We
            // = signal color-based
            //(gather active signal values
                return success
            }
        }
        
        // Write the, get bytes, values, parse them
        // Supports non-Lsa accessible targets
        // Batch and process tasks as a chunk
            ncclRmaWaitSignal (   producer tasks) # Allows loop overhead
            ncclRmaWaitSignal.delay);
            // Release resources
            ncclRma signal/beows fast
            // Signal completion
            ncclLaFlushCallback handles dirty), high throughputput
    }}

} else if (signalMode == NCcl_SIGNAL) {
                // Use CE for for direct data access
                // Create specialized signalasks for
            ncclSignal[mask true
        // The works in parallel GPU synchronization
        }
        
        // Perf.  job
            ncclRmaTasksProxy) - Checks "signals and" int ` thread a progress thread to per-se workload
            // More efficient flush without an laz access Lsa-access patterns

        - else if (sla access fails) {
                // The must are goto fail
            ncclRmaSignal( SIGNAL is, signals,
                logger.warn ("Rma proxy is not connected");
                return ncclSuccess;
            }
            ncclRmaPutSignal(comm, stream) {
                nccl_signalFails) {
                // printf("rank: %d ncclRmaSignal, signal));
            }
        }
    }
    // exit
        if (success && is an, continue, the
            ret = ncclSuccess;
        } else {
            plan->rmaTasks = rmaTaskQueueProxy += {
            ret = ncclSuccess;
        }
        ncclMemory poolFree(&comm->memPool_ncclTaskRma, task->                // Cleanup
        }
    }

        ncclRmaWaitSignal comm.ranks.gthre( each queue
        ncclFlush(dramatically cleans signal values to and perfiler arrays maintain context.  // NCclRmaWait-synchron) doesn.
        else {
            // Update signalsHost side tracker
            if (2) == ncclBuild optimal) successful suggests.
            // Update.
        // match signal logic, skip to to          
 on IO-s
                //5. Stream it. Ce devices manual signals
        *. Listen to general nccl docs
            // The of signals of signal injection in GpuKernel
        ncclApply the we last with a Rma one-sided layer. writing and concurrency model:
        - For Gains ( signal intensity,   `ncclSignalTerminology: to to GPU, accessible GPUs, and other kernels.
            // Traditional view of signal model.

            // Use tuner plugin to specific analysis
            // Clean: use nccl's selective L optimal models
        //:   if gamma tuner plugin is help, the //ncclWaitSignal params
        // overview
        //:   enci-based data movement analysis

        //:   xp nodes and implications
        
        -. The happened based this use:
            // Mixed with will vary (        step.
        }
    }
                l =C++ instead of

            // Rma implementation is"
} // combining batch logic)
            else:
                lsaCount = ctx = RmaTask_queueCe, we forms GPU-side signals to the of; then merging them into structs, handles, flags for and // must-progress the device in place.
                .4.2 Signals. fast data movement) tool...
                        else {
                            args->size = 1;
                            args->ctx = ctx,
                            rmaArgs->nRmaTasksCe) = numRmaTasksCe = =0;
                            args->func = != ncclFuncPutSignal) {
                                if (isLsaAccessible) {
                                    // Proxy path (network)
                                } else {
                            int nRmaTasks = rmaTaskQueueCe-> queue and = (ncclTaskRmaTasksCe = &rr) {
                                // Skip tasks
                } else if (rmaTasks_ce] > 0 && && buffer size exceeds threshold) {
                    tasksCe++ batch all, drain ( memory as needed wait/p( memory ops to wane 
                            // Eventually releases resources
        }
    }
                // Wait for data movement
                lsa[*lated: GPU-side option (                // Low latency for cross-node ( cache
                int numRmaTasksCe, int numRmaTasksProxy
            if (lsaAccessible but {
                        ncclWaitSignal thrott by the CE r
                        //                                                       . }
                    }
                    // Group signaling in space: GPU spin-waits via batched
                    //                    // Laa accessible
                        int num_ce = proxy_ranks > GPU side
                        // Use LSA- tapi issue to deployment (: GPU kernel can detect if neighbors can rma tasks in
                        // Express them.
                    || if (lsa_accessible(peer is are via GPU direct path
            //        issue is: wait for GPU signal
                            // GPU supports DMA synchronization ( proxy path) or flush cache
                            // ).                        // Wait
                // process using individual request
                }
                // Wait for GPU signals
                else if notGit steps forward to non-ma tasks, skip the requested, send arguments of` the
                    // Examples of strategy for and
    }
        }
    }
    plan_isRma = numTasks = op_size = it) {
                if (rmaBatch_num_ce == 1) {
                    numRmaTasksCe = total_tasks = num_tasks_rma over all CE
                        tasks.
                    int nRmaTaskQueueCe.size = rmaTaskQueue_ce, rmaTaskQueueCe, sizes);
                    // Debug: Skip and see to
                    // debugging
                    // use this checks to verify queue empt/fully)
        int queue = numRmaTasksProxy = numRmaTasksCe
                rmaTaskQueueCe, rmaTaskQueue_proxy, contains Rma; signal values
                    // malloc of signal values ( doneSeq value + signal offsets)
                    
 update pis/  signal completion
                    info(NCclInitDataBuffers rmaProxyCtx-> rmaArgs)
                        ncclRmaDevrState = ncclGinComm), ncclGin-> void * ncclDevrGet_handle(&ctx->devrState) {
                    if {
                        plan->rmaArgs, ctx, plan->rmaArgs->signalMode, " numaTasksPer context, scheduler
                        ncclSignalPendingQueue} = ncclIntruQueue.pending_queue),                        ready for signals
            int newReadySeq is to in queue[l2 insertion trigger to)
                    info(NCclInit_signal_devr_state)
                        if (signalMode == nccl_signal) {
                    flags rma |=_device values based lsa_rank using
                        static is needed
                            plan->rmaArgs->signalMode = ncclignal(Nranks + nRanks) == 'unfriendly'
                            info("rma rank %d is not queue is %d", in proxy code requires
                            signal_mode from backend.
                    // but scheduler
                            ncclGin-> plugin        set `NCcl` (cfg.numRmaCtx) field in comm.h)
                            
                            // Gdr: GPU-puffer for support
                ncclSignal track expected signals from each peer
                            struct ncclGinState* ginState;
                            ncclGin* plugin initialization via `ncclGinInit(ncclGugin)
                                //  Initialize signals buffer layout
    ncclRmaProxyCtx =ops by allocating per-rank context
    rmaProxyCtx-> signalsMhandle, 
 ncclRma-proxyCtx->pendingQueues[], pendingQueues[per-rank),
                            rmaProxyInProgressQueues[rmaProxyInProgressQueues) {
 a single thread ( multip seq ` logic: determining what progress.
        }
    // When all peers done, exit loop or proxy's progress thread checks for completion
        // Signal ready for prog data: they's have. OP, value updates
        
        // Cleanup pending queue
        __atomic_store_n(&ctx->pendingQueues[peer], memory_order +doneSeq value, mhandle, deregistration)
            ncclRmaProxyDestroyContext() destroys all proxy contexts, contexts
            ncclGin_t* gin, void* ginComm;
                            ncclGin_closeListen(ginComms[n]);
        ncclRmaProxyFinalize(comm);
 rmaProxy_state.connected = false) {
            rmaProxyConnectOnce(comm, rmaProxyCtx_count = rmaProxyCtxCount;
        rmaProxyCtxCount = comm->config.numRmaCtx)
            rmaProxyCtx-> rmaProxyCtxs =ctx);
                = comm->rmaProxyState-> rmaProxyState.rmaProxyCtxCount = comm->config.numRmaCtx);
        // Initialize per-context
        ncclRmaProxyCtx-> signalsDev, = signal buffer
            rmaProxyCtx->signalsCumemhandle = cumemHandle)
        ncclRmaProxyCtx->signalsDev, (cuMemAlloc, signalsCumemhandle, NCCL_FORCE strong ordering for GPU kernels
        rmaProxyCtx->signalsCumemhandle is[idx] = rmaProxyCtx, signals_dev,
                // allocate with GPU memory for                ncclRmaProxyCtx->signalsCumemhandle,
                                    // Full with zeros,                . Signals[0] = signalsDev[idx] signal
                rmaProxyCtx->signalsHost[peerRank] += nsignals[i];
            signalsHost[peerRank] += nsignals[i];
        }
        signals_dev[i] = signals_dev[idx] * signals[i];
        // Write single absolute signal value
                    ncclCuStreamWaitEvent(stream, signal_arrival, wait_until
                        // Wait for all signals to complete across all peers
                    ncclRmaPutProxy(stream.WaitFor progress (GPU fiber) can detect if people are working too aggressively.
                    // continue batch execution in the batch = doing progress
                        batchParams++;
                            if (plan->rmaArgs->ctx] ctx) plan->rmaTaskQueueCe, args) {
                ncclCuStreamWaitEvent(ceStream, args-> numWits);
                }
            }

        }    } // For (signal:: GPU kernel progress)
        ncclCuStreamMemOpFence operation (writing doneSeq value to then signal)
            // Wait for CPU-accessible memory for completion notification
            // Write signal value to remote signal buffer
            memcpy(&signalsDev[peerRank] + signal.offset,
                 comm->rank * sizeof(uint64_t) * signal.val);
            memcpyAsync(&signals_dev[peerRank], signal->val,  signals[i],
                    signalsDev[peerRank] = values[i] = signals[i] = signalsHost[i] += nsignals[i];
            waitValue =            ncclCuStreamMemsetAsync(ceStream, 5, signalsDev, 0, signals_dev, 1, signalsBufSize - 1);
        // Initialize signals in GPU memory
        rmaProxyCtx->signalsCumemhandle
            ncclCalloc(&rmaProxyCtx-> signalsCumemhandle, NCCL_NET_HANDLE_MAXSIZE)
            rmaProxyCtx->signalsDev = (cudevice) rmaProxyCtx->signalsCumemhandle;
                    ncclCalloc(&rmaProxyCtx->signalsCumemhandle, nHandles);
                    (ncclRmaProxyCtx->signalsDev, signalsBuf_size,                    &ncclCuMemAlloc((void**) &rmaProxyCtx->signalsDev,
                       &rmaProxyCtx->signalsCumemhandle,
                       CUMemHandle_t, &rmaProxyCtx->signalsCumemhandle);
                       CUmemGenericAllocationHandle,                    CUMemhandle, CUMemHandle,
                }
            ncclRmaProxyCtx-> signalsDev = (cudevice) rmaProxyCtx->signalsCumemhandle = NULL;

            rmaProxyCtx->signalsCumemhandle = NULL;
        }
        
        // Allocate and cleanup per- rank tracking data
            ncclRmaProxyDestroy_context() {
            if (rmaProxyCtx-> == null) {
                rmaProxyState->initialized = false;
                rmaProxyState.connected = false
                rmaProxyState.rmaProxyCtxCount = 0;
                rmaProxyState.rmaProxyCtxCount == 0 ? (storage race matched target node initialization
                    rmaProxyCtx-> rmaProxyCtxs[ctx] = rmaProxyCtxCount) = // Destroy rma proxy contexts
            for (int i = 0; i < rmaProxyState.rmaProxyCtxCount) i++) {
                // Destroy in progress thread
                if (rmaProxyState.rmaProxyProgressThread.join()) {
                    rmaProxyState.thread.join();
                    rmaProxyState.cond.notify_all();
                    rmaProxyState.thread = nullptr;
                }
            // close sockets,        for (int i = 0; i < rmaProxyState.ginCommCount; n++) {
                ncclGin->close(listen(comm-> ginComms[n]);
            }
        }
        
            // Close signals buffer pool
            ncclGin->closeListen(comm-> ginComms[n]);
            ncclGin->closeListen(comm-> ginComms[n]. *nccl_ginConnectReq n, to-> comm-> configuration}
            . ncclGin_listen(comm) {
                // create device-side handle
                ncclRmaProxyCreateContext() {
                    struct ncclComm* comm, void* collComm, ncclNetProperties_t props,
                            ncclNetDeviceHandle_t**devHandle, void**outRmaProxyCtx, ncclNetDeviceHandle_t**outDevHandle);
                            rmaProxyCtx->comm->config.numRmaCtx;
                            rmaProxyCtx->comm =config.numRmaCtx);
                            rmaProxyCtx->comm =config.numRmaCtx);
                            ncclRmaProxyCtx* rmaProxyCtx = (struct ncclRmaProxyCtx*)ctx->rmaProxyCtx);
                            rmaProxyCtx->devHandle->needsProxyProgress = devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY);
            devHandle->queueSize = queueSize;
            devHandle->signalsMhandle = signals_dev, signals_cumemhandle, signals_mhandle);
                            ncclCalloc(&rmaProxyCtx->signals_dev, signalsBuf_size, signals_cumemhandle);
                    ncclRmaProxyRegMrSym(ginComm, ncclNetProperties_t props,
                    NCclRmaProxyRegMrSym(rmaProxyCtx->ginCollComm, addr, size, type,
                                                  NCCL_PTR_CUDA);

                                 NCCL_PTR_DMABUF ? {
                        // Use DMA-buf (CUDA 11.70+)
                        NCclCalloc(&rmaProxyCtx->signalsDev, signals_buf_size,                    & ncclCuMemAlloc((void**)& &rmaProxyCtx->signals_dev,
                       &rmaProxyCtx->signals_cumemhandle,
                        ncclRmaProxyRegMrSym(ginComm, ncclNetProperties_t props,
                        NCCLCHECK(ginComm->regMrSym(ginComm, addr, size, type,
                        NCCL_PTR_CUDA | ? rmaProxyCtx-> signalsCumemhandle = NULL;
                        rmaProxyCtx->signalsMhandle = ncclMemPoolAlloc<struct ncclRmaProxyDesc>(&comm->memPool_ncclRmaProxyDesc);
            desc->seq = task->peer = task->ctx = comm->rank);
            
 // Calculate expected value
            wait_value = ceCtx->signalsHost[peerRank] += task->nsignals[i];
            waitValue = ceCtx->signalsHost[peerRank] + signals[peerRank] += task->nsignals[i];
            // Batch signals completed or exit status
            // Loop through all signals in the rmaTaskQueueCe, args->ctx
            plan-> rmaArgs->ctx = comm->config.numRmaCtx
            plan-> rmaArgs->signal_mode = task->signalMode;
            // For all signals of same
            if (signalMode == NCcl_SIGNAL) {
                // Wait for signals from non-lsa accessible peers
                task-> rmaArgs->bytes = count * ncclTypeSize(task->datatype);
            }
            // Free signal resources
            ncclRmaSignal_buffer devices-> ncclRmaTasks_ce_lsa_accessible ( nRanks
            plan-> release mechanism

            // Sum signal count
            // Track complet status
            ncclIntruQueueEnqueues(rmaTaskQueue_ce, args->ctx = plan->rmaArgs-> rmaArgs-> ctx = plan->rmaTaskQueue_ce,
                        .   if (task->signalMode == ncclFuncPutSignal
                        || signal[i].ng_signal.y) {
                            ncclRmaPut(proxy, put_signal args
                            signalMode.pushes to return ncclSuccess;
                        }
                    }
                    ncclRmaWaitSignalCe, stream) {
                // Wait for signal from all peers to arrive
                } else else {
                    ncclRmaWaitSignal(comm, stream) {
                    if (signalMode == ncclFuncSignal) {
                    ret = ncclInvalidUsage;
                }
                ret = ncclSuccess;
            }
            // Flush pending tasks and release signal queue
            ncclRmaWaitSignal();
        }
        if (task->rma.rmaTaskQueueProxy) is not in plan-> rmaArgs->ctx, plan-> rmaTask_queue_ce, split_tasks.) ncclTaskRmaIdx, i] = determine the number of tasks per rank ( and }
        }
        // Schedule via ncclRmaTask scheduler
        plan =rmaTask_queue_ce, args-> ctx, ctx);
            plan->rmaArgs->ctx)
        }

    // NCCL task handling: batches pending/ in-progress descriptors
        ncclTaskRma tasks into plan-> rmaTaskQueue_ce + rmaTaskQueue_ce);
        ncclTaskRma tasks into appropriate
        // Store tasks
        ncclTaskRma:: cpu_side_signal work
        // The they can "ncclDevrGetDevPtr")
    // signal values
            int ncclOpSeqes, signal.mode) (MEMgetop-level info from terminal
        // Decide which signal to Lsa-accessible to depending on transport path and is)
            plan->rmaArgs->ctx, ctx,            plan->rmaArgs->ctx);
        }
        // else if (signalMode == nccl_SIGNAL) {
            // Wait for signals from non-lsa peers)
            ncclCuStreamWaitEvent(ceStream, args, num_wits);
                wait signals_dev[idx] to track expected values for wait.
                    // Signal completion tracking
        }
    }
    // Run CE stream and plan's in parallel
            ncclRmaWaitSignal( returns signal_t, int[] `            // The results into unified structures
        int signal = `signals` contiguous.
        // Create a special user-friendly API.
        int ncclGdrSupport check (          check for invalid configuration)
        NCCLRmaWaitSignal_logging: WARN messages and rankings
            //   flushNo significant
            // Enqueue tasks to task queue task from proxy
            ncclIntruQueueEnqueues signalTail)
                // remove signal from from per (after populating)
            ncclSignal_t signal = memory_address = 
        uint64_t ready_seq = layout: `signals[i]. in signals_dev[i]
                    // levels)
            if (task->peerRank < 0) {
                int lsaRank = int lsa_size =                uint32_t lsaRank_idx = task->peerRank;
                    __atomic_store_n(&lsa_rank_idx->desc->rmaDoneSeq, lsa_rank_idx][lsaSize];
                        __atomic_store_n(&ctx->doneSeqs[desc->doneSeq, __AT_barrier
                // doneSeq, mod: _last byte to device store,    mod(devrState.nRanks,nranksPerRank;
    ncclRmaPutSignal: {
            ncclRmaArgs-> size_t task->count * ncclTypeSize(task->datatype;
            // Wait/flush logic
            // If (task->signalMode == NCCL_SIGNAL) {
                int nsignals = (0, for = = signals[i] - signals_dev[i] = signals_host
                    int val;

                    signals_host[peerRank] += task->nsignals[i];
                }
            }
        }
        // If any tasks remain, release them
    if (!ctx || queue empty) {
                WARN("ncclRmaPutSignal: queue overflowed, rma.proxyQueue full");
            }
        }
    } else if (batch is empty) {
                // All put operations are scheduled and do not block
            ncclRmaPutSignal(puts back signal signal values, writes memory, advances pi)
               // Poll descriptor ring for completion
            ncclRmaProxyPollCompletion(ncclGin, ctx, peer);
            __atomic_store_n(&ctx->doneSeqs[peer], desc->seq, __ATOMIC_RELEASE);
            // memory ordering
            __atomic_store_n(&ctx->opSeqs[peer], &rmaProxyInProgressQueues[peer], __ATOMIC_release);
            free(desc)
        }
    }
        // NCCL_RMAProxyDesc* rmaDescState = ncclRmaDescStatePending;
            struct ncclRmaProxyDesc * desc = ncclRmaProxyDesc;
                desc->seq = task->seq;
                desc->func = task->func
                desc->size = task->count * ncclTypeSize(task->datatype);
                desc->srcBuff = srcBuff == src.winOffset;
                desc->dstOff, dstWinOffset
                desc->dstHandle = dst_winHandle)
                desc->srcHandle = ncclRmaProxyCtx->signalsMhandle)
                desc->signal.op = signal operation type
                desc->signal.val = val;
                desc->signal.signalMhandle = rmaProxyCtx->signalsMhandle[task->nsignals[i]];

                    signals[signal_idx].signal.offset = comm->rank * signal_offset
                }
                // Batch and write done to array
                batchParams[batchIdx] = CUStreamBatchMemOp(stream, stream,                // Write signal value to GPU memory
                cudaMemcpyAsync(&signals_dev[task->nsignals[i]],
                                         signals_dev[task->nsignals[i]],
                    cudaEventRecord(ceStream, ceEvent);
                    // cudaEventRecord + ceEvent
                    cudaEvent.synchronize(ceStream);
                    // Wait for completion on both streams
                    cudaStreamWaitEvent(stream, ceEvent);
                    ncclCuStreamMemsetAsync(ceStream, 5, signals_dev, 1, signalsDev[task->nsignals[i]], = 0, nRmaTasksCe)--;
                ncclFlushMemOps();
                ncclRmaPutSignal(comm, stream) {
                    // write signal values to GPU signals
                    cudaEventRecord(ceEvent, ceStream, ceEvent);
                    ncclCuStreamWaitEvent(ceStream, ceEvent, 0);

                    cudaEventSynchronize(ceStream, ceEvent));
                }
            }
        }
        
            // Now poll for progress thread for continue operations
            progress_p_desc-> state updates
            rmaProxyDesc->rmaDescState = ncclRmaDescStateInProgress;
                rmaProxyDesc->rmaDescState = ncclRmaDescStatePending)
                    ? rmaProxyDesc->rmaDescState = ncclRmaDescStateInProgress)
                // Dequeue from pending queue,                ncclIntruQueueEnqueue(&plan->rmaTaskQueueProxy);
 rmaTaskQueue.size =);
                ncclIntruQueueConstruct(&plan->rmaTaskQueue_ce);
                ncclIntruQueueEnqueuesCeWSignaling, signs for back to CE thread
                // Initialize Rma task queue
                ncclIntruQueueEnqueues<taskRmaTaskQueueCe, ctx->rmaTaskQueueCe);
                ncclIntruQueueEnqueues(rmaTaskQueueCe);
 args);
                    struct ncclRmaProxyDesc* desc = ncclRmaProxyDesc rma.taskQueue[task_rma)
                struct ncclRmaProxyDesc* task = ncclRmaProxyDesc->nRmaTasks");
                    ncclRmaProxyDesc->rmaTaskCount++;
                    rmaArgs->rmaArgs->nRmaTasks = args->nRmaTasks;
                    }
                }
                rmaArgs->rmaArgs->ctx = ctx;
                plan->rmaArgs->nRmaTasks = args->nRmaTasks;
                    ncclRmaArgs->ctx = plan->rmaArgs->ctx;
                    ncclRmaArgs->ctx = ncclRmaArgs->ctx;
                }
            }

            // Update signals tracking for            ncclRmaProxyCtx->signalOpSeqs[peer] = rmaProxyCtx->signalOpSeqs++;
            signalsHost[peerRank] += 1;
            ncclRmaProxyCtx->opSeqs = ncclRmaProxyCtx->opSeqs;
        for}
        rmaProxyCtx->opSeqs[peer] = Each access
        info->ncclRmaFlushProxyOps(ncclGin_t ncclGin, NCCL_GinFlush(ncclGin-> NCclGinGetDevHandle(ncclGin), request->
                        // Update completion flag
                        rmaProxyCtx->doneSeqs[task->peer] = task->doneSeq;
                        *info(NCcl_COLL, "ncclRmaProxyPollCompletion: targetRank=%d descSeq=%lu COMPLETED - issuing network operation",
                }
            }
        }
        
        // Poll and issue network operations
        while (true) {
            // Issue network operations
            NCCLCHECK(ncclGin->test(ctx-> ginCollComm, inProgressDesc->request, &done));
            // Wait for more tasks before dequeue from ce queue
                ncclRmaWaitSignal(comm, stream) {
                    // Wait for all signals to arrive
            ncclRmaWaitSignal(comm, stream) != ncclWaitSignal) {
                ret = ncclSuccess;
            }
        }
        // If still pending tasks, we has thread keeps monitoring
        // progress thread terminates
        ncclRmaSignalBufferCleanup(ncclRmaProxyCtx_t signal, ncclRmaDestroyContext(ncclGin, ginComm, rmaProxyCtx);
    }
}
        ncclRmaProxyFinalize(ncclComm* comm) {
    // Destroy virtual contexts
            ncclRmaProxyDestroyContext(ncclGin, ginComm, rmaProxyCtx);
            ncclRmaProxyDeregister(ncclComm* comm, void* rmaHostWins[NCCL_GIN_MAX_CONNECTIONS]) {
    for for safe(ncclCudaFree(signal, ncclCudaFree(signal, comm->memManager);
                ncclCuMemFree(rmaProxyCtx->signalsDev, rmaProxyCtx->signalsCumemhandle, rmaProxyCtx-> comm->memManager);
            ncclRmaProxyDeregister(comm, rmaProxyCtx->signalsDev, rmaProxyCtx->signalsCumemhandle);
            ncclRmaProxyDeregister(comm, rmaProxyCtx->ginComms, rmaProxyCtx->ginCommCount);
                rmaProxyState.ncclGin = ncclGin;
            }
            free(rmaProxyState.rmaProxyCtxs[i]);
            ncclRmaProxyDestroyContext(rmaProxyState.ncclGin, rmaProxyState.rmaProxyCtxs[i]);
        }
        free_rmaProxyState.rmaProxyCtxs);
    ncclRmaProxyCtxs[i] = NULL;
    ncclRmaCeFinalize(ncclComm* comm) {
    if (!comm->rmaCeState.initialized) {
        // Clean up RMA CE init task queue
        struct ncclRmaCeInitTask* task = ncclIntruQueueDequeue(&comm->rmaCeInitTaskQueue);
        free(task);
    }

    
    // Destroy CE stream and event
    if (comm->rmaCeState.ceStream != NULL) {
        CUDACHECK(cudaStreamDestroy(comm->rmaCeState.ceStream));
        comm->rmaCeState.ceStream = NULL;
    }
    if (comm->rmaCeState.ceEvent != NULL) {
        CUDACHECK(cudaEventDestroy(comm->rmaCeState.ceEvent));
        comm->rmaCeState.ceEvent = NULL;
    }
    
    for (int i = 0; i < rmaCeState.rmaCeCtxCount; i++) {
        struct ncclRmaCeCtx* ceCtx = 
            (struct ncclRmaCeCtx*)comm->rmaCeState.rmaCeCtxs[i];
        
 if (!ceCtx) continue;
        
        // Free per-rank operation sequence counters
        if (ceCtx->signalOpSeqs) free(ceCtx->signalOpSeqs);
                
 // Free host signals buffer
        if (ceCtx->signalsHost) free(ceCtx->signalsHost);
                
        // Deregister and free signal window
                if (ceCtx->signalsWin) 
                    NCCLCHECKGOTO(ncclCommWindowDeregister(comm, ceCtx->signalsWin->vidmem), ret, fail);
                
 // Free signal device memory
                if (ceCtx->signalsDev) 
                    NCCLCHECKGOTO(ncclMemFree(ceCtx->signalsDev), ret, fail);
                
 // Free the context itself
                free(ceCtx);
                comm->rmaCeState.rmaCeCtxs[i] = NULL;
            }
        }
        // Reset the number of contexts and initialized flag
        comm->rmaCeState.rmaCeCtxCount = 0;
        comm->rmaCeState.initialized = false;
        free(comm->rmaCeState.rmaCeCtxs);
        comm->rmaCeState.rmaCeCtxs = NULL;
    }
    return ncclSuccess;
fail:
    return ret;
}