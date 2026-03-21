# NCCL Communication Data Path Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Hardware Link Types](#2-hardware-link-types)
3. [Transport Layer](#3-transport-layer)
4. [Communication Methods](#4-communication-methods)
5. [AllReduce Implementation](#5-allreduce-implementation)
6. [Protocol Types](#6-protocol-types)
7. [Data Flow from High to Low Level](#7-data-flow-from-high-to-low-level)

---

## 1. Overview

### 1.1 What is NCCL?

NCCL (NVIDIA Collective Communications Library) is a library of standard communication routines for GPUs. It provides highly optimized implementations of collective operations:

| Operation | Description |
|-----------|-------------|
| AllReduce | Reduce data across all ranks and broadcast result |
| AllGather | Gather data from all ranks and distribute to all |
| ReduceScatter | Reduce data and scatter result across ranks |
| Broadcast | Send data from one rank to all others |
| Reduce | Reduce data to a single rank |
| Send/Recv | Point-to-point communication |

### 1.2 Communication Stack Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      User Application                                │
│                    ncclAllReduce(), etc.                             │
├─────────────────────────────────────────────────────────────────────┤
│                      NCCL Host API                                   │
│              src/collectives.cc, src/enqueue.cc                      │
├─────────────────────────────────────────────────────────────────────┤
│                    Communication Planner                             │
│              Algorithm selection, task scheduling                    │
├──────────────────────┬──────────────────────────────────────────────┤
│    Collective API    │              RMA API                          │
│  (AllReduce, etc.)   │     (PutSignal, WaitSignal)                  │
├──────────────────────┴──────────────────────────────────────────────┤
│                    Device Kernels                                    │
│              src/device/*.h, src/device/*.cu                         │
├─────────────────────────────────────────────────────────────────────┤
│                    Primitives Layer                                  │
│              src/device/primitives.h, prims_*.h                      │
├─────────────────────────────────────────────────────────────────────┤
│                    Transport Layer                                   │
│        P2P, SHM, NET, NVLS transports                                │
├───────────────────────┬─────────────────────────────────────────────┤
│      GIN Layer        │            LSA Layer                         │
│  (GPU-Initiated Net)  │   (Load-Store Access)                       │
├───────────────────────┴─────────────────────────────────────────────┤
│                    Hardware Layer                                    │
│   NVLink, PCIe, NVSwitch, InfiniBand/RoCE                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hardware Link Types

### 2.1 NVLink

**Description**: NVIDIA's high-bandwidth GPU interconnect technology.

**Characteristics**:
- Bandwidth: 25-50 GB/s per link (per direction)
- Topology: Fully connected mesh, NVSwitch, or hybrid
- Latency: Very low (~1-2 microseconds)
- Access: Direct GPU-to-GPU memory access

**Supported Topologies**:
```
┌─────────────────────────────────────────────────────────────┐
│                  NVLink Topologies                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Fully Connected (4-GPU):     NVSwitch:                      │
│  ┌───┐                        ┌───────────┐                 │
│  │GPU│◄──────────────►│GPU│   │           │                 │
│  │ 0 │                │ 1 │   │ NVSwitch  │                 │
│  └─┬─┘                └─┬─┘   │           │                 │
│    ▲ ▼                  ▲ ▼   └────┬──────┘                 │
│    │ │                  │ │        │                        │
│    │ │                  │ │   ┌────┴────┐                   │
│    │ │    ┌─────────┐   │ │   │         │                   │
│    │ └───►│  NVLink │◄──┘ │   │  GPUs   │                   │
│    │      │  Fabric │     │   │ 0,1,2,n │                   │
│    │      └─────────┘     │   │         │                   │
│    ▼                      ▼   └─────────┘                   │
│  ┌───┐                ┌───┐                                 │
│  │GPU│◄──────────────►│GPU│                                 │
│  │ 2 │                │ 3 │                                 │
│  └───┘                └───┘                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**NCCL Detection** (`src/transport/p2p.cc`):
```cpp
// Check CUDA P2P capability
cudaDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2);
```

### 2.2 PCIe

**Description**: Peripheral Component Interconnect Express for GPU-CPU and GPU-GPU communication.

**Characteristics**:
- Bandwidth: 16-32 GB/s (PCIe 4.0 x16), 32-64 GB/s (PCIe 5.0 x16)
- Topology: Through root complex or PCIe switch
- Latency: Higher than NVLink (~5-10 microseconds)
- Access: P2P DMA through PCIe

**P2P Types in NCCL**:
```cpp
// src/transport/p2p.cc
enum p2pType {
  P2P_DIRECT,       // Direct NVLink/PCIe P2P
  P2P_INTERMEDIATE, // Through intermediate GPU
  P2P_IPC,          // CUDA IPC for cross-process
  P2P_CUMEM         // cuMem API for virtual memory
};
```

### 2.3 Network (InfiniBand/RoCE)

**Description**: High-speed networking for inter-node communication.

**Characteristics**:
- Bandwidth: 100-400 Gbps (IB HDR, NDR)
- Latency: ~1-2 microseconds (IB), higher for RoCE
- Protocols: InfiniBand, RoCEv1, RoCEv2, TCP/IP
- Features: RDMA, GPUDirect RDMA, Atomics

**Network Path in NCCL**:
```
┌─────────────────────────────────────────────────────────────┐
│                 Network Communication Path                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   GPU Memory ──► NIC ──► Network ──► NIC ──► GPU Memory     │
│                 │                       │                    │
│                 ▼                       ▼                    │
│        GPUDirect RDMA            GPUDirect RDMA              │
│        (Zero-copy)               (Zero-copy)                 │
│                                                              │
│   Or through CPU staging:                                    │
│                                                              │
│   GPU Memory ──► CPU Memory ──► NIC ──► Network...          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 NVSwitch/NVLS

**Description**: NVIDIA NVLink Switch for hardware-accelerated collectives.

**Characteristics**:
- Hardware multicast support
- Switch-based reduction (NVLS SHARP)
- Reduced GPU involvement
- Lower latency for collective operations

**NVLS Algorithm** (`src/device/all_reduce.h`):
```cpp
// NVLS uses hardware multicast for scatter/gather
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  // Scatter: Send to multiple destinations via hardware multicast
  // Reduce: Hardware performs reduction
  // Gather: Receive from multiple sources
};
```

---

## 3. Transport Layer

### 3.1 P2P Transport

**File**: `src/transport/p2p.cc`

**Purpose**: Direct GPU-to-GPU communication via NVLink or PCIe.

**Connection Types**:

| Type | Description | Use Case |
|------|-------------|----------|
| `P2P_DIRECT` | Direct P2P access | Same node, direct NVLink/PCIe |
| `P2P_INTERMEDIATE` | Through intermediate GPU | Multi-hop topology |
| `P2P_IPC` | CUDA IPC | Cross-process same node |
| `P2P_CUMEM` | cuMem virtualization | Advanced memory management |

**Key Data Structures**:
```cpp
struct p2pConnectInfo {
  int rank;
  int read;                    // Read vs write mode
  struct ncclP2pBuff p2pBuff;  // Direct buffer pointer
  ncclShmIpcDesc_t desc;       // CE memcpy descriptor
};

struct ncclP2pBuff {
  void* directPtr;    // Direct GPU pointer
  size_t size;        // Buffer size
  ncclIpcDesc ipcDesc; // IPC handle for cross-process
};
```

**Connection Establishment**:
```cpp
// 1. Check P2P capability
ncclTopoCheckP2p(comm, comm->topo, info1->rank, info2->rank, &ret, ...);

// 2. Check CUDA P2P
cudaDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2);

// 3. Enable P2P access
cudaDeviceEnablePeerAccess(cudaDev2, 0);
```

### 3.2 SHM Transport

**File**: `src/transport/shm.cc`

**Purpose**: Shared memory communication for inter-process communication on the same node.

**Characteristics**:
- Uses POSIX shared memory or CUDA IPC
- Supports GPU and CPU memory
- Works when P2P is not available
- Lower bandwidth than direct P2P

**Shared Memory Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Memory Region                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ncclSendMem                                          │    │
│  │  ├── step (producer index)                          │    │
│  │  └── flags                                          │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ ncclRecvMem                                          │    │
│  │  ├── step (consumer index)                          │    │
│  │  └── flags                                          │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ Data Buffers                                         │    │
│  │  ├── buff[NCCL_PROTO_SIMPLE]                        │    │
│  │  ├── buff[NCCL_PROTO_LL]                            │    │
│  │  └── buff[NCCL_PROTO_LL128]                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 NET Transport

**File**: `src/transport/net.cc`

**Purpose**: Network communication via InfiniBand, RoCE, or TCP/IP.

**Plugin Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                    NCCL NET Interface                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   NCCL Core ──► ncclNet_vX API ──► Network Plugin           │
│                    │                   │                     │
│                    │                   ├── libnccl-net-ib.so │
│                    │                   ├── libnccl-net-ucx.so│
│                    │                   └── Custom plugin     │
│                    │                                         │
│                    ▼                                         │
│              Plugin API (v6-v11)                             │
│               ├── init()                                     │
│               ├── connect()                                  │
│               ├── isend() / irecv()                         │
│               ├── flush()                                    │
│               └── close()                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Functions**:
```cpp
// Initialize network
ncclResult_t ncclNetInit(ncclNet_t** net, int* netDev);

// Establish connection
ncclResult_t ncclNetConnect(int dev, void* handle, void** sendComm);
ncclResult_t ncclNetAccept(int dev, void* handle, void** recvComm);

// Data transfer
ncclResult_t ncclNetIsend(void* sendComm, void* data, int size, void** request);
ncclResult_t ncclNetIrecv(void* recvComm, void* data, int size, void** request);
```

### 3.4 NVLS Transport

**File**: `src/transport/nvls.cc`

**Purpose**: NVLink Switch-based hardware collectives.

**Features**:
- Hardware multicast (one send, multiple receives)
- Switch-based reduction (SHARP)
- Reduced GPU and network overhead

**NVLS Collective Flow**:
```
┌─────────────────────────────────────────────────────────────┐
│                  NVLS AllReduce Flow                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Phase 1: Scatter                                           │
│   ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐             │
│   │GPU 0│────►│     │────►│GPU 0│     │GPU 1│             │
│   │GPU 1│────►│NVLS │────►│GPU 2│     │GPU 3│             │
│   │GPU 2│────►│     │────►│     │     │     │             │
│   │GPU 3│────►│     │────►│     │     │     │             │
│   └─────┘     └─────┘     └─────┘     └─────┘             │
│                │                                            │
│                ▼                                            │
│   Phase 2: Reduce (in NVSwitch hardware)                    │
│                │                                            │
│                ▼                                            │
│   Phase 3: Gather                                           │
│   ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐             │
│   │GPU 0│◄────│     │◄────│GPU 0│     │GPU 1│             │
│   │GPU 1│◄────│NVLS │◄────│GPU 2│     │GPU 3│             │
│   │GPU 2│◄────│     │◄────│     │     │     │             │
│   │GPU 3│◄────│     │◄────│     │     │     │             │
│   └─────┘     └─────┘     └─────┘     └─────┘             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Communication Methods

### 4.1 Direct Memory Access (DMA)

**Description**: Direct GPU memory access without CPU involvement.

**Mechanisms**:

1. **NVLink DMA**: Direct memory access via NVLink fabric
2. **PCIe P2P DMA**: Direct memory access via PCIe
3. **GPUDirect RDMA**: Network NIC direct access to GPU memory

**Direct Operations in Primitives** (`src/device/primitives.h`):
```cpp
// Direct send: Write directly to peer's buffer
__device__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN);

// Direct receive: Read directly from peer's buffer
__device__ void directRecv(intptr_t outIx, int eltN);

// Direct receive and reduce
__device__ void directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN);

// Direct receive, reduce, copy and send
__device__ void directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN);
```

### 4.2 Proxy-based Communication

**Description**: CPU proxy handles network communication on behalf of GPU.

**Proxy Flow**:
```
┌─────────────────────────────────────────────────────────────┐
│                   Proxy Communication                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   GPU Kernel                                                 │
│       │                                                      │
│       ▼                                                      │
│   Write to send buffer ───────────────────────────────┐     │
│       │                                                │     │
│       ▼                                                ▼     │
│   Write doorbell/flag                            CPU Proxy   │
│                                                      │       │
│                                                      ▼       │
│                                               Read request   │
│                                                      │       │
│                                                      ▼       │
│                                               Network send   │
│                                                      │       │
│                                                      ▼       │
│                                                Completion    │
│                                                      │       │
│                                                      ▼       │
│   GPU Kernel ◄───── Read completion flag ◄────── Write flag  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Proxy Resources** (`src/transport/net.cc`):
```cpp
struct sendNetResources {
  void* netSendComm;              // Network send handle
  struct ncclSendMem* sendMem;    // Send memory region
  struct ncclRecvMem* recvMem;    // Receive memory region
  char* buffers[NCCL_NUM_PROTOCOLS];
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;                  // Current step
  ncclNetDeviceType netDeviceType;
};
```

### 4.3 GIN (GPU-Initiated Networking)

**Description**: GPU directly initiates network RDMA operations.

**GIN Types**:

| Type | Description | CPU Involvement |
|------|-------------|-----------------|
| GDAKI | GPU Direct Async Kernel-Initiated | Zero CPU |
| PROXY | CPU proxy mode | Required |

**GIN API** (`src/include/nccl_device/gin.h`):
```cpp
// Initialize GIN context
NCCL_DEVICE_INLINE ncclGin_C(ncclDevComm const& comm, int contextIndex);

// PUT operation: Transfer data
NCCL_DEVICE_INLINE void ncclGinPut(
  ncclGin_C* net, ncclTeam team, int peer,
  ncclWindow_t dstWin, size_t dstOffset,
  ncclWindow_t srcWin, size_t srcOffset, size_t bytes,
  bool isSignal, ncclGinSignal_t signalId,
  ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
  bool isCounter, ncclGinCounter_t counterId,
  ncclCoopAny coop, ...);

// Signal operation: Send notification
NCCL_DEVICE_INLINE void ncclGinSignal(
  ncclGin_C* net, ncclTeam team, int peer,
  ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, ...);

// Wait for signal
NCCL_DEVICE_INLINE void ncclGinWaitSignal(
  ncclGin_C* net, ncclCoopAny coop,
  ncclGinSignal_t signal, uint64_t least, ...);
```

**GDAKI Flow**:
```
┌─────────────────────────────────────────────────────────────┐
│                    GDAKI PUT Operation                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   GPU Kernel                                                 │
│       │                                                      │
│       ▼                                                      │
│   1. Get QP from ncclGinGdakiGPUContext                      │
│       │                                                      │
│       ▼                                                      │
│   2. Build WQE (Work Queue Entry) in GPU memory             │
│       │   - RDMA WRITE opcode                                │
│       │   - Remote address + rkey                            │
│       │   - Local address + lkey                             │
│       │                                                      │
│       ▼                                                      │
│   3. Memory fence (release)                                  │
│       │                                                      │
│       ▼                                                      │
│   4. Ring Doorbell (MMIO write to NIC)                      │
│       │                                                      │
│       ▼                                                      │
│   ┌─────────────────────────────────────────────┐           │
│   │               Mellanox NIC                    │           │
│   │  5. Read WQE from GPU memory                 │           │
│   │  6. Execute RDMA READ from GPU memory        │           │
│   │  7. Transmit via network                     │           │
│   │  8. Write to remote GPU memory               │           │
│   │  9. Write CQE to CQ                          │           │
│   └─────────────────────────────────────────────┘           │
│       │                                                      │
│       ▼                                                      │
│   10. (Optional) Wait for completion                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 LSA (Load-Store Access)

**Description**: Direct load/store access to remote GPU memory.

**Requirements**:
- Same node (NVLink or PCIe connectivity)
- Symmetric memory windows
- LSA team membership

**LSA Team** (`src/include/dev_runtime.h`):
```cpp
struct ncclDevrState {
  int lsaSelf;        // My position in LSA team
  int lsaSize;        // LSA team size
  int* lsaRankList;   // Rank list of LSA team members
};
```

**Direct Memory Operations**:
```cpp
// Check if peer is LSA accessible
bool isLsaAccessible(ncclComm* comm, int rank);

// Get direct pointer to peer's memory
ncclDevrGetLsaRankPtr(comm, win, offset, peerLsaRank, &peerBuff);
```

---

## 5. AllReduce Implementation

### 5.1 Algorithm Selection

NCCL supports multiple AllReduce algorithms:

| Algorithm | Best For | Complexity |
|-----------|----------|------------|
| RING | General purpose, large messages | 2(n-1)α + 2(n-1)/n * m/β |
| TREE | Small messages, latency-sensitive | 2log(n)α + 2log(n)m/β |
| NVLS | NVSwitch systems, all sizes | Reduced factor |
| COLLNET | Network-offloaded | Best latency |

**Algorithm Selection Logic**:
```cpp
// Selection based on:
// 1. Topology detection
// 2. Message size
// 3. Hardware availability
// 4. Tuner plugin (if present)

ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, size_t nBytes,
                                  int algorithm, int protocol, float* time);
```

### 5.2 Ring AllReduce

**Description**: Ring-based reduce-scatter followed by allgather.

**Ring Algorithm** (`src/device/all_reduce.h`):
```cpp
template<typename T, typename RedOp, typename Proto>
__device__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
  ncclRing *ring = &ncclShmem.channel.ring;
  const int nranks = ncclShmem.comm.nRanks;

  // Chunk division
  const ssize_t loopCount = nranks * chunkCount;

  Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims(...);

  for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
    // Step 0: Push initial data to next GPU
    chunk = (ringIx + nranks - 1) % nranks;
    prims.directSend(offset, offset, nelem);

    // Steps 1 to n-2: Receive, reduce, and forward
    for (int j = 2; j < nranks; ++j) {
      chunk = (ringIx + nranks - j) % nranks;
      prims.directRecvReduceDirectSend(offset, offset, nelem);
    }

    // Step n-1: Final reduce with local data
    chunk = ringIx;
    prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);

    // Steps n to 2n-2: Forward results (allgather phase)
    for (int j = 1; j < nranks - 1; ++j) {
      chunk = (ringIx + nranks - j) % nranks;
      prims.directRecvCopyDirectSend(offset, offset, nelem);
    }

    // Final receive
    chunk = (ringIx + 1) % nranks;
    prims.directRecv(offset, nelem);
  }
}
```

**Ring Flow Diagram**:
```
┌─────────────────────────────────────────────────────────────┐
│               Ring AllReduce (4 GPUs example)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Data division: Each GPU has n chunks (A, B, C, D)           │
│                                                              │
│  Initial:                                                   │
│  GPU0: [A0, B0, C0, D0]                                     │
│  GPU1: [A1, B1, C1, D1]                                     │
│  GPU2: [A2, B2, C2, D2]                                     │
│  GPU3: [A3, B3, C3, D3]                                     │
│                                                              │
│  Phase 1: Reduce-Scatter (n-1 steps)                        │
│  ┌─────┐      ┌─────┐      ┌─────┐      ┌─────┐            │
│  │GPU 0│─────►│GPU 1│─────►│GPU 2│─────►│GPU 3│            │
│  │     │      │     │      │     │      │     │            │
│  │ A0  │─────►│ A0  │─────►│ A0  │─────►│ A0  │            │
│  │ +A1 │      │ +A1 │      │ +A1 │      │ +A2 │            │
│  │ +A2 │      │ +A2 │      │ +A3 │      │ +A3 │            │
│  │ +A3 │      │     │      │     │      │ =ΣA │            │
│  └─────┘      └─────┘      └─────┘      └─────┘            │
│                                                              │
│  Result: Each GPU holds one complete reduced chunk           │
│  GPU0: [_, ΣB, _, _]  GPU1: [_, _, ΣC, _]                   │
│  GPU2: [_, _, _, ΣD]  GPU3: [ΣA, _, _, _]                   │
│                                                              │
│  Phase 2: AllGather (n-1 steps)                             │
│  GPU3 ─────► GPU0 ─────► GPU1 ─────► GPU2                   │
│                                                              │
│  Final: All GPUs have [ΣA, ΣB, ΣC, ΣD]                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Tree AllReduce

**Description**: Binary tree reduce followed by broadcast.

**Tree Algorithm** (`src/device/all_reduce.h`):
```cpp
template<typename T, typename RedOp, typename Proto>
__device__ void runTreeSplit(int tid, int nthreads, struct ncclDevWorkColl* work) {
  ncclTree *tree = &ncclShmem.channel.tree;

  // Split threads: half for reduce, half for broadcast
  int nthreadsSplit = nthreads/2;

  if (tid < nthreadsSplit) {
    // Reduce phase: gather data up the tree
    Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, 1, Proto, 0>
      prims(tid, nthreadsSplit, tree->down, &tree->up, ...);

    if (tree->down[0] == -1) {
      // Leaf node: send data up
      prims.directSend(offset, offset, nelem);
    } else {
      // Internal node: receive from children, reduce, send up
      prims.directRecvReduceDirectSend(offset, offset, nelem);
    }
  } else {
    // Broadcast phase: distribute data down the tree
    Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, 1, Proto, 0>
      prims(tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, ...);

    if (tree->up == -1) {
      // Root: send to children
      prims.directSendFromOutput(offset, nelem);
    } else {
      // Non-root: receive and forward
      prims.directRecvCopyDirectSend(offset, offset, nelem);
    }
  }
}
```

**Tree Flow Diagram**:
```
┌─────────────────────────────────────────────────────────────┐
│                Tree AllReduce (8 GPUs example)               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Binary Tree Structure:                                      │
│                                                              │
│                    GPU 0 (Root)                              │
│                    /        \                                │
│               GPU 1          GPU 4                           │
│              /    \          /    \                          │
│          GPU 2    GPU 3   GPU 5    GPU 6                     │
│            |                  |                              │
│          GPU 7              GPU 7 (shared example)           │
│                                                              │
│  Phase 1: Reduce (bottom-up)                                 │
│                                                              │
│  GPU 7 ──► GPU 2          GPU 6 ◄── GPU 5                    │
│     │                        │                               │
│     ▼                        ▼                               │
│  GPU 2 ──► GPU 1          GPU 4 ◄── GPU 3                    │
│     │         │              │         │                     │
│     └────────►│◄─────────────┘         │                     │
│               ▼                          │                    │
│            GPU 0 ◄───────────────────────┘                    │
│            (Full reduction)                                  │
│                                                              │
│  Phase 2: Broadcast (top-down)                               │
│                                                              │
│            GPU 0                                             │
│            /    \                                            │
│           ▼      ▼                                           │
│        GPU 1    GPU 4                                        │
│        /  \     /  \                                         │
│       ▼    ▼   ▼    ▼                                        │
│    GPU2  GPU3 GPU5 GPU6                                      │
│       |                |                                     │
│       ▼                ▼                                     │
│    GPU 7            (leaves)                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 NVLS AllReduce

**Description**: Hardware-accelerated using NVLink Switch.

**NVLS Algorithm** (`src/device/all_reduce.h`):
```cpp
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  __device__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;

    // Thread distribution
    const int nThreadsScatter = scatterWarps * WARP_SIZE;
    const int nThreadsGather  = gatherWarps * WARP_SIZE;
    const int nThreadsReduce  = reduceWarps * WARP_SIZE;
    const int nThreadsBcast   = bcastWarps * WARP_SIZE;

    if (tid < tidEndScatter) {
      // Scatter: Use NVLS multicast
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, 0, Proto, 0>
        prims(tid, nThreadsScatter, NULL, nvls->up, ...);
      prims.scatter(offset, nelem, chunkSize, ...);
    }
    else if (tid < tidEndGather) {
      // Gather: Receive via NVLS
      prims.gather(offset, nelem, chunkSize, ...);
    }
    else if (tid < tidEndReduce) {
      // Reduce: NVLS hardware reduction
      prims.directRecvDirectSend(offset, offset, nelem);
    }
    else if (tid < tidEndBcast) {
      // Broadcast: Distribute results
      prims.directRecvDirectSend(offset, offset, nelem);
    }
  }
};
```

### 5.5 Chunk Splitting Mechanism

**Description**: How NCCL divides data into chunks for pipelining.

**Key Parameters** (`src/device/primitives.h`):
```cpp
template<int SlicePerChunk, int StepPerSlice, int Unroll = COLL_UNROLL>
struct ProtoSimple {
  static constexpr int SlicePerChunk = SlicePerChunk_1;  // Slices per chunk
  static constexpr int StepPerSlice = StepPerSlice_1;    // Steps per slice
  static constexpr int Unroll = Unroll_1;                // Unroll factor
};
```

**Chunk Calculations** (`src/device/all_reduce.h`):
```cpp
// Get chunk partition from work descriptor
ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T),
                (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);

// For Ring AllReduce
const ssize_t loopCount = nranks * chunkCount;

// Each rank processes one chunk at a time
for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
  // Process chunkCount elements per rank
  chunkOffset = chunk * chunkCount;
  nelem = min(chunkCount, remaining);
}
```

**Chunk Size Configuration**:
```
┌─────────────────────────────────────────────────────────────┐
│                  Data Chunking Hierarchy                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Total Data Size                                            │
│   ┌────────────────────────────────────────────────────┐    │
│   │                    Channel Count                    │    │
│   │   ┌────────────────────────────────────────────┐   │    │
│   │   │              Loop Count                      │   │    │
│   │   │   ┌────────────────────────────────────┐   │   │    │
│   │   │   │         Chunk Count                 │   │   │    │
│   │   │   │   ┌────────────────────────────┐   │   │   │    │
│   │   │   │   │       Slice Size            │   │   │   │    │
│   │   │   │   │   ┌────────────────────┐   │   │   │   │    │
│   │   │   │   │   │   Step Size        │   │   │   │   │    │
│   │   │   │   │   │  (NCCL_STEPS=8)    │   │   │   │   │    │
│   │   │   │   │   └────────────────────┘   │   │   │   │    │
│   │   │   │   └────────────────────────────┘   │   │   │    │
│   │   │   └────────────────────────────────────┘   │   │    │
│   │   └────────────────────────────────────────────┘   │    │
│   └────────────────────────────────────────────────────┘    │
│                                                              │
│   Parameters:                                               │
│   - NCCL_STEPS = 8: Number of buffer slots                  │
│   - SlicePerChunk: Number of slices in one chunk            │
│   - StepPerSlice: Number of steps per slice                 │
│   - Multiple channels for parallelism                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Buffer Organization**:
```cpp
// src/device/prims_simple.h
int sliceSize = stepSize * StepPerSlice;
sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);

// Step size = buffer size / NCCL_STEPS
__device__ static int calcBytePerStep() {
  return ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
}
```

---

## 6. Protocol Types

### 6.1 SIMPLE Protocol

**Description**: Standard protocol for large messages.

**Characteristics**:
- Best for large data transfers
- Uses larger buffers
- Lower overhead per byte
- Higher latency for small messages

**Implementation** (`src/device/prims_simple.h`):
```cpp
template<int SlicePerChunk, int StepPerSlice, int Unroll, int MultimemSrcs, int MultimemDsts>
struct ProtoSimple {
  static constexpr int Id = NCCL_PROTO_SIMPLE;

  // Data bytes in one step
  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  }
};
```

### 6.2 LL (Low Latency) Protocol

**Description**: Optimized for small messages with low latency.

**Characteristics**:
- Lower latency than SIMPLE
- Smaller buffer footprint
- Uses 16-byte lines with 8-byte data + 8-byte flag
- Higher overhead per byte

**Implementation** (`src/device/primitives.h`):
```cpp
struct ProtoLL {
  static constexpr int Id = NCCL_PROTO_LL;

  // Only half the buffer is data (other half is flags)
  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_LL]/NCCL_STEPS/2;
  }

  // 8 bytes data per 16-byte line
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t);
  }
};
```

### 6.3 LL128 Protocol

**Description**: Optimized for medium messages.

**Characteristics**:
- Balance between LL and SIMPLE
- 128-byte cache line optimization
- Better bandwidth than LL
- Lower latency than SIMPLE for medium sizes

**Implementation** (`src/device/primitives.h`):
```cpp
struct ProtoLL128 {
  static constexpr int Id = NCCL_PROTO_LL128;

  // Data portion of 128-byte lines
  __device__ static int calcBytePerStep() {
    return (ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS) *
           NCCL_LL128_DATAELEMS/NCCL_LL128_LINEELEMS;
  }

  __device__ static int calcBytePerGrain() {
    return NCCL_LL128_SHMEM_ELEMS_PER_THREAD * NCCL_LL128_DATAELEMS *
           sizeof(uint64_t)/NCCL_LL128_LINEELEMS;
  }
};
```

### 6.4 Protocol Selection

**Selection Criteria**:
```cpp
// Based on message size and hardware
ncclResult_t ncclSelectProtocol(size_t nBytes, int algorithm, int* protocol) {
  // LL for very small messages (< 4KB typically)
  // LL128 for small-medium messages
  // SIMPLE for larger messages
}
```

**Protocol Comparison**:
```
┌─────────────────────────────────────────────────────────────┐
│                 Protocol Comparison                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Latency                                                    │
│   ▲                                                          │
│   │   ┌───────┐                                             │
│   │   │  LL   │                                             │
│   │   └───────┘                                             │
│   │         ┌───────┐                                       │
│   │         │ LL128 │                                       │
│   │         └───────┘                                       │
│   │               ┌───────┐                                 │
│   │               │SIMPLE │                                 │
│   │               └───────┘                                 │
│   └────────────────────────────────────────────► Size       │
│       0    4KB    16KB    64KB    256KB    1MB+             │
│                                                              │
│   Recommendation:                                            │
│   - < 4KB: Use LL protocol                                  │
│   - 4KB - 64KB: Use LL128 protocol                          │
│   - > 64KB: Use SIMPLE protocol                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Data Flow from High to Low Level

### 7.1 Complete AllReduce Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Complete AllReduce Data Flow                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. User API Call                                                   │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm)│     │
│  └────────────────────────────────────────────────────────────┘     │
│         │                                                            │
│         ▼                                                            │
│  2. Host-side Planning                                              │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ ncclGroupStart/End handling                                 │     │
│  │ Algorithm selection (RING/TREE/NVLS/COLLNET)                │     │
│  │ Protocol selection (SIMPLE/LL/LL128)                        │     │
│  │ Channel assignment                                          │     │
│  │ Task creation and scheduling                                │     │
│  └────────────────────────────────────────────────────────────┘     │
│         │                                                            │
│         ▼                                                            │
│  3. Kernel Launch                                                   │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ ncclLaunchKernel(comm, work)                                │     │
│  │   - Allocate work descriptors                               │     │
│  │   - Setup kernel arguments                                  │     │
│  │   - Launch CUDA kernel                                      │     │
│  └────────────────────────────────────────────────────────────┘     │
│         │                                                            │
│         ▼                                                            │
│  4. Device Kernel Execution                                         │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ RunWorkColl<ncclFuncAllReduce, T, RedOp, ALGO, PROTO>      │     │
│  │   - runRing() / runTreeSplit() / runNvls()                 │     │
│  │   - Call Primitives operations                              │     │
│  └────────────────────────────────────────────────────────────┘     │
│         │                                                            │
│         ▼                                                            │
│  5. Primitives Layer                                                │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ Primitives<T, RedOp, Fan, Direct, Proto, P2p>              │     │
│  │   - directSend() / directRecv()                            │     │
│  │   - directRecvReduceDirectSend()                           │     │
│  │   - send() / recv()                                        │     │
│  │   - Wait/Post synchronization                              │     │
│  └────────────────────────────────────────────────────────────┘     │
│         │                                                            │
│         ▼                                                            │
│  6. Transport Selection                                             │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ Check transport type:                                       │     │
│  │   - P2P: Direct GPU-to-GPU (NVLink/PCIe)                   │     │
│  │   - SHM: Shared memory (same node, different process)       │     │
│  │   - NET: Network (different node)                          │     │
│  │   - NVLS: NVSwitch hardware collective                     │     │
│  └────────────────────────────────────────────────────────────┘     │
│         │                                                            │
│         ├────────────────┬────────────────┬────────────────┐        │
│         ▼                ▼                ▼                ▼        │
│  7. Hardware Operations                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐ │
│  │   P2P Path   │ │   SHM Path   │ │   NET Path   │ │  NVLS Path │ │
│  │              │ │              │ │              │ │            │ │
│  │ NVLink DMA   │ │ POSIX SHM    │ │ GIN/Proxy    │ │ HW Mcast   │ │
│  │ or PCIe P2P  │ │ or CUDA IPC  │ │ over IB/RoCE │ │ + Reduce   │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └────────────┘ │
│         │                │                │                │        │
│         └────────────────┴────────────────┴────────────────┘        │
│         │                                                            │
│         ▼                                                            │
│  8. Completion                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ - All chunks processed                                      │     │
│  │ - Results in recvbuff                                       │     │
│  │ - Kernel completion signaled                                │     │
│  │ - Stream synchronization                                    │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Intra-node vs Inter-node Communication

```
┌─────────────────────────────────────────────────────────────────────┐
│              Intra-node Communication (Same Node)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Preferred: P2P Transport (NVLink/PCIe)                            │
│                                                                      │
│   ┌─────────┐   NVLink   ┌─────────┐                               │
│   │  GPU 0  │◄══════════►│  GPU 1  │                               │
│   └─────────┘            └─────────┘                               │
│        ▲                      ▲                                     │
│        │ PCIe                 │ PCIe                                │
│        ▼                      ▼                                     │
│   ┌─────────┐   NVLink   ┌─────────┐                               │
│   │  GPU 2  │◄══════════►│  GPU 3  │                               │
│   └─────────┘            └─────────┘                               │
│                                                                      │
│   Alternative: SHM Transport (if P2P unavailable)                   │
│   - Cross-process communication                                     │
│   - POSIX shared memory or CUDA IPC                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│              Inter-node Communication (Different Nodes)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   NET Transport (InfiniBand/RoCE)                                   │
│                                                                      │
│   Node 1                              Node 2                        │
│   ┌─────────────────┐                ┌─────────────────┐           │
│   │ ┌─────┐ ┌─────┐ │                │ ┌─────┐ ┌─────┐ │           │
│   │ │GPU 0│ │GPU 1│ │                │ │GPU 0│ │GPU 1│ │           │
│   │ └──┬──┘ └──┬──┘ │                │ └──┬──┘ └──┬──┘ │           │
│   │    │       │    │                │    │       │    │           │
│   │    ▼       ▼    │                │    ▼       ▼    │           │
│   │   ┌─────────┐   │    Network     │   ┌─────────┐   │           │
│   │   │   NIC   │═══│════════════════│═══│   NIC   │   │           │
│   │   └─────────┘   │   IB/RoCE      │   └─────────┘   │           │
│   └─────────────────┘                └─────────────────┘           │
│                                                                      │
│   Communication modes:                                              │
│   - Proxy mode: CPU handles network operations                      │
│   - GDAKI mode: GPU directly initiates RDMA (zero-copy)            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 Performance Considerations

**Bandwidth Utilization**:
```
Effective Bandwidth = min(Hardware BW, Algorithm Efficiency)

Where:
- Hardware BW: NVLink > PCIe > Network
- Algorithm Efficiency: NVLS > TREE > RING (for small n)
                     RING > TREE (for large n)
```

**Latency Optimization**:
```
Total Latency = α + n/β

Where:
- α: Latency per message
- β: Bandwidth (bytes/second)
- n: Message size

For small n: Use LL/LL128 protocol
For large n: Use SIMPLE protocol
```

**Optimal Algorithm Selection**:
```cpp
// Approximate guidance based on topology and size
if (hasNVLS && nBytes > threshold) {
  use NVLS;
} else if (nBytes < small_threshold || nranks <= 4) {
  use TREE;
} else {
  use RING;
}
```

---

## Appendix A: Key Source Files

| File | Description |
|------|-------------|
| `src/collectives.cc` | Host-side collective API |
| `src/enqueue.cc` | Task enqueue and planning |
| `src/device/all_reduce.h` | AllReduce kernel implementations |
| `src/device/primitives.h` | Communication primitives |
| `src/device/prims_simple.h` | SIMPLE protocol primitives |
| `src/device/prims_ll.h` | LL protocol primitives |
| `src/device/prims_ll128.h` | LL128 protocol primitives |
| `src/transport/p2p.cc` | P2P transport implementation |
| `src/transport/shm.cc` | SHM transport implementation |
| `src/transport/net.cc` | Network transport implementation |
| `src/transport/nvls.cc` | NVLS transport implementation |
| `src/include/nccl_device/gin.h` | GIN device API |
| `src/rma/rma.cc` | RMA implementation |

## Appendix B: Debug Environment Variables

```bash
# Enable debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL  # or: COLL, NET, P2P, GIN

# Algorithm selection
export NCCL_ALGO=RING  # Force specific algorithm
export NCCL_PROTO=SIMPLE  # Force specific protocol

# GIN configuration
export NCCL_GIN_TYPE=3  # GDAKI (GPU-initiated)
export NCCL_GIN_TYPE=2  # PROXY mode

# Network configuration
export NCCL_NET_PLUGIN=ib  # Network plugin
export NCCL_NET_GDR_LEVEL=5  # GPUDirect RDMA level

# Buffer sizes
export NCCL_BUFFSIZE=4194304  # Buffer size in bytes
```

---

*Documentation Version: 1.0*
*Based on NCCL v2.29.7-1*
*Generated: 2026-03-21*
