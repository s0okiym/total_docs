# NCCL 超时检测使用指南

## 快速开始

### 1. 编译带超时检测的NCCL

```bash
cd /root/workspace/nccl

# 应用修改（手动或使用补丁）
# 然后编译
make clean
make -j$(nproc)

# 安装（可选）
make install PREFIX=/usr/local/nccl-timeout
```

### 2. 环境变量配置

```bash
# 启用30秒超时检测
export NCCL_PROXY_TIMEOUT_MS=30000

# 仅打印日志（不中断程序）
export NCCL_PROXY_TIMEOUT_LOG_ONLY=1

# 每5秒检查一次
export NCCL_PROXY_TIMEOUT_CHECK_INTERVAL=5000

# 启用NCCL调试信息（可选）
export NCCL_DEBUG=INFO
```

### 3. 运行程序

```bash
# 使用mpirun运行
mpirun -np 4 ./my_nccl_application

# 或使用torchrun
NCCL_PROXY_TIMEOUT_MS=30000 torchrun --nproc_per_node=4 my_script.py
```

---

## 示例测试程序

### 示例1: 基本超时测试 (test_basic.cu)

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CHECK_NCCL(call) \
    do { \
        ncclResult_t err = call; \
        if (err != ncclSuccess) { \
            fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__, \
                    ncclGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

int main(int argc, char* argv[]) {
    int rank, size;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set device
    int device = rank % 4;  // Assuming 4 GPUs per node
    CHECK_CUDA(cudaSetDevice(device));
    
    // Get NCCL unique ID
    ncclUniqueId id;
    if (rank == 0) {
        CHECK_NCCL(ncclGetUniqueId(&id));
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Initialize NCCL
    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, size, id, rank));
    
    // Allocate buffers
    size_t count = 1024 * 1024;  // 1M elements
    float *sendbuf, *recvbuf;
    CHECK_CUDA(cudaMalloc(&sendbuf, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&recvbuf, count * sizeof(float)));
    
    // Initialize data
    CHECK_CUDA(cudaMemset(sendbuf, rank, count * sizeof(float)));
    CHECK_CUDA(cudaMemset(recvbuf, 0, count * sizeof(float)));
    
    // Synchronize
    CHECK_CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Starting NCCL AllReduce with timeout detection...\n");
    }
    
    // Run AllReduce
    CHECK_NCCL(ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, cudaStreamDefault));
    
    // Synchronize
    CHECK_CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("AllReduce completed successfully!\n");
    }
    
    // Cleanup
    CHECK_NCCL(ncclCommDestroy(comm));
    CHECK_CUDA(cudaFree(sendbuf));
    CHECK_CUDA(cudaFree(recvbuf));
    
    MPI_Finalize();
    return 0;
}
```

编译运行：
```bash
nvcc -o test_basic test_basic.cu -lnccl -lmpi -lcudart
NCCL_PROXY_TIMEOUT_MS=30000 mpirun -np 4 ./test_basic
```

---

### 示例2: 延迟注入测试 (test_delay.cu)

此测试在特定rank中注入延迟，模拟通信超时场景。

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CHECK_NCCL(call) \
    do { \
        ncclResult_t err = call; \
        if (err != ncclSuccess) { \
            fprintf(stderr, "NCCL error: %s\n", ncclGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// 模拟延迟的kernel
__global__ void delay_kernel(int seconds) {
    clock_t start = clock64();
    clock_t wait = seconds * CLOCKS_PER_SEC;
    while (clock64() - start < wait) {}
}

int main(int argc, char* argv[]) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int delay_rank = (argc > 1) ? atoi(argv[1]) : 1;
    int delay_seconds = (argc > 2) ? atoi(argv[2]) : 10;
    
    CHECK_CUDA(cudaSetDevice(rank % 4));
    
    ncclUniqueId id;
    if (rank == 0) CHECK_NCCL(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, size, id, rank));
    
    size_t count = 1024 * 1024;
    float *sendbuf, *recvbuf;
    CHECK_CUDA(cudaMalloc(&sendbuf, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&recvbuf, count * sizeof(float)));
    CHECK_CUDA(cudaMemset(sendbuf, rank, count * sizeof(float)));
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Testing with delay on rank %d for %d seconds...\n", delay_rank, delay_seconds);
    }
    
    // 在指定rank上注入延迟
    if (rank == delay_rank) {
        printf("Rank %d: Injecting %d second delay...\n", rank, delay_seconds);
        delay_kernel<<<1, 1>>>(delay_seconds);
        cudaDeviceSynchronize();
        printf("Rank %d: Delay completed\n", rank);
    }
    
    // 执行AllReduce（应该触发超时检测）
    CHECK_NCCL(ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, cudaStreamDefault));
    
    CHECK_CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Test completed!\n");
    }
    
    CHECK_NCCL(ncclCommDestroy(comm));
    CHECK_CUDA(cudaFree(sendbuf));
    CHECK_CUDA(cudaFree(recvbuf));
    
    MPI_Finalize();
    return 0;
}
```

编译运行：
```bash
nvcc -o test_delay test_delay.cu -lnccl -lmpi -lcudart

# 启用10秒超时，在rank 1注入15秒延迟
NCCL_PROXY_TIMEOUT_MS=10000 NCCL_DEBUG=WARN mpirun -np 4 ./test_delay 1 15
```

预期输出（当超时触发时）：
```
NCCL WARN NCCL Proxy Timeout Detected! Transport: NET, Stage: TRANSMITTING, 
Sub: 0/4, Peer: 1, Channel: 0, 
Progress: posted=4 transmitted=2 received=0 done=0 nsteps=8, 
Elapsed: 10012 ms (stage), 10015 ms (total), Timeout: 10000 ms
```

---

### 示例3: 压力测试 (test_stress.cu)

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#define CHECK_CUDA(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } } while(0)
#define CHECK_NCCL(call) do { ncclResult_t err = call; if (err != ncclSuccess) { fprintf(stderr, "NCCL error: %s\n", ncclGetErrorString(err)); exit(1); } } while(0)

int main(int argc, char* argv[]) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int iterations = (argc > 1) ? atoi(argv[1]) : 100;
    size_t min_count = (argc > 2) ? atoi(argv[2]) : 1024;
    size_t max_count = (argc > 3) ? atoi(argv[3]) : (1024 * 1024);
    
    CHECK_CUDA(cudaSetDevice(rank % 4));
    
    ncclUniqueId id;
    if (rank == 0) CHECK_NCCL(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, size, id, rank));
    
    // Allocate max size buffer
    float *sendbuf, *recvbuf;
    CHECK_CUDA(cudaMalloc(&sendbuf, max_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&recvbuf, max_count * sizeof(float)));
    CHECK_CUDA(cudaMemset(sendbuf, 0, max_count * sizeof(float)));
    
    srand(time(NULL) + rank);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Running stress test: %d iterations, count range [%zu, %zu]\n", 
               iterations, min_count, max_count);
    }
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    float total_ms = 0;
    
    for (int i = 0; i < iterations; i++) {
        // Random size
        size_t count = min_count + rand() % (max_count - min_count);
        
        CHECK_CUDA(cudaEventRecord(start, cudaStreamDefault));
        
        CHECK_NCCL(ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, cudaStreamDefault));
        
        CHECK_CUDA(cudaEventRecord(stop, cudaStreamDefault));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        
        if (rank == 0 && (i + 1) % 10 == 0) {
            printf("Iteration %d/%d completed, avg time: %.3f ms\n", 
                   i + 1, iterations, total_ms / (i + 1));
        }
    }
    
    if (rank == 0) {
        printf("Stress test completed! Average time: %.3f ms\n", total_ms / iterations);
    }
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_NCCL(ncclCommDestroy(comm));
    CHECK_CUDA(cudaFree(sendbuf));
    CHECK_CUDA(cudaFree(recvbuf));
    
    MPI_Finalize();
    return 0;
}
```

编译运行：
```bash
nvcc -o test_stress test_stress.cu -lnccl -lmpi -lcudart
NCCL_PROXY_TIMEOUT_MS=5000 mpirun -np 4 ./test_stress 100 1024 1048576
```

---

### 示例4: Python (PyTorch) 测试

```python
import os
import sys
import torch
import torch.distributed as dist
import time

def run_timeout_test():
    # Initialize
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    torch.cuda.set_device(device)
    
    # Create tensors
    size = 1024 * 1024
    tensor = torch.randn(size, device=device)
    
    print(f"Rank {rank}: Starting test on {device}")
    
    # Warmup
    for _ in range(5):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    
    # Test iterations
    for i in range(10):
        start = time.time()
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        
        if rank == 0:
            print(f"Iteration {i}: {elapsed:.2f} ms")
    
    print(f"Rank {rank}: Test completed")
    dist.destroy_process_group()

if __name__ == '__main__':
    run_timeout_test()
```

运行：
```bash
NCCL_PROXY_TIMEOUT_MS=30000 NCCL_DEBUG=INFO torchrun --nproc_per_node=4 test_torch.py
```

---

## 日志解读

### 正常日志（无超时）

```
NCCL INFO NCCL_PROXY_TIMEOUT_MS set to 30000
NCCL INFO NCCL_PROXY_TIMEOUT_LOG_ONLY set to 1
```

### 超时警告日志

```
NCCL WARN NCCL Proxy Timeout Detected! Transport: NET, Stage: TRANSMITTING, 
Sub: 0/4, Peer: 1, Channel: 0, 
Progress: posted=4 transmitted=2 received=0 done=0 nsteps=8, 
Elapsed: 30012 ms (stage), 60045 ms (total), Timeout: 30000 ms
```

日志字段说明：
- **Transport**: 通信路径类型 (NET/SHM/P2P/COLLNET)
- **Stage**: 当前阶段 (POSTING/TRANSMITTING/RECEIVING/WAITING_ACK)
- **Sub**: 子操作索引/总数
- **Peer**: 对端rank
- **Channel**: 通道ID
- **Progress**: 各阶段进度计数
- **Elapsed**: 当前阶段耗时/总耗时
- **Timeout**: 配置的超时时间

---

## 故障排查

### 问题1: 没有看到超时日志

检查：
1. NCCL是否正确编译了超时检测代码
2. 环境变量是否设置正确
3. 检查NCCL_DEBUG=INFO输出中是否包含超时参数

```bash
NCCL_DEBUG=INFO ./program 2>&1 | grep -i "PROXY_TIMEOUT"
```

### 问题2: 超时日志过多

调整检查间隔：
```bash
export NCCL_PROXY_TIMEOUT_CHECK_INTERVAL=10000  # 10秒检查一次
```

### 问题3: 误报超时

如果正常通信也触发超时，可能是：
1. 超时时间设置过短 - 增加 NCCL_PROXY_TIMEOUT_MS
2. 检查间隔过短 - 增加 NCCL_PROXY_TIMEOUT_CHECK_INTERVAL
3. 系统负载过高 - 检查GPU和网络状态

---

## 性能考虑

### 启用超时检测的性能影响

| 配置 | 预期开销 |
|-----|---------|
| 超时禁用 (0) | 无 |
| 超时启用 | 每次检查间隔一次时间获取 |
| 默认检查间隔(1秒) | <0.1% |

### 优化建议

1. 生产环境建议设置合理的超时时间（30-60秒）
2. 调试时可设置较短的超时时间
3. 检查间隔不宜过短，避免频繁系统调用
