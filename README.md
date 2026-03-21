# NCCL 超时检测实现总结

## 项目概述

本项目为NCCL（NVIDIA Collective Communications Library）添加了一个统一的通信超时检测机制，支持监测各种通信路径（Network、RDMA、TCP、Shared Memory、P2P）的超时情况，通过环境变量配置，超时发生时打印详细日志。

## 文档清单

| 文档 | 描述 |
|-----|------|
| `nccl_timeout_detection_guide.md` | 完整的实现指南，包含架构分析、代码修改详解 |
| `nccl_timeout_patches.patch` | 具体的代码补丁文件（diff格式） |
| `usage_guide.md` | 使用指南和测试示例 |
| `architecture.md` | 详细的架构图和流程图 |
| `FAQ.md` | 常见问题解答 |
| `README.md` | 快速入门指南（本文档） |

## 快速开始

### 1. 应用修改

```bash
cd /root/workspace/nccl

# 方法1: 手动应用补丁
git apply /root/workspace/total_docs/nccl_timeout_patches.patch

# 方法2: 手动修改（参考nccl_timeout_detection_guide.md）
```

### 2. 编译NCCL

```bash
cd /root/workspace/nccl
make clean
make -j$(nproc)
```

### 3. 配置环境变量

```bash
export NCCL_PROXY_TIMEOUT_MS=30000      # 30秒超时
export NCCL_PROXY_TIMEOUT_LOG_ONLY=1     # 仅打印日志
export NCCL_PROXY_TIMEOUT_CHECK_INTERVAL=1000  # 1秒检查间隔
```

### 4. 运行程序

```bash
mpirun -np 4 ./your_nccl_program
```

## 核心修改内容

### 新增环境变量

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `NCCL_PROXY_TIMEOUT_MS` | 0 | 超时时间（毫秒），0表示禁用 |
| `NCCL_PROXY_TIMEOUT_LOG_ONLY` | 1 | 1=仅打印日志，0=超时报错 |
| `NCCL_PROXY_TIMEOUT_CHECK_INTERVAL` | 1000 | 检查间隔（毫秒） |

### 修改的文件

1. **src/include/proxy.h**
   - 添加超时检测字段到 `ncclProxyArgs` 和 `ncclProxySubArgs`
   - 添加时间工具函数和配置结构体

2. **src/proxy.cc**
   - 添加环境变量参数定义
   - 实现超时检测核心函数
   - 修改 `progressOps()` 添加超时检查逻辑

3. **src/transport/net.cc**
   - 在 `sendProxyProgress()` 和 `recvProxyProgress()` 中添加活跃时间更新

4. **src/transport/p2p.cc**
   - 在 `p2pSendProxyProgress()` 中添加活跃时间更新

5. **src/transport/shm.cc**
   - 在 `shmSendProxyProgress()` 和 `shmRecvProxyProgress()` 中添加活跃时间更新

## 超时日志示例

```
NCCL WARN NCCL Proxy Timeout Detected! Transport: NET, Stage: TRANSMITTING, 
Sub: 0/4, Peer: 1, Channel: 0, 
Progress: posted=4 transmitted=2 received=0 done=0 nsteps=8, 
Elapsed: 30012 ms (stage), 60045 ms (total), Timeout: 30000 ms
```

## 性能影响

| 场景 | 开销 |
|-----|------|
| 超时禁用 (0) | 无 |
| 超时启用 | < 0.1%（每次检查间隔） |

## 支持的通信路径

- ✅ **NET**: RDMA/IB、RoCE、TCP、GDRDMA
- ✅ **P2P**: GPU Direct (NVLink、PCIe)
- ✅ **SHM**: Shared Memory
- ✅ **COLLNET**: NVLink SHARP

## 测试验证

参考 `usage_guide.md` 中的测试程序：
- `test_basic.cu` - 基本功能测试
- `test_delay.cu` - 延迟注入测试
- `test_stress.cu` - 压力测试

## 故障排查

### 没有看到超时日志

1. 检查NCCL是否正确编译
2. 确认环境变量已设置
3. 启用调试模式：`NCCL_DEBUG=INFO`
4. 确认超时时间足够长

### 误报超时

1. 增加超时时间：`NCCL_PROXY_TIMEOUT_MS=60000`
2. 增加检查间隔：`NCCL_PROXY_TIMEOUT_CHECK_INTERVAL=10000`
3. 检查系统负载

## 进一步阅读

- 详细实现指南：`nccl_timeout_detection_guide.md`
- 架构设计：`architecture.md`
- 常见问题：`FAQ.md`
- 使用示例：`usage_guide.md`

## 注意事项

1. 超时检测默认禁用，需要显式启用
2. 超时日志可能重复打印，这是设计行为
3. 检查间隔避免设置过短（< 100ms）
4. 生产环境建议使用较长的超时时间（30-60秒）

## 联系方式

如有问题，请参考FAQ文档或查看NCCL官方文档。

---

*文档版本: 1.0*
*最后更新: 2026-03-19*
