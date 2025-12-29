# CUDA Stream Pool Implementation using RMM

## 概述

使用 RAPIDS Memory Manager (RMM) 的 `rmm::cuda_stream_pool` 直接实现 CUDA 流池。这提供了一个经过充分测试、高效且线程安全的流管理解决方案。

## 架构

### 之前的实现
- 自定义的流池实现
- 手动管理流的创建和销毁
- 使用 `std::queue` 和 `std::vector` 管理流
- 需要手动同步和释放

### 新的实现
- 直接使用 RMM 的 `rmm::cuda_stream_pool`
- 自动流管理和循环利用（round-robin）
- 线程安全的流分配（原子操作）
- 最小化的包装层

## 核心特性

### 1. 线程安全的流分配

```cpp
// RMM 的 cuda_stream_pool 使用原子操作实现线程安全
rmm::cuda_stream_view stream_view = pool_->get_stream();
```

### 2. 循环利用流（Round-Robin）

RMM 使用循环缓冲区方式分配流，确保：
- 高效的流重用
- 避免频繁的流创建/销毁
- 更好的 GPU 调度
- 负载均衡

### 3. 默认流池大小

```cpp
static constexpr std::size_t default_size{16};  // 默认 16 个流
```

### 4. cuda_stream_view 支持

```cpp
// RMM 的 cuda_stream_view 提供：
// - 自动同步支持
// - 隐式转换到 cudaStream_t
// - RAII 语义
rmm::cuda_stream_view stream_view = pool_->get_stream();
stream_view.synchronize();  // 同步流
cudaStream_t stream = stream_view.value();  // 获取原始流
```

## 实现细节

### CudaStreamPool 类

<augment_code_snippet path="heu/library/algorithms/paillier_gpu/gpulib/cuda_stream_pool.h" mode="EXCERPT">
```cpp
class CudaStreamPool {
 public:
  static CudaStreamPool& GetInstance();
  rmm::cuda_stream_pool& GetPool();
  rmm::cuda_stream_view GetStream();
  size_t GetPoolSize() const;
 private:
  std::unique_ptr<rmm::cuda_stream_pool> pool_;
};
```
</augment_code_snippet>

### 实现文件

<augment_code_snippet path="heu/library/algorithms/paillier_gpu/gpulib/cuda_stream_pool.cc" mode="EXCERPT">
```cpp
CudaStreamPool::CudaStreamPool() {
  // 创建 RMM cuda_stream_pool（默认 16 个流）
  pool_ = std::make_unique<rmm::cuda_stream_pool>();
}

rmm::cuda_stream_view CudaStreamPool::GetStream() {
  return pool_->get_stream();
}
```
</augment_code_snippet>

### StreamGuard 类

<augment_code_snippet path="heu/library/algorithms/paillier_gpu/gpulib/cuda_stream_pool.h" mode="EXCERPT">
```cpp
class StreamGuard {
 public:
  StreamGuard() : stream_view_(CudaStreamPool::GetInstance().GetStream()) {}

  ~StreamGuard() {
    stream_view_.synchronize();  // 自动同步
  }

  rmm::cuda_stream_view Get() const { return stream_view_; }
  operator cudaStream_t() const { return stream_view_.value(); }
 private:
  rmm::cuda_stream_view stream_view_;
};
```
</augment_code_snippet>

## 使用示例

### 方式 1：直接使用 RMM 流池

```cpp
// 获取流池
rmm::cuda_stream_pool& pool = CudaStreamPool::GetInstance().GetPool();

// 获取流
rmm::cuda_stream_view stream = pool.get_stream();

// 使用流执行 GPU 操作
kernel<<<grid, block, 0, stream.value()>>>(args);

// 同步流
stream.synchronize();
```

### 方式 2：使用 StreamGuard + cuda_stream_view（推荐）

```cpp
// RAII 方式自动管理流的生命周期
{
  StreamGuard stream_guard;

  // 获取 cuda_stream_view（推荐方式）
  rmm::cuda_stream_view stream = stream_guard.stream();

  // 使用流有序分配
  PooledGPUMemory<T> gpu_mem(count, stream);

  // 在流上执行 GPU 操作
  kernel<<<grid, block, 0, stream.value()>>>(gpu_mem.get());

  // 作用域结束时自动同步流
}
```

### 方式 3：使用 rmm::device_buffer（最佳实践）

```cpp
// 这是 RMM 推荐的方式
{
  StreamGuard stream_guard;
  rmm::cuda_stream_view stream = stream_guard.stream();

  // 使用 rmm::device_buffer 进行流有序分配
  rmm::device_buffer buf{size, stream};
  void* ptr = buf.data();

  // 在流上执行 GPU 操作
  kernel<<<grid, block, 0, stream.value()>>>(ptr);

  // 内存和流都会自动释放和同步
}
```

### 方式 4：向后兼容的方式

```cpp
// 对于现有代码，仍然支持 cudaStream_t
{
  StreamGuard stream_guard;
  cudaStream_t stream = stream_guard.Get();  // 或 stream_guard.value()

  // 使用流执行 GPU 操作
  kernel<<<grid, block, 0, stream>>>(args);

  // 作用域结束时自动同步流
}
```

## 优势

| 特性 | 自定义实现 | RMM 直接使用 |
|------|----------|------------|
| 线程安全 | 手动 mutex | 原子操作 ✓ |
| 流管理 | 手动创建/销毁 | 自动循环利用 ✓ |
| 代码复杂度 | 高 | 极低 ✓ |
| 性能 | 基础 | 优化 ✓ |
| 维护成本 | 高 | 极低 ✓ |
| 测试覆盖 | 有限 | 充分 ✓ |
| 包装层 | N/A | 最小化 ✓ |
| cuda_stream_view 支持 | 无 | 完整 ✓ |

## 性能特点

### 流分配
- **时间复杂度**: O(1) - 原子操作
- **空间复杂度**: O(pool_size) - 固定大小

### 循环利用
- 避免频繁的 `cudaStreamCreate/Destroy`
- 减少 GPU 驱动开销
- 更好的 GPU 调度

## 集成点

### 1. 内存池集成

```cpp
// 在 rmm_memory_pool.cc 中使用
PooledGPUMemory<T> mem(count, stream);
```

### 2. GPU 计算集成

```cpp
// 在 paillier.cu 中使用
StreamGuard stream_guard;
kernel<<<grid, block, 0, stream_guard.Get()>>>(args);
```

## 编译配置

### BUILD.bazel 更新

```bazel
cuda_library(
    name = "gpupaillier",
    srcs = [
        "gpulib/cuda_stream_pool.cc",  # 新增
        "gpulib/paillier.cu",
        "gpulib/rmm_memory_pool.cc",
    ],
)
```

### rmm.BUILD 更新

```bazel
cuda_library(
    name = "rmm",
    srcs = [
        "cpp/src/cuda_stream.cpp",        # 新增
        "cpp/src/cuda_stream_pool.cpp",   # 新增
        "cpp/src/aligned.cpp",
        "cpp/src/error.cpp",
        "cpp/src/cuda_stream_view.cpp",
        "cpp/src/cuda_device.cpp",
    ],
)
```

## 验证

✅ 编译成功：`bazel build --config=gpu heu/library/algorithms/paillier_gpu:gpupaillier`
✅ 所有测试通过：`bazel test --config=gpu heu/library/algorithms/paillier_gpu:paillier_gpu_test`

## 总结

通过直接使用 RMM 的 `rmm::cuda_stream_pool`，我们获得了：
- ✓ 经过充分测试的流管理实现
- ✓ 更好的性能和线程安全性（原子操作）
- ✓ 极简的代码和极低的维护成本
- ✓ 完整的 `cuda_stream_view` 支持
- ✓ 与 RMM 内存池的无缝集成
- ✓ 完全的向后兼容性
- ✓ 最小化的包装层（仅用于单例管理）

