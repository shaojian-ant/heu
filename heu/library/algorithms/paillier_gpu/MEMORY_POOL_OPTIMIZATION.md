# Paillier GPU 内存池优化文档

## 📋 优化概述

本文档记录了对 Paillier GPU 实现的内存池优化工作。

### 优化目标
- 减少频繁的 `cudaMalloc`/`cudaFree` 调用开销
- 提升批量操作的性能
- 保持代码简洁和易维护性

### 优化范围
- **只保留**：GPU 内存池优化
- **修改的函数**（10个 - 全部完成！）：
  1. `gpu_paillier_enc`：加密函数
  2. `gpu_paillier_dec`：解密函数
  3. `gpu_paillier_e_add`：密文加法（CT+CT）
  4. `gpu_paillier_e_add_const`：密文+明文（CT+PT）
  5. `gpu_paillier_sub_ct`：密文减法（CT-CT）
  6. `gpu_paillier_e_mul_const`：密文*明文（CT*PT）
  7. `gpu_paillier_e_inverse`：密文取反
  8. `gpu_paillier_sub_ctpt`：CT-PT 减法
  9. `gpu_paillier_sub_ptct`：PT-CT 减法
  10. `gpu_paillier_compare`：比较操作

---

## 🎯 性能提升结果

### 测试环境
- **GPU**: NVIDIA GPU
- **CUDA**: 11.x
- **构建系统**: Bazel
- **测试日期**: 2025-10-28

### 性能对比（与初始版本）

| 操作 | 初始版本 (ms) | 优化后 (ms) | 性能提升 |
|------|------------|-----------|---------|
| **Encrypt(2048,101)** | 5235 | 5179 | 1.1% ⬆️ |
| **Encrypt(512,512)** | 6978 | 6900 | 1.1% ⬆️ |
| **AddCipher(2048,101)** | 177 | 115 | **35.0%** ⬆️ 🎉 |
| **AddCipher(512,512)** | 196 | 150 | **23.5%** ⬆️ 🎉 |
| **AddCipher(101)** | 12.2 | 8.81 | **27.8%** ⬆️ 🎉 |
| **AddCipher(2048)** | 14.9 | 11.6 | **22.1%** ⬆️ 🎉 |
| **SubCipher(2048,101)** | 374 | 335 | **10.4%** ⬆️ ✅ |
| **SubCipher(512,512)** | 514 | 459 | **10.7%** ⬆️ ✅ |
| **AddInt(2048,101)** | 282 | 232 | **17.7%** ⬆️ ✅ |
| **AddInt(512,512)** | 358 | 313 | **12.6%** ⬆️ ✅ |
| **SubInt(2048,101)** | 320 | 266 | **16.9%** ⬆️ ✅ |
| **SubInt(512,512)** | 404 | 362 | **10.4%** ⬆️ ✅ |
| **Matmul((2048,101)@(101))** | 52898 | 25547 | **51.7%** ⬆️ 🎉🎉🎉 |
| **Matmul((2048)@(2048,101))** | 48576 | 26689 | **45.1%** ⬆️ 🎉🎉🎉 |

### 关键发现

#### ✅ **最显著提升的操作（> 40%）**
- **Matmul 操作**：提升 45-52%（最显著！）
- **原因**：Matmul 内部调用的所有函数都使用内存池，消除了所有内存分配瓶颈

#### ✅ **显著提升的操作（20-40%）**
- **AddCipher 操作**：提升 22-35%
- **原因**：内存池消除了频繁的 cudaMalloc/cudaFree 开销

#### ✅ **中等提升的操作（10-20%）**
- **SubCipher 操作**：提升 10-11%
- **AddInt 操作**：提升 13-18%
- **SubInt 操作**：提升 10-17%（从 2-3% 大幅提升！）
- **原因**：所有被调用的函数都使用内存池

#### ⚠️ **提升不明显的操作（< 10%）**
- **Encrypt 操作**：提升约 1%
- **原因**：加密操作的瓶颈在计算（模幂运算）而非内存分配

---

## 🔧 技术实现

### 1. GPU 内存池设计

#### 核心组件

**文件**: `heu/library/algorithms/paillier_gpu/gpulib/gpu_memory_pool.h`

```cpp
class GPUMemoryPool {
 public:
  // 单例模式
  static GPUMemoryPool& GetInstance();
  
  // 分配内存（从池中获取或新分配）
  void* Allocate(size_t size);
  
  // 释放内存（返回到池中）
  void Deallocate(void* ptr);
  
  // 清空内存池
  void Clear();
  
 private:
  GPUMemoryPool();
  ~GPUMemoryPool();
  
  std::mutex mutex_;  // 线程安全
  std::map<size_t, std::vector<void*>> free_blocks_;  // 空闲块
  std::map<void*, size_t> allocated_blocks_;  // 已分配块
  size_t total_allocated_;  // 总分配量
};
```

#### RAII 包装器

```cpp
template<typename T>
class PooledGPUMemory {
 public:
  explicit PooledGPUMemory(size_t count);
  ~PooledGPUMemory();
  
  T* get();
  operator T*();
  
  // 禁止拷贝
  PooledGPUMemory(const PooledGPUMemory&) = delete;
  PooledGPUMemory& operator=(const PooledGPUMemory&) = delete;
  
 private:
  T* ptr_;
  size_t size_;
};
```

### 2. 代码修改示例

#### 修改前（使用 cudaMalloc/cudaFree）

```cpp
int gpu_paillier_enc(h_paillier_ciphertext_t *res, h_paillier_pubkey_t *pub,
                     h_paillier_plaintext_t *pt, h_paillier_random_t *rand,
                     unsigned int count) {
  gpu_paillier_ciphertext_t *gpu_result;
  gpu_paillier_pubkey_t *gpu_pub;
  gpu_paillier_plaintext_t *gpu_pt;
  gpu_paillier_random_t *gpu_random;
  
  // 分配内存
  CUDA_CHECK(cudaMalloc((void **)&gpu_result, 
                        sizeof(gpu_paillier_ciphertext_t) * count));
  CUDA_CHECK(cudaMalloc((void **)&gpu_pub, sizeof(gpu_paillier_pubkey_t)));
  CUDA_CHECK(cudaMalloc((void **)&gpu_pt, 
                        sizeof(gpu_paillier_plaintext_t) * count));
  CUDA_CHECK(cudaMalloc((void **)&gpu_random, 
                        sizeof(gpu_paillier_random_t) * count));
  
  // ... 使用内存 ...
  
  // 释放内存
  CUDA_CHECK(cudaFree(gpu_result));
  CUDA_CHECK(cudaFree(gpu_pub));
  CUDA_CHECK(cudaFree(gpu_pt));
  CUDA_CHECK(cudaFree(gpu_random));
  
  return 0;
}
```

#### 修改后（使用内存池）

```cpp
int gpu_paillier_enc(h_paillier_ciphertext_t *res, h_paillier_pubkey_t *pub,
                     h_paillier_plaintext_t *pt, h_paillier_random_t *rand,
                     unsigned int count) {
  // OPTIMIZATION: Use memory pool to reduce allocation overhead
  PooledGPUMemory<gpu_paillier_ciphertext_t> gpu_result(count);
  PooledGPUMemory<gpu_paillier_pubkey_t> gpu_pub(1);
  PooledGPUMemory<gpu_paillier_plaintext_t> gpu_pt(count);
  PooledGPUMemory<gpu_paillier_random_t> gpu_random(count);
  
  if (!gpu_result.get() || !gpu_pub.get() || !gpu_pt.get() || !gpu_random.get()) {
    return -1;  // Allocation failed
  }
  
  // ... 使用内存 ...
  
  // 内存池会在对象析构时自动返还内存，无需手动释放
  return 0;
}
```

### 3. 修改的文件

#### 新增文件
1. **heu/library/algorithms/paillier_gpu/gpulib/gpu_memory_pool.h**
   - GPU 内存池实现（线程安全的单例）
   - RAII 包装器 `PooledGPUMemory<T>`

#### 修改文件
1. **heu/library/algorithms/paillier_gpu/gpulib/paillier.cu**
   - 添加内存池头文件引用
   - 修改 10 个函数使用内存池（全部函数！）：
     - `gpu_paillier_enc`（加密）
     - `gpu_paillier_dec`（解密）
     - `gpu_paillier_e_add`（CT+CT）
     - `gpu_paillier_e_add_const`（CT+PT）
     - `gpu_paillier_sub_ct`（CT-CT）
     - `gpu_paillier_e_mul_const`（CT*PT）
     - `gpu_paillier_e_inverse`（密文取反）
     - `gpu_paillier_sub_ctpt`（CT-PT）
     - `gpu_paillier_sub_ptct`（PT-CT）
     - `gpu_paillier_compare`（比较）

2. **heu/library/algorithms/paillier_gpu/BUILD.bazel**
   - 添加 `gpu_memory_pool.h` 到头文件列表

---

## 📊 性能分析

### 为什么内存池有效？

#### 1. **消除分配/释放开销**
- `cudaMalloc` 和 `cudaFree` 是昂贵的操作
- 内存池复用已分配的内存，避免重复调用

#### 2. **对频繁调用的操作效果显著**
- **Matmul 操作**：内部大量调用 AddCipher
- **AddCipher 操作**：每次调用都需要分配多个 GPU 内存块
- 内存池使得这些频繁调用的开销大幅降低

#### 3. **对计算密集型操作效果不明显**
- **Encrypt 操作**：瓶颈在模幂运算，内存分配开销占比很小
- 内存池优化无法改善计算瓶颈

### 性能提升公式

```
性能提升 = (内存分配开销 / 总执行时间) × 100%
```

- **AddCipher**：内存分配开销占比高 → 提升显著（25-40%）
- **Matmul**：大量调用 AddCipher → 提升最显著（42-44%）
- **Encrypt**：计算开销占主导 → 提升不明显（< 1%）

---

## ✅ 优点与局限性

### 优点
1. **代码简洁**：使用 RAII 包装器，自动管理内存
2. **显著提升**：Matmul 提升 45-52%，AddCipher 提升 22-35%
3. **全面覆盖**：10个函数全部使用内存池（100% 覆盖！）
4. **易于维护**：清晰的优化策略，易于理解和扩展
5. **线程安全**：使用 `std::mutex` 保护共享数据结构
6. **SubInt 性能大幅改善**：从 2-3% 提升到 10-17%
7. **Matmul 性能进一步提升**：从 43.8% 提升到 51.7%

### 局限性
1. **加密操作提升不明显**：瓶颈在计算而非内存分配（提升约 1%）
2. **内存占用**：内存池会缓存已分配的内存，可能增加内存占用

---

## 🎯 建议与未来优化方向

### 当前建议
- **保留内存池优化**：所有操作都受益，特别是 Matmul（提升 45-52%）和 AddCipher（提升 22-35%）
- **监控内存使用**：定期调用 `GPUMemoryPool::Clear()` 释放缓存的内存

### 未来优化方向

#### 1. **内存池优化已完成**
- ✅ 所有 10 个函数都已使用内存池优化（100% 覆盖！）

#### 2. **针对加密操作的优化**
- **常量内存**：将公钥存储在常量内存中
- **CUDA 流**：使用异步操作重叠计算和内存传输
- **共享内存**：优化 `fixed_window_powm_odd` 函数

#### 3. **Kernel 融合**
- 减少 kernel 启动开销
- 预期收益：10-15% 的性能提升

#### 4. **多流并发**
- 对于超大批量数据使用多个流并发处理
- 预期收益：15-20% 的性能提升（仅限超大批量）

---

## 📝 测试方法

### 运行基准测试

```bash
# 编译并运行基准测试
bazel run -c opt --config=gpu heu/library/benchmark:np -- --schema=gpaillier
```

### 性能对比

```bash
# 保存测试结果
bazel run -c opt --config=gpu heu/library/benchmark:np -- --schema=gpaillier 2>&1 | tee benchmark_results.txt

# 对比不同版本的性能
diff baseline_results.txt optimized_results.txt
```

---

## 📚 参考资料

### CUDA 编程
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### 内存管理
- [CUDA Memory Management](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [Memory Pool Design Patterns](https://en.wikipedia.org/wiki/Memory_pool)

### Paillier 加密
- [CGBN Library](https://github.com/NVlabs/CGBN)
- [Paillier Cryptosystem](https://en.wikipedia.org/wiki/Paillier_cryptosystem)

