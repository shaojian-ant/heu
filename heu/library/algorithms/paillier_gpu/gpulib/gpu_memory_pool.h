#ifndef HEU_LIBRARY_ALGORITHMS_PAILLIER_GPU_GPULIB_GPU_MEMORY_POOL_H_
#define HEU_LIBRARY_ALGORITHMS_PAILLIER_GPU_GPULIB_GPU_MEMORY_POOL_H_

#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <vector>

namespace heu {
namespace lib {
namespace algorithms {
namespace paillier_gpu {

// GPU Memory Pool for reducing cudaMalloc/cudaFree overhead
// Thread-safe singleton implementation
class GPUMemoryPool {
 public:
  static GPUMemoryPool& GetInstance() {
    static GPUMemoryPool instance;
    return instance;
  }

  // Allocate memory from pool
  void* Allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Try to find a free block of sufficient size
    auto it = free_blocks_.lower_bound(size);
    if (it != free_blocks_.end()) {
      void* ptr = it->second.back();
      it->second.pop_back();
      if (it->second.empty()) {
        free_blocks_.erase(it);
      }
      allocated_blocks_[ptr] = size;
      return ptr;
    }
    
    // No suitable block found, allocate new memory
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
      return nullptr;
    }
    
    allocated_blocks_[ptr] = size;
    total_allocated_ += size;
    return ptr;
  }

  // Return memory to pool
  void Deallocate(void* ptr) {
    if (ptr == nullptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
      return;  // Not allocated by this pool
    }
    
    size_t size = it->second;
    allocated_blocks_.erase(it);
    
    // Add to free blocks
    free_blocks_[size].push_back(ptr);
  }

  // Clear all cached memory
  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Free all blocks in free_blocks_
    for (auto& pair : free_blocks_) {
      for (void* ptr : pair.second) {
        cudaFree(ptr);
      }
    }
    free_blocks_.clear();
    
    // Note: We don't free allocated_blocks_ as they are still in use
    total_allocated_ = 0;
    for (const auto& pair : allocated_blocks_) {
      total_allocated_ += pair.second;
    }
  }

  // Get statistics
  size_t GetTotalAllocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_;
  }

  size_t GetFreeBlockCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& pair : free_blocks_) {
      count += pair.second.size();
    }
    return count;
  }

 private:
  GPUMemoryPool() : total_allocated_(0) {}
  ~GPUMemoryPool() { Clear(); }
  
  // Prevent copying
  GPUMemoryPool(const GPUMemoryPool&) = delete;
  GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;

  mutable std::mutex mutex_;
  std::map<size_t, std::vector<void*>> free_blocks_;  // size -> list of free blocks
  std::map<void*, size_t> allocated_blocks_;  // ptr -> size
  size_t total_allocated_;
};

// RAII wrapper for GPU memory pool
template<typename T>
class PooledGPUMemory {
 public:
  explicit PooledGPUMemory(size_t count) : ptr_(nullptr), size_(0) {
    size_ = sizeof(T) * count;
    ptr_ = static_cast<T*>(GPUMemoryPool::GetInstance().Allocate(size_));
  }

  ~PooledGPUMemory() {
    if (ptr_) {
      GPUMemoryPool::GetInstance().Deallocate(ptr_);
    }
  }

  // Prevent copying
  PooledGPUMemory(const PooledGPUMemory&) = delete;
  PooledGPUMemory& operator=(const PooledGPUMemory&) = delete;

  // Allow moving
  PooledGPUMemory(PooledGPUMemory&& other) noexcept
      : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  PooledGPUMemory& operator=(PooledGPUMemory&& other) noexcept {
    if (this != &other) {
      if (ptr_) {
        GPUMemoryPool::GetInstance().Deallocate(ptr_);
      }
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  operator T*() { return ptr_; }
  operator const T*() const { return ptr_; }

 private:
  T* ptr_;
  size_t size_;
};

}  // namespace paillier_gpu
}  // namespace algorithms
}  // namespace lib
}  // namespace heu

#endif  // HEU_LIBRARY_ALGORITHMS_PAILLIER_GPU_GPULIB_GPU_MEMORY_POOL_H_

