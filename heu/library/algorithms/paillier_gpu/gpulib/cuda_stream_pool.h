#ifndef HEU_LIBRARY_ALGORITHMS_PAILLIER_GPU_GPULIB_CUDA_STREAM_POOL_H_
#define HEU_LIBRARY_ALGORITHMS_PAILLIER_GPU_GPULIB_CUDA_STREAM_POOL_H_

#include <cuda_runtime.h>
#include <memory>
#include <rmm/cuda_stream_pool.hpp>

namespace heu {
namespace lib {
namespace algorithms {
namespace paillier_gpu {

// Singleton wrapper for RMM's cuda_stream_pool
// Provides thread-safe access to a pool of CUDA streams
class CudaStreamPool {
 public:
  static CudaStreamPool& GetInstance();

  // Initialize the stream pool with a specific size
  // This should be called before any GPU operations
  // Recommended: set pool_size >= number of concurrent threads
  // If not called, defaults to 16 streams
  void Initialize(size_t pool_size);

  // Get the underlying RMM cuda_stream_pool
  rmm::cuda_stream_pool& GetPool();

  // Convenience method: get a stream from the pool
  rmm::cuda_stream_view GetStream();

  // Get the pool size
  size_t GetPoolSize() const;

  // Check if the pool has been initialized with a custom size
  bool IsInitialized() const { return initialized_; }

 private:
  CudaStreamPool();
  ~CudaStreamPool() = default;

  // Prevent copying
  CudaStreamPool(const CudaStreamPool&) = delete;
  CudaStreamPool& operator=(const CudaStreamPool&) = delete;

  std::unique_ptr<rmm::cuda_stream_pool> pool_;
  bool initialized_ = false;
};

// RAII wrapper for acquiring streams from the pool
// Automatically synchronizes the stream when destroyed
// Usage:
//   StreamGuard stream_guard;
//   rmm::cuda_stream_view stream = stream_guard.stream();
//   rmm::device_buffer buf{size, stream};
//   kernel<<<..., stream.value()>>>(buf.data());
class StreamGuard {
 public:
  StreamGuard() : stream_view_(CudaStreamPool::GetInstance().GetStream()) {}

  ~StreamGuard() {
    // Synchronize the stream before it's returned to the pool
    stream_view_.synchronize();
  }

  // Prevent copying
  StreamGuard(const StreamGuard&) = delete;
  StreamGuard& operator=(const StreamGuard&) = delete;

  // Allow moving
  StreamGuard(StreamGuard&& other) noexcept : stream_view_(other.stream_view_) {}

  StreamGuard& operator=(StreamGuard&& other) noexcept {
    if (this != &other) {
      stream_view_.synchronize();
      stream_view_ = other.stream_view_;
    }
    return *this;
  }

  // Get the underlying cuda_stream_view for stream-ordered operations
  rmm::cuda_stream_view stream() const { return stream_view_; }

  // Implicit conversion to rmm::cuda_stream_view for seamless usage
  // This allows: StreamGuard stream; make_device_buffer<T>(count, stream);
  operator rmm::cuda_stream_view() const { return stream_view_; }

  // Get as cudaStream_t for compatibility with legacy code
  cudaStream_t Get() const { return stream_view_.value(); }

  operator cudaStream_t() const { return stream_view_.value(); }

  cudaStream_t value() const { return stream_view_.value(); }

 private:
  rmm::cuda_stream_view stream_view_;
};

// Convenience function to get a stream from the pool
// Usage: auto stream = get_stream();
inline StreamGuard get_stream() {
  return StreamGuard{};
}

}  // namespace paillier_gpu
}  // namespace algorithms
}  // namespace lib
}  // namespace heu

#endif  // HEU_LIBRARY_ALGORITHMS_PAILLIER_GPU_GPULIB_CUDA_STREAM_POOL_H_

