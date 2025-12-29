#ifndef HEU_LIBRARY_ALGORITHMS_PAILLIER_GPU_GPULIB_RMM_MEMORY_POOL_H_
#define HEU_LIBRARY_ALGORITHMS_PAILLIER_GPU_GPULIB_RMM_MEMORY_POOL_H_

#include <cuda_runtime.h>
#include <memory>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace heu {
namespace lib {
namespace algorithms {
namespace paillier_gpu {

// GPU Memory Pool using RAPIDS Memory Manager (RMM)
// This class provides initialization and configuration of RMM memory resources
// For actual allocations, use rmm::device_buffer directly
class RMMMemoryPool {
 public:
  static RMMMemoryPool& GetInstance();

  // Initialize RMM with a pool memory resource
  // This should be called once at program startup
  void Initialize();

  // Check if RMM has been initialized
  bool IsInitialized() const { return initialized_; }

  // Get memory statistics (requires tracking_resource_adaptor)
  void PrintMemoryStats() const;

 private:
  RMMMemoryPool();
  ~RMMMemoryPool();

  // Prevent copying
  RMMMemoryPool(const RMMMemoryPool&) = delete;
  RMMMemoryPool& operator=(const RMMMemoryPool&) = delete;

  bool initialized_ = false;
};

// Helper function to create a typed device_buffer
// This provides a convenient way to allocate typed device memory using RMM
//
// Usage:
//   auto buffer = make_device_buffer<int>(100, stream);  // Allocate 100 ints
//   int* ptr = static_cast<int*>(buffer.data());
//   kernel<<<..., stream.value()>>>(ptr);
template<typename T>
inline rmm::device_buffer make_device_buffer(std::size_t count,
                                             rmm::cuda_stream_view stream) {
  return rmm::device_buffer(sizeof(T) * count, stream);
}

// Helper function to create a device_buffer from host data
// This copies data from host to device
//
// Usage:
//   std::vector<int> host_data = {1, 2, 3};
//   auto buffer = make_device_buffer_from_host(host_data.data(), host_data.size(), stream);
template<typename T>
inline rmm::device_buffer make_device_buffer_from_host(const T* host_data,
                                                       std::size_t count,
                                                       rmm::cuda_stream_view stream) {
  return rmm::device_buffer(host_data, sizeof(T) * count, stream);
}

}  // namespace paillier_gpu
}  // namespace algorithms
}  // namespace lib
}  // namespace heu

#endif  // HEU_LIBRARY_ALGORITHMS_PAILLIER_GPU_GPULIB_RMM_MEMORY_POOL_H_

