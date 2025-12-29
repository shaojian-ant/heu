// Copyright 2024 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "heu/library/algorithms/paillier_gpu/gpulib/cuda_stream_pool.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "heu/library/algorithms/paillier_gpu/gpulib/rmm_memory_pool.h"

namespace heu {
namespace lib {
namespace algorithms {
namespace paillier_gpu {

CudaStreamPool& CudaStreamPool::GetInstance() {
  static CudaStreamPool instance;
  return instance;
}

CudaStreamPool::CudaStreamPool() {
  // CRITICAL: Initialize RMM memory pool FIRST before creating any CUDA streams
  // This ensures that all RMM allocations use the pool_memory_resource
  // instead of the default cuda_memory_resource (which calls cudaMalloc/cudaFree)
  RMMMemoryPool::GetInstance().Initialize();

  // Check environment variable for custom pool size
  size_t pool_size = 128;  // Default: 128 streams for high concurrency
  const char* env_pool_size = std::getenv("HEU_CUDA_STREAM_POOL_SIZE");
  if (env_pool_size != nullptr) {
    int size = std::atoi(env_pool_size);
    if (size > 0) {
      pool_size = static_cast<size_t>(size);
    } else {
      printf("[CUDA Stream Pool] Warning: Invalid HEU_CUDA_STREAM_POOL_SIZE=%s, using default (128)\n",
             env_pool_size);
    }
  }

  // Create RMM cuda_stream_pool with the determined size
  // This provides efficient stream management with thread-safe access
  // using round-robin distribution and atomic operations
  pool_ = std::make_unique<rmm::cuda_stream_pool>(pool_size);
  printf("[CUDA Stream Pool] Initialized with %zu streams (default=128, set HEU_CUDA_STREAM_POOL_SIZE to customize)\n",
         pool_->get_pool_size());
}

void CudaStreamPool::Initialize(size_t pool_size) {
  static std::mutex init_mutex;
  std::lock_guard<std::mutex> lock(init_mutex);

  if (initialized_) {
    printf("[CUDA Stream Pool] Already initialized with %zu streams, skipping re-initialization\n",
           pool_->get_pool_size());
    return;
  }

  if (pool_size == 0) {
    printf("[CUDA Stream Pool] Warning: pool_size is 0, using default size (16)\n");
    return;
  }

  // Recreate the pool with the specified size
  pool_ = std::make_unique<rmm::cuda_stream_pool>(pool_size);
  initialized_ = true;

  printf("[CUDA Stream Pool] Initialized with custom size: %zu streams\n", pool_size);
  printf("[CUDA Stream Pool] Recommendation: pool_size should be >= number of concurrent threads\n");
}

rmm::cuda_stream_pool& CudaStreamPool::GetPool() {
  return *pool_;
}

rmm::cuda_stream_view CudaStreamPool::GetStream() {
  return pool_->get_stream();
}

size_t CudaStreamPool::GetPoolSize() const {
  return pool_->get_pool_size();
}

}  // namespace paillier_gpu
}  // namespace algorithms
}  // namespace lib
}  // namespace heu

