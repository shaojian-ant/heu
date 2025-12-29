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

#include "heu/library/algorithms/paillier_gpu/gpulib/rmm_memory_pool.h"

#include <cuda_runtime.h>
#include <memory>
#include <mutex>

// Disable RMM logging to avoid dependency on rapids_logger
#define RMM_LOGGING_LEVEL_OFF 0
#define RMM_LOGGING_LEVEL RMM_LOGGING_LEVEL_OFF

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/thread_safe_resource_adaptor.hpp>
#include <rmm/mr/device/tracking_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

namespace heu {
namespace lib {
namespace algorithms {
namespace paillier_gpu {

// Global RMM memory resources with chaining:
// cuda_memory_resource -> pool_memory_resource -> tracking_resource_adaptor -> thread_safe_resource_adaptor
// Using raw pointers to avoid cleanup issues during program shutdown
rmm::mr::cuda_memory_resource* g_cuda_mr = nullptr;
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>* g_pool_mr = nullptr;
rmm::mr::tracking_resource_adaptor<
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>* g_tracking_mr = nullptr;
rmm::mr::thread_safe_resource_adaptor<
    rmm::mr::tracking_resource_adaptor<
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>>*
    g_thread_safe_mr = nullptr;

// Implementation details hidden in this compilation unit
namespace {

std::mutex g_rmm_init_mutex;
bool g_rmm_initialized = false;

void InitializeRMM() {
  std::lock_guard<std::mutex> lock(g_rmm_init_mutex);

  if (g_rmm_initialized) {
    return;
  }

  // Create CUDA memory resource (base)
  g_cuda_mr = new rmm::mr::cuda_memory_resource();

  // Get available device memory
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);

  // Create pool memory resource with 50% of available memory
  // This provides efficient sub-allocation from a pre-allocated pool
  size_t initial_pool_size = free_mem / 2;
  g_pool_mr = new rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>(
      g_cuda_mr, initial_pool_size);

  // Wrap with tracking adaptor to monitor allocations
  g_tracking_mr = new rmm::mr::tracking_resource_adaptor<
      rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(g_pool_mr);

  // Wrap with thread-safe adaptor for multi-threaded access
  // This ensures thread-safe allocation/deallocation
  g_thread_safe_mr = new rmm::mr::thread_safe_resource_adaptor<
      rmm::mr::tracking_resource_adaptor<
          rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>>(g_tracking_mr);

  printf("[RMM] Memory pool initialized: initial_pool_size = %zu MB, free_mem = %zu MB\n",
         initial_pool_size / (1024 * 1024), free_mem / (1024 * 1024));

  g_rmm_initialized = true;
}

rmm::mr::device_memory_resource* GetMemoryResource() {
  if (!g_rmm_initialized) {
    InitializeRMM();
  }
  return g_thread_safe_mr;
}

}  // namespace

// RMMMemoryPool implementation
RMMMemoryPool& RMMMemoryPool::GetInstance() {
  static RMMMemoryPool instance;
  return instance;
}

void RMMMemoryPool::Initialize() {
  if (!initialized_) {
    InitializeRMM();

    // Set the current device resource to our thread-safe pool
    // This allows rmm::device_buffer to use our pool automatically
    rmm::mr::set_current_device_resource(g_thread_safe_mr);

    initialized_ = true;
  }
}

RMMMemoryPool::RMMMemoryPool() {
  // Initialize RMM on construction
  Initialize();
}

RMMMemoryPool::~RMMMemoryPool() {
  // RMM resources are cleaned up automatically
  // Note: In production, you may want to explicitly delete these
  // but for now we let them leak to avoid shutdown order issues
}

void RMMMemoryPool::PrintMemoryStats() const {
  if (!initialized_ || !g_tracking_mr) {
    printf("[RMM] Memory pool not initialized or tracking not enabled\n");
    return;
  }

  auto allocated_bytes = g_tracking_mr->get_allocated_bytes();

  printf("[RMM] Memory Statistics:\n");
  printf("  - Currently allocated: %zu bytes (%.2f MB)\n",
         allocated_bytes, allocated_bytes / (1024.0 * 1024.0));
}

}  // namespace paillier_gpu
}  // namespace algorithms
}  // namespace lib
}  // namespace heu

