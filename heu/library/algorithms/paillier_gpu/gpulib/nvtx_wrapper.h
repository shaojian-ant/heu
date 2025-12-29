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

#pragma once

#include "nvtx3/nvToolsExt.h"
#include <string>

namespace heu::lib::algorithms::paillier_gpu {

/**
 * @brief RAII wrapper for NVTX range profiling
 * 
 * This class provides a convenient RAII (Resource Acquisition Is Initialization)
 * wrapper around NVTX range push/pop operations. It automatically pushes a named
 * range when constructed and pops it when destroyed, ensuring proper cleanup
 * even in the presence of exceptions or early returns.
 * 
 * Usage example:
 * @code
 * void my_function() {
 *     NvtxRange range("my_function_range");
 *     // ... function logic ...
 * } // range is automatically popped when leaving scope
 * @endcode
 */
class NvtxRange {
 public:
  /**
   * @brief Construct and push a named NVTX range
   * @param name The name of the range to be displayed in profiling tools
   */
  explicit NvtxRange(const std::string& name) {
    nvtxRangePushA(name.c_str());
  }

  /**
   * @brief Construct and push a named NVTX range (C-string version)
   * @param name The name of the range to be displayed in profiling tools
   */
  explicit NvtxRange(const char* name) {
    nvtxRangePushA(name);
  }

  /**
   * @brief Destructor that automatically pops the NVTX range
   */
  ~NvtxRange() {
    nvtxRangePop();
  }

  // Disable copy operations
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;

  // Disable move operations (ranges should not be moved)
  NvtxRange(NvtxRange&&) = delete;
  NvtxRange& operator=(NvtxRange&&) = delete;
};

}  // namespace heu::lib::algorithms::paillier_gpu

