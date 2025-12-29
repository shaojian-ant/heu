# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

# Generate logger_macros.hpp stub
genrule(
    name = "generate_logger_macros",
    outs = ["cpp/include/rmm/logger_macros.hpp"],
    cmd = """
cat > $@ << 'EOF'
#pragma once
#define RMM_LOG_TRACE(...)
#define RMM_LOG_DEBUG(...)
#define RMM_LOG_INFO(...)
#define RMM_LOG_WARN(...)
#define RMM_LOG_ERROR(...)
#define RMM_LOG_CRITICAL(...)
#define RMM_EXPECTS(...)
#define RMM_ENSURES(...)
EOF
    """,
)

# Generate rapids_logger stub
genrule(
    name = "generate_rapids_logger",
    outs = ["cpp/include/rapids_logger/logger.hpp"],
    cmd = """
mkdir -p $$(dirname $@)
cat > $@ << 'EOF'
#pragma once
namespace rapids_logger {
  using sink_ptr = void*;
  class logger {};
}
EOF
    """,
)

# RMM library with essential source files
cuda_library(
    name = "rmm",
    srcs = [
        "cpp/src/aligned.cpp",
        "cpp/src/error.cpp",
        "cpp/src/cuda_stream_view.cpp",
        "cpp/src/cuda_device.cpp",
        "cpp/src/cuda_stream.cpp",
        "cpp/src/cuda_stream_pool.cpp",
        "cpp/src/device_buffer.cpp",
    ],
    hdrs = glob([
        "cpp/include/rmm/**/*.hpp",
        "cpp/include/rmm/**/*.h",
    ]) + [
        ":generate_logger_macros",
        ":generate_rapids_logger",
    ],
    includes = ["cpp/include"],
    copts = [
        "-std=c++17",
        "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE",
    ],
    defines = [
        "LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE",
    ],
    visibility = ["//visibility:public"],
)

