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

cuda_library(
    name = "arith",
    srcs = [],
    hdrs = glob([
        "include/cgbn/arith/*.h",
        "include/cgbn/arith/*.cuh",
    ]),
)

cuda_library(
    name = "core",
    srcs = [],
    hdrs = glob([
        "include/cgbn/core/*.cuh",
    ]),
)

cuda_library(
    name = "cgbn",
    srcs = [],
    hdrs = glob([
        "include/cgbn/*.h",
        "include/cgbn/*.cuh",
    ]),
    deps = [
        ":arith",
        ":core",
    ],
)
