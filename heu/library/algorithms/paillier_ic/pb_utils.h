// Copyright 2023 Ant Group Co., Ltd.
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

#include "heu/library/algorithms/util/big_int.h"

#include "interconnection/runtime/phe.pb.h"

namespace heu::lib::algorithms::paillier_ic {

namespace pb_ns = ::org::interconnection::v2::runtime;

pb_ns::Bigint BigInt2PbBigint(const BigInt &bi);
void PbBigint2BigInt(const pb_ns::Bigint &pb_bi, BigInt &bigint);

}  // namespace heu::lib::algorithms::paillier_ic
