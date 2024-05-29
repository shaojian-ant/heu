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

#include <utility>

#include "heu/algorithms/mock_phe/base.h"
#include "heu/spi/he/sketches/scalar/decryptor.h"

namespace heu::algos::mock_phe {

class Decryptor : public spi::DecryptorScalarSketch<Plaintext, Ciphertext> {
 public:
  void Decrypt(const Ciphertext &ct, Plaintext *out) const override;
  Plaintext Decrypt(const Ciphertext &ct) const override;
};

}  // namespace heu::algos::mock_phe