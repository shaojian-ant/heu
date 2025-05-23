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

#include "heu/algorithms/ou/he_kit.h"

#include "heu/algorithms/ou/decryptor.h"
#include "heu/algorithms/ou/encryptor.h"
#include "heu/algorithms/ou/evaluator.h"
#include "heu/spi/he/he.h"

namespace heu::algos::ou {

namespace {
const std::string kLibName = "ou";  // do not change

// according to this paper
// << Accelerating Okamoto-Uchiyama’s Public-Key Cryptosystem >>
// and NIST's recommendation:
// https://www.keylength.com/en/4/
// 160 bits for key size 1024; 224 bits for key size 2048
constexpr size_t kPrimeFactorSize1024 = 160;
constexpr size_t kPrimeFactorSize2048 = 224;
constexpr size_t kPrimeFactorSize3072 = 256;
}  // namespace

std::string HeKit::GetLibraryName() const { return kLibName; }

spi::Schema HeKit::GetSchema() const { return spi::Schema::OU; }

// TODO: print key size?
std::string HeKit::ToString() const {
  return fmt::format("{} schema from {} lib", GetSchema(), GetLibraryName());
}

size_t HeKit::Serialize(uint8_t *, size_t) const {
  // nothing to serialize
  return 0;
}

bool HeKit::Check(spi::Schema schema, const spi::SpiArgs &) {
  return schema == spi::Schema::OU;
}

std::unique_ptr<spi::HeKit> HeKit::Create(spi::Schema schema,
                                          const spi::SpiArgs &args) {
  YACL_ENFORCE(schema == spi::Schema::OU, "Schema {} not supported by {}",
               schema, kLibName);
  YACL_ENFORCE(
      args.Exist(spi::ArgGenNewPkSk) || args.Exist(spi::ArgPkFrom),
      "Neither ArgGenNewPkSk nor ArgPkFrom is set, you must set one of them");

  auto kit = std::make_unique<HeKit>();
  if (args.GetOptional(spi::ArgGenNewPkSk) == true) {
    kit->GenPkSk(args.GetOrDefault(spi::ArgKeySize, 2048));
  } else {
    // recover pk/sk from buffer
    kit->pk_ = PublicKey::LoadFrom(args.GetRequired(spi::ArgPkFrom));
    if (args.Exist(spi::ArgSkFrom)) {
      kit->sk_ = SecretKey::LoadFrom(args.GetRequired(spi::ArgSkFrom));
    }
  }

  kit->InitOperators();
  return kit;
}

void HeKit::InitOperators() {
  item_tool_ = std::make_shared<ItemTool>();

  if (pk_) {
    encryptor_ = std::make_shared<Encryptor>(pk_);
    word_evaluator_ = std::make_shared<Evaluator>(pk_);

    if (sk_) {
      decryptor_ = std::make_shared<Decryptor>(pk_, sk_);
    }
  }
}

void HeKit::GenPkSk(size_t key_size) {
  size_t secret_size = (key_size + 2) / 3;

  auto prime_factor_size = kPrimeFactorSize1024;
  if (key_size >= 3072) {
    prime_factor_size = kPrimeFactorSize3072;
  } else if (key_size >= 2048) {
    prime_factor_size = kPrimeFactorSize2048;
  }

  YACL_ENFORCE(prime_factor_size * 2 <= secret_size,
               "Key size must be larger than {} bits",
               prime_factor_size * 2 * 3 - 2);

  sk_ = std::make_shared<SecretKey>();
  BigInt u, prime_factor;
  do {
    prime_factor = BigInt::RandPrimeOver(prime_factor_size);
    // bits_of(a * b) <= bits_of(a) + bits_of(b),
    // So we add extra two bits to u:
    //    one bit for prime_factor * u; another one bit for p^2;
    // Also, make sure that u > prime_factor
    u = BigInt::RandomMonicExactBits(secret_size - prime_factor_size + 2);
    sk_->p_ = prime_factor * u + 1;  // p - 1 has a large prime factor
  } while (!sk_->p_.IsPrime());
  // since bits_of(a * b) <= bits_of(a) + bits_of(b)
  // add another 1 bit for q
  sk_->q_ = BigInt::RandPrimeOver(secret_size + 1);
  sk_->p2_ = sk_->p_ * sk_->p_;
  sk_->p_half_ = sk_->p_ / 2;
  sk_->t_ = prime_factor;
  sk_->n_ = sk_->p2_ * sk_->q_;

  pk_ = std::make_shared<PublicKey>();
  pk_->n_ = sk_->n_;

  BigInt g, g_, gp, check, gcd;
  do {
    do {
      g = BigInt::RandomLtN(pk_->n_);
      gcd = g.Gcd(sk_->p_);
    } while (gcd != 1);
    gp = (g % sk_->p2_).PowMod(sk_->p_ - 1, sk_->p2_);
    check = gp.PowMod(sk_->p_, sk_->p2_);
  } while (check != 1);

  sk_->gp_inv_ = ((gp - 1) / sk_->p_).InvMod(sk_->p_);

  // make sure 'g_' and 'p^2' are co-prime
  do {
    g_ = BigInt::RandomLtN(pk_->n_);
  } while ((g_ % sk_->p_).IsZero());
  pk_->capital_g_ = g.PowMod(u, pk_->n_);
  pk_->capital_h_ = g_.PowMod(pk_->n_ * u, pk_->n_);

  // Note 1: About plaintext overflow attack:
  // https://staff.csie.ncu.edu.tw/yensm/techreports/1998/LCIS_TR-98-8B.ps.gz
  // proposes an overflow attack method, but due to the existence of plaintext
  // multiplication, No matter how small the 'max_plaintext_' limit is,
  // the plaintext space can overflow, so a small limit of 'max_plaintext_'
  // has little effect against this attack
  // Note 2:
  // A smaller max_plaintext_ can generate a denser cache table.
  // If you modify 'max_plaintext_', please modify the cache table density in
  // public_key.cc as well.
  // Note 3:
  // max_plaintext_ must be a power of 2, for ease of use
  pk_->max_plaintext_ = BigInt(1) << (sk_->p_half_.BitCount() - 1);
  pk_->Init();
}

REGISTER_HE_LIBRARY(kLibName, 120, HeKit::Check, HeKit::Create);

}  // namespace heu::algos::ou
