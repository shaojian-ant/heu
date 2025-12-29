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

#include "heu/library/algorithms/elgamal/utils/lookup_table.h"

#include "yacl/utils/parallel.h"

namespace heu::lib::algorithms::elgamal {

template <typename ImplType, typename HashPointType>
class ImplBase {
 public:
  ~ImplBase() = default;

  void Init(const std::shared_ptr<EcGroup> &curve);

  static const MPInt &MaxSupportedValue() {
    const static MPInt max(kTableMaxValue * kSearchMaxValue - 1);
    return max;
  }

 protected:
  // mG -> m
  std::shared_ptr<HashMap<HashPointType, int64_t>> table_;
  EcPoint table_max_pos_;
  EcPoint table_max_neg_;
  std::shared_ptr<EcGroup> curve_;

  static const int kLookupTableBits;
  static const int kExtraSearchBits;

  static const int64_t kTableMaxValue;
  static const int64_t kSearchMaxValue;
};

class LookupTable::Impl1 : public ImplBase<LookupTable::Impl1, EcPoint> {
 public:
  int64_t Search(const EcPoint &p) const;

 private:
  constexpr static int TableMaxValue() { return 20; };

  constexpr static int ExtraSearchBits() { return 12; };

  const EcPoint &CastPoint(const EcPoint &p) { return p; }

  friend ImplBase<LookupTable::Impl1, EcPoint>;
};

class LookupTable::Impl2
    : public ImplBase<LookupTable::Impl2, AffineAnyEcPoint> {
 public:
  int64_t Search(const EcPoint &p);

 private:
  constexpr static int TableMaxValue() { return 20; };

  constexpr static int ExtraSearchBits() { return 12; };

  AffineAnyEcPoint CastPoint(const EcPoint &p) {
    return curve_->GetAffineAnyPoint(p);
  }

  void InitCacheTable(const EcPoint &p, int current_bits,
                      std::vector<std::any> &mul_tree,
                      std::vector<AffineAnyEcPoint> &cache_table);
  void BuildMulTree(std::vector<std::any> &mul_tree);
  void BuildInvTree(const std::vector<std::any> &mul_tree,
                    std::vector<std::any> &inv_tree);
  std::optional<int64_t> SearchCacheTable(
      int current_bits, std::vector<std::any> &inv_tree,
      std::vector<AffineAnyEcPoint> &cache_table);

  friend ImplBase<LookupTable::Impl2, AffineAnyEcPoint>;
};

template <typename ImplType, typename HashPointType>
const int ImplBase<ImplType, HashPointType>::kLookupTableBits =
    ImplType::TableMaxValue();

template <typename ImplType, typename HashPointType>
const int ImplBase<ImplType, HashPointType>::kExtraSearchBits =
    ImplType::ExtraSearchBits();

template <typename ImplType, typename HashPointType>
const int64_t ImplBase<ImplType, HashPointType>::kTableMaxValue =
    1LL << kLookupTableBits;

template <typename ImplType, typename HashPointType>
const int64_t ImplBase<ImplType, HashPointType>::kSearchMaxValue =
    1LL << kExtraSearchBits;

template <typename ImplType, typename HashPointType>
void ImplBase<ImplType, HashPointType>::Init(
    const std::shared_ptr<EcGroup> &curve) {
  curve_ = curve;

  // lambda: make a copy of curve, so that if LookupTable object moved, these
  // lambdas still work
  auto hash = [curve](const HashPointType &p) { return curve->HashPoint(p); };
  auto equal = [curve](const HashPointType &p1, const HashPointType &p2) {
    return curve->PointEqual(p1, p2);
  };

  // mG -> m
  // m in range [0, MAX_VALUE) U [n - MAX_VALUE, n), n is the order
  table_ = std::make_shared<HashMap<HashPointType, int64_t>>(kTableMaxValue,
                                                             hash, equal);

  yacl::parallel_for(0, kTableMaxValue, 1, [&](int64_t beg, int64_t end) {
    auto g = curve_->GetGenerator();
    auto point = curve_->MulBase(MPInt(beg));
    table_->Insert(static_cast<ImplType *>(this)->CastPoint(point), beg);
    for (int64_t i = beg + 1; i < end; ++i) {
      point = curve_->Add(point, g);
      table_->Insert(static_cast<ImplType *>(this)->CastPoint(point), i);
    }
  });

  table_max_pos_ = curve_->MulBase(MPInt(kTableMaxValue));
  table_max_neg_ = curve_->Negate(table_max_pos_);
}

int64_t LookupTable::Impl1::Search(const EcPoint &p) const {
  auto *it = table_->Find(p);
  if (it != nullptr) {
    return *it;
  }

  auto im_pos = curve_->Add(p, table_max_neg_);  // assume point is positive
  auto im_neg = curve_->Add(p, table_max_pos_);
  for (int64_t i = 1; i < kSearchMaxValue; ++i) {
    it = table_->Find(im_pos);
    if (it != nullptr) {
      return *it + i * kTableMaxValue;
    }

    it = table_->Find(im_neg);
    if (it != nullptr) {
      return *it - i * kTableMaxValue;
    }

    curve_->AddInplace(&im_pos, table_max_neg_);
    curve_->AddInplace(&im_neg, table_max_pos_);
  }

  // last try for negative point
  it = table_->Find(im_neg);
  if (it != nullptr) {
    return *it - kSearchMaxValue * kTableMaxValue;
  }

  YACL_THROW("ElGamal: Cannot decrypt, the plaintext is too big");
}

int64_t LookupTable::Impl2::Search(const EcPoint &p) {
  // https://eprint.iacr.org/2022/1573 IV.A: The Tree-based Montgomery’s Trick
  // The core idea is to replace modular inversions with efficient modular
  // multiplications

  for (int bits = 0; bits <= kExtraSearchBits; ++bits) {
    std::vector<std::any> mul_tree;  // BT1
    std::vector<std::any> inv_tree;  // BT2
    std::vector<AffineAnyEcPoint> cache_table;
    InitCacheTable(p, bits, mul_tree, cache_table);
    BuildMulTree(mul_tree);
    BuildInvTree(mul_tree, inv_tree);
    auto opt = SearchCacheTable(bits, inv_tree, cache_table);
    if (opt.has_value()) {
      return opt.value();
    }
  }

  YACL_THROW("ElGamal: Cannot decrypt, the plaintext is too big");
}

void LookupTable::Impl2::InitCacheTable(
    const EcPoint &p, int current_bits, std::vector<std::any> &mul_tree,
    std::vector<AffineAnyEcPoint> &cache_table) {
  size_t last_num = current_bits > 0 ? 1UL << (current_bits - 1) : 0;
  size_t current_num = 1UL << current_bits;
  size_t total_num = current_num - last_num;
  auto im_pos = curve_->Add(p, curve_->Mul(table_max_neg_, MPInt(last_num)));
  auto im_neg =
      curve_->Add(p, curve_->Mul(table_max_pos_, MPInt(last_num + 1)));

  mul_tree.resize(total_num * 4 - 1);
  cache_table.resize(total_num * 2);

  for (size_t i = 0; i < total_num; ++i) {
    auto [x_pos, y_pos, z_pos] = curve_->GetAnyPoint(im_pos);
    mul_tree[total_num + i] = std::move(z_pos);
    cache_table[total_num + i] = {std::move(x_pos), std::move(y_pos)};

    auto [x_neg, y_neg, z_neg] = curve_->GetAnyPoint(im_neg);
    mul_tree[total_num - 1 - i] = std::move(z_neg);
    cache_table[total_num - 1 - i] = {std::move(x_neg), std::move(y_neg)};

    curve_->AddInplace(&im_pos, table_max_neg_);
    curve_->AddInplace(&im_neg, table_max_pos_);
  }
}

void LookupTable::Impl2::BuildMulTree(std::vector<std::any> &mul_tree) {
  // https://eprint.iacr.org/2022/1573 Algorithm 2 Generation of BT1
  size_t f = (mul_tree.size() + 1) / 2;  // current node index
  size_t h = f / 2;                      // number of nodes in current level
  size_t i = 0;
  while (h > 0) {
    for (size_t j = 1; j <= h; ++j) {
      mul_tree[f] = curve_->FeMul(mul_tree[i], mul_tree[i + 1]);
      ++f;
      i += 2;
    }
    h >>= 1;
  }
}

void LookupTable::Impl2::BuildInvTree(const std::vector<std::any> &mul_tree,
                                      std::vector<std::any> &inv_tree) {
  inv_tree.resize(mul_tree.size());
  // https://eprint.iacr.org/2022/1573 Algorithm 3 Generation of binary tree BT2
  size_t f = inv_tree.size() - 1;  // leftmost node index in previous level
  size_t k = 1;                    // number of nodes in previous level
  inv_tree[f] = curve_->FeInv(mul_tree[f]);
  while (f > 0) {
    size_t h = f - 2 * k;  // leftmost node index in current level
    for (size_t j = 0; j < 2 * k; ++j) {
      size_t v = j + h;  // v is the current node
      size_t w = v ^ 1;  // w is the sibling node of node v
      inv_tree[v] = curve_->FeMul(mul_tree[w], inv_tree[f + (j >> 1)]);
    }
    f = h;
    k <<= 1;
  }
}

std::optional<int64_t> LookupTable::Impl2::SearchCacheTable(
    int current_bits, std::vector<std::any> &inv_tree,
    std::vector<AffineAnyEcPoint> &cache_table) {
  size_t total_num = cache_table.size() / 2;
  for (size_t i = 0; i < cache_table.size(); ++i) {
    auto &p = cache_table[i];
    p.x = curve_->FeMul(p.x, inv_tree[i]);
    p.y = curve_->FeMul(p.y, inv_tree[i]);
    auto it = table_->Find(p);
    if (it != nullptr) {
      if (current_bits == 0) {
        return *it - (1 - i) * kTableMaxValue;
      } else {
        if (i < total_num) {
          return *it - (2 * total_num - i) * kTableMaxValue;
        } else {
          return *it + i * kTableMaxValue;
        }
      }
    }
  }

  return std::nullopt;
}

const MPInt &LookupTable::MaxSupportedValue() {
  const auto &max1 = ImplBase<LookupTable::Impl1, EcPoint>::MaxSupportedValue();
  const auto &max2 =
      ImplBase<LookupTable::Impl2, AffineAnyEcPoint>::MaxSupportedValue();
  return max1 < max2 ? max1 : max2;
}

void LookupTable::Init(const std::shared_ptr<EcGroup> &curve) {
  try {
    pImpl2 = std::make_shared<Impl2>();
    pImpl2->Init(curve);
  } catch (yacl::RuntimeError &e) {
    pImpl2 = nullptr;
    pImpl1 = std::make_shared<Impl1>();
    pImpl1->Init(curve);
  }
}

int64_t LookupTable::Search(const EcPoint &p) const {
  if (pImpl2) {
    return pImpl2->Search(p);
  } else {
    YACL_ENFORCE(pImpl1, "lookup table impl 1 not initialized");
    return pImpl1->Search(p);
  }
}

}  // namespace heu::lib::algorithms::elgamal
