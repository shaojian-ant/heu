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

#include <cstdlib>

#include "heu/library/algorithms/paillier_gpu/gpulib/error.h"
#include "heu/library/algorithms/paillier_gpu/gpulib/gpu_paillier.h"
#include "heu/library/algorithms/paillier_gpu/gpulib/gpupaillier.h"
#include "heu/library/algorithms/paillier_gpu/gpulib/rmm_memory_pool.h"
#include "heu/library/algorithms/paillier_gpu/gpulib/cuda_stream_pool.h"
#include "heu/library/algorithms/paillier_gpu/gpulib/nvtx_wrapper.h"

using heu::lib::algorithms::paillier_gpu::RMMMemoryPool;
using heu::lib::algorithms::paillier_gpu::make_device_buffer;
using heu::lib::algorithms::paillier_gpu::CudaStreamPool;
using heu::lib::algorithms::paillier_gpu::StreamGuard;
using heu::lib::algorithms::paillier_gpu::get_stream;
using heu::lib::algorithms::paillier_gpu::NvtxRange;

template <class params>
__device__ __forceinline__ void paillier_t<params>::fixed_window_powm_odd(
    bn_t &result, const bn_t &x, const bn_t &power, const bn_t &modulus) {
  bn_t t;
  bn_local_t window[1 << window_bits];
  int32_t index, position, offset;
  uint32_t np0;

  // conmpute x^power mod modulus, using the fixed window algorithm
  // requires:  x<modulus,  modulus is odd
  // compute x^0 (in Montgomery space, this is just 2^BITS - modulus)
  cgbn_negate(_env, t, modulus);
  cgbn_store(_env, window + 0, t);

  // convert x into Montgomery space, store into window table
  np0 = cgbn_bn2mont(_env, result, x, modulus);
  cgbn_store(_env, window + 1, result);
  cgbn_set(_env, t, result);

// compute x^2, x^3, ... x^(2^window_bits-1), store into window table
#pragma nounroll
  for (index = 2; index < (1 << window_bits); index++) {
    cgbn_mont_mul(_env, result, result, t, modulus, np0);
    cgbn_store(_env, window + index, result);
  }

  // find leading high bit
  position = params::BITS - cgbn_clz(_env, power);

  // break the exponent into chunks, each window_bits in length
  // load the most significant non-zero exponent chunk
  offset = position % window_bits;
  if (offset == 0)
    position = position - window_bits;
  else
    position = position - offset;
  index = cgbn_extract_bits_ui32(_env, power, position, window_bits);
  cgbn_load(_env, result, window + index);

  // process the remaining exponent chunks
  while (position > 0) {
// square the result window_bits times
#pragma nounroll
    for (int sqr_count = 0; sqr_count < window_bits; sqr_count++)
      cgbn_mont_sqr(_env, result, result, modulus, np0);

    // multiply by next exponent chunk
    position = position - window_bits;
    index = cgbn_extract_bits_ui32(_env, power, position, window_bits);
    cgbn_load(_env, t, window + index);
    cgbn_mont_mul(_env, result, result, t, modulus, np0);
  }

  // we've processed the exponent now, convert back to normal space
  cgbn_mont2bn(_env, result, result, modulus, np0);
}

// Sum the encrypted values by multiplying the ciphertexts
template <class params>
__global__ void kernel_paillier_enc(cgbn_error_report_t *report,
                                    gpu_paillier_ciphertext_t *gpu_res,
                                    gpu_paillier_pubkey_t *gpu_pub,
                                    gpu_paillier_plaintext_t *gpu_pt,
                                    gpu_paillier_random_t *rand,
                                    uint32_t count) {
  // decode an instance number from the blockIdx and threadIdx
  int32_t i;
  i = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
  if (i >= count) return;

  // 4096bit variables
  paillier_t<params> po(cgbn_report_monitor, report, i);
  typename paillier_t<params>::bn_t r1, r2, g, x, p, m, n;
  typename paillier_t<params>::bn_wide_t r;

  cgbn_load(po._env, g, &(gpu_pub->n_plusone));
  cgbn_load(po._env, m, &(gpu_pt[i].m));
  cgbn_load(po._env, n, &(gpu_pub->n_squared));

  // Fake code:paillier.fixed_window_powm_odd(gpu_res[i].c, gpu_pub->n_plusone,
  // gpu_pt[i].m, gpu_pub->n_squared);
  po.fixed_window_powm_odd(r1, g, m, n);

  cgbn_load(po._env, x, &(rand[i].m));
  cgbn_load(po._env, p, &(gpu_pub->n));

  // Fake code:paillier.fixed_window_powm_odd(gpu_x.x, rand[i].m, gpu_pub->n,
  // gpu_pub->n_squared);
  po.fixed_window_powm_odd(r2, x, p, n);
  // cgbn_modular_power(po._env,r2,x,p,n); //the x should less than n

  // Fake code:paillier.mul(&(gpu_res[i].c), &(gpu_res[i].c), &(gpu_x.x));
  cgbn_mul_wide(po._env, r, r1, r2);  // the r is 8192 bit,r=r1*r2;

  // Fake code:paillier.mod(&(gpu_res[i].c), &(gpu_pub->n_squared));
  cgbn_rem_wide(po._env, r1, r,
                n);  // back to 4096 bit for next mod ,r1=r%m ,the high CGBN of
                     // r is less than the denominator

  cgbn_store(po._env, &(gpu_res[i].c), r1);
  return;
}

template <class params>
__global__ void kernel_paillier_dec(cgbn_error_report_t *report,
                                    gpu_paillier_plaintext_t *gpu_res,
                                    gpu_paillier_pubkey_t *gpu_pub,
                                    gpu_paillier_prvkey_t *gpu_prv,
                                    gpu_paillier_ciphertext_t *gpu_ct,
                                    uint32_t count) {
  int32_t i;
  i = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
  if (i >= count) return;

  paillier_t<params> po(cgbn_report_monitor, report, i);
  typename paillier_t<params>::bn_t r, c, l, n, p, x;
  typename paillier_t<params>::bn_wide_t dr;

  cgbn_load(po._env, c, &(gpu_ct[i].c));
  cgbn_load(po._env, l, &(gpu_prv->lambda));
  cgbn_load(po._env, x, &(gpu_prv->x));
  cgbn_load(po._env, n, &(gpu_pub->n_squared));
  cgbn_load(po._env, p, &(gpu_pub->n));

  // Fake code:paillier.fixed_window_powm_odd(gpu_res[i].m, gpu_ct[i].c,
  // gpu_prv[i].lambda, gpu_pub[i].n_squared);
  po.fixed_window_powm_odd(r, c, l, n);
  // Fake code:paillier._env.sub_ui32(gpu_res[i].m, gpu_res[i].m, 1);
  po._env.sub_ui32(r, r, 1);
  // Fake code:paillier.div(gpu_res[i].m, gpu_res[i].m, gpu_pub->n);
  cgbn_div(po._env, r, r, p);
  // Fake code:paillier.mul(gpu_res[i].m, gpu_res[i].m, gpu_prv->x);
  cgbn_mul_wide(po._env, dr, r, x);  // 8192bits,should be fixed
  // Fake code:paillier.mod(gpu_res[i].m, gpu_pub->n);
  cgbn_rem_wide(po._env, r, dr, p);  // back to 4096,the high CGBN of num is
                                     // less than the denominator, denom.
  cgbn_store(po._env, &(gpu_res[i].m), r);
  return;
}

// Sum the encrypted values by multiplying the ciphertexts
template <class params>
__global__ void kernel_paillier_e_add(cgbn_error_report_t *report,
                                      gpu_paillier_ciphertext_t *gpu_res,
                                      gpu_paillier_pubkey_t *gpu_pub,
                                      gpu_paillier_ciphertext_t *gpu_ct0,
                                      gpu_paillier_ciphertext_t *gpu_ct1,
                                      uint32_t count) {
  int32_t i;
  i = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
  if (i >= count) return;

  paillier_t<params> po(cgbn_report_monitor, report, i);

  typename paillier_t<params>::bn_t r, c0, c1, n;
  typename paillier_t<params>::bn_wide_t dr;
  cgbn_load(po._env, c0, &(gpu_ct0[i].c));
  cgbn_load(po._env, c1, &(gpu_ct1[i].c));
  cgbn_load(po._env, n, &(gpu_pub->n_squared));
  // paillier.d_mul(gpu_res[i].c, gpu_ct0[i].c, gpu_ct1[i].c);
  cgbn_mul_wide(po._env, dr, c0, c1);  // dr=c0*c1,  dr is 8192bits
  // paillier.d_mod(gpu_res[i].c, gpu_pub->n_squared);
  cgbn_rem_wide(po._env, r, dr, n);  // back to 4096
  cgbn_store(po._env, &(gpu_res[i].c), r);
  return;
}

template <class params>
__global__ void kernel_paillier_e_sub(cgbn_error_report_t *report,
                                      gpu_paillier_ciphertext_t *gpu_res,
                                      gpu_paillier_pubkey_t *gpu_pub,
                                      gpu_paillier_ciphertext_t *gpu_ct0,
                                      gpu_paillier_ciphertext_t *gpu_ct1,
                                      uint32_t count) {
  int32_t i;
  i = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
  if (i >= count) return;

  paillier_t<params> po(cgbn_report_monitor, report, i);

  typename paillier_t<params>::bn_t r, c0, c1, n;
  typename paillier_t<params>::bn_wide_t dr;
  cgbn_load(po._env, c0, &(gpu_ct0[i].c));
  cgbn_load(po._env, c1, &(gpu_ct1[i].c));
  cgbn_load(po._env, n, &(gpu_pub->n_squared));

  cgbn_modular_inverse(po._env, r, c1, n);  // r=inv(c1)

  cgbn_mul_wide(po._env, dr, c0, r);  // dr=c0*r,  dr is 8192bits
  cgbn_rem_wide(po._env, r, dr, n);   // back to 4096
  cgbn_store(po._env, &(gpu_res[i].c), r);
  return;
}

template <class params>
__global__ void kernel_paillier_e_sub_ctpt(cgbn_error_report_t *report,
                                           gpu_paillier_ciphertext_t *gpu_res,
                                           gpu_paillier_pubkey_t *gpu_pub,
                                           gpu_paillier_ciphertext_t *gpu_ct,
                                           gpu_paillier_plaintext_t *gpu_pt,
                                           uint32_t count) {
  int32_t i;
  i = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
  if (i >= count) return;

  paillier_t<params> po(cgbn_report_monitor, report, i);
  typename paillier_t<params>::bn_t r, c, m, n, g, ri, ro;
  typename paillier_t<params>::bn_wide_t dr;
  cgbn_load(po._env, c, &(gpu_ct[i].c));
  cgbn_load(po._env, m, &(gpu_pt[i].m));
  cgbn_load(po._env, n, &(gpu_pub->n_squared));
  cgbn_load(po._env, g, &(gpu_pub->n_plusone));

  po.fixed_window_powm_odd(r, g, m, n);

  cgbn_modular_inverse(po._env, ri, r, n);  // ri=inv(r)

  cgbn_mul_wide(po._env, dr, c, ri);  // dr=c*ri,  dr is 8192bits

  cgbn_rem_wide(po._env, r, dr, n);  // back to 4096
  cgbn_store(po._env, &(gpu_res[i].c), r);

  return;
}

template <class params>
__global__ void kernel_paillier_e_sub_ptct(cgbn_error_report_t *report,
                                           gpu_paillier_ciphertext_t *gpu_res,
                                           gpu_paillier_pubkey_t *gpu_pub,
                                           gpu_paillier_plaintext_t *gpu_pt,
                                           gpu_paillier_ciphertext_t *gpu_ct,
                                           uint32_t count) {
  int32_t i;
  i = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
  if (i >= count) return;

  paillier_t<params> po(cgbn_report_monitor, report, i);
  typename paillier_t<params>::bn_t r, c, m, n, g, ri, rm;
  typename paillier_t<params>::bn_wide_t dr;
  cgbn_load(po._env, c, &(gpu_ct[i].c));
  cgbn_load(po._env, m, &(gpu_pt[i].m));
  cgbn_load(po._env, n, &(gpu_pub->n_squared));
  cgbn_load(po._env, g, &(gpu_pub->n_plusone));

  po.fixed_window_powm_odd(r, g, m, n);  // r=g^m mod n;

  cgbn_modular_inverse(po._env, ri, c, n);  // ri=inv(c)

  cgbn_mul_wide(po._env, dr, r, ri);  // dr=r*ri,  dr is 8192bits
  cgbn_rem_wide(po._env, r, dr, n);   // back to 4096
  cgbn_store(po._env, &(gpu_res[i].c), r);
  return;
}

// inv
template <class params>
__global__ void kernel_paillier_inv(cgbn_error_report_t *report,
                                    gpu_paillier_ciphertext_t *gpu_res,
                                    gpu_paillier_pubkey_t *gpu_pub,
                                    gpu_paillier_ciphertext_t *gpu_ctx,
                                    uint32_t count) {
  int32_t i;
  i = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
  if (i >= count) return;

  paillier_t<params> po(cgbn_report_monitor, report, i);
  typename paillier_t<params>::bn_t r, c, n;
  // typename paillier_t<params>::bn_wide_t  dr;
  cgbn_load(po._env, c, &(gpu_ctx[i].c));
  cgbn_load(po._env, n, &(gpu_pub->n_squared));

  cgbn_modular_inverse(po._env, r, c, n);

  cgbn_store(po._env, &(gpu_res[i].c), r);
  return;
}

// inv inplace , it can not work, because the memory is not managed by the GPU
template <class params>
__global__ void kernel_paillier_inv_inplace(cgbn_error_report_t *report,
                                            gpu_paillier_pubkey_t *gpu_pub,
                                            gpu_paillier_ciphertext_t *gpu_ctx,
                                            uint32_t count) {
  int32_t i;
  i = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
  if (i >= count) return;

  paillier_t<params> po(cgbn_report_monitor, report, i);
  typename paillier_t<params>::bn_t r, c, n;
  cgbn_load(po._env, c, &(gpu_ctx[i].c));
  cgbn_load(po._env, n, &(gpu_pub->n_squared));

  cgbn_modular_inverse(po._env, r, c, n);

  cgbn_store(po._env, &(gpu_ctx[i].c), r);  // replace the inpute
  return;
}

template <class params>
__global__ void kernel_paillier_e_add_const(cgbn_error_report_t *report,
                                            gpu_paillier_ciphertext_t *gpu_res,
                                            gpu_paillier_pubkey_t *gpu_pub,
                                            gpu_paillier_ciphertext_t *gpu_ct,
                                            gpu_paillier_plaintext_t *gpu_con,
                                            uint32_t count) {
  int32_t i;
  i = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
  if (i >= count) return;

  paillier_t<params> po(cgbn_report_monitor, report, i);
  typename paillier_t<params>::bn_t r, c, n, g, t;
  typename paillier_t<params>::bn_wide_t dr;

  cgbn_load(po._env, c, &(gpu_ct[i].c));
  cgbn_load(po._env, n, &(gpu_pub->n_squared));
  cgbn_load(po._env, g, &(gpu_pub->n_plusone));
  cgbn_load(po._env, t, &(gpu_con[i].m));

  // Fake code: po.d_fixed_window_powm_odd(gpu_res[i].c, gpu_pub->n_plusone,
  // gpu_con[i], gpu_pub->n_squared);
  po.fixed_window_powm_odd(r, g, t, n);
  // Fake code: po.d_mul(gpu_res[i].c, gpu_ct[i].c, gpu_res[i].c);
  cgbn_mul_wide(po._env, dr, c, r);  // dr=c0*c1,  dr is 8192bits
  // Fake code: po.d_mod(gpu_res[i].c,gpu_pub.n_squared);
  cgbn_rem_wide(po._env, r, dr, n);  // back to 4096
  cgbn_store(po._env, &(gpu_res[i].c), r);
  return;
}

template <class params>
__global__ void kernel_paillier_e_mul_const(cgbn_error_report_t *report,
                                            gpu_paillier_ciphertext_t *gpu_res,
                                            gpu_paillier_pubkey_t *gpu_pub,
                                            gpu_paillier_ciphertext_t *gpu_ct,
                                            gpu_paillier_plaintext_t *gpu_con,
                                            uint32_t count) {
  int32_t i;
  i = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
  if (i >= count) return;

  paillier_t<params> po(cgbn_report_monitor, report, i);
  typename paillier_t<params>::bn_t r, c, n, t;
  cgbn_load(po._env, c, &(gpu_ct[i].c));
  cgbn_load(po._env, n, &(gpu_pub->n_squared));
  cgbn_load(po._env, t, &(gpu_con[i].m));
  // Fake code:paillier.d_fixed_window_powm_odd(gpu_res[i].c, gpu_ct[i].c,
  // gpu_con[i].m, gpu_pub->n_squared);
  po.fixed_window_powm_odd(r, c, t, n);
  cgbn_store(po._env, &(gpu_res[i].c), r);
  return;
}

template <class params>
__global__ void kernel_paillier_compare(cgbn_error_report_t *report,
                                        gpu_paillier_plaintext_t *gpu_plain,
                                        uint32_t *gpu_res, uint32_t count) {
  int32_t i;
  i = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
  if (i >= count) return;

  int32_t j = -1;
  paillier_t<params> po(cgbn_report_monitor, report, i);
  typename paillier_t<params>::bn_t r;
  cgbn_load(po._env, r, &(gpu_plain[i].m));
  j = cgbn_compare_ui32(
      po._env, r, gpu_res[i]);  // compare the gpu result and the cpu result
  if (j != 0) printf("instance %d error: %u \n", i, gpu_res[i]);
  return;
  return;
}

void cudainit() {
  int count;
  cudaGetDeviceCount(&count);
  cudaError_t error_t = cudaSetDevice(0);
  if (error_t != cudaSuccess) printf("cuda error\n");
  error_t = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  if (error_t != cudaSuccess) printf("cuda error\n");

  // Ensure CUDA stream pool is initialized (happens automatically on first access)
  // The pool size is determined by HEU_CUDA_STREAM_POOL_SIZE environment variable
  // or defaults to 128 streams
  CudaStreamPool::GetInstance();

  // Initialize RMM memory pool to ensure it's ready before any allocations
  // This sets up the pool_memory_resource as the current device resource
  RMMMemoryPool::GetInstance().Initialize();
}

// Asynchronous version of gpu_paillier_enc using CUDA stream pool
// All threads share a single stream pool for concurrent execution
int gpu_paillier_enc(h_paillier_ciphertext_t *res, h_paillier_pubkey_t *pub,
                           h_paillier_plaintext_t *pt, h_paillier_random_t *rand,
                           unsigned int count) {
  int32_t TPB = (params::TPB == 0)
                    ? 64
                    : params::TPB;  // default threads per block to 128
  int32_t TPI = params::TPI, IPB = TPB / TPI;

  int32_t BPG = 256;

  CUDA_CHECK(cudaSetDevice(0));

  // Acquire a stream from the shared stream pool
  // StreamGuard automatically synchronizes on destruction
  auto stream = get_stream();

  // OPTIMIZATION: Use RMM device_buffer with stream-ordered allocation
  auto gpu_result = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);
  auto gpu_pub = make_device_buffer<gpu_paillier_pubkey_t>(1, stream);
  auto gpu_pt = make_device_buffer<gpu_paillier_plaintext_t>(count, stream);
  auto gpu_random = make_device_buffer<gpu_paillier_random_t>(count, stream);

  // Asynchronous memory copy: host to device
  CUDA_CHECK(cudaMemcpyAsync(gpu_pub.data(), pub, sizeof(gpu_paillier_pubkey_t),
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_pt.data(), (gpu_paillier_plaintext_t *)pt,
                             sizeof(gpu_paillier_plaintext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_random.data(), (gpu_paillier_random_t *)rand,
                             sizeof(gpu_paillier_random_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));

  // Launch kernel on stream
  auto* result_ptr = static_cast<gpu_paillier_ciphertext_t*>(gpu_result.data());
  auto* pub_ptr = static_cast<gpu_paillier_pubkey_t*>(gpu_pub.data());
  auto* pt_ptr = static_cast<gpu_paillier_plaintext_t*>(gpu_pt.data());
  auto* random_ptr = static_cast<gpu_paillier_random_t*>(gpu_random.data());

  unsigned int ps = TPB * BPG;  // kernel parallel
  if (ps < count) {
    unsigned int rep = count / ps;
    for (unsigned int i = 0; i < rep; i++) {
      kernel_paillier_enc<params><<<(ps + IPB - 1) / IPB, TPB, 0, stream.value()>>>(
          nullptr, &result_ptr[i * ps], pub_ptr, &pt_ptr[i * ps],
          &random_ptr[i * ps], ps);
    }
    unsigned int rem = count - ps * rep;
    if (rem > 0) {
      kernel_paillier_enc<params><<<(rem + IPB - 1) / IPB, TPB, 0, stream.value()>>>(
          nullptr, &result_ptr[rep * ps], pub_ptr, &pt_ptr[rep * ps],
          &random_ptr[rep * ps], rem);
    }
  } else {
    kernel_paillier_enc<params><<<(count + IPB - 1) / IPB, TPB, 0, stream.value()>>>(
        nullptr, result_ptr, pub_ptr, pt_ptr, random_ptr, count);
  }

  // Asynchronous memory copy: device to host
  CUDA_CHECK(cudaMemcpyAsync(res, gpu_result.data(),
                             sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyDeviceToHost, stream.value()));

  // Now safe to check errors after GPU operations complete
  CUDA_LAST_CHECK();

  // Stream is automatically returned to pool when stream_guard goes out of scope
  // device_buffer automatically frees memory when it goes out of scope
  return 0;
}

// Asynchronous version of gpu_paillier_dec using CUDA stream pool
// All threads share a single stream pool for concurrent execution
int gpu_paillier_dec(h_paillier_plaintext_t *res, h_paillier_pubkey_t *pub,
                     h_paillier_prvkey_t *prv, h_paillier_ciphertext_t *ct,
                     unsigned int count) {
  // unsigned int TPI, TPB, IPB;
  int32_t TPB = (params::TPB == 0)
                    ? 64
                    : params::TPB;  // default threads per block to 128
  int32_t TPI = params::TPI, IPB = TPB / TPI;

  // Acquire a stream from the shared stream pool
  StreamGuard stream_guard;
  rmm::cuda_stream_view stream = stream_guard.stream();

  // OPTIMIZATION: Use RMM device_buffer with stream-ordered allocation
  auto gpu_result = make_device_buffer<gpu_paillier_plaintext_t>(count, stream);
  auto gpu_pub = make_device_buffer<gpu_paillier_pubkey_t>(1, stream);
  auto gpu_prv = make_device_buffer<gpu_paillier_prvkey_t>(1, stream);
  auto gpu_ct = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);

  // Asynchronous memory copy: host to device
  CUDA_CHECK(cudaMemcpyAsync(gpu_pub.data(), pub, sizeof(gpu_paillier_pubkey_t),
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_prv.data(), prv, sizeof(gpu_paillier_prvkey_t),
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_ct.data(), ct, sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));

  // Launch kernel on stream
  kernel_paillier_dec<params><<<(count + IPB - 1) / IPB, TPB, 0, stream.value()>>>(
      nullptr,
      static_cast<gpu_paillier_plaintext_t*>(gpu_result.data()),
      static_cast<gpu_paillier_pubkey_t*>(gpu_pub.data()),
      static_cast<gpu_paillier_prvkey_t*>(gpu_prv.data()),
      static_cast<gpu_paillier_ciphertext_t*>(gpu_ct.data()),
      count);

  // Asynchronous memory copy: device to host
  CUDA_CHECK(cudaMemcpyAsync(res, gpu_result.data(),
                             sizeof(gpu_paillier_plaintext_t) * count,
                             cudaMemcpyDeviceToHost, stream.value()));

  // Now safe to check errors after GPU operations complete
  CUDA_LAST_CHECK();

  // Stream is automatically returned to pool when stream_guard goes out of scope
  // device_buffer automatically frees memory when it goes out of scope
  return 0;
}

// Asynchronous version of gpu_paillier_e_add using CUDA stream
// Each call creates its own stream for concurrent execution
int gpu_paillier_e_add(h_paillier_pubkey_t *pub, h_paillier_ciphertext_t *res,
                       h_paillier_ciphertext_t *ct0,
                       h_paillier_ciphertext_t *ct1, unsigned int count) {
  NvtxRange chunk_range("gpu_paillier_e_add");
  // unsigned int TPI, TPB, IPB;
  int32_t TPB = (params::TPB == 0)
                    ? 64
                    : params::TPB;  // default threads per block to 128
  int32_t TPI = params::TPI, IPB = TPB / TPI;

  // Acquire a stream from the shared stream pool
  StreamGuard stream_guard;
  rmm::cuda_stream_view stream = stream_guard.stream();

  // OPTIMIZATION: Use RMM device_buffer with stream-ordered allocation
  auto gpu_result = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);
  auto gpu_pub = make_device_buffer<gpu_paillier_pubkey_t>(1, stream);
  auto gpu_ct0 = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);
  auto gpu_ct1 = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);

  // Asynchronous memory copy: host to device
  CUDA_CHECK(cudaMemcpyAsync(gpu_pub.data(), pub, sizeof(gpu_paillier_pubkey_t),
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_ct0.data(), ct0, sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_ct1.data(), ct1, sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));

  // Launch kernel on stream
  kernel_paillier_e_add<params><<<(count + IPB - 1) / IPB, TPB, 0, stream.value()>>>(
      nullptr,
      static_cast<gpu_paillier_ciphertext_t*>(gpu_result.data()),
      static_cast<gpu_paillier_pubkey_t*>(gpu_pub.data()),
      static_cast<gpu_paillier_ciphertext_t*>(gpu_ct0.data()),
      static_cast<gpu_paillier_ciphertext_t*>(gpu_ct1.data()),
      count);

  // Asynchronous memory copy: device to host
  CUDA_CHECK(cudaMemcpyAsync(res, gpu_result.data(),
                             sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyDeviceToHost, stream.value()));

  // Now safe to check errors after GPU operations complete
  CUDA_LAST_CHECK();

  // device_buffer automatically frees memory when it goes out of scope
  return 0;
}

int gpu_paillier_e_inverse(h_paillier_pubkey_t *pub,
                           h_paillier_ciphertext_t *res,
                           h_paillier_ciphertext_t *ct, unsigned int count) {
  // unsigned int TPI, TPB, IPB;
  int32_t TPB = (params::TPB == 0)
                    ? 64
                    : params::TPB;  // default threads per block to 128
  int32_t TPI = params::TPI, IPB = TPB / TPI;

  // Acquire a stream from the shared stream pool
  StreamGuard stream_guard;
  rmm::cuda_stream_view stream = stream_guard.stream();

  // OPTIMIZATION: Use RMM device_buffer with stream-ordered allocation
  auto gpu_result = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);
  auto gpu_pub = make_device_buffer<gpu_paillier_pubkey_t>(1, stream);
  auto gpu_ct = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);

  // Asynchronous memory copy: host to device
  CUDA_CHECK(cudaMemcpyAsync(gpu_pub.data(), pub, sizeof(gpu_paillier_pubkey_t),
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_ct.data(), ct, sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));

  // Launch kernel on stream
  kernel_paillier_inv<params><<<(count + IPB - 1) / IPB, TPB, 0, stream.value()>>>(
      nullptr,
      static_cast<gpu_paillier_ciphertext_t*>(gpu_result.data()),
      static_cast<gpu_paillier_pubkey_t*>(gpu_pub.data()),
      static_cast<gpu_paillier_ciphertext_t*>(gpu_ct.data()),
      count);

  // Asynchronous memory copy: device to host
  CUDA_CHECK(cudaMemcpyAsync(res, gpu_result.data(),
                             sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyDeviceToHost, stream.value()));

  // Now safe to check errors after GPU operations complete
  CUDA_LAST_CHECK();

  // Stream is automatically returned to pool when stream_guard goes out of scope
  // device_buffer automatically frees memory when it goes out of scope
  return 0;
}

int gpu_paillier_e_add_const(h_paillier_pubkey_t *pub,
                             h_paillier_ciphertext_t *res,
                             h_paillier_ciphertext_t *ct,
                             h_paillier_plaintext_t *constant,
                             unsigned int count) {
  int32_t TPB = (params::TPB == 0)
                    ? 64
                    : params::TPB;  // default threads per block to 128
  int32_t TPI = params::TPI, IPB = TPB / TPI;

  // Acquire a stream from the shared stream pool
  StreamGuard stream_guard;
  rmm::cuda_stream_view stream = stream_guard.stream();

  // OPTIMIZATION: Use RMM device_buffer with stream-ordered allocation
  auto gpu_result = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);
  auto gpu_pub = make_device_buffer<gpu_paillier_pubkey_t>(1, stream);
  auto gpu_ct = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);
  auto gpu_constant = make_device_buffer<gpu_paillier_plaintext_t>(count, stream);

  // Asynchronous memory copy: host to device
  CUDA_CHECK(cudaMemcpyAsync(gpu_pub.data(), pub, sizeof(gpu_paillier_pubkey_t),
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_ct.data(), ct, sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_constant.data(), constant,
                             sizeof(gpu_paillier_plaintext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));

  // Launch kernel on stream
  kernel_paillier_e_add_const<params><<<(count + IPB - 1) / IPB, TPB, 0, stream.value()>>>(
      nullptr,
      static_cast<gpu_paillier_ciphertext_t*>(gpu_result.data()),
      static_cast<gpu_paillier_pubkey_t*>(gpu_pub.data()),
      static_cast<gpu_paillier_ciphertext_t*>(gpu_ct.data()),
      static_cast<gpu_paillier_plaintext_t*>(gpu_constant.data()),
      count);

  // Asynchronous memory copy: device to host
  CUDA_CHECK(cudaMemcpyAsync(res, gpu_result.data(),
                             sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyDeviceToHost, stream.value()));

  // Now safe to check errors after GPU operations complete
  CUDA_LAST_CHECK();

  // Stream is automatically returned to pool when stream_guard goes out of scope
  // device_buffer automatically frees memory when it goes out of scope
  return 0;
}

int gpu_paillier_sub_ct(h_paillier_pubkey_t *pub, h_paillier_ciphertext_t *res,
                        h_paillier_ciphertext_t *ct0,
                        h_paillier_ciphertext_t *ct1, unsigned int count) {
  // unsigned int TPI, TPB, IPB;
  int32_t TPB = (params::TPB == 0)
                    ? 64
                    : params::TPB;  // default threads per block to 128
  int32_t TPI = params::TPI, IPB = TPB / TPI;

  // Acquire a stream from the shared stream pool
  StreamGuard stream_guard;
  rmm::cuda_stream_view stream = stream_guard.stream();

  // OPTIMIZATION: Use RMM device_buffer with stream-ordered allocation
  auto gpu_result = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);
  auto gpu_pub = make_device_buffer<gpu_paillier_pubkey_t>(1, stream);
  auto gpu_ct0 = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);
  auto gpu_ct1 = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);

  // Asynchronous memory copy: host to device
  CUDA_CHECK(cudaMemcpyAsync(gpu_pub.data(), pub, sizeof(gpu_paillier_pubkey_t),
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_ct0.data(), ct0, sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_ct1.data(), ct1, sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));

  // Launch kernel on stream
  kernel_paillier_e_sub<params><<<(count + IPB - 1) / IPB, TPB, 0, stream.value()>>>(
      nullptr,
      static_cast<gpu_paillier_ciphertext_t*>(gpu_result.data()),
      static_cast<gpu_paillier_pubkey_t*>(gpu_pub.data()),
      static_cast<gpu_paillier_ciphertext_t*>(gpu_ct0.data()),
      static_cast<gpu_paillier_ciphertext_t*>(gpu_ct1.data()),
      count);

  // Asynchronous memory copy: device to host
  CUDA_CHECK(cudaMemcpyAsync(res, gpu_result.data(),
                             sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyDeviceToHost, stream.value()));

  // Now safe to check errors after GPU operations complete
  CUDA_LAST_CHECK();

  // Stream is automatically returned to pool when stream_guard goes out of scope
  // device_buffer automatically frees memory when it goes out of scope
  return 0;
}

int gpu_paillier_sub_ctpt(h_paillier_pubkey_t *pub,
                          h_paillier_ciphertext_t *res,
                          h_paillier_ciphertext_t *ct,
                          h_paillier_plaintext_t *pt, unsigned int count) {
  // unsigned int TPI, TPB, IPB;
  int32_t TPB = (params::TPB == 0)
                    ? 64
                    : params::TPB;  // default threads per block to 128
  int32_t TPI = params::TPI, IPB = TPB / TPI;

  // Acquire a stream from the shared stream pool
  StreamGuard stream_guard;
  rmm::cuda_stream_view stream = stream_guard.stream();

  // OPTIMIZATION: Use RMM device_buffer with stream-ordered allocation
  auto gpu_result = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);
  auto gpu_pub = make_device_buffer<gpu_paillier_pubkey_t>(1, stream);
  auto gpu_ct = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);
  auto gpu_pt = make_device_buffer<gpu_paillier_plaintext_t>(count, stream);

  // Asynchronous memory copy: host to device
  CUDA_CHECK(cudaMemcpyAsync(gpu_pub.data(), pub, sizeof(gpu_paillier_pubkey_t),
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_ct.data(), ct, sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_pt.data(), pt, sizeof(gpu_paillier_plaintext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));

  // Launch kernel on stream
  kernel_paillier_e_sub_ctpt<params><<<(count + IPB - 1) / IPB, TPB, 0, stream.value()>>>(
      nullptr,
      static_cast<gpu_paillier_ciphertext_t*>(gpu_result.data()),
      static_cast<gpu_paillier_pubkey_t*>(gpu_pub.data()),
      static_cast<gpu_paillier_ciphertext_t*>(gpu_ct.data()),
      static_cast<gpu_paillier_plaintext_t*>(gpu_pt.data()),
      count);

  // Asynchronous memory copy: device to host
  CUDA_CHECK(cudaMemcpyAsync(res, gpu_result.data(),
                             sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyDeviceToHost, stream.value()));

  // Now safe to check errors after GPU operations complete
  CUDA_LAST_CHECK();

  // Stream is automatically returned to pool when stream_guard goes out of scope
  // device_buffer automatically frees memory when it goes out of scope
  return 0;
}

int gpu_paillier_sub_ptct(h_paillier_pubkey_t *pub,
                          h_paillier_ciphertext_t *res,
                          h_paillier_plaintext_t *pt,
                          h_paillier_ciphertext_t *ct, unsigned int count) {
  // unsigned int TPI, TPB, IPB;
  int32_t TPB = (params::TPB == 0)
                    ? 64
                    : params::TPB;  // default threads per block to 128
  int32_t TPI = params::TPI, IPB = TPB / TPI;

  // Acquire a stream from the shared stream pool
  StreamGuard stream_guard;
  rmm::cuda_stream_view stream = stream_guard.stream();

  // OPTIMIZATION: Use memory pool with stream-ordered allocation
  auto gpu_result = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);
  auto gpu_pub = make_device_buffer<gpu_paillier_pubkey_t>(1, stream);
  auto gpu_ct = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);
  auto gpu_pt = make_device_buffer<gpu_paillier_plaintext_t>(count, stream);

  // Asynchronous memory copy: host to device
  CUDA_CHECK(cudaMemcpyAsync(gpu_pub.data(), pub, sizeof(gpu_paillier_pubkey_t),
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_ct.data(), ct, sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_pt.data(), pt, sizeof(gpu_paillier_plaintext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));

  // Launch kernel on stream
  kernel_paillier_e_sub_ptct<params><<<(count + IPB - 1) / IPB, TPB, 0, stream.value()>>>(
      nullptr,
      static_cast<gpu_paillier_ciphertext_t*>(gpu_result.data()),
      static_cast<gpu_paillier_pubkey_t*>(gpu_pub.data()),
      static_cast<gpu_paillier_plaintext_t*>(gpu_pt.data()),
      static_cast<gpu_paillier_ciphertext_t*>(gpu_ct.data()),
      count);

  // Asynchronous memory copy: device to host
  CUDA_CHECK(cudaMemcpyAsync(res, gpu_result.data(),
                             sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyDeviceToHost, stream.value()));

  // Now safe to check errors after GPU operations complete
  CUDA_LAST_CHECK();

  // Stream is automatically returned to pool when stream_guard goes out of scope
  // device_buffer automatically frees memory when it goes out of scope
  return 0;
}

int gpu_paillier_e_mul_const(h_paillier_pubkey_t *pub,
                             h_paillier_ciphertext_t *res,
                             h_paillier_ciphertext_t *ct,
                             h_paillier_plaintext_t *constant,
                             unsigned int count) {
  int32_t TPB = (params::TPB == 0)
                    ? 64
                    : params::TPB;  // default threads per block to 128
  int32_t TPI = params::TPI, IPB = TPB / TPI;

  // Acquire a stream from the shared stream pool
  StreamGuard stream_guard;
  rmm::cuda_stream_view stream = stream_guard.stream();

  // OPTIMIZATION: Use RMM device_buffer with stream-ordered allocation
  auto gpu_result = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);
  auto gpu_pub = make_device_buffer<gpu_paillier_pubkey_t>(1, stream);
  auto gpu_ct = make_device_buffer<gpu_paillier_ciphertext_t>(count, stream);
  auto gpu_constant = make_device_buffer<gpu_paillier_plaintext_t>(count, stream);

  // Asynchronous memory copy: host to device
  CUDA_CHECK(cudaMemcpyAsync(gpu_pub.data(), (gpu_paillier_pubkey_t *)pub,
                             sizeof(gpu_paillier_pubkey_t), cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_ct.data(), (gpu_paillier_ciphertext_t *)ct,
                             sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_constant.data(), constant,
                             sizeof(gpu_paillier_plaintext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));

  // Launch kernel on stream
  kernel_paillier_e_mul_const<params><<<(count + IPB - 1) / IPB, TPB, 0, stream.value()>>>(
      nullptr,
      static_cast<gpu_paillier_ciphertext_t*>(gpu_result.data()),
      static_cast<gpu_paillier_pubkey_t*>(gpu_pub.data()),
      static_cast<gpu_paillier_ciphertext_t*>(gpu_ct.data()),
      static_cast<gpu_paillier_plaintext_t*>(gpu_constant.data()),
      count);

  // Asynchronous memory copy: device to host
  CUDA_CHECK(cudaMemcpyAsync(res, gpu_result.data(),
                             sizeof(gpu_paillier_ciphertext_t) * count,
                             cudaMemcpyDeviceToHost, stream.value()));

  // Now safe to check errors after GPU operations complete
  CUDA_LAST_CHECK();

  // device_buffer automatically frees memory when it goes out of scope
  return 0;
}

int gpu_paillier_compare(h_paillier_plaintext_t *plain, unsigned int *res,
                         unsigned int count) {
  int32_t TPB = (params::TPB == 0)
                    ? 64
                    : params::TPB;  // default threads per block to 128
  int32_t TPI = params::TPI, IPB = TPB / TPI;  // IPB is instances per block

  // Acquire a stream from the shared stream pool
  StreamGuard stream_guard;
  rmm::cuda_stream_view stream = stream_guard.stream();

  // OPTIMIZATION: Use RMM device_buffer with stream-ordered allocation
  auto gpu_plain = make_device_buffer<gpu_paillier_plaintext_t>(count, stream);
  auto gpu_res = make_device_buffer<unsigned int>(count, stream);

  // Asynchronous memory copy: host to device
  CUDA_CHECK(cudaMemcpyAsync(gpu_plain.data(), plain,
                             sizeof(gpu_paillier_plaintext_t) * count,
                             cudaMemcpyHostToDevice, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(gpu_res.data(), res, sizeof(unsigned int) * count,
                             cudaMemcpyHostToDevice, stream.value()));

  // Launch kernel on stream
  kernel_paillier_compare<params><<<(count + IPB - 1) / IPB, TPB, 0, stream.value()>>>(
      nullptr,
      static_cast<gpu_paillier_plaintext_t*>(gpu_plain.data()),
      static_cast<unsigned int*>(gpu_res.data()),
      count);

  // Now safe to check errors after GPU operations complete
  CUDA_LAST_CHECK();

  // Stream is automatically returned to pool when stream_guard goes out of scope
  // device_buffer automatically frees memory when it goes out of scope
  return 0;
}
