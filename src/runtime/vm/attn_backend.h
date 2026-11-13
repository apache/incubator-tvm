/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/vm/attn_backend.h
 * \brief The attention backend classes used by KV cache.
 */

#ifndef TVM_RUNTIME_VM_ATTN_BACKEND_H_
#define TVM_RUNTIME_VM_ATTN_BACKEND_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>

#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "attn_utils.h"

namespace tvm {
namespace runtime {
namespace vm {

/*! \brief The attention backend kinds. */
enum class AttnBackendKind : int {
  kTIR = 0,
  kFlashInfer = 1,
};

/*!
 * \brief Return a zero-copy alias of \p t whose `byte_offset` is folded into the
 * data pointer, so the resulting tensor has `byte_offset == 0`.
 *
 * FlashInfer 0.6.3 kernels read tensors from `data` directly and do NOT honor
 * the DLPack `byte_offset` field. mlc's auxiliary index tensors (qo_indptr,
 * kv_indptr, page_indptr, page_indices, length_info, ...) are views packed into
 * a shared workspace and therefore carry a non-zero `byte_offset`. Passing them
 * as-is makes FlashInfer read the wrong addresses; this helper rebases them.
 */
inline ffi::Tensor ZeroByteOffsetView(const ffi::Tensor& t) {
  TVM_FFI_ICHECK(t.defined());
  if (t.byte_offset() == 0) return t;
  auto* holder = new ffi::Tensor(t);  // keep the underlying storage alive
  auto* managed = new DLManagedTensor();
  managed->manager_ctx = holder;
  managed->deleter = [](DLManagedTensor* self) {
    delete[] self->dl_tensor.shape;
    delete[] self->dl_tensor.strides;
    delete static_cast<ffi::Tensor*>(self->manager_ctx);
    delete self;
  };
  DLTensor& dl = managed->dl_tensor;
  dl.data = static_cast<void*>(static_cast<char*>(t.data_ptr()) + t.byte_offset());
  dl.device = t.device();
  dl.ndim = t.ndim();
  dl.dtype = t.dtype();
  dl.shape = new int64_t[t.ndim()];
  dl.strides = nullptr;
  for (int i = 0; i < t.ndim(); ++i) dl.shape[i] = t.shape()[i];
  if (t.GetDLTensorPtr()->strides != nullptr) {
    dl.strides = new int64_t[t.ndim()];
    for (int i = 0; i < t.ndim(); ++i) dl.strides[i] = t.strides()[i];
  }
  dl.byte_offset = 0;
  return tvm::ffi::Tensor::FromDLPack(managed, /*require_alignment=*/0,
                                      /*require_contiguous=*/false);
}

/*!
 * \brief Build a strided, zero-copy view selecting the key (which=0) or value
 * (which=1) sub-tensor from a combined paged KV tensor of shape
 * (num_pages, 2, num_heads, page_size, head_dim), yielding a
 * (num_pages, num_heads, page_size, head_dim) tensor that shares storage with
 * `pages`. FlashInfer 0.6.3 takes separate key/value paged caches and reads the
 * tensor strides, so a strided view avoids an explicit split/copy.
 */
inline ffi::Tensor PagedKVCacheView(const ffi::Tensor& pages, int64_t which) {
  TVM_FFI_ICHECK(pages.defined());
  TVM_FFI_ICHECK_EQ(pages.ndim(), 5);
  TVM_FFI_ICHECK_EQ(pages.shape()[1], 2);
  int64_t num_pages = pages.shape()[0];
  int64_t num_heads = pages.shape()[2];
  int64_t page_size = pages.shape()[3];
  int64_t head_dim = pages.shape()[4];
  int64_t inner = num_heads * page_size * head_dim;
  int64_t elem_bytes = (pages.dtype().bits * pages.dtype().lanes + 7) / 8;

  auto* holder = new ffi::Tensor(pages);  // keep the underlying storage alive
  auto* managed = new DLManagedTensor();
  managed->manager_ctx = holder;
  managed->deleter = [](DLManagedTensor* self) {
    delete[] self->dl_tensor.shape;
    delete[] self->dl_tensor.strides;
    delete static_cast<ffi::Tensor*>(self->manager_ctx);
    delete self;
  };
  DLTensor& dl = managed->dl_tensor;
  dl.data = static_cast<void*>(static_cast<char*>(pages.data_ptr()) + pages.byte_offset() +
                               which * inner * elem_bytes);
  dl.device = pages.device();
  dl.ndim = 4;
  dl.dtype = pages.dtype();
  dl.shape = new int64_t[4]{num_pages, num_heads, page_size, head_dim};
  dl.strides = new int64_t[4]{2 * inner, page_size * head_dim, head_dim, 1};
  dl.byte_offset = 0;
  return tvm::ffi::Tensor::FromDLPack(managed, /*require_alignment=*/0,
                                      /*require_contiguous=*/false);
}

/*!
 * \brief Return a strided, zero-copy view selecting the `[start, start+length)`
 * slice along the LAST dimension of \p t, preserving all other strides and
 * folding the slice offset into the data pointer (so `byte_offset == 0`).
 *
 * Used to split MLA tensors that store two head components concatenated along
 * the last dim: the query into `q_nope`/`q_pe` and the paged cache into
 * `ckv_cache`/`kpe_cache`. FlashInfer reads tensor strides and ignores
 * `byte_offset`, so a strided slice avoids a copy.
 */
inline ffi::Tensor SliceLastDimView(const ffi::Tensor& t, int64_t start, int64_t length) {
  TVM_FFI_ICHECK(t.defined());
  int ndim = t.ndim();
  int64_t elem_bytes = (t.dtype().bits * t.dtype().lanes + 7) / 8;
  std::vector<int64_t> in_strides(ndim);
  if (t.GetDLTensorPtr()->strides != nullptr) {
    for (int i = 0; i < ndim; ++i) in_strides[i] = t.strides()[i];
  } else {
    int64_t s = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      in_strides[i] = s;
      s *= t.shape()[i];
    }
  }
  auto* holder = new ffi::Tensor(t);  // keep the underlying storage alive
  auto* managed = new DLManagedTensor();
  managed->manager_ctx = holder;
  managed->deleter = [](DLManagedTensor* self) {
    delete[] self->dl_tensor.shape;
    delete[] self->dl_tensor.strides;
    delete static_cast<ffi::Tensor*>(self->manager_ctx);
    delete self;
  };
  DLTensor& dl = managed->dl_tensor;
  dl.data = static_cast<void*>(static_cast<char*>(t.data_ptr()) + t.byte_offset() +
                               start * in_strides[ndim - 1] * elem_bytes);
  dl.device = t.device();
  dl.ndim = ndim;
  dl.dtype = t.dtype();
  dl.shape = new int64_t[ndim];
  dl.strides = new int64_t[ndim];
  for (int i = 0; i < ndim; ++i) {
    dl.shape[i] = t.shape()[i];
    dl.strides[i] = in_strides[i];
  }
  dl.shape[ndim - 1] = length;
  dl.byte_offset = 0;
  return tvm::ffi::Tensor::FromDLPack(managed, /*require_alignment=*/0,
                                      /*require_contiguous=*/false);
}

/*! \brief The base class of attention backends. */
class AttnBackendFunc {
 public:
  explicit AttnBackendFunc(ffi::Function attn_func, AttnKind attn_kind,
                           AttnBackendKind backend_kind)
      : attn_func_(std::move(attn_func)), attn_kind(attn_kind), backend_kind(backend_kind) {}

  virtual ~AttnBackendFunc() = default;

 protected:
  ffi::Function attn_func_;

 public:
  AttnKind attn_kind;
  AttnBackendKind backend_kind;
};

/*! \brief The paged prefill attention function base class. */
class PagedPrefillFunc : public AttnBackendFunc {
 public:
  explicit PagedPrefillFunc(ffi::Function attn_func, AttnKind attn_kind,
                            AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(int depth, ffi::Tensor q, ffi::Tensor qo_indptr, ffi::Tensor pages,
                   ffi::Tensor page_indptr, ffi::Tensor page_indices, ffi::Tensor length_info,
                   ffi::Tensor q_rope_position, ffi::Tensor k_rope_pos_offset, bool causal,
                   RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
                   ffi::Tensor attn_output, ffi::Tensor attn_lse, TVMStreamHandle compute_stream) {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(pages.defined());
    TVM_FFI_ICHECK(page_indptr.defined());
    TVM_FFI_ICHECK(page_indices.defined());
    TVM_FFI_ICHECK(length_info.defined());
    TVM_FFI_ICHECK(q_rope_position.defined());
    TVM_FFI_ICHECK(k_rope_pos_offset.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    TVM_FFI_THROW(InternalError) << "MHA computation is not supported by the current backend";
  }

  virtual void MLA(int depth, ffi::Tensor q, ffi::Tensor qo_indptr, ffi::Tensor pages,
                   ffi::Tensor page_indptr, ffi::Tensor page_indices, ffi::Tensor length_info,
                   bool causal, double sm_scale, ffi::Tensor attn_output, ffi::Tensor attn_lse,
                   TVMStreamHandle compute_stream) {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(pages.defined());
    TVM_FFI_ICHECK(page_indptr.defined());
    TVM_FFI_ICHECK(page_indices.defined());
    TVM_FFI_ICHECK(length_info.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    TVM_FFI_THROW(InternalError) << "MLA computation is not supported by the current backend";
  }

  virtual void BeginForward(int depth, ffi::Tensor float_workspace_buffer,
                            ffi::Tensor int_workspace_buffer,
                            ffi::Tensor page_locked_int_workspace_buffer,
                            HostMemoryVector* qo_indptr, HostMemoryVector* page_indptr,
                            HostMemoryVector* last_page_len, int64_t batch_size,
                            int64_t total_qo_len, int64_t page_size, int64_t num_qo_heads,
                            int64_t num_kv_heads, int64_t qk_head_dim, int64_t v_head_dim,
                            bool causal, TVMStreamHandle copy_stream) {
    TVM_FFI_ICHECK(float_workspace_buffer.defined());
    TVM_FFI_ICHECK(int_workspace_buffer.defined());
    TVM_FFI_ICHECK(page_locked_int_workspace_buffer.defined());
    // Do nothing. Subclasses can override to customize behavior.
  }
};

/*! \brief The TIR-based paged prefill attention function class. */
class TIRPagedPrefillFunc : public PagedPrefillFunc {
 public:
  explicit TIRPagedPrefillFunc(ffi::Function attn_func, AttnKind attn_kind)
      : PagedPrefillFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(int depth, ffi::Tensor q, ffi::Tensor qo_indptr, ffi::Tensor pages,
           ffi::Tensor page_indptr, ffi::Tensor page_indices, ffi::Tensor length_info,
           ffi::Tensor q_rope_position, ffi::Tensor k_rope_pos_offset, bool causal,
           RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
           ffi::Tensor attn_output, ffi::Tensor attn_lse, TVMStreamHandle compute_stream) final {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(pages.defined());
    TVM_FFI_ICHECK(page_indptr.defined());
    TVM_FFI_ICHECK(page_indices.defined());
    TVM_FFI_ICHECK(length_info.defined());
    TVM_FFI_ICHECK(q_rope_position.defined());
    TVM_FFI_ICHECK(k_rope_pos_offset.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    attn_func_(q, qo_indptr, pages, page_indptr, page_indices, length_info, k_rope_pos_offset,
               q_rope_position, attn_output, attn_lse, static_cast<int64_t>(causal),
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale);
  }

  void MLA(int depth, ffi::Tensor q, ffi::Tensor qo_indptr, ffi::Tensor pages,
           ffi::Tensor page_indptr, ffi::Tensor page_indices, ffi::Tensor length_info, bool causal,
           double sm_scale, ffi::Tensor attn_output, ffi::Tensor attn_lse,
           TVMStreamHandle compute_stream) final {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(pages.defined());
    TVM_FFI_ICHECK(page_indptr.defined());
    TVM_FFI_ICHECK(page_indices.defined());
    TVM_FFI_ICHECK(length_info.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    attn_func_(q, qo_indptr, pages, page_indptr, page_indices, length_info, attn_output, attn_lse,
               static_cast<int64_t>(causal), sm_scale);
  }
};

/*! \brief The FlashInfer-based paged prefill attention function class. */
class FlashInferPagedPrefillFunc : public PagedPrefillFunc {
 public:
  explicit FlashInferPagedPrefillFunc(ffi::Function attn_func, ffi::Function plan_func,
                                      AttnKind attn_kind)
      : PagedPrefillFunc(std::move(attn_func), attn_kind, AttnBackendKind::kFlashInfer),
        plan_func_(std::move(plan_func)) {}

  void MHA(int depth, ffi::Tensor q, ffi::Tensor qo_indptr, ffi::Tensor pages,
           ffi::Tensor page_indptr, ffi::Tensor page_indices, ffi::Tensor length_info,
           ffi::Tensor q_rope_position, ffi::Tensor k_rope_pos_offset, bool causal,
           RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
           ffi::Tensor attn_output, ffi::Tensor attn_lse, TVMStreamHandle compute_stream) final {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(pages.defined());
    TVM_FFI_ICHECK(page_indptr.defined());
    TVM_FFI_ICHECK(page_indices.defined());
    TVM_FFI_ICHECK(length_info.defined());
    TVM_FFI_ICHECK(q_rope_position.defined());
    TVM_FFI_ICHECK(k_rope_pos_offset.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    auto [float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
          plan_info_vec] = cached_buffers_[depth].value();
    double rope_rcp_scale = 1 / rotary_scale;
    double rope_rcp_theta = 1 / rotary_theta;
    attn_func_(
        float_workspace_buffer, int_workspace_buffer, plan_info_vec, q, PagedKVCacheView(pages, 0),
        PagedKVCacheView(pages, 1), ZeroByteOffsetView(qo_indptr), ZeroByteOffsetView(page_indptr),
        ZeroByteOffsetView(page_indices), ZeroByteOffsetView(length_info), attn_output, attn_lse,
        /*mask_mode_code=*/static_cast<int64_t>(causal),
        /*layout(HND)=*/1, /*window_left=*/-1, /*enable_pdl=*/false, sm_scale,
        /*rope_rcp_scale=*/rope_rcp_scale, /*rope_rcp_theta=*/rope_rcp_theta);
  }

  void MLA(int depth, ffi::Tensor q, ffi::Tensor qo_indptr, ffi::Tensor pages,
           ffi::Tensor page_indptr, ffi::Tensor page_indices, ffi::Tensor length_info, bool causal,
           double sm_scale, ffi::Tensor attn_output, ffi::Tensor attn_lse,
           TVMStreamHandle compute_stream) final {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(pages.defined());
    TVM_FFI_ICHECK(page_indptr.defined());
    TVM_FFI_ICHECK(page_indices.defined());
    TVM_FFI_ICHECK(length_info.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    auto [float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
          plan_info_vec] = cached_buffers_[depth].value();
    // FlashInfer's MLA run takes the query split into its compressed (nope) and
    // positional-embedding (pe) parts, and the paged cache split into the
    // compressed-kv cache (ckv) and key-positional-embedding cache (kpe). Both
    // q ([n, num_heads, ckv+kpe]) and pages ([num_pages, page_size, ckv+kpe])
    // store the two components concatenated along the last dimension.
    int64_t head_dim_ckv = mla_head_dim_ckv_;
    int64_t head_dim_kpe = mla_head_dim_kpe_;
    TVM_FFI_ICHECK_GE(head_dim_ckv, 0)
        << "MLA head dims are unset; BeginForward must run before MLA.";
    attn_func_(float_workspace_buffer, int_workspace_buffer, plan_info_vec,
               SliceLastDimView(q, 0, head_dim_ckv),
               SliceLastDimView(q, head_dim_ckv, head_dim_kpe),
               SliceLastDimView(pages, 0, head_dim_ckv),
               SliceLastDimView(pages, head_dim_ckv, head_dim_kpe),
               ZeroByteOffsetView(page_indices), attn_output, attn_lse,
               /*mask_mode_code=*/static_cast<int64_t>(causal), /*num_heads=*/q.shape()[1],
               /*page_size=*/pages.shape()[1], sm_scale, /*return_lse_base_on_e=*/false);
  }

  void BeginForward(int depth, ffi::Tensor float_workspace_buffer, ffi::Tensor int_workspace_buffer,
                    ffi::Tensor page_locked_int_workspace_buffer, HostMemoryVector* qo_indptr,
                    HostMemoryVector* page_indptr, HostMemoryVector* last_page_len,
                    int64_t batch_size, int64_t total_qo_len, int64_t page_size,
                    int64_t num_qo_heads, int64_t num_kv_heads, int64_t qk_head_dim,
                    int64_t v_head_dim, bool causal, TVMStreamHandle copy_stream) final {
    TVM_FFI_ICHECK(float_workspace_buffer.defined());
    TVM_FFI_ICHECK(int_workspace_buffer.defined());
    TVM_FFI_ICHECK(page_locked_int_workspace_buffer.defined());
    // FlashInfer expects kv_len as an (int32) tensor rather than a shape tuple.
    HostMemoryVector kv_len_arr(batch_size, DLDataType{kDLInt, 32, 1},
                                qo_indptr->as_tensor().device());
    for (int i = 0; i < static_cast<int>(batch_size); ++i) {
      kv_len_arr.push_back(static_cast<int32_t>(
          (*page_indptr)[i + 1] != (*page_indptr)[i]
              ? ((*page_indptr)[i + 1] - (*page_indptr)[i] - 1) * page_size + (*last_page_len)[i]
              : 0));
    }
    ffi::Array<int64_t> plan_info_vec;
    if (attn_kind == AttnKind::kMHA) {
      // Todo(tvm-team): enable cuda graph
      plan_info_vec =
          plan_func_(float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
                     qo_indptr->as_tensor(), page_indptr->as_tensor(), kv_len_arr.as_tensor(),
                     total_qo_len, batch_size, num_qo_heads, num_kv_heads, page_size,
                     /*enable_cuda_graph=*/false, qk_head_dim, v_head_dim, causal,
                     /*window_left=*/-1, /*fixed_split_size=*/-1, /*disable_split_kv=*/false,
                     /*num_colocated_ctas=*/0)
              .cast<ffi::Array<int64_t>>();
    } else if (attn_kind == AttnKind::kMLA) {
      // For MLA the compressed-kv head dim equals the output (v) head dim, and
      // the remaining part of qk_head_dim is the key positional embedding. Cache
      // them for the run, which must split q/pages into ckv and kpe components.
      mla_head_dim_ckv_ = v_head_dim;
      mla_head_dim_kpe_ = qk_head_dim - v_head_dim;
      plan_info_vec =
          plan_func_(float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
                     qo_indptr->as_tensor(), page_indptr->as_tensor(), kv_len_arr.as_tensor(),
                     num_qo_heads, v_head_dim, causal)
              .cast<ffi::Array<int64_t>>();
    }

    if (cached_buffers_.size() <= static_cast<size_t>(depth)) {
      cached_buffers_.resize(depth + 1);
    }
    cached_buffers_[depth] =
        std::make_tuple(float_workspace_buffer, int_workspace_buffer,
                        page_locked_int_workspace_buffer, std::move(plan_info_vec));
  }

 private:
  ffi::Function plan_func_;
  std::vector<std::optional<std::tuple<ffi::Tensor, ffi::Tensor, ffi::Tensor, ffi::Array<int64_t>>>>
      cached_buffers_;
  // MLA-only: the compressed-kv and key-positional-embedding head dims, used to
  // split q/pages in the run. Set during BeginForward for the kMLA attn kind.
  int64_t mla_head_dim_ckv_ = -1;
  int64_t mla_head_dim_kpe_ = -1;
};

/*! \brief The ragged prefill attention function base class. */
class RaggedPrefillFunc : public AttnBackendFunc {
 public:
  explicit RaggedPrefillFunc(ffi::Function attn_func, AttnKind attn_kind,
                             AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(ffi::Tensor q, ffi::Tensor k, ffi::Tensor v, ffi::Tensor qo_indptr,
                   ffi::Tensor kv_indptr, ffi::Tensor q_rope_position,
                   ffi::Tensor k_rope_pos_offset, bool causal, RoPEMode rope_mode,
                   double rotary_scale, double rotary_theta, double sm_scale,
                   ffi::Tensor attn_output, ffi::Tensor attn_lse, TVMStreamHandle compute_stream) {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(k.defined());
    TVM_FFI_ICHECK(v.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(kv_indptr.defined());
    TVM_FFI_ICHECK(q_rope_position.defined());
    TVM_FFI_ICHECK(k_rope_pos_offset.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    TVM_FFI_THROW(InternalError) << "MHA computation is not supported by the current backend";
  }

  virtual void BeginForward(ffi::Tensor float_workspace_buffer, ffi::Tensor int_workspace_buffer,
                            ffi::Tensor page_locked_int_workspace_buffer,
                            HostMemoryVector* qo_indptr, HostMemoryVector* kv_indptr,
                            int64_t batch_size, int64_t total_qo_len, int64_t num_qo_heads,
                            int64_t num_kv_heads, int64_t qk_head_dim, int64_t v_head_dim,
                            bool causal, TVMStreamHandle copy_stream) {
    TVM_FFI_ICHECK(float_workspace_buffer.defined());
    TVM_FFI_ICHECK(int_workspace_buffer.defined());
    TVM_FFI_ICHECK(page_locked_int_workspace_buffer.defined());
    // Do nothing. Subclasses can override to customize behavior.
  }
};

/*! \brief The TIR-based ragged prefill attention function class. */
class TIRRaggedPrefillFunc : public RaggedPrefillFunc {
 public:
  explicit TIRRaggedPrefillFunc(ffi::Function attn_func, AttnKind attn_kind)
      : RaggedPrefillFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(ffi::Tensor q, ffi::Tensor k, ffi::Tensor v, ffi::Tensor qo_indptr,
           ffi::Tensor kv_indptr, ffi::Tensor q_rope_position, ffi::Tensor k_rope_pos_offset,
           bool causal, RoPEMode rope_mode, double rotary_scale, double rotary_theta,
           double sm_scale, ffi::Tensor attn_output, ffi::Tensor attn_lse,
           TVMStreamHandle compute_stream) final {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(k.defined());
    TVM_FFI_ICHECK(v.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(kv_indptr.defined());
    TVM_FFI_ICHECK(q_rope_position.defined());
    TVM_FFI_ICHECK(k_rope_pos_offset.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    attn_func_(q, qo_indptr, k, v, kv_indptr, q_rope_position, k_rope_pos_offset, attn_output,
               attn_lse, static_cast<int64_t>(causal),
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale);
  }
};

/*! \brief The FlashInfer-based ragged prefill attention function class. */
class FlashInferRaggedPrefillFunc : public RaggedPrefillFunc {
 public:
  explicit FlashInferRaggedPrefillFunc(ffi::Function attn_func, ffi::Function plan_func,
                                       AttnKind attn_kind, int64_t qk_head_dim_override = -1,
                                       int64_t v_head_dim_override = -1)
      : RaggedPrefillFunc(std::move(attn_func), attn_kind, AttnBackendKind::kFlashInfer),
        plan_func_(std::move(plan_func)),
        qk_head_dim_override_(qk_head_dim_override),
        v_head_dim_override_(v_head_dim_override) {}

  void MHA(ffi::Tensor q, ffi::Tensor k, ffi::Tensor v, ffi::Tensor qo_indptr,
           ffi::Tensor kv_indptr, ffi::Tensor q_rope_position, ffi::Tensor k_rope_pos_offset,
           bool causal, RoPEMode rope_mode, double rotary_scale, double rotary_theta,
           double sm_scale, ffi::Tensor attn_output, ffi::Tensor attn_lse,
           TVMStreamHandle compute_stream) final {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(k.defined());
    TVM_FFI_ICHECK(v.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(kv_indptr.defined());
    TVM_FFI_ICHECK(q_rope_position.defined());
    TVM_FFI_ICHECK(k_rope_pos_offset.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    double rope_rcp_scale = 1 / rotary_scale;
    double rope_rcp_theta = 1 / rotary_theta;
    attn_func_(float_workspace_buffer_.value(), int_workspace_buffer_.value(), plan_info_vec_, q, k,
               v, ZeroByteOffsetView(qo_indptr), ZeroByteOffsetView(kv_indptr), attn_output,
               attn_lse,
               /*mask_mode_code=*/static_cast<int64_t>(causal),
               /*layout(NHD)=*/0, /*window_left=*/-1, /*enable_pdl=*/false, sm_scale,
               /*rope_rcp_scale=*/rope_rcp_scale,
               /*rope_rcp_theta=*/rope_rcp_theta);
  }

  void BeginForward(ffi::Tensor float_workspace_buffer, ffi::Tensor int_workspace_buffer,
                    ffi::Tensor page_locked_int_workspace_buffer, HostMemoryVector* qo_indptr,
                    HostMemoryVector* kv_indptr, int64_t batch_size, int64_t total_qo_len,
                    int64_t num_qo_heads, int64_t num_kv_heads, int64_t qk_head_dim,
                    int64_t v_head_dim, bool causal, TVMStreamHandle copy_stream) final {
    TVM_FFI_ICHECK(float_workspace_buffer.defined());
    TVM_FFI_ICHECK(int_workspace_buffer.defined());
    TVM_FFI_ICHECK(page_locked_int_workspace_buffer.defined());
    // For MLA self-attention the ragged kernel operates on different head dims
    // than the (compressed) MLA cache, so they are supplied per-function via the
    // backend spec and override the cache-derived dims passed by the caller. MLA
    // self-attention is full multi-head (one kv head per query head), unlike the
    // single-head compressed cache, so the kv head count is overridden too.
    if (qk_head_dim_override_ >= 0) qk_head_dim = qk_head_dim_override_;
    if (v_head_dim_override_ >= 0) {
      v_head_dim = v_head_dim_override_;
      num_kv_heads = num_qo_heads;
    }
    // FlashInfer expects kv_len as an (int32) tensor rather than a shape tuple.
    HostMemoryVector kv_len_arr(batch_size, DLDataType{kDLInt, 32, 1},
                                qo_indptr->as_tensor().device());
    for (int i = 0; i < static_cast<int>(batch_size); ++i) {
      kv_len_arr.push_back(static_cast<int32_t>((*kv_indptr)[i + 1] - (*kv_indptr)[i]));
    }
    // Todo(tvm-team): enable cuda graph
    float_workspace_buffer_ = float_workspace_buffer;
    int_workspace_buffer_ = int_workspace_buffer;
    page_locked_int_workspace_buffer_ = page_locked_int_workspace_buffer;
    plan_info_vec_ =
        plan_func_(float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
                   qo_indptr->as_tensor(), kv_indptr->as_tensor(), kv_len_arr.as_tensor(),
                   total_qo_len, batch_size, num_qo_heads, num_kv_heads, /*page_size=*/1,
                   /*enable_cuda_graph=*/false, qk_head_dim, v_head_dim, causal,
                   /*window_left=*/-1, /*fixed_split_size=*/-1, /*disable_split_kv=*/false,
                   /*num_colocated_ctas=*/0)
            .cast<ffi::Array<int64_t>>();
  }

 private:
  ffi::Function plan_func_;
  ffi::Optional<ffi::Tensor> float_workspace_buffer_;
  ffi::Optional<ffi::Tensor> int_workspace_buffer_;
  ffi::Optional<ffi::Tensor> page_locked_int_workspace_buffer_;
  ffi::Array<int64_t> plan_info_vec_;
  // MLA self-attention head dims supplied via the backend spec; -1 means use the
  // dims passed by the caller (the regular MHA case).
  int64_t qk_head_dim_override_ = -1;
  int64_t v_head_dim_override_ = -1;
};

/*! \brief The paged decode attention function base class. */
class PagedDecodeFunc : public AttnBackendFunc {
 public:
  explicit PagedDecodeFunc(ffi::Function attn_func, AttnKind attn_kind,
                           AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(int depth, ffi::Tensor q, ffi::Tensor pages, ffi::Tensor page_indptr,
                   ffi::Tensor page_indices, ffi::Tensor length_info, ffi::Tensor k_rope_pos_offset,
                   ffi::Tensor q_rope_position, RoPEMode rope_mode, double rotary_scale,
                   double rotary_theta, double sm_scale, ffi::Tensor attn_output,
                   ffi::Tensor attn_lse, TVMStreamHandle compute_stream) {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(pages.defined());
    TVM_FFI_ICHECK(page_indptr.defined());
    TVM_FFI_ICHECK(page_indices.defined());
    TVM_FFI_ICHECK(length_info.defined());
    TVM_FFI_ICHECK(k_rope_pos_offset.defined());
    TVM_FFI_ICHECK(q_rope_position.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    TVM_FFI_THROW(InternalError) << "MHA computation is not supported by the current backend";
  }

  virtual void MLA(int depth, ffi::Tensor q, ffi::Tensor pages, ffi::Tensor page_indptr,
                   ffi::Tensor page_indices, ffi::Tensor length_info, double sm_scale,
                   ffi::Tensor attn_output, ffi::Tensor attn_lse, TVMStreamHandle compute_stream) {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(pages.defined());
    TVM_FFI_ICHECK(page_indptr.defined());
    TVM_FFI_ICHECK(page_indices.defined());
    TVM_FFI_ICHECK(length_info.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    TVM_FFI_THROW(InternalError) << "MLA computation is not supported by the current backend";
  }

  virtual void BeginForward(int depth, ffi::Tensor float_workspace_buffer,
                            ffi::Tensor int_workspace_buffer,
                            ffi::Tensor page_locked_int_workspace_buffer,
                            HostMemoryVector* page_indptr, int64_t batch_size, int64_t page_size,
                            int64_t num_qo_heads, int64_t num_kv_heads, int64_t qk_head_dim,
                            int64_t v_head_dim, RoPEMode rope_mode, DLDataType q_dtype,
                            DLDataType kv_dtype, TVMStreamHandle copy_stream) {
    TVM_FFI_ICHECK(float_workspace_buffer.defined());
    TVM_FFI_ICHECK(int_workspace_buffer.defined());
    TVM_FFI_ICHECK(page_locked_int_workspace_buffer.defined());
    // Do nothing. Subclasses can override to customize behavior.
  }
};

/*! \brief The TIR-based paged decode attention function class. */
class TIRPagedDecodeFunc : public PagedDecodeFunc {
 public:
  explicit TIRPagedDecodeFunc(ffi::Function attn_func, AttnKind attn_kind)
      : PagedDecodeFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(int depth, ffi::Tensor q, ffi::Tensor pages, ffi::Tensor page_indptr,
           ffi::Tensor page_indices, ffi::Tensor length_info, ffi::Tensor k_rope_pos_offset,
           ffi::Tensor q_rope_position, RoPEMode rope_mode, double rotary_scale,
           double rotary_theta, double sm_scale, ffi::Tensor attn_output, ffi::Tensor attn_lse,
           TVMStreamHandle compute_stream) final {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(pages.defined());
    TVM_FFI_ICHECK(page_indptr.defined());
    TVM_FFI_ICHECK(page_indices.defined());
    TVM_FFI_ICHECK(length_info.defined());
    TVM_FFI_ICHECK(k_rope_pos_offset.defined());
    TVM_FFI_ICHECK(q_rope_position.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    attn_func_(q, pages, page_indptr, page_indices, length_info, k_rope_pos_offset, q_rope_position,
               attn_output, attn_lse,
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale);
  }

  void MLA(int depth, ffi::Tensor q, ffi::Tensor pages, ffi::Tensor page_indptr,
           ffi::Tensor page_indices, ffi::Tensor length_info, double sm_scale,
           ffi::Tensor attn_output, ffi::Tensor attn_lse, TVMStreamHandle compute_stream) final {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(pages.defined());
    TVM_FFI_ICHECK(page_indptr.defined());
    TVM_FFI_ICHECK(page_indices.defined());
    TVM_FFI_ICHECK(length_info.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    attn_func_(q, pages, page_indptr, page_indices, length_info, attn_output, attn_lse, sm_scale);
  }
};

/*! \brief The FlashInfer-based paged decode attention function class. */
class FlashInferPagedDecodeFunc : public PagedDecodeFunc {
 public:
  explicit FlashInferPagedDecodeFunc(ffi::Function attn_func, ffi::Function plan_func,
                                     AttnKind attn_kind)
      : PagedDecodeFunc(std::move(attn_func), attn_kind, AttnBackendKind::kFlashInfer),
        plan_func_(std::move(plan_func)) {}

  void MHA(int depth, ffi::Tensor q, ffi::Tensor pages, ffi::Tensor page_indptr,
           ffi::Tensor page_indices, ffi::Tensor length_info, ffi::Tensor k_rope_pos_offset,
           ffi::Tensor q_rope_position, RoPEMode rope_mode, double rotary_scale,
           double rotary_theta, double sm_scale, ffi::Tensor attn_output, ffi::Tensor attn_lse,
           TVMStreamHandle compute_stream) final {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(pages.defined());
    TVM_FFI_ICHECK(page_indptr.defined());
    TVM_FFI_ICHECK(page_indices.defined());
    TVM_FFI_ICHECK(length_info.defined());
    TVM_FFI_ICHECK(k_rope_pos_offset.defined());
    TVM_FFI_ICHECK(q_rope_position.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    auto [float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
          plan_info_vec] = cached_buffers_[depth].value();
    double rope_rcp_scale = 1 / rotary_scale;
    double rope_rcp_theta = 1 / rotary_theta;
    attn_func_(float_workspace_buffer, int_workspace_buffer, plan_info_vec, q,
               PagedKVCacheView(pages, 0), PagedKVCacheView(pages, 1),
               ZeroByteOffsetView(page_indptr), ZeroByteOffsetView(page_indices),
               ZeroByteOffsetView(length_info), attn_output, attn_lse, /*kv_layout_code(HND)=*/1,
               /*window_left=*/-1, /*enable_pdl=*/false, sm_scale,
               /*rope_rcp_scale=*/rope_rcp_scale, /*rope_rcp_theta=*/rope_rcp_theta);
  }

  void BeginForward(int depth, ffi::Tensor float_workspace_buffer, ffi::Tensor int_workspace_buffer,
                    ffi::Tensor page_locked_int_workspace_buffer, HostMemoryVector* page_indptr,
                    int64_t batch_size, int64_t page_size, int64_t num_qo_heads,
                    int64_t num_kv_heads, int64_t qk_head_dim, int64_t v_head_dim,
                    RoPEMode rope_mode, DLDataType q_dtype, DLDataType kv_dtype,
                    TVMStreamHandle copy_stream) final {
    TVM_FFI_ICHECK(float_workspace_buffer.defined());
    TVM_FFI_ICHECK(int_workspace_buffer.defined());
    TVM_FFI_ICHECK(page_locked_int_workspace_buffer.defined());
    // Todo(tvm-team): enable cuda graph
    // FlashInfer's decode plan takes empty q/kv tensors (used only for dtype
    // dispatch) instead of dtype scalars, adds a logits_soft_cap argument, and
    // no longer takes the pos-encoding mode or an explicit stream.
    DLDevice device = float_workspace_buffer.device();
    ffi::Tensor empty_q_data = tvm::runtime::Empty(ffi::Shape({0}), q_dtype, device);
    ffi::Tensor empty_kv_data = tvm::runtime::Empty(ffi::Shape({0}), kv_dtype, device);
    ffi::Array<int64_t> plan_info_vec =
        plan_func_(float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
                   page_indptr->as_tensor(), batch_size, num_qo_heads, num_kv_heads, page_size,
                   /*enable_cuda_graph=*/false, /*window_left=*/-1, /*logits_soft_cap=*/0.0,
                   qk_head_dim, v_head_dim, empty_q_data, empty_kv_data)
            .cast<ffi::Array<int64_t>>();

    if (cached_buffers_.size() <= static_cast<size_t>(depth)) {
      cached_buffers_.resize(depth + 1);
    }
    cached_buffers_[depth] =
        std::make_tuple(float_workspace_buffer, int_workspace_buffer,
                        page_locked_int_workspace_buffer, std::move(plan_info_vec));
  }

 private:
  ffi::Function plan_func_;
  std::vector<std::optional<std::tuple<ffi::Tensor, ffi::Tensor, ffi::Tensor, ffi::Array<int64_t>>>>
      cached_buffers_;
};

/*! \brief The paged prefill with tree mask attention function base class. */
class PagedPrefillTreeMaskFunc : public AttnBackendFunc {
 public:
  explicit PagedPrefillTreeMaskFunc(ffi::Function attn_func, AttnKind attn_kind,
                                    AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(ffi::Tensor q, ffi::Tensor qo_indptr, ffi::Tensor pages, ffi::Tensor page_indptr,
                   ffi::Tensor page_indices, ffi::Tensor length_info, ffi::Tensor k_rope_pos_offset,
                   ffi::Tensor q_rope_position, ffi::Tensor tree_attn_mn_indptr,
                   ffi::Tensor tree_attn_mask, RoPEMode rope_mode, double rotary_scale,
                   double rotary_theta, double sm_scale, ffi::Tensor attn_output,
                   ffi::Tensor attn_lse, TVMStreamHandle compute_stream) {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(pages.defined());
    TVM_FFI_ICHECK(page_indptr.defined());
    TVM_FFI_ICHECK(page_indices.defined());
    TVM_FFI_ICHECK(length_info.defined());
    TVM_FFI_ICHECK(k_rope_pos_offset.defined());
    TVM_FFI_ICHECK(q_rope_position.defined());
    TVM_FFI_ICHECK(tree_attn_mn_indptr.defined());
    TVM_FFI_ICHECK(tree_attn_mask.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    TVM_FFI_THROW(InternalError) << "MHA computation is not supported by the current backend";
  }

  virtual void MLA(ffi::Tensor q, ffi::Tensor qo_indptr, ffi::Tensor pages, ffi::Tensor page_indptr,
                   ffi::Tensor page_indices, ffi::Tensor length_info,
                   ffi::Tensor tree_attn_mn_indptr, ffi::Tensor tree_attn_mask, double sm_scale,
                   ffi::Tensor attn_output, ffi::Tensor attn_lse, TVMStreamHandle compute_stream) {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(pages.defined());
    TVM_FFI_ICHECK(page_indptr.defined());
    TVM_FFI_ICHECK(page_indices.defined());
    TVM_FFI_ICHECK(length_info.defined());
    TVM_FFI_ICHECK(tree_attn_mn_indptr.defined());
    TVM_FFI_ICHECK(tree_attn_mask.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    TVM_FFI_THROW(InternalError) << "MLA computation is not supported by the current backend";
  }

  virtual void BeginForward(ffi::Tensor temp_float_attn_workspace,
                            ffi::Tensor temp_int_attn_workspace, HostMemoryVector* page_indptr,
                            HostMemoryVector* last_page_len, HostMemoryVector* qo_indptr,
                            int64_t batch_size, int64_t page_size, int64_t num_qo_heads,
                            int64_t num_kv_heads, int64_t qk_head_dim, int64_t v_head_dim,
                            RoPEMode rope_mode, TVMStreamHandle copy_stream) {
    TVM_FFI_ICHECK(temp_float_attn_workspace.defined());
    TVM_FFI_ICHECK(temp_int_attn_workspace.defined());
    // Do nothing. Subclasses can override to customize behavior.
  }
};

/*! \brief The TIR-based paged prefill with tree mask attention function class. */
class TIRPagedPrefillTreeMaskFunc : public PagedPrefillTreeMaskFunc {
 public:
  explicit TIRPagedPrefillTreeMaskFunc(ffi::Function attn_func, AttnKind attn_kind)
      : PagedPrefillTreeMaskFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(ffi::Tensor q, ffi::Tensor qo_indptr, ffi::Tensor pages, ffi::Tensor page_indptr,
           ffi::Tensor page_indices, ffi::Tensor length_info, ffi::Tensor k_rope_pos_offset,
           ffi::Tensor q_rope_position, ffi::Tensor tree_attn_mn_indptr, ffi::Tensor tree_attn_mask,
           RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
           ffi::Tensor attn_output, ffi::Tensor attn_lse, TVMStreamHandle compute_stream) final {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(pages.defined());
    TVM_FFI_ICHECK(page_indptr.defined());
    TVM_FFI_ICHECK(page_indices.defined());
    TVM_FFI_ICHECK(length_info.defined());
    TVM_FFI_ICHECK(k_rope_pos_offset.defined());
    TVM_FFI_ICHECK(q_rope_position.defined());
    TVM_FFI_ICHECK(tree_attn_mn_indptr.defined());
    TVM_FFI_ICHECK(tree_attn_mask.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    attn_func_(q, qo_indptr, pages, page_indptr, page_indices, length_info, k_rope_pos_offset,
               q_rope_position, attn_output, attn_lse,
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale, tree_attn_mn_indptr, tree_attn_mask);
  }
};

/*! \brief The ragged prefill with tree mask function base class. */
class RaggedPrefillTreeMaskFunc : public AttnBackendFunc {
 public:
  explicit RaggedPrefillTreeMaskFunc(ffi::Function attn_func, AttnKind attn_kind,
                                     AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(ffi::Tensor q, ffi::Tensor k, ffi::Tensor v, ffi::Tensor qo_indptr,
                   ffi::Tensor kv_indptr, ffi::Tensor q_rope_position,
                   ffi::Tensor tree_attn_mn_indptr, ffi::Tensor tree_attn_mask, RoPEMode rope_mode,
                   double rotary_scale, double rotary_theta, double sm_scale,
                   ffi::Tensor attn_output, ffi::Tensor attn_lse, TVMStreamHandle compute_stream) {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(k.defined());
    TVM_FFI_ICHECK(v.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(kv_indptr.defined());
    TVM_FFI_ICHECK(q_rope_position.defined());
    TVM_FFI_ICHECK(tree_attn_mn_indptr.defined());
    TVM_FFI_ICHECK(tree_attn_mask.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    TVM_FFI_THROW(InternalError) << "MHA computation is not supported by the current backend";
  }

  virtual void MLA(ffi::Tensor q, ffi::Tensor compressed_kv, ffi::Tensor k_pe,
                   ffi::Tensor qo_indptr, ffi::Tensor kv_indptr, ffi::Tensor tree_attn_mn_indptr,
                   ffi::Tensor tree_attn_mask, double sm_scale, ffi::Tensor attn_output,
                   ffi::Tensor attn_lse, TVMStreamHandle compute_stream) {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(compressed_kv.defined());
    TVM_FFI_ICHECK(k_pe.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(kv_indptr.defined());
    TVM_FFI_ICHECK(tree_attn_mn_indptr.defined());
    TVM_FFI_ICHECK(tree_attn_mask.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    TVM_FFI_THROW(InternalError) << "MLA computation is not supported by the current backend";
  }

  virtual void BeginForward(ffi::Tensor temp_float_attn_workspace,
                            ffi::Tensor temp_int_attn_workspace, HostMemoryVector* page_indptr,
                            HostMemoryVector* last_page_len, HostMemoryVector* qo_indptr,
                            int64_t batch_size, int64_t page_size, int64_t num_qo_heads,
                            int64_t num_kv_heads, int64_t qk_head_dim, int64_t v_head_dim,
                            RoPEMode rope_mode, TVMStreamHandle copy_stream) {
    TVM_FFI_ICHECK(temp_float_attn_workspace.defined());
    TVM_FFI_ICHECK(temp_int_attn_workspace.defined());
    // Do nothing. Subclasses can override to customize behavior.
  }
};

/*! \brief The TIR-based ragged prefill with tree mask attention function class. */
class TIRRaggedPrefillTreeMaskFunc : public RaggedPrefillTreeMaskFunc {
 public:
  explicit TIRRaggedPrefillTreeMaskFunc(ffi::Function attn_func, AttnKind attn_kind)
      : RaggedPrefillTreeMaskFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(ffi::Tensor q, ffi::Tensor k, ffi::Tensor v, ffi::Tensor qo_indptr,
           ffi::Tensor kv_indptr, ffi::Tensor q_rope_position, ffi::Tensor tree_attn_mn_indptr,
           ffi::Tensor tree_attn_mask, RoPEMode rope_mode, double rotary_scale, double rotary_theta,
           double sm_scale, ffi::Tensor attn_output, ffi::Tensor attn_lse,
           TVMStreamHandle compute_stream) final {
    TVM_FFI_ICHECK(q.defined());
    TVM_FFI_ICHECK(k.defined());
    TVM_FFI_ICHECK(v.defined());
    TVM_FFI_ICHECK(qo_indptr.defined());
    TVM_FFI_ICHECK(kv_indptr.defined());
    TVM_FFI_ICHECK(q_rope_position.defined());
    TVM_FFI_ICHECK(tree_attn_mn_indptr.defined());
    TVM_FFI_ICHECK(tree_attn_mask.defined());
    TVM_FFI_ICHECK(attn_output.defined());
    TVM_FFI_ICHECK(attn_lse.defined());
    attn_func_(q, qo_indptr, k, v, kv_indptr, q_rope_position, tree_attn_mn_indptr, tree_attn_mask,
               attn_output, attn_lse,
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale);
  }
};

/*!
 * \brief Create a PagedPrefillFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention
 * ffi::Functions. \param attn_kind The attention kind of the function. \return The created
 * PagedPrefillFunc pointer.
 */
std::unique_ptr<PagedPrefillFunc> ConvertPagedPrefillFunc(ffi::Array<ffi::Any> args,
                                                          AttnKind attn_kind);

/*!
 * \brief Create a PagedDecodeFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention
 * ffi::Functions. \param attn_kind The attention kind of the function. \return The created
 * PagedDecodeFunc pointer.
 */
std::unique_ptr<PagedDecodeFunc> ConvertPagedDecodeFunc(ffi::Array<ffi::Any> args,
                                                        AttnKind attn_kind);

/*!
 * \brief Create a RaggedPrefillFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention
 * ffi::Functions. \param attn_kind The attention kind of the function. \return The created
 * RaggedPrefillFunc pointer.
 */
std::unique_ptr<RaggedPrefillFunc> ConvertRaggedPrefillFunc(ffi::Array<ffi::Any> args,
                                                            AttnKind attn_kind);

/*!
 * \brief Create a PagedPrefillTreeMaskFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention
 * ffi::Functions. \param attn_kind The attention kind of the function. \return The created
 * PagedPrefillTreeMaskFunc pointer.
 */
std::unique_ptr<PagedPrefillTreeMaskFunc> ConvertPagedPrefillTreeMaskFunc(ffi::Array<ffi::Any> args,
                                                                          AttnKind attn_kind);

/*!
 * \brief Create a RaggedPrefillTreeMaskFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention
 * ffi::Functions. \param attn_kind The attention kind of the function. \return The created
 * RaggedPrefillTreeMaskFunc pointer.
 */
std::unique_ptr<RaggedPrefillTreeMaskFunc> ConvertRaggedPrefillTreeMaskFunc(
    ffi::Array<ffi::Any> args, AttnKind attn_kind);

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_ATTN_BACKEND_H_
