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
 * \file src/runtime/vm/paged_kv_cache.cc
 * \brief Runtime paged KV cache object for language models.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/tensor.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <locale>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "attn_backend.h"
#include "attn_utils.h"
#include "kv_state.h"
#include "nvtx.h"

namespace tvm {
namespace runtime {
namespace vm {

namespace {

constexpr const char* kPagedKVCacheCheckpointRuntime = "relax.vm.PagedAttentionKVCache";
constexpr int64_t kPagedKVCacheCheckpointFormatVersion = 1;

const char* AttnKindToString(AttnKind attn_kind) {
  switch (attn_kind) {
    case AttnKind::kMHA:
      return "mha";
    case AttnKind::kMLA:
      return "mla";
    case AttnKind::kLinearAttn:
      return "linear";
    case AttnKind::kMHASliding:
      return "mha_sliding";
  }
  TVM_FFI_ICHECK(false) << "Unknown attention kind: " << static_cast<int>(attn_kind);
  return "unknown";
}

const char* RoPEModeToString(RoPEMode rope_mode) {
  switch (rope_mode) {
    case RoPEMode::kNone:
      return "none";
    case RoPEMode::kNormal:
      return "normal";
    case RoPEMode::kInline:
      return "inline";
  }
  TVM_FFI_ICHECK(false) << "Unknown RoPE mode: " << static_cast<int>(rope_mode);
  return "unknown";
}

std::string Uint64ToHex(uint64_t value) {
  std::ostringstream os;
  os.imbue(std::locale::classic());
  os << std::hex << std::setw(16) << std::setfill('0') << value;
  return os.str();
}

uint64_t FNV1a64(const std::string& value) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char c : value) {
    hash ^= c;
    hash *= 1099511628211ULL;
  }
  return hash;
}

}  // namespace

//-------------------------------------------
// We keep the implementation private as
// they may subject to future changes.
//
// Users can interact with it through the
// runtime API function calls
//-------------------------------------------

/*!
 * \brief The paged KV cache for attention.
 * - It supports managing the K/V data of **multiple sequences**.
 * - It manages K/V values by doing paging along the sequence-length
 * dimension with a configured page size.
 * - To add a sequence to the cache, use AddSequence with a provided
 * unique integer sequence id.
 * - The basic example use of the paged KV cache after initialization
 * in each round of model forwarding is the following:
 *   - step 1. use `BeginForward` to specify the list of sequence ids
 *     together with the lengths of append,
 *   - step 2. use `Attention` to pass in the q/k/v values regarding
 *     the sequences and lengths specified in `BeginForward`. The
 *     attention is computed between input queries and the history
 *     key/values plus the input key/values. The input key/values
 *     will be added into the KV cache as well.
 *   - step 3. use `EndForward` to mark the end of forwarding this round.
 *     After calling `EndForward`, it is required to call `BeginForward`
 *     before calling any `Attention`.
 */
class PagedAttentionKVCacheObj : public AttentionKVCacheObj {
 private:
  /********************* Configuration *********************/

  /*! \brief The page size (the sequence length each page manages) of the cache. */
  const int64_t page_size_;
  /*! \brief The number of layers in the model. */
  const int64_t num_layers_;
  /*! \brief The beginning layer id offset. */
  const int64_t layer_id_begin_offset_;
  /*! \brief The ending layer id offset. */
  const int64_t layer_id_end_offset_;
  /*! \brief The number of query/output heads in the model. */
  const int64_t num_qo_heads_;
  /*! \brief The number of key/value heads in the model. */
  const int64_t num_kv_heads_;
  /*! \brief The number of features each head has. */
  const int64_t qk_head_dim_;
  /*!
   * \brief The number of features each head has for V.
   * For layers that use multi-head attention, this field is overriden by qk_head_dim.
   */
  const int64_t v_head_dim_;
  /*! \brief The number of sequences reserved in the KV cache. */
  const int64_t reserved_num_seqs_;
  /*! \brief The number of total pages allocated in KV cache. */
  const int64_t num_total_pages_;
  /*! \brief The maximum total sequence length in a prefill. */
  const int64_t prefill_chunk_size_;
  /*! \brief A boolean flag indicating if the KV cache supports sliding window. */
  const bool support_sliding_window_;
  /*! \brief A boolean flag indicating if the KV cache has per layer sliding window. */
  const bool support_layer_sliding_window_;
  /*! \brief Window size used by layers with per-layer sliding attention. */
  const int64_t layer_sliding_window_size_;
  /*! \brief The attention kinds for each layer. */
  const std::vector<AttnKind> attn_kinds_;

  /*! \brief The RoPE application mode of KV cache.*/
  const RoPEMode rope_mode_;
  /*! \brief The RoPE scale. */
  const double rotary_scale_;
  /*! \brief The RoPE theta. */
  const double rotary_theta_;
  /*! \brief The optional RoPE extension factors for RoPE scaling. */
  const ffi::Optional<ffi::Tensor> rope_ext_factors_;

  /*! \brief The KV cache dtype. */
  const DLDataType kv_dtype_;
  /*! \brief We fix int32 to be the index dtype of auxiliary data. */
  const DLDataType dtype_aux_ = DLDataType{kDLInt, 32, 1};

  /********************* Page Structures *********************/

  /*!
   * \brief The KV data managed by the KV cache.
   * If KV transfer function is specifed, pages_ will be allocated by NVSHMEM as a whole Tensor.
   * pages_ will contain tensor view of each layer.
   * Otherwise, pages_ has `num_layers` Tensors, each of them
   * has layout (num_pages, 2, num_heads, page_size, qk_head_dim).
   * Along on the "2" dimension, index 0 stands for K and 1 stands for V.
   */
  std::vector<ffi::Tensor> pages_;
  /*! \brief A reusable host-side zero page for deterministic checkpoint padding. */
  ffi::Optional<ffi::Tensor> checkpoint_zero_page_host_;
  /*! \brief The whole KV cache allocated by NVSHMEM*/
  ffi::Optional<ffi::Tensor> nvshmem_pages_;
  /*! \brief The list of ids of released pages for page reuse. */
  std::vector<int32_t> free_page_ids_;
  /*! \brief The mapping from sequence ids to sequences. */
  std::unordered_map<int64_t, Sequence> seq_map_;
  /*! \brief Whether a checkpoint import is waiting for all page groups. */
  bool checkpoint_import_in_progress_ = false;
  /*! \brief The page groups restored by the current checkpoint import. */
  std::vector<bool> checkpoint_imported_groups_;

  /********************* Sequence Block Structures *********************/

  /*! \brief The list of all blocks once allocated. */
  std::vector<Block> global_block_pool_;
  /*! \brief The list of free available blocks (in their indices). */
  std::vector<int32_t> free_block_idx_;

  /*********** Current Batch Info & Auxiliary Arrays on Device ***********/
  //-------------------------------------------
  // The following fields are auxiliary arrays on device.
  // All of them are directly derivable from the fields above.
  // We store them for efficient execution of attentions,
  // cache append, etc.
  //-------------------------------------------
  /*!
   * \brief A boolean flag indicating if the auxiliary arrays are dirty.
   * If it is dirty, an explicit "ComputeStreamWaitForCopyStream" should be invoked.
   */
  bool dirty_aux_data_device_ = false;
  /*! \brief The batch size of the current round of forwarding. */
  int64_t cur_batch_size_;
  /*! \brief The ids of the sequences in the current round of forwarding. */
  ffi::Shape cur_seq_ids_;
  /*! \brief The append lengths of the sequences in the current round of forwarding. */
  ffi::Shape cur_append_lengths_;
  /*! \brief Whether the current batch of sequences are token chains (not token trees). */
  std::vector<bool> is_chain_on_depths_;
  /*! \brief Number of fork depth in the current round of forward. */
  int num_depths_;
  /*! \brief Whether to compute attention after appending KV into cache or not. */
  bool append_before_attn_;
  /*! \brief Whether to use decode kernel for each depth. (see GetChunkedBlockIds) */
  std::vector<bool> use_decode_kernel_;
  /*! \brief Whether the attention request is a decode request, set in BeginForwardFunction. */
  bool is_decode_request_;
  /*! \brief The KV transfer recver disco group's PE offset in this forward.
             If no KV is transfered, recver is -1.
             Assume that all the KV are transfered to the same recver in the forward.
             todo: support multiple recver. */
  bool transfer_kv_;
  bool page_to_page_transfer_kv_;
  /*! \brief The auxiliary data manager for attention. */
  std::unique_ptr<PagedKVCacheAuxDataManager> aux_data_manager_;

  // Temporary arrays to store intermediate attention results.
  ffi::Optional<ffi::Tensor> temp_attn_q_device_;
  ffi::Optional<ffi::Tensor> temp_attn_k_device_;
  ffi::Optional<ffi::Tensor> temp_attn_v_device_;
  ffi::Tensor temp_attn_output_device_;
  ffi::Tensor temp_attn_lse_device_;
  ffi::Tensor merged_attn_lse_device_;
  std::vector<ffi::Tensor> temp_int_attn_workspace_;
  std::vector<ffi::Tensor> temp_int_pinned_attn_workspace_;
  ffi::Optional<ffi::Tensor> temp_float_attn_workspace_;

  std::vector<ffi::Any> retrieve_ret_;

  //-------------------------------------------
  // Below are the auxiliary data structure on CPU.
  // We make them class members to avoid repetitive allocation time in BeginForward.
  //-------------------------------------------
  std::vector<HostMemoryVector> qo_indptr_on_depths_host_;
  std::vector<HostMemoryVector> page_indptr_on_depths_host_;
  std::vector<HostMemoryVector> page_indices_on_depths_host_;
  std::vector<HostMemoryVector> page_indptr_sliding_window_on_depths_host_;
  std::vector<HostMemoryVector> page_indices_sliding_window_on_depths_host_;
  std::vector<HostMemoryVector> last_page_len_on_depths_host_;
  std::vector<HostMemoryVector> sliding_window_offset_on_depths_host_;
  std::vector<HostMemoryVector> sink_size_on_depths_host_;
  std::vector<HostMemoryVector> k_rope_pos_offset_on_depths_host_;
  std::vector<HostMemoryVector> k_rope_pos_offset_sliding_window_on_depths_host_;
  HostMemoryVector kv_len_arr_host_;
  HostMemoryVector k_ragged_rope_pos_offset_host_;
  HostMemoryVector q_rope_position_map_host_;
  HostMemoryVector append_position_map_host_;
  HostMemoryVector cur_append_lengths_indptr_host_;
  std::vector<HostMemoryVector> tree_attn_mask_host_;
  std::vector<HostMemoryVector> tree_attn_mn_indptr_host_;
  HostMemoryVector commit_copy_length_indptr_host_;
  HostMemoryVector commit_copy_src_pos_in_page_table_host_;
  HostMemoryVector commit_copy_dst_pos_in_page_table_host_;
  HostMemoryVector kv_transfer_remote_position_map_host_;
  HostMemoryVector kv_transfer_recver_id_host_;
  HostMemoryVector kv_transfer_page_to_page_local_position_map_host_;
  HostMemoryVector kv_transfer_page_to_page_remote_position_map_host_;
  HostMemoryVector kv_transfer_page_to_page_recver_id_host_;
  //-------------------------------------------
  // For efficient memory management, the actual sizes of the arrays
  // above are over allocated.
  // We create a view for the actual shapes of each of the arrays
  // after each synchronization and pass these views as input for
  // attention/append.
  //-------------------------------------------
  ffi::Optional<ffi::Tensor> cur_append_length_indptr_view_;
  ffi::Optional<ffi::Tensor> k_ragged_rope_pos_offset_view_;
  ffi::Optional<ffi::Tensor> q_rope_position_map_view_;
  ffi::Optional<ffi::Tensor> append_position_map_view_;
  ffi::Optional<ffi::Tensor> kv_transfer_remote_position_map_view_;
  ffi::Optional<ffi::Tensor> kv_transfer_recver_id_view_;
  ffi::Optional<ffi::Tensor> kv_transfer_page_to_page_local_position_map_view_;
  ffi::Optional<ffi::Tensor> kv_transfer_page_to_page_remote_position_map_view_;
  ffi::Optional<ffi::Tensor> kv_transfer_page_to_page_recver_id_view_;
  ffi::Optional<ffi::Tensor> temp_attn_output_view_;
  ffi::Optional<ffi::Tensor> temp_attn_lse_view_;
  ffi::Optional<ffi::Tensor> merged_attn_lse_view_;
  std::vector<ffi::Optional<ffi::Tensor>> qo_indptr_on_depths_view_;
  std::vector<ffi::Optional<ffi::Tensor>> page_indptr_on_depths_view_;
  std::vector<ffi::Optional<ffi::Tensor>> page_indices_on_depths_view_;
  std::vector<ffi::Optional<ffi::Tensor>> page_indptr_sliding_window_on_depths_view_;
  std::vector<ffi::Optional<ffi::Tensor>> page_indices_sliding_window_on_depths_view_;
  std::vector<ffi::Optional<ffi::Tensor>> length_info_on_depths_view_;
  std::vector<ffi::Optional<ffi::Tensor>> layer_sliding_window_length_info_on_depths_view_;
  std::vector<ffi::Optional<ffi::Tensor>> k_rope_pos_offset_view_;
  std::vector<ffi::Optional<ffi::Tensor>> k_rope_pos_offset_sliding_window_view_;
  std::vector<ffi::Optional<ffi::Tensor>> tree_attn_mask_view_;
  std::vector<ffi::Optional<ffi::Tensor>> tree_attn_mn_indptr_view_;

  ffi::Optional<ffi::Function> f_transpose_append_mha_;
  ffi::Optional<ffi::Function> f_transpose_append_mla_;
  ffi::Optional<ffi::Function> f_transfer_kv_;
  ffi::Optional<ffi::Function> f_transfer_kv_page_to_page_ = std::nullopt;
  ffi::Function f_compact_copy_;
  std::unique_ptr<RaggedPrefillFunc> f_attention_prefill_ragged_;
  std::unique_ptr<PagedPrefillFunc> f_attention_prefill_;
  std::unique_ptr<PagedDecodeFunc> f_attention_decode_;
  std::unique_ptr<PagedPrefillFunc> f_attention_prefill_sliding_window_;
  std::unique_ptr<PagedDecodeFunc> f_attention_decode_sliding_window_;
  std::unique_ptr<PagedPrefillTreeMaskFunc> f_attention_prefill_with_tree_mask_paged_kv_;
  std::unique_ptr<RaggedPrefillTreeMaskFunc> f_attention_prefill_with_tree_mask_;
  std::unique_ptr<PagedPrefillFunc> f_mla_prefill_;
  ffi::Array<ffi::Function> f_merge_inplace_;
  ffi::Function f_split_rotary_;
  ffi::Function f_copy_single_page_;
  ffi::Optional<ffi::Function> f_debug_get_kv_;

  /*! \brief The device this PagedKVCache runs on. */
  Device device_;
  /*! \brief The device stream for the default computation operations. */
  TVMStreamHandle compute_stream_ = nullptr;
  /*! \brief The device stream for copying auxiliary data structure to GPU. */
  TVMStreamHandle copy_stream_ = nullptr;
  /*! \brief The device stream for KV transfer */
  TVMStreamHandle kv_transfer_stream_ = nullptr;

  int32_t GetLayerSlidingWindowOffset(int64_t seq_length) const {
    if (seq_length <= layer_sliding_window_size_) {
      return 0;
    }
    return static_cast<int32_t>((seq_length - layer_sliding_window_size_) % page_size_);
  }

  int32_t GetLayerSlidingWindowNumPages(int64_t seq_length) const {
    if (seq_length == 0) {
      return 0;
    }
    int64_t window_length = std::min(seq_length, layer_sliding_window_size_);
    return static_cast<int32_t>(
        (GetLayerSlidingWindowOffset(seq_length) + window_length + page_size_ - 1) / page_size_);
  }

 public:
  /*! \brief Constructor. Take the cache configuration and initialize the Tensors. */
  explicit PagedAttentionKVCacheObj(
      int64_t page_size, int64_t num_layers, int64_t layer_id_begin_offset,
      int64_t layer_id_end_offset, int64_t num_qo_heads, int64_t num_kv_heads, int64_t qk_head_dim,
      int64_t v_head_dim, std::vector<AttnKind> attn_kinds, int64_t reserved_num_seqs,
      int64_t num_total_pages, int64_t prefill_chunk_size, bool support_sliding_window,
      int64_t layer_sliding_window_size, RoPEMode rope_mode, double rotary_scale,
      double rotary_theta, ffi::Optional<ffi::Tensor> rope_ext_factors, bool enable_kv_transfer,
      DLDataType dtype, Device device, ffi::Optional<ffi::Function> f_transpose_append_mha,
      ffi::Optional<ffi::Function> f_transpose_append_mla, ffi::Function f_compact_copy,
      std::unique_ptr<RaggedPrefillFunc> f_attention_prefill_ragged,
      std::unique_ptr<PagedPrefillFunc> f_attention_prefill,
      std::unique_ptr<PagedDecodeFunc> f_attention_decode,
      std::unique_ptr<PagedPrefillFunc> f_attention_prefill_sliding_window,
      std::unique_ptr<PagedDecodeFunc> f_attention_decode_sliding_window,
      std::unique_ptr<PagedPrefillTreeMaskFunc> f_attention_prefill_with_tree_mask_paged_kv,
      std::unique_ptr<RaggedPrefillTreeMaskFunc> f_attention_prefill_with_tree_mask,
      std::unique_ptr<PagedPrefillFunc> f_mla_prefill, ffi::Array<ffi::Function> f_merge_inplace,
      ffi::Function f_split_rotary, ffi::Function f_copy_single_page, ffi::Function f_debug_get_kv)
      : page_size_(page_size),
        num_layers_(num_layers),
        layer_id_begin_offset_(layer_id_begin_offset),
        layer_id_end_offset_(layer_id_end_offset),
        num_qo_heads_(num_qo_heads),
        num_kv_heads_(num_kv_heads),
        qk_head_dim_(qk_head_dim),
        v_head_dim_(v_head_dim),
        reserved_num_seqs_(reserved_num_seqs),
        num_total_pages_(num_total_pages),
        prefill_chunk_size_(prefill_chunk_size),
        support_sliding_window_(std::find(attn_kinds.begin(), attn_kinds.end(),
                                          AttnKind::kMHASliding) != attn_kinds.end()
                                    ? false
                                    : support_sliding_window),
        support_layer_sliding_window_(std::find(attn_kinds.begin(), attn_kinds.end(),
                                                AttnKind::kMHASliding) != attn_kinds.end()),
        layer_sliding_window_size_(layer_sliding_window_size),
        attn_kinds_(std::move(attn_kinds)),
        rope_mode_(support_sliding_window && rope_mode != RoPEMode::kNone ? RoPEMode::kInline
                                                                          : rope_mode),
        rotary_scale_(rotary_scale),
        rotary_theta_(rotary_theta),
        rope_ext_factors_(std::move(rope_ext_factors)),
        kv_dtype_(dtype),
        temp_attn_output_device_(
            tvm::runtime::Empty({prefill_chunk_size_, num_qo_heads, v_head_dim}, dtype, device)),
        temp_attn_lse_device_(tvm::runtime::Empty({prefill_chunk_size_, num_qo_heads},
                                                  DLDataType{kDLFloat, 32, 1}, device)),
        merged_attn_lse_device_(tvm::runtime::Empty({prefill_chunk_size_, num_qo_heads},
                                                    DLDataType{kDLFloat, 32, 1}, device)),
        f_transpose_append_mha_(std::move(f_transpose_append_mha)),
        f_transpose_append_mla_(std::move(f_transpose_append_mla)),
        f_compact_copy_(std::move(f_compact_copy)),
        f_attention_prefill_ragged_(std::move(f_attention_prefill_ragged)),
        f_attention_prefill_(std::move(f_attention_prefill)),
        f_attention_decode_(std::move(f_attention_decode)),
        f_attention_prefill_sliding_window_(std::move(f_attention_prefill_sliding_window)),
        f_attention_decode_sliding_window_(std::move(f_attention_decode_sliding_window)),
        f_attention_prefill_with_tree_mask_paged_kv_(
            std::move(f_attention_prefill_with_tree_mask_paged_kv)),
        f_attention_prefill_with_tree_mask_(std::move(f_attention_prefill_with_tree_mask)),
        f_mla_prefill_(std::move(f_mla_prefill)),
        f_merge_inplace_(std::move(f_merge_inplace)),
        f_split_rotary_(std::move(f_split_rotary)),
        f_copy_single_page_(std::move(f_copy_single_page)),
        f_debug_get_kv_(std::move(f_debug_get_kv)),
        device_(device) {
    TVM_FFI_ICHECK_GT(layer_sliding_window_size_, 0)
        << "Per-layer sliding window size must be positive.";
    // Note: For MLA, sliding window and disaggregation are disabled for now.
    if (std::find(attn_kinds_.begin(), attn_kinds_.end(), AttnKind::kMLA) != attn_kinds_.end()) {
      TVM_FFI_ICHECK(!support_sliding_window_) << "Sliding window not supported yet for MLA";
      TVM_FFI_ICHECK(!enable_kv_transfer) << "KV transfer not supported yet for MLA";
    }

    pages_.reserve(num_layers);
    if (enable_kv_transfer) {
      // For now, KV transfer only supports MHA.
      for (AttnKind attn_kind : attn_kinds_) {
        TVM_FFI_ICHECK(attn_kind == AttnKind::kMHA);
      }
      const auto f_nvshmem_init =
          tvm::ffi::Function::GetGlobal("runtime.disco.nvshmem.init_nvshmem");
      TVM_FFI_ICHECK(f_nvshmem_init.has_value())
          << "NVSHMEM is not enabled. Please make sure NVSHMEM is enabled when compiling TVM.";
      const auto f_nvshmem_empty = tvm::ffi::Function::GetGlobal("runtime.disco.nvshmem.empty");
      TVM_FFI_ICHECK(f_nvshmem_empty.has_value());
      nvshmem_pages_ =
          (*f_nvshmem_empty)(
              ffi::Shape({num_layers, num_total_pages, 2, num_kv_heads, page_size, qk_head_dim}),
              dtype, device)
              .cast<ffi::Tensor>();
      for (int i = 0; i < num_layers; ++i) {
        pages_.push_back(tvm::runtime::CreateView(
            nvshmem_pages_.value(), {num_total_pages_, 2, num_kv_heads_, page_size_, qk_head_dim_},
            nvshmem_pages_.value().dtype(),
            i * num_total_pages_ * 2 * num_kv_heads_ * page_size_ * qk_head_dim_ *
                (nvshmem_pages_.value().dtype().bits + 7) / 8));
      }

      const auto f_transfer_kv_ptr = tvm::ffi::Function::GetGlobal("nvshmem.KVTransfer");
      const auto f_transfer_kv_page_to_page_ptr =
          tvm::ffi::Function::GetGlobal("nvshmem.KVTransferPageToPage");
      TVM_FFI_ICHECK(f_transfer_kv_ptr.has_value());
      TVM_FFI_ICHECK(f_transfer_kv_page_to_page_ptr.has_value());
      f_transfer_kv_ = *f_transfer_kv_ptr;
      f_transfer_kv_page_to_page_ = *f_transfer_kv_page_to_page_ptr;
    } else {
      for (int i = 0; i < num_layers; ++i) {
        ffi::Shape kv_cache_shape =
            GetKVCacheShape(attn_kinds_[layer_id_begin_offset_ + i], num_total_pages,
                            reserved_num_seqs, num_kv_heads, page_size, qk_head_dim, v_head_dim);
        pages_.push_back(tvm::runtime::Empty(kv_cache_shape, dtype, device));
      }
    }

    // Allocate the host memory.
    Device preferred_host_device = GetPreferredHostDevice(device);
    for (int d = 0; d < kPagedKVCacheMaxBlockDepth; ++d) {
      qo_indptr_on_depths_host_.push_back(
          HostMemoryVector(reserved_num_seqs + 1, dtype_aux_, preferred_host_device));
      page_indptr_on_depths_host_.push_back(
          HostMemoryVector(reserved_num_seqs + 1, dtype_aux_, preferred_host_device));
      page_indices_on_depths_host_.push_back(
          HostMemoryVector(num_total_pages, dtype_aux_, preferred_host_device));
      page_indptr_sliding_window_on_depths_host_.push_back(
          HostMemoryVector(reserved_num_seqs + 1, dtype_aux_, preferred_host_device));
      page_indices_sliding_window_on_depths_host_.push_back(
          HostMemoryVector(num_total_pages, dtype_aux_, preferred_host_device));
      last_page_len_on_depths_host_.push_back(
          HostMemoryVector(reserved_num_seqs, dtype_aux_, preferred_host_device));
      sliding_window_offset_on_depths_host_.push_back(
          HostMemoryVector(reserved_num_seqs, dtype_aux_, preferred_host_device));
      sink_size_on_depths_host_.push_back(
          HostMemoryVector(reserved_num_seqs, dtype_aux_, preferred_host_device));
      k_rope_pos_offset_on_depths_host_.push_back(
          HostMemoryVector(reserved_num_seqs, dtype_aux_, preferred_host_device));
      k_rope_pos_offset_sliding_window_on_depths_host_.push_back(
          HostMemoryVector(reserved_num_seqs, dtype_aux_, preferred_host_device));
      tree_attn_mask_host_.push_back(HostMemoryVector(kTreeAttnMaxTreeSize * 2 * reserved_num_seqs,
                                                      dtype_aux_, preferred_host_device));
      tree_attn_mn_indptr_host_.push_back(
          HostMemoryVector(reserved_num_seqs + 1, dtype_aux_, preferred_host_device));
    }
    kv_len_arr_host_ = HostMemoryVector(reserved_num_seqs, dtype_aux_, preferred_host_device);
    k_ragged_rope_pos_offset_host_ =
        HostMemoryVector(reserved_num_seqs, dtype_aux_, preferred_host_device);
    q_rope_position_map_host_ =
        HostMemoryVector(prefill_chunk_size, dtype_aux_, preferred_host_device);
    append_position_map_host_ =
        HostMemoryVector(prefill_chunk_size, dtype_aux_, preferred_host_device);
    kv_transfer_remote_position_map_host_ =
        HostMemoryVector(prefill_chunk_size, dtype_aux_, preferred_host_device);
    kv_transfer_recver_id_host_ =
        HostMemoryVector(prefill_chunk_size, dtype_aux_, preferred_host_device);
    kv_transfer_page_to_page_local_position_map_host_ =
        HostMemoryVector(prefill_chunk_size, dtype_aux_, preferred_host_device);
    kv_transfer_page_to_page_remote_position_map_host_ =
        HostMemoryVector(prefill_chunk_size, dtype_aux_, preferred_host_device);
    kv_transfer_page_to_page_recver_id_host_ =
        HostMemoryVector(prefill_chunk_size, dtype_aux_, preferred_host_device);
    cur_append_lengths_indptr_host_ =
        HostMemoryVector(reserved_num_seqs + 1, dtype_aux_, preferred_host_device);
    commit_copy_length_indptr_host_ =
        HostMemoryVector(reserved_num_seqs + 1, dtype_aux_, preferred_host_device);
    commit_copy_src_pos_in_page_table_host_ =
        HostMemoryVector(std::min(kTreeAttnMaxTreeSize * reserved_num_seqs, prefill_chunk_size),
                         dtype_aux_, preferred_host_device);
    commit_copy_dst_pos_in_page_table_host_ =
        HostMemoryVector(std::min(kTreeAttnMaxTreeSize * reserved_num_seqs, prefill_chunk_size),
                         dtype_aux_, preferred_host_device);

    for (int d = 0; d < kPagedKVCacheMaxBlockDepth; ++d) {
      if (NeedKernelBeginForward()) {
        temp_int_attn_workspace_.push_back(
            tvm::runtime::Empty({kIntAttnWorkspaceByte}, DLDataType{kDLUInt, 8, 1}, device));
        temp_int_pinned_attn_workspace_.push_back(tvm::runtime::Empty(
            {kIntAttnWorkspaceByte}, DLDataType{kDLUInt, 8, 1}, GetPreferredHostDevice(device)));
      }
      qo_indptr_on_depths_view_.push_back(std::nullopt);
      page_indptr_on_depths_view_.push_back(std::nullopt);
      page_indices_on_depths_view_.push_back(std::nullopt);
      page_indptr_sliding_window_on_depths_view_.push_back(std::nullopt);
      page_indices_sliding_window_on_depths_view_.push_back(std::nullopt);
      length_info_on_depths_view_.push_back(std::nullopt);
      layer_sliding_window_length_info_on_depths_view_.push_back(std::nullopt);
      k_rope_pos_offset_view_.push_back(std::nullopt);
      k_rope_pos_offset_sliding_window_view_.push_back(std::nullopt);
      tree_attn_mask_view_.push_back(std::nullopt);
      tree_attn_mn_indptr_view_.push_back(std::nullopt);
      is_chain_on_depths_.push_back(true);
    }
    // Additional workspace for the "prefill with ragged kv" kernel.
    if (NeedKernelBeginForward()) {
      temp_int_attn_workspace_.push_back(
          tvm::runtime::Empty({kIntAttnWorkspaceByte}, DLDataType{kDLUInt, 8, 1}, device));
      temp_int_pinned_attn_workspace_.push_back(tvm::runtime::Empty(
          {kIntAttnWorkspaceByte}, DLDataType{kDLUInt, 8, 1}, GetPreferredHostDevice(device)));
      temp_float_attn_workspace_ =
          tvm::runtime::Empty({kFloatAttnWorkspaceByte}, DLDataType{kDLUInt, 8, 1}, device);
    }

    if (std::find(attn_kinds_.begin(), attn_kinds_.end(), AttnKind::kMHA) != attn_kinds_.end()) {
      temp_attn_q_device_ =
          tvm::runtime::Empty({prefill_chunk_size_, num_qo_heads, qk_head_dim}, dtype, device);
      temp_attn_k_device_ =
          tvm::runtime::Empty({prefill_chunk_size_, num_kv_heads, qk_head_dim}, dtype, device);
      temp_attn_v_device_ =
          tvm::runtime::Empty({prefill_chunk_size_, num_kv_heads, v_head_dim}, dtype, device);
    }
    for (int64_t page_id = num_total_pages - 1; page_id >= 0; --page_id) {
      free_page_ids_.push_back(page_id);
    }

    // If the device is CUDA/ROCm, we create a standalone copy stream, in
    // purpose to hide the latency of auxiliary stream copy.
    if (device.device_type == DLDeviceType::kDLCUDA ||
        device.device_type == DLDeviceType::kDLROCM) {
      // The compute stream is the default stream.
      compute_stream_ = DeviceAPI::Get(device)->GetCurrentStream(device);
      copy_stream_ = DeviceAPI::Get(device)->CreateStream(device);
      kv_transfer_stream_ = DeviceAPI::Get(device)->CreateStream(device);
    }

    // Create the auxiliary data manager for attention.
    // We only use the merged aux data for CUDA, since direct pointer
    // operations may have issues on other platforms.
    if (device_.device_type == DLDeviceType::kDLCUDA ||
        device_.device_type == DLDeviceType::kDLCPU) {
      aux_data_manager_ = std::make_unique<CachedPagedKVCacheAuxDataManager>(
          reserved_num_seqs, num_total_pages, prefill_chunk_size, dtype_aux_, device,
          preferred_host_device, copy_stream_);
    } else {
      aux_data_manager_ = std::make_unique<PlainPagedKVCacheAuxDataManager>(
          reserved_num_seqs, num_total_pages, prefill_chunk_size, dtype_aux_, device,
          preferred_host_device, copy_stream_, support_layer_sliding_window_);
    }

    // Right now only the "normal" RoPE mode supports the RoPE extention factors.
    if (rope_ext_factors_.has_value()) {
      TVM_FFI_ICHECK(rope_mode_ == RoPEMode::kNormal)
          << "The RoPE mode must be normal to support RoPE extension factors.";
    }
  }

  ~PagedAttentionKVCacheObj() {
    // Free the copy stream if defined.
    if (copy_stream_ != nullptr) {
      DeviceAPI::Get(device_)->FreeStream(device_, copy_stream_);
    }
    if (kv_transfer_stream_ != nullptr) {
      DeviceAPI::Get(device_)->FreeStream(device_, kv_transfer_stream_);
    }
  }

  /*! \brief Reset the KV cache. */
  void Clear() final {
    seq_map_.clear();
    checkpoint_import_in_progress_ = false;
    checkpoint_imported_groups_.clear();
    free_page_ids_.clear();
    for (int64_t page_id = num_total_pages_ - 1; page_id >= 0; --page_id) {
      free_page_ids_.push_back(page_id);
    }
    global_block_pool_.clear();
    free_block_idx_.clear();
    dirty_aux_data_device_ = false;
  }

  /************** Sequence Management **************/

  void AddSequence(int64_t seq_id) final {
    TVM_FFI_ICHECK(seq_map_.find(seq_id) == seq_map_.end())
        << "The sequence \"" << seq_id << "\" is already in the KV cache.";
    int32_t block_idx = GetFreeBlock();
    seq_map_.insert({seq_id, Sequence(&global_block_pool_, block_idx)});
    dirty_aux_data_device_ = true;
  }

  void RemoveSequence(int64_t seq_id) final {
    auto it = seq_map_.find(seq_id);
    TVM_FFI_ICHECK(it != seq_map_.end())
        << "The sequence \"" << seq_id << "\" cannot be found in KV cache.";
    int32_t block_idx = it->second.last_block_idx;
    // The block should have at least one reference, which comes from the sequence.
    TVM_FFI_ICHECK_GE(global_block_pool_[block_idx].external_ref_cnt, 1);
    while (block_idx != -1 && global_block_pool_[block_idx].external_ref_cnt == 1) {
      // - Free pages in the last block.
      for (int32_t page_id : global_block_pool_[block_idx].page_ids) {
        free_page_ids_.push_back(page_id);
      }
      free_block_idx_.push_back(block_idx);
      block_idx = global_block_pool_[block_idx].parent_idx;
    }
    // - Decrease the external reference of the parent block.
    if (block_idx != -1) {
      TVM_FFI_ICHECK_GT(global_block_pool_[block_idx].external_ref_cnt, 1);
      --global_block_pool_[block_idx].external_ref_cnt;
    }
    seq_map_.erase(it);
    dirty_aux_data_device_ = true;
  }

  void ForkSequence(int64_t parent_seq_id, int64_t child_seq_id, int64_t fork_pos = -1) final {
    auto parent_it = seq_map_.find(parent_seq_id);
    TVM_FFI_ICHECK(parent_it != seq_map_.end())
        << "The parent sequence \"" << parent_seq_id << "\" cannot be found in KV cache.";
    TVM_FFI_ICHECK(seq_map_.find(child_seq_id) == seq_map_.end())
        << "The child sequence \"" << child_seq_id << "\" is already in the KV cache.";
    TVM_FFI_ICHECK_GE(fork_pos, -1)
        << "The forked position should be non-negative, or -1 for last position as default.";
    TVM_FFI_ICHECK_LE(fork_pos, parent_it->second.seq_length)
        << "The forked position should not exceed the total length of parent sequence.";
    TVM_FFI_ICHECK(parent_it->second.accepted_indices_committed)
        << "The parent sequence's token tree computed in the last round of forward has not been "
           "committed with accepted nodes.";

    if (fork_pos == -1) {
      fork_pos = parent_it->second.seq_length;
    }

    if (parent_it->second.sliding_window_size != -1) {
      // If forked sequence has been enabled sliding window, check the forked position is within
      // sliding window sink size.
      const Sequence& seq = parent_it->second;
      int32_t sink_size = seq.seq_length - global_block_pool_[seq.last_block_idx].seq_length +
                          seq.last_block_attn_sink_size;
      TVM_FFI_ICHECK_LE(fork_pos, sink_size)
          << "The parent sequence \"" << parent_seq_id
          << "\" is enabled with sliding window and thus only can be forked within sink size = "
          << sink_size << ". But the forked position = " << fork_pos << ".";
    }

    if (fork_pos == parent_it->second.seq_length && fork_pos % page_size_ == 0 &&
        global_block_pool_[parent_it->second.last_block_idx].seq_length > 0) {
      // To enable the parent sequence to continue decode after the fork,
      // we add a new empty block at the end of the parent sequence.
      // So the new decoded KV data will go into the new block.
      int32_t new_block_idx = GetFreeBlock();
      global_block_pool_[new_block_idx].start_pos = parent_it->second.seq_length;
      global_block_pool_[new_block_idx].parent_idx = parent_it->second.last_block_idx;
      global_block_pool_[new_block_idx].external_ref_cnt = 1;
      parent_it->second.last_block_idx = new_block_idx;
    }

    int32_t child_block_idx = GetFreeBlock();
    std::vector<int32_t> trace = parent_it->second.GetBlockTrace(global_block_pool_);
    int64_t in_block_offset = fork_pos;
    for (int32_t forked_block_idx : trace) {
      if (forked_block_idx != trace.back()) {
        TVM_FFI_ICHECK_GT(global_block_pool_[forked_block_idx].seq_length, 0);
        TVM_FFI_ICHECK_EQ(global_block_pool_[forked_block_idx].seq_length % page_size_, 0);
        if (global_block_pool_[forked_block_idx].seq_length <= in_block_offset) {
          in_block_offset -= global_block_pool_[forked_block_idx].seq_length;
          continue;
        }
      }
      int32_t in_page_offset = in_block_offset % page_size_;
      int32_t moved_offset = in_block_offset - in_page_offset;
      int32_t moved_pages = moved_offset / page_size_;
      if (moved_pages == 0) {
        // Forked at the first page in block
        int32_t parent_block_idx = global_block_pool_[forked_block_idx].parent_idx;
        if (parent_block_idx != -1) {
          ++global_block_pool_[parent_block_idx].external_ref_cnt;
        }
        // Update child block start position and parent index
        global_block_pool_[child_block_idx].parent_idx = parent_block_idx;
      } else {
        // Forked at the second or latter page in block
        int32_t parent_block_idx = GetFreeBlock();
        // Insert new parent block before forked block and link child block
        global_block_pool_[parent_block_idx].parent_idx =
            global_block_pool_[forked_block_idx].parent_idx;
        global_block_pool_[forked_block_idx].parent_idx = parent_block_idx;
        global_block_pool_[child_block_idx].parent_idx = parent_block_idx;
        global_block_pool_[parent_block_idx].external_ref_cnt = 2;

        // Move common leading pages to new parent block
        auto first_page = global_block_pool_[forked_block_idx].page_ids.begin();
        auto last_page = global_block_pool_[forked_block_idx].page_ids.begin() + moved_pages;
        global_block_pool_[parent_block_idx].page_ids = {first_page, last_page};
        global_block_pool_[forked_block_idx].page_ids.erase(first_page, last_page);

        // Update start position per blocks
        global_block_pool_[parent_block_idx].start_pos =
            global_block_pool_[forked_block_idx].start_pos;
        global_block_pool_[forked_block_idx].start_pos += moved_offset;

        // Update in-block sequence length per blocks
        global_block_pool_[parent_block_idx].seq_length = moved_offset;
        global_block_pool_[forked_block_idx].seq_length -= moved_offset;

        // Update sliding window sink size if sliding window is enabled and the forked block is the
        // last block
        if (parent_it->second.sliding_window_size != -1 &&
            forked_block_idx == parent_it->second.last_block_idx) {
          TVM_FFI_ICHECK_LE(moved_offset, parent_it->second.last_block_attn_sink_size);
          parent_it->second.last_block_attn_sink_size -= moved_offset;
        }
      }
      global_block_pool_[child_block_idx].start_pos = fork_pos - in_page_offset;
      global_block_pool_[child_block_idx].seq_length = in_page_offset;

      if (in_page_offset > 0) {
        // Fork within a page and copy common page to child block partially
        int32_t src_page_id = global_block_pool_[forked_block_idx].page_ids[0];
        int32_t tgt_page_id = GetFreePage();
        global_block_pool_[child_block_idx].page_ids.push_back(tgt_page_id);
        CopySinglePage(src_page_id, tgt_page_id, in_page_offset);
      }
      break;
    }
    // Create the child sequence with the child block.
    seq_map_.insert({child_seq_id, Sequence(&global_block_pool_, child_block_idx)});
    dirty_aux_data_device_ = true;
  }

  void CopySinglePage(int32_t src_page_id, int32_t tgt_page_id, int64_t copy_length) {
    if (copy_stream_ != compute_stream_) {
      // Set the copy stream for copy.
      DeviceAPI::Get(device_)->SetStream(device_, copy_stream_);
    }
    for (int layer = 0; layer < num_layers_; ++layer) {
      ffi::Tensor page_layer_view = pages_[layer];
      f_copy_single_page_(page_layer_view, src_page_id, tgt_page_id, copy_length);
    }
    if (copy_stream_ != compute_stream_) {
      // Set the compute stream back.
      DeviceAPI::Get(device_)->SetStream(device_, compute_stream_);
    }
  }

  void CompactKVCopy() {
    int total_copy_length = commit_copy_length_indptr_host_.back();
    TVM_FFI_ICHECK_GE(total_copy_length, 0);
    if (total_copy_length == 0) {
      return;
    }

    // Copy indptr/src/dst arrays to GPU.
    aux_data_manager_->ResetCompactKVAuxDataCopy();
    ffi::Tensor commit_copy_length_indptr_view =
        aux_data_manager_->CopyCommitLengthIndptrAsync(&commit_copy_length_indptr_host_);
    ffi::Tensor commit_copy_src_dst_pos_in_page_table_view =
        aux_data_manager_->CopyCommitSrcDstPosInPageTableAsync(
            &commit_copy_src_pos_in_page_table_host_, &commit_copy_dst_pos_in_page_table_host_);
    aux_data_manager_->CommitCompactKVAuxDataCopy();

    // Invoke the copy kernel on copy stream.
    if (copy_stream_ != compute_stream_) {
      // Set the copy stream for copy.
      DeviceAPI::Get(device_)->SetStream(device_, copy_stream_);
    }
    TVM_FFI_ICHECK(f_compact_copy_.defined()) << "Function \"f_compact_copy\" is not defined.";
    for (int layer = 0; layer < num_layers_; ++layer) {
      f_compact_copy_(pages_[layer], commit_copy_length_indptr_view,
                      commit_copy_src_dst_pos_in_page_table_view, cur_batch_size_);
    }
    if (copy_stream_ != compute_stream_) {
      // Set the compute stream back.
      DeviceAPI::Get(device_)->SetStream(device_, compute_stream_);
    }

    // Note: We do not explicitly synchronize the copy stream here.
    // The safety is guaranteed by the synchronization pushed by the next round
    // of BeginForward, which also copies auxiliary data structure to GPU.
  }

  void EnableSlidingWindowForSeq(int64_t seq_id, int32_t sliding_window_size,
                                 int32_t attn_sink_size) final {
    // If per layer sliding window exists, enable sliding window for sequence
    TVM_FFI_ICHECK(support_sliding_window_ || support_layer_sliding_window_)
        << "The KV cache does not support sliding window.";
    auto it = seq_map_.find(seq_id);
    TVM_FFI_ICHECK(it != seq_map_.end())
        << "The sequence \"" << seq_id << "\" cannot be found in KV cache.";
    TVM_FFI_ICHECK_GE(attn_sink_size, 0)
        << "The specified attention sink size is expected to be non negative";
    TVM_FFI_ICHECK_GT(sliding_window_size, 0)
        << "The specified sliding window size should be positive.";
    TVM_FFI_ICHECK_LT(attn_sink_size, sliding_window_size)
        << "The attn sink size should be less than the sliding window size.";

    // Set the sliding window flag of the sequence.
    TVM_FFI_ICHECK_EQ(it->second.sliding_window_size, -1)
        << "A sequence cannot be enabled twice for sliding window.";

    // Compute the total length of the prefix blocks of this sequence.
    const Block& last_block = global_block_pool_[it->second.last_block_idx];
    int32_t prefix_length = it->second.seq_length - last_block.seq_length;
    TVM_FFI_ICHECK_GE(prefix_length, 0);
    // Since the prefix blocks cannot sliding, they are natural
    // attention sinks here. When the prefix length is already
    // larger than the specified attn sink size, we do not want to
    // introduce more sink. Therefore, we update the given attn sink size.
    it->second.last_block_attn_sink_size = std::max(attn_sink_size - prefix_length, 0);
    it->second.sliding_window_size = sliding_window_size;
  }

  void PopN(int64_t seq_id, int32_t n) final {
    auto it = seq_map_.find(seq_id);
    TVM_FFI_ICHECK(it != seq_map_.end())
        << "The sequence \"" << seq_id << "\" cannot be found in KV cache.";

    TVM_FFI_ICHECK_GE(n, 0) << "The length of popping " << n << " cannot be negative.";
    TVM_FFI_ICHECK_LE(n, it->second.seq_length)
        << "The sequence only has length " << it->second.seq_length
        << ", while the length of pop is " << n << " which exceeds the whole sequence length.";
    if (n == 0) {
      return;
    }

    int32_t block_idx = it->second.last_block_idx;
    // The block should have at least one reference, which comes from the sequence.
    TVM_FFI_ICHECK_GE(global_block_pool_[block_idx].external_ref_cnt, 1);
    while (block_idx != -1 && global_block_pool_[block_idx].external_ref_cnt == 1) {
      if (n > global_block_pool_[block_idx].seq_length) {
        n -= global_block_pool_[block_idx].seq_length;
        it->second.seq_length -= global_block_pool_[block_idx].seq_length;
        for (int32_t page_id : global_block_pool_[block_idx].page_ids) {
          free_page_ids_.push_back(page_id);
        }
        free_block_idx_.push_back(block_idx);
        block_idx = global_block_pool_[block_idx].parent_idx;
        it->second.last_block_idx = block_idx;
        continue;
      }
      if (n <= global_block_pool_[block_idx].seq_length) {
        int64_t cur_npage = global_block_pool_[block_idx].page_ids.size();
        int64_t tgt_npage =
            (global_block_pool_[block_idx].seq_length - n + page_size_ - 1) / page_size_;
        while (cur_npage > tgt_npage) {
          free_page_ids_.push_back(global_block_pool_[block_idx].page_ids.back());
          global_block_pool_[block_idx].page_ids.pop_back();
          --cur_npage;
        }
        it->second.seq_length -= n;
        global_block_pool_[block_idx].seq_length -= n;
        n = 0;
        break;
      }
    }

    if (n) {
      // We use a temporary sequence id for fork.
      // This temporary seq id will immediately end its effect outside this function.
      int64_t temp_seq_id = -1 - seq_id;
      TVM_FFI_ICHECK(seq_map_.find(temp_seq_id) == seq_map_.end());
      ForkSequence(seq_id, temp_seq_id, it->second.seq_length - n);
      TVM_FFI_ICHECK(seq_map_.find(temp_seq_id) != seq_map_.end());
      RemoveSequence(seq_id);
      TVM_FFI_ICHECK(seq_map_.find(seq_id) == seq_map_.end());
      auto it = seq_map_.find(temp_seq_id);
      seq_map_.insert({seq_id, it->second});
      seq_map_.erase(temp_seq_id);
    }

    dirty_aux_data_device_ = true;
  }

  /************** Raw Info Query **************/

  bool Empty() const final {
    return seq_map_.empty() &&                                     //
           free_block_idx_.size() == global_block_pool_.size() &&  //
           free_page_ids_.size() == static_cast<size_t>(num_total_pages_);
  }

  int32_t GetNumAvailablePages() const final { return free_page_ids_.size(); }

  int32_t GetTotalSequenceLength() const final {
    int32_t total_seq_len = 0;
    for (const auto& it : seq_map_) {
      total_seq_len += it.second.seq_length;
    }
    return total_seq_len;
  }

  ffi::String GetCheckpointMetadata(int64_t seq_id) const final {
    CheckCheckpointSequenceSupported(seq_id);
    const Sequence& seq = seq_map_.at(seq_id);

    namespace json = tvm::ffi::json;
    json::Object metadata = MakeLayoutMetadata();
    metadata.Set("layout_hash", GetLayoutHash());
    metadata.Set("seq_id", seq_id);
    metadata.Set("seq_length", static_cast<int64_t>(seq.seq_length));
    metadata.Set("logical_pages", MakeLogicalPageMetadata(seq));
    metadata.Set("groups", MakePageGroupMetadata(seq));
    return json::Stringify(metadata);
  }

  ffi::String GetLayoutHash() const final {
    CheckCheckpointLayoutSupported();
    return ffi::String(Uint64ToHex(FNV1a64(GetLayoutDescriptor())));
  }

  void ExportPageGroup(int64_t seq_id, int64_t group_id, ffi::Tensor dst) final {
    TVM_FFI_ICHECK(dst.defined());
    CheckCheckpointSequenceSupported(seq_id);
    const Sequence& seq = seq_map_.at(seq_id);
    TVM_FFI_ICHECK_GE(group_id, 0)
        << "PagedAttentionKVCache checkpoint export got invalid group id " << group_id << ".";
    TVM_FFI_ICHECK_LT(group_id, num_layers_)
        << "PagedAttentionKVCache checkpoint export got invalid group id " << group_id
        << ", but only " << num_layers_ << " groups are available.";

    int64_t num_logical_pages = GetNumLogicalPages(seq);
    CheckExportPageGroupTensor(dst, num_logical_pages);
    std::vector<int32_t> page_ids = GetCheckpointPageIds(seq, "checkpoint export");

    if (copy_stream_ != nullptr) {
      DeviceAPI::Get(device_)->SyncStreamFromTo(device_, copy_stream_, compute_stream_);
    }

    ffi::Tensor layer_pages = pages_[group_id];
    int64_t num_full_pages = seq.seq_length / page_size_;
    int64_t partial_page_length = seq.seq_length % page_size_;
    if (partial_page_length != 0) {
      TVM_FFI_ICHECK_LT(num_full_pages, num_logical_pages);
      ExportPartialCheckpointPage(layer_pages, page_ids[num_full_pages], dst, num_full_pages,
                                  partial_page_length);
    }

    int64_t logical_page_index = 0;
    while (logical_page_index < num_full_pages) {
      int64_t run_length = 1;
      while (logical_page_index + run_length < num_full_pages &&
             page_ids[logical_page_index + run_length] ==
                 page_ids[logical_page_index] + run_length) {
        ++run_length;
      }
      CopyCheckpointPageRun(layer_pages, page_ids[logical_page_index], dst, logical_page_index,
                            run_length);
      logical_page_index += run_length;
    }
    if (partial_page_length != 0) {
      ++logical_page_index;
    }
    TVM_FFI_ICHECK_EQ(logical_page_index, num_logical_pages);
  }

  void PrepareImport(int64_t seq_id, ffi::String metadata_json) final {
    CheckCheckpointLayoutSupported();
    TVM_FFI_ICHECK_EQ(seq_id, 0)
        << "PagedAttentionKVCache checkpoint import only supports sequence id 0, got " << seq_id
        << ".";
    ffi::json::Object metadata = ParseCheckpointMetadata(metadata_json);
    int64_t seq_length = CheckCheckpointImportMetadata(seq_id, metadata);
    int64_t num_logical_pages = GetExpectedNumLogicalPages(seq_length);

    Clear();
    int32_t block_idx = GetFreeBlock();
    Block& block = global_block_pool_[block_idx];
    block.start_pos = 0;
    block.seq_length = static_cast<int32_t>(seq_length);
    for (int64_t page_index = 0; page_index < num_logical_pages; ++page_index) {
      block.page_ids.push_back(GetFreePage());
    }
    seq_map_.insert({seq_id, Sequence(&global_block_pool_, block_idx)});
    checkpoint_import_in_progress_ = true;
    checkpoint_imported_groups_.assign(num_layers_, false);
    dirty_aux_data_device_ = true;
  }

  void ImportPageGroup(int64_t seq_id, int64_t group_id, ffi::Tensor src) final {
    TVM_FFI_ICHECK(src.defined());
    CheckCheckpointLayoutSupported();
    TVM_FFI_ICHECK(checkpoint_import_in_progress_)
        << "PagedAttentionKVCache checkpoint import has not been prepared.";
    CheckCheckpointSequenceState(seq_id);
    const Sequence& seq = seq_map_.at(seq_id);
    TVM_FFI_ICHECK_GE(group_id, 0)
        << "PagedAttentionKVCache checkpoint import got invalid group id " << group_id << ".";
    TVM_FFI_ICHECK_LT(group_id, num_layers_)
        << "PagedAttentionKVCache checkpoint import got invalid group id " << group_id
        << ", but only " << num_layers_ << " groups are available.";
    TVM_FFI_ICHECK(!checkpoint_imported_groups_[group_id])
        << "PagedAttentionKVCache checkpoint group " << group_id << " was already imported.";

    int64_t num_logical_pages = GetNumLogicalPages(seq);
    CheckImportPageGroupTensor(src, num_logical_pages);
    std::vector<int32_t> page_ids = GetCheckpointPageIds(seq, "checkpoint import");

    if (copy_stream_ != nullptr) {
      DeviceAPI::Get(device_)->SyncStreamFromTo(device_, copy_stream_, compute_stream_);
    }

    ffi::Tensor layer_pages = pages_[group_id];
    int64_t logical_page_index = 0;
    while (logical_page_index < num_logical_pages) {
      int64_t run_length = 1;
      while (logical_page_index + run_length < num_logical_pages &&
             page_ids[logical_page_index + run_length] ==
                 page_ids[logical_page_index] + run_length) {
        ++run_length;
      }
      CopyCheckpointPageRun(src, logical_page_index, layer_pages, page_ids[logical_page_index],
                            run_length);
      logical_page_index += run_length;
    }
    TVM_FFI_ICHECK_EQ(logical_page_index, num_logical_pages);
    checkpoint_imported_groups_[group_id] = true;
  }

  void FinishImport(int64_t seq_id) final {
    CheckCheckpointLayoutSupported();
    TVM_FFI_ICHECK(checkpoint_import_in_progress_)
        << "PagedAttentionKVCache checkpoint import has not been prepared.";
    CheckCheckpointSequenceState(seq_id);
    for (int64_t group_id = 0; group_id < num_layers_; ++group_id) {
      TVM_FFI_ICHECK(checkpoint_imported_groups_[group_id])
          << "PagedAttentionKVCache checkpoint import is missing group " << group_id << ".";
    }
    checkpoint_import_in_progress_ = false;
    checkpoint_imported_groups_.clear();
  }

  int32_t GetSequenceLength(int64_t seq_id) const final {
    TVM_FFI_ICHECK(!checkpoint_import_in_progress_)
        << "PagedAttentionKVCache sequence length is unavailable until checkpoint import is "
           "finished.";
    auto it = seq_map_.find(seq_id);
    TVM_FFI_ICHECK(it != seq_map_.end())
        << "The sequence \"" << seq_id << "\" cannot be found in KV cache.";
    return it->second.seq_length;
  }

  /************** Attention **************/

  void BeginForward(const ffi::Shape& seq_ids, const ffi::Shape& append_lengths,
                    const ffi::Optional<ffi::Shape>& opt_token_tree_parent_ptr) final {
    TVM_FFI_ICHECK(!checkpoint_import_in_progress_)
        << "PagedAttentionKVCache cannot begin a forward pass before checkpoint import is "
           "finished.";
    // Note: MLA does not supported tree attention for now.
    if (attn_kinds_[0] == AttnKind::kMLA) {
      TVM_FFI_ICHECK(!opt_token_tree_parent_ptr.has_value())
          << "Tree attention is not supported yet for MLA";
    }

    TVM_FFI_ICHECK_EQ(seq_ids.size(), append_lengths.size())
        << "The seq_ids size (" << seq_ids.size() << ") and append_lengths size ("
        << append_lengths.size() << ") mismatch.";
    cur_batch_size_ = seq_ids.size();
    cur_seq_ids_ = seq_ids;
    cur_append_lengths_ = append_lengths;

    // - Collect sequence/block/page information for attention.
    std::vector<Sequence*> sequences;
    std::vector<int32_t> last_block_length_before_append;
    is_decode_request_ = true;
    sequences.reserve(cur_batch_size_);
    last_block_length_before_append.reserve(cur_batch_size_);
    k_ragged_rope_pos_offset_host_.clear();
    for (int i = 0; i < cur_batch_size_; ++i) {
      auto it = seq_map_.find(seq_ids[i]);
      TVM_FFI_ICHECK(it != seq_map_.end())
          << "The sequence \"" << seq_ids[i] << "\" cannot be found in KV cache.";
      sequences.push_back(&it->second);
      last_block_length_before_append.push_back(
          global_block_pool_[it->second.last_block_idx].seq_length);
      int k_rope_offset = it->second.seq_length;
      if (!it->second.accepted_indices_committed) {
        int tree_size = static_cast<int>(it->second.token_tree_parent_ptr.size());
        k_rope_offset -= tree_size;
      }
      k_ragged_rope_pos_offset_host_.push_back(k_rope_offset);
      it->second.seq_length += append_lengths[i];
      if (append_lengths[i] != 1) {
        is_decode_request_ = false;
      }
    }

    auto [block_ids_on_depths, trailing_blocks] =
        GetBlockIdsOnDepth(sequences, global_block_pool_, cur_batch_size_);
    num_depths_ =
        std::min(static_cast<int>(block_ids_on_depths.size()), kPagedKVCacheMaxBlockDepth);
    TVM_FFI_ICHECK_LE(num_depths_, kPagedKVCacheMaxBlockDepth);

    std::vector<std::vector<std::pair<int32_t, int32_t>>> chunked_block_ids_arr;
    chunked_block_ids_arr.reserve(num_depths_);
    use_decode_kernel_.clear();
    for (int d = 0; d < num_depths_; ++d) {
      // We force the blocks at maximum depth not to coalesce, so that it can be concatenated with
      // trailing exceeding blocks.
      auto [chunked_block_ids, use_decode_kernel] = GetChunkedBlockIds(
          block_ids_on_depths[d], /*enable_coalesce=*/d != kPagedKVCacheMaxBlockDepth - 1,
          cur_append_lengths_, global_block_pool_, is_decode_request_);
      chunked_block_ids_arr.push_back(chunked_block_ids);
      use_decode_kernel_.push_back(use_decode_kernel);
    }

    if (num_depths_ == kPagedKVCacheMaxBlockDepth) {
      // Since we force the blocks at maximum depth not to coalesce, the output blocks at maximum
      // depth must have the same size as current batch.
      TVM_FFI_ICHECK_EQ(chunked_block_ids_arr[num_depths_ - 1].size(), cur_batch_size_);
    }

    append_before_attn_ = !support_sliding_window_ && use_decode_kernel_.back();
    if (NeedKernelBeginForward() && num_qo_heads_ / num_kv_heads_ >= 4) {
      // When GQA group size is at least 4 and FlashInfer is enabled,
      // we always use prefill kernel for better performance.
      // Note: For MLA, we always use prefill kernel, so values in `use_decode_kernel` will
      // be ignored for MLA.
      std::fill(use_decode_kernel_.begin(), use_decode_kernel_.end(), /*value=*/false);
    }

    bool has_previous_tree =
        std::any_of(sequences.begin(), sequences.end(),
                    [](const Sequence* sequence) { return !sequence->accepted_indices_committed; });
    if (has_previous_tree) {
      append_before_attn_ = true;
    }

    // - Check token tree validity and process the token tree.
    if (opt_token_tree_parent_ptr.has_value()) {
      TVM_FFI_ICHECK(!support_sliding_window_) << "Tree attention does not support sliding window.";
      TVM_FFI_ICHECK(rope_mode_ != RoPEMode::kInline)
          << "Tree attention does not support inline RoPE mode.";
      ConstructTokenTreeMask(sequences, opt_token_tree_parent_ptr.value(), block_ids_on_depths,
                             trailing_blocks);
    } else {
      // The input batch does not form trees. So each sequence in the batch
      // is required to have all past accepted tokens committed.
      for (int i = 0; i < cur_batch_size_; ++i) {
        Sequence* sequence = sequences[i];
        TVM_FFI_ICHECK(sequence->accepted_indices_committed)
            << "The input batch does not form a tree, in which case the sequences in the input "
               "batch are expected to have their accepted tokens token tree nodes committed. "
               "Please invoke CommitAcceptedTokenTreeNodes for sequence "
            << seq_ids[i];
        sequence->is_chain = true;
        sequence->token_tree_parent_ptr.clear();
        sequence->token_tree_node_depths.clear();
      }
      std::fill(is_chain_on_depths_.begin(), is_chain_on_depths_.end(), true);
    }

    if (append_before_attn_) {
      // Right now we use different kernels when depth is 1 or not 1.
      // For the case where maximum depth is 1, we create the auxiliary
      // data structure with regard to the page table after appending.
      for (int i = 0; i < cur_batch_size_; ++i) {
        ReserveAppendLengthInSeq(sequences[i], append_lengths[i]);
      }
    }

    for (int d = 0; d < num_depths_; ++d) {
      HostMemoryVector& qo_indptr_h = qo_indptr_on_depths_host_[d];
      HostMemoryVector& page_indptr_h = page_indptr_on_depths_host_[d];
      HostMemoryVector& page_indices_h = page_indices_on_depths_host_[d];
      HostMemoryVector& page_indptr_sliding_window_h =
          page_indptr_sliding_window_on_depths_host_[d];
      HostMemoryVector& page_indices_sliding_window_h =
          page_indices_sliding_window_on_depths_host_[d];
      HostMemoryVector& last_page_len_h = last_page_len_on_depths_host_[d];
      HostMemoryVector& sliding_window_offset_h = sliding_window_offset_on_depths_host_[d];
      HostMemoryVector& sink_size_h = sink_size_on_depths_host_[d];
      HostMemoryVector& k_rope_pos_offset_h = k_rope_pos_offset_on_depths_host_[d];
      HostMemoryVector& k_rope_pos_offset_sliding_window_h =
          k_rope_pos_offset_sliding_window_on_depths_host_[d];
      qo_indptr_h.clear();
      page_indptr_h.clear();
      page_indices_h.clear();
      page_indptr_sliding_window_h.clear();
      page_indices_sliding_window_h.clear();
      last_page_len_h.clear();
      sliding_window_offset_h.clear();
      sink_size_h.clear();
      k_rope_pos_offset_h.clear();
      k_rope_pos_offset_sliding_window_h.clear();
      qo_indptr_h.push_back(0);
      page_indptr_h.push_back(0);
      page_indptr_sliding_window_h.push_back(0);
      for (int i = 0; i < static_cast<int>(chunked_block_ids_arr[d].size()); ++i) {
        const auto& [block_id, chunk_append_length] = chunked_block_ids_arr[d][i];
        qo_indptr_h.push_back(qo_indptr_h.back() + chunk_append_length);
        if (block_id == -1) {
          page_indptr_h.push_back(page_indptr_h.back());
          page_indptr_sliding_window_h.push_back(page_indptr_sliding_window_h.back());
          last_page_len_h.push_back(0);
          sliding_window_offset_h.push_back(0);
          sink_size_h.push_back(0);
          k_rope_pos_offset_h.push_back(0);
          k_rope_pos_offset_sliding_window_h.push_back(0);
        } else {
          if (d < kPagedKVCacheMaxBlockDepth - 1) {
            // Blocks not at maximum depth
            const Block& block = global_block_pool_[block_id];
            page_indptr_h.push_back(page_indptr_h.back() + block.page_ids.size());
            for (int32_t page_id : block.page_ids) {
              page_indices_h.push_back(page_id);
              // Do the same for page_indices_sliding_window
            }

            // For sliding window, the first page and last page will both be partially used
            int32_t num_layer_sliding_pages =
                std::min(static_cast<int32_t>(block.page_ids.size()),
                         GetLayerSlidingWindowNumPages(block.seq_length));
            page_indptr_sliding_window_h.push_back(page_indptr_sliding_window_h.back() +
                                                   num_layer_sliding_pages);
            for (int i = page_indices_h.size() - num_layer_sliding_pages;
                 i < static_cast<int32_t>(page_indices_h.size()); i++) {
              page_indices_sliding_window_h.push_back(page_indices_h[i]);
            }
            // set up the page indices properly by choosing the last (sliding_window_size /
            // page_size_) pages (at most)
            last_page_len_h.push_back(
                block.seq_length == 0
                    ? 0
                    : (block.seq_length - block.sink_length + block.sliding_window_offset - 1) %
                              page_size_ +
                          1);
            if (support_layer_sliding_window_) {
              sliding_window_offset_h.push_back(GetLayerSlidingWindowOffset(block.seq_length));
            } else {
              sliding_window_offset_h.push_back(block.sliding_window_offset);
            }
            sink_size_h.push_back(block.sink_length);
            k_rope_pos_offset_h.push_back(block.start_pos);

            // If sliding window, we need to calculate the positional offset
            if (support_layer_sliding_window_) {
              k_rope_pos_offset_sliding_window_h.push_back(std::max<int64_t>(
                  0, block.start_pos + block.seq_length - layer_sliding_window_size_));
            }
          } else {
            // Blocks at maximum depth
            const Block& block = global_block_pool_[block_id];
            int32_t num_pages = static_cast<int32_t>(block.page_ids.size());
            int32_t total_seq_length = static_cast<int32_t>(block.seq_length);
            int32_t last_block_id = block_id;
            for (int32_t page_id : block.page_ids) {
              page_indices_h.push_back(page_id);
            }
            for (int32_t id : trailing_blocks[i]) {
              // Collect trailing blocks if available
              const Block& block = global_block_pool_[id];
              for (int32_t page_id : block.page_ids) {
                page_indices_h.push_back(page_id);
              }
              num_pages += block.page_ids.size();
              total_seq_length += block.seq_length;
              last_block_id = id;
            }
            page_indptr_h.push_back(page_indptr_h.back() + num_pages);
            int32_t num_layer_sliding_pages =
                std::min(num_pages, GetLayerSlidingWindowNumPages(total_seq_length));
            page_indptr_sliding_window_h.push_back(page_indptr_sliding_window_h.back() +
                                                   num_layer_sliding_pages);
            for (int i = page_indices_h.size() - num_layer_sliding_pages;
                 i < static_cast<int32_t>(page_indices_h.size()); i++) {
              page_indices_sliding_window_h.push_back(page_indices_h[i]);
            }
            const Block& last_block = global_block_pool_[last_block_id];
            last_page_len_h.push_back(total_seq_length == 0
                                          ? 0
                                          : (total_seq_length - last_block.sink_length +
                                             last_block.sliding_window_offset - 1) %
                                                    page_size_ +
                                                1);
            if (support_layer_sliding_window_) {
              sliding_window_offset_h.push_back(GetLayerSlidingWindowOffset(total_seq_length));
            } else {
              sliding_window_offset_h.push_back(last_block.sliding_window_offset);
            }
            sink_size_h.push_back(last_block.sink_length);
            k_rope_pos_offset_h.push_back(block.start_pos);
            if (support_layer_sliding_window_) {
              k_rope_pos_offset_sliding_window_h.push_back(std::max<int64_t>(
                  0, block.start_pos + total_seq_length - layer_sliding_window_size_));
            }
          }
        }
      }
    }

    if (!append_before_attn_) {
      // Right now we use different kernels when depth is 1 or not 1.
      // For the case where maximum depth is not 1, we create the auxiliary
      // data structure with regard to the page table before appending.
      for (int i = 0; i < cur_batch_size_; ++i) {
        ReserveAppendLengthInSeq(sequences[i], append_lengths[i]);
      }
    }

    // Map each the token position in the input batch to the position
    // in the global KV cache. The mapping is used in when appending k/v values.
    kv_len_arr_host_.clear();
    q_rope_position_map_host_.clear();
    append_position_map_host_.clear();
    kv_transfer_remote_position_map_host_.clear();
    kv_transfer_recver_id_host_.clear();
    kv_transfer_page_to_page_local_position_map_host_.clear();
    kv_transfer_page_to_page_remote_position_map_host_.clear();
    kv_transfer_page_to_page_recver_id_host_.clear();
    transfer_kv_ = false;
    page_to_page_transfer_kv_ = false;
    for (int i = 0; i < cur_batch_size_; ++i) {
      int64_t append_length = append_lengths[i];
      const Block& block = global_block_pool_[sequences[i]->last_block_idx];
      kv_len_arr_host_.push_back(block.seq_length);
      for (int64_t pos = 0; pos < append_length; ++pos) {
        if (sequences[i]->token_tree_node_depths.empty()) {
          q_rope_position_map_host_.push_back(k_ragged_rope_pos_offset_host_[i] + pos);
        } else {
          int64_t offset_in_tree =
              static_cast<int64_t>(sequences[i]->token_tree_parent_ptr.size()) - append_length;
          TVM_FFI_ICHECK_GE(offset_in_tree, 0);
          q_rope_position_map_host_.push_back(
              k_ragged_rope_pos_offset_host_[i] +
              sequences[i]->token_tree_node_depths[offset_in_tree + pos]);
        }

        int32_t pos_in_block = block.seq_length - append_length + pos;
        if (last_block_length_before_append[i] + pos < block.sink_length) {
          // The location to write is part of the attention sink.
          int32_t offset_in_block = last_block_length_before_append[i] + pos;
          append_position_map_host_.push_back(block.page_ids[offset_in_block / page_size_] *
                                                  page_size_ +
                                              offset_in_block % page_size_);
        } else if (pos_in_block < block.sink_length) {
          // The location to write is pinned by attn sink before the append.
          // Therefore we cannot write into the location.
          append_position_map_host_.push_back(-1);
        } else {
          // The location to write is in the sliding window.
          int32_t offset_in_block = pos_in_block - block.sink_length + block.sliding_window_offset;
          append_position_map_host_.push_back(block.page_ids[offset_in_block / page_size_] *
                                                  page_size_ +
                                              offset_in_block % page_size_);
        }
        int64_t pos_in_seq = sequences[i]->seq_length - append_length + pos;
        int64_t seq_send_start = sequences[i]->kv_transfer_metadata.start;
        if (pos_in_seq < seq_send_start) {
          kv_transfer_remote_position_map_host_.push_back(-1);
          kv_transfer_recver_id_host_.push_back(-1);
        } else {
          transfer_kv_ = true;
          kv_transfer_remote_position_map_host_.push_back(
              sequences[i]->kv_transfer_metadata.remote_position_map[pos_in_seq - seq_send_start]);
          kv_transfer_recver_id_host_.push_back(
              sequences[i]->kv_transfer_metadata.recver_pe_offset);
        }
      }
      if (!sequences[i]->kv_transfer_metadata.local_position_map.empty()) {
        page_to_page_transfer_kv_ = true;
        for (int pos = 0;
             pos < static_cast<int>(sequences[i]->kv_transfer_metadata.local_position_map.size());
             ++pos) {
          kv_transfer_page_to_page_local_position_map_host_.push_back(
              sequences[i]->kv_transfer_metadata.local_position_map[pos]);
          kv_transfer_page_to_page_remote_position_map_host_.push_back(
              sequences[i]->kv_transfer_metadata.remote_position_map[pos]);
          kv_transfer_page_to_page_recver_id_host_.push_back(
              sequences[i]->kv_transfer_metadata.recver_pe_offset);
        }
        sequences[i]->kv_transfer_metadata.local_position_map.clear();
      }
    }
  }

  void EndForward() final {
    if (kv_transfer_stream_ != nullptr) {
      DeviceAPI::Get(device_)->SyncStreamFromTo(device_, kv_transfer_stream_, compute_stream_);
    }
  }

  ffi::Shape DisaggPrepareRecv(int64_t seq_id, int append_length) final {
    // No CPU to GPU copy is needed.
    // Essentially we
    // (step 1.) redirect the preparation to BeginForward.
    BeginForward({seq_id}, {append_length}, /*opt_token_tree_parent_ptr=*/std::nullopt);
    // (step 2.) fetch the append_position_map, compress and return.
    // Compression format: [n, begin_1, length_1, begin_2, length_2, ..., begin_n, length_n]
    // The compressed format will be decompressed to:
    // [begin_1, begin_1+1, ..., begin_1+length_1-1, ..., begin_n, ..., begin_n+length_n-1]
    TVM_FFI_ICHECK_EQ(append_position_map_host_.size(), append_length);
    std::vector<int64_t> compressed_append_pos_map{/*num_segments=*/1,
                                                   append_position_map_host_[0]};
    for (int i = 1; i < append_length; ++i) {
      if (append_position_map_host_[i] != append_position_map_host_[i - 1] + 1) {
        // Terminate the current segment.
        compressed_append_pos_map.push_back(append_position_map_host_[i - 1] -
                                            compressed_append_pos_map.back() + 1);
        // Start a new segment.
        ++compressed_append_pos_map[0];
        compressed_append_pos_map.push_back(append_position_map_host_[i]);
      }
    }
    // Terminate the last segment.
    compressed_append_pos_map.push_back(append_position_map_host_.back() -
                                        compressed_append_pos_map.back() + 1);
    // The compressed array size should be "num_segments * 2 + 1".
    TVM_FFI_ICHECK_EQ(compressed_append_pos_map.size(), compressed_append_pos_map[0] * 2 + 1);
    return ffi::Shape{compressed_append_pos_map};
  }

  void DisaggMarkSend(int64_t seq_id, int64_t begin,
                      const ffi::Shape& compressed_remote_position_map, int32_t recver_pe_offset) {
    TVM_FFI_ICHECK(f_transfer_kv_.has_value());
    auto it = seq_map_.find(seq_id);
    TVM_FFI_ICHECK(it != seq_map_.end())
        << "The sequence \"" << seq_id << "\" cannot be found in KV cache.";
    Sequence* sequence = &it->second;
    sequence->kv_transfer_metadata.start = begin;
    int nsegments = compressed_remote_position_map[0];
    sequence->kv_transfer_metadata.remote_position_map.clear();
    for (int i = 0; i < nsegments; ++i) {
      int begin = compressed_remote_position_map[2 * i + 1];
      int length = compressed_remote_position_map[2 * i + 2];
      for (int j = 0; j < length; ++j) {
        sequence->kv_transfer_metadata.remote_position_map.push_back(begin + j);
      }
    }
    sequence->kv_transfer_metadata.recver_pe_offset = recver_pe_offset;

    sequence->kv_transfer_metadata.local_position_map.clear();
    if (begin >= sequence->seq_length) {
      return;
    }
    // Need to send existing KV.
    TVM_FFI_ICHECK_GT(static_cast<int>(sequence->kv_transfer_metadata.remote_position_map.size()),
                      sequence->seq_length - begin)
        << "Need at least one token to prefill";
    std::vector<int32_t> trace = sequence->GetBlockTrace(global_block_pool_);
    sequence->kv_transfer_metadata.local_position_map.reserve(sequence->seq_length - begin);
    bool done = false;
    for (auto it_block_id = trace.rbegin(); it_block_id != trace.rend(); ++it_block_id) {
      const Block& block = global_block_pool_[*it_block_id];
      for (int i = block.seq_length - 1; i >= 0; --i) {
        int32_t offset =
            i < block.sink_length ? i : i - block.sink_length + block.sliding_window_offset;
        int page_id = block.page_ids[offset / page_size_];
        int page_offset = offset % page_size_;
        sequence->kv_transfer_metadata.local_position_map.push_back(page_id * page_size_ +
                                                                    page_offset);
        if (static_cast<int>(sequence->kv_transfer_metadata.local_position_map.size()) ==
            sequence->seq_length - begin) {
          done = true;
          break;
        }
      }
      if (done) {
        break;
      }
    }
    std::reverse(sequence->kv_transfer_metadata.local_position_map.begin(),
                 sequence->kv_transfer_metadata.local_position_map.end());
  }

  void AttentionWithFusedQKV(int64_t layer_id, ffi::Tensor qkv_data,
                             ffi::Optional<ffi::Tensor> mask, ffi::Tensor o_data,
                             double sm_scale) final {
    TVM_FFI_ICHECK(qkv_data.defined());
    TVM_FFI_ICHECK(o_data.defined());
    // Part 1. Shape and dtype check.
    int64_t local_layer_id = layer_id - layer_id_begin_offset_;
    TVM_FFI_ICHECK_GE(local_layer_id, 0);
    TVM_FFI_ICHECK_LT(local_layer_id, num_layers_);
    ffi::Tensor pages = pages_[local_layer_id];
    TVM_FFI_ICHECK(qkv_data.dtype() == pages.dtype());
    TVM_FFI_ICHECK(o_data.dtype() == pages.dtype());
    TVM_FFI_ICHECK(attn_kinds_[layer_id] == AttnKind::kMHA ||
                   attn_kinds_[layer_id] == AttnKind::kMHASliding);

    // qkv_data: (num_total_length, num_qo_heads + 2 * num_kv_heads, qk_head_dim)
    // o_data: (num_total_length, num_qo_heads, qk_head_dim)

    TVM_FFI_ICHECK_EQ(qkv_data.ndim(), 3);
    TVM_FFI_ICHECK_EQ(o_data.ndim(), 3);
    for (int dim = 0; dim < 3; ++dim) {
      if (dim == 1) {
        TVM_FFI_ICHECK_EQ(qkv_data.shape()[1], num_qo_heads_ + 2 * num_kv_heads_);
        TVM_FFI_ICHECK_EQ(o_data.shape()[1], num_qo_heads_);
      } else {
        TVM_FFI_ICHECK_EQ(o_data.shape()[dim], qkv_data.shape()[dim]);
      }
    }

    TVM_FFI_ICHECK_EQ(qkv_data.shape()[2], qk_head_dim_);
    int64_t total_seq_length = 0;
    for (int64_t seq_id = 0; seq_id < cur_batch_size_; ++seq_id) {
      total_seq_length += cur_append_lengths_[seq_id];
    }
    TVM_FFI_ICHECK_LE(total_seq_length, qkv_data.shape()[0]);
    // Sync the copy stream and the compute stream.
    ComputeStreamWaitForCopyStream();
    // The auxiliary data structure on device must have been synchronized.
    TVM_FFI_ICHECK(!dirty_aux_data_device_);

    ffi::Tensor q_data =
        tvm::runtime::CreateView(temp_attn_q_device_.value(),
                                 {total_seq_length, num_qo_heads_, qk_head_dim_}, qkv_data.dtype());
    ffi::Tensor k_data =
        tvm::runtime::CreateView(temp_attn_k_device_.value(),
                                 {total_seq_length, num_kv_heads_, qk_head_dim_}, qkv_data.dtype());
    ffi::Tensor v_data =
        tvm::runtime::CreateView(temp_attn_v_device_.value(),
                                 {total_seq_length, num_kv_heads_, qk_head_dim_}, qkv_data.dtype());

    ffi::Tensor qkv_data_view = qkv_data;
    ffi::Tensor o_data_view = o_data;
    if (total_seq_length != qkv_data.shape()[0]) {
      qkv_data_view = tvm::runtime::CreateView(
          qkv_data, {total_seq_length, qkv_data.shape()[1], qkv_data.shape()[2]}, qkv_data.dtype());
      o_data_view = tvm::runtime::CreateView(
          o_data, {total_seq_length, num_qo_heads_, qk_head_dim_}, qkv_data.dtype());
    }
    // Part 2. Split fused qkv and apply rotary embedding to q/k data.
    if (transfer_kv_) {
      // The the compute stream needs to wait for the KV transfer stream.
      DeviceAPI::Get(device_)->SyncStreamFromTo(device_, kv_transfer_stream_, compute_stream_);
    }
    if (!rope_ext_factors_.has_value()) {
      f_split_rotary_(qkv_data_view, q_rope_position_map_view_.value(), q_data, k_data, v_data,
                      static_cast<int64_t>(rope_mode_ == RoPEMode::kNormal));
    } else {
      f_split_rotary_(qkv_data_view, q_rope_position_map_view_.value(), q_data, k_data, v_data,
                      rope_ext_factors_.value());
    }

    // Part 3. Append k/v data to kv-cache if flag "append_before_attn" is set.
    TVM_FFI_ICHECK(f_transpose_append_mha_.has_value());
    if (append_before_attn_) {
      f_transpose_append_mha_.value()(pages_[local_layer_id], k_data, v_data,
                                      append_position_map_view_.value());
    }
    // Part 4: KV transfer
    if (page_to_page_transfer_kv_) {
      DeviceAPI::Get(device_)->SyncStreamFromTo(device_, copy_stream_, kv_transfer_stream_);
      // FIXME: if the sender and recver's PP/TP degree do not match, we will need to first
      // get the view of remote pages, and then take the specific remote layer.
      // The KV transfer stream nees to wait for the compute stream.
      f_transfer_kv_page_to_page_.value()(
          pages_[local_layer_id], pages_[local_layer_id],
          kv_transfer_page_to_page_remote_position_map_view_.value(),
          kv_transfer_page_to_page_local_position_map_view_.value(),
          kv_transfer_page_to_page_recver_id_view_.value(), kv_transfer_stream_);
    }
    if (transfer_kv_) {
      // FIXME: if the sender and recver's PP/TP degree do not match, we will need to first
      // get the view of remote pages, and then take the specific remote layer.
      // The KV transfer stream nees to wait for the compute stream.
      DeviceAPI::Get(device_)->SyncStreamFromTo(device_, compute_stream_, kv_transfer_stream_);
      f_transfer_kv_.value()(pages_[local_layer_id], k_data, v_data,
                             kv_transfer_remote_position_map_view_.value(),
                             kv_transfer_recver_id_view_.value(), kv_transfer_stream_);
    }
    // Part 5: perform attention
    AttentionInternal(layer_id, q_data, k_data, v_data, o_data_view, sm_scale);
    // Part 6. Append k/v data to kv-cache if flag "append_before_attn" is not set.
    if (!append_before_attn_) {
      f_transpose_append_mha_.value()(pages_[local_layer_id], k_data, v_data,
                                      append_position_map_view_.value());
    }
  }

  void SelfAttention(int64_t layer_id, ffi::Tensor q_data, ffi::Tensor k_data, ffi::Tensor v_data,
                     ffi::Tensor o_data, ffi::Tensor lse_data, double sm_scale) final {
    TVM_FFI_ICHECK(q_data.defined());
    TVM_FFI_ICHECK(k_data.defined());
    TVM_FFI_ICHECK(v_data.defined());
    TVM_FFI_ICHECK(o_data.defined());
    TVM_FFI_ICHECK(lse_data.defined());
    // Shape and dtype check.
    int64_t local_layer_id = layer_id - layer_id_begin_offset_;
    TVM_FFI_ICHECK_GE(local_layer_id, 0);
    TVM_FFI_ICHECK_LT(local_layer_id, num_layers_);
    ffi::Tensor pages = pages_[local_layer_id];
    TVM_FFI_ICHECK(q_data.dtype() == pages.dtype());
    TVM_FFI_ICHECK(k_data.dtype() == pages.dtype());
    TVM_FFI_ICHECK(v_data.dtype() == pages.dtype());
    TVM_FFI_ICHECK(o_data.dtype() == pages.dtype());
    AttnKind attn_kind = attn_kinds_[layer_id];

    // q_data: (num_total_length, num_qo_heads, qk_head_dim)
    // k_data: (num_total_length, num_kv_heads, qk_head_dim)
    // v_data: (num_total_length, num_kv_heads, v_head_dim)
    // o_data: (num_total_length, num_qo_heads, v_head_dim)

    int64_t total_seq_length = 0;
    for (int64_t seq_id = 0; seq_id < cur_batch_size_; ++seq_id) {
      total_seq_length += cur_append_lengths_[seq_id];
    }
    TVM_FFI_ICHECK_EQ(q_data.ndim(), 3);
    TVM_FFI_ICHECK_EQ(k_data.ndim(), 3);
    TVM_FFI_ICHECK_EQ(v_data.ndim(), 3);
    TVM_FFI_ICHECK_EQ(o_data.ndim(), 3);
    TVM_FFI_ICHECK_EQ(q_data.shape()[0], total_seq_length);
    TVM_FFI_ICHECK_EQ(k_data.shape()[0], total_seq_length);
    TVM_FFI_ICHECK_EQ(v_data.shape()[0], total_seq_length);
    TVM_FFI_ICHECK_EQ(o_data.shape()[0], total_seq_length);

    // Sync the copy stream and the compute stream.
    ComputeStreamWaitForCopyStream();
    // The auxiliary data structure on device must have been synchronized.
    TVM_FFI_ICHECK(!dirty_aux_data_device_);

    if (attn_kind == AttnKind::kMHA) {
      MHASelfAttnInternal(q_data, k_data, v_data, o_data, lse_data, sm_scale);
    } else {
      MLASelfAttnInternal(q_data, k_data, v_data, o_data, lse_data, sm_scale);
    }
  }

  void CrossAttention(int64_t layer_id, ffi::Tensor q_data, ffi::Tensor o_data,
                      ffi::Tensor lse_data, double sm_scale) final {
    TVM_FFI_ICHECK(q_data.defined());
    TVM_FFI_ICHECK(o_data.defined());
    TVM_FFI_ICHECK(lse_data.defined());
    // Shape and dtype check.
    int64_t local_layer_id = layer_id - layer_id_begin_offset_;
    TVM_FFI_ICHECK_GE(local_layer_id, 0);
    TVM_FFI_ICHECK_LT(local_layer_id, num_layers_);
    ffi::Tensor pages = pages_[local_layer_id];
    TVM_FFI_ICHECK(q_data.dtype() == pages.dtype());
    TVM_FFI_ICHECK(o_data.dtype() == pages.dtype());
    AttnKind attn_kind = attn_kinds_[layer_id];

    // q_data: (num_total_length, num_qo_heads, qk_head_dim)
    // o_data: (num_total_length, num_qo_heads, v_head_dim)

    int64_t total_seq_length = 0;
    for (int64_t seq_id = 0; seq_id < cur_batch_size_; ++seq_id) {
      total_seq_length += cur_append_lengths_[seq_id];
    }
    TVM_FFI_ICHECK_EQ(q_data.ndim(), 3);
    TVM_FFI_ICHECK_EQ(o_data.ndim(), 3);
    TVM_FFI_ICHECK_EQ(q_data.shape()[0], total_seq_length);
    TVM_FFI_ICHECK_EQ(o_data.shape()[0], total_seq_length);
    TVM_FFI_ICHECK_EQ(q_data.shape()[1], num_qo_heads_);
    TVM_FFI_ICHECK_EQ(o_data.shape()[1], num_qo_heads_);
    TVM_FFI_ICHECK_EQ(q_data.shape()[2], qk_head_dim_);
    TVM_FFI_ICHECK_EQ(o_data.shape()[2], v_head_dim_);

    // Sync the copy stream and the compute stream.
    ComputeStreamWaitForCopyStream();
    // The auxiliary data structure on device must have been synchronized.
    TVM_FFI_ICHECK(!dirty_aux_data_device_);

    if (attn_kind == AttnKind::kMHA) {
      MHACrossAttnInternal(local_layer_id, q_data, o_data, lse_data, sm_scale,
                           /*is_first_kernel=*/true, /*causal=*/false);
    } else {
      MLACrossAttnInternal(local_layer_id, q_data, o_data, lse_data, sm_scale);
    }
  }

  void AttentionWithSharedKV(int64_t source_layer_id, ffi::Tensor q_data,
                             ffi::Tensor current_k_data, ffi::Tensor current_v_data,
                             ffi::Tensor o_data, double sm_scale) final {
    TVM_FFI_ICHECK(q_data.defined());
    TVM_FFI_ICHECK(current_k_data.defined());
    TVM_FFI_ICHECK(current_v_data.defined());
    TVM_FFI_ICHECK(o_data.defined());
    int64_t local_layer_id = source_layer_id - layer_id_begin_offset_;
    TVM_FFI_ICHECK_GE(local_layer_id, 0);
    TVM_FFI_ICHECK_LT(local_layer_id, num_layers_);
    ffi::Tensor pages = pages_[local_layer_id];
    TVM_FFI_ICHECK(q_data.dtype() == pages.dtype());
    TVM_FFI_ICHECK(current_k_data.dtype() == pages.dtype());
    TVM_FFI_ICHECK(current_v_data.dtype() == pages.dtype());
    TVM_FFI_ICHECK(o_data.dtype() == pages.dtype());
    TVM_FFI_ICHECK(attn_kinds_[source_layer_id] == AttnKind::kMHA ||
                   attn_kinds_[source_layer_id] == AttnKind::kMHASliding)
        << "Querying K/V from another logical layer is only supported for MHA caches.";

    int64_t total_seq_length = 0;
    for (int64_t seq_id = 0; seq_id < cur_batch_size_; ++seq_id) {
      total_seq_length += cur_append_lengths_[seq_id];
    }
    TVM_FFI_ICHECK_EQ(q_data.ndim(), 3);
    TVM_FFI_ICHECK_EQ(current_k_data.ndim(), 3);
    TVM_FFI_ICHECK_EQ(current_v_data.ndim(), 3);
    TVM_FFI_ICHECK_EQ(o_data.ndim(), 3);
    TVM_FFI_ICHECK_EQ(q_data.shape()[0], total_seq_length);
    TVM_FFI_ICHECK_EQ(current_k_data.shape()[0], total_seq_length);
    TVM_FFI_ICHECK_EQ(current_v_data.shape()[0], total_seq_length);
    TVM_FFI_ICHECK_EQ(o_data.shape()[0], total_seq_length);
    TVM_FFI_ICHECK_EQ(q_data.shape()[1], num_qo_heads_);
    TVM_FFI_ICHECK_EQ(current_k_data.shape()[1], num_kv_heads_);
    TVM_FFI_ICHECK_EQ(current_v_data.shape()[1], num_kv_heads_);
    TVM_FFI_ICHECK_EQ(o_data.shape()[1], num_qo_heads_);
    TVM_FFI_ICHECK_EQ(q_data.shape()[2], qk_head_dim_);
    TVM_FFI_ICHECK_EQ(current_k_data.shape()[2], qk_head_dim_);
    TVM_FFI_ICHECK_EQ(current_v_data.shape()[2], v_head_dim_);
    TVM_FFI_ICHECK_EQ(o_data.shape()[2], v_head_dim_);

    ComputeStreamWaitForCopyStream();
    TVM_FFI_ICHECK(!dirty_aux_data_device_);
    AttentionInternal(source_layer_id, q_data, current_k_data, current_v_data, o_data, sm_scale);
  }

  void AppendMLAKV(int64_t layer_id, ffi::Tensor kv_data) final {
    TVM_FFI_ICHECK(kv_data.defined());
    // Shape and dtype check.
    int64_t local_layer_id = layer_id - layer_id_begin_offset_;
    TVM_FFI_ICHECK_GE(local_layer_id, 0);
    TVM_FFI_ICHECK_LT(local_layer_id, num_layers_);
    ffi::Tensor pages = pages_[local_layer_id];
    TVM_FFI_ICHECK(kv_data.dtype() == pages.dtype());
    TVM_FFI_ICHECK(attn_kinds_[layer_id] == AttnKind::kMLA);

    // kv_data: (num_total_length, qk_head_dim)
    TVM_FFI_ICHECK_EQ(kv_data.ndim(), 2);
    int64_t total_seq_length = 0;
    for (int64_t seq_id = 0; seq_id < cur_batch_size_; ++seq_id) {
      total_seq_length += cur_append_lengths_[seq_id];
    }
    TVM_FFI_ICHECK_LE(kv_data.shape()[0], total_seq_length);
    TVM_FFI_ICHECK_EQ(kv_data.shape()[1], qk_head_dim_);
    // Sync the copy stream and the compute stream.
    ComputeStreamWaitForCopyStream();
    // The auxiliary data structure on device must have been synchronized.
    TVM_FFI_ICHECK(!dirty_aux_data_device_);

    TVM_FFI_ICHECK(f_transpose_append_mla_.has_value());
    f_transpose_append_mla_.value()(pages_[local_layer_id], kv_data,
                                    append_position_map_view_.value());
  }

  ffi::Array<ffi::Tensor> MergeAttnOutputInplace(ffi::Tensor o_self_attn, ffi::Tensor lse_self_attn,
                                                 ffi::Tensor o_cross_attn,
                                                 ffi::Tensor lse_cross_attn) final {
    TVM_FFI_ICHECK(o_self_attn.defined());
    TVM_FFI_ICHECK(lse_self_attn.defined());
    TVM_FFI_ICHECK(o_cross_attn.defined());
    TVM_FFI_ICHECK(lse_cross_attn.defined());
    TVM_FFI_ICHECK_GE(f_merge_inplace_.size(), 2)
        << "The general attention merge function is not defined.";
    f_merge_inplace_[1](o_self_attn, lse_self_attn, o_cross_attn, lse_cross_attn);
    return {o_self_attn, lse_self_attn};
  }

  void LinearAttention(int64_t layer_id, ffi::Tensor q_data, ffi::Tensor k_data, ffi::Tensor v_data,
                       double sm_scale) {
    TVM_FFI_ICHECK(q_data.defined());
    TVM_FFI_ICHECK(k_data.defined());
    TVM_FFI_ICHECK(v_data.defined());
    // Todo(ruihang): implement it
  }

  void CommitAcceptedTokenTreeNodes(const ffi::Shape& seq_ids,
                                    const ffi::Shape& leaf_indices) final {
    TVM_FFI_ICHECK_EQ(seq_ids.size(), leaf_indices.size())
        << "The given seq_ids and leaf_indices have different size.";
    int num_seq_to_commit = seq_ids.size();

    std::vector<Sequence*> sequences;
    sequences.reserve(num_seq_to_commit);
    bool is_chain = true;
    for (int i = 0; i < num_seq_to_commit; ++i) {
      auto it = seq_map_.find(seq_ids[i]);
      TVM_FFI_ICHECK(it != seq_map_.end())
          << "The sequence \"" << seq_ids[i] << "\" cannot be found in KV cache.";
      sequences.push_back(&it->second);
      is_chain = it->second.is_chain;
      TVM_FFI_ICHECK(leaf_indices[i] == -1 || !it->second.accepted_indices_committed)
          << "The accepted nodes of sequence " << seq_ids[i] << " are already committed.";
      TVM_FFI_ICHECK_GE(leaf_indices[i], -1)
          << "Invalid tree index " << leaf_indices[i] << " which is less than -1";
      TVM_FFI_ICHECK_LT(leaf_indices[i],
                        static_cast<int64_t>(it->second.token_tree_parent_ptr.size()))
          << "Invalid tree index " << leaf_indices[i]
          << " which is larger than or equals to the append length "
          << it->second.token_tree_parent_ptr.size() << " of the sequence";
    }

    if (!is_chain) {
      commit_copy_length_indptr_host_.clear();
      commit_copy_src_pos_in_page_table_host_.clear();
      commit_copy_dst_pos_in_page_table_host_.clear();
      commit_copy_length_indptr_host_.push_back(0);

      for (int i = 0; i < num_seq_to_commit; ++i) {
        if (leaf_indices[i] == -1) {
          // No node is accepted. All nodes in the token tree need to be popped.
          commit_copy_length_indptr_host_.push_back(commit_copy_length_indptr_host_.back());
          continue;
        }

        // Get the accepted node path on the token tree.
        std::vector<int32_t> path_on_tree;
        path_on_tree.reserve(sequences[i]->token_tree_node_depths[leaf_indices[i]] + 1);
        int node = leaf_indices[i];
        while (node != -1) {
          path_on_tree.push_back(node);
          node = sequences[i]->token_tree_parent_ptr[node];
        }
        TVM_FFI_ICHECK_EQ(path_on_tree.size(),
                          sequences[i]->token_tree_node_depths[leaf_indices[i]] + 1);
        // Get the destination array (range [0, path_length - 1)) of KV cache copy.
        std::vector<int32_t> copy_dst_pos_in_seq;
        copy_dst_pos_in_seq.resize(path_on_tree.size());
        std::iota(copy_dst_pos_in_seq.rbegin(), copy_dst_pos_in_seq.rend(), /*value=*/0);
        // Remove the positions whose KV data do not need copy.
        while (!path_on_tree.empty() && path_on_tree.back() == copy_dst_pos_in_seq.back()) {
          path_on_tree.pop_back();
          copy_dst_pos_in_seq.pop_back();
        }
        // Reverse the position arrays so that they are in ascending order.
        std::reverse(path_on_tree.begin(), path_on_tree.end());
        std::reverse(copy_dst_pos_in_seq.begin(), copy_dst_pos_in_seq.end());

        // Convert the in-sequence src/dst positions to src/dst positions in page table
        // by looking up "append_position_map".
        for (int p = 0; p < static_cast<int>(path_on_tree.size()); ++p) {
          commit_copy_src_pos_in_page_table_host_.push_back(
              append_position_map_host_[cur_append_lengths_indptr_host_[i] + path_on_tree[p]]);
          commit_copy_dst_pos_in_page_table_host_.push_back(
              append_position_map_host_[cur_append_lengths_indptr_host_[i] +
                                        copy_dst_pos_in_seq[p]]);
        }
        commit_copy_length_indptr_host_.push_back(commit_copy_length_indptr_host_.back() +
                                                  path_on_tree.size());
      }

      // Compact the KV data for each sequence by copying KV data.
      CompactKVCopy();
    }

    // - Update the KV cache page data structure.
    //   Note: Function "PopN" only changes the page table structure and does not
    //         change the KV cache data. Therefore, we can directly use it, since
    //         we have already launched all copies.
    for (int i = 0; i < num_seq_to_commit; ++i) {
      int64_t length_to_pop =
          cur_append_lengths_[i] -
          (leaf_indices[i] != -1 ? (sequences[i]->token_tree_node_depths[leaf_indices[i]] + 1) : 0);
      PopN(cur_seq_ids_[i], length_to_pop);
      // Reset the sequence states.
      sequences[i]->accepted_indices_committed = true;
      sequences[i]->token_tree_parent_ptr.clear();
      sequences[i]->token_tree_node_depths.clear();
    }
  }

  ffi::Tensor GetQueryPositions() final {
    // Sync the copy stream and the compute stream.
    ComputeStreamWaitForCopyStream();
    // The auxiliary data structure on device must have been synchronized.
    TVM_FFI_ICHECK(!dirty_aux_data_device_);
    return q_rope_position_map_view_.value();
  };

  void DebugGetKV(int64_t seq_id, int64_t start_pos, int64_t end_pos, ffi::Tensor k_data,
                  ffi::Tensor v_data) final {
    TVM_FFI_ICHECK(k_data.defined());
    TVM_FFI_ICHECK(v_data.defined());
    TVM_FFI_ICHECK(f_debug_get_kv_.has_value())
        << "PageAttentionKVCache requires the `f_debug_get_kv` to be explicitly passed in when "
           "initialization. Please construct the KV cache with `f_debug_get_kv`.";

    const Sequence& seq = seq_map_.at(seq_id);
    TVM_FFI_ICHECK_GE(start_pos, 0)
        << "DebugGetKV does not accept negative start_pos " << start_pos;
    TVM_FFI_ICHECK_LE(end_pos, seq.seq_length) << "DebugGetKV does not accept out-of-range end_pos";
    TVM_FFI_ICHECK_LT(start_pos, end_pos) << "DebugGetKV does not accept \"start_pos >= end_pos\"";

    // k/v_data: (num_layers, seq_length, num_kv_heads, qk_head_dim)
    static constexpr const char* error_msg =
        "DebugGetKV expects the k_data in layout (num_layers, seq_length, num_kv_heads, "
        "qk_head_dim).";
    std::vector<ffi::Tensor*> vec_kv_data = {&k_data, &v_data};
    for (const ffi::Tensor* data_ptr : vec_kv_data) {
      TVM_FFI_ICHECK_EQ((*data_ptr).ndim(), 4) << error_msg;
      TVM_FFI_ICHECK_EQ((*data_ptr).shape()[0], num_layers_)
          << error_msg << " The number of layers mismatches.";
      TVM_FFI_ICHECK_EQ((*data_ptr).shape()[1], end_pos - start_pos)
          << error_msg << " The sequence length mismatches.";
      TVM_FFI_ICHECK_EQ((*data_ptr).shape()[2], num_kv_heads_)
          << error_msg << " The number of heads mismatches.";
      TVM_FFI_ICHECK_EQ((*data_ptr).shape()[3], qk_head_dim_)
          << error_msg << " The number of head features mismatches.";
    }

    std::vector<int32_t> trace = seq.GetBlockTrace(global_block_pool_);
    std::vector<int32_t> append_position_map;
    append_position_map.reserve(seq.seq_length);
    for (int32_t block_id : trace) {
      const Block& block = global_block_pool_[block_id];
      for (int i = 0; i < block.seq_length; ++i) {
        int32_t offset =
            i < block.sink_length ? i : i - block.sink_length + block.sliding_window_offset;
        int page_id = block.page_ids[offset / page_size_];
        int page_offset = offset % page_size_;
        append_position_map.push_back(page_id * page_size_ + page_offset);
      }
    }
    ffi::Tensor position_map_device =
        tvm::runtime::Empty({end_pos - start_pos}, dtype_aux_, device_);
    tvm::runtime::CopyFromBytes(
        position_map_device, append_position_map.data() + start_pos,
        (end_pos - start_pos) * ((dtype_aux_.bits * dtype_aux_.lanes + 7) / 8));
    for (int64_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
      TVM_FFI_ICHECK(attn_kinds_[layer_id] == AttnKind::kMHA)
          << "Only MHA is supported for DebugGetKV";
      f_debug_get_kv_.value()(pages_[layer_id], position_map_device, k_data, v_data, layer_id);
    }
  }

  void DebugGetKVMLA(int64_t seq_id, int64_t start_pos, int64_t end_pos,
                     ffi::Tensor kv_data) final {
    TVM_FFI_ICHECK(kv_data.defined());
    TVM_FFI_ICHECK(f_debug_get_kv_.has_value())
        << "PageAttentionKVCache requires the `f_debug_get_kv` to be explicitly passed in when "
           "initialization. Please construct the KV cache with `f_debug_get_kv`.";

    const Sequence& seq = seq_map_.at(seq_id);
    TVM_FFI_ICHECK_GE(start_pos, 0)
        << "DebugGetKV does not accept negative start_pos " << start_pos;
    TVM_FFI_ICHECK_LE(end_pos, seq.seq_length) << "DebugGetKV does not accept out-of-range end_pos";
    TVM_FFI_ICHECK_LT(start_pos, end_pos) << "DebugGetKV does not accept \"start_pos >= end_pos\"";

    // kv_data: (num_layers, seq_length, qk_head_dim)
    static constexpr const char* error_msg =
        "DebugGetKV expects the kv_data in layout (num_layers, seq_length, qk_head_dim).";
    TVM_FFI_ICHECK_EQ(kv_data.ndim(), 3) << error_msg;
    TVM_FFI_ICHECK_EQ(kv_data.shape()[0], num_layers_)
        << error_msg << " The number of layers mismatches.";
    TVM_FFI_ICHECK_EQ(kv_data.shape()[1], end_pos - start_pos)
        << error_msg << " The sequence length mismatches.";
    TVM_FFI_ICHECK_EQ(kv_data.shape()[2], qk_head_dim_)
        << error_msg << " The number of head features mismatches.";

    std::vector<int32_t> trace = seq.GetBlockTrace(global_block_pool_);
    std::vector<int32_t> append_position_map;
    append_position_map.reserve(seq.seq_length);
    for (int32_t block_id : trace) {
      const Block& block = global_block_pool_[block_id];
      for (int i = 0; i < block.seq_length; ++i) {
        int32_t offset =
            i < block.sink_length ? i : i - block.sink_length + block.sliding_window_offset;
        int page_id = block.page_ids[offset / page_size_];
        int page_offset = offset % page_size_;
        append_position_map.push_back(page_id * page_size_ + page_offset);
      }
    }
    ffi::Tensor position_map_device =
        tvm::runtime::Empty({end_pos - start_pos}, dtype_aux_, device_);
    tvm::runtime::CopyFromBytes(
        position_map_device, append_position_map.data() + start_pos,
        (end_pos - start_pos) * ((dtype_aux_.bits * dtype_aux_.lanes + 7) / 8));
    for (int64_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
      TVM_FFI_ICHECK(attn_kinds_[layer_id] == AttnKind::kMLA)
          << "Only MHA is supported for DebugGetKVMLA";
      f_debug_get_kv_.value()(pages_[layer_id], position_map_device, kv_data, layer_id);
    }
  }

  void DebugSetKV(int64_t seq_id, int64_t start_pos, ffi::Tensor k_data, ffi::Tensor v_data) final {
    TVM_FFI_ICHECK(k_data.defined());
    TVM_FFI_ICHECK(v_data.defined());
    TVM_FFI_ICHECK(false) << "DebugSetKV for PageAttentionKVCache not implemented yet.";
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.vm.PagedAttentionKVCache", PagedAttentionKVCacheObj,
                                    AttentionKVCacheObj);

 private:
  void CheckCheckpointLayoutSupported() const {
    for (int64_t layer_id = layer_id_begin_offset_; layer_id < layer_id_end_offset_; ++layer_id) {
      AttnKind attn_kind = attn_kinds_[layer_id];
      TVM_FFI_ICHECK(attn_kind == AttnKind::kMHA)
          << "PagedAttentionKVCache checkpointing only supports full-context MHA/GQA layers.";
    }
    TVM_FFI_ICHECK_EQ(qk_head_dim_, v_head_dim_)
        << "PagedAttentionKVCache checkpointing requires qk_head_dim to equal v_head_dim.";
    TVM_FFI_ICHECK(!support_sliding_window_ && !support_layer_sliding_window_)
        << "PagedAttentionKVCache checkpointing does not support sliding-window cache layouts.";
    TVM_FFI_ICHECK(!rope_ext_factors_.has_value())
        << "PagedAttentionKVCache checkpointing does not support RoPE extension factors.";
    TVM_FFI_ICHECK(!f_transfer_kv_.has_value() && !f_transfer_kv_page_to_page_.has_value())
        << "PagedAttentionKVCache checkpointing does not support KV transfer/disaggregation.";
  }

  void CheckCheckpointSequenceSupported(int64_t seq_id) const {
    CheckCheckpointLayoutSupported();
    TVM_FFI_ICHECK(!checkpoint_import_in_progress_)
        << "PagedAttentionKVCache checkpoint import must be finished before exporting a "
           "checkpoint.";
    CheckCheckpointSequenceState(seq_id);
  }

  void CheckCheckpointSequenceState(int64_t seq_id) const {
    TVM_FFI_ICHECK_EQ(seq_id, 0)
        << "PagedAttentionKVCache checkpointing only supports sequence id 0, got " << seq_id << ".";
    auto it = seq_map_.find(seq_id);
    TVM_FFI_ICHECK(it != seq_map_.end())
        << "The sequence \"" << seq_id << "\" cannot be found in KV cache.";
    const Sequence& seq = it->second;
    TVM_FFI_ICHECK(seq.accepted_indices_committed && seq.is_chain)
        << "PagedAttentionKVCache checkpointing requires committed token-chain state.";
    TVM_FFI_ICHECK_EQ(seq.sliding_window_size, -1)
        << "PagedAttentionKVCache checkpointing does not support sequences with sliding window.";
  }

  int64_t GetExpectedNumLogicalPages(int64_t seq_length) const {
    return (seq_length + page_size_ - 1) / page_size_;
  }

  ffi::json::Object ParseCheckpointMetadata(const ffi::String& metadata_json) const {
    ffi::String error_msg;
    ffi::json::Value json_info = ffi::json::Parse(metadata_json, &error_msg);
    TVM_FFI_ICHECK(error_msg.empty())
        << "Failed to parse PagedAttentionKVCache checkpoint metadata JSON: " << error_msg << ".";
    TVM_FFI_ICHECK(json_info.as<ffi::json::Object>())
        << "PagedAttentionKVCache checkpoint metadata should be a JSON object.";
    return json_info.cast<ffi::json::Object>();
  }

  ffi::json::Value GetJSONField(const ffi::json::Object& object, const char* field,
                                const char* context) const {
    auto it = object.find(field);
    TVM_FFI_ICHECK(it != object.end()) << context << " missing field \"" << field << "\".";
    return (*it).second;
  }

  int64_t GetJSONIntegerField(const ffi::json::Object& object, const char* field,
                              const char* context) const {
    return GetJSONField(object, field, context).cast<int64_t>();
  }

  double GetJSONNumberField(const ffi::json::Object& object, const char* field,
                            const char* context) const {
    return GetJSONField(object, field, context).cast<double>();
  }

  bool GetJSONBoolField(const ffi::json::Object& object, const char* field,
                        const char* context) const {
    return GetJSONField(object, field, context).cast<bool>();
  }

  std::string GetJSONStringField(const ffi::json::Object& object, const char* field,
                                 const char* context) const {
    return std::string(GetJSONField(object, field, context).cast<ffi::String>());
  }

  ffi::json::Array GetJSONArrayField(const ffi::json::Object& object, const char* field,
                                     const char* context) const {
    return GetJSONField(object, field, context).cast<ffi::json::Array>();
  }

  void CheckJSONIntegerField(const ffi::json::Object& object, const char* field, int64_t expected,
                             const char* context) const {
    int64_t value = GetJSONIntegerField(object, field, context);
    TVM_FFI_ICHECK_EQ(value, expected)
        << context << " field \"" << field << "\" mismatch: expected " << expected << ", got "
        << value << ".";
  }

  void CheckJSONStringField(const ffi::json::Object& object, const char* field,
                            const std::string& expected, const char* context) const {
    std::string value = GetJSONStringField(object, field, context);
    TVM_FFI_ICHECK_EQ(value, expected)
        << context << " field \"" << field << "\" mismatch: expected " << expected << ", got "
        << value << ".";
  }

  void CheckJSONNumberField(const ffi::json::Object& object, const char* field, double expected,
                            const char* context) const {
    double value = GetJSONNumberField(object, field, context);
    TVM_FFI_ICHECK_EQ(value, expected)
        << context << " field \"" << field << "\" mismatch: expected " << expected << ", got "
        << value << ".";
  }

  void CheckJSONBoolField(const ffi::json::Object& object, const char* field, bool expected,
                          const char* context) const {
    bool value = GetJSONBoolField(object, field, context);
    TVM_FFI_ICHECK_EQ(value, expected)
        << context << " field \"" << field << "\" mismatch: expected " << expected << ", got "
        << value << ".";
  }

  void CheckCheckpointImportLayout(const ffi::json::Object& metadata) const {
    static constexpr const char* context = "PagedAttentionKVCache checkpoint import metadata";
    CheckJSONIntegerField(metadata, "format_version", kPagedKVCacheCheckpointFormatVersion,
                          context);
    CheckJSONStringField(metadata, "cache_type", kPagedKVCacheCheckpointRuntime, context);
    CheckJSONIntegerField(metadata, "page_size", page_size_, context);
    CheckJSONIntegerField(metadata, "num_layers", num_layers_, context);
    CheckJSONIntegerField(metadata, "layer_begin", layer_id_begin_offset_, context);
    CheckJSONIntegerField(metadata, "layer_end", layer_id_end_offset_, context);
    CheckJSONIntegerField(metadata, "num_qo_heads", num_qo_heads_, context);
    CheckJSONIntegerField(metadata, "num_kv_heads", num_kv_heads_, context);
    CheckJSONIntegerField(metadata, "qk_head_dim", qk_head_dim_, context);
    CheckJSONIntegerField(metadata, "v_head_dim", v_head_dim_, context);
    CheckJSONStringField(metadata, "dtype", std::string(ffi::DLDataTypeToString(kv_dtype_)),
                         context);
    CheckJSONStringField(metadata, "rope_mode", RoPEModeToString(rope_mode_), context);
    CheckJSONNumberField(metadata, "rotary_scale", rotary_scale_, context);
    CheckJSONNumberField(metadata, "rotary_theta", rotary_theta_, context);
    CheckJSONBoolField(metadata, "has_rope_ext_factors", rope_ext_factors_.has_value(), context);
    CheckJSONBoolField(metadata, "support_sliding_window", support_sliding_window_, context);
    CheckJSONBoolField(metadata, "support_layer_sliding_window", support_layer_sliding_window_,
                       context);
    CheckJSONStringField(metadata, "page_group_layout",
                         "1,num_logical_pages,2,num_kv_heads,page_size,qk_head_dim", context);
    TVM_FFI_ICHECK_GT(GetJSONIntegerField(metadata, "reserved_num_seqs", context), 0)
        << context << " field \"reserved_num_seqs\" must be positive.";
    TVM_FFI_ICHECK_GT(GetJSONIntegerField(metadata, "num_total_pages", context), 0)
        << context << " field \"num_total_pages\" must be positive.";
    TVM_FFI_ICHECK_GT(GetJSONIntegerField(metadata, "prefill_chunk_size", context), 0)
        << context << " field \"prefill_chunk_size\" must be positive.";
    ffi::String expected_layout_hash = GetLayoutHash();
    std::string layout_hash = GetJSONStringField(metadata, "layout_hash", context);
    TVM_FFI_ICHECK_EQ(layout_hash, std::string(expected_layout_hash))
        << "PagedAttentionKVCache checkpoint import layout hash mismatch: expected "
        << expected_layout_hash << ", got " << layout_hash << ".";

    ffi::json::Array attn_kinds = GetJSONArrayField(metadata, "attn_kinds", context);
    TVM_FFI_ICHECK_EQ(attn_kinds.size(), num_layers_)
        << context << " field \"attn_kinds\" size mismatch.";
    for (int64_t local_layer = 0; local_layer < num_layers_; ++local_layer) {
      std::string attn_kind = std::string(attn_kinds[local_layer].cast<ffi::String>());
      std::string expected = AttnKindToString(attn_kinds_[layer_id_begin_offset_ + local_layer]);
      TVM_FFI_ICHECK_EQ(attn_kind, expected)
          << context << " field \"attn_kinds\" mismatch at local layer " << local_layer
          << ": expected " << expected << ", got " << attn_kind << ".";
    }
  }

  void CheckCheckpointImportLogicalPages(const ffi::json::Object& metadata,
                                         int64_t seq_length) const {
    static constexpr const char* context = "PagedAttentionKVCache checkpoint import metadata";
    int64_t num_logical_pages = GetExpectedNumLogicalPages(seq_length);
    ffi::json::Array logical_pages = GetJSONArrayField(metadata, "logical_pages", context);
    TVM_FFI_ICHECK_EQ(static_cast<int64_t>(logical_pages.size()), num_logical_pages)
        << "PagedAttentionKVCache checkpoint import sequence length mismatch: seq_length "
        << seq_length << " requires " << num_logical_pages << " logical pages, but metadata has "
        << logical_pages.size() << ".";

    int64_t expected_start = 0;
    for (int64_t i = 0; i < num_logical_pages; ++i) {
      ffi::json::Object page = logical_pages[i].cast<ffi::json::Object>();
      int64_t expected_length = std::min<int64_t>(page_size_, seq_length - expected_start);
      CheckJSONIntegerField(page, "logical_page_index", i, context);
      CheckJSONIntegerField(page, "start_pos", expected_start, context);
      CheckJSONIntegerField(page, "length", expected_length, context);
      expected_start += expected_length;
    }
    TVM_FFI_ICHECK_EQ(expected_start, seq_length)
        << "PagedAttentionKVCache checkpoint import logical pages do not cover seq_length "
        << seq_length << ".";
  }

  void CheckPageGroupMetadataShape(const ffi::json::Object& group, int64_t num_logical_pages,
                                   const char* context) const {
    ffi::json::Array shape = GetJSONArrayField(group, "shape", context);
    std::vector<int64_t> expected_shape = {1,          num_logical_pages, 2, num_kv_heads_,
                                           page_size_, qk_head_dim_};
    TVM_FFI_ICHECK_EQ(shape.size(), expected_shape.size())
        << context << " field \"shape\" rank mismatch.";
    for (int64_t i = 0; i < static_cast<int64_t>(expected_shape.size()); ++i) {
      int64_t dim = shape[i].cast<int64_t>();
      TVM_FFI_ICHECK_EQ(dim, expected_shape[i])
          << context << " field \"shape\" mismatch at dim " << i << ": expected "
          << expected_shape[i] << ", got " << dim << ".";
    }
  }

  void CheckCheckpointImportGroups(const ffi::json::Object& metadata,
                                   int64_t num_logical_pages) const {
    static constexpr const char* context = "PagedAttentionKVCache checkpoint import group metadata";
    ffi::json::Array groups = GetJSONArrayField(metadata, "groups", context);
    TVM_FFI_ICHECK_EQ(static_cast<int64_t>(groups.size()), num_layers_)
        << context << " size mismatch: expected " << num_layers_ << ", got " << groups.size()
        << ".";
    for (int64_t local_layer = 0; local_layer < num_layers_; ++local_layer) {
      ffi::json::Object group = groups[local_layer].cast<ffi::json::Object>();
      CheckJSONIntegerField(group, "group_index", local_layer, context);
      CheckJSONIntegerField(group, "layer_begin", layer_id_begin_offset_ + local_layer, context);
      CheckJSONIntegerField(group, "layer_end", layer_id_begin_offset_ + local_layer + 1, context);
      CheckJSONIntegerField(group, "num_logical_pages", num_logical_pages, context);
      CheckJSONStringField(group, "dtype", std::string(ffi::DLDataTypeToString(kv_dtype_)),
                           context);
      CheckPageGroupMetadataShape(group, num_logical_pages, context);
      CheckJSONIntegerField(group, "nbytes", GetCheckpointGroupNBytes(num_logical_pages), context);
    }
  }

  int64_t CheckCheckpointImportMetadata(int64_t seq_id, const ffi::json::Object& metadata) const {
    CheckCheckpointImportLayout(metadata);
    CheckJSONIntegerField(metadata, "seq_id", seq_id,
                          "PagedAttentionKVCache checkpoint import metadata");
    int64_t seq_length = GetJSONIntegerField(metadata, "seq_length",
                                             "PagedAttentionKVCache checkpoint import metadata");
    TVM_FFI_ICHECK_GE(seq_length, 0)
        << "PagedAttentionKVCache checkpoint import seq_length cannot be negative.";
    TVM_FFI_ICHECK_LE(seq_length, std::numeric_limits<int32_t>::max())
        << "PagedAttentionKVCache checkpoint import seq_length exceeds int32 range.";
    int64_t num_logical_pages = GetExpectedNumLogicalPages(seq_length);
    int64_t source_num_total_pages = GetJSONIntegerField(
        metadata, "num_total_pages", "PagedAttentionKVCache checkpoint import metadata");
    TVM_FFI_ICHECK_LE(num_logical_pages, source_num_total_pages)
        << "PagedAttentionKVCache checkpoint metadata requires " << num_logical_pages
        << " pages, but reports a source cache with only " << source_num_total_pages << " pages.";
    TVM_FFI_ICHECK_LE(num_logical_pages, num_total_pages_)
        << "PagedAttentionKVCache checkpoint import requires " << num_logical_pages
        << " pages, but this cache only has " << num_total_pages_ << " pages.";
    CheckCheckpointImportLogicalPages(metadata, seq_length);
    CheckCheckpointImportGroups(metadata, num_logical_pages);
    return seq_length;
  }

  void CheckPageGroupTensor(const ffi::Tensor& tensor, int64_t num_logical_pages,
                            const char* api_name) const {
    TVM_FFI_ICHECK(tensor.defined());
    std::string error_msg = std::string(api_name) +
                            " expects the tensor in layout "
                            "(1,num_logical_pages,2,num_kv_heads,page_size,qk_head_dim).";
    TVM_FFI_ICHECK(tensor.defined()) << error_msg;
    TVM_FFI_ICHECK(tensor.dtype() == kv_dtype_) << error_msg << " The dtype mismatches, expected "
                                                << kv_dtype_ << ", got " << tensor.dtype() << ".";
    TVM_FFI_ICHECK_EQ(tensor.ndim(), 6) << error_msg;
    TVM_FFI_ICHECK_EQ(tensor.shape()[0], 1) << error_msg << " The group count mismatches.";
    TVM_FFI_ICHECK_EQ(tensor.shape()[1], num_logical_pages)
        << error_msg << " The number of logical pages mismatches.";
    TVM_FFI_ICHECK_EQ(tensor.shape()[2], 2) << error_msg << " The K/V axis mismatches.";
    TVM_FFI_ICHECK_EQ(tensor.shape()[3], num_kv_heads_)
        << error_msg << " The number of KV heads mismatches.";
    TVM_FFI_ICHECK_EQ(tensor.shape()[4], page_size_) << error_msg << " The page size mismatches.";
    TVM_FFI_ICHECK_EQ(tensor.shape()[5], qk_head_dim_)
        << error_msg << " The head dimension mismatches.";
  }

  void CheckExportPageGroupTensor(const ffi::Tensor& dst, int64_t num_logical_pages) const {
    TVM_FFI_ICHECK(dst.defined());
    CheckPageGroupTensor(dst, num_logical_pages, "ExportPageGroup");
  }

  void CheckImportPageGroupTensor(const ffi::Tensor& src, int64_t num_logical_pages) const {
    TVM_FFI_ICHECK(src.defined());
    CheckPageGroupTensor(src, num_logical_pages, "ImportPageGroup");
  }

  ffi::json::Array MakeAttnKindsMetadata() const {
    ffi::json::Array result;
    for (int64_t layer_id = layer_id_begin_offset_; layer_id < layer_id_end_offset_; ++layer_id) {
      AttnKind attn_kind = attn_kinds_[layer_id];
      result.push_back(ffi::String(AttnKindToString(attn_kind)));
    }
    return result;
  }

  ffi::json::Object MakeLayoutMetadata() const {
    namespace json = tvm::ffi::json;
    json::Object metadata;
    metadata.Set("format_version", kPagedKVCacheCheckpointFormatVersion);
    metadata.Set("cache_type", ffi::String(kPagedKVCacheCheckpointRuntime));
    metadata.Set("page_size", page_size_);
    metadata.Set("num_layers", num_layers_);
    metadata.Set("layer_begin", layer_id_begin_offset_);
    metadata.Set("layer_end", layer_id_end_offset_);
    metadata.Set("num_qo_heads", num_qo_heads_);
    metadata.Set("num_kv_heads", num_kv_heads_);
    metadata.Set("qk_head_dim", qk_head_dim_);
    metadata.Set("v_head_dim", v_head_dim_);
    metadata.Set("reserved_num_seqs", reserved_num_seqs_);
    metadata.Set("num_total_pages", num_total_pages_);
    metadata.Set("prefill_chunk_size", prefill_chunk_size_);
    metadata.Set("dtype", ffi::DLDataTypeToString(kv_dtype_));
    metadata.Set("attn_kinds", MakeAttnKindsMetadata());
    metadata.Set("rope_mode", ffi::String(RoPEModeToString(rope_mode_)));
    metadata.Set("rotary_scale", rotary_scale_);
    metadata.Set("rotary_theta", rotary_theta_);
    metadata.Set("has_rope_ext_factors", rope_ext_factors_.has_value());
    metadata.Set("support_sliding_window", support_sliding_window_);
    metadata.Set("support_layer_sliding_window", support_layer_sliding_window_);
    metadata.Set("page_group_layout",
                 ffi::String("1,num_logical_pages,2,num_kv_heads,page_size,qk_head_dim"));
    return metadata;
  }

  std::string GetLayoutDescriptor() const {
    std::ostringstream os;
    os.imbue(std::locale::classic());
    os << std::setprecision(std::numeric_limits<double>::max_digits10);
    os << "format_version=" << kPagedKVCacheCheckpointFormatVersion << ";";
    os << "cache_type=" << kPagedKVCacheCheckpointRuntime << ";";
    os << "page_size=" << page_size_ << ";";
    os << "num_layers=" << num_layers_ << ";";
    os << "layer_begin=" << layer_id_begin_offset_ << ";";
    os << "layer_end=" << layer_id_end_offset_ << ";";
    os << "num_qo_heads=" << num_qo_heads_ << ";";
    os << "num_kv_heads=" << num_kv_heads_ << ";";
    os << "qk_head_dim=" << qk_head_dim_ << ";";
    os << "v_head_dim=" << v_head_dim_ << ";";
    os << "dtype=" << std::string(ffi::DLDataTypeToString(kv_dtype_)) << ";";
    os << "rope_mode=" << RoPEModeToString(rope_mode_) << ";";
    os << "rotary_scale=" << rotary_scale_ << ";";
    os << "rotary_theta=" << rotary_theta_ << ";";
    os << "has_rope_ext_factors=" << rope_ext_factors_.has_value() << ";";
    os << "support_sliding_window=" << support_sliding_window_ << ";";
    os << "support_layer_sliding_window=" << support_layer_sliding_window_ << ";";
    os << "page_group_layout=1,num_logical_pages,2,num_kv_heads,page_size,qk_head_dim;";
    os << "attn_kinds=";
    for (int64_t layer_id = layer_id_begin_offset_; layer_id < layer_id_end_offset_; ++layer_id) {
      if (layer_id != layer_id_begin_offset_) {
        os << ",";
      }
      os << AttnKindToString(attn_kinds_[layer_id]);
    }
    return os.str();
  }

  ffi::json::Array MakeLogicalPageMetadata(const Sequence& seq) const {
    namespace json = tvm::ffi::json;
    json::Array pages;
    int64_t num_logical_pages = GetExpectedNumLogicalPages(seq.seq_length);
    for (int64_t logical_page_index = 0; logical_page_index < num_logical_pages;
         ++logical_page_index) {
      int64_t page_start = logical_page_index * page_size_;
      int64_t page_length = std::min<int64_t>(page_size_, seq.seq_length - page_start);
      TVM_FFI_ICHECK_GT(page_length, 0);
      json::Object page_json;
      page_json.Set("logical_page_index", logical_page_index);
      page_json.Set("start_pos", page_start);
      page_json.Set("length", page_length);
      pages.push_back(page_json);
    }
    return pages;
  }

  int64_t GetNumLogicalPages(const Sequence& seq) const {
    int64_t num_pages = 0;
    for (int32_t block_id : seq.GetBlockTrace(global_block_pool_)) {
      num_pages += global_block_pool_[block_id].page_ids.size();
    }
    TVM_FFI_ICHECK_EQ(num_pages, GetExpectedNumLogicalPages(seq.seq_length))
        << "PagedAttentionKVCache checkpointing found a page table that does not match the "
           "sequence length.";
    return num_pages;
  }

  ffi::json::Array MakePageGroupMetadata(const Sequence& seq) const {
    namespace json = tvm::ffi::json;
    json::Array groups;
    int64_t num_logical_pages = GetNumLogicalPages(seq);
    for (int64_t local_layer = 0; local_layer < num_layers_; ++local_layer) {
      json::Object group;
      group.Set("group_index", local_layer);
      group.Set("layer_begin", layer_id_begin_offset_ + local_layer);
      group.Set("layer_end", layer_id_begin_offset_ + local_layer + 1);
      group.Set("num_logical_pages", num_logical_pages);
      group.Set("dtype", ffi::DLDataTypeToString(kv_dtype_));
      group.Set("shape",
                json::Array{1, num_logical_pages, 2, num_kv_heads_, page_size_, qk_head_dim_});
      group.Set("nbytes", GetCheckpointGroupNBytes(num_logical_pages));
      groups.push_back(group);
    }
    return groups;
  }

  int64_t GetCheckpointBytesPerScalar() const {
    return (static_cast<int64_t>(kv_dtype_.bits) * kv_dtype_.lanes + 7) / 8;
  }

  int64_t GetCheckpointBytesPerPage() const {
    return 2 * num_kv_heads_ * page_size_ * qk_head_dim_ * GetCheckpointBytesPerScalar();
  }

  int64_t GetCheckpointGroupNBytes(int64_t num_logical_pages) const {
    return num_logical_pages * GetCheckpointBytesPerPage();
  }

  std::vector<int32_t> GetCheckpointPageIds(const Sequence& seq, const char* operation) const {
    std::vector<int32_t> page_ids;
    page_ids.reserve(GetExpectedNumLogicalPages(seq.seq_length));
    for (int32_t block_id : seq.GetBlockTrace(global_block_pool_)) {
      const Block& block = global_block_pool_[block_id];
      for (int32_t page_id : block.page_ids) {
        TVM_FFI_ICHECK_GE(page_id, 0)
            << "PagedAttentionKVCache " << operation << " found invalid page id " << page_id << ".";
        TVM_FFI_ICHECK_LT(page_id, num_total_pages_)
            << "PagedAttentionKVCache " << operation << " found out-of-range page id " << page_id
            << ".";
        page_ids.push_back(page_id);
      }
    }
    TVM_FFI_ICHECK_EQ(static_cast<int64_t>(page_ids.size()),
                      GetExpectedNumLogicalPages(seq.seq_length))
        << "PagedAttentionKVCache " << operation
        << " found a page table that does not match the sequence length.";
    return page_ids;
  }

  void CopyCheckpointPageRun(const ffi::Tensor& src, int64_t src_page_index, const ffi::Tensor& dst,
                             int64_t dst_page_index, int64_t run_length) const {
    TVM_FFI_ICHECK(src.defined());
    TVM_FFI_ICHECK(dst.defined());
    TVM_FFI_ICHECK_GT(run_length, 0);
    int64_t bytes_per_page = GetCheckpointBytesPerPage();
    ffi::Tensor src_pages = tvm::runtime::CreateView(
        src, {run_length, 2, num_kv_heads_, page_size_, qk_head_dim_}, src.dtype(),
        static_cast<uint64_t>(src_page_index * bytes_per_page));
    ffi::Tensor dst_pages = tvm::runtime::CreateView(
        dst, {run_length, 2, num_kv_heads_, page_size_, qk_head_dim_}, dst.dtype(),
        static_cast<uint64_t>(dst_page_index * bytes_per_page));
    DLTensor dst_pages_view = *dst_pages.GetDLTensorPtr();
    tvm::runtime::CopyFromTo(src_pages.GetDLTensorPtr(), &dst_pages_view, compute_stream_);
  }

  void ExportPartialCheckpointPage(const ffi::Tensor& src, int64_t src_page_index,
                                   const ffi::Tensor& dst, int64_t dst_page_index,
                                   int64_t valid_length) {
    TVM_FFI_ICHECK(src.defined());
    TVM_FFI_ICHECK(dst.defined());
    TVM_FFI_ICHECK_GT(valid_length, 0);
    TVM_FFI_ICHECK_LT(valid_length, page_size_);

    int64_t bytes_per_scalar = GetCheckpointBytesPerScalar();
    int64_t bytes_per_page = GetCheckpointBytesPerPage();
    if (!checkpoint_zero_page_host_.has_value()) {
      checkpoint_zero_page_host_ = tvm::runtime::Empty({2, num_kv_heads_, page_size_, qk_head_dim_},
                                                       kv_dtype_, GetPreferredHostDevice(device_));
      std::vector<uint8_t> zero_data(bytes_per_page, 0);
      tvm::runtime::CopyFromBytes(checkpoint_zero_page_host_.value().GetDLTensorPtr(),
                                  zero_data.data(), zero_data.size());
    }

    // Initialize the full destination page with one transfer, then overwrite
    // only the valid prefixes. This makes every unused slot deterministic.
    ffi::Tensor dst_page =
        tvm::runtime::CreateView(dst, {2, num_kv_heads_, page_size_, qk_head_dim_}, dst.dtype(),
                                 static_cast<uint64_t>(dst_page_index * bytes_per_page));
    DLTensor dst_page_view = *dst_page.GetDLTensorPtr();
    tvm::runtime::CopyFromTo(checkpoint_zero_page_host_.value().GetDLTensorPtr(), &dst_page_view,
                             compute_stream_);

    for (int64_t kv_index = 0; kv_index < 2; ++kv_index) {
      for (int64_t head_index = 0; head_index < num_kv_heads_; ++head_index) {
        int64_t head_offset = (kv_index * num_kv_heads_ + head_index) * page_size_ * qk_head_dim_;
        uint64_t src_offset =
            static_cast<uint64_t>(src_page_index * bytes_per_page + head_offset * bytes_per_scalar);
        uint64_t dst_offset =
            static_cast<uint64_t>(dst_page_index * bytes_per_page + head_offset * bytes_per_scalar);
        ffi::Tensor src_valid =
            tvm::runtime::CreateView(src, {valid_length, qk_head_dim_}, src.dtype(), src_offset);
        ffi::Tensor dst_valid =
            tvm::runtime::CreateView(dst, {valid_length, qk_head_dim_}, dst.dtype(), dst_offset);

        DLTensor dst_valid_view = *dst_valid.GetDLTensorPtr();
        tvm::runtime::CopyFromTo(src_valid.GetDLTensorPtr(), &dst_valid_view, compute_stream_);
      }
    }
  }

  /*! \brief Get a new free page and return its id. */
  int32_t GetFreePage() {
    // Find a page from the free page pools.
    TVM_FFI_ICHECK(!free_page_ids_.empty()) << "The KV cache is full. No page can be allocated.";
    int32_t page_id = free_page_ids_.back();
    free_page_ids_.pop_back();
    return page_id;
  }

  /*! \brief Get a new free block and return its index. */
  int32_t GetFreeBlock() {
    if (!free_block_idx_.empty()) {
      int32_t block_idx = free_block_idx_.back();
      free_block_idx_.pop_back();
      global_block_pool_[block_idx].Reset();
      TVM_FFI_ICHECK_EQ(global_block_pool_[block_idx].index, block_idx);
      return block_idx;
    }

    int32_t block_idx = global_block_pool_.size();
    global_block_pool_.push_back(Block(block_idx));
    return block_idx;
  }

  void ConstructTokenTreeMask(const std::vector<Sequence*>& sequences,
                              const ffi::Shape& token_tree_parent_ptr,
                              const std::vector<std::vector<int32_t>>& block_ids_on_depths,
                              const std::vector<std::vector<int32_t>>& trailing_blocks) {
    // Check whether the token tree of a sequence should be handled at the current depth.
    auto check_for_sequence = [&](int seq_i, int depth) -> bool {
      if (!append_before_attn_) {
        return true;
      }
      // Check if the last block of the sequence is on the current depth.
      if (block_ids_on_depths[depth][seq_i] == sequences[seq_i]->last_block_idx ||
          (depth + 1 == kPagedKVCacheMaxBlockDepth && !trailing_blocks[seq_i].empty())) {
        return true;
      }
      return false;
    };
    for (int d = 0; d < num_depths_; ++d) {
      // We check if the token tree deteriorates to a chain,
      // because chain cases can have simplified attention work flow.
      TVM_FFI_ICHECK_LT(d, tree_attn_mask_host_.size());
      TVM_FFI_ICHECK_LT(d, tree_attn_mn_indptr_host_.size());
      HostMemoryVector& tree_attn_mn_indptr = tree_attn_mn_indptr_host_[d];
      HostMemoryVector& tree_attn_mask = tree_attn_mask_host_[d];

      std::vector<bool> seq_in_current_depth(cur_batch_size_, false);

      tree_attn_mn_indptr.clear();
      tree_attn_mask.clear();
      std::fill(is_chain_on_depths_.begin(), is_chain_on_depths_.end(), true);

      bool is_chain = true;
      // - Construct the mn indptr array, which is the indptr of the mask size of each sequence.
      tree_attn_mn_indptr.push_back(0);
      TVM_FFI_ICHECK_EQ(sequences.size(), cur_batch_size_);
      TVM_FFI_ICHECK_EQ(cur_append_lengths_.size(), cur_batch_size_);
      int64_t token_tree_parent_ptr_offset = 0;
      for (int i = 0; i < cur_batch_size_; ++i) {
        int64_t append_length = cur_append_lengths_[i];
        seq_in_current_depth[i] = check_for_sequence(i, d);
        if (!seq_in_current_depth[i]) {
          tree_attn_mn_indptr.push_back(tree_attn_mn_indptr.back());
          token_tree_parent_ptr_offset += append_length;  // Skip the token tree of this sequence.
          continue;
        }
        // Update the token tree parent pointers.
        TVM_FFI_ICHECK_LE(sequences[i]->token_tree_parent_ptr.size(),
                          global_block_pool_[sequences[i]->last_block_idx].seq_length)
            << "The token tree size is larger than the sequence length of the last block.";
        std::copy(token_tree_parent_ptr.begin() + token_tree_parent_ptr_offset,
                  token_tree_parent_ptr.begin() + token_tree_parent_ptr_offset + append_length,
                  std::back_inserter(sequences[i]->token_tree_parent_ptr));
        token_tree_parent_ptr_offset += append_length;

        TVM_FFI_ICHECK_LE(sequences[i]->token_tree_parent_ptr.size(), kTreeAttnMaxTreeSize)
            << "The tree size is " << append_length << " which exceeds the maximum tree size limit "
            << kTreeAttnMaxTreeSize;
        tree_attn_mn_indptr.push_back(tree_attn_mn_indptr.back() +
                                      sequences[i]->token_tree_parent_ptr.size());
      }
      TVM_FFI_ICHECK_EQ(token_tree_parent_ptr.size(), token_tree_parent_ptr_offset)
          << "Invalid token tree size. The sum of \"append_lengths\" is "
          << token_tree_parent_ptr_offset << " while there are " << token_tree_parent_ptr.size()
          << " elements in \"token_tree_parent_ptr\".";

      // - Construct the mask of each sequence.
      for (int i = 0; i < cur_batch_size_; ++i) {
        if (!seq_in_current_depth[i]) {
          continue;
        }
        int64_t tree_size = sequences[i]->token_tree_parent_ptr.size();
        std::vector<std::vector<int32_t>> mask;
        std::vector<int32_t> depth;
        mask.reserve(tree_size);
        depth.reserve(tree_size);
        sequences[i]->is_chain = true;
        sequences[i]->accepted_indices_committed = false;
        std::unordered_map<int, std::vector<int>> tree_parent_to_children;
        std::vector<int> tree_roots;
        for (int n = 0; n < tree_size; ++n) {
          TVM_FFI_ICHECK_LT(sequences[i]->token_tree_parent_ptr[n], n)
              << "Invalid token tree. The parent of node " << n << " in tree " << i << " is "
              << sequences[i]->token_tree_parent_ptr[n] << ", which is not smaller than " << n;
          TVM_FFI_ICHECK_GE(sequences[i]->token_tree_parent_ptr[n], -1)
              << "Invalid token tree. The parent of node " << n << " in tree " << i << " is "
              << sequences[i]->token_tree_parent_ptr[n];
          if (sequences[i]->token_tree_parent_ptr[n] != n - 1) {
            // The parent of the current node is not the last node.
            // Therefore the tree is not a chain.
            sequences[i]->is_chain = false;
            is_chain = false;
          }
          tree_parent_to_children[sequences[i]->token_tree_parent_ptr[n]].push_back(n);

          if (sequences[i]->token_tree_parent_ptr[n] != -1) {
            depth.push_back(depth[sequences[i]->token_tree_parent_ptr[n]] + 1);
          } else {
            depth.push_back(0);
            tree_roots.push_back(n);
          }
        }
        std::vector<std::pair<int, int>> tree_order(tree_size);
        int order = 0;
        std::function<int(int)> tree_dfs = [&order, &tree_order, &tree_parent_to_children,
                                            &tree_dfs](int node) -> int {
          tree_order[node].first = order++;
          int upper_bound = tree_order[node].first + 1;
          for (int child : tree_parent_to_children[node]) {
            upper_bound = std::max(upper_bound, tree_dfs(child));
          }
          tree_order[node].second = upper_bound;
          return upper_bound;
        };
        for (auto root : tree_roots) {
          tree_dfs(root);
        }
        for (int n = 0; n < tree_size; ++n) {
          tree_attn_mask.push_back(tree_order[n].first);
          tree_attn_mask.push_back(tree_order[n].second);
        }
        sequences[i]->token_tree_node_depths = std::move(depth);
      }

      is_chain_on_depths_[d] = is_chain;

      if (!append_before_attn_) {
        break;
      }
    }
  }

  /*!
   * \brief Slide the KV cache window of the given sequence when
   * it has sliding window enabled.
   * \param seq The sequence to be slidden when
   */
  void SlideWindowForSequence(Sequence* seq) {
    // - No action when the sequence is not enabled for sliding window.
    if (seq->sliding_window_size == -1 || !support_sliding_window_) {
      return;
    }
    // - No action when the sequence length does not exceed the window size.
    if (seq->seq_length <= seq->sliding_window_size) {
      return;
    }

    int32_t length_to_slide = seq->seq_length - seq->sliding_window_size;
    // - Get the last block of the sequence.
    Block& block = global_block_pool_[seq->last_block_idx];

    // - If the attention sink exists and the last block has no previous
    // sink length, it means this is the first time we slide the sequence,
    // and thus we set the sink length of the last block, the index of the
    // first sliding page, and starting offset in first sliding page.
    if (seq->last_block_attn_sink_size > 0 && block.sink_length == 0) {
      TVM_FFI_ICHECK_EQ(block.sliding_window_offset, 0);
      block.sink_length = seq->last_block_attn_sink_size;
      block.sliding_window_offset = seq->last_block_attn_sink_size;
    }

    // - The sink pages cannot be slidden.
    int32_t num_sink_pages = (block.sink_length + page_size_ - 1) / page_size_;

    // - Compute the first sliding page index and in-page sliding window
    // start offset in the first sliding page after sliding.
    int32_t page_idx_after_sliding = (block.sliding_window_offset + length_to_slide) / page_size_;
    int32_t page_start_offset_after_sliding =
        (block.sliding_window_offset + length_to_slide) % page_size_;

    // - Free the pages that are fully slidden.
    while (page_idx_after_sliding > num_sink_pages) {
      if (block.page_ids[num_sink_pages] != kPagedKVCacheTempPageId) {
        free_page_ids_.push_back(block.page_ids[num_sink_pages]);
      }
      block.page_ids.erase(block.page_ids.begin() + num_sink_pages);
      --page_idx_after_sliding;
    }
    // - The first sliding page after sliding is either the last sink page,
    // or the page next to the last sink page.
    TVM_FFI_ICHECK(page_idx_after_sliding == num_sink_pages - 1 ||
                   page_idx_after_sliding == num_sink_pages);

    // - Update the length of the sequence and the block.
    seq->seq_length = seq->sliding_window_size;
    block.seq_length -= length_to_slide;
    block.sliding_window_offset =
        page_idx_after_sliding * page_size_ + page_start_offset_after_sliding;
    TVM_FFI_ICHECK_GE(block.seq_length, block.sink_length);
    TVM_FFI_ICHECK_GE(block.sliding_window_offset, block.sink_length);
    TVM_FFI_ICHECK_EQ(
        (block.sliding_window_offset + (block.seq_length - block.sink_length) + page_size_ - 1) /
            page_size_,
        block.page_ids.size());
  }

  /*!
   * \brief Reserve extra append length in the last block of the given
   * sequence, as preparation of the incoming KV cache append.
   * New pages will be allocated to the block until the total
   * capacity can cover the current sequence length (before reservation)
   * plus the required append length.
   * \param block_idx The index of the block to process.
   * \param append_length The extra append length to reserve for the block.
   * \note We apply sliding window in this function.
   */
  void ReserveAppendLengthInSeq(Sequence* seq, int64_t append_length) {
    int32_t block_idx = seq->last_block_idx;
    Block& block = global_block_pool_[block_idx];
    TVM_FFI_ICHECK_GT(append_length, 0) << "Append with length 0 is not allowed.";
    TVM_FFI_ICHECK_EQ(block.external_ref_cnt, 1)
        << "The block is " << block.external_ref_cnt - 1
        << "-time referenced by other blocks, thus cannot accept new KV values.";

    // ==================== Reserve ====================
    // The reservation is based on the current sequence length.
    // If "current sequence + append length" does not exceed the
    // current capacity (number of pages * page size), no action is taken.
    int64_t cur_npage = block.page_ids.size();
    int64_t tgt_npage = (block.seq_length - block.sink_length + block.sliding_window_offset +
                         append_length + page_size_ - 1) /
                        page_size_;
    for (int64_t page_idx = cur_npage; page_idx < tgt_npage; ++page_idx) {
      // When sliding window is enabled for the seq, we can "borrow temporary pages (-1)",
      // since the pages need to be slidden out might not have been released.
      if (free_page_ids_.empty() && seq->sliding_window_size != -1 && support_sliding_window_) {
        block.page_ids.push_back(kPagedKVCacheTempPageId);
      } else {
        block.page_ids.push_back(GetFreePage());
      }
    }
    block.seq_length += append_length;

    // ==================== Slide ====================
    // Slide the sequences so that the pages exceed the sliding window are released.
    SlideWindowForSequence(seq);
    if (support_sliding_window_) {
      for (int i = 0; i < static_cast<int>(block.page_ids.size()); ++i) {
        if (block.page_ids[i] == kPagedKVCacheTempPageId) {
          // Re-allocate the temporary pages after sliding window release.
          block.page_ids[i] = GetFreePage();
        }
      }
    }

    dirty_aux_data_device_ = true;
  }

  /*! \brief Check whether BeginForward for kernels is needed. */
  bool NeedKernelBeginForward() {
    std::vector<AttnBackendFunc*> funcs = {f_attention_prefill_.get(),
                                           f_attention_prefill_ragged_.get(),
                                           f_attention_decode_.get(),
                                           f_attention_prefill_sliding_window_.get(),
                                           f_attention_decode_sliding_window_.get(),
                                           f_attention_prefill_with_tree_mask_.get(),
                                           f_attention_prefill_with_tree_mask_paged_kv_.get(),
                                           f_mla_prefill_.get()};
    for (AttnBackendFunc* func : funcs) {
      if (func != nullptr && func->backend_kind == AttnBackendKind::kFlashInfer) {
        return true;
      }
    }
    return false;
  }

  /*! \brief Invoke the "begin forward" functions of underlying kernels. */
  void KernelBeginForward() {
    if (!NeedKernelBeginForward()) {
      return;
    }

    auto it_layer_begin = attn_kinds_.begin() + layer_id_begin_offset_;
    auto it_layer_end = attn_kinds_.begin() + layer_id_end_offset_;
    if (std::find(it_layer_begin, it_layer_end, AttnKind::kMHA) != it_layer_end) {
      MHAKernelBeginForward();
    }
    if (std::find(it_layer_begin, it_layer_end, AttnKind::kMLA) != it_layer_end) {
      MLAKernelBeginForward();
    }
  }

  /*! \brief KernelBeginForward for multi-head attention. */
  void MHAKernelBeginForward() {
    if (!append_before_attn_) {
      if (is_chain_on_depths_[0] && f_attention_prefill_ragged_ != nullptr &&
          f_attention_prefill_ragged_->backend_kind == AttnBackendKind::kFlashInfer) {
        f_attention_prefill_ragged_->BeginForward(
            temp_float_attn_workspace_.value(), temp_int_attn_workspace_[0],
            temp_int_pinned_attn_workspace_[0], &cur_append_lengths_indptr_host_,
            &cur_append_lengths_indptr_host_, cur_batch_size_,
            cur_append_lengths_indptr_host_.back(), num_qo_heads_, num_kv_heads_, qk_head_dim_,
            v_head_dim_, /*causal=*/true, copy_stream_);
      }
    }
    for (int d = 0; d < num_depths_; ++d) {
      if (page_indices_on_depths_view_[d].value().shape()[0] == 0) {
        continue;
      }
      TVM_FFI_ICHECK(!support_sliding_window_ || !support_layer_sliding_window_)
          << "Kernel BeginForward doesn't support sliding window.";
      if (use_decode_kernel_[d]) {
        if (f_attention_decode_ != nullptr &&
            f_attention_decode_->backend_kind == AttnBackendKind::kFlashInfer) {
          f_attention_decode_->BeginForward(
              d, temp_float_attn_workspace_.value(), temp_int_attn_workspace_[d + 1],
              temp_int_pinned_attn_workspace_[d + 1], &page_indptr_on_depths_host_[d],
              cur_batch_size_, page_size_, num_qo_heads_, num_kv_heads_, qk_head_dim_, v_head_dim_,
              rope_mode_, kv_dtype_, kv_dtype_, copy_stream_);
        }
      } else {
        if (f_attention_prefill_ != nullptr &&
            f_attention_prefill_->backend_kind == AttnBackendKind::kFlashInfer) {
          f_attention_prefill_->BeginForward(
              d, temp_float_attn_workspace_.value(), temp_int_attn_workspace_[d + 1],
              temp_int_pinned_attn_workspace_[d + 1], &qo_indptr_on_depths_host_[d],
              &page_indptr_on_depths_host_[d], &last_page_len_on_depths_host_[d],
              static_cast<int64_t>(qo_indptr_on_depths_host_[d].size()) - 1,
              cur_append_lengths_indptr_host_.back(), page_size_, num_qo_heads_, num_kv_heads_,
              qk_head_dim_, v_head_dim_, /*causal=*/false, copy_stream_);
        }
      }
    }
  }

  /*! \brief KernelBeginForward for multi-head latent attention. */
  void MLAKernelBeginForward() {
    if (!append_before_attn_) {
      if (is_chain_on_depths_[0]) {
        if (f_attention_prefill_ragged_ != nullptr &&
            f_attention_prefill_ragged_->backend_kind == AttnBackendKind::kFlashInfer) {
          f_attention_prefill_ragged_->BeginForward(
              temp_float_attn_workspace_.value(), temp_int_attn_workspace_[0],
              temp_int_pinned_attn_workspace_[0], &cur_append_lengths_indptr_host_,
              &cur_append_lengths_indptr_host_, cur_batch_size_,
              cur_append_lengths_indptr_host_.back(), num_qo_heads_, num_kv_heads_, qk_head_dim_,
              v_head_dim_, /*causal=*/true, copy_stream_);
        }
      }
    }
    for (int d = 0; d < num_depths_; ++d) {
      if (page_indices_on_depths_view_[d].value().shape()[0] == 0) {
        continue;
      }
      TVM_FFI_ICHECK(!support_sliding_window_)
          << "Kernel BeginForward doesn't support sliding window.";
      if (f_mla_prefill_ != nullptr &&
          f_mla_prefill_->backend_kind == AttnBackendKind::kFlashInfer) {
        f_mla_prefill_->BeginForward(
            d, temp_float_attn_workspace_.value(), temp_int_attn_workspace_[d + 1],
            temp_int_pinned_attn_workspace_[d + 1], &qo_indptr_on_depths_host_[d],
            &page_indptr_on_depths_host_[d], &last_page_len_on_depths_host_[d],
            static_cast<int64_t>(qo_indptr_on_depths_host_[d].size()) - 1,
            cur_append_lengths_indptr_host_.back(), page_size_, num_qo_heads_, num_kv_heads_,
            qk_head_dim_, v_head_dim_, /*causal=*/false, copy_stream_);
      }
    }
  }

  /*!
   * \brief Compute attention for between the input q data and the
   * input k/v data and the k/v data in cache on the given layer.
   */
  void AttentionInternal(int64_t layer_id, ffi::Tensor q_data, ffi::Tensor k_data,
                         ffi::Tensor v_data, ffi::Tensor output, double sm_scale) {
    TVM_FFI_ICHECK(q_data.defined());
    TVM_FFI_ICHECK(k_data.defined());
    TVM_FFI_ICHECK(v_data.defined());
    TVM_FFI_ICHECK(output.defined());
    int64_t local_layer_id = layer_id - layer_id_begin_offset_;
    TVM_FFI_ICHECK_GE(local_layer_id, 0);
    TVM_FFI_ICHECK_LT(local_layer_id, num_layers_);

    bool is_first_kernel = true;
    if (!append_before_attn_) {
      // The first part of attention, which only involves the q and the newly appended k/v.
      is_first_kernel = false;
      MHASelfAttnInternal(q_data, k_data, v_data, output, merged_attn_lse_view_.value(), sm_scale);
    }
    bool self_attn_computed = !is_first_kernel;
    bool cross_attn_computed = MHACrossAttnInternal(
        local_layer_id, q_data, output, merged_attn_lse_view_.value(), sm_scale, is_first_kernel,
        /*causal=*/!append_before_attn_ &&
            attn_kinds_[local_layer_id + layer_id_begin_offset_] == AttnKind::kMHASliding);
    TVM_FFI_ICHECK(self_attn_computed || cross_attn_computed)
        << "Both self-attention and cross-attention are not computed.";
  }

  void MHASelfAttnInternal(ffi::Tensor q_data, ffi::Tensor k_data, ffi::Tensor v_data,
                           ffi::Tensor o_data, ffi::Tensor lse_data, double sm_scale) {
    TVM_FFI_ICHECK(q_data.defined());
    TVM_FFI_ICHECK(k_data.defined());
    TVM_FFI_ICHECK(v_data.defined());
    TVM_FFI_ICHECK(o_data.defined());
    TVM_FFI_ICHECK(lse_data.defined());
    if (is_chain_on_depths_[0]) {
      // If the batch does not form a tree, use raggedness prefill kernel.
      TVM_FFI_ICHECK_NOTNULL(f_attention_prefill_ragged_);
      f_attention_prefill_ragged_->MHA(
          q_data, k_data, v_data, cur_append_length_indptr_view_.value(),
          cur_append_length_indptr_view_.value(), q_rope_position_map_view_.value(),
          k_ragged_rope_pos_offset_view_.value(), /*causal=*/true, rope_mode_, rotary_scale_,
          rotary_theta_, sm_scale, o_data, lse_data, compute_stream_);
    } else {
      // The batch requires tree attention.
      TVM_FFI_ICHECK(f_attention_prefill_with_tree_mask_ != nullptr)
          << "Function \"f_attention_prefill_with_tree_mask_\" is not defined.";
      TVM_FFI_ICHECK(tree_attn_mask_view_[0].has_value());
      TVM_FFI_ICHECK(tree_attn_mn_indptr_view_[0].has_value());
      f_attention_prefill_with_tree_mask_->MHA(
          q_data, k_data, v_data, cur_append_length_indptr_view_.value(),
          cur_append_length_indptr_view_.value(), q_rope_position_map_view_.value(),
          tree_attn_mn_indptr_view_[0].value(), tree_attn_mask_view_[0].value(), rope_mode_,
          rotary_scale_, rotary_theta_, sm_scale, o_data, lse_data, compute_stream_);
    }
  }

  void MLASelfAttnInternal(ffi::Tensor q_data, ffi::Tensor k_data, ffi::Tensor v_data,
                           ffi::Tensor o_data, ffi::Tensor lse_data, double sm_scale) {
    TVM_FFI_ICHECK(q_data.defined());
    TVM_FFI_ICHECK(k_data.defined());
    TVM_FFI_ICHECK(v_data.defined());
    TVM_FFI_ICHECK(o_data.defined());
    TVM_FFI_ICHECK(lse_data.defined());
    TVM_FFI_ICHECK(is_chain_on_depths_[0]) << "Tree attn not able for MLA for now.";
    // If the batch does not form a tree, use raggedness prefill kernel.
    TVM_FFI_ICHECK_NOTNULL(f_attention_prefill_ragged_);
    f_attention_prefill_ragged_->MHA(
        q_data, k_data, v_data, cur_append_length_indptr_view_.value(),
        cur_append_length_indptr_view_.value(), q_rope_position_map_view_.value(),
        k_ragged_rope_pos_offset_view_.value(), /*causal=*/true, RoPEMode::kNone, rotary_scale_,
        rotary_theta_, sm_scale, o_data, lse_data, compute_stream_);
  }

  /*! \brief Compute cross-attention for MHA. Return if there is effective computation. */
  bool MHACrossAttnInternal(int64_t local_layer_id, ffi::Tensor q_data, ffi::Tensor o_data,
                            ffi::Tensor lse_data, double sm_scale, bool is_first_kernel,
                            bool causal) {
    TVM_FFI_ICHECK(q_data.defined());
    TVM_FFI_ICHECK(o_data.defined());
    TVM_FFI_ICHECK(lse_data.defined());
    std::unique_ptr<PagedPrefillFunc>& f_prefill =
        (!support_sliding_window_ &&
         attn_kinds_[local_layer_id + layer_id_begin_offset_] != AttnKind::kMHASliding)
            ? f_attention_prefill_
            : f_attention_prefill_sliding_window_;
    std::unique_ptr<PagedDecodeFunc>& f_decode =
        (!support_sliding_window_ &&
         attn_kinds_[local_layer_id + layer_id_begin_offset_] != AttnKind::kMHASliding)
            ? f_attention_decode_
            : f_attention_decode_sliding_window_;
    TVM_FFI_ICHECK_GE(num_depths_, 1)
        << "The number of effective depths must be greater or equal to 1.";

    bool cross_attn_computed = false;
    for (int d = 0; d < num_depths_; ++d) {
      if (page_indices_on_depths_view_[d].value().shape()[0] == 0) {
        continue;
      }
      ffi::Tensor attn_output = is_first_kernel ? o_data : temp_attn_output_view_.value();
      ffi::Tensor attn_lse = is_first_kernel ? lse_data : temp_attn_lse_view_.value();
      // If layer is sliding window, use sliding window index pointer/indices
      bool layer_sliding =
          attn_kinds_[local_layer_id + layer_id_begin_offset_] == AttnKind::kMHASliding;
      ffi::Tensor page_indptr = layer_sliding
                                    ? page_indptr_sliding_window_on_depths_view_[d].value()
                                    : page_indptr_on_depths_view_[d].value();
      ffi::Tensor page_indices = layer_sliding
                                     ? page_indices_sliding_window_on_depths_view_[d].value()
                                     : page_indices_on_depths_view_[d].value();
      ffi::Tensor length_info = layer_sliding
                                    ? layer_sliding_window_length_info_on_depths_view_[d].value()
                                    : length_info_on_depths_view_[d].value();
      ffi::Tensor k_rope_pos = layer_sliding ? k_rope_pos_offset_sliding_window_view_[d].value()
                                             : k_rope_pos_offset_view_[d].value();
      double rotary_theta = layer_sliding ? 10000 : rotary_theta_;
      double rotary_scale = layer_sliding ? 1 : rotary_scale_;

      if (append_before_attn_ && !is_chain_on_depths_[d]) {
        TVM_FFI_ICHECK_NOTNULL(f_attention_prefill_with_tree_mask_paged_kv_);
        f_attention_prefill_with_tree_mask_paged_kv_->MHA(
            q_data, qo_indptr_on_depths_view_[d].value(), pages_[local_layer_id], page_indptr,
            page_indices, length_info, k_rope_pos, q_rope_position_map_view_.value(),
            tree_attn_mn_indptr_view_[d].value(), tree_attn_mask_view_[d].value(), rope_mode_,
            rotary_scale, rotary_theta, sm_scale, attn_output, attn_lse, compute_stream_);
      } else if (use_decode_kernel_[d]) {
        // Use decode kernel for depth d
        TVM_FFI_ICHECK_NOTNULL(f_decode);
        f_decode->MHA(d, q_data, pages_[local_layer_id], page_indptr, page_indices, length_info,
                      k_rope_pos, q_rope_position_map_view_.value(), rope_mode_, rotary_scale,
                      rotary_theta, sm_scale, attn_output, attn_lse, compute_stream_);
      } else {
        // Use prefill kernel for depth d
        TVM_FFI_ICHECK_NOTNULL(f_prefill);
        f_prefill->MHA(d, q_data, qo_indptr_on_depths_view_[d].value(), pages_[local_layer_id],
                       page_indptr, page_indices, length_info, q_rope_position_map_view_.value(),
                       k_rope_pos, causal,
                       /*rotary_mode=*/rope_mode_, rotary_scale, rotary_theta, sm_scale,
                       attn_output, attn_lse, compute_stream_);
      }

      if (!is_first_kernel) {
        f_merge_inplace_[0](o_data, lse_data, temp_attn_output_view_.value(),
                            temp_attn_lse_view_.value());
      } else {
        is_first_kernel = false;
      }
      cross_attn_computed = true;
    }
    return cross_attn_computed;
  }

  /*! \brief Compute cross-attention for MLA. Return if there is effective computation. */
  bool MLACrossAttnInternal(int64_t local_layer_id, ffi::Tensor q_data, ffi::Tensor o_data,
                            ffi::Tensor lse_data, double sm_scale) {
    TVM_FFI_ICHECK(q_data.defined());
    TVM_FFI_ICHECK(o_data.defined());
    TVM_FFI_ICHECK(lse_data.defined());
    TVM_FFI_ICHECK_GE(num_depths_, 1)
        << "The number of effective depths must be greater or equal to 1.";

    bool is_first_kernel = true;
    for (int d = 0; d < num_depths_; ++d) {
      if (page_indices_on_depths_view_[d].value().shape()[0] == 0) {
        continue;
      }
      ffi::Tensor attn_output = is_first_kernel ? o_data : temp_attn_output_view_.value();
      ffi::Tensor attn_lse = is_first_kernel ? lse_data : temp_attn_lse_view_.value();
      TVM_FFI_ICHECK(is_chain_on_depths_[d]) << "Tree attn not able for MLA for now.";
      TVM_FFI_ICHECK_NOTNULL(f_mla_prefill_);
      f_mla_prefill_->MLA(d, q_data, qo_indptr_on_depths_view_[d].value(), pages_[local_layer_id],
                          page_indptr_on_depths_view_[d].value(),
                          page_indices_on_depths_view_[d].value(),
                          length_info_on_depths_view_[d].value(), /*causal=*/false, sm_scale,
                          attn_output, attn_lse, compute_stream_);

      if (!is_first_kernel) {
        f_merge_inplace_[0](o_data, lse_data, temp_attn_output_view_.value(),
                            temp_attn_lse_view_.value());
      } else {
        is_first_kernel = false;
      }
    }
    return !is_first_kernel;
  }

  /*! \brief Synchronize the copy stream and the compute stream. */
  void ComputeStreamWaitForCopyStream() {
    if (!dirty_aux_data_device_) {
      // If the auxiliary data is already synced, return and no need to sync again.
      return;
    }
    // - Sync Tensors to GPU.
    SyncAuxArrayToDevice();
    // FlashInfer's plan kernels no longer take an explicit stream argument; they
    // run on the device's *current* stream. Make the copy stream current around
    // the plan so its workspace writes happen on the copy stream -- matching the
    // aux-array copies above and the copy->compute synchronization below (this
    // preserves the prior behavior where copy_stream_ was passed explicitly).
    bool plan_on_copy_stream = copy_stream_ != nullptr && copy_stream_ != compute_stream_;
    if (plan_on_copy_stream) {
      DeviceAPI::Get(device_)->SetStream(device_, copy_stream_);
    }
    KernelBeginForward();
    if (plan_on_copy_stream) {
      DeviceAPI::Get(device_)->SetStream(device_, compute_stream_);
    }
    // - Clear the dirty flag.
    dirty_aux_data_device_ = false;
    // - If there is no particular copy stream, no action is needed.
    if (copy_stream_ == nullptr) {
      return;
    }
    // - Sync two streams.
    DeviceAPI::Get(device_)->SyncStreamFromTo(device_, copy_stream_, compute_stream_);
  }

  /*!
   * \brief Synchronize auxiliary arrays to device.
   * \note This method resets the dirty flag to false, and needs to be
   * invoked before running attention computation on device.
   */
  void SyncAuxArrayToDevice() {
    support::NVTXScopedRange range("SyncAuxArrayToDevice");
    TVM_FFI_ICHECK(dtype_aux_.bits == 32 && dtype_aux_.code == kDLInt);
    int64_t total_append_length = 0;
    int num_sequences = cur_append_lengths_.size();
    cur_append_lengths_indptr_host_.clear();
    cur_append_lengths_indptr_host_.push_back(0);
    for (int i = 0; i < num_sequences; ++i) {
      cur_append_lengths_indptr_host_.push_back(cur_append_lengths_indptr_host_.back() +
                                                cur_append_lengths_[i]);
    }
    total_append_length = cur_append_lengths_indptr_host_.back();
    TVM_FFI_ICHECK_EQ(total_append_length, append_position_map_host_.size());
    TVM_FFI_ICHECK_EQ(total_append_length, kv_transfer_remote_position_map_host_.size());
    TVM_FFI_ICHECK_EQ(total_append_length, kv_transfer_recver_id_host_.size());

    // - Reset the copy.
    aux_data_manager_->ResetAttnAuxDataCopy();

    // 1. q_rope_position_map
    // q_rope_position_map has to be synced first so that it has a 0 byte offset
    TVM_FFI_ICHECK_EQ(q_rope_position_map_host_.size(), total_append_length);
    q_rope_position_map_view_ = aux_data_manager_->CopyQRoPEPosMapAsync(&q_rope_position_map_host_);
    // 2. qo_indptr_on_depths
    for (int d = 0; d < num_depths_; ++d) {
      qo_indptr_on_depths_view_[d] =
          aux_data_manager_->CopyQOIndptrOnDepthAsync(&qo_indptr_on_depths_host_[d], d);
    }
    // 3. page_indptr_on_depths
    for (int d = 0; d < num_depths_; ++d) {
      TVM_FFI_ICHECK_EQ(page_indptr_on_depths_host_[d].size(), qo_indptr_on_depths_host_[d].size());
      page_indptr_on_depths_view_[d] =
          aux_data_manager_->CopyPageIndptrOnDepthAsync(&page_indptr_on_depths_host_[d], d);
    }
    // 4. page_indices_on_depths
    for (int d = 0; d < num_depths_; ++d) {
      TVM_FFI_ICHECK_EQ(page_indices_on_depths_host_[d].size(),
                        page_indptr_on_depths_host_[d].back());
      page_indices_on_depths_view_[d] =
          aux_data_manager_->CopyPageIndicesOnDepthAsync(&page_indices_on_depths_host_[d], d);
    }

    // If per layer sliding window exists, must copy additional vectors
    if (support_layer_sliding_window_) {
      // 5. page_indptr_sliding_window_on_depths
      for (int d = 0; d < num_depths_; ++d) {
        TVM_FFI_ICHECK_EQ(page_indptr_sliding_window_on_depths_host_[d].size(),
                          qo_indptr_on_depths_host_[d].size());
        page_indptr_sliding_window_on_depths_view_[d] =
            aux_data_manager_->CopyPageIndptrOnDepthAsync(
                &page_indptr_sliding_window_on_depths_host_[d], d, /*layer_sliding_window=*/true);
      }
      // 6. page_indices_sliding_window_on_depths
      for (int d = 0; d < num_depths_; ++d) {
        TVM_FFI_ICHECK_EQ(page_indices_sliding_window_on_depths_host_[d].size(),
                          page_indptr_sliding_window_on_depths_host_[d].back());
        page_indices_sliding_window_on_depths_view_[d] =
            aux_data_manager_->CopyPageIndicesOnDepthAsync(
                &page_indices_sliding_window_on_depths_host_[d], d, /*layer_sliding_window=*/true);
      }
    }
    // 7. length_info_on_depths
    // last_page_len_on_depths_host_;
    // sliding_window_offset_on_depths_host_;
    // sink_size_on_depths_host_;
    for (int d = 0; d < num_depths_; ++d) {
      int num_seq_on_layer = static_cast<int>(qo_indptr_on_depths_host_[d].size()) - 1;
      TVM_FFI_ICHECK_EQ(last_page_len_on_depths_host_[d].size(), num_seq_on_layer);
      TVM_FFI_ICHECK_EQ(sliding_window_offset_on_depths_host_[d].size(), num_seq_on_layer);
      TVM_FFI_ICHECK_EQ(sink_size_on_depths_host_[d].size(), num_seq_on_layer);
      if (!support_sliding_window_) {
        // Sliding window is not enabled, so we first copy "last_page_len".
        length_info_on_depths_view_[d] =
            aux_data_manager_->CopyLastPageLenOnDepthAsync(&last_page_len_on_depths_host_[d], d);
      } else {
        // Sliding window is enabled,
        length_info_on_depths_view_[d] = aux_data_manager_->CopyLengthInfoOnDepthAsync(
            &last_page_len_on_depths_host_[d], &sliding_window_offset_on_depths_host_[d],
            &sink_size_on_depths_host_[d], d);
      }

      if (support_layer_sliding_window_) {
        layer_sliding_window_length_info_on_depths_view_[d] =
            aux_data_manager_->CopyLengthInfoOnDepthAsync(&last_page_len_on_depths_host_[d],
                                                          &sliding_window_offset_on_depths_host_[d],
                                                          &sink_size_on_depths_host_[d], d,
                                                          /*layer_sliding_window=*/true);
      }
    }
    // 6. k_rope_pos_offset_on_depths
    for (int d = 0; d < num_depths_; ++d) {
      TVM_FFI_ICHECK_EQ(k_rope_pos_offset_on_depths_host_[d].size() + 1,
                        qo_indptr_on_depths_host_[d].size());
      k_rope_pos_offset_view_[d] = aux_data_manager_->CopyKRoPEPosOffsetOnDepthAsync(
          &k_rope_pos_offset_on_depths_host_[d], d);
      if (support_layer_sliding_window_) {
        TVM_FFI_ICHECK_EQ(k_rope_pos_offset_sliding_window_on_depths_host_[d].size() + 1,
                          qo_indptr_on_depths_host_[d].size());
        k_rope_pos_offset_sliding_window_view_[d] =
            aux_data_manager_->CopyKRoPEPosOffsetOnDepthAsync(
                &k_rope_pos_offset_sliding_window_on_depths_host_[d], d);
      }
    }
    // 7. cur_append_lengths_indptr
    cur_append_length_indptr_view_ =
        aux_data_manager_->CopyCurAppendLengthIndptrAsync(&cur_append_lengths_indptr_host_);
    // 8. k_ragged_rope_pos_offset
    TVM_FFI_ICHECK_EQ(k_ragged_rope_pos_offset_host_.size(), num_sequences);
    k_ragged_rope_pos_offset_view_ =
        aux_data_manager_->CopyKRaggedRoPEPosOffsetAsync(&k_ragged_rope_pos_offset_host_);
    // 9. append_position_map
    append_position_map_view_ =
        aux_data_manager_->CopyAppendPositionMapAsync(&append_position_map_host_);
    // 10. kv_transfer_remote_position_map
    kv_transfer_remote_position_map_view_ = aux_data_manager_->CopyKVTransferRemotePositionMapAsync(
        &kv_transfer_remote_position_map_host_);
    // 11. kv_transfer_recver_id
    kv_transfer_recver_id_view_ =
        aux_data_manager_->CopyKVTransferRecverIDAsync(&kv_transfer_recver_id_host_);

    // 12. kv_transfer_page_to_page_local_position_map
    kv_transfer_page_to_page_local_position_map_view_ =
        aux_data_manager_->CopyKVTransferPage2PageLocalPositionMapAsync(
            &kv_transfer_page_to_page_local_position_map_host_);
    // 13. kv_transfer_page_to_page_remote_position_map
    kv_transfer_page_to_page_remote_position_map_view_ =
        aux_data_manager_->CopyKVTransferPage2PageRemotePositionMapAsync(
            &kv_transfer_page_to_page_remote_position_map_host_);
    // 14. kv_transfer_page_to_page_recver_id
    kv_transfer_page_to_page_recver_id_view_ =
        aux_data_manager_->CopyKVTransferPage2PageRecverIDAsync(
            &kv_transfer_page_to_page_recver_id_host_);
    // 15. tree_attn_mask and tree_attn_mn_indptr
    for (int d = 0; d < num_depths_; ++d) {
      if (!is_chain_on_depths_[d]) {
        tree_attn_mask_view_[d] =
            aux_data_manager_->CopyTreeAttnMaskOnDepthAsync(&tree_attn_mask_host_[d], d);
        tree_attn_mn_indptr_view_[d] =
            aux_data_manager_->CopyTreeAttnMNIndptrOnDepthAsync(&tree_attn_mn_indptr_host_[d], d);
      }
    }
    // 16. Create view for temporary arrays for attention computation.
    temp_attn_output_view_ = tvm::runtime::CreateView(
        temp_attn_output_device_, {total_append_length, num_qo_heads_, v_head_dim_},
        temp_attn_output_device_.dtype());
    temp_attn_lse_view_ = tvm::runtime::CreateView(
        temp_attn_lse_device_, {total_append_length, num_qo_heads_}, temp_attn_lse_device_.dtype());
    merged_attn_lse_view_ =
        tvm::runtime::CreateView(merged_attn_lse_device_, {total_append_length, num_qo_heads_},
                                 merged_attn_lse_device_.dtype());

    // - Commit the copy.
    aux_data_manager_->CommitAttnAuxDataCopy();
    // - Reset the dirty flag to false.
    dirty_aux_data_device_ = false;
  }
};  // namespace vm

//-------------------------------------------------
//  Register runtime functions
//-------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      "vm.builtin.paged_attention_kv_cache_create", [](ffi::PackedArgs args, ffi::Any* rv) {
        // Todo: cuda graph arg
        TVM_FFI_ICHECK(args.size() == 28 || args.size() == 29)
            << "Invalid number of KV cache constructor args: " << args.size();
        ffi::Shape cache_config = args[0].cast<ffi::Shape>();
        ffi::Shape layer_indptr_tuple = args[1].cast<ffi::Shape>();
        int num_groups = 1;
        int group_id = 0;
        if (DiscoWorker* disco_worker = ThreadLocalDiscoWorker::Get()->worker) {
          // In the Disco worker thread
          num_groups = disco_worker->num_groups;
          group_id = disco_worker->worker_id / (disco_worker->num_workers / num_groups);
        }
        TVM_FFI_ICHECK_EQ(layer_indptr_tuple.size(), num_groups + 1);
        int64_t num_layers = layer_indptr_tuple[group_id + 1] - layer_indptr_tuple[group_id];
        int64_t layer_id_begin_offset = layer_indptr_tuple[group_id];
        int64_t layer_id_end_offset = layer_indptr_tuple[group_id + 1];
        int64_t num_qo_heads = args[2].cast<int64_t>();
        int64_t num_kv_heads = args[3].cast<int64_t>();
        int64_t qk_head_dim = args[4].cast<int64_t>();
        int64_t v_head_dim = args[5].cast<int64_t>();
        ffi::Shape attn_kinds = args[6].cast<ffi::Shape>();
        bool enable_kv_transfer = args[7].cast<bool>();
        int rope_mode = args[8].cast<int>();
        double rotary_scale = args[9].cast<double>();
        double rotary_theta = args[10].cast<double>();
        ffi::Optional<ffi::Tensor> rope_ext_factors = std::nullopt;  // args[11]
        ffi::Tensor init = args[12].cast<ffi::Tensor>();
        TVM_FFI_ICHECK(init.defined());
        ffi::Optional<ffi::Function> f_transpose_append_mha = std::nullopt;  // args[13]
        ffi::Optional<ffi::Function> f_transpose_append_mla = std::nullopt;  // args[14]
        std::unique_ptr<RaggedPrefillFunc> f_attention_prefill_ragged =
            ConvertRaggedPrefillFunc(args[15].cast<ffi::Array<ffi::Any>>(), AttnKind::kMHA);
        std::unique_ptr<PagedPrefillFunc> f_attention_prefill =
            ConvertPagedPrefillFunc(args[16].cast<ffi::Array<ffi::Any>>(), AttnKind::kMHA);
        std::unique_ptr<PagedDecodeFunc> f_attention_decode =
            ConvertPagedDecodeFunc(args[17].cast<ffi::Array<ffi::Any>>(), AttnKind::kMHA);
        std::unique_ptr<PagedPrefillFunc> f_attention_prefill_sliding_window =
            ConvertPagedPrefillFunc(args[18].cast<ffi::Array<ffi::Any>>(), AttnKind::kMHA);
        std::unique_ptr<PagedDecodeFunc> f_attention_decode_sliding_window =
            ConvertPagedDecodeFunc(args[19].cast<ffi::Array<ffi::Any>>(), AttnKind::kMHA);
        std::unique_ptr<PagedPrefillTreeMaskFunc> f_attention_prefill_with_tree_mask_paged_kv =
            ConvertPagedPrefillTreeMaskFunc(args[20].cast<ffi::Array<ffi::Any>>(), AttnKind::kMHA);
        std::unique_ptr<RaggedPrefillTreeMaskFunc> f_attention_prefill_with_tree_mask =
            ConvertRaggedPrefillTreeMaskFunc(args[21].cast<ffi::Array<ffi::Any>>(), AttnKind::kMHA);
        std::unique_ptr<PagedPrefillFunc> f_mla_prefill =
            ConvertPagedPrefillFunc(args[22].cast<ffi::Array<ffi::Any>>(), AttnKind::kMLA);
        ffi::Array<ffi::Function> f_merge_inplace = args[23].cast<ffi::Array<ffi::Function>>();
        ffi::Function f_split_rotary = args[24].cast<ffi::Function>();
        ffi::Function f_copy_single_page = args[25].cast<ffi::Function>();
        ffi::Function f_debug_get_kv = args[26].cast<ffi::Function>();
        ffi::Function f_compact_copy = args[27].cast<ffi::Function>();

        if (auto opt_nd = args[11].as<ffi::Tensor>(); opt_nd && opt_nd->defined()) {
          rope_ext_factors = opt_nd.value();
        }
        auto f_convert_optional_packed_func = [&args](int arg_idx) -> ffi::Optional<ffi::Function> {
          if (auto opt_func = args[arg_idx].as<ffi::Function>()) {
            return opt_func.value();
          }
          return std::nullopt;
        };
        f_transpose_append_mha = f_convert_optional_packed_func(13);
        f_transpose_append_mla = f_convert_optional_packed_func(14);
        TVM_FFI_ICHECK(!f_merge_inplace.empty()) << "Merge inplace function is not defined.";

        std::vector<AttnKind> attn_kinds_vec;
        attn_kinds_vec.reserve(attn_kinds.size());
        for (int64_t attn_kind : attn_kinds) {
          attn_kinds_vec.push_back(static_cast<AttnKind>(attn_kind));
        }

        TVM_FFI_ICHECK(cache_config.size() == 5 || cache_config.size() == 6)
            << "KV cache config must contain five legacy fields and an optional per-layer "
               "sliding window size.";
        int64_t reserved_num_seqs = cache_config[0];
        int64_t total_token_capacity = cache_config[1];
        int64_t prefill_chunk_size = cache_config[2];
        int64_t page_size = cache_config[3];
        bool support_sliding_window = cache_config[4];
        int64_t layer_sliding_window_size = cache_config.size() == 6 ? cache_config[5] : 1024;
        int64_t num_total_pages = (total_token_capacity + page_size - 1) / page_size + 1;
        if (support_sliding_window) {
          // When sliding window is enabled, each sequence may use two more pages at most.
          num_total_pages += reserved_num_seqs * 2;
        }
        // NOTE: We will remove this legacy construction after finishing the transition phase.
        // Some `ffi::Function()` here are placeholders that will be filled.
        ffi::ObjectPtr<PagedAttentionKVCacheObj> n = ffi::make_object<PagedAttentionKVCacheObj>(
            page_size, num_layers, layer_id_begin_offset, layer_id_end_offset, num_qo_heads,
            num_kv_heads, qk_head_dim, v_head_dim, attn_kinds_vec, reserved_num_seqs,
            num_total_pages, prefill_chunk_size, support_sliding_window, layer_sliding_window_size,
            RoPEMode(rope_mode), rotary_scale, rotary_theta, std::move(rope_ext_factors),
            enable_kv_transfer,           //
            init.dtype(), init.device(),  //
            std::move(f_transpose_append_mha), std::move(f_transpose_append_mla),
            std::move(f_compact_copy), std::move(f_attention_prefill_ragged),
            std::move(f_attention_prefill), std::move(f_attention_decode),
            std::move(f_attention_prefill_sliding_window),
            std::move(f_attention_decode_sliding_window),
            std::move(f_attention_prefill_with_tree_mask_paged_kv),  //
            std::move(f_attention_prefill_with_tree_mask),           //
            std::move(f_mla_prefill), std::move(f_merge_inplace), std::move(f_split_rotary),
            std::move(f_copy_single_page), std::move(f_debug_get_kv));
        *rv = ffi::GetRef<AttentionKVCache>(n.get());
      });
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
