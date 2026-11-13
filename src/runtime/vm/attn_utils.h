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
 * \file src/runtime/vm/attn_utils.h
 * \brief Data structure and utilities for KV cache.
 */

#ifndef TVM_RUNTIME_VM_ATTN_UTILS_H_
#define TVM_RUNTIME_VM_ATTN_UTILS_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "nvtx.h"

namespace tvm {
namespace runtime {
namespace vm {

/*!
 * \brief Get the preferred host device from the input device.
 * - For CUDA and ROCm, CUDAHost and ROCMHost will be returned for pinned memory,
 * since pinned memory reduces copy overhead.
 * - For other devices, CPU is returned as a fallback.
 */
inline Device GetPreferredHostDevice(Device device) {
  if (device.device_type == DLDeviceType::kDLCUDA) {
    return Device{DLDeviceType::kDLCUDAHost, 0};
  } else if (device.device_type == DLDeviceType::kDLROCM) {
    return Device{DLDeviceType::kDLROCMHost, 0};
  } else {
    // Fallback to CPU.
    return Device{DLDeviceType::kDLCPU, 0};
  }
}

/*!
 * \brief The maximum allowed block depth (a.k.a. number of common
 * prefixes) in paged KV cache.
 */
constexpr const int kPagedKVCacheMaxBlockDepth = 2;
/*! \brief The maximum tree size of a single sequence in tree attention. */
constexpr const int kTreeAttnMaxTreeSize = 256;
/*! \brief The 1MB workspace size for integer attention auxiliary data. */
constexpr const int kIntAttnWorkspaceByte = 8 * 1024 * 1024;
/*! \brief The 128MB workspace size for floating-point attention auxiliary data. */
constexpr const int kFloatAttnWorkspaceByte = 768 * 1024 * 1024;
/*! \brief The id of the temporary logical page, which is useful for sliding window. */
constexpr const int kPagedKVCacheTempPageId = -1;

/*!
 * \brief The supported attention kinds in PagedKVCache.
 * "MHA" means multi-head attention, multi-query attention and grouped query attention in general.
 * "MLA" means multi-head latent attention.
 * "LinearAttn" means linear attention.
 */
enum class AttnKind : int {
  kMHA = 0,
  kMLA = 1,
  kLinearAttn = 2,
  kMHASliding = 3,
};

/*! \brief Given the attention kind and other metadata, return the one-layer KV cache shape. */
inline ffi::Shape GetKVCacheShape(AttnKind attn_kind, int64_t num_total_pages, int num_sequence,
                                  int64_t num_kv_heads, int64_t page_size, int64_t qk_head_dim,
                                  int64_t v_head_dim) {
  if (attn_kind == AttnKind::kMHA || attn_kind == AttnKind::kMHASliding) {
    // Ignore v_head_dim since multi-head attention requires K/V to have the same head dim.
    return {num_total_pages, 2, num_kv_heads, page_size, qk_head_dim};
  } else if (attn_kind == AttnKind::kMLA) {
    return {num_total_pages, page_size, qk_head_dim};
  } else if (attn_kind == AttnKind::kLinearAttn) {
    return {num_sequence, num_kv_heads, qk_head_dim, v_head_dim};
  }
  TVM_FFI_ICHECK(false);
  return ffi::Shape();
}

/*!
 * \brief The block structure in paged KV cache with common prefix support.
 * Each block contains a list of pages for cached KV data.
 * If a block has `n` pages, the first `n - 1` pages must be
 * full, and only the last page can be partially filled.
 *
 * To support common prefix, each sequence in KV cache is represented
 * as one or more blocks, where the common prefix is a standalone
 * block among.
 *
 * Each block has a parent block when it uses a prefix.
 */
struct Block {
  /*!
   * \brief The ids of the pages in the block.
   * Each page can only be used by a unique block (in other
   * words, different blocks do not share pages).
   */
  std::vector<int32_t> page_ids;
  /*! \brief The total sequence length in the block. */
  int32_t seq_length = 0;
  /*!
   * \brief The start position in sequence of this block.
   * This is the absolute position in the sequence for RoPE computation.
   */
  int32_t start_pos = 0;
  /*!
   * \brief The current attention sink length of the block.
   * It means the **first** sink size elements will be pinned
   * in the KV cache even when sliding window is enabled.
   */
  int32_t sink_length = 0;
  /*!
   * \brief The start offset of the sliding window in the block.
   * It is always 0 when sliding window attn is not enabled.
   */
  int32_t sliding_window_offset = 0;

  /*! \brief The global index of the block. */
  const int32_t index;
  /*!
   * \brief The global index of the parent block of this block, or -1
   * if the block does not have a parent. */
  int32_t parent_idx = -1;
  /*!
   * \brief The external reference counter of the block.
   * When a block is externally referred by some block,
   * we do not allow appending new KV values to this block.
   */
  int external_ref_cnt = 0;

  explicit Block(int32_t index) : index(index) {}

  /*! \brief Reset the block data. */
  void Reset() {
    page_ids.clear();
    seq_length = 0;
    start_pos = 0;
    sink_length = 0;
    sliding_window_offset = 0;
    parent_idx = -1;
    external_ref_cnt = 0;
  }
};

struct KVTransferMetadata {
  int64_t start = std::numeric_limits<int64_t>::max();
  std::vector<int64_t> remote_position_map;
  int32_t recver_pe_offset = -1;
  std::vector<int64_t> local_position_map;
};

/*!
 * \brief The sequence structure in paged KV cache with common prefix support.
 * Each sequence contains one or more blocks to support common prefix.
 */
struct Sequence {
  /*!
   * \brief The global index of the last block of the sequence.
   * We only store the last block, since all the blocks can be
   * tracked with the `parent` field of Block.
   */
  int32_t last_block_idx;
  /*!
   * \brief The total sequence length of the sequence.
   * It is the sum of lengths of all its blocks.
   */
  int32_t seq_length = 0;
  /*!
   * \brief The sliding window size of the sequence, or -1 if sliding window is not enabled.
   * When a sequence is enabled for sliding window, it can no longer be forked.
   */
  int sliding_window_size = -1;
  /*!
   * \brief The attention sink size of the last block of the sequence.
   * The **first** sink size elements of the last block will be pinned
   * in the KV cache even when sliding window is enabled.
   */
  int last_block_attn_sink_size = 0;

  /*! \brief Whether the current appended tokens form a chain (not a tree). */
  bool is_chain = true;
  /*! \brief The token tree parent pointer array of the current appended tokens. */
  std::vector<int32_t> token_tree_parent_ptr;
  /*! \brief The depth of each node in the token tree. */
  std::vector<int32_t> token_tree_node_depths;
  /*! \brief The metadata of kv transfer*/
  KVTransferMetadata kv_transfer_metadata;
  /*!
   * \brief A boolean denoting whether the accepted token tree indices of
   * this sequence are committed
   */
  bool accepted_indices_committed = true;

  explicit Sequence(std::vector<Block>* global_block_pool, int32_t last_block_idx) {
    ++global_block_pool->at(last_block_idx).external_ref_cnt;
    this->last_block_idx = last_block_idx;
    int32_t block_ptr = last_block_idx;
    // Go through each block in the sequence, sum up the length.
    while (true) {
      const Block& block = global_block_pool->at(block_ptr);
      this->seq_length += block.seq_length;
      if (block.parent_idx == -1) {
        break;
      }
      block_ptr = block.parent_idx;
    }
  }

  std::vector<int32_t> GetBlockTrace(const std::vector<Block>& global_block_pool) const {
    std::vector<int32_t> trace;
    // Get the trace from the last block of the sequence to the root block.
    int32_t block_ptr = last_block_idx;
    while (block_ptr != -1) {
      trace.push_back(block_ptr);
      block_ptr = global_block_pool[block_ptr].parent_idx;
    }
    // Reverse the trace so that it starts from the root block.
    std::reverse(trace.begin(), trace.end());
    return trace;
  }
};

/*!
 * \brief For the given list of sequences, check the block trace of
 * each sequence, and return the blocks ids used by the sequences
 * on each depth. And if the depth is larger than the kPagedKVCacheMaxBlockDepth,
 * the exceeding blocks will concatenate and output separately.
 * More precisely, the inner returned vector contains the block ids
 * used by the sequences on a certain depth (or "-1" if a sequence
 * has fewer depth). The outer returned vector contains the inner
 * vectors from the lowest depth to the highest depth.
 */
inline std::pair<std::vector<std::vector<int32_t>>, std::vector<std::vector<int32_t>>>
GetBlockIdsOnDepth(const std::vector<Sequence*>& sequences,
                   const std::vector<Block>& global_block_pool, int64_t batch_size) {
  // - Get the trace of each sequence.
  int64_t num_depths = 0;
  std::vector<std::vector<int32_t>> seq_block_traces;
  std::vector<std::vector<int32_t>> trailing_block_traces;
  seq_block_traces.reserve(batch_size);
  trailing_block_traces.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    std::vector<int32_t> trace = sequences[i]->GetBlockTrace(global_block_pool);
    if (static_cast<int>(trace.size()) <= kPagedKVCacheMaxBlockDepth) {
      seq_block_traces.push_back(std::vector<int32_t>(trace.begin(), trace.end()));
      trailing_block_traces.push_back({});
      num_depths = std::max(num_depths, static_cast<int64_t>(trace.size()));
    } else {
      seq_block_traces.push_back(
          std::vector<int32_t>(trace.begin(), trace.begin() + kPagedKVCacheMaxBlockDepth));
      trailing_block_traces.push_back(
          std::vector<int32_t>(trace.begin() + kPagedKVCacheMaxBlockDepth, trace.end()));
      num_depths = std::max(num_depths, static_cast<int64_t>(kPagedKVCacheMaxBlockDepth));
    }
  }

  // "Transpose" the traces, yielding the block ids used on each depth.
  std::vector<std::vector<int32_t>> block_ids_on_depths;
  block_ids_on_depths.reserve(num_depths);
  for (int d = 0; d < num_depths; ++d) {
    std::vector<int32_t> block_ids;
    block_ids.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      block_ids.push_back(d < static_cast<int>(seq_block_traces[i].size()) ? seq_block_traces[i][d]
                                                                           : -1);
    }
    block_ids_on_depths.push_back(std::move(block_ids));
  }
  return {block_ids_on_depths, trailing_block_traces};
}

/*!
 * \brief This function considers an optimization which coalesces
 * adjacent decode attention computations into a single prefill
 * attention computation if the adjacent decodes attend to the same
 * k/v values under certain conditions.
 * If it decides to coalesce on a certain depth, we need to know
 * the prefill length after coalescing. This function returns
 * - a vector of block ids together with the prefill/decode lengths
 * that attend to the blocks.
 * - a boolean indicating whether to use decode kernel on for the
 * input blocks.
 */
inline std::pair<std::vector<std::pair<int32_t, int32_t>>, bool> GetChunkedBlockIds(
    const std::vector<int32_t>& block_ids, bool enable_coalesce, const ffi::Shape& append_lengths,
    const std::vector<Block>& global_block_pool, bool is_decode_request) {
  std::vector<std::pair<int32_t, int32_t>> uncoalesced_block_ids;
  std::vector<std::pair<int32_t, int32_t>> coalesced_block_ids;

  // Gather the number of pages before/after coalescing respectively.
  int cur_block_id = block_ids[0];
  int chunk_append_length = append_lengths[0];
  int page_counter_coalesced = 0;
  int page_counter_uncoalesced =
      block_ids[0] != -1 ? global_block_pool[block_ids[0]].page_ids.size() : 0;
  for (int i = 1; i < static_cast<int>(block_ids.size()); ++i) {
    if (block_ids[i] != -1) {
      page_counter_uncoalesced += global_block_pool[block_ids[i]].page_ids.size();
    }
    uncoalesced_block_ids.emplace_back(block_ids[i - 1], append_lengths[i - 1]);
    if (block_ids[i] == cur_block_id) {
      chunk_append_length += append_lengths[i];
    } else {
      coalesced_block_ids.emplace_back(cur_block_id, chunk_append_length);
      if (cur_block_id != -1) {
        page_counter_coalesced += global_block_pool[cur_block_id].page_ids.size();
      }
      cur_block_id = block_ids[i];
      chunk_append_length = append_lengths[i];
    }
  }
  uncoalesced_block_ids.emplace_back(block_ids.back(), append_lengths.back());
  coalesced_block_ids.emplace_back(cur_block_id, chunk_append_length);
  if (cur_block_id != -1) {
    page_counter_coalesced += global_block_pool[cur_block_id].page_ids.size();
  }
  double coalesce_ratio =
      page_counter_coalesced > 0 ? 1.0 * page_counter_uncoalesced / page_counter_coalesced : 0.0;
  // Do not coalesce and use batch decode kernel when coalesce ratio is small.
  bool use_decode_kernel = is_decode_request && coalesce_ratio < 32;
  return {use_decode_kernel || !enable_coalesce ? uncoalesced_block_ids : coalesced_block_ids,
          use_decode_kernel};
}

/*!
 * \brief The rotary embedding mode adopted by the paged KV cache
 * when computing attention.
 * "None" means RoPE is never applied to q and k.
 * "Normal" means RoPE is computed in a standalone kernel.
 * "Inline" means RoPE is computed on-the-fly in attention kernels.
 */
enum class RoPEMode : int {
  kNone = 0,
  kNormal = 1,
  kInline = 2,
};

/*!
 * \brief The class of host memory int32 vector in "std::vector" interface.
 * This vector allocates static memory on the specified host memory
 * at the time of construction.
 */
class HostMemoryVector {
 public:
  HostMemoryVector() = default;
  HostMemoryVector(const HostMemoryVector&) = delete;
  HostMemoryVector(HostMemoryVector&& other) = default;
  HostMemoryVector& operator=(const HostMemoryVector&) = delete;
  HostMemoryVector& operator=(HostMemoryVector&& other) = default;

  explicit HostMemoryVector(int64_t reserved_size, DLDataType dtype, Device device)
      : reserved_size_(reserved_size) {
    TVM_FFI_ICHECK((dtype == DLDataType{kDLInt, 32, 1}));
    data_ = tvm::runtime::Empty({reserved_size}, dtype, device);
  }

  void push_back(int32_t value) {
    TVM_FFI_ICHECK_LE(current_size_, reserved_size_);
    if (current_size_ == reserved_size_) {
      reserved_size_ *= 2;
      ffi::Tensor new_data =
          tvm::runtime::Empty({reserved_size_}, data_.value().dtype(), data_.value().device());
      std::memcpy(new_data.data_ptr(), data_.value().data_ptr(),
                  current_size_ * (((data_.value().dtype()).bits + 7) / 8));
      data_ = new_data;
    }
    static_cast<int32_t*>(data_.value().data_ptr())[current_size_++] = value;
  }

  void push_back_vec(const std::vector<int32_t>& values) {
    TVM_FFI_ICHECK_LE(current_size_, reserved_size_);
    int64_t num_new_elements = static_cast<int64_t>(values.size());
    if (current_size_ + num_new_elements > reserved_size_) {
      while (current_size_ + num_new_elements > reserved_size_) {
        reserved_size_ *= 2;
      }
      ffi::Tensor new_data =
          tvm::runtime::Empty({reserved_size_}, data_.value().dtype(), data_.value().device());
      std::memcpy(new_data.data_ptr(), data_.value().data_ptr(),
                  current_size_ * (((data_.value().dtype()).bits + 7) / 8));
      data_ = new_data;
    }
    std::memcpy(static_cast<int32_t*>(data_.value().data_ptr()) + current_size_, values.data(),
                num_new_elements * sizeof(int32_t));
    current_size_ += num_new_elements;
  }

  const int32_t& operator[](int64_t idx) const {
    TVM_FFI_ICHECK_GE(idx, 0) << "Index " << idx << " is negative.";
    TVM_FFI_ICHECK_LT(idx, current_size_) << "Index " << idx << " out of bounds " << current_size_;
    return static_cast<int32_t*>(data_.value().data_ptr())[idx];
  }

  int32_t back() const {
    TVM_FFI_ICHECK_GT(current_size_, 0) << "Vector is empty";
    return static_cast<int32_t*>(data_.value().data_ptr())[current_size_ - 1];
  }

  void fill(int32_t value) {
    std::fill(static_cast<int32_t*>(data_.value().data_ptr()),
              static_cast<int32_t*>(data_.value().data_ptr()) + current_size_, value);
  }

  void resize(size_t new_size) {
    TVM_FFI_ICHECK_LE(new_size, reserved_size_);
    current_size_ = new_size;
  }

  void set(int64_t idx, int32_t value) {
    TVM_FFI_ICHECK_GE(idx, 0) << "Index " << idx << " is negative.";
    TVM_FFI_ICHECK_LT(idx, current_size_) << "Index " << idx << " out of bounds " << current_size_;
    static_cast<int32_t*>(data_.value().data_ptr())[idx] = value;
  }

  size_t size() const { return static_cast<size_t>(current_size_); }

  int32_t* data() const { return static_cast<int32_t*>(data_.value().data_ptr()); }

  void clear() { current_size_ = 0; }

  /*! \brief Return the vector as an Tensor. */
  ffi::Tensor as_tensor() {
    return tvm::runtime::CreateView(data_.value(), {current_size_}, data_.value().dtype());
  }

  ffi::Shape as_int_tuple() const {
    std::vector<int64_t> values;
    values.reserve(current_size_);
    for (int i = 0; i < current_size_; ++i) {
      values.push_back(static_cast<int32_t*>(data_.value().data_ptr())[i]);
    }
    return ffi::Shape(values);
  }

 private:
  int64_t reserved_size_ = 0;
  int64_t current_size_ = 0;
  ffi::Optional<ffi::Tensor> data_;
};

/*!
 * \brief The paged attention auxiliary data manager class.
 * This class manages all the int32 auxiliary data on GPU device, such as
 * page table, position arrays, etc..
 *
 * The core functions of this class is `CopyXXXAsync` and `CommitAttnAuxDataCopy`.
 * `CopyXXXAsync` takes the input data on CPU host, and copy the input data
 * to GPU in an asynchronous way, and returns the Tensor view of the data
 * on GPU device.
 *
 * Being asynchronous here means the `CopyXXXAsync` function may not perform
 * data copy from CPU to GPU at the time of being called. Therefore, the
 * returned Tensor view may have wrong result, until `CommitAttnAuxDataCopy` is
 * explicitly invoked and the data copy stream is synchronized.
 *
 * We design this manager class in order to reduce the data copy overhead.
 */
class PagedKVCacheAuxDataManager {
 public:
  PagedKVCacheAuxDataManager(DLDataType dtype_aux, Device device, Device preferred_host_device,
                             TVMStreamHandle copy_stream)
      : dtype_aux_(dtype_aux),
        device_(device),
        preferred_host_device_(preferred_host_device),
        copy_stream_(copy_stream) {
    TVM_FFI_ICHECK((dtype_aux == DLDataType{kDLInt, 32, 1}));
  }

  virtual ~PagedKVCacheAuxDataManager() = default;
  /*! \brief Reset the attention auxiliary data status of copy manager. */
  virtual void ResetAttnAuxDataCopy() = 0;
  /*! \brief Copy the indptr array of append lengths after coalescing. (see GetChunkedBlockIds) */
  virtual ffi::Tensor CopyQOIndptrOnDepthAsync(HostMemoryVector* data, int depth) = 0;
  /*! \brief Copy the indptr array of page table. */
  virtual ffi::Tensor CopyPageIndptrOnDepthAsync(HostMemoryVector* data, int depth,
                                                 bool layer_sliding_window = false) = 0;
  /*! \brief Copy the indices array of page table. */
  virtual ffi::Tensor CopyPageIndicesOnDepthAsync(HostMemoryVector* data, int depth,
                                                  bool layer_sliding_window = false) = 0;
  /*! \brief Copy the array of KV slot number used in the last page of the seq. */
  virtual ffi::Tensor CopyLastPageLenOnDepthAsync(HostMemoryVector* data, int depth) = 0;
  /*!
   * \brief Copy the length information of the sequences.
   * Each Tensor is in shape `(3, n)`. "n" is the number of sequences.
   * For a sequence "i", location
   * - "(0, i)" is the number of KV slots used in the last page of the seq ("last_page_len"),
   * - "(1, i)" is the starting offset of the sliding window in the seq,
   * - "(2, i)" is the attn sink length of the sequence.
   * \note When sliding window is not enabled, only the
   * "last_page_len" (a.k.a., the first "n" elements) will be effectively used.
   */
  virtual ffi::Tensor CopyLengthInfoOnDepthAsync(HostMemoryVector* last_page_len,
                                                 HostMemoryVector* sliding_window_offset,
                                                 HostMemoryVector* sink_size, int depth,
                                                 bool layer_sliding_window = false) = 0;
  /*! \brief Copy the k position offset of applying RoPE for each sequence. */
  virtual ffi::Tensor CopyKRoPEPosOffsetOnDepthAsync(HostMemoryVector* data, int depth) = 0;
  /*!
   * \brief Copy the append length indptr array on device.
   * \note Since the Q/K/V data may have raggedness in terms of lengths,
   * we represent the append lengths in CSR format.
   */
  virtual ffi::Tensor CopyCurAppendLengthIndptrAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the k position offset of applying RoPE for each sequence. */
  virtual ffi::Tensor CopyKRaggedRoPEPosOffsetAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the q position mapping of applying RoPE for each sequence. */
  virtual ffi::Tensor CopyQRoPEPosMapAsync(HostMemoryVector* data) = 0;
  /*!
   * \brief Copy the corresponding position in global KV cache (pages)
   * for each position along the length dimension of K/V data when
   * appending new K/V data.
   */
  virtual ffi::Tensor CopyAppendPositionMapAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the remote position map for KV transfer. */
  virtual ffi::Tensor CopyKVTransferRemotePositionMapAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the receiver id for KV transfer. */
  virtual ffi::Tensor CopyKVTransferRecverIDAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the local position map for KV page-to-page transfer. */
  virtual ffi::Tensor CopyKVTransferPage2PageLocalPositionMapAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the remote position map for KV page-to-page transfer. */
  virtual ffi::Tensor CopyKVTransferPage2PageRemotePositionMapAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the receiver id for KV page-to-page transfer. */
  virtual ffi::Tensor CopyKVTransferPage2PageRecverIDAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the tree attention mask. */
  virtual ffi::Tensor CopyTreeAttnMaskOnDepthAsync(HostMemoryVector* data, int depth) = 0;
  /*! \brief Copy the mn indptr of the tree attention mask. */
  virtual ffi::Tensor CopyTreeAttnMNIndptrOnDepthAsync(HostMemoryVector* data, int depth) = 0;
  /*! \brief Commit all the attention auxiliary data copy operations since the last commit. */
  virtual void CommitAttnAuxDataCopy() = 0;

  /*! \brief Reset the compact KV auxiliary data status of copy manager. */
  virtual void ResetCompactKVAuxDataCopy() = 0;
  /*! \brief Copy the length indptr array of KV data copy for each sequence. */
  virtual ffi::Tensor CopyCommitLengthIndptrAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the src/dst position arrays for each sequence. */
  virtual ffi::Tensor CopyCommitSrcDstPosInPageTableAsync(HostMemoryVector* src_data,
                                                          HostMemoryVector* dst_data) = 0;
  /*! \brief Commit all the compact KV auxiliary data copy operations since the last commit. */
  virtual void CommitCompactKVAuxDataCopy() = 0;

 protected:
  /*! \brief The dtype of the auxiliary data. It is expected to be int32. */
  const DLDataType dtype_aux_;
  /*! \brief The device this PagedKVCache runs on. */
  const Device device_;
  /*! \brief The preferred host device. */
  const Device preferred_host_device_;
  /*! \brief The device stream for copying auxiliary data structure to GPU. */
  const TVMStreamHandle copy_stream_;
};

/*!
 * \brief The plain auxiliary data manager class.
 * It simply issues one host-to-device copy operation for each `CopyXXXAsync`.
 */
class PlainPagedKVCacheAuxDataManager : public PagedKVCacheAuxDataManager {
 public:
  explicit PlainPagedKVCacheAuxDataManager(int64_t reserved_num_seqs, int64_t num_total_pages,
                                           int64_t prefill_chunk_size, DLDataType dtype_aux,
                                           Device device, Device preferred_host_device,
                                           TVMStreamHandle copy_stream,
                                           bool layer_sliding_window = false)
      : PagedKVCacheAuxDataManager(dtype_aux, device, preferred_host_device, copy_stream),
        cur_append_length_indptr_device_(
            tvm::runtime::Empty({reserved_num_seqs + 1}, dtype_aux_, device)),
        k_ragged_rope_pos_offset_device_(
            tvm::runtime::Empty({reserved_num_seqs}, dtype_aux_, device)),
        q_rope_position_map_device_(tvm::runtime::Empty({prefill_chunk_size}, dtype_aux_, device)),
        append_position_map_device_(tvm::runtime::Empty({prefill_chunk_size}, dtype_aux_, device)),
        kv_transfer_remote_position_map_device(
            tvm::runtime::Empty({prefill_chunk_size}, dtype_aux_, device)),
        kv_transfer_recver_id_device(tvm::runtime::Empty({prefill_chunk_size}, dtype_aux_, device)),
        kv_transfer_page_to_page_local_position_map_device(
            tvm::runtime::Empty({prefill_chunk_size}, dtype_aux_, device)),
        kv_transfer_page_to_page_remote_position_map_device(
            kv_transfer_page_to_page_local_position_map_device),
        kv_transfer_page_to_page_recver_id_device(
            tvm::runtime::Empty({prefill_chunk_size}, dtype_aux_, device)),
        commit_copy_length_indptr_device_(
            tvm::runtime::Empty({reserved_num_seqs + 1}, dtype_aux_, device)),
        commit_copy_src_dst_pos_in_page_table_device_(tvm::runtime::Empty(
            {2, std::min(kTreeAttnMaxTreeSize * reserved_num_seqs, prefill_chunk_size)}, dtype_aux_,
            device)) {
    for (int d = 0; d < kPagedKVCacheMaxBlockDepth; ++d) {
      qo_indptr_on_depths_device_.push_back(
          tvm::runtime::Empty({reserved_num_seqs + 1}, dtype_aux_, device));
      page_indptr_on_depths_device_.push_back(
          tvm::runtime::Empty({reserved_num_seqs + 1}, dtype_aux_, device));
      page_indices_on_depths_device_.push_back(
          tvm::runtime::Empty({num_total_pages}, dtype_aux_, device));
      length_info_on_depths_device_.push_back(
          tvm::runtime::Empty({3, reserved_num_seqs}, dtype_aux_, device));
      if (layer_sliding_window) {
        page_indptr_sliding_on_depths_device_.push_back(
            tvm::runtime::Empty({reserved_num_seqs + 1}, dtype_aux_, device));
        page_indices_sliding_on_depths_device_.push_back(
            tvm::runtime::Empty({num_total_pages}, dtype_aux_, device));
        length_info_sliding_on_depths_device_.push_back(
            tvm::runtime::Empty({3, reserved_num_seqs}, dtype_aux_, device));
      }
      k_rope_pos_offset_on_depths_device_.push_back(
          tvm::runtime::Empty({reserved_num_seqs}, dtype_aux_, device));
      tree_attn_mask_device_.push_back(tvm::runtime::Empty(
          {kTreeAttnMaxTreeSize * kTreeAttnMaxTreeSize * reserved_num_seqs}, dtype_aux_, device));
      tree_attn_mn_indptr_device_.push_back(
          tvm::runtime::Empty({reserved_num_seqs + 1}, dtype_aux_, device));
    }
  }

  // The reset of the plain auxiliary data manager is no-op.
  void ResetAttnAuxDataCopy() final {}
  ffi::Tensor CopyQOIndptrOnDepthAsync(HostMemoryVector* data, int depth) final {
    ffi::Tensor view = tvm::runtime::CreateView(qo_indptr_on_depths_device_[depth],
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyPageIndptrOnDepthAsync(HostMemoryVector* data, int depth,
                                         bool layer_sliding_window = false) final {
    const auto& storage = layer_sliding_window ? page_indptr_sliding_on_depths_device_
                                               : page_indptr_on_depths_device_;
    ffi::Tensor view =
        tvm::runtime::CreateView(storage[depth], {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyPageIndicesOnDepthAsync(HostMemoryVector* data, int depth,
                                          bool layer_sliding_window = false) final {
    const auto& storage = layer_sliding_window ? page_indices_sliding_on_depths_device_
                                               : page_indices_on_depths_device_;
    ffi::Tensor view =
        tvm::runtime::CreateView(storage[depth], {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyLastPageLenOnDepthAsync(HostMemoryVector* data, int depth) final {
    ffi::Tensor view = tvm::runtime::CreateView(length_info_on_depths_device_[depth],
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyKRoPEPosOffsetOnDepthAsync(HostMemoryVector* data, int depth) final {
    ffi::Tensor view = tvm::runtime::CreateView(k_rope_pos_offset_on_depths_device_[depth],
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyCurAppendLengthIndptrAsync(HostMemoryVector* data) final {
    ffi::Tensor view = tvm::runtime::CreateView(cur_append_length_indptr_device_,
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyKRaggedRoPEPosOffsetAsync(HostMemoryVector* data) final {
    ffi::Tensor view = tvm::runtime::CreateView(k_ragged_rope_pos_offset_device_,
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyQRoPEPosMapAsync(HostMemoryVector* data) final {
    ffi::Tensor view = tvm::runtime::CreateView(q_rope_position_map_device_,
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyAppendPositionMapAsync(HostMemoryVector* data) final {
    ffi::Tensor view = tvm::runtime::CreateView(append_position_map_device_,
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyKVTransferRemotePositionMapAsync(HostMemoryVector* data) final {
    ffi::Tensor view = tvm::runtime::CreateView(kv_transfer_remote_position_map_device,
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyKVTransferRecverIDAsync(HostMemoryVector* data) final {
    ffi::Tensor view = tvm::runtime::CreateView(kv_transfer_recver_id_device,
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyKVTransferPage2PageLocalPositionMapAsync(HostMemoryVector* data) final {
    ffi::Tensor view = tvm::runtime::CreateView(kv_transfer_page_to_page_local_position_map_device,
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyKVTransferPage2PageRemotePositionMapAsync(HostMemoryVector* data) final {
    ffi::Tensor view = tvm::runtime::CreateView(kv_transfer_page_to_page_remote_position_map_device,
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyKVTransferPage2PageRecverIDAsync(HostMemoryVector* data) final {
    ffi::Tensor view = tvm::runtime::CreateView(kv_transfer_page_to_page_recver_id_device,
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }

  ffi::Tensor CopyTreeAttnMaskOnDepthAsync(HostMemoryVector* data, int depth) final {
    ffi::Tensor view = tvm::runtime::CreateView(tree_attn_mask_device_[depth],
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyTreeAttnMNIndptrOnDepthAsync(HostMemoryVector* data, int depth) final {
    ffi::Tensor view = tvm::runtime::CreateView(tree_attn_mn_indptr_device_[depth],
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }

  ffi::Tensor CopyLengthInfoOnDepthAsync(HostMemoryVector* last_page_len,
                                         HostMemoryVector* sliding_window_offset,
                                         HostMemoryVector* sink_size, int depth,
                                         bool layer_sliding_window = false) final {
    int n_elem = last_page_len->size();
    TVM_FFI_ICHECK_GT(n_elem, 0);
    const auto& storage = layer_sliding_window ? length_info_sliding_on_depths_device_
                                               : length_info_on_depths_device_;
    ffi::Tensor view = tvm::runtime::CreateView(storage[depth], {3, n_elem}, dtype_aux_);
    ffi::Shape copy_shape{n_elem};
    CopyVecDataToArray(view, last_page_len->data(), copy_shape);
    CopyVecDataToArray(view, sliding_window_offset->data(), copy_shape,
                       /*dst_elem_offset=*/n_elem);
    CopyVecDataToArray(view, sink_size->data(), copy_shape,
                       /*dst_elem_offset=*/2 * n_elem);
    return view;
  }

  // The commit of the plain auxiliary data manager is no-op.
  void CommitAttnAuxDataCopy() final {}

  // The reset of the plain auxiliary data manager is no-op.
  void ResetCompactKVAuxDataCopy() final {}

  ffi::Tensor CopyCommitLengthIndptrAsync(HostMemoryVector* data) final {
    ffi::Tensor view = tvm::runtime::CreateView(commit_copy_length_indptr_device_,
                                                {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  ffi::Tensor CopyCommitSrcDstPosInPageTableAsync(HostMemoryVector* src_data,
                                                  HostMemoryVector* dst_data) final {
    int n_elem = src_data->size();
    TVM_FFI_ICHECK_GT(n_elem, 0);
    ffi::Tensor view = tvm::runtime::CreateView(commit_copy_src_dst_pos_in_page_table_device_,
                                                {2, n_elem}, dtype_aux_);
    ffi::Shape copy_shape{n_elem};
    CopyVecDataToArray(view, src_data->data(), copy_shape);
    CopyVecDataToArray(view, dst_data->data(), copy_shape,
                       /*dst_elem_offset=*/n_elem);
    return view;
  }

  // The commit of the plain auxiliary data manager is no-op.
  void CommitCompactKVAuxDataCopy() final {}

 private:
  /*!
   * \brief Copy a vector of data to the input Tensor.
   * It optionally supports specifying the shape of copy and the element
   * offset to the destination Tensor.
   */
  void CopyVecDataToArray(ffi::Tensor array, int32_t* vec_data,
                          ffi::Optional<ffi::Shape> shape = std::nullopt, int dst_elem_offset = 0) {
    TVM_FFI_ICHECK(array.defined());
    if (array.shape()[0] == 0) {
      return;
    }
    DLTensor copy_dst = *array.GetDLTensorPtr();
    if (shape.has_value()) {
      TVM_FFI_ICHECK_EQ(shape.value().size(), 1);
      copy_dst.ndim = 1;
      copy_dst.shape = const_cast<int64_t*>(shape.value()->data);
      // The strides still describe the multi-row view this copies one row of.
      copy_dst.strides = nullptr;
    }
    copy_dst.byte_offset += dst_elem_offset * sizeof(int32_t);

    if (void* host_ptr =
            DeviceAPI::Get(copy_dst.device)->GetMappedHostPtrIfAvailable(copy_dst.data)) {
      std::memcpy(static_cast<char*>(host_ptr) + copy_dst.byte_offset, vec_data,
                  ffi::GetDataSize(copy_dst));
      return;
    }

    DLTensor copy_src;
    copy_src.data = vec_data;
    copy_src.device = preferred_host_device_;
    copy_src.ndim = 1;
    copy_src.dtype = array.dtype();
    copy_src.shape = copy_dst.shape;
    copy_src.strides = nullptr;
    copy_src.byte_offset = 0;
    tvm::runtime::CopyFromTo(&copy_src, &copy_dst, copy_stream_);
  }

  // Both page tables stay live until all global and sliding layers finish.
  std::vector<ffi::Tensor> page_indptr_sliding_on_depths_device_;
  std::vector<ffi::Tensor> page_indices_sliding_on_depths_device_;
  std::vector<ffi::Tensor> length_info_sliding_on_depths_device_;
  std::vector<ffi::Tensor> qo_indptr_on_depths_device_;
  std::vector<ffi::Tensor> page_indptr_on_depths_device_;
  std::vector<ffi::Tensor> page_indices_on_depths_device_;
  std::vector<ffi::Tensor> length_info_on_depths_device_;
  std::vector<ffi::Tensor> k_rope_pos_offset_on_depths_device_;
  std::vector<ffi::Tensor> tree_attn_mask_device_;
  std::vector<ffi::Tensor> tree_attn_mn_indptr_device_;
  ffi::Tensor cur_append_length_indptr_device_;
  ffi::Tensor k_ragged_rope_pos_offset_device_;
  ffi::Tensor q_rope_position_map_device_;
  ffi::Tensor append_position_map_device_;
  ffi::Tensor kv_transfer_remote_position_map_device;
  ffi::Tensor kv_transfer_recver_id_device;
  ffi::Tensor kv_transfer_page_to_page_local_position_map_device;
  ffi::Tensor kv_transfer_page_to_page_remote_position_map_device;
  ffi::Tensor kv_transfer_page_to_page_recver_id_device;
  ffi::Tensor commit_copy_length_indptr_device_;
  ffi::Tensor commit_copy_src_dst_pos_in_page_table_device_;
};

/*!
 * \brief The cached auxiliary data manager class.
 * It allocates a large on-device array to store all the auxiliary data.
 * For each `CopyXXXAsync`, it copies the input data to a local cache on host.
 * In `CommitAttnAuxDataCopy`, it copies all the data in the local cache to the device
 * array for a single time, and thus reduce the number of host-to-device copies needed.
 */
class CachedPagedKVCacheAuxDataManager : public PagedKVCacheAuxDataManager {
 public:
  explicit CachedPagedKVCacheAuxDataManager(int64_t reserved_num_seqs, int64_t num_total_pages,
                                            int64_t prefill_chunk_size, DLDataType dtype_aux,
                                            Device device, Device preferred_host_device,
                                            TVMStreamHandle copy_stream)
      : PagedKVCacheAuxDataManager(dtype_aux, device, preferred_host_device, copy_stream),
        elem_byte_size_((dtype_aux.bits * dtype_aux.lanes + 7) / 8),
        offset_alignment_(cuda_byte_alignment_ / elem_byte_size_),
        merged_attn_aux_data_device_(tvm::runtime::Empty({32 * 1024 * 1024}, dtype_aux, device)),
        merged_compact_kv_aux_data_device_(tvm::runtime::Empty(
            {CalculateCompactKVAuxDataCacheSize(reserved_num_seqs, prefill_chunk_size)}, dtype_aux,
            device)) {
    // - Calculate cache size of all the attention auxiliary arrays in
    // local cache and the large on-device array.
    // int64_t attn_aux_data_cache_size =
    //     CalculateAttnAuxDataCacheSize(reserved_num_seqs, num_total_pages, prefill_chunk_size);
    int64_t attn_aux_data_cache_size = 32 * 1024 * 1024;
    // - Initialize the host auxiliary data buffer.
    merged_attn_aux_data_host_ =
        HostMemoryVector(attn_aux_data_cache_size, dtype_aux, preferred_host_device);
    // - Calculate cache size of all the compact KV auxiliary arrays in
    // local cache and the large on-device array.
    int64_t compact_kv_aux_data_cache_size =
        CalculateCompactKVAuxDataCacheSize(reserved_num_seqs, prefill_chunk_size);
    // - Initialize the host auxiliary data buffer.
    merged_compact_kv_aux_data_host_ =
        HostMemoryVector(compact_kv_aux_data_cache_size, dtype_aux, preferred_host_device);
  }

  void ResetAttnAuxDataCopy() final { attn_aux_data_copy_offset_ = 0; }
  ffi::Tensor CopyQOIndptrOnDepthAsync(HostMemoryVector* data, int depth) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyPageIndptrOnDepthAsync(HostMemoryVector* data, int depth,
                                         bool layer_sliding_window = false) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyPageIndicesOnDepthAsync(HostMemoryVector* data, int depth,
                                          bool layer_sliding_window = false) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyLastPageLenOnDepthAsync(HostMemoryVector* data, int depth) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyKRoPEPosOffsetOnDepthAsync(HostMemoryVector* data, int depth) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyCurAppendLengthIndptrAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyKRaggedRoPEPosOffsetAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyQRoPEPosMapAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyAppendPositionMapAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyKVTransferRemotePositionMapAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyKVTransferRecverIDAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyKVTransferPage2PageLocalPositionMapAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyKVTransferPage2PageRemotePositionMapAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyKVTransferPage2PageRecverIDAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyTreeAttnMaskOnDepthAsync(HostMemoryVector* data, int depth) final {
    ffi::Tensor mask_1d = CopyAttnAuxVecToCache(data);
    return tvm::runtime::CreateView(mask_1d, {static_cast<int64_t>(data->size() / 2), 2},
                                    mask_1d.dtype());
  }
  ffi::Tensor CopyTreeAttnMNIndptrOnDepthAsync(HostMemoryVector* data, int depth) final {
    return CopyAttnAuxVecToCache(data);
  }
  ffi::Tensor CopyLengthInfoOnDepthAsync(HostMemoryVector* last_page_len,
                                         HostMemoryVector* sliding_window_offset,
                                         HostMemoryVector* sink_size, int depth,
                                         bool layer_sliding_window = false) final {
    int64_t n_elem = last_page_len->size();
    std::memcpy(merged_attn_aux_data_host_.data() + attn_aux_data_copy_offset_,
                last_page_len->data(), n_elem * elem_byte_size_);
    std::memcpy(merged_attn_aux_data_host_.data() + attn_aux_data_copy_offset_ + n_elem,
                sliding_window_offset->data(), n_elem * elem_byte_size_);
    std::memcpy(merged_attn_aux_data_host_.data() + attn_aux_data_copy_offset_ + 2 * n_elem,
                sink_size->data(), n_elem * elem_byte_size_);
    ffi::Tensor view =
        tvm::runtime::CreateView(merged_attn_aux_data_device_, {3, n_elem}, dtype_aux_,
                                 attn_aux_data_copy_offset_ * elem_byte_size_);
    attn_aux_data_copy_offset_ += CeilDivElemAlignment(3 * n_elem);
    return view;
  }

  void CommitAttnAuxDataCopy() final {
    std::vector<int64_t> copy_shape{attn_aux_data_copy_offset_};
    DLTensor copy_dst;
    copy_dst.data = merged_attn_aux_data_device_.data_ptr();
    copy_dst.device = device_;
    copy_dst.ndim = 1;
    copy_dst.dtype = dtype_aux_;
    copy_dst.shape = copy_shape.data();
    copy_dst.strides = nullptr;
    copy_dst.byte_offset = 0;

    DLTensor copy_src = copy_dst;
    copy_src.data = merged_attn_aux_data_host_.data();
    copy_src.device = Device{kDLCPU, 0};
    tvm::runtime::CopyFromTo(&copy_src, &copy_dst, copy_stream_);
  }

  void ResetCompactKVAuxDataCopy() final { compact_kv_aux_data_copy_offset_ = 0; }

  ffi::Tensor CopyCommitLengthIndptrAsync(HostMemoryVector* data) final {
    return CopyCompactKVAuxVecToCache(data);
  }
  ffi::Tensor CopyCommitSrcDstPosInPageTableAsync(HostMemoryVector* src_data,
                                                  HostMemoryVector* dst_data) final {
    int64_t n_elem = src_data->size();
    std::memcpy(merged_compact_kv_aux_data_host_.data() + compact_kv_aux_data_copy_offset_,
                src_data->data(), n_elem * elem_byte_size_);
    std::memcpy(merged_compact_kv_aux_data_host_.data() + compact_kv_aux_data_copy_offset_ + n_elem,
                dst_data->data(), n_elem * elem_byte_size_);
    ffi::Tensor view =
        tvm::runtime::CreateView(merged_compact_kv_aux_data_device_, {2, n_elem}, dtype_aux_,
                                 compact_kv_aux_data_copy_offset_ * elem_byte_size_);
    compact_kv_aux_data_copy_offset_ += CeilDivElemAlignment(2 * n_elem);
    return view;
  }

  void CommitCompactKVAuxDataCopy() final {
    std::vector<int64_t> copy_shape{compact_kv_aux_data_copy_offset_};
    DLTensor copy_dst;
    copy_dst.data = merged_compact_kv_aux_data_device_.data_ptr();
    copy_dst.device = device_;
    copy_dst.ndim = 1;
    copy_dst.dtype = dtype_aux_;
    copy_dst.shape = copy_shape.data();
    copy_dst.strides = nullptr;
    copy_dst.byte_offset = 0;

    DLTensor copy_src = copy_dst;
    copy_src.data = merged_compact_kv_aux_data_host_.data();
    copy_src.device = Device{kDLCPU, 0};
    tvm::runtime::CopyFromTo(&copy_src, &copy_dst, copy_stream_);
  }

 private:
  /*!
   * \brief Calculate the start element offsets of the auxiliary arrays in the local cache.
   * \return Return the local cache size (total number of elements in the local cache).
   */
  int64_t CalculateAttnAuxDataCacheSize(int64_t reserved_num_seqs, int64_t num_total_pages,
                                        int64_t prefill_chunk_size) {
    int64_t cache_size = 0;
    // - Array size of the arrays that every depth has.
    // Corresponding to the following arrays respectively
    //  - qo_indptr_in_depth
    //  - page_indptr_in_depth
    //  - page_indices_in_depth
    //  - length_info_in_depth
    //  - k_rope_pos_offset_in_depth
    cache_size += CeilDivElemAlignment(reserved_num_seqs + 1);
    cache_size += CeilDivElemAlignment(reserved_num_seqs + 1);
    cache_size += CeilDivElemAlignment(num_total_pages);
    cache_size += CeilDivElemAlignment(3 * reserved_num_seqs);
    cache_size += CeilDivElemAlignment(reserved_num_seqs);
    cache_size *= kPagedKVCacheMaxBlockDepth;

    // - Array size of other arrays.
    // Corresponding to the following arrays respectively
    //  - cur_append_length_indptr
    //  - k_ragged_rope_pos_offset
    //  - q_rope_position_map
    //  - append_position_map
    //  - kv_transfer_remote_position_map
    //  - kv_transfer_recver_id
    //  - kv_transfer_page_to_page_local_position_map
    //  - kv_transfer_page_to_page_remote_position_map
    //  - kv_transfer_page_to_page_recver_id
    //  - tree_attn_mask
    //  - tree_attn_mn_indptr
    cache_size += CeilDivElemAlignment(reserved_num_seqs + 1);
    cache_size += CeilDivElemAlignment(reserved_num_seqs);
    cache_size += CeilDivElemAlignment(prefill_chunk_size);
    cache_size += CeilDivElemAlignment(prefill_chunk_size);
    cache_size += CeilDivElemAlignment(prefill_chunk_size);
    cache_size += CeilDivElemAlignment(prefill_chunk_size);
    cache_size += CeilDivElemAlignment(prefill_chunk_size);
    cache_size += CeilDivElemAlignment(prefill_chunk_size);
    cache_size += CeilDivElemAlignment(prefill_chunk_size);
    cache_size +=
        CeilDivElemAlignment(kTreeAttnMaxTreeSize * kTreeAttnMaxTreeSize * reserved_num_seqs);
    cache_size += CeilDivElemAlignment(reserved_num_seqs + 1);

    return cache_size;
  }

  int64_t CalculateCompactKVAuxDataCacheSize(int64_t reserved_num_seqs,
                                             int64_t prefill_chunk_size) {
    int64_t cache_size = 0;
    // Corresponding to the following arrays respectively
    //  - commit_copy_length_indptr
    //  - commit_copy_src_dst_pos_in_page_table
    cache_size += CeilDivElemAlignment(reserved_num_seqs + 1);
    cache_size += CeilDivElemAlignment(
        2 * std::min(kTreeAttnMaxTreeSize * reserved_num_seqs, prefill_chunk_size));

    return cache_size;
  }

  /*!
   * \brief Copy the input data to the cache at the given offset.
   * And return the Tensor view of the cache starting at the offset.
   */
  ffi::Tensor CopyAttnAuxVecToCache(HostMemoryVector* data) {
    int64_t n_elem = data->size();
    std::memcpy(merged_attn_aux_data_host_.data() + attn_aux_data_copy_offset_, data->data(),
                n_elem * elem_byte_size_);
    ffi::Tensor view = tvm::runtime::CreateView(merged_attn_aux_data_device_, {n_elem}, dtype_aux_,
                                                attn_aux_data_copy_offset_ * elem_byte_size_);
    attn_aux_data_copy_offset_ += CeilDivElemAlignment(n_elem);
    return view;
  }

  ffi::Tensor CopyCompactKVAuxVecToCache(HostMemoryVector* data) {
    int64_t n_elem = data->size();
    std::memcpy(merged_compact_kv_aux_data_host_.data() + compact_kv_aux_data_copy_offset_,
                data->data(), n_elem * elem_byte_size_);
    ffi::Tensor view =
        tvm::runtime::CreateView(merged_compact_kv_aux_data_device_, {n_elem}, dtype_aux_,
                                 compact_kv_aux_data_copy_offset_ * elem_byte_size_);
    compact_kv_aux_data_copy_offset_ += CeilDivElemAlignment(n_elem);
    return view;
  }

  /*! \brief For safety, we align the start offset of the arrays to `offset_alignment`. */
  int64_t CeilDivElemAlignment(int n) {
    return (n + offset_alignment_ - 1) / offset_alignment_ * offset_alignment_;
  }

  const int64_t cuda_byte_alignment_ = 16;
  const int64_t elem_byte_size_;
  const int64_t offset_alignment_;

  int64_t attn_aux_data_copy_offset_ = 0;
  int64_t compact_kv_aux_data_copy_offset_ = 0;
  HostMemoryVector merged_attn_aux_data_host_;
  HostMemoryVector merged_compact_kv_aux_data_host_;
  ffi::Tensor merged_attn_aux_data_device_;
  ffi::Tensor merged_compact_kv_aux_data_device_;
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_ATTN_UTILS_H_
