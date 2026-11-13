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
#ifndef TVM_RUNTIME_VM_KV_STATE_H_
#define TVM_RUNTIME_VM_KV_STATE_H_
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/tensor.h>

namespace tvm {
namespace runtime {
namespace vm {

/*! \brief The base class of attention KV cache and rnn state. */
class KVStateObj : public ffi::Object {
 public:
  /*! \brief Reset the KV State. */
  virtual void Clear() = 0;

  /************** Sequence Management **************/

  /*!
   * \brief Add a new sequence with empty K/V state in the cache.
   * Check if the validity of the input sequence id.
   * \param seq_id The id of the new sequence to be added.
   * \throws Error if the given sequence id is not valid.
   */
  virtual void AddSequence(int64_t seq_id) = 0;

  /*!
   * \brief Remove a sequence and its K/V state from the KV cache.
   * \param seq_id The sequence to remove from cache.
   * \throws Error if the given sequence id is not valid.
   */
  virtual void RemoveSequence(int64_t seq_id) = 0;

  /*!
   * \brief Fork the K/V state of parent sequence to the child sequence.
   * After the fork, the child sequence has K/V state of the parent
   * sequence.
   * \param parent_seq_id The parent (source) of the fork.
   * \param child_seq_id The child (destination) of the fork.
   * The child sequence id should not exist in cache prior to fork.
   * \param fork_pos The parent position to fork, the legal forking position is within
   * [0, parent_seq_length] and -1 as default for last position. And if forking position is 0,
   * it equals to add a new sequence with child sequence id.
   * \throws Error if the given sequence ids are not valid.
   */
  virtual void ForkSequence(int64_t parent_seq_id, int64_t child_seq_id, int64_t fork_pos = -1) = 0;

  /*!
   * \brief Pop out the trailing `n` tokens from the KV cache for the
   * specified sequence.
   * \param seq_id The sequence whose trailing tokens are to be popped.
   * \param n The number of tokens to pop from the KV cache.
   * \throws Error if the given sequence id is not valid.
   */
  virtual void PopN(int64_t seq_id, int32_t n) = 0;

  /*!
   * \brief Mark the start of the forward function with the ids of
   * the sequences and the sequence length to forward for each
   * sequence.
   * For example, if we want to prefill 3 sequences "5", "1", "8"
   * with prefill length "10", "15", "20", then we pass `[5, 1, 8]`
   * as the seq_ids and `[10, 15, 20]` as the append_lengths.
   * This method is invoked right before entering the model forward
   * function, and contains operations to prepare the incoming
   * forward. For instance, this method may send auxiliary KV cache
   * data structures to GPUs so that they can be operated
   * in the model forward function.
   * \param seq_ids The ids of the sequence to run in the incoming model forward.
   * \param append_lengths The sequence lengths to run forward for for each sequence.
   * \param token_tree_parent_ptr The parent idx array of the token trees. Its length
   * is the sum of "append_lengths". Nullptr means the token tree of each sequence
   * is a chain.
   */
  virtual void BeginForward(
      const ffi::Shape& seq_ids, const ffi::Shape& append_lengths,
      const ffi::Optional<ffi::Shape>& token_tree_parent_ptr = std::nullopt) = 0;

  /*!
   * \brief Mark the start of the forward function.
   * This method is invoked right after entering the model forward
   * function, and contains post-processing of the forward.
   */
  virtual void EndForward() = 0;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.vm.KVState", KVStateObj, ffi::Object);
};

class KVState : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(KVState, ffi::ObjectRef, KVStateObj);
};

/*!
 * \brief The base class of attention KV cache for efficient
 * k/v data management and attention computation.
 */
class AttentionKVCacheObj : public KVStateObj {
 public:
  /************** Raw Info Query **************/

  /*! \brief Check if the KV cache is empty. */
  virtual bool Empty() const = 0;
  /*!
   * \brief Get the number of available pages in the KV cache.
   * When the underlying KV cache implementation is not
   * paged KV cache, the function falls back to return the
   * number of remaining size (in terms of number of tokens).
   */
  virtual int32_t GetNumAvailablePages() const = 0;

  /*! \brief Get the current total sequence length in the KV cache. */
  virtual int32_t GetTotalSequenceLength() const = 0;

  /*!
   * \brief Get checkpoint metadata for a sequence.
   * \param seq_id The id of the sequence whose checkpoint metadata is requested.
   * \return JSON string describing runtime layout and logical pages.
   */
  virtual ffi::String GetCheckpointMetadata(int64_t seq_id) const = 0;

  /*!
   * \brief Get a stable hash over runtime layout-defining fields.
   * \return Hex string hash of the checkpoint layout metadata.
   */
  virtual ffi::String GetLayoutHash() const = 0;

  /*!
   * \brief Export a checkpoint page group for a sequence.
   * \param seq_id The id of the sequence whose page group is exported.
   * \param group_id The checkpoint group id to export.
   * \param dst The destination tensor for the exported page group.
   */
  virtual void ExportPageGroup(int64_t seq_id, int64_t group_id, ffi::Tensor dst) = 0;

  /*!
   * \brief Prepare sequence state and page tables for checkpoint import.
   * \param seq_id The id of the sequence being imported.
   * \param metadata_json The checkpoint metadata JSON string.
   */
  virtual void PrepareImport(int64_t seq_id, ffi::String metadata_json) = 0;

  /*!
   * \brief Import a checkpoint page group into a prepared sequence.
   * \param seq_id The id of the sequence being imported.
   * \param group_id The checkpoint group id to import.
   * \param src The source tensor containing the page group.
   */
  virtual void ImportPageGroup(int64_t seq_id, int64_t group_id, ffi::Tensor src) = 0;

  /*!
   * \brief Finish a checkpoint import after all page groups have been restored.
   * \param seq_id The id of the sequence being imported.
   */
  virtual void FinishImport(int64_t seq_id) = 0;

  /*!
   * \brief Get the sequence length for a sequence in the KV cache.
   * \param seq_id The id of the sequence whose length is requested.
   * \return The sequence length.
   */
  virtual int32_t GetSequenceLength(int64_t seq_id) const = 0;

  /************** Sequence Management **************/

  /*!
   * \brief Enable sliding window attention for the given sequence.
   * Error will be thrown when the KV cache does not support sliding window.
   * \param seq_id The id of the sequence to enable sliding window for.
   * \param sliding_window_size The sliding window size for the sequence.
   * \param attn_sink_size The attention sink set for the sequence.
   */
  virtual void EnableSlidingWindowForSeq(int64_t seq_id, int32_t sliding_window_size,
                                         int32_t attn_sink_size) = 0;

  /*!
   * \brief Committed the accepted token tree nodes to KV cache.
   * The commit will update the KV cache, by compacting the KV data and discard
   * the KV data of rejected tokens.
   * This is a mandatory step when the BeginForward is given with a token tree.
   * \param seq_ids The ids of the sequences to commit.
   * \param leaf_indices The leaf token tree node index of each sequence.
   */
  virtual void CommitAcceptedTokenTreeNodes(const ffi::Shape& seq_ids,
                                            const ffi::Shape& leaf_indices) = 0;

  /*! \brief Prepare for the disaggregation KV data receive for the specified sequence and length.*/
  virtual ffi::Shape DisaggPrepareRecv(int64_t seq_id, int length) = 0;

  /*! \brief Mark which tokens' KV cache needs to be sent to other devices */
  virtual void DisaggMarkSend(int64_t seq_id, int64_t begin,
                              const ffi::Shape& compressed_remote_position_map,
                              int32_t recver_pe_offset) = 0;

  /************** Attention **************/

  /*!
   * \brief Compute attention with Q/K/V data which are concatenated along
   * the head dimension.
   * \param layer_id The model layer where the attention compute happens.
   * \param qkv_data The input Q/K/V data, in layout
   * `(total_length, num_qo_heads + 2 * num_kv_heads, head_dim)`.
   * \param mask The input mask data, in layout `(total_sqr_length)`.
   * \param o_data The output O data, in layout `(total_length, num_qo_heads, head_dim)`.
   * \param sm_scale The additional attention scaling factor.
   * \sa AttentionKVCache::Attention
   */
  virtual void AttentionWithFusedQKV(int64_t layer_id, ffi::Tensor qkv_data,
                                     ffi::Optional<ffi::Tensor> mask, ffi::Tensor o_data,
                                     double sm_scale) = 0;

  /*!
   * \brief Fine-grained API that computes ragged self attention with Q/K/V data.
   * \param layer_id The model layer where the attention compute happens.
   * \param q_data The input Q data.
   * \param k_data The input K data.
   * \param v_data The input V data.
   * \param o_data The output O data, in layout `(total_length, num_qo_heads, v_head_dim)`.
   * \param lse_data The output attention LSE data, in layout `(total_length, num_qo_heads)`.
   * \param sm_scale The additional attention scaling factor.
   */
  virtual void SelfAttention(int64_t layer_id, ffi::Tensor q_data, ffi::Tensor k_data,
                             ffi::Tensor v_data, ffi::Tensor o_data, ffi::Tensor lse_data,
                             double sm_scale) = 0;

  /*!
   * \brief Fine-grained API that computes paged cross attention with Q and in-cache KV data.
   * \param layer_id The model layer where the attention compute happens.
   * \param q_data The input Q data.
   * \param o_data The output O data.
   * \param lse_data The output attention LSE data, in layout `(total_length, num_qo_heads)`.
   * \param sm_scale The additional attention scaling factor.
   */
  virtual void CrossAttention(int64_t layer_id, ffi::Tensor q_data, ffi::Tensor o_data,
                              ffi::Tensor lse_data, double sm_scale) = 0;

  /*!
   * \brief Compute attention with K/V shared by another logical layer.
   *
   * This operation does not append K/V. The cache-owning source layer must run before this
   * operation in the active forward pass.
   * \param source_layer_id The physical cache layer containing past K/V data.
   * \param q_data The logical layer's input Q data.
   * \param current_k_data The source layer's K data for the active chunk.
   * \param current_v_data The source layer's V data for the active chunk.
   * \param o_data The output O data.
   * \param sm_scale The additional attention scaling factor.
   */
  virtual void AttentionWithSharedKV(int64_t source_layer_id, ffi::Tensor q_data,
                                     ffi::Tensor current_k_data, ffi::Tensor current_v_data,
                                     ffi::Tensor o_data, double sm_scale) = 0;

  /*!
   * \brief Fine-grained API that appends the MLA K/V data to KV cache.
   * \param layer_id The model layer where the attention compute happens.
   * \param kv_data The input KV data to append, in layout `(total_length, qk_head_dim)`.
   */
  virtual void AppendMLAKV(int64_t layer_id, ffi::Tensor kv_data) = 0;

  /*!
   * \brief Fine-grained API that merges the attention output from two sources.
   * \param o1_data The first source O data.
   * \param lse1_data The first source LSE data.
   * \param o2_data The second source O data.
   * \param lse2_data The second source LSE data.
   * \return The merged O and LSE data.
   */
  virtual ffi::Array<ffi::Tensor> MergeAttnOutputInplace(ffi::Tensor o_self_attn,
                                                         ffi::Tensor lse_self_attn,
                                                         ffi::Tensor o_cross_attn,
                                                         ffi::Tensor lse_cross_attn) = 0;

  /*!
   * \brief Compute linear attention with Q/K/V data.
   * \param layer_id The model layer where the attention compute happens.
   * \param q_data The input Q data, in layout `(total_length, num_qo_heads, head_dim)`.
   * \param k_data The input K data, in layout `(total_length, num_kv_heads, head_dim)`.
   * \param v_data The input V data, in layout `(total_length, num_kv_heads, head_dim)`.
   * \param o_data The output O data, in layout `(total_length, num_qo_heads, head_dim)`.
   * \param sm_scale The additional attention scaling factor.
   * \sa AttentionKVCache::Attention
   */
  virtual void LinearAttention(int64_t layer_id, ffi::Tensor q_data, ffi::Tensor k_data,
                               ffi::Tensor v_data, double sm_scale) = 0;

  /************** Positions **************/

  /*!
   * \brief Get the in-sequence positions of each slot in the query.
   * This function is supposed to be invoked after calling BeginForward.
   * \return The in-sequence query positions, in shape `(total_length,)`.
   */
  virtual ffi::Tensor GetQueryPositions() = 0;

  /************** Debug Helpers **************/

  /*!
   * \brief Fetch the compact K/V data of the given sequence.
   * The results are written into `k_data` and `v_data` passed in,
   * conforming to the destination-passing style.
   * `start_pos` (inclusive) and `end_pos` (exclusive) controls the range
   * of K/V data to fetch.
   * - `start_pos` defaults to 0 when undefined,
   * - `end_pos` defaults to the length of sequence when undefined.
   * This method expects `start_pos < end_pos`.
   * The k/v data arrays are expected to have layout
   * `(num_layers, start_pos - end_pos, num_heads, head_dim)`.
   * \param seq_id The sequence whose K/V data is to be fetched.
   * \param start_pos The start position (inclusive) of the K/V data to fetch.
   * \param end_pos The end position (exclusive) of the K/V data to fetch.
   * \param K_data The output K data of the given sequence in layout elaborated above.
   * \param V_data The output V data of the given sequence in layout elaborated above.
   */
  virtual void DebugGetKV(int64_t seq_id,  //
                          int64_t start_pos, int64_t end_pos, ffi::Tensor k_data,
                          ffi::Tensor v_data) = 0;

  /*!
   * \brief Fetch the compact K/V data of the given sequence for MLA cache.
   * \param seq_id The sequence whose K/V data is to be fetched.
   * \param start_pos The start position (inclusive) of the K/V data to fetch.
   * \param end_pos The end position (exclusive) of the K/V data to fetch.
   * \param kv_data The output KV data of the given sequence in layout elaborated above.
   */
  virtual void DebugGetKVMLA(int64_t seq_id, int64_t start_pos, int64_t end_pos,
                             ffi::Tensor kv_data) = 0;

  /*!
   * \brief Set the K/V data of the given sequence from input K/V data.
   * `start_pos` (inclusive) controls starting position of K/V data
   * to set, and defaults to 0 when undefined.
   * The input K/V data is in layout `(num_layers, length, num_heads, head_dim)`,
   * where `length` is the length to set. It implies that the range
   * of set is `[start_pos, start_pos + length)`.
   * It is not allowed that `start_pos + length` exceeds the current
   * length of the given sequence.
   * \param seq_id The sequence whose K/V data is to be set.
   * \param start_pos The start position (inclusive) of the K/V data to set.
   * \param k_data The K data to set in layout elaborated above.
   * \param v_data The V data to set in layout elaborated above.
   */
  virtual void DebugSetKV(int64_t seq_id, int64_t start_pos, ffi::Tensor k_data,
                          ffi::Tensor v_data) = 0;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.vm.AttentionKVCache", AttentionKVCacheObj, KVStateObj);
};

class AttentionKVCache : public KVState {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AttentionKVCache, KVState, AttentionKVCacheObj);
};

/*!
 * \brief The base class of RNN State for efficient
 * State data management and attention computation.
 */
class RNNStateObj : public KVStateObj {
 public:
  /************** Interaction **************/
  /*!
   * \brief Get the State data for the specified sequence.
   * \param layer_id The model layer where the state is set.
   * \param state_id The state id within the layer.
   * \param o_data The output data to be fetched.
   * \return The array of State data, each element corresponds to a state.
   * \throws Error if the given sequence id is not valid.
   */
  virtual void Get(int64_t layer_id, int64_t state_id, ffi::Tensor o_data) = 0;

  /*!
   * \brief Set the State data for the specified sequence.
   * \param layer_id The model layer where the state is set.
   * \param state_id The state id within the layer.
   * \param data The data to be set.
   * \throws Error if the given sequence id is not valid.
   */
  virtual void Set(int64_t layer_id, int64_t state_id, ffi::Tensor data) = 0;

  /*!
   * \brief Fetch the compact rnn state data of the given sequence.
   * \param layer_id The model layer where the state is set.
   * \param state_id The state id within the layer.
   * \param seq_id The sequence whose state data is to be fetched.
   */
  virtual ffi::Tensor DebugGet(int64_t layer_id, int64_t state_id, int64_t seq_id) = 0;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.vm.RNNState", RNNStateObj, KVStateObj);
};

class RNNState : public KVState {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RNNState, KVState, RNNStateObj);
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_KV_STATE_H_
