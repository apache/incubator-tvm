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

#include "kv_state.h"

#include <tvm/ffi/reflection/registry.h>

#include <utility>

namespace tvm {
namespace runtime {
namespace vm {

// Register ffi::Object Type

// KV State base methods
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("vm.builtin.kv_state_clear", &KVStateObj::Clear)
      .def_method("vm.builtin.kv_state_add_sequence", &KVStateObj::AddSequence)
      .def_method("vm.builtin.kv_state_remove_sequence", &KVStateObj::RemoveSequence)
      .def_method("vm.builtin.kv_state_fork_sequence", &KVStateObj::ForkSequence)
      .def_method("vm.builtin.kv_state_popn", &KVStateObj::PopN)
      .def_packed("vm.builtin.kv_state_begin_forward",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    TVM_FFI_ICHECK(args.size() == 3 || args.size() == 4)
                        << "KVState BeginForward only accepts 3 or 4 arguments";
                    KVState kv_state = args[0].cast<KVState>();
                    ffi::Shape seq_ids = args[1].cast<ffi::Shape>();
                    ffi::Shape append_lengths = args[2].cast<ffi::Shape>();
                    ffi::Optional<ffi::Shape> token_tree_parent_ptr;
                    if (args.size() == 4) {
                      token_tree_parent_ptr = args[3].cast<ffi::Optional<ffi::Shape>>();
                    }
                    kv_state->BeginForward(seq_ids, append_lengths, token_tree_parent_ptr);
                  })
      .def_method("vm.builtin.kv_state_end_forward", &KVStateObj::EndForward);
}

// Attention KV Cache methods
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("vm.builtin.kv_cache_disagg_prepare_recv",
                  &AttentionKVCacheObj::DisaggPrepareRecv)
      .def_method("vm.builtin.kv_cache_disagg_mark_send", &AttentionKVCacheObj::DisaggMarkSend)
      .def_method("vm.builtin.attention_kv_cache_enable_sliding_window_for_seq",
                  &AttentionKVCacheObj::EnableSlidingWindowForSeq)
      .def_method("vm.builtin.attention_kv_cache_commit_accepted_token_tree_nodes",
                  &AttentionKVCacheObj::CommitAcceptedTokenTreeNodes)
      .def_method("vm.builtin.attention_kv_cache_empty", &AttentionKVCacheObj::Empty)
      .def_method("vm.builtin.attention_kv_cache_get_num_available_pages",
                  &AttentionKVCacheObj::GetNumAvailablePages)
      .def_method("vm.builtin.attention_kv_cache_get_total_sequence_length",
                  &AttentionKVCacheObj::GetTotalSequenceLength)
      .def_method("vm.builtin.attention_kv_cache_get_checkpoint_metadata",
                  &AttentionKVCacheObj::GetCheckpointMetadata)
      .def_method("vm.builtin.attention_kv_cache_get_layout_hash",
                  &AttentionKVCacheObj::GetLayoutHash)
      .def_method("vm.builtin.attention_kv_cache_export_page_group",
                  &AttentionKVCacheObj::ExportPageGroup)
      .def_method("vm.builtin.attention_kv_cache_prepare_import",
                  &AttentionKVCacheObj::PrepareImport)
      .def_method("vm.builtin.attention_kv_cache_import_page_group",
                  &AttentionKVCacheObj::ImportPageGroup)
      .def_method("vm.builtin.attention_kv_cache_finish_import", &AttentionKVCacheObj::FinishImport)
      .def_method("vm.builtin.attention_kv_cache_get_sequence_length",
                  &AttentionKVCacheObj::GetSequenceLength)
      .def_method("vm.builtin.attention_kv_cache_get_query_positions",
                  &AttentionKVCacheObj::GetQueryPositions)
      .def_method("vm.builtin.attention_kv_cache_debug_get_kv", &AttentionKVCacheObj::DebugGetKV)
      .def_method("vm.builtin.attention_kv_cache_debug_get_kv_mla",
                  &AttentionKVCacheObj::DebugGetKVMLA)
      .def("vm.builtin.attention_kv_cache_attention_with_fused_qkv",
           [](AttentionKVCache kv_cache, int64_t layer_id, double sm_scale, ffi::Tensor qkv_data,
              ffi::Tensor o_data) {
             TVM_FFI_ICHECK(qkv_data.defined());
             TVM_FFI_ICHECK(o_data.defined());
             kv_cache->AttentionWithFusedQKV(layer_id, std::move(qkv_data), std::nullopt,
                                             std::move(o_data), sm_scale);
           })
      .def("vm.builtin.attention_kv_cache_self_attention",
           [](AttentionKVCache kv_cache, int64_t layer_id, double sm_scale, ffi::Tensor q_data,
              ffi::Tensor k_data, ffi::Tensor v_data, ffi::Tensor o_data, ffi::Tensor lse_data) {
             TVM_FFI_ICHECK(q_data.defined());
             TVM_FFI_ICHECK(k_data.defined());
             TVM_FFI_ICHECK(v_data.defined());
             TVM_FFI_ICHECK(o_data.defined());
             TVM_FFI_ICHECK(lse_data.defined());
             kv_cache->SelfAttention(layer_id, std::move(q_data), std::move(k_data),
                                     std::move(v_data), std::move(o_data), std::move(lse_data),
                                     sm_scale);
           })
      .def("vm.builtin.attention_kv_cache_cross_attention",
           [](AttentionKVCache kv_cache, int64_t layer_id, double sm_scale, ffi::Tensor q_data,
              ffi::Tensor o_data, ffi::Tensor lse_data) {
             TVM_FFI_ICHECK(q_data.defined());
             TVM_FFI_ICHECK(o_data.defined());
             TVM_FFI_ICHECK(lse_data.defined());
             kv_cache->CrossAttention(layer_id, std::move(q_data), std::move(o_data),
                                      std::move(lse_data), sm_scale);
           })
      .def("vm.builtin.attention_kv_cache_attention_with_shared_kv",
           [](AttentionKVCache kv_cache, int64_t source_layer_id, double sm_scale,
              ffi::Tensor q_data, ffi::Tensor current_k_data, ffi::Tensor current_v_data,
              ffi::Tensor o_data) {
             TVM_FFI_ICHECK(q_data.defined());
             TVM_FFI_ICHECK(current_k_data.defined());
             TVM_FFI_ICHECK(current_v_data.defined());
             TVM_FFI_ICHECK(o_data.defined());
             kv_cache->AttentionWithSharedKV(source_layer_id, std::move(q_data),
                                             std::move(current_k_data), std::move(current_v_data),
                                             std::move(o_data), sm_scale);
           })
      .def("vm.builtin.attention_kv_cache_append_mla_kv",
           [](AttentionKVCache kv_cache, int64_t layer_id, ffi::Tensor kv_data) {
             TVM_FFI_ICHECK(kv_data.defined());
             kv_cache->AppendMLAKV(layer_id, std::move(kv_data));
             return kv_cache;
           })
      .def("vm.builtin.attention_kv_cache_merge_attn_output_inplace",
           [](AttentionKVCache kv_cache, ffi::Tensor o_self_attn, ffi::Tensor lse_self_attn,
              ffi::Tensor o_cross_attn, ffi::Tensor lse_cross_attn) {
             TVM_FFI_ICHECK(o_self_attn.defined());
             TVM_FFI_ICHECK(lse_self_attn.defined());
             TVM_FFI_ICHECK(o_cross_attn.defined());
             TVM_FFI_ICHECK(lse_cross_attn.defined());
             return kv_cache->MergeAttnOutputInplace(
                 std::move(o_self_attn), std::move(lse_self_attn), std::move(o_cross_attn),
                 std::move(lse_cross_attn));
           });
}

// RNN State methods
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("vm.builtin.rnn_state_get", &RNNStateObj::Get)
      .def("vm.builtin.rnn_state_set",
           [](RNNState state, int64_t layer_id, int64_t state_id, ffi::Tensor data) {
             TVM_FFI_ICHECK(data.defined());
             state->Set(layer_id, state_id, data);
             return state;
           })
      .def_method("vm.builtin.rnn_state_debug_get", &RNNStateObj::DebugGet);
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
