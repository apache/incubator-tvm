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
 * \file src/runtime/vm/rnn_state.cc
 * \brief Runtime RNN state object for space state models.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstdint>
#include <vector>

#include "kv_state.h"

namespace tvm {
namespace runtime {
namespace vm {

//-----------------------------------------------------------------------------
// We keep the implementation private as they may subject to future changes.
//
// Users can interact with it through the runtime API function calls
//-----------------------------------------------------------------------------

class RNNStateImpObj : public RNNStateObj {
 private:
  /********************* Data Structures *********************/

  /*!
   * \brief The sequence structure in paged KV cache with common prefix support.
   * Each sequence contains one or more blocks to support common prefix.
   */
  struct Sequence {
    /*! \brief The total sequence length of the sequence. */
    int64_t seq_length = 0;
    /*! \brief The available history length for rolling back. */
    int64_t available_history_num = 0;
    /*! \brief The index of history slot in the storage. */
    int64_t history_slot_id = 0;
    /*! \brief The index of seq slot in the storage. */
    int64_t seq_slot_id;

    /*! \brief Constructor. */
    explicit Sequence(int64_t seq_slot_id) : seq_slot_id(seq_slot_id) {}

    static Sequence Fork(const Sequence& parent, int64_t seq_slot_id) {
      Sequence child = parent;
      child.seq_slot_id = seq_slot_id;
      return child;
    }
  };

  /********************* Configuration *********************/

  /*! \brief The number of layers in the model. */
  const int64_t num_layers_;
  /*! \brief The max number of sequences in the storage. */
  const int64_t reserved_num_seqs_;
  /*! \brief The number of states per layer. */
  const int64_t num_states_per_layer_;
  /*! \brief The max history length for rolling back. */
  const int64_t max_history_ = 1;
  /*!
   * \brief The init value for ALL layer in the storage.
   * The array has `num_states_per_layer_` Tensors
   */
  const ffi::Array<ffi::Tensor> init_layer_value_;

  /*! \brief We fix int32 to be the index dtype of auxiliary data. */
  const DLDataType dtype_aux_ = DLDataType{kDLInt, 32, 1};

  /******************* Storage Structures *******************/

  /*!
   * \brief The storages of space state models.
   * The array has `num_layers * num_states_per_layer_` Tensors,
   * each of them has layout `(num_seq, max_history, state_size)`.
   * \note As `num_states_per_layer_` may vary for different dtype and shape,
   * we use a 2D array to store the Tensors for each layer.
   */
  ffi::Array<ffi::Array<ffi::Tensor>> storages_;
  /*! \brief The list of ids of released seq slot for reuse. */
  std::vector<int64_t> free_slot_ids_;
  /*! \brief The mapping from sequence ids to sequences. */
  std::unordered_map<int64_t, Sequence> seq_map_;

  /****************** Auxiliary Arrays on Host ******************/

  /*! \brief The batch size of the current round of forwarding. */
  int64_t cur_batch_size_;
  /*! \brief The append lengths of the sequences in the current round of forwarding. */
  ffi::Shape cur_append_lengths_;
  /*! \brief The sequence ids of the current round of forwarding. */
  ffi::Shape cur_seq_ids_;

  /**************** Auxiliary Arrays on Device *****************/

  /*!
   * \brief A boolean flag indicating if the auxiliary arrays are dirty.
   * If it is dirty, an explicit "SyncAuxArrayToDevice" should be invoked.
   */
  bool dirty_aux_data_device_ = false;
  /*! \brief The device array of the sequence ids. */
  ffi::Tensor seq_slot_ids_device_;
  /*!
   * \brief The view of the device array of the sequence ids.
   * The view is used to reuse the memory but with different shape.
   */
  ffi::Optional<ffi::Tensor> seq_slot_ids_view_;
  /*! \brief The device array of the history slot ids. */
  ffi::Tensor history_slot_ids_device_;
  /*!
   * \brief The view of the device array of the history slot ids.
   * The view is used to reuse the memory but with different shape.
   */
  ffi::Optional<ffi::Tensor> history_slot_ids_view_;

  /******************* Interaction Functions *******************/

  /*!
   * \brief The function to get the state data from the storage.
   * The function signature is `f_get_(state, seq_slot_ids, history_slot_ids, out_data)`.
   * and return the contiguous batched state data.
   * \note Each state data per layer may have different dtype and shape, so we use a
   * different function for each state data.
   */
  ffi::Array<ffi::Function> f_gets_;
  /*!
   * \brief The function to set the state data to the storage.
   * The function signature is `f_set_(state, seq_slot_ids, history_slot_ids, data, max_history)`.
   * where `state` is the storage Tensor, `seq_slot_ids` and `history_slot_ids` are
   * 1-D int32 arrays of the same length as the batch size, and `data` is the input data.
   * \note The `history_slot_ids` is the slot of this round, but we need to write to the
   * slot of the next round.
   * \note Each state data per layer may have different dtype and shape, so we use a
   * different function for each state data.
   */
  ffi::Array<ffi::Function> f_sets_;

 public:
  /*! \brief Constructor. Take the cache configuration and initialize the Tensors. */
  explicit RNNStateImpObj(int64_t num_layers,                //
                          int64_t reserved_num_seqs,         //
                          int64_t max_history,               //
                          DLDevice device,                   //
                          ffi::Array<ffi::Function> f_gets,  //
                          ffi::Array<ffi::Function> f_sets,  //
                          ffi::Array<ffi::Tensor> init_layer_value)
      : num_layers_(num_layers),
        reserved_num_seqs_(reserved_num_seqs),
        num_states_per_layer_(init_layer_value.size()),
        max_history_(max_history),
        init_layer_value_(init_layer_value),
        seq_slot_ids_device_(tvm::runtime::Empty({reserved_num_seqs}, dtype_aux_, device)),
        history_slot_ids_device_(tvm::runtime::Empty({reserved_num_seqs}, dtype_aux_, device)),
        f_gets_(std::move(f_gets)),
        f_sets_(std::move(f_sets)) {
    // Allocate the storage for the space state models.
    storages_.reserve(num_layers_);
    for (int64_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
      ffi::Array<ffi::Tensor> layer_storages;
      layer_storages.reserve(num_states_per_layer_);
      for (int64_t state_id = 0; state_id < num_states_per_layer_; ++state_id) {
        ffi::Shape state_shape = init_layer_value[state_id].shape();
        std::vector<ffi::ShapeObj::index_type> storage_shape = {reserved_num_seqs, max_history};
        storage_shape.insert(storage_shape.end(), state_shape.begin(), state_shape.end());
        ffi::Tensor state_storage =
            tvm::runtime::Empty(storage_shape, init_layer_value[state_id].dtype(), device);
        layer_storages.push_back(state_storage);
      }
      storages_.push_back(layer_storages);
    }

    TVM_FFI_ICHECK_GT(max_history_, 0) << "At least 1 history slot to store the current state";

    Clear();
  }

  /*! \brief Reset the KV cache. */
  void Clear() final {
    seq_map_.clear();
    TVM_FFI_ICHECK(!storages_.empty());
    free_slot_ids_.clear();
    for (int64_t slot_id = reserved_num_seqs_ - 1; slot_id >= 0; --slot_id) {
      free_slot_ids_.push_back(slot_id);
    }
    dirty_aux_data_device_ = false;
  }

  /************** Interaction **************/

  void BeginForward(const ffi::Shape& seq_ids, const ffi::Shape& append_lengths,
                    const ffi::Optional<ffi::Shape>& opt_token_tree_parent_ptr) final {
    TVM_FFI_ICHECK_EQ(seq_ids.size(), append_lengths.size())
        << "The seq_ids size (" << seq_ids.size() << ") and append_lengths size ("
        << append_lengths.size() << ") mismatch.";

    if (opt_token_tree_parent_ptr.has_value()) {
      ffi::Shape token_tree_parent_ptr = opt_token_tree_parent_ptr.value();
      int matched_pos = 0;
      for (int64_t append_length : append_lengths) {
        for (int64_t i = 0; i < append_length; ++i) {
          TVM_FFI_ICHECK_EQ(token_tree_parent_ptr[matched_pos], i - 1)
              << "Unexpected token tree for RNN state. RNN state only supports chains as token "
                 "trees.";
          ++matched_pos;
        }
      }
    }
    cur_batch_size_ = seq_ids.size();
    cur_append_lengths_ = append_lengths;
    cur_seq_ids_ = seq_ids;

    if (dirty_aux_data_device_) {
      SyncAuxArrayToDevice();
    }
  }

  void EndForward() final {
    for (int64_t i = 0; i < cur_batch_size_; ++i) {
      int64_t seq_id = cur_seq_ids_[i];
      int64_t seq_length = cur_append_lengths_[i];
      auto it = seq_map_.find(seq_id);
      TVM_FFI_ICHECK(it != seq_map_.end())
          << "The sequence \"" << seq_id << "\" cannot be found in the space state storage.";
      it->second.seq_length += seq_length;
      if (seq_length > 1) {
        // We cannot rollback the prefill input
        it->second.available_history_num = 0;
      } else {
        it->second.available_history_num =
            std::min(it->second.available_history_num + 1, max_history_ - 1);
      }
      it->second.history_slot_id = (it->second.history_slot_id + 1) % max_history_;
    }
    // TODO(Siyuan): We need to update history_slot_id_device_ (on device) as well.
    // There are two ways to do this:
    // 1. Update history_slot_id_device_ on device directly through a explicit kernel
    // 2. Update history_slot_id on host and then sync to device.
    // We choose the second way for now for convenience. But the first way is more efficient.
    dirty_aux_data_device_ = true;
  }

  void Get(int64_t layer_id, int64_t state_id, ffi::Tensor o_data) final {
    TVM_FFI_ICHECK(o_data.defined());
    // The auxiliary data structure on device must have been synchronized.
    TVM_FFI_ICHECK(!dirty_aux_data_device_)
        << "The auxiliary arrays are not synchronized to device. Please call "
           "`BeginForward` to synchronize before calling `Get`.";
    TVM_FFI_ICHECK(cur_batch_size_ == static_cast<int64_t>(cur_seq_ids_.size()))
        << "The batch size is not consistent with the number of sequence ids.";
    TVM_FFI_ICHECK_GT(cur_batch_size_, 0) << "The curent batch size should be greater than 0.";
    // TODO(siyuan): support zero-copy when seq_len is one
    // Copy the state data to the return array.
    ffi::Tensor state = storages_[layer_id][state_id];
    f_gets_[state_id](state, seq_slot_ids_view_.value(), history_slot_ids_view_.value(), o_data);
  }

  void Set(int64_t layer_id, int64_t state_id, ffi::Tensor data) final {
    TVM_FFI_ICHECK(data.defined());
    // The auxiliary data structure on device must have been synchronized.
    TVM_FFI_ICHECK(!dirty_aux_data_device_)
        << "The auxiliary arrays are not synchronized to device. Please call "
           "`BeginForward` to synchronize before calling `Set`.";
    TVM_FFI_ICHECK(cur_batch_size_ == static_cast<int64_t>(cur_seq_ids_.size()))
        << "The batch size is not consistent with the number of sequence ids.";
    TVM_FFI_ICHECK_GT(cur_batch_size_, 0) << "The curent batch size should be greater than 0.";

    ffi::Tensor state = storages_[layer_id][state_id];
    f_sets_[state_id](state, seq_slot_ids_view_.value(), history_slot_ids_view_.value(), data);
  }

  ffi::Tensor DebugGet(int64_t layer_id, int64_t state_id, int64_t seq_id) {
    auto it = seq_map_.find(seq_id);
    TVM_FFI_ICHECK(it != seq_map_.end())
        << "The sequence \"" << seq_id << "\" cannot be found in the space state storage.";
    ffi::Tensor state = storages_[layer_id][state_id];
    int64_t seq_slot_id = it->second.seq_slot_id;
    int64_t history_slot_id = it->second.history_slot_id;

    std::vector<int64_t> shape{state.shape().begin() + 2, state.shape().end()};
    ffi::Tensor result = tvm::runtime::Empty(shape, state.dtype(), state.device());
    DLTensor copy_src = GetStatePtrBySeqHistory(layer_id, state_id, seq_slot_id, history_slot_id);
    DLTensor copy_dst = *result.GetDLTensorPtr();

    tvm::runtime::CopyFromTo(&copy_src, &copy_dst);
    return result;
  }

  /************** Sequence Management **************/

  void AddSequence(int64_t seq_id) final {
    TVM_FFI_ICHECK(seq_map_.find(seq_id) == seq_map_.end())
        << "The sequence \"" << seq_id << "\" is already in the space state storage.";
    int64_t seq_slot_id = GetFreeSlot();
    seq_map_.insert({seq_id, Sequence(seq_slot_id)});

    // Initialize the state data with the init value.
    for (int64_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
      for (int64_t state_id = 0; state_id < num_states_per_layer_; ++state_id) {
        DLTensor dst =
            GetStatePtrBySeqHistory(layer_id, state_id, seq_slot_id, /*history_slot_id=*/0);
        ffi::Tensor init = init_layer_value_[state_id];
        tvm::runtime::CopyFromTo(init.GetDLTensorPtr(), &dst);
      }
    }

    dirty_aux_data_device_ = true;
  }

  void RemoveSequence(int64_t seq_id) final {
    auto it = seq_map_.find(seq_id);
    TVM_FFI_ICHECK(it != seq_map_.end())
        << "The sequence \"" << seq_id << "\" cannot be found in the space state storage.";

    free_slot_ids_.push_back(it->second.seq_slot_id);
    seq_map_.erase(it);

    dirty_aux_data_device_ = true;
  }

  void ForkSequence(int64_t parent_seq_id, int64_t child_seq_id, int64_t fork_pos = -1) final {
    auto parent_it = seq_map_.find(parent_seq_id);
    TVM_FFI_ICHECK(parent_it != seq_map_.end()) << "The parent sequence \"" << parent_seq_id
                                                << "\" cannot be found in space state storage.";
    TVM_FFI_ICHECK(seq_map_.find(child_seq_id) == seq_map_.end())
        << "The child sequence \"" << child_seq_id << "\" is already in the space state storage.";

    // Create a child block with the parent block pointer.
    int64_t child_slot_id = GetFreeSlot();
    seq_map_.insert({child_seq_id, Sequence::Fork(parent_it->second, child_slot_id)});

    // Copy the parent state data to the child state data.
    int64_t parent_slot_id = parent_it->second.seq_slot_id;
    for (int64_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
      for (int64_t state_id = 0; state_id < num_states_per_layer_; ++state_id) {
        DLTensor copy_src = GetStatePtrBySeq(layer_id, state_id, parent_slot_id);
        DLTensor copy_dst = GetStatePtrBySeq(layer_id, state_id, child_slot_id);
        tvm::runtime::CopyFromTo(&copy_src, &copy_dst);
      }
    }
    dirty_aux_data_device_ = true;
  }

  void PopN(int64_t seq_id, int32_t n) final {
    auto it = seq_map_.find(seq_id);
    TVM_FFI_ICHECK(it != seq_map_.end())
        << "The sequence \"" << seq_id << "\" cannot be found in space state.";
    TVM_FFI_ICHECK_GE(n, 0) << "The length of rolling back " << n << " cannot be negative.";
    TVM_FFI_ICHECK_LE(n, it->second.available_history_num)
        << "The sequence only has " << it->second.available_history_num
        << " available history in the space state storage, while the length of rollback is " << n
        << " which exceeds the sequence length.";

    it->second.seq_length -= n;
    it->second.available_history_num -= n;
    it->second.history_slot_id = (it->second.history_slot_id - n + max_history_) % max_history_;
    dirty_aux_data_device_ = true;
  }

 private:
  /*! \brief Get a new free block and return its index. */
  int32_t GetFreeSlot() {
    TVM_FFI_ICHECK(!free_slot_ids_.empty())
        << "The Sequence slot is full, cannot accept new sequence.";
    int32_t seq_slot_id = free_slot_ids_.back();
    free_slot_ids_.pop_back();
    return seq_slot_id;
  }

  DLTensor GetStatePtrBySeqHistory(int64_t layer_id, int64_t state_id, int64_t seq_slot_id,
                                   int64_t history_slot_id) {
    ffi::Tensor state = storages_[layer_id][state_id];
    int64_t state_size = 1;
    for (int64_t i = 2; i < state.ndim(); ++i) {
      state_size *= state.shape()[i];
    }
    int64_t elem_offset = (seq_slot_id * max_history_ + history_slot_id) * state_size;
    // Create a new DLTensor with the same shape and dtype as the state.
    DLTensor _state = *(state.GetDLTensorPtr());
    _state.byte_offset = elem_offset * state.dtype().bits / 8;
    _state.ndim = state.ndim() - 2;
    _state.shape = const_cast<int64_t*>(_state.shape + 2);
    _state.strides = const_cast<int64_t*>(_state.strides + 2);
    return _state;
  }

  DLTensor GetStatePtrBySeq(int64_t layer_id, int64_t state_id, int64_t seq_slot_id) {
    ffi::Tensor state = storages_[layer_id][state_id];
    int64_t state_size = 1;
    for (int64_t i = 1; i < state.ndim(); ++i) {
      state_size *= state.shape()[i];
    }
    int64_t elem_offset = seq_slot_id * state_size;
    // Create a new DLTensor with the same shape and dtype as the state.
    DLTensor _state = *(state.GetDLTensorPtr());
    _state.byte_offset = elem_offset * state.dtype().bits / 8;
    _state.ndim = state.ndim() - 1;
    _state.shape = const_cast<int64_t*>(_state.shape + 1);
    _state.strides = const_cast<int64_t*>(_state.strides + 1);
    return _state;
  }

  /*!
   * \brief Synchronize auxiliary arrays to device.
   * \note This method resets the dirty flag to false, and needs to be
   * invoked before running attention computation on device.
   */
  void SyncAuxArrayToDevice() {
    auto fcopy_from_vec = [](ffi::Tensor array, std::vector<int32_t> vec_data) {
      TVM_FFI_ICHECK(array.defined());
      DLTensor copy_dst = *array.GetDLTensorPtr();
      DLTensor copy_src;
      copy_src.data = vec_data.data();
      copy_src.device = Device{kDLCPU, 0};
      copy_src.ndim = 1;
      copy_src.dtype = array.dtype();
      copy_src.shape = array.GetDLTensorPtr()->shape;
      copy_src.strides = array.GetDLTensorPtr()->strides;
      copy_src.byte_offset = 0;
      tvm::runtime::CopyFromTo(&copy_src, &copy_dst);
    };

    std::vector<int32_t> seq_slot_ids;
    std::vector<int32_t> history_slot_ids;
    seq_slot_ids.reserve(cur_batch_size_);
    history_slot_ids.reserve(cur_batch_size_);
    for (int64_t seq_id : cur_seq_ids_) {
      auto it = seq_map_.find(seq_id);
      TVM_FFI_ICHECK(it != seq_map_.end())
          << "The sequence \"" << seq_id << "\" cannot be found in the space state storage.";
      const Sequence& seq = it->second;
      seq_slot_ids.push_back(seq.seq_slot_id);
      history_slot_ids.push_back(seq.history_slot_id);
    }
    seq_slot_ids_view_ =
        tvm::runtime::CreateView(seq_slot_ids_device_, {cur_batch_size_}, dtype_aux_);
    history_slot_ids_view_ =
        tvm::runtime::CreateView(history_slot_ids_device_, {cur_batch_size_}, dtype_aux_);

    fcopy_from_vec(seq_slot_ids_view_.value(), seq_slot_ids);
    fcopy_from_vec(history_slot_ids_view_.value(), history_slot_ids);

    // Reset the dirty flag to false.
    dirty_aux_data_device_ = false;
  }

 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.vm.RNNStateImp", RNNStateImpObj, RNNStateObj);
};

//-------------------------------------------------
//  Register runtime functions
//-------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "vm.builtin.rnn_state_create", [](int64_t num_layers,                //
                                        int64_t reserved_num_seqs,         //
                                        int64_t max_history,               //
                                        ffi::Array<ffi::Function> f_gets,  //
                                        ffi::Array<ffi::Function> f_sets,  //
                                        ffi::Array<ffi::Tensor> init_layer_value) {
        TVM_FFI_ICHECK_GT(num_layers, 0) << "The number of layers should be greater than 0.";
        TVM_FFI_ICHECK_GT(reserved_num_seqs, 0)
            << "The number of reserved sequences should be greater than 0.";
        TVM_FFI_ICHECK_GE(max_history, 0)
            << "The maximum history length should be greater or equal than 0.";
        TVM_FFI_ICHECK_GT(init_layer_value.size(), 0)
            << "The number of states per layer should be greater than 0.";
        for (const auto& value : init_layer_value) {
          TVM_FFI_ICHECK(value.defined());
        }
        Device device = init_layer_value[0].device();
        for (const ffi::Tensor& state : init_layer_value) {
          TVM_FFI_ICHECK(state.device().device_type == device.device_type &&
                         state.device().device_id == device.device_id)
              << "The device type of all states should be the same.";
        }
        TVM_FFI_ICHECK_EQ(f_gets.size(), init_layer_value.size())
            << "The number of state getters should be the same as the number of states per layer, "
            << "but got " << f_gets.size() << " and " << init_layer_value.size()
            << " respectively.";
        TVM_FFI_ICHECK_EQ(f_sets.size(), init_layer_value.size())
            << "The number of state setters should be the same as the number of states per layer, "
            << "but got " << f_sets.size() << " and " << init_layer_value.size()
            << " respectively.";
        ffi::ObjectPtr<RNNStateImpObj> n = ffi::make_object<RNNStateImpObj>(
            num_layers, reserved_num_seqs, max_history, device, std::move(f_gets),
            std::move(f_sets), init_layer_value);
        return ffi::GetRef<RNNState>(n.get());
      });
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
