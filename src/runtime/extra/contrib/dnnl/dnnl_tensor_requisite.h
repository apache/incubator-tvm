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
 * \file src/runtime/contrib/dnnl/dnnl_tensor_requisite.cc
 * \brief Helper TR wrapper to simplify tensors processing
 */

#ifndef TVM_RUNTIME_CONTRIB_DNNL_DNNL_TENSOR_REQUISITE_H_
#define TVM_RUNTIME_CONTRIB_DNNL_DNNL_TENSOR_REQUISITE_H_

#include <dlpack/dlpack.h>
#include <tvm/tirx/op/memory.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// TODO(@apeskov): Have to mute warning from dnnl headers.
//  -Wzero-as-null-pointer-constant and -Wdocumentation-unknown-command
#include <dnnl.hpp>

#include "dnnl_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace utils;

/*!
 * \brief Helper object to simplify tensor transformation description.
 *
 * Allow to specify original source tensor and future actions which should be applied to it.
 * Can be treated as sequence of reordering or reinterpretation of original source tensor.
 * Finally TR can be solved as proper interpretation of source memory buffer, or sequence of
 * dnnl::reorder operators which will provide desired data.
 *
 * \note Empty TR object allow any manipulation. Empty TR will be returned.
 *
 * \sa TensorRegistry
 *
 * Example:
 * \code
 *   dnnl::memory src_mem = ...;  // 5D tensor, shape {5, 2, 128, 128, 8}
 *
 *   // Construct TR
 *   auto tr = TensorRequisite.AsIs(src_mem, eid);  // 5D
 *
 *   // describe sequence of layout transformation
 *   tr = tr.TreatAs("ABCD8b");  // 4D
 *   tr = tr.Permute({0, 2, 3, 1});  // Permute axes NCHW -> NHWC
 *   tr = tr.Crop({1, 128, 128, 16}, {0, 0, 0});  // extract first batch element
 *   tr = tr.Squeeze(); // 1D
 *
 *   // register TR
 *   TensorRegistry t_reg;
 *   auto t_id = t_reg.register(tr);
 *
 *   // Get final dnnl::memory object
 *   auto solver = t_reg.MakeSolver(ext_tensor_provider);
 *   auto mem = solver(t_id);
 * \endcode
 *
 */
class TensorRequisite {
 public:
  using Tid = uint32_t;
  static constexpr Tid kUndefinedTid = std::numeric_limits<uint32_t>::max() - 1;

  /*! \brief Empty constructor */
  TensorRequisite() {}

  /*! \brief Construct TR on top of existing memory object */
  static TensorRequisite AsIs(const dnnl::memory& mem, Tid id = kUndefinedTid) {
    auto res = AsIs(mem.get_desc(), id);
    if (mem.get_data_handle() != nullptr) res.mem_ = mem;
    return res;
  }

  /*! \brief Construct TR on top of existing memory descriptor object */
  static TensorRequisite AsIs(const dnnl::memory::desc& desc, Tid id = kUndefinedTid) {
    return {desc, {}, false, {}, id, false};
  }

  /*! \brief return logical shape of tensor */
  dnnl::memory::dims dims() const { return t_desc_.dims(); }

  /*! \brief return data type of tensor */
  dnnl::memory::data_type data_type() const { return t_desc_.data_type(); }

  /*! \brief return tensor desc */
  dnnl::memory::desc desc() const { return t_desc_; }

  Tid eid() const {
    auto res = kUndefinedTid;

    if (!defined()) {
      res = kUndefinedTid;
    } else if (eid_ == kUndefinedTid) {
      if (orig_) {
        res = orig_->eid();
      } else {
        res = kUndefinedTid;
      }
    } else {
      res = eid_;
    }
    return res;
  }

  /*! \brief Make TR with backward dataflow */
  TensorRequisite Backward() const {
    if (!defined()) return *this;
    TVM_FFI_ICHECK(orig_ == nullptr);
    return {t_desc_, orig_, reinterpret_, mem_, eid_, true};
  }

  /*! \brief Produce TR with permuted axes */
  TensorRequisite Permute(const std::vector<int>& permutation) const {
    if (!defined()) return *this;  // nothing for empty TR

    auto orig = std::make_shared<TensorRequisite>(*this);
    // reinterpret memory buffer with new strides
    auto desc = t_desc_.permute_axes(permutation);
    return {desc, orig, true, {}, kUndefinedTid, reverse_data_flow_};
  }

  /*! \brief Produce TR with reinterpret data of original tr */
  TensorRequisite Reshape(const dnnl::memory::dims& shape) const {
    if (!defined()) return *this;  // nothing for empty TR
    if (t_desc_.dims() == shape) return *this;

    auto orig = std::make_shared<TensorRequisite>(*this);
    // reinterpret memory buffer with new strides
    auto desc = t_desc_.reshape(shape);
    return {desc, orig, true, {}, kUndefinedTid, reverse_data_flow_};
  }

  /*! \brief Produce TR with broadcasted values */
  TensorRequisite Broadcast(const dnnl::memory::dims& shape) const {
    if (!defined()) return *this;  // nothing for empty TR
    if (t_desc_.dims() == shape) return *this;
    TVM_FFI_ICHECK(!reverse_data_flow_);

    auto orig = std::make_shared<TensorRequisite>(*this);

    // numpy like broadcast
    auto extended_dims = t_desc_.dims();
    auto one_filled = dnnl::memory::dims(shape.size() - extended_dims.size(), 1);
    extended_dims.insert(extended_dims.begin(), one_filled.begin(), one_filled.end());
    auto desc = t_desc_.reshape(extended_dims);
    for (size_t i = 0; i < extended_dims.size(); i++) {
      if (extended_dims[i] == shape[i]) continue;
      TVM_FFI_ICHECK(extended_dims[i] == 1);
      TVM_FFI_ICHECK(desc.data.dims[i] == desc.data.padded_dims[i]);

      desc.data.dims[i] = shape[i];
      desc.data.padded_dims[i] = shape[i];
      desc.data.format_desc.blocking.strides[i] = 0;
    }

    // reinterpret memory buffer with new strides
    return {desc, orig, true, {}, kUndefinedTid, reverse_data_flow_};
  }

  /*! \brief Produce TR with sub memory view (ROI) */
  TensorRequisite Crop(const dnnl::memory::dims& shape, const dnnl::memory::dims& offset) const {
    if (!defined()) return *this;  // nothing for empty TR

    TVM_FFI_ICHECK_EQ(shape.size(), t_desc_.dims().size());
    TVM_FFI_ICHECK_EQ(offset.size(), t_desc_.dims().size());

    auto orig = std::make_shared<TensorRequisite>(*this);
    // reinterpret memory buffer with new strides
    auto desc = t_desc_.submemory_desc(shape, offset, /*allow_empty=*/true);

    // Originally DNNL implementation is very limited. Let's slightly enhance it.
    if (!desc && t_desc_.data.format_kind == dnnl_blocked) {
      bool offset_is_zero =
          std::all_of(offset.begin(), offset.end(), [](auto el) { return el == 0; });

      dnnl::memory::dims block_sizes(t_desc_.dims().size(), 1);
      for (int i = 0; i < t_desc_.data.format_desc.blocking.inner_nblks; i++)
        block_sizes[t_desc_.data.format_desc.blocking.inner_idxs[i]] *=
            t_desc_.data.format_desc.blocking.inner_blks[i];

      bool shape_reduction_less_than_block = true;
      for (int i = 0; i < t_desc_.data.ndims; i++) {
        shape_reduction_less_than_block &= t_desc_.data.dims[i] - shape[i] < block_sizes[i];
      }

      // This is auto padded case. Just update dims value.
      if (offset_is_zero && shape_reduction_less_than_block) {
        desc = t_desc_;
        std::copy(shape.begin(), shape.end(), desc.data.dims);
      }
    }

    TVM_FFI_ICHECK(desc);

    return {desc, orig, true, {}, kUndefinedTid, reverse_data_flow_};
  }

  /*! \brief Produce TR with squeeze shape */
  TensorRequisite Squeeze(const dnnl::memory::dims& dims_to_squeeze = {}) const {
    if (!defined()) return *this;  // nothing for empty TR

    dnnl::memory::dims squeezed_dims;
    if (dims_to_squeeze.empty()) {
      for (auto d : t_desc_.dims())
        if (d != 1) squeezed_dims.push_back(d);
    } else {
      for (size_t i = 0; i < t_desc_.dims().size(); i++)
        if (std::find(dims_to_squeeze.begin(), dims_to_squeeze.end(), i) == dims_to_squeeze.end())
          squeezed_dims.push_back(t_desc_.dims()[i]);
    }

    if (squeezed_dims.empty()) squeezed_dims = {1};

    auto orig = std::make_shared<TensorRequisite>(*this);
    // reinterpret memory buffer with new strides
    auto desc = t_desc_.reshape(squeezed_dims);
    return {desc, orig, true, {}, kUndefinedTid, reverse_data_flow_};
  }

  /*! \brief Produce TR with specified layout descriptor */
  TensorRequisite RequestLayout(dnnl::memory::desc desc) const {
    if (!defined()) return *this;  // nothing for empty TR

    // If it's the same desc just return self
    if (desc == t_desc_) return *this;

    TVM_FFI_ICHECK(t_desc_.dims() == desc.dims()) << "Requested layout is not compatible with "
                                                     "presented shape";

    auto orig = std::make_shared<TensorRequisite>(*this);
    return {desc, orig, false, {}, kUndefinedTid, reverse_data_flow_};
  }

  /*! \brief Define which logical dims ordering is default for particular layout string. */
  static std::string DefaultLogicLayoutFor(const std::string& layout) {
    // Rank is all non digit marked dims
    auto it = layout.begin();
    while (it != layout.end() && !std::isdigit(*it)) it++;
    int rank = std::distance(layout.begin(), it);

    static const std::vector<std::string> sparse_dims = {"W", "HW", "DHW"};
    if (layout.find("N") != std::string::npos) return "NC" + sparse_dims[rank - 3];
    if (layout.find("G") != std::string::npos) return "GOI" + sparse_dims[rank - 4];
    if (layout.find("O") != std::string::npos) return "OI" + sparse_dims[rank - 3];

    TVM_FFI_THROW(InternalError) << "Unknown layout " << layout
                                 << "There is no default scheme to handle it";
  }

  /*!
   * \brief Treat TR shape as described in layout string.
   *
   * Blocked dimensions will be concatenated and put into proper shape position corresponding to  .
   * resulting_layout_logic argument. If desired logic layout was not provided it will be deduced
   * automatically based on some internal heuristics.
   *
   * Limitation 1. Blocking dims should be dense. Dims marked with digits use natural strides.
   * Limitation 2. Blocking dims are innermost. Dims marked like 8c, 4o goes after regular
   *               dimensions. NC8cHW4h4cD is not valid tensor in terms of DNNL. And cannot be
   *               achieved with memory reinterpretation, so data copy is required. Proper layout
   *               looks like NCHWD_8c4h4c, first part is outer dims, second digits marked part is
   *               innermost.
   */
  TensorRequisite TreatAs(const std::string& layout, std::string desired_logic_layout = "") const {
    if (!defined()) return *this;
    if (desired_logic_layout.empty()) desired_logic_layout = DefaultLogicLayoutFor(layout);

    const auto origin_dims = dims();

    // split layout string to tokens {size, tag} like {16, 'C'}, {4, 'O'}
    std::vector<std::pair<int, char>> layout_tokens;
    for (auto it = layout.begin(); it != layout.end();) {
      auto start = it;
      while (std::isdigit(*it)) it++;
      int blk_size = start == it ? -1 : std::stoi(std::string{start, it});
      layout_tokens.push_back({blk_size, std::toupper(*it)});
      it++;
    }

    // check applicability of layout
    auto it = layout_tokens.begin();
    while (it != layout_tokens.end() && it->first == -1) it++;
    int rank = std::distance(layout_tokens.begin(), it);
    while (it != layout_tokens.end()) {
      TVM_FFI_ICHECK_NE(it->first, -1) << "DNNL limitation. Blocking dims should be innermost. "
                                       << "But received layout is " << layout;
      it++;
    }

    TVM_FFI_ICHECK_EQ(layout_tokens.size(), origin_dims.size());
    TVM_FFI_ICHECK_EQ(rank, desired_logic_layout.size()) << layout;

    std::vector<std::pair<int, char>> outermost_tokens(layout_tokens.begin(),
                                                       layout_tokens.begin() + rank);
    std::vector<std::pair<int, char>> innermost_tokens(layout_tokens.begin() + rank,
                                                       layout_tokens.end());
    // define dim resulting dim positions
    std::map<char, int> dim_position_by_tag;
    for (size_t i = 0; i < desired_logic_layout.size(); i++)
      dim_position_by_tag[std::toupper(desired_logic_layout[i])] = i;

    // Construct resulting desc by modifying original one
    dnnl::memory::desc res_desc = t_desc_;

    memset(&res_desc.data.format_desc.blocking, 0, sizeof(res_desc.data.format_desc.blocking));
    std::fill(res_desc.data.dims, res_desc.data.dims + DNNL_MAX_NDIMS, 0);
    std::fill(res_desc.data.padded_dims, res_desc.data.padded_dims + DNNL_MAX_NDIMS, 0);

    res_desc.data.ndims = rank;
    res_desc.data.format_desc.blocking.inner_nblks = innermost_tokens.size();

    auto res_dims = res_desc.data.dims;
    auto res_strides = res_desc.data.format_desc.blocking.strides;
    auto res_inner_blks = res_desc.data.format_desc.blocking.inner_blks;
    auto res_inner_idxs = res_desc.data.format_desc.blocking.inner_idxs;

    std::fill(res_dims, res_dims + rank, 1);

    int orig_dim_idx = 0;
    for (const auto& p : outermost_tokens) {
      auto tag = p.second;
      auto dim_size = origin_dims[orig_dim_idx];

      auto result_dim_position = dim_position_by_tag[tag];
      res_dims[result_dim_position] *= dim_size;
      res_strides[result_dim_position] = t_desc_.data.format_desc.blocking.strides[orig_dim_idx];
      orig_dim_idx++;
    }
    for (const auto& p : innermost_tokens) {
      auto tag = p.second;
      auto dim_size = origin_dims[orig_dim_idx];
      auto result_dim_position = dim_position_by_tag[tag];
      TVM_FFI_ICHECK_EQ(p.first, dim_size)
          << "Blocking layout is not applicable to tensor with shape: " << origin_dims
          << ". Requested layout is " << layout;

      res_dims[result_dim_position] *= dim_size;
      *res_inner_blks++ = dim_size;
      *res_inner_idxs++ = result_dim_position;
      orig_dim_idx++;
    }

    // Assume tensor is dense. There is no additional padding.
    std::copy(res_desc.data.dims, res_desc.data.dims + rank, res_desc.data.padded_dims);

    if (t_desc_ == res_desc) return *this;

    auto orig = std::make_shared<TensorRequisite>(*this);
    return {res_desc, orig, true, {}, kUndefinedTid, reverse_data_flow_};
  }

  /*!
   * \brief Produce TR with unspecified layout.
   *
   * Cannot be registered in TensorRegistry. Only for querying DNNL for preferred layouts.
   */
  TensorRequisite LayoutAny() const {
    auto orig = std::make_shared<TensorRequisite>(*this);
    // Recreate tensor desc with layout 'any'
    dnnl::memory::desc any_desc{t_desc_.dims(), t_desc_.data_type(), dnnl::memory::format_tag::any};
    return {any_desc, orig, false, {}, kUndefinedTid, reverse_data_flow_};
  }

  /*! \brief Check is TR is constant. */
  bool IsConstant() const {
    if (orig_) return orig_->IsConstant();
    return mem_.operator bool();
  }

  /*! \brief Check is tensor is scalar. */
  bool IsScalar() const { return t_desc_.dims().size() == 1 && t_desc_.dims()[0] == 1; }

  /*! \brief Return const data memory if available. */
  dnnl::memory GetConstData() const {
    if (mem_) return mem_;
    if (!orig_) return {};

    if (auto orig_const_data = orig_->GetConstData()) {
      if (reinterpret_) {
        return {t_desc_, orig_const_data.get_engine(), orig_const_data.get_data_handle()};
      } else {
        auto eng = orig_const_data.get_engine();
        auto res = dnnl::memory{t_desc_, eng};
        dnnl::reorder(orig_const_data, res).execute(dnnl::stream(eng), orig_const_data, res);
        return res;
      }
    }
    return {};
  }

  /*!
   * \brief Return const data memory in form of vector.
   *
   * Same as GetConstData but use std::vector instead of dnnl::memory. Works only for 1D tensor
   * and scalar TRs. Useful for specification of 1D DNNL attributes like zero_point or
   * per_channel_scale
   */
  template <typename T>
  std::vector<T> GetConstDataLikeVec() const {
    auto const_data = GetConstData();
    auto desc = const_data.get_desc();
    TVM_FFI_ICHECK(desc.data_type() == utils::DnnlDType<T>());
    TVM_FFI_ICHECK(desc.dims().size() == 1);

    auto size = desc.get_size() / sizeof(T);
    auto ptr = static_cast<T*>(const_data.get_data_handle());

    return std::vector<T>(ptr, ptr + size);
  }

  /*! \brief Get value of constant scalar tensor if possible. */
  template <typename T>
  T GetConstScalarData() const {
    TVM_FFI_ICHECK(IsConstant());
    TVM_FFI_ICHECK(IsScalar());
    auto const_data = GetConstData();
    auto desc = const_data.get_desc();
    TVM_FFI_ICHECK(desc.data_type() == utils::DnnlDType<T>());

    auto ptr = static_cast<T*>(const_data.get_data_handle());
    return *ptr;
  }

  /*! \brief Check if tensor is not empty. */
  bool defined() const { return !t_desc_.is_zero(); }

  /*! \brief Same as defined */
  operator bool() const { return defined(); }

  /*!
   * \brief Check if tensor represent a reversed data flow.
   * Useful for describing output processing
   */
  bool IsReversed() const { return reverse_data_flow_; }

 private:
  TensorRequisite(const dnnl::memory::desc& t_desc, const std::shared_ptr<TensorRequisite>& orig,
                  bool reinterpret, const dnnl::memory& const_mem, uint32_t eid,
                  bool reverse_data_flow)
      : t_desc_(t_desc),
        orig_(orig),
        reinterpret_(reinterpret),
        mem_(const_mem),
        eid_(eid),
        reverse_data_flow_(reverse_data_flow) {
    if (mem_) TVM_FFI_ICHECK(!orig_ && !reverse_data_flow_ && eid_ == kUndefinedTid);
    if (eid_ != kUndefinedTid) TVM_FFI_ICHECK(!orig_);
  }

  /* Descriptor of particular tensor  */
  dnnl::memory::desc t_desc_ = {};
  /* Parent TR object which is referred from this TR */
  std::shared_ptr<TensorRequisite> orig_ = {};
  /* Flag to specify which action should be done with orig TR, reordering or reinterpretation */
  bool reinterpret_ = false;
  /* Const memory object if available */
  dnnl::memory mem_ = {};
  /* Entry ID of tensor if available */
  uint32_t eid_ = kUndefinedTid;

  /*
   * Flag to describe reverse data flow case
   * All operation on queue will be executed in reverse order. Actual for dst tensor description
   */
  bool reverse_data_flow_ = false;

  friend class TensorRegistry;
};

/*!
 * \brief The registry of tensors. Implement matching of provided TRs and real memory buffers.
 *
 * Registration of TR performed by calling method Register(), which will return ArgId object.
 * ArgId can be mapped to real memory via memory solver created by method MakeSolver().
 */
class TensorRegistry {
 private:
  enum ArgReqFlag {
    CONST,        /// < Constant tensor. ExecutionCTX independent
    TMP_STORAGE,  /// < Intermediate tensors. Stored inside TensorRegistry. Inaccessible outside
    EXT_EID,      /// < External data. Input or Output.
  };

 public:
  struct ArgId {
    TensorRegistry::ArgReqFlag flag_;
    uint32_t idx_;
  };

  using Action = std::tuple<dnnl::primitive, std::unordered_map<int, ArgId>>;
  using ActionQue = std::vector<Action>;
  using DLTensorProvider = std::function<const DLTensor*(uint32_t)>;
  using MemSolver = std::function<const dnnl::memory(ArgId)>;

  TensorRegistry() = default;
  TensorRegistry(const dnnl::engine& eng, const std::set<uint32_t>& ext_io_eid)
      : tmp_mem_collection_(1), ext_io_eid_(ext_io_eid), eng_(eng), stream_(eng) {}

  /*!
   * \brief Register TR to registry
   *
   * Resolution of TR may lead to introduction of intermediate memory buffers and additional
   * transformation actions which should be performed before or after usage of corresponding memory
   * buffer. Additional actions will be append to provided actions queue. Corresponding to
   * tr.IsReversed() value actions should be executed before or after usage of resulting ArgId.
   *
   * \param tr tensor requisite sequence to register
   * \param action resulting action queue. If TR resolution is required execution of some
   *               transformation actions they will be put here
   * \return associated ArgId. Should be used as argument for MemSolver.
   */
  ArgId Register(const TensorRequisite& tr, ActionQue* action) {
    // 1) Constant tensor. Direct reference
    if (auto const_data = tr.GetConstData()) {
      auto idx = const_mem_collection_.size();
      const_mem_collection_.push_back(const_data);
      return MakeArgReq(ArgReqFlag::CONST, static_cast<uint32_t>(idx));
    }

    // 2) EID mapped tensor. Direct reference
    if (tr.eid_ != TensorRequisite::kUndefinedTid) {
      if (ext_io_eid_.count(tr.eid_) == 0) {  // Not IO tensor, means it's intermediate
        if (eid2idx_tmp_.count(tr.eid_)) {
          auto idx = eid2idx_tmp_.at(tr.eid_);
          return MakeArgReq(ArgReqFlag::TMP_STORAGE, idx);
        } else {
          // register himself
          auto idx = tmp_mem_collection_.size();
          tmp_mem_collection_.push_back(tr.t_desc_);
          eid2idx_tmp_[tr.eid_] = idx;
          return MakeArgReq(ArgReqFlag::TMP_STORAGE, static_cast<uint32_t>(idx));
        }
      } else {
        auto idx = ext_mem_collection_.size();
        ext_mem_collection_.push_back({tr.eid_, tr.t_desc_});
        return MakeArgReq(ArgReqFlag::EXT_EID, static_cast<uint32_t>(idx));
      }
    }

    // 3) Tensors with transform actions
    if (tr.orig_) {
      // recursive register of orig TR
      auto orig_arg_req = Register(*tr.orig_, action);
      if (tr.reinterpret_) {
        return RegisterReinterpret(orig_arg_req, tr.t_desc_);
      } else {
        return RegisterReorder(orig_arg_req, tr.t_desc_, tr.reverse_data_flow_, action);
      }
    }

    // 4) Scratchpad
    TVM_FFI_ICHECK(!tr.orig_ && !tr.mem_ && tr.eid_ == TensorRequisite::kUndefinedTid);
    auto idx = tmp_mem_collection_.size();
    tmp_mem_collection_.push_back(tr.t_desc_);
    tmp_mem_mapping_[idx] = 0;  // zero position tmp mem object is reserved for scratchpads

    auto scratchpad_size = tr.t_desc_.get_size();
    auto glob_scratchpad_size = tmp_mem_collection_[0].get_size();
    if (scratchpad_size > glob_scratchpad_size) {
      tmp_mem_collection_[0] =
          dnnl::memory::desc({static_cast<dnnl::memory::dim>(scratchpad_size)},
                             dnnl::memory::data_type::u8, dnnl::memory::format_tag::a);
    }
    return MakeArgReq(TMP_STORAGE, static_cast<uint32_t>(idx));
  }

  /*!
   * \brief Construct memory solver for all registered TRs.
   * \param ext_provider callback to resolve external IO buffers
   * \return memory solver object to match ArgId to dnnl::memory objects
   */
  MemSolver MakeSolver(const DLTensorProvider& ext_provider) const {
    return MemSolverImpl(eng_, ext_provider, const_mem_collection_, ext_mem_collection_,
                         tmp_mem_collection_, tmp_mem_mapping_);
  }

  void MarkInplace(const TensorRequisite& tr, const TensorRequisite& shared) {
    const auto tr_id = tr.eid();
    TVM_FFI_ICHECK(tr_id != TensorRequisite::kUndefinedTid);
    const auto shared_id = shared.eid();
    TVM_FFI_ICHECK(shared_id != TensorRequisite::kUndefinedTid);
    eid2idx_tmp_[tr_id] = eid2idx_tmp_[shared_id];
  }

 private:
  ArgId RegisterReinterpret(ArgId src_ar, const dnnl::memory::desc& desc) {
    switch (src_ar.flag_) {
      case TMP_STORAGE: {
        auto idx = tmp_mem_collection_.size();
        tmp_mem_collection_.push_back(desc);
        tmp_mem_mapping_[idx] = src_ar.idx_;
        return MakeArgReq(TMP_STORAGE, idx);
      }
      case EXT_EID: {
        auto ext_req = ext_mem_collection_[src_ar.idx_];
        auto idx = ext_mem_collection_.size();
        ext_mem_collection_.push_back({ext_req.first, desc});
        return MakeArgReq(EXT_EID, idx);
      }
      default:
        TVM_FFI_THROW(InternalError) << "Unknown case";
    }
    return {};
  }

  ArgId RegisterReorder(ArgId src_ar, const dnnl::memory::desc& desc, bool reverse_data_flow,
                        ActionQue* action) {
    TVM_FFI_ICHECK(src_ar.flag_ == TMP_STORAGE || src_ar.flag_ == EXT_EID);

    auto src_desc = src_ar.flag_ == TMP_STORAGE ? tmp_mem_collection_[src_ar.idx_]
                                                : ext_mem_collection_[src_ar.idx_].second;
    auto idx = tmp_mem_collection_.size();
    tmp_mem_collection_.push_back(desc);
    auto dst_ar = MakeArgReq(TMP_STORAGE, idx);

    // reorder action submit
    if (reverse_data_flow) {
      auto reorder_pd = dnnl::reorder::primitive_desc(eng_, desc, eng_, src_desc);
      action->insert(action->begin(),
                     {dnnl::reorder(reorder_pd), {{DNNL_ARG_FROM, dst_ar}, {DNNL_ARG_TO, src_ar}}});
    } else {
      auto reorder_pd = dnnl::reorder::primitive_desc(eng_, src_desc, eng_, desc);
      action->push_back(
          {dnnl::reorder(reorder_pd), {{DNNL_ARG_FROM, src_ar}, {DNNL_ARG_TO, dst_ar}}});
    }
    return dst_ar;
  }
  /*! \brief Implementation of memory solver */
  class MemSolverImpl {
   public:
    MemSolverImpl(const dnnl::engine& eng, const DLTensorProvider& ext_data_provider,
                  const std::vector<dnnl::memory>& const_mems,
                  const std::vector<std::pair<uint32_t, dnnl::memory::desc>>& ext_mems,
                  const std::vector<dnnl::memory::desc>& tmp_mem_descs,
                  const std::map<size_t, size_t>& tmp_mem_mapping)
        : eng_(eng),
          ext_data_provider_(ext_data_provider),
          const_mems_(const_mems),
          ext_mems_(ext_mems) {
      // Construct temp memory objects on the fly. While we have no scratchpads
      // support on VM/GraphExecutor level.
      tmp_mems_.resize(tmp_mem_descs.size());
      for (size_t i = 0; i < tmp_mem_descs.size(); i++) {
        auto found = tmp_mem_mapping.find(i);

        if (found != tmp_mem_mapping.end()) {
          auto reuse_hdl = tmp_mems_[found->second].get_data_handle();
          tmp_mems_[i] = dnnl::memory(tmp_mem_descs[i], eng_, reuse_hdl);
        } else {
          tmp_mems_[i] = dnnl::memory(tmp_mem_descs[i], eng_);
        }
      }
    }

    /*! \brief Find memory object associated with provided ArgId */
    dnnl::memory operator()(const ArgId& ar) const {
      switch (ar.flag_) {
        case CONST:
          return const_mems_.at(ar.idx_);
        case TMP_STORAGE:
          return tmp_mems_.at(ar.idx_);
        case EXT_EID: {
          auto eid_and_desc = ext_mems_.at(ar.idx_);
          auto eid = eid_and_desc.first;
          auto desc = eid_and_desc.second;

          auto ext_dl_tensor = ext_data_provider_(eid);
          TVM_FFI_ICHECK(ext_dl_tensor->data);
          return dnnl::memory{desc, eng_, ext_dl_tensor->data};
        }
      }
      return {};
    }

   private:
    const dnnl::engine& eng_;
    const DLTensorProvider& ext_data_provider_;
    const std::vector<dnnl::memory>& const_mems_;
    const std::vector<std::pair<uint32_t, dnnl::memory::desc>>& ext_mems_;
    std::vector<dnnl::memory> tmp_mems_;
  };

  ArgId MakeArgReq(ArgReqFlag flag, uint32_t idx) { return {flag, idx}; }

  /* Collection of const memory objects. */
  std::vector<dnnl::memory> const_mem_collection_;

  /* Collection of intermediate memory descriptors. Zero position is reserved for scratchpads. */
  std::vector<dnnl::memory::desc> tmp_mem_collection_;

  /* Mapping of some temp buffer on previously registered. */
  std::map<size_t, size_t> tmp_mem_mapping_;

  /* Collection of external_intermediate memory objects.
   *  first  - eid of external buffer to ask
   *  second - t_desc describes how to treat external buffer */
  std::vector<std::pair<uint32_t, dnnl::memory::desc>> ext_mem_collection_;

  /* Map of eid to index of temp buffer in tmp_mem_collection_ */
  std::unordered_map<uint32_t, size_t> eid2idx_tmp_;

  /* List of external eid */
  std::set<uint32_t> ext_io_eid_;

  /* Engine of all tensors existing in this registry */
  dnnl::engine eng_;

  /* Execution stream use to reorder const data */
  dnnl::stream stream_;
};

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_DNNL_DNNL_TENSOR_REQUISITE_H_
