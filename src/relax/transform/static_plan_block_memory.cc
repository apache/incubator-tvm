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
 * \file src/relax/transform/static_plan_block_memory.cc
 * \brief The static memory planning pass on BindingBlock level.
 * \details
 * The core data structure of the planning pass is StorageToken, which denotes
 * reusable memory in this planning pass.
 *
 * The memory planning pass contains three stages:
 *
 * The first stage is initialization. A storage token object will be created
 * for each builtin alloc_tensor as long as the allocated storage satisfies
 * the requirements (which are described in the code). The reference counter
 * (i.e., the times of reference) for each token is recorded.
 *
 * The second stage is allocation planning. We maintain a pool of available
 * allocated storage, in the form of storage tokens. For the storage token of
 * each builtin alloc_tensor, we check if there is appropriate available token
 * in the pool under certain criterion. If there is, we reuse that storage
 * for this alloc_tensor. Otherwise, we decide to allocate a storage for the
 * alloc_tensor.
 *
 * The third stage is IR rewrite. Based on the decision made in the second
 * stage, we insert memory alloc_storage, alloc_tensor.
 *
 * - Insert `memory.alloc_storage` before first usage site of each
 *   storage token.
 *
 * - Insert `memory.alloc_tensor` at the site of the
 *   `builtin.alloc_tensor` that it replaces.
 *
 * We do not insert `memory.kill_storage` or `memory.kill_tensor`, as
 * these are handled in the later `KillAfterLastUse` lowering pass.
 * This ensures that all tensors are killed after their last use,
 * including dynamically-sized tensors, without requiring that
 * `StaticPlanBlockMemory` track these dynamic-sized tensors.
 *
 * The memory planning pass "supports" dynamic shape in the way of TIR variable
 * upper bound annotation. To be more specific, we can annotate the attribute
 * "tir_var_upper_bound" to Relax functions. The attribute value is a dict from
 * strings to integers, denoting the name of TIR variables to the upper bound
 * values of the TIR vars. **The annotated upper bound attribute only applies
 * to TIR vars in the function signature for clarity.**
 *
 * For example, we can annotate a Relax function with
 *   `R.func_attr({"tir_var_upper_bound": {"n": 1024}})`.
 * It means the maximum value of variable that names "n" in the function
 * signature will have upper bound 1024. And we will use 1024 as its value
 * during memory planning.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/transform.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>

#include <map>
#include <set>
#include <vector>

#include "../../s_tir/backend/adreno/texture_utils.h"
#include "utils.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

/*!
 * \brief A representation of a block of reusable memory required at runtime.
 * \details Only the tensors whose memory can be "possibly reused" will have
 * their storage token. In other words, we do not have storage token for tensor
 * - that is a function parameter,
 * - that is a function return value,
 * - one of whose use site is a BindingBlock different from its allocation site,
 * - that is used as a condition or branch return of a IfExprObj,
 * - that is used as the body of a SeqExprObj,
 * - that is used as arguments in a Call whose op is not a tirx::Function.
 *
 * In practice, we do create a storage token for such tensor at first. But at
 * any time we find a tensor satisfying any of the conditions above, we erase
 * its storage token.
 */
class StorageTokenObj : public ffi::Object {
 public:
  explicit StorageTokenObj(PrimExpr bytes) : bytes(std::move(bytes)) {}
  explicit StorageTokenObj(ffi::UnsafeInit) : bytes(ffi::UnsafeInit{}) {}

  /*! \brief Reference counter. */
  int ref_counter{0};
  /*! \brief Number of bytes that this token requires. */
  PrimExpr bytes;
  /*! \brief The dtype of this token. */
  DLDataType dtype;
  /*! \brief The memory scope of the token. */
  runtime::MemoryScope storage_scope;
  /*! \brief The VDevice information. */
  ffi::Optional<VDevice> vdevice;
  /*! \brief The storage id, reserved for debug and demo use. */
  int storage_id{-1};

  /*! \brief Get the constant number of bytes that this token requires, or -1 if the number of bytes
   * is symbolic */
  int64_t const_bytes() const {
    const auto* imm = bytes.as<IntImmObj>();
    auto const_val = imm ? imm->value.as<int64_t>() : std::nullopt;
    if (const_val.has_value()) {
      return *const_val;
    } else {
      return -1;
    }
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.transform.StorageToken", StorageTokenObj, ffi::Object);
};

/*!
 * \brief Managed reference to StorageTokenObj.
 * \sa StorageTokenObj
 */
class StorageToken : public ffi::ObjectRef {
 public:
  explicit StorageToken(ffi::Array<PrimExpr> shape, DLDataType dtype,
                        runtime::MemoryScope storage_scope,
                        ffi::Optional<VDevice> vdevice = std::nullopt) {
    // Compute the tensor size from the shape.
    PrimType dtype_ty(dtype);
    TVM_FFI_ICHECK(!dtype_ty.IsScalableVector())
        << "Cannot statically plan storage size for scalable vector dtype " << dtype_ty;
    ffi::BigInt const_coeff = dtype_ty.StorageBytes();
    PrimExpr size = IntImm::Int64(1);
    bool size_computed = false;

    if (vdevice.has_value()) {
      VDevice vdev = vdevice.value();
      std::string dev_kind = vdev->target.value()->kind->name;

      if (vdev->memory_scope != runtime::MemoryScope::Global()) {
        auto device_size_handler =
            tvm::ffi::Function::GetGlobal(std::string("DeviceGetMemSize." + dev_kind));
        if (device_size_handler.has_value()) {
          size *= (*device_size_handler)(shape, dtype, vdevice.value()).cast<PrimExpr>();
          size_computed = true;
        }
        auto device_scope_handler =
            tvm::ffi::Function::GetGlobal(std::string("DeviceScopeCompatibility." + dev_kind));
        if (device_scope_handler.has_value()) {
          runtime::MemoryScope dev_scope = (*device_scope_handler)(vdevice.value()->target.value(),
                                                                   vdevice.value()->memory_scope)
                                               .cast<runtime::MemoryScope>();
          storage_scope = dev_scope;
        }
      }
    }
    if (!size_computed) {
      for (const PrimExpr& dim_len : shape) {
        if (const IntImmObj* const_dim_len = dim_len.as<IntImmObj>()) {
          const_coeff *= const_dim_len->value;
        } else {
          size *= dim_len;
        }
      }
    }

    size = IntImm::Int64(const_coeff) * size;

    ffi::ObjectPtr<StorageTokenObj> n = ffi::make_object<StorageTokenObj>(size);
    n->dtype = dtype;
    n->storage_scope = std::move(storage_scope);
    n->vdevice = std::move(vdevice);
    data_ = std::move(n);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StorageToken, ffi::ObjectRef, StorageTokenObj);
};

// We use NestedMsg to store the tokens used by each Expr.
using Tokens = NestedMsg<StorageToken>;

/*!
 * \brief Memory manager for flattened 1d memory (buffers)
 * \note We can generalize this implementation to multi-dimensional memory
 * following the same flow in the future.
 */
class TokenAllocatorMixed {
 public:
  explicit TokenAllocatorMixed(sym::AnalyzerObj* analyzer) : analyzer_(analyzer) {}

  /*!
   * \brief Request a storage token from the available token pool for a
   * given prototype, or report no appropriate available token in the pool.
   * \param prototype The requesting prototype storage token.
   * \return The request result token. Return std::nullopt if there is no
   * appropriate available token in the pool.
   */
  ffi::Optional<StorageToken> RequestReuse(StorageToken prototype) {
    // Step 0. Sanity check: the prototype token is supposed not to be allocated with actual storage
    TVM_FFI_ICHECK_EQ(prototype->storage_id, -1)
        << "The token is expected not to be allocated before.";
    // If the prototype has no reference at all, feel free to allocate new storage.
    // The unused binding can be removed by cleaning passes.
    if (prototype->ref_counter == 0) {
      return std::nullopt;
    }

    // Step 1. Get the available pool of the token dtype.
    std::multimap<int64_t, StorageToken>& pool =
        available_pool_[{prototype->storage_scope, prototype->dtype}];

    int64_t size = prototype->const_bytes();
    if (size == -1) {
      // Handle the case where the prototype token has dynamic size. Currently it requires the
      // symbolic size to be the same as the prototype token in order to reuse the storage.
      auto [begin, end] = pool.equal_range(size);
      for (; begin != end; ++begin) {
        StorageToken available_token = begin->second;
        if (analyzer_->CanProveEqual(prototype->bytes, available_token->bytes)) {
          TVM_FFI_ICHECK_EQ(available_token->ref_counter, 0)
              << "Available tokens are expected to have 0 reference.";
          available_token->ref_counter = prototype->ref_counter;
          pool.erase(begin);
          return available_token;
        }
      }
      return std::nullopt;
    }
    // Step 2. Get the range of memory blocks in [size / match_range_, size * match_range_)
    auto begin = pool.lower_bound(size / match_range_);
    auto mid = pool.lower_bound(size);
    auto end = pool.upper_bound(size * match_range_);
    // Step 3. Search for memory block that equals or is larger than the requested size.
    if (mid != end) {
      StorageToken available_token = mid->second;
      TVM_FFI_ICHECK_EQ(available_token->ref_counter, 0)
          << "Available tokens are expected to have 0 reference.";
      TVM_FFI_ICHECK_LE(size, available_token->const_bytes());
      available_token->ref_counter = prototype->ref_counter;
      pool.erase(mid);
      return available_token;
    }
    // Step 4. Then search for memory block that is smaller than the requested size.
    if (mid != begin) {
      --mid;
      StorageToken available_token = mid->second;
      int64_t available_size = available_token->const_bytes();
      TVM_FFI_ICHECK_EQ(available_token->ref_counter, 0)
          << "Available tokens are expected to have 0 reference.";
      TVM_FFI_ICHECK_GE(available_size, 0);
      TVM_FFI_ICHECK_GE(size, available_size);
      // Enlarge the token size.
      available_token->bytes = IntImm::Int64(size);
      available_token->ref_counter = prototype->ref_counter;
      pool.erase(mid);
      return available_token;
    }
    // Return `std::nullopt` indicating that no satisfiable storage token is found in the available
    // pool.
    return std::nullopt;
  }

  /*!
   * \brief Allocate a storage token for the input prototype token.
   * \param prototype The prototype token.
   * \param storage_id The id of this token.
   */
  StorageToken Alloc(StorageToken prototype, int storage_id) {
    // Sanity check: the prototype token is supposed not to be allocated with actual storage yet
    TVM_FFI_ICHECK_EQ(prototype->storage_id, -1)
        << "The token is expected not to be allocated before.";
    prototype->storage_id = storage_id;
    full_pool_.push_back(prototype);
    return prototype;
  }

  /*!
   * \brief Release the input token, putting it into the available pool.
   * \param token The token to be released.
   */
  void Release(StorageToken token) {
    // Sanity check: the token has been allocated with actual storage, and should have 0 reference.
    TVM_FFI_ICHECK_GE(token->storage_id, 0)
        << "The token to be released is expected to be allocated before";
    TVM_FFI_ICHECK_EQ(token->ref_counter, 0)
        << "The token to be released is expected to have 0 reference.";
    available_pool_[{token->storage_scope, token->dtype}].insert({token->const_bytes(), token});
  }

  /*! \brief Clear the allocator. */
  void Clear() {
    available_pool_.clear();
    full_pool_.clear();
  }

 private:
  using PoolKey = std::pair<runtime::MemoryScope, DLDataType>;

  /*! \brief The hash class to enable storage scope and raw dtype as map key class. */
  struct PoolKeyHash {
    std::size_t operator()(const PoolKey& p) const {
      std::size_t h = ffi::ObjectPtrHash{}(p.first);
      h ^= static_cast<std::size_t>(p.second.code) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= static_cast<std::size_t>(p.second.bits) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= static_cast<std::size_t>(p.second.lanes) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  /*! \brief The arithmetic analyzer. */
  sym::AnalyzerObj* analyzer_;
  /*! \brief A constant scale representing the token search range. */
  const int match_range_{16};
  /*! \brief The pool of available storage tokens for each storage scope and dtype. */
  std::unordered_map<PoolKey, std::multimap<int64_t, StorageToken>, PoolKeyHash> available_pool_;
  /*! \brief All the storage tokens that have been allocated with actual storage. */
  std::vector<StorageToken> full_pool_;
};

/*! \brief Check if the input op is a memory op that may return the same buffer. */
bool IsInplaceMemoryOp(const Expr& op) {
  static const Op reshape_op = Op::Get("relax.reshape");
  static const Op view_op = Op::Get("relax.memory.view");
  static const Op ensure_zero_offset_op = Op::Get("relax.memory.ensure_zero_offset");
  const auto* extern_func = op.as<ExternFuncObj>();
  bool is_builtin_reshape =
      extern_func != nullptr && extern_func->global_symbol == "vm.builtin.reshape";
  return op.same_as(reshape_op) || op.same_as(view_op) || op.same_as(ensure_zero_offset_op) ||
         is_builtin_reshape;
}

/*! \brief The base class for the storage allocation visitor. */
class StorageAllocatorBaseVisitor : public ExprVisitor {
 protected:
  using ExprVisitor::VisitExpr_;

  void VisitBindingBlock_(const BindingBlockObj* block) override {
    // We maintain a block stack for token allocation-site and use-site check.
    block_stack_.push_back(block);
    ExprVisitor::VisitBindingBlock_(block);
    TVM_FFI_ICHECK(!block_stack_.empty());
    TVM_FFI_ICHECK(block_stack_.back() == block);
    block_stack_.pop_back();
  }

  void VisitBinding_(const VarBindingObj* binding) override {
    ExprVisitor::VisitBinding_(binding);
    // The binding var has the same tokens as the binding value.
    SetTokens(binding->var.get(), token_map_[binding->value.get()]);
  }

  void VisitBinding_(const MatchCastObj* binding) override {
    ExprVisitor::VisitBinding_(binding);
    // MatchCast refines the type without changing the underlying storage.
    SetTokens(binding->var.get(), token_map_[binding->value.get()]);
  }

  void VisitBindingBlock_(const DataflowBlockObj* block) override {
    // We maintain a block stack for token allocation-site and use-site check.
    block_stack_.push_back(block);
    ExprVisitor::VisitBindingBlock_(block);
    TVM_FFI_ICHECK(!block_stack_.empty());
    TVM_FFI_ICHECK(block_stack_.back() == block);
    block_stack_.pop_back();
  }

  void VisitExpr_(const TupleObj* tuple) final {
    ffi::Array<Tokens> tokens;
    tokens.reserve(tuple->fields.size());
    for (const Expr& field : tuple->fields) {
      Tokens field_tokens = GetTokens(field);
      tokens.push_back(field_tokens);
    }
    SetTokens(tuple, Tokens(tokens));
  }

  void VisitExpr_(const TupleGetItemObj* tuple_item) final {
    Tokens tokens = GetTokens(tuple_item->tuple);
    // If the tuple has no token, every of its field has no token as well.
    if (tokens.IsNull()) {
      token_map_[tuple_item] = Tokens();
      return;
    }
    TVM_FFI_ICHECK(tokens.IsNested());
    ffi::Array<Tokens> field_tokens = tokens.NestedArray();
    TVM_FFI_ICHECK_GT(static_cast<int>(field_tokens.size()), tuple_item->index);
    TVM_FFI_ICHECK_GE(tuple_item->index, 0);
    SetTokens(tuple_item, field_tokens[tuple_item->index]);
  }

  /******************** Utilities ********************/

  Tokens GetTokens(const Expr& expr) {
    this->VisitExpr(expr);
    return token_map_[expr.get()];
  }

  virtual void SetTokens(const ExprObj* expr, Tokens tokens) { token_map_[expr] = tokens; }

  /*! \brief The mapping from each Expr to its corresponding storage tokens. */
  std::unordered_map<const ExprObj*, Tokens> token_map_;
  /*! \brief The binding block stack. */
  std::vector<const BindingBlockObj*> block_stack_;
};

/*!
 * \brief Set the range constraints of the TIR variables that appear in
 * the input function signature in the analyzer.
 * \param func The function to be analyzed.
 * \param ana The analyzer which contains the TIR var upper bounds.
 * \param dom_map The domain map of the TIR variables.
 */
void SetTIRVarRangeConstraints(Function func, sym::AnalyzerObj* ana,
                               ffi::Map<tvm::Var, sym::IntSet>* dom_map) {
  // Use the attribute-annotated TIR var bounds as the TIR var values for
  // memory planning.
  // NOTE: we only apply the annotated bounds to the TIR variables that
  // appear in the **function signature**.
  ffi::Map<ffi::String, IntImm> var_upper_bound_attr_raw =
      func->GetAttr(tvm::relax::attr::kTirVarUpperBound).value_or(ffi::Map<ffi::String, IntImm>());
  ffi::Map<ffi::String, IntImm> var_lower_bound_attr_raw =
      func->GetAttr(tvm::relax::attr::kTirVarLowerBound).value_or(ffi::Map<ffi::String, IntImm>());
  ffi::Array<ffi::String> non_negative_var_attr_raw =
      func->GetAttr(tvm::relax::attr::kTirNonNegativeVar).value_or(ffi::Array<ffi::String>());
  std::unordered_map<ffi::String, IntImm> var_upper_bound_attr;
  std::unordered_map<ffi::String, IntImm> var_lower_bound_attr;
  std::unordered_set<ffi::String> non_negative_var_attr;
  // We manually check the value type to ensure the values are all positive IntImm.
  for (auto [key, value] : var_upper_bound_attr_raw) {
    var_upper_bound_attr.insert_or_assign(key, value);
  }
  for (auto [key, value] : var_lower_bound_attr_raw) {
    var_lower_bound_attr.insert_or_assign(key, value);
  }
  for (const ffi::String& var_name : non_negative_var_attr_raw) {
    non_negative_var_attr.insert(var_name);
  }
  ffi::Array<tvm::Var> var_in_signature = TIRVarsInType(GetType(func));
  for (const tvm::Var& tir_var : var_in_signature) {
    auto it_upper = var_upper_bound_attr.find(tir_var->name);
    auto it_lower = var_lower_bound_attr.find(tir_var->name);

    // Only bind the variable to a range if an upper bound is explicitly provided.
    // Without an upper bound, memory planning cannot determine the required storage size,
    // so we skip binding and let the variable remain unbounded.
    if (it_upper != var_upper_bound_attr.end()) {
      ffi::BigInt lower = (it_lower != var_lower_bound_attr.end()) ? it_lower->second->value : 0;
      const ffi::BigInt& upper = it_upper->second->value;
      tvm::Range range = tvm::Range::FromMinExtent(tvm::IntImm::Int64(lower),
                                                   tvm::IntImm::Int64(upper - lower + 1));
      auto context = ana->GetConstraintContext();
      context->Append(sym::RangeConstraint(tir_var, range));
      ana->SyncConstraintContext(context);
      dom_map->Set(tir_var, sym::IntSet::FromRange(range));
    } else if (it_lower != var_lower_bound_attr.end() && it_lower->second->value >= 0) {
      auto context = ana->GetConstraintContext();
      context->Append(sym::PredicateConstraint(tir_var.as_or_throw<PrimExpr>() >= 0));
      ana->SyncConstraintContext(context);
    } else if (non_negative_var_attr.count(tir_var->name)) {
      auto context = ana->GetConstraintContext();
      context->Append(sym::PredicateConstraint(tir_var.as_or_throw<PrimExpr>() >= 0));
      ana->SyncConstraintContext(context);
    }
  }
}

/*!
 * \brief Use the upper bounds of TIR vars to compute the upper
 * bound of a given shape.
 * \param shape The input shape to be computed.
 * \param ana The arithmetic analyzer that contains the upper bounds
 * of TIR variables
 * \return The upper-bounded shape. When a dimension's upper bound
 * cannot be determined, we keep the dimension unchanged.
 */
ffi::Array<PrimExpr> GetUpperBoundShape(ffi::Array<PrimExpr> shape, sym::AnalyzerObj* ana,
                                        const ffi::Map<tvm::Var, sym::IntSet>& dom_map) {
  // Use the upper bounds of TIR vars as their values.
  ffi::Array<PrimExpr> upper_bounded_shape;
  upper_bounded_shape.reserve(shape.size());
  for (const PrimExpr& dim_len : shape) {
    int64_t max_bound = ana->const_int_bound(dim_len)->max_value;
    if (max_bound == std::numeric_limits<int64_t>::max()) {
      sym::IntSet int_set = ana->int_set(dim_len, dom_map);
      if (int_set.HasUpperBound()) {
        upper_bounded_shape.push_back(int_set.max());
      } else {
        upper_bounded_shape.push_back(dim_len);
      }
    } else {
      upper_bounded_shape.push_back(tvm::IntImm::Int64(max_bound));
    }
  }
  return upper_bounded_shape;
}

/*! \brief Check if a shape is static (a.k.a., has no TIR variable). */
bool IsStaticShape(ffi::Array<PrimExpr> shape) {
  for (const PrimExpr& dim : shape) {
    const auto* int_len = dim.as<IntImmObj>();
    if (!int_len) {
      return false;
    }
  }
  return true;
}

/*!
 * \brief The visitor class for storage token initialization.
 * \details It goes through the entire function to get the storage tokens
 * used by each Expr. After the initialization, we
 * - know the tokens that each Expr is using,
 * - know the number of references for each token,
 * - rule out the builtin alloc_tensors to which the planning does not apply.
 */
class StorageAllocatorInit : public StorageAllocatorBaseVisitor {
 public:
  /*!
   * \brief The entry of the initialization.
   * \param mod The IRModule to be planned
   * \param analyzer The arithmetic analyzer.
   * \return The mapping from each Expr to the token it uses.
   */
  static std::unordered_map<const ExprObj*, Tokens> Initialize(const IRModule& mod,
                                                               sym::AnalyzerObj* analyzer) {
    StorageAllocatorInit initializer(mod, analyzer);

    for (auto it : mod->functions) {
      const auto* func = it.second.as<FunctionObj>();
      if (func == nullptr) {
        continue;
      }
      initializer(ffi::GetRef<Function>(func));
    }
    return initializer.token_map_;
  }

 private:
  using ExprVisitor::VisitExpr_;

  explicit StorageAllocatorInit(const IRModule& ctx_mod, sym::AnalyzerObj* analyzer)
      : ctx_mod_(ctx_mod), analyzer_(analyzer) {}

  void VisitExpr_(const FunctionObj* func) final {
    // Set the range constraints of TIR variables in the analyzer.
    SetTIRVarRangeConstraints(ffi::GetRef<Function>(func), analyzer_, &dom_map_);
    // Recurse into the function to get its tokens.
    Tokens body_tokens = GetTokens(func->body);
    // Discard the tokens used by the function return value, as they are external referenced.
    DiscardTokensIn(body_tokens);
  }

  void VisitExpr_(const CallObj* call) final {
    static const Op alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const Op call_tir_dyn_op = Op::Get("relax.vm.call_tir_dyn");
    static const Op call_tir_packed_op = Op::Get("relax.call_tir_packed");

    if (call->op.same_as(alloc_tensor_op)) {
      // Create a storage token for builtin alloc_tensor.
      this->CreateToken(call);
      return;
    } else if (IsInplaceMemoryOp(call->op)) {
      // Reuse the input's token for builtin reshape.
      SetTokens(call, GetTokens(call->args[0]));
      return;
    }

    // - Increase the reference counters of the arguments when the callee is
    // a tirx::Function of the context module or an external function via 'call_packed'.
    // It assumes external function calls via 'call_packed' do not retain memory
    // from the arguments.
    // - Otherwise, discard the tokens used by the arguments, as there might be
    // potential external reference.
    // A native pointer result may alias an argument, with no lifetime tracked by
    // Relax.  Keep such inputs out of the reusable storage pool.
    bool reusable_tirx_call = call->op.same_as(call_tir_packed_op) && !call->ty.as<AnyTypeObj>();
    if (IsFunctionGlobalVar(call->op) || call->op->IsInstance<ExternFuncObj>() ||
        call->op.same_as(call_tir_dyn_op) || reusable_tirx_call) {
      ffi::Array<Expr> args =
          (call->op.same_as(call_tir_dyn_op) || call->op.same_as(call_tir_packed_op))
              ? call->args[1].as_or_throw<Tuple>()->fields
              : call->args;
      TVM_FFI_ICHECK(!block_stack_.empty());
      for (const Expr& arg : args) {
        Tokens tokens = GetTokensWithAllocSiteCheck(arg, block_stack_.back());
        ForEachLeaf(tokens, [](StorageToken token) { token->ref_counter += 1; });
      }
    } else {
      for (const Expr& arg : call->args) {
        DiscardTokensIn(GetTokens(arg));
      }
    }
  }

  void VisitExpr_(const IfExprObj* if_node) final {
    Tokens cond_tokens = GetTokens(if_node->cond);
    Tokens then_tokens = GetTokens(if_node->true_branch);
    Tokens else_tokens = GetTokens(if_node->false_branch);
    // Discard the tokens used by the condition, then-body and else-body,
    // as the planning works on block level.
    DiscardTokensIn(cond_tokens);
    DiscardTokensIn(then_tokens);
    DiscardTokensIn(else_tokens);
  }

  void VisitExpr_(const SeqExprObj* seq) final {
    for (const BindingBlock& binding_block : seq->blocks) {
      this->VisitBindingBlock(binding_block);
    }
    Tokens body_tokens = GetTokens(seq->body);
    // Discard the tokens used by the body, as the planning works on block level.
    DiscardTokensIn(body_tokens);
  }

  /******************** Utilities ********************/

  /*!
   * \brief Check if the input op is GlobalVar corresponding to a tirx::Function inside the ctx
   * module.
   * \param op The op to be checked
   * \return A boolean indicating if the input op corresponds to a tirx::Function.
   */
  bool IsFunctionGlobalVar(const Expr& op) {
    const auto* global_var = op.as<GlobalVarObj>();
    if (global_var == nullptr) {
      return false;
    }
    auto func_it = ctx_mod_->functions.find(ffi::GetRef<GlobalVar>(global_var));
    if (func_it == ctx_mod_->functions.end()) {
      return false;
    }
    return (*func_it).second->IsInstance<tirx::FunctionObj>();
  }

  /*!
   * \brief Create a storage token for the builtin alloc_tensor call.
   * \param call The call to be processed.
   * \return The created token.
   */
  Tokens CreateToken(const CallObj* call) {
    // Sanity checks about
    // - the call return value is a Tensor;
    // - the shape of the tensor is known, in the form of ShapeExpr;
    // - the tensor has known dtype;
    // - no storage token was created for this call before.
    const auto* ty = call->ty.as<TensorTypeObj>();
    TVM_FFI_ICHECK_NOTNULL(ty);
    const auto* shape = ty->shape.as<ShapeExprObj>();
    TVM_FFI_ICHECK_NOTNULL(shape);
    TVM_FFI_ICHECK(!ty->IsUnknownDtype());
    TVM_FFI_ICHECK(ty->dtype.value()->dtype == call->args[1].as_or_throw<DataTypeImm>()->value);
    TVM_FFI_ICHECK(!token_map_.count(call));

    // Use the upper bounds of TIR vars as their values. The upper bound shape can still be dynamic
    // if the upper bounds of some variables are not provided.
    ffi::Array<PrimExpr> upper_bounded_shape =
        GetUpperBoundShape(shape->values, analyzer_, dom_map_);

    // Create and set token.
    StringImm storage_scope = call->args[3].as_or_throw<StringImm>();

    int64_t vdevice_index = -1;
    if (const auto* int_imm = call->args[2].as<IntImmObj>()) {
      vdevice_index = int_imm->value.as<int>().value();
    }
    ffi::Optional<VDevice> vdevice = GetGlobalVDevice(ctx_mod_, vdevice_index);

    StorageToken token(upper_bounded_shape, ty->dtype.value()->dtype,
                       runtime::MemoryScope::Get(storage_scope->value), vdevice);

    Tokens tokens(token);
    SetTokens(call, tokens);
    TVM_FFI_ICHECK(!block_stack_.empty());
    token2block_[token.get()] = block_stack_.back();
    return tokens;
  }

  /*!
   * \brief Override the token setter in the base visitor.
   * For each token, we keep record of all Expr that are using that token.
   * When we want to discard one token, we use the records to remove the token
   * from the Expr that are using it.
   */
  void SetTokens(const ExprObj* expr, Tokens tokens) final {
    StorageAllocatorBaseVisitor::SetTokens(expr, tokens);
    ForEachLeaf(tokens, [this, expr](StorageToken token) {
      this->token2exprs_[token.get()].push_back(expr);
    });
  }

  /*!
   * \brief Token getter with allocation site check.
   * We first get the tokens used by the input Expr, and check if the allocation
   * site of each token is the input current block.
   * Since the planning works on block level, if some token's allocation site
   * is not the current block, we discard the token so that it will not be planned.
   * \param expr The Expr whose tokens is to be got.
   * \param cur_block The pointer to the current block.
   * \return The tokens used by the input Expr.
   */
  Tokens GetTokensWithAllocSiteCheck(const Expr& expr, const BindingBlockObj* cur_block) {
    Tokens tokens = GetTokens(expr);
    ForEachLeaf(tokens, [this, cur_block](StorageToken token) {
      auto it = this->token2block_.find(token.get());
      TVM_FFI_ICHECK(it != this->token2block_.end());
      if (it->second != cur_block) {
        this->DiscardToken(token);
      }
    });
    return token_map_[expr.get()];
  }

  /*! \brief Discard the input tokens. */
  void DiscardTokensIn(Tokens tokens) {
    ForEachLeaf(tokens, [this](StorageToken token) { this->DiscardToken(token); });
  }

  /*!
   * \brief Discard the input token.
   * For each Expr that is using the input token, remove the token from the Expr's token set.
   * \param token_to_discard The token to be discarded.
   */
  void DiscardToken(StorageToken token_to_discard) {
    const std::vector<const ExprObj*>& exprs = token2exprs_[token_to_discard.get()];
    for (const ExprObj* expr : exprs) {
      token_map_[expr] = MapNestedMsg(token_map_[expr], [token_to_discard](StorageToken token) {
        return token.same_as(token_to_discard) ? Tokens() : Tokens(token);
      });
    }
    token2exprs_.erase(token_to_discard.get());
    token2block_.erase(token_to_discard.get());
  }

  /*!
   * \brief The context IRModule, used for checking if a callee function is
   * a tirx::Function inside the IRModule.
   */
  const IRModule& ctx_mod_;
  /*! \brief The arithmetic analyzer. */
  sym::AnalyzerObj* analyzer_;
  /*! \brief The domain map of dynamic TIR variables for analysis. */
  ffi::Map<tvm::Var, sym::IntSet> dom_map_;
  /*! \brief The mapping from each token to the binding block where it is created. */
  std::unordered_map<const StorageTokenObj*, const BindingBlockObj*> token2block_;
  /*! \brief The mapping from each token to the Exprs that are using this token. */
  std::unordered_map<const StorageTokenObj*, std::vector<const ExprObj*>> token2exprs_;
};

/*!
 * \brief The visitor class for storage token allocation planning.
 * \details
 * - For each builtin alloc_tensor whose token is not discarded in the
 * initialization stage, we request a storage reuse or decide to allocate
 * storage for this token, depending on if there is appropriate available
 * token in the token pool we maintain.
 * - For each VM builtin reshape, we reuse the input's tokens.
 *
 * After the allocation planning, we know the token that each builtin
 * alloc_tensor plans to use. Compared with the initialization, here
 * the token is possibly a reuse of some previous token, rather than
 * we having one token for each alloc_tensor.
 */
class StorageAllocator : public StorageAllocatorBaseVisitor {
 public:
  explicit StorageAllocator(std::unordered_map<const ExprObj*, Tokens> token_map,
                            sym::AnalyzerObj* analyzer)
      : allocator_(analyzer) {
    this->token_map_ = std::move(token_map);
  }

  void Allocate(const IRModule& mod) {
    for (auto it : mod->functions) {
      const auto* func = it.second.as<FunctionObj>();
      if (func == nullptr) {
        continue;
      }
      // Clear the allocator to make the planning of different functions independent.
      allocator_.Clear();
      this->VisitExpr_(func);
    }
  }

  /*!
   * \brief The mapping from each `builtin.alloc_tensor` to its corresponding
   * underlying storage token that it is using.
   */
  std::unordered_map<const ExprObj*, StorageToken> alloc_tensor2token;
  /*! \brief The mapping from each binding block to the storage tokens that are create inside. */
  std::unordered_map<const BindingBlockObj*, std::vector<const StorageTokenObj*>> block2tokens;

 private:
  using ExprVisitor::VisitBinding_;
  using ExprVisitor::VisitExpr_;

  void VisitBindingBlock_(const BindingBlockObj* block) final {
    StorageAllocatorBaseVisitor::VisitBindingBlock_(block);
    // Sanity check: each token allocated inside the block should not be
    // referenced by anyone at the end of the block.
    for (const StorageTokenObj* token : block2tokens[block]) {
      TVM_FFI_ICHECK_EQ(token->ref_counter, 0);
    }
  }

  void VisitBinding_(const VarBindingObj* binding, const CallObj* call) final {
    static const Op alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    if (call->op.same_as(alloc_tensor_op)) {
      auto it = token_map_.find(call);
      TVM_FFI_ICHECK(it != token_map_.end());

      if (it->second.IsNull()) {
        // IsNull being true means the token was discarded, and this alloc_tensor
        // is not considered by the planning.
        return;
      }
      TVM_FFI_ICHECK(it->second.IsLeaf());
      StorageToken new_token = this->RequestReuseOrAlloc(it->second.LeafValue());

      // Record that this alloc_tensor is using the token.
      alloc_tensor2token.insert({call, new_token});
      token2cur_tensor_[new_token.get()].push_back(binding->var);
      SetTokens(call, Tokens(new_token));
      // Record that the token is allocated in the current block.
      TVM_FFI_ICHECK(!block_stack_.empty());
      std::vector<const StorageTokenObj*>& block_tokens = block2tokens[block_stack_.back()];
      if (std::find(block_tokens.begin(), block_tokens.end(), new_token.get()) ==
          block_tokens.end()) {
        block_tokens.push_back(new_token.get());
      }
      return;
    } else if (IsInplaceMemoryOp(call->op)) {
      Tokens tokens = GetTokens(call->args[0]);
      TVM_FFI_ICHECK(!tokens.IsNested());
      if (tokens.IsLeaf()) {
        // If the input is using a token, record that the reshape uses the token as well.
        token2cur_tensor_[tokens.LeafValue().get()].push_back(binding->var);
        SetTokens(call, tokens);
      } else {
        TVM_FFI_ICHECK(token_map_[call].IsNull());
      }
      return;
    }

    // Decrease the reference counter by one for each token that the arguments use.
    // Check if a token can be released (i.e., has no reference) after decrease.
    // And release it if so.
    for (const Expr& arg : call->args) {
      Tokens tokens = GetTokens(arg);
      ForEachLeaf(tokens, [this](StorageToken token) {
        TVM_FFI_ICHECK_GT(token->ref_counter, 0);
        token->ref_counter -= 1;
        this->CheckForRelease(token);
      });
    }
  }

  /*! \brief Request a storage reuse, or allocate storage if no appropriate storage is reusable. */
  StorageToken RequestReuseOrAlloc(StorageToken prototype) {
    ffi::Optional<StorageToken> token = allocator_.RequestReuse(prototype);
    if (!token.has_value()) {
      return allocator_.Alloc(prototype, this->n_storage_++);
    } else {
      return token.value();
    }
  }

  /*!
   * \brief Check if a token has no reference and thus can be released. And release it if so.
   * \param token The token to be checked.
   */
  void CheckForRelease(StorageToken token) {
    // Sanity check: the token was allocated before and has non-negative reference.
    TVM_FFI_ICHECK_GE(token->storage_id, 0);
    TVM_FFI_ICHECK_GE(token->ref_counter, 0);

    if (token->ref_counter == 0) {
      allocator_.Release(token);
      auto it = token2cur_tensor_.find(token.get());
      TVM_FFI_ICHECK(it != token2cur_tensor_.end());
      token2cur_tensor_.erase(it);
    }
  }

  /*! \brief Number of allocated storages. */
  int n_storage_{0};
  /*! \brief The 1D memory allocator. */
  TokenAllocatorMixed allocator_;
  /*! \brief The mapping from each token to the tensors that are currently using it. */
  std::unordered_map<const StorageTokenObj*, std::vector<Var>> token2cur_tensor_;
};

/*!
 * \brief The rewriter class based on the token allocation planning.
 * \details
 * - For each builtin alloc_tensor that was planned, substitute it with a memory
 * alloc_tensor. If no memory alloc_storage was created for it before, create one.
 */
class StorageAllocationRewriter : public ExprMutator {
 public:
  explicit StorageAllocationRewriter(
      IRModule mod, std::unordered_map<const ExprObj*, StorageToken> alloc_tensor2token,
      std::unordered_map<const BindingBlockObj*, std::vector<const StorageTokenObj*>> block2tokens)
      : ExprMutator(std::move(mod)),
        alloc_tensor2token_(std::move(alloc_tensor2token)),
        block2tokens_(std::move(block2tokens)) {}

  IRModule Rewrite() {
    const IRModule& mod = builder_->GetContextIRModule();
    for (const auto& [gv, base_func] : mod->functions) {
      const auto* func_ = base_func.as<FunctionObj>();
      if (func_ == nullptr) {
        continue;
      }
      constexpr static const char* plan_dyn_attr_ = "relax.memory_plan_dynamic_func_output";
      plan_dynamic_output_ = static_cast<bool>(
          func_->GetAttr<IntImm>(plan_dyn_attr_).value_or(IntImm::Int32(0))->value);
      if (plan_dynamic_output_) {
        SetTIRVarRangeConstraints(ffi::GetRef<Function>(func_), ana_.get(), &dom_map_);
      }
      token2storage_var_.clear();
      Function func = this->VisitExpr_(func_).as_or_throw<Function>();
      if (plan_dynamic_output_) {
        func = WithoutAttr(func, plan_dyn_attr_);
      }
      builder_->UpdateFunction(gv, func);
    }
    return builder_->GetContextIRModule();
  }

 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const SeqExprObj* seq) final {
    // A storage var is only visible in the scope it is emitted in, such as an if branch.
    // Forget the vars emitted in this scope on exit, so that a token first used inside a
    // branch gets a new `alloc_storage` where it is reused in another branch or after the if.
    // A token shared by several scopes is allocated in each at its final size, which
    // `RequestReuse` may have enlarged.
    auto saved_token2storage_var = token2storage_var_;
    Expr ret = ExprMutator::VisitExpr_(seq);
    token2storage_var_ = std::move(saved_token2storage_var);
    return ret;
  }

  Expr VisitExpr_(const CallObj* call) final {
    static const Op alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const Op mem_alloc_storage = Op::Get("relax.memory.alloc_storage");
    static const Op mem_alloc_tensor = Op::Get("relax.memory.alloc_tensor");
    auto it = alloc_tensor2token_.find(call);
    if (it != alloc_tensor2token_.end()) {
      // Case 1. This `alloc_tensor` is planned for memory reuse.
      TVM_FFI_ICHECK(call->op.same_as(alloc_tensor_op));
      const auto* ty = call->ty.as<TensorTypeObj>();
      TVM_FFI_ICHECK_NOTNULL(ty);
      TVM_FFI_ICHECK_NOTNULL(ty->shape.as<ShapeExprObj>());
      PrimExpr runtime_device_index = call->args[2].as_or_throw<PrimExpr>();

      // If the token is visited for the first time, create a storage variable using
      // `memory.alloc_storage` for it.
      StorageToken token = it->second;
      Var storage_var = [&]() -> Var {
        auto it_token = token2storage_var_.find(token.get());
        if (it_token == token2storage_var_.end()) {
          ShapeExpr size({token->bytes});
          PrimExpr virtual_device_index = runtime_device_index;
          DLDataType dtype = token->dtype;
          Call alloc_storage(Type::Missing(), mem_alloc_storage,
                             {std::move(size), virtual_device_index,
                              StringImm(token->storage_scope.name()), DataTypeImm(dtype)},
                             std::nullopt);
          Var storage_var = builder_->Emit(alloc_storage, "storage");
          token2storage_var_.insert_or_assign(token.get(), storage_var);
          return storage_var;
        } else {
          return it_token->second;
        }
      }();

      // And always create a `memory.alloc_tensor` for the old `builtin.alloc_tensor`.
      PrimExpr offset = IntImm::Int64(0);
      DLDataType dtype = ty->dtype.value()->dtype;
      return Call(Type::Missing(), mem_alloc_tensor,
                  {storage_var, offset, ty->shape.value(), DataTypeImm(dtype), call->args[2]},
                  std::nullopt);
    } else if (plan_dynamic_output_ && call->op.same_as(alloc_tensor_op)) {
      // Case 2. For a `alloc_tensor` that is not planned for memory reuse,
      // we would still like to allocate **static** memory for the tensor.
      // So in case the tensor shape is dynamic but has an upper bound
      // estimation, we allocate a storage to its upper bound size, and
      // allocate a tensor out from it with the actual symbolic shape.

      const auto* ty = call->ty.as<TensorTypeObj>();
      TVM_FFI_ICHECK_NOTNULL(ty);
      const auto* shape = ty->shape.as<ShapeExprObj>();
      TVM_FFI_ICHECK_NOTNULL(shape);
      ffi::Array<PrimExpr> upper_bounded_shape =
          GetUpperBoundShape(shape->values, ana_.get(), dom_map_);
      if (!IsStaticShape(shape->values)) {
        TVM_FFI_ICHECK(!ty->IsUnknownDtype());
        TVM_FFI_ICHECK_EQ(ty->dtype.value()->dtype,
                          call->args[1].as_or_throw<DataTypeImm>()->value);
        PrimExpr bytes = upper_bounded_shape[0];
        for (int i = 1; i < static_cast<int>(upper_bounded_shape.size()); ++i) {
          bytes *= upper_bounded_shape[i];
        }
        DLDataType dtype = ty->dtype.value()->dtype;
        PrimType dtype_ty(dtype);
        TVM_FFI_ICHECK(!dtype_ty.IsScalableVector())
            << "Cannot statically plan storage size for scalable vector dtype " << dtype_ty;
        bytes *= IntImm::Int64(static_cast<int64_t>(dtype_ty.StorageBytes()));
        Call alloc_storage(Type::Missing(), mem_alloc_storage,
                           {/*size=*/ShapeExpr({bytes}),
                            /*virtual_device_index=*/call->args[2].as_or_throw<PrimExpr>(),
                            /*storage_scope=*/call->args[3].as_or_throw<StringImm>(),  //
                            /*dtype=*/DataTypeImm(dtype)});
        Var storage = builder_->Emit(alloc_storage, "storage");
        return Call(Type::Missing(), mem_alloc_tensor,
                    {storage,  //
                     /*offset=*/IntImm::Int64(0),
                     /*shape=*/ffi::GetRef<ShapeExpr>(shape),  //
                     /*dtype=*/DataTypeImm(dtype),
                     /*vdevice_index=*/call->args[2]});
      }
    }

    return ExprMutator::VisitExpr_(call);
  }

  /*! \brief The arithmetic analyzer. */
  sym::Analyzer ana_;
  /*! \brief The domain map of dynamic TIR variables for analysis. */
  ffi::Map<tvm::Var, sym::IntSet> dom_map_;
  /*! \brief A boolean indicating whether to plan dynamic-shape function output tensors. */
  bool plan_dynamic_output_;
  /*!
   * \brief The mapping from each memory-reusable `builtin.alloc_tensor` to
   its corresponding underlying storage token that it is using.
   */
  std::unordered_map<const ExprObj*, StorageToken> alloc_tensor2token_;
  /*! \brief The mapping from each binding block to the storage tokens that are create inside. */
  std::unordered_map<const BindingBlockObj*, std::vector<const StorageTokenObj*>> block2tokens_;
  /*! \brief The mapping from each token to its storage var visible in the current scope. */
  std::unordered_map<const StorageTokenObj*, Var> token2storage_var_;
};

IRModule StaticPlanBlockMemory(IRModule mod) {
  sym::Analyzer ana;

  // Step 1. Initialize.
  std::unordered_map<const ExprObj*, Tokens> token_map =
      StorageAllocatorInit::Initialize(mod, ana.get());
  // Step 2. Collect the memory allocation info.
  StorageAllocator allocator(std::move(token_map), ana.get());
  allocator.Allocate(mod);
  // Step 3. Rewrite the function.
  StorageAllocationRewriter rewriter(std::move(mod),  //
                                     std::move(allocator.alloc_tensor2token),
                                     std::move(allocator.block2tokens));
  return rewriter.Rewrite();
}

namespace transform {

Pass StaticPlanBlockMemory() {
  auto pass_func = [=](IRModule m, PassContext pc) {
    return relax::StaticPlanBlockMemory(std::move(m));
  };
  return CreateModulePass(pass_func, /*opt_level=*/0, "StaticPlanBlockMemory");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.StaticPlanBlockMemory", StaticPlanBlockMemory);
}

PrimExpr GetTextureMemorySizeFromVDevice(ffi::Array<PrimExpr> pshape, DLDataType dtype,
                                         VDevice vdevice) {
  int image_row_align = static_cast<int>(
      vdevice->target.value()->GetAttr<int64_t>("image_base_address_alignment").value_or(64));

  struct Shape {
    const ffi::Array<PrimExpr>& shape;
    int64_t operator[](size_t i) const {
      const auto* imm = shape[i].as<IntImmObj>();
      auto value = imm ? imm->value.as<int64_t>() : std::nullopt;
      TVM_FFI_ICHECK(value.has_value()) << "Dymamic shapes not suported over texture now";
      return *value;
    }
    int size() { return this->shape.size(); }
  };
  auto shape = Shape{pshape};

  int lanes = static_cast<int16_t>(dtype.lanes);
  TVM_FFI_ICHECK_GE(lanes, 0) << "Can't fetch the bytes of a scalable vector at a compile time.";
  size_t size = texture::GetTextureMemorySize<Shape>(shape, dtype.bits, lanes,
                                                     vdevice->memory_scope, image_row_align);
  return IntImm::Int64(size);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("DeviceGetMemSize.opencl", GetTextureMemorySizeFromVDevice);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
