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
 * \file src/relax/transform/to_non_dataflow.cc
 * \brief Transform all dataflow structure to non-dataflow version.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/tir_pattern.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>

#include "../../s_tir/schedule/ir_comparator.h"

namespace tvm {

static const constexpr char* kCSourceFmtCuda = "cu";

namespace tirx {
using namespace tvm::prim;

using relax::FCodegen;
using relax::MatchResult;
using relax::TIRPattern;
using s_tir::TensorizeComparator;

/*! \brief helper to match a for stmt to a pattern*/
class ForMatcher : public TensorizeComparator {
 public:
  using TensorizeComparator::Dispatch;
  using SymbolMap = std::unordered_map<Var, PrimExpr>;
  explicit ForMatcher(const tirx::Function& pattern, const ffi::Array<Var>& pattern_vars)
      : TensorizeComparator(IRModule({{GlobalVar(""), pattern}}), false), pattern_(pattern) {
    for (const auto& pattern_var : pattern_vars) {
      this->pattern_vars_.insert(pattern_var);
    }
    this->evaluated_symbols.push_back(SymbolMap());
  }

  bool Match(const For& top) {
    TVM_FFI_ICHECK(pattern_->body.has_value() && pattern_->body.value()->size() == 1)
        << "Invalid pattern function";
    const auto* root = pattern_->body.value()->seq[0].as<s_tir::SBlockRealizeObj>();
    TVM_FFI_ICHECK(root && root->block->body->size() == 1) << "Invalid pattern function";
    const ForObj* pattern_top = root->block->body->seq[0].as<ForObj>();
    TVM_FFI_ICHECK(pattern_top) << "Invalid pattern function";
    if (!Dispatch(top, ffi::GetRef<Stmt>(pattern_top))) {
      return false;
    }
    // Get evaluated symbols, tensors from the pattern.
    for (const auto& arg : pattern_->params) {
      if (auto tensor = arg.as<tirx::TensorVar>()) {
        auto itt = rhs_tensor_map_.find(tensor.value());
        TVM_FFI_ICHECK(itt != rhs_tensor_map_.end());
        evaluated_tensors.push_back(itt->second);
      }
    }
    return true;
  }

  std::vector<SymbolMap> evaluated_symbols;
  std::vector<TensorVar> evaluated_tensors;

 private:
  using TensorizeComparator::Dispatch_;

  ffi::Optional<PrimExpr> QueryEvaluatedSymbols(const Var& var) {
    for (const SymbolMap& symbol_map : evaluated_symbols) {
      auto it = symbol_map.find(var);
      if (it != symbol_map.end()) {
        return it->second;
      }
    }
    return std::nullopt;
  }

  bool Dispatch(const Expr& expr, const PrimExpr& rhs) final {
    PrimExpr lhs = expr.as_or_throw<PrimExpr>();
    if (auto rhs_prim_var = rhs.as<PrimVar>()) {
      Var rhs_var = rhs_prim_var.value();
      if (pattern_vars_.count(rhs_var)) {
        // special case for pattern vars
        if (!lhs.as<PrimVar>()) {
          if (lhs->IsInstance<IntImmObj>() || lhs->IsInstance<FloatImmObj>()) {
            ffi::Optional<PrimExpr> value = QueryEvaluatedSymbols(rhs_var);
            if (value.has_value()) {
              if (!analyzer_->CanProveEqual(lhs, value.value())) return false;
            } else {
              evaluated_symbols.back().insert_or_assign(rhs_var, lhs);
            }
            return true;
          } else {
            return false;
          }
        }
      }
    }
    // pattern_var * expr
    if (const auto* rhs_ptr = rhs.as<prim::MulObj>()) {
      auto operand_a = rhs_ptr->a.as<PrimVar>();
      auto operand_b = rhs_ptr->b.as<PrimVar>();
      if (operand_a && pattern_vars_.count(operand_a.value())) {
        Var pattern_var = operand_a.value();
        // pattern var is on the left
        evaluated_symbols.push_back(SymbolMap());
        bool match = Dispatch(lhs, rhs_ptr->b);
        SymbolMap symbol_map = std::move(evaluated_symbols.back());
        evaluated_symbols.pop_back();
        if (match) {
          evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
          evaluated_symbols.back().insert_or_assign(pattern_var,
                                                    MakeConstScalar(rhs_ptr->b.ty(), 1));
          return true;
        }
      }
      if (operand_b && pattern_vars_.count(operand_b.value())) {
        Var pattern_var = operand_b.value();
        // pattern var is on the right
        evaluated_symbols.push_back(SymbolMap());
        bool match = Dispatch(lhs, rhs_ptr->a);
        SymbolMap symbol_map = std::move(evaluated_symbols.back());
        evaluated_symbols.pop_back();
        if (match) {
          evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
          evaluated_symbols.back().insert_or_assign(pattern_var,
                                                    MakeConstScalar(rhs_ptr->a.ty(), 1));
          return true;
        }
      }
    }
    // pattern_Var + expr
    if (const auto* rhs_ptr = rhs.as<prim::AddObj>()) {
      auto operand_a = rhs_ptr->a.as<PrimVar>();
      auto operand_b = rhs_ptr->b.as<PrimVar>();
      if (operand_a && pattern_vars_.count(operand_a.value())) {
        Var pattern_var = operand_a.value();
        // pattern var is on the left
        evaluated_symbols.push_back(SymbolMap());
        bool match = Dispatch(lhs, rhs_ptr->b);
        SymbolMap symbol_map = std::move(evaluated_symbols.back());
        evaluated_symbols.pop_back();
        if (match) {
          evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
          evaluated_symbols.back().insert_or_assign(pattern_var,
                                                    MakeConstScalar(rhs_ptr->b.ty(), 0));
          return true;
        }
      }
      if (operand_b && pattern_vars_.count(operand_b.value())) {
        Var pattern_var = operand_b.value();
        // pattern var is on the right
        evaluated_symbols.push_back(SymbolMap());
        bool match = Dispatch(lhs, rhs_ptr->a);
        SymbolMap symbol_map = std::move(evaluated_symbols.back());
        evaluated_symbols.pop_back();
        if (match) {
          evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
          evaluated_symbols.back().insert_or_assign(pattern_var,
                                                    MakeConstScalar(rhs_ptr->a.ty(), 0));
          return true;
        }
      }
    }
    return TensorizeComparator::Dispatch(lhs, rhs);
  }

  bool Dispatch_(const prim::AddObj* add, const PrimExpr& other) final {
    const auto* rhs = other.as<prim::AddObj>();
    if (rhs == nullptr) return false;
    {
      this->evaluated_symbols.push_back(SymbolMap());
      bool match = Dispatch(add->a, rhs->a) && Dispatch(add->b, rhs->b);
      SymbolMap symbol_map = std::move(evaluated_symbols.back());
      this->evaluated_symbols.pop_back();
      if (match) {
        this->evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
        return true;
      }
    }
    {
      this->evaluated_symbols.push_back(SymbolMap());
      bool match = Dispatch(add->a, rhs->b) && Dispatch(add->b, rhs->a);
      SymbolMap symbol_map = std::move(evaluated_symbols.back());
      this->evaluated_symbols.pop_back();
      if (match) {
        this->evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
        return true;
      }
    }
    return false;
  }

  bool Dispatch_(const prim::MulObj* mul, const PrimExpr& other) final {
    const auto* rhs = other.as<prim::MulObj>();
    if (rhs == nullptr) return false;
    {
      this->evaluated_symbols.push_back(SymbolMap());
      bool match = Dispatch(mul->a, rhs->a) && Dispatch(mul->b, rhs->b);
      SymbolMap symbol_map = std::move(evaluated_symbols.back());
      this->evaluated_symbols.pop_back();
      if (match) {
        this->evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
        return true;
      }
    }
    {
      this->evaluated_symbols.push_back(SymbolMap());
      bool match = Dispatch(mul->a, rhs->b) && Dispatch(mul->b, rhs->a);
      SymbolMap symbol_map = std::move(evaluated_symbols.back());
      this->evaluated_symbols.pop_back();
      if (match) {
        this->evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
        return true;
      }
    }
    return false;
  }

  bool Dispatch_(const CallObj* call, const PrimExpr& other) final {
    const auto* rhs = other.as<CallObj>();
    if (rhs == nullptr) return false;
    const auto* lhs_op = call->op.as<OpObj>();
    const auto* rhs_op = rhs->op.as<OpObj>();
    if (lhs_op == nullptr || rhs_op == nullptr) return false;
    if (lhs_op->name != rhs_op->name) return false;
    if (call->args.size() != rhs->args.size()) return false;
    for (size_t i = 0; i < call->args.size(); ++i) {
      if (!CompareExpr(call->args[i], rhs->args[i])) return false;
    }
    return true;
  }

  bool Dispatch_(const tvm::ForObj* op, const Stmt& other) final {
    const auto* rhs = other.as<ForObj>();
    loop_stack_lhs_.push_back(ffi::GetRef<For>(op));
    loop_stack_rhs_.push_back(ffi::GetRef<For>(rhs));
    // The body of loop must be loop or BlockRealize
    if (op->body->size() != 1 || (!op->body->seq[0]->IsInstance<s_tir::SBlockRealizeObj>() &&
                                  !op->body->seq[0]->IsInstance<ForObj>())) {
      return false;
    }
    if (rhs->body->size() != 1 || (!rhs->body->seq[0]->IsInstance<s_tir::SBlockRealizeObj>() &&
                                   !rhs->body->seq[0]->IsInstance<ForObj>())) {
      return false;
    }
    // Build mapping between the loop vars
    if (!DefEqual(op->var, rhs->var)) return false;
    // Only handle the case where the loop start from 0
    if (!IsZero(op->begin) || !IsZero(rhs->begin)) return false;
    if (op->kind != ForKind::kDefault || op->kind != rhs->kind) return false;
    if (!op->annotations.empty() || !rhs->annotations.empty()) return false;
    // Match the extents of loops
    if (!Dispatch(op->end, rhs->end) || !Dispatch(op->step, rhs->step)) return false;
    return Dispatch(op->body->seq[0], rhs->body->seq[0]);
  }

  bool Dispatch_(const s_tir::SBlockObj* op, const Stmt& other) final {
    const auto* rhs = other.as<s_tir::SBlockObj>();
    // Check block equality.
    // All iter vars and tensor regions including the order should match.
    // When checking iter vars, DefEqual is used to remap variables.
    if (!CompareArray(op->iter_vars, rhs->iter_vars, &ForMatcher::CompareIterVar)) {
      return false;
    }
    // disallow alloc tensors inside the block
    if (!op->alloc_tensors.empty() || !rhs->alloc_tensors.empty()) return false;
    if (!CompareArray(op->writes, rhs->writes, &ForMatcher::CompareTensorRegion)) {
      return false;
    }
    if (!CompareArray(op->reads, rhs->reads, &ForMatcher::CompareTensorRegion)) {
      return false;
    }
    // The body of the block has to be TensorStore
    if (op->body->size() != 1 || rhs->body->size() != 1 ||
        !op->body->seq[0]->IsInstance<TensorStoreObj>() ||
        !rhs->body->seq[0]->IsInstance<TensorStoreObj>()) {
      return false;
    }
    // Handle init block
    if (op->init.has_value() && !rhs->init.has_value()) return false;
    if (!op->init.has_value() && rhs->init.has_value()) return false;
    if (op->init.has_value() && rhs->init.has_value()) {
      if (!Dispatch(op->init.value(), rhs->init.value())) return false;
    }
    return Dispatch(op->body->seq[0], rhs->body->seq[0]);
  }

  bool Dispatch_(const s_tir::SBlockRealizeObj* op, const Stmt& other) final {
    const auto* rhs = other.as<s_tir::SBlockRealizeObj>();
    // Only allow trivial bindings
    for (size_t i = 0; i < op->iter_values.size(); ++i) {
      if (!op->iter_values[i].same_as(loop_stack_lhs_[i]->var)) return false;
    }
    for (size_t i = 0; i < rhs->iter_values.size(); ++i) {
      if (!rhs->iter_values[i].same_as(loop_stack_rhs_[i]->var)) return false;
    }
    // Disallow predicates now
    if (!IsOne(op->predicate) || !IsOne(rhs->predicate)) return false;
    return Dispatch(op->block, rhs->block);
  }

  bool Dispatch_(const TensorStoreObj* op, const Stmt& other) {
    const auto* rhs = other.as<TensorStoreObj>();
    return CompareTensorAccess(op, rhs) && Dispatch(op->value, rhs->value);
  }

  bool Dispatch_(const TensorLoadObj* op, const PrimExpr& other) {
    const auto* rhs = other.as<TensorLoadObj>();
    return CompareTensorAccess(op, rhs);
  }

  bool CompareTensor(const TensorVar& lhs, const TensorVar& rhs) {
    if (lhs.same_as(rhs)) return true;
    auto it = rhs_tensor_map_.find(rhs);
    bool equal;
    if (it != rhs_tensor_map_.end()) {
      equal = (*it).second.same_as(lhs);
    } else {
      // Compare shape
      if (lhs.shape().size() != rhs.shape().size()) return false;
      for (size_t i = 0; i < lhs.shape().size(); ++i) {
        if (!Dispatch(lhs.shape()[i], rhs.shape()[i])) return false;
      }
      equal = DefEqual(lhs, rhs) && lhs.dtype() == rhs.dtype() && lhs.scope() == rhs.scope();
      if (equal) {
        rhs_tensor_map_.insert_or_assign(rhs, lhs);
      }
    }
    return equal;
  }

  bool CompareTensorRegion(const TensorRegion& lhs, const TensorRegion& rhs) {
    if (!CompareTensor(lhs->source.as_or_throw<tvm::tirx::TensorVar>(),
                       rhs->source.as_or_throw<tvm::tirx::TensorVar>())) {
      return false;
    }
    return CompareArray(lhs->region, rhs->region, &ForMatcher::CompareRange);
  }

  bool CompareTensorAccess(const TensorStoreObj* lhs, const TensorStoreObj* rhs) {
    if (!CompareTensor(lhs->dest.as_or_throw<TensorVar>(), rhs->dest.as_or_throw<TensorVar>())) {
      return false;
    }
    return CompareArray(
        lhs->indices, rhs->indices,
        static_cast<bool (ForMatcher::*)(const Expr&, const PrimExpr&)>(&ForMatcher::Dispatch));
  }

  bool CompareTensorAccess(const TensorLoadObj* lhs, const TensorLoadObj* rhs) {
    if (rhs == nullptr) return false;
    if (!CompareTensor(lhs->source.as_or_throw<TensorVar>(),
                       rhs->source.as_or_throw<TensorVar>())) {
      return false;
    }
    return CompareArray(
        lhs->indices, rhs->indices,
        static_cast<bool (ForMatcher::*)(const Expr&, const PrimExpr&)>(&ForMatcher::Dispatch));
  }

  template <typename T, typename Self, typename F>
  bool CompareArray(const ffi::Array<T>& lhs, const ffi::Array<T>& rhs, F Self::* cmp) {
    if (lhs.same_as(rhs)) return true;
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!(static_cast<Self*>(this)->*cmp)(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  sym::Analyzer analyzer_;
  std::vector<For> loop_stack_lhs_, loop_stack_rhs_;
  tirx::Function pattern_;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> pattern_vars_;
};

/*! \brief Analyze the function and match it with a list of patterns */
class TIRPatternMatcher {
 public:
  static ffi::Array<MatchResult> Match(ffi::Array<TIRPattern> patterns, SeqStmt body) {
    TIRPatternMatcher matcher(patterns);
    matcher.OpMatternMatch(body);
    if (matcher.fail_) return {};
    return matcher.match_results_;
  }

 private:
  explicit TIRPatternMatcher(ffi::Array<TIRPattern> patterns) : patterns_(patterns) {}

  // Find an op that matches this block
  bool BlockPatternMatch(const For& top) {
    for (const TIRPattern& pattern : patterns_) {
      tirx::Function pattern_func = pattern;
      ffi::Array<Var> pattern_symbolic_vars;
      int tensor_count = 0;
      while (tensor_count < static_cast<int>(pattern_func->params.size()) &&
             pattern_func->params[tensor_count]->ty.as<tirx::TensorTypeObj>()) {
        ++tensor_count;
      }
      for (int i = tensor_count; i < static_cast<int>(pattern_func->params.size()); i++) {
        pattern_symbolic_vars.push_back(pattern_func->params[i]);
      }
      ForMatcher block_matcher(pattern_func, pattern_symbolic_vars);
      if (block_matcher.Match(top)) {
        // We have found a match
        ffi::Array<PrimExpr> symbol_values;
        for (int i = tensor_count; i < static_cast<int>(pattern_func->params.size()); i++) {
          symbol_values.push_back(
              block_matcher.evaluated_symbols.back().at(pattern_func->params[i]));
        }
        match_results_.push_back(
            MatchResult(pattern, symbol_values, block_matcher.evaluated_tensors));
        return true;
      }
    }
    // The block fails to match any pattern
    return false;
  }

  // For each block in the body, try to find its corresponding pattern one by one
  void OpMatternMatch(const SeqStmt& body) {
    for (const Stmt& stmt : body->seq) {
      const ForObj* loop = stmt.as<ForObj>();
      if (loop == nullptr || !BlockPatternMatch(ffi::GetRef<For>(loop))) {
        break;
      }
    }
    if (match_results_.empty()) {
      fail_ = true;
    }
  }
  /*! \brief Indicate whether we fail to match.*/
  bool fail_ = false;
  /*! \brief The patterns we match the target stmt to.*/
  ffi::Array<TIRPattern> patterns_;
  /*! \brief The results of the matching process.*/
  ffi::Array<MatchResult> match_results_;
};

/*! \brief helper class to partition a function into 2 parts. Return function information which we
 * can use to construct the two partitioned parts.*/
class FunctionPartitioner : public s_tir::StmtExprVisitor {
 public:
  explicit FunctionPartitioner(int num_matched_ops) : num_matched_ops_(num_matched_ops) {}
  /*! \brief alloc_tensors for the first function */
  std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> allocs1;
  /*! \brief alloc_tensors for the second function */
  std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> allocs2;
  /*! \brief whether the current block is in the first function */
  ffi::Map<s_tir::SBlock, bool> block_partition;
  /*! \brief input tensors for the first function */
  std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> input1;
  /*! \brief input tensors for the second function */
  std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> input2;
  /*! \brief The output tensor for the first function, which is also the input tensor for the second
  function */
  ffi::Optional<TensorVar> intermediate_tensor;
  /*! \brief Indicate whether we have failed. If failed, we will not do any further analysis and
  directly return the original one. */
  bool fail = false;

 private:
  ffi::Optional<VisitInterrupt> Visit_(const s_tir::SBlockObj* op) final {
    block_counter_++;
    bool is_matching_ = block_counter_ <= num_matched_ops_;
    if (block_counter_ == num_matched_ops_ && intermediate_tensor.has_value()) {
      allocs1.erase(intermediate_tensor.value());
    }
    for (const auto& read : op->reads) {
      if (is_matching_) {
        input1.insert(read->source.as_or_throw<tvm::tirx::TensorVar>());
      } else {
        input2.insert(read->source.as_or_throw<tvm::tirx::TensorVar>());
      }
    }
    for (const auto& write : op->writes) {
      if (is_matching_) {
        allocs1.insert(write->source.as_or_throw<tvm::tirx::TensorVar>());
      } else if (allocs1.count(write->source.as_or_throw<tvm::tirx::TensorVar>())) {
        fail = true;
        return std::nullopt;
      } else {
        allocs2.insert(write->source.as_or_throw<tvm::tirx::TensorVar>());
      }
      if (is_matching_) {
        intermediate_tensor = write->source.as_or_throw<tvm::tirx::TensorVar>();
      } else {
        input2.insert(write->source.as_or_throw<tvm::tirx::TensorVar>());
      }
    }
    block_partition.Set(ffi::GetRef<s_tir::SBlock>(op), is_matching_);

    return std::nullopt;
  }
  // The number of matched ops in the function
  size_t num_matched_ops_;
  size_t block_counter_ = 0;
};

/*! \brief remove parts according to block partition, and update the alloc_tensors for blocks */
class BlockRemover : public s_tir::StmtExprMutator {
 public:
  static Stmt RemoveBlockByPartition(
      Stmt stmt, const ffi::Map<s_tir::SBlock, bool>& block_partition,
      const std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>& allocs,
      bool is_library_part) {
    auto remover = ffi::make_object<BlockRemover>(block_partition, allocs, is_library_part);
    return remover->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
  }

  BlockRemover(const ffi::Map<s_tir::SBlock, bool>& block_partition,
               const std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>& allocs,
               bool is_library_part)
      : block_partition(block_partition), allocs_(allocs), is_library_part_(is_library_part) {}

 private:
  UnchangedOr<Stmt> Mutate_(const s_tir::SBlockObj* op, InplaceMode inplace_mode) final {
    s_tir::SBlock block = s_tir::StmtExprMutator::Mutate_(op, inplace_mode)
                              .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                              .as_or_throw<s_tir::SBlock>();
    ffi::ObjectPtr<s_tir::SBlockObj> n = ffi::make_object<s_tir::SBlockObj>(*block.operator->());
    if (op->name_hint != "root") {
      TVM_FFI_ICHECK(block_partition.count(ffi::GetRef<s_tir::SBlock>(op)));
      bool block_is_library = block_partition[ffi::GetRef<s_tir::SBlock>(op)];
      if (!(is_library_part_ ^ block_is_library)) {
        n->body = block->body;
      } else {
        erased_ = true;
      }
    }
    ffi::Array<TensorVar> alloc_tensors;
    for (const TensorVar& b : block->alloc_tensors) {
      if (allocs_.count(b)) {
        alloc_tensors.push_back(b);
      }
    }
    n->alloc_tensors = alloc_tensors;
    return s_tir::SBlock(n);
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj* op, InplaceMode inplace_mode) final {
    ffi::Array<Stmt> seq;
    bool erased_all = !op->seq.empty();
    for (const Stmt& s : op->seq) {
      Stmt new_s = Mutate(s).ValueOrUnchanged(s);
      if (erased_) {
        erased_ = false;
      } else {
        erased_all = false;
        seq.push_back(new_s);
      }
    }
    // Let enclosing loops disappear when their only subtree was removed.
    erased_ = erased_all;
    return SeqStmt(seq, op->loc);
  }

  bool erased_ = false;
  ffi::Map<s_tir::SBlock, bool> block_partition;
  std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> allocs_;
  bool is_library_part_ = false;
};

/*!
 * \brief Split the input function into two functions, one for the library kernel and one for the
 * rest.
 * \param func The input function.
 * \param arg_partition The input arg for the functions after split.
 * \param patterns The patterns to match.
 * \param f_codegen The function to generate the code for the library kernel.
 * \return A pair of functions, the first one is the library kernel and the second one is the
 * rest.
 */
std::pair<tirx::Function, ffi::Optional<tirx::Function>> SplitFunctions(
    tirx::Function func, std::vector<std::vector<int>>* arg_partition,
    ffi::Array<TIRPattern> patterns, FCodegen f_codegen) {
  // Step 1. Find the library kernel and the rest.
  if (!func->body.has_value() || func->body.value()->size() != 1) {
    return {func, std::nullopt};
  }
  const auto* root = func->body.value()->seq[0].as<s_tir::SBlockRealizeObj>();
  if (root == nullptr) return {func, std::nullopt};
  SeqStmt body = root->block->body;
  ffi::Array<MatchResult> match_results = TIRPatternMatcher::Match(patterns, body);
  if (match_results.empty()) {
    return {func, std::nullopt};
  }
  ffi::Array<ffi::Any> codegen_result = f_codegen(match_results);
  TVM_FFI_ICHECK(codegen_result.size() == 3);
  ffi::String library_code = codegen_result[0].as_or_throw<ffi::String>();
  int num_matched_ops = codegen_result[1].as_or_throw<IntImm>()->value.as<int>().value();
  ffi::Array<TensorVar> func1_args = codegen_result[2].as_or_throw<ffi::Array<TensorVar>>();
  if (num_matched_ops == 0) {
    return {func, std::nullopt};
  }
  auto partitioner = ffi::make_object<FunctionPartitioner>(num_matched_ops);
  partitioner->Visit(body);
  if (partitioner->fail || !partitioner->intermediate_tensor.has_value()) {
    return {func, std::nullopt};
  }
  bool has_second_func = false;
  for (const auto& pr : partitioner->block_partition) {
    if (!pr.second) {
      has_second_func = true;
      break;
    }
  }
  if (!has_second_func) {
    // No need to split the function.
    return {WithAttr(func, tvm::relax::attr::kLibraryKernel.name, library_code), std::nullopt};
  }
  // Step 2. Split the function into two functions.
  Stmt body1 = BlockRemover::RemoveBlockByPartition(
      func->body.value(), partitioner->block_partition, partitioner->allocs1, true);
  Stmt body2 = BlockRemover::RemoveBlockByPartition(
      func->body.value(), partitioner->block_partition, partitioner->allocs2, false);
  // Step 3. Craft the first function.
  ffi::Array<Var> new_params1;
  std::vector<int> arg_partition1;
  TVM_FFI_ICHECK_LE(func1_args.size(), partitioner->input1.size());
  for (const auto& tensor : func1_args) {
    TVM_FFI_ICHECK(partitioner->input1.find(tensor) != partitioner->input1.end());
    for (size_t i = 0; i < func->params.size(); i++) {
      auto param_tensor = func->params[i].as<tirx::TensorVar>();
      if (param_tensor.has_value() && param_tensor.value().same_as(tensor)) {
        new_params1.push_back(func->params[i]);
        arg_partition1.push_back(i);
        break;
      }
    }
  }
  arg_partition->push_back(arg_partition1);
  new_params1.push_back(partitioner->intermediate_tensor.value());
  tirx::Function func1 = tirx::Function(new_params1, SeqStmt(body1), func->ret_type, func->attrs);
  func1 = WithAttr(func1, tvm::relax::attr::kLibraryKernel.name, library_code);
  // Step 4. Craft the second function.
  ffi::Array<Var> new_params2;
  std::vector<int> arg_partition2;
  new_params2.push_back(partitioner->intermediate_tensor.value());
  for (int i = 0; i < static_cast<int>(func->params.size()); i++) {
    Var param = func->params[i];
    auto param_tensor = param.as<tirx::TensorVar>();
    if (param_tensor.has_value() && partitioner->input2.count(param_tensor.value())) {
      new_params2.push_back(param);
      if (i != static_cast<int>(func->params.size()) - 1) {
        arg_partition2.push_back(i);
      }
    }
  }
  arg_partition->push_back(arg_partition2);
  tirx::Function func2 = tirx::Function(new_params2, SeqStmt(body2), func->ret_type, func->attrs);
  return {func1, func2};
}
}  // namespace tirx

namespace relax {
void StringReplace(std::string* subject, const std::string& search, const std::string& replace) {
  for (size_t pos = 0; (pos = subject->find(search, pos)) != std::string::npos;
       pos += replace.length()) {
    subject->replace(pos, search.length(), replace);
  }
}

tvm::BaseFunc CodegenWithLibrary(const tirx::FunctionObj* pf, ffi::String global_symbol) {
  using namespace tvm::tirx;
  ffi::Optional<ffi::String> library_code = pf->attrs.GetAttr(tvm::relax::attr::kLibraryKernel);
  if (!library_code.has_value()) {
    return ffi::GetRef<tirx::Function>(pf);
  }
  std::string source = library_code.value();
  StringReplace(&source, "{global_symbol}", global_symbol);
  ExternFunc ret(global_symbol);
  ret =
      WithAttrs(std::move(ret),
                ffi::Map<ffi::String, ffi::Any>{
                    {ffi::String(tvm::relax::attr::kCSource.name), ffi::String(source)},
                    {ffi::String(tvm::relax::attr::kCSourceFmt.name), ffi::String(kCSourceFmtCuda)},
                });
  return ret;
}

/*! \brief Emit 2 calls to the library kernel and the rest of the function. */
class SplitMutator : public ExprMutator {
 public:
  SplitMutator(const tvm::IRModule& mod, ffi::Array<TIRPattern> patterns, FCodegen fcodegen)
      : ExprMutator(mod), mod_(mod), patterns_(patterns), fcodegen_(fcodegen) {}
  static IRModule Transform(const IRModule& mod, ffi::Array<TIRPattern> patterns,
                            FCodegen fcodegen) {
    SplitMutator mutator(mod, patterns, fcodegen);
    for (auto& kv : mod->functions) {
      if (auto* func = kv.second.as<FunctionObj>()) {
        Function new_func = mutator(ffi::GetRef<Function>(func)).as_or_throw<Function>();
        mutator.builder_->UpdateFunction(kv.first, new_func);
      }
    }
    return mutator.builder_->GetContextIRModule();
  }

 private:
  using ExprMutator::VisitExpr_;

  inline ffi::Array<Expr> GetCallTIRArgs(Expr args) {
    if (args.as<TupleObj>()) {
      return args.as<TupleObj>()->fields;
    } else {
      return {args};
    }
  }

  Expr VisitExpr_(const CallObj* op) final {
    Call call = ExprMutator::VisitExpr_(op).as_or_throw<Call>();
    static const Op call_tir_op_ = Op::Get("relax.call_tir");
    static const Op call_dps_packed_ = Op::Get("relax.call_dps_packed");
    if (!call->op.same_as(call_tir_op_)) return call;
    // the first argument is the function to be called
    const auto* gv_ptr = call->args[0].as<GlobalVarObj>();
    if (gv_ptr == nullptr) return call;
    GlobalVar gv = ffi::GetRef<GlobalVar>(gv_ptr);
    // retrieve the function from the module and split it
    tirx::Function func = mod_->Lookup(gv).as_or_throw<tirx::Function>();
    std::vector<std::vector<int>> arg_partition;
    // split the function into two functions, one for the library kernel and one for the rest.
    std::pair<tirx::Function, ffi::Optional<tirx::Function>> split_funcs =
        tirx::SplitFunctions(func, &arg_partition, patterns_, fcodegen_);
    if (!split_funcs.second.has_value()) {
      // no need to split, the function itself a library kernel
      tvm::BaseFunc lib_func = CodegenWithLibrary(split_funcs.first.get(), gv->name_hint);
      if (lib_func->IsInstance<tirx::FunctionObj>()) return ffi::GetRef<Call>(op);
      // Update the function in the module with the library kernel
      TVM_FFI_ICHECK(lib_func->IsInstance<ExternFuncObj>());
      builder_->UpdateFunction(gv, lib_func);
      // emit the call to the library kernel
      ffi::ObjectPtr<CallObj> new_call = ffi::make_object<CallObj>(*call.operator->());
      new_call->op = this->call_dps_packed_;
      new_call->args = {lib_func, call->args[1]};
      return Call(new_call);
    }
    tirx::Function func1 = tirx::RenewDef(split_funcs.first);
    tirx::Function func2 = tirx::RenewDef(split_funcs.second.value());
    TVM_FFI_ICHECK(arg_partition.size() == 2);
    // emit the first call to the library kernel
    ffi::Array<Expr> args1;
    for (int p : arg_partition[0]) {
      args1.push_back(GetCallTIRArgs(call->args[1])[p]);
    }
    // replace the function in the module with the library kernel
    tvm::BaseFunc lib_func = CodegenWithLibrary(func1.get(), gv->name_hint);
    if (lib_func->IsInstance<tirx::FunctionObj>()) return ffi::GetRef<Call>(op);
    TVM_FFI_ICHECK(lib_func->IsInstance<ExternFuncObj>());
    builder_->UpdateFunction(gv, lib_func);
    tirx::TensorVar intermediate_tensor = func1->params.back().as_or_throw<tirx::TensorVar>();
    PrimType dtype = intermediate_tensor.dtype();
    Call call1(Type::Missing(), call_dps_packed_, {lib_func, Tuple(args1)}, call->attrs,
               {TensorType(ShapeExpr(intermediate_tensor.shape()), dtype)});
    Var call_var1 = builder_->Emit(call1);
    // emit the second call to the rest of the function
    ffi::Array<Expr> args2;
    args2.push_back(call_var1);
    for (int p : arg_partition[1]) {
      args2.push_back(GetCallTIRArgs(call->args[1])[p]);
    }
    GlobalVar gv2 = builder_->AddFunction(func2, "unfused_epilogue");
    Call call2(Type::Missing(), call_tir_op_, {gv2, Tuple(args2)}, call->attrs, call->ty_args);
    builder_->UpdateFunction(gv, WithoutAttr(func, tvm::attr::kGlobalSymbol.name));
    return call2;
  }

  const Op call_dps_packed_ = Op::Get("relax.call_dps_packed");
  tvm::IRModule mod_;
  ffi::Array<TIRPattern> patterns_;
  FCodegen fcodegen_;
};

namespace transform {
Pass SplitCallTIRByPattern(ffi::Array<TIRPattern> patterns, FCodegen fcodegen) {
  auto pass_func =  //
      [=](IRModule m, PassContext pc) { return SplitMutator::Transform(m, patterns, fcodegen); };
  return CreateModulePass(/*pass_function=*/pass_func,  //
                          /*opt_level=*/0,              //
                          /*pass_name=*/"SplitCallTIRByPattern");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.SplitCallTIRByPattern", SplitCallTIRByPattern);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
