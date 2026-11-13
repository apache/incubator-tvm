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
#include <tvm/ffi/cast.h>
#include <tvm/s_tir/stmt.h>

#include <set>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*!
 * \brief The reorder index is not a valid permutation of
 *   [0, 1, ..., n-1] where n is the number of block iter vars.
 */
class InvalidReorderIndex : public ScheduleErrorContextObj {
 public:
  explicit InvalidReorderIndex(IRModule mod, SBlock block, ffi::Array<int64_t> new_order)
      : mod_(mod), block_(block), new_order_(new_order) {}
  IRModule mod() const final { return mod_; }
  ffi::String FastErrorString() const final {
    return "ScheduleError: The specified reorder indices are invalid.";
  }
  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The user provided block itervar index order " << new_order_
       << " is not a valid permutation of [0, 1, ..., num_block_iter_vars-1] in block {0}.";
    return ffi::String(os.str());
  }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  SBlock block_;
  ffi::Array<int64_t> new_order_;
};

class BlockIterVarRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  ffi::Map<SBlock, SBlock> block_map;
  explicit BlockIterVarRewriter(const SBlockObj* block_n, std::vector<int> order)
      : order_(std::move(order)), block_to_rewrite(block_n) {}

 private:
  std::vector<int> order_;
  const SBlockObj* block_to_rewrite;
  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* op, InplaceMode inplace_mode) final {
    if (op->block.get() == block_to_rewrite) {
      bool inplace_block = inplace_mode == InplaceMode::kAllow && op->block.unique();
      SBlock block = op->block;
      ffi::Array<IterVar> new_iter_vars;
      ffi::Array<PrimExpr> new_iter_values;
      for (int idx : order_) {
        new_iter_vars.push_back(block->iter_vars[idx]);
        new_iter_values.push_back(op->iter_values[idx]);
      }
      SBlock new_block = block;
      if (inplace_block) {
        const_cast<SBlockObj*>(block.get())->iter_vars = std::move(new_iter_vars);
      } else {
        auto copy = ffi::make_object<SBlockObj>(*block.get());
        copy->iter_vars = std::move(new_iter_vars);
        new_block = SBlock(std::move(copy));
      }
      block_map.Set(block, new_block);
      if (inplace_mode == InplaceMode::kAllow) {
        auto* writable = const_cast<SBlockRealizeObj*>(op);
        writable->block = std::move(new_block);
        writable->iter_values = std::move(new_iter_values);
        return ffi::Unchanged();
      } else {
        auto copy = ffi::make_object<SBlockRealizeObj>(*op);
        copy->block = std::move(new_block);
        copy->iter_values = std::move(new_iter_values);
        return SBlockRealize(std::move(copy));
      }
    } else {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }
};

void ReorderBlockIterVar(ScheduleState self, const StmtSRef& block_sref,
                         const ffi::Array<int64_t>& new_order) {
  const SBlockObj* block_n = TVM_SREF_TO_SBLOCK(block_sref);
  std::vector<int> new_order_vec;
  for (int64_t x : new_order) {
    new_order_vec.push_back(static_cast<int>(x));
  }
  // check whether new_order is valid or not;
  size_t num_block_itervars = block_n->iter_vars.size();
  std::set<int> ind_set(new_order_vec.begin(), new_order_vec.end());
  bool is_full = new_order_vec.size() == num_block_itervars;
  bool is_unique = (ind_set.size() == new_order_vec.size());
  bool is_within_boundary = std::all_of(new_order_vec.begin(), new_order_vec.end(), [&](int x) {
    return x >= 0 && x < static_cast<int>(num_block_itervars);
  });
  if (!is_full || !is_unique || !is_within_boundary) {
    throw MakeScheduleError<InvalidReorderIndex>(self->mod, ffi::GetRef<SBlock>(block_n),
                                                 new_order);
  }

  // find parent block
  const SBlockObj* parent_block_n = nullptr;
  const StmtSRefObj* p = block_sref.get()->parent;
  while (p != nullptr) {
    if (p->stmt->IsInstance<SBlockObj>()) {
      parent_block_n = TVM_SREF_TO_SBLOCK(ffi::GetRef<StmtSRef>(p));
      break;
    }
    p = p->parent;
  }
  const StmtSRef parent_block_sref = ffi::GetRef<StmtSRef>(p);
  const SBlock& parent_block = ffi::GetRef<SBlock>(parent_block_n);

  // rewrite block and blockrealize
  auto rewriter = ffi::make_object<BlockIterVarRewriter>(block_n, std::move(new_order_vec));
  SBlock new_parent_block =
      rewriter->Mutate(parent_block).ValueOrUnchanged(parent_block).as_or_throw<SBlock>();
  rewriter->block_map.Set(parent_block, new_parent_block);
  self->Replace(parent_block_sref, new_parent_block, rewriter->block_map);
}

struct ReorderBlockIterVarTraits : public UnpackedInstTraits<ReorderBlockIterVarTraits> {
  static constexpr const char* kName = "ReorderBlockIterVar";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block, ffi::Array<int64_t> new_order) {
    sch->ReorderBlockIterVar(block, new_order);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      ffi::Array<int64_t> new_order) {
    PythonAPICall py("reorder_block_iter_var");
    py.Input("block", block);
    py.Input("new_order", new_order);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() { RegisterInstructionKind<ReorderBlockIterVarTraits>(); }

}  // namespace s_tir
}  // namespace tvm
