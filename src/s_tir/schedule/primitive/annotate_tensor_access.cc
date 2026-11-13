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

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

class AnnotateRegionRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  AnnotateRegionRewriter(TensorVar tensor, int tensor_index, TensorRegion new_region,
                         TensorIndexType tensor_index_type)
      : tensor_(tensor),
        tensor_index_(tensor_index),
        new_region_(new_region),
        tensor_index_type_(tensor_index_type) {}

  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
    SBlock block = StmtExprMutator::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                       .as_or_throw<SBlock>();

    ffi::Array<TensorRegion> regions =
        tensor_index_type_ == TensorIndexType::kWrite ? block->writes : block->reads;
    TVM_FFI_ICHECK_GE(tensor_index_, 0) << "Tensor index must be non-negative";
    TVM_FFI_ICHECK_LT(tensor_index_, static_cast<int>(regions.size()))
        << "Tensor index out of range";
    regions.Set(tensor_index_, new_region_);

    SBlockObj* n = block.CopyOnWrite();
    if (tensor_index_type_ == TensorIndexType::kWrite) {
      n->writes = std::move(regions);
    } else {
      n->reads = std::move(regions);
    }

    // Annotate the block with explicit_read_region or explicit_write_region
    ffi::Map<ffi::String, ffi::Any> new_annotations = n->annotations;
    ffi::String annotation_key = tensor_index_type_ == TensorIndexType::kWrite
                                     ? s_tir::attr::kExplicitWriteRegion.name
                                     : s_tir::attr::kExplicitReadRegion.name;
    if (new_annotations.count(annotation_key)) {
      ffi::Array<int64_t> tensor_indices =
          new_annotations[annotation_key].as_or_throw<ffi::Array<int64_t>>();
      bool found = false;
      for (int64_t index : tensor_indices) {
        if (index == tensor_index_) {
          found = true;
          break;
        }
      }
      if (!found) {
        tensor_indices.push_back(static_cast<int64_t>(tensor_index_));
        new_annotations.Set(annotation_key, tensor_indices);
      }
    } else {
      new_annotations.Set(annotation_key, ffi::Array<int64_t>{static_cast<int64_t>(tensor_index_)});
    }
    n->annotations = std::move(new_annotations);

    return block;
  }

 private:
  TensorVar tensor_;
  int tensor_index_;
  TensorRegion new_region_;
  TensorIndexType tensor_index_type_;
};

void AnnotateTensorAccess(ScheduleState self, const StmtSRef& block_sref, int tensor_index,
                          TensorIndexType tensor_index_type, const IndexMap& index_map) {
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  TensorVar tensor =
      GetNthAccessTensor(self, ffi::GetRef<SBlock>(block), tensor_index, tensor_index_type);

  sym::Analyzer analyzer;
  ffi::Array<PrimExpr> block_iter_vars;
  for (const IterVar& iter_var : block->iter_vars) {
    block_iter_vars.push_back(iter_var->var);
  }
  ffi::Array<PrimExpr> new_indices = index_map->MapIndices(block_iter_vars, analyzer);
  TVM_FFI_ICHECK_EQ(new_indices.size() % 2, 0) << "The size of new_indices should be even.";
  ffi::Array<Range> new_ranges;
  for (size_t i = 0; i < new_indices.size(); i += 2) {
    // (begin, end) represents a region
    new_ranges.push_back(Range::FromMinExtent(
        new_indices[i], analyzer->Simplify(new_indices[i + 1] - new_indices[i])));
  }

  TensorRegion new_region = TensorRegion(tensor, new_ranges, TensorRegionType());

  auto mutator =
      ffi::make_object<AnnotateRegionRewriter>(tensor, tensor_index, new_region, tensor_index_type);
  Stmt new_stmt = mutator->Mutate(ffi::GetRef<Stmt>(block_sref->stmt))
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(block_sref->stmt));

  self->Replace(block_sref, new_stmt,
                {{ffi::GetRef<SBlock>(block), new_stmt.as_or_throw<SBlock>()}});
}

struct AnnotateTensorAccessTraits : public UnpackedInstTraits<AnnotateTensorAccessTraits> {
  static constexpr const char* kName = "AnnotateTensorAccess";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 4;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block, int tensor_index,
                                      int tensor_index_type, IndexMap index_map) {
    return sch->AnnotateTensorAccess(block, tensor_index,
                                     static_cast<TensorIndexType>(tensor_index_type), index_map);
  }

  static ffi::String IndexMap2GenNewRangesLambda(const IndexMap& index_map) {
    std::ostringstream oss;
    auto print_expr = [&oss](const PrimExpr& expr) {
      if (auto var = expr.as<PrimVar>()) {
        oss << var.value()->name;
      } else {
        oss << expr;
      }
    };
    oss << "lambda ";
    for (size_t i = 0; i < index_map->initial_indices.size(); ++i) {
      if (i != 0) oss << ", ";
      oss << index_map->initial_indices[i]->name;
    }
    oss << ": [";
    for (size_t i = 0; i < index_map->final_indices.size(); i += 2) {
      if (i != 0) oss << ", ";
      if (index_map->final_indices[i].same_as(index_map->final_indices[i + 1])) {
        print_expr(index_map->final_indices[i]);
      } else {
        oss << "(";
        print_expr(index_map->final_indices[i]);
        oss << ", ";
        print_expr(index_map->final_indices[i + 1]);
        oss << ")";
      }
    }
    oss << "]";
    return ffi::String(oss.str());
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      int tensor_index, int tensor_index_type, IndexMap index_map) {
    PythonAPICall py("annotate_tensor_access");
    py.Input("block", block);
    py.Input("tensor_index", tensor_index);

    std::ostringstream os;
    os << "\"" << TensorIndexType2Str(static_cast<TensorIndexType>(tensor_index_type)) << "\"";
    py.Input("tensor_type", ffi::String(os.str()));

    py.Input("gen_new_ranges", IndexMap2GenNewRangesLambda(index_map));
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() { RegisterInstructionKind<AnnotateTensorAccessTraits>(); }

}  // namespace s_tir
}  // namespace tvm
