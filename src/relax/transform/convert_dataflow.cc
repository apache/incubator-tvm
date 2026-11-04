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
 * \file tvm/relax/transform/convert_dataflow.cc
 * \brief Pass for extracting groups of pure operations without
 *   dataflow into dataflow blocks.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/utils.h>

#include <optional>

namespace tvm {
namespace relax {

class DataflowBlockExtractor : public ExprMutator {
 public:
  explicit DataflowBlockExtractor(size_t min_size) : ExprMutator(), min_size_(min_size) {}

  Expr VisitExpr_(const SeqExprObj* seq) override {
    ffi::Array<BindingBlock> new_blocks;
    Expr new_body = VisitExpr(seq->body);
    bool changed = !new_body.same_as(seq->body);

    // Accumulated bindings that are not going to be added to a
    // DataflowBlock, either because they would be illegal within a
    // DataflowBlock, or because there were insufficient bindings to
    // make a dataflowblock.  Because these bindings occur prior to
    // `dataflow_bindings`, this array may only be accumulated into
    // when `dataflow_bindings` is empty.
    ffi::Array<Binding> non_dataflow_bindings;

    // Current bindings that may legally be added to a DataflowBlock.
    ffi::Array<Binding> dataflow_bindings;

    // If present, a DataflowBlock whose bindings are currently in
    // `dataflow_bindings`.  Used to propagate DataflowBlock to the
    // output, even if it doesn't meet the minimum size.
    ffi::Optional<DataflowBlock> input_dataflow_block;

    // Handle any bindings currently in `dataflow_bindings`.  These
    // are either pushed to their own block, or to the end of
    // `non_dataflow_bindings`, depending on whether the bindings meet
    // the minimum size requirement.
    auto push_dataflow_bindings = [&]() {
      if (dataflow_bindings.empty()) {
        // No Dataflow bindings, so no action required.
        return;
      }
      if (dataflow_bindings.size() < min_size_ && !input_dataflow_block) {
        // The df block is below the minimum length, and no input
        // DataflowBlock needs to be preserved.  Combine the blocks
        // and reset the dataflow collection.

        non_dataflow_bindings.insert(non_dataflow_bindings.end(), dataflow_bindings.begin(),
                                     dataflow_bindings.end());

      } else {
        // A new DataflowBlock can be generated, with bindings that
        // occur after the non-dataflow bindings.
        new_blocks.push_back(BindingBlock(non_dataflow_bindings));
        new_blocks.push_back(DataflowBlock(dataflow_bindings));
        non_dataflow_bindings = {};

        // Making a dataflow block doesn't imply that the function was
        // changed.  A change requires that this either be a new
        // dataflow block, or have additional dataflow bindings in the
        // current block.
        changed = changed || !input_dataflow_block.has_value() ||
                  input_dataflow_block.value()->bindings.size() != dataflow_bindings.size();
      }

      dataflow_bindings = {};
      input_dataflow_block = std::nullopt;
    };

    for (auto block : seq->blocks) {
      BindingBlock new_block = this->VisitBindingBlock(block);
      changed = changed || !new_block.same_as(block);

      // For an existing dataflow block, we add to the current streak
      // or start a new streak in case there will be more dataflow operations
      // coming up
      if (auto dataflow_block = new_block.as<DataflowBlock>()) {
        dataflow_bindings.insert(dataflow_bindings.end(), new_block->bindings.begin(),
                                 new_block->bindings.end());
        input_dataflow_block = dataflow_block;
        continue;
      }

      // for a binding block, attempt to extract dataflow blocks inside
      auto binding_block = new_block.as_or_throw<BindingBlock>();
      for (const auto& binding : binding_block->bindings) {
        Expr value = GetBoundValue(binding);
        // dataflow values: not an if node and not an impure call
        bool is_dataflow = (!value.as<IfExprObj>()) &&
                           (!(value.as<CallObj>() && IsImpureCall(value.as_or_throw<Call>())));
        if (is_dataflow) {
          // extend the streak
          dataflow_bindings.push_back(binding);
        } else {
          // End the streak, if one currently exists.
          push_dataflow_bindings();
          non_dataflow_bindings.push_back(binding);
        }
      }
    }

    // handle any remaining bindings
    push_dataflow_bindings();
    new_blocks.push_back(BindingBlock(non_dataflow_bindings));

    if (changed) {
      return SeqExpr(new_blocks, new_body);
    } else {
      return ffi::GetRef<SeqExpr>(seq);
    }
  }

 private:
  size_t min_size_;
};

Expr ConvertToDataflow(const Expr& input, size_t min_size) {
  DataflowBlockExtractor extractor(min_size);
  return extractor.VisitExpr(input);
}

namespace transform {

Pass ConvertToDataflow(int min_size) {
  auto pass_func = [=](Function f, IRModule m, PassContext pc) {
    return ConvertToDataflow(f, min_size).as_or_throw<Function>();
  };
  auto pass = CreateFunctionPass(pass_func, 0, "ConvertToDataflow");
  // Canonicalize bindings is included afterwards in order to transform any
  // normal vars in DF blocks that are not used outside the DF block into
  // dataflow vars. This allows us to avoid reimplementing that functionality.
  return tvm::transform::Sequential({pass, CanonicalizeBindings()});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.ConvertToDataflow", ConvertToDataflow);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
