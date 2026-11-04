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
#include <tvm/ir/function.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/script/ir_builder/frame.h>
#include <tvm/relax/script/ir_builder/ir.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  RelaxFrameObj::RegisterReflection();
  SeqExprFrameObj::RegisterReflection();
  FunctionFrameObj::RegisterReflection();
  BindingBlockFrameObj::RegisterReflection();
  IfFrameObj::RegisterReflection();
  ThenFrameObj::RegisterReflection();
  ElseFrameObj::RegisterReflection();
}

void SeqExprFrameObj::ExitWithScope() {
  // At this moment, there should be at most one BindingBlockFrame which hasn't ended. In this case,
  // call its `ExitBindingBlockFrame` and check if there is any more unended BindingBlockFrame.
  if (ffi::Optional<BindingBlockFrame> block_frame =
          IRBuilder::Current()->GetLastFrame<BindingBlockFrame>()) {
    block_frame.value()->ExitWithScope();
    TVM_FFI_CHECK(!IRBuilder::Current()->GetLastFrame<BindingBlockFrame>().has_value(), ValueError)
        << "There is some remaining BindingBlockFrame that is not properly popped out.";
  }
  RelaxFrameObj::ExitWithScope();
}

void SeqExprFrameObj::EnterWithScope() {
  RelaxFrameObj::EnterWithScope();
  BindingBlockFrame block = BindingBlock();
  block->loc = loc;
  block->EnterWithScope();
}

void FunctionFrameObj::EnterWithScope() {
  if (function.has_value()) {
    ffi::Optional<tvm::IRModule> mod = std::nullopt;
    if (auto frame = IRBuilder::Current()->FindFrame<ir::IRModuleFrame>()) {
      mod = tvm::IRModule(frame.value()->functions);
    }
    block_builder = tvm::relax::BlockBuilder::Create(
        mod, tvm::relax::BlockBuilder::DisableOperatorSpecificNormalizationForTVMScript());
  }
  this->block_builder->BeginScope(params);
  if (declaration) {
    RelaxFrameObj::EnterWithScope();
  } else {
    SeqExprFrameObj::EnterWithScope();
  }
}

void FunctionFrameObj::ExitWithScope() {
  using ir::IRModuleFrame;
  using tvm::relax::Expr;
  IRBuilder builder = IRBuilder::Current();
  if (declaration) {
    TVM_FFI_CHECK(name.has_value(), ValueError) << "A function declaration requires a name";
    RelaxFrameObj::ExitWithScope();
    block_builder->EndScope();
    function = tvm::relax::Function::CreateEmpty(params, ret_ty.value_or(tvm::AnyType()),
                                                 is_pure.value_or(true), DictAttrs(attrs), loc);
    if (local) {
      auto ty = tvm::relax::GetType(function.value());
      local_var = CheckBindingBlockFrameExistAndUnended()->is_dataflow
                      ? tvm::relax::DataflowVar(name.value(), ty, loc)
                      : tvm::Var(name.value(), ty, loc);
    } else if (builder->FindFrame<IRModuleFrame>().has_value()) {
      global_var = ir::DeclFunction(name.value(), function.value());
    }
    declaration = false;
    return;
  }
  SeqExprFrameObj::ExitWithScope();
  // Step 1: Create the function.
  TVM_FFI_CHECK(output.has_value(), ValueError)
      << "A Relax function must have a return value. Please use "
         "`return` to return an Expr";

  Expr body =
      this->block_builder->Normalize(tvm::relax::SeqExpr(binding_blocks, output.value(), loc));
  // if the function is not private, add a global symbol to its attributes
  if (!is_private.value_or(false) && name.has_value() &&
      !attrs.count(tvm::attr::kGlobalSymbol.name)) {
    attrs.Set(tvm::attr::kGlobalSymbol.name, name.value());
  }
  this->block_builder->EndScope();
  tvm::relax::Function func(/*params=*/params,
                            /*body=*/body,
                            /*ret_ty=*/ret_ty,
                            /*is_pure=*/is_pure.value_or(true),
                            /*attrs=*/DictAttrs(attrs),
                            /*loc=*/loc);
  function = func;
  // Step 2: Update IRModule.
  if (local) {
    TVM_FFI_CHECK(local_var.has_value(), ValueError)
        << "A local function definition requires its declared reference";
    bool recursive = false;
    for (const tvm::Var& var : tvm::relax::FreeVars(func)) {
      recursive = recursive || var.same_as(local_var.value());
    }
    // Retain the declared reference identity while publishing its inferred type,
    // just as DefFunction refines a module's global reference after definition.
    Type reference_type = tvm::relax::GetType(func);
    if (recursive) {
      // A recursive reference has its own provisional signature.  Its formal
      // primitive parameters must not bind the definition's parameter objects.
      // Keep lexical captures intact while renewing these signature binders.
      ffi::Map<tvm::Var, tvm::Expr> signature_params;
      for (const tvm::Var& param : params) {
        if (param.as<PrimVar>()) {
          signature_params.Set(param, param.CopyWithName(param->name));
        }
      }
      if (!signature_params.empty()) {
        auto signature = reference_type.as_or_throw<tvm::relax::FuncType>();
        auto bind_param = [&](const Type& ty) { return tvm::relax::Bind(ty, signature_params); };
        reference_type =
            tvm::relax::FuncType(signature->params.value().Map(bind_param),
                                 bind_param(signature->ret), signature->purity, signature->loc);
      }
      if (local_var.value()->IsInstance<tvm::relax::DataflowVarObj>()) {
        // The self-reference is captured by the nested function, so it must
        // survive the dataflow region.  Region finalization rewrites every use
        // together, including recursive calls, to the same ordinary Var.
        CheckBindingBlockFrameExistAndUnended()->output_vars.push_back(local_var.value());
      }
    }
    local_var.value()->ty = reference_type;
    EmitVarBinding(tvm::relax::VarBinding(local_var.value(), func, loc));
  } else if (builder->frames.empty()) {
    // Case 0. No outer frame, return function directly
    TVM_FFI_CHECK(!builder->result.has_value(), ValueError)
        << "Builder.result has already been set";
    builder->result = func;
  } else if (ffi::Optional<IRModuleFrame> opt_frame = builder->FindFrame<IRModuleFrame>()) {
    // Case 1. A global function of an IRModule
    TVM_FFI_CHECK(name.has_value(), ValueError)
        << "The function name must be defined before exiting the "
           "function scope, if it's defined in a Module";
    const IRModuleFrame& frame = opt_frame.value();
    const ffi::String& func_name = name.value_or("");
    if (!frame->global_var_map.count(func_name)) {
      // First time visiting the function.
      ir::DeclFunction(func_name, func);
    }
    // Define the function.
    // Note we do checks to disallow redefinition of functions inside the `DefFunction`.
    ir::DefFunction(func_name, func);
    global_var = frame->global_var_map[func_name];
  } else {
    TVM_FFI_THROW(ValueError) << "Cannot find where to insert Relax.Function";
  }
}

void BindingBlockFrameObj::EnterWithScope() {
  // Step 1. If the last frame is a block frame. The start of a new block frame marks the end of the
  // last block frame.
  ffi::Optional<BindingBlockFrame> block_frame =
      IRBuilder::Current()->GetLastFrame<BindingBlockFrame>();
  if (block_frame.has_value()) {
    block_frame.value()->ExitWithScope();
    // Block frames cannot appear consecutively.
    TVM_FFI_ICHECK(!IRBuilder::Current()->GetLastFrame<BindingBlockFrame>());
  }
  // Step 2. Deal with the new block frame.
  RelaxFrameObj::EnterWithScope();
  ffi::Optional<FunctionFrame> func_frame = IRBuilder::Current()->FindFrame<FunctionFrame>();
  TVM_FFI_CHECK(func_frame.has_value(), ValueError)
      << "Cannot find FunctionFrame when creating BindingBlocks, Please ensure "
         "creating the block under Relax function scope.";
  const tvm::relax::BlockBuilder& block_builder = func_frame.value()->block_builder;
  if (is_dataflow) {
    block_builder->BeginDataflowBlock();
  } else {
    block_builder->BeginBindingBlock();
  }
}

class VarReplacer : public tvm::relax::ExprMutator {
 public:
  explicit VarReplacer(
      std::unordered_map<tvm::Var, tvm::Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> var_remap) {
    var_remap_ = std::move(var_remap);
  }

  tvm::Var VisitVarDef(const tvm::Var& var) override {
    // ExprMutator only applies var_remap_ at usage sites.  This
    // applies var_remap_ at each definition site as well.
    if (auto it = var_remap_.find(var); it != var_remap_.end()) {
      return it->second;
    } else {
      return var;
    }
  }
};

void BindingBlockFrameObj::ExitWithScope() {
  // Step 1. Pop the current frame out of the frame stack.
  RelaxFrameObj::ExitWithScope();

  // Step 2. Get the constructed binding block from the block builder. The block should have at
  // lease one binding - otherwise, the block is not supposed to be created.
  const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();
  tvm::relax::BindingBlock block = block_builder->EndBlock();
  if (block->bindings.empty()) {
    return;
  }

  // Step 3. Rewrite the dataflow block.
  if (is_dataflow) {
    // Step 3.0.  Define a map to replace variables
    ffi::Array<tvm::Var> new_output_vars;
    std::unordered_map<tvm::Var, tvm::Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> var_remap;
    for (const auto& output_var : output_vars) {
      if (var_remap.count(output_var)) {
        continue;
      }
      tvm::Var new_output_var(output_var->name, tvm::relax::GetType(output_var), output_var->loc);
      new_output_vars.push_back(new_output_var);
      if (auto loc = binding_locs.Get(output_var)) {
        binding_locs.Set(new_output_var, loc.value());
      }
      var_remap.insert_or_assign(output_var, new_output_var);
    }
    VarReplacer mutator(std::move(var_remap));

    // Step 3.1. Rewrite block binding
    block = mutator.VisitBindingBlock(block);

    // Step 3.3. Rewrite output vars
    output_vars = std::move(new_output_vars);

    // Step 3.4 Rewrite usage of output var, if any
    auto function = FindFunctionFrame("R.dataflow()");
    if (function->output.has_value()) {
      function->output = mutator.VisitExpr(function->output.value());
    }
  }

  // Variable rewriting may rebuild bindings, so attach their own source ranges last.
  block->loc = loc;
  for (const auto& binding : block->bindings) {
    if (auto loc = binding_locs.Get(binding->var)) {
      binding->loc = loc.value();
    }
  }

  // Step 3. Get the last frame from the IRBuilder frame stack.
  ffi::Optional<RelaxFrame> opt_last_frame = IRBuilder::Current()->GetLastFrame<RelaxFrame>();
  TVM_FFI_ICHECK(opt_last_frame.has_value());
  RelaxFrame last_frame = opt_last_frame.value();

  // Step 4. Since we popped out any possible block frame when entering the "with" scope of the
  // current frame, the last frame cannot be a block frame.
  TVM_FFI_ICHECK(!last_frame->IsInstance<BindingBlockFrameObj>());

  // Step 5. Push the block frame into the corresponding field of the last frame.
  if (const auto* seq_frame = last_frame.as<SeqExprFrameObj>()) {
    auto frame = ffi::GetRef<SeqExprFrame>(seq_frame);
    frame->binding_blocks.push_back(block);
  } else {
    TVM_FFI_THROW(ValueError)
        << "Currently the last frame is supposed to be either a function frame "
           "or a block frame. However, the last frame is \""
        << last_frame->GetTypeKey() << "\".";
  }

  // Step 6. Start another binding block when a dataflow block ended.
  if (is_dataflow) {
    BindingBlockFrame block = BindingBlock();
    block->loc = last_frame->loc;
    block->EnterWithScope();
  }
}

void IfFrameObj::EnterWithScope() {
  const ffi::Array<IRBuilderFrame>& frames = IRBuilder::Current()->frames;
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    if ((*it)->IsInstance<FunctionFrameObj>()) {
      break;
    }
    const auto* block_frame = (*it).as<BindingBlockFrameObj>();
    if (block_frame && block_frame->is_dataflow) {
      TVM_FFI_THROW(ValueError) << "Cannot create an IfFrame inside a dataflow block.";
    }
  }
  RelaxFrameObj::EnterWithScope();
}

void IfFrameObj::ExitWithScope() {
  RelaxFrameObj::ExitWithScope();
  TVM_FFI_CHECK(then_expr.has_value(), ValueError)
      << "The body of then part is expected to be defined before exiting.";
  TVM_FFI_CHECK(else_expr.has_value(), ValueError)
      << "The body of else part is expected to be defined before exiting.";
  auto body = tvm::relax::IfExpr(condition, then_expr.value(), else_expr.value(), loc);
  auto binding_var = Emit(body, std::nullopt);
  var = binding_var;
  // Finalization uses the frame's already-composed location, never the exit context.
  if (!loc.as<UnknownLocObj>()) {
    CheckBindingBlockFrameExistAndUnended()->binding_locs.Set(binding_var, loc);
  }
  binding_var->loc = loc;
  IRBuilder::Name(var_name, binding_var);
}

void ThenFrameObj::EnterWithScope() {
  IfFrame frame = FindIfFrame("R.Then");
  TVM_FFI_CHECK(!frame->then_expr.has_value(), ValueError)
      << "Duplicate then branch declaration, previous one is " << frame->then_expr.value();
  SeqExprFrameObj::EnterWithScope();
}

void ThenFrameObj::ExitWithScope() {
  SeqExprFrameObj::ExitWithScope();
  ffi::String var_name;
  output = GetSeqExprForBranch(ffi::GetRef<ThenFrame>(this), &var_name);
  IfFrame frame = FindIfFrame("R.Then");
  frame->then_expr = output;
  frame->var_name = var_name;
}

void ElseFrameObj::EnterWithScope() {
  IfFrame frame = FindIfFrame("R.Else");
  TVM_FFI_ICHECK(frame->then_expr.has_value()) << "The else branch should follow then branch";
  TVM_FFI_CHECK(!frame->else_expr.has_value(), ValueError)
      << "Duplicate else branch declaration, previous one is " << frame->else_expr.value();
  SeqExprFrameObj::EnterWithScope();
}

void ElseFrameObj::ExitWithScope() {
  SeqExprFrameObj::ExitWithScope();
  ffi::String var_name;
  output = GetSeqExprForBranch(ffi::GetRef<ElseFrame>(this), &var_name);
  IfFrame frame = FindIfFrame("R.Else");
  frame->else_expr = output;
  TVM_FFI_ICHECK(frame->var_name == var_name)
      << "This last binding of both branches must provide the same variable.  "
      << "However, the R.Then branch provides variable " << frame->var_name
      << ", while the R.Else branch provides variable " << var_name;
}

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
