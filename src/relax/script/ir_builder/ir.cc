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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/script/ir_builder/ir.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/logging.h>
#include <tvm/script/ir_builder/frame.h>

#include "./utils.h"

namespace tvm {
namespace script {
using namespace tvm::prim;

namespace ir_builder {
namespace relax {

ffi::Optional<tvm::relax::VDevice> LookupVDevice(ffi::String target_kind, int device_index) {
  if (IRBuilder::IsInScope()) {
    ir::IRModuleFrame frame = ir::FindModuleFrame();
    if (frame->global_infos.empty()) {
      TVM_FFI_THROW(ValueError) << "The GlobalInfos in the IRModule is not defined.";
    }
    ffi::Array<GlobalInfo> vdevices = frame->global_infos["vdevice"];
    if (vdevices.empty() || device_index < 0 ||
        static_cast<size_t>(device_index) >= vdevices.size()) {
      TVM_FFI_THROW(ValueError) << "The target VDevice in the GlobalInfos was not found.";
    }
    if (target_kind == "vdevice") {
      return vdevices[device_index].as_or_throw<tvm::relax::VDevice>();
    }
    int count = 0;
    for (auto vdevice : vdevices) {
      auto vdev = vdevice.as_or_throw<tvm::relax::VDevice>();
      if (vdev->target.has_value() && vdev->target.value()->kind->name == target_kind) {
        if (count == device_index) {
          return vdev;
        }
        count++;
      }
    }
  }
  LOG(WARNING) << "The annotated device was not found, please check your vdevice list.";
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.ir_builder.relax.LookupVDevice", LookupVDevice);
}

///////////////////////////////// Vars //////////////////////////////////

using tvm::script::ir_builder::detail::Namer;

TVM_FFI_STATIC_INIT_BLOCK() {
  Namer::vtable().SetDispatch<tvm::relax::DataflowVarObj>(
      [](const ffi::ObjectRef& node, ffi::String name) -> void {
        using tvm::relax::DataflowVarObj;
        DataflowVarObj* var = const_cast<DataflowVarObj*>(node.as<DataflowVarObj>());
        var->name = name;
      });
}

/////////////////////////////// Function ////////////////////////////////

FunctionFrame Function(bool is_pure, bool is_private, bool is_declaration, bool local) {
  const IRBuilder& ir_builder = IRBuilder::Current();
  ffi::Optional<tvm::IRModule> mod = std::nullopt;
  if (const ffi::Optional<ir::IRModuleFrame> mod_frame =
          ir_builder->GetLastFrame<ir::IRModuleFrame>()) {
    mod = tvm::IRModule(mod_frame.value()->functions);
  }
  auto n = ffi::make_object<FunctionFrameObj>(tvm::relax::BlockBuilder::Create(
      /*mod=*/mod, tvm::relax::BlockBuilder::DisableOperatorSpecificNormalizationForTVMScript()));
  n->is_pure = is_pure;
  n->is_private = is_private || local;
  n->declaration = is_declaration;
  n->local = local;
  return FunctionFrame(n);
}

FunctionFrame LocalFunction(bool is_pure, const tvm::Var& reference) {
  FunctionFrame frame = Function(is_pure, true);
  frame->local = true;
  frame->local_var = reference;
  return frame;
}

tvm::Var ArgVar(const ffi::String& name, const tvm::Var& var) {
  FunctionFrame frame = FindFunctionFrame("R.arg");
  TVM_FFI_CHECK(var->name == name, ValueError)
      << "A cached parameter must retain its declaration name";
  for (const auto& param : frame->params) {
    TVM_FFI_CHECK(param->name != name, ValueError) << "Duplicate function parameter: " << name;
  }
  frame->params.push_back(var);
  frame->block_builder->AddDefinitionToScope(var);
  return var;
}

tvm::Var Arg(const ffi::String& name, const tvm::Type& ty) {
  FunctionFrame frame = FindFunctionFrame("R.Arg");
  tvm::Var var(name, ty);
  frame->params.push_back(var);
  frame->block_builder->AddDefinitionToScope(var);

  return var;
}

void FuncName(const ffi::String& name) {
  FunctionFrame frame = FindFunctionFrame("R.func_name");
  if (frame->name.has_value()) {
    TVM_FFI_THROW(ValueError) << "Duplicate function name, previous one is: \""
                              << frame->name.value() << "\"";
  }
  frame->name = name;
}

void FuncAttrs(ffi::Map<ffi::String, ffi::Any> attrs) {
  FunctionFrame frame = FindFunctionFrame("R.func_attr");
  for (const auto& [key, value] : attrs) {
    if (key == tvm::attr::kGlobalSymbol.name && frame->is_private.value_or(false)) {
      TVM_FFI_THROW(ValueError) << "A private function may not have the kGlobalSymbol (\""
                                << tvm::attr::kGlobalSymbol.name << "\") attribute.  "
                                << "However, a private function specified the global symbol as "
                                << value;
    }
    if (auto prev = frame->attrs.Get(key)) {
      TVM_FFI_THROW(ValueError) << "Duplicate R.func_attr annotation for key = \"" << key << "\".  "
                                << "Previous value was " << prev.value()
                                << ", with later definition as " << value;
    } else {
      frame->attrs.Set(key, value);
    }
  }
}

void FuncRetType(const tvm::Type& ret_ty) {
  FunctionFrame frame = FindFunctionFrame("R.func_ret_type");
  if (frame->ret_ty.has_value()) {
    TVM_FFI_THROW(ValueError) << "Duplicate function return type, previous one is:\n "
                              << frame->ret_ty.value();
  }
  frame->ret_ty = ret_ty;
}

void FuncRetValue(const tvm::relax::Expr& value) {
  // Step 0. Normalize the value.
  const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();
  tvm::relax::Expr normalized_value = block_builder->Normalize(value);

  IRBuilder ir_builder = IRBuilder::Current();

  // Step 1. The current Relax TVMScript syntax only allows function return appearing at the end of
  // a function body. Therefore if there is any unended block frame when dealing with function
  // return, we should end the block frame.

  if (auto opt = ir_builder->GetLastFrame<BindingBlockFrame>()) {
    auto block_frame = opt.value();
    for (const auto& var : tvm::relax::FreeVars(normalized_value)) {
      if (var->IsInstance<tvm::relax::DataflowVarObj>()) {
        block_frame->output_vars.push_back(var);
      }
    }
  }
  // Step 2. Add the output value to the function frame.
  FunctionFrame frame = FindFunctionFrame("return");
  TVM_FFI_CHECK(!frame->output.has_value(), ValueError)
      << "Relax functions do not support multiple return statement.  "
      << "However, return of " << normalized_value << " occurred after a return of "
      << frame->output << ".  "
      << "Please make sure function only has a single return statement, "
      << "which appears at the end of function.";

  frame->output = std::move(normalized_value);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.relax.Function", Function)
      .def("script.ir_builder.relax.LocalFunction", LocalFunction)
      .def("script.ir_builder.relax.ArgVar", ArgVar)
      .def("script.ir_builder.relax.Arg", Arg)
      .def("script.ir_builder.relax.FuncName", FuncName)
      .def("script.ir_builder.relax.FuncAttrs", FuncAttrs)
      .def("script.ir_builder.relax.FuncRetType", FuncRetType)
      .def("script.ir_builder.relax.FuncRetValue", FuncRetValue);
}

///////////////////////////// BindingBlock //////////////////////////////

BindingBlockFrame Dataflow() {
  ffi::ObjectPtr<BindingBlockFrameObj> n = ffi::make_object<BindingBlockFrameObj>();
  n->is_dataflow = true;
  n->block_ended = false;
  return BindingBlockFrame(n);
}

BindingBlockFrame BindingBlock() {
  ffi::ObjectPtr<BindingBlockFrameObj> n = ffi::make_object<BindingBlockFrameObj>();
  n->is_dataflow = false;
  n->block_ended = false;
  return BindingBlockFrame(n);
}

void DataflowBlockOutput(const ffi::Array<tvm::Var>& vars) {
  // Step 1. Check that we're in a Dataflow block that is not ended.
  ffi::Optional<BindingBlockFrame> block_frame =
      IRBuilder::Current()->GetLastFrame<BindingBlockFrame>();
  TVM_FFI_CHECK(block_frame.has_value() && block_frame.value()->is_dataflow, ValueError)
      << "`R.output` should appear inside a dataflow block. However, the current "
         "innermost block is not a dataflow block.";
  TVM_FFI_CHECK(!block_frame.value()->block_ended, ValueError)
      << "It is not allowed for a dataflow block to have multiple output operation.";

  // Step 2. Mark the block frame ended of construction, so that any followup binding after this
  // mark in the dataflow block will lead to an error.
  block_frame.value()->block_ended = true;

  // Step 3. All the output variables must be global variables and must be emitted by this dataflow
  // block.
  const ffi::Array<tvm::Var>& emitted_vars = block_frame.value()->emitted_vars;
  for (const tvm::Var& var : vars) {
    TVM_FFI_CHECK(std::find_if(emitted_vars.begin(), emitted_vars.end(),
                               [&](const tvm::Var& emitted) { return emitted.same_as(var); }) !=
                      emitted_vars.end(),
                  ValueError)
        << "An output variable is not emitted by this dataflow block. Please make sure "
           "all dataflow block output variables are emitted exactly by this block.";
    block_frame.value()->output_vars.push_back(var);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.relax.Dataflow", Dataflow)
      .def("script.ir_builder.relax.BindingBlock", BindingBlock)
      .def("script.ir_builder.relax.DataflowBlockOutput", DataflowBlockOutput);
}

/////////////////////////////// Bindings ///////////////////////////////

tvm::Var Emit(const tvm::relax::Expr& expr, const ffi::Optional<tvm::Type>& annotate_ty) {
  using tvm::relax::GetType;
  BindingBlockFrame block_frame = CheckBindingBlockFrameExistAndUnended();
  const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();
  if (annotate_ty.has_value()) {
    const auto& ty = annotate_ty.value();
    if (expr->ty.as<MissingType>().has_value()) {
      tvm::relax::UpdateType(expr, ty);
    } else {
      TVM_FFI_ICHECK(tvm::relax::TypeBaseCheck(ty, GetType(expr)) !=
                     tvm::relax::BaseCheckResult::kFailL0)
          << "Invalid annotation. Got rhs value type: " << GetType(expr) << ", given type: " << ty;
    }
  }
  tvm::Var var = block_builder->Emit(expr);
  block_frame->emitted_vars.push_back(var);
  return var;
}

tvm::Var EmitMatchCast(const tvm::relax::Expr& value, const tvm::Type& ty) {
  BindingBlockFrame block_frame = CheckBindingBlockFrameExistAndUnended();
  const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();

  tvm::Var var = block_builder->EmitMatchCast(value, ty);
  block_frame->emitted_vars.push_back(var);
  return var;
}

tvm::Var EmitVarBinding(const tvm::relax::VarBinding& binding) {
  BindingBlockFrame block_frame = CheckBindingBlockFrameExistAndUnended();
  const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();
  block_builder->EmitNormalized(binding);
  block_frame->emitted_vars.push_back(binding->var);
  return binding->var;
}

namespace {

tvm::Var RecordBindingLoc(tvm::Var var, const Location& name_loc, const Location& statement_loc) {
  Location loc = IRBuilder::Current()->ResolveLoc(statement_loc);
  if (!loc.as<UnknownLocObj>()) {
    CheckBindingBlockFrameExistAndUnended()->binding_locs.Set(var, loc);
  }
  var->loc = name_loc.as<UnknownLocObj>() ? loc : name_loc;
  return var;
}

}  // namespace

tvm::Var EmitWithLoc(const tvm::relax::Expr& value, const ffi::Optional<tvm::Type>& annotate_ty,
                     const Location& name_loc, const Location& loc) {
  return RecordBindingLoc(Emit(value, annotate_ty), name_loc, loc);
}

tvm::Var EmitMatchCastWithLoc(const tvm::relax::Expr& value, const tvm::Type& ty,
                              const Location& name_loc, const Location& loc) {
  return RecordBindingLoc(EmitMatchCast(value, ty), name_loc, loc);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.relax.Emit", Emit)
      .def("script.ir_builder.relax.EmitMatchCast", EmitMatchCast)
      .def("script.ir_builder.relax.EmitVarBinding", EmitVarBinding)
      .def("script.ir_builder.relax.EmitWithLoc", EmitWithLoc)
      .def("script.ir_builder.relax.EmitMatchCastWithLoc", EmitMatchCastWithLoc);
}

/////////////////////////////// SeqExpr ///////////////////////////////

SeqExprFrame SeqExpr() {
  ffi::ObjectPtr<SeqExprFrameObj> n = ffi::make_object<SeqExprFrameObj>();
  return SeqExprFrame(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.ir_builder.relax.SeqExpr", SeqExpr);
}

///////////////////////////// If Then Else /////////////////////////////

IfFrame If(tvm::relax::Expr condition) {
  ffi::ObjectPtr<IfFrameObj> n = ffi::make_object<IfFrameObj>(condition);
  n->then_expr = std::nullopt;
  n->else_expr = std::nullopt;
  return IfFrame(n);
}

ThenFrame Then() {
  ffi::ObjectPtr<ThenFrameObj> n = ffi::make_object<ThenFrameObj>();
  return ThenFrame(n);
}

ElseFrame Else() {
  ffi::ObjectPtr<ElseFrameObj> n = ffi::make_object<ElseFrameObj>();
  return ElseFrame(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.relax.If", If)
      .def("script.ir_builder.relax.Then", Then)
      .def("script.ir_builder.relax.Else", Else);
}

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
