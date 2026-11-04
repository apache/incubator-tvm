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
 * \file src/relax/backend/vm/codegen_vm.cc
 * \brief A codegen to generate VM executable from a Relax IRModule.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/relax/exec_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr.h>
#include <tvm/runtime/vm/bytecode.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../../../runtime/vm/const_loader_module.h"
#include "../../transform/utils.h"

namespace tvm {
namespace relax {
namespace codegen_vm {

using tvm::Target;
using namespace relax;
using namespace tvm::runtime;
using namespace tvm::runtime::vm;

/*!
 * \brief A class to generate VM executable for Relax functions.
 */
class CodeGenVM : public ExprFunctor<Instruction::Arg(const Expr&)> {
 public:
  explicit CodeGenVM(relax::ExecBuilder builder, IRModule ctx_mod)
      : builder_(builder), ctx_mod_(ctx_mod) {}

  static IRModule Run(relax::ExecBuilder builder, IRModule mod) {
    IRModule res_mod = mod;
    res_mod.CopyOnWrite();
    CodeGenVM codegen(builder, mod);
    // Remove relax function and turn into TIR func.
    for (const auto& [gvar, f] : mod->functions) {
      if (auto* func = f.as<FunctionObj>()) {
        codegen.Codegen(ffi::GetRef<Function>(func));
        res_mod->Remove(gvar);
      }
    }
    return res_mod;
  }

 protected:
  size_t NewRegister() { return registers_num_++; }

  // Convert Arg value to a register, trigger copy if needed
  Instruction::Arg EnsureReg(Instruction::Arg arg) {
    if (arg.kind() == Instruction::ArgKind::kRegister) {
      return arg;
    } else {
      RegName dst_reg = NewRegister();
      builder_->EmitCall("vm.builtin.copy", {arg}, dst_reg);
      return Instruction::Arg::Register(dst_reg);
    }
  }

  void Codegen(const Function& func) {
    ffi::Optional<ffi::String> gsymbol = func->GetAttr(tvm::attr::kGlobalSymbol);
    TVM_FFI_ICHECK(gsymbol.has_value())
        << "there should be no local functions in Relax VM codegen phase. "
           "Did you forget to apply LambdaLift or AttachGlobalSymbol Pass?";

    ffi::Array<ffi::String> param_names;
    for (Var param : func->params) {
      param_names.push_back(param->name);
    }

    builder_->EmitFunction(gsymbol.value(), func->params.size(), param_names);

    for (size_t i = 0; i < func->params.size(); ++i) {
      RegName r = NewRegister();
      TVM_FFI_ICHECK_EQ(r, static_cast<RegName>(i));
      this->var_arg_map_.insert({func->params[i], Instruction::Arg::Register(r)});
    }
    Instruction::Arg ret = ExprFunctor::VisitExpr(func->body);
    builder_->EmitRet(EnsureReg(ret));
    builder_->EndFunction(gsymbol.value());
    // reset register number to be 0;
    registers_num_ = 0;
    var_arg_map_.clear();
  }

  Instruction::Arg VisitExpr_(const SeqExprObj* op) final {
    for (auto block : op->blocks) {
      for (Binding binding : block->bindings) {
        Expr expr = GetBoundValue(binding);

        Instruction::Arg value = VisitExpr(expr);
        if (expr.as<VarObj>()) {
          // For a normalized relax module, there should be one
          // register for each relax::Binding.  This makes the Relax
          // semantics of R.vm.kill_* operate the same as the Python
          // "del" operator.  These bindings may be removable by using
          // relax.transform.CanonicalizeBindings earlier in lowering.
          RegName new_reg = NewRegister();
          builder_->EmitCall("vm.builtin.copy", {value}, new_reg);
          value = Instruction::Arg::Register(new_reg);
        }

        this->var_arg_map_.insert({binding->var, value});
      }
    }

    Instruction::Arg ret_reg = this->VisitExpr(op->body);
    return ret_reg;
  }

  Instruction::Arg VisitExpr_(const CallObj* call_node) final {
    Call call = ffi::GetRef<Call>(call_node);

    // ComputePrimValue may introduce pure native calls after purity removal.
    // Their runtime behavior is the explicit bridge underneath the wrapper.
    if (call_node->op.same_as(Op::Get("relax.call_pure_packed")) &&
        call_node->args[0].same_as(Op::Get("relax.call_tir_packed"))) {
      Call inner = Call(call_node->ty, call_node->args[0],
                        ffi::Array<Expr>(call_node->args.begin() + 1, call_node->args.end()),
                        call_node->attrs, call_node->ty_args, call_node->loc);
      return VisitExpr(inner);
    }

    if (call_node->op.same_as(null_value_op_)) {
      return Instruction::Arg::Register(Instruction::kVoidRegister);
    }

    // allocate dst register.
    RegName dst_reg = HasVoidType(call) ? Instruction::kVoidRegister : NewRegister();
    if (call->op.as<OpObj>()) {
      if (call_node->op.same_as(call_builtin_with_ctx_op_)) {
        // TODO(relax-team) migrate most handling of op to
        // directly map to call_builtin_with_ctx before codegen and simplify vm codegen.
        EmitCallBuiltinWithCtx(call, dst_reg);
      } else if (call_node->op.same_as(alloc_storage_op_)) {
        EmitAllocStorage(call, dst_reg);
      } else if (call_node->op.same_as(alloc_tensor_op_)) {
        EmitAllocTensor(call, dst_reg);
      } else if (call_node->op.same_as(Op::Get("relax.call_tir_packed"))) {
        auto arguments = call_node->args[1].as_or_throw<Tuple>();
        builder_->EmitCall(VisitExpr(call_node->args[0]), VisitArray(arguments->fields), dst_reg);
      } else if (call_node->op.same_as(kill_object_op_)) {
        dst_reg = EmitKillObject(call);
      } else {
        // every "normal" operator is lowered to a global var in the IRModule. The Attrs for those
        // ops are handled in a pass when lowering them to TIR.
        TVM_FFI_THROW(InternalError)
            << "CodeGenVM cannot emit this Relax operator directly. "
            << "Run the appropriate lowering pass, or route the operator to an external "
            << "codegen before VM codegen.\nOffending call:\n"
            << call;
      }
    } else {
      EmitNormalCall(call, dst_reg);
    }
    return Instruction::Arg::Register(dst_reg);
  }

  Instruction::Arg VisitExpr_(const IfExprObj* op) final {
    const IfExpr& ife = ffi::GetRef<IfExpr>(op);
    Instruction::Arg cond_value = this->VisitExpr(ife->cond);

    // Reserve a register for cond
    RegName cond_reg = NewRegister();
    builder_->EmitCall("vm.builtin.read_if_cond", {cond_value}, cond_reg);

    // obtain the temp exec in progress.
    vm::VMExecutable* exec = builder_->exec();

    // Record the offset of If instruction
    size_t if_offset = exec->instr_offset.size();

    builder_->EmitIf(Instruction::Arg::Register(cond_reg), 3);
    size_t num_instr = exec->instr_offset.size();
    Instruction::Arg true_value = this->VisitExpr(ife->true_branch);
    // Reserve a register for return
    size_t merge_register = NewRegister();
    // Copy the output from true branch to merge register
    builder_->EmitCall("vm.builtin.copy", {true_value}, merge_register);

    // Record the offset of Goto instruction
    size_t goto_offset = exec->instr_offset.size();

    builder_->EmitGoto(1);

    // Calculate the false offset of If
    size_t false_offset = exec->instr_offset.size() - num_instr + 1;

    Instruction::Arg false_value = this->VisitExpr(ife->false_branch);
    // Copy the output data of false branch to merge register
    builder_->EmitCall("vm.builtin.copy", {false_value}, merge_register);

    // Update the offsets of the If instruction emitted above
    // Jump to the behind of the next goto instruction
    exec->SetInstructionData(if_offset, 2, static_cast<ExecWord>(false_offset));
    // Update the pc_offset of Goto instruction
    // Jump over the false branch
    size_t pc_offset = exec->instr_offset.size() - goto_offset;
    exec->SetInstructionData(goto_offset, 1, static_cast<ExecWord>(pc_offset));
    return Instruction::Arg::Register(merge_register);
  }

  Instruction::Arg VisitExpr_(const VarObj* op) final {
    Var var = ffi::GetRef<Var>(op);
    auto it = this->var_arg_map_.find(var);
    TVM_FFI_ICHECK(it != this->var_arg_map_.end()) << "Var " << var << " is not defined";
    return it->second;
  }

  Instruction::Arg VisitExpr_(const DataflowVarObj* op) final {
    return VisitExpr_(static_cast<const VarObj*>(op));
  }

  Instruction::Arg VisitExpr_(const GenericConstObj* op) final {
    auto arg = builder_->ConvertConstant(op->value);

    if (auto tensor_ty = op->ty.as<TensorTypeObj>()) {
      if (tensor_ty->vdevice.has_value()) {
        VDevice vdev = tensor_ty->vdevice.value();
        builder_->SaveMemoryScope(arg, vdev->memory_scope);
      }
    }
    return arg;
  }

  Instruction::Arg VisitExpr_(const ShapeExprObj* op) final {
    std::vector<int64_t> shape;
    for (PrimExpr e : op->values) {
      if (auto* int_value = e.as<IntImmObj>()) {
        shape.push_back(static_cast<int64_t>(int_value->value));
      } else {
        TVM_FFI_THROW(InternalError)
            << "Should only use constant shape after shape lowering: " << op->values;
      }
    }
    return builder_->ConvertConstant(ffi::Shape(shape));
  }

  Instruction::Arg VisitExpr_(const IntImmObj* op) final {
    return builder_->ConvertConstant(op->value);
  }

  Instruction::Arg VisitExpr_(const FloatImmObj* op) final {
    return builder_->ConvertConstant(op->value);
  }

  Instruction::Arg VisitExpr_(const StringImmObj* op) final {
    return builder_->ConvertConstant(op->value);
  }
  Instruction::Arg VisitExpr_(const DataTypeImmObj* op) final {
    return builder_->ConvertConstant(op->value);
  }

  Instruction::Arg VisitExpr_(const TupleObj* op) final {
    Tuple tuple = ffi::GetRef<Tuple>(op);
    std::vector<Instruction::Arg> args;
    for (Expr arg : tuple->fields) {
      args.push_back(this->VisitExpr(arg));
    }
    size_t dst_register = NewRegister();
    builder_->EmitCall("vm.builtin.make_tuple", args, dst_register);

    return Instruction::Arg::Register(dst_register);
  }

  Instruction::Arg VisitExpr_(const TupleGetItemObj* op) final {
    TupleGetItem expr = ffi::GetRef<TupleGetItem>(op);
    std::vector<Instruction::Arg> args = {this->VisitExpr(expr->tuple)};

    args.push_back(builder_->ConvertConstant(expr->index));

    size_t dst_register = NewRegister();
    builder_->EmitCall("vm.builtin.tuple_getitem", args, dst_register);

    return Instruction::Arg::Register(dst_register);
  }

  Instruction::Arg VisitExpr_(const GlobalVarObj* op) final {
    GlobalVar gvar = ffi::GetRef<GlobalVar>(op);
    ffi::Optional<ffi::String> symbol;
    VMFuncInfo::FuncKind kind = VMFuncInfo::FuncKind::kPackedFunc;

    // Run a look up in the env to see if it maps to an extern func.
    auto it = ctx_mod_->functions.find(gvar);
    if (it != ctx_mod_->functions.end()) {
      BaseFunc func = (*it).second;
      if (auto* efunc = func.as<ExternFuncObj>()) {
        symbol = efunc->global_symbol;
        kind = VMFuncInfo::FuncKind::kPackedFunc;
      } else if (func.as<FunctionObj>()) {
        symbol = gvar->name_hint;
        kind = VMFuncInfo::FuncKind::kVMFunc;
      }
    }
    // GlobalVar can be reference to a Relax function or a TIR function
    // At this point: all global var must corresponds to the right symbol.
    // TODO(relax-team): switch everything to extern before splitting TIR/relax
    // so we do not have idle global var here.
    if (!symbol.has_value()) {
      symbol = gvar->name_hint;
      kind = VMFuncInfo::FuncKind::kPackedFunc;
    }
    // declare the function to be safe.
    TVM_FFI_ICHECK(symbol.has_value());
    builder_->DeclareFunction(symbol.value(), kind);
    return builder_->GetFunction(symbol.value());
  }

  Instruction::Arg VisitExpr_(const ExternFuncObj* op) final {
    if (ffi::Optional<ffi::String> opt_code = op->attrs.GetAttr(tvm::relax::attr::kCSource)) {
      ffi::String sym = op->global_symbol;
      ffi::String fmt = op->attrs.GetAttr(tvm::relax::attr::kCSourceFmt).value_or("c");
      ffi::String code = opt_code.value();
      // Grep hint: grep -rn 'runtime.CSourceModuleCreate' src/
      static const auto create_source =
          ffi::Function::GetGlobalRequired("runtime.CSourceModuleCreate");
      ffi::Module c_source_module =
          create_source(code, fmt, ffi::Array<ffi::String>{sym}, ffi::Array<ffi::String>{})
              .cast<ffi::Module>();
      builder_->exec()->ImportModule(c_source_module);
    }
    builder_->DeclareFunction(op->global_symbol, VMFuncInfo::FuncKind::kPackedFunc);
    return builder_->GetFunction(op->global_symbol);
  }

  void EmitAllocStorage(const Call& call_node, RegName dst_reg) {
    TVM_FFI_ICHECK_EQ(call_node->args.size(), 4);
    // Handle args of the call
    std::vector<Instruction::Arg> args;
    args.push_back(Instruction::Arg::Register(Instruction::kVMRegister));
    // buffer size, dtype, device index
    for (auto arg : call_node->args) {
      args.push_back(this->VisitExpr(arg));
    }
    builder_->EmitCall("vm.builtin.alloc_storage", args, dst_reg);
  }

  void EmitAllocTensor(const Call& call_node, RegName dst_reg) {
    TVM_FFI_ICHECK_EQ(call_node->args.size(), 5);
    std::vector<Instruction::Arg> args;
    args.push_back(Instruction::Arg::Register(Instruction::kVMRegister));
    for (int i = 0; i < 4; ++i) {
      args.push_back(this->VisitExpr(call_node->args[i]));
    }
    int64_t vdevice_index = -1;
    if (const auto* int_imm = call_node->args[4].as<IntImmObj>()) {
      vdevice_index = int_imm->value.as<int>().value();
    }
    auto vdevice = GetGlobalVDevice(ctx_mod_, vdevice_index);

    if (vdevice.has_value()) {
      args.push_back(this->VisitExpr(StringImm(vdevice.value()->memory_scope.name())));
    }
    builder_->EmitCall("vm.builtin.alloc_tensor", args, dst_reg);
  }

  RegName EmitKillObject(const Call& call_node) {
    TVM_FFI_ICHECK_EQ(call_node->args.size(), 1);
    Instruction::Arg arg = this->VisitExpr(call_node->args[0]);
    TVM_FFI_ICHECK(arg.kind() == Instruction::ArgKind::kRegister)
        << "Expected the object to be killed to be stored in a register, "
        << "but argument " << call_node->args[0] << " produced VM instruction of type "
        << arg.kind();
    RegName dst_reg = arg.value();
    builder_->EmitCall("vm.builtin.null_value", {}, dst_reg);
    return dst_reg;
  }

  void EmitCallBuiltinWithCtx(const Call& call_node, RegName dst_reg) {
    std::vector<Instruction::Arg> args;
    args.push_back(Instruction::Arg::Register(Instruction::kVMRegister));

    auto func = this->VisitExpr(call_node->args[0]);
    auto tuple_arg = call_node->args[1].as_or_throw<Tuple>();

    // Handle args of the call
    for (Expr arg : tuple_arg->fields) {
      args.push_back(this->VisitExpr(arg));
    }

    builder_->EmitCall(func, args, dst_reg);
  }
  void EmitNormalCall(const Call& call_node, RegName dst_reg) {
    Instruction::Arg func = VisitExpr(call_node->op);
    std::vector<Instruction::Arg> args = VisitArray(call_node->args);

    if (func.kind() == vm::Instruction::ArgKind::kFuncIdx) {
      builder_->EmitCall(func, args, dst_reg);
    } else {
      std::vector<Instruction::Arg> closure_args = {
          Instruction::Arg::Register(Instruction::kVMRegister), func};
      std::copy(args.begin(), args.end(), std::back_inserter(closure_args));
      builder_->EmitCall("vm.builtin.invoke_closure", closure_args, dst_reg);
    }
  }

  // Emits call to packed function `name` with arguments copied over from `call_node` args
  void EmitPackedFuncCall(const Call& call_node, const ffi::String& name, RegName dst_reg) {
    std::vector<Instruction::Arg> args = VisitArray(call_node->args);
    builder_->EmitCall(name, args, dst_reg);
  }

  std::vector<Instruction::Arg> VisitArray(const ffi::Array<Expr>& arr) {
    std::vector<Instruction::Arg> ret;
    for (size_t i = 0; i < arr.size(); ++i) {
      ret.push_back(this->VisitExpr(arr[i]));
    }
    return ret;
  }

  /*! \brief Internal ExecBuilder. */
  relax::ExecBuilder builder_;
  /*!
   * \brief Total number of virtual registers allocated.
   * \note The first two registers are reserved for special registers.
   */
  size_t registers_num_ = 0;
  /*! \brief Map from var to register number. */
  std::unordered_map<Var, Instruction::Arg> var_arg_map_;
  /*! \brief the context module. */
  IRModule ctx_mod_;
  /*! \brief Cache ops that need to be frequently used later to reduce lookup overhead. */
  const Op alloc_storage_op_ = Op::Get("relax.vm.alloc_storage");
  const Op alloc_tensor_op_ = Op::Get("relax.vm.alloc_tensor");
  const Op kill_object_op_ = Op::Get("relax.vm.kill_object");
  const Op call_builtin_with_ctx_op_ = Op::Get("relax.call_builtin_with_ctx");
  const Op null_value_op_ = Op::Get("relax.null_value");
};

/*!
 * \brief Create the Relax VM executable from all relax.Function in mod.
 *        and add them to exec_builder.
 * \param exec_builder Builder to collect executables.
 * \param mod Input module.
 * \return Left over IRModule that may contain other functions.
 */
IRModule VMCodeGen(ExecBuilder exec_builder, IRModule mod) {
  return CodeGenVM::Run(exec_builder, mod);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.VMCodeGen", VMCodeGen);
}

/*!
 * \brief Link the modules together, possibly create a constant module.
 *
 * \param params The metadata for initialization of all modules.
 * \param lib the internal module that is compiled by tvm.
 * \param ext_libs The external modules that needs to be imported inside the metadata
 * module(s).
 * \return The created module.
 */
void LinkModules(ffi::ObjectPtr<VMExecutable> exec,
                 const ffi::Map<ffi::String, ffi::Tensor>& params, const tvm::ffi::Module& lib,
                 const ffi::Array<ffi::Module>& ext_libs) {
  // query if we need const loader for ext_modules
  // Wrap all submodules in the initialization wrapper.
  ffi::Map<ffi::String, ffi::Array<ffi::String>> const_vars_by_symbol;
  for (tvm::ffi::Module mod : ext_libs) {
    auto pf_sym = mod->GetFunction("get_symbol");
    auto pf_var = mod->GetFunction("get_const_vars");
    if (pf_sym.has_value() && pf_var.has_value()) {
      ffi::String symbol = (*pf_sym)().cast<ffi::String>();
      ffi::Array<ffi::String> variables = (*pf_var)().cast<ffi::Array<ffi::String>>();
      TVM_FFI_ICHECK(!const_vars_by_symbol.count(symbol)) << "Found duplicated symbol: " << symbol;
      const_vars_by_symbol.Set(symbol, variables);
    }
  }
  if (!const_vars_by_symbol.empty() || !params.empty()) {
    // need runtime const information, run link const loader
    ffi::Module const_loader_mod = runtime::ConstLoaderModuleCreate(params, const_vars_by_symbol);
    const_loader_mod->ImportModule(lib);
    for (const auto& it : ext_libs) {
      const_loader_mod->ImportModule(it);
    }
    exec->ImportModule(const_loader_mod);
  } else {
    // directly import the ext_modules as we don't need const loader
    exec->ImportModule(lib);
    for (const auto& it : ext_libs) {
      exec->ImportModule(it);
    }
  }
}

/*!
 * \brief Link the libraries together.
 */
ffi::Module VMLink(ExecBuilder builder, ffi::Optional<Target> target,
                   ffi::Optional<ffi::Module> lib, ffi::Array<ffi::Module> ext_libs,
                   ffi::Map<ffi::String, ffi::Tensor> params) {
  ffi::ObjectPtr<VMExecutable> executable = builder->Get();
  if (!lib.has_value()) {
    // Grep hint: grep -rn 'runtime.CSourceModuleCreate' src/
    static const auto create_source =
        ffi::Function::GetGlobalRequired("runtime.CSourceModuleCreate");
    lib = create_source(";", "c", ffi::Array<ffi::String>{}, ffi::Array<ffi::String>{})
              .cast<ffi::Module>();
  }
  LinkModules(executable, params, lib.value(), ext_libs);
  return ffi::Module(executable);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.VMLink", VMLink);
}

}  // namespace codegen_vm
}  // namespace relax
}  // namespace tvm
