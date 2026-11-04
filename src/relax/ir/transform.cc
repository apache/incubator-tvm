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
 * \file relax/ir/transform.cc
 * \brief Relax specific transformation passes.
 */
#include <tvm/base.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/rvalue_ref.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/logging.h>

#include <unordered_set>

namespace tvm {
namespace relax {
namespace transform {

TVM_FFI_STATIC_INIT_BLOCK() {
  ::tvm::transform::PassContext::RegisterConfigOption<IntImm>("relax.fallback_device_type");
}

class FunctionPass;

/*!
 * \brief Function-level passes are used to implement various global
 * optimizations for a given Relax IRModule. It fetches one function at a time
 * from the function list in the IRModule for optimization.
 *
 * Note that the scope of passes at this level is a Relax function. Therefore,
 * we cannot add or delete a function through these passes as they are not aware
 * of the global information.
 */
class FunctionPassObj : public tvm::transform::PassObj {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The packed pass function sketches the real optimization. For
   * instance, we can implement a pass that works on a Relax function as a
   * `pass_func` and let it run on a given IRModule. The same `pass_func` will
   * then be applied on each function in the IRModule.
   */
  std::function<Function(Function, IRModule, PassContext)> pass_func;

  explicit FunctionPassObj(PassInfo info) : pass_info(std::move(info)) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FunctionPassObj>().def_ro("pass_info", &FunctionPassObj::pass_info);
  }

  /*!
   * \brief Run a function pass on given pass context.
   *
   * \param mod The IRModule that an optimization pass is applied on.
   * \param pass_ctx The context that an optimization pass executes on.
   *
   * \return Return the updated IRModule.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.FunctionPass", FunctionPassObj, PassObj);

 private:
};

class FunctionPass : public Pass {
 public:
  /*!
   * \brief The constructor
   * \param pass_func The packed function which implements a pass.
   * \param pass_info The pass info.
   */
  TVM_DLL FunctionPass(std::function<Function(Function, IRModule, PassContext)> pass_func,
                       PassInfo pass_info);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FunctionPass, Pass, FunctionPassObj);
};

FunctionPass::FunctionPass(std::function<Function(Function, IRModule, PassContext)> pass_func,
                           PassInfo pass_info)
    : Pass(ffi::UnsafeInit{}) {
  auto n = ffi::make_object<FunctionPassObj>(std::move(pass_info));
  n->pass_func = std::move(pass_func);
  data_ = std::move(n);
}

// Perform IRModule -> IRModule optimizations at the Function level.
IRModule FunctionPassObj::operator()(IRModule mod, const PassContext& pass_ctx) const {
  const PassInfo& pass_info = Info();

  TVM_FFI_ICHECK(mod.defined());

  VLOG_CONTEXT << pass_info->name;
  VLOG(0) << "Executing function pass with opt level: " << pass_info->opt_level;
  VLOG(1) << "Input module:" << std::endl << mod;

  IRModule updated_mod = mod->ShallowCopy();

  std::vector<std::pair<GlobalVar, Function> > updates;
  for (const auto& it : updated_mod->functions) {
    // only picks up relax::Function
    if (auto* n = it.second.as<FunctionObj>()) {
      Function func = ffi::GetRef<Function>(n);
      // Enrich at this leaf executor, rendering the location local to the
      // currently-processed function (the access path is function-rooted).
      Function updated_func = [&]() -> Function {
        try {
          return pass_func(func, updated_mod, pass_ctx);
        } catch (ffi::Error& err) {
          throw tvm::transform::EnrichPassErrorWithContext(err, updated_mod, pass_info->name,
                                                           it.first);
        }
      }();
      updates.push_back({it.first, updated_func});
    }
  }

  for (const auto& pair : updates) {
    updated_mod->Add(pair.first, pair.second, true);
  }

  VLOG(1) << "Output module:" << std::endl << updated_mod;

  return updated_mod;
}

Pass CreateFunctionPass(std::function<Function(Function, IRModule, PassContext)> pass_func,
                        int opt_level, ffi::String name) {
  PassInfo pass_info = PassInfo(opt_level, name);
  return FunctionPass(std::move(pass_func), pass_info);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.transform.MakeFunctionPass",
      [](ffi::TypedFunction<Function(ffi::RValueRef<Function>, IRModule, PassContext)> pass_func,
         PassInfo pass_info) {
        auto wrapped_pass_func = [pass_func](Function func, IRModule mod, PassContext ctx) {
          return pass_func(ffi::RValueRef<Function>(std::move(func)), mod, ctx);
        };
        return FunctionPass(wrapped_pass_func, pass_info);
      });
}

// Pattern A (RM): auto-default repr from reflection for FunctionPassObj.

class DataflowBlockPass;

/*!
 * \brief DataflowBlock-level passes are used to implement various dataflow block
 * optimizations for a given Relax IRModule. It fetches one dataflow block at a time
 * from the functions in an IRModule, and yields a rewritten DataflowBlock.
 *
 * Note that the scope of passes at this level is a Relax DataflowBlock. Therefore,
 * we cannot modify the global scope Vars and symbolic shape Vars defined inside the dataflow block.
 */
class DataflowBlockPassObj : public tvm::transform::PassObj {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The packed pass function sketches the real optimization. For
   * instance, we can implement a pass that works on a Relax DataflowBlock as a
   * `pass_func` and let it run on a given IRModule. The same `pass_func` will
   * then be applied on each DataflowBlock in the IRModule.
   */
  std::function<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func;

  explicit DataflowBlockPassObj(PassInfo info) : pass_info(std::move(info)) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DataflowBlockPassObj>().def_ro("pass_info", &DataflowBlockPassObj::pass_info);
  }

  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  PassInfo Info() const override { return pass_info; }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.DataflowBlockPass", DataflowBlockPassObj, PassObj);
};

/*! \brief Helper to apply the passed function to dataflow blocks.*/
class DataflowBlockMutator : public ExprMutator {
 public:
  DataflowBlockMutator(std::function<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func,
                       IRModule mod, PassContext pass_ctx)
      : pass_func_(pass_func), mod_(mod), pass_ctx_(pass_ctx) {}

  /*!
   * \brief Rewrite the DataflowBlockObj with pass_func_
   *
   * This function will check that there are no rewrites of the global-scope
   * Vars or symbolic Vars defined inside the dataflow block.
   */
  BindingBlock VisitBindingBlock_(const DataflowBlockObj* n) final {
    PreservedVars preserved_vars = CollectPreservedVars(n->bindings);

    // apply pass_func_ to the DataflowBlock
    DataflowBlock block = ffi::GetRef<DataflowBlock>(n);
    DataflowBlock updated_block = pass_func_(block, mod_, pass_ctx_);

    PreservedVars updated_vars = CollectPreservedVars(updated_block->bindings);
    for (const VarObj* var : preserved_vars.binding_vars) {
      TVM_FFI_ICHECK(updated_vars.binding_vars.count(var))
          << "Error: DataflowBlock Pass should not rewrite or delete any global-scope Var.";
    }
    for (const VarObj* var : preserved_vars.match_cast_symbolic_vars) {
      TVM_FFI_ICHECK(updated_vars.match_cast_symbolic_vars.count(var))
          << "Error: DataflowBlock Pass should not rewrite or delete any symbolic Var declared "
             "by a MatchCast.";
    }

    return updated_block;
  }

 private:
  using VarSet = std::unordered_set<const VarObj*>;

  struct PreservedVars {
    VarSet binding_vars;
    VarSet match_cast_symbolic_vars;
  };

  static PreservedVars CollectPreservedVars(const ffi::Array<Binding>& bindings) {
    PreservedVars vars;
    for (const Binding& binding : bindings) {
      if (const auto* match_cast = binding.as<MatchCastObj>()) {
        for (const tvm::Var& var : DefinableTIRVarsInType(match_cast->ty)) {
          vars.match_cast_symbolic_vars.insert(var.get());
        }
      }
      if (!binding->var.as<DataflowVarObj>()) {
        vars.binding_vars.insert(binding->var.get());
      }
    }
    return vars;
  }

  std::function<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func_;
  IRModule mod_;
  PassContext pass_ctx_;
};

class DataflowBlockPass : public Pass {
 public:
  /*!
   * \brief The constructor
   * \param pass_func The packed function which implements a pass.
   * \param pass_info The pass info.
   */
  TVM_DLL DataflowBlockPass(
      std::function<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func,
      PassInfo pass_info);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DataflowBlockPass, Pass, DataflowBlockPassObj);
};

DataflowBlockPass::DataflowBlockPass(
    std::function<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func,
    PassInfo pass_info)
    : Pass(ffi::UnsafeInit{}) {
  auto n = ffi::make_object<DataflowBlockPassObj>(std::move(pass_info));
  n->pass_func = std::move(pass_func);
  data_ = std::move(n);
}

// Perform IRModule -> IRModule transformations at the DataflowBlock level.
IRModule DataflowBlockPassObj::operator()(IRModule mod, const PassContext& pass_ctx) const {
  const PassInfo& pass_info = Info();

  TVM_FFI_ICHECK(mod.defined());

  VLOG_CONTEXT << pass_info->name;
  VLOG(0) << "Executing DataflowBlock pass with opt level: " << pass_info->opt_level;
  VLOG(1) << "Input module:" << std::endl << mod;

  IRModule updated_mod = mod->ShallowCopy();

  DataflowBlockMutator dataflow_block_mutator(pass_func, updated_mod, pass_ctx);
  std::vector<std::pair<GlobalVar, Function> > updates;
  for (const auto& it : updated_mod->functions) {
    // only picks up relax::Function
    if (auto* n = it.second.as<FunctionObj>()) {
      Function func = ffi::GetRef<Function>(n);
      // Enrich at this leaf executor, rendering the location local to the
      // currently-processed function.
      Function updated_func = [&]() -> Function {
        try {
          return dataflow_block_mutator.VisitExpr(func).as_or_throw<Function>();
        } catch (ffi::Error& err) {
          throw tvm::transform::EnrichPassErrorWithContext(err, updated_mod, pass_info->name,
                                                           it.first);
        }
      }();
      updates.push_back({it.first, updated_func});
    }
  }

  for (const auto& pair : updates) {
    updated_mod->Add(pair.first, pair.second, true);
  }

  VLOG(1) << "Output module:" << std::endl << updated_mod;

  return updated_mod;
}

Pass CreateDataflowBlockPass(
    std::function<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func, int opt_level,
    ffi::String name) {
  PassInfo pass_info = PassInfo(opt_level, name);
  return DataflowBlockPass(std::move(pass_func), pass_info);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.transform.MakeDataflowBlockPass",
      [](ffi::TypedFunction<DataflowBlock(ffi::RValueRef<DataflowBlock>, IRModule, PassContext)>
             pass_func,
         PassInfo pass_info) {
        auto wrapped_pass_func = [pass_func](DataflowBlock func, IRModule mod, PassContext ctx) {
          return pass_func(ffi::RValueRef<DataflowBlock>(std::move(func)), mod, ctx);
        };
        return DataflowBlockPass(wrapped_pass_func, pass_info);
      });
}

// Pattern A (RM): auto-default repr from reflection for DataflowBlockPassObj.

TVM_FFI_STATIC_INIT_BLOCK() {
  FunctionPassObj::RegisterReflection();
  DataflowBlockPassObj::RegisterReflection();
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
