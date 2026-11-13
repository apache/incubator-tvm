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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/s_tir/meta_schedule/extracted_task.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>

#include "../../s_tir/meta_schedule/module_equality.h"

namespace tvm {
namespace relax {
namespace backend {

using s_tir::meta_schedule::ExtractedTask;
using s_tir::meta_schedule::ModuleEqual;
using s_tir::meta_schedule::ModuleEquality;
using s_tir::meta_schedule::ModuleHash;

/*!
 * \brief Extract the Meta-Schedule tuning task from a given IRModule.
 * \note
 *   1. The task extractor is responsible for task deduplication. The
 *   deduplication is achieved by comparing structural hashes of Functions.
 *   2. For a tirx::Function, the weight of its corresponding task is the number
 *   of times it called by op Call-TIR. Say in an IRModule there are three
 *   Functions `fn1`, `fn2` and `fn3` sharing the same structural hash.
 *   Suppose `fn1` is called by 5 Call-TIR ops among all Relax function,
 *   `fn2` is called by 3 Call-TIR and `fn3` is called by 5 Call-TIR.
 *   Then we will have a ExtractedTask for all three functions, whose weight
 *   is 5 + 3 + 2 = 10.
 */
class BlockCounter : public s_tir::StmtExprVisitor {
 public:
  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<tvm::ExprObj>()) return std::nullopt;
    return s_tir::StmtExprVisitor::Visit(value);
  }

  static size_t GetSBlockCount(const tirx::Function& func) {
    auto counter = ffi::make_object<BlockCounter>();
    counter->Visit(func->body);
    return counter->count;
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const s_tir::SBlockObj* op) final {
    ++count;
    return s_tir::StmtExprVisitor::Visit_(op);
  }
  size_t count{0};
};

class TaskExtractor : public ExprVisitor {
 public:
  static ffi::Array<ExtractedTask> ExtractTask(IRModule mod, Target target,
                                               ffi::String mod_eq_name) {
    TaskExtractor extractor(mod, target, mod_eq_name);
    // We go through each Relax function in the module.
    for (const auto& kv : mod->functions) {
      if (const auto* func = kv.second.as<FunctionObj>()) {
        extractor(ffi::GetRef<Function>(func));
      }
    }
    ffi::Array<ExtractedTask> tasks;
    for (const auto& it : extractor.func2task_) {
      tasks.push_back(it.second);
    }
    return tasks;
  }

 private:
  explicit TaskExtractor(IRModule mod, Target target, ffi::String mod_eq_name)
      : mod_(std::move(mod)),
        target_(std::move(target)),
        mod_eq_(ModuleEquality::Create(mod_eq_name)),
        func2task_(/*bucket_count*/ 0, ModuleHash(*mod_eq_), ModuleEqual(*mod_eq_)) {
    normalize_mod_func_ = tvm::ffi::Function::GetGlobal("tvm.s_tir.meta_schedule.normalize_mod");
    TVM_FFI_ICHECK(normalize_mod_func_.has_value()) << "Normalization function is not found.";
  }

  void VisitExpr_(const CallObj* call) final {
    static const Op call_tir_op = Op::Get("relax.call_tir");

    // TODO(@tvm-team): When we differentiate the call for tirx function and packed function,
    // this logic should be changed accordingly.
    if (!call->op.same_as(call_tir_op)) {
      // Since the Relax function is of A-normal form, the arguments of this call cannot be another
      // Calls. And hence we do not need to recurse into this Call.
      return;
    }

    const GlobalVar& global_var = call->args[0].as_or_throw<GlobalVar>();
    const tirx::Function& func = mod_->Lookup(global_var).as_or_throw<tirx::Function>();
    IRModule mod = (*normalize_mod_func_)(func).cast<IRModule>();
    size_t weight = 1;
    auto it = func2task_.find(mod);
    if (it != func2task_.end()) {
      it->second->weight += 1;
      const tirx::Function& alt_func = it->first->Lookup("main").as_or_throw<tirx::Function>();
      // When anchor-block based equality is used, tuning tasks "nn_conv2d_add_nn_relu" and
      // "nn_conv2d_add_add_nn_relu", for example, can be identified as equal. Thus, one of them
      // will be selected to tune by the code below.
      //
      // To make sure that we tune "nn_conv2d_add_nn_relu" and not "nn_conv2d_add_add_nn_relu", we
      // count the PrinFunc number of blocks and leave only the function with the smallest number of
      // blocks. This way, "nn_conv2d_add_nn_relu" will have a smaller number of blocks than
      // "nn_conv2d_add_add_nn_relu" and will be selected to tune.
      if (BlockCounter::GetSBlockCount(func) < BlockCounter::GetSBlockCount(alt_func)) {
        weight += it->second->weight;
        func2task_.erase(it->first);
      }
    }

    ExtractedTask task(/*task_name=*/global_var->name_hint,  //
                       /*mod=*/mod,                          //
                       /*target=*/target_,                   //
                       /*dispatched=*/{mod},                 //
                       /*weight=*/weight);
    func2task_.emplace(mod, task);
  }

  IRModule mod_;
  Target target_;
  std::unique_ptr<ModuleEquality> mod_eq_;
  std::unordered_map<IRModule, ExtractedTask, ModuleHash, ModuleEqual> func2task_;
  std::optional<tvm::ffi::Function> normalize_mod_func_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.backend.MetaScheduleExtractTask", [](IRModule mod, Target target,
                                                                    ffi::String mod_eq_name) {
    return TaskExtractor::ExtractTask(std::move(mod), std::move(target), std::move(mod_eq_name));
  });
}

}  // namespace backend
}  // namespace relax
}  // namespace tvm
