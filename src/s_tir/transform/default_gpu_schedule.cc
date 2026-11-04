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
#include <tvm/ir/function.h>
#include <tvm/s_tir/function.h>
#include <tvm/s_tir/stmt.h>

#include "../meta_schedule/utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;
namespace transform {
/*!
 * \brief A helper function to do default thread binding for a block.
 * \param sch The schedule to work on.
 * \param block The block to be scheduled.
 * \param max_thread_per_block The maximum number of threads per block.
 * \param max_threadblocks The maximum number of threadblocks.
 */
void ThreadBind(s_tir::Schedule sch, const s_tir::SBlockRV& block, int64_t max_thread_per_block,
                int64_t max_threadblocks = 256) {
  // fetch the loops
  ffi::Array<s_tir::LoopRV> loops = sch->GetLoops(block);
  for (const s_tir::LoopRV& loop : loops) {
    // skip block if already scheduled
    if (tvm::s_tir::GetThreadBinding(sch->Get(loop)).has_value()) {
      return;
    }
  }
  ffi::Array<s_tir::IterVar> iters = sch->Get(block)->iter_vars;

  // when there is no loops, tirx will add a dummy iter var for the block
  // so loops.size() == 0 && iters.size() == 1
  TVM_FFI_ICHECK(loops.size() == iters.size() || (loops.size() == 0 && iters.size() == 1));

  ffi::Array<s_tir::LoopRV> data_parallel_loops;
  // only fuse data parallel loops
  for (size_t i = 0; i < loops.size(); ++i) {
    if (iters[i]->iter_type == s_tir::IterVarType::kDataPar) {
      data_parallel_loops.push_back(loops[i]);
    }
  }

  // Add a dummy loop if there is no data parallel loops
  if (data_parallel_loops.size() == 0) {
    data_parallel_loops.push_back(loops.empty() ? sch->AddUnitLoop(block)
                                                : sch->AddUnitLoop(loops[0]));
  }
  // fuse all data parallel loops
  s_tir::LoopRV fused = sch->Fuse(data_parallel_loops, /*preserve_unit_iters=*/false);
  int64_t product = std::numeric_limits<int64_t>::max();
  if (Range::GetExtentFromBeginEnd(sch->Get(fused)->begin, sch->Get(fused)->end)
          ->IsInstance<IntImmObj>()) {
    product = static_cast<int64_t>(
        Range::GetExtentFromBeginEnd(sch->Get(fused)->begin, sch->Get(fused)->end)
            .as<IntImmObj>()
            ->value);
  }
  // schedule the fused loop
  if (product > max_thread_per_block * max_threadblocks) {
    ffi::Array<s_tir::LoopRV> splits =
        sch->Split(fused,
                   /*factors=*/{std::nullopt, IntImm::Int32(max_threadblocks),
                                IntImm::Int32(max_thread_per_block)});
    sch->Reorder(/*ordered_loop_rvs=*/{splits[1], splits[2], splits[0]});
    sch->Bind(splits[1], "blockIdx.x");
    sch->Bind(splits[2], "threadIdx.x");
  } else {
    ffi::Array<s_tir::LoopRV> splits = sch->Split(
        fused, /*factors=*/{std::nullopt, IntImm::Int32(std::min(product, max_thread_per_block))});
    sch->Bind(splits[0], "blockIdx.x");
    sch->Bind(splits[1], "threadIdx.x");
  }
}

IRModule MarkScheduled(const IRModule& mod) {
  ffi::Map<GlobalVar, BaseFunc> result;

  for (const auto& [gv, base_func] : mod->functions) {
    if (const auto* function_node = base_func.as<tirx::FunctionObj>()) {
      tirx::Function function = ffi::GetRef<tirx::Function>(function_node);
      tirx::Function new_function =
          WithAttr(std::move(function), tvm::s_tir::attr::kIsScheduled.name, true);
      result.Set(gv, new_function);
    } else {
      result.Set(gv, base_func);
    }
  }

  return IRModule(result,              // functions
                  mod->attrs,          // attrs
                  mod->global_infos);  // global_infos
}

/*!
 * \brief Wrap a Function body that is a bare \c SBlockRealize (no enclosing
 * loops, no iter vars) so the realized block is no longer the function's root
 * sref.
 *
 * Without this, \c ThreadBind below calls \c Schedule::AddUnitLoop(block) on
 * a block that is itself the function's root sref, hitting the
 * "Cannot add loops on top of the root block" check in
 * \c s_tir::AddUnitLoop. The schedule infrastructure additionally requires
 * the function body to be an \c SBlockRealize, so we keep that shape and
 * push the original block one level deeper, inside a wrapping root block
 * that holds a unit serial loop. The synthesised data-parallel iter keeps
 * iter_values/iter_vars counts consistent for downstream checks.
 */
tirx::Function WrapBareSBlockBody(const tirx::Function& func) {
  const auto* realize = func->body.has_value() && func->body.value()->size() == 1
                            ? func->body.value()->seq[0].as<s_tir::SBlockRealizeObj>()
                            : nullptr;
  if (realize == nullptr || !realize->block->iter_vars.empty()) {
    return func;
  }
  // Only wrap when the block is a leaf computation. A well-formed Function
  // produced by the rest of the pipeline has an implicit root SBlockRealize
  // whose block body is a For loop (or a nested SBlockRealize) — that case
  // already has somewhere to put thread bindings, so leave it alone.
  const SeqStmt& inner = realize->block->body;
  if (inner->size() == 1 && (inner->seq[0]->IsInstance<ForObj>() ||
                             inner->seq[0]->IsInstance<s_tir::SBlockRealizeObj>())) {
    return func;
  }
  tvm::IntImm zero(tvm::PrimType::Int(32), 0);
  tvm::IntImm one(tvm::PrimType::Int(32), 1);
  tvm::Var loop_var("u", tvm::PrimType::Int(32));
  tvm::Var iter_var_var("vu", tvm::PrimType::Int(32));
  s_tir::IterVar new_iter(tvm::Range::FromMinExtent(zero, one), iter_var_var.as_or_throw<PrimVar>(),
                          s_tir::IterVarType::kDataPar);
  s_tir::SBlock inner_block = realize->block;
  inner_block.CopyOnWrite()->iter_vars = ffi::Array<s_tir::IterVar>{new_iter};
  s_tir::SBlockRealize inner_realize(
      /*iter_values=*/ffi::Array<tvm::PrimExpr>{loop_var.as_or_throw<tvm::PrimExpr>()},
      /*predicate=*/realize->predicate, inner_block);
  Stmt for_stmt =
      For(loop_var.as_or_throw<PrimVar>(), zero, one, one, ForKind::kDefault, inner_realize);
  s_tir::SBlock root_block(/*iter_vars=*/ffi::Array<s_tir::IterVar>{},
                           /*reads=*/ffi::Array<tvm::TensorRegion>{},
                           /*writes=*/ffi::Array<tvm::TensorRegion>{},
                           /*name_hint=*/"root", /*body=*/for_stmt);
  s_tir::SBlockRealize root_realize(/*iter_values=*/ffi::Array<tvm::PrimExpr>{},
                                    /*predicate=*/IntImm::Bool(true), root_block);
  tirx::Function result = func;
  result.CopyOnWrite()->body = std::move(root_realize);
  return result;
}

bool IsScheduledOnGPU(const BaseFunc& func) {
  // the target from context.
  ffi::Optional<tvm::Target> target = tvm::Target::Current();
  // the Target in kTarget attribute of Function
  ffi::Optional<tvm::Target> func_target =
      func->attrs.GetAttr<tvm::Target>(tvm::attr::kTarget.name);
  if (func_target.has_value()) {
    target = func_target.value();
  }

  if (target.has_value()) {
    int dev_type = target.value()->GetTargetDeviceType();
    if (!(dev_type == kDLCUDA || dev_type == kDLMetal || dev_type == kDLROCM ||
          dev_type == kDLVulkan || dev_type == kDLOpenCL || dev_type == kDLWebGPU)) {
      return false;
    }
  }
  return true;
}

Pass DefaultGPUSchedule() {
  auto pass_func =  //
      [=](IRModule m, PassContext pc) {
        // Wrap any GPU-bound Function whose body is a bare SBlockRealize
        // (e.g. a scalar op) so ThreadBind below has a loop to operate on.
        ffi::Map<GlobalVar, BaseFunc> wrapped;
        bool any_wrapped = false;
        for (const auto& [gv, base_func] : m->functions) {
          if (const auto* function_node = base_func.as<tirx::FunctionObj>();
              function_node != nullptr && IsScheduledOnGPU(base_func) &&
              !base_func->HasNonzeroAttr(tvm::s_tir::attr::kIsScheduled.name)) {
            tirx::Function func = ffi::GetRef<tirx::Function>(function_node);
            tirx::Function new_func = WrapBareSBlockBody(func);
            if (!new_func.same_as(func)) {
              wrapped.Set(gv, new_func);
              any_wrapped = true;
              continue;
            }
          }
          wrapped.Set(gv, base_func);
        }
        if (any_wrapped) {
          m = IRModule(wrapped, m->attrs, m->global_infos);
        }
        s_tir::Schedule sch = s_tir::Schedule::Traced(m, /*seed=*/-1, /*debug_mask=*/0,
                                                      s_tir::ScheduleErrorRenderLevel::kDetail);
        for (const auto& [gv, func] : m->functions) {
          if (func->IsInstance<tirx::FunctionObj>() &&
              !func->HasNonzeroAttr(tvm::s_tir::attr::kIsScheduled.name) &&
              IsScheduledOnGPU(func)) {
            // get the target from context.
            ffi::Optional<tvm::Target> target = tvm::Target::Current();
            // get the target from kTarget attribute
            ffi::Optional<tvm::Target> func_target =
                func->attrs.GetAttr<tvm::Target>(tvm::attr::kTarget.name);
            if (func_target.has_value()) {
              target = func_target.value();
            }
            TVM_FFI_ICHECK(target.has_value())
                << "The target is missing either in the current context or in "
                   "the function's attribute.";
            // get the max thread per block from target.
            ffi::Optional<int64_t> opt_max_thread_per_block =
                target.value()->GetAttr<int64_t>("max_num_threads");
            TVM_FFI_ICHECK(opt_max_thread_per_block.has_value())
                << "max_num_threads is not set for target " << target;
            int64_t max_thread_per_block = opt_max_thread_per_block.value();

            sch->WorkOn(gv->name_hint);
            ffi::Array<s_tir::SBlockRV> blocks =
                s_tir::meta_schedule::SBlockCollector::Collect(sch);
            for (const s_tir::SBlockRV& block : blocks) {
              auto childs = sch->GetChildBlocks(block);
              if (!childs.empty()) {
                continue;
              }
              ThreadBind(sch, block, max_thread_per_block);
            }
          }
        }
        return MarkScheduled(sch->mod());
      };
  return tvm::transform::CreateModulePass(/*pass_function=*/pass_func,  //
                                          /*opt_level=*/0,              //
                                          /*pass_name=*/"DefaultGPUSchedule");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.DefaultGPUSchedule", DefaultGPUSchedule);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
