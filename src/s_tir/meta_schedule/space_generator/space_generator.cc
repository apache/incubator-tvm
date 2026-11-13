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
#include <tvm/runtime/logging.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

ffi::String GetRuleKindFromTarget(const Target& target) {
  if (target->kind->name == "llvm") {
    static auto target_has_feature_fn_ptr =
        tvm::ffi::Function::GetGlobalRequired("codegen.target_has_feature");
    bool have_avx512vnni = target_has_feature_fn_ptr("avx512vnni", target).cast<bool>();
    bool have_avxvnni = target_has_feature_fn_ptr("avxvnni", target).cast<bool>();
    if (have_avx512vnni || have_avxvnni) {
      return "vnni";
    } else {
      bool have_avx512f = target_has_feature_fn_ptr("avx512f", target).cast<bool>();
      bool have_avx512bw = target_has_feature_fn_ptr("avx512bw", target).cast<bool>();
      if (have_avx512bw && have_avx512f) {
        return "avx512";
      }
    }
    bool have_rvv = target_has_feature_fn_ptr("v", target).cast<bool>();
    if (have_rvv) {
      return "rvv";
    }

    if (target_has_feature_fn_ptr("dotprod", target).cast<bool>()) {
      return "dotprod";
    }
    if (target_has_feature_fn_ptr("neon", target).cast<bool>()) {
      return "asimd";
    }
    return "llvm";
  }
  if (target->kind->name == "hexagon") {
    return "hexagon";
  }
  if (IsGPUTarget(target->kind->name)) {
    return "cuda";
  }

  if (target->kind->name == "c") {
    return "c";
  }
  TVM_FFI_THROW(InternalError) << "Unsupported target: " << target;
  throw;
}

void SpaceGeneratorObj::InitializeWithTuneContext(const TuneContext& context) {
  if (context->target.has_value() &&  //
      !(sch_rules.has_value() &&      //
        postprocs.has_value() &&      //
        mutator_probs.has_value())) {
    ffi::String kind = GetRuleKindFromTarget(context->target.value());
    ffi::Array<ScheduleRule> default_sch_rules;
    ffi::Array<Postproc> default_postprocs;
    ffi::Map<Mutator, FloatImm> default_mutator_probs;
    // for target with skylake-avx512
    if (kind == "llvm") {
      default_sch_rules = ScheduleRule::DefaultLLVM();
      default_postprocs = Postproc::DefaultLLVM();
      default_mutator_probs = Mutator::DefaultLLVM();
    } else if (kind == "cuda") {
      default_sch_rules = ScheduleRule::DefaultCUDA();
      default_postprocs = Postproc::DefaultCUDA();
      default_mutator_probs = Mutator::DefaultCUDA();
    } else if (kind == "hexagon") {
      default_sch_rules = ScheduleRule::DefaultHexagon();
      default_postprocs = Postproc::DefaultHexagon();
      default_mutator_probs = Mutator::DefaultHexagon();
    } else if (kind == "vnni") {
      default_sch_rules = ScheduleRule::DefaultX86("vnni");
      default_postprocs = Postproc::DefaultCPUTensorization();
      default_mutator_probs = Mutator::DefaultLLVM();
    } else if (kind == "avx512") {
      default_sch_rules = ScheduleRule::DefaultX86("avx512");
      default_postprocs = Postproc::DefaultCPUTensorization();
      default_mutator_probs = Mutator::DefaultLLVM();
    } else if (kind == "rvv") {
      static auto llvm_get_vector_width =
          tvm::ffi::Function::GetGlobalRequired("codegen.llvm_get_vector_width");
      const int vlen = llvm_get_vector_width(context->target.value()).cast<int>();
      default_sch_rules = ScheduleRule::DefaultRISCV(context->target.value(), vlen);
      default_postprocs = Postproc::DefaultRISCV();
      default_mutator_probs = Mutator::DefaultLLVM();
    } else if (kind == "asimd") {
      default_sch_rules = ScheduleRule::DefaultARM("neon");
      default_postprocs = Postproc::DefaultCPUTensorization();
      default_mutator_probs = Mutator::DefaultLLVM();
    } else if (kind == "dotprod") {
      default_sch_rules = ScheduleRule::DefaultARM("dotprod");
      default_postprocs = Postproc::DefaultCPUTensorization();
      default_mutator_probs = Mutator::DefaultLLVM();
    } else {
      TVM_FFI_THROW(InternalError) << "Unsupported kind: " << kind;
      throw;
    }
    if (!sch_rules.has_value()) {
      sch_rules = default_sch_rules;
    }
    if (!postprocs.has_value()) {
      postprocs = default_postprocs;
    }
    if (!mutator_probs.has_value()) {
      mutator_probs = default_mutator_probs;
    }
  }
  if (sch_rules.has_value()) {
    for (ScheduleRule i : sch_rules.value()) {
      i->InitializeWithTuneContext(context);
    }
  }
  if (postprocs.has_value()) {
    for (Postproc i : postprocs.value()) {
      i->InitializeWithTuneContext(context);
    }
  }
  if (mutator_probs.has_value()) {
    for (const auto& kv : mutator_probs.value()) {
      Mutator mutator = kv.first;
      mutator->InitializeWithTuneContext(context);
    }
  }
}

void PySpaceGeneratorObj::InitializeWithTuneContext(const TuneContext& context) {
  TVM_FFI_ICHECK(f_initialize_with_tune_context != nullptr)
      << "PySpaceGenerator's InitializeWithTuneContext method not implemented!";
  f_initialize_with_tune_context(context);
}

ffi::Array<s_tir::Schedule> PySpaceGeneratorObj::GenerateDesignSpace(const IRModule& mod) {
  TVM_FFI_ICHECK(f_generate_design_space != nullptr)
      << "PySpaceGenerator's GenerateDesignSpace method not implemented!";
  return f_generate_design_space(mod);
}

SpaceGenerator PySpaceGeneratorObj::Clone() const {
  TVM_FFI_ICHECK(f_clone != nullptr) << "PySpaceGenerator's Clone method not implemented!";
  return f_clone();
}

SpaceGenerator SpaceGenerator::PySpaceGenerator(
    ffi::Optional<ffi::Array<ScheduleRule>> sch_rules,
    ffi::Optional<ffi::Array<Postproc>> postprocs,
    ffi::Optional<ffi::Map<Mutator, FloatImm>> mutator_probs,
    FInitializeWithTuneContext f_initialize_with_tune_context,
    FGenerateDesignSpace f_generate_design_space, FClone f_clone) {
  ffi::ObjectPtr<PySpaceGeneratorObj> n = ffi::make_object<PySpaceGeneratorObj>();
  n->sch_rules = sch_rules;
  n->postprocs = postprocs;
  n->mutator_probs = mutator_probs;
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_generate_design_space = std::move(f_generate_design_space);
  n->f_clone = std::move(f_clone);
  return SpaceGenerator(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  SpaceGeneratorObj::RegisterReflection();
  PySpaceGeneratorObj::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.meta_schedule.SpaceGeneratorInitializeWithTuneContext",
                  &SpaceGeneratorObj::InitializeWithTuneContext)
      .def_method("s_tir.meta_schedule.SpaceGeneratorGenerateDesignSpace",
                  &SpaceGeneratorObj::GenerateDesignSpace)
      .def("s_tir.meta_schedule.SpaceGeneratorPySpaceGenerator", SpaceGenerator::PySpaceGenerator)
      .def_method("s_tir.meta_schedule.SpaceGeneratorClone", &SpaceGeneratorObj::Clone);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
