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
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/tensor_intrin.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

void PyScheduleRuleObj::InitializeWithTuneContext(const TuneContext& context) {
  TVM_FFI_ICHECK(f_initialize_with_tune_context != nullptr)
      << "PyScheduleRule's InitializeWithTuneContext method not implemented!";
  f_initialize_with_tune_context(context);
}

ffi::Array<s_tir::Schedule> PyScheduleRuleObj::Apply(const s_tir::Schedule& sch,
                                                     const s_tir::SBlockRV& block) {
  TVM_FFI_ICHECK(f_apply != nullptr) << "PyScheduleRule's Apply method not implemented!";
  return f_apply(sch, block);
}

ScheduleRule PyScheduleRuleObj::Clone() const {
  TVM_FFI_ICHECK(f_clone != nullptr) << "PyScheduleRule's Clone method not implemented!";
  return f_clone();
}

ScheduleRule ScheduleRule::PyScheduleRule(
    PyScheduleRuleObj::FInitializeWithTuneContext f_initialize_with_tune_context,  //
    PyScheduleRuleObj::FApply f_apply,                                             //
    PyScheduleRuleObj::FClone f_clone) {
  ffi::ObjectPtr<PyScheduleRuleObj> n = ffi::make_object<PyScheduleRuleObj>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_apply = std::move(f_apply);
  n->f_clone = std::move(f_clone);
  return ScheduleRule(ffi::UnsafeInit{}, n);
}

ffi::Array<ScheduleRule> ScheduleRule::DefaultLLVM() {
  return {
      ScheduleRule::ApplyCustomRule(),
      ScheduleRule::InlineConstantScalars(),
      ScheduleRule::AutoInline(
          /*into_producer=*/false,
          /*into_consumer=*/true,
          /*inline_const_tensor=*/true,
          /*disallow_if_then_else=*/true,
          /*require_injective=*/true,
          /*require_ordered=*/true,
          /*disallow_op=*/ffi::Array<ffi::String>{"prim.exp"}),
      ScheduleRule::AddRFactor(
          /*max_jobs_per_core=*/16,
          /*max_innermost_factor=*/static_cast<int64_t>(64)),
      ScheduleRule::MultiLevelTiling(
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/static_cast<int64_t>(64),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<int64_t>{1, 2}},
                                          {"scope", ffi::String("global")}}),
      ScheduleRule::ParallelizeVectorizeUnroll(
          /*max_jobs_per_core=*/16,
          /*max_vectorize_extent=*/64,
          /*unroll_max_steps=*/ffi::Array<int64_t>{0, 16, 64, 512},
          /*unroll_explicit=*/true),
      ScheduleRule::RandomComputeLocation(),
  };
}

ffi::Array<ScheduleRule> ScheduleRule::DefaultX86(const ffi::String& type) {
  static const ffi::Map<ffi::String, ffi::String> intrins = {{"vnni", "dot_16x4_vnni"},
                                                             {"avx512", "dot_16x4_avx512"}};
  return {
      ScheduleRule::ApplyCustomRule(),
      ScheduleRule::InlineConstantScalars(),
      ScheduleRule::AutoInline(
          /*into_producer=*/false,
          /*into_consumer=*/true,
          /*inline_const_tensor=*/true,
          /*disallow_if_then_else=*/true,
          /*require_injective=*/true,
          /*require_ordered=*/true,
          /*disallow_op=*/ffi::Array<ffi::String>{"prim.exp"}),
      ScheduleRule::AddRFactor(
          /*max_jobs_per_core=*/16,
          /*max_innermost_factor=*/static_cast<int64_t>(64)),
      ScheduleRule::MultiLevelTilingWithIntrin(
          /*intrin_name=*/intrins[type],
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/static_cast<int64_t>(64),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<int64_t>{1, 2}},
                                          {"scope", ffi::String("global")}}),
      ScheduleRule::MultiLevelTiling(
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/static_cast<int64_t>(64),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<int64_t>{1, 2}},
                                          {"scope", ffi::String("global")}}),
      ScheduleRule::ParallelizeVectorizeUnroll(
          /*max_jobs_per_core=*/16,
          /*max_vectorize_extent=*/64,
          /*unroll_max_steps=*/ffi::Array<int64_t>{0, 16, 64, 512},
          /*unroll_explicit=*/true),
      ScheduleRule::RandomComputeLocation(),
  };
}

ffi::Array<ScheduleRule> ScheduleRule::DefaultCUDA() {
  return {
      ScheduleRule::ApplyCustomRule(),
      ScheduleRule::MultiLevelTiling(
          /*structure=*/"SSSRRSRS",
          /*tile_binds=*/ffi::Array<ffi::String>{"blockIdx.x", "vthread.x", "threadIdx.x"},
          /*max_innermost_factor=*/static_cast<int64_t>(64),
          /*vector_load_lens=*/ffi::Array<int64_t>{1, 2, 3, 4, 8, 16},
          /*reuse_read=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("must")},
                                          {"levels", ffi::Array<int64_t>{4}},  //
                                          {"scope", ffi::String("shared")}},
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("must")},
                                          {"levels", ffi::Array<int64_t>{3}},  //
                                          {"scope", ffi::String("local")}}),
      ScheduleRule::InlineConstantScalars(),
      ScheduleRule::AutoInline(
          /*into_producer=*/true,
          /*into_consumer=*/true,
          /*inline_const_tensor=*/true,
          /*disallow_if_then_else=*/false,
          /*require_injective=*/false,
          /*require_ordered=*/false,
          /*disallow_op=*/ffi::Array<ffi::String>{}),
      ScheduleRule::CrossThreadReduction(
          /*thread_extents=*/ffi::Array<int64_t>{4, 8, 16, 32, 64, 128, 256, 512}),
      ScheduleRule::ParallelizeVectorizeUnroll(
          /*max_jobs_per_core=*/-1,
          /*max_vectorize_extent=*/-1,
          /*unroll_max_steps=*/ffi::Array<int64_t>{0, 16, 32, 64, 128, 256, 512, 1024},
          /*unroll_explicit=*/true),
      ScheduleRule::AutoBind(
          /*max_threadblocks=*/256,
          /*thread_extents*/ ffi::Array<int64_t>{32, 64, 128, 256, 512, 1024}),
  };
}

ffi::Array<ScheduleRule> ScheduleRule::DefaultHexagon() {
  return {
      ScheduleRule::ApplyCustomRule(),
      ScheduleRule::InlineConstantScalars(),
      ScheduleRule::AutoInline(
          /*into_producer=*/false,
          /*into_consumer=*/true,
          /*inline_const_tensor=*/true,
          /*disallow_if_then_else=*/true,
          /*require_injective=*/true,
          /*require_ordered=*/true,
          /*disallow_op=*/ffi::Array<ffi::String>{"prim.exp"}),
      ScheduleRule::MultiLevelTilingWideVector(
          /*structure=*/"SRSRS",
          /*vector_length_in_bits=*/1024,
          /*max_innermost_factor=*/static_cast<int64_t>(128),
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<int64_t>{1, 2}},
                                          {"scope", ffi::String("global")}}),
      ScheduleRule::ParallelizeVectorizeUnroll(
          /*max_jobs_per_core=*/16,
          /*max_vectorize_extent=*/128,
          /*unroll_max_steps=*/ffi::Array<int64_t>{0, 16, 64, 512},
          /*unroll_explicit=*/true),
  };
}

ffi::Array<ScheduleRule> ScheduleRule::DefaultRISCV(Target target, const int vlen) {
  ffi::Array<ScheduleRule> rules;
  rules.push_back(ScheduleRule::ApplyCustomRule());
  rules.push_back(ScheduleRule::InlineConstantScalars());
  rules.push_back(ScheduleRule::AutoInline(
      /*into_producer=*/false,
      /*into_consumer=*/true,
      /*inline_const_tensor=*/true,
      /*disallow_if_then_else=*/true,
      /*require_injective=*/true,
      /*require_ordered=*/true,
      /*disallow_op=*/ffi::Array<ffi::String>{"prim.exp"}));
  rules.push_back(ScheduleRule::AddRFactor(
      /*max_jobs_per_core=*/16,
      /*max_innermost_factor=*/static_cast<int64_t>(64)));
  const auto reg_rvv_intrinsics =
      tvm::ffi::Function::GetGlobalRequired("tirx.tensor_intrin.register_rvv_isa_intrinsics");
  const auto rvv_kernels_inventory =
      reg_rvv_intrinsics(target, /* inventory_only */ true).cast<ffi::Map<ffi::String, int>>();
  for (const auto& intrin : rvv_kernels_inventory) {
    if (!TensorIntrin::Get(intrin.first, /*allow_missing*/ true)) {
      // on demand intrinsic register
      reg_rvv_intrinsics(target, /* inventory_only */ false);
    }
    rules.push_back(ScheduleRule::MultiLevelTilingWithIntrin(
        /*intrin_name=*/intrin.first,
        /*structure=*/"SSRSRS",
        /*tile_binds=*/std::nullopt,
        /*max_innermost_factor=*/static_cast<int64_t>(intrin.second),
        /*vector_load_lens=*/std::nullopt,
        /*reuse_read=*/std::nullopt,
        /*reuse_write=*/
        ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                        {"levels", ffi::Array<int64_t>{1, 2}},
                                        {"scope", ffi::String("global")}}));
  }
  rules.push_back(ScheduleRule::MultiLevelTiling(
      /*structure=*/"SSRSRS",
      /*tile_binds=*/std::nullopt,
      /*max_innermost_factor=*/static_cast<int64_t>(64),
      /*vector_load_lens=*/std::nullopt,
      /*reuse_read=*/std::nullopt,
      /*reuse_write=*/
      ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                      {"levels", ffi::Array<int64_t>{1, 2}},
                                      {"scope", ffi::String("global")}}));
  rules.push_back(ScheduleRule::ParallelizeVectorizeUnroll(
      /*max_jobs_per_core=*/16,
      /*max_vectorize_extent=*/64,
      /*unroll_max_steps=*/ffi::Array<int64_t>{0, 16, 64, 512},
      /*unroll_explicit=*/true));
  rules.push_back(ScheduleRule::RandomComputeLocation());

  return rules;
}

ffi::Array<ScheduleRule> GetARMNeonSpecificRules() {
  return {
      ScheduleRule::MultiLevelTilingWithIntrin(
          /*intrin_name=*/ffi::String("dot_4x4_i8i8s32_neon"),
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/static_cast<int64_t>(32),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<int64_t>{1, 2}},
                                          {"scope", ffi::String("global")}}),
  };
}

ffi::Array<ScheduleRule> GetARMDotprodSpecificRules() {
  return {
      ScheduleRule::MultiLevelTilingWithIntrin(
          /*intrin_name=*/ffi::String("dot_4x4_i8i8s32_sdot"),
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/static_cast<int64_t>(32),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<int64_t>{1, 2}},
                                          {"scope", ffi::String("global")}}),
      ScheduleRule::MultiLevelTilingWithIntrin(
          /*intrin_name=*/ffi::String("dot_4x4_u8u8u32_udot"),
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/static_cast<int64_t>(32),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<int64_t>{1, 2}},
                                          {"scope", ffi::String("global")}}),
      ScheduleRule::MultiLevelTilingWithIntrin(
          /*intrin_name=*/ffi::String("dot_4x4_u8u8i32_hdot"),
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/static_cast<int64_t>(32),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<int64_t>{1, 2}},
                                          {"scope", ffi::String("global")}}),
  };
}

ffi::Array<ScheduleRule> ScheduleRule::DefaultARM(const ffi::String& type) {
  return ffi::Array<ScheduleRule>::Agregate(
      ScheduleRule::ApplyCustomRule(), ScheduleRule::InlineConstantScalars(),
      ScheduleRule::AutoInline(
          /*into_producer=*/false,
          /*into_consumer=*/true,
          /*inline_const_tensor=*/true,
          /*disallow_if_then_else=*/true,
          /*require_injective=*/true,
          /*require_ordered=*/true,
          /*disallow_op=*/ffi::Array<ffi::String>{"prim.exp"}),
      ScheduleRule::AddRFactor(
          /*max_jobs_per_core=*/8,
          /*max_innermost_factor=*/static_cast<int64_t>(32)),
      "neon" == type ? GetARMNeonSpecificRules() : ffi::Array<ScheduleRule>{},
      "dotprod" == type ? GetARMDotprodSpecificRules() : ffi::Array<ScheduleRule>{},
      ScheduleRule::MultiLevelTiling(
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/static_cast<int64_t>(32),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<int64_t>{1, 2}},
                                          {"scope", ffi::String("global")}}),
      ScheduleRule::ParallelizeVectorizeUnroll(
          /*max_jobs_per_core=*/8,
          /*max_vectorize_extent=*/32,
          /*unroll_max_steps=*/ffi::Array<int64_t>{0, 8, 32, 256},
          /*unroll_explicit=*/true),
      ScheduleRule::RandomComputeLocation());
}

// Pattern A (RM): auto-default repr from reflection.

TVM_FFI_STATIC_INIT_BLOCK() {
  ScheduleRuleObj::RegisterReflection();
  PyScheduleRuleObj::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.meta_schedule.ScheduleRuleInitializeWithTuneContext",
                  &ScheduleRuleObj::InitializeWithTuneContext)
      .def_method("s_tir.meta_schedule.ScheduleRuleApply", &ScheduleRuleObj::Apply)
      .def_method("s_tir.meta_schedule.ScheduleRuleClone", &ScheduleRuleObj::Clone)
      .def("s_tir.meta_schedule.ScheduleRulePyScheduleRule", ScheduleRule::PyScheduleRule)
      .def("s_tir.meta_schedule.ScheduleRuleDefaultLLVM", ScheduleRule::DefaultLLVM)
      .def("s_tir.meta_schedule.ScheduleRuleDefaultCUDA", ScheduleRule::DefaultCUDA)
      .def("s_tir.meta_schedule.ScheduleRuleDefaultHexagon", ScheduleRule::DefaultHexagon)
      .def("s_tir.meta_schedule.ScheduleRuleDefaultARM", ScheduleRule::DefaultARM);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
