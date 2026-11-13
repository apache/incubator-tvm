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

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

void PyPostprocObj::InitializeWithTuneContext(const TuneContext& context) {
  TVM_FFI_ICHECK(f_initialize_with_tune_context != nullptr)
      << "PyPostproc's InitializeWithTuneContext method not implemented!";
  f_initialize_with_tune_context(context);
}

bool PyPostprocObj::Apply(const s_tir::Schedule& sch) {
  TVM_FFI_ICHECK(f_apply != nullptr) << "PyPostproc's Apply method not implemented!";
  return f_apply(sch);
}

Postproc PyPostprocObj::Clone() const {
  TVM_FFI_ICHECK(f_clone != nullptr) << "PyPostproc's Clone method not implemented!";
  return f_clone();
}

Postproc Postproc::PyPostproc(
    PyPostprocObj::FInitializeWithTuneContext f_initialize_with_tune_context,  //
    PyPostprocObj::FApply f_apply,                                             //
    PyPostprocObj::FClone f_clone) {
  ffi::ObjectPtr<PyPostprocObj> n = ffi::make_object<PyPostprocObj>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_apply = std::move(f_apply);
  n->f_clone = std::move(f_clone);
  return Postproc(ffi::UnsafeInit{}, n);
}

ffi::Array<Postproc> Postproc::DefaultLLVM() {
  return ffi::Array<Postproc>{
      Postproc::DisallowDynamicLoop(),
      Postproc::RewriteParallelVectorizeUnroll(),
      Postproc::RewriteReductionBlock(),
      Postproc::RewriteLayout(),
  };
}

ffi::Array<Postproc> Postproc::DefaultCPUTensorization() {
  return ffi::Array<Postproc>{
      Postproc::DisallowDynamicLoop(),   Postproc::RewriteParallelVectorizeUnroll(),
      Postproc::RewriteReductionBlock(), Postproc::RewriteTensorize(/*vectorize_init_loop=*/true),
      Postproc::RewriteLayout(),
  };
}

ffi::Array<Postproc> Postproc::DefaultRISCV() {
  return ffi::Array<Postproc>{
      Postproc::DisallowDynamicLoop(),   Postproc::RewriteParallelVectorizeUnroll(),
      Postproc::RewriteReductionBlock(), Postproc::RewriteTensorize(/*vectorize_init_loop=*/false),
      Postproc::RewriteLayout(),
  };
}

ffi::Array<Postproc> Postproc::DefaultCUDA() {
  return ffi::Array<Postproc>{
      Postproc::DisallowDynamicLoop(),
      Postproc::RewriteCooperativeFetch(),
      Postproc::RewriteUnboundBlock(/*max_threadblocks=*/256),
      Postproc::RewriteParallelVectorizeUnroll(),
      Postproc::RewriteReductionBlock(),
      Postproc::VerifyGPUCode(),
  };
}

ffi::Array<Postproc> Postproc::DefaultHexagon() {
  return ffi::Array<Postproc>{
      Postproc::DisallowDynamicLoop(),   Postproc::RewriteParallelVectorizeUnroll(),
      Postproc::RewriteReductionBlock(), Postproc::RewriteLayout(),
      Postproc::VerifyVTCMLimit(),
  };
}

// Pattern A (RM): auto-default repr from reflection.

TVM_FFI_STATIC_INIT_BLOCK() {
  PostprocObj::RegisterReflection();
  PyPostprocObj::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.meta_schedule.PostprocInitializeWithTuneContext",
                  &PostprocObj::InitializeWithTuneContext)
      .def_method("s_tir.meta_schedule.PostprocApply", &PostprocObj::Apply)
      .def_method("s_tir.meta_schedule.PostprocClone", &PostprocObj::Clone)
      .def("s_tir.meta_schedule.PostprocPyPostproc", Postproc::PyPostproc)
      .def("s_tir.meta_schedule.PostprocDefaultLLVM", Postproc::DefaultLLVM)
      .def("s_tir.meta_schedule.PostprocDefaultCUDA", Postproc::DefaultCUDA)
      .def("s_tir.meta_schedule.PostprocDefaultHexagon", Postproc::DefaultHexagon);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
