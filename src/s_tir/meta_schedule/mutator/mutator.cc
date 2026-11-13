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

void PyMutatorObj::InitializeWithTuneContext(const TuneContext& context) {
  TVM_FFI_ICHECK(f_initialize_with_tune_context != nullptr)
      << "PyMutator's InitializeWithTuneContext method not implemented!";
  f_initialize_with_tune_context(context);
}

ffi::Optional<s_tir::Trace> PyMutatorObj::Apply(const s_tir::Trace& trace,
                                                LinearCongruentialEngine::TRandState* rand_state) {
  TVM_FFI_ICHECK(f_apply != nullptr) << "PyMutator's Apply method not implemented!";
  return f_apply(trace, *rand_state);
}

Mutator PyMutatorObj::Clone() const {
  TVM_FFI_ICHECK(f_clone != nullptr) << "PyMutator's Clone method not implemented!";
  return f_clone();
}

Mutator Mutator::PyMutator(
    PyMutatorObj::FInitializeWithTuneContext f_initialize_with_tune_context,  //
    PyMutatorObj::FApply f_apply,                                             //
    PyMutatorObj::FClone f_clone) {
  ffi::ObjectPtr<PyMutatorObj> n = ffi::make_object<PyMutatorObj>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_apply = std::move(f_apply);
  n->f_clone = std::move(f_clone);
  return Mutator(ffi::UnsafeInit{}, n);
}

ffi::Map<Mutator, FloatImm> Mutator::DefaultLLVM() {
  return ffi::Map<Mutator, FloatImm>{
      {Mutator::MutateTileSize(), FloatImm(PrimType::Float(64), 0.9)},
      {Mutator::MutateComputeLocation(), FloatImm(PrimType::Float(64), 0.05)},
      {Mutator::MutateUnroll(), FloatImm(PrimType::Float(64), 0.03)},
      {Mutator::MutateParallel(/*max_jobs_per_core=*/16), FloatImm(PrimType::Float(64), 0.02)}};
}

ffi::Map<Mutator, FloatImm> Mutator::DefaultCUDA() {
  return ffi::Map<Mutator, FloatImm>{
      {Mutator::MutateTileSize(), FloatImm(PrimType::Float(64), 0.9)},
      {Mutator::MutateUnroll(), FloatImm(PrimType::Float(64), 0.08)},
      {Mutator::MutateThreadBinding(), FloatImm(PrimType::Float(64), 0.02)}};
}

ffi::Map<Mutator, FloatImm> Mutator::DefaultHexagon() {
  return ffi::Map<Mutator, FloatImm>{
      {Mutator::MutateTileSize(), FloatImm(PrimType::Float(64), 0.9)},
      {Mutator::MutateComputeLocation(), FloatImm(PrimType::Float(64), 0.05)},
      {Mutator::MutateUnroll(), FloatImm(PrimType::Float(64), 0.03)},
      {Mutator::MutateParallel(/*max_jobs_per_core=*/16), FloatImm(PrimType::Float(64), 0.02)}};
}

// Pattern A (RM): auto-default repr from reflection.

TVM_FFI_STATIC_INIT_BLOCK() {
  MutatorObj::RegisterReflection();
  PyMutatorObj::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.meta_schedule.MutatorInitializeWithTuneContext",
                  &MutatorObj::InitializeWithTuneContext)
      .def("s_tir.meta_schedule.MutatorApply",
           [](Mutator self, s_tir::Trace trace, TRandState seed) -> ffi::Optional<s_tir::Trace> {
             TRandState seed_ = (seed != -1) ? seed : LinearCongruentialEngine::DeviceRandom();
             return self->Apply(trace, &seed_);
           })
      .def_method("s_tir.meta_schedule.MutatorClone", &MutatorObj::Clone)
      .def("s_tir.meta_schedule.MutatorPyMutator", Mutator::PyMutator)
      .def("s_tir.meta_schedule.MutatorDefaultLLVM", Mutator::DefaultLLVM)
      .def("s_tir.meta_schedule.MutatorDefaultCUDA", Mutator::DefaultCUDA)
      .def("s_tir.meta_schedule.MutatorDefaultHexagon", Mutator::DefaultHexagon);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
