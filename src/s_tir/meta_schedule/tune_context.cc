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

#include <utility>

#include "./utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

TuneContext::TuneContext(ffi::Optional<IRModule> mod, ffi::Optional<Target> target,
                         ffi::Optional<SpaceGenerator> space_generator,
                         ffi::Optional<SearchStrategy> search_strategy,
                         ffi::Optional<ffi::String> task_name, int num_threads,
                         TRandState rand_state, ffi::Function logger) {
  TVM_FFI_CHECK(rand_state == -1 || rand_state >= 0, ValueError)
      << "Invalid random state: " << rand_state;
  ffi::ObjectPtr<TuneContextObj> n = ffi::make_object<TuneContextObj>();
  n->mod = mod;
  n->target = target;
  n->space_generator = space_generator;
  n->search_strategy = search_strategy;
  n->task_name = task_name;
  n->num_threads = num_threads;
  n->rand_state = LinearCongruentialEngine::NormalizeSeed(rand_state);
  n->logger = logger;
  data_ = std::move(n);
}

TuneContext TuneContextObj::Clone() const {
  ffi::ObjectPtr<TuneContextObj> n = ffi::make_object<TuneContextObj>(*this);
  if (this->space_generator.has_value()) {
    n->space_generator = this->space_generator.value()->Clone();
  }
  if (this->search_strategy.has_value()) {
    n->search_strategy = this->search_strategy.value()->Clone();
  }
  n->rand_state = ForkSeed(&n->rand_state);
  n->Initialize();
  return TuneContext(n);
}

void TuneContextObj::Initialize() {
  if (this->space_generator.has_value()) {
    this->space_generator.value()->InitializeWithTuneContext(ffi::GetRef<TuneContext>(this));
  }
  if (this->search_strategy.has_value()) {
    this->search_strategy.value()->InitializeWithTuneContext(ffi::GetRef<TuneContext>(this));
  }
}

TVM_FFI_STATIC_INIT_BLOCK() { TuneContextObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.meta_schedule.TuneContext",
           [](ffi::Optional<IRModule> mod, ffi::Optional<Target> target,
              ffi::Optional<SpaceGenerator> space_generator,
              ffi::Optional<SearchStrategy> search_strategy, ffi::Optional<ffi::String> task_name,
              int num_threads, TRandState rand_state, ffi::Function logger) -> TuneContext {
             return TuneContext(mod, target, space_generator, search_strategy, task_name,
                                num_threads, rand_state, logger);
           })
      .def("s_tir.meta_schedule._SHash2Hex", SHash2Hex)
      .def_method("s_tir.meta_schedule.TuneContextInitialize", &TuneContextObj::Initialize)
      .def_method("s_tir.meta_schedule.TuneContextClone", &TuneContextObj::Clone);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
