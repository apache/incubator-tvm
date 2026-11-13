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

#ifndef TVM_S_TIR_META_SCHEDULE_MUTATOR_H_
#define TVM_S_TIR_META_SCHEDULE_MUTATOR_H_

#include <tvm/base.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/base.h>
#include <tvm/s_tir/random_engine.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/s_tir/schedule/trace.h>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class TuneContext;
class Mutator;

/*! \brief Mutator is designed to mutate the trace to explore the design space. */
class MutatorObj : public ffi::Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~MutatorObj() = default;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MutatorObj>();
  }

  /*!
   * \brief Initialize the design space generator with tuning context.
   * \param context The tuning context for initialization.
   * \note This method is supposed to be called only once before every other method.
   */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Apply the mutator function to the given trace.
   * \param trace The given trace for mutation.
   * \param rand_state The random state for mutation.
   * \return None if mutator failed, otherwise return the mutated trace.
   */
  virtual ffi::Optional<s_tir::Trace> Apply(const s_tir::Trace& trace,
                                            LinearCongruentialEngine::TRandState* rand_state) = 0;

  /*!
   * \brief Clone the mutator.
   * \return The cloned mutator.
   */
  virtual Mutator Clone() const = 0;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("s_tir.meta_schedule.Mutator", MutatorObj, ffi::Object);
};

/*!
 * \brief Managed reference to MutatorObj
 * \sa MutatorObj
 */
class Mutator : public ffi::ObjectRef {
 public:
  explicit Mutator(ffi::UnsafeInit tag, ffi::ObjectPtr<MutatorObj> data) : ffi::ObjectRef(tag) {
    data_ = std::move(data);
  }

  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param context The tuning context for initialization.
   */
  using FInitializeWithTuneContext = ffi::TypedFunction<void(const TuneContext&)>;
  /*!
   * \brief Apply the mutator function to the given trace.
   * \param trace The given trace for mutation.
   * \return None if mutator failed, otherwise return the mutated trace.
   */
  using FApply = ffi::TypedFunction<ffi::Optional<s_tir::Trace>(
      const s_tir::Trace&, LinearCongruentialEngine::TRandState rand_state)>;
  /*!
   * \brief Clone the mutator.
   * \return The cloned mutator.
   */
  using FClone = ffi::TypedFunction<Mutator()>;
  /*! \brief Create a Mutator that mutates the decision of instruction Sample-Perfect-Tile */
  TVM_DLL static Mutator MutateTileSize();
  /*!
   * \brief Create a Mutator that mutates the parallel extent
   * \param max_jobs_per_core The maximum number of parallel jobs per core.
   * \return The created mutator.
   */
  TVM_DLL static Mutator MutateParallel(int64_t max_jobs_per_core);
  /*!
   * \brief Create a Mutator that mutates auto unroll step
   * \return The mutator created
   */
  TVM_DLL static Mutator MutateUnroll();
  /*!
   * \brief Create a Mutator that mutates the outcome of SampleComputeLocation
   * \return The mutator created
   */
  TVM_DLL static Mutator MutateComputeLocation();
  /*!
   * \brief Create a Mutator that mutates auto thread binding.
   * \return The mutator created
   */
  TVM_DLL static Mutator MutateThreadBinding();
  /*!
   * \brief Create a mutator with customized methods on the python-side.
   * \param f_initialize_with_tune_context The packed function of `InitializeWithTuneContext`.
   * \param f_apply The packed function of `Apply`.
   * \param f_clone The packed function of `Clone`.
   * \return The mutator created.
   */
  TVM_DLL static Mutator PyMutator(FInitializeWithTuneContext f_initialize_with_tune_context,
                                   FApply f_apply, FClone f_clone);
  /*! \brief Create default mutators for LLVM */
  TVM_DLL static ffi::Map<Mutator, FloatImm, void> DefaultLLVM();
  /*! \brief Create default mutators for CUDA */
  TVM_DLL static ffi::Map<Mutator, FloatImm, void> DefaultCUDA();
  /*! \brief Create default mutators for Hexagon */
  TVM_DLL static ffi::Map<Mutator, FloatImm, void> DefaultHexagon();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Mutator, ffi::ObjectRef, MutatorObj);
};

/*! \brief The mutator with customized methods on the python-side. */
class PyMutatorObj : public MutatorObj {
 public:
  using FInitializeWithTuneContext = Mutator::FInitializeWithTuneContext;
  using FApply = Mutator::FApply;
  using FClone = Mutator::FClone;
  /*! \brief The packed function to the `InitializeWithTuneContext` function. */
  FInitializeWithTuneContext f_initialize_with_tune_context;
  /*! \brief The packed function to the `Apply` function. */
  FApply f_apply;
  /*! \brief The packed function to the `Clone` function. */
  FClone f_clone;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PyMutatorObj>();
    // `f_initialize_with_tune_context` is not registered
    // `f_apply` is not registered
    // `f_clone` is not registered
  }

  void InitializeWithTuneContext(const TuneContext& context) final;
  ffi::Optional<s_tir::Trace> Apply(const s_tir::Trace& trace,
                                    LinearCongruentialEngine::TRandState* rand_state) final;
  Mutator Clone() const final;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.PyMutator", PyMutatorObj, MutatorObj);
};

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_MUTATOR_H_
