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

#ifndef TVM_S_TIR_META_SCHEDULE_POSTPROC_H_
#define TVM_S_TIR_META_SCHEDULE_POSTPROC_H_

#include <tvm/base.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/base.h>
#include <tvm/s_tir/schedule/schedule.h>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class TuneContext;
class Postproc;

/*!
 * \brief Rules to apply a postprocessor to a schedule.
 */
class PostprocObj : public ffi::Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~PostprocObj() = default;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PostprocObj>();
  }

  /*!
   * \brief Initialize the design space generator with tuning context.
   * \param context The tuning context for initialization.
   * \note This method is supposed to be called only once before every other method.
   */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Apply a postprocessor to the given schedule.
   * \param sch The schedule to be post processed.
   * \return Whether the postprocessor was successfully applied.
   */
  virtual bool Apply(const s_tir::Schedule& sch) = 0;

  /*!
   * \brief Clone the postprocessor.
   * \return The cloned postprocessor.
   */
  virtual Postproc Clone() const = 0;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("s_tir.meta_schedule.Postproc", PostprocObj, ffi::Object);
};

/*!
 * \brief Managed reference to PostprocObj
 * \sa PostprocObj
 */
class Postproc : public ffi::ObjectRef {
 public:
  explicit Postproc(ffi::UnsafeInit tag, ffi::ObjectPtr<PostprocObj> data) : ffi::ObjectRef(tag) {
    data_ = std::move(data);
  }

  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param context The tuning context for initialization.
   */
  using FInitializeWithTuneContext = ffi::TypedFunction<void(const TuneContext&)>;
  /*!
   * \brief Apply a postprocessor to the given schedule.
   * \param sch The schedule to be post processed.
   * \return Whether the postprocessor was successfully applied.
   */
  using FApply = ffi::TypedFunction<bool(const s_tir::Schedule&)>;
  /*!
   * \brief Clone the postprocessor.
   * \return The cloned postprocessor.
   */
  using FClone = ffi::TypedFunction<Postproc()>;
  /*!
   * \brief Create a postprocessor with customized methods on the python-side.
   * \param f_initialize_with_tune_context The packed function of `InitializeWithTuneContext`.
   * \param f_apply The packed function of `Apply`.
   * \param f_clone The packed function of `Clone`.
   * \return The postprocessor created.
   */
  TVM_DLL static Postproc PyPostproc(FInitializeWithTuneContext f_initialize_with_tune_context,  //
                                     FApply f_apply,                                             //
                                     FClone f_clone);
  /*!
   * \brief Create a postprocessor that checks if all loops are static
   * \return The postprocessor created
   */
  TVM_DLL static Postproc DisallowDynamicLoop();
  /*!
   * \brief Create a postprocessor that checks if all async mem copies are not strided.
   * \return The postprocessor created
   */
  TVM_DLL static Postproc DisallowAsyncStridedMemCopy();
  /*!
   * \brief Create a postprocessor that rewrites the cooperative fetch annotation to
   * actual vectorized cooperative fetching in loop bindings.
   * \return The postprocessor created.
   */
  TVM_DLL static Postproc RewriteCooperativeFetch();
  /*!
   * \brief Creates a postprocessor that applies parallelization, vectorization and auto unrolling
   * according to the annotation of each block
   * \return The postprocessor created
   */
  TVM_DLL static Postproc RewriteParallelVectorizeUnroll();
  /*!
   * \brief Create a postprocessor that rewrites reduction block by moving the init block out.
   * \return The postprocessor created.
   */
  TVM_DLL static Postproc RewriteReductionBlock();
  /*!
   * \brief Create a postprocessor that adds thread binding to unbound blocks
   * \param max_threadblocks The max number of threadblocks in the CUDA device.
   * \return The postprocessor created.
   */
  TVM_DLL static Postproc RewriteUnboundBlock(int max_threadblocks);
  /*!
   * \brief Create a postprocessor that applies tensorization to annotated blocks
   * \param vectorize_init_loop Whether or not vectorize the initialization loop produced by
   * DecomposeReduction
   * \return The postprocessor created.
   */
  TVM_DLL static Postproc RewriteTensorize(bool vectorize_init_loop = false);
  /*!
   * \brief Creates a postprocessor that verifies if the GPU code is correct
   * \return The postprocessor created
   */
  TVM_DLL static Postproc VerifyGPUCode();
  /*!
   * \brief Verifies that the VTCM usage of a given schedule is within the provided limit.
   * \return The postprocessor created
   */
  TVM_DLL static Postproc VerifyVTCMLimit();
  /*!
   * \brief Creates a postprocessor that rewrites the layout of input tensor
   * \note Weight layout rewrite is supported so far, activation layout rewrite will be added.
   * \return The postprocessor created
   */
  TVM_DLL static Postproc RewriteLayout();
  /*! \brief Create default postprocessors for LLVM */
  TVM_DLL static ffi::Array<Postproc, void> DefaultLLVM();
  /*! \brief Create default postprocessors for x86 (AVX512 and VNNI) */
  TVM_DLL static ffi::Array<Postproc, void> DefaultCPUTensorization();
  /*! \brief Create default postprocessors for RISCV */
  TVM_DLL static ffi::Array<Postproc, void> DefaultRISCV();
  /*! \brief Create default postprocessors for CUDA */
  TVM_DLL static ffi::Array<Postproc, void> DefaultCUDA();
  /*! \brief Create default postprocessors for Hexagon */
  TVM_DLL static ffi::Array<Postproc, void> DefaultHexagon();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Postproc, ffi::ObjectRef, PostprocObj);
};

/*! \brief The postprocessor with customized methods on the python-side. */
class PyPostprocObj : public PostprocObj {
 public:
  using FInitializeWithTuneContext = Postproc::FInitializeWithTuneContext;
  using FApply = Postproc::FApply;
  using FClone = Postproc::FClone;
  /*! \brief The packed function to the `InitializeWithTuneContext` function. */
  FInitializeWithTuneContext f_initialize_with_tune_context;
  /*! \brief The packed function to the `Apply` function. */
  FApply f_apply;
  /*! \brief The packed function to the `Clone` function. */
  FClone f_clone;

  static void RegisterReflection() {
    // `f_initialize_with_tune_context` is not registered
    // `f_apply` is not registered
    // `f_clone` is not registered
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PyPostprocObj>();
  }

  void InitializeWithTuneContext(const TuneContext& context) final;
  bool Apply(const s_tir::Schedule& sch) final;
  Postproc Clone() const final;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.PyPostproc", PyPostprocObj, PostprocObj);
};

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_POSTPROC_H_
