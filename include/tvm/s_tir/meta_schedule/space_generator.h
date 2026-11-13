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
#ifndef TVM_S_TIR_META_SCHEDULE_SPACE_GENERATOR_H_
#define TVM_S_TIR_META_SCHEDULE_SPACE_GENERATOR_H_

#include <tvm/base.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/meta_schedule/mutator.h>
#include <tvm/s_tir/meta_schedule/postproc.h>
#include <tvm/s_tir/meta_schedule/schedule_rule.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/target/target.h>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

// Forward declaration
class TuneContext;
class SpaceGenerator;

/*!
 * \brief The abstract class for design space generation.
 * \note The relationship between SpaceGenerator and other classes are as follows:
        +--------------------------------------------------------------+
    +--+-----------------------------------------------------------+    |
  +--+------------------ Tune Context -----------------------------+ |  |
  |                +---------------------+                        |  |  |
  |                |                     |   Generate             |  |  |
  |                |   Space Generator   +--------------+         |  |  |
  |                |                     |              |         |  |  |
  |                +---------------------+              v         |  |  |
  |                                               Design Space    |  |  |
  |                +---------------------+              |         |  |  |
  |      Generate  |                     |   Pretuning  |         |  |  |
  |    +-----------+   Search Strategy   |<-------------+         |  |  |
  |    |           |                     |                        |  +--+
  |    |           +---------------------+                        +--+
  +----+----------------------------------------------------------+
      |
      |
  +----+---------------- Managed By Task Scheduler ---------------------+
  |    |                                 +-----------+                  |
  |    |                      Send to    |           |  Send to         |
  |    v                  +-------------+|  Builder  +----------+       |
  | Measure Candidate     |   Builder    |           |  Runner  |       |
  |    |                  |              +-----------+          |       |
  |    |     +------------+------------+                        |       |
  |    |     |                         |     +-----------+      |       |
  |    +---->|   Task Scheduler        |     |           |      |       |
  |          |                         |     |  Runner   |<-----+       |
  |          +-------------------------+     |           |              |
  |                   ^                      +-----+-----+              |
  |                   |                            |                    |
  |                   +----  Runner Future <-------+                    |
  +---------------------------------------------------------------------+
*/
class SpaceGeneratorObj : public ffi::Object {
 public:
  /*! \brief The schedule rules. */
  ffi::Optional<ffi::Array<ScheduleRule>> sch_rules;
  /*! \brief The postprocessors. */
  ffi::Optional<ffi::Array<Postproc>> postprocs;
  /*! \brief The probability of using certain mutator. */
  ffi::Optional<ffi::Map<Mutator, FloatImm>> mutator_probs;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpaceGeneratorObj>()
        .def_ro("sch_rules", &SpaceGeneratorObj::sch_rules)
        .def_ro("postprocs", &SpaceGeneratorObj::postprocs)
        .def_ro("mutator_probs", &SpaceGeneratorObj::mutator_probs);
  }

  /*! \brief Default destructor */
  virtual ~SpaceGeneratorObj() = default;

  /*!
   * \brief Initialize the design space generator with tuning context.
   * \param context The tuning context for initialization.
   * \note This method is supposed to be called only once before every other method.
   */
  virtual void InitializeWithTuneContext(const TuneContext& context);

  /*!
   * \brief Generate design spaces given a module.
   * \param mod The module used for design space generation.
   * \return The generated design spaces, i.e., schedules.
   */
  virtual ffi::Array<s_tir::Schedule> GenerateDesignSpace(const IRModule& mod) = 0;

  /*!
   * \brief Clone the space generator.
   * \return The cloned space generator.
   */
  virtual SpaceGenerator Clone() const = 0;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("s_tir.meta_schedule.SpaceGenerator", SpaceGeneratorObj, ffi::Object);
};

/*!
 * \brief Managed reference to SpaceGeneratorObj.
 * \sa SpaceGeneratorObj
 */
class SpaceGenerator : public ffi::ObjectRef {
 public:
  /*!
   * \brief Constructor from ffi::ObjectPtr<SpaceGeneratorObj>.
   * \param data The object pointer.
   */
  explicit SpaceGenerator(ffi::ObjectPtr<SpaceGeneratorObj> data) : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param context The tuning context for initialization.
   */
  using FInitializeWithTuneContext = ffi::TypedFunction<void(const TuneContext&)>;
  /*!
   * \brief The function type of `GenerateDesignSpace` method.
   * \param mod The module used for design space generation.
   * \return The generated design spaces, i.e., schedules.
   */
  using FGenerateDesignSpace = ffi::TypedFunction<ffi::Array<s_tir::Schedule>(const IRModule&)>;
  /*!
   * \brief The function type of `Clone` method.
   * \return The cloned space generator.
   */
  using FClone = ffi::TypedFunction<SpaceGenerator()>;

 protected:
  SpaceGenerator() = default;

 public:
  /*!
   * \brief Create a design space generator with customized methods on the python-side.
   * \param sch_rules The schedule rules.
   * \param postprocs The postprocessors.
   * \param mutator_probs The probability of using certain mutator.
   * \param f_initialize_with_tune_context The packed function of `InitializeWithTuneContext`.
   * \param f_generate_design_space The packed function of `GenerateDesignSpace`.
   * \param f_clone The packed function of `Clone`.
   * \return The design space generator created.
   */
  TVM_DLL static SpaceGenerator PySpaceGenerator(
      ffi::Optional<ffi::Array<ScheduleRule>> sch_rules,
      ffi::Optional<ffi::Array<Postproc>> postprocs,
      ffi::Optional<ffi::Map<Mutator, FloatImm>> mutator_probs,
      FInitializeWithTuneContext f_initialize_with_tune_context,
      FGenerateDesignSpace f_generate_design_space, FClone f_clone);
  /*!
   * \brief Create a design space generator with customized schedule function.
   * \param schedule_fn The schedule function, which can have the following signatures:
   * 1) void(Schedule)
   * 2) Schedule(Schedule)
   * 3) ffi::Array<Schedule>(Schedule)
   * \param sch_rules The schedule rules.
   * \param postprocs The postprocessors.
   * \param mutator_probs The probability of using certain mutator.
   */
  TVM_DLL static SpaceGenerator ScheduleFn(
      ffi::Function schedule_fn, ffi::Optional<ffi::Array<ScheduleRule>> sch_rules,
      ffi::Optional<ffi::Array<Postproc>> postprocs,
      ffi::Optional<ffi::Map<Mutator, FloatImm>> mutator_probs);
  /*!
   * \brief Create a design space generator that is union of multiple design space generators.
   * \param space_generators An array of design space generators to be unioned.
   * \param sch_rules The schedule rules.
   * \param postprocs The postprocessors.
   * \param mutator_probs The probability of using certain mutator.
   * \return The design space generator created.
   */
  TVM_DLL static SpaceGenerator SpaceGeneratorUnion(
      ffi::Array<SpaceGenerator, void> space_generators,
      ffi::Optional<ffi::Array<ScheduleRule>> sch_rules,
      ffi::Optional<ffi::Array<Postproc>> postprocs,
      ffi::Optional<ffi::Map<Mutator, FloatImm>> mutator_probs);
  /*!
   * \brief Create a design space generator that generates design spaces by applying schedule
   * rules to blocks in post-DFS order.
   * \param f_block_filter The filter function to filter blocks to be applied with schedule rules.
   * \param sch_rules The schedule rules.
   * \param postprocs The postprocessors.
   * \param mutator_probs The probability of using certain mutator.
   * \return The design space generator created.
   */
  TVM_DLL static SpaceGenerator PostOrderApply(
      ffi::Function f_block_filter, ffi::Optional<ffi::Array<ScheduleRule>> sch_rules,
      ffi::Optional<ffi::Array<Postproc>> postprocs,
      ffi::Optional<ffi::Map<Mutator, FloatImm>> mutator_probs);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SpaceGenerator, ffi::ObjectRef, SpaceGeneratorObj);
};

/*! \brief The design space generator with customized methods on the python-side. */
class PySpaceGeneratorObj : public SpaceGeneratorObj {
 public:
  using FInitializeWithTuneContext = SpaceGenerator::FInitializeWithTuneContext;
  using FGenerateDesignSpace = SpaceGenerator::FGenerateDesignSpace;
  using FClone = SpaceGenerator::FClone;
  /*! \brief The packed function to the `InitializeWithTuneContext` function. */
  FInitializeWithTuneContext f_initialize_with_tune_context;
  /*! \brief The packed function to the `GenerateDesignSpace` function. */
  FGenerateDesignSpace f_generate_design_space;
  /*! \brief The packed function to the `Clone` function. */
  FClone f_clone;

  static void RegisterReflection() {
    // `f_initialize_with_tune_context` is not registered
    // `f_generate_design_space` is not registered
    // `f_clone` is not registered
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PySpaceGeneratorObj>();
  }

  void InitializeWithTuneContext(const TuneContext& context) final;
  ffi::Array<s_tir::Schedule> GenerateDesignSpace(const IRModule& mod) final;
  SpaceGenerator Clone() const final;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.PySpaceGenerator", PySpaceGeneratorObj,
                                    SpaceGeneratorObj);
};

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_SPACE_GENERATOR_H_
