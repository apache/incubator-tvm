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

#ifndef TVM_S_TIR_META_SCHEDULE_COST_MODEL_H_
#define TVM_S_TIR_META_SCHEDULE_COST_MODEL_H_

#include <tvm/base.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/base.h>
#include <tvm/s_tir/meta_schedule/arg_info.h>
#include <tvm/s_tir/meta_schedule/measure_candidate.h>
#include <tvm/s_tir/meta_schedule/runner.h>
#include <tvm/s_tir/schedule/schedule.h>

#include <vector>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class TuneContext;

/*! \brief Cost model. */
class CostModelObj : public ffi::Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~CostModelObj() = default;

  /*!
   * \brief Load the cost model from given file location.
   * \param path The file path.
   */
  virtual void Load(const ffi::String& path) = 0;

  /*!
   * \brief Save the cost model to given file location.
   * \param path The file path.
   */
  virtual void Save(const ffi::String& path) = 0;

  /*!
   * \brief Update the cost model given running results.
   * \param context The tuning context.
   * \param candidates The measure candidates.
   * \param results The running results of the measure candidates.
   */
  virtual void Update(const TuneContext& context, const ffi::Array<MeasureCandidate>& candidates,
                      const ffi::Array<RunnerResult>& results) = 0;

  /*!
   * \brief Predict the normalized score (the larger the better) of given measure candidates.
   * \param context The tuning context.
   * \param candidates The measure candidates.
   * \return The predicted normalized score.
   */
  virtual std::vector<double> Predict(const TuneContext& context,
                                      const ffi::Array<MeasureCandidate>& candidates) = 0;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("s_tir.meta_schedule.CostModel", CostModelObj, ffi::Object);
};

/*! \brief The cost model with customized methods on the python-side. */
class PyCostModelObj : public CostModelObj {
 public:
  /*!
   * \brief Load the cost model from given file location.
   * \param path The file path.
   */
  using FLoad = ffi::TypedFunction<void(ffi::String)>;
  /*!
   * \brief Save the cost model to given file location.
   * \param path The file path.
   */
  using FSave = ffi::TypedFunction<void(ffi::String)>;
  /*!
   * \brief Update the cost model given running results.
   * \param context The tuning context.
   * \param candidates The measure candidates.
   * \param results The running results of the measure candidates.
   * \return Whether cost model was updated successfully.
   */
  using FUpdate = ffi::TypedFunction<void(const TuneContext&, const ffi::Array<MeasureCandidate>&,
                                          const ffi::Array<RunnerResult>&)>;
  /*!
   * \brief Predict the running results of given measure candidates.
   * \param context The tuning context.
   * \param candidates The measure candidates.
   * \param p_addr The address to save the estimated running results.
   */
  using FPredict = ffi::TypedFunction<void(const TuneContext&, const ffi::Array<MeasureCandidate>&,
                                           void* p_addr)>;
  /*! \brief The packed function to the `Load` function. */
  FLoad f_load;
  /*! \brief The packed function to the `Save` function. */
  FSave f_save;
  /*! \brief The packed function to the `Update` function. */
  FUpdate f_update;
  /*! \brief The packed function to the `Predict` function. */
  FPredict f_predict;

  void Load(const ffi::String& path);
  void Save(const ffi::String& path);
  void Update(const TuneContext& context, const ffi::Array<MeasureCandidate>& candidates,
              const ffi::Array<RunnerResult>& results);
  std::vector<double> Predict(const TuneContext& context,
                              const ffi::Array<MeasureCandidate>& candidates);
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.PyCostModel", PyCostModelObj,
                                    CostModelObj);
};

/*!
 * \brief Managed reference to CostModelObj
 * \sa CostModelObj
 */
class CostModel : public ffi::ObjectRef {
 public:
  explicit CostModel(ffi::UnsafeInit tag, ffi::ObjectPtr<CostModelObj> data) : ffi::ObjectRef(tag) {
    data_ = std::move(data);
  }

  /*!
   * \brief Create a cost model with customized methods on the python-side.
   * \param f_load The packed function of `Load`.
   * \param f_save The packed function of `Save`.
   * \param f_update The packed function of `Update`.
   * \param f_predict The packed function of `Predict`.
   * \return The cost model created.
   */
  TVM_DLL static CostModel PyCostModel(PyCostModelObj::FLoad f_load,      //
                                       PyCostModelObj::FSave f_save,      //
                                       PyCostModelObj::FUpdate f_update,  //
                                       PyCostModelObj::FPredict f_predict);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CostModel, ffi::ObjectRef, CostModelObj);
};

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_COST_MODEL_H_
