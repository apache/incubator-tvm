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

#ifndef TVM_S_TIR_META_SCHEDULE_FEATURE_EXTRACTOR_H_
#define TVM_S_TIR_META_SCHEDULE_FEATURE_EXTRACTOR_H_

#include <tvm/base.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/tensor.h>
#include <tvm/s_tir/meta_schedule/measure_candidate.h>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class TuneContext;

/*! \brief Extractor for features from measure candidates for use in cost model. */
class FeatureExtractorObj : public ffi::Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~FeatureExtractorObj() = default;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FeatureExtractorObj>();
  }

  /*!
   * \brief Extract features from the given measure candidate.
   * \param context The tuning context for feature extraction.
   * \param candidates The measure candidates to extract features from.
   * \return The feature tensor extracted.
   */
  virtual ffi::Array<tvm::ffi::Tensor> ExtractFrom(
      const TuneContext& context, const ffi::Array<MeasureCandidate>& candidates) = 0;
  TVM_FFI_DECLARE_OBJECT_INFO("s_tir.meta_schedule.FeatureExtractor", FeatureExtractorObj,
                              ffi::Object);
};

/*! \brief The feature extractor with customized methods on the python-side. */
class PyFeatureExtractorObj : public FeatureExtractorObj {
 public:
  /*!
   * \brief Extract features from the given measure candidate.
   * \param context The tuning context for feature extraction.
   * \param candidates The measure candidates to extract features from.
   * \return The feature tensor extracted.
   */
  using FExtractFrom = ffi::TypedFunction<ffi::Array<tvm::ffi::Tensor>(
      const TuneContext& context, const ffi::Array<MeasureCandidate>& candidates)>;
  /*! \brief The packed function to the `ExtractFrom` function. */
  FExtractFrom f_extract_from;

  static void RegisterReflection() {
    // `f_extract_from` is not registered
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PyFeatureExtractorObj>();
  }

  ffi::Array<tvm::ffi::Tensor> ExtractFrom(const TuneContext& context,
                                           const ffi::Array<MeasureCandidate>& candidates) final;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.PyFeatureExtractor", PyFeatureExtractorObj,
                                    FeatureExtractorObj);
};

/*!
 * \brief Managed reference to FeatureExtractorObj
 * \sa FeatureExtractorObj
 */
class FeatureExtractor : public ffi::ObjectRef {
 public:
  explicit FeatureExtractor(ffi::UnsafeInit tag, ffi::ObjectPtr<FeatureExtractorObj> data)
      : ffi::ObjectRef(tag) {
    data_ = std::move(data);
  }

  /*!
   * \brief Create a feature extractor that extracts features from each TensorStore
   * \param tensors_per_store The number of tensors in each TensorStore; Pad or truncate if
   * necessary.
   * \param arith_intensity_curve_num_samples The number of samples used in the arithmetic intensity
   * curve.
   * \param cache_line_bytes The number of bytes in a cache line.
   * \param extract_workload Whether to extract features in the workload in tuning context or not.
   * \return The feature extractor created.
   */
  TVM_DLL static FeatureExtractor PerStoreFeature(int tensors_per_store = 5,
                                                  int arith_intensity_curve_num_samples = 10,
                                                  int cache_line_bytes = 64,
                                                  bool extract_workload = false);
  /*!
   * \brief Create a feature extractor with customized methods on the python-side.
   * \param f_extract_from The packed function of `ExtractFrom`.
   * \return The feature extractor created.
   */
  TVM_DLL static FeatureExtractor PyFeatureExtractor(
      PyFeatureExtractorObj::FExtractFrom f_extract_from);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FeatureExtractor, ffi::ObjectRef,
                                                FeatureExtractorObj);
};

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_FEATURE_EXTRACTOR_H_
