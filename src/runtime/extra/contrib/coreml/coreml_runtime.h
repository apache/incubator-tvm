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

/*!
 * \brief CoreML runtime that can run coreml model
 *        containing only tvm ffi::Function.
 * \file coreml_runtime.h
 */
#ifndef TVM_RUNTIME_CONTRIB_COREML_COREML_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_COREML_COREML_RUNTIME_H_

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/tensor.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief CoreML model.
 */
class CoreMLModel {
 public:
  /*!
   * \brief constructor
   * \param url The directory where compiled models are located.
   */
  explicit CoreMLModel(NSURL* url) {
    url_ = url;
    model_ = [MLModel modelWithContentsOfURL:url error:nil];
    input_dict_ = [NSMutableDictionary dictionary];
    output_ = nil;
  }
  /*!
   * \brief Invoke the coreml prediction.
   */
  void Invoke();
  /*!
   * \brief set input to the model.
   * \param key The input name.
   * \param data_in The input data.
   */
  void SetInput(const std::string& key, DLTensor* data_in);
  /*!
   * \brief Return Tensor for given output index.
   * \param index The output index.
   *
   * \return Tensor corresponding to given output node index.
   */
  ffi::Tensor GetOutput(int index) const;
  /*!
   * \brief Return the number of outputs
   *
   * \return The number of outputs
   */
  int GetNumOutputs() const;

  // CoreML model url
  NSURL* url_;
  // CoreML model
  MLModel* model_;
  // CoreML model input dictionary
  NSMutableDictionary<NSString*, id>* input_dict_;
  // CoreML model output
  id<MLFeatureProvider> output_;
};

/*!
 * \brief CoreML runtime.
 *
 *  This runtime can be accessed in various language via
 *  TVM runtime ffi::Function API.
 */
class CoreMLRuntime : public ffi::ModuleObj {
 public:
  /*!
   * \brief Get member function to front-end.
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  virtual ffi::Optional<ffi::Function> GetFunction(const ffi::String& name);

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  /*!
   * \brief Serialize the content of the mlmodelc directory and save it to
   *        binary stream.
   * \param stream The binary stream to save to.
   */
  ffi::Bytes SaveToBytes() const final;

  /*!
   * \return The type key of the executor.
   */
  const char* kind() const { return "coreml"; }

  /*!
   * \brief Initialize the coreml runtime with coreml model and context.
   * \param symbol The symbol of this model.
   * \param model_path The compiled model path.
   */
  void Init(const std::string& symbol, const std::string& model_path);

  /*! \brief The symbol that represents the Core ML model. */
  std::string symbol_;

  /*! \brief The Core ML model */
  std::unique_ptr<CoreMLModel> model_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_COREML_COREML_RUNTIME_H_
