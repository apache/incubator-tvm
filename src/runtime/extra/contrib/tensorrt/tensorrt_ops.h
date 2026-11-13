/* * Licensed to the Apache Software Foundation (ASF) under one
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
 * \file runtime/contrib/tensorrt/tensorrt_ops.h
 * \brief Converters from ops into TensorRT layers. Converters should
 * inherit from TensorRTOpConverter and implement the Convert() method.
 */

#ifndef TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_OPS_H_
#define TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_OPS_H_

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../json/json_node.h"
#include "NvInfer.h"
#include "tensorrt_utils.h"

// TensorRT 10 removed implicit-batch mode; every network is explicit-batch. Keep the macro so the
// converters' batch-aware branches read clearly, but it is unconditionally false (and no longer
// calls the deprecated INetworkDefinition::hasImplicitBatchDimension()).
#define TRT_HAS_IMPLICIT_BATCH(params) (false)

namespace tvm {
namespace runtime {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

/*!
 * \brief An input to a op may be either kTensor in the case of nvinfer::ITensor*,
 * a kWeight for nvinfer1::Weights, or ignored (eg for the nn.pad value).
 */
enum TensorRTInputType { kTensor, kWeight, kIgnored };

/*!
 * \brief An input to a TensorRTOpConverter. The type of the input is either kTensor
 * or kWeight. For kTensor, "tensor" contains the input tensor. For kWeight,
 * "weight" contains the input weight and "weight_shape" contains the shape.
 */
struct TensorRTOpInput {
  /*! \brief If type is kTensor, will store input tensor. */
  nvinfer1::ITensor* tensor;

  /*! \brief If type is kWeight, will store input weight. */
  nvinfer1::Weights weight;

  /*! \brief Whether the input is in tensor or weight. */
  TensorRTInputType type;

  /*! \brief If type is kWeight, will store weight shape. */
  std::vector<int> weight_shape;

  explicit TensorRTOpInput(nvinfer1::ITensor* tensor)
      : tensor(tensor), weight({tensor->getType(), nullptr, 0}), type(kTensor) {}
  TensorRTOpInput(nvinfer1::Weights weight, const std::vector<int>& shape)
      : tensor(nullptr), weight(weight), type(kWeight), weight_shape(shape) {}
};

/*! \brief Parameters to convert an Op to TensorRT. */
struct TensorRTOpConverterParams {
  /*! \brief The TRT network that the new layer should be added to. */
  nvinfer1::INetworkDefinition* network;
  /*! \brief Index of JSON node. */
  int nid;
  /*! \brief The corresponding JSON node. */
  const JSONGraphNode& node;
  /*! \brief The type of op. */
  std::string op_name;
  /*! \brief Inputs to the op. */
  std::vector<TensorRTOpInput> inputs;
  /*! \brief Outputs of the op should be populated here during Convert(). */
  std::vector<nvinfer1::ITensor*> outputs;
  /*! \brief Any newly allocated weights should be stored here also. */
  std::vector<nvinfer1::Weights>* trt_weights;

  TensorRTOpConverterParams(nvinfer1::INetworkDefinition* network, int nid,
                            const JSONGraphNode& node, std::vector<nvinfer1::Weights>* trt_weights)
      : network(network), nid(nid), node(node), trt_weights(trt_weights) {
    op_name = node.GetOpName();
  }

  std::string LayerName() const { return op_name + "(" + std::to_string(nid) + ")"; }
};

/*! \brief Base class for an op converter from Operator to TRT. */
class TensorRTOpConverter {
 public:
  virtual ~TensorRTOpConverter() = default;

  /*! \brief Operator name. */
  std::string op_name;
  /*! \brief Used to specify whether each input is tensor or weight. */
  const std::vector<TensorRTInputType> input_types;
  /*! \brief If set to true, any number of tensor inputs can be used for the op. */
  const bool variable_input_count;

  /*!
   * \brief Converter subclasses should call this constructor to set
   * input_types or variable_input_count.
   * \param input_types For each input to the op, there should be a
   * corresponding entry in input_types to determine whether that input should
   * be a tensor or a weight. TensorRTBuilder will prepare inputs in
   * TensorRTOpConverter according to this.
   * \param variable_input_count If the op can have multiple inputs, set this to
   * true. input_types vector will be ignored and any number of input tensors
   * can be used for this op. All inputs will be tensors and not weights.
   */
  TensorRTOpConverter(std::string op_name, const std::vector<TensorRTInputType>& input_types,
                      bool variable_input_count = false);

  /*!
   * \brief Convert to TRT. Implementation should use inputs and attributes
   * from the CallObj to add the corresponding TRT layers to network. Outputs
   * should be pushed to outputs vector.
   * \param params Parameters for this op.
   */
  virtual void Convert(TensorRTOpConverterParams* params) const = 0;

  /*!
   * \brief Helper function to reshape a tensor.
   * \param params Parameters for this op.
   * \param input Tensor to reshape.
   * \param new_shape New shape, does not include batch dim.
   * \return Reshaped tensor
   */
  nvinfer1::ITensor* Reshape(TensorRTOpConverterParams* params, nvinfer1::ITensor* input,
                             const std::vector<int>& new_shape) const;

  /*!
   * \brief Helper function to transpose a tensor.
   * \param params Parameters for this op.
   * \param input Tensor to transpose.
   * \param order New order of axes, does include batch dim.
   * \return Transposed tensor
   */
  nvinfer1::ITensor* Transpose(TensorRTOpConverterParams* params, nvinfer1::ITensor* input,
                               const std::vector<int>& order) const;

  /*!
   * \brief Helper function to convert an axis to TRT format.
   * \param axis Axis from TVM.
   * \param input_rank Rank of input, does not include batch dim.
   * \return Axis in TRT format.
   */
  int ConvertAxis(TensorRTOpConverterParams* params, int axis, int input_rank) const;

  /*!
   * \brief Create constant that is broadcastable.
   * \param params Parameters for this op.
   * \param value Value of scalar.
   * \param broadcast_to_dims Dims that scalar should be broadcastable against.
   * \return Constant tensor.
   */
  nvinfer1::ITensor* CreateScalar(TensorRTOpConverterParams* params, float value,
                                  const nvinfer1::Dims& broadcast_to_dims) const;

  /*!
   * \brief Get pre/post padding values from padding attributes array.
   * \param padding Serialized padding from op attributes.
   * \param padding_is_asymmetric True if both pre and post are needed for asymmetric padding.
   * \param prepadding Prepadding value or symmetric padding values if !padding_is_asymmetric.
   * \param postpadding Postpadding value if padding_is_asymmetric.
   */
  void GetPadding(const ffi::Array<int64_t>& padding, bool* use_asymmetric_padding,
                  nvinfer1::DimsHW* prepadding, nvinfer1::DimsHW* postpadding) const;

  /*!
   * \brief Get pre/post padding values from padding attributes array for volumetric ops.
   * \param padding Serialized padding from op attributes.
   * \param padding_is_asymmetric True if both pre and post are needed for asymmetric padding.
   * \param prepadding Prepadding value or symmetric padding values if !padding_is_asymmetric.
   * \param postpadding Postpadding value if padding_is_asymmetric.
   */
  void GetPadding3D(const ffi::Array<int64_t>& padding, bool* use_asymmetric_padding,
                    nvinfer1::Dims* prepadding, nvinfer1::Dims* postpadding) const;
};

/*!
 * \brief Get the map of available TensorRTOpConverters, where the key is the name of the op.
 * \return Map of TensorRTOpConverters.
 */
const std::unordered_map<std::string, std::unique_ptr<TensorRTOpConverter>>& GetOpConverters();

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_OPS_H_
