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
 * \file Use standard C library call.
 */

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <vector>

#include "builtin_fp16.h"

namespace tvm {
namespace contrib {

template <typename DType, bool stable_comparison = false>
bool CompareAscend(const std::pair<int64_t, DType>& lhs, const std::pair<int64_t, DType>& rhs) {
  if constexpr (stable_comparison) {
    if (lhs.second == rhs.second) {
      return lhs.first < rhs.first;
    }
  }

  return lhs.second < rhs.second;
}

template <typename DType, bool stable_comparison = false>
bool CompareDescend(const std::pair<int64_t, DType>& lhs, const std::pair<int64_t, DType>& rhs) {
  if constexpr (stable_comparison) {
    if (lhs.second == rhs.second) {
      return lhs.first < rhs.first;
    }
  }

  return lhs.second > rhs.second;
}

struct float16 {
  uint16_t bits;
  float to_float() const {
    return __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(bits);
  }

  inline bool operator==(const float16& rhs) const { return to_float() == rhs.to_float(); }
  inline bool operator!=(const float16& rhs) const { return to_float() != rhs.to_float(); }
  inline bool operator<(const float16& rhs) const { return to_float() < rhs.to_float(); }
  inline bool operator>(const float16& rhs) const { return to_float() > rhs.to_float(); }
  inline bool operator<=(const float16& rhs) const { return to_float() <= rhs.to_float(); }
  inline bool operator>=(const float16& rhs) const { return to_float() >= rhs.to_float(); }
};

// Argsort implemented C library sort for nms.
// Return indices of sorted tensor.
// By default, the last axis will be used to sort.
// sort_num specify the number of elements to be sorted.
// If input tensor has dimension (d0, d1, ..., d(k-1), dk, d(k+1), ..., d(n-1))
// and sort axis is dk. sort_num should have dimension of
// (d1, d2, ..., d(k-1), d(k+1), ..., dn).
void RegisterArgsortNMS() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      "tvm.contrib.sort.argsort_nms", [](ffi::PackedArgs args, ffi::Any* ret) {
        auto input = args[0].cast<DLTensor*>();
        auto sort_num = args[1].cast<DLTensor*>();
        auto output = args[2].cast<DLTensor*>();
        int32_t axis = args[3].cast<int>();
        bool is_ascend = args[4].cast<bool>();

        auto dtype = input->dtype;
        auto data_ptr = static_cast<float*>(input->data);
        auto sort_num_ptr = static_cast<int32_t*>(sort_num->data);
        std::vector<std::pair<int32_t, float>> sorter;
        int64_t axis_mul_before = 1;
        int64_t axis_mul_after = 1;

        if (axis < 0) {
          axis = input->ndim + axis;
        }

        // Currently only supports input dtype to be float32.
        TVM_FFI_ICHECK_EQ(dtype.code, 2) << "Currently only supports input dtype "
                                            "to be float.";
#if (__ARM_FEATURE_FP16_SCALAR_ARITHMETIC != 1)
        TVM_FFI_ICHECK_EQ(dtype.bits, 32) << "Currently only supports input dtype "
                                             "to be float32.";
#endif
        TVM_FFI_ICHECK_LT(axis, input->ndim) << "Axis out of boundary for "
                                                "input ndim "
                                             << input->ndim;

        for (int i = 0; i < input->ndim; ++i) {
          if (i < axis) {
            axis_mul_before *= input->shape[i];
          } else if (i > axis) {
            axis_mul_after *= input->shape[i];
          }
        }

        for (int64_t i = 0; i < axis_mul_before; ++i) {
          for (int64_t j = 0; j < axis_mul_after; ++j) {
            sorter.clear();
            int32_t current_sort_num = *(sort_num_ptr + i * axis_mul_after + j);
            int64_t base_idx = i * input->shape[axis] * axis_mul_after + j;
            for (int64_t k = 0; k < current_sort_num; ++k) {
              int64_t full_idx = base_idx + k * axis_mul_after;
              sorter.emplace_back(std::make_pair(k, *(data_ptr + full_idx)));
            }
            if (is_ascend) {
#if (__ARM_FEATURE_FP16_SCALAR_ARITHMETIC == 1)
              if (dtype.bits == 16) {
                std::stable_sort(sorter.begin(), sorter.end(), CompareAscend<__fp16>);
              } else {
#endif
                std::stable_sort(sorter.begin(), sorter.end(), CompareAscend<float>);
#if (__ARM_FEATURE_FP16_SCALAR_ARITHMETIC == 1)
              }
#endif
            } else {
#if (__ARM_FEATURE_FP16_SCALAR_ARITHMETIC == 1)
              if (dtype.bits == 16) {
                std::stable_sort(sorter.begin(), sorter.end(), CompareDescend<__fp16>);
              } else {
#endif
                std::stable_sort(sorter.begin(), sorter.end(), CompareDescend<float>);
#if (__ARM_FEATURE_FP16_SCALAR_ARITHMETIC == 1)
              }
#endif
            }
            for (int32_t k = 0; k < input->shape[axis]; ++k) {
              *(static_cast<int32_t*>(output->data) + base_idx + k * axis_mul_after) =
                  k < static_cast<int32_t>(sorter.size()) ? sorter[k].first : k;
            }
          }
        }
      });
}

template <typename DataType, typename OutType>
void sort_impl(
    DLTensor* input, DLTensor* output, int32_t axis, bool is_ascend,
    std::function<void(OutType*, size_t, const std::pair<int64_t, DataType>&)> epilogue) {
  auto data_ptr = static_cast<DataType*>(input->data);
  auto out_ptr = static_cast<OutType*>(output->data);
  std::vector<std::pair<int64_t, DataType>> sorter;

  int axis_mul_before = 1;
  int axis_mul_after = 1;
  for (int i = 0; i < input->ndim; ++i) {
    if (i < axis) {
      axis_mul_before *= input->shape[i];
    } else if (i > axis) {
      axis_mul_after *= input->shape[i];
    }
  }

  for (int i = 0; i < axis_mul_before; ++i) {
    for (int j = 0; j < axis_mul_after; ++j) {
      sorter.clear();
      int64_t base_idx = i * input->shape[axis] * axis_mul_after + j;
      for (int64_t k = 0; k < input->shape[axis]; ++k) {
        int64_t full_idx = base_idx + k * axis_mul_after;
        sorter.emplace_back(std::make_pair(k, data_ptr[full_idx]));
      }
      if (is_ascend) {
        std::stable_sort(sorter.begin(), sorter.end(), CompareAscend<DataType>);
      } else {
        std::stable_sort(sorter.begin(), sorter.end(), CompareDescend<DataType>);
      }
      for (int64_t k = 0; k < input->shape[axis]; ++k) {
        epilogue(out_ptr, base_idx + k * axis_mul_after, sorter[k]);
      }
    }
  }
}

template <typename DataType, typename OutType>
void argsort(DLTensor* input, DLTensor* output, int32_t axis, bool is_ascend) {
  return sort_impl<DataType, OutType>(
      input, output, axis, is_ascend,
      [](OutType* out_ptr, size_t index, const std::pair<int64_t, DataType>& sort_pair) {
        out_ptr[index] = static_cast<OutType>(sort_pair.first);
      });
}

template <typename DataType>
void sort(DLTensor* input, DLTensor* output, int32_t axis, bool is_ascend) {
  return sort_impl<DataType, DataType>(
      input, output, axis, is_ascend,
      [](DataType* out_ptr, size_t index, const std::pair<int64_t, DataType>& sort_pair) {
        out_ptr[index] = sort_pair.second;
      });
}

// Argsort implemented C library sort.
// Return indices of sorted tensor.
// By default, the last axis will be used to sort.
// sort_num specify the number of elements to be sorted.
// If input tensor has dimension (d0, d1, ..., d(k-1), dk, d(k+1), ..., d(n-1))
// and sort axis is dk. sort_num should have dimension of
// (d1, d2, ..., d(k-1), d(k+1), ..., dn).
void RegisterArgsort() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tvm.contrib.sort.argsort", [](ffi::PackedArgs args, ffi::Any* ret) {
    auto input = args[0].cast<DLTensor*>();
    auto output = args[1].cast<DLTensor*>();
    int32_t axis = args[2].cast<int32_t>();
    bool is_ascend = args[3].cast<bool>();
    if (axis < 0) {
      axis = input->ndim + axis;
    }
    TVM_FFI_ICHECK_LT(axis, input->ndim) << "Axis out of boundary for "
                                            "input ndim "
                                         << input->ndim;

    auto data_dtype = ffi::DLDataTypeToString(input->dtype);
    auto out_dtype = ffi::DLDataTypeToString(output->dtype);

    if (data_dtype == "float32") {
      if (out_dtype == "int32") {
        argsort<float, int32_t>(input, output, axis, is_ascend);
      } else if (out_dtype == "int64") {
        argsort<float, int64_t>(input, output, axis, is_ascend);
      } else if (out_dtype == "float32") {
        argsort<float, float>(input, output, axis, is_ascend);
      } else if (out_dtype == "float64") {
        argsort<float, double>(input, output, axis, is_ascend);
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported output dtype: " << out_dtype;
      }
    } else if (data_dtype == "float64") {
      if (out_dtype == "int32") {
        argsort<double, int32_t>(input, output, axis, is_ascend);
      } else if (out_dtype == "int64") {
        argsort<double, int64_t>(input, output, axis, is_ascend);
      } else if (out_dtype == "float32") {
        argsort<double, float>(input, output, axis, is_ascend);
      } else if (out_dtype == "float64") {
        argsort<double, double>(input, output, axis, is_ascend);
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported output dtype: " << out_dtype;
      }
#if (__ARM_FEATURE_FP16_SCALAR_ARITHMETIC == 1)
    } else if (data_dtype == "float16") {
      if (out_dtype == "float16") {
        argsort<__fp16, __fp16>(input, output, axis, is_ascend);
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported output dtype: " << out_dtype;
      }
#endif
    } else if (data_dtype == "int32") {
      if (out_dtype == "int32") {
        argsort<int32_t, int32_t>(input, output, axis, is_ascend);
      } else if (out_dtype == "int64") {
        argsort<int32_t, int64_t>(input, output, axis, is_ascend);
      } else if (out_dtype == "float32") {
        argsort<int32_t, float>(input, output, axis, is_ascend);
      } else if (out_dtype == "float64") {
        argsort<int32_t, double>(input, output, axis, is_ascend);
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported output dtype: " << out_dtype;
      }
    } else if (data_dtype == "int64") {
      if (out_dtype == "int32") {
        argsort<int64_t, int32_t>(input, output, axis, is_ascend);
      } else if (out_dtype == "int64") {
        argsort<int64_t, int64_t>(input, output, axis, is_ascend);
      } else if (out_dtype == "float32") {
        argsort<int64_t, float>(input, output, axis, is_ascend);
      } else if (out_dtype == "float64") {
        argsort<int64_t, double>(input, output, axis, is_ascend);
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported output dtype: " << out_dtype;
      }
    } else if (data_dtype == "float16") {
      if (out_dtype == "int32") {
        argsort<float16, int32_t>(input, output, axis, is_ascend);
      } else if (out_dtype == "int64") {
        argsort<float16, int64_t>(input, output, axis, is_ascend);
      } else if (out_dtype == "float32") {
        argsort<float16, float>(input, output, axis, is_ascend);
      } else if (out_dtype == "float64") {
        argsort<float16, double>(input, output, axis, is_ascend);
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported output dtype: " << out_dtype;
      }
    } else {
      TVM_FFI_THROW(InternalError) << "Unsupported input dtype: " << data_dtype;
    }
  });
}

// Sort implemented C library sort.
// Return  sorted tensor.
// By default, the last axis will be used to sort.
// sort_num specify the number of elements to be sorted.
// If input tensor has dimension (d0, d1, ..., d(k-1), dk, d(k+1), ..., d(n-1))
// and sort axis is dk. sort_num should have dimension of
// (d1, d2, ..., d(k-1), d(k+1), ..., dn).
void RegisterSort() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tvm.contrib.sort.sort", [](ffi::PackedArgs args, ffi::Any* ret) {
    auto input = args[0].cast<DLTensor*>();
    auto output = args[1].cast<DLTensor*>();
    int32_t axis = args[2].cast<int32_t>();
    bool is_ascend = args[3].cast<bool>();
    if (axis < 0) {
      axis = input->ndim + axis;
    }
    TVM_FFI_ICHECK_LT(axis, input->ndim) << "Axis out of boundary for "
                                            "input ndim "
                                         << input->ndim;

    auto data_dtype = ffi::DLDataTypeToString(input->dtype);
    auto out_dtype = ffi::DLDataTypeToString(output->dtype);

    TVM_FFI_ICHECK_EQ(data_dtype, out_dtype);

    if (data_dtype == "float32") {
      sort<float>(input, output, axis, is_ascend);
    } else if (data_dtype == "float64") {
      sort<double>(input, output, axis, is_ascend);
#if (__ARM_FEATURE_FP16_SCALAR_ARITHMETIC == 1)
    } else if (data_dtype == "float16") {
      sort<__fp16>(input, output, axis, is_ascend);
#endif
    } else if (data_dtype == "int32") {
      sort<int32_t>(input, output, axis, is_ascend);
    } else if (data_dtype == "int64") {
      sort<int64_t>(input, output, axis, is_ascend);
    } else if (data_dtype == "float16") {
      sort<float16>(input, output, axis, is_ascend);
    } else {
      TVM_FFI_THROW(InternalError) << "Unsupported input dtype: " << data_dtype;
    }
  });
}

template <typename DataType, typename IndicesType>
void topk(DLTensor* input, DLTensor* out_values, DLTensor* out_indices, int k, int axis,
          bool is_ascend) {
  DataType* data_ptr = static_cast<DataType*>(input->data);
  DataType* values_ptr =
      (out_values == nullptr) ? nullptr : static_cast<DataType*>(out_values->data);
  IndicesType* indices_ptr =
      (out_indices == nullptr) ? nullptr : static_cast<IndicesType*>(out_indices->data);

  // Maintain a min/max containing the top-k elements
  std::vector<std::pair<int64_t, DataType>> running_heap;

  // Need +1 when inserting new element before maintaining heap invariant
  running_heap.reserve(k + 1);

  int axis_mul_before = 1;
  int axis_mul_after = 1;
  for (int i = 0; i < input->ndim; ++i) {
    if (i < axis) {
      axis_mul_before *= input->shape[i];
    } else if (i > axis) {
      axis_mul_after *= input->shape[i];
    }
  }
  if (k < 1) {
    k = input->shape[axis];
  }

  for (int i = 0; i < axis_mul_before; ++i) {
    for (int j = 0; j < axis_mul_after; ++j) {
      running_heap.clear();
      int64_t src_base_idx = i * input->shape[axis] * axis_mul_after + j;
      int64_t dst_base_idx = i * k * axis_mul_after + j;

      // Start by creating min/max heap with fixed-k elements
      int cur_axis_index = 0;
      for (; cur_axis_index < k && cur_axis_index < input->shape[axis]; cur_axis_index++) {
        int64_t full_idx = src_base_idx + cur_axis_index * axis_mul_after;
        running_heap.emplace_back(std::make_pair(cur_axis_index, data_ptr[full_idx]));
      }
      if (!is_ascend) {
        std::make_heap(running_heap.begin(), running_heap.end(), CompareDescend<DataType, true>);
      } else {
        std::make_heap(running_heap.begin(), running_heap.end(), CompareAscend<DataType, true>);
      }

      // Iterate through all elements, adding to heap along the way
      for (; cur_axis_index < input->shape[axis]; cur_axis_index++) {
        int64_t full_idx = src_base_idx + cur_axis_index * axis_mul_after;
        std::pair<int64_t, DataType> cur_val = {cur_axis_index, data_ptr[full_idx]};

        // Eq. to cur_val.second > running_heap.second
        if (!is_ascend && CompareDescend<DataType, true>(cur_val, running_heap[0])) {
          running_heap.push_back(cur_val);
          std::push_heap(running_heap.begin(), running_heap.end(), CompareDescend<DataType, true>);
          std::pop_heap(running_heap.begin(), running_heap.end(), CompareDescend<DataType, true>);
          running_heap.pop_back();
        } else if (is_ascend && CompareAscend<DataType, true>(cur_val, running_heap[0])) {
          running_heap.push_back(cur_val);
          std::push_heap(running_heap.begin(), running_heap.end(), CompareAscend<DataType, true>);
          std::pop_heap(running_heap.begin(), running_heap.end(), CompareAscend<DataType, true>);
          running_heap.pop_back();
        }
      }

      // finally sort heap and deliver results
      if (is_ascend) {
        std::stable_sort(running_heap.begin(), running_heap.end(), CompareAscend<DataType, true>);
      } else {
        std::stable_sort(running_heap.begin(), running_heap.end(), CompareDescend<DataType, true>);
      }

      for (uint32_t kk = 0; kk < running_heap.size(); ++kk) {
        if (indices_ptr != nullptr) {
          indices_ptr[dst_base_idx + kk * axis_mul_after] =
              static_cast<IndicesType>(running_heap[kk].first);
        }
        if (values_ptr != nullptr) {
          values_ptr[dst_base_idx + kk * axis_mul_after] =
              static_cast<DataType>(running_heap[kk].second);
        }
      }
    }
  }
}

// Argsort implemented C library sort.
// Return indices of sorted tensor.
// By default, the last axis will be used to sort.
// sort_num specify the number of elements to be sorted.
// If input tensor has dimension (d0, d1, ..., d(k-1), dk, d(k+1), ..., d(n-1))
// and sort axis is dk. sort_num should have dimension of
// (d1, d2, ..., d(k-1), d(k+1), ..., dn).
void RegisterTopk() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tvm.contrib.sort.topk", [](ffi::PackedArgs args, ffi::Any* ret) {
    auto input = args[0].cast<DLTensor*>();
    DLTensor* values_out = nullptr;
    DLTensor* indices_out = nullptr;
    int k = args[args.size() - 4].cast<int>();
    int axis = args[args.size() - 3].cast<int>();
    std::string ret_type = args[args.size() - 2].cast<std::string>();
    bool is_ascend = args[args.size() - 1].cast<bool>();
    if (ret_type == "both") {
      values_out = args[1].cast<DLTensor*>();
      indices_out = args[2].cast<DLTensor*>();
    } else if (ret_type == "values") {
      values_out = args[1].cast<DLTensor*>();
    } else if (ret_type == "indices") {
      indices_out = args[1].cast<DLTensor*>();
    } else {
      TVM_FFI_THROW(InternalError) << "Unsupported ret type: " << ret_type;
    }
    if (axis < 0) {
      axis = input->ndim + axis;
    }
    TVM_FFI_ICHECK(axis >= 0 && axis < input->ndim)
        << "Axis out of boundary for input ndim " << input->ndim;

    auto data_dtype = ffi::DLDataTypeToString(input->dtype);
    auto out_dtype =
        (indices_out == nullptr) ? "int64" : ffi::DLDataTypeToString(indices_out->dtype);

    if (data_dtype == "float32") {
      if (out_dtype == "int32") {
        topk<float, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "int64") {
        topk<float, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "float32") {
        topk<float, float>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "float64") {
        topk<float, double>(input, values_out, indices_out, k, axis, is_ascend);
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported output dtype: " << out_dtype;
      }
    } else if (data_dtype == "float64") {
      if (out_dtype == "int32") {
        topk<double, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "int64") {
        topk<double, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "float32") {
        topk<double, float>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "float64") {
        topk<double, double>(input, values_out, indices_out, k, axis, is_ascend);
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported output dtype: " << out_dtype;
      }
    } else if (data_dtype == "uint8") {
      if (out_dtype == "uint8") {
        topk<uint8_t, uint8_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "int32") {
        topk<uint8_t, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "int64") {
        topk<uint8_t, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "float32") {
        topk<uint8_t, float>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "float64") {
        topk<uint8_t, double>(input, values_out, indices_out, k, axis, is_ascend);
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported output dtype: " << out_dtype;
      }
    } else if (data_dtype == "int8") {
      if (out_dtype == "int8") {
        topk<int8_t, int8_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "int32") {
        topk<int8_t, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "int64") {
        topk<int8_t, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "float32") {
        topk<int8_t, float>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "float64") {
        topk<int8_t, double>(input, values_out, indices_out, k, axis, is_ascend);
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported output dtype: " << out_dtype;
      }
    } else if (data_dtype == "int32") {
      if (out_dtype == "int32") {
        topk<int32_t, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "int64") {
        topk<int32_t, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "float32") {
        topk<int32_t, float>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "float64") {
        topk<int32_t, double>(input, values_out, indices_out, k, axis, is_ascend);
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported output dtype: " << out_dtype;
      }
    } else if (data_dtype == "int64") {
      if (out_dtype == "int32") {
        topk<int64_t, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "int64") {
        topk<int64_t, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "float32") {
        topk<int64_t, float>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "float64") {
        topk<int64_t, double>(input, values_out, indices_out, k, axis, is_ascend);
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported output dtype: " << out_dtype;
      }
    } else if (data_dtype == "float16") {
      if (out_dtype == "int32") {
        topk<float16, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "int64") {
        topk<float16, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "float32") {
        topk<float16, float>(input, values_out, indices_out, k, axis, is_ascend);
      } else if (out_dtype == "float64") {
        topk<float16, double>(input, values_out, indices_out, k, axis, is_ascend);
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported output dtype: " << out_dtype;
      }
    } else {
      TVM_FFI_THROW(InternalError) << "Unsupported input dtype: " << data_dtype;
    }
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterArgsortNMS();
  RegisterArgsort();
  RegisterSort();
  RegisterTopk();
}

}  // namespace contrib
}  // namespace tvm
