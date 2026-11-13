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

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>

#include "cutlass_kernels/cutlass_preprocessors.h"

namespace tvm {
namespace runtime {

// This packed function applies the set of preprocessings on the weight, which are required by
// the FT kernel. They consist of permuting / transposing / interleaving the weight elements,
// and changing the weight dtype to be unsigned by adding a bias. The output has the same size
// as the input.
//
// These processes are not well documented, so we wrap them into a packed function and use it as a
// black box.
//
// The preprocessing functions are defined in C++, so we need to copy the input weight to CPU.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "cutlass.ft_preprocess_weight", [](ffi::Tensor packed_weight, int sm, bool is_int4) {
        TVM_FFI_ICHECK(packed_weight.defined());
        TVM_FFI_ICHECK(packed_weight.IsContiguous());
        TVM_FFI_ICHECK(packed_weight.numel() == 0 || packed_weight.IsAligned(kAllocAlignment));
        bool is_2d = packed_weight.ndim() == 2;
        int num_experts = is_2d ? 1 : packed_weight.shape()[0];
        int rows = packed_weight.shape()[is_2d ? 0 : 1];
        int cols = packed_weight.shape()[is_2d ? 1 : 2];

        std::vector<int8_t> input_cpu(num_experts * rows * cols);
        std::vector<int8_t> output_cpu(num_experts * rows * cols);
        tvm::runtime::CopyToBytes(packed_weight, input_cpu.data(), input_cpu.size());
        // multiply cols by 2 since the "col" params in preprocess_weights refers to the column of
        // the unpacked weight.
        if (is_int4) {
          cols *= 2;
        }
        fastertransformer::preprocess_weights(output_cpu.data(), input_cpu.data(), num_experts,
                                              rows, cols, is_int4, sm);
        auto out = tvm::runtime::Empty(packed_weight.shape(), packed_weight.dtype(),
                                       packed_weight.device());
        tvm::runtime::CopyFromBytes(out, output_cpu.data(), output_cpu.size());
        return out;
      });
}

}  // namespace runtime
}  // namespace tvm
