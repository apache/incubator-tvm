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
 * \brief Registration of NN operators
 * \file nn.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/topi/nn.h>
#include <tvm/topi/nn/bias_add.h>
#include <tvm/topi/nn/bnn.h>
#include <tvm/topi/nn/dense.h>
#include <tvm/topi/nn/dilate.h>
#include <tvm/topi/nn/flatten.h>
#include <tvm/topi/nn/group_norm.h>
#include <tvm/topi/nn/instance_norm.h>
#include <tvm/topi/nn/layer_norm.h>
#include <tvm/topi/nn/local_response_norm.h>
#include <tvm/topi/nn/mapping.h>
#include <tvm/topi/nn/pooling.h>
#include <tvm/topi/nn/rms_norm.h>
#include <tvm/topi/nn/softmax.h>

namespace tvm {
namespace topi {

using namespace tvm;
using namespace tvm::runtime;

/* Ops from nn.h */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed(
          "topi.nn.relu",
          [](ffi::PackedArgs args, ffi::Any* rv) { *rv = relu<float>(args[0].cast<te::Tensor>()); })
      .def_packed("topi.nn.leaky_relu",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = leaky_relu(args[0].cast<te::Tensor>(), args[1].cast<double>());
                  })
      .def_packed("topi.nn.prelu",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = prelu(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                                args[2].cast<int>());
                  })
      .def_packed("topi.nn.pad",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    auto pad_value_tensor = args[3].try_cast<te::Tensor>();
                    PrimExpr pad_value = pad_value_tensor
                                             ? pad_value_tensor.value()(ffi::Array<PrimExpr>{})
                                             : args[3].cast<PrimExpr>();
                    *rv = pad(args[0].cast<te::Tensor>(), args[1].cast<ffi::Array<PrimExpr>>(),
                              args[2].cast<ffi::Array<PrimExpr>>(), pad_value);
                  })
      .def_packed(
          "topi.nn.space_to_batch_nd",
          [](ffi::PackedArgs args, ffi::Any* rv) {
            auto pad_value_tensor = args[4].try_cast<te::Tensor>();
            PrimExpr pad_value = pad_value_tensor ? pad_value_tensor.value()(ffi::Array<PrimExpr>{})
                                                  : args[4].cast<PrimExpr>();
            *rv = space_to_batch_nd(args[0].cast<te::Tensor>(), args[1].cast<ffi::Array<int64_t>>(),
                                    args[2].cast<ffi::Array<PrimExpr>>(),
                                    args[3].cast<ffi::Array<PrimExpr>>(), pad_value);
          })
      .def_packed("topi.nn.batch_to_space_nd",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = batch_to_space_nd(
                        args[0].cast<te::Tensor>(), args[1].cast<ffi::Array<int64_t>>(),
                        args[2].cast<ffi::Array<PrimExpr>>(), args[3].cast<ffi::Array<PrimExpr>>(),
                        args[4].cast<std::string>());
                  })
      .def_packed("topi.nn.nll_loss", [](ffi::PackedArgs args, ffi::Any* rv) {
        *rv =
            nll_loss(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                     args[2].cast<te::Tensor>(), args[3].cast<std::string>(), args[4].cast<int>());
      });
}

/* Ops from nn/dense.h */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("topi.nn.dense", [](ffi::PackedArgs args, ffi::Any* rv) {
    *rv = nn::dense(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                    args[2].cast<ffi::Optional<te::Tensor>>(), args[3].cast<PrimType>());
  });
}

/* Ops from nn/bias_add.h */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("topi.nn.bias_add", [](ffi::PackedArgs args, ffi::Any* rv) {
    *rv = nn::bias_add(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(), args[2].cast<int>());
  });
}

/* Ops from nn/dilate.h */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("topi.nn.dilate", [](ffi::PackedArgs args, ffi::Any* rv) {
    *rv = nn::dilate(args[0].cast<te::Tensor>(), args[1].cast<ffi::Array<PrimExpr>>(),
                     args[2].cast<double>());
  });
}

/* Ops from nn/flatten.h */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("topi.nn.flatten", [](ffi::PackedArgs args, ffi::Any* rv) {
    *rv = nn::flatten(args[0].cast<te::Tensor>());
  });
}

/* Ops from nn/mapping.h */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("topi.nn.scale_shift_nchw",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv =
                        nn::scale_shift_nchw(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                                             args[2].cast<te::Tensor>());
                  })
      .def_packed("topi.nn.scale_shift_nhwc", [](ffi::PackedArgs args, ffi::Any* rv) {
        *rv = nn::scale_shift_nhwc(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                                   args[2].cast<te::Tensor>());
      });
}

/* Ops from nn/pooling.h */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("topi.nn.pool_grad",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = nn::pool_grad(
                        args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                        args[2].cast<ffi::Array<PrimExpr>>(), args[3].cast<ffi::Array<PrimExpr>>(),
                        args[4].cast<ffi::Array<PrimExpr>>(),
                        static_cast<nn::PoolType>(args[5].cast<int>()), args[6].cast<bool>(),
                        args[7].cast<std::string>(), args[8].cast<bool>());
                  })
      .def_packed("topi.nn.global_pool",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = nn::global_pool(args[0].cast<te::Tensor>(),
                                          static_cast<nn::PoolType>(args[1].cast<int>()),
                                          args[2].cast<std::string>());
                  })
      .def_packed("topi.nn.adaptive_pool1d",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = nn::adaptive_pool1d(args[0].cast<te::Tensor>(),
                                              args[1].cast<ffi::Array<PrimExpr>>(),
                                              static_cast<nn::PoolType>(args[2].cast<int>()),
                                              args[3].cast<std::string>());
                  })
      .def_packed("topi.nn.adaptive_pool",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = nn::adaptive_pool(args[0].cast<te::Tensor>(),
                                            args[1].cast<ffi::Array<PrimExpr>>(),
                                            static_cast<nn::PoolType>(args[2].cast<int>()),
                                            args[3].cast<std::string>());
                  })
      .def_packed("topi.nn.adaptive_pool3d",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = nn::adaptive_pool3d(args[0].cast<te::Tensor>(),
                                              args[1].cast<ffi::Array<PrimExpr>>(),
                                              static_cast<nn::PoolType>(args[2].cast<int>()),
                                              args[3].cast<std::string>());
                  })
      .def_packed("topi.nn.pool1d",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = nn::pool1d(
                        args[0].cast<te::Tensor>(), args[1].cast<ffi::Array<PrimExpr>>(),
                        args[2].cast<ffi::Array<PrimExpr>>(), args[3].cast<ffi::Array<PrimExpr>>(),
                        args[4].cast<ffi::Array<PrimExpr>>(),
                        static_cast<nn::PoolType>(args[5].cast<int>()), args[6].cast<bool>(),
                        args[7].cast<std::string>(), args[8].cast<bool>());
                  })
      .def_packed("topi.nn.pool2d",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = nn::pool2d(
                        args[0].cast<te::Tensor>(), args[1].cast<ffi::Array<PrimExpr>>(),
                        args[2].cast<ffi::Array<PrimExpr>>(), args[3].cast<ffi::Array<PrimExpr>>(),
                        args[4].cast<ffi::Array<PrimExpr>>(),
                        static_cast<nn::PoolType>(args[5].cast<int>()), args[6].cast<bool>(),
                        args[7].cast<std::string>(), args[8].cast<bool>());
                  })
      .def_packed("topi.nn.pool3d", [](ffi::PackedArgs args, ffi::Any* rv) {
        *rv = nn::pool3d(args[0].cast<te::Tensor>(), args[1].cast<ffi::Array<PrimExpr>>(),
                         args[2].cast<ffi::Array<PrimExpr>>(), args[3].cast<ffi::Array<PrimExpr>>(),
                         args[4].cast<ffi::Array<PrimExpr>>(),
                         static_cast<nn::PoolType>(args[5].cast<int>()), args[6].cast<bool>(),
                         args[7].cast<std::string>(), args[8].cast<bool>());
      });
}

/* Ops from nn/softmax.h */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("topi.nn.softmax",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = nn::softmax(args[0].cast<te::Tensor>(), args[1].cast<int>());
                  })
      .def_packed("topi.nn.log_softmax",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = nn::log_softmax(args[0].cast<te::Tensor>());
                  })
      .def_packed("topi.nn.lrn", [](ffi::PackedArgs args, ffi::Any* rv) {
        *rv = nn::lrn(args[0].cast<te::Tensor>(), args[1].cast<int>(), args[2].cast<int>(),
                      args[3].cast<double>(), args[4].cast<double>(), args[5].cast<double>());
      });
}

/* Ops from nn/bnn.h */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("topi.nn.binarize_pack",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = nn::binarize_pack(args[0].cast<te::Tensor>(), args[1].cast<int>());
                  })
      .def_packed("topi.nn.binary_dense", [](ffi::PackedArgs args, ffi::Any* rv) {
        *rv = nn::binary_dense(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>());
      });
}

/* Ops from nn/layer_norm.h */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("topi.nn.layer_norm", [](ffi::PackedArgs args, ffi::Any* rv) {
    *rv = nn::layer_norm(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                         args[2].cast<ffi::Optional<te::Tensor>>(),
                         args[3].cast<ffi::Array<int64_t>>(), args[4].cast<double>());
  });
}

/* Ops from nn/group_norm.h */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("topi.nn.group_norm", [](ffi::PackedArgs args, ffi::Any* rv) {
    *rv = nn::group_norm(args[0].cast<te::Tensor>(), args[1].cast<ffi::Optional<te::Tensor>>(),
                         args[2].cast<ffi::Optional<te::Tensor>>(), args[3].cast<int>(),
                         args[4].cast<int>(), args[5].cast<ffi::Array<int64_t>>(),
                         args[6].cast<double>());
  });
}

/* Ops from nn/instance_norm.h */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("topi.nn.instance_norm", [](ffi::PackedArgs args, ffi::Any* rv) {
    *rv = nn::instance_norm(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                            args[2].cast<ffi::Optional<te::Tensor>>(), args[3].cast<int>(),
                            args[4].cast<ffi::Array<int64_t>>(), args[5].cast<double>());
  });
}

/* Ops from nn/rms_norm.h */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("topi.nn.rms_norm", [](ffi::PackedArgs args, ffi::Any* rv) {
    *rv = nn::rms_norm(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                       args[2].cast<ffi::Array<int64_t>>(), args[3].cast<double>());
  });
}

}  // namespace topi
}  // namespace tvm
