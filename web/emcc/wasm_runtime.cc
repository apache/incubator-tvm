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

/*
 * \file wasm_runtime.cc
 * \brief TVM wasm runtime library pack.
 */

// configurations for tvm logging
#define TVM_LOG_DEBUG 0
#define TVM_LOG_CUSTOMIZE 1
#define TVM_FFI_USE_LIBBACKTRACE 0
#define TVM_FFI_ALWAYS_LOG_BEFORE_THROW 1

#include <tvm/ffi/any.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <cstring>
#include <limits>

// Keep the amalgamated source initialization order stable.
// clang-format off
#include "src/runtime/extra/contrib/sort/sort.cc"
#include "src/runtime/rpc/rpc_channel.cc"
#include "src/runtime/rpc/rpc_endpoint.cc"
#include "src/runtime/rpc/rpc_event_impl.cc"
#include "src/runtime/rpc/rpc_local_session.cc"
#include "src/runtime/rpc/rpc_module.cc"
#include "src/runtime/rpc/rpc_session.cc"
#include "src/runtime/timer.cc"
#include "src/runtime/vm/params.cc"
#include "3rdparty/tvm-tirx/src/runtime/cpu_device_api.cc"
#include "3rdparty/tvm-tirx/src/runtime/device_api.cc"
#include "3rdparty/tvm-tirx/src/runtime/logging.cc"
#include "3rdparty/tvm-tirx/src/runtime/metadata.cc"
#include "3rdparty/tvm-tirx/src/runtime/tensor.cc"
// relax setup
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/backtrace.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/container.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/custom_allocator.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/dtype.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/error.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/extra/env_c_api.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/extra/env_context.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/extra/json_parser.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/extra/json_writer.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/extra/library_module.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/extra/library_module_system_lib.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/extra/module.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/function.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/init_once.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/object.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/tensor.cc"
#include "3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/testing/testing.cc"
// Enum registration requires the FFI type-attribute registry above.
#include "src/runtime/vm/attn_backend.cc"
#include "src/runtime/vm/builtin.cc"
#include "src/runtime/vm/bytecode.cc"
#include "src/runtime/vm/executable.cc"
#include "src/runtime/vm/kv_state.cc"
#include "src/runtime/vm/lm_support.cc"
#include "src/runtime/vm/paged_kv_cache.cc"
#include "src/runtime/vm/rnn_state.cc"
#include "src/runtime/vm/tensor_cache_support.cc"
#include "src/runtime/vm/vm.cc"
#include "3rdparty/tvm-tirx/src/runtime/memory_scope.cc"
// clang-format on

// --- Implementations of backend and wasm runtime API. ---

int TVMBackendParallelLaunch(FTVMParallelLambda flambda, void* cdata, int num_task) {
  TVMParallelGroupEnv env;
  env.num_task = 1;
  flambda(0, &env, cdata);
  return 0;
}

int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv* penv) { return 0; }

// --- Environment ffi::Functions for testing ---
namespace tvm {
namespace runtime {
namespace detail {
// Override logging mechanism
[[noreturn]] void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
  std::cerr << "[FATAL] " << file << ":" << lineno << ": " << message << std::endl;
  abort();
}

void LogMessageImpl(const std::string& file, int lineno, int level, const std::string& message) {
  static const char* level_strings_[] = {
      "[DEBUG] ",
      "[INFO] ",
      "[WARNING] ",
      "[ERROR] ",
  };
  std::cout << level_strings_[level] << file << ":" << lineno << ": " << message << std::endl;
}

}  // namespace detail

static uint32_t ByteArrayChecksum(const TVMFFIByteArray* bytes) {
  TVM_FFI_ICHECK_NE(bytes, nullptr);
  uint32_t checksum = 2166136261U;
  for (size_t i = 0; i < bytes->size; ++i) {
    checksum ^= static_cast<uint8_t>(bytes->data[i]);
    checksum *= 16777619U;
  }
  return checksum;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("tvmjs.testing.call",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    (args[0].cast<ffi::Function>()).CallPacked(args.Slice(1), ret);
                  })
      .def_packed(
          "tvmjs.testing.log_info_str",
          [](ffi::PackedArgs args, ffi::Any* ret) { LOG(INFO) << args[0].cast<ffi::String>(); })
      .def("tvmjs.testing.add_one", [](int x) { return x + 1; })
      .def_packed("tvmjs.testing.check_byte_array_across_callbacks",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    TVMFFIByteArray* bytes = args[0].cast<TVMFFIByteArray*>();
                    uint32_t expected = ByteArrayChecksum(bytes);
                    (args[1].cast<ffi::Function>())();
                    if (ByteArrayChecksum(bytes) != expected) {
                      *ret = int64_t{1};
                      return;
                    }
                    (args[1].cast<ffi::Function>())();
                    *ret = static_cast<int64_t>(ByteArrayChecksum(bytes) != expected) * 2;
                  })
      .def_packed("tvmjs.testing.return_bytes_across_callbacks",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    TVMFFIByteArray* bytes = args[0].cast<TVMFFIByteArray*>();
                    *ret = ffi::Bytes(bytes->data, bytes->size);
                    (args[1].cast<ffi::Function>())();
                    (args[1].cast<ffi::Function>())();
                  })
      .def_packed("tvmjs.testing.wrap_callback", [](ffi::PackedArgs args, ffi::Any* ret) {
        ffi::Function pf = args[0].cast<ffi::Function>();
        *ret = ffi::TypedFunction<void()>([pf]() { pf(); });
      });
}

size_t GetCheckedTensorElementCount(ffi::TensorView tensor) {
  size_t size = 1;
  for (int i = 0; i < tensor.ndim(); ++i) {
    TVM_FFI_ICHECK_GE(tensor.shape()[i], 0);
    const uint64_t dim_u64 = static_cast<uint64_t>(tensor.shape()[i]);
    TVM_FFI_ICHECK_LE(dim_u64, std::numeric_limits<size_t>::max());
    const size_t dim = static_cast<size_t>(dim_u64);
    if (size != 0) {
      TVM_FFI_ICHECK_LE(dim, std::numeric_limits<size_t>::max() / size);
    }
    size *= dim;
  }
  return size;
}

void CheckF32CPUTensor(ffi::TensorView tensor) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(tensor.IsContiguous());
  TVM_FFI_ICHECK_EQ(tensor.dtype().code, kDLFloat);
  TVM_FFI_ICHECK_EQ(tensor.dtype().bits, 32);
  TVM_FFI_ICHECK_EQ(tensor.dtype().lanes, 1);
}

void ArrayDecodeStorage(ffi::Tensor cpu_arr, TVMFFIByteArray* bytes, const std::string& format,
                        const std::string& dtype) {
  TVM_FFI_ICHECK(cpu_arr.defined());
  TVM_FFI_ICHECK_NE(bytes, nullptr);
  const char* byte_data = bytes->data;
  const size_t byte_size = bytes->size;
  if (format == "f32-to-bf16" && dtype == "float32") {
    CheckF32CPUTensor(cpu_arr);
    const size_t size = GetCheckedTensorElementCount(cpu_arr);
    TVM_FFI_ICHECK_LE(size, std::numeric_limits<size_t>::max() / 4);
    TVM_FFI_ICHECK_EQ(byte_size, size * 2);
    if (size == 0) {
      return;
    }
    TVM_FFI_ICHECK_NE(cpu_arr.data_ptr(), nullptr);
    TVM_FFI_ICHECK_NE(byte_data, nullptr);
    const uint8_t* bf16 = reinterpret_cast<const uint8_t*>(byte_data);
    uint8_t* data = static_cast<uint8_t*>(cpu_arr.data_ptr()) + cpu_arr.byte_offset();
    for (size_t i = 0; i < size; ++i) {
      uint16_t bf16_bits;
      std::memcpy(&bf16_bits, bf16 + 2 * i, sizeof(bf16_bits));
      const uint32_t f32_bits = static_cast<uint32_t>(bf16_bits) << 16;
      std::memcpy(data + 4 * i, &f32_bits, sizeof(f32_bits));
    }
  } else {
    CopyFromBytes(cpu_arr, byte_data, byte_size);
  }
}

void ArrayDecodeBF16ToF32Inplace(ffi::Tensor cpu_arr, int64_t encoded_nbytes) {
  CheckF32CPUTensor(cpu_arr);
  TVM_FFI_ICHECK_GE(encoded_nbytes, 0);

  const size_t size = GetCheckedTensorElementCount(cpu_arr);
  TVM_FFI_ICHECK_LE(size, std::numeric_limits<size_t>::max() / 4);
  TVM_FFI_ICHECK_EQ(static_cast<uint64_t>(encoded_nbytes), static_cast<uint64_t>(size) * 2);
  if (size == 0) {
    return;
  }
  TVM_FFI_ICHECK_NE(cpu_arr.data_ptr(), nullptr);

  uint8_t* data = static_cast<uint8_t*>(cpu_arr.data_ptr()) + cpu_arr.byte_offset();
  for (size_t i = size; i != 0; --i) {
    const size_t j = i - 1;
    uint16_t bf16_bits;
    std::memcpy(&bf16_bits, data + 2 * j, sizeof(bf16_bits));
    const uint32_t f32_bits = static_cast<uint32_t>(bf16_bits) << 16;
    std::memcpy(data + 4 * j, &f32_bits, sizeof(f32_bits));
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("tvmjs.array.decode_storage",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    ffi::Tensor cpu_arr = args[0].cast<ffi::Tensor>();
                    TVMFFIByteArray* bytes = args[1].cast<TVMFFIByteArray*>();
                    std::string format = args[2].cast<ffi::String>().operator std::string();
                    std::string dtype = args[3].cast<ffi::String>().operator std::string();
                    ArrayDecodeStorage(cpu_arr, bytes, format, dtype);
                  })
      .def("tvmjs.array.decode_bf16_to_f32_inplace", ArrayDecodeBF16ToF32Inplace);
}

// Concatenate n TVMArrays
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tvmjs.runtime.ArrayConcat",
                               [](ffi::PackedArgs args, ffi::Any* ret) {
                                 std::vector<Any> data;
                                 for (int i = 0; i < args.size(); ++i) {
                                   // Get i-th TVMArray
                                   auto* arr_i = args[i].as<ffi::ArrayObj>();
                                   TVM_FFI_ICHECK(arr_i != nullptr);
                                   for (size_t j = 0; j < arr_i->size(); ++j) {
                                     // Push back each j-th element of the i-th array
                                     data.push_back(arr_i->at(j));
                                   }
                                 }
                                 *ret = ffi::Array<Any>(data);
                               });
}

ffi::Tensor ConcatEmbeddings(const std::vector<ffi::Tensor>& embeddings) {
  TVM_FFI_ICHECK(!embeddings.empty());
  for (const auto& embedding : embeddings) {
    TVM_FFI_ICHECK(embedding.defined());
    TVM_FFI_ICHECK_EQ(embedding.ndim(), 2);
  }
  // Get output shape
  int64_t hidden_size = embeddings[0].shape()[1];
  DLDataType dtype = embeddings[0].dtype();
  DLDevice device = embeddings[0].device();
  int seqLen = 0;
  for (int i = 0; i < embeddings.size(); ++i) {
    TVM_FFI_ICHECK_EQ(embeddings[i].ndim(), 2);
    TVM_FFI_ICHECK_EQ(embeddings[i].shape()[1], hidden_size);
    seqLen += embeddings[i].shape()[0];
  }

  // Create output
  std::vector<int64_t> shape;
  shape.push_back(seqLen);
  shape.push_back(hidden_size);
  ffi::Tensor result = Empty(shape, dtype, device);

  // Copy
  int offset = 0;
  for (int i = 0; i < embeddings.size(); i++) {
    const DLTensor& copy_src = *(embeddings[i].GetDLTensorPtr());
    const DLTensor* p_copy_dst = result.GetDLTensorPtr();
    DLTensor copy_dst = *p_copy_dst;
    copy_dst.shape = embeddings[i].GetDLTensorPtr()->shape;
    copy_dst.byte_offset =
        offset * hidden_size * ((embeddings[i].dtype().bits * embeddings[i].dtype().lanes + 7) / 8);
    CopyFromTo(&copy_src, &copy_dst);
    offset += embeddings[i].shape()[0];
  }

  return result;
}

// Concatenate n Tensors
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("tvmjs.runtime.ConcatEmbeddings",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    std::vector<ffi::Tensor> embeddings;
                    for (int i = 0; i < args.size(); ++i) {
                      embeddings.push_back(args[i].cast<ffi::Tensor>());
                    }
                    ffi::Tensor result = ConcatEmbeddings(std::move(embeddings));
                    *ret = result;
                  })
      .def("tvmjs.runtime.TensorCopyFromBytes",
           [](ffi::TensorView nd, TVMFFIByteArray* bytes) {
             TVM_FFI_ICHECK_NE(bytes, nullptr);
             CopyFromBytes(nd, bytes->data, bytes->size);
           })
      .def("tvmjs.runtime.TensorCopyToBytes", [](ffi::TensorView nd) -> ffi::Bytes {
        size_t size = ffi::GetDataSize(nd);
        std::string bytes;
        bytes.resize(size);
        CopyToBytes(nd, bytes.data(), size);
        return ffi::Bytes(bytes);
      });
}

}  // namespace runtime
}  // namespace tvm
