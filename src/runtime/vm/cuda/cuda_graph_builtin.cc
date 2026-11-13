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
 * \file src/runtime/vm/cuda/cuda_graph_builtin.cc
 * \brief The CUDA graph related builtin functions for Relax virtual machine.
 */

#include <tvm/ffi/base_detail.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/base.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/vm/vm.h>

namespace tvm {
namespace runtime {
namespace vm {

namespace {

struct CUDAGraphCaptureKey {
  // The unique index of the capture function within the module
  int64_t index;
  // The symbolic variables the capture function depends on. When the capture function is ran with
  // different symbolic variable values, the CUDA graph will be re-captured as a different version,
  // identified by this shape tuple. This is default constructed as an empty tuple.
  ffi::Shape shape_expr;

  CUDAGraphCaptureKey(int64_t index, const ffi::Optional<ffi::Shape>& shape_expr) : index(index) {
    if (shape_expr) {
      this->shape_expr = shape_expr.value();
    }
  }
};

struct CUDAGraphCaptureKeyHash {
  size_t operator()(const CUDAGraphCaptureKey& key) const {
    std::hash<int64_t> hash_fn;
    size_t hash = hash_fn(key.index);
    for (const auto& shape : key.shape_expr) {
      ffi::detail::StableHashCombine(hash, hash_fn(shape));
    }
    return hash;
  }
};

struct CUDAGraphCaptureKeyEqual {
  bool operator()(const CUDAGraphCaptureKey& lhs, const CUDAGraphCaptureKey& rhs) const {
    return lhs.index == rhs.index && std::equal(lhs.shape_expr.begin(), lhs.shape_expr.end(),
                                                rhs.shape_expr.begin(), rhs.shape_expr.end());
  }
};

/*! \brief The captured state of a CUDA graph */
struct CUDAGraphCapturedState {
  CUDAGraphCapturedState() {}

  CUDAGraphCapturedState(const CUDAGraphCapturedState&) = delete;
  CUDAGraphCapturedState(CUDAGraphCapturedState&& other) { *this = std::move(other); }

  CUDAGraphCapturedState& operator=(const CUDAGraphCapturedState&) = delete;
  CUDAGraphCapturedState& operator=(CUDAGraphCapturedState&& other) {
    std::swap(states, other.states);
    std::swap(exec, other.exec);
    return *this;
  }

  ~CUDAGraphCapturedState() {
    if (exec) {
      TVM_FFI_CHECK_CUDA_ERROR(cudaGraphExecDestroy(exec));
    }
  }

  /*!
   * \brief Tuple of intemediate tensors in the capture func that will be used outside the
   * capture func
   */
  ffi::ObjectRef states;
  /*! \brief The instantiated cuda graph */
  cudaGraphExec_t exec = nullptr;
};

class ScopedCUDAStream {
 public:
  ScopedCUDAStream() { TVM_FFI_CHECK_CUDA_ERROR(cudaStreamCreate(&stream_)); }
  ~ScopedCUDAStream() { cudaStreamDestroy(stream_); }
  ScopedCUDAStream(const ScopedCUDAStream&) = delete;
  ScopedCUDAStream(ScopedCUDAStream&&) = delete;
  ScopedCUDAStream& operator=(const ScopedCUDAStream&) = delete;
  ScopedCUDAStream& operator=(ScopedCUDAStream&&) = delete;

  operator cudaStream_t() const { return stream_; }

 private:
  cudaStream_t stream_;
};

class CUDACaptureStream {
 public:
  explicit CUDACaptureStream(cudaGraph_t* graph) : output_graph_(graph) {
    TVM_FFI_CHECK_CUDA_ERROR(cudaGetDevice(&device_id_));
    TVM_FFI_CHECK_SAFE_CALL(
        TVMFFIEnvSetStream(kDLCUDA, device_id_, capture_stream_,
                           reinterpret_cast<TVMFFIStreamHandle*>(&prev_default_stream_)));
    TVM_FFI_CHECK_CUDA_ERROR(cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeGlobal));
  }
  ~CUDACaptureStream() noexcept(false) {
    cudaError_t capture_error = cudaStreamEndCapture(capture_stream_, output_graph_);
    if (capture_error != cudaSuccess) {
      // The capture may have been invalidated by the exception that is
      // currently unwinding the stack.  Do not throw a second exception from
      // this destructor, but clear CUDA's thread-local error so that it is not
      // reported by an unrelated CUDA call later in the same host thread.
      cudaGetLastError();
    }
    TVM_FFI_CHECK_SAFE_CALL(TVMFFIEnvSetStream(kDLCUDA, device_id_, prev_default_stream_, nullptr));
  }

 private:
  int device_id_;
  cudaStream_t prev_default_stream_;
  ScopedCUDAStream capture_stream_;

  cudaGraph_t* output_graph_;
};

}  // namespace

/*! \brief The VM extension of CUDA graph. */
class CUDAGraphExtensionObj : public VMExtensionObj {
 public:
  /*!
   * \brief Launch the cuda graph if it has been cached, otherwise execute it in capture mode.
   * \param vm The virtual machine.
   * \param capture_func The function of type (args...) -> Tuple[ffi::ObjectRef], where 'args' are
   * the static arguments that are the same for all invocations of the capture function, the
   * returned tuple contains the intermediate tensors that will be used outside the capture
   * function.
   * \param args The static arguments of the capture function
   * \param entry_index The unique index of the capture function used for lookup.
   * \return The return value of the capture function.
   */
  ffi::ObjectRef RunOrCapture(VirtualMachine* vm, const ffi::ObjectRef& capture_func, Any args,
                              int64_t entry_index, ffi::Optional<ffi::Shape> shape_expr) {
    CUDAGraphCaptureKey entry_key{entry_index, shape_expr};
    if (auto it = capture_cache_.find(entry_key); it != capture_cache_.end()) {
      // Launch CUDA graph
      const auto& [states, exec] = it->second;
      int device_id;
      TVM_FFI_CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
      TVM_FFI_CHECK_CUDA_ERROR(
          cudaGraphLaunch(exec, static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device_id))));
      return states;
    }

    // Set up arguments for the graph execution
    ffi::Array<Any> tuple_args = args.cast<ffi::Array<Any>>();
    int nargs = static_cast<int>(tuple_args.size());

    std::vector<AnyView> packed_args(nargs);
    for (int i = 0; i < nargs; ++i) {
      packed_args[i] = tuple_args[i];
    }

    ffi::Any capture_func_rv;
    // Run the function without CUDA graph. This is a warm up step to do necessary initialization
    // of the CUDA module such as loading module data, setting kernel attributes.
    vm->InvokeClosurePacked(capture_func, ffi::PackedArgs(packed_args.data(), nargs),
                            &capture_func_rv);

    // Run the graph in capture mode
    cudaGraph_t graph;

    {
      CUDACaptureStream capture_stream(&graph);
      vm->InvokeClosurePacked(capture_func, ffi::PackedArgs(packed_args.data(), nargs),
                              &capture_func_rv);
    }

    CUDAGraphCapturedState entry;
    entry.states = capture_func_rv.cast<ffi::ObjectRef>();
    TVM_FFI_CHECK_CUDA_ERROR(cudaGraphInstantiate(&entry.exec, graph, NULL, NULL, 0));
    TVM_FFI_CHECK_CUDA_ERROR(cudaGraphDestroy(graph));

    ffi::ObjectRef states = entry.states;

    capture_cache_[entry_key] = std::move(entry);

    return states;
  }

  /*!
   * \brief Get the cached allocation from the cache or run the allocation function.
   * \param vm The virtual machine.
   * \param alloc_func The function of type () -> ffi::ObjectRef, where the returned object is the
   * tuple of allocated storage objects.
   * \param entry_index The unique index of the allocation function used for lookup.
   */
  ffi::ObjectRef GetCachedAllocation(VirtualMachine* vm, const ffi::ObjectRef& alloc_func,
                                     int64_t entry_index) {
    if (auto it = alloc_cache_.find(entry_index); it != alloc_cache_.end()) {
      return it->second;
    }
    ffi::Any alloc_func_rv;
    vm->InvokeClosurePacked(alloc_func, ffi::PackedArgs(nullptr, 0), &alloc_func_rv);
    ffi::ObjectRef alloc_result = alloc_func_rv.cast<ffi::ObjectRef>();
    alloc_cache_[entry_index] = alloc_result;
    return alloc_result;
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("vm.CUDAGraphExtension", CUDAGraphExtensionObj, VMExtensionObj);

 private:
  /*!
   * \brief The cache of captured cuda graphs. The key is a unique index for the capture function.
   * The value is the result of the capture.
   */
  std::unordered_map<CUDAGraphCaptureKey, CUDAGraphCapturedState, CUDAGraphCaptureKeyHash,
                     CUDAGraphCaptureKeyEqual>
      capture_cache_;
  /*!
   * \brief The cache of allocations. The key is a unique index for the allocation function.
   * The value is the cached allocations, which is a tuple of storages.
   */
  std::unordered_map<int64_t, ffi::ObjectRef> alloc_cache_;
};

/*! Managed reference to CUDAGraphExtensionObj */
class CUDAGraphExtension : public VMExtension {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CUDAGraphExtension, VMExtension,
                                                CUDAGraphExtensionObj);
  static CUDAGraphExtension Create() {
    auto data_ = ffi::make_object<CUDAGraphExtensionObj>();
    return ffi::GetRef<CUDAGraphExtension>(data_.get());
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("vm.builtin.cuda_graph.run_or_capture",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    TVM_FFI_ICHECK(args.size() == 5 || args.size() == 4);
                    VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
                    auto extension = vm->GetOrCreateExtension<CUDAGraphExtension>();
                    auto capture_func = args[1].cast<ffi::ObjectRef>();
                    Any func_args = args[2];
                    int64_t entry_index = args[3].cast<int64_t>();
                    ffi::Optional<ffi::Shape> shape_expr = std::nullopt;
                    if (args.size() == 5) {
                      shape_expr = args[4].cast<ffi::Shape>();
                    }
                    *rv = extension->RunOrCapture(vm, capture_func, func_args, entry_index,
                                                  shape_expr);
                  })
      .def_packed("vm.builtin.cuda_graph.get_cached_alloc", [](ffi::PackedArgs args, ffi::Any* rv) {
        TVM_FFI_ICHECK_EQ(args.size(), 3);
        VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
        auto extension = vm->GetOrCreateExtension<CUDAGraphExtension>();
        auto alloc_func = args[1].cast<ffi::ObjectRef>();
        int64_t entry_index = args[2].cast<int64_t>();
        *rv = extension->GetCachedAllocation(vm, alloc_func, entry_index);
      });
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
