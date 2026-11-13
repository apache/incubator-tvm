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
 * \file src/runtime/vm/builtin.cc
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/tensor.h>
#include <tvm/runtime/vm/builtin.h>
#include <tvm/runtime/vm/bytecode.h>
#include <tvm/runtime/vm/vm.h>

#include <cstring>
#include <unordered_map>

namespace tvm {
namespace runtime {
namespace vm {

using tvm::ffi::Tensor;

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("vm.builtin.initialize_default", [](ffi::Module mod, DLDevice dev) {
    auto init = mod->GetFunction("vm_initialization");
    TVM_FFI_CHECK(init.has_value(), ValueError) << "Module has no VM initialization function";
    (*init)(static_cast<int>(dev.device_type), dev.device_id, static_cast<int>(kDLCPU), 0);
  });
}

//-------------------------------------------------
//  Shape/Type handling.
//-------------------------------------------------
/*!
 * \brief Builtin function to allocate shape heap.
 * \param ctx_ptr The context module pointer.
 * \param size the size of the heap.
 * \return An allocate Tensor as shape heap.
 */
ffi::Tensor AllocShapeHeap(void* ctx_ptr, int64_t size) {
  VirtualMachine* vm = static_cast<VirtualMachine*>(ctx_ptr);
  // Use the host pool, which is always the last element.
  size_t host_device_index = vm->devices.size() - 1;
  // specially handle hexagon on-device RT.
  // TODO(relax-team): visit and consider other possible choices.
  if (vm->devices[0].device_type == kDLHexagon) {
    host_device_index = 0;
  } else {
    TVM_FFI_ICHECK_EQ(vm->devices[host_device_index].device_type, kDLCPU);
  }
  auto& pool = vm->memory_pools[host_device_index];
  return pool->Empty({size}, DLDataType{kDLInt, 64, 1}, vm->devices[host_device_index]);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("vm.builtin.alloc_shape_heap", AllocShapeHeap);
}

/*!
 * \brief Builtin match R.Prim function.
 *
 * \param input_value The runtime value provided by the user
 *
 * \param heap The VM storage for symbolic shapes
 *
 * \param code_value The op code, defined in MatchShapeCode,
 *     indicating how this value should be interpreted.
 *
 * \param reg The register, if using kStoreToHeap or
 *     kAssertEqualToLoad, or a literal value if using kAssertEqualToImm
 *
 * \param err_ctx An optional string used in error messages, providing
 *     additional context
 *
 * \sa MatchShape
 */
void MatchPrimValue(int64_t input_value, DLTensor* heap, int code_value, int64_t reg,
                    ffi::Optional<ffi::String> err_ctx) {
  int64_t* heap_data = heap == nullptr ? nullptr : static_cast<int64_t*>(heap->data);
  MatchShapeCode code = static_cast<MatchShapeCode>(code_value);

  if (code == MatchShapeCode::kAssertEqualToImm) {
    TVM_FFI_CHECK_EQ(input_value, reg, RuntimeError)
        << err_ctx.value_or("") << " match_cast error, "
        << " PrimExpr mismatch to specified constant.";
  } else if (code == MatchShapeCode::kStoreToHeap) {
    heap_data[reg] = input_value;
  } else if (code == MatchShapeCode::kNoOp) {
  } else if (code == MatchShapeCode::kAssertEqualToLoad) {
    TVM_FFI_CHECK_EQ(input_value, heap_data[reg], RuntimeError)
        << err_ctx.value_or("") << " match_cast error, "
        << " PrimExpr mismatch to a previous populated value.";
  } else {
    TVM_FFI_THROW(InternalError) << "Unknown match shape code: " << static_cast<int>(code);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("vm.builtin.match_prim_value", MatchPrimValue);
}

/*!
 * \brief Builtin match shape function.
 * \param args The packed function arguments.
 * \param rv The return value.
 *
 * \sa MatchShapeCode
 */
void MatchShape(ffi::PackedArgs args, ffi::Any* rv) {
  // input shape the first argument can take in tensor, DLTensor* or shape.
  ffi::Shape input_shape;
  if (auto opt_tensor = args[0].as<ffi::Tensor>(); opt_tensor && opt_tensor->defined()) {
    input_shape = opt_tensor.value().shape();
  } else if (auto opt_dltensor = args[0].try_cast<DLTensor*>()) {
    DLTensor* ptr = opt_dltensor.value();
    input_shape = ffi::Shape(ptr->shape, ptr->shape + ptr->ndim);
  } else {
    input_shape = args[0].cast<ffi::Shape>();
  }
  auto heap = args[1].try_cast<DLTensor*>();
  int64_t* heap_data = heap.has_value() ? static_cast<int64_t*>((*heap)->data) : nullptr;
  int64_t size = args[2].cast<int64_t>();
  const int64_t kBeginCode = 3;
  TVM_FFI_ICHECK_LE(kBeginCode + size * 2, args.size());
  // a function that lazily get context for error reporting
  const int64_t kErrorContextOffset = kBeginCode + size * 2;
  ffi::Optional<ffi::String> err_ctx = args[kErrorContextOffset].cast<ffi::String>();

  TVM_FFI_CHECK_EQ(input_shape.size(), size, RuntimeError)
      << err_ctx.value_or("") << " match_cast shape size mismatch.";

  for (int64_t i = 0; i < size; ++i) {
    MatchShapeCode code = static_cast<MatchShapeCode>(args[kBeginCode + i * 2].cast<int>());
    int64_t reg = args[kBeginCode + i * 2 + 1].cast<int64_t>();

    if (code == MatchShapeCode::kAssertEqualToImm) {
      TVM_FFI_CHECK_EQ(input_shape[i], reg, RuntimeError)
          << err_ctx.value_or("") << " match_cast error, "
          << " shape[" << i << "]"
          << " mismatch to specified constant.";
    } else if (code == MatchShapeCode::kStoreToHeap) {
      heap_data[reg] = input_shape[i];
    } else if (code == MatchShapeCode::kNoOp) {
    } else {
      TVM_FFI_ICHECK(code == MatchShapeCode::kAssertEqualToLoad);
      TVM_FFI_CHECK_EQ(input_shape[i], heap_data[reg], RuntimeError)
          << err_ctx.value_or("") << " match_cast error, "
          << " shape[" << i << "]"
          << " mismatch to a previous populated value.";
    }
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("vm.builtin.match_shape", MatchShape);
}

/*!
 * \brief Builtin make prim value function.
 * \param heap The shape heap to use
 * \param shape_code The shape code of the value
 * \param rv The return value.
 *
 * \sa MakeShape
 */
int64_t MakePrimValue(DLTensor* heap, int shape_code, int64_t reg) {
  // NOTE: heap can be nullptr
  int64_t* heap_data = heap == nullptr ? nullptr : static_cast<int64_t*>(heap->data);

  MakeShapeCode code = static_cast<MakeShapeCode>(shape_code);
  if (code == MakeShapeCode::kUseImm) {
    return reg;
  } else if (code == MakeShapeCode::kLoadShape) {
    return heap_data[reg];
  } else {
    TVM_FFI_THROW(InternalError) << "Invalid shape code: " << shape_code;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("vm.builtin.make_prim_value", MakePrimValue);
}

/*!
 * \brief Builtin make shape function.
 * \param args The packed function arguments.
 * \param rv The return value.
 *
 * \sa MakeShapeCode
 */
void MakeShape(ffi::PackedArgs args, ffi::Any* rv) {
  // NOTE: heap can be nullptr
  auto heap = args[0].try_cast<DLTensor*>();
  int64_t* heap_data = heap.has_value() ? static_cast<int64_t*>((*heap)->data) : nullptr;
  int64_t size = args[1].cast<int64_t>();
  const int64_t kBeginCode = 2;

  std::vector<int64_t> shape(size);

  for (int64_t i = 0; i < size; ++i) {
    MakeShapeCode code = static_cast<MakeShapeCode>(args[kBeginCode + i * 2].cast<int>());
    int64_t reg = args[kBeginCode + i * 2 + 1].cast<int64_t>();
    if (code == MakeShapeCode::kUseImm) {
      shape[i] = reg;
    } else {
      TVM_FFI_ICHECK(code == MakeShapeCode::kLoadShape);
      shape[i] = heap_data[reg];
    }
  }
  *rv = ffi::Shape(std::move(shape));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("vm.builtin.make_shape", MakeShape);
}

/*!
 * \brief Builtin function to check if arg is Tensor(dtype, ndim)
 * \param arg The input argument.
 * \param ndim Expected ndim of the Tensor, can be -1 (indicate unknown).
 * \param dtype The expected content data type.
 * \param err_ctx Additional context if error occurs.
 */
void CheckTensorInfo(ffi::PackedArgs args, ffi::Any* rv) {
  ffi::AnyView arg = args[0];
  int ndim = args[1].cast<int>();
  ffi::Optional<DLDataType> dtype;
  ffi::Optional<ffi::String> err_ctx;

  if (args.size() == 3) {
    dtype = std::nullopt;
    err_ctx = args[2].cast<ffi::Optional<ffi::String>>();
  } else {
    dtype = args[2].cast<ffi::Optional<DLDataType>>();
    err_ctx = args[3].cast<ffi::Optional<ffi::String>>();
  }

  auto opt_ptr = arg.try_cast<DLTensor*>();
  TVM_FFI_CHECK(opt_ptr.has_value(), TypeError)
      << err_ctx.value_or("") << " expect a Tensor but get " << arg.GetTypeKey();

  DLTensor* ptr = opt_ptr.value();
  if (ndim != -1) {
    TVM_FFI_CHECK(ptr->ndim == ndim, ValueError)
        << err_ctx.value_or("") << " expect Tensor with ndim " << ndim << " but get " << ptr->ndim;
  }

  if (dtype) {
    TVM_FFI_CHECK(ptr->dtype == dtype.value(), ValueError)
        << err_ctx.value_or("") << " expect Tensor with dtype " << dtype.value() << " but get "
        << ptr->dtype;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("vm.builtin.check_tensor_info", CheckTensorInfo);
}

/*!
 * \brief Builtin function to check if arg is Shape(ndim)
 * \param arg The input argument.
 * \param ndim Expected size of the shape, can be -1 (indicate unknown).
 * \param err_ctx Additional context if error occurs.
 */
void CheckShapeInfo(ffi::ObjectRef arg, int ndim, ffi::Optional<ffi::String> err_ctx) {
  // a function that lazily get context for error reporting
  auto* ptr = arg.as<ffi::Shape::ContainerType>();
  TVM_FFI_CHECK(ptr != nullptr, TypeError)
      << err_ctx.value_or("") << " expect a Shape but get " << arg->GetTypeKey();
  if (ndim != -1) {
    TVM_FFI_CHECK(ptr->size == static_cast<uint64_t>(ndim), ValueError)
        << err_ctx.value_or("") << " expect Shape with ndim " << ndim << " but get " << ptr->size;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("vm.builtin.check_shape_info", CheckShapeInfo);
}

/*!
 * \brief Builtin function to check if arg is PrimExpr(dtype)
 * \param arg The input argument.
 * \param dtype Expected dtype of the PrimExpr.  Can be DLDataType{kDLOpaqueHandle, 0, 0} for
 * unknown dtype.
 * \param err_ctx Additional context if error occurs.
 */
void CheckPrimValueInfo(ffi::AnyView arg, DLDataType dtype, ffi::Optional<ffi::String> err_ctx) {
  if (auto opt_obj = arg.as<ffi::ObjectRef>()) {
    TVM_FFI_THROW(TypeError) << err_ctx.value_or("") << ", expected dtype " << dtype
                             << ", but received ObjectRef of type "
                             << opt_obj.value()->GetTypeKey();
  } else if (dtype.code == kDLBool) {
    arg.cast<bool>();
  } else if (dtype.code == kDLInt) {
    arg.cast<int64_t>();
  } else if (dtype.code == kDLUInt) {
    arg.cast<uint64_t>();
  } else if (dtype.code == kDLFloat) {
    arg.cast<double>();
  } else if (dtype.code == kDLOpaqueHandle && !(dtype.bits == 0 && dtype.lanes == 0)) {
    arg.cast<void*>();
  } else {
    TVM_FFI_THROW(TypeError) << err_ctx.value_or("") << ", unsupported dtype " << dtype;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("vm.builtin.check_prim_value_info", CheckPrimValueInfo);
  refl::GlobalDef().def("vm.builtin.check_string_info",
                        [](ffi::AnyView value, ffi::Optional<ffi::String> err_ctx) {
                          TVM_FFI_CHECK(value.as<ffi::String>().has_value(), TypeError)
                              << err_ctx.value_or("") << " expect a string";
                        });
}

/*!
 * \brief Builtin function to check if arg is Tuple with size elements.
 * \param arg The input argument.
 * \param size The expected size of the tuple.
 * \param err_ctx Additional context if error occurs.
 */
void CheckTupleInfo(ffi::ObjectRef arg, int64_t size, ffi::Optional<ffi::String> err_ctx) {
  // a function that lazily get context for error reporting
  auto* ptr = arg.as<ffi::ArrayObj>();
  TVM_FFI_CHECK(ptr != nullptr, TypeError)
      << err_ctx.value_or("") << " expect a Tuple but get " << arg->GetTypeKey();
  TVM_FFI_CHECK(static_cast<int64_t>(ptr->size()) == size, ValueError)
      << err_ctx.value_or("") << " expect a Tuple with " << size << " elements, "
      << " but get a Tuple with " << ptr->size() << " elements.";
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("vm.builtin.check_tuple_info", CheckTupleInfo);
}

/*!
 * \brief Builtin function to check if arg is a callable function.
 * \param arg The input argument.
 * \param err_ctx Additional context if error occurs.
 */
void CheckFuncInfo(ffi::ObjectRef arg, ffi::Optional<ffi::String> err_ctx) {
  // a function that lazily get context for error reporting
  bool is_func = arg.as<ffi::Function::ContainerType>() || arg.as<VMClosure::ContainerType>();
  TVM_FFI_CHECK(is_func, TypeError)
      << err_ctx.value_or("") << " expect a Function but get " << arg->GetTypeKey();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("vm.builtin.check_func_info", CheckFuncInfo);
}

//-------------------------------------------------
//  Storage management.
//-------------------------------------------------
ffi::Tensor VMAllocStorage(void* ctx_ptr, ffi::Shape buffer_shape, Index device_index,
                           DLDataType dtype_hint, MemoryScope mem_scope) {
  VirtualMachine* vm = static_cast<VirtualMachine*>(ctx_ptr);
  if (device_index == -1) {
    // Allocate on host. Host is always the last element of vm->devices.
    device_index = vm->devices.size() - 1;
  }
  TVM_FFI_ICHECK_GE(device_index, 0);
  TVM_FFI_ICHECK_LT(device_index, vm->devices.size())
      << "The device index is out of VM physical devices list";
  auto& pool = vm->memory_pools[device_index];
  TVM_FFI_ICHECK(pool) << "Did you forget to init the VirtualMachine with devices?";
  return pool->Empty(buffer_shape, dtype_hint, vm->devices[device_index], mem_scope);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("vm.builtin.alloc_storage", VMAllocStorage)
      .def_packed("vm.builtin.alloc_tensor", [](ffi::PackedArgs args, ffi::Any* rv) {
        VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
        ffi::Tensor backing = args[1].cast<ffi::Tensor>();
        int64_t offset = args[2].cast<int64_t>();
        ffi::Shape shape = args[3].cast<ffi::Shape>();
        DLDataType dtype = args[4].cast<DLDataType>();
        ffi::Optional<MemoryScope> scope = std::nullopt;
        if (args.size() == 6) {
          scope = args[5].cast<MemoryScope>();
        }
        TVM_FFI_CHECK_GE(offset, 0, ValueError) << "Tensor byte offset must be nonnegative";
        DLDevice device = backing.device();
        for (size_t i = 0; i < vm->devices.size(); ++i) {
          if (vm->devices[i].device_type == device.device_type &&
              vm->devices[i].device_id == device.device_id) {
            *rv = vm->memory_pools[i]->CreateView(std::move(backing), shape, dtype, offset, scope);
            return;
          }
        }
        TVM_FFI_THROW(ValueError) << "Tensor backing device is not initialized in the VM";
      });
}

//-------------------------------------------------
//  Closure function handling, calling convention
//-------------------------------------------------
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("vm.builtin.make_closure",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    VMClosure clo = args[0].cast<VMClosure>();
                    std::vector<ffi::Any> saved_args;
                    saved_args.resize(args.size() - 1);
                    for (size_t i = 0; i < saved_args.size(); ++i) {
                      saved_args[i] = args[i + 1];
                    }
                    auto impl = VMClosure::BindLastArgs(clo->impl, saved_args);
                    *rv = VMClosure(clo->func_name, impl);
                  })
      .def_packed("vm.builtin.invoke_closure",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    // args[0]: vm; args[1]: closure; args[2, 3, ...]: function arguments
                    VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
                    ffi::ObjectRef vm_closure = args[1].cast<ffi::ObjectRef>();
                    vm->InvokeClosurePacked(vm_closure, args.Slice(2), rv);
                  })
      .def_packed("vm.builtin.call_tir_dyn", [](ffi::PackedArgs args, ffi::Any* rv) {
        ffi::Function func = args[0].cast<ffi::Function>();
        ffi::Shape to_unpack = args[args.size() - 1].cast<ffi::Shape>();
        size_t num_tensor_args = args.size() - 2;

        std::vector<ffi::AnyView> packed_args(num_tensor_args + to_unpack.size());
        std::copy(args.data() + 1, args.data() + args.size() - 1, packed_args.data());

        for (size_t i = 0; i < to_unpack.size(); ++i) {
          packed_args[i + num_tensor_args] = to_unpack[i];
        }
        func.CallPacked(ffi::PackedArgs(packed_args.data(), packed_args.size()), rv);
      });
}

//-------------------------------------
//  Python function call support
//-------------------------------------

// Global registry for Python functions
static std::unordered_map<std::string, ffi::Function> py_func_registry;

/*!
 * \brief Clear the Python function registry on shutdown
 */
void ClearPyFuncRegistry() { py_func_registry.clear(); }

/*!
 * \brief Register a Python function for call_py_func
 * \param name The function name
 * \param func The Python function wrapped as ffi::Function
 */
void RegisterPyFunc(const std::string& name, ffi::Function func) { py_func_registry[name] = func; }

/*!
 * \brief Unregister a Python function registered with RegisterPyFunc
 * \param name The function name
 */
void UnregisterPyFunc(const std::string& name) { py_func_registry.erase(name); }

/*!
 * \brief Get a registered Python function
 * \param name The function name
 * \return The Python function
 */
ffi::Function GetPyFunc(const std::string& name) {
  auto it = py_func_registry.find(name);
  if (it == py_func_registry.end()) {
    TVM_FFI_THROW(InternalError) << "Python function '" << name << "' not found in registry";
  }
  return it->second;
}

/*!
 * \brief Call a Python function from VM
 * \param args The packed function arguments (tuple containing function name and arguments)
 * \param rv The return value
 */
void CallPyFunc(ffi::PackedArgs args, ffi::Any* rv) {
  // args[0] should be a tuple containing (func_name, args_tuple)
  if (args.size() != 1) {
    TVM_FFI_THROW(InternalError) << "vm.builtin.call_py_func expects exactly 1 argument (tuple)";
  }

  auto tuple_arg = args[0].cast<ffi::Array<ffi::Any>>();
  if (tuple_arg.size() != 2) {
    TVM_FFI_THROW(InternalError)
        << "vm.builtin.call_py_func tuple should contain (func_name, args)";
  }

  // Get function name
  std::string func_name = tuple_arg[0].cast<ffi::String>();

  // Get arguments tuple
  auto func_args = tuple_arg[1].cast<ffi::Array<ffi::Any>>();

  // Look up Python function in registry
  ffi::Function py_func = GetPyFunc(func_name);

  // Call the Python function with the arguments
  std::vector<ffi::AnyView> py_args_vec(func_args.begin(), func_args.end());
  ffi::PackedArgs py_args(py_args_vec.data(), py_args_vec.size());
  py_func.CallPacked(py_args, rv);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("vm.builtin.call_py_func", CallPyFunc)
      .def("vm.builtin.register_py_func", RegisterPyFunc)
      .def("vm.builtin.unregister_py_func", UnregisterPyFunc)
      .def("vm.builtin.get_py_func", GetPyFunc)
      .def("vm.builtin.clear_py_func_registry", ClearPyFuncRegistry);
}

//-------------------------------------
//  Builtin runtime operators.
//-------------------------------------
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("vm.builtin.shape_of",
                  [](ffi::Any any) -> ffi::Shape {
                    if (auto opt_tensor = any.try_cast<ffi::Tensor>();
                        opt_tensor && opt_tensor->defined()) {
                      return opt_tensor.value().shape();
                    } else if (auto opt_dltensor = any.try_cast<DLTensor*>()) {
                      DLTensor* ptr = opt_dltensor.value();
                      return ffi::Shape(ptr->shape, ptr->shape + ptr->ndim);
                    } else {
                      TVM_FFI_THROW(TypeError)
                          << "vm.builtin.shape_of expects a Tensor or DLTensor*, but get "
                          << any.GetTypeKey();
                    }
                  })
      .def("vm.builtin.copy", [](ffi::Any a) -> ffi::Any { return a; })
      .def("vm.builtin.reshape",
           [](ffi::Any any, ffi::Shape new_shape) {
             if (auto opt_tensor = any.try_cast<ffi::Tensor>();
                 opt_tensor && opt_tensor->defined()) {
               ffi::Tensor data = opt_tensor.value();
               return tvm::runtime::CreateView(data, new_shape, data.dtype());
             } else if (auto opt_dltensor = any.try_cast<DLTensor*>()) {
               DLTensor* ptr = opt_dltensor.value();
               auto tmp = std::make_unique<DLManagedTensor>();
               tmp->dl_tensor = *ptr;
               tmp->manager_ctx = nullptr;
               tmp->deleter = nullptr;
               ffi::Tensor data = ffi::Tensor::FromDLPack(tmp.release());
               return tvm::runtime::CreateView(data, new_shape, data.dtype());
             } else {
               TVM_FFI_THROW(TypeError)
                   << "vm.builtin.reshape expects a Tensor or DLTensor*, but get "
                   << any.GetTypeKey();
             }
           })
      .def("vm.builtin.null_value", []() -> std::nullptr_t { return nullptr; })
      .def_packed("vm.builtin.to_device", [](ffi::PackedArgs args, ffi::Any* rv) {
        ffi::Tensor data = args[0].cast<ffi::Tensor>();
        TVM_FFI_ICHECK(data.defined());
        int dev_type = args[1].cast<int>();
        int dev_id = args[2].cast<int>();
        Device dst_device = {(DLDeviceType)dev_type, dev_id};
        MemoryScope mem_scope = MemoryScope::Global();
        if (args.size() == 4) {
          mem_scope = args[3].cast<MemoryScope>();
        }
        *rv = tvm::runtime::CopyTo(data, dst_device, mem_scope);
      });
}

/*!
 * \brief Load the scalar value in cond and return the result value.
 * \param cond The condition
 * \return Bool
 */
bool ReadIfCond(ffi::AnyView cond) {
  if (auto opt_int = cond.try_cast<bool>()) {
    return opt_int.value();
  }
  ffi::Tensor arr = cond.cast<tvm::ffi::Tensor>();
  TVM_FFI_ICHECK(arr.defined());
  if (arr.device().device_type != kDLCPU) {
    arr = tvm::runtime::CopyTo(arr, DLDevice{kDLCPU, 0});
  }
  TVM_FFI_ICHECK(arr.dtype().code == kDLInt || arr.dtype().code == kDLUInt ||
                 arr.dtype().code == kDLBool);
  TVM_FFI_ICHECK_GT(arr.numel(), 0);
  TVM_FFI_ICHECK(arr.data_ptr() != nullptr);
  const char* scalar_data = static_cast<const char*>(arr.data_ptr()) + arr.byte_offset();
  int64_t result;
  switch (arr.dtype().bits) {
    case 1: {
      int8_t value;
      std::memcpy(&value, scalar_data, sizeof(value));
      result = value;
      break;
    }
    case 8: {
      int8_t value;
      std::memcpy(&value, scalar_data, sizeof(value));
      result = value;
      break;
    }
    case 16: {
      int16_t value;
      std::memcpy(&value, scalar_data, sizeof(value));
      result = value;
      break;
    }
    case 32: {
      int32_t value;
      std::memcpy(&value, scalar_data, sizeof(value));
      result = value;
      break;
    }
    case 64: {
      int64_t value;
      std::memcpy(&value, scalar_data, sizeof(value));
      result = value;
      break;
    }
    default:
      TVM_FFI_THROW(InternalError)
          << "Unknown scalar int type: " << ffi::DLDataTypeToString(arr.dtype());
      throw;
  }
  return result != 0;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("vm.builtin.read_if_cond", ReadIfCond);
}

//-------------------------------------
//  Debugging API
//-------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      "vm.builtin.invoke_debug_func", [](ffi::PackedArgs args, ffi::Any* rv) -> void {
        TVM_FFI_ICHECK_GE(args.size(), 3);
        int num_args = args.size() - 3;
        ffi::ObjectRef io_effect = args[0].cast<ffi::ObjectRef>();
        TVM_FFI_CHECK(!io_effect.defined(), ValueError)
            << "IOEffect is expected to be lowered to None.";
        ffi::String debug_func_name = args[1].cast<ffi::String>();
        const auto debug_func = tvm::ffi::Function::GetGlobal(debug_func_name);
        TVM_FFI_CHECK(debug_func.has_value(), ValueError)
            << debug_func_name << " is not found. "
            << "Use the decorator `@tvm_ffi.register_global_func(\"" << debug_func_name
            << "\")` to register it.";
        ffi::String line_info = args[2].cast<ffi::String>();
        std::vector<ffi::AnyView> call_args(num_args + 1);
        {
          call_args[0] = line_info;
          for (int i = 0; i < num_args; ++i) {
            call_args[i + 1] = args[i + 3];
          }
        }
        debug_func->CallPacked(ffi::PackedArgs(call_args.data(), call_args.size()), rv);
        *rv = io_effect;
      });
}

//-------------------------------------
//  Data structure API
//-------------------------------------
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("vm.builtin.tuple_getitem",
           [](ffi::Array<ffi::Any> arr, int64_t index) { return arr[index]; })
      .def("vm.builtin.tuple_reset_item",
           [](const ffi::ArrayObj* arr, int64_t index) {
             const_cast<ffi::ArrayObj*>(arr)->SetItem(index, nullptr);
           })
      .def_packed("vm.builtin.make_tuple",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    ffi::Array<ffi::Any> arr;
                    for (int i = 0; i < args.size(); ++i) {
                      arr.push_back(args[i]);
                    }
                    *rv = arr;
                  })
      .def("vm.builtin.tensor_to_shape",
           [](ffi::Tensor data) {
             TVM_FFI_ICHECK(data.defined());
             ffi::Tensor arr = data;
             if (data.device().device_type != kDLCPU) {
               arr = tvm::runtime::CopyTo(data, DLDevice{kDLCPU, 0});
             }

             TVM_FFI_ICHECK_EQ(arr.ndim(), 1);
             TVM_FFI_ICHECK_EQ(arr.dtype().code, kDLInt);
             TVM_FFI_ICHECK(arr.IsContiguous());
             TVM_FFI_ICHECK(arr.numel() == 0 || arr.data_ptr() != nullptr);

             std::vector<int64_t> out_shape;
             for (int i = 0; i < arr.shape()[0]; ++i) {
               int64_t result;
               switch (arr.dtype().bits) {
                 case 16: {
                   int16_t value;
                   const char* element = static_cast<const char*>(arr.data_ptr()) +
                                         arr.byte_offset() + i * sizeof(value);
                   std::memcpy(&value, element, sizeof(value));
                   result = value;
                   break;
                 }
                 case 32: {
                   int32_t value;
                   const char* element = static_cast<const char*>(arr.data_ptr()) +
                                         arr.byte_offset() + i * sizeof(value);
                   std::memcpy(&value, element, sizeof(value));
                   result = value;
                   break;
                 }
                 case 64: {
                   int64_t value;
                   const char* element = static_cast<const char*>(arr.data_ptr()) +
                                         arr.byte_offset() + i * sizeof(value);
                   std::memcpy(&value, element, sizeof(value));
                   result = value;
                   break;
                 }
                 default:
                   TVM_FFI_THROW(InternalError)
                       << "Unknown scalar int type: " << ffi::DLDataTypeToString(arr.dtype());
                   throw;
               }
               out_shape.push_back(result);
             }
             return ffi::Shape(out_shape);
           })
      .def("vm.builtin.ensure_zero_offset", [](ffi::Tensor data) {
        TVM_FFI_ICHECK(data.defined());
        if (data.byte_offset() == 0) {
          return data;
        }
        auto* device_api = DeviceAPI::Get(data.device());
        if (device_api->SupportsDevicePtrArithmeticOnHost() &&
            data.byte_offset() % tvm::runtime::kAllocAlignment == 0) {
          DLManagedTensor* dl_tensor = data.ToDLPack();
          dl_tensor->dl_tensor.data =
              reinterpret_cast<char*>(dl_tensor->dl_tensor.data) + dl_tensor->dl_tensor.byte_offset;
          dl_tensor->dl_tensor.byte_offset = 0;
          return ffi::Tensor::FromDLPack(dl_tensor);
        } else {
          auto new_array = tvm::runtime::Empty(data.shape(), data.dtype(), data.device());
          tvm::runtime::CopyFromTo(data, new_array);
          return new_array;
        }
      });
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
