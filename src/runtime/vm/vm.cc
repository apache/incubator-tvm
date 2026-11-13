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
 * \file src/runtime/vm/vm.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/vm/vm.h>

#include <thread>

#include "./module_utils.h"
#include "nvtx.h"

namespace tvm {
namespace runtime {
namespace vm {

//---------------------------------------------
// VM Closure object
//---------------------------------------------

VMClosure::VMClosure(ffi::String func_name, ffi::Function impl) {
  auto ptr = ffi::make_object<VMClosureObj>();
  ptr->func_name = func_name;
  ptr->impl = std::move(impl);
  data_ = std::move(ptr);
}

/*!
 * \brief Create another ffi::Function with last arguments already bound to last_args.
 * \param func The input func, can be a VMClosure or ffi::Function.
 * \param last_args The arguments to bound to in the end of the function.
 * \note The new function takes in arguments and append the last_args in the end.
 */
ffi::Function VMClosure::BindLastArgs(ffi::Function func, std::vector<ffi::Any> last_args) {
  return ffi::Function([func, last_args](ffi::PackedArgs args, ffi::Any* rv) {
    std::vector<ffi::AnyView> packed_args(args.size() + last_args.size());
    std::copy(args.data(), args.data() + args.size(), packed_args.data());
    for (size_t i = 0; i < last_args.size(); ++i) {
      packed_args[args.size() + i] = last_args[i];
    }
    func.CallPacked(ffi::PackedArgs(packed_args.data(), packed_args.size()), rv);
  });
}

//-----------------------------------------------------------
// Utility functions.
//-----------------------------------------------------------
// Use the args after `starting_arg_idx` as a series of indices into `obj`,
// indexing into nested Array and returning the final indexed object.
ffi::Any IndexIntoNestedObject(ffi::Any obj, ffi::PackedArgs args, int starting_arg_idx) {
  for (int i = starting_arg_idx; i < args.size(); i++) {
    // the object must be an Array to be able to index into it
    if (!obj.as<ffi::ArrayObj>()) {
      TVM_FFI_THROW(ValueError) << "Attempted to index into an object that is not an Array.";
    }
    int index = args[i].cast<int>();
    auto arr = obj.cast<ffi::Array<ffi::Any>>();
    // make sure the index is in bounds
    if (index >= static_cast<int>(arr.size())) {
      TVM_FFI_THROW(IndexError) << "Invalid index (" << index << " >= " << arr.size() << ").";
    }
    obj = arr[index];
  }
  return obj;
}

ffi::Tensor ConvertTensorToDevice(ffi::Tensor src, const DLDevice& dev, TensorMemoryPool* pool,
                                  MemoryScope scope = {}) {
  TVM_FFI_ICHECK(src.defined());
  if (src.device().device_type == dev.device_type && src.device().device_id == dev.device_id) {
    return src;
  } else {
    auto res = pool->Empty(src.shape(), src.dtype(), dev, scope);
    tvm::runtime::CopyFromTo(src, res);
    return res;
  }
}

Any ConvertObjectToDevice(Any src, const Device& dev, TensorMemoryPool* pool,
                          MemoryScope scope = {}) {
  if (src.as<ffi::Tensor::ContainerType>()) {
    return ConvertTensorToDevice(src.cast<ffi::Tensor>(), dev, pool, scope);
  } else if (src.as<ffi::ArrayObj>()) {
    std::vector<Any> ret;
    auto arr = src.cast<ffi::Array<Any>>();
    for (size_t i = 0; i < arr.size(); i++) {
      ret.push_back(ConvertObjectToDevice(arr[i], dev, pool));
    }
    return ffi::Array<Any>(ret.begin(), ret.end());
  } else {
    return src;
  }
}

ffi::Any ConvertArgToDevice(ffi::AnyView input, Device dev, TensorMemoryPool* pool) {
  // in terms of memory-behavior.
  // To be extra careful, we copy DLTensor.
  // The developer can still explicitly allocate Tensor
  // in TVM Native API or Tensor::FromDLPack to regain zero copy behavior.
  ffi::Any ret;
  if (auto opt_obj = input.as<ffi::ObjectRef>()) {
    ret = ConvertObjectToDevice(opt_obj.value(), dev, pool);
  } else if (auto opt_dltensor = input.as<DLTensor*>()) {
    DLTensor* tensor = opt_dltensor.value();
    ffi::ShapeView shape(tensor->shape, tensor->ndim);
    auto dst = pool->Empty(shape, tensor->dtype, dev);
    tvm::runtime::CopyFromTo(tensor, dst);
    ret = dst;
  } else {
    ret = input;
  }
  return ret;
}

ffi::Any ConvertRegToDevice(ffi::Any input, Device dev, TensorMemoryPool* pool,
                            MemoryScope scope = {}) {
  ffi::Any ret;
  if (auto opt_obj = input.as<ffi::ObjectRef>()) {
    ret = ConvertObjectToDevice(opt_obj.value(), dev, pool, scope);
  } else {
    ret = input;
  }
  return ret;
}

//-----------------------------------------------------------
// VM implementations.
//-----------------------------------------------------------
/*!
 * \brief The register type.
 */
using RegType = ffi::Any;

/*!
 * \brief A representation of a stack frame.
 *
 * A stack frame is a record containing the information needed
 * to restore the caller's virtual machine state after returning
 * from a function call.
 */
struct VMFrame {
  /*! \brief The return program counter. */
  Index return_pc;
  /*! \brief Statically allocated space for objects */
  std::vector<RegType> register_file;
  /*! \brief Register in caller's frame to put return value */
  RegName caller_return_register;
  /*! \brief Temporary argument tcode stack for packed func call. */
  std::vector<ffi::AnyView> call_args;

  VMFrame(Index pc, Index register_file_size)
      : return_pc(pc), register_file(register_file_size), caller_return_register(0) {}

  void Clear() {
    this->caller_return_register = 0;
    this->call_args.clear();
    for (RegType& reg : register_file) {
      reg = nullptr;
    }
  }

  void ResetForRecycle(Index pc, Index register_file_size) {
    this->return_pc = pc;
    this->register_file.resize(register_file_size);
  }
};

class VirtualMachineImpl : public VirtualMachine {
 public:
  //---------------------------------------------------
  // Public facing functions overloading
  //---------------------------------------------------
  void LoadExecutable(ffi::ObjectPtr<VMExecutable> exec) final;
  void Init(const std::vector<Device>& devices) final;
  VMClosure GetClosure(const ffi::String& func_name) final {
    return this->GetClosureInternal(func_name, false).value();
  }
  void InvokeClosurePacked(const ffi::ObjectRef& closure_or_packedfunc, ffi::PackedArgs args,
                           ffi::Any* rv) final;
  void SetInstrument(ffi::Function instrument) final { this->instrument_ = instrument; }

  //---------------------------------------------------
  // Functions in the vtable of Module
  //---------------------------------------------------
  void _Init(ffi::PackedArgs args, ffi::Any* rv);
  void _SaveClosure(ffi::PackedArgs args, ffi::Any* rv);
  void _InvokeClosure(ffi::PackedArgs args, ffi::Any* rv);
  void _InvokeClosureStateful(std::string func_name);
  void _SetInstrument(ffi::PackedArgs args, ffi::Any* rv);
  void _GetOutputArity(ffi::PackedArgs args, ffi::Any* rv);
  void _GetOutput(ffi::PackedArgs args, ffi::Any* rv);
  void _SetInputWithoutParamModule(ffi::PackedArgs args, ffi::Any* rv);
  void _SetInputWithParamModule(ffi::PackedArgs args, ffi::Any* rv);
  int _GetFunctionArity(std::string func_name);
  std::string _GetFunctionParamName(std::string func_name, int index);
  ffi::Function _LookupFunction(const ffi::String& name);

  TVM_MODULE_VTABLE_BEGIN("relax.VirtualMachine");
  TVM_MODULE_VTABLE_ENTRY_PACKED("vm_initialization", &VirtualMachineImpl::_Init);
  TVM_MODULE_VTABLE_ENTRY_PACKED("save_function", &VirtualMachineImpl::_SaveClosure);
  TVM_MODULE_VTABLE_ENTRY_PACKED("invoke_closure", &VirtualMachineImpl::_InvokeClosure);
  TVM_MODULE_VTABLE_ENTRY("invoke_stateful", &VirtualMachineImpl::_InvokeClosureStateful);
  TVM_MODULE_VTABLE_ENTRY_PACKED("set_instrument", &VirtualMachineImpl::_SetInstrument);
  TVM_MODULE_VTABLE_ENTRY_PACKED("get_output_arity", &VirtualMachineImpl::_GetOutputArity);
  TVM_MODULE_VTABLE_ENTRY_PACKED("get_output", &VirtualMachineImpl::_GetOutput);
  TVM_MODULE_VTABLE_ENTRY_PACKED("set_input", &VirtualMachineImpl::_SetInputWithoutParamModule);
  TVM_MODULE_VTABLE_ENTRY_PACKED("set_input_with_param_module",
                                 &VirtualMachineImpl::_SetInputWithParamModule);
  TVM_MODULE_VTABLE_ENTRY("get_function_arity", &VirtualMachineImpl::_GetFunctionArity);
  TVM_MODULE_VTABLE_ENTRY("get_function_param_name", &VirtualMachineImpl::_GetFunctionParamName);
  TVM_MODULE_VTABLE_END_WITH_DEFAULT(&VirtualMachineImpl::_LookupFunction);

  //--------------------------------------------------
  // Additional support arguments functions for VM
  //--------------------------------------------------
  /*!
   * \brief Internal implementation of GetClosure which also allow none.
   * \param func_name The name of the function.
   * \param allow_missing Whether none is allowed.
   * \return The result
   */
  ffi::Optional<VMClosure> GetClosureInternal(const ffi::String& func_name, bool allow_missing);

  /*!
   * \brief Set inputs to a function.
   * \param func_name The function name.
   * \param args args[offset:] are arguments to the function. If the arguments are not of the
   * correct device for the function, they will be copied to the device.
   * \param with_param_module If set to true, the last argument will be a module and can be invoked
   *        to get the argument, this is mainly used for debugging purposes and setting composite
   * objects. \note This interface works when using VM over RPC by internally converting Tensor in
   * the arguments to DLTensor, which is supported in RPC where remote could only have a minimal C
   * runtime.
   */
  void SetInput(std::string func_name, bool with_param_module, ffi::PackedArgs args);

  /*!
   * \brief Look up whether the VM has a function by the given name.
   * \param func_name the function's name
   * \return The function, if it exists. Logs a fatal error if not.
   */
  VMFuncInfo LookupVMFuncInfo(const std::string& func_name);

  /*!
   * \brief Look up whether the VM has outputs for the given function.
   * \param func_name the function's name
   * \return The output, if it exists. Logs a fatal error if not.
   */
  RegType LookupVMOutput(const std::string& func_name);

  /*!
   * \brief Fully bind the argument of a global function and save it in the env.
   * \param func_name The global function name to be saved.
   * \param save_name The saved name of the function.
   * \param include_return Whether forward the return value, set it to false allows
   *        us to ignore forwarding return value, which can be helpful to do benchmarking
   *        in RPC environment when return value is complicated Array.
   *
   * \param args The arguments to bound to the function.
   * \note This function is used by RPC server to help benchmarking.
   */
  void SaveClosure(const ffi::String& func_name, const ffi::String& save_name, bool include_return,
                   ffi::PackedArgs args);
  /*!
   * \brief Internal function to invoke a closure.
   * \param closure_or_packed The closure to be invoked.
   * \param args The arguments to the function.
   * \return The result value.
   */
  RegType InvokeClosureInternal(const ffi::ObjectRef& closure_or_packed,
                                const std::vector<RegType>& args);
  /*!
   * \brief Invoke a VM function by interpreting bytecode.
   * \param fidx The function index.
   * \param args The arguments to the function.
   * \return The object representing the result.
   */
  RegType InvokeBytecode(Index fidx, const std::vector<RegType>& args);

 protected:
  /*!
   * \brief Get function by querying all of the current module's imports.
   * \param name The name of the function.
   * \return The result function, can return ffi::Function(nullptr) if nothing is found.
   */
  ffi::Optional<ffi::Function> GetFuncFromImports(const ffi::String& name) {
    for (auto& lib : this->imports_) {
      if (auto opt_func = lib.cast<ffi::Module>()->GetFunction(name, true)) {
        return *opt_func;
      }
    }
    return std::nullopt;
  }
  /*!
   * \brief Initialize function pool.
   */
  void InitFuncPool();

  /*!
   * \brief A RAII wrapper that pushes and pops VM frames.
   */
  class FrameGuard {
   public:
    VirtualMachineImpl* vm;
    explicit FrameGuard(VirtualMachineImpl* vm, std::unique_ptr<VMFrame> frame) : vm(vm) {
      vm->frames_.emplace_back(std::move(frame));
    }
    ~FrameGuard() {
      TVM_FFI_ICHECK_GT(vm->frames_.size(), 0);
      vm->pc_ = vm->frames_.back()->return_pc;
      vm->frames_.back()->Clear();
      vm->frame_free_list_.emplace_back(std::move(vm->frames_.back()));
      vm->frames_.pop_back();
    }
  };
  //-------------------------------------------------
  // Instruction interpretations.
  //-------------------------------------------------
  /*!
   * \brief Push a call frame onto the call stack.
   * \param ret_pc The program counter to return to.
   * \param vm_func The function to be pushed to the call stack.
   * \return A RAII wrapper that pops the frame when going out of scope.
   */
  FrameGuard PushFrame(Index ret_pc, const VMFuncInfo& vm_func) {
    std::unique_ptr<VMFrame> new_frame;
    if (!frame_free_list_.empty()) {
      new_frame = std::move(frame_free_list_.back());
      frame_free_list_.pop_back();
      new_frame->ResetForRecycle(ret_pc, vm_func.register_file_size);
    } else {
      new_frame = std::make_unique<VMFrame>(ret_pc, vm_func.register_file_size);
    }
    return FrameGuard(this, std::move(new_frame));
  }
  /*!
   * \brief Write to a VM register.
   * \param frame current vm frame.
   * \param reg The register to write to.
   * \param obj The object to write to.
   */
  TVM_FFI_INLINE void WriteRegister(VMFrame* frame, RegName reg, const RegType& obj) {
    TVM_FFI_ICHECK_LT(reg, frame->register_file.size());
    frame->register_file[reg] = obj;
  }
  /*!
   * \brief Read a VM register.
   * \param frame current vm frame.
   * \param reg The register to read from.
   * \return The value of the register.
   */
  TVM_FFI_INLINE RegType ReadRegister(VMFrame* frame, RegName reg) {
    if (reg < Instruction::kBeginSpecialReg) {
      return frame->register_file[reg];
    }
    RegType ret;
    if (reg == Instruction::kVoidRegister) {
      ret = nullptr;
    } else {
      TVM_FFI_ICHECK_EQ(reg, Instruction::kVMRegister);
      // per convention, ctx ptr must be VirtualMachine* casted to void.
      // this and VirtualMachine* may or may not be the same
      // do first cast to VirtualMachine* then to void*
      ret = static_cast<void*>(static_cast<VirtualMachine*>(this));
    }
    return ret;
  }
  /*!
   * \brief Run call instruction.
   * \param curr_frame The current frame.
   * \param inst The call instruction.
   */
  virtual void RunInstrCall(VMFrame* curr_frame, Instruction inst);

  /*! \brief Run VM dispatch loop. */
  void RunLoop();

  /*!
   * \brief Retrieve the name of the function identified by the given index.
   * \param idx The index into the VM executable function table.
   * \return The name of the function.
   */
  const std::string& GetFuncName(int idx) { return exec_->func_table[idx].name; }

  /*!
   * \brief Retrieve the inputs for a function.
   * \param func_name The name of the function.
   * \return The function inputs.
   */
  const std::vector<RegType>& GetInputsFor(const std::string& func_name) {
    return inputs_[func_name];
  }

  void ClearInputsFor(const std::string& func_name) { inputs_.erase(func_name); }

  //--------------------------------------------------------
  // Internal states for execution.
  //--------------------------------------------------------
  /*! \brief The loaded executable. */
  ffi::ObjectPtr<VMExecutable> exec_;
  /*! \brief The global constant pool */
  std::vector<ffi::Any> const_pool_;
  /*!
   * \brief Function pool to cache functions in func_table
   */
  std::vector<ffi::Any> func_pool_;
  //--------------------------------------------------------
  // Executor interface support
  //--------------------------------------------------------
  /*! \brief The function name to input register mapping. */
  std::unordered_map<std::string, std::vector<RegType>> inputs_;
  /*! \brief The function name to output register. */
  std::unordered_map<std::string, RegType> outputs_;
  /*! \brief A store of closures created by `save_function`. */
  std::unordered_map<std::string, VMClosure> saved_closures_;
  //------------------------------------------------------------
  // VM Instruction execution.
  //------------------------------------------------------------
  /*!
   * \brief The current stack of call frames.
   * \note: Use unique ptr to avoid re-allocation and copy when frames_ get resized.
   */
  std::vector<std::unique_ptr<VMFrame>> frames_;
  /*!
   * \brief A free list of frame
   */
  std::vector<std::unique_ptr<VMFrame>> frame_free_list_;

  /*! \brief The virtual machine PC. */
  Index pc_{0};
  /*! \brief The special return register. */
  RegType return_value_;
  /*!\ brief instrument function. */
  ffi::Function instrument_ = nullptr;
};

void VirtualMachineImpl::LoadExecutable(ffi::ObjectPtr<VMExecutable> exec) {
  this->exec_ = exec;
  this->imports_ = exec->imports();
}

void VirtualMachineImpl::Init(const std::vector<Device>& devices) {
  this->devices.reserve(devices.size());
  this->memory_pools.reserve(devices.size());
  for (Device device : devices) {
    this->devices.push_back(device);
    this->memory_pools.push_back(CreateTensorMemoryPool(device));
  }
  // Setup constant sections.
  this->const_pool_.reserve(exec_->constants.size());
  for (size_t i = 0; i < exec_->constants.size(); ++i) {
    if (auto opt_nd = exec_->constants[i].as<ffi::Tensor>(); opt_nd && opt_nd->defined()) {
      this->const_pool_.push_back(ConvertRegToDevice(
          opt_nd.value(), devices[0], memory_pools[0].get(), exec_->memory_scopes[i]));
    } else {
      this->const_pool_.push_back(exec_->constants[i]);
    }
  }
  // Setup function sections.
  this->InitFuncPool();
}

VMFuncInfo VirtualMachineImpl::LookupVMFuncInfo(const std::string& func_name) {
  TVM_FFI_ICHECK(exec_) << "The executable is not created yet.";
  auto it = this->exec_->func_map.find(func_name);
  TVM_FFI_CHECK(it != this->exec_->func_map.end(), ValueError) << "Unknown function: " << func_name;

  return exec_->func_table[it->second];
}

RegType VirtualMachineImpl::LookupVMOutput(const std::string& func_name) {
  if (!outputs_.count(func_name)) {
    TVM_FFI_THROW(ValueError) << "No output saved for call of \"" << func_name
                              << "\"; use `invoke_stateful` to call it first.";
  }
  return outputs_[func_name];
}

void VirtualMachineImpl::SetInput(std::string func_name, bool with_param_module,
                                  ffi::PackedArgs args) {
  const auto& m = exec_->func_map;
  if (m.find(func_name) != m.end()) {
    Index gf_idx = m.at(func_name);
    const VMFuncInfo& vm_func = exec_->func_table[gf_idx];
    size_t params_num = vm_func.num_args;
    TVM_FFI_ICHECK_EQ(args.size(), params_num)
        << "The number of provided parameters doesn't match the number of arguments for";
    std::vector<RegType> func_args(params_num);
    for (int i = 0; i < args.size(); ++i) {
      if (with_param_module && i == args.size() - 1) {
        // call param func to get the arguments(usually corresponds to param pack.)
        func_args[i] = (args[i].cast<ffi::Module>())->GetFunction("get_params").value()();
      } else {
        func_args[i] = ConvertArgToDevice(args[i], devices[0], memory_pools[0].get());
      }
    }
    inputs_[func_name] = func_args;
  } else {
    TVM_FFI_THROW(ValueError) << "Unknown function: " << func_name;
  }
}

//------------------------------------------
// Closure handling
//------------------------------------------
void VirtualMachineImpl::InvokeClosurePacked(const ffi::ObjectRef& closure_or_packedfunc,
                                             ffi::PackedArgs args, ffi::Any* rv) {
  // run packed call if it is a packed func.
  if (auto* packed = closure_or_packedfunc.as<ffi::Function::ContainerType>()) {
    packed->CallPacked(args.data(), args.size(), rv);
    return;
  }
  // run closure call.
  auto* clo = closure_or_packedfunc.as<VMClosureObj>();
  TVM_FFI_ICHECK(clo != nullptr) << "Function expects a closure or ffi::Function ";

  std::vector<ffi::AnyView> packed_args(args.size() + 1);
  // per convention, ctx ptr must be VirtualMachine* casted to void.
  // this and VirtualMachine* may or maynot be the same
  // do first cast to VirtualMachine* then to void*
  packed_args[0] = static_cast<void*>(static_cast<VirtualMachine*>(this));
  std::copy(args.data(), args.data() + args.size(), packed_args.begin() + 1);
  {
    support::NVTXScopedRange scope("RelaxVM: " + clo->func_name);
    clo->impl.CallPacked(ffi::PackedArgs(packed_args.data(), packed_args.size()), rv);
  }
}

// internal variant version of invoke closurepacked
RegType VirtualMachineImpl::InvokeClosureInternal(const ffi::ObjectRef& closure_or_packed,
                                                  const std::vector<RegType>& args) {
  RegType ret;
  auto* packed = closure_or_packed.as<ffi::Function::ContainerType>();
  auto* clo = closure_or_packed.as<VMClosureObj>();
  int clo_offset = clo != nullptr ? 1 : 0;

  std::vector<ffi::AnyView> packed_args(args.size() + clo_offset);

  if (clo != nullptr) {
    packed_args[0] = static_cast<void*>(static_cast<VirtualMachine*>(this));
  }
  for (size_t i = 0; i < args.size(); ++i) {
    packed_args[i + clo_offset] = args[i];
  }

  if (packed != nullptr) {
    packed->CallPacked(packed_args.data(), packed_args.size(), &ret);
  } else {
    TVM_FFI_ICHECK(clo != nullptr);
    clo->impl.CallPacked(packed_args.data(), packed_args.size(), &ret);
  }
  return ret;
}

void VirtualMachineImpl::SaveClosure(const ffi::String& func_name, const ffi::String& save_name,
                                     bool include_return, ffi::PackedArgs args) {
  VMClosure clo = this->GetClosure(func_name);
  std::vector<RegType> inputs(args.size());
  for (int i = 0; i < args.size(); ++i) {
    inputs[i] = ConvertArgToDevice(args[i], this->devices[0], this->memory_pools[0].get());
  }
  ffi::Function impl = VMClosure::BindLastArgs(clo->impl, inputs);
  if (!include_return) {
    impl = ffi::Function([impl](ffi::PackedArgs args, ffi::Any* rv) {
      ffi::Any temp;
      impl.CallPacked(args, &temp);
    });
  }
  saved_closures_.insert_or_assign(save_name, VMClosure(save_name, impl));
}

ffi::Optional<VMClosure> VirtualMachineImpl::GetClosureInternal(const ffi::String& func_name,
                                                                bool allow_missing) {
  // look up saved closures.
  auto saved_it = saved_closures_.find(func_name);
  if (saved_it != saved_closures_.end()) {
    return saved_it->second;
  }
  auto it = exec_->func_map.find(func_name);
  if (it == exec_->func_map.end()) {
    if (allow_missing) return std::nullopt;
    TVM_FFI_THROW(ValueError) << "Unknown function: " << func_name;
  }

  Index gf_idx = it->second;
  const VMFuncInfo& finfo = exec_->func_table[gf_idx];

  TVM_FFI_ICHECK(finfo.kind == VMFuncInfo::FuncKind::kVMFunc)
      << "Cannot support closure with function kind " << static_cast<int>(finfo.kind);
  // NOTE: should not capture strong ref to self and avoid cyclic ref.
  auto impl = ffi::Function([gf_idx](ffi::PackedArgs args, ffi::Any* rv) {
    // Per convention, ctx ptr is a VirtualMachine*
    VirtualMachine* ctx_ptr = static_cast<VirtualMachine*>(args[0].cast<void*>());

    std::vector<RegType> inputs(args.size() - 1);
    for (size_t i = 0; i < inputs.size(); ++i) {
      inputs[i] = args[i + 1];
    }
    *rv = static_cast<VirtualMachineImpl*>(ctx_ptr)->InvokeBytecode(gf_idx, inputs);
  });
  return VMClosure(func_name, impl);
}

//--------------------------------------------------------------------
// Instruction interpretations.
//--------------------------------------------------------------------
RegType VirtualMachineImpl::InvokeBytecode(Index gf_idx, const std::vector<RegType>& args) {
  const VMFuncInfo& gfunc = exec_->func_table[gf_idx];
  TVM_FFI_ICHECK(gfunc.kind == VMFuncInfo::FuncKind::kVMFunc);

  // Get the curr instr which might be a potential caller.
  Instruction curr_instr = exec_->GetInstruction(pc_);
  auto guard = PushFrame(this->pc_, gfunc);
  // Get new frame and set the caller info.
  VMFrame* curr_frame = frames_.back().get();
  if (curr_instr.op == Opcode::Call) {
    curr_frame->caller_return_register = curr_instr.dst;
  }

  // load arguments to the register file
  TVM_FFI_ICHECK_EQ(static_cast<size_t>(gfunc.num_args), args.size())
      << "Invoking function " << gfunc.name << " expects " << gfunc.num_args << " arguments" <<
      [&]() {
        std::stringstream ss;
        if (gfunc.param_names.size()) {
          ss << " (";
          for (size_t i = 0; i < gfunc.param_names.size(); i++) {
            if (i) {
              ss << ", ";
            }
            ss << gfunc.param_names[i];
          }
          ss << ")";
        }
        return ss.str();
      }()
      << ", but " << args.size() << " arguments were provided.";
  for (size_t i = 0; i < args.size(); ++i) {
    WriteRegister(frames_.back().get(), i, args[i]);
  }
  // set program counter
  pc_ = gfunc.start_instr;
  RunLoop();
  return return_value_;
}

void VirtualMachineImpl::InitFuncPool() {
  func_pool_.resize(exec_->func_table.size());

  for (size_t func_index = 0; func_index < exec_->func_table.size(); ++func_index) {
    const VMFuncInfo& info = exec_->func_table[func_index];
    if (info.kind == VMFuncInfo::FuncKind::kPackedFunc) {
      // only look through imports first
      ffi::Optional<ffi::Function> func = GetFuncFromImports(info.name);
      if (!func.has_value()) {
        const auto p_func = tvm::ffi::Function::GetGlobal(info.name);
        if (p_func.has_value()) func = *p_func;
      }
      TVM_FFI_ICHECK(func.has_value())
          << "Error: Cannot find ffi::Function " << info.name
          << " in either Relax VM kernel library, or in TVM runtime ffi::Function registry, or in "
             "global Relax functions of the VM executable";
      func_pool_[func_index] = *func;

    } else {
      TVM_FFI_ICHECK(info.kind == VMFuncInfo::FuncKind::kVMFunc);
      auto clo = this->GetClosure(info.name);
      func_pool_[func_index] = clo;
    }
  }
}

void VirtualMachineImpl::RunInstrCall(VMFrame* curr_frame, Instruction instr) {
  DLOG(INFO) << "\n  pc = " << pc_ << ", execute: " << GetFuncName(instr.func_idx);
  int args_begin_offset = instrument_ != nullptr ? 4 : 0;
  // Use the call arg stack from the current frame to increase reuse
  // and avoid re-allocation
  curr_frame->call_args.resize(args_begin_offset + instr.num_args);

  // NOTE: no changes and resize to those vector ref(otherwise can leads to segfault)
  //       in the remainder part of the function.
  std::vector<ffi::AnyView>& call_args = curr_frame->call_args;

  for (Index i = 0; i < instr.num_args; ++i) {
    Instruction::Arg arg = instr.args[i];
    int arg_index = args_begin_offset + i;
    switch (arg.kind()) {
      case Instruction::ArgKind::kRegister: {
        call_args[arg_index] = ReadRegister(curr_frame, arg.value());
        break;
      }
      case Instruction::ArgKind::kImmediate: {
        call_args[arg_index] = arg.value();
        break;
      }
      case Instruction::ArgKind::kConstIdx: {
        call_args[arg_index] = this->const_pool_[arg.value()];
        break;
      }
      case Instruction::ArgKind::kFuncIdx: {
        TVM_FFI_ICHECK_LT(static_cast<size_t>(arg.value()), this->func_pool_.size());
        call_args[arg_index] = this->func_pool_[arg.value()];
        break;
      }
      default: {
        TVM_FFI_THROW(ValueError) << "Unknown argument kind: " << int(arg.kind());
      }
    }
  }
  ffi::PackedArgs args(call_args.data() + args_begin_offset, instr.num_args);
  ffi::Any ret;

  TVM_FFI_ICHECK_LT(static_cast<size_t>(instr.func_idx), this->func_pool_.size());

  if (instrument_ == nullptr) {
    this->InvokeClosurePacked(func_pool_[instr.func_idx].cast<ffi::ObjectRef>(), args, &ret);
  } else {
    // insert light-weight instrument callback
    call_args[0] = func_pool_[instr.func_idx];
    call_args[1] = GetFuncName(instr.func_idx);
    call_args[2] = true;
    call_args[3] = nullptr;

    ffi::Any rv;
    // store dtype to str since py callback cannot handle dtype atm.
    std::vector<std::unique_ptr<std::string>> temp_dtype;
    for (int i = 0; i < instr.num_args; ++i) {
      if (call_args[i + args_begin_offset].type_index() == ffi::TypeIndex::kTVMFFIDataType) {
        std::string str_dtype =
            ffi::DLDataTypeToString(call_args[i + args_begin_offset].cast<DLDataType>());
        temp_dtype.emplace_back(std::make_unique<std::string>(str_dtype));
        call_args[i + args_begin_offset] = *temp_dtype.back();
      }
    }
    int ret_kind = static_cast<int>(VMInstrumentReturnKind::kNoOp);
    instrument_.CallPacked(call_args.data(), call_args.size(), &rv);
    if (auto opt_int = rv.try_cast<int64_t>()) {
      ret_kind = opt_int.value();
    }
    if (ret_kind != static_cast<int>(VMInstrumentReturnKind::kSkipRun)) {
      this->InvokeClosurePacked(func_pool_[instr.func_idx].cast<ffi::ObjectRef>(), args, &ret);
      call_args[2] = false;
      call_args[3] = ret;
      instrument_.CallPacked(call_args.data(), call_args.size(), &rv);
    }
  }

  // save the return value to the register
  // saving to special register is a NOP
  if (instr.dst < Instruction::kBeginSpecialReg) {
    WriteRegister(curr_frame, instr.dst, ret);
  }
  // increment pc
  pc_++;
}

void VirtualMachineImpl::RunLoop() {
  VMFrame* curr_frame = frames_.back().get();

  while (true) {
    TVM_FFI_ICHECK_LT(static_cast<size_t>(pc_), exec_->instr_offset.size())
        << "run into invalid section";
    Instruction instr = exec_->GetInstruction(pc_);
    switch (instr.op) {
      case Opcode::Call: {
        this->RunInstrCall(curr_frame, instr);
        break;
      }
      case Opcode::Ret: {
        // If we have hit the point from which we started
        // running, we should return to the caller breaking
        // the dispatch loop.
        return_value_ = ReadRegister(curr_frame, instr.result);
        RegName caller_return_register = curr_frame->caller_return_register;
        if (frames_.size() <= 1) {
          // directly return if no other frame in the call stack.
        } else {
          // return from a local call.
          // Update the current frame to be the parent frame.
          VMFrame* parent_frame = frames_.end()[-2].get();
          WriteRegister(parent_frame, caller_return_register, return_value_);
        }
        return;
      }
      case Opcode::Goto: {
        pc_ += instr.pc_offset;
        break;
      }
      case Opcode::If: {
        int64_t cond_val = ReadRegister(curr_frame, instr.cond).cast<int64_t>();
        if (cond_val != 0) {
          pc_++;
        } else {
          TVM_FFI_ICHECK_GT(instr.false_offset, 1);
          pc_ += instr.false_offset;
        }
        break;
      }
    }
  }
}

ffi::ObjectPtr<VirtualMachine> VirtualMachine::Create() {
  return ffi::make_object<VirtualMachineImpl>();
}

//--------------------------------------------------------------------
// FFI related code
//--------------------------------------------------------------------

void VirtualMachineImpl::_Init(ffi::PackedArgs args, ffi::Any* rv) {
  TVM_FFI_ICHECK_EQ(args.size() % 2, 0);
  std::vector<Device> devices;
  for (int i = 0; i < args.size(); i += 2) {
    int device_type = args[i].cast<int>();
    int device_id = args[i + 1].cast<int>();
    devices.push_back(Device{DLDeviceType(device_type), device_id});
  }
  this->Init(devices);
}

void VirtualMachineImpl::_SaveClosure(ffi::PackedArgs args, ffi::Any* rv) {
  TVM_FFI_ICHECK_GE(args.size(), 3);
  std::string func_name = args[0].cast<std::string>();
  this->SaveClosure(func_name, args[1].cast<ffi::String>(), args[2].cast<bool>(), args.Slice(3));
}

void VirtualMachineImpl::_InvokeClosure(ffi::PackedArgs args, ffi::Any* rv) {
  this->InvokeClosurePacked(args[0].cast<ffi::ObjectRef>(), args.Slice(1), rv);
}

void VirtualMachineImpl::_InvokeClosureStateful(std::string func_name) {
  const std::unordered_map<std::string, Index>& m = this->exec_->func_map;
  if (m.find(func_name) == m.end()) {
    TVM_FFI_THROW(ValueError) << "Unknown function: " << func_name;
  }
  if (!inputs_.count(func_name)) {
    TVM_FFI_THROW(ValueError) << "No inputs set for stateful call of " << func_name
                              << "; use `set_input` first.";
    return;
  }
  outputs_[func_name] = this->InvokeClosureInternal(
      func_pool_[m.at(func_name)].cast<ffi::ObjectRef>(), inputs_[func_name]);
}

void VirtualMachineImpl::_SetInstrument(ffi::PackedArgs args, ffi::Any* rv) {
  if (args[0].as<ffi::Function>()) {
    this->SetInstrument(args[0].cast<ffi::Function>());
  } else {
    ffi::String func_name = args[0].cast<ffi::String>();
    const auto factory = tvm::ffi::Function::GetGlobal(func_name);
    TVM_FFI_ICHECK(factory.has_value()) << "Cannot find factory " << func_name;
    ffi::Any rv;
    factory->CallPacked(args.Slice(1), &rv);
    this->SetInstrument(rv.cast<ffi::Function>());
  }
}

void VirtualMachineImpl::_GetOutputArity(ffi::PackedArgs args, ffi::Any* rv) {
  std::string func_name = args[0].cast<std::string>();
  RegType out = LookupVMOutput(func_name);
  ffi::Any obj = IndexIntoNestedObject(out, args, 1);
  if (const auto* arr = obj.as<ffi::ArrayObj>()) {
    *rv = static_cast<int>(arr->size());
  } else {
    *rv = -1;
  }
}

void VirtualMachineImpl::_GetOutput(ffi::PackedArgs args, ffi::Any* rv) {
  std::string func_name = args[0].cast<std::string>();
  RegType out = LookupVMOutput(func_name);
  ffi::Any obj = IndexIntoNestedObject(out, args, 1);
  if (obj.as<ffi::ArrayObj>()) {
    TVM_FFI_THROW(ValueError) << "`get_output` cannot return a tuple for RPC compatibility. "
                                 "Please specify another index argument.";
    return;
  }
  *rv = obj;
}

void VirtualMachineImpl::_SetInputWithoutParamModule(ffi::PackedArgs args, ffi::Any* rv) {
  std::string func_name = args[0].cast<std::string>();
  this->SetInput(func_name, false, args.Slice(1));
}

void VirtualMachineImpl::_SetInputWithParamModule(ffi::PackedArgs args, ffi::Any* rv) {
  std::string func_name = args[0].cast<std::string>();
  this->SetInput(func_name, true, args.Slice(1));
}

int VirtualMachineImpl::_GetFunctionArity(std::string func_name) {
  const VMFuncInfo& vm_func = LookupVMFuncInfo(func_name);
  return vm_func.param_names.size();
}

std::string VirtualMachineImpl::_GetFunctionParamName(std::string func_name, int index) {
  const VMFuncInfo& vm_func = LookupVMFuncInfo(func_name);
  if (static_cast<size_t>(index) >= vm_func.param_names.size()) {
    TVM_FFI_THROW(ValueError) << "Invalid index for " << func_name << " (" << index << " out of "
                              << vm_func.param_names.size() << ")";
  }
  return vm_func.param_names[index];
}

ffi::Function VirtualMachineImpl::_LookupFunction(const ffi::String& name) {
  if (ffi::Optional<VMClosure> opt = this->GetClosureInternal(name, true)) {
    return ffi::Function([clo = opt.value(), _self = ffi::GetRef<ffi::Module>(this)](
                             ffi::PackedArgs args, ffi::Any* rv) -> void {
      auto* self = const_cast<VirtualMachineImpl*>(_self.as<VirtualMachineImpl>());
      TVM_FFI_ICHECK(self);
      self->InvokeClosurePacked(clo, args, rv);
    });
  }
  return ffi::Function(nullptr);
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
