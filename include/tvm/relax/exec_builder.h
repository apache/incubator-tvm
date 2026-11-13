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
 * \file tvm/relax/exec_builder.h
 */
#ifndef TVM_RELAX_EXEC_BUILDER_H_
#define TVM_RELAX_EXEC_BUILDER_H_

#include <tvm/base.h>
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/vm/bytecode.h>
#include <tvm/runtime/vm/executable.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relax {

namespace vm = tvm::runtime::vm;

class ExecBuilder;

/*!
 * \brief A builder provides api to build VM executable with instructions.
 */
class ExecBuilderObj : public ffi::Object {
 public:
  /*!
   * \brief Declare a function, it is OK to have multiple declarations.
   * \param func The function name.
   * \param kind The kind of the function.
   */
  void DeclareFunction(const std::string& func, vm::VMFuncInfo::FuncKind kind);
  /*!
   * \brief To annotate the start of a vm function.
   * \param func The function name.
   * \param num_inputs The number of inputs.
   * \param param_names The function parameter names.
   * \param kind The kind of the function.
   * \param init_register_size Initial setting of register file size.
   */
  void EmitFunction(const std::string& func, int64_t num_inputs,
                    ffi::Optional<ffi::Array<ffi::String>> param_names,
                    vm::VMFuncInfo::FuncKind kind = vm::VMFuncInfo::FuncKind::kVMFunc,
                    int64_t init_register_size = 0);
  /*!
   * \brief Annotate the end of a vm function.
   * \param func The function name.
   */
  void EndFunction(const std::string& func);
  /*!
   * \brief Emit a call instruction for a packed function.
   * \param func The packed function name.
   * \param args The arguments of the function.
   * \param ret The return register.
   */
  void EmitCall(const std::string& func, std::vector<vm::Instruction::Arg> args, vm::RegName ret);
  /*!
   * \brief Emit a call instruction with func as argument.
   * \param func The packed function index.
   * \param args The arguments of the function.
   * \param ret The return register.
   */
  void EmitCall(vm::Instruction::Arg func, std::vector<vm::Instruction::Arg> args, vm::RegName ret);
  /*!
   * \brief Emit a ret instruction.
   * \param result The return result.
   * \note result must be a register.
   */
  void EmitRet(vm::Instruction::Arg result);
  /*!
   * \brief Emit a goto instruction.
   * \param pc_offset The program counter offset as the jump offset.
   */
  void EmitGoto(vm::Index pc_offset);
  /*!
   * \brief Emit an If instruction.
   * \param cond The register containing the cond value.
   * \param false_offset The program counter offset for the false branch.
   * \note result must be a register.
   */
  void EmitIf(vm::Instruction::Arg cond, vm::Index false_offset);
  /*!
   * \brief Get function index by its name.
   * \param name The name of the function.
   * \return The argument corresponding to the function index.
   */
  vm::Instruction::Arg GetFunction(const std::string& name);
  /*!
   * \brief Convert a constant value something that exec builder can understand.
   *
   * This function may update the constant pool to include the obj value.
   *
   * \param value The input constant value
   * \return An Arg that represents the result of constant argument.
   */
  template <typename T>
  vm::Instruction::Arg ConvertConstant(T value) {
    ffi::Any rv;
    rv = value;
    return ConvertConstant_(rv);
  }
  /*!
   * \brief update memory scopes.
   *
   * This function builds the memory scopes for constants.
   *
   * \param idx Index of the constant
   * \param scope The memory scope.
   */
  void SaveMemoryScope(vm::Instruction::Arg idx, runtime::MemoryScope scope);
  /*!
   * \brief Raw access to underlying executable build in progress.
   */
  vm::VMExecutable* exec() const;
  /*!
   * \brief Finalize the build, run formalize and get the final result.
   * \note This function should not be called during construction.
   */
  ffi::ObjectPtr<vm::VMExecutable> Get();
  /*!
   * \brief Create an ExecBuilder.
   * \return The ExecBuilder.
   */
  TVM_DLL static ExecBuilder Create();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExecBuilderObj>();
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.ExecBuilder", ExecBuilderObj, ffi::Object);

 private:
  /*!
   * \brief Convert a constant value something that exec builder can understand.
   *
   * This function may update the constant pool to include the obj value.
   *
   * \param obj The constant value to be emitted
   * \return An Arg that represents the result of constant argument.
   */
  vm::Instruction::Arg ConvertConstant_(ffi::Any obj);

  /*!
   * \brief A helper function to check if an executable is legal by checking if registers are used
   * properly
   */
  void CheckExecutable();
  /*!
   * \brief Formalize the executable.
   */
  void Formalize();

  /*! \brief The mutable internal executable. */
  ffi::ObjectPtr<vm::VMExecutable> exec_;  // mutable
  /*! \brief internal dedup map when creating index for a new constant */
  std::unordered_map<ffi::Any, vm::Index, ffi::StructuralHash, ffi::StructuralEqual>
      const_dedup_map_;
};

class ExecBuilder : public ffi::ObjectRef {
 public:
  explicit ExecBuilder(ffi::UnsafeInit tag, ffi::ObjectPtr<ExecBuilderObj> n)
      : ffi::ObjectRef(tag) {
    data_ = std::move(n);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExecBuilder, ffi::ObjectRef, ExecBuilderObj);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_EXEC_BUILDER_H_
