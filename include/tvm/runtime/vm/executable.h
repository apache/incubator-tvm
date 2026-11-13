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
 * \file tvm/runtime/vm/executable.h
 */
#ifndef TVM_RUNTIME_VM_EXECUTABLE_H_
#define TVM_RUNTIME_VM_EXECUTABLE_H_

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/extra_base.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/support/io.h>
#include <tvm/support/serializer.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "./bytecode.h"

// Convention: this version should set to minimum TVM version it support
// NOTE: this file only changes if we change relax vm format
// for example if relax vm format do not change in 0.15, this should remain as 0.14
// if it changes in 0.16, we will change it to 0.16
#define VM_VERSION "0.14"

namespace tvm {
namespace runtime {
namespace vm {

/*!
 * \brief Information entry in executable function table.
 *
 * Contains metadata about the compiled function, as
 * well as the compiled VM instructions.
 */
struct VMFuncInfo {
  /*! \brief kind of the function. */
  enum class FuncKind : int {
    /*! \brief system level packed function */
    kPackedFunc = 0,
    /*! \brief VM function. */
    kVMFunc = 1,
  };
  /*! \brief The kind of function. */
  FuncKind kind;
  /*! \brief The function's name, global symbol */
  std::string name;
  /*! \brief The start instruction index of the function. */
  Index start_instr = 0;
  /*! \brief The end instruction index of the function. */
  Index end_instr = 0;
  /*! \brief The number of arguments of the function. */
  Index num_args = 0;
  /*! \brief The register file size of the function. */
  Index register_file_size = 0;
  /*! \brief The function parameter names.*/
  std::vector<std::string> param_names;

  // defined customized loader save
  void Save(support::Stream* writer) const;
  bool Load(support::Stream* reader);
};

/*!
 * \brief The virtual machine executable emitted by the VM compiler.
 *
 * The executable contains information (e.g. data in different memory regions)
 * to run in a virtual machine.
 */
class TVM_RUNTIME_EXTRA_DLL VMExecutable : public ffi::ModuleObj {
 public:
  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return ffi::Module::kBinarySerializable; };

  /*!
   * \brief Print the detailed statistics of the given code, i.e. number of
   * globals and constants, etc.
   * \return The statistics represented by a string.
   */
  std::string Stats() const;
  /*!
   * \brief Get the i-th instruction from the executable.
   * \param i The index of the instruction to be fetched.
   * \return The instruction.
   */
  Instruction GetInstruction(Index i) const;
  /*!
   * \brief Set j-th byte data of i-th instruction to val.
   * \param i The index of the instruction to be updated.
   * \param j The index of the byte data of the instruction to be updated.
   * \param val The value to be set
   */
  void SetInstructionData(Index i, Index j, ExecWord val);
  /*!
   * \brief Print the instructions as text format.
   * \return The text format of the instructions.
   */
  ffi::String AsText() const;
  /*!
   * \brief Print the instructions as python program.
   * \return The python program of the instructions, represented by a string.
   */
  ffi::String AsPython() const;
  /*!
   * \brief Write the VMExecutable to the binary stream in serialized form.
   * \return The binary bytes that save the executable to.
   */
  ffi::Bytes SaveToBytes() const final;
  /*!
   * \brief Load VMExecutable from the binary stream in serialized form.
   * \param bytes The binary bytes that load the executable from.
   * \return The loaded executable, in the form of a `runtime::Module`.
   */
  static ffi::Module LoadFromBytes(const ffi::Bytes& bytes);
  /*!
   * \brief Write the VMExecutable to the provided path as a file containing its serialized content.
   * \param file_name The name of the file to write the serialized data to.
   * \param format The target format of the saved file.
   */
  void WriteToFile(const ffi::String& file_name, const ffi::String& format) const final;
  /*! \brief Create a Relax virtual machine and load `this` as the executable. */
  ffi::Module VMLoadExecutable() const;
  /*! \brief Check if the VMExecutable contains a specific function. */
  bool HasFunction(const ffi::String& name) const;
  /*!
   * \brief Load VMExecutable from the file.
   * \param file_name The path of the file that load the executable from.
   * \return The loaded executable, in the form of a `runtime::Module`.
   */
  static ffi::Module LoadFromFile(const ffi::String& file_name);

  /*! \brief The virtual machine's function table. */
  std::vector<VMFuncInfo> func_table;
  /*! \brief A map from globals (as strings) to their index in the function map. */
  std::unordered_map<std::string, Index> func_map;
  /*! \brief The global constant pool. */
  std::vector<ffi::Any> constants;
  /*! \brief The VDevice memory scopes */
  std::unordered_map<Index, MemoryScope> memory_scopes;
  /*! \brief The offset of instruction. */
  std::vector<Index> instr_offset;
  /*! \brief The byte data of instruction. */
  std::vector<ExecWord> instr_data;

  virtual ~VMExecutable() {}

  /*! \brief Module type key. */
  const char* kind() const final;
  /*!
   * \brief Look up an exported function by name.
   * \param name The function name.
   * \return The function if found, otherwise std::nullopt.
   */
  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) override;

 private:
  /*!
   * \brief Save the globals.
   * \param strm The input stream.
   */
  void SaveGlobalSection(support::Stream* strm) const;
  /*!
   * \brief Save the memory scopes.
   * \param strm The output stream.
   */
  void SaveMemoryScopeSection(support::Stream* strm) const;
  /*!
   * \brief Save the constant pool.
   * \param strm The input stream.
   */
  void SaveConstantSection(support::Stream* strm) const;
  /*!
   * \brief Save the instructions.
   * \param strm The input stream.
   */
  void SaveCodeSection(support::Stream* strm) const;
  /*!
   * \brief Save the packed functions.
   * \param strm The input stream.
   */
  void SavePackedFuncNames(support::Stream* strm) const;
  /*!
   * \brief Load the globals.
   * \param strm The input stream.
   */
  void LoadGlobalSection(support::Stream* strm);
  /*!
   * \brief Load the memory scopes.
   * \param strm The input stream.
   */
  void LoadMemoryScopeSection(support::Stream* strm);
  /*!
   * \brief Load the constant pool.
   * \param strm The input stream.
   */
  void LoadConstantSection(support::Stream* strm);
  /*!
   * \brief Load the instructions.
   * \param strm The input stream.
   */
  void LoadCodeSection(support::Stream* strm);
  /*!
   * \brief Save the packed functions.
   * \param strm The input stream.
   */
  void LoadPackedFuncNames(support::Stream* strm);
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

namespace tvm {
namespace support {
template <>
struct Serializer<::tvm::runtime::vm::VMFuncInfo> {
  static constexpr bool enabled = true;
  static void Write(Stream* strm, const ::tvm::runtime::vm::VMFuncInfo& data) { data.Save(strm); }
  static bool Read(Stream* strm, ::tvm::runtime::vm::VMFuncInfo* data) { return data->Load(strm); }
};
}  // namespace support
}  // namespace tvm
#endif  // TVM_RUNTIME_VM_EXECUTABLE_H_
