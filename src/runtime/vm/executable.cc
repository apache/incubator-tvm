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
 * \file src/runtime/vm/executable.cc
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/support/bytes_io.h>
#include <tvm/support/file_utils.h>
#include <tvm/support/io.h>

#include <functional>
#include <sstream>

#include "./module_utils.h"

namespace tvm {
namespace runtime {
namespace vm {

/*! \brief The magic number for the serialized VM bytecode file  */
constexpr uint64_t kTVMVMBytecodeMagic = 0xD225DE2F4214151D;
constexpr uint64_t kTVMVMBytecodeMagicV2 = 0xD225DE2F4214151E;

#define STREAM_CHECK(val, section)                                                  \
  TVM_FFI_ICHECK(val) << "Invalid VM file format in the " << section << " section." \
                      << "\n";

const char* VMExecutable::kind() const { return "relax.VMExecutable"; }

ffi::Optional<ffi::Function> VMExecutable::GetFunction(const ffi::String& _name) {
  using SelfPtr = std::remove_cv_t<decltype(this)>;
  ::tvm::ffi::ObjectPtr<::tvm::ffi::Object> _self =
      ::tvm::ffi::GetObjectPtr<::tvm::ffi::Object>(this);
  TVM_MODULE_VTABLE_ENTRY("stats", &VMExecutable::Stats);
  TVM_MODULE_VTABLE_ENTRY("as_text", &VMExecutable::AsText);
  TVM_MODULE_VTABLE_ENTRY("as_python", &VMExecutable::AsPython);
  TVM_MODULE_VTABLE_ENTRY("vm_load_executable", &VMExecutable::VMLoadExecutable);
  TVM_MODULE_VTABLE_ENTRY("has_function", &VMExecutable::HasFunction);
  return std::nullopt;
}

std::string VMExecutable::Stats() const {
  std::ostringstream oss;
  oss << "Relax VM executable statistics:" << std::endl;

  // Get the number of constants.
  // If the constant is an Tensor, get the shape of each of them.
  // If the constant is an DLDataType, get the data type of each of them.
  oss << "  Constant pool (# " << constants.size() << "): [";
  for (const auto& it : constants) {
    if (auto opt_nd = it.as<ffi::Tensor>(); opt_nd && opt_nd->defined()) {
      const auto ndarray = opt_nd.value();
      const auto& shape = ndarray.shape();
      // Scalar
      if (shape.empty()) {
        oss << "scalar, ";
        continue;
      }
      oss << "[";
      for (auto s : shape) {
        oss << s << ", ";
      }
      oss.seekp(-2, oss.cur);
      oss << "], ";
    } else if (auto opt_shape = it.as<ffi::Shape>()) {
      ffi::Shape shape = opt_shape.value();
      oss << "shapetuple[";
      for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape.at(i) << ", ";
      }
      oss.seekp(-2, oss.cur);
      oss << "], ";
    } else if (auto opt_str = it.as<ffi::String>()) {
      std::string f = opt_str.value();
      oss << "\"";
      oss << f;
      oss << "\", ";
    } else if (auto opt_int = it.as<int64_t>()) {
      oss << opt_int.value();
      oss << ", ";
    } else if (auto opt_dtype = it.as<DLDataType>()) {
      oss << opt_dtype.value();
      oss << ", ";
    } else {
      TVM_FFI_THROW(InternalError) << "Unsupported constant pool type " << it.GetTypeKey();
    }
  }
  if (!constants.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  // Get the number of globals and the name of each of them.
  oss << "  Globals (#" << func_table.size() << "): [";
  for (const auto& it : func_table) {
    oss << it.name << ", ";
  }
  if (!func_map.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  return oss.str();
}

void VMExecutable::SetInstructionData(Index i, Index j, ExecWord val) {
  TVM_FFI_ICHECK_LT(i, instr_offset.size());
  Index instr_idx = instr_offset[i];
  TVM_FFI_ICHECK_LT(instr_idx + j, instr_data.size());
  instr_data[instr_idx + j] = val;
}

Instruction VMExecutable::GetInstruction(Index i) const {
  Index offset = instr_offset[i];
  Opcode op = static_cast<Opcode>(instr_data[offset]);
  switch (op) {
    case Opcode::Call: {
      RegName dst = instr_data[offset + 1];
      Index func_idx = instr_data[offset + 2];
      Index num_args = instr_data[offset + 3];
      ExecWord* args = const_cast<ExecWord*>(&instr_data[offset + 4]);
      return Instruction::Call(func_idx, num_args, reinterpret_cast<Instruction::Arg*>(args), dst);
    }
    case Opcode::Ret: {
      RegName result = instr_data[offset + 1];
      return Instruction::Ret(result);
    }
    case Opcode::Goto: {
      Index pc_offset = instr_data[offset + 1];
      return Instruction::Goto(pc_offset);
    }
    case Opcode::If: {
      RegName cond = instr_data[offset + 1];
      Index false_offset = instr_data[offset + 2];
      return Instruction::If(cond, false_offset);
    }
    default:
      TVM_FFI_THROW(InternalError) << "should never hit this case: " << static_cast<int>(op);
      break;
  }
  return Instruction();
}

void SaveHeader(support::Stream* strm) {
  uint64_t header = kTVMVMBytecodeMagicV2;
  strm->Write(header);
  std::string version = VM_VERSION;
  strm->Write(version);
}

uint64_t LoadHeader(support::Stream* strm) {
  // Check header.
  uint64_t header;
  STREAM_CHECK(strm->Read(&header), "header");
  STREAM_CHECK((header == kTVMVMBytecodeMagic) || (header == kTVMVMBytecodeMagicV2), "header");

  // Check version.
  std::string version;
  STREAM_CHECK(strm->Read(&version), "version");
  STREAM_CHECK(version == VM_VERSION, "version");

  return header;
}

ffi::Bytes VMExecutable::SaveToBytes() const {
  std::string result;
  support::BytesOutStream strm(&result);

  // Save header
  SaveHeader(&strm);

  // Global section.
  SaveGlobalSection(&strm);

  // Memory Scopes
  SaveMemoryScopeSection(&strm);

  // Constant section.
  SaveConstantSection(&strm);

  // Code section.
  SaveCodeSection(&strm);

  return ffi::Bytes(std::move(result));
}

void VMExecutable::WriteToFile(const ffi::String& file_name, const ffi::String& format) const {
  support::WriteBytesToFile(file_name, VMExecutable::SaveToBytes());
}

ffi::Module VMExecutable::LoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream strm(bytes);

  ffi::ObjectPtr<VMExecutable> exec = ffi::make_object<VMExecutable>();

  // Load header.
  uint64_t header_magic = LoadHeader(&strm);

  // Global section.
  exec->LoadGlobalSection(&strm);

  if (kTVMVMBytecodeMagicV2 == header_magic) {
    // Memory Scopes
    exec->LoadMemoryScopeSection(&strm);
  }

  // Constant section.
  exec->LoadConstantSection(&strm);

  // Code section.
  exec->LoadCodeSection(&strm);

  return ffi::Module(exec);
}

ffi::Module VMExecutable::LoadFromFile(const ffi::String& file_name) {
  std::string data = support::ReadBytesFromFile(file_name);
  return VMExecutable::LoadFromBytes(ffi::Bytes(data));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ffi.Module.load_from_file.relax.VMExecutable", VMExecutable::LoadFromFile)
      .def("ffi.Module.load_from_bytes.relax.VMExecutable", VMExecutable::LoadFromBytes);
}

void VMFuncInfo::Save(support::Stream* strm) const {
  int32_t temp_kind = static_cast<int32_t>(kind);
  strm->Write(temp_kind);
  strm->Write(name);
  strm->Write(start_instr);
  strm->Write(end_instr);
  strm->Write(num_args);
  strm->Write(register_file_size);
  strm->Write(param_names);
}

bool VMFuncInfo::Load(support::Stream* strm) {
  int32_t temp_kind;
  if (!strm->Read(&temp_kind)) return false;
  this->kind = static_cast<VMFuncInfo::FuncKind>(temp_kind);
  if (!strm->Read(&name)) return false;
  if (!strm->Read(&start_instr)) return false;
  if (!strm->Read(&end_instr)) return false;
  if (!strm->Read(&num_args)) return false;
  if (!strm->Read(&register_file_size)) return false;
  if (!strm->Read(&param_names)) return false;
  return true;
}

void VMExecutable::SaveGlobalSection(support::Stream* strm) const { strm->Write(func_table); }

void VMExecutable::SaveMemoryScopeSection(support::Stream* strm) const {
  strm->Write(static_cast<uint64_t>(this->memory_scopes.size()));
  for (auto it = this->memory_scopes.begin(); it != this->memory_scopes.end(); it++) {
    LOG(WARNING) << "Scope Saving:" << it->second;
    strm->Write(it->first);
    strm->Write(std::string(it->second.name()));
  }
}

void VMExecutable::SaveConstantSection(support::Stream* strm) const {
  // NOTE: pay close attention to the explicit type in write here
  // so the load/save is 32/64 bit compatible
  strm->Write(static_cast<uint64_t>(this->constants.size()));
  for (const auto& it : this->constants) {
    if (auto opt_nd = it.as<ffi::Tensor>(); opt_nd && opt_nd->defined()) {
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFITensor);
      runtime::SaveTensor(opt_nd.value(), strm);
    } else if (auto opt_shape = it.as<ffi::Shape>()) {
      ffi::Shape shape = opt_shape.value();
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIShape);
      strm->Write<uint64_t>(shape.size());
      strm->WriteArray<int64_t>(shape.data(), shape.size());
    } else if (auto opt_str = it.as<ffi::String>()) {
      ffi::String str = opt_str.value();
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIStr);
      strm->Write<uint64_t>(str.size());
      strm->WriteArray<uint8_t>(reinterpret_cast<const uint8_t*>(str.data()), str.size());
    } else if (auto opt_int = it.as<int64_t>()) {
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIInt);
      strm->Write<int64_t>(opt_int.value());
    } else if (auto opt_float = it.as<double>()) {
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIFloat);
      strm->Write<double>(opt_float.value());
    } else if (auto opt_dtype = it.as<DLDataType>()) {
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIDataType);
      strm->Write(opt_dtype.value());
    } else {
      TVM_FFI_THROW(InternalError) << "Unsupported constant pool type " << it.GetTypeKey();
    }
  }
}

void VMExecutable::SaveCodeSection(support::Stream* strm) const {
  strm->Write(instr_offset);
  strm->Write(instr_data);
}

void VMExecutable::LoadGlobalSection(support::Stream* strm) {
  STREAM_CHECK(strm->Read(&func_table), "Global Section");
  // setup func map
  for (size_t i = 0; i < func_table.size(); ++i) {
    this->func_map[func_table[i].name] = i;
  }
}

void VMExecutable::LoadMemoryScopeSection(support::Stream* strm) {
  uint64_t sz;
  // Load the number of memory scope entries.
  STREAM_CHECK(strm->Read(&sz, sizeof(sz)), "memory scopes");

  size_t size = static_cast<size_t>(sz);
  // Load each of the scopes.
  for (size_t i = 0; i < size; i++) {
    Index const_idx;
    std::string scope;
    STREAM_CHECK(strm->Read(&const_idx), "memory scopes");
    STREAM_CHECK(strm->Read(&scope), "memory scopes");
    LOG(WARNING) << "Scope Loaded:" << scope;
    this->memory_scopes[const_idx] = MemoryScope::Get(scope);
  }
}

void VMExecutable::LoadConstantSection(support::Stream* strm) {
  uint64_t sz;
  // Load the number of constants.
  STREAM_CHECK(strm->Read(&sz, sizeof(sz)), "constant");

  size_t size = static_cast<size_t>(sz);
  DLDataType dtype;
  // Load each of the constants.
  for (size_t i = 0; i < size; i++) {
    int constant_type;
    STREAM_CHECK(strm->Read(&constant_type, sizeof(constant_type)), "constant");
    if (constant_type == ffi::TypeIndex::kTVMFFITensor) {
      ffi::Tensor ndarray = LoadTensor(strm);
      ffi::Any cell;
      cell = ndarray;
      this->constants.push_back(cell);
    } else if (constant_type == ffi::TypeIndex::kTVMFFIShape) {
      uint64_t size;
      strm->Read(&size);
      std::vector<ffi::Shape::index_type> data(size);
      strm->ReadArray(data.data(), size);
      ffi::Any cell;
      cell = ffi::Shape(data);
      this->constants.push_back(cell);
    } else if (constant_type == ffi::TypeIndex::kTVMFFIDataType) {
      strm->Read(&dtype);
      ffi::Any cell;
      cell = dtype;
      this->constants.push_back(cell);
    } else if (constant_type == ffi::TypeIndex::kTVMFFIStr) {
      uint64_t size;
      strm->Read(&size);
      std::vector<char> data(size);
      STREAM_CHECK(strm->ReadArray(reinterpret_cast<uint8_t*>(data.data()), size),
                   "constant string");
      ffi::Any cell;
      cell = ffi::String(std::string(data.begin(), data.end()));
      this->constants.push_back(cell);
    } else if (constant_type == ffi::TypeIndex::kTVMFFIInt) {
      int64_t value;
      strm->Read(&value);
      ffi::Any cell;
      cell = value;
      this->constants.push_back(cell);
    } else if (constant_type == ffi::TypeIndex::kTVMFFIFloat) {
      double value;
      strm->Read(&value);
      ffi::Any cell;
      cell = value;
      this->constants.push_back(cell);
    } else {
      TVM_FFI_THROW(InternalError)
          << "Constant pool can only contain Tensor and DLDataType, but got "
          << ffi::TypeIndexToTypeKey(constant_type) << " when loading the VM constant pool.";
    }
  }
}

void VMExecutable::LoadCodeSection(support::Stream* strm) {
  STREAM_CHECK(strm->Read(&(this->instr_offset)), "instr offset");
  STREAM_CHECK(strm->Read(&(this->instr_data)), "instr data");
}

template <typename T>
std::string StrJoin(T* items, int offset, int cnt, std::string delim = ", ",
                    std::function<std::string(T)> repr = std::to_string) {
  if (cnt == 0) {
    return "";
  }
  std::ostringstream oss;
  oss << repr(items[offset]);
  for (int i = 1; i < cnt; ++i) {
    oss << delim << repr(items[offset + i]);
  }
  return oss.str();
}

std::string RegNameToStr(RegName reg) {
  if (reg == Instruction::kVoidRegister) {
    return "%void";
  }
  if (reg == Instruction::kVMRegister) {
    return "%vm";
  }
  return "%" + std::to_string(reg);
}

ffi::Module VMExecutable::VMLoadExecutable() const {
  ffi::ObjectPtr<VirtualMachine> vm = VirtualMachine::Create();
  vm->LoadExecutable(ffi::GetObjectPtr<VMExecutable>(const_cast<VMExecutable*>(this)));
  return ffi::Module(vm);
}

bool VMExecutable::HasFunction(const ffi::String& name) const { return func_map.count(name); }

ffi::String VMExecutable::AsText() const {
  auto get_func_name = [&](Index index) -> std::string {
    if (static_cast<size_t>(index) < func_table.size()) {
      return func_table[index].name;
    } else {
      return "unknown_func_index(" + std::to_string(index) + ")";
    }
  };

  auto instr_to_str = [&](Instruction::Arg arg) -> std::string {
    // only for argument
    switch (arg.kind()) {
      case Instruction::ArgKind::kRegister:
        return RegNameToStr(arg.value());
      case Instruction::ArgKind::kImmediate:
        return "i" + std::to_string(arg.value());
      case Instruction::ArgKind::kConstIdx:
        return "c[" + std::to_string(arg.value()) + "]";
      case Instruction::ArgKind::kFuncIdx:
        return "f[" + get_func_name(arg.value()) + "]";
      default:
        TVM_FFI_THROW(InternalError) << "Wrong instruction kind: " << static_cast<int>(arg.kind());
        return "";
    }
  };

  // print the text format
  std::ostringstream os;
  for (size_t fidx = 0; fidx < this->func_table.size(); ++fidx) {
    const VMFuncInfo& gfunc = this->func_table[fidx];
    if (gfunc.kind == VMFuncInfo::FuncKind::kPackedFunc) {
      os << "@" << gfunc.name << " packed_func;\n\n";
      continue;
    }
    TVM_FFI_ICHECK(gfunc.kind == VMFuncInfo::FuncKind::kVMFunc);
    os << "@" << gfunc.name << ":\n";
    size_t start_instr = gfunc.start_instr;
    size_t end_instr = gfunc.end_instr;

    for (size_t idx = start_instr; idx < end_instr; ++idx) {
      os << "  ";
      Instruction instr = this->GetInstruction(idx);
      switch (instr.op) {
        case Opcode::Call: {
          os << std::setw(6) << std::left << "call" << std::setw(16) << std::left
             << get_func_name(instr.func_idx) << " in: " << std::setw(12) << std::left
             << StrJoin<Instruction::Arg>(instr.args, 0, instr.num_args, ", ", instr_to_str)
             << " dst: " << RegNameToStr(instr.dst) << "\n";
          break;
        }
        case Opcode::Ret: {
          os << std::setw(6) << std::left << "ret " << RegNameToStr(instr.result) << "\n";
          break;
        }
        case Opcode::Goto: {
          os << std::setw(6) << std::left << "goto" << instr.pc_offset << "\n";
          break;
        }
        case Opcode::If: {
          os << std::setw(6) << std::left << "If" << RegNameToStr(instr.cond) << ", "
             << instr.false_offset << "\n";
          break;
        }
        default:
          TVM_FFI_THROW(InternalError)
              << "should never hit this case: " << static_cast<int>(instr.op);
          break;
      }
    }
    os << "\n";
  }
  return ffi::String(os.str());
}

ffi::String VMExecutable::AsPython() const {
  auto get_func_name = [&](Index index) -> std::string {
    if (static_cast<size_t>(index) < func_table.size()) {
      return "\"" + func_table[index].name + "\"";
    } else {
      return "ib.unknown_func_index(" + std::to_string(index) + ")";
    }
  };

  auto arg_to_py_str = [&](Instruction::Arg arg) -> std::string {
    switch (arg.kind()) {
      case Instruction::ArgKind::kRegister:
        if (arg.value() == Instruction::kVMRegister) {
          return "ib.r(vm)";
        }
        return "ib.r(" + std::to_string(arg.value()) + ")";
      case Instruction::ArgKind::kImmediate:
        return "ib.imm(" + std::to_string(arg.value()) + ")";
      case Instruction::ArgKind::kConstIdx:
        return "ib.c(" + std::to_string(arg.value()) + ")";
      case Instruction::ArgKind::kFuncIdx: {
        return "ib.f(" + get_func_name(arg.value()) + ")";
      }
      default:
        TVM_FFI_THROW(InternalError) << "Wrong instruction kind: " << static_cast<int>(arg.kind());
        return "";
    }
  };

  // print the python format
  std::ostringstream os;
  os << "ib = rx.Builder()\n";
  for (size_t fidx = 0; fidx < this->func_table.size(); ++fidx) {
    const VMFuncInfo& gfunc = this->func_table[fidx];
    if (gfunc.kind == VMFuncInfo::FuncKind::kPackedFunc) {
      continue;
    }
    TVM_FFI_ICHECK(gfunc.kind == VMFuncInfo::FuncKind::kVMFunc);

    os << "with ib.function(\"" << gfunc.name << "\", num_inputs=" << gfunc.num_args << "):\n";
    size_t start_instr = gfunc.start_instr;
    size_t end_instr = gfunc.end_instr;

    for (size_t idx = start_instr; idx < end_instr; ++idx) {
      Instruction instr = this->GetInstruction(idx);
      switch (instr.op) {
        case Opcode::Call: {
          os << "    ib.emit_call(" << get_func_name(instr.func_idx) << ", args=["
             << StrJoin<Instruction::Arg>(instr.args, 0, instr.num_args, ", ", arg_to_py_str)
             << "]";
          if (instr.dst != Instruction::kVoidRegister) os << ", dst=ib.r(" << instr.dst << ")";
          os << ")\n";
          break;
        }
        case Opcode::Ret: {
          os << "    ib.emit_ret(ib.r(" << instr.result << "))\n";
          break;
        }
        case Opcode::Goto: {
          os << "    ib.emit_goto(" << instr.pc_offset << ")\n";
          break;
        }
        case Opcode::If: {
          os << "    ib.emit_if(ib.r(" << instr.cond << "), " << instr.false_offset << ")\n";
          break;
        }
        default:
          TVM_FFI_THROW(InternalError)
              << "should never hit this case: " << static_cast<int>(instr.op);
          break;
      }
    }
  }
  return ffi::String(os.str());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ExecutableLoadFromFile", VMExecutable::LoadFromFile);
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
