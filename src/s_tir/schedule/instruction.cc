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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>

#include "./utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

namespace {

ffi::String InstructionAsPythonRepr(const InstructionObj* self) {
  ffi::Array<Any> inputs;
  inputs.reserve(self->inputs.size());
  for (const Any& obj : self->inputs) {
    if (obj == nullptr) {
      inputs.push_back(ffi::String("None"));
    } else if (auto opt_str = obj.as<ffi::String>()) {
      inputs.push_back(ffi::String('"' + (*opt_str).operator std::string() + '"'));
    } else if (obj.as<SBlockRVObj>() || obj.as<LoopRVObj>()) {
      inputs.push_back(ffi::String("_"));
    } else if (obj.type_index() < ffi::TypeIndex::kTVMFFISmallStr) {
      inputs.push_back(obj);
    } else if (obj.as<IntImmObj>() || obj.as<FloatImmObj>()) {
      inputs.push_back(obj);
    } else if (auto expr = obj.as<PrimExpr>()) {
      auto f_substitute = [](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        return ffi::Any(Var("_", var->ty, var->loc).as_or_throw<PrimExpr>());
      };
      PrimExpr new_expr = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(expr.value(), f_substitute)
                              .as_or_throw<PrimExpr>();
      std::ostringstream os;
      os << new_expr;
      inputs.push_back(ffi::String(os.str()));
    } else if (obj.as<IndexMapObj>()) {
      inputs.push_back(obj);
    } else {
      TVM_FFI_THROW(TypeError) << "Stringifying is not supported for type: " << obj.GetTypeKey();
      throw;
    }
  }
  return self->kind->f_as_python(
      /*inputs=*/inputs,
      /*attrs=*/self->attrs,
      /*decision=*/Any(nullptr),
      /*outputs=*/ffi::Array<ffi::String>(self->outputs.size(), ffi::String("_")));
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() { InstructionKindObj::RegisterReflection(); }

bool InstructionKindObj::IsPostproc() const {
  static InstructionKind inst_enter_postproc = InstructionKind::Get("EnterPostproc");
  return this == inst_enter_postproc.get();
}

Instruction::Instruction(InstructionKind kind, ffi::Array<Any> inputs, ffi::Array<Any> attrs,
                         ffi::Array<Any> outputs) {
  ffi::ObjectPtr<InstructionObj> n = ffi::make_object<InstructionObj>(std::move(kind));
  n->inputs = std::move(inputs);
  n->attrs = std::move(attrs);
  n->outputs = std::move(outputs);
  this->data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  InstructionObj::RegisterReflection();
  refl::TypeAttrDef<InstructionObj>().def(ffi::type_attr::kRepr.name,
                                          [](Instruction inst, ffi::Function) -> ffi::String {
                                            return InstructionAsPythonRepr(inst.get());
                                          });

  refl::GlobalDef().def("s_tir.schedule.Instruction",
                        [](InstructionKind kind, ffi::Array<Any> inputs, ffi::Array<Any> attrs,
                           ffi::Array<Any> outputs) -> Instruction {
                          return Instruction(kind, inputs, attrs, outputs);
                        });
}

namespace {
class InstructionKindRegistry {
 public:
  static InstructionKindRegistry* Global() {
    static InstructionKindRegistry registry;
    return &registry;
  }

  InstructionKind Get(const ffi::String& name) const {
    auto kind = kinds_.Get(name);
    TVM_FFI_CHECK(kind.has_value(), AttributeError)
        << "Instruction kind " << name << " is not registered";
    return *kind;
  }

  InstructionKind RegisterOrGet(const ffi::String& name) {
    if (auto kind = kinds_.Get(name)) {
      return *kind;
    }
    InstructionKind kind(ffi::UnsafeInit{}, ffi::make_object<InstructionKindObj>());
    kinds_.Set(name, kind);
    return kind;
  }

 private:
  ffi::Dict<ffi::String, InstructionKind> kinds_;
};
}  // namespace

InstructionKind InstructionKind::Get(const ffi::String& name) {
  return InstructionKindRegistry::Global()->Get(name);
}

InstructionKindDef::InstructionKindDef(const ffi::String& name)
    : inst_kind_(InstructionKindRegistry::Global()->RegisterOrGet(name)) {
  get_mutable()->name = name;
}

/**************** Repr ****************/

// AC: kRepr already registered below in TVM_FFI_STATIC_INIT_BLOCK.

/**************** FFI ****************/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.schedule.InstructionKindGet", InstructionKind::Get);
}

}  // namespace s_tir
}  // namespace tvm
