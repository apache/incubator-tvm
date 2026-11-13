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
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>

#include <sstream>

#include "./utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

namespace {

ffi::String TraceAsPythonRepr(const TraceObj* self) {
  std::ostringstream os;
  os << "# from tvm import s_tir\n";
  os << "def apply_trace(sch: s_tir.Schedule) -> None:\n";
  ffi::Array<ffi::String> repr = self->AsPython(/*remove_postproc=*/false);
  bool is_first = true;
  for (const ffi::String& line : repr) {
    if (is_first) {
      is_first = false;
    } else {
      os << '\n';
    }
    os << "  " << std::string(line);
  }
  if (is_first) {
    os << "  pass";
  }
  return os.str();
}

}  // namespace

/**************** Constructors  ****************/

Trace::Trace() { data_ = ffi::make_object<TraceObj>(); }

Trace::Trace(ffi::Array<Instruction> insts, ffi::Map<Instruction, Any> decisions) {
  ffi::ObjectPtr<TraceObj> n = ffi::make_object<TraceObj>();
  n->insts = std::move(insts);
  n->decisions = std::move(decisions);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TraceObj::RegisterReflection();
  // Register __ffi_repr__ so str(trace) returns the Python script format
  refl::TypeAttrDef<TraceObj>().def(
      ffi::type_attr::kRepr.name,
      [](Trace trace, ffi::Function) -> ffi::String { return TraceAsPythonRepr(trace.get()); });

  refl::GlobalDef().def(
      "s_tir.schedule.Trace", [](ffi::Optional<ffi::Array<Instruction>> insts,
                                 ffi::Optional<ffi::Map<Instruction, Any>> decisions) {
        return Trace(insts.value_or(ffi::Array<Instruction>()), decisions.value_or({}));
      });
}

/**************** Utilities  ****************/

int GetNumValidInstructions(const ffi::Array<Instruction>& insts, bool remove_postproc) {
  if (!remove_postproc) {
    return insts.size();
  }
  int n_insts = 0;
  for (const Instruction& inst : insts) {
    if (!inst->kind->IsPostproc()) {
      ++n_insts;
    } else {
      break;
    }
  }
  return n_insts;
}

/**************** TranslateInputRVs  ****************/

ffi::Array<Any> TranslateInputRVs(
    const ffi::Array<Any>& inputs,
    const std::unordered_map<const ffi::Object*, const ffi::Object*>& rv_map) {
  ffi::Array<Any> result;
  result.reserve(inputs.size());
  auto f_subst_with_rv_map = [&rv_map](const Var& var) -> ffi::Optional<Expr> {
    auto it = rv_map.find(var.get());
    if (it == rv_map.end()) {
      return std::nullopt;
    }
    const ffi::Object* dst = it->second;
    TVM_FFI_CHECK(dst->IsInstance<VarObj>(), TypeError)
        << "Expect 'tirx.Var', but gets: " << dst->GetTypeKey();
    return ffi::GetRef<Var>(static_cast<const VarObj*>(dst));
  };
  auto f_subst_with_rv_map_prim = [&](const Var& var) -> ffi::Optional<PrimExpr> {
    if (auto replacement = f_subst_with_rv_map(var)) {
      return replacement.value().as_or_throw<PrimExpr>();
    }
    return std::nullopt;
  };
  auto f_substitute =
      [&f_subst_with_rv_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = f_subst_with_rv_map(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  auto f_substitute_prim =
      [&f_subst_with_rv_map_prim](
          const Var& var, TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (kind != kTVMFFIDefRegionKindNone) return ffi::Unchanged();
    if (auto repl = f_subst_with_rv_map_prim(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };

  for (const Any& input : inputs) {
    if (input.type_index() < ffi::TypeIndex::kTVMFFISmallStr) {
      // directly put back POD type
      result.push_back(input);
    } else if (auto expr = input.as<ffi::String>()) {
      result.push_back(expr.value());
    } else if (input.as<SBlockRVObj>() ||  // RV: block
               input.as<LoopRVObj>() ||    // RV: loop
               input.as<VarObj>()) {       // RV: var
      auto it = rv_map.find(input.as<ffi::Object>());
      TVM_FFI_CHECK(it != rv_map.end(), IndexError) << "Random variable doesn't exist: " << input;
      result.push_back(ffi::GetRef<ffi::ObjectRef>(it->second));
    } else if (auto expr = input.try_cast<PrimExpr>()) {  // RV: Expr
      result.push_back(ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(expr.value(), f_substitute)
                           .as_or_throw<PrimExpr>());
    } else if (auto index_map = input.as<IndexMap>()) {
      result.push_back(
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index_map.value(), f_substitute_prim)
              .as_or_throw<IndexMap>());
    } else if (auto arr = input.as<ffi::Array<Any>>()) {
      // Recursively convert elements of the array into a new list of ObjectRefs.
      result.push_back(TranslateInputRVs(arr.value(), rv_map));
    } else {
      TVM_FFI_CHECK(false, TypeError)
          << "Cannot recognize the type of an input random variable: " << input.GetTypeKey();
      throw;
    }
  }
  return result;
}

// translate rv to string
ffi::Array<Any> TranslateInputRVs(
    const ffi::Array<Any>& inputs,
    const std::unordered_map<ffi::ObjectRef, ffi::String, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>&
        rv_names) {
  ffi::Array<Any> results;
  results.reserve(inputs.size());
  for (const Any& input : inputs) {
    if (input == nullptr) {
      // Case 0. nullptr => None
      results.push_back(ffi::String("None"));
      continue;
    }
    // string => "content"
    if (auto opt_str = input.as<ffi::String>()) {
      results.push_back(ffi::String('"' + (*opt_str).operator std::string() + '"'));
    } else if (input.type_index() < ffi::TypeIndex::kTVMFFISmallStr) {
      // directly put back POD type and not string
      results.push_back(input);
    } else if (input.as<SBlockRVObj>() ||  // RV: block
               input.as<LoopRVObj>() ||    // RV: loop
               input.as<VarObj>()) {       // RV: var
      auto it = rv_names.find(input.cast<ffi::ObjectRef>());
      if (it != rv_names.end()) {
        // Case 1. SBlockRV, LoopRV, VarRV
        results.push_back(it->second);
      } else {
        TVM_FFI_THROW(IndexError) << "Random variable is not defined " << input;
        throw;
      }
    } else if (input.as<IntImmObj>() || input.as<FloatImmObj>()) {
      // Case 3. integer or floating-point number
      results.push_back(input);
    } else if (input.as<ffi::ArrayObj>()) {
      // Case 4: array
      results.push_back(TranslateInputRVs(Any(input).as_or_throw<ffi::Array<Any>>(), rv_names));
    } else if (input.as<ffi::MapObj>()) {
      // Case 5: dict
      results.push_back(input);
    } else if (input.as<IndexMapObj>()) {
      // // Case 6: IndexMap
      IndexMap index_map = input.as_or_throw<IndexMap>();
      index_map =
          index_map.RenameVariables([&rv_names](const Var& var) -> ffi::Optional<ffi::String> {
            if (auto it = rv_names.find(var); it != rv_names.end()) {
              return it->second;
            }
            return std::nullopt;
          });
      results.push_back(index_map);
    } else {
      TVM_FFI_THROW(TypeError) << "Stringifying is not supported for type: " << input.GetTypeKey();
      throw;
    }
  }
  return results;
}

ffi::Array<Any> TranslateInputRVs(
    const ffi::Array<Any>& inputs,
    const std::unordered_map<std::string, ffi::ObjectRef>& named_rvs) {
  ffi::Array<Any> results;
  results.reserve(inputs.size());
  auto f_substitute = [&named_rvs](
                          const Var& var,
                          TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (kind != kTVMFFIDefRegionKindNone) return ffi::Unchanged();
    auto it = named_rvs.find(var->name);
    if (it != named_rvs.end()) {
      return ffi::Any(it->second.as_or_throw<Var>().as_or_throw<PrimExpr>());
    }
    return ffi::Unchanged();
  };
  for (const Any& input : inputs) {
    if (input.type_index() < ffi::TypeIndex::kTVMFFISmallStr) {
      // directly put back POD type
      results.push_back(input);
      continue;
    }
    // Case 3. integer or floating-point immediate
    if (input.as<IntImmObj>() || input.as<FloatImmObj>()) {
      results.push_back(input);
      continue;
    }
    // Case 4. array
    if (input.as<ffi::ArrayObj>()) {
      results.push_back(TranslateInputRVs(input.as_or_throw<ffi::Array<Any>>(), named_rvs));
      continue;
    }
    // Case 5. dict
    if (input.as<ffi::MapObj>()) {
      results.push_back(input);
      continue;
    }
    auto opt_str = input.as<ffi::String>();
    TVM_FFI_CHECK(opt_str, TypeError) << "Expect String, but gets: " << input.GetTypeKey();
    TVM_FFI_CHECK_GT((*opt_str).size(), 0, ValueError)
        << "Empty string is not allowed in input names";
    const char* name = (*opt_str).data();
    int64_t size = (*opt_str).size();
    if (name[0] == '{' && name[size - 1] == '}') {
      Any obj = ffi::FromJSONGraph(ffi::json::Parse(name));
      // Case 6. IndexMap
      if (obj.as<IndexMapObj>()) {
        IndexMap index_map = obj.as_or_throw<IndexMap>();
        index_map = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index_map, f_substitute)
                        .as_or_throw<IndexMap>();
        results.push_back(index_map);
        continue;
      } else {
        TVM_FFI_THROW(TypeError) << "Unexpected object: " << obj.GetTypeKey();
        throw;
      }
    }
    // Case 2. string
    if (size >= 2 && name[0] == '"' && name[size - 1] == '"') {
      results.push_back(ffi::String(std::string(name + 1, size - 2)));
      continue;
    }
    // Case 0 & 1. None, SBlockRV, LoopRV, VarRV
    auto it = named_rvs.find(name);
    TVM_FFI_CHECK(it != named_rvs.end(), ValueError)
        << "The random variable is not defined: " << name;
    results.push_back(it->second);
  }
  return results;
}

// Compatibility is limited to schema-declared integer metadata. In particular,
// do not coerce expression-valued attributes or nested arbitrary containers.
int64_t ReadIntegerMetadata(const Any& value) {
  if (auto integer = value.as<IntImm>()) return Any(integer.value()->value).cast<int64_t>();
  return value.cast<int64_t>();
}

/**************** TranslateAddOutputRVs  ****************/

void TranslateAddOutputRVs(const ffi::Array<Any>& old_outputs, const ffi::Array<Any>& new_outputs,
                           std::unordered_map<const ffi::Object*, const ffi::Object*>* rv_map) {
  TVM_FFI_ICHECK_EQ(old_outputs.size(), new_outputs.size());
  int n = old_outputs.size();
  for (int i = 0; i < n; ++i) {
    const ffi::Object* old_rv = old_outputs[i].as<ffi::Object>();
    const ffi::Object* new_rv = new_outputs[i].as<ffi::Object>();
    TVM_FFI_ICHECK(old_rv != nullptr && new_rv != nullptr);
    (*rv_map)[old_rv] = new_rv;
  }
}

ffi::Array<ffi::String> TranslateAddOutputRVs(
    const ffi::Array<Any>& outputs,
    std::unordered_map<ffi::ObjectRef, ffi::String, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>*
        rv_names) {
  ffi::Array<ffi::String> results;
  results.reserve(outputs.size());
  for (const Any& output : outputs) {
    int i = rv_names->size();
    TVM_FFI_CHECK(!rv_names->count(output.cast<ffi::ObjectRef>()), ValueError)
        << "The random variable has been produced once: "
        << rv_names->at(output.cast<ffi::ObjectRef>());
    ffi::String result;
    if (output == nullptr) {
      result = "_";
    } else if (output.as<SBlockRVObj>()) {
      result = "b" + std::to_string(i);
    } else if (output.as<LoopRVObj>()) {
      result = "l" + std::to_string(i);
    } else if (output.as<VarObj>()) {
      result = "v" + std::to_string(i);
    } else {
      TVM_FFI_THROW(TypeError) << "Cannot recognize the type of the random variable: "
                               << output.GetTypeKey();
      throw;
    }
    results.push_back(result);
    rv_names->emplace(output.cast<ffi::ObjectRef>(), std::move(result));
  }
  return results;
}

void TranslateAddOutputRVs(const ffi::Array<ffi::String>& old_outputs,
                           const ffi::Array<Any>& new_outputs,
                           std::unordered_map<std::string, ffi::ObjectRef>* named_rvs) {
  TVM_FFI_ICHECK_EQ(old_outputs.size(), new_outputs.size());
  int n = old_outputs.size();
  for (int i = 0; i < n; ++i) {
    named_rvs->emplace(old_outputs[i], new_outputs[i].cast<ffi::ObjectRef>());
  }
}

/**************** Add/Remove/Get ****************/

Any TraceObj::GetDecision(const Instruction& inst) const {
  return this->decisions.Get(inst).value_or(Any{nullptr});
}

void TraceObj::Append(Instruction inst) { insts.push_back(std::move(inst)); }

void TraceObj::Append(Instruction inst, Any decision) {
  decisions.Set(inst, std::move(decision));
  insts.push_back(std::move(inst));
}

ffi::Optional<Instruction> TraceObj::Pop() {
  if (insts.empty()) {
    return std::nullopt;
  }
  Instruction inst = insts.back();
  insts.pop_back();
  if (decisions.count(inst)) {
    decisions.erase(inst);
  }
  return inst;
}

/**************** Interfacing with InstructionKind ****************/

void TraceObj::ApplyToSchedule(
    Schedule sch, bool remove_postproc,
    ffi::TypedFunction<Any(const Instruction& inst, const ffi::Array<Any>& inputs,  //
                           const ffi::Array<Any>& attrs,                            //
                           const Any& decision)>
        decision_provider) const {
  std::unordered_map<const ffi::Object*, const ffi::Object*> rv_map;
  for (const Instruction& inst : this->insts) {
    if (remove_postproc && inst->kind->IsPostproc()) {
      break;
    }
    ffi::Array<Any> inputs = TranslateInputRVs(inst->inputs, rv_map);
    ffi::Array<Any> attrs = inst->attrs;
    Any decision = this->GetDecision(inst);
    if (decision_provider != nullptr) {
      decision = decision_provider(inst, inputs, attrs, decision);
    }
    ffi::Array<Any> outputs = inst->kind->f_apply_to_schedule(sch, inputs, attrs, decision);
    TranslateAddOutputRVs(inst->outputs, outputs, &rv_map);
  }
}

ffi::ObjectRef TraceObj::AsJSON(bool remove_postproc) const {
  std::unordered_map<ffi::ObjectRef, ffi::String, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> rv_names;
  ffi::Array<ffi::Any> json_insts;
  ffi::Array<ffi::Any> json_decisions;
  json_insts.reserve(this->insts.size());
  json_decisions.reserve(this->insts.size());

  int i = 0;
  for (const Instruction& inst : this->insts) {
    const InstructionKind& kind = inst->kind;
    if (remove_postproc && kind->IsPostproc()) {
      break;
    }
    json_insts.push_back(ffi::Array<ffi::Any>{
        /* 0: inst name */ kind->name,
        /* 1: inputs    */ TranslateInputRVs(inst->inputs, rv_names),
        /* 2: attrs     */ kind->f_attrs_as_json != nullptr ? kind->f_attrs_as_json(inst->attrs)
                                                            : ffi::ObjectRef(inst->attrs),
        /* 3: outputs   */ TranslateAddOutputRVs(inst->outputs, &rv_names),
    });
    // Decisions may now hold POD types (e.g. int64_t for SampleCategorical /
    // SampleComputeLocation, or Array<int64_t> for SamplePerfectTile) after
    // the Integer phase-out. Treat them uniformly as Any.
    Any decision = this->GetDecision(inst);
    if (decision != nullptr) {
      json_decisions.push_back(ffi::Array<ffi::Any>{
          /* 0: index    */ i,
          /* 1: decision */ decision,
      });
    }
    ++i;
  }
  return ffi::Array<ffi::Any>{
      /* 0: trace    */ std::move(json_insts),
      /* 1: decision */ std::move(json_decisions),
  };
}

ffi::Array<ffi::String> TraceObj::AsPython(bool remove_postproc) const {
  std::unordered_map<ffi::ObjectRef, ffi::String, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> rv_names;
  ffi::Array<ffi::String> py_trace;
  py_trace.reserve(this->insts.size());
  for (const Instruction& inst : this->insts) {
    if (remove_postproc && inst->kind->IsPostproc()) {
      break;
    }
    ffi::Array<Any> attrs;
    attrs.reserve(inst->attrs.size());
    for (const Any& obj : inst->attrs) {
      if (auto opt_str = obj.as<ffi::String>()) {
        attrs.push_back(ffi::String('"' + (*opt_str).operator std::string() + '"'));
      } else {
        attrs.push_back(obj);
      }
    }
    py_trace.push_back(
        inst->kind->f_as_python(/*inputs=*/TranslateInputRVs(inst->inputs, rv_names),
                                /*attrs=*/attrs,
                                /*decision=*/this->GetDecision(inst),
                                /*outputs=*/TranslateAddOutputRVs(inst->outputs, &rv_names)));
  }
  return py_trace;
}

void Trace::ApplyJSONToSchedule(ffi::ObjectRef json, Schedule sch) {
  ffi::Array<Any> json_insts{nullptr};
  ffi::Array<Any> json_decisions{nullptr};
  // Parse `json` into `json_insts` and `json_decisions`
  try {
    const ffi::ArrayObj* arr = json.as<ffi::ArrayObj>();
    TVM_FFI_ICHECK(arr && arr->size() == 2);
    const auto* arr0 = arr->at(0).as<ffi::ArrayObj>();
    const auto* arr1 = arr->at(1).as<ffi::ArrayObj>();
    TVM_FFI_ICHECK(arr0 && arr1);
    json_insts = ffi::GetRef<ffi::ObjectRef>(arr0).as_or_throw<ffi::Array<Any>>();
    json_decisions = ffi::GetRef<ffi::ObjectRef>(arr1).as_or_throw<ffi::Array<Any>>();
  } catch (const tvm::ffi::Error& e) {
    TVM_FFI_THROW(ValueError) << "The json entry of a trace should contain two arrays, an array of "
                                 "instructions and an array of decisions, but gets: "
                              << json;
    throw;
  }
  // Parse `json_decisions`
  std::vector<Any> decisions(json_insts.size(), Any{nullptr});
  for (const Any& decision_entry : json_decisions) {
    int index = -1;
    Any decision{nullptr};
    try {
      const ffi::ArrayObj* arr = decision_entry.as<ffi::ArrayObj>();
      TVM_FFI_ICHECK(arr && arr->size() == 2);
      int64_t decision_index = ReadIntegerMetadata(arr->at(0));
      TVM_FFI_ICHECK(decision_index >= 0 &&
                     decision_index < static_cast<int64_t>(decisions.size()));
      index = static_cast<int>(decision_index);
      decision = arr->at(1);
    } catch (const tvm::ffi::Error& e) {
      TVM_FFI_THROW(ValueError) << "Each entry of a json decision should be a tuple [index, "
                                   "decision], but gets: "
                                << decision_entry;
      throw;
    }
    decisions[index] = std::move(decision);
  }
  // Parse `json_insts`
  std::unordered_map<std::string, ffi::ObjectRef> named_rvs{{"None", ffi::ObjectRef{nullptr}}};
  int i = 0;
  for (const Any& inst_entry : json_insts) {
    InstructionKind kind{ffi::UnsafeInit{}};
    ffi::Array<Any> inputs{nullptr};
    ffi::Array<Any> attrs{nullptr};
    ffi::Array<ffi::String> outputs;
    // Parse the entry
    try {
      const auto* arr = inst_entry.as<ffi::ArrayObj>();
      TVM_FFI_ICHECK(arr && arr->size() == 4);
      ffi::String arr0 = arr->at(0).cast<ffi::String>();
      kind = InstructionKind::Get(arr0);
      inputs = arr->at(1).cast<ffi::Array<Any>>();
      attrs = arr->at(2).cast<ffi::Array<Any>>();
      outputs = arr->at(3).cast<ffi::Array<ffi::String>>();
    } catch (const tvm::ffi::Error& e) {
      TVM_FFI_THROW(ValueError) << "Each entry of a json instruction should be a tuple [inst_name, "
                                   "inputs, attrs, outputs], but gets: "
                                << inst_entry << "\nThe error is: " << e.what();
      throw;
    }
    // Parse inputs
    inputs = TranslateInputRVs(inputs, named_rvs);
    // Parse attrs
    if (kind->f_attrs_from_json != nullptr) {
      attrs = kind->f_attrs_from_json(attrs);
    }
    // Apply to the schedule
    ffi::Array<Any> new_outputs = kind->f_apply_to_schedule(sch, inputs, attrs, decisions[i]);
    // Parse outputs
    TranslateAddOutputRVs(outputs, new_outputs, &named_rvs);
    ++i;
  }
}

/**************** Creation ****************/

Trace TraceObj::WithDecision(Instruction inst, Any decision, bool remove_postproc) const {
  int n_insts = GetNumValidInstructions(this->insts, remove_postproc);
  ffi::Array<Instruction> new_insts =
      ffi::Array<Instruction>{this->insts.begin(), this->insts.begin() + n_insts};
  ffi::Map<Instruction, Any> new_decisions{this->decisions.begin(), this->decisions.end()};
  new_decisions.Set(std::move(inst), std::move(decision));
  return Trace(new_insts, new_decisions);
}

Trace TraceObj::Simplified(bool remove_postproc) const {
  int n_insts = GetNumValidInstructions(this->insts, remove_postproc);
  std::unordered_set<const ffi::Object*> used_rvs;
  std::vector<Instruction> new_insts;
  std::unordered_map<Instruction, Any, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> new_decisions;
  new_insts.reserve(n_insts);
  new_decisions.reserve(this->decisions.size());
  for (int inst_idx = n_insts - 1; inst_idx >= 0; --inst_idx) {
    const Instruction& inst = this->insts[inst_idx];
    // Check if all the variables the instruction defined are dead
    // If so, and the instruction is pure, we can safely remove this instruction
    bool all_defs_dead = inst->kind->is_pure;
    if (all_defs_dead) {
      for (const Any& obj : inst->outputs) {
        if (auto* obj_ptr = obj.as<ffi::Object>()) {
          if (used_rvs.count(obj_ptr)) {
            all_defs_dead = false;
            break;
          }
        }
      }
    }
    // Remove this instruction
    if (all_defs_dead) {
      continue;
    }
    // Otherwise this instruction is not dead
    new_insts.push_back(inst);
    Any decision = this->GetDecision(inst);
    if (decision != nullptr) {
      new_decisions.emplace(inst, std::move(decision));
    }
    // Add its inputs as "used" ones
    for (const Any& obj : inst->inputs) {
      if (obj == nullptr) {
        continue;
      } else if (obj.as<SBlockRVObj>() || obj.as<LoopRVObj>() || obj.as<VarObj>()) {
        used_rvs.insert(obj.as<ffi::Object>());
        continue;
      } else if (auto prim_expr = obj.as<PrimExpr>()) {
        auto walk_fn = [&used_rvs](const Var& var) -> ffi::Expected<ffi::WalkResult> {
          used_rvs.insert(var.get());
          return ffi::WalkResult::Advance();
        };
        ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(*prim_expr, walk_fn);
      }
    }
  }
  return Trace(ffi::Array<Instruction>(new_insts.rbegin(), new_insts.rend()),
               ffi::Map<Instruction, Any>(new_decisions));
}

/**************** Repr ****************/

/**************** Instruction Registration ****************/

struct EnterPostprocTraits : public UnpackedInstTraits<EnterPostprocTraits> {
  static constexpr const char* kName = "EnterPostproc";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 0;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch) { return sch->EnterPostproc(); }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs) {
    PythonAPICall py("enter_postproc");
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() { RegisterInstructionKind<EnterPostprocTraits>(); }

/**************** FFI ****************/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.TraceGetDecision", &TraceObj::GetDecision)
      .def("s_tir.schedule.TraceAppend",
           [](Trace self, Instruction inst, ffi::Optional<ffi::ObjectRef> decision) {
             if (decision.has_value()) {
               return self->Append(inst, decision.value());
             } else {
               return self->Append(inst);
             }
           })
      .def_method("s_tir.schedule.TracePop", &TraceObj::Pop)
      .def_method("s_tir.schedule.TraceApplyToSchedule", &TraceObj::ApplyToSchedule)
      .def_method("s_tir.schedule.TraceAsJSON", &TraceObj::AsJSON)
      .def_method("s_tir.schedule.TraceAsPython", &TraceObj::AsPython)
      .def_method("s_tir.schedule.TraceWithDecision", &TraceObj::WithDecision)
      .def_method("s_tir.schedule.TraceSimplified", &TraceObj::Simplified)
      .def("s_tir.schedule.TraceApplyJSONToSchedule", Trace::ApplyJSONToSchedule);
}

}  // namespace s_tir
}  // namespace tvm
