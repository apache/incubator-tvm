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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/function.h>

#include <sstream>

#include "./utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*!
 * \brief Find the entry function of the given IRModule, i.e, functions marked by
 * `tirx::attr::kIsEntryFunc.name`, whose name is `main` or being the only Function.
 * \param mod The IRModule to find the entry function.
 * \return The entry function.
 */
inline tirx::Function FindEntryFunc(const IRModule& mod) {
  // Priority 1: Function marked as `tirx::attr::kIsEntryFunc.name`
  int num_function = 0;
  const tirx::FunctionObj* main_func = nullptr;
  const tirx::FunctionObj* last_func = nullptr;
  for (const auto& kv : mod->functions) {
    GlobalVar gv = kv.first;
    BaseFunc base_func = kv.second;
    if (const auto* func = base_func.as<tirx::FunctionObj>()) {
      last_func = func;
      if (func->HasNonzeroAttr(tvm::tirx::attr::kIsEntryFunc.name)) {
        return ffi::GetRef<tirx::Function>(func);
      }
      if (gv->name_hint == "main") {
        main_func = func;
      }
      ++num_function;
    }
  }
  // Priority 2: Function whose name is `main`
  if (main_func != nullptr) {
    return ffi::GetRef<tirx::Function>(main_func);
  }
  // Priority 3: The only Function in the IRModule
  if (num_function == 0) {
    TVM_FFI_THROW(ValueError) << "Cannot find any Function in the given IRModule: " << mod;
  }
  if (num_function > 1) {
    TVM_FFI_THROW(ValueError) << "Multiple Functions exist in the IRModule, but none of them are "
                                 "annotated with `kIsEntryFunc`, i.e. `tirx.is_entry_func`"
                              << mod;
  }
  return ffi::GetRef<tirx::Function>(last_func);
}
/******** ArgInfo ********/

ArgInfo ArgInfo::FromJSON(const ffi::ObjectRef& json_obj) {
  // The JSON object is always an array whose first element is a tag. For example:
  // `['TENSOR', 'float32', [1, 224, 224, 3]]
  // Step 1. Extract the tag
  ffi::Optional<ffi::String> tag{std::nullopt};
  try {
    const ffi::ArrayObj* json_array = json_obj.as<ffi::ArrayObj>();
    TVM_FFI_ICHECK(json_array && json_array->size() >= 1);
    tag = json_array->at(0).cast<ffi::String>();
  } catch (const std::runtime_error& e) {  // includes tvm::Error and dmlc::Error
    TVM_FFI_THROW(ValueError) << "Unable to parse the JSON object: " << json_obj
                              << "\nThe error is: " << e.what();
  }
  // Step 2. Dispatch the tag to corresponding subclass of ArgInfo
  if (tag == "TENSOR") {
    return TensorInfo::FromJSON(json_obj);
  }
  TVM_FFI_THROW(ValueError) << "Unable to parse the JSON object: " << json_obj;
  throw;
}

ffi::Array<ArgInfo> ArgInfo::FromFunction(const tirx::Function& func) {
  using support::AsVector;
  ffi::Array<ArgInfo> result;
  result.reserve(func->params.size());
  for (const tvm::Var& arg : func->params) {
    if (auto tensor = arg.as<tirx::TensorVar>()) {
      result.push_back(TensorInfo(/*dtype=*/tensor.value().dtype()->dtype,
                                  /*shape=*/AsVector<PrimExpr, int64_t>(tensor.value().shape())));
    } else {
      TVM_FFI_THROW(ValueError) << "Unsupported argument type: " << arg;
    }
  }
  return result;
}

ffi::Array<ArgInfo> ArgInfo::FromEntryFunc(const IRModule& mod, bool remove_preproc) {
  if (remove_preproc) {
    IRModule new_mod = transform::RemoveWeightLayoutRewriteBlock(/*skip_tensor_rewrite*/ true)(mod);
    return ArgInfo::FromFunction(FindEntryFunc(new_mod));
  }
  return ArgInfo::FromFunction(FindEntryFunc(mod));
}

/******** TensorInfo ********/

TensorInfo::TensorInfo(DLDataType dtype, ffi::Shape shape) {
  ffi::ObjectPtr<TensorInfoObj> n = ffi::make_object<TensorInfoObj>();
  n->dtype = dtype;
  n->shape = shape;
  this->data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TensorInfoObj::RegisterReflection();
  refl::TypeAttrDef<TensorInfoObj>().def(ffi::type_attr::kRepr.name,
                                         [](TensorInfo ti, ffi::Function fn_repr) -> ffi::String {
                                           std::ostringstream os;
                                           os << "TensorInfo(\"" << ti->dtype << "\", [";
                                           bool first = true;
                                           for (int64_t v : ti->shape) {
                                             if (!first) os << ", ";
                                             os << v;
                                             first = false;
                                           }
                                           os << "])";
                                           return os.str();
                                         });

  refl::GlobalDef().def(
      "s_tir.meta_schedule.TensorInfo",
      [](DLDataType dtype, ffi::Shape shape) -> TensorInfo { return TensorInfo(dtype, shape); });
}

ffi::ObjectRef TensorInfoObj::AsJSON() const {
  static ffi::String tag = "TENSOR";
  ffi::String dtype = ffi::DLDataTypeToString(this->dtype);
  ffi::Array<int64_t> shape = support::AsArray(this->shape);
  return ffi::Array<ffi::Any>{tag, dtype, shape};
}

TensorInfo TensorInfo::FromJSON(const ffi::ObjectRef& json_obj) {
  DLDataType dtype;
  ffi::Array<int64_t> shape;
  try {
    const ffi::ArrayObj* json_array = json_obj.as<ffi::ArrayObj>();
    TVM_FFI_ICHECK(json_array && json_array->size() == 3);
    // Load json[1] => dtype
    {
      ffi::String dtype_str = json_array->at(1).cast<ffi::String>();
      dtype = ffi::StringToDLDataType(dtype_str);
    }
    // Load json[2] => shape
    shape = AsIntArray(json_array->at(2).cast<ffi::ObjectRef>());
  } catch (const std::runtime_error& e) {  // includes tvm::Error and dmlc::Error
    TVM_FFI_THROW(ValueError) << "Unable to parse the JSON object: " << json_obj
                              << "\nThe error is: " << e.what();
  }
  std::vector<int64_t> s;
  std::transform(shape.begin(), shape.end(), std::back_inserter(s), [](int64_t i) { return i; });
  return TensorInfo(dtype, ffi::Shape(s.begin(), s.end()));
}

/******** Repr ********/

/******** FFI ********/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.meta_schedule.ArgInfoAsJSON", &ArgInfoObj::AsJSON)
      .def("s_tir.meta_schedule.ArgInfoFromFunction", ArgInfo::FromFunction)
      .def("s_tir.meta_schedule.ArgInfoFromEntryFunc", ArgInfo::FromEntryFunc)
      .def("s_tir.meta_schedule.ArgInfoFromJSON", ArgInfo::FromJSON);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
