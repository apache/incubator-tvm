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
#ifndef TVM_S_TIR_SCHEDULE_INSTRUCTION_TRAITS_H_
#define TVM_S_TIR_SCHEDULE_INSTRUCTION_TRAITS_H_

#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/schedule/instruction.h>
#include <tvm/s_tir/schedule/schedule.h>

#include <algorithm>
#include <array>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*! \brief Register a named instruction kind using its typed traits.
 * \tparam Traits Traits with kName, kIsPure, and the instruction callbacks.
 *
 * Example:
 * \code
 * TVM_FFI_STATIC_INIT_BLOCK() {
 *   RegisterInstructionKind<ComputeInlineTraits>();
 *   RegisterInstructionKind<ReverseComputeInlineTraits>();
 * }
 * \endcode
 */
template <typename Traits>
void RegisterInstructionKind() {
  InstructionKindDef(Traits::kName)
      .set_is_pure(Traits::kIsPure)
      .set_apply_to_schedule(Traits::ApplyToSchedule)
      .set_attrs_as_json(Traits::AttrsAsJSON)
      .set_attrs_from_json(Traits::AttrsFromJSON)
      .set_as_python(Traits::AsPython);
}

/*!
 * \brief A helper to conveniently register an InstructionKind. When inherited in curiously
 * recursive template pattern, the derived class `TTraits` only needs to define two functions on the
 * unpacked inputs, and the helper handles unpacking and downcasting. See the example for more
 * details.
 *
 * \tparam TTraits The derived class
 *
 * Example:
 *
 * \code
 *
 * struct SamplePerfectTileTraits : public UnpackedInstTraits<SamplePerfectTileTraits> {
 *   // The name of this kind of instruction
 *   static constexpr const char* kName = "SamplePerfectTile";
 *   // A boolean indicating if the instruction is pure, i.e. change nothing in the schedule state
 *   static constexpr bool kIsPure = true;
 *   // The number of inputs in this kind of instruction
 *   static constexpr size_t kNumInputs = 1;
 *   // The number of attributes in this kind of instruction
 *   static constexpr size_t kNumAttrs = 2;
 *   // The number of decisions in this kind of instruction (only 0 or 1 is allowed)
 *   static constexpr size_t kNumDecisions = 1;
 *
 *   // Calling convention:
 *   // - Arguments use their FFI-compatible schema types
 *   // - The 1st argument is Schedule
 *   // - The next `kNumInputs` arguments are input random variables
 *   // - The next `kNumAttrs` arguments are attributes
 *   // - The next argument is decision, if `kNumDecisions == 1`
 *   static ffi::Array<Var> UnpackedApplyToSchedule(
 *      Schedule sch,
 *      LoopRV loop_rv,
 *      int n,
 *      int max_innermost_factor,
 *      ffi::Optional<ffi::Array<int64_t>> decision) {
 *     return sch->SamplePerfectTile(loop_rv, n, max_innermost_factor, decision);
 *   }
 *
 *   // Calling convention:
 *   // - Arguments use their FFI-compatible schema types
 *   // - The 1st argument is an array containing names of output random variables
 *   // - The next `kNumInputs` arguments are names of input random variables
 *   // - The next `kNumAttrs` arguments are attributes
 *   // - The next argument is decision, if `kNumDecisions == 1`
 *   static ffi::String UnpackedAsPython(
 *      ffi::Array<ffi::String> outputs,
 *      ffi::String loop_rv,
 *      int n,
 *      int max_innermost_factor,
 *      ffi::Optional<ffi::Array<int64_t>> decision) {
 *     PythonAPICall py("sample_perfect_tile");
 *     py.Input("loop", loop_rv);
 *     py.Input("n", n);
 *     py.Input("max_innermost_factor", max_innermost_factor);
 *     py.Decision(decision);
 *     py.OutputList(outputs);
 *     return py.Str();
 *   }
 *
 *   template <typename>
 *   friend struct UnpackedInstTraits;
 * };
 *
 * RegisterInstructionKind<SamplePerfectTileTraits>();
 * \endcode
 */
template <class TTraits>
struct UnpackedInstTraits {
  /*!
   * \brief Unpack the arguments in the calling convention, and feed them into
   * `TTraits::UnpackedApplyToSchedule`
   * \sa InstructionKindObj::f_apply_to_schedule
   */
  static ffi::Array<Any> ApplyToSchedule(const Schedule& sch, const ffi::Array<Any>& inputs,
                                         const ffi::Array<Any>& attrs, const Any& decision);

  /*!
   * \brief Unpack the arguments in the calling convention, and feed them into
   * `TTraits::UnpackedAsPython`
   * \sa InstructionKindObj::f_as_python
   */
  static ffi::String AsPython(const ffi::Array<Any>& inputs, const ffi::Array<Any>& attrs,
                              const Any& decision, const ffi::Array<ffi::String>& outputs);

  /*! \brief No customized serializer by default */
  static constexpr std::nullptr_t AttrsAsJSON = nullptr;

  /*! \brief No customized deserializer by default */
  static constexpr std::nullptr_t AttrsFromJSON = nullptr;

 protected:
  template <size_t index_offset>
  TVM_FFI_INLINE static void _SetInputs(AnyView* packed_args, const ffi::Array<Any>& inputs);
  template <size_t index_offset>
  TVM_FFI_INLINE static void _SetAttrs(AnyView* packed_args, const ffi::Array<Any>& attrs);
  template <size_t index_offset>
  TVM_FFI_INLINE static void _SetDecision(AnyView* packed_args, const Any& decision);
  TVM_FFI_INLINE static ffi::Array<Any> _ConvertOutputs(const ffi::Any& rv);
};

/*!
 * \brief A helper class that constructs schedule API call in python syntax,
 * which helps convert an Inst to a python statement.
 * \sa InstructionKindObj::f_as_python
 */
class PythonAPICall {
 public:
  /*!
   * \brief Constructor
   * \param method_name The name of the schedule API to be called
   */
  explicit PythonAPICall(ffi::String method_name)
      : method_name_(method_name), output_(std::nullopt) {}
  /*! \brief Add an integer input */
  inline void Input(ffi::String arg_name, int arg);
  /*! \brief Add an integer input */
  inline void Input(ffi::String arg_name, int64_t arg);
  /*! \brief Add a bool input */
  inline void Input(ffi::String arg_name, bool arg);
  /*! \brief Add a double input */
  inline void Input(ffi::String arg_name, double arg);
  /*! \brief Add an input random variable */
  inline void Input(ffi::String arg_name, ffi::String arg);
  /*! \brief Add an input random variable */
  inline void Input(ffi::String arg_name, std::string arg);
  /*! \brief Add an input, dispatched to different implementations according to the object's type */
  inline void Input(ffi::String arg_name, Any arg);
  /*! \brief Add the decision */
  inline void Decision(Any decision);
  /*!
   * \brief Add a single output random variable
   * \param unit_array An array containing only one element
   */
  inline void SingleOutput(ffi::Array<ffi::String> unit_array);
  /*! \brief Add a list of output random variables */
  inline void OutputList(ffi::Array<ffi::String> outputs);
  /*! \returns The schedule API call in python syntax */
  inline ffi::String Str() const;

 private:
  /*! \brief Converts a TVM object to python string and print to the output stream */
  inline void AsPythonString(const Any& obj, std::ostream& os);

 private:
  /*! \brief The name of the API to call */
  ffi::String method_name_;
  /*! \brief The output of the instruction */
  ffi::Optional<ffi::String> output_;
  /*! \brief The names of input arguments */
  std::vector<ffi::String> arg_names_;
  /*! \brief The values of input arguments */
  std::vector<ffi::String> args_;
};

/********** implementation details **********/

// forward declaration
namespace details {

template <typename... Args>
struct _ArgsPacker;

template <>
struct _ArgsPacker<> {
  static constexpr bool checked = true;
};

template <typename TObjectRef, typename... Args>
struct _ArgsPacker<TObjectRef, Args...> {
  static constexpr bool checked =
      std::is_base_of<ffi::ObjectRef, TObjectRef>::value && _ArgsPacker<Args...>::checked;
};

template <typename T>
struct _MethodType {};

template <typename TReturn, typename... Args>
struct _MethodType<TReturn(Args...)> {
  using return_type = TReturn;
  using argument_types = std::tuple<Args...>;
  using argument_type = _ArgsPacker<Args...>;
};

template <typename T>
struct _NumArgs {};

template <typename TReturn, typename... Args>
struct _NumArgs<TReturn(Args...)> {
  static constexpr size_t value = sizeof...(Args);
};

template <typename>
struct _IsTVMArray : std::false_type {};

template <typename T>
struct _IsTVMArray<ffi::Array<T>> : std::true_type {};

template <typename T>
struct _IsSingleObject
    : std::integral_constant<bool, !_IsTVMArray<T>::value && !std::is_void<T>::value> {};

template <class T>
using ReturnType = typename _MethodType<std::remove_cv_t<T>>::return_type;

template <class T>
static constexpr bool ArgumentAreAllObjects =
    _MethodType<std::remove_cv_t<T>>::argument_type::checked;

template <class T>
static constexpr size_t NumArgs = _NumArgs<std::remove_cv_t<T>>::value;

template <class T>
static constexpr int IsTVMArray = _IsTVMArray<std::remove_cv_t<T>>::value;

template <class T>
static constexpr int IsSingleObject = _IsSingleObject<std::remove_cv_t<T>>::value;

// Legacy traces may carry IntImm wrappers in schema-declared integer slots.
// Normalize by the trait signature, never by recursively walking arbitrary IR.
template <typename T>
Any NormalizeIntegerMetadataArgument(AnyView value) {
  using U = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (std::is_integral_v<U>) {
    if (auto integer = value.as<IntImm>()) {
      if constexpr (std::is_same_v<U, bool>) {
        return integer.value()->value != 0;
      } else {
        return integer.value()->value;
      }
    }
  } else if constexpr (std::is_same_v<U, ffi::Array<int64_t>>) {
    if (!value.try_cast<U>()) {
      ffi::Array<int64_t> result;
      for (AnyView item : value.cast<ffi::Array<Any>>()) {
        result.push_back(NormalizeIntegerMetadataArgument<int64_t>(item).cast<int64_t>());
      }
      return result;
    }
  } else if constexpr (std::is_same_v<U, ffi::Optional<int64_t>>) {
    if (value != nullptr) return NormalizeIntegerMetadataArgument<int64_t>(value);
  } else if constexpr (std::is_same_v<U, ffi::Optional<ffi::Array<int64_t>>>) {
    if (value != nullptr) return NormalizeIntegerMetadataArgument<ffi::Array<int64_t>>(value);
  }
  return Any(value);
}

template <typename Method, size_t... I>
std::array<Any, sizeof...(I)> NormalizeIntegerMetadataArguments(const AnyView* args,
                                                                std::index_sequence<I...>) {
  using Arguments = typename _MethodType<Method>::argument_types;
  return {NormalizeIntegerMetadataArgument<std::tuple_element_t<I, Arguments>>(args[I])...};
}

};  // namespace details

template <class TTraits>
ffi::Array<Any> UnpackedInstTraits<TTraits>::ApplyToSchedule(const Schedule& sch,
                                                             const ffi::Array<Any>& inputs,
                                                             const ffi::Array<Any>& attrs,
                                                             const Any& decision) {
  using method_type = decltype(TTraits::UnpackedApplyToSchedule);
  using return_type = details::ReturnType<method_type>;
  // static_assert(details::ArgumentAreAllObjects<method_type>,
  //              "All arguments to `UnpackedApplyToSchedule` must be subclasses of ObjectRef");
  constexpr size_t kNumArgs = details::NumArgs<method_type>;
  constexpr size_t kNumInputs = TTraits::kNumInputs;
  constexpr size_t kNumAttrs = TTraits::kNumAttrs;
  constexpr size_t kNumDecisions = TTraits::kNumDecisions;
  static_assert(kNumArgs == 1 + kNumInputs + kNumAttrs + kNumDecisions,
                "length of argument list mismatch");
  AnyView packed_args[kNumArgs];
  packed_args[0] = sch;
  TTraits::template _SetInputs<1>(packed_args, inputs);
  TTraits::template _SetAttrs<1 + kNumInputs>(packed_args, attrs);
  TTraits::template _SetDecision<1 + kNumInputs + kNumAttrs>(packed_args, decision);
  auto normalized = details::NormalizeIntegerMetadataArguments<method_type>(
      packed_args, std::make_index_sequence<kNumArgs>{});
  for (size_t i = 0; i < kNumArgs; ++i) packed_args[i] = normalized[i];
  ffi::Function pf([](const ffi::PackedArgs& args, ffi::Any* rv) -> void {
    constexpr size_t kNumArgs = details::NumArgs<method_type>;
    TVM_FFI_ICHECK_EQ(args.size(), kNumArgs);
    ffi::detail::unpack_call<return_type>(std::make_index_sequence<kNumArgs>{}, nullptr,
                                          TTraits::UnpackedApplyToSchedule, args.data(),
                                          args.size(), rv);
  });
  ffi::Any rv;
  pf.CallPacked(ffi::PackedArgs(packed_args, kNumArgs), &rv);
  return TTraits::_ConvertOutputs(rv);
}

template <class TTraits>
ffi::String UnpackedInstTraits<TTraits>::AsPython(const ffi::Array<Any>& inputs,
                                                  const ffi::Array<Any>& attrs, const Any& decision,
                                                  const ffi::Array<ffi::String>& outputs) {
  using method_type = decltype(TTraits::UnpackedAsPython);
  using return_type = details::ReturnType<method_type>;
  // static_assert(details::ArgumentAreAllObjects<method_type>,
  //               "All arguments to `UnpackedAsPython` must be subclasses of ObjectRef");
  constexpr size_t kNumArgs = details::NumArgs<method_type>;
  constexpr size_t kNumInputs = TTraits::kNumInputs;
  constexpr size_t kNumAttrs = TTraits::kNumAttrs;
  constexpr size_t kNumDecisions = TTraits::kNumDecisions;
  static_assert(kNumArgs == 1 + kNumInputs + kNumAttrs + kNumDecisions,
                "length of argument list mismatch");

  AnyView packed_args[kNumArgs];
  packed_args[0] = outputs;
  TTraits::template _SetInputs<1>(packed_args, inputs);
  TTraits::template _SetAttrs<1 + kNumInputs>(packed_args, attrs);
  TTraits::template _SetDecision<1 + kNumInputs + kNumAttrs>(packed_args, decision);
  auto normalized = details::NormalizeIntegerMetadataArguments<method_type>(
      packed_args, std::make_index_sequence<kNumArgs>{});
  for (size_t i = 0; i < kNumArgs; ++i) packed_args[i] = normalized[i];
  ffi::Function pf([](const ffi::PackedArgs& args, ffi::Any* rv) -> void {
    constexpr size_t kNumArgs = details::NumArgs<method_type>;
    TVM_FFI_ICHECK_EQ(args.size(), kNumArgs);
    ffi::detail::unpack_call<return_type>(std::make_index_sequence<kNumArgs>{}, nullptr,
                                          TTraits::UnpackedAsPython, args.data(), args.size(), rv);
  });
  ffi::Any rv;
  pf.CallPacked(ffi::PackedArgs(packed_args, kNumArgs), &rv);
  return rv.cast<ffi::String>();
}

template <class TTraits>
template <size_t index_offset>
TVM_FFI_INLINE void UnpackedInstTraits<TTraits>::_SetInputs(AnyView* packed_args,
                                                            const ffi::Array<Any>& inputs) {
  constexpr size_t kNumInputs = TTraits::kNumInputs;
  TVM_FFI_CHECK_EQ(kNumInputs, inputs.size(), ValueError)
      << "Incorrect kNumInputs for instruction: " << TTraits::kName;
  for (size_t i = 0; i < kNumInputs; ++i) {
    packed_args[i + index_offset] = inputs[i];
  }
}

template <class TTraits>
template <size_t index_offset>
TVM_FFI_INLINE void UnpackedInstTraits<TTraits>::_SetAttrs(AnyView* packed_args,
                                                           const ffi::Array<Any>& attrs) {
  constexpr size_t kNumAttrs = TTraits::kNumAttrs;
  TVM_FFI_CHECK_EQ(kNumAttrs, attrs.size(), ValueError)
      << "Incorrect kNumAttrs for instruction: " << TTraits::kName;
  for (size_t i = 0; i < kNumAttrs; ++i) {
    packed_args[i + index_offset] = attrs[i];
  }
}

template <class TTraits>
template <size_t index_offset>
TVM_FFI_INLINE void UnpackedInstTraits<TTraits>::_SetDecision(AnyView* packed_args,
                                                              const Any& decision) {
  constexpr size_t kNumDecisions = TTraits::kNumDecisions;
  static_assert(kNumDecisions <= 1, "an instruction is supposed to have at most 1 decision");
  if (kNumDecisions == 1) {
    packed_args[index_offset] = decision;
  } else {
    TVM_FFI_ICHECK(decision == nullptr);
  }
}

template <class TTraits>
TVM_FFI_INLINE ffi::Array<Any> UnpackedInstTraits<TTraits>::_ConvertOutputs(const ffi::Any& rv) {
  using method_type = decltype(TTraits::UnpackedApplyToSchedule);
  using return_type = details::ReturnType<method_type>;
  constexpr int is_array = details::IsTVMArray<return_type>;
  constexpr int is_single_obj = details::IsSingleObject<return_type>;
  constexpr int is_void = std::is_void<return_type>::value;
  static_assert(is_array || is_single_obj || is_void, "return type not supported");
  static_assert(is_array + is_single_obj + is_void == 1, "internal template error");
  if (is_void) {
    return {};
  } else if (is_single_obj) {
    return {rv};
  } else if (is_array) {
    return rv.cast<ffi::Array<Any>>();
  }
}

/********** PythonAPICall **********/

inline void PythonAPICall::AsPythonString(const Any& obj, std::ostream& os) {
  if (obj == nullptr) {
    os << "None";
  } else if (auto opt_str = obj.as<ffi::String>()) {
    os << *opt_str;
  } else if (const auto opt_int_imm = obj.try_cast<IntImm>()) {
    os << (*opt_int_imm)->value;
  } else if (const auto opt_float_imm = obj.try_cast<FloatImm>()) {
    os.precision(17);
    os << (*opt_float_imm)->value;
  } else if (const auto* array = obj.as<ffi::ArrayObj>()) {
    os << '[';
    bool is_first = true;
    for (Any e : *array) {
      if (is_first) {
        is_first = false;
      } else {
        os << ", ";
      }
      AsPythonString(e, os);
    }
    os << ']';
  } else if (const auto* dict = obj.as<ffi::MapObj>()) {
    os << '{';
    bool is_first = true;
    std::vector<std::pair<std::string, std::string>> dict_items;
    for (auto it = dict->begin(); it != dict->end(); ++it) {
      std::ostringstream ks;
      AsPythonString(it->first, ks);
      std::ostringstream vs;
      AsPythonString(it->second, vs);
      dict_items.emplace_back(ks.str(), vs.str());
    }
    std::sort(dict_items.begin(), dict_items.end(),
              [](const auto& p1, const auto& p2) { return p1.first < p2.first; });
    for (const auto& kv : dict_items) {
      if (is_first) {
        is_first = false;
      } else {
        os << ", ";
      }
      os << '\"' << kv.first << "\": " << kv.second;
    }
    os << '}';
  } else {
    TVM_FFI_THROW(ValueError) << "Cannot translate type '" << obj.GetTypeKey()
                              << "' to python. Its value is: " << obj;
    throw;
  }
}

void PythonAPICall::Input(ffi::String arg_name, int arg) {
  arg_names_.emplace_back(std::move(arg_name));
  args_.push_back(std::to_string(arg));
}

void PythonAPICall::Input(ffi::String arg_name, int64_t arg) {
  arg_names_.emplace_back(std::move(arg_name));
  args_.push_back(std::to_string(arg));
}

void PythonAPICall::Input(ffi::String arg_name, bool arg) {
  static const char* true_str = "True";
  static const char* false_str = "False";
  arg_names_.emplace_back(std::move(arg_name));
  if (arg) {
    args_.push_back(true_str);
  } else {
    args_.push_back(false_str);
  }
}

void PythonAPICall::Input(ffi::String arg_name, double arg) {
  arg_names_.emplace_back(std::move(arg_name));
  std::ostringstream os;
  os.precision(17);
  os << arg;
  args_.push_back(os.str());
}

void PythonAPICall::Input(ffi::String arg_name, ffi::String arg) {
  arg_names_.emplace_back(std::move(arg_name));
  args_.emplace_back(std::move(arg));
}

void PythonAPICall::Input(ffi::String arg_name, std::string arg) {
  arg_names_.emplace_back(std::move(arg_name));
  args_.emplace_back(std::move(arg));
}

void PythonAPICall::Input(ffi::String arg_name, Any arg) {
  arg_names_.emplace_back(std::move(arg_name));
  std::ostringstream os;
  AsPythonString(arg, os);
  args_.push_back(os.str());
}

void PythonAPICall::Decision(Any decision) {
  if (decision != nullptr) {
    this->Input("decision", decision);
  }
}

void PythonAPICall::SingleOutput(ffi::Array<ffi::String> unit_array) {
  TVM_FFI_ICHECK_EQ(unit_array.size(), 1);
  this->output_ = unit_array[0];
}

void PythonAPICall::OutputList(ffi::Array<ffi::String> outputs) {
  if (outputs.empty()) {
    return;
  }
  if (outputs.size() == 1) {
    this->output_ = outputs[0] + ",";
    return;
  }
  std::ostringstream os;
  os << outputs[0];
  for (int i = 1, n = outputs.size(); i < n; ++i) {
    os << ", " << outputs[i];
  }
  this->output_ = os.str();
}

ffi::String PythonAPICall::Str() const {
  std::ostringstream os;
  if (output_.has_value()) {
    os << output_.value() << " = ";
  }
  os << "sch." << method_name_ << '(';
  int n = args_.size();
  for (int i = 0; i < n; ++i) {
    if (i > 0) {
      os << ", ";
    }
    if (arg_names_[i].empty()) {
      os << args_[i];
    } else {
      os << arg_names_[i] << '=' << args_[i];
    }
  }
  os << ')';
  return os.str();
}

}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_SCHEDULE_INSTRUCTION_TRAITS_H_
