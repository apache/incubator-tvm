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
#ifndef TVM_S_TIR_SCHEDULE_INSTRUCTION_H_
#define TVM_S_TIR_SCHEDULE_INSTRUCTION_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>

#include <utility>

namespace tvm {

namespace s_tir {

// Forward declaration
class Schedule;

/*!
 * \brief Type of the functor that applies the instruction to a TensorIR schedule
 * \param sch The schedule to be applied on
 * \param inputs The input random variables
 * \param attrs Instruction attributes
 * \param decision Decisions made on the instruction
 * \return The functor returns an array of output random variables
 */
using FInstructionApply =
    ffi::TypedFunction<ffi::Array<Any>(Schedule sch, const ffi::Array<Any>& inputs,
                                       const ffi::Array<Any>& attrs, const Any& decision)>;

/*!
 * \brief Type of the functor that converts the instruction to a statement in python syntax
 * \param inputs Names of the input random variables
 * \param attrs Instruction attributes
 * \param decisions Decisions made on the instruction
 * \param outputs Names of the output random variables
 * \return A string representing the python api call
 */
using FInstructionAsPython =
    ffi::TypedFunction<ffi::String(const ffi::Array<Any>& inputs, const ffi::Array<Any>& attrs,
                                   const Any& decision, const ffi::Array<ffi::String>& outputs)>;

/*!
 * \brief Type of the functor that serialize its attributes to JSON
 * \param attrs The attributes to be serialized
 * \return An array, serialized attributes
 * \note This functor is nullable
 */
using FInstructionAttrsAsJSON = ffi::TypedFunction<ffi::ObjectRef(ffi::Array<Any> attrs)>;

/*!
 * \brief Type of the functor that deserialize its attributes from JSON
 * \param json_attrs The attributes to be serialized
 * \return An array, deserialized attributes
 * \note This functor is nullable
 */
using FInstructionAttrsFromJSON = ffi::TypedFunction<ffi::Array<Any>(ffi::ObjectRef json_attrs)>;

/*!
 * \brief Kind of an instruction, e.g. Split, Reorder, etc.
 * Besides the name, every kind of instruction has its own properties, including:
 * 1) A boolean indicating if the instruction is pure, i.e. change nothing in the schedule state
 * 2) A functor that applies the instruction to a TensorIR schedule
 * 3) A functor that converts the instruction to a statement in python syntax
 * 4) A functor that serialize its attributes to JSON
 * 5) A functor that deserialize its attributes from JSON
 *
 * Unlike `tvm::OpObj`, `InstructionKindObj` doesn't support unstructured properties,
 * mainly because there is no such usecase yet to add any other property.
 */
class InstructionKindObj : public ffi::Object {
 public:
  /*! \brief The name of a kind of instructions */
  ffi::String name;
  /*!
   * \brief Indicates if the instruction is pure, i.e. removing it alone doesn't mutate the schedule
   * state. For example, the instruction `GetSBlock` is pure because it changes
   * nothing, while `ComputeInline` is not because removing it leads to a different resulting
   * schedule.
   */
  bool is_pure{false};
  /*! \brief A functor that applies the instruction to a TensorIR schedule */
  FInstructionApply f_apply_to_schedule{nullptr};
  /*! \brief A functor that converts the instruction to a statement in python syntax */
  FInstructionAsPython f_as_python{nullptr};
  /*!
   * \brief A functor that serialize its attributes to JSON
   * \note If the functor is null, it means no conversion is needed
   */
  FInstructionAttrsAsJSON f_attrs_as_json{nullptr};
  /*!
   * \brief A functor that deserialize its attributes from JSON
   * \note If the functor is null, it means no conversion is needed
   */
  FInstructionAttrsFromJSON f_attrs_from_json{nullptr};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<InstructionKindObj>()
        .def_ro("name", &InstructionKindObj::name)
        .def_ro("_is_pure", &InstructionKindObj::is_pure);
  }

  /*! \brief Checks if the instruction kind is EnterPostproc */
  bool IsPostproc() const;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.InstructionKind", InstructionKindObj, ffi::Object);
};

/*!
 * \brief Managed reference to InstructionKindObj
 * \sa InstructionKindObj
 */
class InstructionKind : public ffi::ObjectRef {
 public:
  explicit InstructionKind(ffi::UnsafeInit tag, ffi::ObjectPtr<InstructionKindObj> data)
      : ffi::ObjectRef(tag) {
    data_ = std::move(data);
  }

  /*!
   * \brief Retrieve an InstructionKind using its name
   * \param name The registered name of the InstructionKind
   * \return The InstructionKind retrieved
   */
  static InstructionKind Get(const ffi::String& name);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(InstructionKind, ffi::ObjectRef,
                                                InstructionKindObj);
};

/*! \brief Schedule instructions each corresponds to a schedule primitive */
class InstructionObj : public ffi::Object {
 public:
  explicit InstructionObj(ffi::UnsafeInit tag) : kind(tag) {}

  explicit InstructionObj(InstructionKind kind) : kind(std::move(kind)) {}
  /*! \brief The kind of the instruction */
  InstructionKind kind;
  /*!
   * \brief The input random variables of the instruction, and the type of each element can be one
   * of the following:
   * - SBlockRV
   * - LoopRV
   * - ExprRV
   * - double
   * - int64_t
   * - String
   * - null pointer
   */
  ffi::Array<Any> inputs;
  /*!
   * \brief The attributes of the instruction. Similar to attributes of an operator,
   * attributes of an instruction are arbitrary constant metadata required by the instructions.
   * For example, the name of the block to be retrieved in `GetSBlock`.
   */
  ffi::Array<Any> attrs;
  /*! \brief The output random variables of the instruction, and the type of each element can be one
   * of the following:
   * - SBlockRV
   * - LoopRV
   * - ExprRV, atomic variables only, won't be constants or composite PrimExpr
   */
  ffi::Array<Any> outputs;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<InstructionObj>()
        .def_ro("kind", &InstructionObj::kind)
        .def_ro("inputs", &InstructionObj::inputs)
        .def_ro("attrs", &InstructionObj::attrs)
        .def_ro("outputs", &InstructionObj::outputs);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.Instruction", InstructionObj, ffi::Object);
};

/*!
 * \brief Managed reference to InstructionObj
 * \sa InstructionObj
 */
class Instruction : public ffi::ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param kind The kind of the instruction
   * \param inputs The input random variables of the instruction
   * \param attrs The attributes of the instruction
   * \param outputs The output random variables of the instruction
   */
  explicit Instruction(InstructionKind kind, ffi::Array<Any> inputs, ffi::Array<Any> attrs,
                       ffi::Array<Any> outputs);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Instruction, ffi::ObjectRef, InstructionObj);
};

/*! \brief Temporary builder for a canonical named instruction kind. */
class InstructionKindDef {
 public:
  /*! \brief Find or create the canonical kind and assign its name.
   * \param name The instruction kind name.
   */
  explicit InstructionKindDef(const ffi::String& name);

  /*! \brief Set whether removing this instruction leaves schedule state unchanged.
   * \param is_pure Whether the instruction is pure.
   * \return This builder for chaining.
   */
  InstructionKindDef& set_is_pure(bool is_pure) {
    get_mutable()->is_pure = is_pure;
    return *this;
  }

  /*! \brief Set the schedule application callback.
   * \param f_apply_to_schedule Callback that applies the instruction.
   * \return This builder for chaining.
   */
  InstructionKindDef& set_apply_to_schedule(FInstructionApply f_apply_to_schedule) {
    get_mutable()->f_apply_to_schedule = std::move(f_apply_to_schedule);
    return *this;
  }

  /*! \brief Set the Python trace formatting callback.
   * \param f_as_python Callback that formats an instruction.
   * \return This builder for chaining.
   */
  InstructionKindDef& set_as_python(FInstructionAsPython f_as_python) {
    get_mutable()->f_as_python = std::move(f_as_python);
    return *this;
  }

  /*! \brief Set the optional JSON serialization callback.
   * \param f_attrs_as_json Callback, or null when conversion is unnecessary.
   * \return This builder for chaining.
   */
  InstructionKindDef& set_attrs_as_json(FInstructionAttrsAsJSON f_attrs_as_json) {
    get_mutable()->f_attrs_as_json = std::move(f_attrs_as_json);
    return *this;
  }

  /*! \brief Set the optional JSON deserialization callback.
   * \param f_attrs_from_json Callback, or null when conversion is unnecessary.
   * \return This builder for chaining.
   */
  InstructionKindDef& set_attrs_from_json(FInstructionAttrsFromJSON f_attrs_from_json) {
    get_mutable()->f_attrs_from_json = std::move(f_attrs_from_json);
    return *this;
  }

 private:
  InstructionKindObj* get_mutable() const {
    return const_cast<InstructionKindObj*>(inst_kind_.get());
  }

  InstructionKind inst_kind_;
};

}  // namespace s_tir
}  // namespace tvm

#endif  //  TVM_S_TIR_SCHEDULE_INSTRUCTION_H_
