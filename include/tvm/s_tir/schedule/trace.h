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
#ifndef TVM_S_TIR_SCHEDULE_TRACE_H_
#define TVM_S_TIR_SCHEDULE_TRACE_H_

#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/schedule/instruction.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

// Forward declaration
class Trace;

/*!
 * \brief A callback that allows users to mutate decisions on the fly
 * when applying instructions. The signature of the callback is:
 * \param inst The instruction
 * \param inputs The input random variables
 * \param attrs The attributes
 * \param decision The original decision
 * \return A new decision
 */
using FTraceDecisionProvider =
    ffi::TypedFunction<Any(const Instruction& inst, const ffi::Array<Any>& inputs,
                           const ffi::Array<Any>& attrs, const Any& decision)>;

/*!
 * \brief An execution trace of a scheduling program
 *
 * A trace has two parts:
 * 1) The instructions invoked so far in the program execution
 * 2) The random decisions made upon those instructions, if any
 *
 * A trace can be serialized to:
 * 1) Roundtrippable JSON format: can be saved to file and loaded back
 * 2) Python syntax: allows users to copy-paste the trace to reproduce the scheduling process
 *
 * A trace can be applied to a TensorIR schedule by re-applying all its instructions possibly with
 * their decisions accordingly. Re-sampling is invoked if a sampling instruction doesn't have its
 * corresponding decision; Otherwise the existing decision will be reused accordingly.
 */
class TraceObj : public ffi::Object {
 public:
  /*! \brief The instructions invoked so far in the program execution */
  ffi::Array<Instruction> insts;
  /*! \brief The random decisions made upon those instructions */
  ffi::Map<Instruction, Any> decisions;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TraceObj>()
        .def_ro("insts", &TraceObj::insts)
        .def_ro("decisions", &TraceObj::decisions);
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.Trace", TraceObj, ffi::Object);

 public:
  /*!
   * \brief Retrieve the decision made on a specific instruction
   * \param inst The instruction whose decision is to be retrieved
   * \return The corresponding decision; nullptr if there is no decision made on the instruction
   */
  Any GetDecision(const Instruction& inst) const;
  /*!
   * \brief Append a new instruction to the trace
   * \param inst The new instruction to be appended
   */
  void Append(Instruction inst);
  /*!
   * \brief Append a new instruction with a random decision to the trace
   * \param inst The new instruction to be appended
   * \param decision The random decision made on this instruction
   * The type of `decision` depends on the instruction, e.g.
   * the decision of `SamplePerfectTile` has type `ffi::Array<IntImm>`
   */
  void Append(Instruction inst, Any decision);
  /*!
   * \brief Remove the last instruction, along with the decision made on that instruction, if any
   * \return The instruction removed; std::nullopt if the trace is empty
   */
  ffi::Optional<Instruction> Pop();
  /*!
   * \brief Apply the trace to a TensorIR schedule
   * \param sch The schedule to be applied onto
   * \param remove_postproc If postprocessing instructions are removed
   * \param decision_provider A callback that allows users to mutate decisions on the fly
   * when applying instructions.
   * \sa FTraceDecisionProvider
   */
  void ApplyToSchedule(Schedule sch, bool remove_postproc,
                       FTraceDecisionProvider decision_provider = nullptr) const;
  /*!
   * \brief Serialize the trace as a JSON-style object
   * \param remove_postproc If postprocessing instructions are removed
   * \return The JSON-style object
   */
  ffi::ObjectRef AsJSON(bool remove_postproc) const;
  /*!
   * \brief Serialize the trace as a sequence of python statements
   * \param remove_postproc If postprocessing instructions are removed
   * \return A sequence of python statements
   */
  ffi::Array<ffi::String> AsPython(bool remove_postproc) const;
  /*!
   * \brief Create a new trace with an instruction whose decision is changed,
   * assuming this instruction exists in the resulting trace
   * \param inst The instruction whose decision is to be changed
   * \param decision The decision to be changed to
   * \param remove_postproc If postprocessing instructions are removed
   * \return The new trace with the decision changed
   */
  Trace WithDecision(Instruction inst, Any decision, bool remove_postproc) const;
  /*!
   * \brief Simplify the trace with dead-code elimination
   * \param remove_postproc If postprocessing instructions are removed
   * \return A simplified trace
   */
  Trace Simplified(bool remove_postproc) const;
};

/*!
 * \brief Managed reference to TraceObj
 * \sa TraceObj
 */
class Trace : public ffi::ObjectRef {
 public:
  /*! \brief Default constructor. Creating an empty trace. */
  Trace();
  /*!
   * \brief Constructor. Creating a trace from existing instructions and their decisions
   * \param insts The instructions used
   * \param decisions The decisions made in sampling
   */
  explicit Trace(ffi::Array<Instruction> insts, ffi::Map<Instruction, Any> decisions);
  /*!
   * \brief Apply a JSON-serialized trace to a TensorIR schedule
   * \param json The JSON-serialized trace
   * \param sch The TensorIR schedule
   */
  static void ApplyJSONToSchedule(ffi::ObjectRef json, Schedule sch);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Trace, ffi::ObjectRef, TraceObj);
};

}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_SCHEDULE_TRACE_H_
