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
#ifndef TVM_RELAX_SCRIPT_IR_BUILDER_IR_H_
#define TVM_RELAX_SCRIPT_IR_BUILDER_IR_H_

#include <tvm/base.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/script/ir_builder/frame.h>
#include <tvm/relax/type.h>
#include <tvm/script/ir_builder/base.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

/*! \brief Find a module virtual device by target kind and index. */
TVM_DLL ffi::Optional<tvm::relax::VDevice> LookupVDevice(ffi::String target_kind, int device_index);

/////////////////////////////// Function ////////////////////////////////

/*!
 * \brief Start a function frame.
 * \param is_pure Whether the function is annotated as pure.
 * \param is_private Whether the function is annotated as private.
 * \return The created ir_builder Function frame.
 */
TVM_DLL FunctionFrame Function(bool is_pure, bool is_private, bool is_declaration = false,
                               bool local = false);

/*! \brief Define a local function under its cached declaration identity. */
TVM_DLL FunctionFrame LocalFunction(bool is_pure, const tvm::Var& reference);

/*! \brief Add a cached parameter without changing its identity. */
TVM_DLL tvm::Var ArgVar(const ffi::String& name, const tvm::Var& var);

/*!
 * \brief Add a parameter to the last function frame.
 * \param name The name of the parameter.
 * \param ty The ty of the parameter.
 * \return The created function parameter var.
 */
TVM_DLL tvm::Var Arg(const ffi::String& name, const tvm::Type& ty);

/*!
 * \brief Specify the name of the last function frame.
 * \param name The function name.
 */
TVM_DLL void FuncName(const ffi::String& name);

/*!
 * \brief Specify the attrs of the last function frame.
 * \param attrs The function attrs.
 */
TVM_DLL void FuncAttrs(ffi::Map<ffi::String, Any> attrs);

/*!
 * \brief Specify the return type of the last function frame.
 * \param ret_ty The return type.
 */
TVM_DLL void FuncRetType(const tvm::Type& ret_ty);

/*!
 * \brief Specify the return value of the last function frame.
 * \param value The return value.
 */
TVM_DLL void FuncRetValue(const tvm::relax::Expr& value);

///////////////////////////// BindingBlock //////////////////////////////

/*!
 * \brief Start a binding block frame.
 * \return The created ir_builder Block frame.
 */
TVM_DLL BindingBlockFrame BindingBlock();

/*!
 * \brief Start a dataflow binding block frame.
 * \return The created ir_builder Block frame.
 */
TVM_DLL BindingBlockFrame Dataflow();

/*!
 * \brief Expose the dataflow block output variables as global ones
 * \param vars The output variables of a dataflow block
 */
TVM_DLL void DataflowBlockOutput(const ffi::Array<tvm::Var>& vars);

////////////////////////////// Bindings ////////////////////////////////

/*!
 * \brief Emit a binding to the last binding block frame.
 * \param value The right side value of the bindings to be emitted.
 * \param annotate_ty The optional type annotation for the emitted value.
 * \return The left side var of the emitted binding.
 */
TVM_DLL tvm::Var Emit(const tvm::relax::Expr& value,
                      const ffi::Optional<tvm::Type>& annotate_ty = std::nullopt);

/*!
 * \brief Emit a match_cast binding to the last binding block frame.
 * \param value The value of the MatchCast to be emitted.
 * \param ty The type of the MatchCast to be emitted.
 * \return The left side var of the emitted binding.
 */
TVM_DLL tvm::Var EmitMatchCast(const tvm::relax::Expr& value, const tvm::Type& ty);

/*!
 * \brief Emit a binding to the last binding block frame.
 * \param binding The binding to be emitted.
 * \return The left side var of the emitted binding.
 */
TVM_DLL tvm::Var EmitVarBinding(const tvm::relax::VarBinding& binding);

/*! \brief Emit a binding with separate statement and variable-name ranges. */
TVM_DLL tvm::Var EmitWithLoc(const tvm::relax::Expr& value,
                             const ffi::Optional<tvm::Type>& annotate_ty, const Location& name_loc,
                             const Location& loc = UnknownLoc());

/*! \brief Emit a match cast with separate statement and variable-name ranges. */
TVM_DLL tvm::Var EmitMatchCastWithLoc(const tvm::relax::Expr& value, const tvm::Type& ty,
                                      const Location& name_loc, const Location& loc = UnknownLoc());

///////////////////////////// If Then Else /////////////////////////////

/*!
 * \brief Create an if statement.
 * \param condition The condition of if statement.
 * \return The result IfFrame.
 */
IfFrame If(tvm::relax::Expr condition);
/*!
 * \brief Create a then.
 * \return The result ThenFrame.
 */
ThenFrame Then();
/*!
 * \brief Create an else.
 * \return The result ElseFrame.
 */
ElseFrame Else();

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_RELAX_SCRIPT_IR_BUILDER_IR_H_
