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
 * \file tvm/relax/block_builder.h
 * \brief The utility for constructing Relax binding blocks.
 */
#ifndef TVM_RELAX_BLOCK_BUILDER_H_
#define TVM_RELAX_BLOCK_BUILDER_H_

#include <tvm/base.h>
#include <tvm/ir/unique_name_supply.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/utils.h>
#include <tvm/runtime/base.h>
#include <tvm/sym/analyzer.h>
#include <tvm/target/target.h>

namespace tvm {
namespace relax {

/*!
 * \brief A builder to build Relax binding blocks.
 *
 * BlockBuilder provides the following three categories
 * of main functionalities for IR building and transformations:
 *
 * - Global context management: manages the IRModule,
 *   allowing query, update the surrounding global context.
 *   Provide context tools for analysis.
 * - Scope management:
 *   - Manages block scopes for bulding nested blocks.
 *   - Emit bindings to the current scope.
 *   - Construct blocks by calling EndScope.
 * - Normalization: Take an Expr, normalize it
 *   to deduce shape/type, turn things into normal forms.
 *
 * Importantly, these three categories of features can be dependent
 * on each other. For example, when we emit into scope we will call
 * normalize to ensure the code is in normal form. Similarly, when we
 * normalize we could choose to emit into the current context.
 *
 * We would encourage the developers to keep these three category
 * in mind when using and developing BlockBuilder, we can group
 * the code in a logically clean way.
 *
 * BlockBuilderObj is implemented as a virtual interface to
 * allow logically grouped implementation and internal data
 * structures that are hidden from the users.
 */
class BlockBuilderObj : public ffi::Object {
 public:
  /*! \brief Explicit target for the operation being legalized, if one was supplied. */
  ffi::Optional<Target> target;

  //-------------------------------
  // Global Context management
  //-------------------------------
  /*!
   * \brief Get the unique name supply for generating unique names.
   *
   * \return The unique name supply.
   */
  virtual UniqueNameSupply name_supply() = 0;

  /*!
   * \brief Get the context IRModule in this builder.
   *
   * \note The context
   * \return The IRModule in this BlockBuilder.
   */
  virtual IRModule GetContextIRModule() const = 0;

  /*!
   * \brief Finalize the building process and return the result IRModule. Possibly rename
   * GlobalVars in the IRModule to ensure name uniqueness and the invariant:
   * every public function has the same name as its "global_symbol" attribute.
   *
   * \note this method should be called only once at the end of the building process, since it may
   * invalidate global vars previously returned by this builder. See also
   * transform::NormalizeGlobalVar.
   *
   * \return The result IRModule.
   */
  virtual IRModule Finalize() = 0;

  /*!
   * \brief Add a Relax function or a TIR tirx::Function to internal context module.
   * \param func The function to be added.
   * \param func_name_hint The name hint of the function to be added.
   * \note If the function to be added already exists, return its
   * GlobalVar directly.
   * \return The global var bound to the added function.
   */
  virtual GlobalVar AddFunction(const BaseFunc& func, ffi::String func_name_hint) = 0;

  /*!
   * \brief Update a Relax function or a TIR tirx::Function in the internal context module.
   * \param gv The global var referring the function to be updated.
   * \param function The updated function.
   */
  virtual void UpdateFunction(const GlobalVar& gv, BaseFunc function) = 0;

  //-------------------------------
  // Scope management
  //-------------------------------
  /*!
   * \brief Lookup the binding value that var binds to in the current emitted sequences.
   * \param var The input var.
   * \return The Expr bound to the input \p var.
   * \note For function parameters, this function returns std::nullopt.
   */
  virtual ffi::Optional<Expr> LookupBinding(const Var& var) = 0;

  /*!
   * \brief Begin a new scope, with optional parameters that
   *        are visible within the scope.
   *
   * Symbolic variables from the parent scope are not available.
   *
   * \param params Parameters that are visible within the scope.
   *
   * \note This function should be called when new scope is introduced
   *       (e.g. function bodies) to properly track the variable
   *       availability and help the best effort deduction.
   *
   * \sa EndScope
   */
  virtual void BeginScope(ffi::Optional<ffi::Array<Var>> params) = 0;

  /*!
   * \brief Begin a new scope, which inherits visible parameters from
   * its parent scope.
   *
   * Symbolic variables from the parent scope are available.
   *
   * \note This function should be called when an inner scope is
   *       introduced (e.g. conditional branches) to properly track
   *       the variable availability and help the best effort
   *       deduction.
   *
   * \sa EndScope
   */
  virtual void BeginInnerScope() = 0;

  /*!
   * \brief Append a definition to the current scope.
   *
   * \param var A variable within the current scope.
   *
   * \note This function should be called when a new variable is
   *       defined that may impact struct inference (e.g. MatchCast)
   *       to properly track the variable availability and help the
   *       best effort deduction.
   *
   * \sa EndScope
   */
  virtual void AddDefinitionToScope(Var var) = 0;

  /*! \brief End the previously defined scope. */
  virtual void EndScope() = 0;

  /*! \brief Begin to build a DataflowBlock. */
  virtual void BeginDataflowBlock() = 0;

  /*! \brief Begin to build a BindingBlock. */
  virtual void BeginBindingBlock() = 0;
  /*!
   * \brief End building a BindingBlock.
   * \return The BindingBlock being built.
   */
  virtual BindingBlock EndBlock() = 0;

  /*!
   * \brief Check if the block being built is DataflowBlock or not.
   * \return A boolean that indicates if the block being built is DataflowBlock or not.
   */
  virtual bool CurrentBlockIsDataFlow() = 0;

  /*!
   * \brief Emits an Expr, and returns the variable it is bound to.
   * \param expr The Expr to be emitted.
   * \param name_hint Name hint for the bound variable.
   * \return The new variable that \p expr is bound to.
   *
   * \note This Emit function normalizes the \p expr, and
   *       performs shape and type deductions by calling Normalize.
   */
  virtual Var Emit(Expr expr, ffi::String name_hint = "") = 0;

  /*!
   * \brief Emit a MatchCast.
   * \param value The input value.
   * \param ty The type to be matched.
   * \param name_hint Name hint for the bound variable.
   * \return The variable bound to the MatchCast.
   */
  virtual Var EmitMatchCast(Expr value, Type ty, ffi::String name_hint = "") = 0;

  /*!
   * \brief Generate an output for the current dataflow block.
   * \param output The output variable of the block.
   * \param name_hint Name hint for the bound variable.
   * \return The variable bound to \p output.
   */
  virtual Var EmitOutput(Expr output, ffi::String name_hint = "") = 0;

  /*!
   * \brief Emit a binding that is already normalized.
   *
   * \param normalized_binding A binding whose value is already normalized.
   *
   * \note This function requires binding to be pre-normalized.
   */
  virtual void EmitNormalized(Binding normalized_binding) = 0;

  /*!
   * \brief Convert an expression to normal form, and try to eagerly infer types and shapes.
   * \param expr The input expression.
   * \return The normalized expression.
   *
   * \note Invariant: If any of the sub expr have ty field.
   *       they must have already been normalized.
   */
  virtual Expr Normalize(const Expr& expr) = 0;

  /*!
   * \brief Normalize argument to a call or another IRNode.
   * \param expr The input expression.
   * \return The normalized expression.
   *
   * \note This function will create a binding var for non-leaf expressions such as Call.
   */
  virtual Expr NormalizeArgument(const Expr& expr) = 0;

  /*!
   * \brief Get the analyzer of the BlockBuilder.
   * \return The BlockBuilder's arithmetic analyzer.
   */
  virtual sym::Analyzer GetAnalyzer() = 0;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.BlockBuilder", BlockBuilderObj, ffi::Object);
};

class BlockBuilder : public ffi::ObjectRef {
 public:
  /*!
   * \brief Create a BlockBuilder.
   *
   * \param ctx_mod Optional before-transformation context module for rewriting.
   *
   * \return The created BlockBuilder.
   *
   * \note When rewriting an existing IRModule, it is important to pass it in as
   *       ctx_mod so you can lookup the context functions for cross function
   *       call analysis.
   */
  TVM_DLL static BlockBuilder Create(ffi::Optional<IRModule> ctx_mod);

  /*! \brief A marker struct to disable FNormalize
   *
   * This struct is used as a marker to disable the use of FNormalize
   * by this block builder.  This should only be used for TVMScript
   * parsing, which may require producing un-normalized Relax IR for
   * testing purposes, and to ensure that round-trips are unchanged.
   *
   * The name is deliberately verbose to draw attention during a code
   * review.  The explicit default constructor prevents aggregate
   * initialization, ensuring that the full name of the marker struct
   * appears at the callsite.
   *
   * This constructor is marked as no-lint to allow a zero-parameter
   * constructor to be marked as explicit.  The constructor must be
   * explicit in order to disable aggregate initialization in C++17.
   * While C++20 disables aggregate initialization when a
   * user-declared constructor is present, C++17 only disables
   * aggregate initialization when a user-defined constructor is
   * present.  Therefore, we need to mark the zero-parameter
   * constructor as explicit in order to prevent aggregate
   * initialization, and to ensure that the name appears at all
   * callsites.
   */
  struct DisableOperatorSpecificNormalizationForTVMScript {
    explicit DisableOperatorSpecificNormalizationForTVMScript() = default;  // NOLINT(*)
  };
  /*!
   * \brief Create a BlockBuilder.
   *
   * \param ctx_mod Optional before-transformation context module for rewriting.
   *
   * \param tag An instance of DisableOperatorSpecificNormalizationForTVMScript
   *
   * \return The created BlockBuilder.
   *
   * \note When rewriting an existing IRModule, it is important to pass it in as
   *       ctx_mod so you can lookup the context functions for cross function
   *       call analysis.
   */
  TVM_DLL static BlockBuilder Create(ffi::Optional<IRModule> ctx_mod,
                                     DisableOperatorSpecificNormalizationForTVMScript tag);

  explicit BlockBuilder(ffi::UnsafeInit tag, ffi::ObjectPtr<BlockBuilderObj> n)
      : ffi::ObjectRef(tag) {
    data_ = std::move(n);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BlockBuilder, ffi::ObjectRef, BlockBuilderObj);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BLOCK_BUILDER_H_
