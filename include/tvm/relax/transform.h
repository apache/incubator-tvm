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
 * \file tvm/relax/transform.h
 * \brief Relax specific transformation passes.
 */
#ifndef TVM_RELAX_TRANSFORM_H_
#define TVM_RELAX_TRANSFORM_H_

#include <tvm/base.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/index_map.h>

namespace tvm {
namespace relax {
namespace transform {

using Pass = tvm::transform::Pass;
using PassInfo = tvm::transform::PassInfo;
using PassContext = tvm::transform::PassContext;
using Function = tvm::relax::Function;
using DataflowBlock = tvm::relax::DataflowBlock;
using tvm::transform::CreateModulePass;
using LayoutCb = ffi::TypedFunction<ffi::Map<ffi::String, ffi::Array<ffi::String>>(Call)>;

/*!
 * \brief Create a function pass.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the function pass.
 * \param name The name of the function pass.
 *
 * \return The created function pass.
 */
TVM_DLL Pass CreateFunctionPass(std::function<Function(Function, IRModule, PassContext)> pass_func,
                                int opt_level, ffi::String name);

/*!
 * \brief Create a dataflowblock pass.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the dataflowblock pass.
 * \param name The name of the dataflowblock pass.
 *
 * \return The created dataflowblock pass.
 */
TVM_DLL Pass CreateDataflowBlockPass(
    std::function<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func, int opt_level,
    ffi::String name);

/*!
 * \brief Perform lambda lifting to lift functions from nested into global.
 *
 * \return The Pass.
 */
TVM_DLL Pass LambdaLift();

/*!
 * \brief Transform all dataflow structure to non-dataflow version.
 *
 * \return The Pass.
 */
TVM_DLL Pass ToNonDataflow();

/*!
 * \brief Activate force_pure on all pure functions in the module
 * and unwrap all pure override ops into the normal versions.
 *
 * This effectively means that there will be no more purity tracking,
 * useful for low-level code generation.
 *
 * \return The Pass.
 *
 * \note Should be used after ToNonDataflow()
 */
TVM_DLL Pass RemovePurityChecking();

/*!
 * \brief Perform explicit tensor allocation for call_tir and call_dps_packed.
 *
 * \return The Pass.
 */
TVM_DLL Pass CallTIRRewrite();

/*!
 * \brief Convert all reshape-like call_tir whose corresponding binding
 * vars are DataflowVars to relax.reshape operator calls. The relax.reshape
 * calls will be lowered an external builtin function call in a subsequent
 * pass, where the external builtin function does a CreateView operation
 * at runtime, instead of doing real data copy.
 * Here "reshape-like" includes reshape, expand_dims, flatten, etc.
 *
 * \return The Pass.
 * \note The pass is applied at the first stage of Relax VM build, before
 * rewriting call_tir, as this pass requires dataflow information.
 */
TVM_DLL Pass RewriteDataflowReshape();

/*!
 * \brief The static memory planning pass on BindingBlock level.
 * The pass will reuse allocated memory to its best effort, in order to
 * reduce the total amount of allocated memory size.
 *
 * The pass "supports" dynamic shape in the way of TIR variable bound
 * annotations. We can optionally annotate the attributes "tir_var_upper_bound"
 * and "tir_var_lower_bound" to Relax functions. The attribute values are dicts
 * from strings to integers, denoting the name of TIR variables to the bound
 * values of the TIR vars.
 * Note: The annotated bound attributes only apply to TIR vars in the
 * function signature for clarity.
 *
 * For example, we can annotate a Relax function with
 *   `R.func_attr({"tir_var_lower_bound": {"n": 1}, "tir_var_upper_bound": {"n": 1024}})`.
 * It means the variable that names "n" in the function signature will have
 * range [1, 1024]. And we will use these bounds during memory planning.
 * If lower bound is not specified, it defaults to 0.
 *
 * \return The pass.
 */
TVM_DLL Pass StaticPlanBlockMemory();

/*!
 * \brief Attach global_symbol to Relax functions and TIR Functions for codegen.
 *
 * \return The Pass.
 */
TVM_DLL Pass AttachGlobalSymbol();

/*!
 * \brief Transform Relax IR to normal form: transform AST to A-normal form, and fill the
 * ty of expressions.
 *
 * \return The Pass.
 */
TVM_DLL Pass Normalize();

/*!
 * \brief Possibly rename the GlobalVar in an IRModule to ensure these properties:
 * 1. (Invariant) First ensure every public function has the same name as its "global_symbol"
 *    attribute;
 * 2. To ensure 1., we may need to rename private functions with conflicting names;
 * 3. Finally, the name of every GlobalVar is unique in the IRModule.
 */
TVM_DLL Pass NormalizeGlobalVar();

/*!
 * \brief Simplify a Relax module by folding var bindings and match shape nodes,
 * as well as tuple indices.
 * Best used alongside constant folding and eliminating unused bindings.
 *
 * \note If a dataflow var is used only in a binding to the dataflow block
 * output var (i.e., a non-dataflow var), this pass will also remove the dataflow var
 * and replaces the output var's binding with the dataflow var's direct definition.
 *
 * \return The Pass.
 */
TVM_DLL Pass CanonicalizeBindings();

/*!
 * Eliminate common subexpressions within functions.
 * \return The pass that eliminates common subexpressions.
 *
 * \note For nested functions, this pass performs CSE *within* those functions.
 * \param call_only If true, enable eliminating only call nodes.
 */
TVM_DLL Pass EliminateCommonSubexpr(bool call_only = false);

/*!
 * \brief Bind params of function of the module to constant tensors.
 *
 * \param func_name The name of the function to bind parameters.
 * \param params The parameters to bind.
 *
 * \return The Pass.
 */
TVM_DLL Pass BindParams(ffi::String func_name, ffi::Map<Any, ffi::ObjectRef> params);

/*!
 * \brief Bind symbolic vars to constant shape values.
 *
 * \param binding_map The dictionary of symbolic variables and their
 *      constant shape values.  Dictionary keys may be either a
 *      `tirx.Var` or a string name of the variable.  If the variables
 *      are referred to by name, the name must uniquely identify a
 *      symbolic variable in each function where it is used.
 *
 * \param func_name The name of the function in which to bind shape
 *      values.  If std::nullopt, all functions in the module will be
 *      updated.
 *
 * \return The Pass.
 */
TVM_DLL Pass BindSymbolicVars(ffi::Map<ffi::Variant<PrimVar, ffi::String>, PrimExpr> binding_map,
                              ffi::Optional<ffi::String> func_name = std::nullopt);

/*!
 * \brief Fold constant expressions within dataflow blocks.
 *
 * \note ConvertToDataflow may need to be called first to provide dataflow blocks.
 *
 * \return The Pass.
 */
TVM_DLL Pass FoldConstant();

/*!
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR Functions.
 *
 * For each high-level operator, we register the way of legalizing it as a
 * function, which takes a context BlockBuilder and the Call being legalized
 * as input, and returns the legalized call. Here the input BlockBuilder is
 * mainly used for adding the tirx::Function created by call_te into the context
 * IRModule.
 *
 * The legalization function for each operator is registered as an attribute (with
 * attribute key `FLegalize`) of the operator.
 *
 * For customizability, the user can pass their own legalization by an optional customized map,
 * with the key to be the operator name and value to be the legalization function.
 * The default legalization function will be overridden by the customized one.
 *
 * \param cmap The customized operator legalization function map. The customized function
 * will override the default one.
 * \param skip_ops The list operator names which need to be skipped from legalization
 * \param enable_warning A boolean value indicating if to print warnings for TIR functions not
 * showing up in the database.
 * \param target Default compilation target; virtual-device annotations take precedence.
 * \return The Pass.
 */
TVM_DLL Pass LegalizeOps(ffi::Optional<ffi::Map<ffi::String, ffi::Function>> cmap,
                         ffi::Optional<ffi::Array<ffi::String>> skip_ops,
                         bool enable_warning = false, ffi::Optional<Target> target = std::nullopt);

/*!
 * \brief Propagate virtual device information.
 * \return The Pass.
 */
TVM_DLL Pass RealizeVDevice();

/*!
 * \brief Attach layout free tensors to the tirx::Function.
 *
 * This pass is used to attach layout free tensors to the tirx::Function according to
 * the function usage in the relax function. Currently, the layout free tensors are the model
 * weights and relax constants.
 *
 * \note We recommend applying CanonicalizeBindings before this pass.
 * \return The Pass.
 */
TVM_DLL Pass AttachAttrLayoutFreeTensors();

/*!
 * \brief Split the layout rewrite preproc block to a separate tirx::Function.
 *
 * This pass is used in the prepack weight after meta_schedule tuning.
 *
 * \return The Pass.
 */
TVM_DLL Pass SplitLayoutRewritePreproc();

/*!
 * \brief Lift transformation of the parameters of a function.
 *
 * When some inputs of the function is marked as 'parameters' (the model weights), this pass
 * identifies the transformation of the parameters and lifts them to a separate function called
 * `transform_params`. `transform_params` takes a tuple of the original parameters as input and
 * returns a tuple of the transformed parameters. The original function will be rewritten to accept
 * a tuple of transformed parameters as input.
 *
 * Users are expected to invoke the `transform_params` function in runtime and pass the transformed
 * parameters to the original function as input.
 *
 * \param shared_transform Indicates how the parameter transformation function will be produced.
 *    - `False` (default): A separate parameter transformation function will be produced for each
 *      function with the `"num_input"` attribute.
 *
 *    - `True`: A single parameter transformation function will be produced, containing the
 *      preprocessing steps common across all functions with the `"num_input"` attribute.
 *
 *    - List[str]: A single parameter transformation function will be produced, containing the
 *      preprocessing steps common across each function whose name is in the list. Passing a list of
 *      all functions with the `"num_input"` attribute or an empty list is equivalent to passing
 *      `True`.
 *
 * \return The Pass.
 */
TVM_DLL Pass
LiftTransformParams(ffi::Variant<bool, ffi::Array<ffi::String>> shared_transform = false);

/*!
 * \brief Update virtual device.
 * \param new_vdevice The new virtual device.
 * \param index The device index indicates the device on which the update will be performed.
 * \return The Pass.
 */
TVM_DLL Pass UpdateVDevice(VDevice new_vdevice, int64_t index);

/*! \brief Expand tuple arguments to internal functions
 *
 * \return The Pass
 */
TVM_DLL Pass ExpandTupleArguments();

/*! \brief Remove unused parameters to internal functions
 *
 * \return The Pass
 */
TVM_DLL Pass RemoveUnusedParameters();

/*! \brief Remove unused outputs from internal functions
 *
 * \return The Pass
 */
TVM_DLL Pass RemoveUnusedOutputs();

/*!
 * \brief Annotate Op Pattern Kind for TIR functions, which is used in FuseOps.
 * \note It is an auto-detect pass for "unscheduled functions", the op_pattern will be
 *       "opaque" of we can't detect it. Users can manually annotate the attr `op_pattern`
 *       to function.
 * \return The Pass.
 */
TVM_DLL Pass AnnotateTIROpPattern();

/*!
 * \brief This pass groups bindings in a dataflow block of Relax functions and generates a new
 * grouped Relax function for each group, according to the fusion algorithm described in the pass
 * implementation. By grouping bindings into new Relax functions, we substitute the bindings in the
 * function being manipulated into function calls to the new grouped function.
 *
 * A follow-up pass named "FuseTIR" will generate a TIR tirx::Function for each grouped function.
 * \param fuse_opt_level The level of fuse optimization.
 *        -1 indicates that the level will be inferred from pass context.
 * \return The Pass.
 */
TVM_DLL Pass FuseOps(int fuse_opt_level = -1);

/*!
 * \brief The pattern object used as the input of FuseOpsByPattern. For bindings to be
 * fused, it needs to be matched with `pattern` and the `check` function needs to return
 * true.
 */
class FusionPatternObj : public ffi::Object {
 public:
  explicit FusionPatternObj(DFPattern value) : pattern(std::move(value)) {}
  explicit FusionPatternObj(ffi::UnsafeInit) : pattern(ffi::UnsafeInit{}) {}

  /*!
   * \brief The name of pattern. It becomes the value of the kComposite attribute
   * of a fused function after successful matching
   */
  ffi::String name;

  /*!
   * \brief The dataflow pattern that will be used to match expression in the DataflowBlock.
   * All the call nodes covered by the pattern will be extracted into the fused function.
   */
  DFPattern pattern;

  /*!
   * \brief The map which is used to extract important expressions from the pattern match
   * result. All DFPattern in this map should be part of the `pattern`.
   */
  ffi::Map<ffi::String, DFPattern> annotation_patterns;

  /*!
   * \brief The function to determine whether the match result is accepted. This can be
   * std::nullopt if check function is not necessary for this pattern.
   *
   * It should have signature
   * bool(const PatternCheckContext& context)
   */
  ffi::Optional<ffi::Function> check;

  /*!
   * \brief The function to get attributes for fused function
   *
   * It should have signature
   * ffi::Map<ffi::String, Any>(const ffi::Map<ffi::String, Expr>& context)
   */
  ffi::Optional<ffi::Function> attrs_getter;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FusionPatternObj>()
        .def_ro("name", &FusionPatternObj::name)
        .def_ro("pattern", &FusionPatternObj::pattern)
        .def_ro("annotation_patterns", &FusionPatternObj::annotation_patterns)
        .def_ro("check", &FusionPatternObj::check)
        .def_ro("attrs_getter", &FusionPatternObj::attrs_getter);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.transform.FusionPattern", FusionPatternObj, ffi::Object);
};

class FusionPattern : public ffi::ObjectRef {
 public:
  FusionPattern(ffi::String name, DFPattern pattern,
                ffi::Map<ffi::String, DFPattern> annotation_patterns,
                ffi::Optional<ffi::Function> check, ffi::Optional<ffi::Function> attrs_getter);

  FusionPattern(ffi::String name, DFPattern pattern)
      : FusionPattern(name, pattern, {}, std::nullopt, std::nullopt) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FusionPattern, ffi::ObjectRef, FusionPatternObj);
};

/*!
 * \brief The input of FusionPattern::check.
 */
class PatternCheckContextObj : public ffi::Object {
 public:
  explicit PatternCheckContextObj(Expr matched_expr) : matched_expr(std::move(matched_expr)) {}
  explicit PatternCheckContextObj(ffi::UnsafeInit) : matched_expr(ffi::UnsafeInit{}) {}

  /*!
   * \brief The expression that's matched with the FusionPattern::pattern.
   */
  Expr matched_expr;

  /*!
   * \brief A map which contains all expressions matched by the sub patterns in
   * FusionPattern::annotation_patterns.
   */
  ffi::Map<ffi::String, Expr> annotated_expr;

  /*!
   * \brief Map from variable to its value. It contains variables from bindings that
   * is being fused by FuseOpsByPattern.
   */
  ffi::Map<Var, Expr> matched_bindings;

  /*!
   * \brief A map mapping variable definitions to a set of uses. It has all variables
   * used in the function.
   */
  ffi::Map<Var, ffi::Array<Var>> var_usages;

  /*!
   * \brief Map from value to its bound variable. It doesn't have variables after the
   * matched expression.
   */
  ffi::Map<Expr, Var> value_to_bound_var;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PatternCheckContextObj>()
        .def_ro("matched_expr", &PatternCheckContextObj::matched_expr)
        .def_ro("annotated_expr", &PatternCheckContextObj::annotated_expr)
        .def_ro("matched_bindings", &PatternCheckContextObj::matched_bindings)
        .def_ro("var_usages", &PatternCheckContextObj::var_usages)
        .def_ro("value_to_bound_var", &PatternCheckContextObj::value_to_bound_var);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.transform.PatternCheckContext", PatternCheckContextObj,
                                    ffi::Object);
};

class PatternCheckContext : public ffi::ObjectRef {
 public:
  PatternCheckContext(Expr matched_expr, ffi::Map<ffi::String, Expr> annotated_expr,
                      ffi::Map<Var, Expr> matched_bindings,
                      ffi::Map<Var, ffi::Array<Var>> var_usages,
                      ffi::Map<Expr, Var> value_to_bound_var);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PatternCheckContext, ffi::ObjectRef,
                                                PatternCheckContextObj);
};

/*!
 * \brief Reverse-mode automatic differentiation.
 *
 * This pass will differentiate one function in the IRModule. Now the input function must have only
 * one dataflow block.
 *
 * For a given function specified by `func_name`, it generates a new function with the name
 * `func_name + "_adjoint"`. The new function computes the gradient of the **differentiation
 * target** with respect to the arguments specified by `require_grads` of the original function.
 *
 * If the function has only one return value, the return value will be specified as target. If the
 * function has more than one return values, the target will be specified as the target_index-th
 * return value. The target must be a scalar (0-dim tensor).
 *
 * \param func_name The name of the specified function.
 * \param require_grads The relax variables whose adjoints is needed. Must be parameters of the
 * given function and should not be duplicate. If it is not specified, adjoints of all parameters
 * would be computed.
 * \param target_index If the specified function has more than one return values, specify the index
 * of the return value as the target. If it is not specified, the first return value will be the
 * target.
 * \return The Pass.
 *
 * \note ConvertToDataflow may need to be called first to provide dataflow blocks.
 */
TVM_DLL Pass Gradient(ffi::String func_name,
                      ffi::Optional<ffi::Array<Var>> require_grads = std::nullopt,
                      int target_index = 0);

/*!
 * \brief Apply pattern matching to each function in the given module, and group matched
 * expressions into a new function. The end result is similar to FuseOps, but fusion is driven
 * completely by the provided patterns.
 *
 * \param patterns The patterns to detect. The order of the patterns determines the order
 * of priority in which they are matched. Higher-priority patterns should come earlier in the list.
 * \param bind_constants Whether or not to keep bound constants of the grouped function.
 * \param annotate_codegen If true, wrap each created composite function with another function,
 * whose body consists only of a call to the composite function, and annotate the outer function
 * with kCodegen and kGlobalSymbol attributes. The kCodegen attribute is set as the prefix of the
 * corresponding pattern name. For example, "dnnl" if the pattern name is "dnnl.conv2d_relu".
 * This must be True if the created composite functions are intended to be offloaded to
 * an external backend without using the MergeCompositeFunctions pass.
 * \param entry_function_names The names of functions that should be considered as entry points. If
 * not specified, all externally exposed functions will be considered as entry points.
 * \return The Pass.
 *
 * \note Only operates within dataflow blocks. ConvertToDataflow may need to be called first.
 */
TVM_DLL Pass FuseOpsByPattern(const tvm::ffi::Array<FusionPattern>& patterns,
                              bool bind_constants = true, bool annotate_codegen = false,
                              const tvm::ffi::Array<ffi::String>& entry_function_names = {});

/*!
 * \brief Group one or multiple composite functions created by FuseOpsByPattern into a new
 *  function. The new function will be annotated with kCodegen and GlobalSymbol attributes,
 *  and it is intented to be offloaded to an external backend.
 *
 * \return The Pass.
 */
TVM_DLL Pass MergeCompositeFunctions();

/*!
 * \brief Fuse relax sub-function into a larger TIR function if possible.
    this pass works together with FuseOps to perform operator fusion.

 * \return The Pass.
 */
TVM_DLL Pass FuseTIR();

/*!
 * \brief Run codegen.
 * \param target_options pairs of target name and compilation options
 * \param entry_functions list of entry functions
 * \return The Pass.
 */
TVM_DLL Pass
RunCodegen(ffi::Optional<ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>>> target_options,
           ffi::Array<ffi::String> entry_functions);

/*!
 * \brief Decompose composite operators during inference. For example, The result of batch norm (a
 * triple) will be simplified. Operators like Attention, Erf, etc. can be also simplified into
 * several operators as well.
 *
 * \param func_name The name of the specified function. If not specified, the pass will run in
 * all functions.
 */
TVM_DLL Pass DecomposeOpsForInference(ffi::Optional<ffi::String> func_name);

/*!
 * \brief Decompose composite operators during training. For example, The result of batch norm (a
 * triple) will be simplified. Operators like Attention, Erf, etc. can be also simplified into
 * several operators as well.
 *
 * \param func_name The name of the specified function. If not specified, the pass will run in
 * all functions.
 */
TVM_DLL Pass DecomposeOpsForTraining(ffi::Optional<ffi::String> func_name);

/*!
 * \brief Returns a pass which replaces Functions which have matching kOperatorName attribute in \p
 * op_impl_map, with replacement tirx::Function that could possibly have different layouts on i/o
 * tensors. The layout transformations on i/o tensors is present in the \p op_tensor_transforms. The
 * pass inserts the layout transformations in the call sites of Functions being replaced to
 * transform i/o tensors into expected layout.
 *
 * \param op_impl_map Map from kOperatorName attr (e.g., relax.conv2d) to replacement tirx::Function
 * \param op_tensor_transforms Map from kOperatorName attr to layout transformations on each of the
 * tirx::Function i/o tensors.
 * \return The Pass.
 */
TVM_DLL Pass AlterOpImpl(
    const ffi::Map<ffi::String, tirx::Function>& op_impl_map,
    const ffi::Map<ffi::String, ffi::Array<ffi::Optional<tirx::IndexMap>>>& op_tensor_transforms);

/*!
 * \brief Layout conversion pass.
 * \param desired_layouts The desired layouts for some operators.
 * \param layout_cb custom call back to define layouts dynamically.
 * \return The Pass.
 * \note Operates only on dataflow blocks. ConvertToDataflow may need to be called first.
 */
TVM_DLL Pass ConvertLayout(ffi::Map<ffi::String, ffi::Array<ffi::String>> desired_layouts,
                           LayoutCb layout_cb);

/*!
 * \brief A pass that converts consecutive dataflow operations
 *   inside binding blocks into dataflow blocks.
 * \param min_size The minimum number of consecutive dataflow bindings
 *   required for the pass to create a new dataflow block
 * \return The Pass.
 */
TVM_DLL Pass ConvertToDataflow(int min_size = 2);

/*!
 * \brief Dead code elimination.
 * \sa RemoveAllUnused
 * Currently it removes:
 *   1. Unused local VarBindings
 *      (those where the bound var is unused and no impure operation is used).
 *   2. Unused Relax functions in the module.
 *      We detect the call chain from the entry function, and remove all unused functions.
 *
 * Any binding blocks that are left empty will be removed by the normalizer.
 *
 * \param entry_functions Names of functions that should be considered
 *     as entry points, in addition to any externally exposed functions.
 *
 * \return The Pass.
 */
TVM_DLL Pass DeadCodeElimination(ffi::Array<ffi::String> entry_functions = {});

/*!
 * \brief Pass that changes calls to operators that can be done in-place
 * (generally, these are elementwise operations) in dataflow blocks into in-place implementations.
 * Supported operators will be replaced by calls to `call_tir_inplace` that invoke in-place
 * tirx::Function implementations of those operators (which are based on the legalizations of those
 * operators).
 * \note ConvertToDataflow may need to be called first to provide dataflow blocks.
 * \return The pass.
 */
TVM_DLL Pass DataflowUseInplaceCalls();

/*!
 * \brief Automatic mixed precision pass. Currently the pass assumes the input module to be fp32
 * only, and will automatically cast fp32 to fp16 for certain ops.
 * \param out_dtype The output data type of gemm/conv, which is the data type of the accumulator.
 * \param fp16_input_names The names of function parameters whose dtype should become fp16. The
 * function signature would change accordingly.
 * \return The Pass.
 *
 * \note Mainly operates within dataflow blocks. ConvertToDataflow may need to be called first.
 */
TVM_DLL Pass ToMixedPrecision(
    DLDataType out_dtype, ffi::Optional<ffi::Array<ffi::String>> fp16_input_names = std::nullopt);

/*!
 * \brief Rewrite a Relax module for executing with CUDA graph. This pass identifies
 * the regions that can be executed with CUDA graph and lifts them into new functions for runtime
 * graph capturing.
 */
TVM_DLL Pass RewriteCUDAGraph();

/*!
 * \brief This pass updates the var_tensor mapping of Functiontions from the call_tir info.
 * Primarily used to update the VDevice information if any changes occurred from the caller.
 * This pass recreates the tensors and updates the map.
 */
TVM_DLL Pass SpecializeFunctionBasedOnCallSite();

}  // namespace transform
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_H_
