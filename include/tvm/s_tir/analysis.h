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
 * \file tvm/s_tir/analysis.h
 * \brief Analysis utilities for Schedulable TensorIR (S-TIR).
 */
#ifndef TVM_S_TIR_ANALYSIS_H_
#define TVM_S_TIR_ANALYSIS_H_

#include <tvm/base.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/transform.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <optional>

namespace tvm {
namespace tirx {
/*!
 * \brief Auto detect the block access region according to its body stmt
 *        It will detect the access region as an array in order of appearance in AST
 * \param block The block to be detected
 * \param tensor_var_map The outside tensors which may be accessed the block.
 *                       It is a map from tensor var to the tensor.
 * \return Array of access regions.
 *         There are three arrays of TensorRegion:
 *           - first: read regions
 *           - second: write regions
 *           - third: opaque regions
 */
TVM_DLL ffi::Array<ffi::Array<TensorRegion>> GetSBlockAccessRegion(
    const s_tir::SBlock& block, const ffi::Map<Var, TensorVar>& tensor_var_map);

/*!
 * \brief Auto detect the block read/write region according to its body stmt. An opaque access will
 *        be counted as both a read and a write access
 * \param block The block to be detected
 * \param tensor_var_map The outside tensors which may be accessed the block.
 *                       It is a map from tensor var to the tensor
 * \return An array only consisting of the read regions and write regions of the input block
 */
TVM_DLL ffi::Array<ffi::Array<TensorRegion>> GetSBlockReadWriteRegion(
    const s_tir::SBlock& block, const ffi::Map<Var, TensorVar>& tensor_var_map);

/*!
 * \brief Detect the lowest common ancestor(LCA) of tensor access, including both high-level
 *        access(TensorLoad, TensorStore) and low-level access(Load, Store and opaque access).
 *        The LCA may be a For loop or a Block.
 * \param func The Function to be detected.
 * \return The Map from tensor to the LCA of all access to it. The lca is function root if the
 *         return stmt is std::nullopt.
 */
TVM_DLL ffi::Map<TensorVar, ffi::Optional<Stmt>> DetectTensorAccessLCA(const Function& func);

/*!
 * \brief Find the "anchor block" of the given module.
 * We define the anchor block to be the block with (1) an init statement and (2) having
 * the biggest flops count. The latter condition is only used when there are multiple blocks
 * with an init statement.
 * For example, if the input module is conv2d + fused spatial blocks, conv2d is the anchor block.
 * The input module may not contain more than one such block. For example, a module having
 * two conv2d is not allowed as an input.
 * However, a module created from winograd convolution has multiple blocks with an init statement
 * (input transform, batched GEMM, and output transform). We use the second condition, the flops
 * count, to determine that the batched GEMM block is the anchor block.
 * \param mod The input TIR module.
 * \return The anchor block if found, nullptr otherwise.
 */
const s_tir::SBlockObj* FindAnchorBlock(const IRModule& mod);

}  // namespace tirx

namespace sym {
class AnalyzerObj;
class Analyzer;
}  // namespace sym

namespace s_tir {
using namespace tvm::tirx;

/*! \brief Verify variable/tensor definitions, load types and schedulable block boundaries. */
TVM_DLL void CheckWellFormed(const tirx::Function& func);
/*! \brief Verify S-TIR or mixed modules, including definitions shared across functions. */
TVM_DLL void CheckWellFormed(const IRModule& mod);

/*!
 * \brief Estimate the FLOPs of a TIR fragment.
 * \param stmt The TIR fragment to be estimated.
 * \return The estimated FLOPs.
 */
TVM_DLL double EstimateTIRFlops(const Stmt& stmt);

/*!
 * \brief Estimate the FLOPs of TIRs in an IRModule.
 * \param mod The IRModule to be estimated.
 * \return The estimated FLOPs.
 */
TVM_DLL double EstimateTIRFlops(const IRModule& mod);

/*!
 * \brief Analyze the side effect of a function
 * \param func The function to be checked.
 * \param assert_on_error If true, an error will be thrown for an impure function.
 * \return The purity of the function.
 */
TVM_DLL bool IsPureFunction(const Function& func, bool assert_on_error = false);

/*!
 * \brief Verify the correctness of a GPU code
 * \param func The function to be checked.
 * \param constraints The dict to specify constraints to check.
 * \return valid Whether it is a valid GPU code.
 */
TVM_DLL bool VerifyGPUCode(const Function& func, ffi::Map<ffi::String, PrimExpr> constraints);

/*! \brief Helper struct for return value of IdentifyMemCpy */
struct MemCpyDetails {
  TensorRegion source;
  TensorRegion dest;
};

/*! \brief Identify whether a For loop is semantically equivalent to MemCpy
 * \param loop The loop to be checked
 * \param analyzer The analyzer with which to check any algebraic expressions
 * \returns The source and destination regions being copied, if the loop is equivalent to memcpy.
 */
TVM_DLL std::optional<MemCpyDetails> IdentifyMemCpy(const For& loop, const sym::Analyzer& analyzer);

/*!
 * \brief Infer the domain touched by tensor accesses within a statement.
 * \param body The statement to analyze.
 * \param tensor The tensor whose accesses are analyzed.
 * \param consider_loads Whether to include loads.
 * \param consider_stores Whether to include stores.
 * \return The domain covering the selected accesses.
 */
TVM_DLL ffi::Array<ffi::Optional<Range>> DomainTouched(const Stmt& body, const TensorVar& tensor,
                                                       bool consider_loads, bool consider_stores);

/*!
 * \brief Calculate the allocated memory per scope in bytes needed inside the TIR Function
 * \param func The TIR Function for which the allocated memory size to be calculated
 * \return Allocated memory size per scope in bytes.
 */
TVM_DLL ffi::Map<ffi::String, ffi::Map<ffi::String, int64_t>> CalculateAllocatedBytes(
    const Function& func);

/*!
 * \brief Calculate the allocated memory per scope in bytes for each function inside the module
 * \param mod The IRModule for which the allocated memory size has to be calculated
 * \return Allocated memory size per scope in bytes for each function.
 */
TVM_DLL ffi::Map<ffi::String, ffi::Map<ffi::String, int64_t>> CalculateAllocatedBytes(
    const IRModule& mod);

/**
 * \brief Get the list of lowering passes to calculate the compacted VTCM allocation size.
 * \return The list of passes.
 */
TVM_DLL ffi::Array<tvm::transform::Pass> GetVTCMCompactionPasses();

/*!
 * \brief Verifies that the VTCM usage for all functions in the given IRModule.
 * \param mod The module to be checked.
 * \param limit The limit to check.
 * \return true if the VTCM usage is within the provided limit.
 */
TVM_DLL bool VerifyVTCMLimit(const IRModule& mod, int64_t limit);

/*!
 * \brief Verifies that the VTCM usage of the given function is within the provided limit.
 * \param func The function to be checked.
 * \param limit The limit to check.
 * \return true if the VTCM usage is within the provided limit.
 */
TVM_DLL bool VerifyVTCMLimit(const Function& func, int64_t limit);

namespace transform {

using tvm::transform::Pass;
using tvm::transform::PassContext;

/*!
 * \brief Pass to verify GPU code constraints.
 * \param constraints The dict to specify constraints.
 * \return The pass.
 */
TVM_DLL Pass VerifyGPUCode(ffi::Map<ffi::String, PrimExpr> constraints);

/*!
 * \brief Pass to check if VTCM usage is within limit.
 * \param default_target The default target for functions without target attribute.
 * \return The pass.
 */
TVM_DLL Pass VerifyVTCMLimit(ffi::Optional<Target> default_target = std::nullopt);

/*!
 * \brief Statically check TIR code for out of bounds array access.
 * \return The pass.
 */
TVM_DLL Pass OOBChecker();

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_ANALYSIS_H_
