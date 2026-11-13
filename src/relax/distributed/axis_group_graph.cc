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
#include <tvm/relax/analysis.h>
#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/op/distributed.h>
#include <tvm/relax/op/linear_algebra.h>
#include <tvm/relax/op/manipulate.h>
#include <tvm/relax/op/nn.h>
#include <tvm/relax/op/statistical.h>

#include <numeric>

namespace tvm {
namespace tirx {
using namespace tvm::prim;

ffi::Optional<Var> GetShardingVarFromIndex(PrimExpr index, ffi::Map<Var, Range> var_range,
                                           const sym::Analyzer& analyzer) {
  if (auto prim_var = index.as<PrimVar>()) {
    return Var(prim_var.value());
  }
  ffi::Map<PrimVar, Range> primitive_var_range;
  for (const auto& [var, range] : var_range) {
    primitive_var_range.Set(var.as_or_throw<PrimVar>(), range);
  }
  sym::IterSumExpr iter_sum = sym::NormalizeToIterSum(index, primitive_var_range, analyzer);
  if (!IsZero(iter_sum->base)) {
    return std::nullopt;
  }
  if (iter_sum->args.empty()) {
    return std::nullopt;
  }
  // floormod(floordiv(source, lower_factor), extent) * scale
  sym::IterSplitExpr highest_iter_split = iter_sum->args[0];
  auto source_var = highest_iter_split->source->source.as<PrimVar>();
  if (!source_var) {
    return std::nullopt;
  }
  Var var = source_var.value();
  // the floormod must take no effect
  if (!analyzer->CanProve(floordiv(var_range[var]->extent, highest_iter_split->lower_factor) <=
                          highest_iter_split->extent)) {
    return std::nullopt;
  }
  return var;
}
}  // namespace tirx
}  // namespace tvm

namespace tvm {

namespace relax {
namespace distributed {

const TensorTypeObj* GetTensorType(Expr tensor) {
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(tensor);
  if (tensor_ty) {
    return tensor_ty;
  }
  const auto* dtensor_ty = GetTypeAs<DTensorTypeObj>(tensor);
  if (dtensor_ty) {
    return dtensor_ty->tensor_ty.get();
  }
  TVM_FFI_THROW(InternalError) << tensor << " must be either Tensor or DTesor";
  throw;
}

void UnaryOpHelper(ffi::Array<Expr> tensor_list, distributed::AxisGroupGraph* axis_group_graph) {
  int n_dim = GetTensorType(tensor_list[0])->ndim;
  for (const auto& tensor : tensor_list) {
    TVM_FFI_ICHECK(GetTensorType(tensor)->ndim == n_dim);
  }
  for (int i = 0; i < n_dim; i++) {
    TVM_FFI_ICHECK(tensor_list.size() <= 2);
    for (int j = 0; j < static_cast<int>(tensor_list.size()) - 1; j++) {
      axis_group_graph->JoinAxis({tensor_list[j].get(), i}, {tensor_list[j + 1].get(), i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    }
  }
}

void BuildAxisGraphUnary(const Var& output_var, const Call& call,
                         distributed::AxisGroupGraph* axis_group_graph) {
  ffi::Array<Expr> tensor_list;  // vars in param and output
  if (call->args[0]->IsInstance<VarObj>()) {
    tensor_list.push_back(call->args[0]);
  }
  tensor_list.push_back(output_var);
  UnaryOpHelper(tensor_list, axis_group_graph);
}

void BuildAxisGraphBinary(const Var& output_var, const Call& call,
                          distributed::AxisGroupGraph* axis_group_graph) {
  ffi::Array<Expr> tensor_list;  // vars in param and output
  if (call->args[0]->ty.as<TensorTypeObj>() || call->args[0]->ty.as<DTensorTypeObj>()) {
    tensor_list.push_back(call->args[0]);
  }
  if (call->args[1]->ty.as<TensorTypeObj>() || call->args[1]->ty.as<DTensorTypeObj>()) {
    tensor_list.push_back(call->args[1]);
  }
  tensor_list.push_back(output_var);
  if (tensor_list.size() <= 2) {
    UnaryOpHelper(tensor_list, axis_group_graph);
    return;
  }
  const auto* x1_ty = GetTensorType(tensor_list[0]);
  const auto* x2_ty = GetTensorType(tensor_list[1]);
  int x1_ndim = x1_ty->ndim;
  int x2_ndim = x2_ty->ndim;
  const auto* x1_shape = x1_ty->shape.as<ShapeExprObj>();
  const auto* x2_shape = x2_ty->shape.as<ShapeExprObj>();
  TVM_FFI_ICHECK(x1_shape && x2_shape);
  sym::Analyzer analyzer;
  for (int i = 1; i <= std::min(x1_ndim, x2_ndim); ++i) {
    const PrimExpr& dim0 = x1_shape->values[x1_ndim - i];
    const PrimExpr& dim1 = x2_shape->values[x2_ndim - i];
    if (analyzer->CanProveEqual(dim0, dim1)) {
      // join batch dim
      axis_group_graph->JoinAxis({tensor_list[0].get(), x1_ndim - i},
                                 {tensor_list[2].get(), std::max(x1_ndim, x2_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
      axis_group_graph->JoinAxis({tensor_list[1].get(), x2_ndim - i},
                                 {tensor_list[2].get(), std::max(x1_ndim, x2_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    } else if (analyzer->CanProveEqual(dim0, 1)) {
      axis_group_graph->JoinAxis({tensor_list[1].get(), x2_ndim - i},
                                 {tensor_list[2].get(), std::max(x1_ndim, x2_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    } else if (analyzer->CanProveEqual(dim1, 1)) {
      axis_group_graph->JoinAxis({tensor_list[0].get(), x1_ndim - i},
                                 {tensor_list[2].get(), std::max(x1_ndim, x2_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid broadcast, dim0: " << dim0 << ", dim1: " << dim1;
    }
  }
  if (x1_ndim > x2_ndim) {
    for (int i = 0; i < x1_ndim - x2_ndim; i++) {
      axis_group_graph->JoinAxis({tensor_list[0].get(), i}, {tensor_list[2].get(), i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    }
  } else if (x1_ndim < x2_ndim) {
    for (int i = 0; i < x2_ndim - x1_ndim; i++) {
      axis_group_graph->JoinAxis({tensor_list[1].get(), i}, {tensor_list[2].get(), i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    }
  }
}

void BuildAxisGraphReduce(const Var& output_var, const Call& call,
                          distributed::AxisGroupGraph* axis_group_graph) {
  Expr input_tensor = call->args[0];
  ffi::Array<int64_t> axes;
  bool keepdims;
  if (const auto* attrs = call->attrs.as<StatisticalAttrs>()) {
    if (attrs->axis.has_value()) {
      axes = attrs->axis.value();
    }
    keepdims = attrs->keepdims;
  } else if (const auto* attrs = call->attrs.as<SoftmaxAttrs>()) {
    axes = {attrs->axis};
    keepdims = true;
  } else {
    TVM_FFI_THROW(InternalError) << "Unsupported reduce op: " << call->op;
  }

  int ndim = GetTensorType(input_tensor)->ndim;

  std::unordered_set<int> normalized_axes;
  for (int64_t i : axes) {
    int val = static_cast<int>(i);
    TVM_FFI_ICHECK(val < ndim && val >= -ndim);
    if (val < 0) {
      val = ndim + val;
    }
    normalized_axes.insert(val);
  }
  if (keepdims) {
    for (int i = 0; i < ndim; i++) {
      if (!normalized_axes.count(i)) {
        axis_group_graph->JoinAxis({input_tensor.get(), i}, {output_var.get(), i},
                                   distributed::AxisGroupGraph::EdgeType::kDescend);
      }
    }
  } else {
    for (int i = 0, j = 0; i < ndim; i++) {
      if (!normalized_axes.count(i)) {
        axis_group_graph->JoinAxis({input_tensor.get(), i}, {output_var.get(), j},
                                   distributed::AxisGroupGraph::EdgeType::kDescend);
        j++;
      }
    }
  }
}

void BuildAxisGraphMatmul(const Var& output_var, const Call& call,
                          distributed::AxisGroupGraph* axis_group_graph) {
  Expr x1 = call->args[0];
  Expr x2 = call->args[1];
  Var x3 = output_var;
  const auto* x1_ty = GetTensorType(x1);
  const auto* x2_ty = GetTensorType(x2);
  int x1_ndim = x1_ty->ndim;
  int x2_ndim = x2_ty->ndim;
  TVM_FFI_ICHECK(x1_ndim > 0 && x2_ndim > 0);
  int x1_prepended = 0;
  int x2_appended = 0;
  if (x1_ndim == 1) {
    x1_ndim = 2;
    x1_prepended = 1;
  }
  if (x2_ndim == 1) {
    x2_ndim = 2;
    x2_appended = 1;
  }
  const auto* x1_shape = x1_ty->shape.as<ShapeExprObj>();
  const auto* x2_shape = x2_ty->shape.as<ShapeExprObj>();
  TVM_FFI_ICHECK(x1_shape && x2_shape);
  ffi::Array<PrimExpr> x1_shape_prefix{x1_shape->values.begin(),
                                       x1_shape->values.end() - 2 + x1_prepended};
  ffi::Array<PrimExpr> x2_shape_prefix{x2_shape->values.begin(),
                                       x2_shape->values.end() - 2 + x2_appended};

  int x1_prefix_ndim = x1_shape_prefix.size();
  int x2_prefix_ndim = x2_shape_prefix.size();
  sym::Analyzer analyzer;
  for (int i = 1; i <= std::min(x1_prefix_ndim, x2_prefix_ndim); ++i) {
    const PrimExpr& dim0 = x1_shape_prefix[x1_prefix_ndim - i];
    const PrimExpr& dim1 = x2_shape_prefix[x2_prefix_ndim - i];
    // join batch dim
    if (analyzer->CanProveEqual(dim0, dim1)) {
      axis_group_graph->JoinAxis({x1.get(), x1_prefix_ndim - i},
                                 {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
      axis_group_graph->JoinAxis({x2.get(), x2_prefix_ndim - i},
                                 {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    } else if (analyzer->CanProveEqual(dim0, 1)) {
      axis_group_graph->JoinAxis({x2.get(), x2_prefix_ndim - i},
                                 {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    } else if (analyzer->CanProveEqual(dim1, 1)) {
      axis_group_graph->JoinAxis({x1.get(), x1_prefix_ndim - i},
                                 {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    } else {
      TVM_FFI_THROW(InternalError) << "Cannot broadcast " << dim0 << " and " << dim1;
    }
  }
  // join reduction dim
  axis_group_graph->JoinAxis({x1.get(), x1_ty->ndim - 1}, {x2.get(), x2_ndim - 2},
                             distributed::AxisGroupGraph::EdgeType::kSimbling);
  // join lhs_spatial dim and rhs_spatial dim
  if (!x1_prepended) {
    axis_group_graph->JoinAxis({x1.get(), x1_ndim - 2},
                               {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim)},
                               distributed::AxisGroupGraph::EdgeType::kDescend);
    if (!x2_appended) {
      axis_group_graph->JoinAxis({x2.get(), x2_ndim - 1},
                                 {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) + 1},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    }
  } else if (!x2_appended) {
    axis_group_graph->JoinAxis({x2.get(), x2_ndim - 1},
                               {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim)},
                               distributed::AxisGroupGraph::EdgeType::kDescend);
  }
}

void BuildAxisGraphPermuteDims(const Var& output_var, const Call& call,
                               distributed::AxisGroupGraph* axis_group_graph) {
  Expr input_tensor = call->args[0];
  const auto* attrs = call->attrs.as<PermuteDimsAttrs>();
  TVM_FFI_ICHECK(attrs);
  int ndim = GetTensorType(input_tensor)->ndim;
  std::vector<int> normalized_axes;
  if (attrs->axes.has_value()) {
    for (int64_t i : attrs->axes.value()) {
      int val = static_cast<int>(i);
      TVM_FFI_ICHECK(val < ndim && val >= -ndim);
      if (val < 0) {
        val = ndim + val;
      }
      normalized_axes.push_back(val);
    }
  } else {
    normalized_axes.resize(ndim);
    std::iota(normalized_axes.rbegin(), normalized_axes.rend(), 0);
  }
  for (int i = 0; i < ndim; i++) {
    axis_group_graph->JoinAxis({input_tensor.get(), normalized_axes[i]}, {output_var.get(), i},
                               distributed::AxisGroupGraph::EdgeType::kDescend);
  }
}
void BuildAxisGraphReshape(const Var& output_var, const Call& call,
                           distributed::AxisGroupGraph* axis_group_graph) {
  Expr input_tensor = call->args[0];
  const auto* tensor_ty = GetTensorType(input_tensor);
  const auto* new_shape_ty = GetTypeAs<ShapeTypeObj>(call->args[1]);
  const auto* old_shape_ty = GetTypeAs<ShapeTypeObj>(tensor_ty->shape.value());
  TVM_FFI_ICHECK_NOTNULL(old_shape_ty);
  ffi::Array<PrimExpr> old_shape_values = old_shape_ty->values.value();
  ffi::Array<PrimExpr> new_shape_values = new_shape_ty->values.value();
  int i = old_shape_values.size();
  int j = new_shape_values.size();
  PrimExpr old_shape_product = 1, new_shape_product = 1;
  sym::Analyzer analyzer_;
  while (i > 0 && j > 0) {
    if (analyzer_->CanProve(new_shape_product > old_shape_product)) {
      i--;
      old_shape_product *= old_shape_values[i];
    } else if (analyzer_->CanProve(new_shape_product < old_shape_product)) {
      j--;
      new_shape_product *= new_shape_values[j];
    } else {
      if (i != static_cast<int>(old_shape_values.size())) {
        axis_group_graph->JoinAxis({input_tensor.get(), i}, {output_var.get(), j},
                                   distributed::AxisGroupGraph::EdgeType::kDescend);
      }
      i--;
      j--;
      old_shape_product *= old_shape_values[i];
      new_shape_product *= new_shape_values[j];
    }
  }
}

inline int GetNumOutput(Call call) {
  Type output_ty = call->ty_args[0];
  if (const auto* tuple_ty = output_ty.as<TupleTypeObj>()) {
    return tuple_ty->fields.size();
  } else {
    return 1;
  }
}

void BuildAxisGraphCallTIR(const Var& output_var, const Call& call, const tirx::Function& func,
                           distributed::AxisGroupGraph* axis_group_graph) {
  auto tir_var_axis_group_list = tirx::TensorAxisGraphExtractor::GetTIRVarAxisGraph(func);
  ffi::Map<tvm::Var, Expr> input_var_to_relax_expr;
  ffi::Array<Expr> input_list = call->args[1].as_or_throw<Tuple>()->fields;
  input_list.push_back(output_var);
  for (int i = 0; i < static_cast<int>(input_list.size()); i++) {
    if (func->params[i]->ty.as<tirx::TensorTypeObj>()) {
      input_var_to_relax_expr.Set(func->params[i], input_list[i]);
    }
  }
  int num_params = func->params.size();
  int num_outputs = GetNumOutput(call);
  for (const auto& var_axis_group : tir_var_axis_group_list) {
    std::unordered_map<int, int> output_tensor_indices;
    for (int i = 0; i < static_cast<int>(var_axis_group.size()); i++) {
      for (int j = num_params - num_outputs; j < num_params; j++) {
        if (func->params[j].same_as(var_axis_group[i].first)) {
          output_tensor_indices[i] = j - num_params + num_outputs;
          break;
        }
      }
    }
    if (output_tensor_indices.empty()) {
      for (int i = 1; i < static_cast<int>(var_axis_group.size()); i++) {
        axis_group_graph->JoinAxis(
            {input_var_to_relax_expr[var_axis_group[i].first].get(), var_axis_group[i].second},
            {input_var_to_relax_expr[var_axis_group[0].first].get(), var_axis_group[0].second},
            distributed::AxisGroupGraph::EdgeType::kSimbling);
      }
    } else {
      for (const auto& pr : output_tensor_indices) {
        for (int i = 0; i < static_cast<int>(var_axis_group.size()); i++) {
          if (!output_tensor_indices.count(i)) {
            axis_group_graph->JoinAxis(
                {input_var_to_relax_expr[var_axis_group[i].first].get(), var_axis_group[i].second},
                {output_var.get(), var_axis_group[pr.first].second, pr.second},
                distributed::AxisGroupGraph::EdgeType::kDescend);
          }
        }
      }
    }
  }
}
}  // namespace distributed
}  // namespace relax
}  // namespace tvm
