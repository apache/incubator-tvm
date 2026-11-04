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

#ifndef TVM_RELAX_DISTRIBUTED_AXIS_GROUP_GRAPH_H_
#define TVM_RELAX_DISTRIBUTED_AXIS_GROUP_GRAPH_H_

#include <tvm/relax/distributed/type.h>
#include <tvm/relax/expr.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/function.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {
// (var, axis)
using TIRVarAxis = std::pair<Var, int>;
// (tensor, axis)
using TensorAxis = std::pair<TensorVar, int>;
class TensorAxisHash {
 public:
  size_t operator()(const TensorAxis& tensor_axis) const {
    size_t const h1(ffi::ObjectPtrHash()(tensor_axis.first));
    size_t const h2(std::hash<int>()(tensor_axis.second));
    return h1 ^ (h2 << 1);
  }
};
/*!
 * \brief Suppose we want to shard a tensor along a specific dimension, we need to know how
 * to rewrite the access index of the tensor. To make it simple, we only support the case that
 * the access can be rewritten by changing the extent of an iter var.
 * \param index The access index
 * \param var_range The range of each iter var
 * \param analyzer The analyzer
 * \return The iter var whose extent to be changed
 */
ffi::Optional<Var> GetShardingVarFromIndex(PrimExpr index, ffi::Map<Var, Range> var_range,
                                           const sym::Analyzer& analyzer);

/*!
 * \brief Construct an axis group graph from a tirx::Function. Two tensor axis are connected if they
 * are accessed by the same index.
 */
class TensorAxisGraphExtractor : public s_tir::StmtExprVisitor {
 public:
  static std::vector<std::vector<TIRVarAxis>> GetTIRVarAxisGraph(const tirx::Function& function) {
    auto extractor = ffi::make_object<TensorAxisGraphExtractor>();
    extractor->Visit(function->body);
    ffi::Map<TensorVar, Var> inverse_tensor_map;
    for (const Var& param : function->params) {
      if (param->ty.as<TensorTypeObj>()) {
        inverse_tensor_map.Set(param.as_or_throw<TensorVar>(), param);
      }
    }
    std::vector<std::vector<TIRVarAxis>> tir_var_axis_group_list;
    std::unordered_set<TensorAxis, TensorAxisHash> visited;
    for (const Var& param : function->params) {
      if (!param->ty.as<TensorTypeObj>()) {
        continue;
      }
      TensorVar tensor = param.as_or_throw<TensorVar>();
      for (int i = 0; i < static_cast<int>(tensor.shape().size()); i++) {
        if (extractor->tensor_axis_graph_.count({tensor, i})) {
          std::vector<TensorAxis> tensor_axis_group;
          extractor->DFSGraph({tensor, i}, &visited, &tensor_axis_group);
          if (tensor_axis_group.size() <= 1) {
            continue;
          }
          std::vector<TIRVarAxis> tir_var_axis_group;
          for (const auto& tensor_axis : tensor_axis_group) {
            if (!inverse_tensor_map.count(tensor_axis.first)) {
              continue;
            }
            tir_var_axis_group.push_back(
                {inverse_tensor_map[tensor_axis.first], tensor_axis.second});
          }
          tir_var_axis_group_list.push_back(tir_var_axis_group);
        }
      }
    }
    return tir_var_axis_group_list;
  }

  void DFSGraph(TensorAxis cur, std::unordered_set<TensorAxis, TensorAxisHash>* visited,
                std::vector<TensorAxis>* tensor_axis_group) {
    if (visited->count(cur)) {
      return;
    }
    visited->insert(cur);
    tensor_axis_group->push_back(cur);
    for (const auto& next : tensor_axis_graph_[cur]) {
      DFSGraph(next, visited, tensor_axis_group);
    }
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit_(op));
    tensor_access_indices_.push_back({op->dest.as_or_throw<tvm::tirx::TensorVar>(), op->indices});

    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit_(op));
    tensor_access_indices_.push_back({op->source.as_or_throw<tvm::tirx::TensorVar>(), op->indices});

    return std::nullopt;
  }

  bool Match(PrimExpr a, PrimExpr tensor_shape_a, PrimExpr b, PrimExpr tensor_shape_b,
             const sym::Analyzer& analyzer) {
    if (b.as<PrimVar>()) {
      std::swap(a, b);
      std::swap(tensor_shape_a, tensor_shape_b);
    }
    auto prim_var = a.as<PrimVar>();
    if (!prim_var) {
      return false;
    }
    Var var = *prim_var;
    b = analyzer->Simplify(b);
    // index var `a` must access whole range of a specific tensor dimension
    sym::IntSet intset_b = sym::EvalSet(b, sym::AsIntSet(iter_var_range_));
    if (!analyzer->CanProveEqual(tensor_shape_a, iter_var_range_[var]->extent) ||
        !intset_b.MatchRange(Range::FromMinExtent(0, tensor_shape_b))) {
      return false;
    }
    auto matched_var = GetShardingVarFromIndex(b, iter_var_range_, analyzer);
    if (!matched_var.has_value() || !matched_var.value().same_as(var)) {
      return false;
    }
    return true;
  }

  ffi::Optional<VisitInterrupt> Visit_(const s_tir::SBlockObj* op) final {
    if (op->name_hint == "root") {
      return s_tir::StmtExprVisitor::Visit_(op);
    }
    tensor_access_indices_.clear();
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit_(op));
    iter_var_range_.clear();
    for (const auto& iter_var : op->iter_vars) {
      iter_var_range_.Set(iter_var->var, iter_var->dom.value());
    }
    sym::Analyzer analyzer;
    sym::ConstraintContext context;
    for (const auto& [var, range] : iter_var_range_) {
      context->Append(sym::RangeConstraint(var, range));
    }
    analyzer->SyncConstraintContext(context);
    for (const auto& access_pr : tensor_access_indices_) {
      TensorVar tensor = access_pr.first;
      ffi::Array<PrimExpr> indices = access_pr.second;
      for (int i = 0; i < static_cast<int>(indices.size()); i++) {
        for (const auto& another_access_pr : tensor_access_indices_) {
          if (another_access_pr.first.same_as(tensor)) {
            continue;
          }
          TensorVar another_tensor = another_access_pr.first;
          ffi::Array<PrimExpr> another_indices = another_access_pr.second;
          for (int j = 0; j < static_cast<int>(another_indices.size()); j++) {
            if (Match(indices[i], tensor.shape()[i], another_indices[j], another_tensor.shape()[j],
                      analyzer)) {
              JoinTensorAxis({tensor, i}, {another_tensor, j});
            }
          }
        }
      }
    }

    return std::nullopt;
  }

  void JoinTensorAxis(TensorAxis axis1, TensorAxis axis2) {
    if (!tensor_axis_graph_.count(axis1)) {
      tensor_axis_graph_[axis1] = {};
    }
    if (!tensor_axis_graph_.count(axis2)) {
      tensor_axis_graph_[axis2] = {};
    }
    tensor_axis_graph_[axis1].push_back(axis2);
    tensor_axis_graph_[axis2].push_back(axis1);
  }

  std::vector<std::pair<TensorVar, ffi::Array<PrimExpr>>> tensor_access_indices_;
  std::unordered_map<TensorAxis, std::vector<TensorAxis>, TensorAxisHash> tensor_axis_graph_;
  ffi::Map<Var, Range> iter_var_range_;
  std::string func_name;
};
}  // namespace tirx
}  // namespace tvm

namespace tvm {
namespace relax {
namespace distributed {

/*! \brief tensor axis*/
struct Axis {
  const ExprObj* tensor;
  int dim = 0;
  int tuple_index = 0;

  Axis(const ExprObj* tensor, int dim, int tuple_index = 0)
      : tensor(tensor), dim(dim), tuple_index(tuple_index) {
    TVM_FFI_ICHECK(tensor->IsInstance<GenericConstObj>() || tensor->IsInstance<VarObj>());
  }

  bool operator==(const Axis& other) const {
    return tensor == other.tensor && dim == other.dim && tuple_index == other.tuple_index;
  }
};

class AxisHash {
 public:
  size_t operator()(const Axis& axis) const {
    size_t const h1(std::hash<const ExprObj*>()(axis.tensor));
    size_t const h2(std::hash<int>()(axis.dim));
    size_t const h3(std::hash<int>()(axis.tuple_index));
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

using AxisGroup = std::unordered_set<Axis, AxisHash>;

class AxisGroupHash {
 public:
  size_t operator()(const AxisGroup& axis_group) const {
    size_t seed = 0;
    for (auto axis : axis_group) {
      seed ^= AxisHash()(axis) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

using ShardingSpec = std::pair<DeviceMesh, Placement>;

// device mesh and the device mesh axis that the tensor axis maps to
using AxisShardingSpec = std::pair<DeviceMesh, int>;
class AxisShardingSpecEqual {
 public:
  bool operator()(const AxisShardingSpec& lhs, const AxisShardingSpec& rhs) const {
    return ffi::StructuralEqual()(lhs.first, rhs.first) && lhs.second == rhs.second;
  }
};

class AxisShardingSpecHash {
 public:
  size_t operator()(const AxisShardingSpec& sharding_spec) const {
    size_t seed = 0;
    seed ^= ffi::StructuralHash()(sharding_spec.first);
    seed ^= std::hash<int>()(sharding_spec.second) << 1;
    return seed;
  }
};

/*!
 * \brief A graph whose nodes are tensor axes, and the edge means some information can be propagated
 * through the two axes. Although it only does sharding propagation, this data structure can be
 * extended to perform all kinds of propagation that happens on tensor axes.
 */
class AxisGroupGraph {
 public:
  enum class EdgeType { kAscend, kDescend, kSimbling };

 private:
  static EdgeType ReverseEdgeType(EdgeType type) {
    switch (type) {
      case EdgeType::kAscend:
        return EdgeType::kDescend;
      case EdgeType::kDescend:
        return EdgeType::kAscend;
      case EdgeType::kSimbling:
        return EdgeType::kSimbling;
    }
    TVM_FFI_THROW(InternalError) << "Unreachable code";
    throw;
  }

  static int GetEdgePriority(EdgeType type) {
    switch (type) {
      case EdgeType::kAscend:
        return 0;
      case EdgeType::kDescend:
        return 2;
      case EdgeType::kSimbling:
        return 1;
    }
    TVM_FFI_THROW(InternalError) << "Unreachable code";
    throw;
  }

  struct AxisGraphEdge {
    Axis src;
    Axis dst;

    // the producer-consumer relationship between src tensor and dst tensor
    // kAscend means consumer->producer
    // kDescend means producer->consumer
    // kSimbling means other cases
    EdgeType type;

    bool operator==(const AxisGraphEdge& other) const {
      return src == other.src && dst == other.dst && type == other.type;
    }
  };

  struct Path {
    int direction = 0;

    Path AddEdge(EdgeType type) { return {direction |= (1 << GetEdgePriority(type))}; }

    int GetPriority() const {
      switch (direction) {
        case 1:  // ascend only
          return 0;
        case 4:  // descend only
          return 2;
        case 0:      // empty path (source node)
          return 3;  // source node must have max priority
        default:     // mixed path
          return 1;
      }
    }
  };

 public:
  AxisGroupGraph() = default;

  /*!
   * \brief add edge between two axes
   * \param axis1 The src axis
   * \param axis2 The dst axis
   * \param type  The producer-consumer relationship between src tensor and dst tensor
   *              kAscend means consumer->producer
   *              kDescend means producer->consumer
   *              kSimbling means other cases
   */
  void JoinAxis(Axis axis1, Axis axis2, EdgeType type) {
    AddEdge(axis1, axis2, type);
    AddEdge(axis2, axis1, ReverseEdgeType(type));
  }

  /*!
   * \brief add a source shardingspec to propagate
   * \param axis The source axis
   * \param spec The axis's sharding spec
   */
  void AddSrcShardingPoint(Axis axis, AxisShardingSpec spec) {
    src_axis_sharding_spec_.insert_or_assign(axis, spec);
  }

  /*!
   * \brief propagate sharding specs from source axes
   */
  void PropagateShardingSpec() {
    axis_sharding_specs_priority_.clear();
    for (const auto& pr : src_axis_sharding_spec_) {
      std::unordered_set<Axis, AxisHash> visited;
      PropagateShardingSpec(pr.first, pr.second, Path(), &visited);
    }
    ChooseAxisShardingSpec();
  }

  /*!
   * \brief add a cut point that stops the propagation of a certain sharding spec
   *
   * \param axis The cut point
   * \param spec The spec to stop propagation
   */
  void AddPropagationCutPoint(Axis axis, AxisShardingSpec spec) {
    cutpoint_axis_sharding_spec_.insert_or_assign(axis, spec);
  }

  /*!
   * \brief Get the Sharding Spec of an axis after propagation
   *
   * \param axis the specified axis
   * \return if a sharding spec is found, return (axis_sharding_spec, true)
   *         otherwise, return (std::nullopt, false)
   */
  std::tuple<std::optional<AxisShardingSpec>, bool> GetAxisShardingSpec(Axis axis) {
    if (axis_sharding_specs_priority_.count(axis)) {
      return {axis_sharding_specs_priority_[axis].begin()->first, true};
    } else {
      return {std::nullopt, false};
    }
  }

 private:
  void AddEdge(Axis src, Axis dst, EdgeType type) {
    if (!graph_.count(src)) {
      graph_[src] = {};
    }
    graph_[src].push_back({src, dst, type});
  }

  void PropagateShardingSpec(Axis axis, AxisShardingSpec spec, Path path,
                             std::unordered_set<Axis, AxisHash>* visited) {
    if (cutpoint_axis_sharding_spec_.count(axis) ||
        (src_axis_sharding_spec_.count(axis) &&
         !AxisShardingSpecEqual()(src_axis_sharding_spec_.at(axis), spec)) ||
        visited->count(axis)) {
      return;
    }
    visited->insert(axis);
    if (!axis_sharding_specs_priority_.count(axis)) {
      axis_sharding_specs_priority_[axis] = {};
    }
    axis_sharding_specs_priority_[axis][spec] = path.GetPriority();
    for (auto edge : graph_[axis]) {
      PropagateShardingSpec(edge.dst, spec, path.AddEdge(edge.type), visited);
    }
  }

  void ChooseAxisShardingSpec() {
    for (auto& pr : axis_sharding_specs_priority_) {
      auto& axis = pr.first;
      auto& specs = pr.second;
      int max_priority = std::numeric_limits<int>::min();
      for (auto& pr2 : specs) {
        max_priority = std::max(max_priority, pr2.second);
      }
      for (auto it = specs.begin(); it != specs.end();) {
        if (it->second != max_priority) {
          it = specs.erase(it);
        } else {
          it++;
        }
      }
      TVM_FFI_ICHECK(specs.size() == 1)
          << "multiple possible sharding for axis: (" << ffi::GetRef<Expr>(axis.tensor) << ", "
          << axis.dim << ")";
    }
  }

  // union set
  std::unordered_map<Axis, std::vector<AxisGraphEdge>, AxisHash> graph_;
  std::unordered_map<Axis, AxisShardingSpec, AxisHash> src_axis_sharding_spec_;
  std::unordered_map<Axis, AxisShardingSpec, AxisHash> cutpoint_axis_sharding_spec_;
  std::unordered_map<
      Axis, std::unordered_map<AxisShardingSpec, int, AxisShardingSpecHash, AxisShardingSpecEqual>,
      AxisHash>
      axis_sharding_specs_priority_;
};

using FBuildAxisGraph = std::function<void(const Var& output_var, const Call& call,
                                           distributed::AxisGroupGraph* axis_group_graph)>;

void BuildAxisGraphUnary(const Var& output_var, const Call& call,
                         distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphBinary(const Var& output_var, const Call& call,
                          distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphReduce(const Var& output_var, const Call& call,
                          distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphMatmul(const Var& output_var, const Call& call,
                          distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphPermuteDims(const Var& output_var, const Call& call,
                               distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphReshape(const Var& output_var, const Call& call,
                           distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphCallTIR(const Var& output_var, const Call& call, const tirx::Function& func,
                           distributed::AxisGroupGraph* axis_group_graph);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DISTRIBUTED_AXIS_GROUP_GRAPH_H_
