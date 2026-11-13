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
 * \file src/relax/transform/convert_layout.cc
 * \brief Automatic layout conversion pass, especially for axis swapping.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/transform.h>
#include <tvm/tirx/index_map.h>

#include "../op/tensor/manipulate.h"
#include "infer_layout_utils.h"
#include "utils.h"

namespace tvm {
namespace relax {

using tirx::IndexMap;
using tirx::SLayout;
using LayoutCb = tvm::relax::transform::LayoutCb;

/*!
 * \brief Main logic to convert the layout of conv2d. Other ops
 * can adapt to such layout conversion following conv2d accordingly.
 *
 * Structurally speaking, a Relax function is composed of a series of VarBinding and
 * MatchCast. And a specific class of VarBindings is the basic unit we want to rewrite.
 * Formally, they are of the form:
 *
 * var = Call(Op, [args], attrs)
 *
 * where Op is a specific op we want to rewrite, and attrs is the attributes of the op.
 * var and args are all exprs with type Tensor or Tuple of Tensors. They might
 * be vars, constants, or Tuple of vars and constants.
 *
 * We register the layout inference function for each op (FRelaxInferLayout), which accepts the
 * current call, the desired layout of conv2d ops, and the layout map of previous vars. The result
 * of the layout inference function is contained in an InferLayoutOutput object, which contains 3
 * fields: input_layouts, output_layouts, and attr, which represents the expected input layout,
 * output_layout and converted attrs of the new op call.
 *
 * The rewrite pass does the rewriting in a single forward pass, where for each Call(Op),
 * we collect the current SLayout of each input var, and let the InferLayout function to infer the
 * desired layout of the output. The rewriter will use these info to convert
 * the layout of inputs and attrs of the op call, and note down the new layout of the output.
 *
 * The desired layout of conv2d ops is a map from the name of the op to the desired layout of the
 * desired feature map, weight and output. For example, if we want to convert the layout of conv2d
 * from NCHW to NHWC, we can set the desired layout of conv2d to be {"conv2d": ["NHWC", "OHWI"]}.
 *
 * The way we represent the layout of a var is a NLayout object, which is a nested tuple of SLayout.
 * The incoming layout of the module will be set as the default layout (We use ABCD... as the
 * default) Note that for operators like conv, pool, people typically use NHWC to refer to the axes.
 * But to be generic and support more operators, we use ABCD... to refer to the axes.
 *
 * Note that currently the layout conversion of conv2d only support axis swapping, such as NCHW to
 * NWHC. Packed layout such as NCHW to NCHW4c is not supported now.
 */
class LayoutConvertMutator : public ExprMutator {
 public:
  explicit LayoutConvertMutator(
      const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts, LayoutCb layout_cb)
      : desired_layouts_(desired_layouts), layout_cb_(layout_cb) {}

 private:
  ffi::Array<int64_t> LayoutToIntegers(const SLayout& layout) {
    ffi::Array<int64_t> ret;
    LayoutDecision src = InitialLayoutDecision(layout.ndim());
    for (size_t i = 0; i < layout.ndim(); ++i) {
      ret.push_back(static_cast<int64_t>(src->layout.value().IndexOf(layout[i])));
    }
    return ret;
  }

  IndexMap LayoutIndexMap(int ndim, const SLayout& src_layout, const SLayout& desired_layout) {
    tirx::SBijectiveLayout todesired(src_layout, desired_layout);
    ffi::Optional<IndexMap> inverse_index_map;

    ffi::Array<tvm::Var> initial_indices;
    ffi::Array<PrimExpr> initial_indices_expr;
    initial_indices.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      auto var = tvm::Var("i" + std::to_string(i), PrimType::Int(32));
      initial_indices.push_back(var);
      initial_indices_expr.push_back(var.as_or_throw<PrimExpr>());
    }
    ffi::Array<PrimExpr> desired_shape = todesired.ForwardIndex(initial_indices_expr);
    return IndexMap(
        initial_indices.Map([](tvm::Var var) { return var.as_or_throw<tvm::PrimVar>(); }),
        desired_shape, std::move(inverse_index_map));
  }

  Expr RewriteExpr(const Expr& expr, const NLayout& to) {
    auto fvisitleaf = [&](const Expr& expr, std::array<NLayout, 2> layouts) -> Expr {
      NLayout from = layouts[0], to = layouts[1];
      if (NLayoutEqual()(from, to) || (layouts[0].LeafValue()->layout.has_value()
                                           ? layouts[0].LeafValue()->layout.value().name()
                                           : "__undef__") == "")
        return expr;
      // If not both from and to are unknown, then none of them can be unknown.
      TVM_FFI_ICHECK(!NLayoutEqual()(from, LayoutDecision::InitUnknownDim()) &&
                     !NLayoutEqual()(to, LayoutDecision::InitUnknownDim()))
          << "Cannot convert when exactly one of the layouts is unknown";
      const auto* tensor = GetTypeAs<TensorTypeObj>(expr);
      TVM_FFI_ICHECK(tensor != nullptr) << "Expect a tensor, but got: " << expr;

      if ((from.LeafValue()->layout.has_value() ? from.LeafValue()->layout.value().ndim() : 0) ==
          (to.LeafValue()->layout.has_value() ? to.LeafValue()->layout.value().ndim() : 0)) {
        SLayout axes = TransposeLike(InitialLayoutDecision(tensor->ndim)->layout,
                                     from.LeafValue()->layout, to.LeafValue()->layout);
        return permute_dims(expr, LayoutToIntegers(axes));
      } else {
        auto index_map = LayoutIndexMap(
            (from.LeafValue()->layout.has_value() ? from.LeafValue()->layout.value().ndim() : 0),
            from.LeafValue()->layout.value(), to.LeafValue()->layout.value());
        ffi::ObjectPtr<LayoutTransformAttrs> attrs = ffi::make_object<LayoutTransformAttrs>(
            ffi::FromJSONGraph(ffi::ToJSONGraph(index_map)).as_or_throw<IndexMap>());
        const Op layout_transform_op_ = Op::Get("relax.layout_transform");
        auto ret_expr =
            Call(Type::Missing(), layout_transform_op_, {expr}, Attrs{std::move(attrs)}, {});
        return ret_expr;
      }
    };
    return TransformTupleLeaf<LayoutDecision>(
        VarReplacer::Replace(expr, var_remap_),
        std::array<NLayout, 2>({GetNLayout(var_layout_map_, expr), to}), fvisitleaf);
  }

  ffi::Array<Expr> RewriteArgs(const ffi::Array<Expr>& args, const ffi::Array<NLayout>& to) {
    // The `ffi::Array<Expr> args` array contains both tensor and
    // non-tensor arguments, where the `ffi::Array<NLayout> to` array only
    // contains tensor arguments.  The number of tensor arguments in
    // `args` should match the full extent of `to`.

    TVM_FFI_ICHECK_LE(to.size(), args.size());

    std::vector<Expr> new_args;
    for (size_t i = 0; i < args.size(); ++i) {
      Expr arg = args[i];
      if (i < to.size()) {
        arg = RewriteExpr(arg, to[i]);
      }
      new_args.push_back(arg);
    }

    return new_args;
  }

  void VisitBinding(const Binding& binding) final {
    // Emit the binding
    ExprMutator::VisitBinding(binding);
    // The layout is default to be initial if not rewritten.
    if (var_layout_map_.find(binding->var) == var_layout_map_.end()) {
      var_layout_map_[binding->var] = InitialNLayout(binding->var);
    }
  }

  Expr VisitVars_(const Var& var) {
    // We encounter a var use outside of inferrable regions, we rewrite it to initial layout.
    return RewriteExpr(var, InitialNLayout(var));
  }

  Expr VisitExpr_(const VarObj* op) final { return VisitVars_(ffi::GetRef<Var>(op)); }

  bool HasUnknownDimTensor(const NLayout& nlayout) {
    bool find = false;
    auto fvisit = [&](const LayoutDecision& layout) {
      find = find | (NLayoutEqual()(layout, LayoutDecision::InitUnknownDim()));
    };
    ForEachLeaf<LayoutDecision>(nlayout, fvisit);
    return find;
  }

  bool HasUnknownDimTensor(const ffi::Array<Expr>& args) {
    for (const auto& arg : args) {
      if (IsNestedTensor(arg)) {
        if (HasUnknownDimTensor(GetNLayout(var_layout_map_, arg))) {
          return true;
        }
      }
    }
    return false;
  }

  ffi::Optional<InferLayoutOutput> GetInferLayoutInfo(
      const CallObj* call_node,
      const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
      const LayoutCb& layout_cb, const VarLayoutMap& var_layout_map) {
    const OpObj* op_node = call_node->op.as<OpObj>();
    if (op_node == nullptr) return std::nullopt;
    Op op = ffi::GetRef<Op>(op_node).as_or_throw<Op>();
    const auto attr_map = Op::GetAttrMap(tvm::relax::op_attr::kInferLayout);
    if (attr_map.count(op) && !HasUnknownDimTensor(call_node->args)) {
      // If the op has FRelaxInferLayout, and all the input tensors have known ndim
      auto f = attr_map[op];
      auto call = ffi::GetRef<Call>(call_node);
      if (layout_cb != nullptr) {
        auto custom_layouts = layout_cb(call);
        return f(call, custom_layouts, var_layout_map);
      } else {
        return f(call, desired_layouts, var_layout_map);
      }
    } else {
      // Otherwise, we use the default policy.
      return std::nullopt;
    }
  }

  void VisitBinding_(const VarBindingObj* binding, const CallObj* call_node) final {
    ffi::Optional<InferLayoutOutput> res =
        GetInferLayoutInfo(call_node, desired_layouts_, layout_cb_, var_layout_map_);
    ffi::ObjectPtr<CallObj> new_call = ffi::make_object<CallObj>(*call_node);
    new_call->ty = Type::Missing();
    if (!res.has_value() ||
        (!IsNestedTensor(binding->var) && !binding->var->IsInstance<DataflowVarObj>())) {
      // Default policy: use the initial layout.
      // When we don't have the infer layout info, or it's a non-tensor global var binding.
      std::vector<NLayout> input_layout;
      for (const auto& arg : call_node->args) {
        input_layout.push_back(InitialNLayout(arg));
      }
      ffi::Array<Expr> new_args = RewriteArgs(call_node->args, std::move(input_layout));
      new_call->args = std::move(new_args);
      ReEmitBinding(binding, builder_->Normalize(Call(new_call)));
      // update the layout map
      var_layout_map_[binding->var] = InitialNLayout(binding->var);
    } else {
      // Convert the layout according to the inferred layout output.
      ffi::Array<Expr> new_args = RewriteArgs(call_node->args, res.value()->input_layouts);
      for (const auto& [i, arg] : res.value()->new_args) {
        new_args.Set(i->value.as<size_t>().value(), arg);
      }
      new_call->args = std::move(new_args);

      new_call->attrs = std::move(res.value()->new_attrs);
      Expr cur_call = builder_->Normalize(Call(new_call));
      if (binding->var->IsInstance<DataflowVarObj>()) {
        // Dataflow var, we emit the rewritten call.
        ReEmitBinding(binding, cur_call);
        // update the layout map
        var_layout_map_[binding->var] = res.value()->output_layouts[0];
      } else {
        // Global var (tensor), we rewrite it to initial layout
        TVM_FFI_ICHECK(IsNestedTensor(binding->var));
        if (!NLayoutEqual()(res.value()->output_layouts[0], InitialNLayout(binding->var))) {
          Var new_var = builder_->Emit(cur_call);
          var_layout_map_[new_var] = res.value()->output_layouts[0];
          cur_call = builder_->Normalize(RewriteExpr(new_var, InitialNLayout(binding->var)));
        }
        ReEmitBinding(binding, cur_call);
        // update the layout map
        var_layout_map_[binding->var] = InitialNLayout(binding->var);
      }
    }
  }

  void VisitBinding_(const VarBindingObj* binding, const TupleObj* val) final {
    std::vector<NLayout> input_layout;
    for (const auto& field : val->fields) {
      if (binding->var->IsInstance<DataflowVarObj>()) {
        // Df var: Use the current realized layout to group the tuple;
        input_layout.push_back(GetNLayout(var_layout_map_, field));
      } else {
        // Global var: Use the initial layout to group the tuple;
        input_layout.push_back(InitialNLayout(field));
      }
    }
    ffi::Array<Expr> new_fields = RewriteArgs(val->fields, std::move(input_layout));
    if (IsNestedTensor(binding->var)) {
      ReEmitBinding(binding, builder_->Normalize(Tuple(new_fields)));
      var_layout_map_[binding->var] = input_layout;
    }
  }

  void VisitBinding_(const VarBindingObj* binding, const TupleGetItemObj* val) final {
    NLayout input_layout = binding->var->IsInstance<DataflowVarObj>()
                               ? GetNLayout(var_layout_map_, val->tuple)
                               : InitialNLayout(val->tuple);
    ReEmitBinding(binding, builder_->Normalize(
                               TupleGetItem(RewriteExpr(val->tuple, input_layout), val->index)));
    // update the layout map
    var_layout_map_[binding->var] = input_layout.NestedArray()[val->index];
  }

  void VisitBinding_(const MatchCastObj* binding) final {
    if (!binding->var->IsInstance<DataflowVarObj>()) {
      ExprMutator::VisitBinding_(binding);
      return;
    }
    NLayout from_layout = InitialNLayout(binding->value);
    NLayout input_layout = GetNLayout(var_layout_map_, binding->value);
    auto fvisitleaf = [&](const Type& ty, std::array<NLayout, 2> layouts) -> Type {
      NLayout from = layouts[0], to = layouts[1];
      if (NLayoutEqual()(from, to)) return ty;
      // If not both from and to are unknown, then none of them can be unknown.
      TVM_FFI_ICHECK(!NLayoutEqual()(from, LayoutDecision::InitUnknownDim()) &&
                     !NLayoutEqual()(to, LayoutDecision::InitUnknownDim()))
          << "Cannot convert when exactly one of the layouts is unknown";
      const TensorTypeObj* tensor_ty = ty.as<TensorTypeObj>();
      TVM_FFI_ICHECK(tensor_ty != nullptr) << "We can not set layout for non-tensor struct";
      if (!tensor_ty->shape.has_value()) return ty;
      const ShapeExprObj* shape = tensor_ty->shape.value().as<ShapeExprObj>();
      if (shape == nullptr) return ty;
      TVM_FFI_ICHECK_EQ(
          shape->values.size(),
          (to.LeafValue()->layout.has_value() ? to.LeafValue()->layout.value().ndim() : 0));
      std::vector<PrimExpr> new_shape;
      for (size_t i = 0; i < shape->values.size(); ++i) {
        new_shape.push_back(shape->values[from.LeafValue()->layout.value().IndexOf(
            to.LeafValue()->layout.value()[i])]);
      }
      ffi::Optional<VDevice> vdev = tensor_ty->vdevice;
      return TensorType(ShapeExpr(new_shape), tensor_ty->dtype, vdev, tensor_ty->loc);
    };
    Type new_ty = TransformTupleLeaf<LayoutDecision>(
        binding->ty, std::array<NLayout, 2>({from_layout, input_layout}), fvisitleaf);
    // re-emit old binding if nothing changes
    if (new_ty.same_as(binding->ty)) {
      builder_->EmitNormalized(ffi::GetRef<MatchCast>(binding));
    } else {
      Var new_var = builder_->EmitMatchCast(RewriteExpr(binding->value, input_layout), new_ty);
      var_layout_map_[binding->var] = input_layout;
      this->var_remap_.insert_or_assign(binding->var, new_var);
    }
  }

  std::unordered_map<Var, NLayout> var_layout_map_;
  ffi::Map<ffi::String, ffi::Array<ffi::String>> desired_layouts_;
  LayoutCb layout_cb_;
};  // namespace relax

DataflowBlock ConvertLayoutPass(const DataflowBlock& df_block,
                                ffi::Map<ffi::String, ffi::Array<ffi::String>> desired_layouts,
                                LayoutCb layout_cb) {
  LayoutConvertMutator mutator(desired_layouts, layout_cb);
  return mutator.VisitBindingBlock(df_block).as_or_throw<DataflowBlock>();
}

namespace transform {

Pass ConvertLayout(ffi::Map<ffi::String, ffi::Array<ffi::String>> desired_layouts,
                   LayoutCb layout_cb) {
  ffi::TypedFunction<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func =
      [=](DataflowBlock df_block, IRModule m, PassContext pc) {
        return ConvertLayoutPass(df_block, desired_layouts, layout_cb);
      };
  return CreateDataflowBlockPass(pass_func, 0, "ConvertLayout");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.ConvertLayout", ConvertLayout);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
