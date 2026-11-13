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
 * \file src/relax/backend/adreno/annotate_texture_storage.cc
 * \brief Texture Storage Annotation Pass for Adreno GPU targets.
 *
 * Texture realization for Adreno GPU targets requires fundamentally follows
 * Stage 1: Transforming the shapes with inner most dimension being 4
 * Stage 2: Annotate appropriate memory_scope hint in VDevice of Type
 * Stage 3: TIR lowering does injects texture load/store builtins looking at this scope
 * Stage 4: Finally codegen handles appropriate code looking at buffer types and load/store
 *          builtins.
 *
 * Stage 1 is generic and straight forward by using convert_layout pass that transforms the
 * shapes as well as injecting layout_transform ops as needed.
 *
 * Stage 2 This pass is responsible for injeting appropriate VDevice into Type and
 * adding any copies if there is a conflict between producer and consuner scopes.
 *
 * After convert_layout the mod looks like  below
 * @I.ir_module
 * class Module:
 *   @R.function
 *   def main(
 *     x: R.Tensor((2, 64, 56, 56), dtype="float32"),
 *     w: R.Tensor((32, 64, 3, 3), dtype="float32")
 *   ) -> R.Tensor((2, 32, 54, 54), dtype="float32"):
 *      with R.dataflow():
 *          lv: R.Tensor((2, 16, 56, 56, 4), dtype="float32") = R.layout_transform(
 *              x,
 *              index_map=T.index_map(
 *                  lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4)))
 *          lv1: R.Tensor((8, 64, 3, 3, 4), dtype="float32") = R.layout_transform(
 *              w,
 *              index_map=T.index_map(
 *                  lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4)))
 *          lv2: R.Tensor((2, 8, 54, 54, 4), dtype="float32") = R.nn.conv2d(
 *              lv,
 *              lv1,
 *              data_layout="NCHW4c",
 *              kernel_layout="OIHW4o",
 *              out_layout="NCHW4c",
 *              out_dtype="float32"
 *          )
 *          gv: R.Tensor((2, 32, 54, 54), dtype="float32") = R.layout_transform(
 *              lv2,
 *              index_map=T.index_map(
 *                  lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3)))
 *          R.output(gv)
 *      return gv
 *
 * Here, the param layout transforms are injected properly and the conv2d op is operating
 * in 5D shapes.
 *
 * Now, the scope annotation decisions are done by
 * - For op_pattern < kCommReduce we just look for shape being 5D and inner dimsion = 4
 * - For op_pattern > kCommReduce we make decisions selectively. Currently we do enable texture
 *   scope for Conv2D, PoolOps.
 * The trick here is whiel this pass is in action we need op_pattern information for ops that are
 * below kCommReduce as well op attrbuted for seletive ops like Conv2D and PoolOps.
 * op_pattern is available after legalization and TIROpPattern pass does an analysis. However,
 * op specific attributes doesn't exist after legalization.
 *
 * To solve this issue, we go legalization in parts.
 * At first, we call legalization by skipping the list of ops we wanted not to legalize.
 * LigalizeOps is enhanced to accept skip_ops for this purpose.
 * After legalization and AnnotateTIROpPattern this way the mod liiks like
 *
 * class Module:
 *   @R.function
 *   def main(
 *     x: R.Tensor((2, 64, 56, 56), dtype="float32"),
 *     w: R.Tensor((32, 64, 3, 3), dtype="float32")
 *   ) -> R.Tensor((2, 32, 54, 54), dtype="float32"):
 *      with R.dataflow():
 *           lv = R.call_tir(cls.te_layout_transform, (x,),
 *               out_ty=R.Tensor((2, 16, 56, 56, 4), dtype="float32")
 *           )
 *           lv1 = R.call_tir(cls.te_layout_transform1, (w,),
 *               out_ty=R.Tensor((8, 64, 3, 3, 4), dtype="float32")
 *           )
 *           lv2: R.Tensor((2, 8, 54, 54, 4), dtype="float32") = R.nn.conv2d(
 *               lv,
 *               lv1,
 *               data_layout="NCHW4c",
 *               kernel_layout="OIHW4o",
 *               out_layout="NCHW4c",
 *               out_dtype="float32"
 *           )
 *           gv = R.call_tir(cls.te_layout_transform2, (lv2,),
 *               out_ty=R.Tensor((2, 32, 54, 54), dtype="float32")
 *           )
 *           R.output(gv)
 *       return gv
 *
 * Here, the legalized functiontions does have op_pattern attribute.
 * We now have what we wanted to run this pass.
 *
 * This pass in principle does scope annotation based on sonsumer priotiry. i.e.
 * For any tensor object we tries to assign scope based on the sonsuner requirement.
 * The conflicts and multiple consumers for same tensor are handled by injecting
 * appropriate copies.
 * 1: CollectConsumerScopeInfo: Visitor collects all consumer demand for each input
 * 2: CollectProducerScopeInfo: Visitor does finalizes the scope for each input and output based
 *    on consumer scope information. It does evaluating mutiple consumer cases and conflicts.
 * 3: DefineVDevice: Pass does injects hint_on_device for each argument. It also tries to update
 *    out Type containing VDevice information. This update for tirx calls is straight forward
 *    as ty_args in CallObj is meant for this purpose. This ty_args for other calls by
 *    design is invalid as we do this by per-op type inference.
 *    Another issue with type inference per op is that it cannot decide this
 *    memory scope information which is done by this pass based on consumer demand.
 *    Hence, we are going to use the ty_args to indicate this information.
 *    So, this pass attributes ty_args for regular calls too, and type inference implementations
 *    take VDevice information from this hint. This also solves the issue of mixed VDevice
 *    for arguments of an op.
 * After these steps the mod looks like
 *
 * class Module:
 *   @R.function
 *   def main(
 *     x: R.Tensor((2, 64, 56, 56), dtype="float32"),
 *     w: R.Tensor((32, 64, 3, 3), dtype="float32")
 *   ) -> R.Tensor((2, 32, 54, 54), dtype="float32"):
 *      with R.dataflow():
 *          lv: R.Tensor((2, 64, 56, 56), dtype="float32") = R.hint_on_device(
 *               x, R.device(dev_type=4, dev_id=0), "global"
 *          )
 *          lv_1 = R.call_tir(cls.te_layout_transform, (lv,),
 *              out_ty=R.Tensor((2, 16, 56, 56, 4), dtype="float32",
 *                  vdevice="opencl:0:global.texture-nhwc"
 *              )
 *          )
 *          lv1: R.Tensor((32, 64, 3, 3), dtype="float32") = R.hint_on_device(
 *              w, R.device(dev_type=4, dev_id=0), "global"
 *          )
 *          lv1_1 = R.call_tir(cls.te_layout_transform1, (lv1,),
 *              out_ty=R.Tensor((8, 64, 3, 3, 4), dtype="float32",
 *                  vdevice="opencl:2:global.texture-weight"
 *              )
 *          )
 *          lv2: R.Tensor((2, 16, 56, 56, 4), dtype="float32",
 *              vdevice="opencl:0:global.texture-nhwc"
 *          ) = R.hint_on_device(lv_1, R.device(dev_type=4, dev_id=0), "global.texture-nhwc")
 *          lv3: R.Tensor((8, 64, 3, 3, 4), dtype="float32",
 *              vdevice="opencl:2:global.texture-weight"
 *          ) = R.hint_on_device(lv1_1, R.device(dev_type=4, dev_id=0), "global.texture-weight")
 *          lv2_1: R.Tensor((2, 8, 54, 54, 4), dtype="float32",
 *              vdevice="opencl:1:global"
 &          ) = R.nn.conv2d(
 *              lv2, lv3,
 *              data_layout="NCHW4c", kernel_layout="OIHW4o",
 *              out_layout="NCHW4c", out_dtype="float32",
 *              ty_args=(R.Tensor((2, 8, 54, 54, 4), dtype="float32",
 *                  vdevice="opencl:1:global"),
 *              )
 *          )
 *          lv4: R.Tensor((2, 8, 54, 54, 4), dtype="float32",
 *              vdevice="opencl:1:global"
 *          ) = R.hint_on_device(lv2_1, R.device(dev_type=4, dev_id=0), "global")
 *          gv = R.call_tir(cls.te_layout_transform2, (lv4,),
 *              out_ty=R.Tensor((2, 32, 54, 54), dtype="float32", vdevice="opencl:1:global")
 *          )
 *          R.output(gv)
 *      return gv
 *
 * What we have above is hint_on_device injections and out_ty for all calls.
 * Now, we apply RealizeVDevice to formalize the hints. Follwed by we also call
 * CanonicalizeBindings that removes redundant assignments like
 *
 * lv: R.Tensor((2, 64, 56, 56), dtype="float32", vdevice="opencl:1:global") = x
 * lv1: R.Tensor((32, 64, 3, 3), dtype="float32", vdevice="opencl:1:global") = w
 *
 * These assignments are result of hint_on_device not realizing any copy while consumer and
 * producer has same memory scope or vdevice. These assignments do impact operator fusion.
 *
 * Now the mod looks like,
 *
 * class Module:
 *    @R.function
 *    def main(
 *      x: R.Tensor((2, 64, 56, 56), dtype="float32"),
 *      w: R.Tensor((32, 64, 3, 3), dtype="float32")
 *    ) -> R.Tensor((2, 32, 54, 54), dtype="float32"):
 *        with R.dataflow():
 *           lv = R.call_tir(cls.te_layout_transform, (x,),
 *               out_ty=R.Tensor((2, 16, 56, 56, 4), dtype="float32",
 *                   vdevice="opencl:0:global.texture-nhwc"
 *               )
 *           )
 *           lv1 = R.call_tir(cls.te_layout_transform1, (w,),
 *               out_ty=R.Tensor((8, 64, 3, 3, 4), dtype="float32",
 *                   vdevice="opencl:2:global.texture-weight"
 *               )
 *           )
 *           lv2: R.Tensor((2, 8, 54, 54, 4), dtype="float32",
 *               vdevice="opencl:1:global"
 *           ) = R.nn.conv2d(
 *               lv2, lv3,
 *               data_layout="NCHW4c", kernel_layout="OIHW4o",
 *               out_layout="NCHW4c", out_dtype="float32",
 *               ty_args=(R.Tensor((2, 8, 54, 54, 4), dtype="float32",
 *                   vdevice="opencl:1:global"),
 *               )
 *           )
 *           gv = R.call_tir(cls.te_layout_transform2, (lv4,),
 *               out_ty=R.Tensor((2, 32, 54, 54), dtype="float32", vdevice="opencl:1:global")
 *           )
 *           R.output(gv)
 *       return gv
 *
 * Followed by, the compilation pipeline calls
 * - legalization of the remainng ops: This legalization do forwards the annotated out_ty
 *   VDevice information to tir_calls
 * - AnnotateTIROpPattern : TIROp Patterns for newly legalizes ops
 * - Fusion
 * - FoldVDeviceScopeChange: There existed some ToVDevice copies from texture to buffer
 *   This pass removes the copes and updates producer scope to global.
 * - SpecializeFunctionBasedOnCallSite: Finally we update the buffer parameter annotations
 *   according to VDevice scopes.
 *
 */

#include <tvm/ffi/cast.h>
#include <tvm/ir/op.h>
#include <tvm/relax/backend/adreno/transform.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op/op.h>
#include <tvm/relax/op_attr.h>
#include <tvm/tirx/index_map.h>

#include <tuple>

#include "../../op/tensor/manipulate.h"
#include "../../transform/infer_layout_utils.h"
#include "../../transform/utils.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

namespace backend {
namespace adreno {

using tvm::tirx::TensorVar;

static ffi::Array<PrimExpr> GetShapeFromTensorType(const TensorType& tensor_ty) {
  auto shape = tensor_ty->GetShape();
  TVM_FFI_ICHECK(shape.has_value());
  return shape.value();
}

/*
 * \brief generates consumer information for each var
 * \return scope_info is a map which contain for each var the corresponding call nodes that
 * consume it and corresponding scope it expects this input to be.
 * \return call_scope_info is a map of each call_node and array holding scope infor for each input.
 */
class CollectConsumerScopeInfo : public ExprVisitor {
 public:
  explicit CollectConsumerScopeInfo(Target target) : target_(std::move(target)) {}
  using ExprVisitor::VisitExpr_;

  std::pair<ffi::Map<Expr, ffi::Array<ffi::String>>,
            ffi::Map<Expr, ffi::Map<Expr, ffi::Array<ffi::String>>>>
  Collect(const IRModule& mod, Function func, const Target& target) {
    mod_ = mod;
    target_ = target;
    VisitExpr(func->body);

    for (const auto& val : tuple_item_to_binding) {
      if (scope_info.find(val.first) != scope_info.end()) {
        for (const auto& item_val : val.second) {
          // TODO(Siva): How about ops that generate tuples like split ?
          if (tuples_to_binding.find(item_val.first) == tuples_to_binding.end()) {
            continue;
          }
          auto producer_var = tuples_to_binding[item_val.first][item_val.second];
          if (scope_info.find(producer_var) == scope_info.end()) {
            scope_info.Set(producer_var, scope_info[val.first]);
          } else {
            auto ent = scope_info[producer_var];
            for (auto ent_val : scope_info[val.first]) {
              ent.Set(ent_val.first, ent_val.second);
            }
            scope_info.Set(producer_var, ent);
          }
        }
      }
    }
    return std::make_pair(call_scope_info, scope_info);
  }

  void VisitBinding_(const VarBindingObj* binding,
                     const TupleGetItemObj* tuple_get_item_node) final {
    /*
     * lv9 = R.call_tir(add, (lv7, m
     * lv: R.Tuple(
     *     R.Tensor((1, 3, 224, 224), dtype="float32"),
     *     R.Tensor((3,), dtype="float32"),
     *     R.Tensor((3,), dtype="float32")
     * ) = lv9, metadata["ir.GenericConst"][4], metadata["ir.GenericConst"][5]
     * lv1_1: R.Tensor((1, 3, 224, 224), dtype="float32") = lv[0]
     * lv4: R.Tensor((1, 64, 112, 112), dtype="float32") = R.nn.conv2d(lv1_1, .....
     *
     * lv1_1 scope is requested by conv2d now we need to populate the same to lv9
     * Capture essestial information here as
     *
     * lv => {(0, lv1_1), (1, ) ...}
     */
    ffi::Map<Expr, int> field_map;
    if (tuple_item_to_binding.find(ffi::GetRef<Expr>(binding->var.get())) !=
        tuple_item_to_binding.end()) {
      field_map = tuple_item_to_binding[ffi::GetRef<Expr>(binding->var.get())];
    }
    field_map.Set(ffi::GetRef<Expr>(tuple_get_item_node->tuple.get()), tuple_get_item_node->index);
    tuple_item_to_binding.Set(ffi::GetRef<Expr>(binding->var.get()), field_map);
  }

  void VisitBinding_(const VarBindingObj* binding, const TupleObj* tuple) final {
    tuples_to_binding.Set(ffi::GetRef<Expr>(binding->var.get()), tuple->fields);
  }

  void VisitExpr_(const CallObj* call) final {
    static const Op call_tir_op = Op::Get("relax.call_tir");
    ffi::Optional<GlobalVar> gv;
    ffi::Array<Attrs> op_attrs;
    ffi::Optional<int64_t> op_pattern = static_cast<int64_t>(OpPatternKind::kOpaque);
    Tuple func_args =
        call->op.same_as(call_tir_op) ? call->args[1].as_or_throw<Tuple>() : Tuple(call->args);

    if (call->op.same_as(call_tir_op)) {
      gv = call->args[0].as_or_throw<GlobalVar>();
      tirx::Function pfunc = mod_->Lookup(gv.value()).as_or_throw<tirx::Function>();
      op_attrs = ExtractAttrs<tirx::Function>(pfunc);
      op_pattern = ExtractPattern<tirx::Function>(pfunc);
    } else {
      if (call->attrs.has_value()) op_attrs = {call->attrs.value()};
      op_pattern = static_cast<int64_t>(OpPatternKind::kOpaque);
    }

    auto is_texture_supported = SupportsTexture(op_attrs, op_pattern.value());

    ffi::Array<ffi::String> arg_scope;
    for (uint32_t i = 0; i < func_args->fields.size(); ++i) {
      auto ty = GetType(func_args->fields[i]);
      if (auto tensor_ty = ty.as<TensorType>()) {
        bool is_texture =
            i < is_texture_supported.size() ? is_texture_supported[i] : is_texture_supported[0];
        auto scope = is_texture ? Scope(GetShapeFromTensorType(tensor_ty.value())) : "global";
        ffi::Map<Expr, ffi::Array<ffi::String>> ent_call;
        const VarObj* arg_var = func_args->fields[i].as<VarObj>();
        if (scope_info.find(ffi::GetRef<Expr>(arg_var)) != scope_info.end()) {
          ent_call = scope_info[ffi::GetRef<Expr>(arg_var)];
        }
        ent_call.Set(ffi::GetRef<Expr>(call), {scope});
        scope_info.Set(ffi::GetRef<Expr>(arg_var), ent_call);
        arg_scope.push_back(scope);
      }
    }
    call_scope_info.Set(ffi::GetRef<Expr>(call), arg_scope);
  }

 private:
  template <typename T>
  ffi::Array<Attrs> ExtractAttrs(const T& func) {
    ffi::Array<Attrs> op_attrs;
    ffi::Optional<ffi::ObjectRef> attrs = func->GetAttr(tvm::relax::attr::kOpAttrs);
    if (attrs) {
      if (auto val = attrs.value().as<Attrs>()) {
        op_attrs.push_back(val.value());
      } else if (auto val = attrs.value().as<ffi::Array<Attrs>>()) {
        op_attrs = val.value();
      }
    }
    return op_attrs;
  }

  template <typename T>
  ffi::Optional<int64_t> ExtractPattern(const T& func) {
    ffi::Optional<int64_t> op_pat = func->GetAttr(tvm::relax::attr::kOpPattern);
    return op_pat;
  }

  std::vector<bool> SupportsTexture(const ffi::Array<Attrs>& op_attrs, int64_t op_pattern) {
    if (op_pattern < OpPatternKind::kCommReduce) return {true};

    for (auto attr : op_attrs) {
      if (auto conv_attr = attr.as<Conv2DAttrs>()) {
        if (conv_attr->data_layout == "NCHW4c" && conv_attr->kernel_layout == "OIHW4o") {
          // No Texture for weights
          return {true, false};
        }
      } else if (auto pool_attrs = attr.as<Pool2DAttrs>()) {
        if (pool_attrs->layout == "NCHW4c") {
          return {true};
        }
      } else if (auto avg_attrs = attr.as<AdaptivePool2DAttrs>()) {
        if (avg_attrs->layout == "NCHW4c") {
          return {true};
        }
      } else if (attr.as<LayerNormAttrs>()) {
        return {true};
      }
    }

    return {false};
  }

  std::string Scope(ffi::Array<PrimExpr> shape) {
    // currently we support only textures been made from 5d tensors
    // 5d requirement is not limitation of textures in general, it is limitation how
    // we are representing memory scopes/layout and flattening of textures in tirx
    if (shape.size() == 5 && shape[4].as<IntImmObj>()->value == 4) {
      for (auto ind : shape) {
        if (!ind.as<IntImmObj>()) {
          // Dynamic tensors
          return "global.texture-nchw";
        }
      }
      std::map<int, std::string> diffs;
      int spatial_limit =
          static_cast<int>(target_->GetAttr<int64_t>("texture_spatial_limit").value_or(16384));
      int depth_limit =
          static_cast<int>(target_->GetAttr<int64_t>("texture_depth_limit").value_or(2048));
      int a0 = shape[0].as<IntImmObj>()->value.as<int>().value();
      int a1 = shape[1].as<IntImmObj>()->value.as<int>().value();
      int a2 = shape[2].as<IntImmObj>()->value.as<int>().value();
      int a3 = shape[3].as<IntImmObj>()->value.as<int>().value();

      int d1r = a0 * a1;
      int d2r = a2 * a3;
      int d3r = a1 * a2 * a3;
      std::string scope = "global";
      if (a0 < spatial_limit && d3r < spatial_limit)
        scope += ".texture-weight";
      else if (a0 < depth_limit && a1 < spatial_limit && d2r < spatial_limit)
        scope += ".texture-nhwc";
      else if (d1r < depth_limit && a2 < spatial_limit && a3 < spatial_limit)
        scope += ".texture";
      return scope;
    }
    return "global";
  }

  /* Map of each Var consumption by a call node and its scope */
  ffi::Map<Expr, ffi::Map<Expr, ffi::Array<ffi::String>>> scope_info;
  /* A map of call node and scope info for each argument it consunes */
  ffi::Map<Expr, ffi::Array<ffi::String>> call_scope_info;
  ffi::Map<Expr, ffi::Map<Expr, int>> tuple_item_to_binding;
  ffi::Map<Expr, ffi::Array<Expr>> tuples_to_binding;
  IRModule mod_;
  Target target_;
};

/*
 * \brief producer scope information consolidated based on consumer demands.
 * \return producer_info which is a map of each call node and corresponding out Type
 * This pass considers all consumers and their scope demand.
 * Any mismatches here introduces copies as needed.
 */
class CollectProducerScopeInfo : public ExprVisitor {
 public:
  CollectProducerScopeInfo(Target target, BlockBuilder builder)
      : target_(std::move(target)), builder_(std::move(builder)) {}
  using ExprVisitor::VisitExpr_;

  ffi::Map<Expr, Type> Collect(
      const IRModule& mod, Function func,
      const ffi::Map<Expr, ffi::Map<Expr, ffi::Array<ffi::String>>>& scope_info,
      const Target& target, const BlockBuilder& builder) {
    mod_ = mod;
    scope_info_ = scope_info;
    target_ = target;
    builder_ = builder;
    VisitExpr(func->body);

    return producer_ty;
  }

  void VisitBinding_(const VarBindingObj* binding, const CallObj* call) final {
    ExprVisitor::VisitBinding_(binding, call);

    static const Op call_tir_op = Op::Get("relax.call_tir");
    Type out_ty = Type::Missing();

    if (call->op.same_as(call_tir_op)) {
      out_ty = call->ty_args[0];
    } else {
      auto* op_ptr = call->op.as<OpObj>();
      Op op = ffi::GetRef<Op>(op_ptr);
      static auto op_map_context_free = Op::GetAttrMap(tvm::op_attr::kInferType);
      if (op_map_context_free.count(op)) {
        out_ty = Call::ReinferType(call);
      } else {
        static auto op_map_infer_ty = Op::GetAttrMap(tvm::relax::op_attr::kInferTypeWithBuilder);
        TVM_FFI_ICHECK(op_map_infer_ty.count(op))
            << " Cannot find a type inference attribute registered to op: " << op->name;
        out_ty = op_map_infer_ty[op](ffi::GetRef<Call>(call), builder_);
      }
    }

    std::unordered_map<ffi::String, int> scope_count;

    // Decide the final scope based on the max consumer demand. Rest will use to_device.
    auto arg_var = binding->var.as<VarObj>();
    if (scope_info_.find(ffi::GetRef<Expr>(arg_var)) != scope_info_.end()) {
      for (const auto& val : scope_info_[ffi::GetRef<Expr>(arg_var)]) {
        auto call_node = val.first.as_or_throw<Call>();
        if (scope_count.find(val.second[0]) == scope_count.end()) {
          scope_count.insert({val.second[0], 1});
        } else {
          auto curr_count = scope_count[val.second[0]];
          scope_count.emplace(val.second[0], curr_count + 1);
        }
      }
    }
    ffi::String final_scope = "global";
    int count = 0;
    for (const auto& sval : scope_count) {
      if (sval.second > count) {
        final_scope = sval.first;
        count = sval.second;
      }
    }
    // Applying same scope for outputs
    Type updated_ret_ty = UpdateOutputType(out_ty, {final_scope});
    producer_ty.Set(ffi::GetRef<Expr>(call), updated_ret_ty);
  }

 private:
  Type UpdateOutputType(const Type& out_ty, ffi::Array<ffi::String> scope) {
    if (out_ty->IsInstance<TensorTypeObj>()) {
      auto tensor_ty = out_ty.as_or_throw<TensorType>();
      auto shape_arr = GetShapeFromTensorType(tensor_ty);
      return TensorType(ShapeExpr(shape_arr), tensor_ty->dtype,
                        VDevice(target_, 0, runtime::MemoryScope::Get(scope[0])));
    }

    TVM_FFI_ICHECK(out_ty->IsInstance<TupleTypeObj>())
        << "Expect output type of call_tir to be either TupleType or "
           "TensorType, but got "
        << out_ty;

    const auto& tuple_ty = out_ty.as_or_throw<TupleType>();
    ffi::Array<Type> ty_fields;
    for (const auto& si : tuple_ty->fields) {
      TVM_FFI_ICHECK(si->IsInstance<TensorTypeObj>())
          << "Fields of TupleType must be TensorType for call_tir "
             "output structinfo, but got "
          << si;
      auto ty = si.as_or_throw<TensorType>();
      auto shape_arr = GetShapeFromTensorType(ty);
      ty_fields.push_back(TensorType(ShapeExpr(shape_arr), ty->dtype,
                                     VDevice(target_, 0, runtime::MemoryScope::Get(scope[0]))));
    }
    return TupleType(ty_fields);
  }

  ffi::Map<Expr, ffi::Map<Expr, ffi::Array<ffi::String>>> scope_info_;
  ffi::Map<Expr, Type> producer_ty;
  IRModule mod_;
  Target target_;
  BlockBuilder builder_;
};

/*
 * \brief main pass that injects hint_on_device for each argument based on producer,
 * consumer indormations. This also attributes ret Type for each call node.
 * This pass also calls the ReliaseVdevice that formalizes the hints by appropriately injecting
 * Vdevice copies as needed.
 */

class DefineVDevice : ExprMutator {
 public:
  explicit DefineVDevice(const Target& target) : target_(target) {}

  IRModule Run(IRModule& mod) {
    mod_ = mod;
    for (const auto& [gv, func] : mod_->functions) {
      if (func->IsInstance<relax::FunctionObj>()) {
        const auto& base_func = mod_->Lookup(gv);
        // Only non primitive relax functions
        if (base_func->HasNonzeroAttr(tvm::relax::attr::kPrimitive.name)) {
          continue;
        }
        auto info =
            CollectConsumerScopeInfo(target_).Collect(mod_, func.as_or_throw<Function>(), target_);
        call_scope_info_ = info.first;
        scope_info_ = info.second;
        producer_ty_ =
            CollectProducerScopeInfo(target_, builder_)
                .Collect(mod_, func.as_or_throw<Function>(), scope_info_, target_, builder_);
        relax::Function update_func = VisitExpr(func).as_or_throw<Function>();
        updates_->Add(gv, update_func);
      }
    }
    mod_.CopyOnWrite()->Update(updates_);

    ffi::Array<GlobalInfo> global_vdevices_;
    for (auto vdev : vdevices_) {
      global_vdevices_.push_back(vdev.as<GlobalInfo>().value());
    }
    mod_.CopyOnWrite()->global_infos.Set("vdevice", global_vdevices_);

    mod_ = relax::transform::DeadCodeElimination()(mod_);
    mod_ = relax::transform::RealizeVDevice()(mod_);
    mod_ = relax::transform::CanonicalizeBindings()(mod_);

    return mod_;
  }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallObj* call_node) override {
    auto call = ExprMutator::VisitExpr_(call_node).as_or_throw<Call>();
    static const Op call_tir_op = Op::Get("relax.call_tir");

    ffi::Optional<GlobalVar> gv;
    Tuple func_args =
        call->op.same_as(call_tir_op) ? call->args[1].as_or_throw<Tuple>() : Tuple(call->args);

    Type out_ty = Type::Missing();

    if (call->op.same_as(call_tir_op)) {
      gv = call->args[0].as_or_throw<GlobalVar>();
    }

    ffi::Array<Expr> new_args;
    Type updated_ret_ty = producer_ty_[ffi::GetRef<Expr>(call_node)];

    if (updated_ret_ty->IsInstance<TensorTypeObj>()) {
      auto tensor_ty = updated_ret_ty.as_or_throw<TensorType>();
      auto shape = tensor_ty->shape.value();
      auto dtype = tensor_ty->dtype;
      if (tensor_ty->vdevice.has_value()) {
        auto vdev = tensor_ty->vdevice.value();
        const VDevice& vdev_global = MakeGlobalVDevice(vdev);
        updated_ret_ty = TensorType(shape, dtype, vdev_global);
      }
    } else {
      TVM_FFI_ICHECK(updated_ret_ty->IsInstance<TupleTypeObj>())
          << "Expect output type of call_tir to be either TupleType or "
             "TensorType, but got "
          << updated_ret_ty;

      const auto& tuple_ty = updated_ret_ty.as_or_throw<TupleType>();
      ffi::Array<Type> ty_fields;
      for (const auto& si : tuple_ty->fields) {
        TVM_FFI_ICHECK(si->IsInstance<TensorTypeObj>())
            << "Fields of TupleType must be TensorType for call_tir "
               "output structinfo, but got "
            << si;
        auto ty = si.as_or_throw<TensorType>();

        auto shape_arr = GetShapeFromTensorType(ty);

        auto shape = ty->shape.value();
        auto dtype = ty->dtype;
        if (ty->vdevice.has_value()) {
          auto vdev = ty->vdevice.value();
          const VDevice& vdev_global = MakeGlobalVDevice(vdev);
          ty_fields.push_back(TensorType(shape, dtype, vdev_global));
        } else {
          ty_fields.push_back(ty);
        }
      }
      updated_ret_ty = TupleType(ty_fields);
    }

    int arg_idx = 0;
    for (auto arg : func_args->fields) {
      auto ty = GetType(arg);
      if (auto tensor_ty = ty.as<TensorType>()) {
        ffi::String scope = "global";
        if (call_scope_info_.find(ffi::GetRef<Expr>(call_node)) != call_scope_info_.end()) {
          scope = call_scope_info_[ffi::GetRef<Expr>(call_node)][arg_idx];
        }
        new_args.push_back(HintArg(arg, scope));
        arg_idx++;
      } else {
        new_args.push_back(arg);
      }
    }

    if (call->op.same_as(call_tir_op)) {
      return builder_->Normalize(Call(Type::Missing(), call_tir_op, {gv.value(), Tuple(new_args)},
                                      call->attrs, {updated_ret_ty}));
    } else {
      return builder_->Normalize(
          Call(Type::Missing(), call->op, new_args, call->attrs, {updated_ret_ty}));
    }
  }

 private:
  VDevice MakeGlobalVDevice(VDevice vdev) {
    int device_type = vdev->target.value()->GetTargetDeviceType();
    for (size_t i = 0; i < vdevices_.size(); ++i) {
      int dev_type = vdevices_[i]->target.value()->GetTargetDeviceType();
      if (dev_type == device_type && vdevices_[i]->vdevice_id == vdev->vdevice_id &&
          vdevices_[i]->memory_scope == vdev->memory_scope) {
        return vdevices_[i];
      }
    }
    vdevices_.push_back(vdev);
    return (vdevices_.back());
  }

  Expr HintArg(const Expr& arg, ffi::String scope) {
    if (arg->IsInstance<GenericConstObj>()) {
      if (auto tensor_ty = arg->ty.as<TensorTypeObj>()) {
        if (!tensor_ty->vdevice.has_value()) {
          const VDevice& vdev =
              MakeGlobalVDevice(VDevice(target_, 0, runtime::MemoryScope::Get(scope)));
          TVM_FFI_ICHECK(tensor_ty->shape.has_value())
              << "Shape not defined for a constant tensor ..!";
          arg->ty = TensorType(tensor_ty->shape.value(), tensor_ty->dtype, vdev, tensor_ty->loc);
          return arg;
        }
      }
    }
    ffi::ObjectPtr<HintOnDeviceAttrs> attrs = ffi::make_object<HintOnDeviceAttrs>();
    const VDevice& vdev = MakeGlobalVDevice(VDevice(target_, 0, runtime::MemoryScope::Get(scope)));
    attrs->device_type = vdev->target.value()->GetTargetDeviceType();
    attrs->index = vdev->vdevice_id;
    attrs->memory_scope = vdev->memory_scope;

    Expr new_arg = Call(Type::Missing(), hint_on_device_op_, {arg}, Attrs{std::move(attrs)}, {});

    return new_arg;
  }

  ffi::Optional<Target> GetTarget(const Type& ty) {
    auto tinfo = ty.as<TensorTypeObj>();
    if (tinfo->vdevice.has_value()) {
      auto vdevice = tinfo->vdevice.value();
      if (vdevice->target.has_value()) {
        return vdevice->target;
      }
    }
    return std::nullopt;
  }

  const Op hint_on_device_op_ = Op::Get("relax.hint_on_device");
  IRModule mod_;
  IRModule updates_;
  Target target_;
  ffi::Array<VDevice> vdevices_;
  ffi::Map<Expr, ffi::Map<Expr, ffi::Array<ffi::String>>> scope_info_;
  ffi::Map<Expr, Type> producer_ty_;
  ffi::Map<Expr, ffi::Array<ffi::String>> call_scope_info_;
};

namespace transform {

Pass AnnotateCustomMemoryScope(Target target) {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    return tvm::relax::backend::adreno::DefineVDevice(target).Run(mod);
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"AnnotateCustomMemoryScope");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.backend.adreno.transform.AnnotateCustomMemoryScope",
                        AnnotateCustomMemoryScope);
}
}  // namespace transform
}  // namespace adreno
}  // namespace backend
}  // namespace relax
}  // namespace tvm
