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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/script/ir_builder/ir.h>
#include <tvm/tirx/type.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace s_tir {

using ir::ForFrameObj;
using tvm::s_tir::IterVar;
using tvm::s_tir::IterVarType;

FunctionFrame Function(bool is_private, bool persistent, bool is_declaration) {
  auto n = ffi::make_object<FunctionFrameObj>();
  n->is_private = is_private;
  n->is_declaration = is_declaration;
  n->persistent = persistent;
  n->attrs = {};
  return FunctionFrame(std::move(n));
}

TensorVar MatchTensor(ffi::ObjectRef param, ffi::Array<PrimExpr> shape, PrimType dtype,
                      ffi::Optional<Expr> data, runtime::MemoryScope storage_scope, int align,
                      ffi::Optional<Layout> layout, ffi::Array<PrimExpr> allocated_addr) {
  if (data.has_value()) {
    storage_scope = data.value()->ty.as_or_throw<PtrType>()->scope;
  }

  tvm::TensorRegion region{ffi::UnsafeInit{}};
  if (auto load = param.as<TensorLoad>()) {
    region = TensorRegion::FromTensorLoad(load.value());
  } else if (auto view = param.as<tvm::TensorRegion>()) {
    region = view.value();
  } else {
    TVM_FFI_THROW(ValueError) << "Unexpected type for MatchTensor";
  }
  TensorVar tensor("", tvm::tirx::TensorType(shape, dtype, storage_scope, align, layout));
  TVM_FFI_CHECK(tvm::tirx::MatchShapeStridePattern(tensor.ty()).has_value(), ValueError)
      << "S-TIR tensors require a regular memory layout without swizzles or thread axes";
  FunctionFrameObj::BindTensorRegion(tensor, std::move(region), std::move(allocated_addr));
  return tensor;
}

SBlockFrame Block(ffi::String name, bool no_realize, ffi::String exec_scope) {
  ffi::ObjectPtr<SBlockFrameObj> n = ffi::make_object<SBlockFrameObj>();
  n->name = name;
  n->iter_vars.clear();
  n->reads = std::nullopt;
  n->writes = std::nullopt;
  n->init = std::nullopt;
  n->alloc_tensors.clear();
  n->match_tensors.clear();
  n->annotations = std::nullopt;
  n->iter_values.clear();
  n->predicate = std::nullopt;
  n->no_realize = no_realize;
  return SBlockFrame(n);
}

BlockInitFrame Init() { return BlockInitFrame(ffi::make_object<BlockInitFrameObj>()); }

void Where(PrimExpr predicate) {
  SBlockFrame frame = FindSBlockFrame("Ts.where");
  if (frame->predicate.has_value()) {
    TVM_FFI_THROW(InternalError)
        << "ValueError: Duplicate block predicate declaration, previous one is "
        << frame->predicate;
  }
  frame->predicate = predicate;
}

void Reads(ffi::Array<ffi::ObjectRef> tensor_slices) {
  using namespace tvm::tirx;
  SBlockFrame frame = FindSBlockFrame("Ts.reads");
  if (frame->reads.has_value()) {
    TVM_FFI_THROW(InternalError)
        << "ValueError: Duplicate read region declaration, previous one is " << frame->reads;
  }
  ffi::Array<TensorRegion> reads;
  for (const ffi::ObjectRef& obj : tensor_slices) {
    if (auto tensor_region = obj.as<TensorRegion>()) {
      reads.push_back(tensor_region.value());
    } else if (auto tensor_load = obj.as<TensorLoad>()) {
      reads.push_back(TensorRegion::FromTensorLoad(tensor_load.value()));
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid type for tensor reads.";
    }
  }
  frame->reads = reads;
}

void Writes(ffi::Array<ffi::ObjectRef> tensor_slices) {
  using namespace tvm::tirx;
  SBlockFrame frame = FindSBlockFrame("Ts.writes");
  if (frame->writes.has_value()) {
    TVM_FFI_THROW(InternalError)
        << "ValueError: Duplicate write region declaration, previous one is " << frame->writes;
  }
  ffi::Array<TensorRegion> writes;
  for (const ffi::ObjectRef& obj : tensor_slices) {
    if (auto tensor_region = obj.as<TensorRegion>()) {
      writes.push_back(tensor_region.value());
    } else if (auto tensor_load = obj.as<TensorLoad>()) {
      writes.push_back(TensorRegion::FromTensorLoad(tensor_load.value()));
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid type for tensor writes.";
    }
  }
  frame->writes = writes;
}

/*! \brief Recursively merge two annotations, the new attrs will override the old ones */
ffi::Map<ffi::String, Any> MergeAnnotations(const ffi::Map<ffi::String, Any>& new_attrs,
                                            const ffi::Map<ffi::String, Any>& old_attrs) {
  ffi::Map<ffi::String, Any> result = old_attrs;
  for (const auto& [key, value] : new_attrs) {
    auto old_value = old_attrs.Get(key);
    // Case 1: the key is not in the old annotations, set the key to the new value
    if (!old_value) {
      result.Set(key, value);
      continue;
    }

    // Case 2: the key is in the old annotations
    // Case 2.1: both are dicts
    auto old_dict = old_value->try_cast<ffi::Map<ffi::String, Any>>();
    auto new_dict = value.try_cast<ffi::Map<ffi::String, Any>>();
    if (old_dict && new_dict) {
      // Recursively merge the two dicts
      auto merged_dict = MergeAnnotations(*old_dict, *new_dict);
      result.Set(key, merged_dict);
      continue;
    }
    // Case 2.2: the values are not both dicts, check if the keys are the same
    if (!ffi::AnyEqual()(old_value.value(), value)) {
      TVM_FFI_THROW(InternalError)
          << "ValueError: Try to merge two annotations with different values for key `" << key
          << "`, previous one is " << old_value.value() << ", new one is " << value;
    }
  }
  return result;
}

void BlockAttrs(ffi::Map<ffi::String, Any> attrs) {
  // First try to find an SBlockFrame
  ffi::Optional<SBlockFrame> sblock_frame = IRBuilder::Current()->FindFrame<SBlockFrame>();
  if (sblock_frame.has_value()) {
    if (!sblock_frame.value()->annotations.has_value()) {
      sblock_frame.value()->annotations = attrs;
    } else {
      sblock_frame.value()->annotations =
          MergeAnnotations(attrs, sblock_frame.value()->annotations.value());
    }
    return;
  }
  TVM_FFI_THROW(InternalError)
      << "ValueError: Ts.sblock_attr must be called at the top of a Ts.sblock() "
      << "frame, but Ts.sblock_attr occurred outside of any such frame";
}

TensorVar SBlockAllocTensor(ffi::Array<PrimExpr> shape, PrimType dtype, ffi::Optional<Expr> data,
                            runtime::MemoryScope storage_scope, int align,
                            ffi::Optional<Layout> layout, ffi::Array<PrimExpr> allocated_addr) {
  static const auto scope_shared = runtime::MemoryScope::Get("shared");
  static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");
  static const auto scope_local = runtime::MemoryScope::Get("local");

  auto scope = storage_scope;
  if (scope == runtime::MemoryScope::Global() || scope == scope_shared ||
      scope == scope_shared_dyn || scope == scope_local) {
    TVM_FFI_ICHECK(allocated_addr.empty())
        << "ValueError: For `" << scope
        << "` scope, Ts.alloc_tensor does not accept `allocated_addr`";
  }

  TensorVar tensor("", tvm::tirx::TensorType(shape, dtype, storage_scope, align, layout));
  TVM_FFI_CHECK(tvm::tirx::MatchShapeStridePattern(tensor.ty()).has_value(), ValueError)
      << "S-TIR tensors require a regular memory layout without swizzles or thread axes";
  IRBuilder builder = IRBuilder::Current();
  auto opt_func_frame = builder->FindFrame<tirx::FunctionFrame>();
  if (opt_func_frame.has_value()) {
    TVM_FFI_CHECK(opt_func_frame.value().as<FunctionFrameObj>() != nullptr, ValueError)
        << "ValueError: `Ts.alloc_tensor()` is only for s_tir Functions. "
           "Use `T.alloc_tensor()` inside default (tirx) Functions.";
  }

  // Walk up the frame stack: attach to the innermost enclosing s_tir::SBlock (lifting
  // the allocation past any intermediate For/If/While frames). Fall back to the
  // Function root when no sblock is in scope. When neither is present (raw
  // IRBuilder construction used by tests), just return the tensor.
  if (ffi::Optional<SBlockFrame> block_frame = builder->FindFrame<SBlockFrame>()) {
    block_frame.value()->alloc_tensors.push_back(tensor);
    if (!allocated_addr.empty()) {
      block_frame.value()->allocated_addresses.push_back(
          ffi::Tuple<Var, ffi::Array<PrimExpr>>(tensor, allocated_addr));
    }
  } else if (opt_func_frame.has_value()) {
    auto frame = ffi::GetRef<FunctionFrame>(opt_func_frame.value().as<FunctionFrameObj>());
    frame->root_alloc_tensors.push_back(tensor);
    if (!allocated_addr.empty()) {
      frame->root_allocated_addresses.push_back(
          ffi::Tuple<Var, ffi::Array<PrimExpr>>(tensor, allocated_addr));
    }
  } else {
    TVM_FFI_CHECK(allocated_addr.empty(), ValueError)
        << "sblock_alloc_tensor placement requires an owning block or function";
  }
  return tensor;
}
namespace axis {

IterVar PushBlockVar(IterVar iter_var, ffi::Optional<PrimExpr> binding) {
  if (ffi::Optional<SBlockFrame> opt_frame = IRBuilder::Current()->GetLastFrame<SBlockFrame>()) {
    SBlockFrame frame = opt_frame.value();
    TVM_FFI_CHECK(binding.has_value() || frame->no_realize, TypeError)
        << "Block axes require a binding unless no_realize=True";
    frame->iter_vars.push_back(iter_var);
    if (binding.has_value()) frame->iter_values.push_back(binding.value());
  } else {
    TVM_FFI_THROW(InternalError) << "TypeError: The last frame is not SBlockFrame";
  }
  return iter_var;
}

#define TVM_S_TIR_IR_BUILDER_AXIS(Method, Kind, Name)                                              \
  Var Method(Range dom, ffi::Optional<PrimExpr> binding, PrimType dtype) {                         \
    TVM_FFI_ICHECK(dom.defined()) << Name << " axis must have a domain";                           \
    PrimType min_ty = dom->begin.ty();                                                             \
    PrimType extent_ty = dom->extent.ty();                                                         \
    int bits = std::max({min_ty.bits(), extent_ty.bits(), dtype.bits()});                          \
    PrimType var_ty = dtype.WithBits(bits);                                                        \
    return PushBlockVar(IterVar(/*dom=*/dom, /*var=*/tvm::PrimVar("", var_ty), /*iter_type=*/Kind, \
                                /*thread_tag=*/""),                                                \
                        binding)                                                                   \
        ->var;                                                                                     \
  }
TVM_S_TIR_IR_BUILDER_AXIS(Spatial, tvm::s_tir::IterVarType::kDataPar, "Spatial");
TVM_S_TIR_IR_BUILDER_AXIS(Reduce, tvm::s_tir::IterVarType::kCommReduce, "Reduction");
TVM_S_TIR_IR_BUILDER_AXIS(Scan, tvm::s_tir::IterVarType::kOrdered, "Scan");
TVM_S_TIR_IR_BUILDER_AXIS(Opaque, tvm::s_tir::IterVarType::kOpaque, "Opaque");
#undef TVM_S_TIR_IR_BUILDER_AXIS

ffi::Array<Var> Remap(ffi::String kinds, ffi::Array<PrimExpr> bindings, PrimType dtype) {
  using namespace tvm::tirx;
  ffi::Array<Var> results;
  TVM_FFI_ICHECK_EQ(kinds.size(), bindings.size());
  int n = bindings.size();
  results.reserve(n);
  for (int i = 0; i < n; ++i) {
    char c = kinds.c_str()[i];
    PrimExpr e = bindings[i];
    auto v = e.as<PrimVar>();
    TVM_FFI_ICHECK(v) << "TypeError: Only Var is supported in T.axis.remap";
    ffi::Optional<Range> dom;
    for (const auto& frame : IRBuilder::Current()->frames) {
      if (const auto* for_frame = frame.as<ForFrameObj>()) {
        TVM_FFI_ICHECK_EQ(for_frame->doms.size(), for_frame->vars.size());
        int n = for_frame->doms.size();
        for (int i = 0; i < n; ++i) {
          if (for_frame->vars[i].same_as(v.value())) {
            dom = for_frame->doms[i];
            break;
          }
        }
        if (dom.has_value()) {
          break;
        }
      }
    }
    TVM_FFI_ICHECK(dom.has_value()) << "TypeError: Variable is not in the loop: " << v.value();
    PrimType dtype = v.value().ty();
    if (c == 'S') {
      results.push_back(PushBlockVar(IterVar(/*dom=*/dom,
                                             /*var=*/tvm::PrimVar("", dtype),
                                             /*iter_type=*/IterVarType::kDataPar,
                                             /*thread_tag=*/""),
                                     e)
                            ->var);
    } else if (c == 'R') {
      results.push_back(PushBlockVar(IterVar(/*dom=*/dom,
                                             /*var=*/tvm::PrimVar("", dtype),
                                             /*iter_type=*/IterVarType::kCommReduce,
                                             /*thread_tag=*/""),
                                     e)
                            ->var);
    } else {
      TVM_FFI_THROW(InternalError) << "Unknown axis kind: " << c;
    }
  }
  return results;
}

}  // namespace axis

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::script::ir_builder::detail::Namer::vtable().SetDispatch<tvm::s_tir::IterVarObj>(
      [](const ffi::ObjectRef& node, ffi::String name) -> void {
        using namespace tvm::s_tir;
        IterVarObj* var = const_cast<IterVarObj*>(node.as<IterVarObj>());
        tvm::script::ir_builder::detail::Namer::Name(var->var, name);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.MatchShapeStridePattern", tvm::tirx::MatchShapeStridePattern);
  refl::GlobalDef()
      .def("script.ir_builder.s_tir.Function", Function)
      .def("script.ir_builder.s_tir.MatchTensor", MatchTensor)
      .def("script.ir_builder.s_tir.Block", Block)
      .def("script.ir_builder.s_tir.Init", Init)
      .def("script.ir_builder.s_tir.Where", Where)
      .def("script.ir_builder.s_tir.Reads", Reads)
      .def("script.ir_builder.s_tir.Writes", Writes)
      .def("script.ir_builder.s_tir.BlockAttrs", BlockAttrs)
      .def("script.ir_builder.s_tir.SBlockAllocTensor", SBlockAllocTensor)
      .def("script.ir_builder.s_tir.AxisSpatial", axis::Spatial)
      .def("script.ir_builder.s_tir.AxisReduce", axis::Reduce)
      .def("script.ir_builder.s_tir.AxisScan", axis::Scan)
      .def("script.ir_builder.s_tir.AxisOpaque", axis::Opaque)
      .def("script.ir_builder.s_tir.AxisRemap", axis::Remap);
}

}  // namespace s_tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
