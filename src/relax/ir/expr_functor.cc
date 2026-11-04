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
 * \file src/relax/expr_functor.cc
 * \brief A wrapper around ExprFunctor which functionally updates the AST.
 *
 * ExprMutator uses memoization and self return in order to amortize
 * the cost of using functional updates.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>

// functions to be overriden.
#define RELAX_VISIT_BINDING_DISPATCH(OP)                                       \
  vtable.template SetDispatch<OP>(                                             \
      [](const ffi::ObjectRef& n, TSelf* self, const VarBindingObj* binding) { \
        self->VisitBinding_(binding, static_cast<const OP*>(n.get()));         \
      });

#define RELAX_VAR_BINDING_DISPATCH_IMPL(Type)                                           \
  Type::VisitBindingVTable Type::InitVisitBindingVTable() {                             \
    VisitBindingVTable vtable;                                                          \
    RELAX_VISIT_BINDING_DISPATCH(GenericConstObj);                                      \
    RELAX_VISIT_BINDING_DISPATCH(TupleObj);                                             \
    RELAX_VISIT_BINDING_DISPATCH(VarObj);                                               \
    RELAX_VISIT_BINDING_DISPATCH(DataflowVarObj);                                       \
    RELAX_VISIT_BINDING_DISPATCH(ShapeExprObj);                                         \
    RELAX_VISIT_BINDING_DISPATCH(ExternFuncObj);                                        \
    RELAX_VISIT_BINDING_DISPATCH(GlobalVarObj);                                         \
    RELAX_VISIT_BINDING_DISPATCH(FunctionObj);                                          \
    RELAX_VISIT_BINDING_DISPATCH(CallObj);                                              \
    RELAX_VISIT_BINDING_DISPATCH(SeqExprObj);                                           \
    RELAX_VISIT_BINDING_DISPATCH(IfExprObj);                                            \
    RELAX_VISIT_BINDING_DISPATCH(OpObj);                                                \
    RELAX_VISIT_BINDING_DISPATCH(TupleGetItemObj);                                      \
    RELAX_VISIT_BINDING_DISPATCH(StringImmObj);                                         \
    RELAX_VISIT_BINDING_DISPATCH(DataTypeImmObj);                                       \
    return vtable;                                                                      \
  }                                                                                     \
  void Type::VisitBinding_(const VarBindingObj* binding) {                              \
    static VisitBindingVTable vtable = InitVisitBindingVTable();                        \
    const Expr& value = binding->value;                                                 \
    TVM_FFI_ICHECK(value.defined()) << "Found null pointer node while traversing AST."; \
    if (vtable.CanDispatch(value)) {                                                    \
      vtable(value, this, binding);                                                     \
    } else {                                                                            \
      VisitBinding_(binding, value.get());                                              \
    }                                                                                   \
  }

// functions to be overriden.
#define RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(OP)                                  \
  void ExprVisitor::VisitBinding_(const VarBindingObj* binding, const OP* value) { \
    this->VisitExpr(binding->value);                                               \
    this->VisitVarDef(binding->var);                                               \
  }

// functions to be overriden.
#define RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(OP)                                  \
  void ExprMutator::VisitBinding_(const VarBindingObj* binding, const OP* value) { \
    Expr new_value = this->VisitExpr(binding->value);                              \
    this->ReEmitBinding(binding, new_value);                                       \
  }

namespace tvm {
namespace relax {
using namespace tvm::prim;

// ==================
// ExprVisitor

void ExprVisitor::VisitExprDepTypeField(const Type& ty) {
  // recurse into type in case they depend on value
  // under the current scope.
  default_tyfield_visitor_.VisitType(ty);
}

ExprVisitor::DefaultTypeFieldVisitor::DefaultTypeFieldVisitor(ExprVisitor* parent)
    : parent_(parent) {}

void ExprVisitor::DefaultTypeFieldVisitor::VisitTypeExprField(const Expr& expr) {
  parent_->VisitExpr(expr);
}

void ExprVisitor::DefaultTypeFieldVisitor::VisitTypeExprField(const PrimExpr& expr) {
  parent_->VisitTypePrimExprField(expr);
}

void ExprVisitor::DefaultTypeFieldVisitor::VisitType_(const FuncTypeObj* op) {
  // Do not recurse into function type
  // as they won't contain ref to values in current scope.
}

void VisitExprDepTypeFieldIfNeeded(ExprVisitor* visitor, const Type& ty) {
  if (!ty.as<MissingType>().has_value()) {
    auto* ty_node = ty.as<TypeObj>();
    TVM_FFI_DCHECK(ty_node != nullptr);
    visitor->VisitExprDepTypeField(ffi::GetRef<Type>(ty_node));
  }
}

void ExprVisitor::VisitExpr(const Expr& expr) { ExprFunctor::VisitExpr(expr); }

void ExprVisitor::VisitExpr_(const GenericConstObj* op) {
  this->VisitLoc(op->loc);
  // GenericConst's Type does not depend on Expr.
}

void ExprVisitor::VisitExpr_(const GlobalVarObj* op) {
  this->VisitLoc(op->loc);
  // FuncType is not value-dep
}

void ExprVisitor::VisitExpr_(const TupleObj* op) {
  this->VisitLoc(op->loc);
  for (Expr field : op->fields) {
    this->VisitExpr(field);
  }
  VisitExprDepTypeFieldIfNeeded(this, op->ty);
}

// Visit the use-site of a defined Var
void ExprVisitor::VisitExpr_(const VarObj* op) {
  this->VisitLoc(op->loc);
  VisitExprDepTypeFieldIfNeeded(this, op->ty);
}

// Visit the use-site of a defined DataflowVar
void ExprVisitor::VisitExpr_(const DataflowVarObj* op) {
  VisitExpr_(static_cast<const VarObj*>(op));
}

void ExprVisitor::VisitExpr_(const FunctionObj* op) {
  this->VisitLoc(op->loc);
  for (Var param : op->params) {
    this->VisitVarDef(param);
  }

  this->VisitExpr(op->body);
  // FuncType does not depend on Expr.
}

void ExprVisitor::VisitExpr_(const CallObj* op) {
  this->VisitLoc(op->loc);
  this->VisitExpr(op->op);

  for (Type ty_arg : op->ty_args) {
    this->VisitExprDepTypeField(ty_arg);
  }

  for (Expr arg : op->args) {
    this->VisitExpr(arg);
  }

  VisitExprDepTypeFieldIfNeeded(this, op->ty);
}

void ExprVisitor::VisitExpr_(const TensorLoadObj* op) {
  this->VisitLoc(op->loc);
  for (const PrimExpr& index : op->indices) {
    this->VisitExpr(index);
  }
  VisitExprDepTypeFieldIfNeeded(this, op->ty);
}

#define RELAX_VISIT_TIRX_BINOP(OP)                   \
  void ExprVisitor::VisitExpr_(const prim::OP* op) { \
    this->VisitLoc(op->loc);                         \
    this->VisitExpr(op->a);                          \
    this->VisitExpr(op->b);                          \
    VisitExprDepTypeFieldIfNeeded(this, op->ty);     \
  }

RELAX_VISIT_TIRX_BINOP(AddObj);
RELAX_VISIT_TIRX_BINOP(LShiftObj);
RELAX_VISIT_TIRX_BINOP(RShiftObj);
RELAX_VISIT_TIRX_BINOP(BitwiseAndObj);
RELAX_VISIT_TIRX_BINOP(BitwiseOrObj);
RELAX_VISIT_TIRX_BINOP(BitwiseXorObj);
RELAX_VISIT_TIRX_BINOP(SubObj);
RELAX_VISIT_TIRX_BINOP(MulObj);
RELAX_VISIT_TIRX_BINOP(DivObj);
RELAX_VISIT_TIRX_BINOP(ModObj);
RELAX_VISIT_TIRX_BINOP(FloorDivObj);
RELAX_VISIT_TIRX_BINOP(FloorModObj);
RELAX_VISIT_TIRX_BINOP(MinObj);
RELAX_VISIT_TIRX_BINOP(MaxObj);
RELAX_VISIT_TIRX_BINOP(EQObj);
RELAX_VISIT_TIRX_BINOP(NEObj);
RELAX_VISIT_TIRX_BINOP(LTObj);
RELAX_VISIT_TIRX_BINOP(LEObj);
RELAX_VISIT_TIRX_BINOP(GTObj);
RELAX_VISIT_TIRX_BINOP(GEObj);
RELAX_VISIT_TIRX_BINOP(AndObj);
RELAX_VISIT_TIRX_BINOP(OrObj);

#undef RELAX_VISIT_TIRX_BINOP

void ExprVisitor::VisitExpr_(const prim::CastObj* op) {
  this->VisitLoc(op->loc);
  this->VisitExpr(op->value);
  VisitExprDepTypeFieldIfNeeded(this, op->ty);
}

void ExprVisitor::VisitExpr_(const prim::NotObj* op) {
  this->VisitLoc(op->loc);
  this->VisitExpr(op->a);
  VisitExprDepTypeFieldIfNeeded(this, op->ty);
}

void ExprVisitor::VisitExpr_(const prim::BitwiseNotObj* op) {
  this->VisitLoc(op->loc);
  this->VisitExpr(op->a);
  VisitExprDepTypeFieldIfNeeded(this, op->ty);
}

void ExprVisitor::VisitExpr_(const prim::SelectObj* op) {
  this->VisitLoc(op->loc);
  this->VisitExpr(op->condition);
  this->VisitExpr(op->true_value);
  this->VisitExpr(op->false_value);
  VisitExprDepTypeFieldIfNeeded(this, op->ty);
}

void ExprVisitor::VisitExpr_(const prim::RampObj* op) {
  this->VisitLoc(op->loc);
  this->VisitExpr(op->base);
  this->VisitExpr(op->stride);
  this->VisitExpr(op->lanes);
  VisitExprDepTypeFieldIfNeeded(this, op->ty);
}

void ExprVisitor::VisitExpr_(const prim::BroadcastObj* op) {
  this->VisitLoc(op->loc);
  this->VisitExpr(op->value);
  this->VisitExpr(op->lanes);
  VisitExprDepTypeFieldIfNeeded(this, op->ty);
}

void ExprVisitor::VisitExpr_(const prim::ShuffleObj* op) {
  this->VisitLoc(op->loc);
  for (const PrimExpr& vector : op->vectors) {
    this->VisitExpr(vector);
  }
  for (const PrimExpr& index : op->indices) {
    this->VisitExpr(index);
  }
  VisitExprDepTypeFieldIfNeeded(this, op->ty);
}

void ExprVisitor::VisitExpr_(const tvm::IntImmObj* op) { this->VisitLoc(op->loc); }

void ExprVisitor::VisitExpr_(const tvm::FloatImmObj* op) { this->VisitLoc(op->loc); }

void ExprVisitor::VisitExpr_(const IfExprObj* op) {
  this->VisitLoc(op->loc);
  this->VisitExpr(op->cond);
  this->VisitExpr(op->true_branch);
  this->VisitExpr(op->false_branch);

  VisitExprDepTypeFieldIfNeeded(this, op->ty);
}

void ExprVisitor::VisitExpr_(const OpObj* op) { this->VisitLoc(op->loc); }

void ExprVisitor::VisitExpr_(const TupleGetItemObj* op) {
  this->VisitLoc(op->loc);
  this->VisitExpr(op->tuple);

  VisitExprDepTypeFieldIfNeeded(this, op->ty);
}

void ExprVisitor::VisitExpr_(const ShapeExprObj* op) {
  for (PrimExpr val : op->values) {
    this->VisitExpr(val);
  }
  this->VisitLoc(op->loc);
}

void ExprVisitor::VisitExpr_(const ExternFuncObj* op) {
  this->VisitLoc(op->loc);
  // FuncType does not depend on Expr.
}

void ExprVisitor::VisitExpr_(const SeqExprObj* op) {
  this->VisitLoc(op->loc);
  for (BindingBlock block : op->blocks) {
    this->VisitBindingBlock(block);
  }
  this->VisitExpr(op->body);

  VisitExprDepTypeFieldIfNeeded(this, op->ty);
}

void ExprVisitor::VisitExprFallback_(const ExprObj* op) {
  if (op->IsInstance<OpaqueExprObj>()) {
    return;
  }
  this->VisitExprDefault_(op);
}

void ExprVisitor::VisitExpr_(const StringImmObj* op) { this->VisitLoc(op->loc); }
void ExprVisitor::VisitExpr_(const DataTypeImmObj* op) { this->VisitLoc(op->loc); }

void ExprVisitor::VisitLoc(const Location& loc) {}

void ExprVisitor::VisitTypePrimExprField(const PrimExpr& expr) { this->VisitExpr(expr); }

// implementations of binding visitor dispatch
RELAX_VAR_BINDING_DISPATCH_IMPL(ExprVisitor);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(GenericConstObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(TupleObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(VarObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(DataflowVarObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(ShapeExprObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(ExternFuncObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(GlobalVarObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(FunctionObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(CallObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(SeqExprObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(IfExprObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(OpObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(TupleGetItemObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(ExprObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(StringImmObj);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(DataTypeImmObj);

void ExprVisitor::VisitBinding_(const MatchCastObj* binding) {
  this->VisitExpr(binding->value);
  this->VisitExprDepTypeField(binding->ty);
  this->VisitVarDef(binding->var);
}

void ExprVisitor::VisitBindingBlock_(const BindingBlockObj* block) {
  for (Binding binding : block->bindings) {
    this->VisitBinding(binding);
  }
}

void ExprVisitor::VisitBindingBlock_(const DataflowBlockObj* block) {
  for (Binding binding : block->bindings) {
    this->VisitBinding(binding);
  }
}

void ExprVisitor::VisitVarDef_(const DataflowVarObj* var) {
  VisitVarDef_(static_cast<const VarObj*>(var));
}

void ExprVisitor::VisitVarDef_(const VarObj* var) { this->VisitLoc(var->loc); }

void ExprVisitor::VisitBinding(const Binding& binding) {
  if (const auto* node = binding.as<VarBindingObj>()) {
    VisitBinding_(node);
  } else if (const auto* node = binding.as<MatchCastObj>()) {
    VisitBinding_(node);
  } else {
    TVM_FFI_THROW(TypeError) << "Invalid type: " << binding->GetTypeKey();
  }
}

void ExprVisitor::VisitBindingBlock(const BindingBlock& block) {
  if (const auto* node = block.as<DataflowBlockObj>()) {
    VisitBindingBlock_(node);
  } else if (const auto* node = block.as<BindingBlockObj>()) {
    VisitBindingBlock_(node);
  } else {
    TVM_FFI_THROW(TypeError) << "Invalid type: " << block->GetTypeKey();
  }
}

void ExprVisitor::VisitVarDef(const Var& var) {
  if (const auto* node = var.as<DataflowVarObj>()) {
    VisitVarDef_(node);
  } else if (const auto* node = var.as<VarObj>()) {
    VisitVarDef_(node);
  } else {
    TVM_FFI_THROW(TypeError) << "Invalid type: " << var->GetTypeKey();
  }
}

class ExprApplyVisit : public ExprVisitor {
 public:
  explicit ExprApplyVisit(std::function<void(const Expr&)> f) : f_(f) {}

  void VisitExpr(const Expr& e) final {
    ExprVisitor::VisitExpr(e);
    f_(e);
  }

 private:
  std::function<void(const Expr&)> f_;
};

void PostOrderVisit(const Expr& e, std::function<void(const Expr&)> fvisit) {
  ExprApplyVisit(fvisit).VisitExpr(e);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.analysis.post_order_visit", [](Expr expr, ffi::Function f) {
    PostOrderVisit(expr, [f](const Expr& n) { f(n); });
  });
}

// ==================
// ExprMutatorBase

Type ExprMutatorBase::VisitExprDepTypeField(const Type& ty) {
  // recurse into type in case they depend on value
  // under the current scope.
  return default_tyfield_mutator_.VisitType(ty);
}

ExprMutatorBase::DefaultTypeFieldMutator::DefaultTypeFieldMutator(ExprMutatorBase* parent)
    : parent_(parent) {}

Expr ExprMutatorBase::DefaultTypeFieldMutator::VisitTypeExprField(const Expr& expr) {
  return parent_->VisitExpr(expr);
}

PrimExpr ExprMutatorBase::DefaultTypeFieldMutator::VisitTypeExprField(const PrimExpr& expr) {
  return parent_->VisitTypePrimExprField(expr);
}

Type ExprMutatorBase::DefaultTypeFieldMutator::VisitType_(const FuncTypeObj* op) {
  // Do not recurse into function type
  // as they won't contain ref to values in current scope.
  return ffi::GetRef<Type>(op);
}

Expr ExprMutatorBase::VisitExpr(const Expr& expr) { return ExprFunctor::VisitExpr(expr); }

Expr ExprMutatorBase::VisitExpr_(const GenericConstObj* op) {
  // GenericConst' type won't be affected by Expr/PrimExpr change.
  return ffi::GetRef<Expr>(op);
}

Expr ExprMutatorBase::VisitExpr_(const GlobalVarObj* op) {
  // FuncType won't be affected by Expr/PrimExpr change.
  return ffi::GetRef<Expr>(op);
}

Expr ExprMutatorBase::VisitExpr_(const TupleObj* op) {
  bool unchanged = true;
  tvm::ffi::Array<Expr> fields;
  for (Expr field : op->fields) {
    Expr new_field = this->VisitExpr(field);
    fields.push_back(new_field);
    unchanged &= new_field.same_as(field);
  }

  if (unchanged) {
    // If tuple's type change it means that
    // one of its fields' type will change
    // so un-changed already implies that type won't change
    return ffi::GetRef<Expr>(op);
  } else {
    // when there is a change return a new tuple node
    return Tuple(fields, op->loc);
  }
}

// Visit the use-site of a defined Var
Expr ExprMutatorBase::VisitExpr_(const VarObj* op) { return ffi::GetRef<Expr>(op); }

// Visit the use-site of a defined DataflowVar
Expr ExprMutatorBase::VisitExpr_(const DataflowVarObj* op) {
  return VisitExpr_(static_cast<const VarObj*>(op));
}

Expr ExprMutatorBase::VisitExpr_(const FunctionObj* op) {
  // type of function is not value dependent
  // so no need to check ty field
  Expr body = this->VisitExpr(op->body);

  if (body.same_as(op->body)) {
    return ffi::GetRef<Expr>(op);
  } else {
    return Function(op->params, body, op->ret_ty, op->is_pure, op->attrs);
  }
}

Expr ExprMutatorBase::VisitExpr_(const CallObj* call_node) {
  Expr new_op = this->VisitExpr(call_node->op);
  bool unchanged = call_node->op.same_as(new_op);

  ffi::Array<Type> ty_args;
  for (Type ty_arg : call_node->ty_args) {
    Type new_ty_arg = this->VisitExprDepTypeField(ty_arg);
    ty_args.push_back(new_ty_arg);
    unchanged &= new_ty_arg.same_as(ty_arg);
  }

  tvm::ffi::Array<Expr> call_args;
  for (Expr arg : call_node->args) {
    Expr new_arg = this->VisitExpr(arg);
    call_args.push_back(new_arg);
    unchanged &= new_arg.same_as(arg);
  }

  Type ret_ty = call_node->ty;
  if (!ret_ty.as<MissingType>().has_value()) {
    ret_ty = this->VisitExprDepTypeField(ret_ty);
  }
  bool ret_ty_unchanged = ret_ty.same_as(call_node->ty);

  if (unchanged && ret_ty_unchanged) {
    return ffi::GetRef<Expr>(call_node);
  } else {
    // A result-type rewrite is authoritative only while the semantic inputs
    // are unchanged.  Once the callee, arguments, or type arguments change,
    // the old result may contain both rewritten dependent dimensions and
    // stale derived metadata (for example an unknown dtype).  Invalidate the
    // complete result so BlockBuilder can infer it from the rebuilt call.
    Type rebuilt_ret_ty = unchanged ? ret_ty : Type::Missing();
    return Call(rebuilt_ret_ty, new_op, call_args, call_node->attrs, ty_args, call_node->loc);
  }
}

Expr ExprMutatorBase::VisitExpr_(const TensorLoadObj* op) {
  ffi::Array<PrimExpr> indices = op->indices.Map(
      [this](const PrimExpr& e) { return this->VisitExpr(e).as_or_throw<PrimExpr>(); });
  if (indices.same_as(op->indices)) {
    return ffi::GetRef<Expr>(op);
  }
  return TensorLoad(op->source.as_or_throw<tirx::TensorVar>(), indices, op->loc);
}

#define RELAX_MUTATE_TIRX_BINOP(OP)                              \
  Expr ExprMutatorBase::VisitExpr_(const prim::OP##Obj* op) {    \
    PrimExpr a = this->VisitExpr(op->a).as_or_throw<PrimExpr>(); \
    PrimExpr b = this->VisitExpr(op->b).as_or_throw<PrimExpr>(); \
    if (a.same_as(op->a) && b.same_as(op->b)) {                  \
      return ffi::GetRef<Expr>(op);                              \
    }                                                            \
    return prim::OP(a, b, op->loc);                              \
  }

RELAX_MUTATE_TIRX_BINOP(Add);
RELAX_MUTATE_TIRX_BINOP(LShift);
RELAX_MUTATE_TIRX_BINOP(RShift);
RELAX_MUTATE_TIRX_BINOP(BitwiseAnd);
RELAX_MUTATE_TIRX_BINOP(BitwiseOr);
RELAX_MUTATE_TIRX_BINOP(BitwiseXor);
RELAX_MUTATE_TIRX_BINOP(Sub);
RELAX_MUTATE_TIRX_BINOP(Mul);
RELAX_MUTATE_TIRX_BINOP(Div);
RELAX_MUTATE_TIRX_BINOP(Mod);
RELAX_MUTATE_TIRX_BINOP(FloorDiv);
RELAX_MUTATE_TIRX_BINOP(FloorMod);
RELAX_MUTATE_TIRX_BINOP(Min);
RELAX_MUTATE_TIRX_BINOP(Max);
RELAX_MUTATE_TIRX_BINOP(EQ);
RELAX_MUTATE_TIRX_BINOP(NE);
RELAX_MUTATE_TIRX_BINOP(LT);
RELAX_MUTATE_TIRX_BINOP(LE);
RELAX_MUTATE_TIRX_BINOP(GT);
RELAX_MUTATE_TIRX_BINOP(GE);
RELAX_MUTATE_TIRX_BINOP(And);
RELAX_MUTATE_TIRX_BINOP(Or);

#undef RELAX_MUTATE_TIRX_BINOP

Expr ExprMutatorBase::VisitExpr_(const prim::CastObj* op) {
  PrimExpr value = this->VisitExpr(op->value).as_or_throw<PrimExpr>();
  return value.same_as(op->value)
             ? ffi::GetRef<Expr>(op)
             : Expr(prim::Cast(op->ty.as_or_throw<PrimType>(), value, op->loc));
}

Expr ExprMutatorBase::VisitExpr_(const prim::NotObj* op) {
  PrimExpr a = this->VisitExpr(op->a).as_or_throw<PrimExpr>();
  return a.same_as(op->a) ? ffi::GetRef<Expr>(op) : Expr(prim::Not(a, op->loc));
}

Expr ExprMutatorBase::VisitExpr_(const prim::BitwiseNotObj* op) {
  PrimExpr a = this->VisitExpr(op->a).as_or_throw<PrimExpr>();
  return a.same_as(op->a) ? ffi::GetRef<Expr>(op) : Expr(prim::BitwiseNot(a, op->loc));
}

Expr ExprMutatorBase::VisitExpr_(const prim::SelectObj* op) {
  PrimExpr condition = this->VisitExpr(op->condition).as_or_throw<PrimExpr>();
  PrimExpr true_value = this->VisitExpr(op->true_value).as_or_throw<PrimExpr>();
  PrimExpr false_value = this->VisitExpr(op->false_value).as_or_throw<PrimExpr>();
  if (condition.same_as(op->condition) && true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value)) {
    return ffi::GetRef<Expr>(op);
  }
  return prim::Select(condition, true_value, false_value, op->loc);
}

Expr ExprMutatorBase::VisitExpr_(const prim::RampObj* op) {
  PrimExpr base = this->VisitExpr(op->base).as_or_throw<PrimExpr>();
  PrimExpr stride = this->VisitExpr(op->stride).as_or_throw<PrimExpr>();
  PrimExpr lanes = this->VisitExpr(op->lanes).as_or_throw<PrimExpr>();
  if (base.same_as(op->base) && stride.same_as(op->stride) && lanes.same_as(op->lanes)) {
    return ffi::GetRef<Expr>(op);
  }
  return prim::Ramp(base, stride, lanes, op->loc);
}

Expr ExprMutatorBase::VisitExpr_(const prim::BroadcastObj* op) {
  PrimExpr value = this->VisitExpr(op->value).as_or_throw<PrimExpr>();
  PrimExpr lanes = this->VisitExpr(op->lanes).as_or_throw<PrimExpr>();
  if (value.same_as(op->value) && lanes.same_as(op->lanes)) {
    return ffi::GetRef<Expr>(op);
  }
  return prim::Broadcast(value, lanes, op->loc);
}

Expr ExprMutatorBase::VisitExpr_(const prim::ShuffleObj* op) {
  ffi::Array<PrimExpr> vectors = op->vectors.Map(
      [this](const PrimExpr& e) { return this->VisitExpr(e).as_or_throw<PrimExpr>(); });
  ffi::Array<PrimExpr> indices = op->indices.Map(
      [this](const PrimExpr& e) { return this->VisitExpr(e).as_or_throw<PrimExpr>(); });
  if (vectors.same_as(op->vectors) && indices.same_as(op->indices)) {
    return ffi::GetRef<Expr>(op);
  }
  return prim::Shuffle(vectors, indices, op->loc);
}

Expr ExprMutatorBase::VisitExpr_(const tvm::IntImmObj* op) { return ffi::GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const tvm::FloatImmObj* op) { return ffi::GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const IfExprObj* op) {
  Expr guard = this->VisitExpr(op->cond);
  Expr true_b = this->VisitExpr(op->true_branch);
  Expr false_b = this->VisitExpr(op->false_branch);
  if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b) && VisitAndCheckTypeFieldUnchanged(op->ty)) {
    return ffi::GetRef<Expr>(op);
  } else {
    return IfExpr(guard, true_b, false_b, op->loc);
  }
}

Expr ExprMutatorBase::VisitExpr_(const OpObj* op) { return ffi::GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const TupleGetItemObj* op) {
  auto t = this->VisitExpr(op->tuple);
  if (op->tuple.same_as(t)) {
    // type can be deterministically derived by tuple and index
    // if t does not change, then type won't change.
    return ffi::GetRef<Expr>(op);
  } else {
    return TupleGetItem(t, op->index, op->loc);
  }
}

Expr ExprMutatorBase::VisitExprFallback_(const ExprObj* op) {
  if (op->IsInstance<OpaqueExprObj>()) {
    return ffi::GetRef<Expr>(op);
  }
  return this->VisitExprDefault_(op);
}

Expr ExprMutatorBase::VisitExpr_(const StringImmObj* op) { return ffi::GetRef<Expr>(op); }
Expr ExprMutatorBase::VisitExpr_(const DataTypeImmObj* op) { return ffi::GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const ShapeExprObj* op) {
  auto values = op->values.Map(
      [this](const PrimExpr& e) { return this->VisitExpr(e).as_or_throw<PrimExpr>(); });

  if (values.same_as(op->values)) {
    // If values does not change, type won't change.
    return ffi::GetRef<Expr>(op);
  } else {
    return ShapeExpr(values, op->loc);
  }
}

Expr ExprMutatorBase::VisitExpr_(const ExternFuncObj* op) {
  // Type of function remains value independent.
  return ffi::GetRef<Expr>(op);
}

Expr ExprMutatorBase::VisitExpr_(const SeqExprObj* op) {
  bool all_blocks_unchanged = true;
  ffi::Array<BindingBlock> blocks;
  for (auto block : op->blocks) {
    BindingBlock new_block = this->VisitBindingBlock(block);
    if (!new_block->bindings.empty()) {
      blocks.push_back(new_block);
    }
    all_blocks_unchanged &= block.same_as(new_block);
  }

  Expr body = this->VisitExpr(op->body);

  if (all_blocks_unchanged && body.same_as(op->body) && VisitAndCheckTypeFieldUnchanged(op->ty)) {
    return ffi::GetRef<Expr>(op);
  }
  return SeqExpr(blocks, body);
}

BindingBlock ExprMutatorBase::VisitBindingBlock(const BindingBlock& block) {
  ffi::Array<Binding> bindings;
  if (const auto* node = block.as<BindingBlockObj>()) {
    for (auto binding : node->bindings) {
      if (auto var_binding = binding.as<VarBindingObj>()) {
        Expr new_value = this->VisitExpr(var_binding->value);
        bindings.push_back(VarBinding(var_binding->var, new_value));
      } else if (auto match_cast = binding.as<MatchCastObj>()) {
        Expr new_value = this->VisitExpr(match_cast->value);
        bindings.push_back(MatchCast(match_cast->var, new_value, match_cast->ty));
      } else {
        TVM_FFI_THROW(TypeError) << "Invalid type: " << binding->GetTypeKey();
      }
    }
  } else {
    TVM_FFI_THROW(TypeError) << "Invalid type: " << block->GetTypeKey();
  }

  if (block.as<DataflowBlockObj>()) {
    return DataflowBlock(bindings);
  } else {
    return BindingBlock(bindings);
  }
}

PrimExpr ExprMutatorBase::VisitTypePrimExprField(const PrimExpr& expr) {
  return this->VisitExpr(expr).as_or_throw<PrimExpr>();
}

// ==================
// ExprMutator

Expr ExprMutator::VisitExpr(const Expr& expr) {
  return builder_->Normalize(ExprFunctor::VisitExpr(expr));
}

// Visit the use-site of a defined Var
Expr ExprMutator::VisitExpr_(const VarObj* op) {
  Var var = ffi::GetRef<Var>(op);
  auto it = var_remap_.find(var);
  if (it != var_remap_.end()) {
    return it->second;
  }

  // default case return self.
  return var;
}

// Visit the use-site of a defined DataflowVar
Expr ExprMutator::VisitExpr_(const DataflowVarObj* op) {
  return VisitExpr_(static_cast<const VarObj*>(op));
}

Expr ExprMutator::VisitExpr_(const FunctionObj* op) {
  tvm::ffi::Array<Var> params;
  bool all_params_unchanged = true;
  for (Var param : op->params) {
    Var new_param = this->VisitVarDef(param);
    params.push_back(new_param);
    if (!param.same_as(new_param)) {
      var_remap_.insert_or_assign(param, new_param);
      all_params_unchanged = false;
    }
  }

  Type ret_ty = all_params_unchanged ? op->ret_ty : this->VisitExprDepTypeField(op->ret_ty);

  Expr body = this->VisitWithNewScope(op->body, params);

  if (all_params_unchanged && body.same_as(op->body)) {
    // No changes to the function, return the original object
    return ffi::GetRef<Expr>(op);
  } else if (IsBaseOf(GetType(body), ret_ty)) {
    // If the function was mutated into a form that can no longer
    // propagate shape information all the way to the return value, we
    // may keep the return type.  This is only allowed when the
    // body produces a return value that is the same as, or more
    // specific than, the pre-mutation type.  For example, if
    // the previous return value was `TensorType(shape=[16,16])`
    // but the body only produced `TensorType(ndim=2)`, we can
    // keep the more specific information.
    return Function(params, body, ret_ty, op->is_pure, op->attrs);
  } else {
    // If the function was mutated such that the body produces an
    // output that is incompatible with the original return struct
    // info, the original return type should not be used.  For
    // example, if the previous return value was
    // `TensorType(shape=[16,16])`, but the new return value is
    // `TensorType(shape=[8,8])`.
    return Function(params, body, std::nullopt, op->is_pure, op->attrs);
  }
}

Expr ExprMutator::VisitExpr_(const IfExprObj* op) {
  Expr guard = this->VisitExpr(op->cond);
  Expr true_b = this->VisitWithInnerScope(op->true_branch);
  Expr false_b = this->VisitWithInnerScope(op->false_branch);
  if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b) && VisitAndCheckTypeFieldUnchanged(op->ty)) {
    return ffi::GetRef<Expr>(op);
  } else {
    return IfExpr(guard, true_b, false_b, op->loc);
  }
}

Expr ExprMutator::VisitExpr_(const SeqExprObj* op) {
  bool all_blocks_unchanged = true;
  ffi::Array<BindingBlock> blocks;
  for (auto block : op->blocks) {
    BindingBlock new_block = this->VisitBindingBlock(block);
    if (!new_block->bindings.empty()) {
      blocks.push_back(new_block);
    }
    all_blocks_unchanged &= block.same_as(new_block);
  }

  builder_->BeginBindingBlock();
  Expr body = this->VisitExpr(op->body);
  BindingBlock prologue = builder_->EndBlock();
  if (!prologue->bindings.empty()) {
    blocks.push_back(prologue);
    all_blocks_unchanged = false;
  }

  if (all_blocks_unchanged && body.same_as(op->body) && VisitAndCheckTypeFieldUnchanged(op->ty)) {
    return ffi::GetRef<Expr>(op);
  } else {
    return SeqExpr(blocks, body);
  }
}

RELAX_VAR_BINDING_DISPATCH_IMPL(ExprMutator);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(GenericConstObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(TupleObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(VarObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(DataflowVarObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(ShapeExprObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(ExternFuncObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(GlobalVarObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(FunctionObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(CallObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(SeqExprObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(IfExprObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(OpObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(TupleGetItemObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(ExprObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(StringImmObj);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(DataTypeImmObj);

void ExprMutator::ReEmitBinding(const VarBindingObj* binding, Expr new_value) {
  Var new_var = this->VisitVarDef(binding->var);
  Var visited_var = new_var;

  // fast path: re-emit binding if nothing changes
  if (new_var.same_as(binding->var) && new_value.same_as(binding->value)) {
    builder_->EmitNormalized(ffi::GetRef<VarBinding>(binding));
    return;
  }

  auto new_ty = new_value->ty.as<Type>();

  TVM_FFI_CHECK(new_ty, InternalError)
      << "In binding of variable " << binding->var << ", the value " << new_value
      << " does not have Type.  "
      << "This typically occurs when ReEmitBinding is called without first calling Normalize.";

  Var temp = WithType(new_var, new_ty.value());
  if (!temp.same_as(new_var)) {
    new_var = temp;
  }

  this->var_remap_.insert_or_assign(binding->var, new_var);
  this->var_remap_.insert_or_assign(visited_var, new_var);
  this->var_remap_.insert_or_assign(new_var, new_var);

  builder_->EmitNormalized(VarBinding(new_var, new_value));
}

void ExprMutator::VisitBinding_(const MatchCastObj* binding) {
  Expr new_value = this->VisitExpr(binding->value);
  Type new_ty = this->VisitExprDepTypeField(binding->ty);

  Var new_var = this->VisitVarDef(binding->var);
  Var visited_var = new_var;

  MatchCast new_binding = [&]() -> MatchCast {
    if (new_var.same_as(binding->var) && new_value.same_as(binding->value) &&
        new_ty.same_as(binding->ty)) {
      // re-emit old binding if nothing changes
      return ffi::GetRef<MatchCast>(binding);
    } else {
      new_value = builder_->NormalizeArgument(new_value);
      new_var = WithType(new_var, new_ty);

      var_remap_.insert_or_assign(binding->var, new_var);
      var_remap_.insert_or_assign(visited_var, new_var);
      var_remap_.insert_or_assign(new_var, new_var);

      return MatchCast(new_var, new_value, new_ty, binding->loc);
    }
  }();

  builder_->EmitNormalized(new_binding);
  builder_->AddDefinitionToScope(new_binding->var);
}

BindingBlock ExprMutator::VisitBindingBlock_(const BindingBlockObj* block) {
  builder_->BeginBindingBlock();
  for (Binding binding : block->bindings) {
    this->VisitBinding(binding);
  }
  return builder_->EndBlock();
}

BindingBlock ExprMutator::VisitBindingBlock_(const DataflowBlockObj* block) {
  builder_->BeginDataflowBlock();
  for (auto binding : block->bindings) {
    this->VisitBinding(binding);
  }
  return builder_->EndBlock();
}

Var ExprMutator::VisitVarDef_(const DataflowVarObj* var) {
  Var output = VisitVarDef_(static_cast<const VarObj*>(var));
  // Because we delegate from DataflowVar visitor to Var visitor to
  // provide default behavior in subclasses, we may produce a Var
  // where we should produce a DataflowVar.
  if (!output->IsInstance<DataflowVarObj>()) {
    Var delegated_output = output;
    output = DataflowVar(output->name, GetType(output), output->loc);
    var_remap_.insert_or_assign(delegated_output, output);
  }
  return output;
}

Var ExprMutator::VisitVarDef_(const VarObj* var) {
  if (auto* ty_node = var->ty.as<TypeObj>()) {
    Type ty = this->VisitExprDepTypeField(ffi::GetRef<Type>(ty_node));
    if (ty.same_as(var->ty)) {
      return ffi::GetRef<Var>(var);
    } else {
      return Var(var->name, ty, var->loc);
    }
  } else {
    return ffi::GetRef<Var>(var);
  }
}

void ExprMutator::VisitBinding(const Binding& binding) {
  if (const auto* node = binding.as<VarBindingObj>()) {
    VisitBinding_(node);
  } else if (const auto* node = binding.as<MatchCastObj>()) {
    VisitBinding_(node);
  } else {
    TVM_FFI_THROW(TypeError) << "Invalid type: " << binding->GetTypeKey();
  }
}

BindingBlock ExprMutator::VisitBindingBlock(const BindingBlock& block) {
  BindingBlock ret(ffi::UnsafeInit{});
  if (const auto* node = block.as<DataflowBlockObj>()) {
    ret = VisitBindingBlock_(node);
  } else if (const auto* node = block.as<BindingBlockObj>()) {
    ret = VisitBindingBlock_(node);
  } else {
    TVM_FFI_THROW(TypeError) << "Invalid type: " << block->GetTypeKey();
  }
  return ret;
}

Var ExprMutator::VisitVarDef(const Var& var) {
  if (const auto* node = var.as<DataflowVarObj>()) {
    return VisitVarDef_(node);
  } else if (const auto* node = var.as<VarObj>()) {
    return VisitVarDef_(node);
  }
  TVM_FFI_THROW(TypeError) << "Invalid type: " << var->GetTypeKey();
  throw;
}

Expr ExprMutator::VisitWithNewScope(const Expr& expr, ffi::Optional<ffi::Array<Var>> params) {
  TVM_FFI_ICHECK(expr->IsInstance<SeqExprObj>())
      << "Normal form requires all new scope is stored as SeqExpr";

  PrimExpr constraint = IntImm::Bool(true);
  if (params.has_value()) {
    auto non_negative_expressions =
        CollectNonNegativeExpressions(TupleType(params.value().Map(GetType)));
    for (const auto& expr : non_negative_expressions) {
      constraint = constraint && (expr >= 0);
    }
  }

  builder_->BeginScope(params);
  // Outer scope only includes TIR variables that can be inferred from
  // the function parameters.
  auto analyzer = builder_->GetAnalyzer();
  auto saved_context = analyzer->GetConstraintContext();
  auto context = saved_context->Snapshot();
  context->Append(sym::PredicateConstraint(constraint));
  analyzer->SyncConstraintContext(context);
  builder_->BeginInnerScope();
  // Inner scope also includes any TIR variables that are defined by
  // MatchCast nodes, and are internal to the scope.
  Expr ret = this->VisitExpr(expr);

  builder_->EndScope();

  // Normalization (and the resulting Type inference) of the
  // expr occurs outside of the body's parameters, but inside the
  // function signature's scope.  This keeps variables that are
  // inferable based on the function signature, to allow callers to
  // propagate Type across the function.
  ret = builder_->Normalize(ret);
  builder_->EndScope();
  analyzer->SyncConstraintContext(saved_context);
  return ret;
}

Expr ExprMutator::VisitWithInnerScope(const Expr& expr) {
  TVM_FFI_ICHECK(expr->IsInstance<SeqExprObj>())
      << "Normal form requires all new scope is stored as SeqExpr";

  builder_->BeginInnerScope();
  Expr ret = this->VisitExpr(expr);
  builder_->EndScope();
  return ret;
}

ffi::Optional<Expr> ExprMutator::LookupBinding(const Var& var) {
  if (auto it = var_remap_.find(var); it != var_remap_.end()) {
    return builder_->LookupBinding(it->second);
  }
  return builder_->LookupBinding(var);
}

Var ExprMutator::WithType(Var var, Type ty) {
  TVM_FFI_ICHECK(!ty.as<MissingType>().has_value());

  // TODO(relax-team) add TypeEqual check
  if (!var->ty.as<MissingType>().has_value()) {
    // use same-as as a quick path
    if (var->ty.same_as(ty) || ffi::StructuralEqual()(var->ty, ty)) {
      return var;
    } else {
      Var new_var = var.as<DataflowVarObj>() ? DataflowVar(var->name, ty, var->loc)
                                             : Var(var->name, ty, var->loc);
      return new_var;
    }
  } else {
    UpdateType(var, ty);
    return var;
  }
}

}  // namespace relax
}  // namespace tvm
