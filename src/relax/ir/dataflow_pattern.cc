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
 * \file src/relax/ir/dataflow_pattern.cc
 * \brief The dataflow pattern language for Relax
 */

#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/dataflow_pattern_functor.h>

#include <sstream>
#include <stack>
#include <string>

namespace tvm {
namespace relax {

// Helper used inside RELAX_PATTERN_PRINTER_DEF lambdas.
// Mimics the ReprPrinter interface (p->stream, p->Print) so that existing
// REPR_LAMBDA bodies compile without modification.
// fn_repr is the canonical dispatch callback from the kRepr machinery; Print
// routes through it rather than calling ffi::ReprPrint directly so that any
// recursion context threaded by the printer is preserved.
struct PatternReprPrinterHelper {
  std::ostringstream stream;
  ffi::Function fn_repr;
  explicit PatternReprPrinterHelper(ffi::Function fn_repr_) : fn_repr(std::move(fn_repr_)) {}
  void Print(const ffi::ObjectRef& x) { stream << fn_repr(ffi::AnyView(x)).cast<ffi::String>(); }
};

#define RELAX_PATTERN_PRINTER_DEF(NODE_TYPE, REPR_LAMBDA)                                          \
  TVM_FFI_STATIC_INIT_BLOCK() {                                                                    \
    namespace refl = tvm::ffi::reflection;                                                         \
    refl::TypeAttrDef<NODE_TYPE>().def(                                                            \
        ffi::type_attr::kRepr.name, [](ffi::ObjectRef ref, ffi::Function fn_repr) -> ffi::String { \
          auto* node = static_cast<const NODE_TYPE*>(ref.get());                                   \
          PatternReprPrinterHelper printer(fn_repr);                                               \
          auto* p = &printer;                                                                      \
          REPR_LAMBDA(p, node);                                                                    \
          return printer.stream.str();                                                             \
        });                                                                                        \
  }

ExternFuncPattern::ExternFuncPattern(ffi::String global_symbol) : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<ExternFuncPatternObj> n = ffi::make_object<ExternFuncPatternObj>();
  n->global_symbol_ = std::move(global_symbol);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ExternFuncPatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.ExternFuncPattern",
                        [](ffi::String global_symbol) { return ExternFuncPattern(global_symbol); });
}
RELAX_PATTERN_PRINTER_DEF(ExternFuncPatternObj, [](auto p, auto node) {
  p->stream << "ExternFuncPattern(" << node->global_symbol() << ")";
});

VarPattern::VarPattern(ffi::String name_hint) : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<VarPatternObj> n = ffi::make_object<VarPatternObj>();
  n->name = std::move(name_hint);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  VarPatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.VarPattern",
                        [](ffi::String name_hint) { return VarPattern(name_hint); });
}
RELAX_PATTERN_PRINTER_DEF(VarPatternObj, [](auto p, auto node) {
  p->stream << "VarPattern(" << node->name_hint() << ")";
});

DataflowVarPattern::DataflowVarPattern(ffi::String name_hint) : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<DataflowVarPatternObj> n = ffi::make_object<DataflowVarPatternObj>();
  n->name = std::move(name_hint);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  DataflowVarPatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.DataflowVarPattern",
                        [](ffi::String name_hint) { return DataflowVarPattern(name_hint); });
}

RELAX_PATTERN_PRINTER_DEF(DataflowVarPatternObj, [](auto p, auto node) {
  p->stream << "DataflowVarPattern(" << node->name_hint() << ")";
});

GlobalVarPattern::GlobalVarPattern(ffi::String name_hint) : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<GlobalVarPatternObj> n = ffi::make_object<GlobalVarPatternObj>();
  n->name = std::move(name_hint);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.dpl.GlobalVarPattern",
                        [](ffi::String name_hint) { return GlobalVarPattern(name_hint); });
}
RELAX_PATTERN_PRINTER_DEF(GlobalVarPatternObj, [](auto p, auto node) {
  p->stream << "GlobalVarPattern(" << node->name_hint() << ")";
});

ExprPattern::ExprPattern(Expr expr) : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<ExprPatternObj> n = ffi::make_object<ExprPatternObj>(std::move(expr));
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ExprPatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.ExprPattern", [](Expr e) { return ExprPattern(e); });
}
RELAX_PATTERN_PRINTER_DEF(ExprPatternObj, [](auto p, auto node) { p->Print(node->expr); });

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ConstantPatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.ConstantPattern", []() {
    auto c = ConstantPattern(ffi::UnsafeInit{}, ffi::make_object<ConstantPatternObj>());
    return c;
  });
}
RELAX_PATTERN_PRINTER_DEF(ConstantPatternObj,
                          [](auto p, auto node) { p->stream << "ConstantPattern()"; });

CallPattern::CallPattern(DFPattern op, ffi::Array<DFPattern> args, bool varg_default_wildcard)
    : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<CallPatternObj> n = ffi::make_object<CallPatternObj>(std::move(op));
  n->args = std::move(args);
  n->varg_default_wildcard = varg_default_wildcard;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  CallPatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.CallPattern",
                        [](DFPattern op, ffi::Array<DFPattern> args, bool varg_default_wildcard) {
                          return CallPattern(op, args, varg_default_wildcard);
                        });
}
RELAX_PATTERN_PRINTER_DEF(CallPatternObj, [](auto p, auto node) {
  p->stream << node->op << "(";
  for (size_t i = 0; i < node->args.size(); ++i) {
    if (i != 0) p->stream << ", ";
    p->stream << node->args[i];
  }
  if (node->varg_default_wildcard) {
    if (node->args.size() != 0) p->stream << ", ";
    p->stream << "...";
  }
  p->stream << ")";
});

PrimArrPattern::PrimArrPattern(ffi::Array<PrimExpr> arr) : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<PrimArrPatternObj> n = ffi::make_object<PrimArrPatternObj>();
  n->fields = std::move(arr);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  PrimArrPatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.PrimArrPattern",
                        [](ffi::Array<PrimExpr> arr) { return PrimArrPattern(std::move(arr)); });
}
RELAX_PATTERN_PRINTER_DEF(PrimArrPatternObj, [](auto p, auto node) {
  p->stream << "PrimArrPattern(" << node->fields << ")";
});

FunctionPattern::FunctionPattern(ffi::Array<DFPattern> params, DFPattern body)
    : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<FunctionPatternObj> n = ffi::make_object<FunctionPatternObj>(std::move(body));
  n->params = std::move(params);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  FunctionPatternObj::RegisterReflection();
  refl::GlobalDef().def(
      "relax.dpl.FunctionPattern",
      [](ffi::Array<DFPattern> params, DFPattern body) { return FunctionPattern(params, body); });
}
RELAX_PATTERN_PRINTER_DEF(FunctionPatternObj, [](auto p, auto node) {
  p->stream << "FunctionPattern(" << node->params << ", " << node->body << ")";
});

TuplePattern::TuplePattern(tvm::ffi::Array<DFPattern> fields) : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<TuplePatternObj> n = ffi::make_object<TuplePatternObj>();
  n->fields = std::move(fields);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TuplePatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.TuplePattern",
                        [](tvm::ffi::Array<DFPattern> fields) { return TuplePattern(fields); });
}
RELAX_PATTERN_PRINTER_DEF(TuplePatternObj, [](auto p, auto node) {
  p->stream << "TuplePattern(" << node->fields << ")";
});

UnorderedTuplePattern::UnorderedTuplePattern(tvm::ffi::Array<DFPattern> fields)
    : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<UnorderedTuplePatternObj> n = ffi::make_object<UnorderedTuplePatternObj>();
  n->fields = std::move(fields);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  UnorderedTuplePatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.UnorderedTuplePattern", [](tvm::ffi::Array<DFPattern> fields) {
    return UnorderedTuplePattern(fields);
  });
}
RELAX_PATTERN_PRINTER_DEF(UnorderedTuplePatternObj, [](auto p, auto node) {
  p->stream << "UnorderedTuplePattern(" << node->fields << ")";
});

TupleGetItemPattern::TupleGetItemPattern(DFPattern tuple, int index)
    : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<TupleGetItemPatternObj> n =
      ffi::make_object<TupleGetItemPatternObj>(std::move(tuple));
  n->index = index;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TupleGetItemPatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.TupleGetItemPattern", [](DFPattern tuple, int index) {
    return TupleGetItemPattern(tuple, index);
  });
}
RELAX_PATTERN_PRINTER_DEF(TupleGetItemPatternObj, [](auto p, auto node) {
  p->stream << "TupleGetItemPattern(" << node->tuple << ", " << node->index << ")";
});

AndPattern::AndPattern(DFPattern left, DFPattern right) : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<AndPatternObj> n =
      ffi::make_object<AndPatternObj>(std::move(left), std::move(right));
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AndPatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.AndPattern",
                        [](DFPattern left, DFPattern right) { return AndPattern(left, right); });
}
RELAX_PATTERN_PRINTER_DEF(AndPatternObj, [](auto p, auto node) {
  p->stream << "AndPattern(" << node->left << " & " << node->right << ")";
});

OrPattern::OrPattern(DFPattern left, DFPattern right) : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<OrPatternObj> n =
      ffi::make_object<OrPatternObj>(std::move(left), std::move(right));
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  OrPatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.OrPattern",
                        [](DFPattern left, DFPattern right) { return OrPattern(left, right); });
}
RELAX_PATTERN_PRINTER_DEF(OrPatternObj, [](auto p, auto node) {
  p->stream << "OrPattern(" << node->left << " | " << node->right << ")";
});

NotPattern::NotPattern(DFPattern reject) : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<NotPatternObj> n = ffi::make_object<NotPatternObj>(std::move(reject));
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  NotPatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.NotPattern",
                        [](DFPattern reject) { return NotPattern(reject); });
}
RELAX_PATTERN_PRINTER_DEF(NotPatternObj,
                          [](auto p, auto node) { p->stream << "!(" << node->reject << ")"; });

WildcardPattern::WildcardPattern() : DFPattern(ffi::UnsafeInit{}) {
  data_ = ffi::make_object<WildcardPatternObj>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  WildcardPatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.WildcardPattern", []() { return WildcardPattern(); });
}
RELAX_PATTERN_PRINTER_DEF(WildcardPatternObj, [](auto p, auto node) { p->stream << "*"; });

TypePattern::TypePattern(DFPattern pattern, Type ty) : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<TypePatternObj> n = ffi::make_object<TypePatternObj>(std::move(pattern));
  n->ty = std::move(ty);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TypePatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.TypePattern",
                        [](DFPattern pattern, Type ty) { return TypePattern(pattern, ty); });
}
RELAX_PATTERN_PRINTER_DEF(TypePatternObj, [](auto p, auto node) {
  p->stream << "TypePattern(" << node->pattern << " has relax Type " << node->ty << ")";
});

ShapePattern::ShapePattern(DFPattern pattern, ffi::Array<PrimExpr> shape)
    : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<ShapePatternObj> n = ffi::make_object<ShapePatternObj>(std::move(pattern));
  n->shape = std::move(shape);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ShapePatternObj::RegisterReflection();
  refl::GlobalDef().def(
      "relax.dpl.ShapePattern",
      [](DFPattern pattern, ffi::Array<PrimExpr> shape) { return ShapePattern(pattern, shape); });
}
RELAX_PATTERN_PRINTER_DEF(ShapePatternObj, [](auto p, auto node) {
  p->stream << "ShapePattern(" << node->pattern << " has shape " << node->shape << ")";
});

SameShapeConstraint::SameShapeConstraint(ffi::Array<DFPattern> args)
    : DFConstraint(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<SameShapeConstraintObj> n = ffi::make_object<SameShapeConstraintObj>();
  n->args = std::move(args);
  data_ = std::move(n);

  if (auto ctx = PatternContext::Current()) {
    ctx.value().add_constraint(*this);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SameShapeConstraintObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.SameShapeConstraint",
                        [](ffi::Array<DFPattern> args) { return SameShapeConstraint(args); });
}
RELAX_PATTERN_PRINTER_DEF(SameShapeConstraintObj, [](auto p, auto node) {
  p->stream << "SameShapeConstraint(";
  for (size_t i = 0; i < node->args.size(); i++) {
    if (i) {
      p->stream << ", ";
    }
    p->stream << node->args;
  }
  p->stream << ")";
});

DataTypePattern::DataTypePattern(DFPattern pattern, DLDataType dtype)
    : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<DataTypePatternObj> n = ffi::make_object<DataTypePatternObj>(std::move(pattern));
  n->dtype = dtype;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  DataTypePatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.DataTypePattern", [](DFPattern pattern, DLDataType dtype) {
    return DataTypePattern(pattern, dtype);
  });
}
RELAX_PATTERN_PRINTER_DEF(DataTypePatternObj, [](auto p, auto node) {
  p->stream << "DataTypePattern(" << node->pattern << " has dtype " << node->dtype << ")";
});

AttrPattern::AttrPattern(DFPattern pattern, DictAttrs attrs) : DFPattern(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<AttrPatternObj> n = ffi::make_object<AttrPatternObj>(std::move(pattern));
  n->attrs = std::move(attrs);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AttrPatternObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.AttrPattern", [](DFPattern pattern, DictAttrs attrs) {
    return AttrPattern(pattern, attrs);
  });
}
RELAX_PATTERN_PRINTER_DEF(AttrPatternObj, [](auto p, auto node) {
  p->stream << "AttrPattern(" << node->pattern << " has attributes " << node->attrs << ")";
});

class DFPatternDuplicator : public DFPatternFunctor<DFPattern(const DFPattern&)> {
 public:
  DFPattern VisitDFPattern(const DFPattern& pattern) override {
    return DFPatternFunctor::VisitDFPattern(pattern);
  }
  DFPattern VisitDFPattern_(const OrPatternObj* op) override {
    return OrPattern(op->left, op->right);
  }
  DFPattern VisitDFPattern_(const AndPatternObj* op) override {
    return AndPattern(op->left, op->right);
  }
  DFPattern VisitDFPattern_(const NotPatternObj* op) override { return NotPattern(op->reject); }
  DFPattern VisitDFPattern_(const VarPatternObj* op) override { return VarPattern(op->name); }
  DFPattern VisitDFPattern_(const ConstantPatternObj* op) override {
    return ConstantPattern(ffi::UnsafeInit{}, ffi::make_object<ConstantPatternObj>());
  }
  DFPattern VisitDFPattern_(const WildcardPatternObj* op) override {
    return WildcardPattern(ffi::make_object<WildcardPatternObj>());
  }
  DFPattern VisitDFPattern_(const ExprPatternObj* op) override { return ExprPattern(op->expr); }
  DFPattern VisitDFPattern_(const GlobalVarPatternObj* op) override {
    return GlobalVarPattern(op->name);
  }
  DFPattern VisitDFPattern_(const TuplePatternObj* op) override { return TuplePattern(op->fields); }
  DFPattern VisitDFPattern_(const UnorderedTuplePatternObj* op) override {
    return UnorderedTuplePattern(op->fields);
  }
  DFPattern VisitDFPattern_(const TupleGetItemPatternObj* op) override {
    return TupleGetItemPattern(op->tuple, op->index);
  }
  DFPattern VisitDFPattern_(const CallPatternObj* op) override {
    return CallPattern(op->op, op->args);
  }
  DFPattern VisitDFPattern_(const DataTypePatternObj* op) override {
    return DataTypePattern(op->pattern, op->dtype);
  }
  DFPattern VisitDFPattern_(const FunctionPatternObj* op) override {
    return FunctionPattern(op->params, op->body);
  }
  DFPattern VisitDFPattern_(const ShapePatternObj* op) override {
    return ShapePattern(op->pattern, op->shape);
  }
  DFPattern VisitDFPattern_(const TypePatternObj* op) override {
    return TypePattern(op->pattern, op->ty);
  }

  DFPattern VisitDFPattern_(const DataflowVarPatternObj* op) override {
    return DataflowVarPattern(op->name);
  }
  DFPattern VisitDFPattern_(const ExternFuncPatternObj* op) override {
    return ExternFuncPattern(op->global_symbol());
  }
  DFPattern VisitDFPattern_(const PrimArrPatternObj* op) override {
    return PrimArrPattern(op->fields);
  }
};

// Syntatic Sugar
CallPattern DFPattern::operator()(const std::vector<DFPattern>& args) const {
  return CallPattern(*this, ffi::Array<DFPattern>(args));
}
OrPattern DFPattern::operator|(const DFPattern& other) const { return OrPattern(*this, other); }

AndPattern DFPattern::operator&(const DFPattern& other) const { return AndPattern(*this, other); }

NotPattern DFPattern::operator~() const { return NotPattern(*this); }

AttrPattern DFPattern::HasAttr(const ffi::Map<ffi::String, Any>& attrs) const {
  return AttrPattern(*this, DictAttrs(attrs));
}
TypePattern DFPattern::HasType(const Type& ty) const { return TypePattern(*this, ty); }
DataTypePattern DFPattern::HasDtype(DLDataType dtype) const {
  return DataTypePattern(*this, dtype);
}
DataTypePattern DFPattern::HasDtype(const std::string& dtype) const {
  return HasDtype(ffi::StringToDLDataType(dtype));
}
ShapePattern DFPattern::HasShape(const ffi::Array<PrimExpr>& shape) const {
  return ShapePattern(*this, shape);
}

DFPattern::operator PatternSeq() const { return PatternSeq{{*this}}; }

std::stack<PatternContext>& pattern_ctx_stack() {
  thread_local std::stack<PatternContext> graph_pattern_managers;
  return graph_pattern_managers;
}

ffi::Optional<PatternContext> PatternContext::Current() {
  if (pattern_ctx_stack().empty()) return std::nullopt;
  return pattern_ctx_stack().top();
}

PatternContext::PatternContext(bool incremental) {
  auto n = ffi::make_object<PatternContextObj>();
  if (incremental) {
    TVM_FFI_ICHECK(!pattern_ctx_stack().empty())
        << "Incremental context needs to be built inside a existing context.";
    n->allow_extern_use = pattern_ctx_stack().top()->allow_extern_use;
    n->edge_constraints = pattern_ctx_stack().top()->edge_constraints;
    n->src_ordered = pattern_ctx_stack().top()->src_ordered;
  }

  data_ = std::move(n);
}

void PatternContext::EnterWithScope() const { pattern_ctx_stack().push(*this); }

void PatternContext::ExitWithScope() const {
  TVM_FFI_ICHECK(pattern_ctx_stack().top().same_as(*this));
  pattern_ctx_stack().pop();
}

static void sync_graph_constraints(const DFPattern& lhs, const DFPattern& rhs, PairCons pcon) {
  if (auto ctx = PatternContext::Current()) {
    ctx.value().add_constraint(lhs, rhs, pcon);
  }
}

PatternSeq::PatternSeq(DFPattern init_pattern) {
  ffi::ObjectPtr<PatternSeqObj> n = ffi::make_object<PatternSeqObj>();
  n->patterns = {init_pattern};
  n->pair_constraints = {};
  data_ = std::move(n);
}
PatternSeq::PatternSeq(tvm::ffi::Array<DFPattern> patterns, bool only_used_by) {
  TVM_FFI_ICHECK_GE(patterns.size(), 1) << "PatternSeq must have at least one pattern";
  const auto cons = PairCons(only_used_by ? PairCons::kOnlyUsedBy : PairCons::kUsedBy);

  ffi::ObjectPtr<PatternSeqObj> n = ffi::make_object<PatternSeqObj>();
  n->patterns = std::move(patterns);
  n->pair_constraints = std::vector<PairCons>(n->patterns.size() - 1, cons);
  data_ = std::move(n);
}

PatternSeq PatternSeq::UsedBy(PatternSeq other, int index) const {
  return relax::UsedBy(*this, other, index);
}

PatternSeq PatternSeq::OnlyUsedBy(PatternSeq other, int index) const {
  return relax::OnlyUsedBy(*this, other, index);
}

PatternSeq PatternSeq::dup() const {
  PatternSeq ret(ffi::UnsafeInit{});

  ffi::ObjectPtr<PatternSeqObj> n = ffi::make_object<PatternSeqObj>();
  n->patterns = ffi::Array<DFPattern>{};
  n->patterns.reserve(get()->patterns.size());
  n->pair_constraints = this->get()->pair_constraints;

  for (size_t i = 0; i < get()->patterns.size(); ++i) {
    n->patterns.push_back(get()->patterns[i].dup());
    if (i >= 1)
      sync_graph_constraints(n->patterns[i - 1], n->patterns[i], n->pair_constraints[i - 1]);
  }

  ret.data_ = std::move(n);

  return ret;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  PatternSeqObj::RegisterReflection();
  refl::GlobalDef().def("relax.dpl.PatternSeq",
                        [](ffi::Array<DFPattern> patterns, bool only_used_by) {
                          return PatternSeq(std::move(patterns), only_used_by);
                        });
}
RELAX_PATTERN_PRINTER_DEF(PatternSeqObj, [](auto p, auto node) {
  p->stream << "[";
  for (size_t i = 0; i < node->patterns.size(); ++i) {
    if (i != 0)
      p->stream << (PairCons::kOnlyUsedBy == node->pair_constraints[i].type ? " >> " : " ^ ");
    p->stream << node->patterns[i];
  }
  p->stream << "]";
});

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.dpl.used_by",
           [](PatternSeq lhs, PatternSeq rhs, int index) { return lhs.UsedBy(rhs, index); })
      .def("relax.dpl.only_used_by",
           [](PatternSeq lhs, PatternSeq rhs, int index) { return lhs.OnlyUsedBy(rhs, index); });
}

PatternSeq UsedBy(const PatternSeq& lhs, const PatternSeq& rhs, int index) {
  PatternSeq ret(ffi::UnsafeInit{});

  const auto constraint = PairCons{PairCons::kOnlyUsedBy, index};

  sync_graph_constraints(lhs->patterns.back(), rhs->patterns.front(),
                         PairCons{PairCons::kUsedBy, index});

  ffi::Array<DFPattern> patterns;
  patterns.reserve(lhs->patterns.size() + rhs->patterns.size());
  patterns.insert(patterns.end(), lhs->patterns.begin(), lhs->patterns.end());
  patterns.insert(patterns.end(), rhs->patterns.begin(), rhs->patterns.end());

  std::vector<PairCons> pair_constraints = lhs->pair_constraints;
  pair_constraints.reserve(pair_constraints.size() + rhs->pair_constraints.size() + 1);
  pair_constraints.push_back(constraint);
  pair_constraints.insert(pair_constraints.end(), rhs->pair_constraints.begin(),
                          rhs->pair_constraints.end());

  ffi::ObjectPtr<PatternSeqObj> n = ffi::make_object<PatternSeqObj>();
  n->patterns = std::move(patterns);
  n->pair_constraints = std::move(pair_constraints);
  ret.data_ = std::move(n);

  return ret;
}
PatternSeq operator^(const PatternSeq& lhs, const PatternSeq& rhs) { return lhs.UsedBy(rhs); }

PatternSeq OnlyUsedBy(const PatternSeq& lhs, const PatternSeq& rhs, int index) {
  PatternSeq ret(ffi::UnsafeInit{});

  const auto constraint = PairCons{PairCons::kOnlyUsedBy, index};

  sync_graph_constraints(lhs->patterns.back(), rhs->patterns.front(), constraint);

  ffi::Array<DFPattern> patterns;
  patterns.reserve(lhs->patterns.size() + rhs->patterns.size());
  patterns.insert(patterns.end(), lhs->patterns.begin(), lhs->patterns.end());
  patterns.insert(patterns.end(), rhs->patterns.begin(), rhs->patterns.end());

  std::vector<PairCons> pair_constraints = lhs->pair_constraints;
  pair_constraints.reserve(pair_constraints.size() + rhs->pair_constraints.size() + 1);
  pair_constraints.push_back(constraint);
  pair_constraints.insert(pair_constraints.end(), rhs->pair_constraints.begin(),
                          rhs->pair_constraints.end());

  ffi::ObjectPtr<PatternSeqObj> n = ffi::make_object<PatternSeqObj>();
  n->patterns = std::move(patterns);
  n->pair_constraints = std::move(pair_constraints);
  ret.data_ = std::move(n);

  return ret;
}
PatternSeq operator>>(const PatternSeq& lhs, const PatternSeq& rhs) { return lhs.OnlyUsedBy(rhs); }

VarPattern IsVar(const ffi::String& name) { return VarPattern(name); }
ConstantPattern IsConst() {
  return ConstantPattern(ffi::UnsafeInit{}, ffi::make_object<ConstantPatternObj>());
}
WildcardPattern Wildcard() { return WildcardPattern(ffi::make_object<WildcardPatternObj>()); }
ExprPattern IsExpr(const Expr& expr) { return ExprPattern(expr); }
ExprPattern IsOp(const ffi::String& op_name) { return IsExpr(Op::Get(op_name)); }
CallPattern IsCallTIR(const ffi::String& name, ffi::Optional<TuplePattern> var_args) {
  DFPattern arg_pattern =
      var_args.has_value() ? DFPattern(var_args.value()) : DFPattern(Wildcard());

  return IsOp("relax.call_tir")(GlobalVarPattern(name), arg_pattern);
}

CallPattern IsCallTIR(const ffi::String& name, TuplePattern var_args) {
  return IsOp("relax.call_tir")(GlobalVarPattern(name), var_args);
}
CallPattern IsCallDPSPacked(const ffi::String& name, ffi::Optional<TuplePattern> var_args) {
  DFPattern arg_pattern =
      var_args.has_value() ? DFPattern(var_args.value()) : DFPattern(Wildcard());

  return IsOp("relax.call_dps_packed")(GlobalVarPattern(name), arg_pattern);
}

CallPattern IsCallDPSPacked(const ffi::String& name, TuplePattern var_args) {
  return IsOp("relax.call_dps_packed")(GlobalVarPattern(name), var_args);
}

DFPattern IsTuple(const ffi::Array<DFPattern>& fields, bool unordered) {
  if (unordered)
    return UnorderedTuplePattern(fields);
  else
    return TuplePattern(fields);
}
TupleGetItemPattern IsTupleGetItem(const DFPattern tuple, int index) {
  return TupleGetItemPattern(tuple, index);
}

DFPattern DFPattern::dup() const {
  auto pattern = DFPatternDuplicator().VisitDFPattern(*this);
  return pattern;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.dpl.dup_pattern", [](DFPattern pattern) { return pattern.dup(); })
      .def("relax.dpl.dup_seq", [](PatternSeq seq) { return seq.dup(); })
      .def("relax.dpl.PatternContext", [](bool incre) { return PatternContext(incre); })
      .def("relax.dpl.current_context", [] { return PatternContext::Current(); })
      .def("relax.dpl.enter_context", [](const PatternContext& ctx) { ctx.EnterWithScope(); })
      .def("relax.dpl.exit_context", [](const PatternContext& ctx) { ctx.ExitWithScope(); });
}

}  // namespace relax
}  // namespace tvm
