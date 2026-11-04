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
 * \file tvm/relax/dataflow_pattern.h
 * \brief A pattern language for matching dataflow properties.
 */
#ifndef TVM_RELAX_DATAFLOW_PATTERN_H_
#define TVM_RELAX_DATAFLOW_PATTERN_H_

#include <tvm/base.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/with_context.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tvm {

namespace sym {
class AnalyzerObj;
class Analyzer;
}  // namespace sym

namespace relax {

class PatternSeq;
class CallPattern;
class OrPattern;
class AndPattern;
class NotPattern;
class ShapePattern;
class TypePattern;
class DataTypePattern;
class AttrPattern;
class SameShapeConstraint;

/*!
 * \brief Create used-by relationship between lhs[-1] and rhs[0], with [*lhs, *rhs] returned.
 *
 * \param lhs Left hand side of the used-by relationship.
 * \param rhs Right hand side of the used-by relationship.
 * \param index lhs[-1] is used as the index'th argument of rhs[0].
 * \return PatternSeq The concatenated sequence of [*lhs, *rhs].
 */
TVM_DLL PatternSeq UsedBy(const PatternSeq& lhs, const PatternSeq& rhs, int index = -1);
/*! \brief Syntax sugar of UsedBy(lhs, rhs, -1). */
TVM_DLL PatternSeq operator^(const PatternSeq& lhs, const PatternSeq& rhs);

/*!
 * \brief Create only-used-by relationship between lhs[-1] and rhs[0], with [*lhs, *rhs] returned.
 *
 * \param lhs Left hand side of the used-by relationship.
 * \param rhs Right hand side of the used-by relationship.
 * \param index lhs[-1] is used as the index'th argument of rhs[0].
 * \return PatternSeq The concatenated sequence of [*lhs, *rhs].
 */
TVM_DLL PatternSeq OnlyUsedBy(const PatternSeq& lhs, const PatternSeq& rhs, int index = -1);
/*! \brief Syntax sugar of OnlyUsedBy(lhs, rhs, -1). */
TVM_DLL PatternSeq operator>>(const PatternSeq& lhs, const PatternSeq& rhs);

/*!
 * \brief Base type of all dataflow patterns.
 * \sa DFPattern
 */
class DFPatternObj : public ffi::Object {
 public:
  static constexpr const uint32_t _type_child_slots = 21;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.dpl.DFPattern", DFPatternObj, ffi::Object);
};

/*!
 * \brief Managed reference to dataflow patterns.
 * \sa DFPatternObj
 */
class DFPattern : public ffi::ObjectRef {
 public:
  /*! \brief Syntatic Sugar for creating a CallPattern */
  template <typename... Args>
  CallPattern operator()(Args&&... args) const;
  /*! \brief Syntatic Sugar for creating a CallPattern */
  TVM_DLL CallPattern operator()(const std::vector<DFPattern>& args) const;
  /*! \brief Syntatic Sugar for creating an OrPattern */
  TVM_DLL OrPattern operator|(const DFPattern& other) const;
  /*! \brief Syntatic Sugar for creating an AndPattern */
  TVM_DLL AndPattern operator&(const DFPattern& other) const;
  /*! \brief Syntatic Sugar for creating a NotPattern */
  TVM_DLL NotPattern operator~() const;
  /*! \brief Syntatic Sugar for creating an AttrPattern */
  TVM_DLL AttrPattern HasAttr(const ffi::Map<ffi::String, Any>& attrs) const;
  /*! \brief Syntatic Sugar for creating a TypePattern */
  TVM_DLL TypePattern HasType(const Type& ty) const;
  /*! \brief Syntatic Sugar for creating a DataTypePattern with a dtype */
  TVM_DLL DataTypePattern HasDtype(DLDataType dtype) const;
  /*! \brief Syntatic Sugar for creating a DataTypePattern with a data type's name */
  TVM_DLL DataTypePattern HasDtype(const std::string& dtype) const;
  /*! \brief Syntatic Sugar for creating a ShapePattern */
  TVM_DLL ShapePattern HasShape(const ffi::Array<PrimExpr>& shape) const;
  /*! \brief Syntatic Sugar for creating a ShapePattern */
  TVM_DLL SameShapeConstraint HasSameShapeAs(const DFPattern& other) const;
  /*! \brief Syntatic Sugar for duplicating the current pattern */
  TVM_DLL DFPattern dup() const;

  /*! \brief Implicit conversion from DFPattern to PatternSeq */
  TVM_DLL operator PatternSeq() const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DFPattern, ffi::ObjectRef, DFPatternObj);
};

/*! \brief Constraint of a DFPattern edge (producer -> consumer) in graph-level matching */
struct PairCons {
  /*! \brief Constraint types of the edge */
  enum Type {
    kUsedBy,     /*!< producer ^ consumer */
    kOnlyUsedBy, /*!< producer >> consumer */
  } type = kUsedBy;
  int index = -1; /*!< The argument index of the producer in the consumer caller site */

  /*!
   * \brief Construct a new PairCons object
   *
   * \param t The constraint type
   * \param index The producer is called as the index'th argument of the consumer function.
   */
  TVM_DLL explicit PairCons(Type t, int index = -1) : type(t), index(index) {}

  bool operator==(const PairCons& other) const {
    return type == other.type && index == other.index;
  }
};

/*! \brief Additional constraints on the graph
 *
 * Unlike PairCons, these may relate nodes that are not directly
 * connected by a DFPattern edge from producer to consumer.  For
 * example, constraining the two branches of an elementwise operation
 * to have the same shape.
 */
class DFConstraintObj : public ffi::Object {
 public:
  /*! \brief Return the patterns on which the constraint depends */
  virtual ffi::Array<DFPattern> GetDependentPatterns() const = 0;

  /*! \brief Convert the constraint to a PrimExpr
   *
   * If the returned boolean parameter is true, then the returned
   * expression is a necessary-and-sufficient condition for evaluating
   * the constraint.  In this case, the matcher may either mark the
   * constraint as satisfied (no need to re-check later), or as failed
   * (need to back-track).
   *
   * If the returned boolean parameter is false, then the returned
   * expression is a necessary-but-not-sufficient condition for
   * evaluating the constraint.  In this case, the matcher may start
   * backtracking as a result of a failed condition, but may not mark
   * the constraint as satisfied.  This typically occurs when the
   * constraint involves a parameter that the matcher has not yet
   * filled.
   *
   * \param match_state A function that can be called to check the
   *    current state of the match.  The function takes as argument a
   *    pattern on which the constraint depends, and returns the relax
   *    variable matched by that pattern, or std::nullopt if the pattern
   *    has not yet been matched.
   *
   * \return A tuple of `PrimExpr` and `bool`.  The first element is a
   *    necessary condition for the constraint to be satisfied.  The
   *    second tuple element indicates whether the condition is also
   *    sufficient for the constraint to be satisfied.
   */
  virtual std::tuple<PrimExpr, bool> AsCondition(
      std::function<ffi::Optional<Var>(const DFPatternObj*)> match_state) const = 0;

  static constexpr const uint32_t _type_child_slots = 1;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.dpl.DFConstraint", DFConstraintObj, ffi::Object);
};

class DFConstraint : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DFConstraint, ffi::ObjectRef, DFConstraintObj);
};

/*!
 * \brief A sequence of DFPatterns that the previous DFPattern is connected to the next one.
 * \sa PatternSeq
 */
class PatternSeqObj final : public ffi::Object {
 public:
  tvm::ffi::Array<DFPattern> patterns;    /*!< The sequence of DFPatterns */
  std::vector<PairCons> pair_constraints; /*!< Constraints between the previous and next patterns */

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PatternSeqObj>().def_ro("patterns", &PatternSeqObj::patterns);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("relax.dpl.PatternSeq", PatternSeqObj, ffi::Object);
};

/*!
 * \brief Managed reference to pattern sequences.
 * \sa PatternSeqObj
 */
class PatternSeq final : public ffi::ObjectRef {
 public:
  TVM_DLL explicit PatternSeq(DFPattern init_pattern);
  TVM_DLL explicit PatternSeq(tvm::ffi::Array<DFPattern> patterns, bool only_used_by = false);

  PatternSeq UsedBy(PatternSeq other, int index = -1) const;
  PatternSeq OnlyUsedBy(PatternSeq other, int index = -1) const;

  /*! \brief Syntatic Sugar for duplicating the current pattern sequence */
  PatternSeq dup() const;

  // friend functions
  friend PatternSeq UsedBy(const PatternSeq& lhs, const PatternSeq& rhs, int index);
  friend PatternSeq OnlyUsedBy(const PatternSeq& lhs, const PatternSeq& rhs, int index);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PatternSeq, ffi::ObjectRef, PatternSeqObj);
};

/*!
 * \brief A context to manage the graph-level pattern matching.
 * \sa PatternContext
 */
class PatternContextObj : public ffi::Object {
 public:
  /*! \brief Constrainting matched graph with assertion to external uses */
  enum ExternUse {
    kMay,     /*!< No constraints */
    kMustNot, /*!< All nodes except outputs only have internal depedencies in the matched graph. */
  } allow_extern_use = kMay;

  // src node -> <dst node, constraint type> constraints.
  // Dst nodes are kept in a vector to keep them ordered.
  std::map<DFPattern, std::vector<std::pair<DFPattern, std::vector<PairCons>>>> edge_constraints;

  // Underlying DFPattern nodes which the edge constraints may reference
  // Kept as a separate vector of patterns to process constraints in a fixed order.
  std::vector<DFPattern> src_ordered;

  // Non-edge constraints
  std::vector<DFConstraint> validation_constraints;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.PatternContext", PatternContextObj, ffi::Object);
};

/*!
 * \brief Managed reference to a pattern context.
 * \sa PatternContextObj
 */
class PatternContext : public ffi::ObjectRef {
 public:
  explicit PatternContext(ffi::UnsafeInit tag) : ffi::ObjectRef(tag) {}
  static constexpr bool _type_is_nullable = false;
  TVM_DLL explicit PatternContext(bool incremental = false);

  const PatternContextObj* operator->() const {
    TVM_FFI_ICHECK(get() != nullptr);
    return static_cast<const PatternContextObj*>(get());
  }

  PatternContextObj* operator->() {
    TVM_FFI_ICHECK(get() != nullptr);
    return static_cast<PatternContextObj*>(get_mutable());
  }

  /*!
   * \brief Build an edge constraint between two patterns (producer and consumer).
   *
   * \param producer The pattern corresponding to the producer node.
   * \param consumer The pattern corresponding to the consumer node.
   * \param cons The constraint type. \sa PairCons
   */
  void add_constraint(DFPattern producer, DFPattern consumer, PairCons cons) {
    auto& pairs = (*this)->edge_constraints[producer];
    auto it = std::find_if(pairs.begin(), pairs.end(),
                           [consumer](auto p) { return p.first == consumer; });
    if (it == pairs.end()) {
      pairs.emplace_back(consumer, std::vector{cons});
    } else {
      auto& vec = it->second;
      TVM_FFI_ICHECK(std::find(vec.cbegin(), vec.cend(), cons) == vec.cend())
          << "Constraint already exists";
      vec.push_back(cons);
    }

    auto& patterns = (*this)->src_ordered;
    if (std::find(patterns.begin(), patterns.end(), producer) == patterns.end()) {
      patterns.push_back(producer);
    }
  }

  /*!
   * \brief Add a validation constraint
   *
   * \param constraint The new constraint
   */
  void add_constraint(DFConstraint constraint) {
    (*this)->validation_constraints.push_back(constraint);
  }

  /*! \brief Get the constraint context object on the top of the stack */
  TVM_DLL static ffi::Optional<PatternContext> Current();

  /*! \brief The RAII-like entry of a constraint context scope */
  TVM_DLL void EnterWithScope() const;
  /*! \brief The RAII-like exit of a constraint context scope */
  TVM_DLL void ExitWithScope() const;

 private:
  friend class With<PatternContext>;
};

/*!
 * \brief Pattern for Relax Expression.
 * \sa ExprPattern
 */
class ExprPatternObj : public DFPatternObj {
 public:
  explicit ExprPatternObj(Expr expr) : expr(std::move(expr)) {}
  explicit ExprPatternObj(ffi::UnsafeInit) : expr(ffi::UnsafeInit{}) {}

  Expr expr; /*!< The expression to match */

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExprPatternObj>().def_ro("expr", &ExprPatternObj::expr);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.ExprPattern", ExprPatternObj, DFPatternObj);
};

/*!
 * \brief Managed reference to an ExprPattern.
 * \sa ExprPatternObj
 */
class ExprPattern : public DFPattern {
 public:
  TVM_DLL explicit ExprPattern(Expr expr);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExprPattern, DFPattern, ExprPatternObj);
};

/*!
 * \brief A Pattern to Match a Relax Variable.
 * \note The name field matches any string if it is empty.
 * \sa VarPattern
 */
class VarPatternObj : public DFPatternObj {
 public:
  ffi::String name;
  const ffi::String& name_hint() const { return name; }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VarPatternObj>().def_ro("name", &VarPatternObj::name);
  }

  static constexpr const uint32_t _type_child_slots = 1;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.dpl.VarPattern", VarPatternObj, DFPatternObj);
};

/*!
 * \brief Managed reference to a VarPattern.
 * \sa VarPatternObj
 */
class VarPattern : public DFPattern {
 public:
  /*!
   * \brief Create a pattern matching by variable name.
   *
   * \param name_hint Variable name to match. Any if empty ("").
   */
  TVM_DLL VarPattern(ffi::String name_hint);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VarPattern, DFPattern, VarPatternObj);
};

/*!
 * \brief A Pattern to Match a Relax Dataflow Variable
 * \sa DataflowVarPattern
 */
class DataflowVarPatternObj : public VarPatternObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DataflowVarPatternObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.DataflowVarPattern", DataflowVarPatternObj,
                                    VarPatternObj);
};

/*!
 * \brief Managed reference to a DataflowVarPattern.
 * \sa DataflowVarPatternObj
 */
class DataflowVarPattern : public DFPattern {
 public:
  /*! \sa VarPattern::VarPattern */
  TVM_DLL DataflowVarPattern(ffi::String name_hint);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DataflowVarPattern, DFPattern,
                                                DataflowVarPatternObj);
};

/*!
 * \brief A Pattern to Match a Relax Global Variable
 * \sa GlobalVarPattern
 */
class GlobalVarPatternObj : public VarPatternObj {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.GlobalVarPattern", GlobalVarPatternObj,
                                    DFPatternObj);
};

/*!
 * \brief Managed reference to a GlobalVarPattern.
 * \sa GlobalVarPatternObj
 */
class GlobalVarPattern : public DFPattern {
 public:
  TVM_DLL GlobalVarPattern(ffi::String name_hint);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GlobalVarPattern, DFPattern, GlobalVarPatternObj);
};

/*!
 * \brief A Pattern to Match a Relax GenericConst.
 * \sa ConstantPattern
 */
class ConstantPatternObj : public DFPatternObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ConstantPatternObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.ConstantPattern", ConstantPatternObj, DFPatternObj);
};

/*!
 * \brief Managed reference to a ConstantPattern.
 * \sa ConstantPatternObj
 */
class ConstantPattern : public DFPattern {
 public:
  explicit ConstantPattern(ffi::UnsafeInit tag, ffi::ObjectPtr<ConstantPatternObj> n)
      : DFPattern(tag) {
    data_ = std::move(n);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ConstantPattern, DFPattern, ConstantPatternObj);
};

/*!
 * \brief A pattern to match a callable node in Relax.
 * \sa CallPattern
 */
class CallPatternObj : public DFPatternObj {
 public:
  explicit CallPatternObj(DFPattern op) : op(std::move(op)) {}
  explicit CallPatternObj(ffi::UnsafeInit) : op(ffi::UnsafeInit{}) {}

  /*!
   * \note The op field can be:
   *  - relax::Op which corresponds to the primitive operators.
   *  - user defined functions (Function, GlobalVar, Var).
   */
  DFPattern op;                    /*!< The operator (function) being invoked */
  tvm::ffi::Array<DFPattern> args; /*!< The arguments of the function call */
  /*!
   * \note If varg_default_wildcard is true. Given args of [pA, pB], when matching a call whose
   * arguments are [A, B, ...], the pattern will still match despite N(args) < N(call.args). That
   * said, with varg_default_wildcard set to true, we match the args in the order we have, and
   * regard the rest of the arguments as wildcards.
   */
  bool varg_default_wildcard; /*!< N(args) can be < N(real args) by the padding of Wildcard */

  // Todo(relax-team): Dataflow pattern for Type, and match ty_args

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CallPatternObj>()
        .def_ro("op", &CallPatternObj::op)
        .def_ro("args", &CallPatternObj::args);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.CallPattern", CallPatternObj, DFPatternObj);
};

class CallPattern : public DFPattern {
 public:
  TVM_DLL CallPattern(DFPattern op, ffi::Array<DFPattern> args, bool varg_default_wildcard = false);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CallPattern, DFPattern, CallPatternObj);
};

/*!
 * \brief A pattern to match an array of PrimExpr.
 * \sa PrimArrPattern
 * \note This is often used to match shapes specified as arguments to a function.
 */
class PrimArrPatternObj : public DFPatternObj {
 public:
  ffi::Array<PrimExpr> fields; /*!< The array to match */

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PrimArrPatternObj>().def_ro("fields", &PrimArrPatternObj::fields);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.PrimArrPattern", PrimArrPatternObj, DFPatternObj);
};

/*!
 * \brief Managed reference to a PrimArrPattern.
 * \sa PrimArrPatternObj
 */
class PrimArrPattern : public DFPattern {
 public:
  TVM_DLL PrimArrPattern(ffi::Array<PrimExpr> arr);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimArrPattern, DFPattern, PrimArrPatternObj);
};

/*!
 * \brief A pattern to match a Relax Function
 * \sa Function
 * \sa FunctionPattern
 */
class FunctionPatternObj : public DFPatternObj {
 public:
  explicit FunctionPatternObj(DFPattern body) : body(std::move(body)) {}
  explicit FunctionPatternObj(ffi::UnsafeInit) : body(ffi::UnsafeInit{}) {}

  tvm::ffi::Array<DFPattern> params; /*!< The parameters of the function */
  /*!
   * \note Note that in Relax, the function body is a SeqExpr which contains
   * 1) SeqExprObj::blocks, which is a list of blocks of statements; and 2)
   * SeqExprObj::body, which is an Expr that can be anything. FunctionPattern
   * only matches the body of the function (writing patterns to statements is tricky).
   */
  DFPattern body; /*!< The body of the function */

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FunctionPatternObj>()
        .def_ro("params", &FunctionPatternObj::params)
        .def_ro("body", &FunctionPatternObj::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.FunctionPattern", FunctionPatternObj, DFPatternObj);
};

/*!
 * \brief Managed reference to FunctionPatternObj.
 * \sa FunctionPatternObj
 */
class FunctionPattern : public DFPattern {
 public:
  /*!
   * \brief Constructor
   * \param params The parameters of the function.
   * \param body The body of the function.
   */
  TVM_DLL FunctionPattern(tvm::ffi::Array<DFPattern> params, DFPattern body);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FunctionPattern, DFPattern, FunctionPatternObj);
};

/*!
 * \brief Pattern to match a tuple of ordered expressions.
 * \sa TuplePattern
 */
class TuplePatternObj : public DFPatternObj {
 public:
  tvm::ffi::Array<DFPattern> fields; /*!< The fields of the tuple */

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TuplePatternObj>().def_ro("fields", &TuplePatternObj::fields);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.TuplePattern", TuplePatternObj, DFPatternObj);
};

/*!
 * \brief Managed reference to TuplePatternObj.
 * \sa TuplePatternObj
 */
class TuplePattern : public DFPattern {
 public:
  TVM_DLL explicit TuplePattern(tvm::ffi::Array<DFPattern> fields);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TuplePattern, DFPattern, TuplePatternObj);
};

/*!
 * \brief A pattern to match multiple expressions unorderedly.
 * \sa UnorderedTuplePattern
 */
class UnorderedTuplePatternObj : public DFPatternObj {
 public:
  tvm::ffi::Array<DFPattern> fields; /*!< The fields of the tuple */

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<UnorderedTuplePatternObj>().def_ro("fields", &UnorderedTuplePatternObj::fields);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.UnorderedTuplePattern", UnorderedTuplePatternObj,
                                    DFPatternObj);
};

/*!
 * \brief Managed reference to UnorderedTuplePatternObj.
 * \sa UnorderedTuplePatternObj
 */
class UnorderedTuplePattern : public DFPattern {
 public:
  TVM_DLL explicit UnorderedTuplePattern(tvm::ffi::Array<DFPattern> fields);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(UnorderedTuplePattern, DFPattern,
                                                UnorderedTuplePatternObj);
};

/*!
 * \brief A pattern to match n'th indexing to a tuple.
 * \sa TupleGetItem
 * \sa TupleGetItemPattern
 */
class TupleGetItemPatternObj : public DFPatternObj {
 public:
  explicit TupleGetItemPatternObj(DFPattern tuple) : tuple(std::move(tuple)) {}
  explicit TupleGetItemPatternObj(ffi::UnsafeInit) : tuple(ffi::UnsafeInit{}) {}

  DFPattern tuple; /*!< The tuple Expression */
  int index;       /*!< The index of the tuple with -1 meaning arbitrary */

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TupleGetItemPatternObj>()
        .def_ro("tuple", &TupleGetItemPatternObj::tuple)
        .def_ro("index", &TupleGetItemPatternObj::index);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.TupleGetItemPattern", TupleGetItemPatternObj,
                                    DFPatternObj);
};

/*!
 * \brief Managed reference to TupleGetItemPatternObj.
 * \sa TupleGetItemPatternObj
 */
class TupleGetItemPattern : public DFPattern {
 public:
  TVM_DLL TupleGetItemPattern(DFPattern tuple, int index);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TupleGetItemPattern, DFPattern,
                                                TupleGetItemPatternObj);
};

/*!
 * \brief Match a conjunction of other patterns.
 * \sa AndPattern
 */
class AndPatternObj : public DFPatternObj {
 public:
  explicit AndPatternObj(DFPattern left, DFPattern right)
      : left(std::move(left)), right(std::move(right)) {}
  explicit AndPatternObj(ffi::UnsafeInit) : left(ffi::UnsafeInit{}), right(ffi::UnsafeInit{}) {}

  DFPattern left;  /*!< The left hand side of the conjunction */
  DFPattern right; /*!< The right hand side of the conjunction */

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AndPatternObj>()
        .def_ro("left", &AndPatternObj::left)
        .def_ro("right", &AndPatternObj::right);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.AndPattern", AndPatternObj, DFPatternObj);
};

/*!
 * \brief Managed reference to AndPatternObj.
 * \sa AndPatternObj
 */
class AndPattern : public DFPattern {
 public:
  TVM_DLL AndPattern(DFPattern lhs, DFPattern rhs);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AndPattern, DFPattern, AndPatternObj);
};

/*!
 * \brief Match a disjunction of other patterns.
 * \sa OrPattern
 */
class OrPatternObj : public DFPatternObj {
 public:
  explicit OrPatternObj(DFPattern left, DFPattern right)
      : left(std::move(left)), right(std::move(right)) {}
  explicit OrPatternObj(ffi::UnsafeInit) : left(ffi::UnsafeInit{}), right(ffi::UnsafeInit{}) {}

  DFPattern left;  /*!< The left hand side of the disjunction */
  DFPattern right; /*!< The right hand side of the disjunction */

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OrPatternObj>()
        .def_ro("left", &OrPatternObj::left)
        .def_ro("right", &OrPatternObj::right);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.OrPattern", OrPatternObj, DFPatternObj);
};

/*!
 * \brief Managed reference to OrPatternObj.
 * \sa OrPatternObj
 */
class OrPattern : public DFPattern {
 public:
  TVM_DLL OrPattern(DFPattern left, DFPattern right);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(OrPattern, DFPattern, OrPatternObj);
};

/*!
 * \brief Pattern for rejecting a certain pattern.
 * \sa NotPattern
 */
class NotPatternObj : public DFPatternObj {
 public:
  explicit NotPatternObj(DFPattern reject) : reject(std::move(reject)) {}
  explicit NotPatternObj(ffi::UnsafeInit) : reject(ffi::UnsafeInit{}) {}

  DFPattern reject; /*!< The pattern to reject */

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<NotPatternObj>().def_ro("reject", &NotPatternObj::reject);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.NotPattern", NotPatternObj, DFPatternObj);
};

/*!
 * \brief Managed reference to NotPatternObj.
 * \sa NotPatternObj
 */
class NotPattern : public DFPattern {
 public:
  TVM_DLL NotPattern(DFPattern reject);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(NotPattern, DFPattern, NotPatternObj);
};

/*!
 * \brief Wildcard Pattern is a pattern that can match anything.
 * \sa WildcardPattern
 */
class WildcardPatternObj : public DFPatternObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<WildcardPatternObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.WildcardPattern", WildcardPatternObj, DFPatternObj);
};

/*!
 * \brief Managed reference to WildcardPatternObj.
 * \sa WildcardPatternObj
 */
class WildcardPattern : public DFPattern {
 public:
  WildcardPattern();
  explicit WildcardPattern(ffi::ObjectPtr<WildcardPatternObj> data) : DFPattern(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }

  // Declaring WildcardPattern declared as non-nullable avoids the
  // default zero-parameter constructor for ffi::ObjectRef with `data_ =
  // nullptr`.  This allows a zero-parameter constructor to be
  // declared here, to create a valid wildcard instance.

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(WildcardPattern, DFPattern, WildcardPatternObj);
};

/*!
 * \brief Pattern for matching a certain type.
 * \sa TypePattern
 */
class TypePatternObj : public DFPatternObj {
 public:
  explicit TypePatternObj(DFPattern pattern) : pattern(std::move(pattern)) {}
  explicit TypePatternObj(ffi::UnsafeInit) : pattern(ffi::UnsafeInit{}) {}

  DFPattern pattern;         /*!< The pattern to match */
  Type ty = Type::Missing(); /*!< The type to match */

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TypePatternObj>()
        .def_ro("pattern", &TypePatternObj::pattern)
        .def_ro("ty", &TypePatternObj::ty);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.TypePattern", TypePatternObj, DFPatternObj);
};

class TypePattern : public DFPattern {
 public:
  TVM_DLL TypePattern(DFPattern pattern, Type ty);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TypePattern, DFPattern, TypePatternObj);
};

/*!
 * \brief A pattern that asserting a root pattern has a certain shape.
 * \sa ShapePattern
 */
class ShapePatternObj : public DFPatternObj {
 public:
  explicit ShapePatternObj(DFPattern pattern) : pattern(std::move(pattern)) {}
  explicit ShapePatternObj(ffi::UnsafeInit) : pattern(ffi::UnsafeInit{}) {}

  DFPattern pattern;          /*!< The root pattern to match */
  ffi::Array<PrimExpr> shape; /*!< The shape to match */

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ShapePatternObj>()
        .def_ro("pattern", &ShapePatternObj::pattern)
        .def_ro("shape", &ShapePatternObj::shape);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.ShapePattern", ShapePatternObj, DFPatternObj);
};

/*!
 * \brief Managed reference to ShapePatternObj.
 * \sa ShapePatternObj
 */
class ShapePattern : public DFPattern {
 public:
  TVM_DLL ShapePattern(DFPattern pattern, ffi::Array<PrimExpr> type);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ShapePattern, DFPattern, ShapePatternObj);
};

/*!
 * \brief A pattern that asserting multiple root patterns have the same shape
 * \sa SameShapePattern
 */
class SameShapeConstraintObj : public DFConstraintObj {
 public:
  ffi::Array<DFPattern> args; /*!< The patterns with matching shapes */

  ffi::Array<DFPattern> GetDependentPatterns() const override { return args; }

  std::tuple<PrimExpr, bool> AsCondition(
      std::function<ffi::Optional<Var>(const DFPatternObj*)> match_state) const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SameShapeConstraintObj>().def_ro("args", &SameShapeConstraintObj::args);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.SameShapeConstraint", SameShapeConstraintObj,
                                    DFConstraintObj);
};

/*!
 * \brief Managed reference to SameShapePatternNode.
 * \sa SameShapePatternNode
 */
class SameShapeConstraint : public DFConstraint {
 public:
  TVM_DLL SameShapeConstraint(ffi::Array<DFPattern> args);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SameShapeConstraint, DFConstraint,
                                                SameShapeConstraintObj);
};

/*!
 * \brief A pattern that asserting a root pattern has a certain data type.
 * \sa DataTypePattern
 */
class DataTypePatternObj : public DFPatternObj {
 public:
  explicit DataTypePatternObj(DFPattern pattern) : pattern(std::move(pattern)) {}
  explicit DataTypePatternObj(ffi::UnsafeInit) : pattern(ffi::UnsafeInit{}) {}

  DFPattern pattern; /*!< The root pattern to match */
  DLDataType dtype;  /*!< The data type to match */

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DataTypePatternObj>()
        .def_ro("pattern", &DataTypePatternObj::pattern)
        .def_ro("dtype", &DataTypePatternObj::dtype);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.DataTypePattern", DataTypePatternObj, DFPatternObj);
};

/*!
 * \brief Managed reference to DataTypePatternObj.
 * \sa DataTypePatternObj
 */
class DataTypePattern : public DFPattern {
 public:
  TVM_DLL DataTypePattern(DFPattern pattern, DLDataType dtype);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DataTypePattern, DFPattern, DataTypePatternObj);
};

/*!
 * \brief A pattern that asserting a root pattern has certain attributes.
 * \sa AttrPattern
 */
class AttrPatternObj : public DFPatternObj {
 public:
  explicit AttrPatternObj(DFPattern pattern) : pattern(std::move(pattern)) {}
  explicit AttrPatternObj(ffi::UnsafeInit) : pattern(ffi::UnsafeInit{}) {}

  DFPattern pattern; /*!< The root pattern to match */
  DictAttrs attrs;   /*!< The attributes (a map/dictionary) to match */

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AttrPatternObj>()
        .def_ro("pattern", &AttrPatternObj::pattern)
        .def_ro("attrs", &AttrPatternObj::attrs);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.AttrPattern", AttrPatternObj, DFPatternObj);
};

/*!
 * \brief Managed reference to AttrPatternObj.
 * \sa AttrPatternObj
 */
class AttrPattern : public DFPattern {
 public:
  TVM_DLL AttrPattern(DFPattern pattern, DictAttrs attrs);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AttrPattern, DFPattern, AttrPatternObj);
};

/*!
 * \brief A pattern of external function.
 * \sa ExternFunc
 * \sa ExternFuncPattern
 */
class ExternFuncPatternObj : public DFPatternObj {
 public:
  ffi::String global_symbol_; /*!< The global symbol name of the external function */

  /*! \brief The external function name */
  const ffi::String& global_symbol() const { return global_symbol_; }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExternFuncPatternObj>().def_ro("global_symbol",
                                                   &ExternFuncPatternObj::global_symbol_);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.ExternFuncPattern", ExternFuncPatternObj,
                                    DFPatternObj);
};

/*!
 * \brief Managed reference to ExternFuncPatternObj.
 * \sa ExternFuncPatternObj
 */
class ExternFuncPattern : public DFPattern {
 public:
  TVM_DLL ExternFuncPattern(ffi::String global_symbol);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExternFuncPattern, DFPattern, ExternFuncPatternObj);
};

/*! \brief Syntatic Sugar for creating a VarPattern with a name */
VarPattern IsVar(const ffi::String& name);
/*! \brief Syntatic Sugar for creating a ConstantPattern */
ConstantPattern IsConst();
/*! \brief Syntatic Sugar for creating a WildcardPattern */
WildcardPattern Wildcard();
/*! \brief Syntatic Sugar for creating a ExprPattern */
ExprPattern IsExpr(const Expr& expr);
/*! \brief Syntatic Sugar for creating a ExprPattern base on an Op */
ExprPattern IsOp(const ffi::String& op_name);
/*! \brief Syntatic Sugar for call_tir (return a tensor) */
// Todo(relax-team): Dataflow pattern for Type, and match out_ty
CallPattern IsCallTIR(const ffi::String& name, ffi::Optional<TuplePattern> args = std::nullopt);
/*! \brief Syntatic Sugar for call_tir (return a tuple of tensor) */
CallPattern IsCallTIR(const ffi::String& name, TuplePattern var_args);
/*! \brief Syntatic Sugar for call_dps_packed (return a tensor) */
CallPattern IsCallDPSPacked(const ffi::String& name,
                            ffi::Optional<TuplePattern> args = std::nullopt);
/*! \brief Syntatic Sugar for call_dps_packed (return a tuple of tensor) */
CallPattern IsCallDPSPacked(const ffi::String& name, TuplePattern var_args);
/*! \brief Syntatic Sugar for creating TuplePattern or UnorderedTuplePattern (unordered=true) */
DFPattern IsTuple(const ffi::Array<DFPattern>& fields, bool unordered = false);
/*! \brief Syntatic Sugar for creating a TupleGetItemPattern */
TupleGetItemPattern IsTupleGetItem(const DFPattern tuple, int index = -1);

/*! \brief Implementation of the templated CallPattern syntax sugar */
template <typename... Args>
CallPattern DFPattern::operator()(Args&&... args) const {
  return CallPattern(ffi::GetRef<DFPattern>(this->get()),
                     ffi::Array<DFPattern>({std::forward<Args>(args)...}));
}

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_DATAFLOW_PATTERN_H_
