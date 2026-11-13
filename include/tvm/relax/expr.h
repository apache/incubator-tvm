/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_RELAX_EXPR_H_
#define TVM_RELAX_EXPR_H_

#include <tvm/base.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/location.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/tensor.h>

#include <functional>

namespace tvm {
namespace relax {

// Compatibility aliases. Tuple expressions are defined in the common IR.
using ::tvm::Tuple;
using ::tvm::TupleGetItem;
using ::tvm::TupleGetItemObj;
using ::tvm::TupleObj;

/*! \brief A shape expression which allows users to construct a shape containing PrimExpr.
 */
class ShapeExprObj : public ExprObj {
 public:
  /*! The values of the shape expression. */
  ffi::Array<PrimExpr> values;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ShapeExprObj>().def_ro("values", &ShapeExprObj::values);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.ShapeExpr", ShapeExprObj, ExprObj);
};

class ShapeExpr : public Expr {
 public:
  TVM_DLL explicit ShapeExpr(ffi::Array<PrimExpr> values, Location loc = UnknownLoc());
  explicit ShapeExpr(ffi::ObjectPtr<ShapeExprObj> node) : Expr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ShapeExpr, Expr, ShapeExprObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ShapeExprObj);
};

/*! \brief A sub-type of the variable node used to mark dataflow variables from
 * normal visible "function local" bindings.
 */
class DataflowVarObj : public VarObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DataflowVarObj>();
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.DataflowVar", DataflowVarObj, VarObj);
};

class DataflowVar : public Var {
 public:
  TVM_DLL explicit DataflowVar(ffi::String name, ffi::Optional<Type> ty_annotation,
                               Location loc = UnknownLoc());

  explicit DataflowVar(ffi::ObjectPtr<DataflowVarObj> node) : Var(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DataflowVar, Var, DataflowVarObj);
};

/*! \brief Construct a tensor constant, inferring shape and dtype when type is omitted. */
TVM_DLL GenericConst MakeTensorConst(ffi::Tensor data,
                                     ffi::Optional<Type> ty_annotation = std::nullopt,
                                     Location loc = UnknownLoc());

/*! \brief The base class of a variable binding in Relax. */
class BindingObj : public ffi::Object {
 public:
  explicit BindingObj(Var var) : var(std::move(var)) {}
  explicit BindingObj(ffi::UnsafeInit) : var(ffi::UnsafeInit{}) {}

  mutable Location loc = UnknownLoc();
  /*! \brief The return variable to bound to. */
  Var var;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BindingObj>()
        .def_ro("loc", &BindingObj::loc, refl::AttachFieldFlag::SEqHashIgnore())
        // TODO(tqchen): use SEqHashDefSimple after the next pypi tvm-ffi release
        .def_ro("var", &BindingObj::var, refl::AttachFieldFlag::SEqHashDefPattern());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  TVM_FFI_DECLARE_OBJECT_INFO("relax.expr.Binding", BindingObj, ffi::Object);
};

class Binding : public ffi::ObjectRef {
 protected:
  Binding() = default;

 public:
  explicit Binding(ffi::UnsafeInit tag) : ffi::ObjectRef(tag) {}
  Binding(const Binding&) = default;
  Binding(Binding&&) = default;
  Binding& operator=(const Binding&) = default;
  Binding& operator=(Binding&&) = default;
  const BindingObj* operator->() const { return static_cast<const BindingObj*>(data_.get()); }
  const BindingObj* get() const { return operator->(); }
  static constexpr bool _type_is_nullable = false;
  using ContainerType = BindingObj;
};

/*!
 * \brief Runtime-match the value to the type.
 *
 * This operation does runtime check, populates the un-defined symbolic shape vars
 * and vars in ty in first occurance, and insert equality assertions in
 * other cases.
 */
class MatchCastObj : public BindingObj {
 public:
  explicit MatchCastObj(Var var, Expr value)
      : BindingObj(std::move(var)), value(std::move(value)) {}
  explicit MatchCastObj(ffi::UnsafeInit)
      : BindingObj(ffi::UnsafeInit{}), value(ffi::UnsafeInit{}) {}

  /*! \brief The input value to match cast. */
  Expr value;
  /*! \brief The type pattern to match to. */
  Type ty = Type::Missing();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MatchCastObj>()
        .def_ro("value", &MatchCastObj::value)
        // TODO(tqchen): use SEqHashDefSimple after the next pypi tvm-ffi release
        .def_ro("ty", &MatchCastObj::ty, refl::AttachFieldFlag::SEqHashDefPattern());
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.MatchCast", MatchCastObj, BindingObj);
};

/*!
 * \brief Managed reference to MatchCastObj.
 * \sa MatchCastObj
 */
class MatchCast : public Binding {
 public:
  TVM_DLL explicit MatchCast(Var var, Expr value, Type ty, Location loc = UnknownLoc());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MatchCast, Binding, MatchCastObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MatchCastObj);
};

class VarBindingObj : public BindingObj {
 public:
  explicit VarBindingObj(Var var, Expr value)
      : BindingObj(std::move(var)), value(std::move(value)) {}
  explicit VarBindingObj(ffi::UnsafeInit)
      : BindingObj(ffi::UnsafeInit{}), value(ffi::UnsafeInit{}) {}

  /*! \brief The binding value. */
  Expr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VarBindingObj>().def_ro("value", &VarBindingObj::value);
    // customize the SEqual and SHash methods for better error messages
    refl::TypeAttrDef<VarBindingObj>()
        .def(tvm::ffi::type_attr::kSEqual.name, &VarBindingObj::SEqual)
        .def(tvm::ffi::type_attr::kSHash.name, &VarBindingObj::SHash);
  }

  bool SEqual(const VarBindingObj* other,
              ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const;
  int64_t SHash(int64_t init_hash, ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.VarBinding", VarBindingObj, BindingObj);
};

class VarBinding : public Binding {
 public:
  TVM_DLL explicit VarBinding(Var var, Expr value, Location loc = UnknownLoc());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VarBinding, Binding, VarBindingObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VarBindingObj);
};

class BindingBlockObj : public ffi::Object {
 public:
  ffi::Array<Binding> bindings;
  mutable Location loc = UnknownLoc();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BindingBlockObj>()
        .def_ro("bindings", &BindingBlockObj::bindings)
        .def_ro("loc", &BindingBlockObj::loc, refl::AttachFieldFlag::SEqHashIgnore(),
                refl::DefaultValue(UnknownLoc()));
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.expr.BindingBlock", BindingBlockObj, ffi::Object);
};

class BindingBlock : public ffi::ObjectRef {
 public:
  TVM_DLL explicit BindingBlock(ffi::Array<Binding> bindings, Location loc = UnknownLoc());
  explicit BindingBlock(ffi::UnsafeInit tag, ffi::ObjectPtr<BindingBlockObj> n)
      : ffi::ObjectRef(tag) {
    data_ = std::move(n);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BindingBlock, ffi::ObjectRef, BindingBlockObj);

  BindingBlockObj* CopyOnWrite();
};

class DataflowBlockObj : public BindingBlockObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DataflowBlockObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.DataflowBlock", DataflowBlockObj, BindingBlockObj);
};

class DataflowBlock : public BindingBlock {
 public:
  TVM_DLL explicit DataflowBlock(ffi::Array<Binding> bindings, Location loc = UnknownLoc());
  explicit DataflowBlock(ffi::UnsafeInit tag, ffi::ObjectPtr<DataflowBlockObj> n)
      : BindingBlock(tag) {
    data_ = std::move(n);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DataflowBlock, BindingBlock, DataflowBlockObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataflowBlockObj);
};

/*! \brief A sequence of blocks followed by an expression.
 *
 * The order of blocks enforces scoping and ordering.
 */
class SeqExprObj : public ExprObj {
 public:
  explicit SeqExprObj(Expr body) : body(std::move(body)) {}
  explicit SeqExprObj(ffi::UnsafeInit) : body(ffi::UnsafeInit{}) {}

  ffi::Array<BindingBlock> blocks;
  Expr body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SeqExprObj>()
        .def_ro("blocks", &SeqExprObj::blocks)
        .def_ro("body", &SeqExprObj::body);
    refl::TypeAttrDef<SeqExprObj>()
        .def(tvm::ffi::type_attr::kSEqual.name, &SeqExprObj::SEqual)
        .def(tvm::ffi::type_attr::kSHash.name, &SeqExprObj::SHash);
  }

  bool SEqual(const SeqExprObj* other,
              ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const {
    // Establish mappings for symbolic variables defined by bindings before
    // comparing their uses in the SeqExpr result type and body.
    return equal(blocks, other->blocks, false, "blocks") && equal(ty, other->ty, false, "ty") &&
           equal(body, other->body, false, "body");
  }

  int64_t SHash(int64_t init_hash, ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
    int64_t hash_value = init_hash;
    hash_value = hash(blocks, hash_value, false);
    hash_value = hash(ty, hash_value, false);
    hash_value = hash(body, hash_value, false);
    return hash_value;
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.SeqExpr", SeqExprObj, ExprObj);
};

class SeqExpr : public Expr {
 public:
  /* \brief Implicit conversion constructor
   *
   * Relax nodes that introduce a new scope (e.g. `relax::Function`)
   * are required to be held as SeqExpr.  This implicit conversion
   * provides allows callsites to use these member variables when the
   * C++ compile-time type is a `relax::Expr`.  For example,
   * a transform may use `func.CopyOnWrite()->body = expr;`.
   *
   * If the expression is already a `relax::SeqExpr`, the same
   * underlying `relax::SeqExprObj` is used, and no copies are made.
   */
  TVM_DLL SeqExpr(Expr body);  // NOLINT(*)

  TVM_DLL explicit SeqExpr(ffi::Array<BindingBlock> blocks, Expr body, Location loc = UnknownLoc());
  explicit SeqExpr(ffi::ObjectPtr<SeqExprObj> node) : Expr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SeqExpr, Expr, SeqExprObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SeqExprObj);
};

/*!
 * \brief Condition expression
 *
 * Unlike traditional statement `if`s, the if evalutes
 * to the result of the branch taken.
 *
 * x = if (true) { 1 } else { 0 }; // x is 1
 * y = if (false) { 1 } else { 0 }; // y is 0
 *
 * \note This is similar to C's ternary operator.
 */
class IfExprObj : public ExprObj {
 public:
  explicit IfExprObj(Expr cond, SeqExpr true_branch, SeqExpr false_branch)
      : cond(std::move(cond)),
        true_branch(std::move(true_branch)),
        false_branch(std::move(false_branch)) {}
  explicit IfExprObj(ffi::UnsafeInit)
      : cond(ffi::UnsafeInit{}), true_branch(ffi::UnsafeInit{}), false_branch(ffi::UnsafeInit{}) {}

  /*! \brief The condition. */
  Expr cond;
  /*! \brief The expression evaluated when condition is true. */
  SeqExpr true_branch;
  /*! \brief The expression evaluated when condition is false */
  SeqExpr false_branch;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IfExprObj>()
        .def_ro("cond", &IfExprObj::cond)
        .def_ro("true_branch", &IfExprObj::true_branch)
        .def_ro("false_branch", &IfExprObj::false_branch);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindDAGNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.IfExpr", IfExprObj, ExprObj);
};

class IfExpr : public Expr {
 public:
  /*!
   * \brief The constructor
   *
   * \param cond The condition of a if node.
   *
   * \param true_branch The fall through branch.  If this is not a
   *     SeqExpr, it will be wrapped in a SeqExpr, to satisfy the
   *     Relax IR requirement that all scopes be contained in a
   *     SeqExpr.
   *
   * \param false_branch The branch for execution when condition is
   *     false.  If this is not a SeqExpr, it will be wrapped in a
   *     SeqExpr, to satisfy the Relax IR requirement that all scopes
   *     be contained in a SeqExpr.
   *
   * \param loc The source location of the expression.
   */
  TVM_DLL IfExpr(Expr cond, Expr true_branch, Expr false_branch, Location loc = UnknownLoc());

  explicit IfExpr(ffi::ObjectPtr<IfExprObj> node) : Expr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IfExpr, Expr, IfExprObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IfExprObj);
};

/*! \brief A Relax function. */
class FunctionObj : public BaseFuncObj {
 public:
  explicit FunctionObj(SeqExpr body) : body(std::move(body)) {}
  explicit FunctionObj(ffi::UnsafeInit) : body(ffi::UnsafeInit{}) {}

  /*! \brief The parameters to the function. */
  ffi::Array<Var> params;
  /*! \brief The body of the function. */
  SeqExpr body;
  /*! \brief The return type of the function. */
  Type ret_ty = Type::Missing();
  /*! \brief Whether the function is annotated as pure or not. */
  bool is_pure;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FunctionObj>()
        .def_ro("params", &FunctionObj::params, refl::AttachFieldFlag::SEqHashDefPattern())
        .def_ro("body", &FunctionObj::body)
        .def_ro("ret_ty", &FunctionObj::ret_ty)
        .def_ro("is_pure", &FunctionObj::is_pure);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindDAGNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.Function", FunctionObj, BaseFuncObj);
};

class Function : public BaseFunc {
 public:
  /*!
   * \brief Construct a Relax Function
   *
   * \param params The parameters accepted by the function
   *
   * \param body The body of the function.  If this is not a
   *     SeqExpr, it will be wrapped in a SeqExpr, to satisfy the
   *     Relax IR requirement that all scopes be contained in a
   *     SeqExpr.
   *
   * \param ret_ty The Type returned by the function.
   *     If std::nullopt, will be inferred from the Type of the
   *     function's body.
   *
   * \param is_pure The purity of the function.
   *
   * \param attrs Any attributes associated with the function.
   *     Defaults to an empty dictionary.
   *
   * \param loc The source location of the expression.
   */
  TVM_DLL explicit Function(ffi::Array<Var> params, Expr body, ffi::Optional<Type> ret_ty,
                            bool is_pure = true, DictAttrs attrs = DictAttrs(),
                            Location loc = UnknownLoc());

  /*!
   * \brief Mimics the constructor but without body Expr.
   * \note ret_ty is required, since it can not deduced by the body.
   */
  TVM_DLL static Function CreateEmpty(ffi::Array<Var> params, Type ret_ty, bool is_pure = true,
                                      DictAttrs attrs = DictAttrs(), Location loc = UnknownLoc());

  explicit Function(ffi::ObjectPtr<FunctionObj> node) : BaseFunc(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Function, BaseFunc, FunctionObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FunctionObj);
};

/*! \brief The extern function, which can represent packed function. */
class ExternFuncObj : public BaseFuncObj {
 public:
  /*! \brief The name of global symbol. */
  ffi::String global_symbol;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExternFuncObj>().def_ro("global_symbol", &ExternFuncObj::global_symbol);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.ExternFunc", ExternFuncObj, BaseFuncObj);
};

class ExternFunc : public BaseFunc {
 public:
  TVM_DLL ExternFunc(ffi::String global_symbol, Location loc = UnknownLoc());
  TVM_DLL ExternFunc(ffi::String global_symbol, Type ty, Location loc = UnknownLoc());

  explicit ExternFunc(ffi::ObjectPtr<ExternFuncObj> node) : BaseFunc(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExternFunc, BaseFunc, ExternFuncObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ExternFuncObj);
};

/*!
 * \brief Get the shape of Expr.
 * \param expr The input expr.
 * \return The corresonding shape.
 *
 * \note This function requires expr to be normalized.
 *       The function will report an error if expr's Type is not TensorType.
 *       It will try to return symbolic function when possible. If the tensor do not
 *       have a compile-time symbolic shape, the function will then choose to return
 *       Call(relax.op.shape_of, [expr]).
 */
TVM_DLL Expr GetShapeOf(const Expr& expr);

// TODO(@sunggg): Investigate the exact usage of kComposite, kPartitionedFromPattern, and
// kPrimitive.
namespace attr {
/*!
 * \brief All the named Tensors accumulated during compilation by external codegen.
 * Generally the associated runtime::Module will indicate it requires bindings for these names,
 * and during module initialization these bindings will be recovered from a ConstLoaderModule.
 *
 * Type: ffi::Map<ffi::String, Tensor>
 */
inline constexpr ffi::reflection::AttrKey<ffi::Map<ffi::String, ffi::Tensor>> kConstNameToConstant{
    "const_name_to_constant"};

/*! \brief Mark the function as a primitive function. */
inline constexpr ffi::reflection::AttrKey<bool> kPrimitive{"Primitive"};
/*!
 * \brief Indicate the codegen that should be used for building this function.
 * When this is unset or set to "default", the default compilation pipeline will be used.
 */
inline constexpr ffi::reflection::AttrKey<ffi::String> kCodegen{"Codegen"};
/*! \brief Treat the function as a composite operator. */
inline constexpr ffi::reflection::AttrKey<ffi::String> kComposite{"Composite"};
/*! \brief Indicate the function was created by the Pattern Partitioning Pass. */
inline constexpr ffi::reflection::AttrKey<ffi::String> kPartitionedFromPattern{
    "PartitionedFromPattern"};
/*! \brief The required workspace for an external function. */
inline constexpr ffi::reflection::AttrKey<int64_t> kWorkspaceSize{"WorkspaceSize"};

// Note: in the future, we prefer snake_case instead of CamelCase for attributes.
// Past ones will be kept for backwards compatibility.
/*! \brief Override checking purity for this function and treat as pure
 * (is_pure must be set to true) */
inline constexpr ffi::reflection::AttrKey<bool> kForcePure{"relax.force_pure"};

/*!
 * \brief The number of inputs of a function.
 * If a function has the num_input attribute, the last func->params.size() - num_inputs
 * arguments are assumed to be weights that are fixed across invocations.
 */
inline constexpr ffi::reflection::AttrKey<int64_t> kNumInput{"num_input"};
inline constexpr ffi::reflection::AttrKey<int64_t> kOpPattern{"op_pattern"};

inline constexpr ffi::reflection::AttrKey<ffi::ObjectRef> kOpAttrs{"op_attrs"};

inline constexpr ffi::reflection::AttrKey<ffi::String> kOperatorName{"operator_name"};

inline constexpr ffi::reflection::AttrKey<ffi::String> kLibraryKernel{"library_kernel"};

inline constexpr ffi::reflection::AttrKey<ffi::String> kCSource{"c_source"};

inline constexpr ffi::reflection::AttrKey<ffi::String> kCSourceFmt{"c_source_fmt"};

inline constexpr ffi::reflection::AttrKey<ffi::Map<ffi::String, IntImm>> kTirVarUpperBound{
    "tir_var_upper_bound"};

inline constexpr ffi::reflection::AttrKey<ffi::Map<ffi::String, IntImm>> kTirVarLowerBound{
    "tir_var_lower_bound"};

inline constexpr ffi::reflection::AttrKey<ffi::Array<ffi::String>> kTirNonNegativeVar{
    "tir_non_negative_var"};

constexpr const char* kMemoryPlanDynamicFuncOutput = "relax.memory_plan_dynamic_func_output";

inline constexpr ffi::reflection::AttrKey<ffi::Array<ffi::String>> kCaptureSymbolicVars{
    "relax.rewrite_cuda_graph.capture_symbolic_vars"};

inline constexpr ffi::reflection::AttrKey<int64_t> kScoped{"scoped"};

}  // namespace attr

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_EXPR_H_
