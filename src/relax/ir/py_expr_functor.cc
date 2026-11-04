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
 * \file src/relax/py_expr_functor.cc
 * \brief The backbone of PyExprVisitor/PyExprMutator.
 */
#include <tvm/base.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr_functor.h>

namespace tvm {
namespace relax {

/*!
 * \brief The abstract interface of ExprVisitor.
 */
class PyExprVisitorObj : public ffi::Object, public ExprVisitor {
 private:
  using TSelf = PyExprVisitorObj;
  using FType = tvm::ObjectFunctor<void(const ffi::ObjectRef& n, TSelf* self)>;

 public:
  /*! \brief The packed function to the `VisitExpr(const Expr& expr)` function. */
  ffi::Function f_visit_expr{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const GenericConstObj* op)` function. */
  ffi::Function f_visit_generic_const_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const TupleObj* op)` function. */
  ffi::Function f_visit_tuple_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const VarObj* op)` function. */
  ffi::Function f_visit_var_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const DataflowVarObj* op)` function. */
  ffi::Function f_visit_dataflow_var_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ShapeExprObj* op)` function. */
  ffi::Function f_visit_shape_expr_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ExternFuncObj* op)` function. */
  ffi::Function f_visit_extern_func_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const GlobalVarObj* op)` function. */
  ffi::Function f_visit_global_var_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const FunctionObj* op)` function. */
  ffi::Function f_visit_function_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const CallObj* op)` function. */
  ffi::Function f_visit_call_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const SeqExprObj* op)` function. */
  ffi::Function f_visit_seq_expr_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const IfExprObj* op)` function. */
  ffi::Function f_visit_if_expr_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const OpObj* op)` function. */
  ffi::Function f_visit_op_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const TupleGetItemObj* op)` function. */
  ffi::Function f_visit_tuple_getitem_{nullptr};
  /*! \brief The packed function to the generic expression fallback. */
  ffi::Function f_visit_expr_fallback_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const StringImmObj* op)` function. */
  ffi::Function f_visit_string_imm_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const DataTypeImmObj* op)` function. */
  ffi::Function f_visit_data_type_imm_{nullptr};
  /*! \brief The packed function to the `VisitBinding(const Binding& binding)` function. */
  ffi::Function f_visit_binding{nullptr};
  /*! \brief The packed function to the `VisitBinding_(const VarBindingObj* binding)`
   * function. */
  ffi::Function f_visit_var_binding_{nullptr};
  /*! \brief The packed function to the `VisitBinding_(const MatchCastObj* binding)`
   * function. */
  ffi::Function f_visit_match_cast_{nullptr};
  /*! \brief The packed function to the `VisitBindingBlock(const BindingBlock& block)`
   * function. */
  ffi::Function f_visit_binding_block{nullptr};
  /*! \brief The packed function to the `VisitBindingBlock_(const BindingBlockObj* block)`
   * function. */
  ffi::Function f_visit_binding_block_{nullptr};
  /*! \brief The packed function to the `VisitBindingBlock_(const DataflowBlockObj* block)`
   * function. */
  ffi::Function f_visit_dataflow_block_{nullptr};
  /*! \brief The packed function to the `VisitVarDef(const Var& var)` function. */
  ffi::Function f_visit_var_def{nullptr};
  /*! \brief The packed function to the `VisitVarDef_(const VarObj* var)` function. */
  ffi::Function f_visit_var_def_{nullptr};
  /*! \brief The packed function to the `VisitVarDef_(const DataflowVarObj* var)` function. */
  ffi::Function f_visit_dataflow_var_def_{nullptr};
  /*! \brief The packed function to the `VisitLoc(const Location& loc)` function. */
  ffi::Function f_visit_loc{nullptr};

  void VisitExpr(const Expr& expr) override {
    if (f_visit_expr != nullptr) {
      f_visit_expr(expr);
    } else {
      // Need to init the overwrite VTable
      static FType vtable = InitVTable();
      if (vtable.CanDispatch(expr)) {
        vtable(expr, this);
      } else {
        ExprVisitor::VisitExpr(expr);
      }
    }
  }

  void VisitExprFallback_(const ExprObj* op) override
      PY_EXPR_VISITOR_DEFAULT(ffi::GetRef<Expr>(op), f_visit_expr_fallback_,
                              ExprVisitor::VisitExprFallback_(op));

  void VisitBinding(const Binding& binding) override
      PY_EXPR_VISITOR_DEFAULT(binding, f_visit_binding, ExprVisitor::VisitBinding(binding));

  void VisitBinding_(const VarBindingObj* binding) override
      PY_EXPR_VISITOR_DEFAULT(ffi::GetRef<VarBinding>(binding), f_visit_var_binding_,
                              ExprVisitor::VisitBinding_(binding));
  void VisitBinding_(const MatchCastObj* binding) override
      PY_EXPR_VISITOR_DEFAULT(ffi::GetRef<MatchCast>(binding), f_visit_match_cast_,
                              ExprVisitor::VisitBinding_(binding));

  void VisitBindingBlock(const BindingBlock& block) override
      PY_EXPR_VISITOR_DEFAULT(block, f_visit_binding_block, ExprVisitor::VisitBindingBlock(block));

  void VisitBindingBlock_(const BindingBlockObj* block) override
      PY_EXPR_VISITOR_DEFAULT(ffi::GetRef<BindingBlock>(block), f_visit_binding_block_,
                              ExprVisitor::VisitBindingBlock_(block));
  void VisitBindingBlock_(const DataflowBlockObj* block) override
      PY_EXPR_VISITOR_DEFAULT(ffi::GetRef<DataflowBlock>(block), f_visit_dataflow_block_,
                              ExprVisitor::VisitBindingBlock_(block));

  void VisitVarDef(const Var& var) override
      PY_EXPR_VISITOR_DEFAULT(var, f_visit_var_def, ExprVisitor::VisitVarDef(var));
  void VisitVarDef_(const VarObj* var) override
      PY_EXPR_VISITOR_DEFAULT(ffi::GetRef<Var>(var), f_visit_var_def_,
                              ExprVisitor::VisitVarDef_(var));
  void VisitVarDef_(const DataflowVarObj* var) override
      PY_EXPR_VISITOR_DEFAULT(ffi::GetRef<DataflowVar>(var), f_visit_dataflow_var_def_,
                              ExprVisitor::VisitVarDef_(var));

  void VisitLoc(const Location& loc) override
      PY_EXPR_VISITOR_DEFAULT(loc, f_visit_loc, ExprVisitor::VisitLoc(loc));

  static void RegisterReflection() {
    // PyExprVisitorObj has no fields to register
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("expr_functor.PyExprVisitor", PyExprVisitorObj, ffi::Object);

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    PY_EXPR_VISITOR_DISPATCH(GenericConstObj, f_visit_generic_const_);
    PY_EXPR_VISITOR_DISPATCH(TupleObj, f_visit_tuple_);
    PY_EXPR_VISITOR_DISPATCH(VarObj, f_visit_var_);
    PY_EXPR_VISITOR_DISPATCH(DataflowVarObj, f_visit_dataflow_var_);
    PY_EXPR_VISITOR_DISPATCH(ShapeExprObj, f_visit_shape_expr_);
    PY_EXPR_VISITOR_DISPATCH(ExternFuncObj, f_visit_extern_func_);
    PY_EXPR_VISITOR_DISPATCH(GlobalVarObj, f_visit_global_var_);
    PY_EXPR_VISITOR_DISPATCH(FunctionObj, f_visit_function_);
    PY_EXPR_VISITOR_DISPATCH(CallObj, f_visit_call_);
    PY_EXPR_VISITOR_DISPATCH(SeqExprObj, f_visit_seq_expr_);
    PY_EXPR_VISITOR_DISPATCH(IfExprObj, f_visit_if_expr_);
    PY_EXPR_VISITOR_DISPATCH(OpObj, f_visit_op_);
    PY_EXPR_VISITOR_DISPATCH(TupleGetItemObj, f_visit_tuple_getitem_);
    PY_EXPR_VISITOR_DISPATCH(StringImmObj, f_visit_string_imm_);
    PY_EXPR_VISITOR_DISPATCH(DataTypeImmObj, f_visit_data_type_imm_);
    vtable.Finalize();
    return vtable;
  }
};

/*!
 * \brief Managed reference to PyExprVisitorObj.
 * \sa PyExprVisitorObj
 */
class PyExprVisitor : public ffi::ObjectRef {
 public:
  explicit PyExprVisitor(ffi::ObjectPtr<PyExprVisitorObj> data) : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  /*!
   * \brief Create a PyExprVisitor with customized methods on the python-side.
   * \param f_visit_expr The packed function of `VisitExpr(const Expr& expr)`.
   * \param f_visit_generic_const_ The packed function of `VisitExpr_(const GenericConstObj* op)`.
   * \param f_visit_tuple_ The packed function of `VisitExpr_(const TupleObj* op)`.
   * \param f_visit_var_ The packed function of `VisitExpr_(const VarObj* op)`.
   * \param f_visit_dataflow_var_ The packed function of `VisitExpr_(const DataflowVarObj* op)`.
   * \param f_visit_shape_expr_ The packed function of `VisitExpr_(const ShapeExprObj* op)`.
   * \param f_visit_extern_func_ The packed function of `VisitExpr_(const ExternFuncObj* op)`.
   * \param f_visit_global_var_ The packed function of `VisitExpr_(const GlobalVarObj* op)`.
   * \param f_visit_function_ The packed function of `VisitExpr_(const FunctionObj* op)`.
   * \param f_visit_call_ The packed function of `VisitExpr_(const CallObj* op)`.
   * \param f_visit_seq_expr_ The packed function of `VisitExpr_(const SeqExprObj* op)`.
   * \param f_visit_if_expr_ The packed function of `VisitExpr_(const IfExprObj* op)`.
   * \param f_visit_op_ The packed function of `VisitExpr_(const OpObj* op)`.
   * \param f_visit_tuple_getitem_ The packed function of `VisitExpr_(const TupleGetItemObj* op)`.
   * \param f_visit_expr_fallback_ The packed function of the generic expression fallback.
   * \param f_visit_string_imm_ The packed function of `VisitExpr_(const StringImmObj* op)`.
   * \param f_visit_data_type_imm_ The packed function of `VisitExpr_(const DataTypeImmObj* op)`.
   * \param f_visit_binding The packed function of `VisitBinding(const Binding& binding)`.
   * \param f_visit_var_binding_ The packed function of `VisitBinding_(const VarBindingObj*
   * binding)`.
   * \param f_visit_match_cast_ The packed function of `VisitBinding_(const MatchCastObj*
   * binding)`.
   * \param f_visit_binding_block The packed function of `VisitBindingBlock(const BindingBlock&
   * block)`.
   * \param f_visit_binding_block_ The packed function of `VisitBindingBlock_(const
   * BindingBlockObj* block)`.
   * \param f_visit_dataflow_block_ The packed function of `VisitBindingBlock_(const
   * DataflowBlockObj* block)`.
   * \param f_visit_var_def The packed function of `VisitVarDef(const Var& var)`.
   * \param f_visit_var_def_ The packed function of `VisitVarDef_(const VarObj* var)`.
   * \param f_visit_dataflow_var_def_ The packed function of `VisitVarDef_(const DataflowVarObj*
   * var)`.
   * \param f_visit_loc The packed function of `VisitLoc(const Location& loc)`.
   * \return The PyVisitor created.
   */
  TVM_DLL static PyExprVisitor MakePyExprVisitor(
      ffi::Function f_visit_expr, ffi::Function f_visit_generic_const_,
      ffi::Function f_visit_tuple_, ffi::Function f_visit_var_, ffi::Function f_visit_dataflow_var_,
      ffi::Function f_visit_shape_expr_, ffi::Function f_visit_extern_func_,
      ffi::Function f_visit_global_var_, ffi::Function f_visit_function_,
      ffi::Function f_visit_call_, ffi::Function f_visit_seq_expr_, ffi::Function f_visit_if_expr_,
      ffi::Function f_visit_op_, ffi::Function f_visit_tuple_getitem_,
      ffi::Function f_visit_expr_fallback_, ffi::Function f_visit_string_imm_,
      ffi::Function f_visit_data_type_imm_, ffi::Function f_visit_binding,
      ffi::Function f_visit_var_binding_, ffi::Function f_visit_match_cast_,
      ffi::Function f_visit_binding_block, ffi::Function f_visit_binding_block_,
      ffi::Function f_visit_dataflow_block_, ffi::Function f_visit_var_def,
      ffi::Function f_visit_var_def_, ffi::Function f_visit_dataflow_var_def_,
      ffi::Function f_visit_loc) {
    ffi::ObjectPtr<PyExprVisitorObj> n = ffi::make_object<PyExprVisitorObj>();
    n->f_visit_expr = f_visit_expr;
    n->f_visit_binding = f_visit_binding;
    n->f_visit_binding_block = f_visit_binding_block;
    n->f_visit_var_def = f_visit_var_def;
    n->f_visit_loc = f_visit_loc;
    n->f_visit_generic_const_ = f_visit_generic_const_;
    n->f_visit_tuple_ = f_visit_tuple_;
    n->f_visit_var_ = f_visit_var_;
    n->f_visit_dataflow_var_ = f_visit_dataflow_var_;
    n->f_visit_shape_expr_ = f_visit_shape_expr_;
    n->f_visit_extern_func_ = f_visit_extern_func_;
    n->f_visit_global_var_ = f_visit_global_var_;
    n->f_visit_function_ = f_visit_function_;
    n->f_visit_call_ = f_visit_call_;
    n->f_visit_seq_expr_ = f_visit_seq_expr_;
    n->f_visit_if_expr_ = f_visit_if_expr_;
    n->f_visit_op_ = f_visit_op_;
    n->f_visit_tuple_getitem_ = f_visit_tuple_getitem_;
    n->f_visit_expr_fallback_ = f_visit_expr_fallback_;
    n->f_visit_string_imm_ = f_visit_string_imm_;
    n->f_visit_data_type_imm_ = f_visit_data_type_imm_;
    n->f_visit_var_binding_ = f_visit_var_binding_;
    n->f_visit_match_cast_ = f_visit_match_cast_;
    n->f_visit_binding_block_ = f_visit_binding_block_;
    n->f_visit_dataflow_block_ = f_visit_dataflow_block_;
    n->f_visit_var_def_ = f_visit_var_def_;
    n->f_visit_dataflow_var_def_ = f_visit_dataflow_var_def_;
    return PyExprVisitor(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PyExprVisitor, ffi::ObjectRef, PyExprVisitorObj);
};

/*!
 * \brief The abstract interface of ExprMutator.
 */
class PyExprMutatorObj : public ffi::Object, public ExprMutator {
 private:
  using TSelf = PyExprMutatorObj;
  using FType = tvm::ObjectFunctor<Expr(const ffi::ObjectRef& n, TSelf* self)>;

 public:
  /*! \brief The packed function to the `VisitExpr(const Expr& expr)` function. */
  ffi::Function f_visit_expr{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const GenericConstObj* op)` function. */
  ffi::Function f_visit_generic_const_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const TupleObj* op)` function. */
  ffi::Function f_visit_tuple_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const VarObj* op)` function. */
  ffi::Function f_visit_var_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const DataflowVarObj* op)` function. */
  ffi::Function f_visit_dataflow_var_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ShapeExprObj* op)` function. */
  ffi::Function f_visit_shape_expr_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ExternFuncObj* op)` function. */
  ffi::Function f_visit_extern_func_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const GlobalVarObj* op)` function. */
  ffi::Function f_visit_global_var_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const FunctionObj* op)` function. */
  ffi::Function f_visit_function_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const CallObj* op)` function. */
  ffi::Function f_visit_call_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const SeqExprObj* op)` function. */
  ffi::Function f_visit_seq_expr_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const IfExprObj* op)` function. */
  ffi::Function f_visit_if_expr_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const OpObj* op)` function. */
  ffi::Function f_visit_op_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const TupleGetItemObj* op)` function. */
  ffi::Function f_visit_tuple_getitem_{nullptr};
  /*! \brief The packed function to the generic expression fallback. */
  ffi::Function f_visit_expr_fallback_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const StringImmObj* op)` function. */
  ffi::Function f_visit_string_imm_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const DataTypeImmObj* op)` function. */
  ffi::Function f_visit_data_type_imm_{nullptr};
  /*! \brief The packed function to the `VisitBinding(const Binding& binding)` function. */
  ffi::Function f_visit_binding{nullptr};
  /*! \brief The packed function to the `VisitBinding_(const VarBindingObj* binding)`
   * function. */
  ffi::Function f_visit_var_binding_{nullptr};
  /*! \brief The packed function to the `VisitBinding_(const MatchCastObj* binding)`
   * function. */
  ffi::Function f_visit_match_cast_{nullptr};
  /*! \brief The packed function to the `VisitBindingBlock(const BindingBlock& block)`
   * function. */
  ffi::Function f_visit_binding_block{nullptr};
  /*! \brief The packed function to the `VisitBindingBlock_(const BindingBlockObj* block)`
   * function. */
  ffi::Function f_visit_binding_block_{nullptr};
  /*! \brief The packed function to the `VisitBindingBlock_(const DataflowBlockObj* block)`
   * function. */
  ffi::Function f_visit_dataflow_block_{nullptr};
  /*! \brief The packed function to the `VisitVarDef(const Var& var)` function. */
  ffi::Function f_visit_var_def{nullptr};
  /*! \brief The packed function to the `VisitVarDef_(const VarObj* var)` function. */
  ffi::Function f_visit_var_def_{nullptr};
  /*! \brief The packed function to the `VisitVarDef_(const DataflowVarObj* var)` function. */
  ffi::Function f_visit_dataflow_var_def_{nullptr};
  /*! \brief The packed function to the `VisitLoc(const Location& loc)` function. */
  ffi::Function f_visit_loc{nullptr};

  Expr VisitExpr(const Expr& expr) override {
    if (f_visit_expr != nullptr) {
      return builder_->Normalize(f_visit_expr(expr).cast<Expr>());
    } else {
      static FType vtable = InitVTable();
      if (vtable.CanDispatch(expr)) {
        return builder_->Normalize(vtable(expr, this));
      }
      return ExprMutator::VisitExpr(expr);
    }
  }

  Expr VisitExprFallback_(const ExprObj* op) override
      PY_EXPR_MUTATOR_DEFAULT(ffi::GetRef<Expr>(op), f_visit_expr_fallback_,
                              ExprMutator::VisitExprFallback_(op), Expr);

  void VisitBinding(const Binding& binding) override {
    if (f_visit_binding != nullptr)
      f_visit_binding(binding);
    else
      ExprMutator::VisitBinding(binding);
  }

  void VisitBinding_(const VarBindingObj* binding) override {
    if (f_visit_var_binding_ != nullptr)
      f_visit_var_binding_(ffi::GetRef<VarBinding>(binding));
    else
      ExprMutator::VisitBinding_(binding);
  }

  void VisitBinding_(const MatchCastObj* binding) override {
    if (f_visit_match_cast_ != nullptr)
      f_visit_match_cast_(ffi::GetRef<MatchCast>(binding));
    else
      ExprMutator::VisitBinding_(binding);
  }

  BindingBlock VisitBindingBlock(const BindingBlock& block) override
      PY_EXPR_MUTATOR_DEFAULT(block, f_visit_binding_block, ExprMutator::VisitBindingBlock(block),
                              BindingBlock);

  BindingBlock VisitBindingBlock_(const BindingBlockObj* block) override
      PY_EXPR_MUTATOR_DEFAULT(ffi::GetRef<BindingBlock>(block), f_visit_binding_block_,
                              ExprMutator::VisitBindingBlock_(block), BindingBlock);
  BindingBlock VisitBindingBlock_(const DataflowBlockObj* block) override
      PY_EXPR_MUTATOR_DEFAULT(ffi::GetRef<DataflowBlock>(block), f_visit_dataflow_block_,
                              ExprMutator::VisitBindingBlock_(block), BindingBlock);

  Var VisitVarDef(const Var& var) override
      PY_EXPR_MUTATOR_DEFAULT(var, f_visit_var_def, ExprMutator::VisitVarDef(var), Var);
  Var VisitVarDef_(const VarObj* var) override
      PY_EXPR_MUTATOR_DEFAULT(ffi::GetRef<Var>(var), f_visit_var_def_,
                              ExprMutator::VisitVarDef_(var), Var);
  Var VisitVarDef_(const DataflowVarObj* var) override
      PY_EXPR_MUTATOR_DEFAULT(ffi::GetRef<DataflowVar>(var), f_visit_dataflow_var_def_,
                              ExprMutator::VisitVarDef_(var), Var);

  /*!
   * \brief Dispatcher for post-order rewrite.
   * \param expr The Expr to be rewritten.
   * \return The Expr after post-order rewritten.
   */
  Expr VisitExprPostOrder(const Expr& expr) {
    static FType post_order_vtable = InitPostOrderVTable();
    if (post_order_vtable.CanDispatch(expr)) {
      return post_order_vtable(expr, this);
    }
    return builder_->Normalize(ExprMutator::VisitExprFallback_(expr.get()));
  }

  using ExprMutator::builder_;
  using ExprMutator::LookupBinding;
  using ExprMutator::var_remap_;
  using ExprMutator::VisitWithNewScope;
  using ExprMutator::WithType;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PyExprMutatorObj>().def_ro("builder_", &PyExprMutatorObj::builder_);
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("expr_functor.PyExprMutator", PyExprMutatorObj, ffi::Object);

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    PY_EXPR_MUTATOR_DISPATCH(GenericConstObj, f_visit_generic_const_);
    PY_EXPR_MUTATOR_DISPATCH(TupleObj, f_visit_tuple_);
    PY_EXPR_MUTATOR_DISPATCH(VarObj, f_visit_var_);
    PY_EXPR_MUTATOR_DISPATCH(DataflowVarObj, f_visit_dataflow_var_);
    PY_EXPR_MUTATOR_DISPATCH(ShapeExprObj, f_visit_shape_expr_);
    PY_EXPR_MUTATOR_DISPATCH(ExternFuncObj, f_visit_extern_func_);
    PY_EXPR_MUTATOR_DISPATCH(GlobalVarObj, f_visit_global_var_);
    PY_EXPR_MUTATOR_DISPATCH(FunctionObj, f_visit_function_);
    PY_EXPR_MUTATOR_DISPATCH(CallObj, f_visit_call_);
    PY_EXPR_MUTATOR_DISPATCH(SeqExprObj, f_visit_seq_expr_);
    PY_EXPR_MUTATOR_DISPATCH(IfExprObj, f_visit_if_expr_);
    PY_EXPR_MUTATOR_DISPATCH(OpObj, f_visit_op_);
    PY_EXPR_MUTATOR_DISPATCH(TupleGetItemObj, f_visit_tuple_getitem_);
    PY_EXPR_MUTATOR_DISPATCH(StringImmObj, f_visit_string_imm_);
    PY_EXPR_MUTATOR_DISPATCH(DataTypeImmObj, f_visit_data_type_imm_);
    vtable.Finalize();
    return vtable;
  }

  // initialize the vtable for post order visit.
  static FType InitPostOrderVTable() {
    FType post_order_vtable;
    // Set dispatch
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(GenericConstObj);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(TupleObj);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(VarObj);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(DataflowVarObj);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(ShapeExprObj);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(ExternFuncObj);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(GlobalVarObj);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(FunctionObj);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(CallObj);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(SeqExprObj);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(IfExprObj);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(OpObj);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(TupleGetItemObj);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(StringImmObj);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(DataTypeImmObj);
    post_order_vtable.Finalize();
    return post_order_vtable;
  }
};

/*!
 * \brief Managed reference to PyExprMutatorObj.
 * \sa PyExprMutatorObj
 */
class PyExprMutator : public ffi::ObjectRef {
 public:
  explicit PyExprMutator(ffi::ObjectPtr<PyExprMutatorObj> data) : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  /*!
   * \brief Create a PyExprMutator with customized methods on the python-side.
   * \param f_visit_expr The packed function of `VisitExpr(const Expr& expr)`.
   * \param f_visit_generic_const_ The packed function of `VisitExpr_(const GenericConstObj* op)`.
   * \param f_visit_tuple_ The packed function of `VisitExpr_(const TupleObj* op)`.
   * \param f_visit_var_ The packed function of `VisitExpr_(const VarObj* op)`.
   * \param f_visit_dataflow_var_ The packed function of `VisitExpr_(const DataflowVarObj* op)`.
   * \param f_visit_shape_expr_ The packed function of `VisitExpr_(const ShapeExprObj* op)`.
   * \param f_visit_extern_func_ The packed function of `VisitExpr_(const ExternFuncObj* op)`.
   * \param f_visit_global_var_ The packed function of `VisitExpr_(const GlobalVarObj* op)`.
   * \param f_visit_function_ The packed function of `VisitExpr_(const FunctionObj* op)`.
   * \param f_visit_call_ The packed function of `VisitExpr_(const CallObj* op)`.
   * \param f_visit_seq_expr_ The packed function of `VisitExpr_(const SeqExprObj* op)`.
   * \param f_visit_if_expr_ The packed function of `VisitExpr_(const IfExprObj* op)`.
   * \param f_visit_op_ The packed function of `VisitExpr_(const OpObj* op)`.
   * \param f_visit_tuple_getitem_ The packed function of `VisitExpr_(const TupleGetItemObj* op)`.
   * \param f_visit_expr_fallback_ The packed function of the generic expression fallback.
   * \param f_visit_string_imm_ The packed function of `VisitExpr_(const StringImmObj* op)`.
   * \param f_visit_data_type_imm_ The packed function of `VisitExpr_(const DataTypeImmObj* op)`.
   * \param f_visit_binding The packed function of `VisitBinding(const Binding& binding)`.
   * \param f_visit_var_binding_ The packed function of `VisitBinding_(const VarBindingObj*
   * binding)`.
   * \param f_visit_match_cast_ The packed function of `VisitBinding_(const MatchCastObj*
   * binding)`.
   * \param f_visit_binding_block The packed function of `VisitBindingBlock(const BindingBlock&
   * block)`.
   * \param f_visit_binding_block_ The packed function of `VisitBindingBlock_(const
   * BindingBlockObj* block)`.
   * \param f_visit_dataflow_block_ The packed function of `VisitBindingBlock_(const
   * DataflowBlockObj* block)`.
   * \param f_visit_var_def The packed function of `VisitVarDef(const Var& var)`.
   * \param f_visit_var_def_ The packed function of `VisitVarDef_(const VarObj* var)`.
   * \param f_visit_dataflow_var_def_ The packed function of `VisitVarDef_(const DataflowVarObj*
   * var)`.
   * \param f_visit_loc The packed function of `VisitLoc(const Location& loc)`.
   * \return The PyExprMutator created.
   */
  TVM_DLL static PyExprMutator MakePyExprMutator(
      BlockBuilder builder_, ffi::Function f_visit_expr, ffi::Function f_visit_generic_const_,
      ffi::Function f_visit_tuple_, ffi::Function f_visit_var_, ffi::Function f_visit_dataflow_var_,
      ffi::Function f_visit_shape_expr_, ffi::Function f_visit_extern_func_,
      ffi::Function f_visit_global_var_, ffi::Function f_visit_function_,
      ffi::Function f_visit_call_, ffi::Function f_visit_seq_expr_, ffi::Function f_visit_if_expr_,
      ffi::Function f_visit_op_, ffi::Function f_visit_tuple_getitem_,
      ffi::Function f_visit_expr_fallback_, ffi::Function f_visit_string_imm_,
      ffi::Function f_visit_data_type_imm_, ffi::Function f_visit_binding,
      ffi::Function f_visit_var_binding_, ffi::Function f_visit_match_cast_,
      ffi::Function f_visit_binding_block, ffi::Function f_visit_binding_block_,
      ffi::Function f_visit_dataflow_block_, ffi::Function f_visit_var_def,
      ffi::Function f_visit_var_def_, ffi::Function f_visit_dataflow_var_def_,
      ffi::Function f_visit_loc) {
    ffi::ObjectPtr<PyExprMutatorObj> n = ffi::make_object<PyExprMutatorObj>();
    n->builder_ = builder_;
    n->f_visit_expr = f_visit_expr;
    n->f_visit_generic_const_ = f_visit_generic_const_;
    n->f_visit_tuple_ = f_visit_tuple_;
    n->f_visit_var_ = f_visit_var_;
    n->f_visit_dataflow_var_ = f_visit_dataflow_var_;
    n->f_visit_shape_expr_ = f_visit_shape_expr_;
    n->f_visit_extern_func_ = f_visit_extern_func_;
    n->f_visit_global_var_ = f_visit_global_var_;
    n->f_visit_function_ = f_visit_function_;
    n->f_visit_call_ = f_visit_call_;
    n->f_visit_seq_expr_ = f_visit_seq_expr_;
    n->f_visit_if_expr_ = f_visit_if_expr_;
    n->f_visit_op_ = f_visit_op_;
    n->f_visit_tuple_getitem_ = f_visit_tuple_getitem_;
    n->f_visit_expr_fallback_ = f_visit_expr_fallback_;
    n->f_visit_string_imm_ = f_visit_string_imm_;
    n->f_visit_data_type_imm_ = f_visit_data_type_imm_;
    n->f_visit_binding = f_visit_binding;
    n->f_visit_var_binding_ = f_visit_var_binding_;
    n->f_visit_match_cast_ = f_visit_match_cast_;
    n->f_visit_binding_block = f_visit_binding_block;
    n->f_visit_binding_block_ = f_visit_binding_block_;
    n->f_visit_dataflow_block_ = f_visit_dataflow_block_;
    n->f_visit_var_def = f_visit_var_def;
    n->f_visit_var_def_ = f_visit_var_def_;
    n->f_visit_dataflow_var_def_ = f_visit_dataflow_var_def_;
    n->f_visit_loc = f_visit_loc;
    return PyExprMutator(n);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PyExprMutator, ffi::ObjectRef, PyExprMutatorObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<PyExprVisitorObj>();
  refl::GlobalDef()
      .def("relax.MakePyExprVisitor", PyExprVisitor::MakePyExprVisitor)
      .def("relax.PyExprVisitorVisitExpr",
           [](PyExprVisitor visitor, const Expr& expr) { visitor->VisitExpr(expr); })
      .def("relax.PyExprVisitorVisitBinding",
           [](PyExprVisitor visitor, const Binding& binding) { visitor->VisitBinding(binding); })
      .def("relax.PyExprVisitorVisitBindingBlock",
           [](PyExprVisitor visitor, const BindingBlock& block) {
             visitor->VisitBindingBlock(block);
           })
      .def("relax.PyExprVisitorVisitVarDef",
           [](PyExprVisitor visitor, const Var& var) { visitor->VisitVarDef(var); })
      .def("relax.ExprVisitorVisitExpr",
           [](PyExprVisitor visitor, const Expr& expr) { visitor->ExprVisitor::VisitExpr(expr); })
      .def("relax.ExprVisitorVisitBinding",
           [](PyExprVisitor visitor, const Binding& binding) {
             if (const auto* ptr = binding.as<VarBindingObj>()) {
               visitor->ExprVisitor::VisitBinding_(ptr);
             } else if (const auto* ptr = binding.as<MatchCastObj>()) {
               visitor->ExprVisitor::VisitBinding_(ptr);
             } else {
               TVM_FFI_THROW(InternalError) << "unreachable";
             }
           })
      .def("relax.ExprVisitorVisitBindingBlock",
           [](PyExprVisitor visitor, const BindingBlock& block) {
             if (const auto* ptr = block.as<DataflowBlockObj>()) {
               visitor->ExprVisitor::VisitBindingBlock_(ptr);
             } else if (const auto* ptr = block.as<BindingBlockObj>()) {
               visitor->ExprVisitor::VisitBindingBlock_(ptr);
             } else {
               TVM_FFI_THROW(TypeError) << "Invalid type: " << block->GetTypeKey();
             }
           })
      .def("relax.ExprVisitorVisitVarDef",
           [](PyExprVisitor visitor, const Var& var) {
             if (const auto* node = var.as<DataflowVarObj>()) {
               visitor->ExprVisitor::VisitVarDef_(node);
             } else if (const auto* node = var.as<VarObj>()) {
               visitor->ExprVisitor::VisitVarDef_(node);
             } else {
               TVM_FFI_THROW(TypeError) << "Invalid type: " << var->GetTypeKey();
             }
           })
      .def("relax.ExprVisitorVisitLoc",
           [](PyExprVisitor visitor, const Location& loc) { visitor->ExprVisitor::VisitLoc(loc); })
      .def("relax.ExprVisitorVisitExprFallback",
           [](PyExprVisitor visitor, const Expr& expr) {
             visitor->ExprVisitor::VisitExprFallback_(expr.get());
           })
      .def("relax.MakePyExprMutator", PyExprMutator::MakePyExprMutator)
      .def("relax.PyExprMutatorVisitExpr",
           [](PyExprMutator mutator, const Expr& expr) { return mutator->VisitExpr(expr); })
      .def("relax.PyExprMutatorVisitBinding",
           [](PyExprMutator mutator, const Binding& binding) { mutator->VisitBinding(binding); })
      .def("relax.PyExprMutatorVisitBindingBlock",
           [](PyExprMutator mutator, const BindingBlock& block) {
             return mutator->VisitBindingBlock(block);
           })
      .def("relax.PyExprMutatorVisitVarDef",
           [](PyExprMutator mutator, const Var& var) { return mutator->VisitVarDef(var); })
      .def("relax.ExprMutatorVisitExpr",
           [](PyExprMutator mutator, const Expr& expr) {
             return mutator->ExprMutator::VisitExpr(expr);
           })
      .def("relax.ExprMutatorVisitBinding",
           [](PyExprMutator mutator, const Binding& binding) {
             if (const auto* ptr = binding.as<VarBindingObj>()) {
               return mutator->ExprMutator::VisitBinding_(ptr);
             } else if (const auto* ptr = binding.as<MatchCastObj>()) {
               return mutator->ExprMutator::VisitBinding_(ptr);
             } else {
               TVM_FFI_THROW(InternalError) << "unreachable";
             }
           })
      .def("relax.ExprMutatorVisitBindingBlock",
           [](PyExprMutator mutator, const BindingBlock& block) {
             if (const auto* node = block.as<DataflowBlockObj>()) {
               return mutator->ExprMutator::VisitBindingBlock_(node);
             } else if (const auto* node = block.as<BindingBlockObj>()) {
               return mutator->ExprMutator::VisitBindingBlock_(node);
             } else {
               TVM_FFI_THROW(TypeError) << "Invalid type: " << block->GetTypeKey();
             }
           })
      .def("relax.ExprMutatorVisitVarDef",
           [](PyExprMutator mutator, const Var& var) {
             if (const auto* node = var.as<DataflowVarObj>()) {
               return mutator->ExprMutator::VisitVarDef_(node);
             } else if (const auto* node = var.as<VarObj>()) {
               return mutator->ExprMutator::VisitVarDef_(node);
             } else {
               TVM_FFI_THROW(TypeError) << "Invalid type: " << var->GetTypeKey();
             }
           })
      .def("relax.ExprMutatorVisitExprFallback",
           [](PyExprMutator mutator, const Expr& expr) {
             return mutator->ExprMutator::VisitExprFallback_(expr.get());
           })
      .def(
          "relax.PyExprMutatorVisitExprPostOrder",
          [](PyExprMutator mutator, const Expr& expr) { return mutator->VisitExprPostOrder(expr); })
      .def("relax.PyExprMutatorVisitWithNewScope",
           [](PyExprMutator mutator, const Expr& expr) { return mutator->VisitWithNewScope(expr); })
      .def("relax.PyExprMutatorLookupBinding",
           [](PyExprMutator mutator, const Var& var) { return mutator->LookupBinding(var); })
      .def("relax.PyExprMutatorWithType",
           [](PyExprMutator mutator, Var var, Type ty) { return mutator->WithType(var, ty); })
      .def("relax.PyExprMutatorSetVarRemap",
           [](PyExprMutator mutator, Var old_var, Var new_var) {
             mutator->var_remap_.insert_or_assign(old_var, new_var);
             return new_var;
           })
      .def("relax.PyExprMutatorGetVarRemap",
           [](PyExprMutator mutator, Var var) -> ffi::Optional<Var> {
             auto it = mutator->var_remap_.find(var);
             if (it != mutator->var_remap_.end()) return it->second;
             return std::nullopt;
           });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  PyExprVisitorObj::RegisterReflection();
  PyExprMutatorObj::RegisterReflection();
}

}  // namespace relax
}  // namespace tvm
