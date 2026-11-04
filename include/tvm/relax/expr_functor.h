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
 * \file tvm/relax/expr_functor.h
 * \brief A more powerful visitor which enables defining arbitrary function
 * signatures with type based dispatch on first argument.
 */
#ifndef TVM_RELAX_EXPR_FUNCTOR_H_
#define TVM_RELAX_EXPR_FUNCTOR_H_

#include <tvm/ir/object_functor.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/relax/type_functor.h>
#include <tvm/tirx/function.h>

#include <unordered_map>
#include <utility>
namespace tvm {
namespace relax {

/*!
 * \brief A dynamical functor that dispatches on in the first Expr argument.
 *  You can use this as a more powerful Visitor, since it allows you to
 *  define function signatures of Visit Function.
 *
 * \sa tvm/ir_functor.h
 *
 * \tparam FType function signature
 *  This type is only defined for FType with function signature R(const Expr&,
 * Args...)
 */
template <typename FType>
class ExprFunctor;

// functions to be overriden.
#define EXPR_FUNCTOR_DEFAULT                                   \
  {                                                            \
    return VisitExprDefault_(op, std::forward<Args>(args)...); \
  }

#define EXPR_FUNCTOR_DISABLED                                                                    \
  final {                                                                                        \
    TVM_FFI_THROW(TypeError) << "Relax does not support " << op->GetTypeKey() << " expressions"; \
    throw;                                                                                       \
  }

#define RELAX_EXPR_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template SetDispatch<OP>([](const ffi::ObjectRef& n, TSelf* self, Args... args) { \
    return self->VisitExpr_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

#define PY_EXPR_VISITOR_DEFAULT(N, PY_FUNC, DEFAULT_FUNC) \
  {                                                       \
    if (PY_FUNC != nullptr)                               \
      PY_FUNC(N);                                         \
    else                                                  \
      DEFAULT_FUNC;                                       \
  }

#define PY_EXPR_MUTATOR_DEFAULT(N, PY_FUNC, DEFAULT_FUNC, RET_TYPE) \
  {                                                                 \
    if (PY_FUNC != nullptr) {                                       \
      RET_TYPE ret = PY_FUNC(N).cast<RET_TYPE>();                   \
      return ret;                                                   \
    } else {                                                        \
      return DEFAULT_FUNC;                                          \
    }                                                               \
  }

#define PY_EXPR_VISITOR_DISPATCH(OP, PY_FUNC)                                \
  vtable.template SetDispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) { \
    if (self->PY_FUNC != nullptr)                                            \
      self->PY_FUNC(n);                                                      \
    else                                                                     \
      self->VisitExpr_(static_cast<const OP*>(n.get()));                     \
  });

#define PY_EXPR_MUTATOR_DISPATCH(OP, PY_FUNC)                                \
  vtable.template SetDispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) { \
    if (self->PY_FUNC != nullptr) {                                          \
      Expr expr = self->PY_FUNC(n).cast<Expr>();                             \
      return expr;                                                           \
    } else {                                                                 \
      return self->VisitExpr_(static_cast<const OP*>(n.get()));              \
    }                                                                        \
  });

#define PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(OP)                              \
  post_order_vtable.template SetDispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) { \
    return self->VisitExprPostOrder_(static_cast<const OP*>(n.get()));                  \
  });

template <typename R, typename... Args>
class ExprFunctor<R(const Expr& n, Args...)> {
 private:
  using TSelf = ExprFunctor<R(const Expr& n, Args...)>;
  using FType = tvm::ObjectFunctor<R(const ffi::ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~ExprFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Expr& n, Args... args) { return VisitExpr(n, std::forward<Args>(args)...); }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitExpr(const Expr& n, Args... args) {
    TVM_FFI_ICHECK(n.defined())
        << "Found null pointer node while traversing AST. The previous pass may "
           "have generated invalid data.";
    static FType vtable = InitVTable();
    if (vtable.CanDispatch(n)) {
      return vtable(n, this, std::forward<Args>(args)...);
    }
    return VisitExprFallback_(n.get(), std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  // NOTE: cross dialect calls are invoked through global var
  // We do not expect inline tirx::Function to appear in relax IR.
  virtual R VisitExpr_(const GenericConstObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TupleObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const VarObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const DataflowVarObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ShapeExprObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ExternFuncObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const GlobalVarObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FunctionObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const CallObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TensorLoadObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const tvm::IntImmObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const tvm::FloatImmObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const SeqExprObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const IfExprObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const OpObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TupleGetItemObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const StringImmObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const DataTypeImmObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::LetObj* op, Args...) EXPR_FUNCTOR_DISABLED;
  virtual R VisitExpr_(const prim::AddObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::LShiftObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::RShiftObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::BitwiseAndObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::BitwiseOrObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::BitwiseXorObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::BitwiseNotObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::SubObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::MulObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::DivObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::ModObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::FloorDivObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::FloorModObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::MinObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::MaxObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::EQObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::NEObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::LTObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::LEObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::GTObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::GEObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::AndObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::OrObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::CastObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::NotObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::SelectObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::RampObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::BroadcastObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::ShuffleObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExprFallback_(const ExprObj* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExprDefault_(const ffi::Object* op, Args...) {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << op->GetTypeKey();
    throw;
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAX_EXPR_FUNCTOR_DISPATCH(GenericConstObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(TupleObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(VarObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(DataflowVarObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(ShapeExprObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(ExternFuncObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(GlobalVarObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(FunctionObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(CallObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::LetObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(TensorLoadObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::AddObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::LShiftObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::RShiftObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::BitwiseAndObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::BitwiseOrObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::BitwiseXorObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::BitwiseNotObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::SubObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::MulObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::DivObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::ModObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::FloorDivObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::FloorModObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::MinObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::MaxObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::EQObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::NEObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::LTObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::LEObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::GTObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::GEObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::AndObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::OrObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::CastObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::NotObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::SelectObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::RampObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::BroadcastObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(prim::ShuffleObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(tvm::IntImmObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(tvm::FloatImmObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(SeqExprObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(IfExprObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(OpObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(TupleGetItemObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(StringImmObj);
    RELAX_EXPR_FUNCTOR_DISPATCH(DataTypeImmObj);
    vtable.Finalize();
    return vtable;
  }
};

/*!
 * \brief A simple visitor wrapper around ExprFunctor.
 *  Recursively visit the content.
 */
class ExprVisitor : public ExprFunctor<void(const Expr&)> {
 public:
  /*!
   * \brief Generic dispatcher for Expr.
   * \param expr The expr to be visited.
   */
  void VisitExpr(const Expr& expr) override;
  // specific leaf level visitor functions
  void VisitExpr_(const GenericConstObj* op) override;
  void VisitExpr_(const TupleObj* op) override;
  void VisitExpr_(const VarObj* op) override;
  void VisitExpr_(const DataflowVarObj* op) override;
  void VisitExpr_(const ShapeExprObj* op) override;
  void VisitExpr_(const ExternFuncObj* op) override;
  void VisitExpr_(const GlobalVarObj* op) override;
  void VisitExpr_(const FunctionObj* op) override;
  void VisitExpr_(const CallObj* op) override;
  void VisitExpr_(const TensorLoadObj* op) override;
  void VisitExpr_(const tvm::IntImmObj* op) override;
  void VisitExpr_(const tvm::FloatImmObj* op) override;
  void VisitExpr_(const SeqExprObj* op) override;
  void VisitExpr_(const IfExprObj* op) override;
  void VisitExpr_(const OpObj* op) override;
  void VisitExpr_(const TupleGetItemObj* op) override;
  void VisitExpr_(const StringImmObj* op) override;
  void VisitExpr_(const DataTypeImmObj* op) override;
  void VisitExpr_(const prim::AddObj* op) override;
  void VisitExpr_(const prim::LShiftObj* op) override;
  void VisitExpr_(const prim::RShiftObj* op) override;
  void VisitExpr_(const prim::BitwiseAndObj* op) override;
  void VisitExpr_(const prim::BitwiseOrObj* op) override;
  void VisitExpr_(const prim::BitwiseXorObj* op) override;
  void VisitExpr_(const prim::BitwiseNotObj* op) override;
  void VisitExpr_(const prim::SubObj* op) override;
  void VisitExpr_(const prim::MulObj* op) override;
  void VisitExpr_(const prim::DivObj* op) override;
  void VisitExpr_(const prim::ModObj* op) override;
  void VisitExpr_(const prim::FloorDivObj* op) override;
  void VisitExpr_(const prim::FloorModObj* op) override;
  void VisitExpr_(const prim::MinObj* op) override;
  void VisitExpr_(const prim::MaxObj* op) override;
  void VisitExpr_(const prim::EQObj* op) override;
  void VisitExpr_(const prim::NEObj* op) override;
  void VisitExpr_(const prim::LTObj* op) override;
  void VisitExpr_(const prim::LEObj* op) override;
  void VisitExpr_(const prim::GTObj* op) override;
  void VisitExpr_(const prim::GEObj* op) override;
  void VisitExpr_(const prim::AndObj* op) override;
  void VisitExpr_(const prim::OrObj* op) override;
  void VisitExpr_(const prim::CastObj* op) override;
  void VisitExpr_(const prim::NotObj* op) override;
  void VisitExpr_(const prim::SelectObj* op) override;
  void VisitExpr_(const prim::RampObj* op) override;
  void VisitExpr_(const prim::BroadcastObj* op) override;
  void VisitExpr_(const prim::ShuffleObj* op) override;
  void VisitExprFallback_(const ExprObj* op) override;

  /*!
   * \brief Generic dispatcher for bindings.
   * \param binding The binding to be visited.
   */
  virtual void VisitBinding(const Binding& binding);
  // specific leaf level visitor functions
  virtual void VisitBinding_(const VarBindingObj* binding);
  virtual void VisitBinding_(const MatchCastObj* binding);
  // second level dispatching based on binding value type.
  // these dispatching functions get called from first-level dispatch on VarBinding
  virtual void VisitBinding_(const VarBindingObj* binding, const GenericConstObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const TupleObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const VarObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const DataflowVarObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const ShapeExprObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const ExternFuncObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const GlobalVarObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const FunctionObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const CallObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const SeqExprObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const IfExprObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const OpObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const TupleGetItemObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const ExprObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const StringImmObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const DataTypeImmObj* val);
  /*!
   * \brief Generic dispatcher for binding blocks.
   * \param block The binding block to be visited.
   */
  virtual void VisitBindingBlock(const BindingBlock& block);
  // specific leaf level visitor functions
  virtual void VisitBindingBlock_(const BindingBlockObj* block);
  virtual void VisitBindingBlock_(const DataflowBlockObj* block);

  /*!
   * \brief Generic dispatcher for visiting the var definition site.
   * \param var The var to be visited.
   * \note VisitExpr_(const VarObj*) will only visit the usage site of an Var
   */
  virtual void VisitVarDef(const Var& var);

  /*!
   * \brief Visit ty may recursively contain Expr/PrimExpr.
   *
   * By default, this function recurse into type such as
   * TensorType and ShapeType and call VisitExpr/VisitTypePrimExprField
   * accordingly. It does not
   * recurse into FunctionType as it does
   * not contain Expr defined in the current scope.
   *
   * Pass writers can overload this function to change to other behaviors.
   * For example, if we are not interested in Expr in Type, we can
   * override this function by a no-op.
   *
   * \param ty Input type field.
   */
  virtual void VisitExprDepTypeField(const Type& ty);

  // specific leaf level visitor functions
  virtual void VisitVarDef_(const VarObj* var);
  virtual void VisitVarDef_(const DataflowVarObj* var);

  virtual void VisitLoc(const Location& loc);
  virtual void VisitTypePrimExprField(const PrimExpr& expr);

 private:
  using TSelf = ExprVisitor;
  using VisitBindingVTable = tvm::ObjectFunctor<void(const ffi::ObjectRef& n, ExprVisitor* self,
                                                     const VarBindingObj* binding)>;
  // initialize the vtable.
  static VisitBindingVTable InitVisitBindingVTable();
  /*!
   * \brief Private internal type field visitor.
   *
   *  Support default visiting of type field and recursive into
   *  their Expr fields.
   *
   *  We use component instead of sub-classing so there can be other
   *  joint inheritance between ExprVisitor and TypeVisitor.
   */
  class DefaultTypeFieldVisitor : public TypeVisitor {
   public:
    explicit DefaultTypeFieldVisitor(ExprVisitor* parent);

    // Override defaults in type visitor.
    void VisitTypeExprField(const Expr& expr) final;
    void VisitTypeExprField(const PrimExpr& expr) final;
    void VisitType_(const FuncTypeObj* op) final;

   private:
    ExprVisitor* parent_;
  };
  // This visitor is not visible to child classes and only
  // used to supported default visiting behavior.
  DefaultTypeFieldVisitor default_tyfield_visitor_{this};
};

void PostOrderVisit(const Expr& node, std::function<void(const Expr&)> fvisit);

/*!
 * \brief A mutator works in unnormalized form.
 *
 * ExprMutatorBase expects input AST to be in the unnormalized form, i.e., ty
 * of expressions can be nullptr, and the expressions may nest(and as a result the AST is not in
 * ANF).
 */

class ExprMutatorBase : public ExprFunctor<Expr(const Expr&)> {
 public:
  Expr VisitExpr(const Expr& expr) override;
  Expr VisitExpr_(const GenericConstObj* op) override;
  Expr VisitExpr_(const TupleObj* op) override;
  Expr VisitExpr_(const VarObj* op) override;
  Expr VisitExpr_(const DataflowVarObj* op) override;
  Expr VisitExpr_(const ShapeExprObj* op) override;
  Expr VisitExpr_(const ExternFuncObj* op) override;
  Expr VisitExpr_(const GlobalVarObj* op) override;
  Expr VisitExpr_(const FunctionObj* op) override;
  Expr VisitExpr_(const CallObj* op) override;
  Expr VisitExpr_(const TensorLoadObj* op) override;
  Expr VisitExpr_(const tvm::IntImmObj* op) override;
  Expr VisitExpr_(const tvm::FloatImmObj* op) override;
  Expr VisitExpr_(const SeqExprObj* op) override;
  Expr VisitExpr_(const IfExprObj* op) override;
  Expr VisitExpr_(const OpObj* op) override;
  Expr VisitExpr_(const TupleGetItemObj* op) override;
  Expr VisitExpr_(const StringImmObj* op) override;
  Expr VisitExpr_(const DataTypeImmObj* op) override;
  Expr VisitExpr_(const prim::AddObj* op) override;
  Expr VisitExpr_(const prim::LShiftObj* op) override;
  Expr VisitExpr_(const prim::RShiftObj* op) override;
  Expr VisitExpr_(const prim::BitwiseAndObj* op) override;
  Expr VisitExpr_(const prim::BitwiseOrObj* op) override;
  Expr VisitExpr_(const prim::BitwiseXorObj* op) override;
  Expr VisitExpr_(const prim::BitwiseNotObj* op) override;
  Expr VisitExpr_(const prim::SubObj* op) override;
  Expr VisitExpr_(const prim::MulObj* op) override;
  Expr VisitExpr_(const prim::DivObj* op) override;
  Expr VisitExpr_(const prim::ModObj* op) override;
  Expr VisitExpr_(const prim::FloorDivObj* op) override;
  Expr VisitExpr_(const prim::FloorModObj* op) override;
  Expr VisitExpr_(const prim::MinObj* op) override;
  Expr VisitExpr_(const prim::MaxObj* op) override;
  Expr VisitExpr_(const prim::EQObj* op) override;
  Expr VisitExpr_(const prim::NEObj* op) override;
  Expr VisitExpr_(const prim::LTObj* op) override;
  Expr VisitExpr_(const prim::LEObj* op) override;
  Expr VisitExpr_(const prim::GTObj* op) override;
  Expr VisitExpr_(const prim::GEObj* op) override;
  Expr VisitExpr_(const prim::AndObj* op) override;
  Expr VisitExpr_(const prim::OrObj* op) override;
  Expr VisitExpr_(const prim::CastObj* op) override;
  Expr VisitExpr_(const prim::NotObj* op) override;
  Expr VisitExpr_(const prim::SelectObj* op) override;
  Expr VisitExpr_(const prim::RampObj* op) override;
  Expr VisitExpr_(const prim::BroadcastObj* op) override;
  Expr VisitExpr_(const prim::ShuffleObj* op) override;
  Expr VisitExprFallback_(const ExprObj* op) override;

  /*!
   * \brief Mutate BindingBlock.
   * \param block The binding block to be visited.
   * \return The binding block after transformation.
   */
  virtual BindingBlock VisitBindingBlock(const BindingBlock& block);

  /*!
   * \brief Used to visit the PrimExpr inside of dependent type fields.
   *
   * Can be overloaded to transform shape expressions without changing
   * runtime PrimExpr traversal.
   */
  virtual PrimExpr VisitTypePrimExprField(const PrimExpr& expr);

  /*!
   * \brief Visit ty that may recursively contain Expr/PrimExpr.
   *
   * By default, this function recurse into type such as
   * TensorType and ShapeType and call VisitExpr/VisitTypePrimExprField
   * accordingly. It does not
   * recurse into FunctionType as it does
   * not contain Expr defined in the current scope.
   *
   * Pass writers can overload this function to change to other behaviors.
   * For example, if in Expr in Type won't change, we can
   * override this function by an identity function.
   *
   * \param ty Input type field.
   * \return The updated type.
   */
  virtual Type VisitExprDepTypeField(const Type& ty);

 protected:
  /*!
   * \brief Check whether VisitExprDepTypeField change ty.
   * \return Whether type changed.
   * \note This function is used by mutator implementations to check if
   *       previous Expr update will trigger a change in ty.
   *       If change is detected, the implementation can generate a fresh
   *       node without ty, and trigger normalizer to re-derive.
   */
  bool VisitAndCheckTypeFieldUnchanged(const ffi::ObjectRef& ty) {
    if (const TypeObj* ty_node = ty.as<TypeObj>()) {
      Type type = ffi::GetRef<Type>(ty_node);
      return type.as<MissingType>().has_value() || this->VisitExprDepTypeField(type).same_as(ty);
    } else {
      return true;
    }
  }

 private:
  /*!
   * \brief Private internal type field visitor to support
   *  Default visiting of type field and recursive into their Expr fields.
   *
   *  We use component instead of sub-classing so there can be other
   *  joint inheritance between ExprMutator and TypeMutator.
   */
  class DefaultTypeFieldMutator : public TypeMutator {
   public:
    explicit DefaultTypeFieldMutator(ExprMutatorBase* parent);

    // Override defaults in type visitor.
    Expr VisitTypeExprField(const Expr& expr) final;
    PrimExpr VisitTypeExprField(const PrimExpr& expr) final;
    Type VisitType_(const FuncTypeObj* op) final;

   private:
    ExprMutatorBase* parent_;
  };
  // This visitor is not visible to child classes and only
  // used to supported default visiting behavior.
  DefaultTypeFieldMutator default_tyfield_mutator_{this};
};

/*!
 * \brief A mutator works in normal form.
 *
 * ExprMutator expects input AST to be in the normal form, i.e., the expressions are normalized(no
 * nesting and hence the AST is in ANF), and all ty of expressions are
 * available.
 */
class ExprMutator : public ExprMutatorBase {
 public:
  using ExprMutatorBase::VisitExpr_;

  ExprMutator(ffi::Optional<IRModule> mod = std::nullopt) : builder_(BlockBuilder::Create(mod)) {}
  Expr VisitExpr(const Expr& expr) override;
  Expr VisitExpr_(const VarObj* op) override;
  Expr VisitExpr_(const DataflowVarObj* op) override;
  Expr VisitExpr_(const FunctionObj* op) override;
  Expr VisitExpr_(const SeqExprObj* op) override;
  Expr VisitExpr_(const IfExprObj* op) override;

  /*!
   * \brief Generic dispatcher for bindings.
   * \param binding The binding to be visited.
   */
  virtual void VisitBinding(const Binding& binding);
  // specific leaf level visitor functions
  virtual void VisitBinding_(const VarBindingObj* binding);
  virtual void VisitBinding_(const MatchCastObj* binding);
  // second level dispatching based on binding value type.
  // these dispatching functions get called from first-level dispatch on VarBinding
  virtual void VisitBinding_(const VarBindingObj* binding, const GenericConstObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const TupleObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const VarObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const DataflowVarObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const ShapeExprObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const ExternFuncObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const GlobalVarObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const FunctionObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const CallObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const SeqExprObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const IfExprObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const OpObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const TupleGetItemObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const ExprObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const StringImmObj* val);
  virtual void VisitBinding_(const VarBindingObj* binding, const DataTypeImmObj* val);
  /*!
   * \brief Generic dispatcher for binding blocks.
   * \param block The binding block to be visited.
   * \return The binding block after transformation.
   */
  virtual BindingBlock VisitBindingBlock(const BindingBlock& block) override;  // NOLINT(*)
  // specific leaf level visitor functions
  virtual BindingBlock VisitBindingBlock_(const BindingBlockObj* block);
  virtual BindingBlock VisitBindingBlock_(const DataflowBlockObj* block);

  /*!
   * \brief Generic dispatcher for rewriting the var definition site.
   * \param var The var to be visited.
   * \return The var after post-order rewritten.
   * \note VisitExpr_(const VarObj*) will only visit the usage site of an Var
   */
  virtual Var VisitVarDef(const Var& var);
  // specific leaf level visitor functions
  virtual Var VisitVarDef_(const VarObj* var);
  virtual Var VisitVarDef_(const DataflowVarObj* var);

 protected:
  /*!
   * \brief Try to remit binding and bind it to a new_value
   *
   * This function is called after VisitExpr(binding->value) in
   * VisitBinding_(const VarBinding*).
   * It will try to reuse the current binding when the new value's shape/type
   * matches the original binding and no changes in var is needed.
   *
   * Otherwise, a new binding will be emitted to replace the var specified in
   * the current binding.
   */
  void ReEmitBinding(const VarBindingObj* binding, Expr new_value);

  /*!
   * \brief Rewrite the expr with a new scope, used in a Function's body.
   *
   * Visit an expression that may neither access variables from the
   * current scope, nor may export definitions into the current scope.
   *
   * \param body_expr The body to be visited.
   * \param params Optional parameters that are visible within the scope.
   * \return The expr after visiting.
   *
   * \note The body_expr must be an SeqExpr in the normal form.
   */
  Expr VisitWithNewScope(const Expr& body_expr,
                         ffi::Optional<ffi::Array<Var>> params = std::nullopt);

  /*!
   * \brief Rewrite the expr with a new scope, used in the branches of IfExpr.
   *
   * Visit an expression that may access variables from the current
   * scope, but may not export definitions into the current scope.
   *
   * \param body_expr The body to be visited.
   *
   * \return The expr after visiting.
   *
   * \sa VisitWithNewScope
   *
   * \note The body_expr must be an SeqExpr in the normal form.
   */
  Expr VisitWithInnerScope(const Expr& body_expr);

  /*!
   * \brief Look up the value bound to a variable.
   * \param var The var to be looked up.
   * \return The value bound to the input \p var.
   * \note For function parameters, this function returns std::nullopt.
   */
  ffi::Optional<Expr> LookupBinding(const Var& var);

  /*!
   * \brief Post-order rewrite a node and normalize.
   * \tparam T The node type to be rewritten.
   * \param op The node to be rewritten.
   * \return The node after post rewritten.
   */
  template <typename T>
  Expr VisitExprPostOrder_(const T* op) {
    return builder_->Normalize(ExprMutator::VisitExpr_(op));
  }

  /*!
   * \brief Create a new var with specified type if the original var's shape or type does not
   * match with the specified ones.
   * \param var The var to be updated.
   * \param ty The type to be updated.
   * \return The var filled with type information.
   */
  Var WithType(Var var, Type ty);

  /*! \brief Internal block builder to emit bindings during rewriting. */
  BlockBuilder builder_;

  /*! \brief Remap a var to a new var in use-site. */
  std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> var_remap_;

 private:
  using TSelf = ExprMutator;
  using VisitBindingVTable = tvm::ObjectFunctor<void(const ffi::ObjectRef& n, ExprMutator* self,
                                                     const VarBindingObj* binding)>;
  // initialize the vtable.
  static VisitBindingVTable InitVisitBindingVTable();
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_EXPR_FUNCTOR_H_
