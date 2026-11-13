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
 * \file tvm/relax/dataflow_pattern_functor.h
 * \brief Functors and visitors for dataflow patterns.
 */
#ifndef TVM_RELAX_DATAFLOW_PATTERN_FUNCTOR_H_
#define TVM_RELAX_DATAFLOW_PATTERN_FUNCTOR_H_

#include <tvm/relax/dataflow_pattern.h>

#include <unordered_set>
#include <utility>

namespace tvm {
namespace relax {

/*!
 * \brief A dynamical functor that dispatches on in the first DFPattern argument.
 *
 * \tparam FType function signature
 *  This type is only defined for FType with function signature R(const DFPattern&,
 * Args...)
 */
template <typename FType>
class DFPatternFunctor;

// functions to be overriden.
#define DFPATTERN_FUNCTOR_DEFAULT                                   \
  {                                                                 \
    return VisitDFPatternDefault_(op, std::forward<Args>(args)...); \
  }

#define RELAX_DFPATTERN_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template SetDispatch<OP>([](const ffi::ObjectRef& n, TSelf* self, Args... args) {      \
    return self->VisitDFPattern_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class DFPatternFunctor<R(const DFPattern& n, Args...)> {
 private:
  using TSelf = DFPatternFunctor<R(const DFPattern& n, Args...)>;
  using FType = tvm::ObjectFunctor<R(const ffi::ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief virtual destructor */
  virtual ~DFPatternFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const DFPattern& n, Args... args) {
    return VisitDFPattern(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitDFPattern(const DFPattern& n, Args... args) {
    TVM_FFI_ICHECK(n.defined());
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitDFPattern_(const OrPatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const AndPatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const NotPatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const AttrPatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const CallPatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const ConstantPatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const DataTypePatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const ExprPatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const FunctionPatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const ShapePatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const TupleGetItemPatternObj* op,
                            Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const TuplePatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const TypePatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const WildcardPatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const VarPatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;

  virtual R VisitDFPattern_(const DataflowVarPatternObj* op,
                            Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const GlobalVarPatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const ExternFuncPatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const PrimArrPatternObj* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const UnorderedTuplePatternObj* op,
                            Args... args) DFPATTERN_FUNCTOR_DEFAULT;

  virtual R VisitDFPatternDefault_(const ffi::Object* op, Args...) {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << op->GetTypeKey();
    throw;
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(OrPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(AndPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(NotPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(AttrPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(CallPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(ConstantPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(DataTypePatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(ExprPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(FunctionPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(ShapePatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(TupleGetItemPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(TuplePatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(TypePatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(WildcardPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(VarPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(DataflowVarPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(GlobalVarPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(ExternFuncPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(PrimArrPatternObj);
    RELAX_DFPATTERN_FUNCTOR_DISPATCH(UnorderedTuplePatternObj);
    vtable.Finalize();
    return vtable;
  }
};

/*!
 * \brief A simple visitor wrapper around DFPatternFunctor.
 *  Recursively visit the content.
 *
 *  DFPatternVisitor treats the Pattern as dataflow graph,and only visit each Expr node once.
 */
class DFPatternVisitor : public DFPatternFunctor<void(const DFPattern&)> {
 public:
  void VisitDFPattern(const DFPattern& pattern) override;
  void VisitDFPattern_(const OrPatternObj* op) override;
  void VisitDFPattern_(const AndPatternObj* op) override;
  void VisitDFPattern_(const NotPatternObj* op) override;
  void VisitDFPattern_(const AttrPatternObj* op) override;
  void VisitDFPattern_(const CallPatternObj* op) override;
  void VisitDFPattern_(const ConstantPatternObj* op) override;
  void VisitDFPattern_(const DataTypePatternObj* op) override;
  void VisitDFPattern_(const ExprPatternObj* op) override;
  void VisitDFPattern_(const FunctionPatternObj* op) override;
  void VisitDFPattern_(const ShapePatternObj* op) override;
  void VisitDFPattern_(const TupleGetItemPatternObj* op) override;
  void VisitDFPattern_(const TuplePatternObj* op) override;
  void VisitDFPattern_(const TypePatternObj* op) override;
  void VisitDFPattern_(const WildcardPatternObj* op) override;
  void VisitDFPattern_(const VarPatternObj* op) override;

  void VisitDFPattern_(const DataflowVarPatternObj* op) override;
  void VisitDFPattern_(const GlobalVarPatternObj* op) override;
  void VisitDFPattern_(const ExternFuncPatternObj* op) override;
  void VisitDFPattern_(const PrimArrPatternObj* op) override;
  void VisitDFPattern_(const UnorderedTuplePatternObj* op) override;

 protected:
  // set of already-visited nodes
  std::unordered_set<const ffi::Object*> visited_;
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_DATAFLOW_PATTERN_FUNCTOR_H_
