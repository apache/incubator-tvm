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
#ifndef TVM_S_TIR_SCHEDULE_IR_COMPARATOR_H_
#define TVM_S_TIR_SCHEDULE_IR_COMPARATOR_H_

#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/stmt.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

using ExprComparator = tvm::ExprFunctor<bool(const Expr& n, const PrimExpr& other)>;
using StmtComparator = StmtFunctor<bool(const Stmt& n, const Stmt& other)>;

/*! \brief Deep comparison to check if two IR ASTs are equivalent for tensorization*/
class TensorizeComparator : public ExprComparator, public StmtComparator {
 public:
  /*!
   * \brief Constructor of TensorizeComparator
   * \param assert_mode Whether to raise an error if the two IR ASTs do not match.
   * \param lhs_mod The IRModule of the LHS. This is used for error reporting.
   */
  explicit TensorizeComparator(IRModule lhs_mod, bool assert_mode = true)
      : lhs_mod_(std::move(lhs_mod)), assert_mode_(assert_mode) {}

  bool Dispatch(const Expr& n, const PrimExpr& other) override;
  bool Dispatch(const Stmt& n, const Stmt& other) override;

  bool Dispatch_(const CallObj* op, const PrimExpr& other) override;
  bool Dispatch_(const ForObj* op, const Stmt& other) override;
  bool Dispatch_(const SeqStmtObj* op, const Stmt& other) override;
  bool Dispatch_(const TensorStoreObj* op, const Stmt& other) override;
  bool Dispatch_(const SBlockRealizeObj* op, const Stmt& other) override;
  bool Dispatch_(const SBlockObj* op, const Stmt& other) override;

  bool Dispatch_(const prim::LShiftObj* op, const PrimExpr& other) override;
  bool Dispatch_(const prim::RShiftObj* op, const PrimExpr& other) override;
  bool Dispatch_(const prim::BitwiseAndObj* op, const PrimExpr& other) override;
  bool Dispatch_(const prim::BitwiseOrObj* op, const PrimExpr& other) override;
  bool Dispatch_(const prim::BitwiseXorObj* op, const PrimExpr& other) override;
  bool Dispatch_(const prim::BitwiseNotObj* op, const PrimExpr& other) override;
  bool Dispatch_(const AddObj* op, const PrimExpr& other) override;
  bool Dispatch_(const SubObj* op, const PrimExpr& other) override;
  bool Dispatch_(const MulObj* op, const PrimExpr& other) override;
  bool Dispatch_(const DivObj* op, const PrimExpr& other) override;
  bool Dispatch_(const ModObj* op, const PrimExpr& other) override;
  bool Dispatch_(const EQObj* op, const PrimExpr& other) override;
  bool Dispatch_(const NEObj* op, const PrimExpr& other) override;
  bool Dispatch_(const LTObj* op, const PrimExpr& other) override;
  bool Dispatch_(const LEObj* op, const PrimExpr& other) override;
  bool Dispatch_(const GTObj* op, const PrimExpr& other) override;
  bool Dispatch_(const GEObj* op, const PrimExpr& other) override;
  bool Dispatch_(const AndObj* op, const PrimExpr& other) override;
  bool Dispatch_(const OrObj* op, const PrimExpr& other) override;
  bool Dispatch_(const MinObj* op, const PrimExpr& other) override;
  bool Dispatch_(const MaxObj* op, const PrimExpr& other) override;
  bool Dispatch_(const FloorDivObj* op, const PrimExpr& other) override;
  bool Dispatch_(const FloorModObj* op, const PrimExpr& other) override;
  bool Dispatch_(const IntImmObj* op, const PrimExpr& other) override;
  bool Dispatch_(const FloatImmObj* op, const PrimExpr& other) override;
  bool Dispatch_(const CastObj* op, const PrimExpr& other) override;
  bool Dispatch_(const VarObj* op, const PrimExpr& other) override;
  bool Dispatch_(const TensorLoadObj* op, const PrimExpr& other) override;
  bool Dispatch_(const SelectObj* op, const PrimExpr& other) override;

  /*! \brief Map from RHS tensor to LHS tensor */
  std::unordered_map<TensorVar, TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> rhs_tensor_map_;
  /*! \brief Base indices of the LHS tensor. */
  std::unordered_map<TensorVar, std::vector<PrimExpr>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      tensor_indices_;

 protected:
  bool DefEqual(const Var& lhs, const Var& rhs);
  bool CompareExpr(const Expr& lhs, const Expr& rhs);
  virtual bool CompareTensor(const TensorVar& lhs, const TensorVar& rhs);
  bool CompareTensorRegion(const TensorRegion& lhs, const TensorRegion& rhs);
  bool CompareAnnotation(const std::pair<ffi::String, ffi::Any>& lhs,
                         const std::pair<ffi::String, ffi::Any>& rhs);
  bool CompareAnnotationMap(const ffi::Map<ffi::String, ffi::Any>& lhs,
                            const ffi::Map<ffi::String, ffi::Any>& rhs);
  template <typename T>
  bool CompareTensorAccess(const T* lhs, const T* rhs);
  template <typename T, typename Self, typename F>
  bool CompareArray(const ffi::Array<T>& lhs, const ffi::Array<T>& rhs, F Self::* cmp);
  bool CompareRange(const Range& lhs, const Range& rhs);
  bool CompareIterVar(const IterVar& lhs, const IterVar& rhs);
  void EmitError(const std::string& error_message);

  /*! \brief IRModule of the LHS stmt. */
  IRModule lhs_mod_;
  /*! \brief Whether assertion mode is enabled. */
  bool assert_mode_;
  /*! \brief Whether it is visiting the scope block (the outermost block). */
  bool is_scope_block = true;
  /*! \brief The arithmetic analyzer for comparing LHS and RHS */
  sym::Analyzer analyzer_;
  /*!
   * \brief The arithmetic analyzer for simplifying expressions on LHS.
   *  This analyzer only contains the domains of the iterators on LHS.
   */
  sym::Analyzer lhs_analyzer_;
  /*! \brief Additional error messages. Only used when assert_mode is true. */
  std::vector<std::string> error_messages_;
  // variable remap if any
  std::unordered_map<ffi::ObjectRef, ffi::ObjectRef, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      equal_map_;
};

/*!
 * \brief IR comparator for auto tensorization.
 * This comparator is used to extract correspondence between the IR of the workload (LHS) and the
 * tensor intrin (RHS). Unlike `TensorizeComparator`, this comparator has relaxed requirements
 * during comparison. It ignores the loop structure (number of loops and their extents) and tensor
 * indices. It only requires the LHS and the RHS to have the same arithmetic operations and the same
 * dtype. With such relaxed requirements, workloads that can only match the tensor intrin after
 * certain transformations (e.g. im2col for conv2d) are allowed for auto tensorization.
 */
class AutoTensorizeComparator : public TensorizeComparator {
 public:
  explicit AutoTensorizeComparator(const IRModule& lhs_mod)
      : TensorizeComparator(lhs_mod, /* assert_mode=*/false) {}

 private:
  bool DispatchDefault_(const ffi::Object* op, const PrimExpr& other) override;
  bool DispatchDefault_(const ffi::Object* op, const Stmt& other) override;

  bool Dispatch_(const SBlockObj* op, const Stmt& other) override;
  bool Dispatch_(const TensorStoreObj* op, const Stmt& other) override;

  bool Dispatch_(const TensorLoadObj* op, const PrimExpr& other) override;

  bool CompareTensor(const TensorVar& lhs, const TensorVar& rhs) override;
  template <typename T>
  bool CompareTensorAccess(const T* lhs, const T* rhs);

 public:
  // Additional information extracted from LHS (the workload) and RHS (the tensor intrin).

  /*! \brief SBlock iters in the LHS stmt. */
  std::vector<IterVar> lhs_iters_;
  /*! \brief SBlock iters in the RHS stmt. */
  std::vector<IterVar> rhs_iters_;
  /*! \brief The tensor and its access indices in the LHS stmt. */
  std::unordered_map<TensorVar, ffi::Array<PrimExpr>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      lhs_tensor_indices_map_;
  /*! \brief The tensor and its access indices in the RHS stmt. */
  std::unordered_map<TensorVar, ffi::Array<PrimExpr>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      rhs_tensor_indices_map_;
  /*! \brief Map from LHS tensor to RHS tensor */
  std::unordered_map<TensorVar, TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> lhs_tensor_map_;

 private:
  /*! \brief The domain of the inner block iters. */
  ffi::Map<Var, sym::IntSet> inner_iter_dom_map_;
};

}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_SCHEDULE_IR_COMPARATOR_H_
