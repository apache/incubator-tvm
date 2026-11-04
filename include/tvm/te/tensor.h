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
 * \file tvm/te/tensor.h
 * \brief Dataflow tensor object
 */
#ifndef TVM_TE_TENSOR_H_
#define TVM_TE_TENSOR_H_

#include <tvm/base.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/iter_var.h>
#include <tvm/sym/int_set.h>
#include <tvm/tirx/expr.h>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tvm {
namespace te {
using s_tir::IterVar;
using s_tir::IterVarObj;
using s_tir::IterVarType;
using s_tir::IterVarType2String;
using s_tir::kCommReduce;
using s_tir::kDataPar;
using s_tir::kOpaque;
using s_tir::kOrdered;
using s_tir::kParallelized;
using s_tir::kTensorized;
using s_tir::kThreadIndex;
using s_tir::kUnrolled;
using s_tir::kVectorized;
using sym::IntSet;

using namespace tvm::tirx;

// internal node container for Operation
class OperationObj;
class Tensor;

/*! \brief Operation that produces tensors */
class Operation : public ffi::ObjectRef {
 public:
  static constexpr bool _type_is_nullable = false;
  explicit Operation(ffi::UnsafeInit tag) : ffi::ObjectRef(tag) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const OperationObj* operator->() const;
  /*!
   * \brief get the i-th output of the operation.
   * \param i the output index.
   * \return The i-th output.
   */
  TVM_DLL Tensor output(size_t i) const;
  /*! \brief specify container node */
  using ContainerType = OperationObj;
};

/*! \brief Opaque construction-time node that represents a tensor. */
class TensorObj : public OpaqueExprObj {
 public:
  /*! \brief The shape of the tensor */
  ffi::Array<PrimExpr> shape;
  /*! \brief dtype in the content of the tensor */
  PrimType dtype = PrimType::Void();
  /*! \brief the source operation, can be None */
  ffi::Optional<Operation> op;
  /*! \brief the output index from source operation */
  int value_index{0};

  static void RegisterReflection();

  ffi::Array<PrimExpr> GetShape() const { return shape; }

  PrimType GetDataType() const { return dtype; }

  TVM_DLL ffi::String GetNameHint() const;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindConstTreeNode;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("te.Tensor", TensorObj, OpaqueExprObj);
};

/*!
 * \brief Tensor structure representing a possible input,
 *  or intermediate computation result.
 */
class Tensor : public OpaqueExpr {
 private:
  /*!
   * \brief Helper for indexing operations into tensors
   * \param indices The indices
   * \param support_negative_indices Whether to normalize indices in the case of negative indices.
   * \return the result expression representing tensor read.
   */
  inline PrimExpr IndexTensor(ffi::Array<PrimExpr> indices, bool support_negative_indices) const;

 public:
  TVM_DLL Tensor(ffi::Array<PrimExpr> shape, PrimType dtype, ffi::Optional<Operation> op,
                 int value_index);

  /*!
   * \brief check if two tensors equals each other.
   * \param other tensor to be checked.
   * \return whether the two tensors equals each other.
   */
  inline bool operator==(const Tensor& other) const;
  /*!
   * \brief check if two tensors are different.
   * \param other tensor to be checked.
   * \return whether the two tensors are different.
   */
  inline bool operator!=(const Tensor& other) const;
  /*! \return The dimension of the tensor */
  inline size_t ndim() const;
  /*!
   * \brief Take elements from the tensor
   * \param args The indices
   * \return the result expression representing tensor read.
   */
  template <typename... Args>
  inline PrimExpr operator()(Args&&... args) const {
    ffi::Array<PrimExpr> indices{std::forward<Args>(args)...};
    return operator()(indices);
  }
  /*!
   * \brief Take elements from the tensor
   * \param indices the indices.
   * \return the result expression representing tensor read.
   */
  TVM_DLL PrimExpr operator()(ffi::Array<PrimExpr> indices) const;
  /*!
   * \brief Take elements from the tensor
   * \param indices the indices.
   * \return the result expression representing tensor read.
   */
  TVM_DLL PrimExpr operator()(ffi::Array<PrimVar> indices) const;
  /*!
   * \brief Take elements from the tensor with support for negative indices.
   * \param args The indices
   * \return the result expression representing tensor read.
   */
  template <typename... Args>
  TVM_DLL PrimExpr IndexWithNegativeIndices(Args&&... args) const {
    ffi::Array<PrimExpr> indices{std::forward<Args>(args)...};
    return IndexWithNegativeIndices(indices);
  }
  /*!
   * \brief Take elements from the tensor with support for negative indices.
   * \param indices the indices.
   * \return the result expression representing tensor read.
   */
  TVM_DLL PrimExpr IndexWithNegativeIndices(ffi::Array<PrimExpr> indices) const;
  /*!
   * \brief Take elements from the tensor with support for negative indices.
   * \param indices the indices.
   * \return the result expression representing tensor read.
   */
  TVM_DLL PrimExpr IndexWithNegativeIndices(ffi::Array<PrimVar> indices) const;

  /*!
   * \brief data structure to represent a slice that fixes first k coordinates.
   *  This is used to enable syntax sugar of Tensor[x][y][z] to get the element.
   */
  class Slice {
   public:
    // construct via tensor and indices
    Slice(const Tensor& tensor, std::vector<PrimExpr> indices)
        : tensor_(tensor), indices_(indices) {}
    /*!
     * \brief get i-th slice from the current slice.
     * \param i the index of the coordinate
     * \return the subsequent slice.
     */
    inline Slice operator[](PrimExpr i) {
      std::vector<PrimExpr> other = indices_;
      other.emplace_back(i);
      return Slice(tensor_, other);
    }
    /*!
     * \brief Convert slice to expression.
     *  This is only valid when all the coordinates are fully specified.
     * \return the corresponding expression of this slice.
     */
    inline operator PrimExpr() const { return tensor_(indices_); }

   private:
    const Tensor& tensor_;
    std::vector<PrimExpr> indices_;
  };
  /*!
   * \brief get i-th slice from the current Tensor.
   * \param i the index of the coordinate
   * \return the subsequent slice.
   */
  inline Slice operator[](PrimExpr i) const { return Slice(*this, {i}); }

  explicit Tensor(ffi::ObjectPtr<TensorObj> node) : OpaqueExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Tensor, OpaqueExpr, TensorObj);
};

/*! \brief Return whether an expression calls the registered te.tensor_load operator. */
TVM_DLL bool IsTensorLoad(const Expr& expr);

/*! \brief Recover and validate the first Tensor argument of a tensor-load Call. */
TVM_DLL Tensor GetTensorFromLoad(const Call& call);

/*! \brief Recover and validate the primitive indices of a tensor-load Call. */
TVM_DLL ffi::Array<PrimExpr> GetTensorLoadIndices(const Call& call);

// Implementations of inline functions
inline size_t Tensor::ndim() const { return (*this)->shape.size(); }

inline bool Tensor::operator==(const Tensor& other) const {
  if (get() == other.get()) return true;
  if (get() == nullptr || other.get() == nullptr) return false;
  if ((*this)->op.has_value() || other->op.has_value()) {
    return (*this)->op == other->op && (*this)->value_index == other->value_index;
  } else {
    return false;
  }
}

inline bool Tensor::operator!=(const Tensor& other) const { return !(*this == other); }

// macro to turn every operation of slice to expression
#define DEFINE_OVERLOAD_SLICE_UNARY_OP(Op) \
  inline PrimExpr operator Op(const Tensor::Slice& a) { return Op a.operator PrimExpr(); }

#define DEFINE_OVERLOAD_SLICE_BINARY_OP(Op)                                     \
  template <typename T>                                                         \
  inline PrimExpr operator Op(const Tensor::Slice& a, const T& b) {             \
    return a.operator PrimExpr() Op b;                                          \
  }                                                                             \
  template <typename T>                                                         \
  inline PrimExpr operator Op(const T& a, const Tensor::Slice& b) {             \
    return a Op b.operator PrimExpr();                                          \
  }                                                                             \
  inline PrimExpr operator Op(const Tensor::Slice& a, const Tensor::Slice& b) { \
    return a.operator PrimExpr() Op b.operator PrimExpr();                      \
  }

DEFINE_OVERLOAD_SLICE_UNARY_OP(!);
DEFINE_OVERLOAD_SLICE_UNARY_OP(-);
DEFINE_OVERLOAD_SLICE_BINARY_OP(+);
DEFINE_OVERLOAD_SLICE_BINARY_OP(-);
DEFINE_OVERLOAD_SLICE_BINARY_OP(*);
DEFINE_OVERLOAD_SLICE_BINARY_OP(==);
DEFINE_OVERLOAD_SLICE_BINARY_OP(<=);
DEFINE_OVERLOAD_SLICE_BINARY_OP(>=);
DEFINE_OVERLOAD_SLICE_BINARY_OP(!=);
DEFINE_OVERLOAD_SLICE_BINARY_OP(&&);
DEFINE_OVERLOAD_SLICE_BINARY_OP(||);
DEFINE_OVERLOAD_SLICE_BINARY_OP(>>);
DEFINE_OVERLOAD_SLICE_BINARY_OP(<<);
DEFINE_OVERLOAD_SLICE_BINARY_OP(>);  // NOLINT(*)
DEFINE_OVERLOAD_SLICE_BINARY_OP(<);  // NOLINT(*)

}  // namespace te
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::te::Operation> : public ::tvm::ffi::ObjectPtrHash {};

template <>
struct hash<::tvm::te::Tensor> {
  std::size_t operator()(const ::tvm::te::Tensor& k) const {
    ::tvm::ffi::ObjectPtrHash hasher;
    if (k.defined() && k->op.has_value()) {
      return hasher(k->op.value());
    } else {
      return hasher(k);
    }
  }
};
}  // namespace std
#endif  // TVM_TE_TENSOR_H_
