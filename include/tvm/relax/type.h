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
 * \file tvm/relax/type.h
 * \brief Relax types, including the richer dependent Relax type nodes.
 */
#ifndef TVM_RELAX_TYPE_H_
#define TVM_RELAX_TYPE_H_

#include <tvm/base.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/type.h>
#include <tvm/relax/global_info.h>

#include <string>
#include <utility>

namespace tvm {
namespace relax {

using Expr = tvm::Expr;
using ExprObj = tvm::ExprObj;

class BlockBuilder;

/*! \brief Indicates the number of dimensions of a tensor is unknown at compile time. */
static constexpr int kUnknownNDim = -1;

using tvm::AnyType;
using tvm::AnyTypeObj;
using tvm::TupleType;
using tvm::TupleTypeObj;

// Compatibility aliases for existing C++ callers.  New code should use AnyType.
using ObjectTypeObj = AnyTypeObj;
using ObjectType = AnyType;

class PackedFuncTypeObj : public TypeObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PackedFuncTypeObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.PackedFuncType", PackedFuncTypeObj, TypeObj);
};

class PackedFuncType : public Type {
 public:
  TVM_DLL PackedFuncType(Location loc = UnknownLoc());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PackedFuncType, Type, PackedFuncTypeObj);
};

/*!
 * \brief Base type of all Relax type information.
 *
 * Type stores possible type information deduced during compile-time.
 * It encapsulates both static type and runtime information such as shape.
 *
 * Type of each non-primitive Expr can be deduced during compilation in a
 * "best-effort" manner.
 *
 * When ty appears in function parameter and return signatures, it
 * implies a runtime check that matches the type information with the value.
 *
 * When it appears in Expr, it follows "assume-semantics", which means the
 * compiler will take the deduced information as it is and only do best effort
 * proofs and checks.
 *
 * Each type can be uniquely erased to a static-type.  The compiler will
 * still compile the code, with less information, when we erase to the static
 * type.
 *
 * If a Type contains an Expr field, then that field must already be
 * normalized through NormalizeArg.  This invariant is checked in constructors
 * and simplifies assumptions during type deduction.
 */

/*!
 * \brief Type of shape value.
 */
class ShapeTypeObj : public TypeObj {
 public:
  /*! \brief optionally stores the symbolic value patterns of the shape */
  ffi::Optional<ffi::Array<PrimExpr>> values;
  /*!
   * \brief The number of dimension of the shape, can be unknown.
   * \sa kUnknownNDim
   */
  int ndim{kUnknownNDim};

  /*! \return Whether the type contains unknown ndim. */
  bool IsUnknownNdim() const { return ndim == kUnknownNDim; }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ShapeTypeObj>()
        .def_ro("values", &ShapeTypeObj::values)
        .def_ro("ndim", &ShapeTypeObj::ndim);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.ShapeType", ShapeTypeObj, TypeObj);
};

/*!
 * \brief Managed reference to ShapeTypeObj.
 * \sa ShapeTypeObj
 */
class ShapeType : public Type {
 public:
  /*!
   * \brief Construction with known symbolic shape patterns
   * \param values The symbolic shape values
   * \param loc The loc of the AST.
   */
  TVM_DLL ShapeType(ffi::Array<PrimExpr> values, Location loc = UnknownLoc());
  /*!
   * \brief Construction with known unknown symbolic shape patterns.
   * \param ndim Number of dimensions -- can be kUnknownNDim
   * \param loc The loc of the AST.
   */
  TVM_DLL ShapeType(int ndim, Location loc = UnknownLoc());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ShapeType, Type, ShapeTypeObj);
};

/*!
 * \brief Type of Tensor.
 */
class TensorTypeObj : public TypeObj {
 public:
  /*!
   * \brief optionally store the shape expression of the tensor.
   * \note shape must be normalized: it can only be std::nullopt or ShapeExpr or Var.
   */
  ffi::Optional<Expr> shape;
  /*! \brief The virtual device, indicates where the tensor
   *  is expected to be executed.
   */
  ffi::Optional<VDevice> vdevice;
  /*! \brief The content dtype, or nullopt if the dtype is unknown. */
  ffi::Optional<tvm::PrimType> dtype{std::nullopt};
  /*!
   * \brief The number of dimension of the tensor, can be unknown.
   * \sa kUnknownNDim
   */
  int ndim{kUnknownNDim};

  /*! \return Whether the type contains unknown ndim. */
  bool IsUnknownNdim() const { return ndim == kUnknownNDim; }

  /*! \return Whether the type contains unknown dtype. */
  bool IsUnknownDtype() const { return !dtype.has_value(); }

  /*! \return Shape if it is known. */
  ffi::Optional<ffi::Array<PrimExpr>> GetShape() const {
    if (!shape.has_value()) return {};
    const Expr& shape_expr = this->shape.value();
    if (shape_expr->ty.as<MissingType>().has_value()) return {};
    if (const auto* shape_ty = shape_expr->ty.as<ShapeTypeObj>()) {
      return shape_ty->values;
    }
    return {};
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorTypeObj>()
        .def_ro("shape", &TensorTypeObj::shape)
        .def_ro("dtype", &TensorTypeObj::dtype)
        .def_ro("vdevice", &TensorTypeObj::vdevice)
        .def_ro("ndim", &TensorTypeObj::ndim);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.TensorType", TensorTypeObj, TypeObj);
};

/*!
 * \brief Managed reference to TensorTypeObj.
 * \sa TensorTypeObj
 */
class TensorType : public Type {
 public:
  explicit TensorType(ffi::ObjectPtr<TensorTypeObj> data) : Type(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }

  /*!
   * \brief Construction with a known shape expression.
   * \param shape The shape of the tensor.
   * \param dtype The data type of tensor's elements.
   * \param vdevice The virtual device.
   * \param loc The loc of the AST.
   *
   * \note shape must already be normalized.
   */
  TVM_DLL TensorType(Expr shape, ffi::Optional<tvm::PrimType> dtype = std::nullopt,
                     ffi::Optional<VDevice> vdevice = std::nullopt, Location loc = UnknownLoc());

  /*!
   * \brief Construction with an unknown shape expression.
   * \param dtype The data type of tensor's elements.
   * \param ndim The number of dimensions
   * \param vdevice The virtual device.
   * \param loc The loc of the AST.
   */
  TVM_DLL TensorType(ffi::Optional<tvm::PrimType> dtype, int ndim,
                     ffi::Optional<VDevice> vdevice = std::nullopt, Location loc = UnknownLoc());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorType, Type, TensorTypeObj);
};

/*!
 * \brief custom-defined Type derivation function.
 * \param call The call expression to be derived.
 * \param ctx The builder context.
 * \return The derived type of the call.
 */
using TypeDeriveFunc = TypedEnvFunc<Type(const Call& call, const BlockBuilder& ctx)>;

/*!
 * \brief Function type information.
 *
 * This data structure contains enough information for us to do best-effort
 * type deduction.
 */
class FuncTypeObj : public TypeObj {
 public:
  /*!
   * \brief The parameter type of the function.
   * \note When params is std::nullopt means the function can take arbitrary number of arguments.
   *       We define such functions as Opaque function.
   */
  ffi::Optional<ffi::Array<Type>> params;
  /*!
   * \brief The type of the function's return value.
   */
  Type ret = Type::Missing();
  /*!
   * \brief Derivation function of opaque functions that may take any number of parameters.
   * \note When derive_func is not empty, then params should be std::nullopt,
   *       ret should be AnyType()
   */
  ffi::Optional<TypeDeriveFunc> derive_func;
  /*!
   * \brief Whether the function is pure.
   * \note This parameter should be set to true only if the function is pure on all inputs.
   *   If the function _may_ have visible side effects, set it to false.
   */
  bool purity;

  /*!
   * \return Whether the func type is opaque.
   * \note We define a function as opaque we have no constraints on params.
   */
  bool IsOpaque() const { return !params.has_value(); }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FuncTypeObj>()
        .def_ro("params", &FuncTypeObj::params, refl::AttachFieldFlag::SEqHashDefPattern())
        .def_ro("ret", &FuncTypeObj::ret)
        .def_ro("derive_func", &FuncTypeObj::derive_func)
        .def_ro("purity", &FuncTypeObj::purity);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.FuncType", FuncTypeObj, TypeObj);
};

/*!
 * \brief Managed reference to FuncTypeObj.
 * \sa FuncTypeObj
 */
class FuncType : public Type {
 public:
  explicit FuncType(ffi::ObjectPtr<FuncTypeObj> data) : Type(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  /*!
   * \brief Constructor from parameter type and return value type.
   * \param params The type of function parameters.
   * \param ret The return value type.
   * \param purity The purity of the function (true by default).
   * \param loc The loc of the AST.
   *
   * \note If the ret contains canonical tvm::Var values, they must be deducible from
   * params. If you are unsure, you can always erase ret to static.
   */
  TVM_DLL FuncType(ffi::Array<Type> params, Type ret, bool purity = true,
                   Location loc = UnknownLoc());

  /*!
   * \brief Constructing an opaque function type using derive_func.
   *
   * \param derive_func Derivation function.
   * \param purity The purity of the function
   *   (false by default: most external functions are not pure).
   * \param loc The loc of the AST.
   *
   * \return The FuncType for opaque packedfunc.
   * \note Defaults to an derive func that always return AnyType if not specified.
   */
  TVM_DLL static FuncType OpaqueFunc(TypeDeriveFunc derive_func, bool purity = false,
                                     Location loc = UnknownLoc());

  /*!
   * \brief Construct an opaque function using from return type.
   *
   * \param ret The type of the return value.
   * \param purity The purity of the function
   *   (false by default: most external functions are not pure).
   * \param loc The loc of the AST.
   *
   * \return The FuncType for opaque packedfunc.
   * \note Defaults to an derive func that always return AnyType if not specified.
   */
  TVM_DLL static FuncType OpaqueFunc(Type ret = AnyType(), bool purity = false,
                                     Location loc = UnknownLoc());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FuncType, Type, FuncTypeObj);
};

/*!
 * \brief Match and check if expr has Relax type T and return it.
 *
 * \param expr The input expression.
 * \return The result of match.
 * \tparam T the underlying Relax type
 */
template <typename T>
inline ffi::Optional<T> MatchType(const Expr& expr) {
  if (!expr.defined()) {
    return std::nullopt;
  }
  using TNode = typename T::ContainerType;
  if (const TNode* ptr = expr->ty.as<TNode>()) {
    return ffi::GetRef<T>(ptr);
  } else {
    return std::nullopt;
  }
}

/*!
 * \brief Get the type of a given expr and try to cast it as const T*.
 *
 * \param expr The input expression.
 * \return The pointer. Returns nullptr if the type does not match.
 * \tparam T the underlying Relax type node
 */
template <typename T>
inline const T* GetTypeAs(const Expr& expr) {
  TVM_FFI_ICHECK(!expr->ty.as<MissingType>().has_value())
      << "The type is not populated, check if you have normalized the expr";
  return expr->ty.as<T>();
}

/*!
 * \brief Get the underlying Relax type of expr.
 *
 * \param expr The input expression.
 * \return underlying Relax type.
 */
inline Type GetType(const Expr& expr) {
  TVM_FFI_ICHECK(!expr->ty.as<MissingType>().has_value())
      << "The type is not populated, check if you have normalized the expr";
  return expr->ty;
}

/*!
 * \brief Whether the expr has void type.
 *
 * \param expr The input expression.
 * \return Whether the expr has void type.
 */
inline bool HasVoidType(const Expr& expr) {
  auto* ptr = expr->ty.as<TupleTypeObj>();
  return ptr != nullptr && ptr->fields.size() == 0;
}

/*!
 * \brief Update the type of an Expr.
 * \param expr The Expr whose type to be updated.
 * \param ty The type assigned.
 * \note We ensure idempotence, that is we can only update the type of an Expr only
 *  if the original one is nullptr.
 */
TVM_DLL void UpdateType(Expr expr, Type ty);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TYPE_H_
