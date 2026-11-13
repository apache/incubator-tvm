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
 * \file tvm/te/operation.h
 * \brief Operation node can generate one or multiple Tensors
 */
#ifndef TVM_TE_OPERATION_H_
#define TVM_TE_OPERATION_H_

#include <tvm/base.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/te/reduction.h>
#include <tvm/te/tensor.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
/*! \brief Tensor expression language DSL. */
namespace te {

// Reduce operator
/*!
 * \brief A commutative reducer node to represent a commutative
 *  binary operator with identity element
 */
class CommReducerObj : public ffi::Object {
 public:
  /*! \brief The left argument of reducer */
  ffi::Array<PrimVar> lhs;
  /*! \brief The right argument of reducer */
  ffi::Array<PrimVar> rhs;
  /*! \brief The result of reducer */
  ffi::Array<PrimExpr> result;
  /*!
   * \brief The identity element of reducer, which leaves other
   *  elements unchanged when combined with it, with respect to
   *  the binary operation of this reducer uses.
   */
  ffi::Array<PrimExpr> identity_element;
  /*! \brief Function call operator to combine a and b */
  ffi::Array<PrimExpr> operator()(ffi::Array<PrimExpr> a, ffi::Array<PrimExpr> b) const;
  /*!
   * \brief Location that points to the original source code.
   *        Reserved debug information.
   */
  mutable Location loc = UnknownLoc();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CommReducerObj>()
        .def_ro("lhs", &CommReducerObj::lhs, refl::AttachFieldFlag::SEqHashDefPattern())
        .def_ro("rhs", &CommReducerObj::rhs, refl::AttachFieldFlag::SEqHashDefPattern())
        .def_ro("result", &CommReducerObj::result)
        .def_ro("identity_element", &CommReducerObj::identity_element)
        .def_ro("loc", &CommReducerObj::loc, refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("te.CommReducer", CommReducerObj, ffi::Object);
};

/*!
 * \brief Managed reference to CommReducerObj
 * \sa CommReducerObj
 */
class CommReducer : public ffi::ObjectRef {
 public:
  TVM_DLL CommReducer(ffi::Array<PrimVar> lhs, ffi::Array<PrimVar> rhs, ffi::Array<PrimExpr> result,
                      ffi::Array<PrimExpr> identity_element, Location loc = UnknownLoc());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CommReducer, ffi::ObjectRef, CommReducerObj);
};

/*! \brief Reduction operator */
class ReduceObj : public OpaqueExprObj {
 public:
  explicit ReduceObj(CommReducer combiner, PrimExpr condition)
      : combiner(std::move(combiner)), condition(std::move(condition)) {}
  explicit ReduceObj(ffi::UnsafeInit) : combiner(ffi::UnsafeInit{}), condition(ffi::UnsafeInit{}) {}

  /*! \brief The commutative combiner */
  CommReducer combiner;
  /*! \brief The source operand */
  ffi::Array<PrimExpr> source;
  /*! \brief The init operand */
  ffi::Array<PrimExpr> init;
  /*! \brief The reduction axis */
  ffi::Array<s_tir::IterVar> axis;
  /*!
   * \brief Predicate on the reduction
   *  Only add the body to reduction if condition is true.
   */
  PrimExpr condition;
  /*! \brief the index of this reduce node */
  int value_index;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReduceObj>()
        .def_ro("combiner", &ReduceObj::combiner)
        .def_ro("source", &ReduceObj::source)
        .def_ro("init", &ReduceObj::init)
        .def_ro("axis", &ReduceObj::axis)
        .def_ro("condition", &ReduceObj::condition)
        .def_ro("value_index", &ReduceObj::value_index);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("te.Reduce", ReduceObj, OpaqueExprObj);
};

/*!
 * \brief Managed reference to ReduceObj
 * \sa ReduceObj
 */
class Reduce : public PrimExpr {
 public:
  TVM_DLL Reduce(CommReducer combiner, ffi::Array<PrimExpr> src, ffi::Array<s_tir::IterVar> rdom,
                 ffi::Optional<PrimExpr> condition, int value_index, ffi::Array<PrimExpr> init,
                 Location loc = UnknownLoc());
  explicit Reduce(ffi::ObjectPtr<ReduceObj> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Reduce, PrimExpr, ReduceObj);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ReduceObj);
};

/*!
 * \brief Temporary data structure to store union
 *  of bounds of each axis of Tensor.
 */
struct TensorDom {
  // constructor
  explicit TensorDom(int ndim) : data(ndim) {}
  /*! \brief The domain data */
  std::vector<std::vector<IntSet>> data;
};

/*!
 * \brief Base class of all operation nodes
 */
class TVM_DLL OperationObj : public ffi::Object {
 public:
  /*! \brief optional name of the operation */
  std::string name;
  /*! \brief optional tag of the operation */
  std::string tag;
  /*! \brief additional attributes of the operation*/
  ffi::Map<ffi::String, ffi::Any> attrs;
  // virtual destructor.
  virtual ~OperationObj() {}
  /*! \return number of outputs */
  virtual int num_outputs() const = 0;
  /*!
   * \brief Get the primitive element type of the i-th output tensor.
   * \param i The output index.
   * \return primitive element type of i-th output.
   */
  virtual PrimType output_dtype(size_t i) const = 0;
  /*!
   * \brief Get shape of i-th output tensor.
   * \param i The output index.
   * \return shape of i-th output.
   */
  virtual ffi::Array<PrimExpr> output_shape(size_t i) const = 0;
  /*!
   * \brief List all the input Tensors.
   * \return List of input tensors.
   */
  virtual ffi::Array<Tensor> InputTensors() const = 0;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OperationObj>()
        .def_ro("name", &OperationObj::name)
        .def_ro("tag", &OperationObj::tag)
        .def_ro("attrs", &OperationObj::attrs);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("te.Operation", OperationObj, ffi::Object);
};

/*!
 * \brief A placeholder op represents an input placeholder.
 */
class PlaceholderOpObj : public OperationObj {
 public:
  /*! \brief The shape of the input */
  ffi::Array<PrimExpr> shape;
  /*! \brief The dtype of the input. */
  PrimType dtype = PrimType::Void();
  // override behavior.
  int num_outputs() const final;
  PrimType output_dtype(size_t i) const final;
  ffi::Array<PrimExpr> output_shape(size_t i) const final;
  ffi::Array<Tensor> InputTensors() const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PlaceholderOpObj>()
        .def_ro("shape", &PlaceholderOpObj::shape)
        .def_ro("dtype", &PlaceholderOpObj::dtype);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("te.PlaceholderOp", PlaceholderOpObj, OperationObj);
};

/*!
 * \brief Managed reference to PlaceholderOpObj
 * \sa PlaceholderOpObj
 */
class PlaceholderOp : public Operation {
 public:
  TVM_DLL PlaceholderOp(std::string name, ffi::Array<PrimExpr> shape, PrimType dtype);

  explicit PlaceholderOp(ffi::UnsafeInit tag, ffi::ObjectPtr<PlaceholderOpObj> node)
      : Operation(tag) {
    data_ = std::move(node);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PlaceholderOp, Operation, PlaceholderOpObj);
};

/*!
 * \brief A Compute op that compute a tensor on certain domain.
 * This is the base class for ComputeOp (operating on a scalar at a time)
 */
class TVM_DLL BaseComputeOpObj : public OperationObj {
 public:
  /*! \brief IterVar on each axis */
  ffi::Array<IterVar> axis;
  /*! \brief IterVar on each reduction axis, if the body is a Reduce */
  ffi::Array<IterVar> reduce_axis;
  // override functions
  ffi::Array<PrimExpr> output_shape(size_t idx) const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BaseComputeOpObj>()
        .def_ro("axis", &BaseComputeOpObj::axis)
        .def_ro("reduce_axis", &BaseComputeOpObj::reduce_axis);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("te.BaseComputeOp", BaseComputeOpObj, OperationObj);
};

/*!
 * \brief A Compute op that compute a tensor on certain domain.
 */
class TVM_DLL ComputeOpObj : public BaseComputeOpObj {
 public:
  /*! \brief the compute expression */
  ffi::Array<PrimExpr> body;
  /*! \brief constructor */
  ComputeOpObj() {}
  // override functions
  int num_outputs() const final;
  PrimType output_dtype(size_t i) const final;
  ffi::Array<Tensor> InputTensors() const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ComputeOpObj>().def_ro("body", &ComputeOpObj::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("te.ComputeOp", ComputeOpObj, BaseComputeOpObj);
};

/*!
 * \brief Managed reference to ComputeOpObj
 * \sa ComputeOpObj
 */
class ComputeOp : public Operation {
 public:
  TVM_DLL ComputeOp(std::string name, std::string tag, ffi::Map<ffi::String, ffi::Any> attrs,
                    ffi::Array<IterVar> axis, ffi::Array<PrimExpr> body);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ComputeOp, Operation, ComputeOpObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ComputeOpObj);
};

/*!
 * \brief Symbolic scan.
 */
class ScanOpObj : public OperationObj {
 public:
  /*! \brief IterVar to scan over */
  IterVar scan_axis{ffi::UnsafeInit{}};
  /*! \brief the initialization tensors */
  ffi::Array<Tensor> init;
  /*! \brief the update function represented by tensor */
  ffi::Array<Tensor> update;
  /*! \brief The placeholder to refer as states in update. */
  ffi::Array<Tensor> state_placeholder;
  /*!
   * \brief the inputs to the scan, these are optionally provided
   *  But they can be helpful to provide hints to speedup get of scan body.
   */
  ffi::Array<Tensor> inputs;
  /*!
   * \brief Spatial axis to indicate spatial dimension of each output.
   *  They corresponds to flattened spatial axis of the outputs.
   *
   *  [output[0].axis[1], output[0].axis[2]... output[k].axis[j]...]
   *  These are auxiliary data structure for storing result of bound inference.
   *  They do not corresponds to splittable iterations, thus the name comes
   *  with underscore.
   */
  ffi::Array<IterVar> spatial_axis_;
  /*! \brief constructor */
  ScanOpObj() {}
  // override behavior.
  int num_outputs() const final;
  PrimType output_dtype(size_t i) const final;
  ffi::Array<PrimExpr> output_shape(size_t i) const final;
  ffi::Array<Tensor> InputTensors() const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScanOpObj>()
        .def_ro("scan_axis", &ScanOpObj::scan_axis)
        .def_ro("init", &ScanOpObj::init)
        .def_ro("update", &ScanOpObj::update)
        .def_ro("state_placeholder", &ScanOpObj::state_placeholder)
        .def_ro("inputs", &ScanOpObj::inputs)
        .def_ro("spatial_axis_", &ScanOpObj::spatial_axis_);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("te.ScanOp", ScanOpObj, OperationObj);
};

/*!
 * \brief Managed reference to ScanOpObj
 * \sa ScanOpObj
 */
class ScanOp : public Operation {
 public:
  TVM_DLL ScanOp(std::string name, std::string tag,
                 ffi::Optional<ffi::Map<ffi::String, ffi::Any>> attrs, IterVar axis,
                 ffi::Array<Tensor> init, ffi::Array<Tensor> update,
                 ffi::Array<Tensor> state_placeholder, ffi::Array<Tensor> input);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ScanOp, Operation, ScanOpObj);
};

/*!
 * \brief External computation that cannot be splitted.
 */
class ExternOpObj : public OperationObj {
 public:
  explicit ExternOpObj(SeqStmt body) : body(std::move(body)) {}
  explicit ExternOpObj(ffi::UnsafeInit) : body(ffi::UnsafeInit{}) {}

  /*! \brief The input tensors */
  ffi::Array<Tensor> inputs;
  /*! \brief Symbolic placeholder representation of inputs */
  ffi::Array<TensorVar> input_placeholders;
  /*! \brief Symbolic placeholder representation of outputs */
  ffi::Array<TensorVar> output_placeholders;
  /*! \brief the statement that generates the computation. */
  SeqStmt body;

  // override functions
  int num_outputs() const final;
  PrimType output_dtype(size_t i) const final;
  ffi::Array<PrimExpr> output_shape(size_t i) const final;
  ffi::Array<Tensor> InputTensors() const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExternOpObj>()
        .def_ro("inputs", &ExternOpObj::inputs)
        .def_ro("input_placeholders", &ExternOpObj::input_placeholders)
        .def_ro("output_placeholders", &ExternOpObj::output_placeholders)
        .def_ro("body", &ExternOpObj::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("te.ExternOp", ExternOpObj, OperationObj);
};

/*!
 * \brief Managed reference to ExternOpObj
 * \sa ExternOpObj
 */
class ExternOp : public Operation {
 public:
  TVM_DLL ExternOp(std::string name, std::string tag, ffi::Map<ffi::String, ffi::Any> attrs,
                   ffi::Array<Tensor> inputs, ffi::Array<TensorVar> input_placeholders,
                   ffi::Array<TensorVar> output_placeholders, SeqStmt body);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExternOp, Operation, ExternOpObj);
};

/*!
 * \brief Construct a new Var expression
 * \param name_hint The name hint for the expression
 * \param t The type of the expression
 */
TVM_DLL PrimVar var(std::string name_hint, PrimType t = PrimType::Int(32));

/*!
 * \brief Create a new IterVar that represents an axis in thread.
 *
 * \param dom Optional, domain of the thread axis.
 * \param tag The thread tag of the axis.
 */
TVM_DLL IterVar thread_axis(Range dom, std::string tag);

/*!
 * \brief Create a new IterVar for reduction operations.
 *
 * \param dom The domain of the reduction axis.
 * \param name The name of the reduction axis.
 */
TVM_DLL IterVar reduce_axis(Range dom, std::string name = "rv");

/*! \brief The compute function to specify the input source of a Tensor */
using FCompute = std::function<PrimExpr(const ffi::Array<PrimVar>& i)>;

/*! \brief The compute function to specify the inputs source of Tensors */
using FBatchCompute = std::function<ffi::Array<PrimExpr>(const ffi::Array<PrimVar>& i)>;

/*!
 * \brief create a place holder tensor.
 * \param shape The shape of the tensor.
 * \param dtype the data type of the tensor.
 * \param name The name of the Tensor.
 */
TVM_DLL Tensor placeholder(ffi::Array<PrimExpr> shape, PrimType dtype = PrimType::Float(32),
                           std::string name = "placeholder");

/*!
 * \brief Construct a new tensor by computing over shape,
 *  using the computation rule: result_tensor[axis] = fcompute(axis)
 * \param shape Shape of the tensor.
 * \param fcompute The compute function to create the tensor.
 * \param name The optional name of the tensor.
 * \param tag The optional tag of the tensor.
 * \param attrs Optional additional attributes of the compute.
 */
TVM_DLL Tensor compute(ffi::Array<PrimExpr> shape, FCompute fcompute, std::string name = "tensor",
                       std::string tag = "", ffi::Map<ffi::String, ffi::Any> attrs = {});

/*!
 * \brief Construct a new tensor by computing over shape,
 *  using the computation rule: result_tensor[axis] = fcompute(axis)
 * \param shape Shape of the tensor.
 * \param fcompute The compute function to create the tensors.
 * \param name The optional name of the tensor.
 * \param tag The optional tag of the tensor.
 * \param attrs Optional additional attributes of the compute.
 */
TVM_DLL ffi::Array<Tensor> compute(ffi::Array<PrimExpr> shape, FBatchCompute fcompute,
                                   std::string name = "tensor", std::string tag = "",
                                   ffi::Map<ffi::String, ffi::Any> attrs = {});

/*!
 * \brief Construct new tensors by scan.
 *
 * \param init The intialize tensor of first K steps.
 * \param update The update tensor indicated the updated result after each timestamp.
 * \param state_placeholder The placeholder for the states.
 * \param inputs The inputs to the scan body, this is optional,
 *    but recommended to provide concrete information about scan body.
 * \param name The optional name of the tensor.
 * \param tag The optional tag of the tensor.
 * \param attrs Optional additional attributes of the compute.
 */
TVM_DLL ffi::Array<Tensor> scan(ffi::Array<Tensor> init, ffi::Array<Tensor> update,
                                ffi::Array<Tensor> state_placeholder,
                                ffi::Array<Tensor> inputs = ffi::Array<Tensor>(),
                                std::string name = "scan", std::string tag = "",
                                ffi::Map<ffi::String, ffi::Any> attrs = {});

// same as compute, specialized for different fcompute function
inline Tensor compute(ffi::Array<PrimExpr> shape, std::function<PrimExpr(PrimVar)> f,
                      std::string name = "tensor", std::string tag = "",
                      ffi::Map<ffi::String, ffi::Any> attrs = {}) {
  FCompute fc = [f](const ffi::Array<PrimVar>& i) { return f(i[0]); };
  return compute(shape, fc, name, tag, attrs);
}
inline Tensor compute(ffi::Array<PrimExpr> shape, std::function<PrimExpr(PrimVar, PrimVar)> f,
                      std::string name = "tensor", std::string tag = "",
                      ffi::Map<ffi::String, ffi::Any> attrs = {}) {
  FCompute fc = [f](const ffi::Array<PrimVar>& i) { return f(i[0], i[1]); };
  return compute(shape, fc, name, tag, attrs);
}
inline Tensor compute(ffi::Array<PrimExpr> shape,
                      std::function<PrimExpr(PrimVar, PrimVar, PrimVar)> f,
                      std::string name = "tensor", std::string tag = "",
                      ffi::Map<ffi::String, ffi::Any> attrs = {}) {
  FCompute fc = [f](const ffi::Array<PrimVar>& i) { return f(i[0], i[1], i[2]); };
  return compute(shape, fc, name, tag, attrs);
}
inline Tensor compute(ffi::Array<PrimExpr> shape,
                      std::function<PrimExpr(PrimVar, PrimVar, PrimVar, PrimVar)> f,
                      std::string name = "tensor", std::string tag = "",
                      ffi::Map<ffi::String, ffi::Any> attrs = {}) {
  FCompute fc = [f](const ffi::Array<PrimVar>& i) { return f(i[0], i[1], i[2], i[3]); };
  return compute(shape, fc, name, tag, attrs);
}

// inline function.
inline const OperationObj* Operation::operator->() const {
  return static_cast<const OperationObj*>(get());
}
}  // namespace te

namespace ffi {

template <>
inline constexpr bool object_ref_contains_v<PrimExpr, te::ReduceObj> = true;

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_TE_OPERATION_H_
