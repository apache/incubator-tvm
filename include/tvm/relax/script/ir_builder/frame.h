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
#ifndef TVM_RELAX_SCRIPT_IR_BUILDER_FRAME_H_
#define TVM_RELAX_SCRIPT_IR_BUILDER_FRAME_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/script/ir_builder/base.h>
#include <tvm/script/ir_builder/frame.h>
#include <tvm/script/ir_builder/ir.h>

#include <utility>

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

/*! \brief The base ir_builder frame for the relax dialect. */
class RelaxFrameObj : public IRBuilderFrameObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RelaxFrameObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO("script.ir_builder.relax.RelaxFrame", RelaxFrameObj,
                              IRBuilderFrameObj);
};

class RelaxFrame : public IRBuilderFrame {
 public:
  explicit RelaxFrame(ffi::ObjectPtr<RelaxFrameObj> data) : IRBuilderFrame(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RelaxFrame, IRBuilderFrame, RelaxFrameObj);

 protected:
  RelaxFrame() = default;
};

/*! \brief The base ir_builder frame for frames with SeqExpr
           i.e. Functions, IfExpr branches
  */
class SeqExprFrameObj : public RelaxFrameObj {
 public:
  /*! \brief The binding blocks inside the frame. */
  ffi::Array<tvm::relax::BindingBlock> binding_blocks;
  /*! \brief The frame output expr. `std::nullopt` when undefined. */
  ffi::Optional<tvm::relax::Expr> output;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SeqExprFrameObj>()
        .def_ro("binding_blocks", &SeqExprFrameObj::binding_blocks)
        .def_ro("output", &SeqExprFrameObj::output);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("script.ir_builder.relax.SeqExprFrame", SeqExprFrameObj,
                              RelaxFrameObj);

 public:
  void EnterWithScope() override;
  void ExitWithScope() override;
};

class SeqExprFrame : public RelaxFrame {
 public:
  explicit SeqExprFrame(ffi::ObjectPtr<SeqExprFrameObj> data) : RelaxFrame(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SeqExprFrame, RelaxFrame, SeqExprFrameObj);
};

/*! \brief The ir_builder frame for the relax function. */
class FunctionFrameObj : public SeqExprFrameObj {
 public:
  explicit FunctionFrameObj(tvm::relax::BlockBuilder builder) : block_builder(std::move(builder)) {}
  explicit FunctionFrameObj(ffi::UnsafeInit) : block_builder(ffi::UnsafeInit{}) {}
  /*!
   * \brief The function name.
   * \note The name will not be specified in constructor, so it is "Optional",
   *       However, we must specify the name by `R.func_name` before exit this frame.
   */
  ffi::Optional<ffi::String> name;
  /*! \brief The function params. */
  ffi::Array<tvm::Var> params;
  /*!
   * \brief The function return type.
   * \note Usually the function return type can be deduced by the function body.
   *       But we can use this field to specify a more "accurate" return type.
   *       i.e. If the `ret_ty` is None, try to use the deduced type from body
   *       If the `ret_ty` is not None, we can still take body.ty
   *       if we ret_ty is base of body.ty. If not, we will
   *       take the specified `ret_ty`.
   */
  ffi::Optional<tvm::Type> ret_ty;
  /*! \brief Whether the function is annotated as pure */
  ffi::Optional<bool> is_pure;
  /*! \brief Whether the function is annotated as private */
  ffi::Optional<bool> is_private;
  /*! \brief The function attributes. */
  ffi::Map<ffi::String, Any> attrs;
  /*! \brief The block builder to create Relax function. */
  tvm::relax::BlockBuilder block_builder;
  /*! \brief Whether this frame constructs only a function signature. */
  bool declaration = false;
  bool local = false;
  ffi::Optional<tvm::Var> local_var;
  /*! \brief Finalized function and its stable module reference. */
  ffi::Optional<tvm::relax::Function> function;
  ffi::Optional<tvm::GlobalVar> global_var;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FunctionFrameObj>()
        .def_ro("name", &FunctionFrameObj::name)
        .def_ro("params", &FunctionFrameObj::params)
        .def_ro("ret_ty", &FunctionFrameObj::ret_ty)
        .def_ro("is_pure", &FunctionFrameObj::is_pure)
        .def_ro("attrs", &FunctionFrameObj::attrs)
        .def_ro("function", &FunctionFrameObj::function)
        .def_ro("global_var", &FunctionFrameObj::global_var)
        .def_ro("local_var", &FunctionFrameObj::local_var);
    // `binding_blocks` and `output` are inherited from SeqExprFrameObj.
    // `block_builder` is not registered as it's not visited.
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.relax.FunctionFrame", FunctionFrameObj,
                                    SeqExprFrameObj);

 public:
  void EnterWithScope() final;
  void ExitWithScope() final;
};

class FunctionFrame : public SeqExprFrame {
 public:
  explicit FunctionFrame(ffi::ObjectPtr<FunctionFrameObj> data) : SeqExprFrame(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FunctionFrame, SeqExprFrame, FunctionFrameObj);
};

/*! \brief The ir_builder frame for relax binding blocks. */
class BindingBlockFrameObj : public RelaxFrameObj {
 public:
  /*! \brief The flag that indicates whether the block is a dataflow block. */
  bool is_dataflow;
  /*! \brief The variables emitted in this block. */
  ffi::Array<tvm::Var> emitted_vars;
  /*!
   * \brief A boolean indicating if the dataflow block is ended of construction.
   * If it is true, any new binding trying to be emitted into this block will cause an error.
   * \note Only used for a dataflow block.
   */
  bool block_ended;
  /*!
   * \brief The output vars of the dataflow block.
   * \note Only used for a dataflow block.
   */
  ffi::Array<tvm::Var> output_vars;
  /*! \brief Statement ranges for explicitly emitted bindings. */
  ffi::Map<tvm::Var, Location> binding_locs;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BindingBlockFrameObj>()
        .def_ro("is_dataflow", &BindingBlockFrameObj::is_dataflow)
        .def_ro("emitted_vars", &BindingBlockFrameObj::emitted_vars)
        .def_ro("output_vars", &BindingBlockFrameObj::output_vars);
    // `block_ended` is not registered as it's not visited.
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.relax.BindingBlockFrame",
                                    BindingBlockFrameObj, RelaxFrameObj);

 public:
  void EnterWithScope() final;
  void ExitWithScope() final;
};

class BindingBlockFrame : public RelaxFrame {
 public:
  explicit BindingBlockFrame(ffi::ObjectPtr<BindingBlockFrameObj> data) : RelaxFrame(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BindingBlockFrame, RelaxFrame,
                                                BindingBlockFrameObj);
};

/*!
 * \brief A frame that represents if statement.
 *
 * \sa IfFrame
 */
class IfFrameObj : public RelaxFrameObj {
 public:
  explicit IfFrameObj(tvm::relax::Expr condition) : condition(std::move(condition)) {}
  explicit IfFrameObj(ffi::UnsafeInit) : condition(ffi::UnsafeInit{}) {}

  /*! \brief The condition of the if statement. */
  tvm::relax::Expr condition;
  /*! \brief The Bindings in the true branch. */
  ffi::Optional<tvm::relax::Expr> then_expr;
  /*! \brief The Bindings in the false branch. */
  ffi::Optional<tvm::relax::Expr> else_expr;
  /*! \brief The Binding var. */
  ffi::Optional<tvm::Var> var;
  /*! \brief The binding var name. */
  ffi::String var_name;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IfFrameObj>()
        .def_ro("condition", &IfFrameObj::condition)
        .def_ro("then_expr", &IfFrameObj::then_expr)
        .def_ro("else_expr", &IfFrameObj::else_expr)
        .def_ro("var", &IfFrameObj::var)
        .def_ro("var_name", &IfFrameObj::var_name);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.relax.IfFrame", IfFrameObj, RelaxFrameObj);

 public:
  /*!
   * \brief The method called when entering RAII scope.
   * \sa tvm::support::With
   */
  void EnterWithScope() final;
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to IfFrameObj.
 *
 * \sa IfFrameObj
 */
class IfFrame : public RelaxFrame {
 public:
  explicit IfFrame(ffi::ObjectPtr<IfFrameObj> data) : RelaxFrame(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IfFrame, RelaxFrame, IfFrameObj);
};

/*!
 * \brief A frame that represents then.
 *
 * \sa ThenFrame
 */
class ThenFrameObj : public SeqExprFrameObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ThenFrameObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.relax.ThenFrame", ThenFrameObj,
                                    SeqExprFrameObj);

 public:
  /*!
   * \brief The method called when entering RAII scope.
   * \sa tvm::support::With
   */
  void EnterWithScope() final;
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to ThenFrameObj.
 *
 * \sa ThenFrameObj
 */
class ThenFrame : public SeqExprFrame {
 public:
  explicit ThenFrame(ffi::ObjectPtr<ThenFrameObj> data) : SeqExprFrame(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ThenFrame, SeqExprFrame, ThenFrameObj);
};

/*!
 * \brief A frame that represents else.
 *
 * \sa ElseFrame
 */
class ElseFrameObj : public SeqExprFrameObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ElseFrameObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.relax.ElseFrame", ElseFrameObj,
                                    SeqExprFrameObj);

 public:
  /*!
   * \brief The method called when entering RAII scope.
   * \sa tvm::support::With
   */
  void EnterWithScope() final;
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to ElseFrameObj.
 *
 * \sa ElseFrameObj
 */
class ElseFrame : public SeqExprFrame {
 public:
  explicit ElseFrame(ffi::ObjectPtr<ElseFrameObj> data) : SeqExprFrame(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ElseFrame, SeqExprFrame, ElseFrameObj);
};

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_RELAX_SCRIPT_IR_BUILDER_FRAME_H_
