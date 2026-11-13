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
#ifndef TVM_S_TIR_META_SCHEDULE_BUILDER_H_
#define TVM_S_TIR_META_SCHEDULE_BUILDER_H_

#include <tvm/base.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/tensor.h>
#include <tvm/target/target.h>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*! \brief The builder's input, containing an IRModule and the target. */
class BuilderInputObj : public ffi::Object {
 public:
  explicit BuilderInputObj(ffi::UnsafeInit tag) : target(tag) {}

  explicit BuilderInputObj(Target target) : target(std::move(target)) {}
  /*! \brief The IRModule to be built. */
  IRModule mod;
  /*! \brief The target to be built for. */
  Target target;
  /*! \brief Parameters for Relax build module. */
  ffi::Optional<ffi::Map<ffi::String, ffi::Tensor>> params;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BuilderInputObj>()
        .def_ro("mod", &BuilderInputObj::mod)
        .def_ro("target", &BuilderInputObj::target)
        .def_ro("params", &BuilderInputObj::params);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.BuilderInput", BuilderInputObj,
                                    ffi::Object);
};

/*!
 * \brief Managed reference to BuilderInputObj
 * \sa BuilderInputObj
 */
class BuilderInput : public ffi::ObjectRef {
 public:
  /*!
   * \brief Constructor of BuilderInput.
   * \param mod The IRModule to be built.
   * \param target The target to be built for.
   * \param params Parameters for Relax build module.
   */
  TVM_DLL explicit BuilderInput(
      IRModule mod, Target target,
      ffi::Optional<ffi::Map<ffi::String, ffi::Tensor>> params = std::nullopt);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BuilderInput, ffi::ObjectRef, BuilderInputObj);
};

/*! \brief The builder's output, containing the artifact path or error message if any. */
class BuilderResultObj : public ffi::Object {
 public:
  /*! \brief The path to the built artifact. */
  ffi::Optional<ffi::String> artifact_path;
  /*! \brief The error message if any. */
  ffi::Optional<ffi::String> error_msg;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BuilderResultObj>()
        .def_ro("artifact_path", &BuilderResultObj::artifact_path)
        .def_ro("error_msg", &BuilderResultObj::error_msg);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.BuilderResult", BuilderResultObj,
                                    ffi::Object);
};

/*!
 * \brief Managed reference to BuilderResultObj
 * \sa BuilderResultObj
 */
class BuilderResult : public ffi::ObjectRef {
 public:
  /*!
   * \brief Constructor of BuilderResult.
   * \param artifact_path The path to the built artifact.
   * \param error_msg The error message if any.
   */
  TVM_DLL explicit BuilderResult(ffi::Optional<ffi::String> artifact_path,
                                 ffi::Optional<ffi::String> error_msg);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BuilderResult, ffi::ObjectRef, BuilderResultObj);
};

/*! \brief The abstract builder interface. */
class BuilderObj : public ffi::Object {
 public:
  /*! \brief Default destructor */
  virtual ~BuilderObj() = default;
  /*!
   * \brief Generate the build results from build inputs.
   * \param build_inputs The inputs to be built.
   * \return The build results.
   */
  virtual ffi::Array<BuilderResult> Build(const ffi::Array<BuilderInput>& build_inputs) = 0;
  /*!
   * \brief The function type of `Build` method.
   * \param build_inputs The inputs to be built.
   * \return The build results.
   */
  using FBuild = ffi::TypedFunction<ffi::Array<BuilderResult>(const ffi::Array<BuilderInput>&)>;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("s_tir.meta_schedule.Builder", BuilderObj, ffi::Object);
};

/*!
 * \brief Managed reference to BuilderObj
 * \sa BuilderObj
 */
class Builder : public ffi::ObjectRef {
 public:
  /*!
   * \brief Constructor from ffi::ObjectPtr<BuilderObj>.
   * \param data The object pointer.
   */
  explicit Builder(ffi::ObjectPtr<BuilderObj> data) : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  /*!
   * \brief Create a builder with customized build method on the python-side.
   * \param f_build The packed function to the `Build` function..
   * \return The Builder created.
   */
  static Builder PyBuilder(BuilderObj::FBuild f_build);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Builder, ffi::ObjectRef, BuilderObj);
};

/*! \brief An abstract builder with customized build method on the python-side. */
class PyBuilderObj : public BuilderObj {
 public:
  /*! \brief The packed function to the `Build` function. */
  FBuild f_build;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PyBuilderObj>().def_ro("f_build", &PyBuilderObj::f_build);
  }

  ffi::Array<BuilderResult> Build(const ffi::Array<BuilderInput>& build_inputs) final {
    TVM_FFI_ICHECK(f_build != nullptr) << "PyBuilder's Build method not implemented!";
    return f_build(build_inputs);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.PyBuilder", PyBuilderObj, BuilderObj);
};

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_BUILDER_H_
