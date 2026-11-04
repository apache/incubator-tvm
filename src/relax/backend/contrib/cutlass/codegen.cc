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
 * \file src/relax/backend/contrib/cutlass/codegen.cc
 * \brief Implementation of the CUTLASS code generator for Relax.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/unique_name_supply.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/op/nn.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/logging.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../codegen_c/codegen_c.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace contrib {

std::string EmitSignature(const std::vector<Output>& out, const std::string& func_id,
                          const std::vector<std::string>& arg_names) {
  std::ostringstream code_stream_;
  code_stream_ << "void " << func_id << "_(";
  for (const auto& arg_name : arg_names) {
    code_stream_ << "DLTensor* " << arg_name << ", ";
  }
  for (size_t i = 0; i < out.size() - 1; ++i) {
    code_stream_ << "DLTensor* out" << i << ", ";
  }
  code_stream_ << "DLTensor* out" << out.size() - 1 << ")";
  return code_stream_.str();
}

ffi::Module Finalize(const std::string& code, const ffi::Array<ffi::String>& func_names) {
  TVM_FFI_ICHECK(!func_names.empty())
      << "Should only create CUTLASS CSourceModule if there is at least one CUTLASS partition";

  std::ostringstream default_headers;
  default_headers << "#include <tvm/ffi/function.h>\n";
  default_headers << "#include <dlpack/dlpack.h>\n";
  default_headers << "#include <cuda_fp16.h>\n";
  default_headers << "#include <cutlass/cutlass.h>\n";
  default_headers << "#include <cutlass/coord.h>\n";
  default_headers << "#include <cutlass/tensor_ref.h>\n";
  default_headers << "#include <cutlass/util/host_tensor.h>\n";

  const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.CSourceModuleCreate");
  VLOG(1) << "Generated CUTLASS code:" << std::endl << code;
  return pf(default_headers.str() + code, "cu", func_names,
            /*const_vars=*/ffi::Array<ffi::String>())
      .cast<ffi::Module>();
}

class CodegenResultObj : public ffi::Object {
 public:
  ffi::String code;
  ffi::Array<ffi::String> headers;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CodegenResultObj>()
        .def_ro("code", &CodegenResultObj::code)
        .def_ro("headers", &CodegenResultObj::headers);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("contrib.cutlass.CodegenResult", CodegenResultObj, ffi::Object);
};

class CodegenResult : public ffi::ObjectRef {
 public:
  CodegenResult(ffi::String code, ffi::Array<ffi::String> headers) {
    auto n = ffi::make_object<CodegenResultObj>();
    n->code = std::move(code);
    n->headers = std::move(headers);
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CodegenResult, ffi::ObjectRef, CodegenResultObj);
};

TVM_FFI_STATIC_INIT_BLOCK() { CodegenResultObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("contrib.cutlass.CodegenResult",
                        [](ffi::String code, ffi::Array<ffi::String> headers) {
                          return CodegenResult(code, headers);
                        });
}

GenerateBodyOutput GenerateBody(const std::string& func_name, const std::string& ext_func_id,
                                const std::vector<std::string>& output_types,
                                const ffi::Array<ffi::String>& func_args,
                                const ffi::Map<ffi::String, ffi::Any>& attrs, int* buf_idx) {
  // Make function call with input buffers when visiting arguements
  TVM_FFI_ICHECK_GT(func_args.size(), 0);
  std::ostringstream decl_stream;
  decl_stream << "(" << func_args[0];
  for (size_t i = 1; i < func_args.size(); ++i) {
    decl_stream << ", " << func_args[i];
  }
  GenerateBodyOutput ret;
  for (const auto& out_type : output_types) {
    const std::string out = "out" + std::to_string(*buf_idx++);
    decl_stream << ", " << out;
    Output output;
    output.name = out;
    output.dtype = out_type;
    output.need_copy = false;
    ret.outputs.push_back(output);
  }
  decl_stream << ");";

  const auto instantiate_template_func =
      tvm::ffi::Function::GetGlobalRequired("contrib.cutlass.instantiate_template");
  CodegenResult codegen_res =
      instantiate_template_func(func_name, attrs, func_args).cast<CodegenResult>();
  ret.decl = codegen_res->code;
  ret.headers = codegen_res->headers;

  return ret;
}

using OutputType = std::vector<Output>;

class CodegenCutlass : public relax::MemoizedExprTranslator<OutputType>,
                       public relax::contrib::CodegenCBase {
 public:
  using relax::MemoizedExprTranslator<OutputType>::VisitExpr_;

  CodegenCutlass(const std::string& id, const ffi::Map<Var, Expr>& bindings)
      : ext_func_id_(id), bindings_(bindings) {}

  void AddParm(Var param) {
    ext_func_args_.push_back(param);
    auto v_name = name_sup_->FreshName(param->name);
    var_name_map_[param.get()] = v_name;
  }

  std::string JIT(const OutputType& out) final {
    std::vector<std::string> arg_types, arg_names;

    for (const auto& arg : ext_func_args_) {
      auto ty = GetType(arg);
      if (const auto* tensor_ty = ty.as<TensorTypeObj>()) {
        arg_types.emplace_back(backend::DType2String(tensor_ty->dtype.value()->dtype));
      } else if (const auto* shape_ty = ty.as<ShapeTypeObj>()) {
        arg_types.emplace_back(backend::DType2String(shape_ty->values.value()[0].ty()->dtype));
      } else {
        TVM_FFI_THROW(InternalError) << "Unimplemented";
      }
      arg_names.push_back(var_name_map_.at(arg.get()));
    }

    code_stream_ << EmitSignature(out, ext_func_id_, arg_names) << "{\n";

    this->EnterScope();

    for (auto decl : buf_decl_) {
      this->PrintIndents();
      code_stream_ << decl << "\n";
    }
    code_stream_ << "\n";
    for (auto stmt : ext_func_body_) {
      this->PrintIndents();
      code_stream_ << stmt << "\n";
    }

    this->ExitScope();
    code_stream_ << "}\n";

    this->GenerateBackendCFunc(ext_func_id_, arg_types, /*const_arr_name=*/"", out, true);
    return code_stream_.str();
  }

  ffi::Array<ffi::String> GetHeaders() { return headers_; }

 protected:
  OutputType VisitExpr_(const VarObj* node) final {
    Output output;
    auto it = var_name_map_.find(node);
    TVM_FFI_ICHECK(it != var_name_map_.end());
    output.name = it->second;
    return {output};
  }

  OutputType VisitExpr_(const CallObj* call) final {
    const auto* fn_var = call->op.as<VarObj>();
    TVM_FFI_ICHECK(fn_var);
    const auto func = bindings_[ffi::GetRef<Var>(fn_var)].as_or_throw<Function>();
    const auto pattern_name_opt = func->GetAttr(tvm::relax::attr::kComposite);
    TVM_FFI_ICHECK(pattern_name_opt) << "Only composite function is supported for CUTLASS.";
    auto ret = GenerateBody(call, pattern_name_opt.value(), func->attrs->dict);
    ext_func_body_.push_back(ret.decl);
    headers_ = ret.headers;
    return ret.outputs;
  }

  OutputType VisitExpr_(const FunctionObj* fn) final {
    TVM_FFI_ICHECK(fn->GetAttr(tvm::relax::attr::kComposite).has_value())
        << "JSON runtime only supports composite functions";
    // FunctionObj should be handled by the caller.
    return {};
  }

  OutputType VisitBinding(const Binding& binding) {
    OutputType outputs;
    if (const auto* node = binding.as<VarBindingObj>()) {
      auto from_b = VisitBinding_(node);
      outputs.insert(outputs.end(), from_b.begin(), from_b.end());
    } else {
      TVM_FFI_THROW(InternalError) << "Unimplemented type: " << binding->GetTypeKey();
    }
    return outputs;
  }

  OutputType VisitBindingBlock(const BindingBlock& block) {
    OutputType outputs;
    if (const auto* node = block.as<DataflowBlockObj>()) {
      auto from_bb = VisitBindingBlock_(node);
      outputs.insert(outputs.end(), from_bb.begin(), from_bb.end());
    } else if (const auto* node = block.as<BindingBlockObj>()) {
      auto from_bb = VisitBindingBlock_(node);
      outputs.insert(outputs.end(), from_bb.begin(), from_bb.end());
    } else {
      TVM_FFI_THROW(InternalError) << "Unimplemented type: " << block->GetTypeKey();
    }
    return outputs;
  }

  OutputType VisitBindingBlock_(const BindingBlockObj* block) {
    OutputType outputs;
    for (Binding binding : block->bindings) {
      auto from_b = VisitBinding(binding);
      outputs.insert(outputs.end(), from_b.begin(), from_b.end());
    }
    return outputs;
  }

  OutputType VisitBindingBlock_(const DataflowBlockObj* block) {
    OutputType outputs;
    for (Binding binding : block->bindings) {
      auto from_b = VisitBinding(binding);
      outputs.insert(outputs.end(), from_b.begin(), from_b.end());
    }
    return outputs;
  }

  OutputType VisitExpr_(const SeqExprObj* op) final {
    OutputType outputs;

    for (BindingBlock block : op->blocks) {
      VisitBindingBlock(block);
    }

    auto from_body = VisitExpr(op->body);
    outputs.insert(outputs.end(), from_body.begin(), from_body.end());

    return outputs;
  }

 private:
  ffi::Array<ffi::String> GetArgumentNames(const CallObj* call) {
    ffi::Array<ffi::String> arg_names;
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto res = VisitExpr(call->args[i]);
      for (const auto& out : res) {
        arg_names.push_back(out.name);
      }
    }
    return arg_names;
  }

  GenerateBodyOutput GenerateBody(const CallObj* call, const std::string& func_name,
                                  const ffi::Map<ffi::String, ffi::Any>& attrs) {
    auto func_args = GetArgumentNames(call);
    auto ty = GetType(ffi::GetRef<Call>(call));

    std::vector<std::string> out_types;
    if (const auto* tensor_ty = ty.as<TensorTypeObj>()) {
      out_types.emplace_back(backend::DType2String(tensor_ty->dtype.value()->dtype));
    } else {
      TVM_FFI_THROW(InternalError) << "Unimplemented ty type: " << ty;
    }

    return contrib::GenerateBody(func_name, ext_func_id_, out_types, func_args, attrs, &buf_idx_);
  }

  /*! \brief The id of the external cutlass ext_func. */
  std::string ext_func_id_;
  /*!
   * \brief The index to track the output buffer. Each kernel will redirect the
   * output to a buffer that may be consumed by other kernels.
   */
  int buf_idx_{0};
  /*! \brief The arguments used by a wrapped function that calls CUTLASS kernels. */
  ffi::Array<Var> ext_func_args_;
  /*! \brief The statements of the function that will be compiled using CUTLASS kernels. */
  std::vector<std::string> ext_func_body_;
  /*! \brief The declaration of intermediate buffers. */
  std::vector<std::string> buf_decl_;
  /*! \brief The binding to look up composite functions. */
  ffi::Map<Var, Expr> bindings_;
  /*! \brief Required header-file names. */
  ffi::Array<ffi::String> headers_;
  /*!
   * \brief A mapping from a variable to its unique name.
   * We use this since sometimes different parameters to the same function end up having the same
   * name.
   */
  std::unordered_map<const VarObj*, std::string> var_name_map_;
  /*! \brief A unique name supply to generate a unique name for each parameter. */
  UniqueNameSupply name_sup_;
};

class CutlassModuleCodegen {
 public:
  ffi::Module CreateCSourceModule(ffi::Array<Function> functions,
                                  const ffi::Map<ffi::String, ffi::Any>& options) {
    std::string headers = "";
    std::string code = "";
    for (const auto& f : functions) {
      auto [f_code, op_headers] = GenCutlassFunc(f, options);
      code += "\n" + f_code;
      for (const auto& header : op_headers) {
        headers += "#include <" + header + ">\n";
      }
    }
    return Finalize(headers + "\n" + code, func_names_);
  }

 private:
  std::pair<std::string, ffi::Array<ffi::String>> GenCutlassFunc(
      const Function& function, const ffi::Map<ffi::String, ffi::Any>& options) {
    TVM_FFI_ICHECK(function.defined()) << "Input error: expect a Relax function.";

    auto sid = GetExtSymbol(function);
    func_names_.push_back(sid);

    CodegenCutlass builder(sid, AnalyzeVar2Value(function));

    for (const auto& p : function->params) {
      builder.AddParm(p);
    }

    auto out = builder.VisitExpr(function->body);
    return {builder.JIT(out), builder.GetHeaders()};
  }

  /*! \brief The accumulated function names. */
  ffi::Array<ffi::String> func_names_;
};

ffi::Array<ffi::Module> CUTLASSCompiler(ffi::Array<Function> functions,
                                        ffi::Map<ffi::String, ffi::Any> options,
                                        ffi::Map<GenericConst, ffi::String> /*unused*/) {
  const auto tune_func = tvm::ffi::Function::GetGlobal("contrib.cutlass.tune_relax_function");
  TVM_FFI_ICHECK(tune_func.has_value())
      << "The packed function contrib.cutlass.tune_relax_function not found, "
         "please import tvm.contrib.cutlass.build";

  auto annotated_functions = (*tune_func)(functions, options).cast<ffi::Array<Function>>();

  auto source_mod = CutlassModuleCodegen().CreateCSourceModule(annotated_functions, options);
  const auto pf = tvm::ffi::Function::GetGlobal("contrib.cutlass.compile");
  TVM_FFI_ICHECK(pf.has_value())
      << "The packed function contrib.cutlass.compile not found, please import "
         "tvm.contrib.cutlass.build";
  ffi::Module cutlass_mod = (*pf)(source_mod, options).cast<ffi::Module>();

  return {cutlass_mod};
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ext.cutlass", CUTLASSCompiler);
}

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
