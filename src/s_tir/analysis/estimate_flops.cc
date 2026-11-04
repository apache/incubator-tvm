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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/function.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/tirx/analysis.h>

#include "tvm/sym/analyzer.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

int32_t DataType2Int(DLDataType dtype) {
  static_assert(sizeof(DLDataType) == sizeof(int32_t), "Incorrect size of DLDataType");
  union {
    DLDataType src;
    int32_t dst;
  } converter;
  converter.src = dtype;
  return converter.dst;
}

ffi::String Int2DataTypeStr(int32_t dtype) {
  union {
    DLDataType dst;
    int32_t src;
  } converter;
  converter.src = dtype;
  static std::string type_code_tab[] = {"int", "uint", "float", "handle", "bfloat"};
  std::ostringstream os;
  os << type_code_tab[converter.dst.code];
  os << static_cast<int>(converter.dst.bits);
  if (converter.dst.lanes != 1) {
    os << "x" << static_cast<int>(converter.dst.lanes);
  }
  return os.str();
}

struct TResult {
  TResult() = default;

  void Add(DLDataType dtype) { data_[DataType2Int(dtype)] += 1; }

  TResult operator+=(const TResult& rhs) {
    for (const auto& kv : rhs.data_) {
      data_[kv.first] += kv.second;
    }
    return *this;
  }

  TResult operator*=(int64_t rhs) {
    for (auto& kv : data_) {
      kv.second *= rhs;
    }
    return *this;
  }

  TResult MaxWith(const TResult& rhs) {
    for (const auto& kv : rhs.data_) {
      double& v = data_[kv.first];
      if (v < kv.second) {
        v = kv.second;
      }
    }
    return *this;
  }

  std::unordered_map<int32_t, double> data_;
};

class FlopEstimator : private tvm::ExprFunctor<TResult(const Expr& n)>,
                      private StmtFunctor<TResult(const Stmt& n)> {
  sym::Analyzer ana;

 public:
  using tvm::ExprFunctor<TResult(const Expr&)>::Dispatch;
  TResult Dispatch(const Stmt& stmt) override { return StmtFunctor::Dispatch(stmt); }

#define TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(Node)       \
  TResult Dispatch_(const Node* op) final {            \
    TResult result = Dispatch(op->a);                  \
    result += Dispatch(op->b);                         \
    result.Add(op->ty.as_or_throw<PrimType>()->dtype); \
    return result;                                     \
  }
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::AddObj);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::SubObj);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::MulObj);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::DivObj);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::ModObj);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::FloorDivObj);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::FloorModObj);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::MinObj);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::MaxObj);
#undef TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY
  TResult Dispatch_(const prim::EQObj* op) override { return TResult(); }
  TResult Dispatch_(const prim::NEObj* op) override { return TResult(); }
  TResult Dispatch_(const prim::LTObj* op) override { return TResult(); }
  TResult Dispatch_(const prim::LEObj* op) override { return TResult(); }
  TResult Dispatch_(const prim::GTObj* op) override { return TResult(); }
  TResult Dispatch_(const prim::GEObj* op) override { return TResult(); }

  int64_t GetLoopExtent(const ForObj* node, const sym::Analyzer& ana) {
    int64_t bound =
        ana->const_int_bound(Range::GetExtentFromBeginEnd(node->begin, node->end))->max_value;
    if (bound == sym::ConstIntBound::kPosInf) {
      return 1;  // Analyzer could not determine a valid bound, use 1 instead.
    } else {
      return bound;
    }
  }

  TResult Dispatch_(const prim::LShiftObj* op) final {
    TResult result = Dispatch(op->a);
    result += Dispatch(op->b);
    return result;
  }
  TResult Dispatch_(const prim::RShiftObj* op) final {
    TResult result = Dispatch(op->a);
    result += Dispatch(op->b);
    return result;
  }
  TResult Dispatch_(const prim::BitwiseAndObj* op) final {
    TResult result = Dispatch(op->a);
    result += Dispatch(op->b);
    return result;
  }
  TResult Dispatch_(const prim::BitwiseOrObj* op) final {
    TResult result = Dispatch(op->a);
    result += Dispatch(op->b);
    return result;
  }
  TResult Dispatch_(const prim::BitwiseXorObj* op) final {
    TResult result = Dispatch(op->a);
    result += Dispatch(op->b);
    return result;
  }
  TResult Dispatch_(const prim::BitwiseNotObj* op) final { return Dispatch(op->a); }
  TResult Dispatch_(const prim::NotObj* op) override { return Dispatch(op->a); }
  TResult Dispatch_(const prim::AndObj* op) final {
    TResult result = Dispatch(op->a);
    result += Dispatch(op->b);
    return result;
  }
  TResult Dispatch_(const prim::OrObj* op) final {
    TResult result = Dispatch(op->a);
    result += Dispatch(op->b);
    return result;
  }

  TResult Dispatch_(const TensorLoadObj* op) override { return TResult(); }
  TResult Dispatch_(const RegionStmtObj* op) override {
    TResult result = Dispatch(op->body);
    for (const Expr& arg : op->args) result += Dispatch(arg);
    return result;
  }
  TResult Dispatch_(const TensorStoreObj* store) override { return Dispatch(store->value); }
  TResult Dispatch_(const SBlockRealizeObj* block) override { return Dispatch(block->block->body); }
  TResult Dispatch_(const SBlockObj* block) override {
    TResult result;
    if (block->init.has_value()) {
      result += Dispatch(block->init.value());
    }
    result += Dispatch(block->body);
    return result;
  }
  TResult Dispatch_(const ForObj* loop) override {
    auto outer_context = ana->GetConstraintContext();
    auto context = outer_context->Snapshot();
    context->Append(sym::RangeConstraint(loop->var, Range(loop->begin, loop->end)));
    ana->SyncConstraintContext(context);
    const auto int_imm = GetLoopExtent(loop, ana);
    TResult result = Dispatch(loop->body);
    ana->SyncConstraintContext(outer_context);
    result *= int_imm;
    return result;
  }

  TResult Dispatch_(const IfObj* branch) override {
    TResult cond = Dispatch(branch->condition);
    if (branch->else_case) {
      cond += Dispatch(branch->then_case).MaxWith(Dispatch(branch->else_case.value()));
    } else {
      cond += Dispatch(branch->then_case);
    }
    return cond;
  }

  TResult Dispatch_(const WhileObj* op) override {
    // TODO(jikechao): Improve while loop FLOP estimation with loop bound analysis
    TResult result = Dispatch(op->condition);
    result += Dispatch(op->body);
    return result;
  }

  TResult Dispatch_(const BindObj* let) override {
    if (auto value = let->value.as<PrimExpr>()) return Dispatch(value.value());
    return TResult();
  }

  TResult Dispatch_(const prim::SelectObj* op) override {
    TResult cond = Dispatch(op->condition);
    cond += Dispatch(op->true_value).MaxWith(Dispatch(op->false_value));
    return cond;
  }

  TResult Dispatch_(const AssertStmtObj* op) override {
    TResult result = Dispatch(op->condition);
    return result;
  }

  TResult Dispatch_(const VarObj* op) override { return TResult(); }
  TResult Dispatch_(const IntImmObj* op) override { return TResult(); }
  TResult Dispatch_(const FloatImmObj* op) override { return TResult(); }
  TResult Dispatch_(const StringImmObj* op) override { return TResult(); }
  TResult Dispatch_(const prim::CastObj* op) override { return Dispatch(op->value); }
  TResult Dispatch_(const EvaluateObj* op) override { return TResult(); }

  TResult Dispatch_(const SeqStmtObj* seq) override {
    TResult result;
    for (const Stmt& stmt : seq->seq) {
      result += Dispatch(stmt);
    }
    return result;
  }

  TResult Dispatch_(const CallObj* op) override {
    TResult ret;
    for (const Expr& arg : op->args) {
      ret += Dispatch(arg);
    }
    return ret;
  }
};

double PostprocessResults(const TResult& result) {
  double cnt = 0.0;
  for (const auto& kv : result.data_) {
    cnt += kv.second;
  }
  return cnt;
}

double EstimateTIRFlops(const Stmt& stmt) {
  FlopEstimator counter;
  return PostprocessResults(counter.Dispatch(stmt));
}

double EstimateTIRFlops(const IRModule& mod) {
  FlopEstimator counter;
  TResult result;
  double cached_result = 0;
  for (const auto& [_, base_func] : mod->functions) {
    if (const auto* func = base_func.as<FunctionObj>()) {
      if (auto cached = func->attrs.GetAttr(tvm::s_tir::attr::kEstimatedFlops)) {
        cached_result += cached.value();
      } else if (func->body.has_value()) {
        result += counter.Dispatch(func->body.value());
      }
    }
  }
  return PostprocessResults(result) + cached_result;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.analysis.EstimateTIRFlops", [](ffi::ObjectRef obj) -> double {
    if (auto mod = obj.as<IRModule>()) {
      return EstimateTIRFlops(mod.value());
    } else if (auto stmt = obj.as<Stmt>()) {
      return EstimateTIRFlops(stmt.value());
    } else {
      TVM_FFI_THROW(TypeError) << "Expect the input to be either IRModule or Stmt, but gets: "
                               << obj->GetTypeKey();
      throw;
    }
  });
}

}  // namespace s_tir
}  // namespace tvm
