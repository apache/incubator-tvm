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
 * \file verify_gpu_code.cc
 * \brief Verify the correctness of a GPU IR.
 *        It will check the whether the amount of memory usage or the number of threads
 *        in a block exceeds the limit
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt.h>

#include <algorithm>

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

class GPUCodeVerifier : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  std::vector<ffi::String> Verify(Stmt stmt, int64_t max_local_memory_per_block,
                                  int64_t max_shared_memory_per_block,
                                  int64_t max_threads_per_block, int64_t max_thread_x,
                                  int64_t max_thread_y, int64_t max_thread_z, int64_t max_vthread,
                                  int64_t max_vector_bytes, int64_t max_kernels) {
    max_local_memory_per_block_ = static_cast<size_t>(max_local_memory_per_block);
    max_shared_memory_per_block_ = static_cast<size_t>(max_shared_memory_per_block);
    max_threads_per_block_ = static_cast<size_t>(max_threads_per_block);
    max_thread_x_ = static_cast<size_t>(max_thread_x);
    max_thread_y_ = static_cast<size_t>(max_thread_y);
    max_thread_z_ = static_cast<size_t>(max_thread_z);
    max_vthread_ = static_cast<size_t>(max_vthread);
    max_vector_bytes_ = static_cast<size_t>(max_vector_bytes);
    max_kernels_ = static_cast<size_t>(max_kernels);
    Reset_();

    // TODO(jcf94): Add support of detecting CUDA Misaligned Address error
    this->Visit(stmt);

    return errors_;
  }

  ffi::Optional<VisitInterrupt> Visit_(const BindObj* op) final {
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      return DispatchAllocTensor(op, call);
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> DispatchAllocTensor(const BindObj* op, const CallObj* call) {
    static const auto scope_local = runtime::MemoryScope::Get("local");
    static const auto scope_shared = runtime::MemoryScope::Get("shared");
    static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    tvm::Tuple shape = tvm::Tuple(call->ty_args[0].as_or_throw<TensorType>()->shape);
    DLDataType dtype = call->ty_args[0].as_or_throw<TensorType>()->dtype->dtype;
    runtime::MemoryScope scope = call->ty_args[0].as_or_throw<TensorType>()->scope;
    runtime::MemoryScope storage_scope = scope;
    int64_t const_size = 1;
    for (const Expr& e : shape->fields) {
      if (auto* imm = e.as<IntImmObj>()) {
        const_size = static_cast<int64_t>(const_size * imm->value);
      } else {
        const_size = 0;
        break;
      }
    }
    PrimType dtype_ty(dtype);
    TVM_FFI_ICHECK(!dtype_ty.IsScalableVector())
        << "Cannot verify GPU memory usage for scalable vector dtype " << dtype_ty;
    if (storage_scope == scope_local) {
      local_memory_per_block_ += static_cast<size_t>(const_size) * ElementBytes(dtype_ty);
    } else if ((storage_scope == scope_shared || storage_scope == scope_shared_dyn)) {
      shared_memory_per_block_ += static_cast<size_t>(const_size) * ElementBytes(dtype_ty);
    }
    if (dtype_ty.IsFixedLengthVector()) {
      if (ElementBytes(dtype_ty) > max_vector_bytes_) {
        std::stringstream s;
        s << "Number of lanes (" << dtype_ty.lanes() << ") times number of bytes ("
          << dtype_ty.StorageBytes() << ") for dtype " << dtype_ty
          << " is greater than the maximum number of vector bytes (" << max_vector_bytes_ << ")";
        errors_.push_back(s.str());
      }
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj* op) final {
    if (op->op.same_as(tirx::launch_thread_op())) {
      if (nest_level_ == 0) {
        // enter a new kernel, reset statistics
        Reset_();
        kernels_launched_++;
      }

      const auto* extent = op->args[1].as_or_throw<PrimExpr>().as<IntImmObj>();
      TVM_FFI_ICHECK(extent);

      std::string name = op->args[0].as_or_throw<StringImm>()->value;
      size_t previous_virtual_extent = active_virtual_extent_;
      if (name.rfind("vthread", 0) == 0) {
        // Virtual launches are lexical loops; equal tags do not identify the
        // same binding. Sibling loops contribute their maximum, nested loops
        // their product.
        size_t length = extent->value.as<size_t>().value();
        if (length > max_vthread_) {
          std::stringstream s;
          s << "Extent of " << name << " (" << length << ") is greater than maximum allowed ("
            << max_vthread_ << ");";
          errors_.push_back(s.str());
        }
        active_virtual_extent_ *= length;
        max_virtual_extent_ = std::max(max_virtual_extent_, active_virtual_extent_);
      } else if (name == "threadIdx.x" || name == "threadIdx.y" || name == "threadIdx.z") {
        size_t length = extent->value.as<size_t>().value();
        if (!visited_threads_.count(name)) {
          visited_threads_.insert(name);
          thread_per_block_ *= length;

          auto err = [this](std::string id, size_t ext, size_t m) {
            if (ext > m) {
              std::stringstream s;
              s << "Extent of " << id << " (" << ext << ") is greater than maximum allowed (" << m
                << ");";
              errors_.push_back(s.str());
            }
          };

          if (name == "threadIdx.x") {
            err("threadIdx.x", length, max_thread_x_);
            thread_x_extent_ = length;
          } else if (name == "threadIdx.y") {
            err("threadIdx.y", length, max_thread_y_);
            thread_y_extent_ = length;
          } else if (name == "threadIdx.z") {
            err("threadIdx.z", length, max_thread_z_);
            thread_z_extent_ = length;
          }
        } else {
          // the thread should be bound to axes with the same length
          auto err = [this, name](std::string id, size_t ext, size_t m) {
            if (name == id && ext != m) {
              std::stringstream s;
              s << "Extent of " << id << " (" << ext << ") does not match the bound " << m;
              errors_.push_back(s.str());
            }
          };
          err("threadIdx.x", length, thread_x_extent_);
          err("threadIdx.y", length, thread_y_extent_);
          err("threadIdx.z", length, thread_z_extent_);
        }
      }

      nest_level_++;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
      nest_level_--;
      active_virtual_extent_ = previous_virtual_extent;

      if (nest_level_ == 0) {
        // exit a kernel, check the validity
        auto err = [this](std::string id, size_t num, size_t m) {
          if (num > m) {
            std::stringstream s;
            s << "Used " << id << " (" << num << ") is greater than the allowed maximum (" << m
              << ")";
            errors_.push_back(s.str());
          }
        };
        err("threads per block", thread_per_block_ * max_virtual_extent_, max_threads_per_block_);
        err("local memory per block", local_memory_per_block_, max_local_memory_per_block_);
        err("shared memory per block", shared_memory_per_block_, max_shared_memory_per_block_);

        if (kernels_launched_ > max_kernels_) {
          std::stringstream s;
          s << "Number of launched kernels (" << kernels_launched_
            << ") is greater than the allowed maximum (" << max_kernels_ << ")";
          errors_.push_back(s.str());
        }
      }
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) {
    if (op->var->name == "vthread.s") {
      PrimExpr extent_expr = Range::GetExtentFromBeginEnd(op->begin, op->end);
      const auto* extent = extent_expr.as<IntImmObj>();
      TVM_FFI_ICHECK(extent);

      size_t num_vthread = extent->value.as<size_t>().value();
      if (num_vthread > max_vthread_) {
        std::stringstream s;
        s << "Number of vthreads (" << num_vthread << ") is greater than the allowed maximum ("
          << max_vthread_ << ")";
        errors_.push_back(s.str());
      }
    }

    return StmtExprVisitor::Visit_(op);
  }

  void CheckTensorIndicesVectorizable(const ffi::Array<PrimExpr> indices) {
    for (const auto index : indices) {
      if (const auto* ramp = index.as<RampObj>()) {
        PrimType ramp_ty = ramp->ty.as_or_throw<PrimType>();
        if (!IsOne(ramp->stride) && ramp_ty.IsFixedLengthVector() &&
            ElementBytes(ramp_ty) > max_vector_bytes_) {
          std::stringstream s;
          s << "Number of lanes (" << ramp_ty.lanes() << ") times number of bytes ("
            << ramp_ty.StorageBytes() << ") for dtype " << ramp_ty
            << " is greater than the maximum number of vector bytes (" << max_vector_bytes_ << ")";
          errors_.push_back(s.str());
        }
      }
    }
  }

  ffi::Optional<VisitInterrupt> Visit_(const CastObj* op) {
    PrimType op_ty = op->ty.as_or_throw<PrimType>();
    if (op_ty.IsFixedLengthVector()) {
      if (ElementBytes(op_ty) > max_vector_bytes_) {
        std::stringstream s;
        s << "Number of lanes (" << op_ty.lanes() << ") times number of bytes ("
          << op_ty.StorageBytes() << ") for dtype " << op_ty
          << " is greater than the maximum number of vector bytes (" << max_vector_bytes_ << ")";
        errors_.push_back(s.str());
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) {
    PrimType op_ty = op->ty.as_or_throw<PrimType>();
    if (op_ty.IsFixedLengthVector()) {
      if (ElementBytes(op_ty) > max_vector_bytes_) {
        std::stringstream s;
        s << "Number of lanes (" << op_ty.lanes() << ") times number of bytes ("
          << op_ty.StorageBytes() << ") for dtype " << op_ty
          << " is greater than the maximum number of vector bytes (" << max_vector_bytes_ << ")";
        errors_.push_back(s.str());
      }
      CheckTensorIndicesVectorizable(op->indices);
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) {
    PrimType value_ty = op->value.ty();
    if (value_ty.IsFixedLengthVector()) {
      if (ElementBytes(value_ty) > max_vector_bytes_) {
        std::stringstream s;
        s << "Number of lanes (" << value_ty.lanes() << ") times number of bytes ("
          << value_ty.StorageBytes() << ") for dtype " << value_ty
          << " is greater than the maximum number of vector bytes (" << max_vector_bytes_ << ")";
        errors_.push_back(s.str());
      }
      CheckTensorIndicesVectorizable(op->indices);
    }
    return StmtExprVisitor::Visit_(op);
  }

 private:
  int nest_level_{0};

  std::unordered_set<std::string> visited_threads_;

  size_t thread_x_extent_, thread_y_extent_, thread_z_extent_;

  size_t local_memory_per_block_;
  size_t shared_memory_per_block_;
  size_t thread_per_block_;
  size_t active_virtual_extent_;
  size_t max_virtual_extent_;
  size_t kernels_launched_{0};

  size_t max_local_memory_per_block_;
  size_t max_shared_memory_per_block_;
  size_t max_threads_per_block_;
  size_t max_thread_x_, max_thread_y_, max_thread_z_, max_vthread_;
  size_t max_vector_bytes_;
  size_t max_kernels_;

  std::vector<ffi::String> errors_;

  static size_t ElementBytes(const PrimType& ty) { return ty.StorageBytes(); }

  void Reset_() {
    local_memory_per_block_ = 0;
    shared_memory_per_block_ = 0;

    visited_threads_.clear();
    thread_per_block_ = 1;
    thread_x_extent_ = thread_y_extent_ = thread_z_extent_ = 1;
    active_virtual_extent_ = 1;
    max_virtual_extent_ = 1;
  }
};

std::vector<ffi::String> VerifyGPUCode_(const Function& func,
                                        ffi::Map<ffi::String, PrimExpr> constraints) {
  auto verifier = ffi::make_object<GPUCodeVerifier>();

  int64_t max_local_memory_per_block = INT64_MAX;
  int64_t max_shared_memory_per_block = INT64_MAX;
  int64_t max_threads_per_block = INT64_MAX;
  int64_t max_thread_x = INT64_MAX;
  int64_t max_thread_y = INT64_MAX;
  int64_t max_thread_z = INT64_MAX;
  int64_t max_vthread = INT64_MAX;
  int64_t max_vector_bytes = INT64_MAX;
  int64_t max_kernels = INT64_MAX;

  for (auto iter : constraints) {
    const IntImmObj* val = iter.second.as<IntImmObj>();
    if (iter.first == "max_local_memory_per_block") {
      max_local_memory_per_block = static_cast<int64_t>(val->value);
    } else if (iter.first == "max_shared_memory_per_block") {
      max_shared_memory_per_block = static_cast<int64_t>(val->value);
    } else if (iter.first == "max_threads_per_block") {
      max_threads_per_block = static_cast<int64_t>(val->value);
    } else if (iter.first == "max_thread_x") {
      max_thread_x = static_cast<int64_t>(val->value);
    } else if (iter.first == "max_thread_y") {
      max_thread_y = static_cast<int64_t>(val->value);
    } else if (iter.first == "max_thread_z") {
      max_thread_z = static_cast<int64_t>(val->value);
    } else if (iter.first == "max_vthread") {
      max_vthread = static_cast<int64_t>(val->value);
    } else if (iter.first == "max_vector_bytes") {
      max_vector_bytes = static_cast<int64_t>(val->value);
    } else if (iter.first == "max_kernels") {
      max_kernels = static_cast<int64_t>(val->value);
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid check item: " << iter.first;
    }
  }

  if (!func->body.has_value()) return {};
  return verifier->Verify(func->body.value(), max_local_memory_per_block,
                          max_shared_memory_per_block, max_threads_per_block, max_thread_x,
                          max_thread_y, max_thread_z, max_vthread, max_vector_bytes, max_kernels);
}

bool VerifyGPUCode(const Function& func, ffi::Map<ffi::String, PrimExpr> constraints) {
  auto errs = VerifyGPUCode_(func, constraints);
  return errs.size() == 0;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.analysis.verify_gpu_code", VerifyGPUCode);
}

namespace transform {

Pass VerifyGPUCode(ffi::Map<ffi::String, PrimExpr> constraints) {
  auto pass_func = [=](IRModule mod, PassContext ctx) {
    for (auto kv : mod->functions) {
      if (auto func = kv.second.as<Function>()) {
        auto errs = VerifyGPUCode_(func.value(), constraints);
        if (errs.size() != 0) {
          std::stringstream s;
          for (auto& err : errs) {
            s << "    " << err << std::endl;
          }
          TVM_FFI_THROW(RuntimeError) << "GPU constraint(s) violated:\n"
                                      << s.str() << "  In function\n"
                                      << func.value();
        }
      }
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "s_tir.VerifyGPUCode");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.VerifyGPUCode", VerifyGPUCode);
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
