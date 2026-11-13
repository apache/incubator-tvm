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

/*! \brief Check if schedulable TIR is well-formed. */

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/tirx/type.h>

#include <algorithm>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace s_tir {
using namespace tirx;
namespace {

/*! \brief Verify all Expr inside the block does not contain:
 *    1. loop vars outside the current block.
 *    2. block vars of parent blocks.
 */
class BlockVarAccessVerifier : public StmtExprVisitor {
 public:
  static bool Verify(const Function& func, bool assert_mode) {
    auto verifier = ffi::make_object<BlockVarAccessVerifier>(assert_mode);
    verifier->Visit(func->body);
    return !verifier->has_error_;
  }

  explicit BlockVarAccessVerifier(bool assert_mode) : assert_mode_(assert_mode) {}

 private:
  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView stmt) final {
    if (!has_error_) {
      return StmtExprVisitor::Visit(stmt);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarObj* op) final {
    auto it = loop_vars_.find(op);
    if (it != loop_vars_.end() && it->second < block_stack_.size()) {
      has_error_ = true;
      if (assert_mode_) {
        if (it->second == 0) {
          TVM_FFI_THROW(InternalError)
              << "Well-formedness check failed: "
              << "Loop iterator var " << op->name << " is defined outside of any block, "
              << "but is used inside the non-opaque current block \""
              << block_stack_.back()->name_hint << "\".";
        } else {
          TVM_FFI_THROW(InternalError)
              << "Well-formedness check failed: "
              << "Loop iterator var " << op->name << " is defined in block \""
              << block_stack_[it->second - 1]->name_hint << "\", "
              << "but is used inside the non-opaque current block \""
              << block_stack_.back()->name_hint << "\".";
        }
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) final {
    TVM_FFI_ICHECK(loop_vars_.find(op->var.get()) == loop_vars_.end());
    loop_vars_[op->var.get()] = block_stack_.size();
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    loop_vars_.erase(op->var.get());
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* op) final {
    // Do not check boundary if it's a opaque block.
    bool is_non_opaque = op->iter_vars.size();
    if (is_non_opaque) {
      block_stack_.push_back(op);
    }

    // Step 0. Skip block iter var's domain

    // Step 1. Visit read/write regions
    auto fvisit_tensor_region = [this](const TensorRegion& s) -> ffi::Optional<VisitInterrupt> {
      for (const auto& range : s->region) {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(range->begin));
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(range->extent));
      }
      return std::nullopt;
    };
    for (const auto& region : op->reads) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(fvisit_tensor_region(region));
    }
    for (const auto& region : op->writes) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(fvisit_tensor_region(region));
    }

    // Step 2. Visit match tensors
    for (const auto& match_tensor_region : op->match_tensors) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(fvisit_tensor_region(match_tensor_region->source));
    }

    // Step 3. Visit init and body
    if (op->init.has_value()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->init.value()));
    }
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->body));

    if (is_non_opaque) {
      block_stack_.pop_back();
    }
    return std::nullopt;
  }

 private:
  /*! \brief The map from outside loop vars to its corresponding block level. */
  std::unordered_map<const VarObj*, size_t> loop_vars_;
  /*! \brief Whether it's in assert mode. */
  bool assert_mode_;
  /*! \brief Current nested block stack level. */
  std::vector<const SBlockObj*> block_stack_;
  /*! \brief Whether there is error. */
  bool has_error_{false};
};

// This state and traversal belong to well-formedness checking. In particular,
// ordinary visitors do not impose a definition/use or lexical-scope policy.
class WellFormednessVerifier : public StmtExprVisitor {
  friend class tvm::ObjectVisitor;

 public:
  enum class Check { kVariables, kTensors, kTensorLoadType, kLoopControl, kThreadBinding };
  using AccessPath = ffi::reflection::AccessPath;
  using Result = ffi::Optional<VisitInterrupt>;

  WellFormednessVerifier(Check check, bool assert_mode)
      : WellFormednessVerifier(GlobalVTable(), check, assert_mode) {}

  bool Run(const ffi::ObjectRef& root) {
    root_ = root;
    try {
      this->Visit(root);
    } catch (const ffi::Error& error) {
      // Resolve at the verifier boundary for both direct and pass-driven calls.
      // Once reported, consume the visit context to avoid a second enrichment.
      auto context = ffi::VisitErrorContext::TryGetFromError(error);
      if (!context) throw;
      auto normalized = ffi::make_object<ffi::VisitErrorContextObj>();
      normalized->prev_error_context = context.value()->prev_error_context;
      for (const auto& object : context.value()->reverse_visit_pattern) {
        if (!IsContainer(object)) normalized->reverse_visit_pattern.push_back(object);
      }
      auto location = ffi::VisitErrorContext(std::move(normalized));
      auto paths = ffi::VisitErrorContext::FindAccessPaths(root, location);
      if (paths.empty()) throw;
      std::ostringstream path;
      path << paths[0];
      std::string message = error.message();
      const std::string marker = "<current location>";
      size_t pos = 0;
      bool replaced = false;
      while ((pos = message.find(marker, pos)) != std::string::npos) {
        message.replace(pos, marker.size(), path.str());
        pos += path.str().size();
        replaced = true;
      }
      if (!replaced) message += " at " + path.str();
      throw ffi::Error(error.kind(), message, error.backtrace(), error.cause_chain(),
                       location->prev_error_context.has_value()
                           ? std::optional<ffi::ObjectRef>(location->prev_error_context.value())
                           : std::nullopt);
    }
    return !has_error_;
  }

 protected:
  using VTable = StmtExprVisitor::VTable;
  using StmtExprVisitor::Visit_;
  explicit WellFormednessVerifier(const VTable* table, Check check, bool assert_mode)
      : StmtExprVisitor(table), check_(check), table_(table), assert_mode_(assert_mode) {}

  static void InitVTable(VTable* table) {
    StmtExprVisitor::InitVTable(table);
    StmtExprVisitor::SetDispatch<WellFormednessVerifier, IRModuleObj>(table);
    StmtExprVisitor::SetDispatch<WellFormednessVerifier, FunctionObj>(table);
    StmtExprVisitor::SetDispatch<WellFormednessVerifier, LambdaExprObj>(table);
  }

  Result Visit(ffi::AnyView value) override {
    auto object = value.as<ffi::ObjectRef>();
    if (!object) return StmtExprVisitor::Visit(value);
    visit_stack_.push_back(object.value());
    struct Pop {
      std::vector<ffi::ObjectRef>* stack;
      ~Pop() { stack->pop_back(); }
    } pop{&visit_stack_};
    // Function entrypoints and module members are roots of verification;
    // a Function embedded as an expression remains unsupported.
    if (value.as<Function>() && visit_stack_.size() != 1 &&
        !(visit_stack_.size() == 2 && root_.as<IRModule>())) {
      TVM_FFI_VISIT_THROW(TypeError, object.value())
          << "Unsupported non-primitive TIR expression " << object.value()->GetTypeKey();
    }
    if (value.as<Stmt>() && !table_->CanDispatch(value)) {
      TVM_FFI_VISIT_BEGIN();
      Verify(false) << "TIR verifier does not support statement " << object.value()->GetTypeKey()
                    << " at <current location>";
      return std::nullopt;
      TVM_FFI_VISIT_END(object.value());
    }
    if (value.as<Expr>() && !table_->CanDispatch(value)) {
      if (value.as<OpaqueExprObj>()) return std::nullopt;
      if (auto var = value.as<VarObj>()) {
        TVM_FFI_VISIT_BEGIN();
        return this->Visit_(var);
        TVM_FFI_VISIT_END(object.value());
      }
      TVM_FFI_VISIT_THROW(TypeError, object.value())
          << "Unsupported non-primitive TIR expression " << object.value()->GetTypeKey();
    }
    return StmtExprVisitor::Visit(value);
  }

  class VerifyStream {
   public:
    explicit VerifyStream(bool fail) {
      if (fail) message_.emplace();
    }
    VerifyStream(const VerifyStream&) = delete;
    VerifyStream(VerifyStream&& other) { std::swap(message_, other.message_); }
    template <typename T>
    VerifyStream& operator<<(T&& value) {
      if (message_) *message_ << std::forward<T>(value);
      return *this;
    }
    ~VerifyStream() noexcept(false) {
      if (message_) LOG(FATAL) << message_->str();
    }

   private:
    std::optional<std::ostringstream> message_;
  };

  VerifyStream Verify(bool condition) {
    has_error_ |= !condition;
    return VerifyStream(!condition && assert_mode_);
  }

  // Only earlier definitions need a saved occurrence. Keep their enclosing
  // objects and the defining field, rather than searching for a Var that may
  // also occur at arbitrarily many use sites.
  struct DefinitionSite {
    ffi::VisitErrorContext context;
    AccessPath field;
  };

  // FindAccessPaths descends through containers without matching their identity.
  // Native visiting can record them; keep only matchable object breadcrumbs.
  static bool IsContainer(const ffi::ObjectRef& object) {
    switch (object->type_index()) {
      case ffi::TypeIndex::kTVMFFIArray:
      case ffi::TypeIndex::kTVMFFIList:
      case ffi::TypeIndex::kTVMFFIMap:
      case ffi::TypeIndex::kTVMFFIDict:
        return true;
      default:
        return false;
    }
  }

  DefinitionSite DefinitionAt(AccessPath field) const {
    auto context = ffi::make_object<ffi::VisitErrorContextObj>();
    for (auto it = visit_stack_.rbegin(); it != visit_stack_.rend(); ++it) {
      if (!IsContainer(*it)) context->reverse_visit_pattern.push_back(*it);
    }
    return {ffi::VisitErrorContext(std::move(context)), std::move(field)};
  }

  AccessPath DefinitionPath(const DefinitionSite& site) const {
    auto paths = ffi::VisitErrorContext::FindAccessPaths(root_, site.context);
    auto path = paths.empty() ? AccessPath::Root() : paths[0];
    for (const auto& step : site.field->ToSteps()) path = path->Extend(step);
    return path;
  }

  template <typename F>
  Result WithScope(F&& body) {
    return scopes_.WithNewScope([&]() -> Result {
      Result result = body();
      // Definition exit never diagnoses an error. On an exception the entire
      // verifier is discarded, so only normal completion needs history updates.
      for (auto it = scopes_.Current().rbegin(); it != scopes_.Current().rend(); ++it) {
        current_.erase(it->first);
        previous_.insert(*it);
      }
      return result;
    });
  }

  void Define(const Var& var, AccessPath field, bool visit_type = true) {
    if (visit_type) this->Visit(var->ty);
    auto site = DefinitionAt(std::move(field));
    if (check_ == Check::kVariables) {
      {
        auto it = current_.find(var);
        auto verify = Verify(it == current_.end());
        verify << "ValueError: TIR is ill-formed, due to multiple nested definitions of variable "
               << var->name << ".";
        if (it != current_.end()) {
          verify << " It was first defined at " << DefinitionPath(it->second)
                 << ", and was re-defined at <current location>";
        }
      }
      {
        auto it = previous_.find(var);
        auto verify = Verify(it == previous_.end());
        verify << "ValueError: TIR is ill-formed, due to multiple definitions of variable "
               << var->name << ".";
        if (it != previous_.end()) {
          verify << " It was first defined at " << DefinitionPath(it->second)
                 << ", and was later re-defined at <current location>";
        }
      }
    }
    current_.insert({var, site});
    scopes_.Current().emplace_back(var, std::move(site));
  }

  void MatchTensorDefinitions(const TensorVar& tensor, AccessPath field) {
    auto define_if_needed = [&](const Var& var, AccessPath site) {
      if (!current_.count(var)) Define(var, std::move(site));
    };
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
      auto site = field->Attr("shape")->ArrayItem(i);
      auto walk = [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
        define_if_needed(var, site);
        return ffi::WalkResult::Advance();
      };
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(tensor.shape()[i], walk);
    }
    if (const auto* tile = tensor.ty()->layout.as<TileLayoutObj>();
        tile && MatchShapeStridePattern(tensor.ty()).has_value()) {
      auto site = field->Attr("layout");
      for (size_t i = 0; i < tile->shard.size(); ++i) {
        if (auto var = tile->shard[i]->stride.as<Var>()) {
          define_if_needed(var.value(), site->Attr("shard")->ArrayItem(i)->Attr("stride"));
        }
      }
      for (const auto& [axis, offset] : tile->offset) {
        auto walk = [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
          define_if_needed(var, site->Attr("offset"));
          return ffi::WalkResult::Advance();
        };
        ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(offset, walk);
      }
    }
  }

  Result Visit_(const IRModuleObj* op) {
    std::vector<GlobalVar> globals;
    for (const auto& [global, function] : op->functions) globals.push_back(global);
    std::sort(globals.begin(), globals.end(), [&](const GlobalVar& a, const GlobalVar& b) {
      bool a_public = op->functions[a]->GetAttr(tvm::attr::kGlobalSymbol).has_value();
      bool b_public = op->functions[b]->GetAttr(tvm::attr::kGlobalSymbol).has_value();
      return a_public != b_public ? a_public > b_public : a->name_hint < b->name_hint;
    });
    for (const auto& global : globals) {
      if (auto function = op->functions[global].as<Function>()) this->Visit(function.value());
    }
    return std::nullopt;
  }

  Result Visit_(const FunctionObj* op) {
    int enclosing_depth = loop_depth_;
    loop_depth_ = 0;
    WithScope([&]() -> Result {
      auto params = AccessPath::Root()->Attr("params");
      for (size_t i = 0; i < op->params.size(); ++i) {
        if (!op->params[i]->ty.as<TensorTypeObj>()) Define(op->params[i], params->ArrayItem(i));
      }
      for (size_t i = 0; i < op->params.size(); ++i) {
        if (auto tensor = op->params[i].as<TensorVar>()) {
          MatchTensorDefinitions(tensor.value(), params->ArrayItem(i)->Attr("ty"));
        }
      }
      for (size_t i = 0; i < op->params.size(); ++i) {
        if (op->params[i].as<TensorVar>()) Define(op->params[i], params->ArrayItem(i));
      }
      return WithScope([&]() { return this->Visit(op->body); });
    });
    loop_depth_ = enclosing_depth;
    if (check_ == Check::kTensors) {
      current_.clear();
      previous_.clear();
    }
    return std::nullopt;
  }

  Result Visit_(const VarObj* op) override {
    auto var = ffi::GetRef<Var>(op);
    if (check_ == Check::kVariables) {
      auto verify = Verify(current_.count(var));
      verify << "ValueError: Invalid use of undefined variable " << var->name
             << " at <current location>.";
      if (auto it = previous_.find(var); it != previous_.end()) {
        verify << ".  While this variable was previously defined at " << DefinitionPath(it->second)
               << ", this definition is no longer in-scope.";
      }
    }
    return std::nullopt;
  }

  void TensorUse(const TensorVar& tensor) {
    if (check_ != Check::kTensors) return;
    if (current_.count(tensor)) return;
    auto it = previous_.find(tensor);
    auto verify = Verify(false);
    verify << "TIR is ill-formed: tensor " << tensor->name << " is used at <current location>";
    if (it != previous_.end()) {
      verify << " but its declaration is no longer in-scope. It was declared at "
             << DefinitionPath(it->second) << ".";
    } else {
      verify << " without a prior DeclTensor or other declaration.";
    }
  }

  Result Visit_(const BindObj* op) override {
    this->Visit(op->value);
    Define(op->var, AccessPath::Root()->Attr("var"));
    return std::nullopt;
  }

  Result Visit_(const RegionStmtObj* op) override {
    this->Visit(op->args);
    this->Visit(op->attrs);
    for (const auto& param : op->body_params) this->Visit(param->ty);
    WithScope([&]() -> Result {
      for (size_t i = 0; i < op->body_params.size(); ++i) {
        Define(op->body_params[i], AccessPath::Root()->Attr("body_params")->ArrayItem(i), false);
      }
      return this->Visit(op->body);
    });
    for (size_t i = 0; i < op->result_vars.size(); ++i) {
      Define(op->result_vars[i], AccessPath::Root()->Attr("result_vars")->ArrayItem(i));
    }
    return std::nullopt;
  }

  Result Visit_(const ForObj* op) override {
    if (check_ == Check::kThreadBinding) {
      if (auto tag = op->annotations.Get(attr::kThreadBinding.name)) {
        Verify(op->kind == ForKind::kParallel)
            << "ValueError: thread_binding requires a parallel loop at <current location>";
        Verify(tag->as<ffi::String>().has_value())
            << "TypeError: thread_binding annotation must be a string at <current location>";
        Verify(op->HasTrivialStep())
            << "ValueError: Thread binding loops require a unit step at <current location>";
      }
    }
    this->Visit(op->begin);
    this->Visit(op->end);
    this->Visit(op->step);
    if (check_ == Check::kLoopControl) {
      ++loop_depth_;
      this->Visit(op->body);
      --loop_depth_;
      return std::nullopt;
    }
    return WithScope([&]() -> Result {
      Define(op->var, AccessPath::Root()->Attr("var"));
      return WithScope([&]() { return this->Visit(op->body); });
    });
  }

  Result Visit_(const WhileObj* op) override {
    this->Visit(op->condition);
    ++loop_depth_;
    WithScope([&]() { return this->Visit(op->body); });
    --loop_depth_;
    return std::nullopt;
  }

  Result Visit_(const IfObj* op) override {
    this->Visit(op->condition);
    WithScope([&]() { return this->Visit(op->then_case); });
    return WithScope([&]() { return this->Visit(op->else_case); });
  }

  Result Visit_(const prim::LetObj* op) override {
    this->Visit(op->value);
    return WithScope([&]() -> Result {
      Define(op->var, AccessPath::Root()->Attr("var"));
      return this->Visit(op->body);
    });
  }

  Result Visit_(const LambdaExprObj* op) {
    this->Visit(op->ty);
    return WithScope([&]() -> Result {
      for (size_t i = 0; i < op->vars.size(); ++i) {
        Define(op->vars[i], AccessPath::Root()->Attr("vars")->ArrayItem(i));
      }
      return this->Visit(op->body);
    });
  }

  Result Visit_(const CallObj* op) override {
    if (op->op.as<GlobalVar>() || op->op.as<OpaqueExprObj>()) this->Visit(op->op);
    return this->Visit(op->args);
  }

  Result Visit_(const TensorStoreObj* op) override {
    this->Visit(op->value);
    TensorUse(op->dest.as_or_throw<TensorVar>());
    return this->Visit(op->indices);
  }

  Result Visit_(const TensorRegionObj* op) override {
    if (auto tensor = op->source.as<TensorVar>()) {
      TensorUse(tensor.value());
    } else {
      this->Visit(op->source);
    }
    return this->Visit(op->region);
  }

  Result Visit_(const GenericConstObj* op) override { return std::nullopt; }

  Result Visit_(const AssertStmtObj* op) override {
    this->Visit(op->condition);
    this->Visit(op->error_kind);
    return this->Visit(op->message_parts);
  }

  Result Visit_(const prim::RampObj* op) override {
    this->Visit(op->base);
    this->Visit(op->stride);
    return this->Visit(op->lanes);
  }

  Result Visit_(const prim::BroadcastObj* op) override {
    this->Visit(op->value);
    return this->Visit(op->lanes);
  }

  Result Visit_(const BreakObj* op) override {
    if (check_ == Check::kLoopControl) {
      Verify(loop_depth_ > 0) << "ValueError: break at <current location> requires an enclosing "
                                 "loop in the same function.";
    }
    return std::nullopt;
  }

  Result Visit_(const ContinueObj* op) override {
    if (check_ == Check::kLoopControl) {
      Verify(loop_depth_ > 0) << "ValueError: continue at <current location> requires an enclosing "
                                 "loop in the same function.";
    }
    return std::nullopt;
  }

  Result Visit_(const TensorLoadObj* op) override;

  Result Visit_(const SBlockObj* op) override {
    return WithScope([&]() -> Result {
      for (size_t i = 0; i < op->iter_vars.size(); ++i) {
        const auto& axis = op->iter_vars[i];
        this->Visit(axis->dom);
        Define(axis->var.as_or_throw<Var>(),
               AccessPath::Root()->Attr("iter_vars")->ArrayItem(i)->Attr("var"));
      }
      for (size_t i = 0; i < op->alloc_tensors.size(); ++i) {
        Define(op->alloc_tensors[i], AccessPath::Root()->Attr("alloc_tensors")->ArrayItem(i));
      }
      for (size_t i = 0; i < op->match_tensors.size(); ++i) {
        const auto& match = op->match_tensors[i];
        this->Visit(match->source);
        auto site = AccessPath::Root()->Attr("match_tensors")->ArrayItem(i)->Attr("tensor");
        MatchTensorDefinitions(match->tensor, site->Attr("ty"));
        Define(match->tensor, site);
      }
      this->Visit(op->reads);
      this->Visit(op->writes);
      WithScope([&]() { return this->Visit(op->init); });
      return WithScope([&]() { return this->Visit(op->body); });
    });
  }

  Check check_;

 private:
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }

  const VTable* table_;
  bool assert_mode_;
  bool has_error_{false};
  int loop_depth_{0};
  ffi::ObjectRef root_;
  std::vector<ffi::ObjectRef> visit_stack_;
  ScopeStack<std::vector<std::pair<Var, DefinitionSite>>> scopes_;
  std::unordered_map<Var, DefinitionSite> current_;
  std::unordered_map<Var, DefinitionSite> previous_;
};

auto WellFormednessVerifier::Visit_(const TensorLoadObj* op) -> Result {
  if (check_ != Check::kTensorLoadType) {
    TensorUse(op->source.as_or_throw<TensorVar>());
    return this->Visit(op->indices);
  }
  auto tensor = op->source.as<TensorVar>();
  auto valid_source = Verify(tensor.has_value());
  valid_source << "TypeError: TIR TensorLoad source at " << "<current location>.source"
               << " must be a TensorVar.";
  if (!tensor.has_value()) {
    this->Visit(op->indices);
    return std::nullopt;
  }

  bool valid_indices = tensor.value().shape().size() == op->indices.size();
  auto valid_rank = Verify(valid_indices);
  valid_rank << "ValueError: TIR TensorLoad at " << "<current location>" << " indexes "
             << tensor.value().shape().size() << "-dimensional tensor " << tensor.value()->name
             << " with " << op->indices.size() << " indices.";
  if (!valid_indices) {
    this->Visit(op->indices);
    return std::nullopt;
  }

  for (size_t i = 0; i + 1 < op->indices.size(); ++i) {
    bool is_scalar = op->indices[i].ty().IsScalar();
    auto valid_index = Verify(is_scalar);
    valid_index << "TypeError: TIR TensorLoad index " << i << " at "
                << "<current location>.indices[" << i << "]"
                << " must be scalar because only the final index may be vector-valued.";
    valid_indices = valid_indices && is_scalar;
  }
  if (!valid_indices) {
    this->Visit(op->indices);
    return std::nullopt;
  }

  ffi::Optional<PrimType> index_ty = op->indices.empty()
                                         ? ffi::Optional<PrimType>(tensor.value().dtype())
                                         : op->indices.back().ty().as<PrimType>();
  auto valid_index_type = Verify(index_ty.has_value());
  valid_index_type << "TypeError: TIR TensorLoad final index at <current location>.indices";
  if (!op->indices.empty()) valid_index_type << "[" << op->indices.size() - 1 << "]";
  valid_index_type << " must have a primitive type.";
  if (!index_ty.has_value()) {
    this->Visit(op->indices);
    return std::nullopt;
  }

  bool scalable_compatible = op->indices.empty() || !(tensor.value().dtype().IsScalableVector() &&
                                                      index_ty.value().IsScalableVector());
  auto valid_scalability = Verify(scalable_compatible);
  valid_scalability << "TypeError: TIR TensorLoad at " << "<current location>"
                    << " cannot combine a scalable tensor dtype with a scalable index.";
  if (!scalable_compatible) {
    this->Visit(op->indices);
    return std::nullopt;
  }

  TensorLoad expected = TensorLoad(tensor.value(), op->indices, op->loc);
  ffi::Optional<PrimType> asserted_ty = op->ty.as<PrimType>();
  ffi::Optional<PrimType> expected_ty = expected->ty.as<PrimType>();
  auto valid_type = Verify(asserted_ty.has_value() && expected_ty.has_value() &&
                           asserted_ty.value() == expected_ty.value());
  valid_type << "TypeError: TIR TensorLoad at " << "<current location>" << " asserts result type "
             << op->ty << ", but its source and indices imply " << expected->ty << ".";
  TensorUse(tensor.value());
  return this->Visit(op->indices);
}

bool CheckWellFormedCommon(const ffi::ObjectRef& node, bool assert_mode) {
  using Check = WellFormednessVerifier::Check;
  for (Check check :
       {Check::kVariables, Check::kTensors, Check::kTensorLoadType, Check::kLoopControl}) {
    if (!ffi::make_object<WellFormednessVerifier>(check, assert_mode)->Run(node)) return false;
  }
  return true;
}

bool VerifyThreadBindings(const ffi::ObjectRef& node, bool assert_mode) {
  return ffi::make_object<WellFormednessVerifier>(WellFormednessVerifier::Check::kThreadBinding,
                                                  assert_mode)
      ->Run(node);
}

}  // namespace

void CheckWellFormed(const tirx::Function& func) {
  VerifyThreadBindings(func, true);
  BlockVarAccessVerifier::Verify(func, true);
  CheckWellFormedCommon(func, true);
}

void CheckWellFormed(const IRModule& mod) {
  VerifyThreadBindings(mod, true);
  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto func = base_func.as<tirx::Function>()) {
      BlockVarAccessVerifier::Verify(func.value(), true);
    }
  }
  CheckWellFormedCommon(mod, true);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("s_tir.analysis.CheckWellFormed", [](const ffi::ObjectRef& obj) {
    if (auto func = obj.as<tirx::Function>()) {
      return s_tir::CheckWellFormed(func.value());
    }
    if (auto mod = obj.as<IRModule>()) {
      return s_tir::CheckWellFormed(mod.value());
    }
    TVM_FFI_THROW(TypeError) << "Expected a Function or IRModule, but received "
                             << obj->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  });
}
}  // namespace s_tir
}  // namespace tvm
