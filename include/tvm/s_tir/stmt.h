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
 * \file tvm/s_tir/stmt.h
 * \brief S-TIR (Schedulable TIR) statements and attributes.
 *
 * This file contains attribute keys that are specific to the schedulable TIR
 * (S-TIR) layer, including meta_schedule annotations and schedule primitive /
 * SBlock annotations.
 */
#ifndef TVM_S_TIR_STMT_H_
#define TVM_S_TIR_STMT_H_

#include <tvm/base.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/s_tir/iter_var.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace s_tir {

/*! \brief Placement entries owned by a block, traversable with their tensor references. */
using TensorAllocatedAddresses = ffi::Array<ffi::Tuple<Var, ffi::Array<PrimExpr>>>;

/*!
 * \brief Match introduces a constraint that the source tensor region can be remapped to the data
 * layout specified by the tensor field. The constraint can be checked in later part of lowering (or
 * optionally during runtime).
 *
 * MatchTensorRegion provides a mechanism to represent data layout and compactness constraints in
 * low-level hardware primitives in the IR and defer the check after the sequence of
 * transformations.
 */
class MatchTensorRegionObj : public ffi::Object {
 public:
  explicit MatchTensorRegionObj(ffi::UnsafeInit tag) : tensor(tag), source(tag) {}

  MatchTensorRegionObj(tirx::TensorVar tensor, TensorRegion source)
      : tensor(std::move(tensor)), source(std::move(source)) {}

  /*! \brief The target tensor. */
  tirx::TensorVar tensor;
  /*! \brief The source tensor region. */
  TensorRegion source;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MatchTensorRegionObj>()
        .def_ro("tensor", &MatchTensorRegionObj::tensor, refl::AttachFieldFlag::SEqHashDefPattern())
        .def_ro("source", &MatchTensorRegionObj::source);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.MatchTensorRegion", MatchTensorRegionObj, ffi::Object);
};

/*!
 * \brief Managed reference to MatchTensorRegionObj.
 * \sa MatchTensorRegionObj
 */
class MatchTensorRegion : public ffi::ObjectRef {
 public:
  explicit MatchTensorRegion(ffi::UnsafeInit tag, ffi::ObjectPtr<MatchTensorRegionObj> data)
      : ffi::ObjectRef(tag) {
    data_ = std::move(data);
  }

  TVM_DLL explicit MatchTensorRegion(tirx::TensorVar tensor, TensorRegion source);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MatchTensorRegion, ffi::ObjectRef,
                                                MatchTensorRegionObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MatchTensorRegionObj);
};

/*!
 * \brief A block is a basic schedule unit in TIR.
 * \note SBlock's body is parameterized by iter vars.
 * \code
 *
 *  with T.sblock(name):
 *      v0 = T.axis.S(domain, value0)
 *      v1 = T.axis.R(domain, value1)
 *      ...
 *      T.reads([tensor0[start:end, ...], ...])
 *      T.writes([tensor1[start:end, ...], ...])
 *      T.where(predicate)
 *      tensor2 = T.alloc_tensor(ty_args=[T.Tensor(shape, dtype)])
 *      tensor3 = Ts.match_tensor(source_tensor[start:end, ...])
 *      Ts.sblock_attr({attr_key: attr_value, ...})
 *      with T.init():
 *          // init body
 *      // body
 *
 * \endcode
 */
class SBlockObj : public StmtObj {
 public:
  explicit SBlockObj(ffi::UnsafeInit tag) : body(tag) {}

  explicit SBlockObj(SeqStmt body) : body(std::move(body)) {}

  /*! \brief The variables of the block. */
  ffi::Array<s_tir::IterVar> iter_vars;
  /*! \brief The read tensor regions of the block. */
  ffi::Array<TensorRegion> reads;
  /*! \brief The write tensor regions of the block. */
  ffi::Array<TensorRegion> writes;
  /*! \brief The name_hint of the block. */
  ffi::String name_hint;
  /*! \brief The tensor allocated in the block. */
  ffi::Array<tirx::TensorVar> alloc_tensors;
  /*! \brief The match tensor regions. */
  ffi::Array<MatchTensorRegion> match_tensors;
  /*! \brief The annotation of the block. */
  ffi::Map<ffi::String, ffi::Any> annotations;
  /*!
   * \brief The init statement is executed during the first iteration of reduction loops in a
   *  reduction block. The optional init field allows us to represent initialization and
   *  reduction update in a single block and transform them collectively.
   *  We also provide primitives to decompose the init into a separate block during scheduling.
   *  Init field is `std::nullopt` if there is no reduction iter_vars
   */
  ffi::Optional<SeqStmt> init;
  /*! \brief The body of the block. */
  SeqStmt body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SBlockObj>()
        .def_ro("iter_vars", &SBlockObj::iter_vars)
        .def_ro("reads", &SBlockObj::reads)
        .def_ro("writes", &SBlockObj::writes)
        .def_ro("name_hint", &SBlockObj::name_hint, refl::AttachFieldFlag::SEqHashIgnore())
        .def_ro("alloc_tensors", &SBlockObj::alloc_tensors,
                refl::AttachFieldFlag::SEqHashDefSimple())
        .def_ro("match_tensors", &SBlockObj::match_tensors)
        .def_ro("annotations", &SBlockObj::annotations)
        .def_ro("init", &SBlockObj::init)
        .def_ro("body", &SBlockObj::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.SBlock", SBlockObj, StmtObj);
};

/*!
 * \brief Managed reference to SBlockObj.
 * \sa SBlockObj
 */
class SBlock : public Stmt {
 public:
  TVM_DLL explicit SBlock(
      ffi::Array<s_tir::IterVar> iter_vars, ffi::Array<TensorRegion> reads,
      ffi::Array<TensorRegion> writes, ffi::String name_hint, SeqStmt body,
      ffi::Optional<SeqStmt> init = std::nullopt,
      ffi::Array<tirx::TensorVar> alloc_tensors = ffi::Array<tirx::TensorVar>(),
      ffi::Array<MatchTensorRegion> match_tensors = ffi::Array<MatchTensorRegion>(),
      ffi::Map<ffi::String, ffi::Any> annotations = ffi::Map<ffi::String, ffi::Any>(),
      Location loc = UnknownLoc());

  TVM_DLL explicit SBlock(ffi::String name_hint, SeqStmt body,
                          ffi::Array<tirx::TensorVar> alloc_tensors = ffi::Array<tirx::TensorVar>(),
                          Location loc = UnknownLoc());

  explicit SBlock(ffi::ObjectPtr<SBlockObj> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SBlock, Stmt, SBlockObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SBlockObj);
};

/*!
 * \brief A block realization node represents execution of the block at the binding values.
 */
class SBlockRealizeObj : public StmtObj {
 public:
  explicit SBlockRealizeObj(ffi::UnsafeInit tag) : predicate(tag), block(tag) {}

  SBlockRealizeObj(PrimExpr predicate, SBlock block)
      : predicate(std::move(predicate)), block(std::move(block)) {}

  /*! \brief The corresponding values of the iter vars. */
  ffi::Array<PrimExpr> iter_values;
  /*!
   * \brief The predicate of the block realization, the block will only be executed when the
   * predicate is true.
   */
  PrimExpr predicate;
  /*! \brief The block to be realized. */
  SBlock block;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SBlockRealizeObj>()
        .def_ro("iter_values", &SBlockRealizeObj::iter_values)
        .def_ro("predicate", &SBlockRealizeObj::predicate)
        .def_ro("block", &SBlockRealizeObj::block);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.SBlockRealize", SBlockRealizeObj, StmtObj);
};

/*!
 * \brief Managed reference to BlockRealizeNode
 * \sa BlockRealizeNode
 */
class SBlockRealize : public Stmt {
 public:
  TVM_DLL explicit SBlockRealize(ffi::Array<PrimExpr> iter_values, PrimExpr predicate, SBlock block,
                                 Location loc = UnknownLoc());

  explicit SBlockRealize(ffi::ObjectPtr<SBlockRealizeObj> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SBlockRealize, Stmt, SBlockRealizeObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SBlockRealizeObj);
};

/*! \brief Region marking copies eligible for asynchronous lowering. */
TVM_DLL const Op& async_copy_scope();
/*! \brief Commit the preceding asynchronous copies to a queue. */
TVM_DLL const Op& async_commit();
/*! \brief Wait until at most the given number of committed groups remain in flight. */
TVM_DLL const Op& async_wait();
/*! \brief Region whose author manages cross-thread synchronization explicitly. */
TVM_DLL const Op& manual_sync();

namespace attr {
/*! \brief Hardware thread axis bound to a parallel loop during S-TIR scheduling. */
inline constexpr ffi::reflection::AttrKey<ffi::String> kThreadBinding{"thread_binding"};

/*!
 * \brief SBlock annotation selecting a write-tensor index for double buffering in
 * InjectSoftwarePipeline.
 */
inline constexpr ffi::reflection::AttrKey<int64_t> kDoubleBufferScope{"double_buffer_scope"};

/*!
 * \brief String-valued allocation Call attribute containing the TensorCore fragment shape
 */
inline constexpr ffi::reflection::AttrKey<ffi::String> kFragmentShape{"fragment_shape"};

/*!
 * \brief String-valued allocation Call attribute containing the TensorCore fragment layout
 */
inline constexpr ffi::reflection::AttrKey<ffi::String> kFragmentLayout{"fragment_layout"};

/*!
 * \brief For annotation requesting partitioning when its PrimExpr value is provably true.
 */
inline constexpr ffi::reflection::AttrKey<PrimExpr> kLoopPartitionHint{"loop_partition_hint"};

// -----------------------------------------------------------------------
// meta_schedule annotations
// -----------------------------------------------------------------------

/*! \brief Mark the tiling structure of blocks that are applied by rule Multi-Level-Tiling */
inline constexpr ffi::reflection::AttrKey<ffi::String> kMetaScheduleTilingStructure{
    "meta_schedule.tiling_structure"};

/*!
 * \brief Mark that the loop should be further skip and bound to environment threads to enable
 * cooperative fetching.
 */
inline constexpr ffi::reflection::AttrKey<int64_t> kMetaScheduleCooperativeFetch{
    "meta_schedule.cooperative_fetch"};

/*! \brief The allowed range of thread extent in thread bindings */
inline constexpr ffi::reflection::AttrKey<int64_t> kMetaScheduleThreadExtentLowInclusive{
    "meta_schedule.thread_extent_low_inclusive"};

/*! \brief The allowed range of thread extent in thread bindings */
inline constexpr ffi::reflection::AttrKey<int64_t> kMetaScheduleThreadExtentHighInclusive{
    "meta_schedule.thread_extent_high_inclusive"};

/*! \brief Mark the block whose producer needs to be applied by rule Random-Compute-Location */
inline constexpr ffi::reflection::AttrKey<int64_t> kMetaScheduleRandomComputeProducer{
    "meta_schedule.random_compute_producer"};

/*! \brief Mark auto-parallel setting on the block. */
inline constexpr ffi::reflection::AttrKey<int64_t> kMetaScheduleParallel{"meta_schedule.parallel"};

/*! \brief Mark auto-vectorize setting on the block. */
inline constexpr ffi::reflection::AttrKey<int64_t> kMetaScheduleVectorize{
    "meta_schedule.vectorize"};

/*! \brief Mark auto-unroll setting on the block. */
inline constexpr ffi::reflection::AttrKey<int64_t> kMetaScheduleUnrollExplicit{
    "meta_schedule.unroll_explicit"};

/*! \brief Mark auto-unroll setting on the block. */
inline constexpr ffi::reflection::AttrKey<int64_t> kMetaScheduleUnrollImplicit{
    "meta_schedule.unroll_implicit"};

/*! \brief Mark that a block should be further rewritten using tensorization. */
inline constexpr ffi::reflection::AttrKey<ffi::String> kMetaScheduleAutoTensorize{
    "meta_schedule.auto_tensorize"};

/*! \brief Mark that a block is a preprocessor block for layout rewrite. */
inline constexpr ffi::reflection::AttrKey<bool> kMetaScheduleLayoutRewritePreproc{
    "meta_schedule.layout_rewrite_preproc"};

/*!
 * \brief Mark that the init statement of a block should be further rewritten using tensorization.
 */
inline constexpr ffi::reflection::AttrKey<ffi::String> kMetaScheduleAutoTensorizeInit{
    "meta_schedule.auto_tensorize_init"};

/*! \brief Mark that a block is disallowed in auto inline. */
inline constexpr ffi::reflection::AttrKey<ffi::String> kMetaScheduleInlineRule{
    "meta_schedule.inline_rule"};

// -----------------------------------------------------------------------
// Schedule primitive / SBlock annotations
// -----------------------------------------------------------------------

/*! \brief TensorAllocatedAddresses for block-owned allocations and match tensors. */
inline constexpr ffi::reflection::AttrKey<TensorAllocatedAddresses> kTensorAllocatedAddr{
    "s_tir.tensor_allocated_addr"};

/*!
 * \brief Mark whether the script-completer need to fill in missing access region
 *        during script parsing.
 * \note The result should be a integer mask with range [0, 4).
 *       if (mask & 1) the read region should be detected,
 *       if (mask & 2) the write region should be detected.
 */
inline constexpr ffi::reflection::AttrKey<int64_t> kScriptParsingDetectAccess{
    "tirx.script_parsing_detect_access"};

/*!
 * \brief Mark that the block need to add predicate for block var bounds during lowering
 */
constexpr const char* kRequireBlockVarBoundPredicate = "require_bound_predicate";

/*! \brief Mark the stage of a statement in the software pipeline */
inline constexpr ffi::reflection::AttrKey<ffi::Array<int64_t>> kSoftwarePipelineStage{
    "software_pipeline_stage"};

/*! \brief Mark the order of a statement in the software pipeline */
inline constexpr ffi::reflection::AttrKey<ffi::Array<int64_t>> kSoftwarePipelineOrder{
    "software_pipeline_order"};

/*! \brief List stages in the software pipeline that should run asynchronously
 * \note All statements in the provided stages are assumed to have asynchronous
 *       semantics (e.g. CUDA async global to shared memory copy).
 */
inline constexpr ffi::reflection::AttrKey<ffi::Array<int64_t>> kSoftwarePipelineAsyncStages{
    "software_pipeline_async_stages"};

/*! \brief Mark the local stage for the shared memory access should be added. */
inline constexpr ffi::reflection::AttrKey<int64_t> kManifestSharedMemoryLocalStage{
    "tirx.manifest_shared_memory_local_stage"};

/*!
 * \brief Mark alignment of tensor dimension
 *  The annotation value is an array of explicit tuples
 *  (tensor_index, axis, factor, offset).
 *  This requires the stride of an axis to be k * factor + offset.
 */
inline constexpr ffi::reflection::AttrKey<
    ffi::Array<ffi::Tuple<int32_t, int32_t, int32_t, int32_t>>>
    kTensorDimAlign{"tensor_dim_align"};

/*! \brief Mark that a block has an explicitly specified read region.
 * This is used to override the default read region inference in TIR.
 */
inline constexpr ffi::reflection::AttrKey<ffi::Array<int64_t>> kExplicitReadRegion{
    "explicit_read_region"};

/*! \brief Mark that a block has an explicitly specified write region.
 * This is used to override the default write region inference in TIR.
 */
inline constexpr ffi::reflection::AttrKey<ffi::Array<int64_t>> kExplicitWriteRegion{
    "explicit_write_region"};

/*! \brief ,ark a ForObj represent an irregular loop of non-structural control flow edges. */
inline constexpr ffi::reflection::AttrKey<int64_t> kIrregularLoopMark{"irregular_loop_mark"};

/*! \brief Mark auto copy for memhammer */
inline constexpr ffi::reflection::AttrKey<int64_t> kAutoCopy{"auto_copy"};

/*! \brief Mark local stage constraint on data copy */
inline constexpr ffi::reflection::AttrKey<int64_t> kLocalStage{"local_stage"};

/*! \brief Mark vectorization length constraint on block */
inline constexpr ffi::reflection::AttrKey<int64_t> kVectorBytes{"vector_bytes"};

/*!
 * \brief Mark that a block is executed by a warp. This implies the extend of threadIdx.x is
 * warp size.
 */
inline constexpr ffi::reflection::AttrKey<int64_t> kWarpExecution{"warp_execution"};

constexpr const char* kPermutedLayout = "permuted_layout";

inline constexpr ffi::reflection::AttrKey<ffi::String> kScheduleRule{"schedule_rule"};

inline constexpr ffi::reflection::AttrKey<ffi::Array<int64_t>> kMetaScheduleWriteCacheLevel{
    "s_tir.meta_schedule.write_cache_level"};

}  // namespace attr

/*! \brief Interpret and validate S-TIR thread placement on a shared loop. */
inline ffi::Optional<ffi::String> GetThreadBinding(const ForObj* loop) {
  if (auto tag = loop->annotations.Get(tvm::s_tir::attr::kThreadBinding.name)) {
    TVM_FFI_CHECK(loop->kind == ForKind::kParallel, ValueError)
        << "thread_binding requires a parallel loop";
    TVM_FFI_CHECK(tag->as<ffi::String>().has_value(), TypeError)
        << "thread_binding annotation must be a string";
    TVM_FFI_CHECK(loop->HasTrivialStep(), ValueError) << "Thread binding loops require a unit step";
    return tag->cast<ffi::String>();
  }
  return std::nullopt;
}
inline ffi::Optional<ffi::String> GetThreadBinding(const For& loop) {
  return GetThreadBinding(loop.get());
}

}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_STMT_H_
