# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name
"""The S-TIR backend compilation pipeline."""

import tvm_ffi

import tvm_tirx
from tvm import s_tir
from tvm_tirx import tirx
from tvm_tirx.tirx import compilation_pipeline as tir_pipeline

tir = tirx  # alias for backward compat


def default_s_tir_pipeline(*, prepare_only=False, target=None):
    """Lower S-TIR, binding targets after thread launches become explicit."""
    return _s_tir_pipeline(prepare_only=prepare_only, target=target)


def _lower_s_tir(*, adreno=False):
    passes = []
    if adreno:
        passes.append(s_tir.backend.adreno.transform.TextureFlatten())
    passes.extend(
        [
            s_tir.transform.CanonicalizeLoop(),
            s_tir.transform.LowerCrossThreadReduction(),
            s_tir.transform.LowerInitBlock(),
            s_tir.transform.PlanAndUpdateTensorAllocationLocation(),
            s_tir.transform.ConvertBlocksToOpaque(),
            s_tir.transform.LiftThreadBinding(),
            s_tir.transform.ManifestSharedMemoryLocalStage(),
            s_tir.transform.CompactTensorAllocation(),
            s_tir.transform.LowerAutoCopy(),
            s_tir.transform.LowerMatchTensor(),
            s_tir.transform.StmtSimplify(),
            s_tir.transform.InjectPermutedLayout(),
            s_tir.transform.AnnotateIrregularLoop(),
            s_tir.transform.InjectSoftwarePipeline(),
            s_tir.transform.TransformMmaTensorLayout(),
            s_tir.transform.LoopPartition(),
            s_tir.transform.LowerOpaqueBlock(),
            s_tir.transform.LowerThreadBinding(),
        ]
    )
    return tvm_tirx.ir.transform.Sequential(passes)


def _lower_s_tir_tail(*, adreno=False):
    @tvm_tirx.ir.transform.module_pass(opt_level=0)
    def _pipeline(mod, _ctx):
        config = tvm_tirx.ir.transform.PassContext.current().config
        passes = []
        if adreno:
            passes.append(s_tir.backend.adreno.transform.InjectTextureAlloc())
        passes.extend(
            [
                tirx.transform.FlattenTensor(),
                tirx.transform.BF16ComputeLegalize(),
                tirx.transform.NarrowDataType(32),
                tirx.transform.VectorizeLoop(not bool(config.get("tirx.disable_vectorize", False))),
                s_tir.transform.InjectVirtualThread(),
            ]
        )
        if not bool(config.get("tirx.disable_storage_rewrite", False)):
            passes.append(tirx.transform.StorageRewrite())
        passes.extend(
            [
                tirx.transform.HoistIf(),
                tirx.transform.UnrollLoop(),
                tirx.transform.RenormalizeSplitPattern(),
                s_tir.transform.StmtSimplify(),
                tirx.transform.RemoveNoOp(),
                s_tir.transform.RewriteUnsafeSelect(),
            ]
        )
        # Additional passes based on configuration.
        if not bool(config.get("tirx.disable_cse_tir", False)):
            passes.append(tirx.transform.CommonSubexprElim())
        passes.extend(
            [
                # Bind the target first so that target-specific attributes are available.
                tirx.transform.FP8ComputeLegalize(),
                # VerifyVTCMLimit must occur before LowerVtcmAlloc.
                s_tir.transform.VerifyVTCMLimit(),
                s_tir.transform.LowerVtcmAlloc(),
                tirx.transform.VerifyMemory(),
            ]
        )
        passes.extend(
            [
                s_tir.transform.ThreadSync("shared"),
                s_tir.transform.ThreadSync("shared.dyn"),
                s_tir.transform.ThreadSync("warp"),
                s_tir.transform.InferFragment(),
                tirx.transform.LowerThreadAllreduce(),
            ]
        )
        if bool(config.get("tirx.use_async_copy", False)):
            passes.append(s_tir.transform.InjectPTXAsyncCopy())
        passes.append(s_tir.transform.LowerSynchronization())
        passes.append(s_tir.transform.MergeSharedMemoryAllocations())
        return tvm_tirx.ir.transform.Sequential(passes)(mod)

    return _pipeline


def _partition_dialects(mod):
    scheduled = {}
    native = {}
    for gv, func in mod.functions.items():
        group = (
            scheduled
            if isinstance(func, tirx.Function) and func.attrs.get("script.namespace") == "s_tir"
            else native
        )
        group[gv] = func
    return scheduled, native


def _submodule(mod, functions):
    return tvm_tirx.IRModule(functions, attrs=mod.attrs, global_infos=mod.global_infos)


def _s_tir_pipeline(*, prepare_only=False, target=None, adreno=False):
    @tvm_tirx.ir.transform.module_pass(opt_level=0)
    def _pipeline(mod, _ctx):
        scheduled, native = _partition_dialects(mod)
        prepared = _submodule(mod, {})
        if scheduled:
            prepared.update(_lower_s_tir(adreno=adreno)(_submodule(mod, scheduled)))
        if native:
            prepared.update(_submodule(mod, native))

        # Classification sees every caller after each dialect has explicit launches.
        prepared = tir_pipeline._bind_target(prepared, target)
        # Target binding and specialization can clone helpers. Select their tails
        # from the current function dialect, not from the original GlobalVars.
        scheduled, native = _partition_dialects(prepared)
        lowered = _submodule(prepared, {})
        if scheduled:
            lowered.update(_lower_s_tir_tail(adreno=adreno)(_submodule(prepared, scheduled)))
        if native:
            lower_native, _, _ = tir_pipeline.tirx_pipeline(prepare_only=True)
            lowered.update(lower_native(_submodule(prepared, native)))
        if prepare_only:
            return lowered
        if native:
            return tir_pipeline.finalize_tir_pipeline()(lowered)
        return tvm_tirx.ir.transform.Sequential(
            [
                tirx.transform.AnnotateEntryFunc(),
                tirx.transform.SplitHostDevice(),
                tirx.transform.MakePackedAPI(),
                tirx.transform.FP8StorageLegalize(),
                tirx.transform.BF16StorageLegalize(),
            ]
        )(lowered)

    return _pipeline, finalize_host_passes, finalize_device_passes


def _prepare_legacy_target_binding(mod):
    """Keep custom scheduling pipelines intact when prebinding is unambiguous."""
    scheduled, _ = _partition_dialects(mod)
    in_binding = False
    in_device_region = False

    def visit_loop(loop, visitor):
        nonlocal in_binding
        visitor.visit(loop.begin)
        visitor.visit(loop.end)
        visitor.visit(loop.step)
        visitor.visit(loop.annotations)
        previous = in_binding
        in_binding = previous or loop.annotations.get("thread_binding") is not None
        visitor.visit(loop.body)
        in_binding = previous

    def visit_region(region, visitor):
        nonlocal in_device_region
        visitor.visit(region.args)
        visitor.visit(region.attrs)
        previous = in_device_region
        in_device_region = previous or region.op.name in ("tirx.launch_thread", "tirx.device_entry")
        visitor.visit(region.body)
        in_device_region = previous

    def visit_call(call, visitor):
        if in_binding and not in_device_region and isinstance(call.op, tvm_tirx.ir.GlobalVar):
            raise ValueError(
                "A custom pipeline with calls inside S-TIR thread bindings must accept "
                "an explicit target keyword and bind it after LowerThreadBinding. "
                'Delegate to get_tir_pipeline("s_tir", target=target) or lower the '
                "complete mixed module before BindTarget."
            )
        return visitor.default_visit(call)

    for func in scheduled.values():
        tvm_ffi.structural_visit(
            func.body,
            [
                (tvm_tirx.ir.For, visit_loop),
                (tvm_tirx.ir.RegionStmt, visit_region),
                (tvm_tirx.ir.Call, visit_call),
            ],
        )
    return mod


def finalize_host_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    host_pass_list = [
        tirx.transform.LowerTVMBuiltin(),
        tirx.transform.LowerIntrin(),
    ]
    return tvm_tirx.ir.transform.Sequential(host_pass_list)


def finalize_device_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    device_pass_list = [
        tirx.transform.StmtSimplify(),
        tirx.transform.LowerIntrin(),
    ]
    return tvm_tirx.ir.transform.Sequential(device_pass_list)


tir_pipeline.PIPELINE_MAP["s_tir"] = default_s_tir_pipeline


def _select_default_pipeline(mod, target, *, target_to_bind=None):
    """Select S-TIR lowering only for functions constructed in this dialect."""
    scheduled, _ = _partition_dialects(mod)
    if not scheduled:
        return None
    name = "s_tir"
    if target is not None and target.kind.name == "opencl" and "adreno" in target.keys:
        name = "adreno"
    return tir_pipeline.get_tir_pipeline(name, target=target_to_bind)


tir_pipeline.register_default_tir_pipeline_selector(_select_default_pipeline)
tir_pipeline._LEGACY_TARGET_PREPARERS.append(_prepare_legacy_target_binding)
