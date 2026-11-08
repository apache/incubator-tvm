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
# pylint: disable=invalid-name, unused-argument, redefined-argument-from-local
"""Dispatch sort and scan operators to platform dependent implementation."""

from functools import reduce
from operator import mul

import tvm_ffi

from tvm import relax, topi
from tvm.contrib.thrust import can_use_thrust
from tvm.relax import expr_functor
from tvm_tirx.ir import GlobalVar, Op
from tvm_tirx.ir.module import IRModule
from tvm_tirx.ir.transform import PassContext, module_pass
from tvm_tirx.target import Target

from .utils import BackendDispatcher


@expr_functor.mutator
class SortScanDispatcher(BackendDispatcher):
    """Dispatcher to dispatch sort and scan."""

    calls_to_update: dict[GlobalVar, Target]

    def __init__(self, mod, index_bits: int | None = None):
        super().__init__(mod)
        self.calls_to_update = {}
        self.index_bits = index_bits

    def apply_dlight_gpu_fallback(
        self,
    ) -> None:
        """Apply DLight rules for all the calls that need to be updated."""
        from tvm.s_tir import dlight  # pylint: disable=import-outside-toplevel

        for gvar, target in self.calls_to_update.items():
            func = self.builder_.get()[gvar]
            sch = dlight.base.transform._apply_rules(
                func,
                target,
                rules=[dlight.gpu.Fallback()],
                tunable=False,
            )
            if sch is not None:
                assert len(sch) == 1
                self.builder_.update_func(
                    gvar, sch[0].mod["main"].with_attr("tirx.is_scheduled", True)
                )

    def _append_calls_to_update(self, tir_call: relax.Call, target: Target) -> None:
        gvar = tir_call.args[0]
        assert isinstance(gvar, GlobalVar)
        existing_tgt = self.calls_to_update.get(gvar, None)
        if existing_tgt is not None and existing_tgt != target:
            raise ValueError(
                f"Multiple targets detected for function {gvar}. "
                f"Existing target: {existing_tgt}, new target: {target}"
            )
        self.calls_to_update[gvar] = target

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        if not isinstance(call.op, Op):
            return super().visit_call_(call)

        if call.op.name == "relax.bucketize":
            input_tensor = call.args[0]
            boundaries = call.args[1]
            right = call.attrs.right
            tgt = self._get_target(call.ty)
            te_func = topi.searchsorted
            with tgt:
                if self.is_gpu_target(tgt):
                    te_func = topi.gpu.searchsorted
            out_dtype = "int32" if call.attrs.out_int32 else "int64"
            return self.builder_.call_te(te_func, boundaries, input_tensor, right, out_dtype)
        if call.op.name == "relax.searchsorted":
            sorted_sequence = call.args[0]
            values = call.args[1]
            right = call.attrs.right
            tgt = self._get_target(call.ty)
            te_func = topi.searchsorted
            with tgt:
                if self.is_gpu_target(tgt):
                    te_func = topi.gpu.searchsorted
            out_dtype = "int32" if call.attrs.out_int32 else "int64"
            return self.builder_.call_te(te_func, sorted_sequence, values, right, out_dtype)
        if call.op.name == "relax.sort":
            tgt = self._get_target(call.ty)
            te_func = topi.sort
            kwargs = {}
            with tgt:
                if can_use_thrust(tgt, "tvm.contrib.thrust.sort"):
                    te_func = topi.gpu.sort_thrust
                    kwargs["workspace"] = self.allocate_workspace(call)
                elif self.is_gpu_target(tgt):
                    te_func = topi.gpu.sort
            return self.builder_.call_te(
                te_func, call.args[0], call.attrs.axis, not call.attrs.descending, **kwargs
            )
        if call.op.name == "relax.argsort":
            tgt = self._get_target(call.ty)
            te_func = topi.argsort
            kwargs = {}
            with tgt:
                if can_use_thrust(tgt, "tvm.contrib.thrust.sort"):
                    te_func = topi.gpu.argsort_thrust
                    kwargs["workspace"] = self.allocate_workspace(call)
                elif self.is_gpu_target(tgt):
                    te_func = topi.gpu.argsort
            return self.builder_.call_te(
                te_func,
                call.args[0],
                axis=call.attrs.axis,
                is_ascend=not call.attrs.descending,
                dtype=call.attrs.dtype,
                **kwargs,
            )
        if call.op.name == "relax.topk":
            tgt = self._get_target(call.ty)
            te_func = topi.topk
            kwargs = {}
            if can_use_thrust(tgt, "tvm.contrib.thrust.sort"):
                te_func = topi.gpu.topk_thrust
                kwargs["workspace"] = self.allocate_workspace(call)
            elif self.is_gpu_target(tgt):
                te_func = topi.gpu.topk
            tir_call = self.builder_.call_te(
                te_func,
                call.args[0],
                k=call.attrs.k,
                axis=call.attrs.axis,
                ret_type=call.attrs.ret_type,
                is_ascend=not call.attrs.largest,
                dtype=call.attrs.dtype,
                **kwargs,
            )
            self._append_calls_to_update(tir_call, tgt)
            return tir_call
        if call.op.name in ("relax.cumprod", "relax.cumsum"):
            tgt = self._get_target(call.ty)
            axis = int(call.attrs.axis) if call.attrs.axis is not None else call.attrs.axis
            shape = call.ty.shape
            # TODO(tvm-team): Support fully dynamic case with `shape=None`
            if shape is None:
                raise ValueError("non-symbolic shape is not supported for now")
            shape_values = [shape[i] for i in range(len(shape))]
            kwargs = {}
            normalized_axis = axis
            if normalized_axis is not None and normalized_axis < 0:
                normalized_axis += len(shape)
            if (
                normalized_axis is not None
                and (normalized_axis == len(shape) - 1 or tgt.kind.name == "webgpu")
                and self.is_gpu_target(tgt)
                and not can_use_thrust(tgt, "tvm.contrib.thrust.sum_scan")
                and call.op.name == "relax.cumsum"
                and call.attrs.exclusive == 0
            ):
                from tvm.relax.backend.gpu_generic import (  # pylint: disable=import-outside-toplevel
                    gpu_2d_continuous_cumsum,
                    gpu_3d_axis_1_cumsum,
                )

                input_tensor = call.args[0]
                in_dtype = call.args[0].ty.dtype
                out_dtype = call.attrs.dtype
                out_dtype = out_dtype or in_dtype

                if normalized_axis == len(shape) - 1:
                    outer = reduce(mul, shape_values[:-1], 1)
                    kernel_shape = relax.ShapeExpr([outer, shape[-1]])
                    index_bits = self.index_bits
                    if index_bits is None:
                        index_bits = 32 if tgt.kind.name == "webgpu" else 64
                    if tgt.kind.name == "webgpu" and index_bits != 32:
                        raise ValueError("WebGPU scan kernels require index_bits=32")
                    kernel = gpu_2d_continuous_cumsum(
                        in_dtype=in_dtype,
                        out_dtype=out_dtype,
                        index_bits=index_bits,
                    )
                    kernel_name = "gpu_2d_continuous_cumsum"
                else:
                    outer = reduce(mul, shape_values[:normalized_axis], 1)
                    inner = reduce(mul, shape_values[normalized_axis + 1 :], 1)
                    kernel_shape = relax.ShapeExpr([outer, shape[normalized_axis], inner])
                    kernel = gpu_3d_axis_1_cumsum(
                        in_dtype=in_dtype,
                        out_dtype=out_dtype,
                    )
                    kernel_name = "gpu_3d_axis_1_cumsum"

                reshape = relax.call_pure_packed(
                    "vm.builtin.reshape",
                    input_tensor,
                    kernel_shape,
                    ty_args=relax.TensorType(kernel_shape, in_dtype, vdevice=call.ty.vdevice),
                )
                gv = self.builder_.add_func(kernel, kernel_name)
                cumsum = relax.call_tir(
                    gv,
                    reshape,
                    ty_args=[relax.TensorType(kernel_shape, out_dtype, vdevice=call.ty.vdevice)],
                )
                return relax.call_pure_packed(
                    "vm.builtin.reshape",
                    cumsum,
                    shape,
                    ty_args=call.ty,
                )

            with tgt:
                if call.op.name == "relax.cumsum":
                    te_func = topi.gpu.cumsum if self.is_gpu_target(tgt) else topi.cumsum
                    if can_use_thrust(tgt, "tvm.contrib.thrust.sum_scan"):
                        kwargs["workspace"] = self.allocate_workspace(call)
                elif call.op.name == "relax.cumprod":
                    te_func = topi.gpu.cumprod if self.is_gpu_target(tgt) else topi.cumprod
                else:
                    raise ValueError(f"Unsupported op: {call.op.name}")
                tir_call = self.builder_.call_te(
                    te_func,
                    call.args[0],
                    axis,
                    call.attrs.dtype,
                    call.attrs.exclusive,
                    **kwargs,
                )
            self._append_calls_to_update(tir_call, tgt)
            return tir_call
        return super().visit_call_(call)

    def estimate_thrust_workspace_size(self, call: relax.Call) -> int:
        """
        Estimate the workspace size for thrust sort/argsort/topk/cumsum
        """
        input_shape = call.args[0].ty.shape
        input_byte_per_elem = tvm_ffi.dtype(call.args[0].ty.dtype.dtype).bits // 8
        int64_byte_per_elem = tvm_ffi.dtype("int64").bits // 8
        int32_byte_per_elem = tvm_ffi.dtype("int32").bits // 8
        num_elem = reduce(mul, input_shape, 1)
        input_size = num_elem * input_byte_per_elem
        # Most GPU algorithms take O(n) space or less, we choose 8N + 8MB as a safe estimation
        # for algorithm workspace.
        # The current thrust sort implementation may need extra int64 and int32 arrays
        # for temporary data, so we further add this part to the workspace.
        return (
            8 * input_size
            + 8 * 1024 * 1024
            + num_elem * (int64_byte_per_elem + int32_byte_per_elem)
        )

    def allocate_workspace(self, call: relax.Call) -> relax.Var:
        """
        Allocate workspace for thrust sort/argsort/topk.
        """
        workspace_size = self.estimate_thrust_workspace_size(call)
        alloc = relax.op.builtin.alloc_tensor(
            relax.ShapeExpr((workspace_size,)), "uint8", runtime_device_index=0
        )
        return self.builder_.emit(alloc)


@module_pass(opt_level=0, name="DispatchSortScan")
class DispatchSortScan:
    """
    Pass to dispatch scan and sort operators to platform dependent implementation.

    Parameters
    ----------
    index_bits : Optional[int]
        Signed index-width budget for the generated continuous GPU cumsum hierarchy.
        Must be 32 or 64. By default, use 32 for WebGPU and 64 for other targets.
        WebGPU does not support an explicit 64-bit budget.

        Pipelines that subsequently force indices to int32 should request 32 to
        avoid generating hierarchy thresholds outside the signed int32 range.
        The caller must ensure runtime indices fit the requested width; this
        option does not insert runtime bounds checks.
        This option does not narrow the generated TIR, change tensor dtypes, or
        affect other sort/scan implementations.
    """

    def __init__(self, index_bits: int | None = None):
        if index_bits not in (None, 32, 64):
            raise ValueError("index_bits must be either 32 or 64")
        self.index_bits = index_bits

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        sort_scan_dispater = SortScanDispatcher(mod, self.index_bits)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                func = sort_scan_dispater.visit_expr(func)
                sort_scan_dispater.builder_.update_func(gv, func)
        sort_scan_dispater.apply_dlight_gpu_fallback()
        return sort_scan_dispater.builder_.finalize()
