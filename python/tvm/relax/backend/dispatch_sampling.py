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
"""Dispatch sampling operators to platform dependent implementation."""

from tvm import relax
from tvm.relax import expr_functor
from tvm_tirx.ir import Op
from tvm_tirx.ir.module import IRModule
from tvm_tirx.ir.transform import PassContext, module_pass

from .utils import BackendDispatcher


@expr_functor.mutator
class SamplingDispatcher(BackendDispatcher):
    """Dispatcher to dispatch sampling op."""

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        if not isinstance(call.op, Op):
            return super().visit_call_(call)

        if call.op.name == "relax.multinomial_from_uniform":
            from tvm.relax.backend.gpu_generic import (  # pylint: disable=import-outside-toplevel
                generic_get_sample_index,
                gpu_multinomial_from_uniform,
            )

            prob, uniform_sample, sample_indices = call.args
            tgt = self._get_target(call.ty)
            dtype = call.attrs.dtype
            _, prob_dtype = self.get_shape_dtype(prob)
            sample_shape, sample_dtype = self.get_shape_dtype(uniform_sample)
            sample_indices_shape, sample_indices_dtype = self.get_shape_dtype(sample_indices)

            if len(sample_shape) != 2 or sample_shape[1] != 1:
                raise ValueError("uniform_sample should be a 2D tensor with shape (N, 1)")

            if len(sample_indices_shape) != 2 or sample_indices_shape[1] != 1:
                raise ValueError("sample_indices should be a 2D tensor with shape (N, 1)")

            if self.is_gpu_target(tgt):
                gv = self.builder_.add_func(
                    gpu_multinomial_from_uniform(
                        prob_dtype, sample_dtype, sample_indices_dtype, dtype, target=tgt
                    ),
                    "gpu_multinomial_from_uniform",
                )
                return relax.call_tir(
                    gv,
                    [prob, uniform_sample, sample_indices],
                    ty_args=[call.ty],
                )
            else:
                cumsum_prob = relax.op.cumsum(prob, axis=1, dtype=prob_dtype.dtype, exclusive=False)
                gv = self.builder_.add_func(
                    generic_get_sample_index(prob_dtype, sample_dtype, sample_indices_dtype, dtype),
                    "get_sample_index",
                )
                return relax.call_tir(
                    gv,
                    [cumsum_prob, uniform_sample, sample_indices],
                    ty_args=[call.ty],
                )

        return super().visit_call_(call)


@module_pass(opt_level=0, name="DispatchSampling")
class DispatchSampling:
    """Pass to dispatch scan and sort operators to platform dependent implementation."""

    def __init__(self, target=None):
        self.target = target

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        sampling_dispatcher = SamplingDispatcher(mod, self.target)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                func = sampling_dispatcher.visit_expr(func)
                sampling_dispatcher.builder_.update_func(gv, func)
        return sampling_dispatcher.builder_.finalize()
