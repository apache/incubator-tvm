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
# ruff: noqa: E501, F401

import os
import tempfile

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax, s_tir
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.relax.dpl import is_op, wildcard
from tvm.relax.testing import transform
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.support import utils
from tvm_tirx.testing import env

env_checker_codegen = tvm_ffi.get_global_func("relax.ext.tensorrt", True)
env_checker_runtime = tvm_ffi.get_global_func("relax.is_tensorrt_runtime_enabled", True)

requires_tensorrt_codegen = pytest.mark.skipif(
    not env_checker_codegen,
    reason="TensorRT codegen not available",
)
requires_tensorrt_runtime = pytest.mark.skipif(
    not env_checker_runtime or not env_checker_runtime(),
    reason="TensorRT runtime not available",
)

# Global variable in pytest that applies markers to all tests.
pytestmark = [
    requires_tensorrt_codegen,
    pytest.mark.gpu,
    pytest.mark.skipif(not env.has_cuda(), reason="need cuda"),
]

# Target gpu
target_str = "nvidia/nvidia-t4"
target = tvm_tirx.target.Target(target_str)


def check_executable(exec, dev, inputs, expected, entry_func_name):
    vm = relax.VirtualMachine(exec, dev)
    out = vm[entry_func_name](*inputs)
    tvm_tirx.testing.assert_allclose(out.numpy(), expected, atol=1e-5, rtol=1e-5)


def check_roundtrip(exec0, reference_exec, input_arrays, entry_func_name="main"):
    with utils.tempdir() as temp:
        exec0.export_library(temp.relpath("exec.so"))
        exec1 = tvm_tirx.runtime.load_module(temp.relpath("exec.so"))
    assert exec0.stats() == exec1["stats"]()
    assert exec0.as_text() == exec1["as_text"]()

    def run_and_check():
        dev = tvm_tirx.cuda()
        inputs = [tvm_tirx.runtime.tensor(array, dev) for array in input_arrays]
        reference_vm = relax.VirtualMachine(reference_exec, dev)
        expected = reference_vm["main"](*inputs).numpy()
        check_executable(exec0, dev, inputs, expected, entry_func_name)
        check_executable(exec1, dev, inputs, expected, entry_func_name)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def gen_ground_truth(mod, target):
    # Lower and run tuning
    # Since there is no default schedule for GPU in MS yet, this is necessary
    seq = tvm_tirx.ir.transform.Sequential(
        [
            relax.transform.LegalizeOps(target=target),
            s_tir.transform.DefaultGPUSchedule(target=target),
        ]
    )
    new_mod = seq(mod)
    relax.analysis.check_well_formed(new_mod)
    return tvm.compile(new_mod, target, params={})


@tvm_tirx.script.ir_module
class InputModule:
    @R.function
    def main(x: R.Tensor((16, 16), "float32"), y: R.Tensor((16, 16), "float32")) -> R.Tensor(
        (16, 16), "float32"
    ):
        with R.dataflow():
            z1 = R.multiply(x, y)
            z2 = R.add(z1, x)
            z3 = R.add(z1, z2)
            z4 = R.multiply(z3, z2)
            z5 = R.add(z4, z1)
            R.output(z5)
        return z5


def setup_test():
    # Prepare IRModule and its input
    mod = InputModule
    assert isinstance(mod, tvm_tirx.IRModule)

    np0 = np.random.rand(16, 16).astype(np.float32)
    np1 = np.random.rand(16, 16).astype(np.float32)
    inputs = [np0, np1]

    # Ground truth should be generated before annotation
    # due to the conflict with MS task extraction
    # TODO(@sunggg): Sort this out
    reference_exec = gen_ground_truth(mod, target)
    return mod, inputs, reference_exec


@pytest.fixture(params=["main", "func"], scope="session")
def entry_func_name(request):
    return request.param


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
@requires_tensorrt_runtime
def test_tensorrt_only(entry_func_name):
    mod, inputs, reference_exec = setup_test()

    if entry_func_name != "main":
        mod[entry_func_name] = mod
        del mod["main"]

    # Define patterns that we want to offload to byoc
    # This test will offload entire model
    # Thus, define patterns for both `multiply` and `add` ops
    patterns = [
        ("tensorrt.multiply", is_op("relax.multiply")(wildcard(), wildcard())),
        ("tensorrt.add", is_op("relax.add")(wildcard(), wildcard())),
    ]

    new_mod = tvm_tirx.ir.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern(patterns),
            relax.transform.MergeCompositeFunctions(),
            relax.transform.RunCodegen(),
        ]
    )(mod)

    ex0 = tvm.compile(new_mod, target, params={})
    # Sanity check for the correctness and roundtrip
    check_roundtrip(ex0, reference_exec, inputs, entry_func_name)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
@requires_tensorrt_runtime
def test_mix_use_tensorrt_and_tvm():
    mod, inputs, reference_exec = setup_test()

    # Define patterns that we want to offload to byoc
    # This test will only offload `add` op to tensorrt
    # and tune `multiply` op with MetaSchedule
    patterns = [
        ("tensorrt.add", is_op("relax.add")(wildcard(), wildcard())),
    ]

    # Run Codegen pass
    with tempfile.TemporaryDirectory() as work_dir:
        with tvm_tirx.ir.transform.PassContext(opt_level=0):
            new_mod = tvm_tirx.ir.transform.Sequential(
                [
                    relax.transform.FuseOpsByPattern(patterns),
                    relax.transform.MergeCompositeFunctions(),
                    relax.transform.RunCodegen(),
                    relax.transform.LegalizeOps(target=target),
                    relax.transform.MetaScheduleTuneIRMod(
                        params={}, work_dir=work_dir, max_trials_global=8, target=target
                    ),
                    relax.transform.MetaScheduleApplyDatabase(work_dir, target=target),
                ]
            )(mod)
    relax.analysis.check_well_formed(new_mod)
    with transform.PassContext(opt_level=0):
        ex0 = tvm.compile(new_mod, target, params={})

    # Sanity check for the correctness and roundtrip
    check_roundtrip(ex0, reference_exec, inputs)


@tvm_tirx.script.ir_module
class Conv2dx2:
    @R.function
    def main(
        data: R.Tensor((16, 32, 32, 16), dtype="float16"),
        weight1: R.Tensor((16, 3, 3, 16), dtype="float16"),
        weight2: R.Tensor((16, 3, 3, 16), dtype="float16"),
    ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
        cls = Conv2dx2
        with R.dataflow():
            lv: R.Tensor((16, 32, 32, 16), dtype="float16") = cls.fused_relax_nn_conv2d_tensorrt(
                data, weight1
            )
            gv: R.Tensor((16, 32, 32, 16), dtype="float16") = cls.fused_relax_nn_conv2d_tensorrt(
                lv, weight2
            )
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_conv2d_tensorrt(
        data: R.Tensor((16, 32, 32, 16), dtype="float16"),
        weight1: R.Tensor((16, 3, 3, 16), dtype="float16"),
    ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
        R.func_attr({"Codegen": "tensorrt", "global_symbol": "fused_relax_nn_conv2d_tensorrt"})

        @R.function
        def gv(
            data_1: R.Tensor((16, 32, 32, 16), dtype="float16"),
            weight1_1: R.Tensor((16, 3, 3, 16), dtype="float16"),
        ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
            R.func_attr({"Composite": "tensorrt.conv2d", "Primitive": True})
            with R.dataflow():
                gv_1: R.Tensor((16, 32, 32, 16), dtype="float16") = R.nn.conv2d(
                    data_1,
                    weight1_1,
                    padding=[1, 1, 1, 1],
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                )
                R.output(gv_1)
            return gv_1

        gv1: R.Tensor((16, 32, 32, 16), dtype="float16") = gv(data, weight1)
        return gv1


@tvm_tirx.script.ir_module
class Conv2dx2_after:
    @R.function
    def main(
        data: R.Tensor((16, 32, 32, 16), dtype="float16"),
        weight1: R.Tensor((16, 3, 3, 16), dtype="float16"),
        weight2: R.Tensor((16, 3, 3, 16), dtype="float16"),
    ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
        with R.dataflow():
            lv = R.call_dps_packed(
                _ExternFunc("fused_relax_nn_conv2d_tensorrt"),
                (data, weight1),
                ty_args=[R.Tensor((16, 32, 32, 16), dtype="float16")],
            )
            gv = R.call_dps_packed(
                _ExternFunc("fused_relax_nn_conv2d_tensorrt"),
                (lv, weight2),
                ty_args=[R.Tensor((16, 32, 32, 16), dtype="float16")],
            )
            R.output(gv)
        return gv


def test_multiple_calls_same_extern():
    mod = relax.transform.RunCodegen()(Conv2dx2)
    tvm_tirx.ir.assert_structural_equal(mod["main"], Conv2dx2_after["main"])


def test_default_entry_func():
    """The entry function is not necessarily named "main"

    Like `test_multiple_calls_same_extern`, but the main function is
    named "func".
    """
    before_with_main = Conv2dx2
    after_with_main = relax.transform.RunCodegen()(before_with_main)

    def rename_main(mod):
        mod = mod.clone()
        mod["func"] = mod["main"].with_attr("global_symbol", "func")
        del mod["main"]
        return mod

    before_with_func = rename_main(before_with_main)
    expected_with_func = rename_main(after_with_main)
    after_with_func = relax.transform.RunCodegen()(before_with_func)

    tvm_tirx.ir.assert_structural_equal(expected_with_func["func"], after_with_func["func"])


def test_dynamic_shape():
    import tvm.relax.backend.cuda.cublas

    r1_main = T.dynamic("r1", "int64")
    r2 = T.dynamic("r2", "int64")
    r1_fused_relax_matmul_cublas = T.dynamic("r1", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 4096), dtype="float16"),
            w1: R.Tensor((4096, r1_main), dtype="float16"),
            w2: R.Tensor((4096, r2), dtype="float16"),
        ) -> R.Tuple(R.Tensor((1, r1_main), dtype="float16"), R.Tensor((1, r2), dtype="float16")):
            cls = Before
            with R.dataflow():
                lv: R.Tensor((1, r1_main), dtype="float16") = cls.fused_relax_matmul_cublas(x, w1)
                lv1: R.Tensor((1, r2), dtype="float16") = cls.fused_relax_matmul_cublas(x, w2)
                gv: R.Tuple(
                    R.Tensor((1, r1_main), dtype="float16"), R.Tensor((1, r2), dtype="float16")
                ) = (lv, lv1)
                R.output(gv)
            return gv

        @R.function
        def fused_relax_matmul_cublas(
            x: R.Tensor((1, 4096), dtype="float16"),
            w1: R.Tensor((4096, r1_fused_relax_matmul_cublas), dtype="float16"),
        ) -> R.Tensor((1, r1_fused_relax_matmul_cublas), dtype="float16"):
            R.func_attr({"Codegen": "cublas"})

            @R.function
            def gv(
                x_1: R.Tensor((1, 4096), dtype="float16"),
                w1_1: R.Tensor((4096, r1_fused_relax_matmul_cublas), dtype="float16"),
            ) -> R.Tensor((1, r1_fused_relax_matmul_cublas), dtype="float16"):
                R.func_attr({"Composite": "cublas.matmul"})
                with R.dataflow():
                    gv_1: R.Tensor((1, r1_fused_relax_matmul_cublas), dtype="float16") = R.matmul(
                        x_1, w1_1, out_dtype=None
                    )
                    R.output(gv_1)
                return gv_1

            gv1: R.Tensor((1, r1_fused_relax_matmul_cublas), dtype="float16") = gv(x, w1)
            return gv1

    r1 = T.dynamic("r1", "int64")
    r2 = T.dynamic("r2", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 4096), dtype="float16"),
            w1: R.Tensor((4096, r1), dtype="float16"),
            w2: R.Tensor((4096, r2), dtype="float16"),
        ) -> R.Tuple(R.Tensor((1, r1), dtype="float16"), R.Tensor((1, r2), dtype="float16")):
            with R.dataflow():
                lv = R.call_dps_packed(
                    _ExternFunc("fused_relax_matmul_cublas"),
                    (x, w1),
                    ty_args=[R.Tensor((1, r1), dtype="float16")],
                )
                lv1 = R.call_dps_packed(
                    _ExternFunc("fused_relax_matmul_cublas"),
                    (x, w2),
                    ty_args=[R.Tensor((1, r2), dtype="float16")],
                )
                gv: R.Tuple(
                    R.Tensor((1, r1), dtype="float16"), R.Tensor((1, r2), dtype="float16")
                ) = (lv, lv1)
                R.output(gv)
            return gv

    after = relax.transform.RunCodegen()(Before)
    tvm_tirx.ir.assert_structural_equal(after["main"], Expected["main"])


def test_no_op_for_call_to_tir():
    """Calls to Function are ignored

    RunCodegen should only update calls to Relax functions annotated
    with the `"Codegen"` attribute.  Calls to any other function type
    should be ignored.

    This is a regression test.  Previous implementations performed an
    unconditional cast from `tvm::BaseFunc` to `tvm::relax::Function`,
    which produced an error.
    """

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor([4], "int64")):
            R.func_attr({"relax.force_pure": True})
            _ = R.call_tir_packed(Before.shape_func, (x,))
            return x

        @Ts.function(private=True)
        def shape_func(H: T.Tensor(T.int64(4), "int64")):
            H[T.int64(0)] = H[T.int64(0)] + T.int64(1)

    Expected = Before
    After = relax.transform.RunCodegen()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


# TODO(@sunggg):  test with more complex patterns (e.g., multiple annots, mixed codegens, different ops, const binding)

if __name__ == "__main__":
    pytest.main([__file__])
