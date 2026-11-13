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
# ruff: noqa: F401, F811
import pytest

pytest.importorskip("torch._dynamo")


import torch
import torch._dynamo as dynamo
from packaging import version

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.frontend.torch import relax_dynamo
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

torch_version = torch.__version__


def test_relax_dynamo():
    class Input1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(100, 10)

        def forward(self, x):
            return torch.nn.functional.relu(self.lin(x))

    model = Input1()

    ### construct the database
    @tvm_tirx.script.ir_module
    class Input1_ir:
        @Ts.function
        def main(
            inp_0: T.Tensor((T.int64(10), T.int64(100)), "float32"),
            param_0: T.Tensor((T.int64(100), T.int64(10)), "float32"),
            param_1: T.Tensor(T.int64(10), "float32"),
            compute: T.Tensor((T.int64(10), T.int64(10)), "float32"),
        ):
            # function attr dict
            T.func_attr({"tirx.noalias": True, "global_symbol": "main"})
            # body
            # with Ts.sblock("root")
            matmul = Ts.sblock_alloc_tensor([T.int64(10), T.int64(10)], dtype="float32")
            T_add = Ts.sblock_alloc_tensor([T.int64(10), T.int64(10)], dtype="float32")
            for i0, i1, k in T.grid(T.int64(10), T.int64(10), T.int64(100)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(inp_0[v_i0, v_k], param_0[v_k, v_i1])
                    Ts.writes(matmul[v_i0, v_i1])
                    with Ts.init():
                        matmul[v_i0, v_i1] = T.float32(0)
                    matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + inp_0[v_i0, v_k] * param_0[v_k, v_i1]
            for ax0, ax1 in T.grid(T.int64(10), T.int64(10)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(matmul[v_ax0, v_ax1], param_1[v_ax1])
                    Ts.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = matmul[v_ax0, v_ax1] + param_1[v_ax1]
            for i0, i1 in T.grid(T.int64(10), T.int64(10)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(T_add[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.max(T_add[v_i0, v_i1], T.float32(0))

    db = ms.Database.create("memory")
    workload = db.commit_workload(Input1_ir)

    sch = tvm.s_tir.Schedule(Input1_ir, debug_mask="all")
    b0 = sch.get_sblock(name="matmul", func_name="main")
    b1 = sch.get_sblock(name="T_add", func_name="main")
    b2 = sch.get_sblock(name="root", func_name="main")
    sch.compute_inline(block=b1)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
    l3, l4, l5 = sch.get_loops(block=b0)
    v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l3, n=4, max_innermost_factor=64, decision=[1, 2, 5, 1]
    )
    l10, l11, l12, l13 = sch.split(loop=l3, factors=[v6, v7, v8, v9], preserve_unit_iters=True)
    v14, v15, v16, v17 = sch.sample_perfect_tile(
        loop=l4, n=4, max_innermost_factor=64, decision=[1, 1, 10, 1]
    )
    l18, l19, l20, l21 = sch.split(loop=l4, factors=[v14, v15, v16, v17], preserve_unit_iters=True)
    v22, v23 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64, decision=[100, 1])
    l24, l25 = sch.split(loop=l5, factors=[v22, v23], preserve_unit_iters=True)
    sch.reorder(l10, l18, l11, l19, l24, l12, l20, l25, l13, l21)
    (b26,) = sch.get_consumers(block=b0)
    sch.reverse_compute_at(block=b26, loop=l18, preserve_unit_loops=True, index=-1)
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.parallel", ann_val=96)
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.vectorize", ann_val=64)
    v27 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v27)

    tuning_record = ms.database.TuningRecord(sch.trace, workload, run_secs=[0.0])
    db.commit_tuning_record(tuning_record)
    ### Optimize the model with tuned-log
    with db:
        opt_model = torch.compile(model, backend=relax_dynamo())
    inp = torch.randn(10, 100)

    default_output = model(inp).detach().numpy()
    optimized_output = opt_model(inp).detach().numpy()
    tvm_tirx.testing.assert_allclose(optimized_output, default_output, rtol=1e-5, atol=1e-5)


def test_relax_dynamo_scalar_params():
    class ScalarParams(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.nn.Parameter(torch.tensor(1.0))
            self.y = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self):
            return self.x + self.y

    model = ScalarParams()

    opt_model = torch.compile(model, backend=relax_dynamo())

    default_output = model().detach().numpy()
    optimized_output = opt_model().detach().numpy()
    tvm_tirx.testing.assert_allclose(optimized_output, default_output, rtol=1e-5, atol=1e-5)


def test_relax_dynamo_dynamic():
    class Input1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(100, 10)

        def forward(self, x):
            return torch.nn.functional.relu(self.lin(x))

    model = Input1()

    opt_model = torch.compile(model, backend=relax_dynamo(), dynamic=True)

    inp = torch.randn(10, 100)
    tvm_tirx.testing.assert_allclose(
        opt_model(inp).detach().numpy(), model(inp).detach().numpy(), rtol=1e-5, atol=1e-5
    )

    def Func1(x, y):
        z = torch.cat([x, y])
        if z.size(0) > 5:
            return z.mul(2)
        else:
            return z.add(2)

    opt_func = torch.compile(Func1, backend=relax_dynamo(), dynamic=True)

    for s in (2, 4):
        x = torch.randn(s, 100)
        y = torch.randn(s, 100)
        with torch.no_grad():
            tvm_tirx.testing.assert_allclose(Func1(x, y), opt_func(x, y))


def test_relax_dynamo_dynamic_sym_input_reference():
    class ViewModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x).view(x.size(0), -1)

    model = ViewModel()
    opt_model = torch.compile(model, backend=relax_dynamo(), dynamic=True)

    with torch.no_grad():
        for s in (1, 2, 4):
            inp = torch.randn(s, 3, 8, 8)
            tvm_tirx.testing.assert_allclose(
                opt_model(inp).detach().numpy(), model(inp).detach().numpy(), rtol=1e-5, atol=1e-5
            )


def test_subgraph_capture():
    import torch

    from tvm.relax.frontend.torch.dynamo import dynamo_capture_subgraphs

    class Input1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(100, 10)

        def forward(self, x):
            return torch.nn.functional.relu(self.lin(x))

    @tvm_tirx.script.ir_module
    class Expected1:
        @R.function
        def subgraph_0(
            inp_0: R.Tensor((10, 100), dtype="float32"),
            w1: R.Tensor((10,), dtype="float32"),
            w0: R.Tensor((10, 100), dtype="float32"),
        ) -> R.Tensor((10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((100, 10), dtype="float32") = R.permute_dims(inp_0, axes=None)
                lv1: R.Tensor((10, 10), dtype="float32") = R.matmul(w0, lv, out_dtype="float32")
                lv2: R.Tensor((10, 10), dtype="float32") = R.add(lv1, w1)
                lv3: R.Tensor((10, 10), dtype="float32") = R.nn.relu(lv2)
                gv: R.Tensor((10, 10), dtype="float32") = lv3
                R.output(gv)
            return gv

    model = Input1()
    mod = dynamo_capture_subgraphs(model, torch.randn(10, 100))
    tvm_tirx.ir.assert_structural_equal(mod, Expected1)

    def Input2(a, b):
        x = a / (torch.sin(a) + 1)
        if torch.sum(b) < 1:
            b = b * -1
        return x * b

    @tvm_tirx.script.ir_module
    class Expected2:
        @R.function
        def subgraph_0(
            inp_0: R.Tensor((10,), dtype="float32"), inp_1: R.Tensor((10,), dtype="float32")
        ) -> R.Tuple(R.Tensor((), dtype="bool"), R.Tensor((10,), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10,), dtype="float32") = R.sin(inp_0)
                lv1: R.Tensor((10,), dtype="float32") = R.add(lv, R.const(1.0, "float32"))
                lv2: R.Tensor((10,), dtype="float32") = R.divide(inp_0, lv1)
                lv3: R.Tensor((), dtype="float32") = R.sum(inp_1, axis=None, keepdims=False)
                lv4: R.Tensor((), dtype="bool") = R.less(lv3, R.const(1.0, "float32"))
                gv: R.Tuple(R.Tensor((), dtype="bool"), R.Tensor((10,), dtype="float32")) = (
                    lv4,
                    lv2,
                )
                R.output(gv)
            return gv

        @R.function
        def subgraph_1(
            inp_0: R.Tensor((10,), dtype="float32"), inp_1: R.Tensor((10,), dtype="float32")
        ) -> R.Tensor((10,), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10,), dtype="float32") = R.multiply(inp_0, inp_1)
                gv: R.Tensor((10,), dtype="float32") = lv
                R.output(gv)
            return gv

    mod = dynamo_capture_subgraphs(Input2, torch.randn(10), torch.ones(10))
    tvm_tirx.ir.assert_structural_equal(mod, Expected2)

    class Input3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(100, 10)

        def forward(self, x, add_one=False):
            if add_one:
                x = x + 1
            return torch.nn.functional.relu(self.lin(x))

    @tvm_tirx.script.ir_module
    class Expected3:
        @R.function
        def subgraph_0(
            inp_0: R.Tensor((10, 100), dtype="float32"),
            w0: R.Tensor((10, 100), dtype="float32"),
            w1: R.Tensor((10,), dtype="float32"),
        ) -> R.Tensor((10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 100), dtype="float32") = R.add(inp_0, R.const(1.0, "float32"))
                lv1: R.Tensor((100, 10), dtype="float32") = R.permute_dims(w0, axes=None)
                lv2: R.Tensor((10, 10), dtype="float32") = R.matmul(lv, lv1, out_dtype="float32")
                lv3: R.Tensor((10, 10), dtype="float32") = R.add(lv2, w1)
                lv4: R.Tensor((10, 10), dtype="float32") = R.nn.relu(lv3)
                gv: R.Tensor((10, 10), dtype="float32") = lv4
                R.output(gv)
            return gv

    model = Input3()
    mod = dynamo_capture_subgraphs(model, torch.randn(10, 100), add_one=True)
    tvm_tirx.ir.assert_structural_equal(mod, Expected3)


def verify_dynamo_model(torch_model, input_info, binding, expected):
    import torch
    import torch._dynamo as dynamo

    from tvm.relax.frontend.torch import from_fx

    args = []
    for info in input_info:
        args.append(torch.zeros(*info[0], dtype=_convert_data_type(info[1])))
    graph_model = dynamo.export(torch_model)(*args)[0]
    mod = from_fx(graph_model, input_info, unwrap_unit_return_tuple=True)
    binding = {k: tvm_tirx.runtime.tensor(v) for k, v in binding.items()}
    expected = relax.transform.BindParams("main", binding)(expected)
    tvm_tirx.ir.assert_structural_equal(mod, expected)


def _convert_data_type(input_type):
    """converts the PyTorch scalar type input_type to a TVM dtype."""
    import torch  # type: ignore

    input_type = input_type.lower() if isinstance(input_type, str) else input_type
    # Float types
    if input_type == "float16":
        return torch.float16
    elif input_type == "float32":
        return torch.float32
    elif input_type == "float64":
        return torch.float64
    elif input_type == "bfloat16":
        return torch.bfloat16
    # Signed integer types
    elif input_type == "int8":
        return torch.int8
    elif input_type == "int16":
        return torch.int16
    elif input_type == "int32":
        return torch.int32
    elif input_type == "int64":
        return torch.int64
    # Unsigned integer types
    elif input_type == "uint8":
        return torch.uint8
    elif input_type == "uint16":
        return torch.uint16
    elif input_type == "uint32":
        return torch.uint32
    elif input_type == "uint64":
        return torch.uint64
    # Boolean
    elif input_type == "bool":
        return torch.bool
    else:
        raise NotImplementedError(f"input_type {input_type} is not handled yet")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_ones():
    import torch
    from torch.nn import Module

    class Ones(Module):
        def forward(self, input):
            return torch.ones((10, 10), dtype=torch.float32)

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32"),
        ) -> R.Tensor((10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.full(
                    R.shape([10, 10]), R.const(1, "float32"), dtype="float32"
                )
                gv: R.Tensor((10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_dynamo_model(
        Ones(),
        [([256, 256], "float32")],
        {},
        Expected1,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_full():
    import torch
    from torch.nn import Module

    class Full(Module):
        def forward(self, input):
            return torch.full((10, 10), 1, dtype=torch.float32)

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32"),
        ) -> R.Tensor((10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.full(
                    R.shape([10, 10]), R.const(1, "float32"), dtype="float32"
                )
                gv: R.Tensor((10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_dynamo_model(
        Full(),
        [([256, 256], "float32")],
        {},
        Expected1,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_gelu():
    import torch
    from torch.nn import Module

    class GeLU(Module):
        def forward(self, input):
            return torch.nn.functional.gelu(input)

    class GeLUTanh(Module):
        def forward(self, input):
            return torch.nn.functional.gelu(input, approximate="tanh")

    @I.ir_module
    class ExpectedGeLU:
        @R.function
        def main(
            inp_0: R.Tensor((128, 256), dtype="float32"),
        ) -> R.Tensor((128, 256), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((128, 256), dtype="float32") = R.nn.gelu(inp_0)
                gv: R.Tensor((128, 256), dtype="float32") = lv
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedGeLUTanh:
        @R.function
        def main(
            inp_0: R.Tensor((128, 256), dtype="float32"),
        ) -> R.Tensor((128, 256), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((128, 256), dtype="float32") = R.nn.gelu_tanh(inp_0)
                gv: R.Tensor((128, 256), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_dynamo_model(
        GeLU(),
        [([128, 256], "float32")],
        {},
        ExpectedGeLU,
    )

    verify_dynamo_model(
        GeLUTanh(),
        [([128, 256], "float32")],
        {},
        ExpectedGeLUTanh,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_masked_fill():
    import torch
    from torch.nn import Module

    class MaskedFill(Module):
        def forward(self, mask, input):
            return input.masked_fill(mask, 0)

    class InplaceMaskedFill(Module):
        def forward(self, mask, input):
            input.masked_fill_(mask, 0)
            return input

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="bool"), inp_1: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tensor((256, 256), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((256, 256), dtype="float32") = R.full_like(
                    inp_1, R.const(0.0, "float32"), dtype=None
                )
                lv1: R.Tensor((256, 256), dtype="float32") = R.where(inp_0, lv, inp_1)
                gv: R.Tensor((256, 256), dtype="float32") = lv1
                R.output(gv)
            return gv

    verify_dynamo_model(
        MaskedFill(), [([256, 256], "bool"), ([256, 256], "float32")], {}, Expected1
    )
    verify_dynamo_model(
        InplaceMaskedFill(), [([256, 256], "bool"), ([256, 256], "float32")], {}, Expected1
    )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_getitem():
    import torch
    from torch.nn import Module

    class Select1(Module):
        def forward(self, input1, input2):
            result = input1[:, input2.argmax(dim=-1), :]
            return result

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((1, 77, 1280), dtype="float32"),
            inp_1: R.Tensor((1, 77), dtype="float32"),
        ) -> R.Tensor((1, 1, 1280), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1,), dtype="int64") = R.argmax(inp_1, axis=-1, keepdims=False)
                lv1: R.Tensor((1, 1, 1280), dtype="float32") = R.take(inp_0, lv, axis=1)
                lv2: R.Tensor((1, 1, 1280), dtype="float32") = R.strided_slice(
                    lv1,
                    axes=[0, 2],
                    begin=[0, 0],
                    end=[1, 1280],
                    strides=[1, 1],
                    assume_inbound=False,
                )
                lv3: R.Tensor((1, 1, 1280), dtype="float32") = R.reshape(lv2, R.shape([1, 1, 1280]))
                gv: R.Tensor((1, 1, 1280), dtype="float32") = lv3
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((1, 77, 1280), dtype="float32"),
        ) -> R.Tensor((1, 77, 1280), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(1), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((1, 77, 1280), dtype="float32") = R.take(inp_0, lv, axis=0)
                lv2: R.Tensor((1, 77, 1280), dtype="float32") = R.strided_slice(
                    lv1,
                    axes=[1, 2],
                    begin=[0, 0],
                    end=[77, 1280],
                    strides=[1, 1],
                    assume_inbound=False,
                )
                lv3: R.Tensor((1, 77, 1280), dtype="float32") = R.reshape(
                    lv2, R.shape([1, 77, 1280])
                )
                gv: R.Tensor((1, 77, 1280), dtype="float32") = lv3
                R.output(gv)
            return gv

    class Select2(Module):
        def forward(self, input1):
            result = input1[torch.arange(1),]
            return result

    verify_dynamo_model(
        Select1(), [([1, 77, 1280], "float32"), ([1, 77], "float32")], {}, Expected1
    )
    verify_dynamo_model(Select2(), [([1, 77, 1280], "float32")], {}, Expected2)


@pytest.mark.skipif(
    version.parse(torch_version) >= version.parse("2.6.0"),
    reason="Need to support dynamic arange in Relax",
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_arange():
    import torch
    from torch.nn import Module

    class Arange1(Module):
        def forward(self, input0):
            mask_cond = torch.arange(input0.size(-1))
            result = mask_cond + 1
            return result

    @I.ir_module
    class Expected1:
        @R.function
        def main(inp_0: R.Tensor((1, 77), dtype="float32")) -> R.Tensor((77,), dtype="int64"):
            with R.dataflow():
                lv: R.Tensor((77,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(77), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((77,), dtype="int64") = R.add(lv, R.const(1, "int64"))
                gv: R.Tensor((77,), dtype="int64") = lv1
                R.output(gv)
            return gv

    verify_dynamo_model(Arange1(), [([1, 77], "float32")], {}, Expected1)


if __name__ == "__main__":
    tvm_tirx.testing.main()
