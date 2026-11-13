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
# ruff: noqa: F841
"""S-TIR script basic usage."""

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import assert_structural_equal_ignore_global_symbol
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@Ts.function
def get_valid_counts(
    data_buf: T.Tensor((1, 2500, 6), "float32"),
    valid_count_buf: T.Tensor((1,), "int32"),
    out_buf: T.Tensor((1, 2500, 6), "float32"),
    out_indices_buf: T.Tensor((1, 2500), "int32"),
    score_threshold: T.float32,
    id_index: T.int32,
    score_index: T.int32,
) -> None:
    with Ts.sblock("init"):
        vi = Ts.axis.S(1, 0)
        valid_count_buf[vi] = T.int32(0)
        for j in range(2500):
            with Ts.sblock("update"):
                vj = Ts.axis.S(2500, j)
                Ts.reads([data_buf[vi, vj, 6]])
                Ts.writes([valid_count_buf[vi], out_indices_buf[vi, vj], out_buf[vi, vj, 6]])
                if (data_buf[vi, vj, score_index] > score_threshold) and (
                    (id_index < 0) or (data_buf[vi, vj, id_index] >= T.float32(0))
                ):
                    for k in T.serial(0, 6):
                        out_buf[vi, valid_count_buf[vi], k] = data_buf[vi, vj, k]
                    out_indices_buf[vi, valid_count_buf[vi]] = vj
                    valid_count_buf[vi] = valid_count_buf[vi] + 1
                if vj >= valid_count_buf[vi]:
                    for k in T.serial(0, 6):
                        out_buf[vi, vj, k] = T.float32(-1)
                    out_indices_buf[vi, vj] = T.int32(-1)


def _check_get_valid_counts_with_numpy(f, dshape, score_threshold, id_index, score_index):
    dtype = "float32"
    ctx = tvm_tirx.cpu()
    batch_size, num_anchor, elem_length = dshape
    np_data = np.random.uniform(low=-2, high=2, size=dshape).astype(dtype)
    np_out1 = np.zeros(shape=(batch_size,), dtype="int32")
    np_out2 = np.zeros(shape=dshape).astype(dtype)
    np_out3 = np.zeros(shape=(batch_size, num_anchor), dtype="int32")
    for i in range(batch_size):
        np_out1[i] = 0
        inter_idx = 0
        for j in range(num_anchor):
            score = np_data[i, j, score_index]
            if score > score_threshold and (id_index < 0 or np_data[i, j, id_index] >= 0):
                for k in range(elem_length):
                    np_out2[i, inter_idx, k] = np_data[i, j, k]
                np_out1[i] += 1
                np_out3[i, inter_idx] = j
                inter_idx += 1
            if j >= np_out1[i]:
                for k in range(elem_length):
                    np_out2[i, j, k] = -1.0
                np_out3[i, j] = -1

    in_data = tvm_tirx.runtime.tensor(np_data, ctx)
    out1 = tvm_tirx.runtime.tensor(np_out1, ctx)
    out2 = tvm_tirx.runtime.tensor(np_out2, ctx)
    out3 = tvm_tirx.runtime.tensor(np_out3, ctx)
    f(in_data, out1, out2, out3, score_threshold, id_index, score_index)
    tvm_tirx.testing.assert_allclose(out1.numpy(), np_out1, rtol=1e-5)
    tvm_tirx.testing.assert_allclose(out2.numpy(), np_out2, rtol=1e-5)
    tvm_tirx.testing.assert_allclose(out3.numpy(), np_out3, rtol=1e-5)
    print("test get_valid_counts end")


def test_get_valid_counts_script_func():
    device = "llvm"
    # check lowering
    print(get_valid_counts.script())
    mod = tvm_tirx.ir.IRModule({"get_valid_counts": get_valid_counts})
    print(mod.script())
    # check building
    f = tvm.compile(mod["get_valid_counts"], target=device)
    _check_get_valid_counts_with_numpy(f, (1, 2500, 6), 0.0, 0, 1)


def test_tir_func_name():
    @Ts.function
    def matmul(
        A: T.Tensor([128, 128], "float32"),
        B: T.Tensor([128, 128], "float32"),
        C: T.Tensor([128, 128], "float32"),
    ) -> None:
        for i, j, k in T.grid(128, 128, 128):
            with Ts.sblock("update"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    assert matmul.__name__ == "matmul"
    assert matmul.attrs["global_symbol"] == "matmul"


def test_tir_func_private_attrs():
    @Ts.function(private=True)
    def matmul(
        A: T.Tensor([128, 128], "float32"),
        B: T.Tensor([128, 128], "float32"),
        C: T.Tensor([128, 128], "float32"),
    ) -> None:
        T.func_attr({"attr": "value"})

        for i, j, k in T.grid(128, 128, 128):
            with Ts.sblock("update"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    assert "global_symbol" not in matmul.attrs


def test_reinterpret_preserves_explicit_call():
    """The shared constructor retains operands even when result types match."""

    @Ts.function
    def func(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")) -> None:
        T.func_attr({"global_symbol": "main"})
        for i in T.serial(0, 32):
            with Ts.sblock():
                vi = Ts.axis.remap("S", [i])
                B[vi] = T.reinterpret(A[vi], ty="float32")

    @Ts.function
    def expected(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")) -> None:
        T.func_attr({"global_symbol": "main"})
        for i in T.serial(0, 32):
            with Ts.sblock():
                vi = Ts.axis.remap("S", [i])
                B[vi] = I.Call(
                    "tirx.reinterpret",
                    [A[vi]],
                    ty="float32",
                )

    tvm_tirx.ir.assert_structural_equal(func, expected)


def test_block_annotation_merge():
    def _to_dict(anno: tvm_ffi.container.Map):
        result = {}
        for k, v in anno.items():
            result[k] = _to_dict(v) if isinstance(v, tvm_ffi.container.Map) else v
        return result

    @Ts.function
    def func0():
        with Ts.sblock():
            Ts.sblock_attr({"key1": "block1"})
            Ts.sblock_attr({"key2": "block2"})
            T.evaluate(0)

    assert _to_dict(func0.body[0].block.annotations) == {"key1": "block1", "key2": "block2"}

    @Ts.function
    def func1():
        with Ts.sblock():
            Ts.sblock_attr({"key": {"key1": "block1"}})
            Ts.sblock_attr({"key": {"key2": "block2"}})
            T.evaluate(0)

    assert _to_dict(func1.body[0].block.annotations) == {
        "key": {"key1": "block1", "key2": "block2"}
    }

    @Ts.function
    def func2():
        with Ts.sblock():
            Ts.sblock_attr({"key1": "block1"})
            Ts.sblock_attr({"key1": "block1"})
            T.evaluate(0)

    assert _to_dict(func2.body[0].block.annotations) == {"key1": "block1"}

    with pytest.raises(RuntimeError):

        @Ts.function
        def func3():
            with Ts.sblock():
                Ts.sblock_attr({"key1": "block1"})
                Ts.sblock_attr({"key1": "block2"})
                T.evaluate(0)


def test_alloc_inside_block():
    @Ts.function(private=True)
    def func() -> None:
        with Ts.sblock():
            A = Ts.sblock_alloc_tensor([10], "float32")
            for i in T.serial(0, 10):
                B = Ts.sblock_alloc_tensor([10], "float32")
                for j in T.serial(0, 10):
                    B[j] = T.float32(j)
                    A[i] += B[j]

    @Ts.function(private=True)
    def expected() -> None:
        with Ts.sblock():
            A = Ts.sblock_alloc_tensor([10], "float32")
            B = Ts.sblock_alloc_tensor([10], "float32")
            for i, j in T.grid(10, 10):
                B[j] = T.float32(j)
                A[i] += B[j]

    tvm_tirx.ir.assert_structural_equal(func, expected)


def while_loop():
    @Ts.function
    def while_loop(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")) -> None:
        i = Ts.sblock_alloc_tensor((), "int32", scope="local")
        for ii in range(16):
            with Ts.sblock():
                vi = Ts.axis.S(16, ii)
                B[vi] = 0
            while i[()] < 10:
                for j in range(16):
                    B[j] += A[j]

    return while_loop


@Ts.function
def elementwise_tensor_default_dtype(
    A: T.Tensor((128, 128, 128, 128), "float32"),
    B: T.Tensor((128, 128, 128, 128), "float32"),
) -> None:
    for i, j, k, l in T.grid(128, 128, 128, 128):  # noqa: E741
        with Ts.sblock("B"):
            vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@Ts.function
def elementwise_tensor_kwargs(
    a: T.Tensor(shape=(128, 128, 128, 128), dtype="float32"),
    b: T.Tensor(shape=(128, 128, 128, 128), dtype="float32"),
) -> None:
    for i, j, k, l in T.grid(128, 128, 128, 128):  # noqa: E741
        with Ts.sblock("B"):
            vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
            b[vi, vj, vk, vl] = a[vi, vj, vk, vl] * 2.0


@Ts.function
def elementwise_tensor_no_kwargs(
    a: T.Tensor((128, 128, 128, 128), "float32"),
    b: T.Tensor((128, 128, 128, 128), "float32"),
) -> None:
    for i, j, k, l in T.grid(128, 128, 128, 128):  # noqa: E741
        with Ts.sblock("B"):
            vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
            b[vi, vj, vk, vl] = a[vi, vj, vk, vl] * 2.0


def test_tensor_signature_syntax_sugar():
    # with kwargs
    assert_structural_equal_ignore_global_symbol(
        elementwise_tensor_default_dtype, elementwise_tensor_kwargs
    )
    # without kwargs
    assert_structural_equal_ignore_global_symbol(
        elementwise_tensor_default_dtype, elementwise_tensor_no_kwargs
    )


@pytest.mark.parametrize(
    "ir_generator",
    [while_loop],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_basic_usage(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm_tirx.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
            "Ts": tvm.s_tir.script,
            "R": tvm.relax.script,
        },
    )
    tvm_tirx.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)


# Import-time construction also checks the annotated S-TIR API.
@Ts.function
def element_wise_env_thread_x(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    i = T.launch_thread("blockIdx.x", 128)
    j0_0 = T.launch_thread("threadIdx.x", 4)
    j1_0 = T.launch_thread("threadIdx.x", 4)

    for blockIdx_x in T.parallel(0, 128, annotations={"thread_binding": "blockIdx.x"}):
        for threadIdx_x in T.parallel(0, 4, annotations={"thread_binding": "threadIdx.x"}):
            for j0_1 in T.serial(0, 32):
                with Ts.sblock(""):
                    B[blockIdx_x, threadIdx_x * 32 + j0_1] = (
                        A[blockIdx_x, threadIdx_x * 32 + j0_1] * 2.0
                    )
            for j1_1 in T.serial(0, 32):
                with Ts.sblock(""):
                    C[blockIdx_x, threadIdx_x * 32 + j1_1] = (
                        B[blockIdx_x, threadIdx_x * 32 + j1_1] + 1.0
                    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
