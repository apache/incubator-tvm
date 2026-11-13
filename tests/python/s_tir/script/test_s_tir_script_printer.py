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
"""S-TIR script printer."""

import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.ir import Range, TensorRegion, TensorRegionType
from tvm_tirx.script import tirx as T
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as TB


def _assert_print(obj, expected):
    assert obj.script(verbose_expr=True).strip() == expected.strip()


def test_block_realize():
    i = tirx.Var("i", "int32")
    j = tirx.Var("j", "int32")
    k = tirx.Var("k", "int32")
    with IRBuilder() as ib:
        with Ts.sblock(name="block", no_realize=False):
            vi = ib.name("vi", Ts.axis.spatial(128, i))
            vj = ib.name("vj", Ts.axis.spatial(64, j))
            vk = ib.name("vk", Ts.axis.reduce(32, k))
            Ts.reads()
            Ts.writes()
            TB.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
i = T.dynamic("i", dtype="int32")
j = T.dynamic("j", dtype="int32")
k = T.dynamic("k", dtype="int32")
with Ts.sblock("block"):
    vi = Ts.axis.spatial(128, i)
    vj = Ts.axis.spatial(64, j)
    vk = Ts.axis.reduce(32, k)
    Ts.reads()
    Ts.writes()
""",
    )


def test_block():
    i = tirx.Var("i", "int32")
    j = tirx.Var("j", "int32")
    k = tirx.Var("k", "int32")
    with IRBuilder() as ib:
        with Ts.sblock(name="block", no_realize=False):
            vi = ib.name("vi", Ts.axis.spatial(128, i))
            vj = ib.name("vj", Ts.axis.spatial(64, j))
            vk = ib.name("vk", Ts.axis.reduce(32, k))
            Ts.reads()
            Ts.writes()
            TB.evaluate(0)
    obj = ib.get().block
    _assert_print(
        obj,
        """
with Ts.sblock("block", no_realize=True):
    vi = Ts.axis.spatial(128)
    vj = Ts.axis.spatial(64)
    vk = Ts.axis.reduce(32)
    Ts.reads()
    Ts.writes()
""",
    )


def test_match_tensor_region():
    src = tirx.Var("src", tirx.TensorType((128, 128), "float32"))
    tgt = tirx.Var("tgt", tirx.TensorType((64, 64), "float32"))
    obj = s_tir.MatchTensorRegion(
        tgt,
        TensorRegion(
            src,
            [
                Range(64, 128),
                Range(64, 128),
            ],
            ty=TensorRegionType(),
        ),
    )
    _assert_print(
        obj,
        """
src = T.Var("src", T.Tensor((128, 128), T.float32))
tgt = Ts.match_tensor(src[64:128, 64:128], (64, 64), T.float32)
""",
    )


def test_remap():
    from tvm_tirx.script import tirx as TB

    @Ts.function
    def block_with_remap_implicitly():
        for i0, i1, i2, i3, i4, i5 in TB.grid(128, 128, 128, 128, 128, 128):
            with Ts.sblock("update"):
                v0 = Ts.axis.spatial(128, i0 + 1)
                v1 = Ts.axis.spatial(128, i1)
                v2 = Ts.axis.reduce(128, i2)
                v3 = Ts.axis.spatial(128, i3 - 1)
                v4 = Ts.axis.reduce(128, i4)
                v5 = Ts.axis.spatial(128, i5)

    @Ts.function
    def block_with_remap_explicitly():
        for i0, i1, i2, i3, i4, i5 in TB.grid(128, 128, 128, 128, 128, 128):
            with Ts.sblock("update"):
                v0 = Ts.axis.spatial(128, i0 + 1)
                v1, v2 = Ts.axis.remap("SR", [i1, i2])
                v3 = Ts.axis.spatial(128, i3 - 1)
                v4, v5 = Ts.axis.remap("RS", [i4, i5])

    expected_output = """
# from tvm.s_tir import script as Ts
# from tvm_tirx.script import tirx as T

@Ts.function
def main():
    T.func_attr({"script.namespace": "s_tir"})
    with Ts.sblock("root"):
        Ts.reads()
        Ts.writes()
        for i0 in range(128):
            for i1 in range(128):
                for i2 in range(128):
                    for i3 in range(128):
                        for i4 in range(128):
                            for i5 in range(128):
                                with Ts.sblock("update"):
                                    v = Ts.axis.spatial(128, i0 + 1)
                                    v_1 = Ts.axis.spatial(128, i1)
                                    v_2 = Ts.axis.reduce(128, i2)
                                    v_3 = Ts.axis.spatial(128, i3 - 1)
                                    v_4 = Ts.axis.reduce(128, i4)
                                    v_5 = Ts.axis.spatial(128, i5)
                                    Ts.reads()
                                    Ts.writes()
"""
    _assert_print(block_with_remap_explicitly.with_attr("global_symbol", "main"), expected_output)
    _assert_print(block_with_remap_implicitly.with_attr("global_symbol", "main"), expected_output)


def test_root_block():
    from tvm_tirx.script import tirx as TB

    @Ts.function
    def root_block_implicitly():
        a = Ts.sblock_alloc_tensor([128, 128])
        for i, j in TB.grid(128, 128):
            with Ts.sblock():
                TB.evaluate(0)

    @Ts.function
    def root_block_explicitly():
        with Ts.sblock("root"):
            a = Ts.sblock_alloc_tensor([128, 128])
            for i, j in TB.grid(128, 128):
                with Ts.sblock():
                    TB.evaluate(0)

    expected_output = """
# from tvm.s_tir import script as Ts
# from tvm_tirx.script import tirx as T

@Ts.function
def main():
    T.func_attr({"script.namespace": "s_tir"})
    with Ts.sblock("root"):
        Ts.reads()
        Ts.writes()
        v = Ts.sblock_alloc_tensor((128, 128), T.float32)
        for i in range(128):
            for j in range(128):
                with Ts.sblock(""):
                    Ts.reads()
                    Ts.writes()
"""
    _assert_print(root_block_implicitly.with_attr("global_symbol", "main"), expected_output)
    _assert_print(root_block_explicitly.with_attr("global_symbol", "main"), expected_output)


def abs():
    @Ts.function
    def abs(A: T.Tensor((128, 128), "float32")) -> None:
        for i, j in T.grid(128, 128):
            with Ts.sblock("A"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                A[vi, vj] = T.abs(A[vi, vj])

    return abs


def var_with_same_name():
    @Ts.function
    def var_with_same_name(A: T.Tensor((16, 16), "float32")) -> None:
        for i, j in T.grid(16, 16):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                A[vi, vj] = 0
        for i, j in T.grid(16, 16):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                A[vi, vj] = 0

    return var_with_same_name


def test_same_name_var():
    func = var_with_same_name()
    out_str = func.script()
    rt_func = tvm_tirx.script.from_source(
        out_str,
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
            "Ts": tvm.s_tir.script,
            "R": tvm.relax.script,
        },
    )
    tvm_tirx.ir.assert_structural_equal(func, rt_func)
    assert out_str.count("for i in range(16):\n            for j in range(16):") == 1
    assert out_str.count("for i_1 in range(16):\n            for j_1 in range(16):") == 1
    assert out_str.find("i_2") == -1
    assert out_str.find("j_2") == -1


def multiple_commreducer():
    # normal_reduce_temp0 is treated as uninitialized value
    @Ts.function(check_well_formed=False)
    def multiple_commreducer() -> None:
        normal_reduce_temp0 = T.Var(
            "normal_reduce_temp0",
            T.Tensor([1], dtype="float32", layout=T.TileLayout(T.S[[1] : [1]]), scope="local"),
        )
        normal_reduce_temp1 = T.Var(
            "normal_reduce_temp1",
            T.Tensor([1], dtype="float32", layout=T.TileLayout(T.S[[1] : [1]]), scope="local"),
        )
        reduce_temp0 = T.Var(
            "reduce_temp0",
            T.Tensor([1], dtype="float32", layout=T.TileLayout(T.S[[1] : [1]]), scope="local"),
        )
        reduce_temp1 = T.Var(
            "reduce_temp1",
            T.Tensor([1], dtype="float32", layout=T.TileLayout(T.S[[1] : [1]]), scope="local"),
        )
        for ax0_1 in T.parallel(0, 32, annotations={"thread_binding": "threadIdx.x"}):
            with Ts.sblock("T_softmax_maxelem_cross_thread_reduction"):
                T.evaluate(
                    T.gpu_thread_allreduce(
                        T.Lambda([T.float32, T.float32], lambda x, y: (T.max(x, y),)),
                        (T.min_value(ty="float32"),),
                        (normal_reduce_temp0[0],),
                        True,
                        (reduce_temp0[0],),
                        (ax0_1,),
                    )
                )
        for ax0_1 in T.parallel(0, 32, annotations={"thread_binding": "threadIdx.x"}):
            with Ts.sblock("T_softmax_expsum_cross_thread_reduction"):
                T.evaluate(
                    T.gpu_thread_allreduce(
                        T.Lambda([T.float32, T.float32], lambda x, y: (x + y,)),
                        (T.float32(0),),
                        (normal_reduce_temp1[0],),
                        True,
                        (reduce_temp1[0],),
                        (ax0_1,),
                    )
                )

    return multiple_commreducer


def llvm_intrin_call():
    @Ts.function
    def ctpop(A: T.Tensor((16,), "uint8"), B: T.Tensor((16,), "uint8")) -> None:
        for i in range(0, 16):
            with Ts.sblock("A"):
                vi = Ts.axis.remap(
                    "S",
                    [
                        i,
                    ],
                )
                B[vi] = T.call_llvm_pure_intrin(
                    T.llvm_lookup_intrinsic_id("llvm.ctpop.i8"), A[vi], ty="uint8"
                )

    return ctpop


def float_infinity():
    @Ts.function
    def func(
        placeholder: T.Tensor((1, 512, 768), "float32"), T_isinf: T.Tensor((1, 512, 768), "bool")
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        for i0, i1, i2 in T.grid(1, 512, 768):
            with Ts.sblock("T_isinf"):
                ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                Ts.reads(placeholder[ax0, ax1, ax2])
                Ts.writes(T_isinf[ax0, ax1, ax2])
                T_isinf[ax0, ax1, ax2] = T.fabs(placeholder[ax0, ax1, ax2]) == T.float32(
                    "inf"
                ) and not (T.isnan(placeholder[ax0, ax1, ax2]))

    return func


def multi_env_threads():
    @Ts.function
    def func(A: T.Tensor(128, "float32"), C: T.Tensor(128, "float32")):
        B = Ts.sblock_alloc_tensor([128], dtype="float32")
        for i in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
            B[i] = A[i] + 1.0
        for i in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
            C[i] = B[i] + 2.0

    mod = tvm.s_tir.transform.LowerOpaqueBlock()(
        tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    )
    return mod["main"]


@pytest.mark.parametrize(
    "ir_generator",
    [multiple_commreducer, llvm_intrin_call, multi_env_threads],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_lowered(ir_generator):
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


@pytest.mark.parametrize(
    "ir_generator",
    [abs, float_infinity],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_expressions(ir_generator):
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
def lowered_loop_split(
    A: T.Tensor([128, 128], dtype="float32"), B: T.Tensor([128], dtype="float32")
) -> None:
    reduce_temp0 = Ts.sblock_alloc_tensor(
        [1], dtype="float32", layout=T.TileLayout(T.S[[1] : [1]]), scope="local"
    )
    normal_reduce_temp0 = Ts.sblock_alloc_tensor(
        [1], dtype="float32", layout=T.TileLayout(T.S[[1] : [1]]), scope="local"
    )
    for i in T.serial(0, 128):
        for ki in T.parallel(0, 32, annotations={"thread_binding": "threadIdx.x"}):
            normal_reduce_temp0[0] = T.float32(0)
            for ko in T.serial(0, 4):
                with Ts.sblock("B_normal_reduction"):
                    vi = Ts.axis.S(128, i)
                    vk = Ts.axis.R(128, ko * 32 + ki)
                    Ts.reads([A[vi, vk], normal_reduce_temp0[0]])
                    Ts.writes([normal_reduce_temp0[0]])
                    normal_reduce_temp0[0] = normal_reduce_temp0[0] + A[vi, vk]
            with Ts.sblock("B_cross_thread_reduction"):
                Ts.reads([normal_reduce_temp0[0]])
                Ts.writes([reduce_temp0[0]])
                T.evaluate(
                    T.gpu_thread_allreduce(
                        T.Lambda([T.float32, T.float32], lambda x, y: (x + y,)),
                        (T.float32(0),),
                        (normal_reduce_temp0[0],),
                        True,
                        (reduce_temp0[0],),
                        (ki,),
                    )
                )
            with Ts.sblock("B_write_back"):
                vi = Ts.axis.S(128, i)
                Ts.reads([reduce_temp0[0]])
                Ts.writes([B[vi]])
                B[vi] = reduce_temp0[0]


if __name__ == "__main__":
    tvm_tirx.testing.main()
