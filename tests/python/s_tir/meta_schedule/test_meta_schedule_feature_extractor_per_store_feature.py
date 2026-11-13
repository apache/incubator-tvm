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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
# ruff: noqa: E501, F401
import sys
from collections.abc import Callable

import pytest
from numpy.testing import assert_allclose

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir, te
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

N_FEATURES = 164


@Ts.function
def matmul(
    A: T.Tensor((512, 512), "float32"),
    B: T.Tensor((512, 512), "float32"),
    C: T.Tensor((512, 512), "float32"),
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})
    # body
    # with Ts.sblock("root")
    for i0, i1, i2 in T.grid(512, 512, 512):
        with Ts.sblock("C"):
            i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
            Ts.reads(C[i, j], A[i, k], B[k, j])
            Ts.writes(C[i, j])
            with Ts.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument
# fmt: off

# from tvm.script import tirx as T
@tvm_tirx.script.ir_module
class LayoutTransform:
    @Ts.function
    def main(placeholder: T.Tensor((1, 16, 7, 7, 32), "float32"), placeholder_1: T.Tensor((25088,), "float32"), T_layout_trans: T.Tensor((1, 1, 7, 7, 512), "float32")) -> None:
        # function attr dict
        T.func_attr({"tirx.noalias": True, "global_symbol": "main"})
        # body
        # with Ts.sblock("root")
        for i0_i1_i2_i3_i4_fused in T.parallel(25088, annotations={"tirx.auto_unroll_max_step":64, "tirx.unroll_explicit":1}):
            with Ts.sblock("T_layout_trans_1"):
                ax0 = Ts.axis.spatial(1, 0)
                ax1 = Ts.axis.spatial(1, 0)
                ax2 = Ts.axis.spatial(7, i0_i1_i2_i3_i4_fused // 3584)
                ax3 = Ts.axis.spatial(7, i0_i1_i2_i3_i4_fused % 3584 // 512)
                ax4 = Ts.axis.spatial(512, i0_i1_i2_i3_i4_fused % 512)
                Ts.reads(placeholder[0, (ax4 * 49 + ax2 * 7 + ax3) % 25088 // 1568, (ax2 * 7 + ax3) % 49 // 7, ax3 % 7, (ax4 * 49 + ax2 * 7 + ax3) % 1568 // 49], placeholder_1[(ax4 * 49 + ax2 * 7 + ax3) % 25088])
                Ts.writes(T_layout_trans[ax0, ax1, ax2, ax3, ax4])
                T_layout_trans[ax0, ax1, ax2, ax3, ax4] = T.if_then_else(ax0 < 1 and ax1 * 512 + ax4 < 512 and ax2 < 7 and ax3 < 7, T.Select(T.float32(0) < T.if_then_else(T.LT(0, 1) and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 < 512 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 49 // 7 < 7 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 7 < 7, placeholder[0, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 // 32, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 49 // 7, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 7, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 % 32], T.float32(0)), T.if_then_else(T.LT(0, 1) and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 < 512 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 49 // 7 < 7 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 7 < 7, placeholder[0, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 // 32, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 49 // 7, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 7, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 % 32], T.float32(0)), T.if_then_else(T.LT(0, 1) and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 < 512 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 49 // 7 < 7 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 7 < 7, placeholder[0, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 // 32, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 49 // 7, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 7, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 % 32], T.float32(0)) * placeholder_1[((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088]), T.float32(0))

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def _make_context(target) -> ms.TuneContext:
    return ms.TuneContext(
        target=target,
        num_threads=1,
    )


def _make_candidate(f_sch: Callable[[], s_tir.Schedule]) -> ms.MeasureCandidate:
    return ms.MeasureCandidate(sch=f_sch(), args_info=[])


def _feature_names(  # pylint: disable=invalid-name
    tensors_per_store: int = 5,
    arith_intensity_curve_num_samples: int = 10,
) -> list[str]:
    result = [
        "float_mad",
        "float_addsub",
        "float_mul",
        "float_divmod",
        "float_cmp",
        "float_mathfunc",
        "float_otherfunc",
        "int_mad",
        "int_addsub",
        "int_mul",
        "int_divmod",
        "int_cmp",
        "int_mathfunc",
        "int_otherfunc",
        "bool_op",
        "select_op",
        "vec_num",
        "vec_prod",
        "vec_len",
        "vec_type.kPosNone",
        "vec_type.kPosInnerSpatial",
        "vec_type.kPosMiddleSpatial",
        "vec_type.kPosOuterSpatial",
        "vec_type.kPosInnerReduce",
        "vec_type.kPosMiddleReduce",
        "vec_type.kPosOuterReduce",
        "vec_type.kPosMixed",
        "unroll_num",
        "unroll_prod",
        "unroll_len",
        "unroll_type.kPosNone",
        "unroll_type.kPosInnerSpatial",
        "unroll_type.kPosMiddleSpatial",
        "unroll_type.kPosOuterSpatial",
        "unroll_type.kPosInnerReduce",
        "unroll_type.kPosMiddleReduce",
        "unroll_type.kPosOuterReduce",
        "unroll_type.kPosMixed",
        "parallel_num",
        "parallel_prod",
        "parallel_len",
        "parallel_type.kPosNone",
        "parallel_type.kPosInnerSpatial",
        "parallel_type.kPosMiddleSpatial",
        "parallel_type.kPosOuterSpatial",
        "parallel_type.kPosInnerReduce",
        "parallel_type.kPosMiddleReduce",
        "parallel_type.kPosOuterReduce",
        "parallel_type.kPosMixed",
        "is_gpu",
        "blockIdx_x_len",
        "blockIdx_y_len",
        "blockIdx_z_len",
        "threadIdx_x_len",
        "threadIdx_y_len",
        "threadIdx_z_len",
        "vthread_len",
    ]
    for i in range(tensors_per_store):
        result.extend(
            f"B{i}.{s}"
            for s in [
                "acc_type.kRead",
                "acc_type.kWrite",
                "acc_type.kReadWrite",
                "bytes",
                "unique_bytes",
                "lines",
                "unique_lines",
                "reuse_type.kLoopMultipleRead",
                "reuse_type.kSerialMultipleReadWrite",
                "reuse_type.kNoReuse",
                "reuse_dis_iter",
                "reuse_dis_bytes",
                "reuse_ct",
                "bytes_d_reuse_ct",
                "unique_bytes_d_reuse_ct",
                "lines_d_reuse_ct",
                "unique_lines_d_reuse_ct",
                "stride",
            ]
        )
    result.extend(f"arith_intensity_curve_{i}" for i in range(arith_intensity_curve_num_samples))
    result.extend(
        [
            "alloc_size",
            "alloc_prod",
            "alloc_outer_prod",
            "alloc_inner_prod",
            "outer_prod",
            "num_loops",
            "tirx.auto_unroll_max_step",
        ]
    )
    # 57 + 18 * 5 + 10 + 4 + 3
    assert len(result) == N_FEATURES
    return result


def _zip_feature(feature, names):
    assert feature.ndim == 1
    assert feature.shape[0] == N_FEATURES
    assert len(names) == N_FEATURES
    return list(zip(names, feature))


def _print_feature(feature, st, ed):  # pylint: disable=invalid-name
    named_feature = _zip_feature(feature, _feature_names())
    for k, v in named_feature[st:ed]:
        print("\t", k, v)


def test_cpu_matmul():
    def _create_schedule():
        func = matmul
        sch = tvm.s_tir.Schedule(func, debug_mask="all")
        block = sch.get_sblock("C")
        i, j, k = sch.get_loops(block)
        i_o, i_i = sch.split(i, factors=[None, 16])  # outer: 32
        j_o, j_i = sch.split(j, factors=[None, 8])  # outer: 64
        sch.reorder(i_o, j_o, k, j_i, i_i)
        sch.vectorize(j_i)
        sch.parallel(i_o)
        sch.parallel(j_o)
        sch.unroll(k)
        return sch

    extractor = ms.feature_extractor.PerStoreFeature()
    (feature,) = extractor.extract_from(
        _make_context(tvm_tirx.target.Target("llvm")),
        candidates=[_make_candidate(_create_schedule)],
    )
    feature = feature.numpy()
    assert feature.shape == (1, N_FEATURES)
    f = feature[0]
    # Group 1.1: arith
    assert_allclose(
        actual=f[0:16],
        desired=[
            # float math ops
            0,
            27,
            27,
            0,
            0,
            0,
            0,
            # int math ops
            0,
            29,
            29,
            0,
            0,
            0,
            0,
            # bool/select ops
            0,
            0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.2: vectorize
    assert_allclose(
        actual=f[16:27],
        desired=[1.0, 3.169924, 3.169924, 0, 0, 0, 0, 0, 0, 0, 1],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.3: unroll
    assert_allclose(
        actual=f[27:38],
        desired=[1.0, 9.002815, 9.002815, 0, 0, 0, 0, 0, 0, 0, 1],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.4: parallel
    assert_allclose(
        actual=f[38:49],
        desired=[1.58496, 11.0007, 6.022368, 0, 0, 0, 0, 0, 0, 0, 1],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.5: is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
    assert_allclose(
        actual=f[49:57],
        desired=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.1: Tensor A
    assert_allclose(
        actual=f[57:75],
        desired=[
            1,
            0,
            0,
            29,
            20,
            27,
            14,
            1,
            0,
            0,
            4.087463,
            7.0552826,
            3.169925,
            26,
            17,
            24,
            11.0007038,
            9.002815,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.2: Tensor C
    assert_allclose(
        actual=f[75:93],
        desired=[
            0.0,
            0.0,
            1.0,
            29.0,
            20.000001907348633,
            27.0,
            14.00008773803711,
            1.0,
            0.0,
            0.0,
            7.011227130889893,
            9.250298500061035,
            9.002815246582031,
            20.000001907348633,
            11.000703811645508,
            18.0000057220459,
            5.044394016265869,
            9.002815246582031,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3: Tensor B
    assert_allclose(
        actual=f[93:111],
        desired=[
            1.0,
            0.0,
            0.0,
            29.0,
            20.000001907348633,
            19.000001907348633,
            14.00008773803711,
            1.0,
            0.0,
            0.0,
            1.0,
            3.700439691543579,
            4.087462902069092,
            25.0,
            16.000022888183594,
            15.000043869018555,
            10.001408194392809,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.4: Dummy padding
    assert_allclose(
        actual=f[111:129],
        desired=[0.0] * 18,
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.5: Dummy padding
    assert_allclose(
        actual=f[129:147],
        desired=[0.0] * 18,
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 3: Arithmetic intensity
    # arithmetic intensity = flops/bytes touched = 2*512*512*512/(3 * 4 * 512*512)
    #                             add and multiply ^     3 arrays ^   ^ 4 bytes per f32
    # = 85.3 but log2 is used so values should be around 6.4
    assert_allclose(
        actual=f[147:157],
        desired=[
            3.812599,
            4.464822,
            4.912349,
            5.253426,
            5.529086,
            5.76043,
            5.959752,
            6.134849,
            6.290977,
            6.431846,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 4 & 5
    assert_allclose(
        actual=f[157:164],
        desired=[
            20.000001907348633,
            18.0000057220459,
            1.0,
            27.0,
            27.0,
            2.5849626064300537,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )


def test_cpu_fusion():
    # pylint: disable=all
    @Ts.function
    def func(
        A: T.Tensor([64, 32], dtype="float32"),
        B: T.Tensor([64, 32], dtype="float32"),
        C: T.Tensor([64, 32], dtype="float32"),
    ) -> None:
        for i, j in T.grid(64, 32):  # type: ignore
            with Ts.sblock():
                Ts.reads([A[i, j], B[i, j]])  # type: ignore
                Ts.writes([B[i, j], C[i, j]])  # type: ignore
                with Ts.sblock("B"):
                    Ts.reads([A[i, j]])  # type: ignore
                    Ts.writes([B[i, j]])  # type: ignore
                    B[i, j] = A[i, j]  # type: ignore
                with Ts.sblock("C"):
                    Ts.reads([B[i, j]])  # type: ignore
                    Ts.writes([C[i, j]])  # type: ignore
                    C[i, j] = B[i, j]  # type: ignore

    # pylint: enable=all

    def _create_schedule():
        return s_tir.Schedule(func, debug_mask="all")

    extractor = ms.feature_extractor.PerStoreFeature()
    (feature,) = extractor.extract_from(
        _make_context(tvm_tirx.target.Target("llvm")),
        candidates=[_make_candidate(_create_schedule)],
    )
    feature = feature.numpy()
    assert feature.shape == (2, N_FEATURES)
    ## Features for TensorStore(B)
    f = feature[0]
    # Group 1.1: arith
    assert_allclose(
        actual=f[0:16],
        desired=[0.0] * 16,
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.2: vectorize
    assert_allclose(
        actual=f[16:27],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.3: unroll
    assert_allclose(
        actual=f[27:38],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.4: parallel
    assert_allclose(
        actual=f[38:49],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.5: is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
    assert_allclose(
        actual=f[49:57],
        desired=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.1: Tensor A
    assert_allclose(
        actual=f[57:75],
        desired=[
            1.0,
            0.0,
            0.0,
            13.000176429748535,
            13.000176429748535,
            7.011227130889893,
            7.011227130889893,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            14.00008773803711,
            14.00008773803711,
            8.005624771118164,
            8.005624771118164,
            1.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.2: Tensor B
    assert_allclose(
        actual=f[75:93],
        desired=[
            0.0,
            1.0,
            0.0,
            13.000176429748535,
            13.000176429748535,
            7.011227130889893,
            7.011227130889893,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            14.00008773803711,
            14.00008773803711,
            8.005624771118164,
            8.005624771118164,
            1.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3: Dummy padding
    assert_allclose(
        actual=f[93:111],
        desired=[0.0] * 18,
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.4: Dummy padding
    assert_allclose(
        actual=f[111:129],
        desired=[0.0] * 18,
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.5: Dummy padding
    assert_allclose(
        actual=f[129:147],
        desired=[0.0] * 18,
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 3: Arithmetic intensity
    assert_allclose(
        actual=f[147:157],
        desired=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 4 & 5
    assert_allclose(
        actual=f[157:164],
        desired=[
            13.000176,
            11.000703811645508,
            1.0,
            11.000703811645508,
            11.000703811645508,
            1.5849624872207642,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    ## Features for TensorStore(C)
    f = feature[1]
    # Group 1.1: arith
    assert_allclose(
        actual=f[0:16],
        desired=[0.0] * 16,
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.2: vectorize
    assert_allclose(
        actual=f[16:27],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.3: unroll
    assert_allclose(
        actual=f[27:38],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.4: parallel
    assert_allclose(
        actual=f[38:49],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.5: is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
    assert_allclose(
        actual=f[49:57],
        desired=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.1: Tensor B
    assert_allclose(
        actual=f[57:75],
        desired=[
            1.0,
            0.0,
            0.0,
            13.000176429748535,
            13.000176429748535,
            7.011227130889893,
            7.011227130889893,
            0.0,
            1.0,
            0.0,
            1.0,
            4.087462902069092,
            1.0,
            13.000176429748535,
            13.000176429748535,
            7.011227130889893,
            7.011227130889893,
            1.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.2: Tensor C
    assert_allclose(
        actual=f[75:93],
        desired=[
            0.0,
            1.0,
            0.0,
            13.000176429748535,
            13.000176429748535,
            7.011227130889893,
            7.011227130889893,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            14.00008773803711,
            14.00008773803711,
            8.005624771118164,
            8.005624771118164,
            1.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3: Dummy padding
    assert_allclose(
        actual=f[93:111],
        desired=[0.0] * 18,
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.4: Dummy padding
    assert_allclose(
        actual=f[111:129],
        desired=[0.0] * 18,
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.5: Dummy padding
    assert_allclose(
        actual=f[129:147],
        desired=[0.0] * 18,
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 3: Arithmetic intensity
    assert_allclose(
        actual=f[147:157],
        desired=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 4 & 5
    assert_allclose(
        actual=f[157:164],
        desired=[
            13.000176429748535,
            11.000703811645508,
            1.0,
            11.000703811645508,
            11.000703811645508,
            1.5849624872207642,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )


def test_empty_feature():
    @Ts.function
    def full(T_full: T.Tensor((T.int64(2), T.int64(3)), "float32")):
        for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
            with Ts.sblock("T_full"):
                v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads()
                Ts.writes(T_full[v_ax0, v_ax1])
                T_full[v_ax0, v_ax1] = T.float32(1)

    def _create_schedule():
        return s_tir.Schedule(full, debug_mask="all")

    extractor = ms.feature_extractor.PerStoreFeature()
    (feature,) = extractor.extract_from(
        _make_context(tvm_tirx.target.Target("llvm")),
        candidates=[_make_candidate(_create_schedule)],
    )
    feature = feature.numpy()
    assert feature.shape == (0, N_FEATURES)


def test_gpu():
    def _create_schedule():
        func = matmul
        sch = tvm.s_tir.Schedule(func, debug_mask="all")
        c = sch.get_sblock("C")
        c_local = sch.cache_write(c, 0, "local")
        i, j, k = sch.get_loops(c)
        # pylint: disable=invalid-name
        i0, i1, i2, i3, i4 = sch.split(i, factors=[None, 1, 16, 32, 1])  # outer: 1
        j0, j1, j2, j3, j4 = sch.split(j, factors=[None, 4, 1, 1, 16])  # outer: 8
        k0, k1, k2 = sch.split(k, factors=[None, 1, 2])  # outer: 256
        # pylint: enable=invalid-name
        # fmt: off
        sch.reorder(
            i0, j0,  # S
            i1, j1,  # S
            i2, j2,  # S
            k0,      # R
            k1,      # R
            i3, j3,  # S
            k2,      # R
            i4, j4,  # S
        )
        # fmt: on
        # thread binding
        i0_j0 = sch.fuse(i0, j0)
        i1_j1 = sch.fuse(i1, j1)
        i2_j2 = sch.fuse(i2, j2)
        sch.bind(i0_j0, "blockIdx.x")
        sch.bind(i1_j1, "vthread.x")
        sch.bind(i2_j2, "threadIdx.x")
        # fusion
        sch.reverse_compute_at(c_local, i2_j2)
        # cache read 'A'
        a_shared = sch.cache_read(c, 1, "shared")
        sch.compute_at(a_shared, k0)
        _, _, _, _, a_i, a_j = sch.get_loops(a_shared)
        a_ij = sch.fuse(a_i, a_j)
        _, a_j = sch.split(a_ij, factors=[None, 16])  # outer: 64
        sch.bind(a_j, "threadIdx.x")
        # cache read 'B'
        b_shared = sch.cache_read(c, 2, "shared")
        sch.compute_at(b_shared, k0)
        _, _, _, _, b_i, b_j = sch.get_loops(b_shared)
        b_ij = sch.fuse(b_i, b_j)
        _, b_j = sch.split(b_ij, factors=[None, 16])  # outer: 8
        sch.bind(b_j, "threadIdx.x")
        # auto unroll
        sch.annotate(i0_j0, "tirx.auto_unroll_max_step", tvm_tirx.ir.IntImm("int32", 1024))
        sch.annotate(i0_j0, "tirx.unroll_explicit", tvm_tirx.ir.IntImm("int32", 1))
        return sch

    extractor = ms.feature_extractor.PerStoreFeature()
    (feature,) = extractor.extract_from(
        _make_context(tvm_tirx.target.Target("cuda")),
        candidates=[_make_candidate(_create_schedule)],
    )
    feature = feature.numpy()
    assert feature.shape == (4, N_FEATURES)
    ### Check feature[0]: TensorStore(A_shared) <= A[...]
    f = feature[0]
    # Group 1.1: arith
    assert_allclose(
        actual=f[0:16],
        desired=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            24.000000085991324,
            24.000000085991324,
            24.000000085991324,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.2: vectorize
    assert_allclose(
        actual=f[16:27],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.3: unroll
    assert_allclose(
        actual=f[27:38],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.4: parallel
    assert_allclose(
        actual=f[38:49],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.5: is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
    assert_allclose(
        actual=f[49:57],
        desired=[1.0, 3.169925001442312, 1.0, 1.0, 4.087462841250339, 1.0, 1.0, 2.321928094887362],
        rtol=1e-5,
        atol=1e-5,
    )
    # Region estimates replay each loop-range context, without retaining the
    # temporary constant bindings used while relaxing inner loops.  In particular,
    # the full 512x512 float32 input contributes 2**20 unique bytes.
    # Group 2.1: Tensor A
    assert_allclose(
        actual=f[57:75],
        desired=[
            1.0,
            0.0,
            0.0,
            25.00000004,
            20.000001375860553,
            23.00000017,
            14.000088052430122,
            1.0,
            0.0,
            0.0,
            18.0000055,
            20.00562591970089,
            2.32192809,
            23.00000017,
            18.00000550343433,
            21.00000069,
            12.0003521774803,
            12.00035218,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.2: Tensor A.shared
    assert_allclose(
        actual=f[75:93],
        desired=[
            0.0,
            1.0,
            0.0,
            25.00000004,
            12.0003521774803,
            23.00000017,
            9.002815015607053,
            1.0,
            0.0,
            0.0,
            6.02236781,
            11.98049664,
            8.00562455,
            17.00001101,
            4.087462841250339,
            15.00004403,
            1.584962500721156,
            4.08746284,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3: Dummy padding
    assert_allclose(
        actual=f[93:111],
        desired=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.4: Dummy padding
    assert_allclose(
        actual=f[111:129],
        desired=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.5: Dummy padding
    assert_allclose(
        actual=f[129:147],
        desired=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 3: Arithmetic intensity
    assert_allclose(
        actual=f[147:157],
        desired=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 4 & 5
    assert_allclose(
        actual=f[157:164],
        desired=[
            12.0003521774803,
            27.000000010748916,
            17.000011006847668,
            6.022367813028454,
            23.00000017198264,
            2.584962500721156,
            10.001408,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    ### Check feature[1]: TensorStore(B_shared) <= B[...]
    f = feature[1]
    # Group 1.1: arith
    assert_allclose(
        actual=f[0:16],
        desired=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            21.584962959341485,
            21.584962959341485,
            21.000000687930438,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.2: vectorize
    assert_allclose(
        actual=f[16:27],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.3: unroll
    assert_allclose(
        actual=f[27:38],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.4: parallel
    assert_allclose(
        actual=f[38:49],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.5: is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
    assert_allclose(
        actual=f[49:57],
        desired=[1.0, 3.169925001442312, 1.0, 1.0, 4.087462841250339, 1.0, 1.0, 2.321928094887362],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.1: Tensor B
    assert_allclose(
        actual=f[57:75],
        desired=[
            1.0,
            0.0,
            0.0,
            22.00000034,
            20.000001375860553,
            20.00000138,
            14.000088052430122,
            1.0,
            0.0,
            0.0,
            15.00004403,
            20.17555076886471,
            2.32192809,
            20.00000138,
            18.00000550343433,
            18.0000055,
            12.0003521774803,
            4.08746284,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.2: Tensor B.shared
    assert_allclose(
        actual=f[75:93],
        desired=[
            0.0,
            1.0,
            0.0,
            22.00000034,
            9.002815015607053,
            20.00000138,
            3.169925001442312,
            1.0,
            0.0,
            0.0,
            3.169925001442312,
            9.61654884377899,
            8.005624549193879,
            14.000088052430122,
            1.584962500721156,
            12.00035218,
            0.044394119358453436,
            4.087462841250339,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3: Dummy padding
    assert_allclose(
        actual=f[93:111],
        desired=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.4: Dummy padding
    assert_allclose(
        actual=f[111:129],
        desired=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.5: Dummy padding
    assert_allclose(
        actual=f[129:147],
        desired=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 3: Arithmetic intensity
    assert_allclose(
        actual=f[147:157],
        desired=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 4 & 5
    assert_allclose(
        actual=f[157:164],
        desired=[
            9.002815015607053,
            24.000000085991324,
            17.000011006847668,
            3.169925001442312,
            20.000001375860553,
            2.584962500721156,
            10.001408,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    ### Check feature[2]: TensorStore(C_local) <= C_local[...] + A_shared[...] * B_shared[...]
    f = feature[2]
    # Group 1.1: arith
    assert_allclose(
        actual=f[0:16],
        desired=[
            0.0,
            27.000000010748916,
            27.000000010748916,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            28.000000005374456,
            28.000000005374456,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.2: vectorize
    assert_allclose(
        actual=f[16:27],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.3: unroll
    assert_allclose(
        actual=f[27:38],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.4: parallel
    assert_allclose(
        actual=f[38:49],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.5: is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
    assert_allclose(
        actual=f[49:57],
        desired=[1.0, 3.169925001442312, 1.0, 1.0, 4.087462841250339, 1.0, 1.0, 2.321928094887362],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.1: Tensor B.shared
    assert_allclose(
        actual=f[57:75],
        desired=[
            1.0,
            0.0,
            0.0,
            29.00000000268723,
            9.002815015607053,
            23.00000017198264,
            3.169925001442312,
            1.0,
            0.0,
            0.0,
            5.044394119358453,
            7.651051691178929,
            5.044394119358453,
            24.000000085991324,
            4.087462841250339,
            18.00000550343433,
            0.32192809488736235,
            1.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.2: Tensor C.local
    assert_allclose(
        actual=f[75:93],
        desired=[
            0.0,
            0.0,
            1.0,
            29.00000000268723,
            11.000704269011246,
            23.00000017198264,
            5.044394119358453,
            1.0,
            0.0,
            0.0,
            4.087462841250339,
            7.05528243550119,
            1.584962500721156,
            28.000000005374456,
            10.001408194392809,
            22.00000034396526,
            4.087462841250339,
            1.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3: Tensor A.shared
    assert_allclose(
        actual=f[93:111],
        desired=[
            1.0,
            0.0,
            0.0,
            29.00000000268723,
            12.0003521774803,
            19.00000275171979,
            9.002815015607053,
            1.0,
            0.0,
            0.0,
            1.0,
            3.700439718141092,
            4.087462841250339,
            25.000000042995662,
            8.005624549193879,
            15.000044026886828,
            5.044394119358453,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.4: Dummy padding
    assert_allclose(
        actual=f[111:129],
        desired=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.5: Dummy padding
    assert_allclose(
        actual=f[129:147],
        desired=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 3: Arithmetic intensity
    # Arithmetic intensity is high here because of repeated use of a shared
    # tensor. Multiple accesses to the same memory location are counted as a
    # single byte, skewing these numbers towards higher intensity.
    assert_allclose(
        actual=f[147:157],
        desired=[
            11.98533,
            12.977811,
            13.562714,
            13.977722,
            14.299632,
            14.562654,
            14.785038,
            14.977677,
            15.147597,
            15.299596,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 4 & 5
    assert_allclose(
        actual=f[157:164],
        desired=[
            11.000704269011246,
            18.00000550343433,
            9.002815015607053,
            18.00000550343433,
            27.000000010748916,
            3.0,
            10.001408,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    ### Check feature[3]: TensorStore(C) <= C_local[...]
    f = feature[3]
    # Group 1.1: arith
    assert_allclose(
        actual=f[0:16],
        desired=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.2: vectorize
    assert_allclose(
        actual=f[16:27],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.3: unroll
    assert_allclose(
        actual=f[27:38],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.4: parallel
    assert_allclose(
        actual=f[38:49],
        desired=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.5: is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
    assert_allclose(
        actual=f[49:57],
        desired=[1.0, 3.169925001442312, 1.0, 1.0, 4.087462841250339, 1.0, 1.0, 2.321928094887362],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.1: Tensor C
    assert_allclose(
        actual=f[57:75],
        desired=[
            0.0,
            1.0,
            0.0,
            20.000001375860553,
            20.000001375860553,
            14.000088052430122,
            14.000088052430122,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            21.000000687930438,
            21.000000687930438,
            15.000044026886828,
            15.000044026886828,
            1.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.2: Tensor C.local
    assert_allclose(
        actual=f[75:93],
        desired=[
            1.0,
            0.0,
            0.0,
            20.000001375860553,
            11.000704269011246,
            14.000088052430122,
            5.044394119358453,
            1.0,
            0.0,
            0.0,
            9.002815015607053,
            12.0003521774803,
            4.087462841250339,
            16.00002201361136,
            7.011227255423254,
            10.001408194392809,
            1.584962500721156,
            1.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3: Dummy padding
    assert_allclose(
        actual=f[93:111],
        desired=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.4: Dummy padding
    assert_allclose(
        actual=f[111:129],
        desired=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.5: Dummy padding
    assert_allclose(
        actual=f[129:147],
        desired=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 3: Arithmetic intensity
    assert_allclose(
        actual=f[147:157],
        desired=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 4 & 5
    assert_allclose(
        actual=f[157:164],
        desired=[
            20.000001375860553,
            18.00000550343433,
            1.0,
            18.00000550343433,
            18.00000550343433,
            2.584962500721156,
            10.001408,
        ],
        rtol=1e-5,
        atol=1e-5,
    )


def test_cpu_layout_transform():
    extractor = ms.feature_extractor.PerStoreFeature()
    (feature,) = extractor.extract_from(
        _make_context(tvm_tirx.target.Target("llvm")),
        candidates=[_make_candidate(lambda: s_tir.Schedule(LayoutTransform))],
    )


@Ts.function
def negative_extent(A: T.Tensor((1,), "float32")):
    for j in range(0, -1):
        A[j] = A[j] + 1.0


def test_negative_extent():
    extractor = ms.feature_extractor.PerStoreFeature()
    (features,) = extractor.extract_from(
        _make_context(tvm_tirx.target.Target("llvm")),
        candidates=[_make_candidate(lambda: s_tir.Schedule(negative_extent))],
    )
    named_features = dict(zip(_feature_names(), list(features.numpy()[0, :])))
    assert named_features["B0.unique_bytes"] == 0


if __name__ == "__main__":
    tvm_tirx.testing.main()
