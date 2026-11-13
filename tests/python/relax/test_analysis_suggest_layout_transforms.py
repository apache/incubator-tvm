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
# ruff: noqa: E741

import pytest

import tvm
import tvm_tirx.testing
from tvm import relax, s_tir
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T


def apply_transformations(func, suggested_transfoms, print_transformation=False):
    sch = tvm.s_tir.Schedule(func)
    for block, per_block_transformations in suggested_transfoms.items():
        blockrv = sch.get_sblock(block.name_hint)
        for obj, index_map in per_block_transformations.items():
            if isinstance(obj, s_tir.SBlock):
                block_name = obj.name_hint
                if print_transformation:
                    print("Block transformation: ", block_name, " :: ", index_map)
                sch.transform_block_layout(block_name, index_map)
            else:
                assert tirx.is_tensor_var(obj)
                tensor = obj
                if print_transformation:
                    print("Tensor transformation: ", tensor, " :: ", index_map)
                sch.transform_layout(blockrv, tensor, index_map)
    return sch.mod["main"]


def test_nested_blocks():
    @Ts.function(private=True)
    def nested_block(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        relu: T.Tensor((32, 64, 224, 224), "float32"),
    ):
        for i, j in T.grid(32, 64):
            with Ts.sblock("outer"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                Ts.reads(arg[v_i, v_j, 0:224, 0:224])
                Ts.writes(relu[v_i, v_j, 0:224, 0:224])
                for k, l in T.grid(224, 224):
                    with Ts.sblock("inner"):
                        v_k, v_l = Ts.axis.remap("SS", [k, l])
                        Ts.reads(arg[v_i, v_j, v_k, v_l])
                        Ts.writes(relu[v_i, v_j, v_k, v_l])
                        relu[v_i, v_j, v_k, v_l] = T.max(arg[v_i, v_j, v_k, v_l], T.float32(0))

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=nested_block, write_tensor_transforms=[lambda n, c, h, w: (n, h, w, c)]
    )
    # no suggestions for nested block.
    assert len(suggested_transforms.items()) == 0


def test_mismatch_transformations_and_num_params():
    @Ts.function(private=True)
    def elemwise(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        relu: T.Tensor((32, 64, 224, 224), "float32"),
    ):
        for i0, i1, i2, i3 in T.grid(32, 64, 224, 224):
            with Ts.sblock("compute"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(arg[v_i0, v_i1, v_i2, v_i3])
                Ts.writes(relu[v_i0, v_i1, v_i2, v_i3])
                relu[v_i0, v_i1, v_i2, v_i3] = T.max(arg[v_i0, v_i1, v_i2, v_i3], T.float32(0))

    with pytest.raises(RuntimeError, match="Incompatible Function and write_transformations"):
        _ = relax.analysis.suggest_layout_transforms(
            func=elemwise,
            write_tensor_transforms=[
                lambda n, c, h, w: (n, h, w, c),
                lambda n, c, h, w: (n, h, w, c),
                lambda n, c, h, w: (n, h, w, c),
            ],
        )


def test_empty_write_transformations():
    @Ts.function(private=True)
    def elemwise(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        relu: T.Tensor((32, 64, 224, 224), "float32"),
    ):
        for i0, i1, i2, i3 in T.grid(32, 64, 224, 224):
            with Ts.sblock("compute"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(arg[v_i0, v_i1, v_i2, v_i3])
                Ts.writes(relu[v_i0, v_i1, v_i2, v_i3])
                relu[v_i0, v_i1, v_i2, v_i3] = T.max(arg[v_i0, v_i1, v_i2, v_i3], T.float32(0))

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=elemwise, write_tensor_transforms=[]
    )
    assert len(suggested_transforms.items()) == 0


def test_non_bijective_block_transform():
    @Ts.function(private=True)
    def before(
        arg: T.Tensor((32, 64), "float32"),
        output: T.Tensor((32, 64), "float32"),
    ):
        for ax0, ax1 in T.grid(32, 64):
            with Ts.sblock("compute"):
                v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(arg[v_ax0, v_ax1])
                Ts.writes(output[v_ax0, v_ax1])
                output[v_ax0, v_ax1] = arg[v_ax0, v_ax1]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_tensor_transforms=[lambda n, c: (n, c // 5, c % 5)]
    )
    assert len(suggested_transforms.items()) == 0


def test_non_affine_access():
    @Ts.function(private=True)
    def before(
        arg: T.Tensor((32, 64), "float32"),
        output: T.Tensor((32 * 64, 10), "float32"),
    ):
        for ax0, ax1, ax2 in T.grid(32, 64, 10):
            with Ts.sblock("compute"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                Ts.reads(arg[v_ax0, v_ax1])
                Ts.writes(output[v_ax0 * v_ax1, v_ax2])
                output[v_ax0 * v_ax1, v_ax2] = arg[v_ax0, v_ax1]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_tensor_transforms=[lambda a, b: (b, a)]
    )
    assert len(suggested_transforms.items()) == 0


def test_unsupported_write_spatial_layout():
    @Ts.function(private=True)
    def before(
        arg: T.Tensor((4, 4), "float32"),
        output: T.Tensor((16), "float32"),
    ):
        for ax0, ax1 in T.grid(4, 4):
            with Ts.sblock("flatten"):
                v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(arg[v_ax0, v_ax1])
                Ts.writes(output[v_ax0 * 4 + v_ax1])
                output[v_ax0 * 4 + v_ax1] = arg[v_ax0, v_ax1]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_tensor_transforms=[lambda a: (a // 4, a % 4)]
    )
    assert len(suggested_transforms.items()) == 0


def test_unpacked_iter_used_in_read_access():
    @Ts.function(private=True)
    def before(
        arg: T.Tensor((8, 4), "float32"),
        output: T.Tensor((4, 8), "float32"),
    ):
        for ax0, ax1, ax2 in T.grid(4, 8, 4):
            with Ts.sblock("compute"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                Ts.reads(arg[v_ax1, v_ax2])
                Ts.writes(output[v_ax0, v_ax1])
                output[v_ax0, v_ax1] = arg[v_ax1, v_ax2]

    @Ts.function(private=True)
    def expected(
        arg: T.Tensor((8, 4), "float32"),
        output: T.Tensor((32), "float32"),
    ):
        for ax0, ax2 in T.grid(32, 4):
            with Ts.sblock("compute"):
                v_ax0, v_ax2 = Ts.axis.remap("SS", [ax0, ax2])
                Ts.reads(arg[v_ax0 % 8, v_ax2])
                Ts.writes(output[v_ax0])
                output[v_ax0] = arg[v_ax0 % 8, v_ax2]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_tensor_transforms=[lambda a, b: a * 8 + b]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_invalid_index_map():
    @Ts.function(private=True)
    def elemwise(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        relu: T.Tensor((32, 64, 224, 224), "float32"),
    ):
        for i0, i1, i2, i3 in T.grid(32, 64, 224, 224):
            with Ts.sblock("compute"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(arg[v_i0, v_i1, v_i2, v_i3])
                Ts.writes(relu[v_i0, v_i1, v_i2, v_i3])
                relu[v_i0, v_i1, v_i2, v_i3] = T.max(arg[v_i0, v_i1, v_i2, v_i3], T.float32(0))

    with pytest.raises(RuntimeError, match="Mismatch between output tensor shape and index map"):
        _ = relax.analysis.suggest_layout_transforms(
            func=elemwise, write_tensor_transforms=[lambda n, h, w: (n, w, h)]
        )
    with pytest.raises(AssertionError):
        _ = relax.analysis.suggest_layout_transforms(func=elemwise, write_tensor_transforms=[2])


def test_SRSR_block():
    @Ts.function(private=True)
    def before(
        arg: T.Tensor((32, 224, 64, 224), "float32"),
        sum: T.Tensor((32, 64), "float32"),
    ):
        for ax0, k2, ax1, k3 in T.grid(32, 224, 64, 224):
            with Ts.sblock("rxplaceholder_red"):
                v_ax0, v_k2, v_ax1, v_k3 = Ts.axis.remap("SRSR", [ax0, k2, ax1, k3])
                Ts.reads(arg[v_ax0, v_ax1, v_k2, v_k3])
                Ts.writes(sum[v_ax0, v_ax1])
                with Ts.init():
                    sum[v_ax0, v_ax1] = T.float32(0)
                sum[v_ax0, v_ax1] = sum[v_ax0, v_ax1] + arg[v_ax0, v_k2, v_ax1, v_k3]

    @Ts.function(private=True)
    def expected(
        arg: T.Tensor((32, 224, 16, 224, 4), "float32"),
        sum: T.Tensor((32, 16, 4), "float32"),
    ):
        for ax0, ax1, ax2, ax3, ax4 in T.grid(32, 224, 16, 224, 4):
            with Ts.sblock("rxplaceholder_red"):
                v0, v1, v2, v3, v4 = Ts.axis.remap("SRSRS", [ax0, ax1, ax2, ax3, ax4])
                Ts.reads(arg[v0, v1, v2, v3, v4])
                Ts.writes(sum[v0, v2, v4])
                with Ts.init():
                    sum[v0, v2, v4] = T.float32(0)
                sum[v0, v2, v4] = sum[v0, v2, v4] + arg[v0, v1, v2, v3, v4]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_tensor_transforms=[lambda n, c: (n, c // 4, c % 4)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_op_elemwise_symbolic():
    N = T.dynamic("N", "int64")
    C = T.dynamic("C", "int64")
    H = T.dynamic("H", "int64")
    W = T.dynamic("W", "int64")

    @Ts.function(private=True)
    def before(Arg: T.Tensor((N, C, H, W), "float32"), Relu: T.Tensor((N, C, H, W), "float32")):
        for i0, i1, i2, i3 in T.grid(N, C, H, W):
            with Ts.sblock("compute"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(Arg[v_i0, v_i1, v_i2, v_i3])
                Ts.writes(Relu[v_i0, v_i1, v_i2, v_i3])
                Relu[v_i0, v_i1, v_i2, v_i3] = T.max(Arg[v_i0, v_i1, v_i2, v_i3], T.float32(0))

    N = T.dynamic("N", "int64")
    C = T.dynamic("C", "int64")
    H = T.dynamic("H", "int64")
    W = T.dynamic("W", "int64")

    @Ts.function(private=True)
    def expected(Arg: T.Tensor((N, H, W, C), "float32"), Relu: T.Tensor((N, H, W, C), "float32")):
        # with Ts.sblock("root"):
        for ax0, ax1, ax2, ax3 in T.grid(N, H, W, C):
            with Ts.sblock("compute"):
                v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(Arg[v0, v1, v2, v3])
                Ts.writes(Relu[v0, v1, v2, v3])
                Relu[v0, v1, v2, v3] = T.max(Arg[v0, v1, v2, v3], T.float32(0))

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_tensor_transforms=[lambda n, c, h, w: (n, h, w, c)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_op_elemwise():
    @Ts.function(private=True)
    def before(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        relu: T.Tensor((32, 64, 224, 224), "float32"),
    ):
        for i0, i1, i2, i3 in T.grid(32, 64, 224, 224):
            with Ts.sblock("compute"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(arg[v_i0, v_i1, v_i2, v_i3])
                Ts.writes(relu[v_i0, v_i1, v_i2, v_i3])
                relu[v_i0, v_i1, v_i2, v_i3] = T.max(arg[v_i0, v_i1, v_i2, v_i3], T.float32(0))

    @Ts.function(private=True)
    def expected(
        arg: T.Tensor((32, 224, 224, 64), "float32"),
        relu: T.Tensor((32, 224, 224, 64), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 224, 224, 64):
            with Ts.sblock("compute"):
                v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(arg[v0, v1, v2, v3])
                Ts.writes(relu[v0, v1, v2, v3])
                relu[v0, v1, v2, v3] = T.max(arg[v0, v1, v2, v3], T.float32(0))

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_tensor_transforms=[lambda n, c, h, w: (n, h, w, c)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_op_pool_nchw_nhwc():
    @Ts.function(private=True)
    def before(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        pool_max: T.Tensor((32, 64, 111, 223), "float32"),
    ):
        for ax0, ax1, ax2, ax3, rv0, rv1 in T.grid(32, 64, 111, 223, 2, 2):
            with Ts.sblock("pool_max"):
                v_ax0, v_ax1, v_ax2, v_ax3, v_rv0, v_rv1 = Ts.axis.remap(
                    "SSSSRR", [ax0, ax1, ax2, ax3, rv0, rv1]
                )
                Ts.reads(
                    arg[
                        v_ax0,
                        v_ax1,
                        v_ax2 * 2 + v_rv0 * 2,
                        v_ax3 + v_rv1,
                    ]
                )
                Ts.writes(pool_max[v_ax0, v_ax1, v_ax2, v_ax3])
                Ts.sblock_attr({"schedule_rule": "meta_schedule.pool_max"})
                with Ts.init():
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(-3.4028234663852886e38)
                pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3],
                    arg[
                        v_ax0,
                        v_ax1,
                        v_ax2 * 2 + v_rv0 * 2,
                        v_ax3 + v_rv1,
                    ],
                )

    @Ts.function(private=True)
    def expected(
        arg: T.Tensor((32, 224, 224, 64), "float32"),
        pool_max: T.Tensor((32, 111, 223, 64), "float32"),
    ):
        # with Ts.sblock("root"):
        for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(32, 111, 223, 64, 2, 2):
            with Ts.sblock("pool_max"):
                v0, v1, v2, v3, v4, v5 = Ts.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, ax4, ax5])
                Ts.reads(arg[v0, v1 * 2 + v4 * 2, v2 + v5, v3])
                Ts.writes(pool_max[v0, v1, v2, v3])
                Ts.sblock_attr({"schedule_rule": "meta_schedule.pool_max"})
                with Ts.init():
                    pool_max[v0, v1, v2, v3] = T.float32(-3.4028234663852886e38)
                pool_max[v0, v1, v2, v3] = T.max(
                    pool_max[v0, v1, v2, v3],
                    arg[v0, v1 * 2 + v4 * 2, v2 + v5, v3],
                )

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before,
        write_tensor_transforms=[lambda n, c, h, w: (n, h, w, c)],
    )
    after = apply_transformations(before, suggested_transforms)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_op_pool_nchw16c_nhwc():
    @Ts.function(private=True)
    def before(
        arg: T.Tensor(
            (32, 4, 224, 224, 16),
            "float32",
        ),
        pool_max: T.Tensor(
            (32, 4, 110, 220, 16),
            "float32",
        ),
    ):
        for ax0, ax1, ax2, ax3, ax4, rv0, rv1 in T.grid(32, 4, 110, 220, 16, 5, 5):
            with Ts.sblock("pool_max"):
                v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_rv0, v_rv1 = Ts.axis.remap(
                    "SSSSSRR", [ax0, ax1, ax2, ax3, ax4, rv0, rv1]
                )
                Ts.reads(arg[v_ax0, v_ax1, v_ax2 * 2 + v_rv0, v_ax3 + v_rv1, v_ax4])
                Ts.writes(pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                Ts.sblock_attr({"schedule_rule": "meta_schedule.pool_max"})
                with Ts.init():
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.float32(-3.4028234663852886e38)
                pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.max(
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4],
                    arg[v_ax0, v_ax1, v_ax2 * 2 + v_rv0, v_ax3 + v_rv1, v_ax4],
                )

    @Ts.function(private=True)
    def expected(
        arg: T.Tensor((32, 224, 224, 64), "float32"),
        pool_max: T.Tensor((32, 110, 220, 64), "float32"),
    ):
        for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(32, 110, 220, 64, 5, 5):
            with Ts.sblock("pool_max"):
                v0, v1, v2, v3, v4, v5 = Ts.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, ax4, ax5])
                Ts.reads(arg[v0, v1 * 2 + v4, v2 + v5, v3])
                Ts.writes(pool_max[v0, v1, v2, v3])
                Ts.sblock_attr({"schedule_rule": "meta_schedule.pool_max"})
                with Ts.init():
                    pool_max[v0, v1, v2, v3] = T.float32(-3.4028234663852886e38)
                pool_max[v0, v1, v2, v3] = T.max(
                    pool_max[v0, v1, v2, v3],
                    arg[v0, v1 * 2 + v4, v2 + v5, v3],
                )

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before,
        write_tensor_transforms=[lambda n, C, h, w, c: (n, h, w, C * 16 + c)],
    )
    after = apply_transformations(before, suggested_transforms)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_op_reduce():
    @Ts.function(private=True)
    def before(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        sum: T.Tensor((32, 64), "float32"),
    ):
        for ax0, ax1, k2, k3 in T.grid(32, 64, 224, 224):
            with Ts.sblock("rxplaceholder_red"):
                v_ax0, v_ax1, v_k2, v_k3 = Ts.axis.remap("SSRR", [ax0, ax1, k2, k3])
                Ts.reads(arg[v_ax0, v_ax1, v_k2, v_k3])
                Ts.writes(sum[v_ax0, v_ax1])
                with Ts.init():
                    sum[v_ax0, v_ax1] = T.float32(0)
                sum[v_ax0, v_ax1] = sum[v_ax0, v_ax1] + arg[v_ax0, v_ax1, v_k2, v_k3]

    @Ts.function(private=True)
    def expected(
        arg: T.Tensor((32, 4, 224, 224, 16), "float32"),
        sum: T.Tensor((32, 4, 16), "float32"),
    ):
        for ax0, ax1, ax2, ax3, ax4 in T.grid(32, 4, 224, 224, 16):
            with Ts.sblock("rxplaceholder_red"):
                v0, v1, v2, v3, v4 = Ts.axis.remap("SSRRS", [ax0, ax1, ax2, ax3, ax4])
                Ts.reads(arg[v0, v1, v2, v3, v4])
                Ts.writes(sum[v0, v1, v4])
                with Ts.init():
                    sum[v0, v1, v4] = T.float32(0)
                sum[v0, v1, v4] = sum[v0, v1, v4] + arg[v0, v1, v2, v3, v4]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_tensor_transforms=[lambda n, c: (n, c // 16, c % 16)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_op_upsampling():
    # relax materializes the layout if H, W or D dimensions are moved or tiled.
    @Ts.function(private=True)
    def before(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        resize: T.Tensor((32, 64, 202, 246), "float32"),
    ):
        for i0, i1, i2, i3 in T.grid(32, 64, 202, 246):
            with Ts.sblock("resize"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(arg[v_i0, v_i1, 0:224, 0:224])
                Ts.writes(resize[v_i0, v_i1, v_i2, v_i3])
                resize[v_i0, v_i1, v_i2, v_i3] = arg[
                    v_i0,
                    v_i1,
                    T.max(
                        T.min(
                            T.Cast(
                                "int64",
                                T.floor(
                                    T.float32(1.1089109182357788) * T.Cast("float32", v_i2)
                                    + T.float32(1.0000000000000001e-05)
                                ),
                            ),
                            223,
                        ),
                        0,
                    ),
                    T.max(
                        T.min(
                            T.Cast(
                                "int64",
                                T.floor(
                                    T.float32(0.91056913137435913) * T.Cast("float32", v_i3)
                                    + T.float32(1.0000000000000001e-05)
                                ),
                            ),
                            223,
                        ),
                        0,
                    ),
                ]

    @Ts.function(private=True)
    def expected(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        resize: T.Tensor((32, 202, 246, 64), "float32"),
    ):
        # with Ts.sblock("root"):
        for ax0, ax1, ax2, ax3 in T.grid(32, 202, 246, 64):
            with Ts.sblock("resize"):
                v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(arg[v0, v3, 0:224, 0:224])
                Ts.writes(resize[v0, v1, v2, v3])
                resize[v0, v1, v2, v3] = arg[
                    v0,
                    v3,
                    T.max(
                        T.min(
                            T.Cast(
                                "int64",
                                T.floor(
                                    T.float32(1.1089109182357788) * T.Cast("float32", v1)
                                    + T.float32(1.0000000000000001e-05)
                                ),
                            ),
                            T.int64(223),
                        ),
                        T.int64(0),
                    ),
                    T.max(
                        T.min(
                            T.Cast(
                                "int64",
                                T.floor(
                                    T.float32(0.91056913137435913) * T.Cast("float32", v2)
                                    + T.float32(1.0000000000000001e-05)
                                ),
                            ),
                            T.int64(223),
                        ),
                        T.int64(0),
                    ),
                ]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_tensor_transforms=[lambda n, c, h, w: (n, h, w, c)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_op_strided_slice():
    @Ts.function(private=True)
    def before(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        T_strided_slice_with_axes: T.Tensor((32, 64, 10, 8), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 64, 10, 8):
            with Ts.sblock("T_strided_slice_with_axes"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(
                    arg[
                        v_ax0,
                        v_ax1,
                        v_ax2 * 5 + 2,
                        v_ax3 * 7 + 4,
                    ]
                )
                Ts.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3])
                T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3] = arg[
                    v_ax0,
                    v_ax1,
                    v_ax2 * 5 + 2,
                    v_ax3 * 7 + 4,
                ]

    @Ts.function(private=True)
    def expected(
        arg: T.Tensor((32, 224, 224, 16, 4), "float32"),
        T_strided_slice_with_axes: T.Tensor((32, 10, 8, 16, 4), "float32"),
    ):
        # with Ts.sblock("root"):
        for ax0, ax1, ax2, ax3, ax4 in T.grid(32, 10, 8, 16, 4):
            with Ts.sblock("T_strided_slice_with_axes"):
                v0, v1, v2, v3, v4 = Ts.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                Ts.reads(arg[v0, v1 * 5 + 2, v2 * 7 + 4, v3, v4])
                Ts.writes(T_strided_slice_with_axes[v0, v1, v2, v3, v4])
                T_strided_slice_with_axes[v0, v1, v2, v3, v4] = arg[
                    v0, v1 * 5 + 2, v2 * 7 + 4, v3, v4
                ]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_tensor_transforms=[lambda n, c, h, w: (n, h, w, c // 4, c % 4)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_op_binary_broadcast():
    @Ts.function(private=True)
    def before(
        arg0: T.Tensor((32, 64, 224, 224), "float32"),
        arg1: T.Tensor((64, 224, 224), "float32"),
        T_add: T.Tensor((32, 64, 224, 224), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        # with Ts.sblock("root"):
        for ax0, ax1, ax2, ax3 in T.grid(32, 64, 224, 224):
            with Ts.sblock("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(
                    arg0[v_ax0, v_ax1, v_ax2, v_ax3],
                    arg1[v_ax1, v_ax2, v_ax3],
                )
                Ts.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = (
                    arg0[v_ax0, v_ax1, v_ax2, v_ax3] + arg1[v_ax1, v_ax2, v_ax3]
                )

    @Ts.function(private=True)
    def expected(
        arg0: T.Tensor((32, 224, 224, 16, 4), "float32"),
        arg1: T.Tensor((224, 224, 16, 4), "float32"),
        T_add: T.Tensor((32, 224, 224, 16, 4), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        # with Ts.sblock("root"):
        for ax0, ax1, ax2, ax3, ax4 in T.grid(32, 224, 224, 16, 4):
            with Ts.sblock("T_add"):
                v0, v1, v2, v3, v4 = Ts.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                Ts.reads(arg0[v0, v1, v2, v3, v4], arg1[v1, v2, v3, v4])
                Ts.writes(T_add[v0, v1, v2, v3, v4])
                T_add[v0, v1, v2, v3, v4] = arg0[v0, v1, v2, v3, v4] + arg1[v1, v2, v3, v4]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_tensor_transforms=[lambda n, c, h, w: (n, h, w, c // 4, c % 4)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_op_transpose():
    @Ts.function(private=True)
    def before(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        T_transpose: T.Tensor((32, 224, 224, 64), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 224, 224, 64):
            with Ts.sblock("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(arg[v_ax0, v_ax3, v_ax1, v_ax2])
                Ts.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = arg[v_ax0, v_ax3, v_ax1, v_ax2]

    @Ts.function(private=True)
    def expected(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        T_transpose: T.Tensor((32, 224, 64, 224), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 224, 64, 224):
            with Ts.sblock("T_transpose"):
                v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(arg[v0, v2, v3, v1])
                Ts.writes(T_transpose[v0, v1, v2, v3])
                T_transpose[v0, v1, v2, v3] = arg[v0, v2, v3, v1]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_tensor_transforms=[lambda n, c, h, w: (n, h, w, c)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_op_pad():
    @Ts.function(private=True)
    def before(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        PadInput: T.Tensor((32, 64, 230, 230), "float32"),
    ):
        for i0, i1, i2, i3 in T.grid(32, 64, 230, 230):
            with Ts.sblock("PadInput"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(arg[v_i0, v_i1, v_i2 - 2, v_i3 - 2])
                Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(
                    2 <= v_i2 and v_i2 < 226 and 2 <= v_i3 and v_i3 < 226,
                    arg[v_i0, v_i1, v_i2 - 2, v_i3 - 2],
                    T.float32(2),
                )

    @Ts.function(private=True)
    def expected(
        arg: T.Tensor((32, 224, 224, 16, 4), "float32"),
        PadInput: T.Tensor((32, 230, 230, 16, 4), "float32"),
    ):
        for ax0, ax1, ax2, ax3, ax4 in T.grid(32, 230, 230, 16, 4):
            with Ts.sblock("PadInput"):
                v0, v1, v2, v3, v4 = Ts.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                Ts.reads(arg[v0, v1 - 2, v2 - 2, v3, v4])
                Ts.writes(PadInput[v0, v1, v2, v3, v4])
                PadInput[v0, v1, v2, v3, v4] = T.if_then_else(
                    2 <= v1 and v1 < 226 and 2 <= v2 and v2 < 226,
                    arg[v0, v1 - 2, v2 - 2, v3, v4],
                    T.float32(2),
                )

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_tensor_transforms=[lambda n, c, h, w: (n, h, w, c // 4, c % 4)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_op_split():
    @Ts.function(private=True)
    def before(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        split0: T.Tensor((32, 32, 224, 224), "float32"),
        split1: T.Tensor((32, 32, 224, 224), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 32, 224, 224):
            with Ts.sblock("T_split_sections"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(arg[v_ax0, v_ax1, v_ax2, v_ax3])
                Ts.writes(split0[v_ax0, v_ax1, v_ax2, v_ax3])
                split0[v_ax0, v_ax1, v_ax2, v_ax3] = arg[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0, ax1, ax2, ax3 in T.grid(32, 32, 224, 224):
            with Ts.sblock("T_split_sections_1"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(arg[v_ax0, v_ax1 + 32, v_ax2, v_ax3])
                Ts.writes(split1[v_ax0, v_ax1, v_ax2, v_ax3])
                split1[v_ax0, v_ax1, v_ax2, v_ax3] = arg[v_ax0, v_ax1 + 32, v_ax2, v_ax3]

    @Ts.function(private=True)
    def expected(
        arg: T.Tensor((32, 224, 224, 64), "float32"),
        split0: T.Tensor((32, 224, 224, 32), "float32"),
        split1: T.Tensor((32, 224, 224, 32), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 224, 224, 32):
            with Ts.sblock("T_split_sections"):
                v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(arg[v0, v1, v2, v3])
                Ts.writes(split0[v0, v1, v2, v3])
                split0[v0, v1, v2, v3] = arg[v0, v1, v2, v3]
        for ax0, ax1, ax2, ax3 in T.grid(32, 224, 224, 32):
            with Ts.sblock("T_split_sections_1"):
                v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(arg[v0, v1, v2, v3 + 32])
                Ts.writes(split1[v0, v1, v2, v3])
                split1[v0, v1, v2, v3] = arg[v0, v1, v2, v3 + 32]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before,
        write_tensor_transforms=[lambda n, c, h, w: (n, h, w, c), lambda n, c, h, w: (n, h, w, c)],
    )
    after = apply_transformations(before, suggested_transforms)
    tvm_tirx.ir.assert_structural_equal(after, expected)


@pytest.mark.skip("temp disable, due to minor sym regression")
def test_op_split_tiling_split_dim():
    @Ts.function(private=True)
    def before(
        arg: T.Tensor((32, 64, 224, 224), "float32"),
        split0: T.Tensor((32, 32, 224, 224), "float32"),
        split1: T.Tensor((32, 32, 224, 224), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 32, 224, 224):
            with Ts.sblock("T_split_sections"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(arg[v_ax0, v_ax1, v_ax2, v_ax3])
                Ts.writes(split0[v_ax0, v_ax1, v_ax2, v_ax3])
                split0[v_ax0, v_ax1, v_ax2, v_ax3] = arg[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0, ax1, ax2, ax3 in T.grid(32, 32, 224, 224):
            with Ts.sblock("T_split_sections_1"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(arg[v_ax0, v_ax1 + 32, v_ax2, v_ax3])
                Ts.writes(split1[v_ax0, v_ax1, v_ax2, v_ax3])
                split1[v_ax0, v_ax1, v_ax2, v_ax3] = arg[v_ax0, v_ax1 + 32, v_ax2, v_ax3]

    @Ts.function(private=True)
    def expected(
        arg: T.Tensor((32, 224, 224, 16, 4), "float32"),
        split0: T.Tensor((32, 224, 224, 8, 4), "float32"),
        split1: T.Tensor((32, 224, 224, 8, 4), "float32"),
    ):
        # with Ts.sblock("root"):
        for ax0, ax1, ax2, ax3, ax4 in T.grid(32, 224, 224, 8, 4):
            with Ts.sblock("T_split_sections"):
                v0, v1, v2, v3, v4 = Ts.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                Ts.reads(arg[v0, v1, v2, v3, v4])
                Ts.writes(split0[v0, v1, v2, v3, v4])
                split0[v0, v1, v2, v3, v4] = arg[v0, v1, v2, v3, v4]
        for ax0, ax1, ax2, ax3, ax4 in T.grid(32, 224, 224, 8, 4):
            with Ts.sblock("T_split_sections_1"):
                v0, v1, v2, v3, v4 = Ts.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                Ts.reads(arg[v0, v1, v2, v3 + 8, v4])
                Ts.writes(split1[v0, v1, v2, v3, v4])
                split1[v0, v1, v2, v3, v4] = arg[v0, v1, v2, v3 + 8, v4]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before,
        write_tensor_transforms=[
            lambda n, c, h, w: (n, h, w, c // 4, c % 4),
            lambda n, c, h, w: (n, h, w, c // 4, c % 4),
        ],
    )
    after = apply_transformations(before, suggested_transforms)
    tvm_tirx.ir.assert_structural_equal(after, expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
