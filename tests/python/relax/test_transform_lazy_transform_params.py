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
import numpy as np
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.transform import LazyTransformParams
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_lazy_transform_params():
    @I.ir_module
    class Before:
        @Ts.function
        def transform_layout_IOHW_to_OIHW(
            w1: T.Tensor((3, 16, 3, 3), "float32"), out: T.Tensor((16, 3, 3, 3), "float32")
        ):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with Ts.sblock("layout_transform"):
                    o, i, h, w = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(w1[i, o, h, w])
                    Ts.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((3, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
            ),
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
        ):
            # we expect ToNonDataflow and RemovePurityTracking to be invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Before
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
            lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = params[0]
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                ty_args=[R.Tensor((16, 3, 3, 3), dtype="float32")],
            )
            gv: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((16, 3, 3, 3), dtype="float32"),
            ) = (lv, lv2)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function
        def transform_layout_IOHW_to_OIHW(
            w1: T.Tensor((3, 16, 3, 3), "float32"), out: T.Tensor((16, 3, 3, 3), "float32")
        ):
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with Ts.sblock("layout_transform"):
                    o, i, h, w = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(w1[i, o, h, w])
                    Ts.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function(pure=False)
        def main_transform_params() -> R.Tuple:
            cls = Expected
            lv: R.Any = R.call_packed("get_item", R.prim_value(1), ty_args=(R.Any,))
            gv1: R.Tensor((16, 16, 3, 3), dtype="float32") = R.match_cast(
                lv, R.Tensor((16, 16, 3, 3), dtype="float32")
            )
            lv_m: R.Tensor((16, 16, 3, 3), dtype="float32") = gv1
            _: R.Any = R.call_packed("set_item", R.prim_value(0), lv_m, ty_args=(R.Any,))
            _1: R.Tuple = R.vm.kill_object(lv_m)
            lv1: R.Any = R.call_packed("get_item", R.prim_value(0), ty_args=(R.Any,))
            gv3: R.Tensor((3, 16, 3, 3), dtype="float32") = R.match_cast(
                lv1, R.Tensor((3, 16, 3, 3), dtype="float32")
            )
            lv1_m: R.Tensor((3, 16, 3, 3), dtype="float32") = gv3
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1_m,),
                ty_args=[R.Tensor((16, 3, 3, 3), dtype="float32")],
            )
            _2: R.Tuple = R.vm.kill_object(lv1_m)
            _3: R.Any = R.call_packed("set_item", R.prim_value(1), lv2, ty_args=(R.Any,))
            gv: R.Tuple = ()
            return gv

    after = LazyTransformParams()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected, map_free_vars=True)


def test_get_item_only():
    @I.ir_module
    class Before:
        @Ts.function
        def transform_layout_IOHW_to_OIHW(
            w1: T.Tensor((3, 16, 3, 3), "float32"), out: T.Tensor((16, 3, 3, 3), "float32")
        ):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with Ts.sblock("layout_transform"):
                    o, i, h, w = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(w1[i, o, h, w])
                    Ts.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((3, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
            ),
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
        ):
            # we expect ToNonDataflow and RemovePurityTracking to be invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Before
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
            lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = params[0]
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                ty_args=[R.Tensor((16, 3, 3, 3), dtype="float32")],
            )
            lv3 = R.add(lv2, R.const(1, "float32"))
            gv: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((16, 3, 3, 3), dtype="float32"),
            ) = (lv, lv3)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function
        def transform_layout_IOHW_to_OIHW(
            w1: T.Tensor((3, 16, 3, 3), "float32"), out: T.Tensor((16, 3, 3, 3), "float32")
        ):
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with Ts.sblock("layout_transform"):
                    o, i, h, w = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(w1[i, o, h, w])
                    Ts.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function(pure=False)
        def main_transform_params() -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
        ):
            cls = Expected
            gv: R.Any = R.call_packed("get_item_0", R.prim_value(1), ty_args=(R.Any,))
            gv1: R.Tensor((16, 16, 3, 3), dtype="float32") = R.match_cast(
                gv, R.Tensor((16, 16, 3, 3), dtype="float32")
            )
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = gv1
            gv2: R.Any = R.call_packed("get_item_0", R.prim_value(0), ty_args=(R.Any,))
            gv3: R.Tensor((3, 16, 3, 3), dtype="float32") = R.match_cast(
                gv2, R.Tensor((3, 16, 3, 3), dtype="float32")
            )
            lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = gv3
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                ty_args=[R.Tensor((16, 3, 3, 3), dtype="float32")],
            )
            lv3: R.Tensor((16, 3, 3, 3), dtype="float32") = R.add(lv2, R.const(1, "float32"))
            gv_1: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
            ) = (lv, lv3)
            return gv_1

    after = LazyTransformParams(fget_item="get_item_0", fset_item=None)(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected, map_free_vars=True)


def test_extra_get_item_params():
    @I.ir_module
    class Before:
        @Ts.function
        def transform_layout_IOHW_to_OIHW(
            w1: T.Tensor((3, 16, 3, 3), "float32"), out: T.Tensor((16, 3, 3, 3), "float32")
        ):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with Ts.sblock("layout_transform"):
                    o, i, h, w = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(w1[i, o, h, w])
                    Ts.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((3, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
            ),
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
        ):
            # we expect ToNonDataflow and RemovePurityTracking to be invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Before
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
            lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = params[0]
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                ty_args=[R.Tensor((16, 3, 3, 3), dtype="float32")],
            )
            lv3 = R.add(lv2, R.const(1, "float32"))
            gv: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((16, 3, 3, 3), dtype="float32"),
            ) = (lv, lv3)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function
        def transform_layout_IOHW_to_OIHW(
            w1: T.Tensor((3, 16, 3, 3), "float32"), out: T.Tensor((16, 3, 3, 3), "float32")
        ):
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with Ts.sblock("layout_transform"):
                    o, i, h, w = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(w1[i, o, h, w])
                    Ts.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function(pure=False)
        def main_transform_params(loader: R.Any) -> R.Tuple:
            cls = Expected
            gv: R.Any = R.call_packed("get_item", loader, R.prim_value(1), ty_args=(R.Any,))
            gv1: R.Tensor((16, 16, 3, 3), dtype="float32") = R.match_cast(
                gv, R.Tensor((16, 16, 3, 3), dtype="float32")
            )
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = gv1
            _: R.Any = R.call_packed("set_item", R.prim_value(0), lv, ty_args=(R.Any,))
            _1: R.Tuple = R.vm.kill_object(lv)
            gv2: R.Any = R.call_packed("get_item", loader, R.prim_value(0), ty_args=(R.Any,))
            gv3: R.Tensor((3, 16, 3, 3), dtype="float32") = R.match_cast(
                gv2, R.Tensor((3, 16, 3, 3), dtype="float32")
            )
            lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = gv3
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                ty_args=[R.Tensor((16, 3, 3, 3), dtype="float32")],
            )
            _2: R.Tuple = R.vm.kill_object(lv1)
            lv3: R.Tensor((16, 3, 3, 3), dtype="float32") = R.add(lv2, R.const(1, "float32"))
            _3: R.Any = R.call_packed("set_item", R.prim_value(1), lv3, ty_args=(R.Any,))
            gv_1: R.Tuple = ()
            return gv_1

    after = LazyTransformParams(extra_get_item_params=[relax.Var("loader", relax.AnyType())])(
        Before
    )
    tvm_tirx.ir.assert_structural_equal(after, Expected, map_free_vars=True)


def test_extra_set_item_params():
    @I.ir_module
    class Before:
        @Ts.function
        def transform_layout_IOHW_to_OIHW(
            w1: T.Tensor((3, 16, 3, 3), "float32"), out: T.Tensor((16, 3, 3, 3), "float32")
        ):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with Ts.sblock("layout_transform"):
                    o, i, h, w = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(w1[i, o, h, w])
                    Ts.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((3, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
            ),
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
        ):
            # we expect ToNonDataflow and RemovePurityTracking to be invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Before
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
            lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = params[0]
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                ty_args=[R.Tensor((16, 3, 3, 3), dtype="float32")],
            )
            lv3 = R.add(lv2, R.const(1, "float32"))
            gv: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((16, 3, 3, 3), dtype="float32"),
            ) = (lv, lv3)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function
        def transform_layout_IOHW_to_OIHW(
            w1: T.Tensor((3, 16, 3, 3), "float32"), out: T.Tensor((16, 3, 3, 3), "float32")
        ):
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with Ts.sblock("layout_transform"):
                    o, i, h, w = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(w1[i, o, h, w])
                    Ts.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function(pure=False)
        def main_transform_params(setter: R.Any) -> R.Tuple:
            cls = Expected
            gv: R.Any = R.call_packed("get_item", R.prim_value(1), ty_args=(R.Any,))
            gv1: R.Tensor((16, 16, 3, 3), dtype="float32") = R.match_cast(
                gv, R.Tensor((16, 16, 3, 3), dtype="float32")
            )
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = gv1
            _: R.Any = R.call_packed("set_item", setter, R.prim_value(0), lv, ty_args=(R.Any,))
            _1: R.Tuple = R.vm.kill_object(lv)
            gv2: R.Any = R.call_packed("get_item", R.prim_value(0), ty_args=(R.Any,))
            gv3: R.Tensor((3, 16, 3, 3), dtype="float32") = R.match_cast(
                gv2, R.Tensor((3, 16, 3, 3), dtype="float32")
            )
            lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = gv3
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                ty_args=[R.Tensor((16, 3, 3, 3), dtype="float32")],
            )
            _2: R.Tuple = R.vm.kill_object(lv1)
            lv3: R.Tensor((16, 3, 3, 3), dtype="float32") = R.add(lv2, R.const(1, "float32"))
            _3: R.Any = R.call_packed("set_item", setter, R.prim_value(1), lv3, ty_args=(R.Any,))
            gv_1: R.Tuple = ()
            return gv_1

    after = LazyTransformParams(extra_set_item_params=[relax.Var("setter", relax.AnyType())])(
        Before
    )
    tvm_tirx.ir.assert_structural_equal(after, Expected, map_free_vars=True)


def test_extra_set_item_params_with_const_output():
    @I.ir_module
    class Before:
        @R.function
        def main_transform_params(
            params: R.Tuple(),
        ) -> R.Tuple(R.Tensor([2], dtype="float32"), R.Tensor([3], dtype="float32")):
            R.func_attr({"relax.force_pure": True})
            gv = (
                R.const(np.array([1, 2]).astype("float32")),
                R.const(np.array([3, 4]).astype("float32")),
            )
            return gv

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main_transform_params(setter: R.Any) -> R.Tuple:
            output = ()
            _ = R.call_packed(
                "set_item",
                setter,
                R.prim_value(0),
                R.const(np.array([1, 2]).astype("float32")),
                ty_args=(R.Any,),
            )
            _ = R.call_packed(
                "set_item",
                setter,
                R.prim_value(1),
                R.const(np.array([3, 4]).astype("float32")),
                ty_args=(R.Any,),
            )
            return output

    after = LazyTransformParams(extra_set_item_params=[relax.Var("setter", relax.AnyType())])(
        Before
    )
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_lazy_transform_params_with_symbolic_vars():
    slice_index = T.dynamic("slice_index", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((16, 16), dtype="float32"),
                R.Shape(
                    [slice_index],
                ),
            ),
        ):
            # we expect ToNonDataflow and RemovePurityTracking to be invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Before

            param = params[0]
            transformed = R.call_tir(
                cls.slice_buffer,
                (param, slice_index),
                ty_args=[R.Tensor((16,), dtype="float32")],
            )
            output = (transformed,)
            return output

        @Ts.function(private=True)
        def slice_buffer(
            Input: T.Tensor((16, 16), "float32"),
            slice_index: T.int64,
            Output: T.Tensor(16, "float32"),
        ):
            for i in T.grid(16):
                with Ts.sblock("slice_buffer"):
                    vi = Ts.axis.remap("S", [i])
                    Output[vi] = Input[slice_index, vi]

    slice_index = T.dynamic("slice_index", "int64")

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main_transform_params(slice_shape_expr: R.Shape([slice_index])):
            cls = Expected

            param = R.call_packed("get_item", R.prim_value(0), ty_args=(R.Any,))
            gv: R.Tensor((16, 16), dtype="float32") = R.match_cast(
                param, R.Tensor((16, 16), dtype="float32")
            )
            param_m: R.Tensor((16, 16), dtype="float32") = gv
            transformed = R.call_tir(
                cls.slice_buffer,
                (param_m, slice_index),
                ty_args=[R.Tensor((16,), dtype="float32")],
            )
            unused_1_ = R.vm.kill_object(param_m)
            unused_2_ = R.call_packed("set_item", R.prim_value(0), transformed, ty_args=(R.Any,))

            output = ()
            return output

        @Ts.function(private=True)
        def slice_buffer(
            Input: T.Tensor((16, 16), "float32"),
            slice_index: T.int64,
            Output: T.Tensor(16, "float32"),
        ):
            for i in T.grid(16):
                with Ts.sblock("slice_buffer"):
                    vi = Ts.axis.remap("S", [i])
                    Output[vi] = Input[slice_index, vi]

    after = LazyTransformParams()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected, map_free_vars=True)


def test_param_shape_symbolic():
    ic_transform_layout_IOHW_to_OIHW = T.dynamic("ic", "int32")
    ic_main_transform_params = T.dynamic("ic", "int64")

    @I.ir_module
    class Before:
        @Ts.function
        def transform_layout_IOHW_to_OIHW(
            w1: T.Tensor((ic_transform_layout_IOHW_to_OIHW, 16, 3, 3), "float32"),
            out: T.Tensor((16, ic_transform_layout_IOHW_to_OIHW, 3, 3), "float32"),
        ):
            for ax0, ax1, ax2, ax3 in T.grid(16, ic_transform_layout_IOHW_to_OIHW, 3, 3):
                with Ts.sblock("layout_transform"):
                    o, i, h, w = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(w1[i, o, h, w])
                    Ts.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((3, ic_main_transform_params, 3, 3), dtype="float32"),
                R.Tensor((16, 16, 3, 3), dtype="float32"),
            ),
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"),
            R.Tensor((ic_main_transform_params, 3, 3, 3), dtype="float32"),
        ):
            # we expect ToNonDataflow and RemovePurityTracking to be invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Before
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
            lv1: R.Tensor((3, ic_main_transform_params, 3, 3), dtype="float32") = params[0]
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                ty_args=[R.Tensor((ic_main_transform_params, 3, 3, 3), dtype="float32")],
            )
            gv: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((ic_main_transform_params, 3, 3, 3), dtype="float32"),
            ) = (lv, lv2)
            return gv

    ic_transform_layout_IOHW_to_OIHW = T.dynamic("ic", "int32")
    ic_main_transform_params = T.dynamic("ic", "int64")

    @I.ir_module
    class Expected:
        @Ts.function
        def transform_layout_IOHW_to_OIHW(
            w1: T.Tensor((ic_transform_layout_IOHW_to_OIHW, 16, 3, 3), "float32"),
            out: T.Tensor((16, ic_transform_layout_IOHW_to_OIHW, 3, 3), "float32"),
        ):
            for ax0, ax1, ax2, ax3 in T.grid(16, ic_transform_layout_IOHW_to_OIHW, 3, 3):
                with Ts.sblock("layout_transform"):
                    o, i, h, w = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(w1[i, o, h, w])
                    Ts.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function(pure=False)
        def main_transform_params() -> R.Tuple:
            cls = Expected
            gv: R.Any = R.call_packed("get_item", R.prim_value(1), ty_args=(R.Any,))
            gv1: R.Tensor((16, 16, 3, 3), dtype="float32") = R.match_cast(
                gv, R.Tensor((16, 16, 3, 3), dtype="float32")
            )
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = gv1
            _: R.Any = R.call_packed("set_item", R.prim_value(0), lv, ty_args=(R.Any,))
            _1: R.Tuple = R.vm.kill_object(lv)
            gv2: R.Any = R.call_packed("get_item", R.prim_value(0), ty_args=(R.Any,))
            gv3: R.Tensor((3, ic_main_transform_params, 3, 3), dtype="float32") = R.match_cast(
                gv2, R.Tensor((3, ic_main_transform_params, 3, 3), dtype="float32")
            )
            lv1: R.Tensor((3, ic_main_transform_params, 3, 3), dtype="float32") = gv3
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                ty_args=[R.Tensor((ic_main_transform_params, 3, 3, 3), dtype="float32")],
            )
            _2: R.Tuple = R.vm.kill_object(lv1)
            _3: R.Any = R.call_packed("set_item", R.prim_value(1), lv2, ty_args=(R.Any,))
            gv4: R.Tuple = ()
            return gv4

    after = LazyTransformParams()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected, map_free_vars=True)


def test_output_with_use_site():
    @I.ir_module
    class Module:
        @Ts.function
        def copy(x: T.Tensor((), "float32"), y: T.Tensor((), "float32")):
            with Ts.sblock("block"):
                Ts.reads(x[()])
                Ts.writes(y[()])
                y[()] = x[()]

        @R.function
        def main_transform_params(params: R.Tuple(R.Tensor((), dtype="float32"))) -> R.Tuple(
            R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32")
        ):
            # we expect ToNonDataflow and RemovePurityTracking to be invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Module
            x: R.Tensor((), dtype="float32") = params[0]
            y = R.call_tir(cls.copy, (x,), ty_args=[R.Tensor((), dtype="float32")])
            z = R.call_tir(cls.copy, (y,), ty_args=[R.Tensor((), dtype="float32")])
            gv: R.Tuple(R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32")) = (y, z)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function
        def copy(x: T.Tensor((), "float32"), y: T.Tensor((), "float32")):
            with Ts.sblock("block"):
                Ts.reads(x[()])
                Ts.writes(y[()])
                y[()] = x[()]

        @R.function(pure=False)
        def main_transform_params() -> R.Tuple:
            cls = Expected
            x: R.Any = R.call_packed("get_item", R.prim_value(0), ty_args=(R.Any,))
            gv: R.Tensor((), dtype="float32") = R.match_cast(x, R.Tensor((), dtype="float32"))
            x_m: R.Tensor((), dtype="float32") = gv
            y = R.call_tir(cls.copy, (x_m,), ty_args=[R.Tensor((), dtype="float32")])
            _: R.Tuple = R.vm.kill_object(x_m)
            z = R.call_tir(cls.copy, (y,), ty_args=[R.Tensor((), dtype="float32")])
            _1: R.Any = R.call_packed("set_item", R.prim_value(0), y, ty_args=(R.Any,))
            _2: R.Any = R.call_packed("set_item", R.prim_value(1), z, ty_args=(R.Any,))
            gv: R.Tuple = ()
            return gv

    after = LazyTransformParams()(Module)
    tvm_tirx.ir.assert_structural_equal(after, Expected, map_free_vars=True)


def test_output():
    target = "llvm"
    dev = tvm_tirx.cpu()

    ic = T.dynamic("ic", "int64")

    @I.ir_module
    class TransformModule:
        @R.function
        def transform_params(
            params: R.Tuple(
                R.Tensor((3, ic, 3, 3), dtype="float32"),
                R.Tensor((16, 16, 3, 3), dtype="float32"),
            ),
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((ic, 3, 3, 3), dtype="float32")
        ):
            R.func_attr({"relax.force_pure": True})
            param0 = params[0]
            param1 = params[1]
            transformed0 = R.permute_dims(param0, [1, 0, 2, 3])
            transformed = (transformed0, param1)
            return transformed

    mod = TransformModule
    mod = relax.transform.LazyTransformParams()(mod)
    mod = relax.transform.LegalizeOps()(mod)
    built = tvm.compile(mod, target=target)

    params = [
        np.random.random(size=(3, 64, 3, 3)).astype("float32"),
        np.random.random(size=(16, 16, 3, 3)).astype("float32"),
    ]
    transformed = {}
    expected = [params[0].transpose(1, 0, 2, 3), params[1]]

    @tvm_ffi.register_global_func("get_item", override=True)
    def get_item(i):
        return tvm_tirx.runtime.tensor(params[i], dev)

    @tvm_ffi.register_global_func("set_item", override=True)
    def set_item(i, value):
        assert i not in transformed, f"Set item called multiple times for index {i}"
        transformed[i] = value.numpy()

    vm = relax.VirtualMachine(built, dev)
    vm["transform_params"]()

    assert sorted(transformed) == list(range(len(transformed)))
    transformed = [value for i, value in sorted(transformed.items())]
    assert len(transformed) == len(expected)

    for expected_i, transformed_i in zip(expected, transformed):
        tvm_tirx.testing.assert_allclose(expected_i, transformed_i)


def test_duplicate_outputs():
    """A tensor may be repeated in the output

    This is something that should be avoided upstream, but is a legal
    parameter transformation, and should produce correct output.
    """

    @I.ir_module
    class Before:
        @R.function
        def main_transform_params(
            params: R.Tuple(R.Tensor([16], dtype="int32"), R.Tensor([16], dtype="int32")),
        ):
            R.func_attr({"relax.force_pure": True})
            param0 = params[0]
            param1 = params[1]
            transformed0 = R.add(param0, R.const(1, "int32"))
            transformed1 = R.add(param1, R.const(2, "int32"))
            output = (transformed0, transformed1, transformed0)
            return output

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main_transform_params() -> R.Tuple:
            gv: R.Any = R.call_packed("get_item", R.prim_value(0), ty_args=(R.Any,))
            gv1: R.Tensor((16,), dtype="int32") = R.match_cast(gv, R.Tensor((16,), dtype="int32"))
            param0: R.Tensor((16,), dtype="int32") = gv1

            gv2: R.Any = R.call_packed("get_item", R.prim_value(1), ty_args=(R.Any,))
            gv3: R.Tensor((16,), dtype="int32") = R.match_cast(gv2, R.Tensor((16,), dtype="int32"))
            param1: R.Tensor((16,), dtype="int32") = gv3

            transformed0: R.Tensor((16,), dtype="int32") = R.add(param0, R.const(1, "int32"))
            _: R.Tuple = R.vm.kill_object(param0)
            _: R.Any = R.call_packed("set_item", R.prim_value(0), transformed0, ty_args=(R.Any,))
            _: R.Any = R.call_packed("set_item", R.prim_value(2), transformed0, ty_args=(R.Any,))

            transformed1: R.Tensor((16,), dtype="int32") = R.add(param1, R.const(2, "int32"))
            _ = R.vm.kill_object(param1)
            _ = R.call_packed("set_item", R.prim_value(1), transformed1, ty_args=(R.Any,))
            output = ()
            return output

    after = LazyTransformParams()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_params_without_tuple():
    @I.ir_module
    class Before:
        @R.function
        def transform_params(A: R.Tensor([16, 16], "float32"), B: R.Tensor([16, 16], "float32")):
            C = R.multiply(A, R.const(2, "float32"))
            D = R.add(C, B)
            return (D, B)

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def transform_params():
            A = R.call_packed("get_item", R.prim_value(0), ty_args=[R.Any])
            A = R.match_cast(A, R.Tensor([16, 16], "float32"))
            C = R.multiply(A, R.const(2, "float32"))

            B = R.call_packed("get_item", R.prim_value(1), ty_args=[R.Any])
            B = R.match_cast(B, R.Tensor([16, 16], "float32"))
            D = R.add(C, B)
            return (D, B)

    After = LazyTransformParams(fset_item=None)(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_params_without_tuple_with_symbolic_var():
    @I.ir_module
    class Before:
        @R.function
        def transform_params(A: R.Any):
            return (A,)

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def transform_params():
            A = R.call_packed("get_item", R.prim_value(0), ty_args=[R.Any])
            A = R.match_cast(A, R.Any)

            return (A,)

    After = LazyTransformParams(fset_item=None)(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_get_item_callback():
    @I.ir_module
    class Before:
        @R.function
        def transform_params(A: R.Tensor([16, 16], "float32"), B: R.Tensor([16, 16], "float32")):
            C = R.multiply(A, R.const(2, "float32"))
            D = R.add(C, B)
            return (D, B)

    @I.ir_module
    class Expected:
        @R.function
        def transform_params(fget_param: R.Callable([T.int64, R.Any], R.Any)):
            R.func_attr({"num_input": 1})
            A = fget_param(R.prim_value(0), R.str("A"))
            A = R.match_cast(A, R.Tensor([16, 16], "float32"))
            C = R.multiply(A, R.const(2, "float32"))

            B = fget_param(R.prim_value(1), R.str("B"))
            B = R.match_cast(B, R.Tensor([16, 16], "float32"))
            D = R.add(C, B)
            return (D, B)

    After = relax.transform.LazyGetInput()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_get_item_callback_dynamic_shape():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @R.function
        def transform_params(
            A: R.Tensor([m, n], "float32"), B: R.Tensor([m, n], "float32")
        ) -> R.Tuple(R.Tensor([m, n], "float32"), R.Tensor([m, n], "float32")):
            C = R.multiply(A, R.const(2, "float32"))
            D = R.add(C, B)
            return (D, B)

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def transform_params(
            fget_param: R.Callable([T.int64, R.Any], R.Any),
        ) -> R.Tuple(R.Tensor(ndim=2, dtype="float32"), R.Tensor(ndim=2, dtype="float32")):
            R.func_attr({"num_input": 1})

            A = fget_param(R.prim_value(0), R.str("A"))
            A = R.match_cast(A, R.Tensor([m, n], "float32"))
            C = R.multiply(A, R.const(2, "float32"))

            B = fget_param(R.prim_value(1), R.str("B"))
            B = R.match_cast(B, R.Tensor([m, n], "float32"))
            D = R.add(C, B)
            return (D, B)

    After = relax.transform.LazyGetInput()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_set_output_callback():
    """fset_output is called for each element of the output tuple

    The call is placed immediately after the corresponding
    `VarBinding`.
    """

    @I.ir_module
    class Before:
        @R.function
        def transform_params(A: R.Tensor([16, 16], "float32"), B: R.Tensor([16, 16], "float32")):
            C = R.multiply(A, R.const(2, "float32"))
            D = R.add(C, B)
            return (D, C)

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def transform_params(
            A: R.Tensor([16, 16], "float32"),
            B: R.Tensor([16, 16], "float32"),
            fset_output: R.Callable([T.int64, R.Any], R.Tuple([]), purity=False),
        ):
            C = R.multiply(A, R.const(2, "float32"))
            fset_output(R.prim_value(1), C)
            D = R.add(C, B)
            fset_output(R.prim_value(0), D)
            return ()

    After = relax.transform.LazySetOutput()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_set_output_callback_of_param():
    """fset_output may need to be called for parameters

    A function parameter does not have a `VarBinding`.  If a parameter
    is returned in the output tuple, the `fset_output` call is
    generated at the beginning of the function.
    """

    @I.ir_module
    class Before:
        @R.function
        def transform_params(A: R.Tensor([16, 16], "float32"), B: R.Tensor([16, 16], "float32")):
            C = R.multiply(A, R.const(2, "float32"))
            D = R.add(C, B)
            return (D, B)

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def transform_params(
            A: R.Tensor([16, 16], "float32"),
            B: R.Tensor([16, 16], "float32"),
            fset_output: R.Callable([T.int64, R.Any], R.Tuple([]), purity=False),
        ):
            fset_output(R.prim_value(1), B)
            C = R.multiply(A, R.const(2, "float32"))
            D = R.add(C, B)
            fset_output(R.prim_value(0), D)
            return ()

    After = relax.transform.LazySetOutput()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_set_output_callback_num_input():
    """The parameter transformation may have other runtime parameters

    The new `fset_output` parameter is placed after the other runtime
    parameters, before any model weights.
    """

    @I.ir_module
    class Before:
        @R.function
        def transform_params(A: R.Tensor([16, 16], "float32"), B: R.Tensor([16, 16], "float32")):
            R.func_attr({"num_input": 1})
            C = R.multiply(A, R.const(2, "float32"))
            D = R.add(C, B)
            return (D, B)

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def transform_params(
            A: R.Tensor([16, 16], "float32"),
            fset_output: R.Callable([T.int64, R.Any], R.Tuple([]), purity=False),
            B: R.Tensor([16, 16], "float32"),
        ):
            R.func_attr({"num_input": 2})
            fset_output(R.prim_value(1), B)
            C = R.multiply(A, R.const(2, "float32"))
            D = R.add(C, B)
            fset_output(R.prim_value(0), D)
            return ()

    After = relax.transform.LazySetOutput()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_set_output_callback_with_duplicate_output():
    """fset_output may be called more than once for a variable

    A variable may occur multiple times in the output tuple.  The
    `fset_output` callback should be called once for each tuple
    element, even if they reuse the same variable.
    """

    @I.ir_module
    class Before:
        @R.function
        def transform_params(A: R.Tensor([16, 16], "float32"), B: R.Tensor([16, 16], "float32")):
            C = R.multiply(A, R.const(2, "float32"))
            D = R.add(C, B)
            return (D, D)

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def transform_params(
            A: R.Tensor([16, 16], "float32"),
            B: R.Tensor([16, 16], "float32"),
            fset_output: R.Callable([T.int64, R.Any], R.Tuple([]), purity=False),
        ):
            C = R.multiply(A, R.const(2, "float32"))
            D = R.add(C, B)
            fset_output(R.prim_value(0), D)
            fset_output(R.prim_value(1), D)
            return ()

    After = relax.transform.LazySetOutput()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_set_output_callback_with_inline_const():
    """fset_output may be called for inline objects

    The return tuple may contain inline leaf nodes, such as
    `Expr` or `tvm_tirx.ir.GenericConst`.  A call to `fset_output`
    must be generated, even though they do not have an associated
    `relax.VarBinding`.
    """

    @I.ir_module
    class Before:
        @R.function
        def transform_params(A: R.Tensor([16, 16], "float32"), B: R.Tensor([16, 16], "float32")):
            C = R.multiply(A, R.const(2, "float32"))
            D = R.add(C, B)
            return (C, D, R.prim_value(42), R.const(17.5, "float16"))

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def transform_params(
            A: R.Tensor([16, 16], "float32"),
            B: R.Tensor([16, 16], "float32"),
            fset_output: R.Callable([T.int64, R.Any], R.Tuple([]), purity=False),
        ):
            C = R.multiply(A, R.const(2, "float32"))
            fset_output(R.prim_value(0), C)
            D = R.add(C, B)
            fset_output(R.prim_value(1), D)
            fset_output(R.prim_value(2), R.prim_value(42))
            fset_output(R.prim_value(3), R.const(17.5, "float16"))
            return ()

    After = relax.transform.LazySetOutput()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_set_output_callback_with_non_tuple_output():
    """Non-tuple outputs produce a single call to fset_output"""

    @I.ir_module
    class Before:
        @R.function
        def transform_params(A: R.Tensor([16, 16], "float32"), B: R.Tensor([16, 16], "float32")):
            C = R.multiply(A, R.const(2, "float32"))
            D = R.add(C, B)
            return D

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def transform_params(
            A: R.Tensor([16, 16], "float32"),
            B: R.Tensor([16, 16], "float32"),
            fset_output: R.Callable([T.int64, R.Any], R.Tuple([]), purity=False),
        ):
            C = R.multiply(A, R.const(2, "float32"))
            D = R.add(C, B)
            fset_output(R.prim_value(0), D)
            return ()

    After = relax.transform.LazySetOutput()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
