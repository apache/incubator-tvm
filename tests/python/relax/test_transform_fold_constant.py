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

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.ir import TupleType as _TupleType
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def gen_mod(mod, name, binding):
    """Select relax function with name, rename to main and bind constant.

    Parameters
    ----------
    mod: IRModule
        The input module

    name: str
        The name of relax function to preserve and rename to main

    binding: Dict[str, array]
        The const parameter bindings
    """
    funcs = {}
    binding = {k: tvm_tirx.runtime.tensor(v) for k, v in binding.items()}

    for k, v in mod.functions.items():
        if isinstance(v, tvm.relax.Function):
            if k.name_hint == name:
                # rename to main
                gv = tvm_tirx.ir.GlobalVar("main")
                funcs[gv] = tvm.relax.Function(v.params, v.body, v.ret_ty).with_attr(
                    "global_symbol", "main"
                )
        else:
            funcs[k] = v
    mod = tvm_tirx.IRModule(funcs)
    return relax.transform.BindParams("main", binding)(mod)


def test_one_fold_addone():
    # put before after in a single module
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def addone(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")) -> None:
            for i, j in T.grid(16, 16):
                with Ts.sblock("addone"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.float32(1)

        @R.function
        def before(c0: R.Tensor((16, 16), "float32")):
            cls = Module
            lv0 = relax.call_tir(cls.addone, (c0,), ty_args=[R.Tensor((16, 16), dtype="float32")])
            return lv0

        @R.function
        def expected(c1: R.Tensor((16, 16), "float32")):
            return c1

    c0_np = np.arange(16 * 16).astype("float32").reshape(16, 16)
    c1_np = c0_np + 1
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np})

    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_one_fold_transpose():
    # put before after in a single module
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def func(A: T.Tensor((2, 3), "float32"), B: T.Tensor((3, 2), "float32")) -> None:
            for i, j in T.grid(3, 2):
                with Ts.sblock("transpose"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vj, vi]

        @R.function
        def before(c0: R.Tensor((2, 3), "float32")):
            cls = Module
            lv0 = relax.call_tir(cls.func, (c0,), ty_args=[R.Tensor((3, 2), dtype="float32")])
            return lv0

        @R.function
        def expected(c1: R.Tensor((3, 2), "float32")):
            return c1

    c0_np = np.arange(2 * 3).astype("float32").reshape(2, 3)
    c1_np = c0_np.T
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np})

    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_two_hop_addone():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def addone(A: T.Tensor((2, 2), "float32"), B: T.Tensor((2, 2), "float32")) -> None:
            for i, j in T.grid(2, 2):
                with Ts.sblock("addone"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.float32(1)

        @R.function
        def before(c0: R.Tensor((2, 2), "float32")):
            cls = Module
            lv0 = relax.call_tir(cls.addone, (c0,), ty_args=[R.Tensor((2, 2), dtype="float32")])
            lv1 = relax.call_tir(cls.addone, (lv0,), ty_args=[R.Tensor((2, 2), dtype="float32")])
            return lv1

        @R.function
        def expected(c1: R.Tensor((2, 2), "float32"), c2: R.Tensor((2, 2), "float32")):
            return c2

    c0_np = np.arange(2 * 2).astype("float32").reshape(2, 2)
    c1_np = c0_np + 1
    c2_np = c1_np + 1
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np, "c2": c2_np})

    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_dataflow_fold():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def identity(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")) -> None:
            for i, j in T.grid(16, 16):
                with Ts.sblock("identity"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]

        @R.function
        def before(c0: R.Tensor((16, 16), "float32")):
            cls = Module
            with R.dataflow():
                gv0 = relax.call_tir(
                    cls.identity, (c0,), ty_args=[R.Tensor((16, 16), dtype="float32")]
                )
                R.output(gv0)
            return gv0

        @R.function
        def expected(c1: R.Tensor((16, 16), "float32")):
            return c1

    c0_np = np.arange(16 * 16).astype("float32").reshape(16, 16)
    c1_np = c0_np
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np})
    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_fold_mixed_case():
    n_addone = T.dynamic("n", "int32")
    m_addone = T.dynamic("m", "int32")
    n_before = T.dynamic("n", "int64")
    m_before = T.dynamic("m", "int64")
    n_expected = T.dynamic("n", "int64")
    m_expected = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class Module:
        # TIR function can handle different cases.
        @Ts.function
        def addone(
            A: T.Tensor((n_addone, m_addone), "float32"),
            B: T.Tensor((n_addone, m_addone), "float32"),
        ) -> None:
            for i, j in T.grid(n_addone, m_addone):
                with Ts.sblock("addone"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.float32(1)

        @Ts.function
        def sub(
            A: T.Tensor((16, 16), "float32"),
            B: T.Tensor((16, 16), "float32"),
            C: T.Tensor((16, 16), "float32"),
        ) -> None:
            for i, j in T.grid(16, 16):
                with Ts.sblock("sub"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    C[vi, vj] = A[vi, vj] - B[vi, vj]

        @R.function
        def before(c0: R.Tensor((16, 16), "float32"), x: R.Tensor("float32", ndim=2)):
            cls = Module
            x0 = R.match_cast(x, R.Tensor((n_before, m_before), "float32"))
            # this line cannot be folded because n is unknown
            lv0 = relax.call_tir(
                cls.addone, (c0,), ty_args=[R.Tensor((n_before, 16), dtype="float32")]
            )
            # this line can be folded
            lv1 = relax.call_tir(cls.addone, (c0,), ty_args=[R.Tensor((16, 16), dtype="float32")])
            # this line can be folded because all inputs are const
            lv2 = relax.call_tir(cls.sub, (c0, lv1), ty_args=[R.Tensor((16, 16), dtype="float32")])
            # this line can not be folded because x's shape is unknown
            lv3 = relax.call_tir(cls.sub, (lv2, x), ty_args=[R.Tensor((16, 16), dtype="float32")])
            return (lv0, lv3)

        @R.function
        def expected(
            c0: R.Tensor((16, 16), "float32"),
            c1: R.Tensor((16, 16), "float32"),
            c2: R.Tensor((16, 16), "float32"),
            x: R.Tensor("float32", ndim=2),
        ):
            cls = Module
            x0 = R.match_cast(x, R.Tensor((n_expected, m_expected), "float32"))
            # this line cannot be folded because n is unknown
            lv0 = relax.call_tir(
                cls.addone, (c0,), ty_args=[R.Tensor((n_expected, 16), dtype="float32")]
            )
            # this line can not be folded because x's shape is unknown
            lv3 = relax.call_tir(cls.sub, (c2, x), ty_args=[R.Tensor((16, 16), dtype="float32")])
            return (lv0, lv3)

    c0_np = np.arange(16 * 16).astype("float32").reshape(16, 16)
    c1_np = c0_np + 1
    c2_np = c0_np - c1_np

    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c0": c0_np, "c1": c1_np, "c2": c2_np})
    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_int32_fold():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def addone(A: T.Tensor((16, 16), "int32"), B: T.Tensor((16, 16), "int32")) -> None:
            for i, j in T.grid(16, 16):
                with Ts.sblock("addone"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.int32(1)

        @R.function
        def before(c0: R.Tensor((16, 16), "int32")):
            cls = Module
            lv0 = relax.call_tir(cls.addone, (c0,), ty_args=[R.Tensor((16, 16), dtype="int32")])
            return lv0

        @R.function
        def expected(c1: R.Tensor((16, 16), "int32")):
            return c1

    c0_np = np.arange(16 * 16).astype("int32").reshape(16, 16)
    c1_np = c0_np + 1
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np})

    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_fold_single_relax_op():
    # put before after in a single module
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def before(c0: R.Tensor((16, 16), "float32")):
            with R.dataflow():
                gv = R.add(c0, c0)
                R.output(gv)
            return gv

        @R.function
        def expected(c1: R.Tensor((16, 16), "float32")):
            return c1

    c0_np = np.arange(16 * 16).astype("float32").reshape(16, 16)
    c1_np = c0_np + c0_np
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np})

    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_fold_multiple_relax_ops():
    # put before after in a single module
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def before(c0: R.Tensor((16, 16), "float32"), c1: R.Tensor((16, 16), "float32")):
            with R.dataflow():
                lv0 = R.add(c0, c1)
                lv1 = R.multiply(c0, lv0)
                gv = R.subtract(lv1, c1)
                R.output(gv)
            return gv

        @R.function
        def expected(c4: R.Tensor((16, 16), "float32")):
            return c4

    c0_np = np.arange(16 * 16).astype("float32").reshape(16, 16)
    c1_np = np.arange(16 * 16).astype("float32").reshape(16, 16)
    c2_np = c0_np + c1_np
    c3_np = c0_np * c2_np
    c4_np = c3_np - c1_np
    before = gen_mod(Module, "before", {"c0": c0_np, "c1": c1_np})
    expected = gen_mod(Module, "expected", {"c4": c4_np})

    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_do_not_fold_ops_outside_dataflow():
    # put before after in a single module
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def before(c0: R.Tensor((16, 16), "float32")):
            gv = R.add(c0, c0)
            return gv

    c0_np = np.arange(16 * 16).astype("float32").reshape(16, 16)
    before = gen_mod(Module, "before", {"c0": c0_np})

    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, before)


def test_fold_multiple_relax_ops_with_data_dependent_reshape():
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def before(
            data: R.Tensor((256,), "float32"),
            c0: R.Tensor((2,), "int64"),
            c1: R.Tensor((2,), "int64"),
        ):
            with R.dataflow():
                lv0 = R.add(c0, c0)
                target_shape = R.multiply(lv0, c1)
                lv2: R.Shape(ndim=2) = R.tensor_to_shape(target_shape)
                gv: R.Tensor(ndim=2, dtype="float32") = R.reshape(data, lv2)
                R.output(gv)
            return gv

        @R.function
        def expected(data: R.Tensor((256,), "float32")) -> R.Tensor((16, 16), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((16, 16), dtype="float32") = R.reshape(data, R.shape([16, 16]))
                R.output(gv)
            return gv

    c0_np = [8, 8]
    c1_np = [1, 1]
    before = gen_mod(Module, "before", {"c0": c0_np, "c1": c1_np})
    relax.analysis.check_well_formed(before)

    expected = gen_mod(Module, "expected", {})

    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_unsupported_fold_ops_legalized_to_multiple_calls():
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def before(c0: R.Tensor((16, 16), "float32")):
            with R.dataflow():
                gv = R.nn.relu(c0)
                R.output(gv)
            return gv

    c0_np = np.arange(16 * 16).astype("float32").reshape(16, 16)
    before = gen_mod(Module, "before", {"c0": c0_np})

    from tvm.relax.transform.legalize_ops.common import register_legalize

    def customized_legalize_relu(bb: relax.BlockBuilder, call: relax.Call):
        from tvm import topi  # pylint: disable=import-outside-toplevel

        x = bb.emit_te(topi.nn.relu, *call.args)
        return bb.call_te(topi.identity, x)

    # register custom legalization for relu that emits multiple bindings for testing
    relu_legalize = tvm_tirx.ir.Op.get("relax.nn.relu").get_attr("FLegalize")
    tvm_tirx.ir.Op.get("relax.nn.relu").reset_attr("FLegalize")
    register_legalize("relax.nn.relu", customized_legalize_relu)

    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, before)

    # revert to correct legalization of relu
    tvm_tirx.ir.Op.get("relax.nn.relu").reset_attr("FLegalize")
    register_legalize("relax.nn.relu", relu_legalize)


def test_fold_shape_computation():
    @I.ir_module
    class Module:
        @R.function
        def before(
            data: R.Tensor((5, 4, 3, 2), dtype="float32"),
            indices: R.Tensor((1,), dtype="int64"),
        ) -> R.Tensor((1, 1), dtype="int64"):
            with R.dataflow():
                lv: R.Tensor((4,), dtype="int64") = R.shape_to_tensor(R.shape([5, 4, 3, 2]))
                lv1: R.Tensor((1,), dtype="int64") = R.take(lv, indices, axis=0)
                lv2: R.Tensor((1, 1), dtype="int64") = R.expand_dims(lv1, axis=[0])
                gv: R.Tensor((1, 1), dtype="int64") = R.concat((lv2,), axis=0)
                R.output(gv)
            return gv

        @R.function
        def expected(
            data: R.Tensor((5, 4, 3, 2), dtype="float32"), new_shape: R.Tensor((1, 1), "int64")
        ) -> R.Tensor((1, 1), dtype="int64"):
            return new_shape

    before = gen_mod(
        Module, "before", {"indices": tvm_tirx.runtime.tensor(np.array([0]).astype("int64"))}
    )
    after = relax.transform.FoldConstant()(before)
    np_take = np.take([5, 4, 3, 2], [0], axis=0)
    np_expand = np.expand_dims(np_take, axis=[0])
    np_concat = np.concatenate([np_expand], axis=0)
    expected = gen_mod(Module, "expected", {"new_shape": tvm_tirx.runtime.tensor(np_concat)})
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_fold_tuple_output():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def split(
            A: T.Tensor((4, 4), "float32"),
            B: T.Tensor((2, 4), "float32"),
            C: T.Tensor((2, 4), "float32"),
        ) -> None:
            for i, j in T.grid(2, 4):
                with Ts.sblock("upper"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]
            for i, j in T.grid(2, 4):
                with Ts.sblock("lower"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    C[vi, vj] = A[vi + 2, vj]

        @R.function
        def before(c0: R.Tensor((4, 4), "float32")):
            cls = Module
            lv0 = relax.call_tir(
                cls.split,
                (c0,),
                ty_args=[
                    _TupleType(
                        [R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")]
                    )
                ],
            )
            return lv0

        @R.function
        def expected(c1: R.Tensor((2, 4), "float32"), c2: R.Tensor((2, 4), "float32")) -> R.Tuple(
            R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")
        ):
            lv0: R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")) = (
                c1,
                c2,
            )
            return lv0

    c0_np = np.arange(16).astype("float32").reshape(4, 4)
    c1_np = c0_np[:2]
    c2_np = c0_np[2:]
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np, "c2": c2_np})

    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_skip_folding_large_creation_op():
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def before():
            with R.dataflow():
                # 2048 elements > 1024 threshold, no tensor input
                gv = R.zeros((2048,), "float32")
                R.output(gv)
            return gv

    before = Module
    after = relax.transform.FoldConstant()(before)
    # The zeros op should NOT be folded because the output is large
    tvm_tirx.ir.assert_structural_equal(after, before)


def test_fold_small_creation_op():
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def before():
            with R.dataflow():
                # 16 elements <= 1024 threshold
                gv = R.zeros((4, 4), "float32")
                R.output(gv)
            return gv

        @R.function
        def expected(c0: R.Tensor((4, 4), "float32")):
            return c0

    before = gen_mod(Module, "before", {})
    expected = gen_mod(Module, "expected", {"c0": np.zeros((4, 4), dtype="float32")})
    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_fold_boundary_creation_op():
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def before():
            with R.dataflow():
                # Exactly 1024 elements == threshold, should fold
                gv = R.zeros((1024,), "float32")
                R.output(gv)
            return gv

        @R.function
        def expected(c0: R.Tensor((1024,), "float32")):
            return c0

    before = gen_mod(Module, "before", {})
    expected = gen_mod(Module, "expected", {"c0": np.zeros((1024,), dtype="float32")})
    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_fold_large_op_with_tensor_input():
    """Ops with tensor inputs should be folded even if output is large."""

    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def addone(A: T.Tensor((2048,), "float32"), B: T.Tensor((2048,), "float32")) -> None:
            for i in range(2048):
                with Ts.sblock("addone"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float32(1)

        @R.function
        def before(c0: R.Tensor((2048,), "float32")):
            cls = Module
            lv0 = relax.call_tir(cls.addone, (c0,), ty_args=[R.Tensor((2048,), dtype="float32")])
            return lv0

        @R.function
        def expected(c1: R.Tensor((2048,), "float32")):
            return c1

    c0_np = np.arange(2048).astype("float32")
    c1_np = c0_np + 1
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np})
    after = relax.transform.FoldConstant()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_call_tir_with_primitive_args_not_folded():
    """call_tir with symbolic primitive arguments cannot be const-evaluated."""

    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function(private=True)
        def shape_to_tensor(m: T.int64, out: T.Tensor((T.int64(1),), "int64")):
            for i in range(T.int64(1)):
                with Ts.sblock("out"):
                    vi = Ts.axis.remap("S", [i])
                    out[vi] = m

        @R.function
        def main(x: R.Tensor((m,), "float32")):
            cls = Module
            gv = relax.call_tir(cls.shape_to_tensor, (m,), ty_args=[R.Tensor((1,), "int64")])
            return gv

    after = relax.transform.FoldConstant()(Module)
    tvm_tirx.ir.assert_structural_equal(after, Module)


if __name__ == "__main__":
    tvm_tirx.testing.main()
