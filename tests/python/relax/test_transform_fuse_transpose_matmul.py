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
# pylint: disable=invalid-name, missing-docstring

import numpy as np

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_transform_fuse_transpose_matmul():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((128, 256), "float32"),
            w: R.Tensor((128, 256), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            with R.dataflow():
                wT = R.permute_dims(w, [1, 0])
                o = R.matmul(x, wT)
                R.output(o)
            return o

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def NT_matmul(
            x: T.Tensor((T.int64(128), T.int64(256)), "float32"),
            w: T.Tensor((T.int64(128), T.int64(256)), "float32"),
            NT_matmul: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, k in T.grid(T.int64(128), T.int64(128), T.int64(256)):
                with Ts.sblock("NT_matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(x[v_i0, v_k], w[v_i1, v_k])
                    Ts.writes(NT_matmul[v_i0, v_i1])
                    with Ts.init():
                        NT_matmul[v_i0, v_i1] = T.float32(0)
                    NT_matmul[v_i0, v_i1] = NT_matmul[v_i0, v_i1] + x[v_i0, v_k] * w[v_i1, v_k]

        @R.function
        def main(
            x: R.Tensor((128, 256), dtype="float32"), w: R.Tensor((128, 256), dtype="float32")
        ) -> R.Tensor((128, 128), dtype="float32"):
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(
                    cls.NT_matmul, (x, w), ty_args=[R.Tensor((128, 128), dtype="float32")]
                )
                R.output(gv)
            return gv

    after = tvm_tirx.ir.transform.Sequential(
        [
            relax.transform.FuseTransposeMatmul(),
            relax.transform.FuseTIR(),  # Only used for remove unused primitive function
        ]
    )(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_transform_fuse_transpose_matmul_const():
    w = relax.const(np.random.uniform(-1e-3, 1e-3, (128, 256)), "float32")

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((128, 256), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            with R.dataflow():
                wT = R.permute_dims(w, [1, 0])
                o = R.matmul(x, wT)
                R.output(o)
            return o

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def NT_matmul(
            x: T.Tensor((T.int64(128), T.int64(256)), "float32"),
            w: T.Tensor((T.int64(128), T.int64(256)), "float32"),
            NT_matmul: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, k in T.grid(T.int64(128), T.int64(128), T.int64(256)):
                with Ts.sblock("NT_matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(x[v_i0, v_k], w[v_i1, v_k])
                    Ts.writes(NT_matmul[v_i0, v_i1])
                    with Ts.init():
                        NT_matmul[v_i0, v_i1] = T.float32(0)
                    NT_matmul[v_i0, v_i1] = NT_matmul[v_i0, v_i1] + x[v_i0, v_k] * w[v_i1, v_k]

        @R.function
        def main(x: R.Tensor((128, 256), dtype="float32")) -> R.Tensor((128, 128), dtype="float32"):
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(
                    cls.NT_matmul, (x, w), ty_args=[R.Tensor((128, 128), dtype="float32")]
                )
                R.output(gv)
            return gv

    after = tvm_tirx.ir.transform.Sequential(
        [
            relax.transform.FuseTransposeMatmul(),
            relax.transform.FuseTIR(),  # Only used for remove unused primitive function
        ]
    )(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def _execute(module):
    inputs = []
    for parameter in module["main"].params:
        shape = tuple(int(dim) for dim in parameter.ty.shape)
        values = np.arange(1, int(np.prod(shape)) + 1, dtype="float32").reshape(shape) / 10
        inputs.append(tvm_tirx.runtime.tensor(values))
    executable = relax.build(module, target="llvm")
    virtual_machine = relax.VirtualMachine(executable, tvm_tirx.cpu())
    return virtual_machine["main"](*inputs).numpy()


def _assert_fusion_skipped(before):
    """The pass must not crash, and must preserve both structure and numerics."""
    after = relax.transform.FuseTransposeMatmul()(before)
    tvm_tirx.ir.assert_structural_equal(after, before)
    np.testing.assert_array_equal(_execute(before), _execute(after))


def test_fuse_transpose_matmul_rank1_weight():
    """permute_dims of a vector is an identity and has no second-to-last axis."""

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((2,), "float32"),
            weight: R.Tensor((2,), "float32"),
        ) -> R.Tensor((), "float32"):
            with R.dataflow():
                transposed = R.permute_dims(weight)
                product = R.matmul(x, transposed)
                R.output(product)
            return product

    _assert_fusion_skipped(Before)


def test_fuse_transpose_matmul_reused_transposed_operand():
    """matmul(t, t) reuses one transposed value and is not an NT fusion."""

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((2, 2), "float32")) -> R.Tensor((2, 2), "float32"):
            with R.dataflow():
                t = R.permute_dims(x, axes=[1, 0])
                out = R.matmul(t, t)
                R.output(out)
            return out

    module = relax.transform.ConvertToDataflow(min_size=1)(Before)
    _assert_fusion_skipped(module)


def test_fuse_transpose_matmul_reused_weight_operand():
    """matmul(w, permute_dims(w)) shares the pre-transpose tensor across operands."""

    @I.ir_module
    class Before:
        @R.function
        def main(w: R.Tensor((2, 2), "float32")) -> R.Tensor((2, 2), "float32"):
            with R.dataflow():
                w_t = R.permute_dims(w, axes=[1, 0])
                out = R.matmul(w, w_t)
                R.output(out)
            return out

    _assert_fusion_skipped(Before)


if __name__ == "__main__":
    tvm_tirx.testing.main()
