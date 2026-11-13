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
# ruff: noqa: F401
import numpy as np

import tvm
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.script import ir_builder as relax_builder
from tvm.relax.transform import CombineParallelMatmul
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.script.ir_builder import IRBuilder


def test_param():
    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def matmul(
            A: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            B: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            C: T.Tensor((T.int64(32), T.int64(32)), "float32"),
        ):
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with Ts.sblock("C"):
                    with Ts.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(x: R.Tensor((32, 32), "float32"), y: R.Tensor((32, 32), "float32")):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                gv = R.call_tir(cls.matmul, (x, y), ty_args=[R.Tensor((32, 32), "float32")])
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def matmul1(
            A: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            B: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            C: T.Tensor((T.int64(32), T.int64(32)), "float32"),
        ):
            T.func_attr({"layout_free_tensors": [1]})
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with Ts.sblock("C"):
                    with Ts.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(x: R.Tensor((32, 32), "float32"), y: R.Tensor((32, 32), "float32")):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(cls.matmul1, (x, y), ty_args=[R.Tensor((32, 32), "float32")])
                R.output(gv)
            return gv

    after = relax.transform.AttachAttrLayoutFreeTensors()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_const():
    const_value = np.ones((32, 32), dtype="float32")

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def matmul(
            A: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            B: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            C: T.Tensor((T.int64(32), T.int64(32)), "float32"),
        ):
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with Ts.sblock("C"):
                    with Ts.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(x: R.Tensor((32, 32), "float32")):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                gv = R.call_tir(
                    cls.matmul,
                    (x, relax.const(const_value)),
                    ty_args=[R.Tensor((32, 32), "float32")],
                )
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def matmul1(
            A: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            B: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            C: T.Tensor((T.int64(32), T.int64(32)), "float32"),
        ):
            T.func_attr({"layout_free_tensors": [1]})
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with Ts.sblock("C"):
                    with Ts.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(x: R.Tensor((32, 32), "float32")):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(
                    cls.matmul1,
                    (x, relax.const(const_value)),
                    ty_args=[R.Tensor((32, 32), "float32")],
                )
                R.output(gv)
            return gv

    after = relax.transform.AttachAttrLayoutFreeTensors()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_multiple_same_func():
    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def matmul(
            A: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            B: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            C: T.Tensor((T.int64(32), T.int64(32)), "float32"),
        ):
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with Ts.sblock("C"):
                    with Ts.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(
            x: R.Tensor((32, 32), "float32"),
            w1: R.Tensor((32, 32), "float32"),
            w2: R.Tensor((32, 32), "float32"),
        ):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                lv1 = R.call_tir(
                    cls.matmul,
                    (x, w1),
                    ty_args=[R.Tensor((32, 32), "float32")],
                )
                gv = R.call_tir(
                    cls.matmul,
                    (lv1, w2),
                    ty_args=[R.Tensor((32, 32), "float32")],
                )
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def matmul1(
            A: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            B: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            C: T.Tensor((T.int64(32), T.int64(32)), "float32"),
        ):
            T.func_attr({"layout_free_tensors": [1]})
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with Ts.sblock("C"):
                    with Ts.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(
            x: R.Tensor((32, 32), "float32"),
            w1: R.Tensor((32, 32), "float32"),
            w2: R.Tensor((32, 32), "float32"),
        ):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                lv1 = R.call_tir(
                    cls.matmul1,
                    (x, w1),
                    ty_args=[R.Tensor((32, 32), "float32")],
                )
                gv = R.call_tir(
                    cls.matmul1,
                    (lv1, w2),
                    ty_args=[R.Tensor((32, 32), "float32")],
                )
                R.output(gv)
            return gv

    after = relax.transform.AttachAttrLayoutFreeTensors()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_multiple_same_func_with_different_free_buffers():
    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def matmul(
            A: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            B: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            C: T.Tensor((T.int64(32), T.int64(32)), "float32"),
        ):
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with Ts.sblock("C"):
                    with Ts.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(
            x: R.Tensor((32, 32), "float32"),
            w1: R.Tensor((32, 32), "float32"),
            w2: R.Tensor((32, 32), "float32"),
        ):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                lv1 = R.call_tir(
                    cls.matmul,
                    (x, w1),
                    ty_args=[R.Tensor((32, 32), "float32")],
                )
                gv = R.call_tir(
                    cls.matmul,
                    (w2, lv1),
                    ty_args=[R.Tensor((32, 32), "float32")],
                )
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def matmul1(
            A: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            B: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            C: T.Tensor((T.int64(32), T.int64(32)), "float32"),
        ):
            T.func_attr({"layout_free_tensors": [1]})
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with Ts.sblock("C"):
                    with Ts.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @Ts.function(private=True)
        def matmul2(
            A: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            B: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            C: T.Tensor((T.int64(32), T.int64(32)), "float32"),
        ):
            T.func_attr({"layout_free_tensors": [0]})
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with Ts.sblock("C"):
                    with Ts.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(
            x: R.Tensor((32, 32), "float32"),
            w1: R.Tensor((32, 32), "float32"),
            w2: R.Tensor((32, 32), "float32"),
        ):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                lv1 = R.call_tir(
                    cls.matmul1,
                    (x, w1),
                    ty_args=[R.Tensor((32, 32), "float32")],
                )
                gv = R.call_tir(
                    cls.matmul2,
                    (w2, lv1),
                    ty_args=[R.Tensor((32, 32), "float32")],
                )
                R.output(gv)
            return gv

    after = relax.transform.AttachAttrLayoutFreeTensors()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
