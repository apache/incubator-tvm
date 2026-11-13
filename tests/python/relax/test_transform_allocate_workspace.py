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
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx.script import ir as I


@I.ir_module
class Module:
    @R.function
    def fused_relax_nn_attention_cutlass(
        q: R.Tensor((32, 8, 16, 8), dtype="float16"),
        k: R.Tensor((32, 8, 16, 8), dtype="float16"),
        v: R.Tensor((32, 8, 16, 8), dtype="float16"),
    ) -> R.Tensor((32, 8, 16, 8), dtype="float16"):
        R.func_attr(
            {
                "Codegen": "cutlass",
                "WorkspaceSize": 65536,
                "global_symbol": "fused_relax_nn_attention_cutlass",
            }
        )

        @R.function
        def gv(
            q_1: R.Tensor((32, 8, 16, 8), dtype="float16"),
            k_1: R.Tensor((32, 8, 16, 8), dtype="float16"),
            v_1: R.Tensor((32, 8, 16, 8), dtype="float16"),
        ) -> R.Tensor((32, 8, 16, 8), dtype="float16"):
            R.func_attr(
                {"Composite": "cutlass.attention", "Primitive": True, "WorkspaceSize": 65536}
            )
            with R.dataflow():
                gv_2: R.Tensor((32, 8, 16, 8), dtype="float16") = R.nn.attention(
                    q_1, k_1, v_1, scale=None
                )
                R.output(gv_2)
            return gv_2

        gv1: R.Tensor((32, 8, 16, 8), dtype="float16") = gv(q, k, v)
        return gv1

    @R.function
    def entry_a(
        q: R.Tensor((32, 8, 16, 8), dtype="float16"),
        k: R.Tensor((32, 8, 16, 8), dtype="float16"),
        v: R.Tensor((32, 8, 16, 8), dtype="float16"),
    ) -> R.Tensor((32, 8, 16, 8), dtype="float16"):
        cls = Module
        with R.dataflow():
            gv: R.Tensor((32, 8, 16, 8), dtype="float16") = cls.fused_relax_nn_attention_cutlass(
                q, k, v
            )
            R.output(gv)
        return gv

    @R.function
    def entry_b(
        q: R.Tensor((32, 8, 16, 8), dtype="float16"),
        k: R.Tensor((32, 8, 16, 8), dtype="float16"),
        v: R.Tensor((32, 8, 16, 8), dtype="float16"),
    ) -> R.Tensor((32, 8, 16, 8), dtype="float16"):
        cls = Module
        with R.dataflow():
            gv: R.Tensor((32, 8, 16, 8), dtype="float16") = cls.fused_relax_nn_attention_cutlass(
                q, k, v
            ) + R.const(1, dtype="float16")
            R.output(gv)
        return gv


@I.ir_module
class Expected:
    @R.function
    def fused_relax_nn_attention_cutlass1(
        q: R.Tensor((32, 8, 16, 8), dtype="float16"),
        k: R.Tensor((32, 8, 16, 8), dtype="float16"),
        v: R.Tensor((32, 8, 16, 8), dtype="float16"),
        workspace: R.Tensor((65536,), dtype="uint8"),
    ) -> R.Tensor((32, 8, 16, 8), dtype="float16"):
        R.func_attr(
            {
                "Codegen": "cutlass",
                "global_symbol": "fused_relax_nn_attention_cutlass1",
            }
        )

        @R.function
        def gv(
            q_1: R.Tensor((32, 8, 16, 8), dtype="float16"),
            k_1: R.Tensor((32, 8, 16, 8), dtype="float16"),
            v_1: R.Tensor((32, 8, 16, 8), dtype="float16"),
            workspace_1: R.Tensor((65536,), dtype="uint8"),
        ) -> R.Tensor((32, 8, 16, 8), dtype="float16"):
            R.func_attr({"Composite": "cutlass.attention", "Primitive": True})
            with R.dataflow():
                gv_2: R.Tensor((32, 8, 16, 8), dtype="float16") = R.nn.attention(
                    q_1, k_1, v_1, scale=None
                )
                R.output(gv_2)
            return gv_2

        gv1: R.Tensor((32, 8, 16, 8), dtype="float16") = gv(q, k, v, workspace)
        return gv1

    @R.function
    def entry_a(
        q: R.Tensor((32, 8, 16, 8), dtype="float16"),
        k: R.Tensor((32, 8, 16, 8), dtype="float16"),
        v: R.Tensor((32, 8, 16, 8), dtype="float16"),
    ) -> R.Tensor((32, 8, 16, 8), dtype="float16"):
        cls = Expected
        with R.dataflow():
            workspace_main: R.Tensor((65536,), dtype="uint8") = R.builtin.alloc_tensor(
                R.shape([65536]), R.dtype("uint8"), R.prim_value(0)
            )
            gv: R.Tensor((32, 8, 16, 8), dtype="float16") = cls.fused_relax_nn_attention_cutlass1(
                q, k, v, workspace_main
            )
            R.output(gv)
        return gv

    @R.function
    def entry_b(
        q: R.Tensor((32, 8, 16, 8), dtype="float16"),
        k: R.Tensor((32, 8, 16, 8), dtype="float16"),
        v: R.Tensor((32, 8, 16, 8), dtype="float16"),
    ) -> R.Tensor((32, 8, 16, 8), dtype="float16"):
        cls = Expected
        with R.dataflow():
            workspace_main: R.Tensor((65536,), dtype="uint8") = R.builtin.alloc_tensor(
                R.shape([65536]), R.dtype("uint8"), R.prim_value(0)
            )
            gv: R.Tensor((32, 8, 16, 8), dtype="float16") = cls.fused_relax_nn_attention_cutlass1(
                q, k, v, workspace_main
            ) + R.const(1, dtype="float16")
            R.output(gv)
        return gv


def test_single_attention():
    rewritten = relax.transform.AllocateWorkspace()(Module)
    tvm_tirx.ir.assert_structural_equal(rewritten, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
