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
# ruff: noqa: E501, F401, F841
import tvm
import tvm_tirx
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_rewrite_Select():
    @I.ir_module
    class ModuleY:
        @Ts.function
        def main(i: T.int32):
            A = T.alloc_tensor(ty_args=[T.Tensor((100,), "float32")])
            T.evaluate(T.Select(i > 1, A[i - 1], T.float32(1.0)))

    yy = tvm.s_tir.transform.RewriteUnsafeSelect()(ModuleY)["main"].body.seq[-1].value

    @I.ir_module
    class ModuleZ:
        @Ts.function
        def main(i: T.int32):
            A = T.alloc_tensor(ty_args=[T.Tensor((100,), "float32")])
            T.evaluate(
                T.Select(
                    T.Select(i > 1, A[i - 1], T.float32(1.0)) > T.float32(0.0), A[i], T.float32(0.1)
                )
            )

    zz = tvm.s_tir.transform.RewriteUnsafeSelect()(ModuleZ)["main"].body.seq[-1].value

    @I.ir_module
    class ModuleA:
        @Ts.function
        def main(i: T.int32):
            A = T.alloc_tensor(ty_args=[T.Tensor((100,), "float32")])
            # Inline y and z to avoid Let bindings - outer Select condition is safe (no tensor access)
            T.evaluate(
                T.Select(
                    T.floordiv(i, 4) > 10,
                    T.Select(i > 1, A[i - 1], T.float32(1.0)),
                    T.Select(
                        T.Select(i > 1, A[i - 1], T.float32(1.0)) > T.float32(0.0),
                        A[i],
                        T.float32(0.1),
                    ),
                )
            )

    aa = tvm.s_tir.transform.RewriteUnsafeSelect()(ModuleA)["main"].body.seq[-1].value
    builtin_if_then_else = tvm_tirx.ir.Op.get("prim.if_then_else")

    assert yy.op.same_as(builtin_if_then_else)
    assert yy.op.same_as(builtin_if_then_else)
    assert isinstance(aa, tvm_tirx.tirx.Select)


def test_scalar_address_survives_generic_transforms():
    @I.ir_module
    class Before:
        @Ts.function
        def main(value: T.uint32, condition: T.bool):
            T.evaluate(T.Select(condition, T.isnullptr(T.address_of(value)), T.bool(False)))

    transforms = [
        tvm.s_tir.transform.RewriteUnsafeSelect(),
        tvm.s_tir.transform.ThreadSync("shared"),
        tvm.s_tir.transform.MergeSharedMemoryAllocations(),
        tvm_tirx.tirx.transform.StorageRewrite(),
    ]
    for transform in transforms:
        transformed = transform(Before)
        assert transformed["main"] is not None


if __name__ == "__main__":
    test_rewrite_Select()
