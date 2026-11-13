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

from __future__ import annotations

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.relax import VDevice
from tvm.relax.transform import UpdateVDevice
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.script.parser import tirx as T


def verify(input, new_vdevice, vdevice_index, expected):
    tvm_tirx.ir.assert_structural_equal(UpdateVDevice(new_vdevice, vdevice_index)(input), expected)


def test_update():
    vdevices = [
        VDevice("llvm"),
        VDevice("cuda", 0),
        VDevice("metal", 0, "global"),
        VDevice({"kind": "cuda", "arch": "sm_80"}, 0),
        VDevice("metal", 1, "global"),
        VDevice("llvm", 1),
    ]

    @I.ir_module
    class Input1:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    R.vdevice("llvm"),
                    R.vdevice("cuda", 0),
                    R.vdevice("metal", 0, "global"),
                    R.vdevice({"kind": "cuda", "arch": "sm_80"}, 0),
                ]
            }
        )

        @R.function
        def main(
            a: R.Tensor((128, 128), "float32", "cuda:1"),
            c: R.Tensor((128, 128), "float32", "vdevice:3"),
        ) -> R.Tensor((128, 128), "float32"):
            s = R.add(a, c)
            return s

    @I.ir_module
    class Expect1:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    R.vdevice("llvm"),
                    R.vdevice("cuda", 0),
                    R.vdevice("metal", 0, "global"),
                    R.vdevice("metal", 1, "global"),
                ]
            }
        )

        @R.function
        def main(
            a: R.Tensor((128, 128), dtype="float32", vdevice="metal:1"),
            c: R.Tensor((128, 128), dtype="float32", vdevice="metal:1"),
        ) -> R.Tensor((128, 128), dtype="float32", vdevice="metal:1"):
            s: R.Tensor((128, 128), dtype="float32", vdevice="metal:1") = R.add(a, c)
            return s

    @I.ir_module
    class Input2:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    R.vdevice("llvm"),
                    R.vdevice("cuda", 0),
                ]
            }
        )

        @R.function
        def main(
            a: R.Tensor((128, 128), "float32", "cuda:0"),
            c: R.Tensor((128, 128), "float32", "cuda:0"),
        ) -> R.Tensor((128, 128), "float32"):
            s = R.add(a, c)
            return s

    @I.ir_module
    class Expect2:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    R.vdevice("llvm"),
                    R.vdevice("llvm", 1),
                ]
            }
        )

        @R.function
        def main(
            a: R.Tensor((128, 128), "float32", "llvm:1"),
            c: R.Tensor((128, 128), "float32", "llvm:1"),
        ) -> R.Tensor((128, 128), "float32", "llvm:1"):
            s: R.Tensor((128, 128), "float32", "llvm:1") = R.add(a, c)
            return s

    verify(Input1, vdevices[4], 3, Expect1)
    verify(Input2, vdevices[5], 1, Expect2)


if __name__ == "__main__":
    tvm_tirx.testing.main()
