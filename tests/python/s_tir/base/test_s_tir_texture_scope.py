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

import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.backend.adreno import pipeline as adreno_pipeline
from tvm_tirx import tirx
from tvm_tirx.ir.module import IRModule
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx._compile import _split_host_device_modules
from tvm_tirx.tirx.compilation_pipeline import finalize_device_passes


def test_texture_scope():
    @tvm_tirx.script.ir_module
    class PlusOneMultTwo:
        @Ts.function
        def main(
            A: T.Tensor((128, 128, 4), dtype="float32", scope="global.texture"),
            C: T.Tensor((128, 128, 4), dtype="float32", scope="global.texture"),
        ) -> None:
            T.func_attr({"tirx.noalias": True})

            B = Ts.sblock_alloc_tensor((128, 128, 4), dtype="float32", scope="global.texture")

            for block_idx in T.parallel(0, 128, annotations={"thread_binding": "blockIdx.x"}):
                for thread_idx in T.parallel(0, 128, annotations={"thread_binding": "threadIdx.x"}):
                    for k in T.serial(4):
                        with Ts.sblock("B"):
                            vb, vt, vk = Ts.axis.remap("SSS", [block_idx, thread_idx, k])
                            B[vb, vt, vk] = A[vb, vt, vk] + T.float32(1)
            for block_idx in T.parallel(0, 128, annotations={"thread_binding": "blockIdx.x"}):
                for thread_idx in T.parallel(0, 128, annotations={"thread_binding": "threadIdx.x"}):
                    for k in T.serial(4):
                        with Ts.sblock("C"):
                            vb, vt, vk = Ts.axis.remap("SSS", [block_idx, thread_idx, k])
                            C[vb, vt, vk] = B[vb, vt, vk] * T.float32(2)

    sch = tvm.s_tir.Schedule(PlusOneMultTwo, debug_mask="all")

    def schedule_block(block):
        _, _, inner = sch.get_loops(block)
        sch.vectorize(inner)

    schedule_block(sch.get_sblock("B"))
    schedule_block(sch.get_sblock("C"))

    target = tvm_tirx.target.Target({"kind": "opencl", "keys": ["adreno"]})
    lowered = tirx.transform.BindTarget(target.with_host("c"))(sch.mod)
    pipeline, _, _ = adreno_pipeline.default_tir_pipeline()
    lowered = pipeline(lowered)
    _, device_mods = _split_host_device_modules(lowered)
    assert len(device_mods) == 1
    (device_target, _backend_config), device_mod = next(iter(device_mods.items()))
    device_mod = finalize_device_passes()(device_mod)
    source = tvm_ffi.get_global_func("target.build.opencl")(
        device_mod, device_target
    ).inspect_source()
    assert "__read_only image2d_array_t" in source
    assert "__write_only image2d_array_t" in source
    assert "READ_IMAGEF" in source
    assert "write_imagef" in source


if __name__ == "__main__":
    tvm_tirx.testing.main()
