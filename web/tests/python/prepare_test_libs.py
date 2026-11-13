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
# Prepare test library for standalone wasm runtime test.

import os

import tvm_tirx
from tvm import relax, te
from tvm.contrib import emcc
from tvm.relax import script as R
from tvm_tirx.script import ir as I


def prepare_relax_lib(base_path):
    pipeline = relax.get_pipeline()
    n = I.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Mod:
        @R.function
        def main(x: R.Tensor([n], "float32"), y: R.Tensor([n], "float32")):
            lv0 = R.add(x, y)
            return lv0

    target = tvm_tirx.target.Target({"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"})

    mod = pipeline(Mod)
    ex = relax.build(mod, target)
    wasm_path = os.path.join(base_path, "test_relax.wasm")
    ex.export_library(wasm_path, fcompile=emcc.create_wasm)


def prepare_tir_lib(base_path):
    target = {"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
    if not tvm_tirx.runtime.enabled("llvm"):
        raise RuntimeError(f"Target {target} is not enbaled")
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    mod = tvm_tirx.IRModule.from_expr(
        te.create_function([A, B]).with_attr("global_symbol", "add_one")
    ).with_attr("system_lib_prefix", "")

    fadd = tvm_tirx.tirx.compile(mod, target)
    wasm_path = os.path.join(base_path, "test_addone.wasm")
    fadd.export_library(wasm_path, fcompile=emcc.create_wasm)


if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    base_path = os.path.join(curr_path, "../../dist/wasm")
    prepare_tir_lib(base_path)
    prepare_relax_lib(base_path)
