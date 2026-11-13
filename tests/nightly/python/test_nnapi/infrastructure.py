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
# ruff: noqa: RUF005

import numpy as np

import tvm
import tvm_tirx
import tvm_tirx.script.relax as R
from tvm.relax.backend.contrib.nnapi import partition_for_nnapi
from tvm_tirx.support import ndk, utils


# pylint: disable=import-outside-toplevel,missing-function-docstring
def reshape_matmul(mod: tvm_tirx.IRModule):
    from tvm.relax import Expr
    from tvm.relax.dpl import DFPattern, rewrite_call
    from tvm.relax.dpl.pattern import is_op, wildcard

    input0 = wildcard()
    input1 = wildcard()
    pattern = is_op("relax.matmul")(input0, input1)

    def _rewriter(expr: Expr, matches: dict[DFPattern, Expr]):
        i0 = matches[input0]
        i1 = matches[input1]
        if len(i0.ty.shape) == 2 and len(i1.ty.shape) == 2:
            i0_shape = [1] + [*i0.ty.shape.values]
            i1_shape = [1] + [*i1.ty.shape.values]
            oshape = matches[pattern].ty.shape
            return R.reshape(R.matmul(R.reshape(i0, i0_shape), R.reshape(i1, i1_shape)), oshape)
        return expr

    mod["main"] = rewrite_call(pattern, _rewriter, mod["main"])
    return mod


def decompose_clip(mod: tvm_tirx.IRModule) -> tvm_tirx.IRModule:
    from tvm.relax import Expr
    from tvm.relax.dpl import DFPattern, rewrite_call
    from tvm.relax.dpl.pattern import is_op, wildcard

    input_pattern = wildcard()
    min_pattern = wildcard()
    max_pattern = wildcard()
    pattern = is_op("relax.clip")(input_pattern, min_pattern, max_pattern)

    def _rewriter(expr: Expr, matches: dict[DFPattern, Expr]) -> Expr:  # pylint: disable=unused-argument
        dtype = matches[input_pattern].ty.dtype
        return R.minimum(
            R.maximum(
                matches[input_pattern],
                R.const(np.array(matches[min_pattern].value.value).astype(dtype), dtype),
            ),
            R.const(np.array(matches[max_pattern].value.value).astype(dtype), dtype),
        )

    mod["main"] = rewrite_call(pattern, _rewriter, mod["main"])
    return mod


def _build(mod, enable_nnapi):
    if isinstance(mod, tvm_tirx.ir.Call):
        mod = tvm_tirx.IRModule.from_expr(mod)

    if enable_nnapi:
        mod = tvm.relax.transform.FoldConstant()(mod)
        mod = reshape_matmul(mod)
        mod = decompose_clip(mod)
        mod = partition_for_nnapi(mod)

        mod = tvm.relax.transform.RunCodegen()(mod)
    ex = tvm.compile(mod, target={"kind": "llvm", "mtriple": "aarch64-linux-android"})

    return ex


def _run(remote, tracker, ex, inputs):
    tmp = utils.tempdir()
    so_name = "test_mod.so"
    so_path = tmp / so_name
    ex.export_library(str(so_path), fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"])

    remote.upload(so_path)
    dev = remote.cpu(0)

    try:
        # Execute the model on the remote.
        remote_ex = remote.load_module(so_name)
        vm = tvm.rpc.virtual_machine(remote_ex, dev)

        inputs = [dev.tensor(x) for x in inputs]

        vm.set_input("main", *inputs)
        vm.invoke_stateful("main")
        output = vm.get_outputs("main")
        output = output.numpy()
    except Exception as e:
        # Re-raise all exceptions
        raise e
    finally:
        # Manually close the connection.
        # See https://discuss.tvm.apache.org/t/trouble-with-rpc-session/14008/.
        #
        # TODO: Remove if it does not happen on Python 3.11.
        remote._sess.get_function("CloseRPCConnection")()
        tracker.close()
        pass

    return output


def build_and_run(
    remote,
    tracker,
    mod,
    inputs,
    enable_nnapi=False,
):
    ex = _build(mod, enable_nnapi)
    return _run(remote, tracker, ex, inputs)
