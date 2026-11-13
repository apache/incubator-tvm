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
import importlib.util

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax

target, dev = "llvm", tvm_tirx.cpu()


def _has_xcode():
    try:
        import tvm_tirx.support.xcode

        tvm_tirx.support.xcode.xcrun([])
        return True
    except FileNotFoundError:
        pass
    return False


requires_coreml_runtime = pytest.mark.skipif(
    not (importlib.util.find_spec("coremltools") and _has_xcode()),
    reason="coreml is not enabled.",
)


def test_partition_for_coreml_uses_current_relax_passes():
    from tvm.relax.backend.metal.coreml import partition_for_coreml

    x = relax.Var("x", relax.TensorType([10, 10], "float32"))
    y = relax.Var("y", relax.TensorType([10, 10], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.add(x, y))
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)

    partitioned = partition_for_coreml(bb.get())

    relax.analysis.check_well_formed(partitioned)
    assert any(
        getattr(func, "attrs", None) is not None
        and "Codegen" in func.attrs
        and str(func.attrs["Codegen"]) == "coreml"
        for func in partitioned.functions.values()
    )


def verify(mod, inputs):
    from tvm.relax.backend.metal.coreml import partition_for_coreml

    mod1 = partition_for_coreml(mod)
    mod1 = relax.transform.RunCodegen()(mod1)
    relax.analysis.check_well_formed(mod1)
    assert mod1.attrs, "Should exist if offloaded successfully."
    assert "external_mods" in mod1.attrs, "Should exist if offloaded successfully."
    mod1 = relax.transform.LegalizeOps()(mod1)
    relax.analysis.check_well_formed(mod1)

    ex1 = tvm.compile(mod1, target=target)
    vm1 = relax.VirtualMachine(ex1, dev)
    out1 = vm1["main"](*inputs)

    mod2 = relax.transform.LegalizeOps()(mod)
    ex2 = tvm.compile(mod2, target=target)
    vm2 = relax.VirtualMachine(ex2, dev)
    out2 = vm2["main"](*inputs)

    tvm_tirx.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-3, atol=1e-3)


@requires_coreml_runtime
def test_add():
    x = relax.Var("x", relax.TensorType([10, 10], "float32"))
    y = relax.Var("y", relax.TensorType([10, 10], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.add(x, y))
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)
    mod = bb.get()
    x_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10).astype("float32"), dev)
    y_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10).astype("float32"), dev)
    verify(mod, [x_data, y_data])


@requires_coreml_runtime
def test_add_const():
    x = relax.Var("x", relax.TensorType([10, 10], "float32"))
    y = relax.const(np.ones([10, 10]), "float32")
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.add(x, y))
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)
    mod = bb.get()
    x_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10).astype("float32"), dev)
    verify(mod, [x_data])


@requires_coreml_runtime
def test_multiply():
    x = relax.Var("x", relax.TensorType([10, 10], "float32"))
    y = relax.Var("y", relax.TensorType([10, 10], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.multiply(x, y))
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)
    mod = bb.get()

    x_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10).astype("float32"), dev)
    y_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10).astype("float32"), dev)
    verify(mod, [x_data, y_data])


@requires_coreml_runtime
def test_matmul():
    x = relax.Var("x", relax.TensorType([8, 10], "float32"))
    y = relax.const(tvm_tirx.runtime.tensor(np.random.rand(10, 8).astype("float32"), dev))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.matmul(x, y))
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)
    mod = bb.get()

    x_data = tvm_tirx.runtime.tensor(np.random.rand(8, 10).astype("float32"), dev)
    verify(mod, [x_data])

    x = relax.Var("x", relax.TensorType([8, 10], "float32"))
    y = relax.Var("y", relax.TensorType([10, 8], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.matmul(x, y))
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)
    mod = bb.get()

    x_data = tvm_tirx.runtime.tensor(np.random.rand(8, 10).astype("float32"), dev)
    y_data = tvm_tirx.runtime.tensor(np.random.rand(10, 8).astype("float32"), dev)
    verify(mod, [x_data, y_data])


@requires_coreml_runtime
def test_clip():
    x = relax.Var("x", relax.TensorType([10, 10], "float32"))
    bb = relax.BlockBuilder()

    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.clip(x, 0, 4))
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    mod = bb.get()

    x_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10).astype("float32"), dev)
    verify(mod, [x_data])

    x = relax.Var("x", relax.TensorType([10, 10], "float32"))
    bb = relax.BlockBuilder()

    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.clip(x, 0, 4))
            lv1 = bb.emit(relax.op.clip(x, 1, 3))
            gv0 = bb.emit_output(lv0)
            gv1 = bb.emit_output(lv1)
        bb.emit_func_output([gv0, gv1])

    x_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10).astype("float32"), dev)
    verify(mod, [x_data])


@requires_coreml_runtime
def test_expand_dims():
    def get_mod(axis):
        x = relax.Var("x", relax.TensorType([10, 10], "float32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit(relax.op.expand_dims(x, axis=axis))
                gv = bb.emit_output(lv0)
            bb.emit_func_output(gv)
        return bb.get()

    x_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10).astype("float32"), dev)
    verify(get_mod(axis=0), [x_data])
    verify(get_mod(axis=1), [x_data])


@requires_coreml_runtime
def test_relu():
    x = relax.Var("x", relax.TensorType([10, 10], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.nn.relu(x))
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)
    mod = bb.get()

    x_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10).astype("float32"), dev)
    verify(mod, [x_data])


@requires_coreml_runtime
def test_batch_flatten():
    x = relax.Var("x", relax.TensorType([10, 10, 10], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.nn.batch_flatten(x))
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)
    mod = bb.get()

    x_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10, 10).astype("float32"), dev)
    verify(mod, [x_data])


@requires_coreml_runtime
def test_softmax():
    x = relax.Var("x", relax.TensorType([10, 10], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.nn.softmax(x))
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)
    mod = bb.get()

    x_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10).astype("float32"), dev)
    verify(mod, [x_data])


@requires_coreml_runtime
def test_conv2d():
    x = relax.Var("x", relax.TensorType([1, 3, 224, 224], "float32"))
    w = relax.const(np.zeros((16, 3, 3, 3), dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.nn.conv2d(x, w, strides=[2, 2], padding=[1, 1, 1, 1]))
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)
    mod = bb.get()
    x_data = tvm_tirx.runtime.tensor(np.random.rand(1, 3, 224, 224).astype("float32"), dev)
    verify(mod, [x_data])


@requires_coreml_runtime
def test_global_avg_pool2d():
    x = relax.Var("x", relax.TensorType([1, 1, 10, 10], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.nn.avg_pool2d(x))
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)
    mod = bb.get()
    x_data = tvm_tirx.runtime.tensor(np.random.rand(1, 1, 10, 10).astype("float32"), dev)
    verify(mod, [x_data])


@requires_coreml_runtime
def test_subgraph1():
    x = relax.Var("x", relax.TensorType([10, 10], "float32"))
    y = relax.Var("y", relax.TensorType([10, 10], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.multiply(x, y))
            lv1 = bb.emit(relax.op.nn.softmax(lv0))
            gv = bb.emit_output(lv1)
        bb.emit_func_output(gv)
    mod = bb.get()
    x_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10).astype("float32"), dev)
    y_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10).astype("float32"), dev)
    verify(mod, [x_data, y_data])


@requires_coreml_runtime
def test_subgraph2():
    x = relax.Var("x", relax.TensorType([10, 10], "float32"))
    y = relax.Var("y", relax.TensorType([10, 10], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        with bb.dataflow():
            # multiply+relu will be offloaded to coreml
            lv0 = bb.emit(relax.op.multiply(x, y))
            lv1 = bb.emit(relax.op.nn.relu(lv0))
            # gelu wouldn't be offloaded to coreml
            lv2 = bb.emit(relax.op.nn.gelu(lv1))
            # relu would be offloaded to coreml
            lv3 = bb.emit(relax.op.nn.relu(lv2))
            gv = bb.emit_output(lv3)
        bb.emit_func_output(gv)
    mod = bb.get()
    x_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10).astype("float32"), dev)
    y_data = tvm_tirx.runtime.tensor(np.random.rand(10, 10).astype("float32"), dev)
    verify(mod, [x_data, y_data])


if __name__ == "__main__":
    pytest.main([__file__])
