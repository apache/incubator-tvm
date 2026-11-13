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
"""Compile-time TensorRT engines and their exported runtime artifacts."""

import json
import os
import struct
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax.backend.contrib.tensorrt import partition_for_tensorrt
from tvm.relax.dpl import is_op, wildcard
from tvm_tirx import tirx
from tvm_tirx.support import cc
from tvm_tirx.testing import env

_has_codegen = tvm_ffi.get_global_func("relax.ext.tensorrt", True) is not None
_runtime_enabled = tvm_ffi.get_global_func("relax.is_tensorrt_runtime_enabled", True)
_has_runtime = _runtime_enabled is not None and _runtime_enabled()

pytestmark = pytest.mark.skipif(not _has_codegen, reason="TensorRT codegen is not enabled")
requires_runtime = pytest.mark.skipif(not _has_runtime, reason="TensorRT runtime is not enabled")
requires_gpu = pytest.mark.skipif(not env.has_cuda(), reason="CUDA is not available")

# Version 1 extends the legacy JSON runtime prefix with a private engine trailer.
_ENGINE_MAGIC = 0x31545254564D5445


@pytest.fixture(autouse=True)
def isolated_tensorrt_options(monkeypatch):
    for name in (
        "TVM_TENSORRT_CACHE_DIR",
        "TVM_TENSORRT_USE_INT8",
        "TENSORRT_NUM_CALI_INT8",
        "TVM_TENSORRT_MULTI_ENGINE",
        "TVM_TENSORRT_USE_FP16",
        "TVM_TENSORRT_MAX_WORKSPACE_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)


def _string(value):
    value = value.encode() if isinstance(value, str) else value
    return struct.pack("<Q", len(value)) + value


def _strings(values):
    return struct.pack("<Q", len(values)) + b"".join(_string(value) for value in values)


def _legacy_bytes(module):
    return (
        _string(module["get_symbol"]())
        + _string(module.inspect_source())
        + _strings(module["get_const_vars"]())
    )


def _module_bytes(module):
    """Extract a leaf's payload from the same import packer used by export_library."""
    assert not module.imports
    packed = tvm_ffi.get_global_func("runtime.ModulePackImportsToTensor")(module).numpy().tobytes()
    assert struct.unpack_from("<Q", packed)[0] == len(packed) - 8
    offset = 8
    # The import tree consists of row pointers followed by child indices.
    for _ in range(2):
        count = struct.unpack_from("<Q", packed, offset)[0]
        offset += 8 + 8 * count
    kind_size = struct.unpack_from("<Q", packed, offset)[0]
    offset += 8
    assert packed[offset : offset + kind_size] == b"tensorrt"
    offset += kind_size
    payload_size = struct.unpack_from("<Q", packed, offset)[0]
    offset += 8
    assert offset + payload_size == len(packed)
    return packed[offset:]


def _load_bytes(payload):
    return tvm_ffi.get_global_func("ffi.Module.load_from_bytes.tensorrt")(payload)


def _relu_partition(shape=(2, 4)):
    data = relax.Var("data", relax.TensorType(shape, "float32"))
    builder = relax.BlockBuilder()
    with builder.function("main", [data]):
        with builder.dataflow():
            output = builder.emit_output(relax.op.nn.relu(data))
        builder.emit_func_output(output)
    return tvm_tirx.ir.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern(
                [("tensorrt.nn.relu", is_op("relax.nn.relu")(wildcard()))]
            ),
            relax.transform.MergeCompositeFunctions(),
        ]
    )(builder.get())


def _codegen(partitioned, eager=None):
    config = {}
    if eager is not None:
        config["relax.ext.tensorrt.options"] = {"build_at_compile_time": eager}
    with tvm_tirx.ir.transform.PassContext(config=config):
        return relax.transform.RunCodegen()(partitioned)


def _external_modules(mod):
    modules = list(mod.attrs["external_mods"])
    assert modules and all(module.kind == "tensorrt" for module in modules)
    return modules


def test_default_preserves_legacy_bytes():
    partitioned = _relu_partition()
    default = _external_modules(_codegen(partitioned))[0]
    disabled = _external_modules(_codegen(partitioned, eager=False))[0]
    payload = _legacy_bytes(default)
    assert _module_bytes(default) == payload
    assert _module_bytes(disabled) == payload
    # A real old-format record must remain loadable and exportable without a GPU.
    restored = _load_bytes(payload)
    assert restored["get_symbol"]() == default["get_symbol"]()
    assert restored.inspect_source() == default.inspect_source()
    assert _module_bytes(restored) == payload


@pytest.mark.skipif(_has_runtime, reason="This test requires a codegen-only TensorRT build")
def test_eager_requires_runtime():
    with pytest.raises(RuntimeError, match="TVM_USE_TENSORRT_RUNTIME"):
        _codegen(_relu_partition(), eager=True)


@pytest.mark.parametrize(
    "trailer, message",
    [
        (b"partial", "Truncated TensorRT engine trailer"),
        (struct.pack("<Q", 0), "Invalid TensorRT engine trailer"),
        (
            struct.pack("<QI", _ENGINE_MAGIC, 999),
            "Unsupported TensorRT engine version",
        ),
        (
            struct.pack("<QI", _ENGINE_MAGIC, 1) + _string(b""),
            "embedded TensorRT engine is empty",
        ),
    ],
)
def test_reject_malformed_engine_trailer(trailer, message):
    module = _external_modules(_codegen(_relu_partition()))[0]
    with pytest.raises(RuntimeError, match=message):
        _load_bytes(_legacy_bytes(module) + trailer)


@requires_runtime
@pytest.mark.parametrize("batch", ["dynamic", 0])
def test_eager_requires_static_positive_dimensions(batch):
    dimension = tirx.Var("batch", "int64") if batch == "dynamic" else batch
    with pytest.raises(ValueError, match="static positive"):
        _codegen(_relu_partition((dimension, 4)), eager=True)


@requires_runtime
def test_eager_rejects_int8_calibration(monkeypatch):
    monkeypatch.setenv("TVM_TENSORRT_USE_INT8", "1")
    monkeypatch.setenv("TENSORRT_NUM_CALI_INT8", "1")
    with pytest.raises(RuntimeError, match="INT8 calibration"):
        _codegen(_relu_partition(), eager=True)


def _two_partitions(dtype):
    data, first, second, third = [
        relax.Var(name, relax.TensorType((2, 4), dtype))
        for name in ("data", "first", "second", "third")
    ]
    builder = relax.BlockBuilder()
    with builder.function("main", [data, first, second, third]):
        with builder.dataflow():
            difference = builder.emit(relax.op.subtract(second, data))
            shifted = builder.emit(relax.op.add(difference, first))
            separated = builder.emit(relax.op.sin(shifted))
            result = builder.emit_output(relax.op.multiply(separated, third))
        builder.emit_func_output(result)

    parameters = {
        "first": np.linspace(-0.3, 0.4, 8, dtype=dtype).reshape(2, 4),
        "second": np.linspace(0.8, 2.2, 8, dtype=dtype).reshape(2, 4),
        "third": np.linspace(1.2, 2.9, 8, dtype=dtype).reshape(2, 4),
    }
    patterns = [
        ("tensorrt.subtract", is_op("relax.subtract")(wildcard(), wildcard())),
        ("tensorrt.add", is_op("relax.add")(wildcard(), wildcard())),
        ("tensorrt.multiply", is_op("relax.multiply")(wildcard(), wildcard())),
    ]
    partitioned = tvm_tirx.ir.transform.Sequential(
        [
            relax.transform.BindParams("main", parameters),
            # Leave sin on the TVM side to force two TensorRT partitions.
            relax.transform.FuseOpsByPattern(patterns, bind_constants=True),
            relax.transform.MergeCompositeFunctions(),
        ]
    )(builder.get())
    return partitioned, parameters


def _conv2d_relu_partition():
    data = relax.Var("data", relax.TensorType((1, 2, 5, 5), "float32"))
    weight = relax.Var("weight", relax.TensorType((3, 2, 3, 3), "float32"))
    builder = relax.BlockBuilder()
    with builder.function("main", [data, weight]):
        with builder.dataflow():
            convolution = builder.emit(relax.op.nn.conv2d(data, weight, padding=(1, 1)))
            result = builder.emit_output(relax.op.nn.relu(convolution))
        builder.emit_func_output(result)

    weight_np = np.linspace(-0.3, 0.4, 54, dtype="float32").reshape(3, 2, 3, 3)
    bound = relax.transform.BindParams("main", {"weight": weight_np})(builder.get())
    partitioned = partition_for_tensorrt(bound)
    data_np = np.linspace(-1.7, 1.3, 100, dtype="float32").reshape(2, 1, 2, 5, 5)
    padded = np.pad(data_np, ((0, 0), (0, 0), (0, 0), (1, 1), (1, 1)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(-2, -1))
    expected = np.maximum(np.einsum("snihwkl,oikl->snohw", windows, weight_np), 0)
    return partitioned, data_np, expected


@requires_runtime
@requires_gpu
@pytest.mark.gpu
def test_failed_engine_build_can_be_retried():
    """A failure after copying weights must not leave a partial engine or borrowed inputs."""
    partitioned, _ = _two_partitions("float32")
    offloaded = _codegen(partitioned, eager=False)
    module = max(_external_modules(offloaded), key=lambda mod: len(mod["get_const_vars"]()))
    graph = json.loads(module.inspect_source())
    kernels = [node for node in graph["nodes"] if node["op"] == "kernel"]
    kernels[-1]["name"] = "tensorrt.unsupported_test_op"
    names = module["get_const_vars"]()
    assert len(names) == 2
    failing = tvm_ffi.get_global_func("runtime.tensorrt_runtime_create")(
        module["get_symbol"](), json.dumps(graph), names
    )
    constants_by_name = dict(offloaded.attrs["const_name_to_constant"])
    constants = [constants_by_name[name] for name in names]
    legacy = _legacy_bytes(failing)

    def check():
        for _ in range(2):
            with pytest.raises(RuntimeError, match="Unsupported operator"):
                failing["build_engine"](constants)
            assert _module_bytes(failing) == legacy
            assert _module_bytes(_load_bytes(legacy)) == legacy

    tvm_tirx.testing.run_with_gpu_lock(check)


@requires_runtime
@requires_gpu
@pytest.mark.skipif(not env.has_llvm(), reason="LLVM is not available")
@pytest.mark.gpu
@pytest.mark.skipif(sys.platform != "linux", reason="The builder guard uses LD_PRELOAD")
@pytest.mark.parametrize(
    "dtype, device_id, model, execution_device",
    [
        ("float32", 0, "elementwise", "cuda"),
        ("float16", 0, "elementwise", "cuda"),
        ("float16", 1, "elementwise", "cuda"),
        ("float32", 0, "conv2d_relu", "cuda"),
        ("float32", 0, "elementwise", "cpu"),
    ],
)
def test_exported_engines_run_without_builder(
    tmp_path, monkeypatch, dtype, device_id, model, execution_device
):
    """Build weighted partitions before VM creation, then load and run without a builder."""
    if not tvm_tirx.cuda(device_id).exist:
        pytest.skip(f"CUDA device {device_id} is not available")

    target = "llvm" if execution_device == "cpu" else "cuda"
    guard_source = tmp_path / "builder_guard.cc"
    guard_library = tmp_path / "builder_guard.so"
    guard_source.write_text(
        "#include <cstdlib>\n"
        'extern "C" void* createInferBuilder_INTERNAL(void*, int) { std::_Exit(86); }\n'
    )
    cc.create_shared(str(guard_library), [str(guard_source)])

    if model == "conv2d_relu":
        partitioned, data, expected = _conv2d_relu_partition()
        expected_partition_count = 1
        # Match the numerical tolerance of the existing TensorRT convolution tests.
        tolerance = 1e-3
    else:
        partitioned, parameters = _two_partitions(dtype)
        data = np.linspace(-1.7, 1.3, 16, dtype=dtype).reshape(2, 2, 4)
        expected = np.sin(parameters["second"] - data + parameters["first"]) * parameters["third"]
        expected_partition_count = 2
        tolerance = 1e-2 if dtype == "float16" else 1e-5
    np.savez(tmp_path / "reference.npz", data=data, expected=expected)

    runner = textwrap.dedent(
        """
        import sys
        import numpy as np
        import tvm, tvm_tirx
        from tvm import relax

        artifact = tvm_tirx.runtime.load_module(sys.argv[1])
        device = tvm_tirx.cpu() if sys.argv[5] == "cpu" else tvm_tirx.cuda(int(sys.argv[3]))
        tolerance = float(sys.argv[4])
        vm = relax.VirtualMachine(artifact, device)
        reference = np.load(sys.argv[2])
        for data, expected in zip(reference["data"], reference["expected"]):
            result = vm["main"](tvm_tirx.runtime.tensor(data, device))
            assert result.device == device
            np.testing.assert_allclose(result.numpy(), expected, rtol=tolerance, atol=tolerance)
        """
    )
    child_env = os.environ.copy()
    child_env.pop("TVM_TENSORRT_CACHE_DIR", None)
    previous_preload = child_env.get("LD_PRELOAD", "")
    child_env["LD_PRELOAD"] = str(guard_library) + (
        ":" + previous_preload if previous_preload else ""
    )

    def run_child(path):
        return subprocess.run(
            [
                sys.executable,
                "-c",
                runner,
                str(path),
                str(tmp_path / "reference.npz"),
                str(device_id),
                str(tolerance),
                execution_device,
            ],
            env=child_env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def check():
        # Negative control: prove the preload intercepts the existing lazy builder.
        lazy = _codegen(partitioned, eager=False)
        assert len(_external_modules(lazy)) == expected_partition_count
        lazy_path = tmp_path / "lazy.so"
        tvm.compile(lazy, target).export_library(str(lazy_path))
        result = run_child(lazy_path)
        assert result.returncode == 86, result.stdout + result.stderr

        eager = _codegen(partitioned, eager=True)
        modules = _external_modules(eager)
        assert len(modules) == expected_partition_count
        for module in modules:
            payload = _module_bytes(module)
            # No VirtualMachine has been created in this process.
            assert len(payload) > len(_legacy_bytes(module))
            restored = _load_bytes(payload)
            assert _module_bytes(restored) == payload

        eager_path = tmp_path / "eager.so"
        tvm.compile(eager, target).export_library(str(eager_path))
        result = run_child(eager_path)
        assert result.returncode == 0, result.stdout + result.stderr

        module = modules[0]
        constants_by_name = dict(eager.attrs["const_name_to_constant"])
        constants = [constants_by_name[name] for name in module["get_const_vars"]()]
        # Embedded plans cannot accept either an INT8 flag or a captured calibration count.
        for use_int8 in ("1", "0"):
            with monkeypatch.context() as calibration:
                calibration.setenv("TVM_TENSORRT_USE_INT8", use_int8)
                calibration.setenv("TENSORRT_NUM_CALI_INT8", "1")
                loaded = _load_bytes(_module_bytes(module))
                with pytest.raises(RuntimeError, match="prebuilt TensorRT.*INT8 calibration"):
                    loaded["__init_" + loaded["get_symbol"]()](constants)

        # A corrupt plan must fail deserialization instead of rebuilding from the JSON.
        payload = bytearray(_module_bytes(module))
        plan_start = len(_legacy_bytes(module)) + 8 + 4 + 8
        payload[plan_start : plan_start + 32] = b"\0" * 32
        corrupt = _load_bytes(bytes(payload))
        symbol = corrupt["get_symbol"]()
        corrupt["__init_" + symbol](constants)
        # Engine deserialization is deferred until the input device is known.
        device = tvm_tirx.cpu() if execution_device == "cpu" else tvm_tirx.cuda(device_id)
        argument = tvm_tirx.runtime.tensor(np.zeros(data.shape[1:], dtype=dtype), device)
        output = tvm_tirx.runtime.tensor(np.zeros(expected.shape[1:], dtype=dtype), device)
        with pytest.raises(RuntimeError, match="deserialize the embedded TensorRT engine"):
            corrupt[symbol](argument, output)

    tvm_tirx.testing.run_with_gpu_lock(check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
