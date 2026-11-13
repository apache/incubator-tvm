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

"""Test contrib.tvmjs"""

import json
import os
import tempfile

import numpy as np
import pytest

import tvm_tirx.testing
from tvm.contrib import tvmjs

_dtype_params = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "float8_e4m3fn",
    "float8_e5m2",
)


@pytest.fixture(params=_dtype_params, scope="session")
def dtype(request):
    return request.param


def test_save_load_float8(dtype):
    if "float8" in dtype or "bfloat16" in dtype:
        ml_dtypes = pytest.importorskip("ml_dtypes")
        np_dtype = np.dtype(getattr(ml_dtypes, dtype))
    else:
        np_dtype = np.dtype(dtype)

    arr = np.arange(16, dtype=np_dtype)

    with tempfile.TemporaryDirectory(prefix="tvm_") as temp_dir:
        tvmjs.dump_tensor_cache({"arr": arr}, temp_dir)
        cache, _ = tvmjs.load_tensor_cache(temp_dir, tvm_tirx.cpu())

    after_roundtrip = cache["arr"].numpy()

    np.testing.assert_array_equal(arr, after_roundtrip)


def _records_by_name(manifest_path):
    with open(manifest_path, encoding="utf-8") as source:
        manifest = json.load(source)
    records = {
        record["name"]: record for shard in manifest["records"] for record in shard["records"]
    }
    return records, manifest


def test_dump_tensor_cache_supports_per_parameter_encoding_roundtrip():
    raw = np.array([0.1234567, -0.7654321], dtype="float32")
    compressed = np.array([0.2345678, -0.8765432], dtype="float32")

    with tempfile.TemporaryDirectory(prefix="tvm_") as temp_dir:
        tvmjs.dump_tensor_cache(
            {"raw": raw, "compressed": compressed},
            temp_dir,
            encode_format={"raw": "raw", "*": "f32-to-bf16"},
        )
        cache, _ = tvmjs.load_tensor_cache(temp_dir, tvm_tirx.cpu())
        b16_cache, _ = tvmjs.load_tensor_cache(
            os.path.join(temp_dir, "tensor-cache-b16.json"), tvm_tirx.cpu()
        )

        records, _ = _records_by_name(os.path.join(temp_dir, "tensor-cache.json"))
        b16_records, _ = _records_by_name(os.path.join(temp_dir, "tensor-cache-b16.json"))

    assert records["raw"]["format"] == "raw"
    assert records["raw"]["dtype"] == "float32"
    assert records["compressed"]["format"] == "f32-to-bf16"
    assert records["compressed"]["dtype"] == "float32"
    assert b16_records["raw"]["format"] == "raw"
    assert b16_records["raw"]["dtype"] == "float32"
    assert b16_records["compressed"]["format"] == "raw"
    assert b16_records["compressed"]["dtype"] == "bfloat16"
    np.testing.assert_array_equal(cache["raw"].numpy(), raw)
    np.testing.assert_allclose(cache["compressed"].numpy(), compressed, rtol=4e-3, atol=1e-3)
    np.testing.assert_array_equal(b16_cache["raw"].numpy(), raw)
    np.testing.assert_allclose(b16_cache["compressed"].numpy(), compressed, rtol=4e-3, atol=1e-3)


def test_dump_tensor_cache_supports_generator_input():
    params = (
        item
        for item in [
            ("raw", np.arange(4, dtype="float32")),
            ("compressed", np.linspace(-1, 1, 4, dtype="float32")),
        ]
    )

    with tempfile.TemporaryDirectory(prefix="tvm_") as temp_dir:
        tvmjs.dump_tensor_cache(
            params,
            temp_dir,
            encode_format={"raw": "raw", "*": "f32-to-bf16"},
        )
        cache, _ = tvmjs.load_tensor_cache(temp_dir, tvm_tirx.cpu())

    np.testing.assert_array_equal(cache["raw"].numpy(), np.arange(4, dtype="float32"))
    np.testing.assert_allclose(
        cache["compressed"].numpy(),
        np.linspace(-1, 1, 4, dtype="float32"),
        rtol=4e-3,
        atol=1e-3,
    )


def test_dump_tensor_cache_updates_mixed_encoding_manifests():
    original_raw = np.array([1.0, 2.0], dtype="float32")
    updated_raw = np.array([3.0, 4.0], dtype="float32")
    compressed = np.array([0.2345678, -0.8765432], dtype="float32")

    with tempfile.TemporaryDirectory(prefix="tvm_") as temp_dir:
        tvmjs.dump_tensor_cache(
            {"raw": original_raw, "compressed": compressed},
            temp_dir,
            encode_format={"raw": "raw", "*": "f32-to-bf16"},
        )
        _, old_manifest = _records_by_name(os.path.join(temp_dir, "tensor-cache.json"))
        old_md5 = old_manifest["records"][0]["md5sum"]

        tvmjs.dump_tensor_cache(
            iter([("raw", updated_raw)]),
            temp_dir,
            encode_format={"raw": "raw"},
            update_if_exists=True,
        )

        cache, _ = tvmjs.load_tensor_cache(temp_dir, tvm_tirx.cpu())
        b16_cache, _ = tvmjs.load_tensor_cache(
            os.path.join(temp_dir, "tensor-cache-b16.json"), tvm_tirx.cpu()
        )
        _, manifest = _records_by_name(os.path.join(temp_dir, "tensor-cache.json"))
        _, b16_manifest = _records_by_name(os.path.join(temp_dir, "tensor-cache-b16.json"))

    np.testing.assert_array_equal(cache["raw"].numpy(), updated_raw)
    np.testing.assert_array_equal(b16_cache["raw"].numpy(), updated_raw)
    np.testing.assert_allclose(cache["compressed"].numpy(), compressed, rtol=4e-3, atol=1e-3)
    assert manifest["records"][0]["md5sum"] != old_md5
    assert manifest["records"][0]["md5sum"] == b16_manifest["records"][0]["md5sum"]


def test_dump_tensor_cache_removes_stale_b16_manifest():
    with tempfile.TemporaryDirectory(prefix="tvm_") as temp_dir:
        tvmjs.dump_tensor_cache(
            {"compressed": np.ones(2, dtype="float32")},
            temp_dir,
            encode_format="f32-to-bf16",
        )
        b16_manifest = os.path.join(temp_dir, "tensor-cache-b16.json")
        assert os.path.exists(b16_manifest)

        tvmjs.dump_tensor_cache(
            {"raw": np.ones(2, dtype="float32")},
            temp_dir,
            encode_format="raw",
        )
        assert not os.path.exists(b16_manifest)


def test_dump_tensor_cache_requires_a_format_for_every_parameter():
    with tempfile.TemporaryDirectory(prefix="tvm_") as temp_dir:
        with pytest.raises(ValueError, match="parameter arr"):
            tvmjs.dump_tensor_cache(
                {"arr": np.ones(2, dtype="float32")},
                temp_dir,
                encode_format={"other": "raw"},
            )


if __name__ == "__main__":
    tvm_tirx.testing.main()
