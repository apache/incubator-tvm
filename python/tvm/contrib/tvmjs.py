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
# ruff: noqa: E501, F401
"""Namespace to store utilities for building web runtime."""

import copy
import hashlib
import json
import math
import os
import shutil

# pylint: disable=unused-import
import sys
from collections.abc import Iterator, Mapping
from typing import Any, Optional, Union

import numpy as np
import tvm_ffi

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None

import tvm
import tvm_tirx

from .emcc import create_wasm as create_tvmjs_wasm
from .emcc import find_wasm_lib


def _convert_f32_to_bf16(value):
    cap = np.finfo("float32").max
    assert -np.finfo("float32").max == np.finfo("float32").min
    bf16_limit = ((np.array([cap.view("uint32")]) >> 16) << 16).view("float32")[0]
    # When the value is in [-bf16_limit, bf16_limit], round to nearest even.
    # We can afford to do it in dumping phase to reduce overall rounding error.
    #
    # When the value is out of bound(usually mask values in attention), use truncation
    # so it is equivalent to clip to the limit values
    data = value.view("uint32")
    rounding_bias = np.where(
        np.logical_and(value < bf16_limit, value > -bf16_limit),
        ((data >> 16) & 1) + 0x7FFF,
        np.zeros_like(data),
    )
    return ((data + rounding_bias) >> 16).astype("uint16")


def _convert_bf16_to_f32(value):
    data = value.view("uint16")
    return (data.astype("uint32") << 16).view("float32")


def _calculate_md5(filename):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class TensorCacheShardingManager:
    """Internal helper to shard ndarrays."""

    def __init__(
        self,
        cache_dir: str,
        prefix: str,
        shard_cap_nbytes: int,
        initial_shard_records: Mapping[str, Any] | None = None,
    ):
        self.cache_dir = cache_dir
        self.prefix = prefix
        self.curr_records = []
        self.curr_data = bytearray()
        self.shard_records = []
        self.shard_cap_nbytes = shard_cap_nbytes
        self.counter = 0
        self.name_to_record: Mapping[str, tuple[int, Mapping[str, Any]]] = {}
        self.updated_shards: set[int] = set()

        if initial_shard_records is not None:
            self.shard_records = initial_shard_records
            self.counter = len(initial_shard_records)
            for idx, shard in enumerate(initial_shard_records):
                for rec in shard["records"]:
                    self.name_to_record[rec["name"]] = (idx, rec)

    def append_or_update(self, data, name, shape, dtype, encode_format, allow_update: bool = False):
        """Commit a record to the manager.

        Parameters
        ----------
        data: bytes
            Raw bytes to be appended.

        name: str
            The name of the parameter

        shape: tuple
            The shape of the array

        dtype: str
            The dtype information

        encode_format:
            The encode format of the entry

        allow_update: bool
            If the record already exists, update the record. Otherwise, raise an error.
        """
        rec = {
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "format": encode_format,
            "nbytes": len(data),
        }
        if name in self.name_to_record:
            if not allow_update:
                raise ValueError(f"Duplicate name {name} found in the cache.")
            self.update_single_record(rec, data)
            return

        self.name_to_record[name] = (self.counter, rec)

        if self.pending_nbytes + len(data) >= self.shard_cap_nbytes:
            if len(data) * 2 >= self.shard_cap_nbytes:
                # out of band data
                rec["byteOffset"] = 0
                self._commit_internal(data, [rec])
                return
            self.commit()
        rec["byteOffset"] = self.pending_nbytes
        self.curr_records.append(rec)
        self.curr_data += data

    def update_single_record(self, rec, data):
        """Update a single record in a shard file."""
        name = rec["name"]
        idx, old_rec = self.name_to_record[name]
        if old_rec["nbytes"] != rec["nbytes"]:
            raise ValueError(f"Cannot update record {name}, size mismatch.")
        data_path = self.shard_records[idx]["dataPath"]
        full_path = os.path.join(self.cache_dir, data_path)
        with open(full_path, "r+b") as outfile:
            outfile.seek(old_rec["byteOffset"])
            outfile.write(data)
        self.name_to_record[name] = (idx, rec)
        self.updated_shards.add(idx)

    def commit(self):
        """Commit a record"""
        if self.pending_nbytes != 0:
            self._commit_internal(self.curr_data, self.curr_records)
            self.curr_data = bytearray()
            self.curr_records = []

    def finish(self):
        """Finish building and return shard records."""
        self.commit()
        for idx in self.updated_shards:
            full_path = os.path.join(self.cache_dir, self.shard_records[idx]["dataPath"])
            self.shard_records[idx]["md5sum"] = _calculate_md5(full_path)
        return self.shard_records

    def _commit_internal(self, data, records):
        data_path = f"{self.prefix}_{self.counter}.bin"
        full_path = os.path.join(self.cache_dir, data_path)
        self.counter += 1
        with open(full_path, "wb") as outfile:
            outfile.write(data)

        shard_record = {
            "dataPath": data_path,
            "format": "raw-shard",
            "nbytes": len(data),
            "records": records,
            "md5sum": _calculate_md5(full_path),
        }
        self.shard_records.append(shard_record)

    @property
    def pending_nbytes(self):
        """Return total bytes stored so far"""
        return len(self.curr_data)


def dump_tensor_cache(
    params: Mapping[str, np.ndarray | tvm_tirx.runtime.Tensor]
    | Iterator[tuple[str, np.ndarray | tvm_tirx.runtime.Tensor]],
    cache_dir: str,
    encode_format: str | Mapping[str, str] = "f32-to-bf16",
    meta_data=None,
    shard_cap_mb=32,
    show_progress: bool = True,
    update_if_exists: bool = False,
):
    """Dump parameters to Tensor cache.

    Parameters
    ----------
    params: Union[
        Mapping[str, Union[np.ndarray, tvm_tirx.runtime.Tensor]],
        Iterator[Tuple[str, Union[np.ndarray, tvm_tirx.runtime.Tensor]]],
    ]
        The parameter dictionary or generator

    cache_dir: str
        The path to the cache

    encode_format: Union[
        {"f32-to-bf16", "raw"},
        Mapping[str, {"f32-to-bf16", "raw"}]
    ]
        Encoding format. A mapping selects the format by parameter name and may
        use ``"*"`` as the fallback for names not explicitly listed.

    meta_data: json-compatible-struct or Callable[[], Any]
        Extra meta_data to be stored in the cache json file,
        or a callable that returns the metadata.

    shard_cap_mb: int
        Maxinum number of MB to be kept per shard

    show_progress: bool
        A boolean indicating if to show the dump progress.

    update_if_exists: bool
        If the cache already exists, update the cache. When set to False, it will overwrite the
        existing files.
    """
    if isinstance(encode_format, str):
        if encode_format not in ("raw", "f32-to-bf16"):
            raise ValueError(f"Invalid encode_format {encode_format}")
    elif not isinstance(encode_format, Mapping):
        raise TypeError("encode_format must be a string or parameter-name mapping")
    else:
        for name, param_format in encode_format.items():
            if param_format not in ("raw", "f32-to-bf16"):
                raise ValueError(f"Invalid encode_format for parameter {name}: {param_format}")

    def resolve_encode_format(name):
        if isinstance(encode_format, str):
            return encode_format
        param_format = encode_format.get(name, encode_format.get("*"))
        if param_format not in ("raw", "f32-to-bf16"):
            raise ValueError(f"Invalid encode_format for parameter {name}: {param_format}")
        return param_format

    records = []
    total_bytes = 0
    counter = 0
    max_out_length = 0

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    print(f"Start storing to cache {cache_dir}")
    shard_cap_nbytes = shard_cap_mb * (1 << 20)

    nd_cache_json = os.path.join(cache_dir, "tensor-cache.json")
    if update_if_exists and os.path.exists(nd_cache_json):
        with open(nd_cache_json) as infile:
            old_data = json.load(infile)
            if meta_data is None:
                meta_data = old_data["metadata"]
            records = old_data["records"]

    shard_manager = TensorCacheShardingManager(
        cache_dir, "params_shard", shard_cap_nbytes, initial_shard_records=records
    )

    param_generator = params.items() if isinstance(params, Mapping) else params
    for k, origin_v in param_generator:
        param_encode_format = resolve_encode_format(k)
        shape = list(origin_v.shape)
        v = origin_v
        if not isinstance(v, np.ndarray):
            v = v.numpy()

        # prefer to preserve original dtype, especially if the format was bfloat16
        dtype = origin_v.dtype if isinstance(origin_v, tvm_tirx.runtime.Tensor) else v.dtype

        if dtype in tvm_ffi.dtype._NUMPY_DTYPE_TO_STR:
            dtype = tvm_ffi.dtype._NUMPY_DTYPE_TO_STR[dtype]
        else:
            dtype = str(dtype)

        total_bytes += math.prod(v.shape) * np.dtype(v.dtype).itemsize

        # convert fp32 to bf16
        if param_encode_format == "f32-to-bf16" and dtype == "float32":
            data = _convert_f32_to_bf16(v).tobytes()
        else:
            data = v.tobytes()

        shard_manager.append_or_update(
            data,
            name=k,
            shape=shape,
            dtype=dtype,
            encode_format=param_encode_format,
            allow_update=update_if_exists,
        )

        counter += 1
        if show_progress:
            last_cmd = f"[{counter:04d}] saving {k}"
            flush = "\r" + (" " * max_out_length) + "\r"
            max_out_length = max(len(last_cmd), max_out_length)
            sys.stdout.write(flush + last_cmd)

    records = shard_manager.finish()
    meta_data = {} if meta_data is None else meta_data if not callable(meta_data) else meta_data()

    with open(nd_cache_json, "w") as outfile:
        json.dump({"metadata": meta_data, "records": records}, outfile, indent=4)
    print(
        f"\nAll finished, {shard_manager.counter} total shards committed, record saved to {nd_cache_json}"
    )

    b16_nd_cache_json = os.path.join(cache_dir, "tensor-cache-b16.json")
    has_f32_to_bf16 = any(
        item["format"] == "f32-to-bf16" and item["dtype"] == "float32"
        for shard in records
        for item in shard["records"]
    )
    if has_f32_to_bf16:
        b16_records = copy.deepcopy(records)
        for shard in b16_records:
            for item in shard["records"]:
                if item["format"] == "f32-to-bf16" and item["dtype"] == "float32":
                    item["format"] = "raw"
                    item["dtype"] = "bfloat16"
        # also dump a file that contains bf16
        with open(b16_nd_cache_json, "w") as outfile:
            json.dump({"metadata": meta_data, "records": b16_records}, outfile, indent=4)
        print(f"Also saved a bf16 record to {b16_nd_cache_json}")
    elif os.path.exists(b16_nd_cache_json):
        os.remove(b16_nd_cache_json)


def load_tensor_cache(cachepath: str, device: tvm_tirx.runtime.Device):
    """Load the tensor cache from the directory or json.


    Parameters
    ----------
    cachepath: str
        Path to the location or json file.

    device: tvm_tirx.runtime.Device
        The device we would like to load the data from.
    """
    if not cachepath.endswith(".json"):
        cachepath = os.path.join(cachepath, "tensor-cache.json")

    cachedir = os.path.dirname(cachepath)
    json_info = json.loads(open(cachepath).read())
    result_dict = {}

    for shard_rec in json_info["records"]:
        data_path = shard_rec["dataPath"]
        full_data_path = os.path.join(cachedir, data_path)
        raw_data = open(full_data_path, "rb").read()
        assert shard_rec["format"] == "raw-shard"
        assert shard_rec["nbytes"] == len(raw_data)

        for rec in shard_rec["records"]:
            name = rec["name"]
            shape = rec["shape"]
            dtype = rec["dtype"]
            encode_format = rec["format"]
            offset = rec["byteOffset"]
            nbytes = rec["nbytes"]

            arr = tvm_tirx.runtime.empty(shape, dtype, device=device)
            assert offset + nbytes <= len(raw_data)
            buffer_source = raw_data[offset : offset + nbytes]
            if dtype == "float8_e4m3fn":
                if ml_dtypes is not None:
                    dtype = ml_dtypes.float8_e4m3fn
                else:
                    raise RuntimeError(
                        "ml_dtypes is not installed, cannot convert float8_e4m3fn array to numpy."
                    )
            if dtype == "float8_e5m2":
                if ml_dtypes is not None:
                    dtype = ml_dtypes.float8_e5m2
                else:
                    raise RuntimeError(
                        "ml_dtypes is not installed, cannot convert float8_e5m2 array to numpy."
                    )
            if encode_format == "f32-to-bf16" and dtype == "float32":
                data = np.frombuffer(buffer_source, dtype="uint16").reshape(shape)
                arr.copyfrom(_convert_bf16_to_f32(data))
            elif dtype == "bfloat16":
                data = np.frombuffer(buffer_source, dtype="uint16").reshape(shape)
                arr.copyfrom(data)
            else:
                data = np.frombuffer(buffer_source, dtype=dtype).reshape(shape)
                arr.copyfrom(data)
            result_dict[name] = arr
    return result_dict, json_info["metadata"]


def export_runtime(runtime_dir):
    """Export TVMJS runtime to the runtime_dir

    Parameters
    ----------
    runtime_dir: str
        The runtime directory
    """
    web_hint = (
        "make sure you setup tvm web runtime correctly."
        + " obtain a copy of TVM source code, set TVM_HOME env variable:\n"
        + " cd /path/to/tvm/web; make; npm run bundle"
    )

    jsbundle = find_wasm_lib("tvmjs.bundle.js", optional=True)
    if not jsbundle:
        raise RuntimeError("Cannot find tvmjs.bundle.js, " + web_hint)

    wasi = find_wasm_lib("tvmjs_runtime.wasi.js", optional=True)
    if not wasi:
        raise RuntimeError("Cannot find tvmjs_runtime.wasi.js, " + web_hint)

    print(f"Copy {jsbundle[0]} to {runtime_dir}")
    shutil.copy(jsbundle[0], runtime_dir)
    print(f"Copy {wasi[0]} to {runtime_dir}")
    shutil.copy(wasi[0], runtime_dir)
