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

"""FlashInfer JIT compilation module for CUDA backend"""

import re
from pathlib import Path

import tvm_ffi

import tvm_tirx
from tvm_tirx.target import Target


def _rename_exported_func_names(source_paths: list[Path], prefix: str):
    """Rename the ffi-exported function names in the source files to the given prefix."""
    pattern = re.compile(r"^(\s*TVM_FFI_DLL_EXPORT_TYPED_FUNC\()([A-Za-z0-9_]+)(,.*)$")
    for source_path in source_paths:
        if not source_path.name.endswith("_binding.cu"):
            continue

        original_text = source_path.read_text(encoding="utf-8")
        lines = original_text.splitlines(keepends=True)
        updated = False
        for idx, line in enumerate(lines):
            line_body = line.rstrip("\r\n")
            line_ending = line[len(line_body) :]
            match = pattern.match(line_body)
            if not match:
                continue
            new_body = f"{match.group(1)}{prefix}_{match.group(2)}{match.group(3)}"
            lines[idx] = new_body + line_ending
            updated = True

        if updated:
            source_path.write_text("".join(lines), encoding="utf-8")


def _load_flashinfer_modules(object_files: list[Path]) -> list[tvm_ffi.Module]:
    return [
        tvm_tirx.codegen.load_static_library(str(obj_path.absolute()), func_names=[])
        for obj_path in object_files
    ]


def gen_flashinfer_prefill_module(
    dtype_q: str,
    dtype_kv: str,
    dtype_o: str,
    qk_head_dim: int,
    v_head_dim: int,
    enable_inline_rope: bool,
    return_static_libs: bool = False,
) -> list[tvm_ffi.Module]:
    """Generate a FlashInfer module for prefill.

    Parameters
    ----------
    dtype_q : str
        The data type of the query tensor.
    dtype_kv : str
        The data type of the key/value tensors.
    dtype_o : str
        The data type of the output tensor.
    qk_head_dim : int
        The head dimension of the query and key tensors.
    v_head_dim : int
        The head dimension of the value tensor.
    enable_inline_rope : bool
        Whether to enable inline rotary positional embedding.
    return_static_libs : bool
        Whether to return static library modules instead of compiled modules.
        When it is False, it returns the loaded shared library that links all the object files.
        When it is True, it returns the static libraries of each compiled object files.

    Returns
    -------
    A list of compiled static library modules for FlashInfer prefill kernels.
    """
    try:
        from flashinfer.jit import (  # pylint: disable=import-outside-toplevel
            gen_customize_batch_prefill_module,
        )
    except ImportError:
        raise ImportError(
            "FlashInfer is not installed. Please follow instructions "
            "in https://docs.flashinfer.ai to install FlashInfer."
        )
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError("PyTorch is not installed. Please install PyTorch to use FlashInfer.")

    if enable_inline_rope and qk_head_dim != v_head_dim:
        raise ValueError("Inline rope mode is not supported when qk_head_dim == v_head_dim")

    torch_dtype_q = getattr(torch, dtype_q)
    torch_dtype_kv = getattr(torch, dtype_kv)
    torch_dtype_o = getattr(torch, dtype_o)
    # Todo(tvm-team): decide which backend ("fa2/fa3") to use
    backend = "fa2"
    variant_name = (
        "DefaultAttention<false, false, false, false>"
        if backend == "fa2"
        else "DefaultAttention<false>"
    )
    variant_decl = (
        "#include <flashinfer/attention/variants.cuh>"
        if backend == "fa2"
        else "#include <flashinfer/attention/hopper/variants.cuh>"
    )
    jit_spec = gen_customize_batch_prefill_module(
        backend=backend,
        uri=f"batch_prefill_tvm_dtype_q_{dtype_q}_"
        + f"dtype_kv_{dtype_kv}_"
        + f"dtype_o_{dtype_o}_"
        + f"qk_head_dim_{qk_head_dim}_"
        + f"v_head_dim_{v_head_dim}_"
        + f"enable_inline_rope_{enable_inline_rope}",
        dtype_q=torch_dtype_q,
        dtype_kv=torch_dtype_kv,
        dtype_o=torch_dtype_o,
        idtype=torch.int32,
        head_dim_qk=qk_head_dim,
        head_dim_vo=v_head_dim,
        pos_encoding_mode=int(enable_inline_rope),
        additional_tensor_names=[],
        additional_tensor_dtypes=[],
        additional_scalar_names=["sm_scale", "rope_rcp_scale", "rope_rcp_theta"],
        additional_scalar_dtypes=["double", "double", "double"],
        variant_name=variant_name,
        variant_decl=variant_decl,
    )
    _rename_exported_func_names(jit_spec.sources, "batch_prefill")
    if return_static_libs:
        jit_spec.build(verbose=False)
        return _load_flashinfer_modules(jit_spec.get_object_paths())
    return [jit_spec.build_and_load()]


def gen_flashinfer_decode_module(
    dtype_q: str,
    dtype_kv: str,
    dtype_o: str,
    qk_head_dim: int,
    v_head_dim: int,
    enable_inline_rope: bool,
    return_static_libs: bool = False,
) -> list[tvm_ffi.Module]:
    """Generate a FlashInfer module for decode.

    Parameters
    ----------
    dtype_q : str
        The data type of the query tensor.
    dtype_kv : str
        The data type of the key/value tensors.
    dtype_o : str
        The data type of the output tensor.
    qk_head_dim : int
        The head dimension of the query and key tensors.
    v_head_dim : int
        The head dimension of the value tensor.
    enable_inline_rope : bool
        Whether to enable inline rotary positional embedding.
    return_static_libs : bool
        Whether to return static library modules instead of compiled modules.
        When it is False, it returns the loaded shared library that links all the object files.
        When it is True, it returns the static libraries of each compiled object files.

    Returns
    -------
    A list of compiled static library modules for FlashInfer decode kernels.
    """
    try:
        from flashinfer.jit import (  # pylint: disable=import-outside-toplevel
            gen_customize_batch_decode_module,
        )
    except ImportError:
        raise ImportError(
            "FlashInfer is not installed. Please follow instructions "
            "in https://docs.flashinfer.ai to install FlashInfer."
        )
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError("PyTorch is not installed. Please install PyTorch to use FlashInfer.")

    torch_dtype_q = getattr(torch, dtype_q)
    torch_dtype_kv = getattr(torch, dtype_kv)
    torch_dtype_o = getattr(torch, dtype_o)
    jit_spec = gen_customize_batch_decode_module(
        uri=f"batch_decode_tvm_dtype_q_{dtype_q}_"
        + f"dtype_kv_{dtype_kv}_"
        + f"dtype_o_{dtype_o}_"
        + f"qk_head_dim_{qk_head_dim}_"
        + f"v_head_dim_{v_head_dim}_"
        + f"enable_inline_rope_{enable_inline_rope}",
        dtype_q=torch_dtype_q,
        dtype_kv=torch_dtype_kv,
        dtype_o=torch_dtype_o,
        idtype=torch.int32,
        head_dim_qk=qk_head_dim,
        head_dim_vo=v_head_dim,
        pos_encoding_mode=int(enable_inline_rope),
        additional_tensor_names=[],
        additional_tensor_dtypes=[],
        additional_scalar_names=["sm_scale", "rope_rcp_scale", "rope_rcp_theta"],
        additional_scalar_dtypes=["double", "double", "double"],
        variant_name="DefaultAttention<false, false, false, false>",
        variant_decl="#include <flashinfer/attention/variants.cuh>",
    )
    _rename_exported_func_names(jit_spec.sources, "batch_decode")
    if return_static_libs:
        jit_spec.build(verbose=False)
        return _load_flashinfer_modules(jit_spec.get_object_paths())
    return [jit_spec.build_and_load()]


def gen_flashinfer_mla_module(
    dtype_q: str,
    dtype_kv: str,
    dtype_o: str,
    head_dim_ckv: int,
    head_dim_kpe: int,
    return_static_libs: bool = False,
) -> list[tvm_ffi.Module]:
    """Generate a FlashInfer module for MLA.

    Parameters
    ----------
    dtype_q : str
        The data type of the query tensor.
    dtype_kv : str
        The data type of the key/value tensors.
    dtype_o : str
        The data type of the output tensor.
    head_dim_ckv : int
        The head dimension of the compressed key/value tensors.
    head_dim_kpe : int
        The head dimension of the query/key positional embedding.
    target : Target
        The target device to compile for.
    num_threads : int
        The number of threads to use for compilation.
    return_static_libs : bool
        Whether to return static library modules instead of compiled modules.
        When it is False, it returns the loaded shared library that links all the object files.
        When it is True, it returns the static libraries of each compiled object files.

    Returns
    -------
    A list of compiled static library modules for FlashInfer MLA kernels.
    """
    try:
        from flashinfer.jit import (  # pylint: disable=import-outside-toplevel
            gen_batch_mla_module,
        )
    except ImportError:
        raise ImportError(
            "FlashInfer is not installed. Please follow instructions "
            "in https://docs.flashinfer.ai to install FlashInfer."
        )
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError("PyTorch is not installed. Please install PyTorch to use FlashInfer.")

    torch_dtype_q = getattr(torch, dtype_q)
    torch_dtype_kv = getattr(torch, dtype_kv)
    torch_dtype_o = getattr(torch, dtype_o)
    jit_spec = gen_batch_mla_module(
        backend="fa2",
        dtype_q=torch_dtype_q,
        dtype_kv=torch_dtype_kv,
        dtype_o=torch_dtype_o,
        dtype_idx=torch.int32,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        use_profiler=False,
    )
    _rename_exported_func_names(jit_spec.sources, "batch_mla")
    if return_static_libs:
        jit_spec.build(verbose=False)
        return _load_flashinfer_modules(jit_spec.get_object_paths())
    return [jit_spec.build_and_load()]


def gen_grouped_gemm_module(
    target: Target, return_static_libs: bool = False
) -> list[tvm_ffi.Module]:
    """Generate a FlashInfer module for FP8 grouped GEMM.

    Parameters
    ----------
    target : Target
        The target device to compile for.
    return_static_libs : bool
        Whether to return static library modules instead of compiled modules.
        When it is False, it returns the loaded shared library that links all the object files.
        When it is True, it returns the static libraries of each compiled object files.

    Returns
    -------
    List[tvm_ffi.Module]
        A list of compiled static library modules for FlashInfer FP8 grouped GEMM kernels.

    Note
    _____
    when apply grouped gemm on A: (total_m, k), B: (batch_size, n, k), m_indptr: (batch_size, )
    requires all m in m_indptr to be multiple of 4
    """
    # NOTE: This function is still under development,
    # and we currently only support SM100 grouped gemm
    try:
        from flashinfer.gemm import (  # pylint: disable=import-outside-toplevel
            gen_gemm_sm100_module,
        )
    except ImportError:
        raise ImportError(
            "FlashInfer is not installed. Please follow instructions "
            "in https://docs.flashinfer.ai to install FlashInfer."
        )

    compute_version = "".join(tvm_tirx.support.nvcc.get_target_compute_version(target).split("."))
    if compute_version == "100":
        jit_spec = gen_gemm_sm100_module()
    else:
        raise ValueError(f"Unsupported compute version: {compute_version}")
    if return_static_libs:
        jit_spec.build(verbose=False)
        return _load_flashinfer_modules(jit_spec.get_object_paths())
    return [jit_spec.build_and_load()]
