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
# pylint: disable=invalid-name
"""Common functions and classes for CUTLASS GEMM and Conv2d geneator."""

import logging
import math
import multiprocessing
import os
import re
import subprocess
import tempfile

import tvm_ffi

import tvm_tirx
from tvm_tirx.ir import IntImm

from . import _ffi_api as ffi
from .attention_operation import (
    instantiate_attention_template,
    instantiate_flash_attention_template,
    instantiate_flash_attention_var_len_template,
)
from .conv2d_operation import instantiate_conv2d_template
from .gemm_operation import emit_fp16A_intB_matmul, instantiate_gemm_template
from .layer_norm_operation import instantiate_layer_norm_template
from .library import (
    DataType,
    DataTypeSize,
    DataTypeTag,
    EpilogueFunctor,
    MathInstruction,
    MathOperation,
    OpcodeClass,
    TileDescription,
)
from .rms_norm_operation import instantiate_rms_norm_template

logger = logging.getLogger("cutlass")


dtype_map = {
    "int8": DataType.s8,
    "uint8": DataType.u8,
    "int32": DataType.s32,
    "float32": DataType.f32,
    "float16": DataType.f16,
}


def generate_tensor_op_common(
    math_instructions, alignment_constraints, get_tile_descriptions, op_creator
):
    """Common kernel generator to be used by archtecture specific generators."""
    ops = []
    for math_inst in math_instructions:
        tile_descriptions = get_tile_descriptions(math_inst)
        data_type = [
            math_inst.element_a,
            math_inst.element_b,
            math_inst.element_c,
            math_inst.element_accumulator,
        ]

        out = op_creator(tile_descriptions, data_type, alignment_constraints)

        ops.extend(out)

    return ops


def generate_sm50_simt(out_dtype, arg0_dtype, arg1_dtype, op_creator, accumulator_dtype="float32"):
    """Gemerate GEMM or Conv2D SIMT kernels"""
    # pylint: disable=unused-argument
    min_cc = 50
    max_cc = 1024
    if arg0_dtype == "float32" and arg1_dtype == "float32":
        assert out_dtype == "float32" and accumulator_dtype == "float32"
        math_instructions = [
            MathInstruction(
                [1, 1, 1],
                DataType.f32,
                DataType.f32,
                DataType.f32,
                DataType.f32,
                OpcodeClass.Simt,
                MathOperation.multiply_add,
            )
        ]
        alignment_constraints = [1]
        tile_descriptions = [
            ([128, 128, 8], 2, [4, 2, 1], min_cc, max_cc),
            ([128, 64, 8], 2, [2, 2, 1], min_cc, max_cc),
            ([64, 128, 8], 2, [2, 2, 1], min_cc, max_cc),
            ([64, 64, 8], 2, [2, 1, 1], min_cc, max_cc),
            ([128, 32, 8], 2, [2, 1, 1], min_cc, max_cc),
            ([32, 128, 8], 2, [1, 2, 1], min_cc, max_cc),
        ]

        def get_tile_descriptions(math_inst):
            return [
                TileDescription(threadblock_shape, stages, warp_count, math_inst, min_cc, max_cc)
                for threadblock_shape, stages, warp_count, min_cc, max_cc in tile_descriptions
            ]

        return generate_tensor_op_common(
            math_instructions, alignment_constraints, get_tile_descriptions, op_creator
        )
    else:
        raise NotImplementedError()


def generate_sm75_tensor_op_1688(
    out_dtype,
    arg0_dtype,
    arg1_dtype,
    op_creator,
    check_align,
    _,
    profile_all_alignments=False,
    accumlator_dtype="float32",
):
    """Generate GEMM or Conv2D kernels for Turing."""
    assert out_dtype in ["float32", "float16", "int32"]
    min_cc = 75
    max_cc = 1024

    if arg0_dtype == "float16" and arg1_dtype == "float16":
        math_instructions = [
            MathInstruction(
                [16, 8, 8],
                DataType.f16,
                DataType.f16,
                dtype_map[out_dtype],
                dtype_map[accumlator_dtype],
                OpcodeClass.TensorOp,
                MathOperation.multiply_add,
            )
        ]
        alignment_constraints = [8, 4, 2, 1]
        tile_descriptions = [
            ([256, 128, 32], 2, [4, 2, 1], min_cc, max_cc),
            ([128, 256, 32], 2, [2, 4, 1], min_cc, max_cc),
            ([128, 128, 32], 2, [2, 2, 1], min_cc, max_cc),
            ([64, 128, 32], 2, [2, 2, 1], min_cc, max_cc),
            ([128, 64, 32], 2, [2, 2, 1], min_cc, max_cc),
            ([64, 64, 32], 2, [2, 2, 1], min_cc, max_cc),
            ([64, 128, 64], 2, [1, 2, 2], min_cc, max_cc),
        ]

    elif "int8" in arg0_dtype and "int8" in arg1_dtype:
        assert out_dtype == "int32"
        math_instructions = [
            MathInstruction(
                [8, 8, 16],
                dtype_map[arg0_dtype],
                dtype_map[arg1_dtype],
                DataType.s32,
                DataType.s32,
                OpcodeClass.TensorOp,
                MathOperation.multiply_add_saturate,
            )
        ]
        alignment_constraints = [16, 8, 4, 2, 1]
        tile_descriptions = [
            ([256, 128, 64], 2, [4, 2, 1], min_cc, max_cc),
            ([128, 256, 64], 2, [2, 4, 1], min_cc, max_cc),
            ([128, 128, 64], 2, [2, 2, 1], min_cc, max_cc),
            ([64, 256, 64], 2, [1, 4, 1], min_cc, max_cc),
            ([256, 64, 64], 2, [4, 1, 1], min_cc, max_cc),
            ([64, 128, 64], 2, [2, 2, 1], min_cc, max_cc),
            ([128, 64, 64], 2, [2, 2, 1], min_cc, max_cc),
            ([64, 64, 64], 2, [2, 2, 1], min_cc, max_cc),
        ]
    elif arg0_dtype == "float32" and arg1_dtype == "float32" and out_dtype == "float32":
        return generate_sm50_simt(out_dtype, arg0_dtype, arg1_dtype, op_creator, accumlator_dtype)
    else:
        raise NotImplementedError()

    alignment_constraints = [align for align in alignment_constraints if check_align(align)]
    assert len(alignment_constraints) > 0

    if not profile_all_alignments:
        alignment_constraints = [alignment_constraints[0]]

    def get_tile_descriptions(math_inst):
        return [
            TileDescription(threadblock_shape, stages, warp_count, math_inst, min_cc, max_cc)
            for threadblock_shape, stages, warp_count, min_cc, max_cc in tile_descriptions
        ]

    return generate_tensor_op_common(
        math_instructions, alignment_constraints, get_tile_descriptions, op_creator
    )


def generate_sm80_tensor_op_16816(
    out_dtype,
    arg0_dtype,
    arg1_dtype,
    op_creator,
    check_align,
    use_3xtf32=True,
    profile_all_alignments=False,
    accumlator_dtype="float32",
):
    """Generate GEMM or Conv2D kernels for Ampere."""
    min_cc = 80
    max_cc = 1024
    max_cc_smem_limited = 80

    def get_default_tile_descriptions(block_k_factor):
        return [
            ([128, 256, int(32 * block_k_factor)], 3, [2, 4, 1], min_cc, max_cc),
            ([256, 128, int(32 * block_k_factor)], 3, [4, 2, 1], min_cc, max_cc),
            ([256, 64, int(32 * block_k_factor)], 3, [4, 1, 1], min_cc, max_cc),
            ([256, 64, int(32 * block_k_factor)], 4, [4, 1, 1], min_cc, max_cc),
            ([64, 256, int(32 * block_k_factor)], 4, [1, 4, 1], min_cc, max_cc),
            ([128, 128, int(32 * block_k_factor)], 3, [2, 2, 1], min_cc, max_cc),
            ([128, 128, int(32 * block_k_factor)], 4, [2, 2, 1], min_cc, max_cc),
            ([128, 128, int(32 * block_k_factor)], 5, [2, 2, 1], min_cc, max_cc),
            ([128, 64, int(32 * block_k_factor)], 6, [2, 2, 1], min_cc, max_cc),
            ([64, 128, int(32 * block_k_factor)], 6, [2, 2, 1], min_cc, max_cc),
            ([64, 64, int(32 * block_k_factor)], 10, [2, 2, 1], min_cc, max_cc),
            ([256, 128, int(64 * block_k_factor)], 3, [4, 2, 1], min_cc, max_cc_smem_limited),
            ([128, 256, int(64 * block_k_factor)], 3, [2, 4, 1], min_cc, max_cc_smem_limited),
            ([256, 64, int(64 * block_k_factor)], 4, [4, 1, 1], min_cc, max_cc_smem_limited),
            ([64, 256, int(64 * block_k_factor)], 4, [1, 4, 1], min_cc, max_cc_smem_limited),
            ([128, 128, int(64 * block_k_factor)], 4, [2, 2, 1], min_cc, max_cc),
            ([256, 64, int(64 * block_k_factor)], 3, [4, 1, 1], min_cc, max_cc),
            ([64, 256, int(64 * block_k_factor)], 3, [1, 4, 1], min_cc, max_cc),
            ([128, 128, int(64 * block_k_factor)], 3, [2, 2, 1], min_cc, max_cc),
            ([128, 64, int(64 * block_k_factor)], 3, [2, 2, 1], min_cc, max_cc),
            ([64, 128, int(64 * block_k_factor)], 3, [2, 2, 1], min_cc, max_cc),
            ([64, 64, int(64 * block_k_factor)], 5, [2, 2, 1], min_cc, max_cc),
        ]

    if arg0_dtype == "float16" and arg1_dtype == "float16":
        math_instructions = [
            MathInstruction(
                [16, 8, 16],
                DataType.f16,
                DataType.f16,
                dtype_map[out_dtype],
                dtype_map[accumlator_dtype],
                OpcodeClass.TensorOp,
                MathOperation.multiply_add,
            )
        ]
        alignment_constraints = [8, 4, 2]
        tile_descriptions = get_default_tile_descriptions(1)
    elif arg0_dtype == "float32" and arg1_dtype == "float32":
        math_instructions = [
            MathInstruction(
                [16, 8, 8],
                DataType.f32,
                DataType.f32,
                DataType.f32,
                DataType.f32,
                OpcodeClass.TensorOp,
                MathOperation.multiply_add_fast_f32 if use_3xtf32 else MathOperation.multiply_add,
            )
        ]
        alignment_constraints = [4, 2, 1]

        if use_3xtf32:
            # tf32
            tile_descriptions = [
                ([128, 128, 16], 4, [4, 2, 1], min_cc, max_cc),
                ([128, 128, 16], 3, [4, 2, 1], min_cc, max_cc),
                ([256, 64, 16], 3, [4, 2, 1], min_cc, max_cc),
                ([64, 256, 16], 3, [2, 4, 1], min_cc, max_cc),
                ([128, 64, 16], 4, [2, 2, 1], min_cc, max_cc),
                ([64, 128, 16], 4, [2, 2, 1], min_cc, max_cc),
                ([64, 64, 16], 3, [2, 2, 1], min_cc, max_cc),
                ([128, 128, 32], 3, [4, 2, 1], min_cc, max_cc),
                ([256, 64, 32], 3, [4, 2, 1], min_cc, max_cc_smem_limited),
                ([64, 256, 32], 3, [2, 4, 1], min_cc, max_cc_smem_limited),
                ([128, 64, 32], 3, [2, 2, 1], min_cc, max_cc),
                ([64, 128, 32], 3, [2, 2, 1], min_cc, max_cc),
                ([64, 64, 32], 3, [2, 2, 1], min_cc, max_cc),
            ]
        else:
            tile_descriptions = get_default_tile_descriptions(0.5)
    else:
        assert out_dtype == "int32"
        math_instructions = [
            MathInstruction(
                [16, 8, 32],
                dtype_map[arg0_dtype],
                dtype_map[arg1_dtype],
                DataType.s32,
                DataType.s32,
                OpcodeClass.TensorOp,
                MathOperation.multiply_add_saturate,
            )
        ]
        alignment_constraints = [16, 8, 4]
        tile_descriptions = get_default_tile_descriptions(2)

    def get_tile_descriptions(math_inst):
        return [
            TileDescription(threadblock_shape, stages, warp_count, math_inst, min_cc, max_cc)
            for threadblock_shape, stages, warp_count, min_cc, max_cc in tile_descriptions
        ]

    alignment_constraints = [align for align in alignment_constraints if check_align(align)]

    if len(alignment_constraints) > 0 and not profile_all_alignments:
        alignment_constraints = [alignment_constraints[0]]

    if arg0_dtype != "float32" and arg1_dtype != "float32":
        sm75_kernels = generate_sm75_tensor_op_1688(
            out_dtype,
            arg0_dtype,
            arg1_dtype,
            op_creator,
            check_align,
            False,
            profile_all_alignments,
            accumlator_dtype=accumlator_dtype,
        )
    else:
        # TF32 (float32 + float32 case) is only supported on sm80
        sm75_kernels = []

    if len(alignment_constraints) > 0:
        sm80_kernels = generate_tensor_op_common(
            math_instructions, alignment_constraints, get_tile_descriptions, op_creator
        )
    else:
        sm80_kernels = []

    # TODO(masahi): For int8 kernels, The CUTLASS generator modifies the output tensor alignment
    # after ops are created. Revisit how important this modification is.
    # for op in operations:
    #     if op.tile_description.threadblock_shape[1] >= 128:
    #         op.C.alignment = 16
    #     else:
    #         op.C.alignment = 8

    return sm75_kernels + sm80_kernels


GENERATOR_FUNC_TABLE = {75: generate_sm75_tensor_op_1688, 80: generate_sm80_tensor_op_16816}


# (Epilogue functor name, no_beta_scaling)
EPILOGUE_MAP = {
    "cutlass.dense": (EpilogueFunctor.LinearCombination, False),
    "cutlass.dense_bias": (EpilogueFunctor.LinearCombinationBias, True),
    "cutlass.dense_bias_relu": (EpilogueFunctor.LinearCombinationRelu, True),
    "cutlass.dense_bias_gelu_fp16": (EpilogueFunctor.LinearCombinationGelu, False),
    "cutlass.dense_bias_gelu_fp32": (EpilogueFunctor.LinearCombinationGelu, False),
    "cutlass.matmul": (EpilogueFunctor.LinearCombination, False),
    "cutlass.matmul_bias": (EpilogueFunctor.LinearCombinationBias, True),
    "cutlass.matmul_bias_relu": (EpilogueFunctor.LinearCombinationRelu, True),
    "cutlass.matmul_bias_gelu": (EpilogueFunctor.LinearCombinationGelu, False),
    "cutlass.matmul_transposed": (EpilogueFunctor.LinearCombination, False),
    "cutlass.matmul_transposed_bias": (EpilogueFunctor.LinearCombinationBias, True),
    "cutlass.matmul_transposed_bias_relu": (EpilogueFunctor.LinearCombinationRelu, True),
    "cutlass.matmul_transposed_bias_gelu": (EpilogueFunctor.LinearCombinationGelu, False),
    "cutlass.batch_matmul": (EpilogueFunctor.LinearCombination, False),
    "cutlass.conv2d_bias_hardswish": (EpilogueFunctor.LinearCombinationHardSwish, False),
    "cutlass.conv2d_bias_silu": (EpilogueFunctor.LinearCombinationSilu, False),
    "cutlass.conv2d_bias_sigmoid": (EpilogueFunctor.LinearCombinationSigmoid, False),
    "cutlass.conv2d_bias_relu": (EpilogueFunctor.LinearCombinationRelu, True),
    "cutlass.conv2d_bias": (EpilogueFunctor.LinearCombinationBias, True),
    "cutlass.conv2d": (EpilogueFunctor.LinearCombination, False),
    "cutlass.conv2d_transpose": (EpilogueFunctor.LinearCombination, False),
    "cutlass.conv2d_backward_weight": (EpilogueFunctor.LinearCombination, False),
}


class ProfilerEngine:
    """Compile and run a given profiler executable."""

    def __init__(self, cuda_arch, cutlass_path, binary_prefix):
        self.cuda_arch = cuda_arch
        self.binary_prefix = binary_prefix
        self.cutlass = cutlass_path
        self.cflags = f"-I{cutlass_path}/include -I{cutlass_path}/tools/util/include -O3 -std=c++17"
        self.cflags += " -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
        self.cflags += (
            f" -gencode=arch=compute_{cuda_arch},code=[sm_{cuda_arch},compute_{cuda_arch}]"
        )
        self.cflags += " -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing"
        self.cmd = "nvcc {cflags} {src} -o {output}"

    def _compile(self, op):
        os.makedirs(self.binary_prefix, exist_ok=True)
        opath = os.path.join(self.binary_prefix, op["name"])
        if os.path.exists(opath):
            return
        fi = tempfile.NamedTemporaryFile("w", delete=False, prefix=self.binary_prefix, suffix=".cu")
        fi.write(op["src"])
        fi.close()
        cmd = self.cmd.format(cflags=self.cflags, src=fi.name, output=opath)
        logger.info("invoking compilation %s", cmd)
        os.system(cmd)
        os.unlink(fi.name)

    def compile_all(self, ops, use_multiprocessing=False):
        """Compile all profiler executables."""
        if use_multiprocessing:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            pool.map(self._compile, ops)
        else:
            for op in ops:
                self._compile(op)

    def evaluate(self, op, args):
        """Run the profiler executable corresponding to op_name with args."""
        op_name = op["name"]
        opath = os.path.join(self.binary_prefix, op_name)
        if not os.path.exists(opath):
            self._compile(op)
        if not os.path.exists(opath):
            # Bail out if compilation fails for a whatever reason (e.g. static assert failure)
            return float("inf")
        cmd = [opath]
        for arg in args:
            cmd.append(str(arg))
        try:
            logger.info("invoking evaluation %s", cmd)
            sp = subprocess.run(cmd, capture_output=True, check=True)
            rt = float(sp.stdout)
            if rt == 0.0:
                # This seems to happen with split-k using invalid split-k-slices
                rt = float("inf")
            logger.info("%s, %f", op_name, rt)
        except subprocess.CalledProcessError:
            rt = float("inf")
        return rt


class CodegenResult(tvm_ffi.Object):
    """The holder for the generated code and required headers."""

    def __init__(self, code, headers):
        self.__init_handle_by_constructor__(ffi.CodegenResult, code, headers)


def _get_optional_int_annotation(annotations, key, default=None):
    value = annotations.get(key, default)
    if value is None:
        return default
    return int(value)


@tvm_ffi.register_global_func("contrib.cutlass.instantiate_template")
def instantiate_template(func_name, annotations, func_args):
    """Return CUTLASS host code based on a template and the provided annotations.

    Parameters
    ----------
    func_name: str
        A string to identify the type of the kernel (dense/matmul, batched_matmul, or conv2d).

    annotations: tvm_ffi.Map
        Key and value pairs annotated during kernel selection.

    func_args: list
        Names of the function arguments.

    Returns
    -------
    codegen_result : CodegenResult
        Generated CUTLASS host code and required header-file names.
    """
    attrs = {}

    for k in ["lda", "ldb", "ldc", "cutlass_op_def", "cutlass_op_name", "op_type"]:
        if k in annotations:
            attrs[k] = annotations[k]

    headers = ["tvm/ffi/function.h", "tvm/ffi/extra/c_env_api.h"]

    if "relu" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_bias_relu.h")
    elif "gelu" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_gelu.h")
    elif "sigmoid" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_sigmoid.h")
    elif "silu" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_silu.h")
    elif "hardswish" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_hardswish.h")
    else:
        headers.append("cutlass/epilogue/thread/linear_combination.h")

    if "residual" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_residual_block.h")

    def get_dim(shape_annot, var_name, axis_idx, batched_offset=0):
        if isinstance(shape_annot, IntImm):
            return str(int(shape_annot))
        return f"{var_name}->shape[{batched_offset + axis_idx}]"

    def get_batch_stride(stride_annot, arg0_idx, arg1_idx, arg0_axis_idx, arg1_axis_idx):
        if isinstance(stride_annot, IntImm):
            return str(int(stride_annot))
        dim1 = func_args[arg0_idx] + f"->shape[{arg0_axis_idx}]"
        dim2 = func_args[arg1_idx] + f"->shape[{arg1_axis_idx}]"
        return dim1 + " * " + dim2

    def get_flattened_batch_dim(arg_name, batch_rank):
        return " * ".join([f"{arg_name}->shape[{i}]" for i in range(batch_rank)])

    if "decode_matmul" in func_name:
        headers.append("cutlass_kernels/fpA_intB_gemm.h")
        lhs_arg_idx = _get_optional_int_annotation(annotations, "lhs_arg_idx", 0)
        rhs_arg_idx = _get_optional_int_annotation(annotations, "rhs_arg_idx", 1)
        scales_arg_idx = _get_optional_int_annotation(annotations, "scales_arg_idx", 2)
        bias_arg_idx = _get_optional_int_annotation(annotations, "bias_arg_idx", None)
        residual_arg_idx = _get_optional_int_annotation(annotations, "residual_arg_idx", None)

        attrs["A_arg"] = func_args[lhs_arg_idx]
        attrs["B_arg"] = func_args[rhs_arg_idx]
        attrs["scales_arg"] = func_args[scales_arg_idx]
        attrs["activation"] = annotations.get("activation", "identity")
        attrs["bias_stride"] = annotations["bias_stride"]
        attrs["M"] = annotations["M"]
        attrs["group_size"] = annotations["group_size"]

        if not isinstance(attrs["M"], tvm_tirx.ir.IntImm):
            attrs["M"] = get_flattened_batch_dim(
                func_args[lhs_arg_idx], int(annotations["batch_rank"])
            )

        if bias_arg_idx is not None:
            attrs["bias_arg"] = func_args[bias_arg_idx]

        if residual_arg_idx is not None:
            attrs["residual_arg"] = func_args[residual_arg_idx]
            attrs["binary_op"] = annotations["binary_op"]
            attrs["unary_op"] = annotations["unary_op"]

        if annotations["weight_nbit"] == 4:
            attrs["weight_dtype"] = "cutlass::uint4b_t"
            attrs["float_per_int"] = 2
        else:
            assert annotations["weight_nbit"] == 8
            attrs["weight_dtype"] = "uint8_t"
            attrs["float_per_int"] = 1

        code = emit_fp16A_intB_matmul(attrs)
        return CodegenResult(code, headers)

    elif "dense" in func_name or "matmul" in func_name:
        batched = "batch" in annotations
        # dense is equal to transposed_matmul
        transposed = "transposed" in func_name or "dense" in func_name
        lhs_arg_idx = _get_optional_int_annotation(annotations, "lhs_arg_idx", 0)
        rhs_arg_idx = _get_optional_int_annotation(annotations, "rhs_arg_idx", 1)
        if "bias" in func_name:
            bias_arg_idx = _get_optional_int_annotation(annotations, "bias_arg_idx", 2)
        else:
            bias_arg_idx = _get_optional_int_annotation(annotations, "bias_arg_idx", None)
        residual_arg_idx = _get_optional_int_annotation(annotations, "residual_arg_idx", None)

        lhs_arg = func_args[lhs_arg_idx]
        rhs_arg = func_args[rhs_arg_idx]
        lhs_shape = annotations[f"arg{lhs_arg_idx}_shape"]
        rhs_shape = annotations[f"arg{rhs_arg_idx}_shape"]
        lhs_batched_offset = len(lhs_shape) - 2
        rhs_batched_offset = len(rhs_shape) - 2

        attrs["lhs_arg"] = lhs_arg
        attrs["rhs_arg"] = rhs_arg

        if bias_arg_idx is not None:
            attrs["bias_arg"] = func_args[bias_arg_idx]
        if residual_arg_idx is not None:
            attrs["residual_arg"] = func_args[residual_arg_idx]

        attrs["ElementInputA"] = DataTypeTag[dtype_map[annotations[f"arg{lhs_arg_idx}_dtype"]]]
        attrs["ElementInputB"] = DataTypeTag[dtype_map[annotations[f"arg{rhs_arg_idx}_dtype"]]]
        attrs["ElementOutput"] = DataTypeTag[dtype_map[annotations["ret_dtype"]]]

        attrs["K"] = lhs_shape[lhs_batched_offset + 1]
        attrs["M"] = get_dim(lhs_shape[lhs_batched_offset], lhs_arg, 0, lhs_batched_offset)

        if transposed:
            attrs["N"] = get_dim(rhs_shape[rhs_batched_offset], rhs_arg, 0, rhs_batched_offset)
        else:
            attrs["N"] = get_dim(rhs_shape[rhs_batched_offset + 1], rhs_arg, 1, rhs_batched_offset)

        if batched:
            headers.append("cutlass/gemm/device/gemm_batched.h")

            def get_batch_on_arg(arg_name, arg_shape):
                return " * ".join(f"{arg_name}->shape[{i}]" for i in range(len(arg_shape) - 2))

            if isinstance(annotations["batch"], IntImm):
                attrs["batch"] = str(int(annotations["batch"]))
            elif annotations["batch_stride_A"] == 0:
                # 2D x ND
                attrs["batch"] = get_batch_on_arg(rhs_arg, rhs_shape)
            else:
                # ND x 2D or ND x ND
                attrs["batch"] = get_batch_on_arg(lhs_arg, lhs_shape)

            attrs["batch_stride_A"] = get_batch_stride(
                annotations["batch_stride_A"],
                lhs_arg_idx,
                lhs_arg_idx,
                lhs_batched_offset,
                lhs_batched_offset + 1,
            )
            attrs["batch_stride_B"] = get_batch_stride(
                annotations["batch_stride_B"],
                rhs_arg_idx,
                rhs_arg_idx,
                rhs_batched_offset,
                rhs_batched_offset + 1,
            )

            if transposed:
                attrs["batch_stride_C"] = get_batch_stride(
                    annotations["batch_stride_C"],
                    lhs_arg_idx,
                    rhs_arg_idx,
                    lhs_batched_offset,
                    rhs_batched_offset,
                )
            else:
                attrs["batch_stride_C"] = get_batch_stride(
                    annotations["batch_stride_C"],
                    lhs_arg_idx,
                    rhs_arg_idx,
                    lhs_batched_offset,
                    rhs_batched_offset + 1,
                )
        else:
            headers.append("cutlass/gemm/device/gemm.h")

        if "residual" in func_name:
            headers.append("cutlass/gemm/device/gemm_universal_with_broadcast.h")

        code = instantiate_gemm_template(attrs)
        return CodegenResult(code, headers)

    elif "conv2d" in func_name:
        data_arg_idx = _get_optional_int_annotation(annotations, "data_arg_idx", 0)
        weight_arg_idx = _get_optional_int_annotation(annotations, "weight_arg_idx", 1)
        bias_arg_idx = _get_optional_int_annotation(annotations, "bias_arg_idx", None)
        residual_arg_idx = _get_optional_int_annotation(annotations, "residual_arg_idx", None)

        attrs["data_arg"] = func_args[data_arg_idx]
        attrs["weight_arg"] = func_args[weight_arg_idx]

        if bias_arg_idx is not None:
            attrs["bias_arg"] = func_args[bias_arg_idx]
        if residual_arg_idx is not None:
            attrs["residual_arg"] = func_args[residual_arg_idx]

        activation_shape = annotations[f"arg{data_arg_idx}_shape"]
        weight_shape = annotations[f"arg{weight_arg_idx}_shape"]
        output_shape = annotations["ret_shape"]

        if "conv2d_transpose" in func_name:
            headers.append("cutlass/conv/kernel/default_conv2d_dgrad.h")
            activation_shape = output_shape
            output_shape = annotations["arg0_shape"]
        elif "backward" in func_name:
            headers.append("cutlass/conv/kernel/default_conv2d_wgrad.h")
            activation_shape = annotations["arg1_shape"]
            weight_shape = output_shape
            output_shape = annotations["arg0_shape"]
        elif "residual" in func_name:
            headers.append("cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h")
        else:
            headers.append("cutlass/conv/kernel/default_conv2d_fprop.h")

        headers.append("cutlass/conv/device/implicit_gemm_convolution.h")

        op_name = attrs["cutlass_op_name"]

        if "splitk" in op_name:
            headers += [
                "cutlass/reduction/device/reduce_split_k.h",
                "cutlass/reduction/thread/reduction_operators.h",
            ]

        data_arg = attrs["data_arg"]
        attrs["N"] = get_dim(activation_shape[0], data_arg, 0)
        attrs["H"] = get_dim(activation_shape[1], data_arg, 1)
        attrs["W"] = get_dim(activation_shape[2], data_arg, 2)
        attrs["C"] = activation_shape[3]
        attrs["P"] = get_dim(output_shape[1], "out0", 1)
        attrs["Q"] = get_dim(output_shape[2], "out0", 2)
        attrs["K"] = output_shape[3]
        attrs["R"] = weight_shape[1]
        attrs["S"] = weight_shape[2]
        attrs["pad_h"] = annotations["padding"][0]
        attrs["pad_w"] = annotations["padding"][1]
        attrs["stride_h"] = annotations["strides"][0]
        attrs["stride_w"] = annotations["strides"][1]
        attrs["dilation_h"] = annotations["dilation"][0]
        attrs["dilation_w"] = annotations["dilation"][1]

        if "splitk" in op_name:
            attrs["split_k_mode"] = "kParallel"
            attrs["split_k_slices"] = str(re.search(r"splitk(\d+)", op_name).group(1))
        else:
            attrs["split_k_mode"] = "kSerial"
            attrs["split_k_slices"] = 1

        if "residual_shape" in annotations:
            attrs["residual_shape"] = annotations["residual_shape"]

        code = instantiate_conv2d_template(attrs)
        return CodegenResult(code, headers)

    elif "attention" in func_name:
        is_var_len = "var_len" in func_name
        data_type = dtype_map[annotations["arg0_dtype"]]

        attrs["qkv_layout"] = annotations["qkv_layout"]
        if attrs["qkv_layout"] == "default":
            attrs["query"] = func_args[0]
            attrs["key"] = func_args[1]
            attrs["value"] = func_args[2]
            attrs["num_queries"] = s = get_dim(annotations["num_queries"], func_args[0], 1)
            attrs["num_keys"] = get_dim(annotations["num_keys"], func_args[1], 1)
            if len(func_args) > 4 and not is_var_len:  # +1 for workspace, the last arg
                attrs["bias"] = func_args[3]
        elif attrs["qkv_layout"] == "qkv_stacked":
            attrs["qkv"] = func_args[0]
            attrs["num_queries"] = s = annotations["num_queries"]
            attrs["num_keys"] = annotations["num_keys"]
            if len(func_args) > 2 and not is_var_len:  # +1 for workspace, the last arg
                attrs["bias"] = func_args[1]
        else:
            raise NotImplementedError()

        attrs["data_type"] = DataTypeTag[data_type]
        attrs["num_batches"] = b = annotations["num_batches"]
        attrs["head_dim"] = h = annotations["head_dim"]
        attrs["head_dim_value"] = h_v = annotations["head_dim_value"]
        attrs["kMaxK"] = max(int(attrs["head_dim"]), int(attrs["head_dim_value"]))
        attrs["scale"] = (
            float(1 / math.sqrt(h.value)) if annotations["scale"] is None else annotations["scale"]
        )

        if is_var_len:
            attrs["seqstart_q"] = func_args[int(annotations["seqstart_q_idx"])]
            attrs["seqstart_k"] = func_args[int(annotations["seqstart_k_idx"])]
            attrs["max_seqlen_q"] = func_args[int(annotations["max_seqlen_q_idx"])]
            attrs["max_seqlen_k"] = func_args[int(annotations["max_seqlen_k_idx"])]

        is_mqa = annotations["num_q_heads"] != annotations["num_kv_heads"]

        use_flash = (
            annotations["ret_dtype"] == "float16"
            and "bias" not in attrs
            and int(attrs["head_dim"]) <= 256
            and int(attrs["head_dim"]) % 8 == 0
            and int(attrs["head_dim"]) == int(attrs["head_dim_value"])
            # For the causal case (custom mask = "BottomRight"), only use flash for multi-query
            # attention workloads. Otherwise, CUTLASS fMHA seems faster for causal attention
            # with a single query.
            # In addition, sliding-window attention is only supported by flash.
            and (
                int(annotations["custom_mask_type"]) == 0
                or (int(annotations["custom_mask_type"]) == 2 and is_mqa)
                or (int(annotations["custom_mask_type"]) == 2 and "window_size" in annotations)
            )
            # Flash v2 is currently not supported for sm < 80
            and int(annotations["arch"]) >= 80
        )

        # See https://github.com/Dao-AILab/flash-attention/blob/
        # 92dd5703ecdb99aa4a4aee9817f28557907403a2/csrc/flash_attn/flash_api.cpp#L111-L116
        if "window_size" in annotations:
            assert use_flash, "Sliding-window attention is supported only by Flash Attention."
            assert int(annotations["custom_mask_type"]) == 2, (
                "Sliding-window attention is only supported for causal with bottom right mask."
            )
            attrs["window_size_left"] = int(annotations["window_size"]) - 1
            attrs["window_size_right"] = 0
            attrs["is_causal"] = False
        else:
            if int(annotations["custom_mask_type"]) == 2:
                attrs["window_size_left"] = attrs["num_keys"]
                attrs["window_size_right"] = 0
                attrs["is_causal"] = True
            else:
                attrs["window_size_left"] = -1
                attrs["window_size_right"] = -1
                attrs["is_causal"] = False

        if use_flash:
            headers.append("flash.h")
            attrs["num_q_heads"] = annotations["num_q_heads"]
            attrs["num_kv_heads"] = annotations["num_kv_heads"]

            if is_var_len:
                code = instantiate_flash_attention_var_len_template(attrs)
            else:
                code = instantiate_flash_attention_template(attrs)
        else:
            headers.append("kernel_forward.h")

            assert not is_mqa, (
                "The number of query and KV heads need to be the same for CUTLASS fMHA."
            )

            attrs["num_heads"] = n = annotations["num_q_heads"]

            data_type_size = DataTypeSize[data_type]
            if (data_type_size * h // 8) % 16 == 0 and (data_type_size * h_v // 8) % 16 == 0:
                attrs["kIsAligned"] = True
            elif (h % 4 == 0) and (h_v % 4 == 0):
                attrs["kIsAligned"] = False
            else:
                raise NotImplementedError()
            if h_v > 64:
                attrs["kQueriesPerBlock"] = 32
                attrs["kKeysPerBlock"] = 128
                attrs["kSingleValueIteration"] = h_v <= 128
            else:
                attrs["kQueriesPerBlock"] = 64
                attrs["kKeysPerBlock"] = 64
                attrs["kSingleValueIteration"] = True

            assert attrs["scale"] > 0 or attrs["scale"] < 0, (
                "Cutlass may generate nan occasionally when scale == 0.0"
            )
            attrs["arch"] = "cutlass::arch::Sm{}".format(annotations["arch"])
            attrs["kSupportsDropout"] = False

            attrs["output_size"] = f"{b} * {s} * {n} * {h_v}"

            attrs["custom_mask_type"] = annotations["custom_mask_type"]

            for arg in func_args:
                if "workspace" in arg:
                    attrs["workspace"] = arg
            if "bias" in attrs:
                attrs["kSupportsBias"] = True
                if len(annotations["bias_shape"]) == 4:
                    strides = "p.num_keys"
                    if annotations["bias_shape"][2] == 1:
                        attrs["bias_strideM"] = 0
                    else:
                        attrs["bias_strideM"] = strides
                        strides = f"p.num_queries * {strides}"
                    if annotations["bias_shape"][1] == 1:
                        attrs["bias_strideH"] = 0
                    else:
                        attrs["bias_strideH"] = strides
                        strides = f"p.num_heads * {strides}"
                    if annotations["bias_shape"][0] == 1:
                        attrs["bias_strideB"] = 0
                    else:
                        attrs["bias_strideB"] = strides
                else:
                    raise NotImplementedError()
            else:
                # To support negative scale in current Cutlass implementation,
                # kSupportsBias should be set true, or there are nan's as result.
                attrs["kSupportsBias"] = attrs["scale"] < 0

            code = instantiate_attention_template(attrs)

        return CodegenResult(code, headers)
    elif "layer_norm" in func_name:
        headers.append("cutlass/util/device_layernorm.h")
        headers.append("cutlass/layout/matrix.h")
        attrs = {"input": func_args[0], "gamma": func_args[1], "beta": func_args[2]}
        attrs.update(dict(annotations))

        if not isinstance(attrs["M"], tvm_tirx.ir.IntImm):
            attrs["M"] = get_flattened_batch_dim(func_args[0], int(attrs["batch_rank"]))

        code = instantiate_layer_norm_template(attrs)
        return CodegenResult(code, headers)
    elif "rms_norm" in func_name:
        headers.append("cutlass/util/device_rmsnorm.h")
        headers.append("cutlass/layout/matrix.h")
        attrs = {"input": func_args[0], "weight": func_args[1]}
        attrs.update(dict(annotations))

        if not isinstance(attrs["M"], tvm_tirx.ir.IntImm):
            attrs["M"] = get_flattened_batch_dim(func_args[0], int(attrs["batch_rank"]))

        code = instantiate_rms_norm_template(attrs)
        return CodegenResult(code, headers)

    raise ValueError(f"Do not have a template for {func_name}")
