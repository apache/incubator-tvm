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
# pylint: disable=invalid-name,line-too-long
"""Intrinsics for RISCV tensorization"""

import logging

import tvm_ffi

from tvm.s_tir import script as Ts
from tvm_tirx.codegen import llvm, target_has_features
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

from .. import TensorIntrin

logger = logging.getLogger(__name__)


def get_max_elems(vlen: int, lmul: int, sew: int) -> int:
    """Returns number of elements of a given data type (SEW)
    that fits multiple (LMUL) of the vector registers (VLEN).

    Args:
        vlen (int): VLEN vector length in bits
        lmul (int): LMUL vector lenght multiplier
        sew (int): SEW standard (single) element width

    Returns:
        int: Number of elements
    """
    return (vlen // sew) * lmul


def rvv_vec_dot_product_kernels(
    n_elems: int,
    n_lanes: int,
    data_dtype: str,
    weight_dtype: str,
    out_dtype: str,
    lmul: int,
):
    """Dot product of vector and matrix rows using RISC-V vector instructions.

    These kernels takes two arrays A[ELEMS] and B[ELEMS][MACS] and computes
    dot product of A[ELEMS] with each row of B[LANES], accumulating results
    with C[LANES].

    The pseudo code is as follows:

    .. code-block:: c

        void vec_dot_prod(A[ELEMS], B[LANES][ELEMS], C[LANES]){
            for (j = 0; j < LANES; j++) {
                for (k = 0; k < ELEMS; k++) {
                    C[j] += A[k] * B[j][k]
                }
            }
        }
    """

    rvv_vec_dot_prod_desc_a_offset = T.dynamic("rvv_vec_dot_prod_desc_a_offset", "int32")
    rvv_vec_dot_prod_desc_b_offset = T.dynamic("rvv_vec_dot_prod_desc_b_offset", "int32")
    rvv_vec_dot_prod_desc_c_offset = T.dynamic("rvv_vec_dot_prod_desc_c_offset", "int32")

    @Ts.function
    def rvv_vec_dot_prod_desc(
        A: T.Tensor(
            (n_elems,),
            data_dtype,
            layout=T.TileLayout(T.S[(n_elems,)] + (rvv_vec_dot_prod_desc_a_offset)),
        ),
        B: T.Tensor(
            (n_lanes, n_elems),
            weight_dtype,
            layout=T.TileLayout(T.S[(n_lanes, n_elems)] + (rvv_vec_dot_prod_desc_b_offset)),
        ),
        C: T.Tensor(
            (n_lanes,),
            out_dtype,
            layout=T.TileLayout(T.S[(n_lanes,)] + (rvv_vec_dot_prod_desc_c_offset)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:n_lanes], A[0:n_elems], B[0:n_lanes, 0:n_elems])
            Ts.writes(C[0:n_lanes])
            for j in T.serial(0, n_lanes):
                for k in T.serial(0, n_elems):
                    with Ts.sblock("update"):
                        vj, vk = Ts.axis.remap("SR", [j, k])
                        C[vj] = C[vj] + T.cast(A[vk], ty=out_dtype) * T.cast(
                            B[vj, vk], ty=out_dtype
                        )

    # LLVM only supports ELEN=32 or ELEN=64
    # https://llvm.org/docs//RISCV/RISCVVectorExtension.html
    d_dtype_lanes = (64 // tvm_ffi.dtype(data_dtype).bits) * lmul
    w_dtype_lanes = (64 // tvm_ffi.dtype(weight_dtype).bits) * lmul
    # reduction lanes narrows
    o_dtype_lanes = (64 // tvm_ffi.dtype(out_dtype).bits) * lmul // n_lanes
    # data type widening case
    o_dtype_lanes = max(o_dtype_lanes, 2)

    weight_bytes = tvm_ffi.dtype(weight_dtype).bits // 8
    output_bytes = tvm_ffi.dtype(out_dtype).bits // 8
    mask_args = () if data_dtype[0] in ("i", "u") else (T.uint64(7),)

    wide_dtype = out_dtype
    if tvm_ffi.dtype(out_dtype).bits > tvm_ffi.dtype(data_dtype).bits:
        wide_dtype = "".join(c for c in data_dtype if not c.isdigit())
        wide_dtype += str(tvm_ffi.dtype(data_dtype).bits * 2)

    # fmt: off
    rvv_vec_dot_prod_impl_a_offset = T.dynamic("rvv_vec_dot_prod_impl_a_offset", "int32")
    rvv_vec_dot_prod_impl_b_offset = T.dynamic("rvv_vec_dot_prod_impl_b_offset", "int32")
    rvv_vec_dot_prod_impl_c_offset = T.dynamic("rvv_vec_dot_prod_impl_c_offset", "int32")

    @Ts.function
    def rvv_vec_dot_prod_impl(
        A: T.Tensor(
            (n_elems,), data_dtype,
            layout=T.TileLayout(T.S[(n_elems,)] + rvv_vec_dot_prod_impl_a_offset),
        ),
        B: T.Tensor(
            (n_lanes, n_elems), weight_dtype,
            layout=T.TileLayout(T.S[(n_lanes, n_elems)] + rvv_vec_dot_prod_impl_b_offset),
        ),
        C: T.Tensor(
            (n_lanes,), out_dtype,
            layout=T.TileLayout(T.S[(n_lanes,)] + rvv_vec_dot_prod_impl_c_offset),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:n_lanes], A[0:n_elems], B[0:n_lanes, 0:n_elems])
            Ts.writes(C[0:n_lanes])

            vec_A = T.call_llvm_intrin(
                "llvm.riscv.vle",
                T.broadcast(T.Cast(data_dtype, 0), T.vscale() * d_dtype_lanes),
                A.data,
                T.int64(n_elems), ty=f"{data_dtype}xvscalex{d_dtype_lanes}")

            for i in range(n_lanes):
                with Ts.sblock("update"):
                    Ts.reads(B[i, 0:n_elems])
                    Ts.writes(C[i])

                    vec_B_row = T.call_llvm_intrin(
                        "llvm.riscv.vle",
                        T.broadcast(T.Cast(data_dtype, 0), T.vscale() * w_dtype_lanes),
                        T.ptr_byte_offset(B.data, i * n_elems * weight_bytes, ty=B.data.ty),
                        T.int64(n_elems), ty=f"{weight_dtype}xvscalex{w_dtype_lanes}")

                    product = T.call_llvm_intrin(
                        "llvm.riscv.vfmul" if out_dtype[0] == "f" else \
                        "llvm.riscv.vwmulsu" if (data_dtype[0] != weight_dtype[0]) else \
                        "llvm.riscv.vwmul",
                        T.broadcast(T.Cast(wide_dtype, 0), T.vscale() * w_dtype_lanes),
                        vec_B_row,
                        vec_A,
                        *mask_args,
                        T.uint64(n_elems), ty=f"{wide_dtype}xvscalex{w_dtype_lanes}")

                    ini_acc = T.call_llvm_intrin(
                        "llvm.riscv.vle",
                        T.broadcast(T.Cast(out_dtype, 0), T.vscale() * o_dtype_lanes),
                        T.ptr_byte_offset(C.data, i * output_bytes, ty=C.data.ty),
                        T.int64(1), ty=f"{out_dtype}xvscalex{o_dtype_lanes}")

                    red_sum = T.call_llvm_intrin(
                        "llvm.riscv.vfredusum" if out_dtype[0] == "f" else \
                        "llvm.riscv.vwredsum",
                        T.broadcast(T.Cast(out_dtype, 0), T.vscale() * o_dtype_lanes),
                        product,
                        ini_acc,
                        *mask_args,
                        T.uint64(n_elems), ty=f"{out_dtype}xvscalex{o_dtype_lanes}")

                    C[i] = T.call_llvm_intrin(
                        "llvm.riscv.vfmv.f.s" if out_dtype[0] == "f" else \
                        "llvm.riscv.vmv.x.s",
                        red_sum, ty=out_dtype)
    # fmt: on
    return rvv_vec_dot_prod_desc, rvv_vec_dot_prod_impl


@tvm_ffi.register_global_func("tirx.tensor_intrin.register_rvv_isa_intrinsics")
def register_rvv_isa_intrinsics(target: Target, inventory_only=False) -> dict():
    """Register RISCV V (vector) intrinsics
    [x] Implementation follows version 1.0 vector specifications:
        https://github.com/riscvarchive/riscv-v-spec/releases/tag/v1.0

    Args:
        target (Target): TVM target
        inventory_only (bool): No registration inventory only

    Returns:
        dict(): A catalog with registered kernel names and properties
    """
    if not target_has_features("v", target):
        raise RuntimeError("Current target does not support `v` extension.")

    vlen = llvm.get_vector_width(target)
    # get maximum reduction lanes (without grouping)
    n_lanes = get_max_elems(vlen, lmul=1, sew=32)

    kernels_inventory = {}

    data_dtype = ["uint8", "int8", "float16", "float32"]
    weight_dtype = ["int8", "int8", "float16", "float32"]
    output_dtype = ["int32", "int32", "float16", "float32"]

    for d_dtype, w_dtype, o_dtype in zip(data_dtype, weight_dtype, output_dtype):
        # max elements to grouped registers
        max_elems = get_max_elems(vlen, lmul=8, sew=tvm_ffi.dtype(d_dtype).bits)
        # data widening halves available vector registers
        if tvm_ffi.dtype(o_dtype).bits > tvm_ffi.dtype(d_dtype).bits:
            max_elems //= 2
        # compute optimal LMUL for full load
        lmul = max_elems // (vlen // tvm_ffi.dtype(d_dtype).bits)

        n_elems = max_elems
        while n_elems >= 4:
            dt = tvm_ffi.dtype(d_dtype)
            wt = tvm_ffi.dtype(w_dtype)
            ot = tvm_ffi.dtype(o_dtype)
            kernel_name = "rvv_dot"
            kernel_name += f"_{n_elems}{dt[0]}{dt.bits}"
            kernel_name += f"_{n_lanes}x{n_elems}{wt[0]}{wt.bits}"
            kernel_name += f"_{n_lanes}{ot[0]}{ot.bits}"
            kernels_inventory[kernel_name] = n_elems

            if not inventory_only:
                logger.debug(f"Registering kernel {kernel_name}")
                desc, impl = rvv_vec_dot_product_kernels(
                    n_elems, n_lanes, d_dtype, w_dtype, o_dtype, lmul
                )
                TensorIntrin.register(kernel_name, desc, impl, override=True)

            n_elems //= 2

    return kernels_inventory


def register_riscv_intrinsics(target: Target):
    """Register RISCV intrinsics

    Args:
        target (Target): TVM target
    """

    # RISCV `v` 1.0 extension templates
    _ = register_rvv_isa_intrinsics(target)
    logger.debug("Finished registering riscv intrinsics.")
