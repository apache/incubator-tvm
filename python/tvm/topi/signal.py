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
# pylint: disable=invalid-name, too-many-arguments, too-many-nested-blocks, unused-argument
"""STFT operator"""

from math import pi

import tvm_tirx
from tvm import te
from tvm_tirx import ir, tirx
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as T


def stft(
    data,
    n_fft,
    hop_length,
    win_length,
    window,
    normalized,
    onesided,
    output_shape,
):
    """
    The STFT computes the Fourier transform of short overlapping windows of the input.
    This gives frequency components of the signal as they change over time.
    Parameters
    ----------
    data : te.Tensor
        Either a 1-D tensor or a 2-D batch tensor.
    n_fft : int
        The size of Fourier transform
    hop_length : int
        The distance between neighboring sliding window frames
    win_length : int
        The size of window frame and STFT filter
    window : te.Tensor
        A 1-D tensor window frame
    normalized : bool
        Whether to return the normalized STFT results
    onesided : bool
        Whether to return onesided result or fill with conjugate symmetry
    Returns
    -------
    output : te.Tensor
        Tensor containing the STFT result
    Examples
    --------
    .. code-block:: python

        data = [1, 2, 3, 4, 5, 6]
        window = [4, 3, 2]
        [n_fft, hop_length, win_length, normalized, onesided] = [3, 3, 3, False, True]
        topi.stft(data, n_fft, hop_length, win_length, window, normalized, onesided)
        -> [[[15.0000,  0.0000], [34.0000,  0.0000]], [[ 4.5000,  0.8660], [ 1.0000, -1.7321]]]
    """

    def gen_ir(
        data_ptr,
        n_fft,
        hop_length,
        win_length,
        window_ptr,
        normalized,
        onesided,
        output_ptr,
        loop_kind,
    ):
        col_loop = T.vectorized if loop_kind == "vectorize" else T.serial

        with IRBuilder() as ib:
            data = data_ptr
            window = window_ptr
            output = output_ptr
            # https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
            with T.parallel(0, output_ptr.shape[0] * output_ptr.shape[1]) as batch_row:
                with col_loop(0, output_ptr.shape[2]) as col:
                    batch = tirx.floordiv(batch_row, output_ptr.shape[1])
                    row = tirx.floormod(batch_row, output_ptr.shape[1])
                    T.tensor_store(
                        output,
                        (batch, row, col, 0),
                        tirx.Cast(data_ptr.dtype, 0),
                    )
                    T.tensor_store(
                        output,
                        (batch, row, col, 1),
                        tirx.Cast(data_ptr.dtype, 0),
                    )
                    with T.serial(0, win_length) as wlen:
                        T.tensor_store(
                            output,
                            (batch, row, col, 0),
                            output[(batch, row, col, 0)]
                            + (
                                window[T.tensor_indices(window, wlen)]
                                * data[(batch, col * hop_length + wlen)]
                                * tirx.cos(2 * pi * row * wlen / win_length)
                            ),
                        )
                        T.tensor_store(
                            output,
                            (batch, row, col, 1),
                            output[(batch, row, col, 1)]
                            - (
                                window[T.tensor_indices(window, wlen)]
                                * data[(batch, col * hop_length + wlen)]
                                * tirx.sin(2 * pi * row * wlen / win_length)
                            ),
                        )
                    with T.if_(normalized):
                        with T.then_():
                            T.tensor_store(
                                output,
                                (batch, row, col, 0),
                                output[(batch, row, col, 0)]
                                / (tirx.sqrt(tirx.const(n_fft, "float32"))),
                            )
                            T.tensor_store(
                                output,
                                (batch, row, col, 1),
                                output[(batch, row, col, 1)]
                                / (tirx.sqrt(tirx.const(n_fft, "float32"))),
                            )

            return ib.get()

    output_buf = tirx.Var("output_buf", tirx.TensorType(output_shape, data.dtype, layout=None))
    loop_kind = "vectorize"
    if ir.is_prim_var(output_shape[2]):  # any_dim
        loop_kind = "serial"

    return te.extern(
        output_shape,
        [data, window],
        lambda ins, outs: gen_ir(
            ins[0], n_fft, hop_length, win_length, ins[1], normalized, onesided, outs[0], loop_kind
        ),
        dtype=[data.dtype],
        out_tensors=[output_buf],
        name="stft_cpu",
        tag="stft_cpu",
    )


def dft(
    re_data: te.Tensor,
    im_data: te.Tensor,
    inverse: tvm_tirx.ir.IntImm,
):
    """
    Computes the discrete Fourier transform of input (calculation along the last axis).
    This gives frequency components of the signal as they change over time.

    Parameters
    ----------
    re_data : te.Tensor
        N-D tensor, real part of the input signal.

    im_data : te.Tensor
        N-D tensor, imaginary part of the input signal.
        If the signal is real, then the values of this tensor are zeros.

    inverse : bool
        Whether to perform the inverse discrete fourier transform.

    Returns
    -------
    re_output : te.Tensor
        The Fourier Transform of the input (Real part).
    im_output : te.Tensor
        The Fourier Transform of the input (Imaginary part).
    """

    def gen_ir(
        re_data_buf,
        im_data_buf,
        re_output_buf,
        im_output_buf,
    ):
        with IRBuilder() as ib:
            re_data_ptr = re_data_buf
            im_data_ptr = im_data_buf
            re_output_ptr = re_output_buf
            im_output_ptr = im_output_buf

            shape = re_data.shape
            n_fft = shape[len(shape) - 1]
            base_range = 1
            for i in range(len(shape) - 1):
                base_range *= shape[i]

            sign = -1 if inverse else 1
            factor = 1.0 / n_fft if inverse else 1.0

            with T.parallel(0, base_range) as i:
                base_idx = i * n_fft
                with T.serial(0, n_fft) as n:
                    n_idx = base_idx + n
                    T.tensor_store(
                        re_output_ptr,
                        T.tensor_indices(re_output_ptr, n_idx),
                        tirx.Cast(re_output_ptr.dtype, 0),
                    )
                    T.tensor_store(
                        im_output_ptr,
                        T.tensor_indices(im_output_ptr, n_idx),
                        tirx.Cast(im_output_ptr.dtype, 0),
                    )
                    _w = sign * -2 * pi * n / n_fft
                    with T.serial(0, n_fft) as k:
                        k_idx = base_idx + k
                        w = _w * k
                        cos_w = tirx.Cast(re_output_ptr.dtype, tirx.cos(w))
                        sin_w = tirx.Cast(re_output_ptr.dtype, tirx.sin(w))
                        T.tensor_store(
                            re_output_ptr,
                            T.tensor_indices(re_output_ptr, n_idx),
                            re_output_ptr[T.tensor_indices(re_output_ptr, n_idx)]
                            + (
                                re_data_ptr[T.tensor_indices(re_data_ptr, k_idx)] * cos_w
                                - im_data_ptr[T.tensor_indices(im_data_ptr, k_idx)] * sin_w
                            ),
                        )
                        T.tensor_store(
                            im_output_ptr,
                            T.tensor_indices(im_output_ptr, n_idx),
                            im_output_ptr[T.tensor_indices(im_output_ptr, n_idx)]
                            + (
                                re_data_ptr[T.tensor_indices(re_data_ptr, k_idx)] * sin_w
                                + im_data_ptr[T.tensor_indices(im_data_ptr, k_idx)] * cos_w
                            ),
                        )

                    T.tensor_store(
                        re_output_ptr,
                        T.tensor_indices(re_output_ptr, n_idx),
                        re_output_ptr[T.tensor_indices(re_output_ptr, n_idx)]
                        * (tirx.Cast(re_output_ptr.dtype, factor)),
                    )
                    T.tensor_store(
                        im_output_ptr,
                        T.tensor_indices(im_output_ptr, n_idx),
                        im_output_ptr[T.tensor_indices(im_output_ptr, n_idx)]
                        * (tirx.Cast(im_output_ptr.dtype, factor)),
                    )

            return ib.get()

    output_shape = [re_data.shape] * 2

    return te.extern(
        shape=output_shape,
        inputs=[re_data, im_data],
        fcompute=lambda ins, outs: gen_ir(ins[0], ins[1], outs[0], outs[1]),
        dtype=[re_data.dtype, im_data.dtype],
        name="dft_cpu",
        tag="dft_cpu",
    )
