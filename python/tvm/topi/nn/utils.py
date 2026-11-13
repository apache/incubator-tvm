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
# pylint: disable=invalid-name, unused-variable
"""NN operator common utilities"""

import tvm
import tvm_tirx

from ..utils import get_const_int


def infer_pad(data, data_pad):
    """Infer the padding from stages in reverse.

    Parameters
    ----------
    data : Tensor
        data stage.

    data_pad : Tensor
        pad stage.

    Returns
    -------
    hpad : int
        padding size on height
    wpad : int
        padding size on width
    """
    if data_pad is None:
        return 0, 0
    _, _, IH, IW = data.shape
    _, _, TH, TW = data_pad.shape
    hpad = (TH - IH) // 2
    wpad = (TW - IW) // 2
    return get_const_int(hpad), get_const_int(wpad)


def infer_pad3d(data, data_pad, layout):
    """Infer the padding from stages in reverse.

    Parameters
    ----------
    data : Tensor
        data stage.

    data_pad : Tensor
        pad stage.

    Returns
    -------
    dpad : int
        padding depth
    hpad : int
        padding height
    wpad : int
        padding width
    """
    if data_pad is None:
        return 0, 0, 0

    if layout == "NDHWC":
        _, ID, IH, IW, _ = data.shape
        _, TD, TH, TW, _ = data_pad.shape
    elif layout == "NCDHW":
        _, _, ID, IH, IW = data.shape
        _, _, TD, TH, TW = data_pad.shape
    else:
        raise ValueError(f"Layout {layout} is not supported")
    dpad = TD - ID
    hpad = TH - IH
    wpad = TW - IW
    return get_const_int(dpad), get_const_int(hpad), get_const_int(wpad)


def infer_stride(data, kernel, out):
    """Infer the stride from stages in reverse.

    Parameters
    ----------
    data : Tensor
        data stage.

    kernel : Tensor
        kernel stage.

    out : Tensor
        output stage.

    Returns
    -------
    hstride : int
        stride size on height
    wstride : int
        stride size on width
    """
    _, _, IH, IW = data.shape
    _, _, KH, KW = kernel.shape
    _, _, OH, OW = out.shape
    hstride = (IH - KH) // tvm.te.max(OH - 1, 1) + tvm_tirx.tirx.Select(OH == 1, 1, 0)
    wstride = (IW - KW) // tvm.te.max(OW - 1, 1) + tvm_tirx.tirx.Select(OW == 1, 1, 0)
    return get_const_int(hstride), get_const_int(wstride)


def get_pad_tuple(padding, kernel):
    """Common code to get the pad option

    Parameters
    ----------
    padding : int or str
        Padding size, or ['VALID', 'SAME']

    kernel : tuple of int
        Conv kernel size

    Returns
    -------
    pad_top : int
        Padding size on top

    pad_left : int
        Padding size on left

    pad_down : int
        Padding size on down.

    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, tuple | list):
        if len(padding) == 2:
            pad_h = padding[0] * 2
            pad_w = padding[1] * 2
        elif len(padding) == 4:
            return padding[0], padding[1], padding[2], padding[3]
        else:
            raise ValueError("Size of padding can only be 2 or 4")
    elif isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif padding == "VALID":
        pad_h = 0
        pad_w = 0
    elif padding == "SAME":
        pad_h = kernel[0] - 1
        pad_w = kernel[1] - 1
    else:
        raise ValueError(f"Unknown padding option {padding}")
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left


def get_pad_tuple_generic(padding, kernel):
    """Common code to get the pad option

    Parameters
    ----------
    padding : int or str
        Padding size, or ['VALID', 'SAME']

    kernel : tuple of int
        Conv kernel size

    Returns
    -------
    pad_top : int
        Padding size on top

    pad_down : int
        Padding size on down.

    pad_left : int
        Padding size on left

    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, tuple | list):
        if len(padding) == len(kernel):
            pad_dimensions = [p * 2 for p in padding]
        elif len(padding) == len(kernel) * 2:
            return (
                [padding[i] for i in range(len(kernel))],
                [padding[len(kernel) + i] for i in range(len(kernel))],
            )
        else:
            raise ValueError("Size of padding can only be len(kernel) or len(kernel) * 2")
    elif isinstance(padding, int):
        pad_dimensions = [padding * 2 for _ in range(len(kernel))]
    elif padding == "VALID":
        pad_dimensions = [0 for _ in range(len(kernel))]
    elif padding == "SAME":
        pad_dimensions = [k - 1 for k in kernel]
    else:
        raise ValueError(f"Unknown padding option {padding}")
    pad_begin = [(p + 1) // 2 for p in pad_dimensions]
    return [pad_begin, [pd - pb for pb, pd in zip(pad_begin, pad_dimensions)]]


def get_pad_tuple3d(padding, kernel):
    """Common code to get the pad option

    Parameters
    ----------
    padding : int or str
        Padding size, or ['VALID', 'SAME']

    kernel : tuple of int
        Conv kernel size

    Returns
    -------
    pad_front : int
        Padding size on front.

    pad_top : int
        Padding size on top

    pad_left : int
        Padding size on left

    pad_back : int
        Padding size on back.

    pad_down : int
        Padding size on down.

    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, tuple | list):
        if len(padding) == 3:
            pad_d = padding[0] * 2
            pad_h = padding[1] * 2
            pad_w = padding[2] * 2
        elif len(padding) == 6:
            return padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]
        else:
            raise ValueError("Size of padding can only be 3 or 6")
    elif isinstance(padding, int):
        pad_d = pad_w = pad_h = padding * 2
    elif padding == "VALID":
        pad_h = 0
        pad_w = 0
        pad_d = 0
    elif padding == "SAME":
        pad_d = kernel[0] - 1
        pad_h = kernel[1] - 1
        pad_w = kernel[2] - 1
    else:
        raise ValueError(f"Unknown padding option {padding}")
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    pad_front = (pad_d + 1) // 2
    return pad_front, pad_top, pad_left, pad_d - pad_front, pad_h - pad_top, pad_w - pad_left


def get_pad_tuple1d(padding, kernel):
    """Common code to get the pad option

    Parameters
    ----------
    padding : int or str
        Padding size, or ['VALID', 'SAME']

    kernel : tuple of int
        Conv kernel size

    Returns
    -------
    pad_left : int
        Padding size on left

    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, tuple | list):
        if len(padding) == 1:
            pad_w = padding[0] * 2
        elif len(padding) == 2:
            return padding[0], padding[1]
        else:
            raise ValueError("Size of padding can only be 2 or 4")
    elif isinstance(padding, int):
        pad_w = padding * 2
    elif padding == "VALID":
        pad_w = 0
    elif padding == "SAME":
        pad_w = kernel[0] - 1
    else:
        raise ValueError(f"Unknown padding option {padding}")
    pad_left = (pad_w + 1) // 2
    return pad_left, pad_w - pad_left
