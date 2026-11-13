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
"""Commons for Relax frontend."""

import numpy as _np

import tvm_tirx
from tvm import topi


def detach_params(
    mod: tvm_tirx.IRModule,
) -> tuple[tvm_tirx.IRModule, dict[str, list[tvm_tirx.runtime.Tensor]]]:
    """Detach the attribute "params" in the functions of the input IRModule as
    separate dictionary of params.

    Parameters
    ----------
    mod : tvm_tirx.IRModule
        The IRModule whose functions' "param" attribute is going to be detached.

    Returns
    -------
    detached_mod : tvm_tirx.IRModule
        The IRModule after the detachment.

    params_dict : Dict[str, List[tvm_tirx.runtime.Tensor]]
        The detached params. The dict keys corresponds to the names of the
        functions in the input IRModule that have attribute "params".
    """
    detached_mod = tvm_tirx.IRModule()
    params_dict = dict()
    for gv, func in mod.functions_items():
        if "params" in func.attrs:
            params = list(func.attrs["params"])
            if not all([isinstance(param, tvm_tirx.runtime.Tensor) for param in params]):
                raise ValueError('The value "params" attribute is expected to be a list of Tensor.')
            params_dict[gv.name_hint] = params
            detached_mod[gv] = func.without_attr("params")
        else:
            detached_mod[gv] = func
    return detached_mod, params_dict


def autopad(
    bb,
    data,
    strides,
    kernel_shape,
    dilations=(1, 1),
    pad_type="constant",
    deconv=False,
    mode="SAME_UPPER",
    pad_value=0.0,
):
    """
    Perform autopadding with dynamic input shapes
    """
    # get attributes as constants
    strides = _np.array(strides)
    dilated_kernel_shape = _np.array(
        [(kernel - 1) * dilation + 1 for kernel, dilation in zip(kernel_shape, dilations)]
    )
    # get input shape
    ndim = data.ty.ndim
    data_shape = list(data.ty.shape)
    shape = data_shape[2:ndim]

    # set up integer constants
    zero = 0
    one = 1
    two = 2

    # Calculate total padding
    mod = shape % strides

    left = _np.maximum(dilated_kernel_shape - strides, zero)
    right = _np.maximum(dilated_kernel_shape - mod, zero)

    total_pad = _np.where(_np.equal(mod, zero), left, right)
    if deconv:
        total_pad = _np.array(kernel_shape) - one - total_pad

    # split total padding into before and after
    pad_before = _np.floor_divide(total_pad, two)
    pad_after = total_pad - pad_before

    # combine
    if "LOWER" in mode:
        pad = _np.concatenate(
            [_np.reshape(pad_after, [-1, 1]), _np.reshape(pad_before, [-1, 1])], axis=1
        )
    else:
        pad = _np.concatenate(
            [_np.reshape(pad_before, [-1, 1]), _np.reshape(pad_after, [-1, 1])], axis=1
        )

    # pad N and C with zeros
    pad = _np.concatenate([_np.zeros([2, 2], dtype="int64"), pad], axis=0)

    if pad_type not in ["constant", "edge", "reflect"]:
        raise AttributeError(
            "Value " + pad_type + ' in attribute "mode" is invalid for operator Pad.'
        )

    if pad_type == "constant":
        return bb.emit_te(topi.nn.pad, data, pad[:, 0].tolist(), pad[:, 1].tolist(), pad_value)
    elif pad_type == "reflect":
        return bb.emit_te(
            topi.nn.mirror_pad, data, pad[:, 0].tolist(), pad[:, 1].tolist(), "REFLECT"
        )
    else:
        # edge mode - replicate border values
        return bb.emit_te(topi.nn.replicate_pad, data, pad[:, 0].tolist(), pad[:, 1].tolist())
