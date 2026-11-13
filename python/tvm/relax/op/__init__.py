# isort: skip_file
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
# pylint: disable= redefined-builtin
"""Relax core operators."""

# Operator families register their attrs before exposure.
from . import builtin, ccl, distributed, grad, image, memory, nn

# Operators
from .base import (
    CallTIRWithGradAttrs,
    CallInplacePackedAttrs,
    CallTIRInplaceAttrs,
    ToVDeviceAttrs,
    HintOnDeviceAttrs,
    assert_op,
    call_builtin_with_ctx,
    call_dps_packed,
    call_inplace_packed,
    call_pure_packed,
    call_py_func,
    call_tir,
    call_tir_inplace,
    call_tir_packed,
    call_tir_with_grad,
    hint_on_device,
    invoke_closure,
    invoke_pure_closure,
    make_closure,
    null_value,
    print,
    register_gradient,
    shape_of,
    shape_to_tensor,
    size,
    tensor_to_shape,
    to_vdevice,
)
from .binary import (
    add,
    atan2,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    divide,
    equal,
    floor_divide,
    log_add_exp,
    floor_mod,
    greater,
    greater_equal,
    left_shift,
    less,
    less_equal,
    logical_and,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    mod,
    multiply,
    not_equal,
    power,
    right_shift,
    subtract,
)
from .create import (
    InitAttrs,
    TriluAttrs,
    arange,
    full,
    full_like,
    hamming_window,
    ones,
    ones_like,
    eye,
    eye_like,
    tril,
    triu,
    zeros,
    zeros_like,
)
from .datatype import (
    AstypeAttrs,
    WrapParamAttrs,
    astype,
    wrap_param,
)
from .index import (
    TakeAttrs,
    StridedSliceAttrs,
    dynamic_strided_slice,
    strided_slice,
    take,
)
from .linear_algebra import (
    MatmulAttrs,
    EinsumAttrs,
    einsum,
    linear,
    matmul,
    outer,
)
from .manipulate import (
    ConcatAttrs,
    ExpandDimsAttrs,
    PermuteDimsAttrs,
    SplitAttrs,
    SqueezeAttrs,
    StackAttrs,
    IndexPutAttrs,
    LayoutTransformAttrs,
    RepeatAttrs,
    TileAttrs,
    FlipAttrs,
    ReverseSequenceAttrs,
    GatherElementsAttrs,
    GatherNDAttrs,
    MeshgridAttrs,
    ScatterElementsAttrs,
    ScatterNDAttrs,
    SliceScatterAttrs,
    OneHotAttrs,
    broadcast_to,
    collapse_sum_like,
    collapse_sum_to,
    concat,
    expand_dims,
    flatten,
    flip,
    gather_elements,
    gather_nd,
    index_put,
    index_tensor,
    meshgrid,
    layout_transform,
    one_hot,
    permute_dims,
    repeat,
    reshape,
    reverse_sequence,
    scatter_elements,
    scatter_nd,
    slice_scatter,
    split,
    squeeze,
    stack,
    tile,
)
from .mask import masked_fill
from .qdq import (
    QuantizeAttrs,
    dequantize,
    quantize,
)
from .sampling import (
    MultinomialFromUniformAttrs,
    multinomial_from_uniform,
)
from .search import (
    ArgmaxArgminAttrs,
    argmax,
    argmin,
    where,
    bucketize,
)
from .set import nonzero, unique
from .sorting import (
    SortAttrs,
    ArgsortAttrs,
    TopKAttrs,
    argsort,
    sort,
    topk,
)
from .statistical import (
    StatisticalAttrs,
    ScanopAttrs,
    cumprod,
    cumsum,
    max,
    mean,
    min,
    prod,
    std,
    sum,
    variance,
    median,
)
from .ternary import ewise_fma
from .unary import (
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    bitwise_not,
    ceil,
    clip,
    cos,
    cosh,
    erf,
    exp,
    floor,
    isfinite,
    isinf,
    isnan,
    log,
    logical_not,
    negative,
    round,
    rsqrt,
    sigmoid,
    sign,
    sin,
    sinh,
    sqrt,
    square,
    tan,
    tanh,
    trunc,
)
from .vision import (
    all_class_non_max_suppression,
    get_valid_counts,
    multibox_transform_loc,
    non_max_suppression,
    roi_align,
    roi_pool,
)


# Register gradients after the operator families and their attrs are available.
from . import _op_gradient


def _register_op_make():
    # pylint: disable=import-outside-toplevel
    from .. import expr
    from tvm_tirx.ir import _tensor_expr_overload
    from . import _ffi_api

    def _add(lhs, rhs):
        if isinstance(lhs.ty, expr.tvm.relax.TupleType) and isinstance(rhs, tuple):
            return tuple([*lhs, *rhs])
        return expr._binary_op_helper(lhs, rhs, _ffi_api.add)

    def _rhs(lhs, rhs):
        if not expr._is_tensor_or_missing_type(lhs.ty):
            return NotImplemented
        return expr._binary_rhs_helper(rhs)

    def _unary(lhs, op):
        if not expr._is_tensor_or_missing_type(lhs.ty):
            return NotImplemented
        return op(lhs)

    def _call(func, *args, attrs=None):
        if not isinstance(
            func.ty,
            expr.tvm_tirx.ir.FuncType | expr.tvm.relax.FuncType | expr.tvm_tirx.ir.MissingType,
        ):
            return NotImplemented
        return expr.tvm_tirx.ir.Call(func, args, attrs=attrs)

    operators = {}
    operators["astype"] = (
        lambda lhs, dtype, _loc=None: _ffi_api.astype(lhs, dtype)
        if expr._is_tensor_or_missing_type(lhs.ty)
        else NotImplemented
    )
    operators["__call__"] = _call
    operators["__neg__"] = lambda lhs: _unary(lhs, _ffi_api.negative)
    operators["__lt__"] = lambda lhs, rhs: expr._binary_op_helper(lhs, rhs, _ffi_api.less)
    operators["__le__"] = lambda lhs, rhs: expr._binary_op_helper(lhs, rhs, _ffi_api.less_equal)
    operators["__gt__"] = lambda lhs, rhs: expr._binary_op_helper(lhs, rhs, _ffi_api.greater)
    operators["__ge__"] = lambda lhs, rhs: expr._binary_op_helper(lhs, rhs, _ffi_api.greater_equal)
    operators["__add__"] = _add
    operators["__radd__"] = _add
    operators["__sub__"] = lambda lhs, rhs: expr._binary_op_helper(lhs, rhs, _ffi_api.subtract)
    operators["__rsub__"] = _rhs
    operators["__mul__"] = lambda lhs, rhs: expr._binary_op_helper(lhs, rhs, _ffi_api.multiply)
    operators["__rmul__"] = operators["__mul__"]
    operators["__div__"] = lambda lhs, rhs: expr._binary_op_helper(lhs, rhs, _ffi_api.divide)
    operators["__rdiv__"] = _rhs
    operators["__truediv__"] = operators["__div__"]
    operators["__rtruediv__"] = _rhs
    operators["__floordiv__"] = lambda lhs, rhs: expr._binary_op_helper(
        lhs, rhs, _ffi_api.floor_divide
    )
    operators["__rfloordiv__"] = _rhs
    operators["__mod__"] = lambda lhs, rhs: expr._binary_op_helper(lhs, rhs, _ffi_api.mod)
    operators["__rmod__"] = _rhs
    operators["__pow__"] = lambda lhs, rhs: expr._binary_op_helper(lhs, rhs, _ffi_api.power)
    operators["__rpow__"] = _rhs
    _tensor_expr_overload.register_tensor_expr_operators(operators)


_register_op_make()
