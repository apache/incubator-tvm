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
# ruff: noqa: F401, F811, RUF005
from collections.abc import Callable
from typing import Union

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax.testing.gradient import check_numerical_grads
from tvm.relax.transform import LegalizeOps
from tvm.relax.type import TensorType, TupleType
from tvm.testing import device_enabled
from tvm_tirx.ir import Call
from tvm_tirx.ir.op import Op


def relax_check_gradients(
    op_func: Callable,
    inputs_numpy: list[np.ndarray],
    target: str | tvm_tirx.target.Target,
    tuple_input: bool = False,
    ignore_grads: list[int] = [],
    **kwargs,  # attr for operators
):
    """Generate the forward and the gradient module. Then run them and check numeric gradients.

    Parameters
    ----------
    op_func : Callable
        The forward operator function. Should be a function in package relax.op.

    inputs_numpy : List[np.array]
        The np array inputs for op_func. inputs_numpy will be transformed into TVM Tensor inside
        this function.

        If op_func takes a tuple of tensors as input, you can set tuple_input as True, and pass the
        tuple input (or list) as inputs_numpy. See test_concat().

    target : Union[str, tvm_tirx.target.Target]
        The building target.

    tuple_input : bool
        Whether the operator accepts a tuple as input. If true, operator will accept exactly one
        tuple of tensors as input; otherwise, operator accept one or more tensors as input. See
        test_concat(). Default: False.

    ignore_grads: List[int]
        Specifies which input we do not need to find gradient.

        Sometimes the input is not differentiable, such as shape, boolean values, positions, etc.
        We can specify the index of these inputs to check the gradient of them is no_grad, and
        prevent computing numeric gradient.

    kwargs : Any
        The keyword arguments for the op_func. Will be passed to op_func directly.
    """

    func_name = "main"
    dev = tvm_tirx.device_from_target(target)

    # Helper functions
    def _numpy_to_ty(data):
        if isinstance(data, list):
            return relax.TupleType([_numpy_to_ty(d) for d in data])
        return relax.TensorType(data.shape, str(data.dtype))

    def _numpy_to_tvm(data):
        if isinstance(data, list):
            return [_numpy_to_tvm(d) for d in data]
        return tvm_tirx.runtime.tensor(data)

    def _tvm_to_numpy(data, ignore_idx=[]):
        if isinstance(data, tvm_ffi.Array):
            return [_tvm_to_numpy(d) for i, d in enumerate(data) if i not in ignore_idx]
        if isinstance(data, tvm_tirx.runtime.Tensor):
            return data.numpy()
        return data

    def _gen_weights(out_ty):
        if isinstance(out_ty, TupleType):
            return [_gen_weights(ty) for ty in out_ty.fields]
        else:
            assert isinstance(out_ty, TensorType)
            return np.random.uniform(size=[int(i) for i in out_ty.shape]).astype(out_ty.dtype)

    def _is_call_no_grad(expr):
        return isinstance(expr, Call) and expr.op == Op.get("relax.grad.no_grad")

    # Generate parameter relax Vars
    param_vars = [
        relax.Var("x_" + str(i), _numpy_to_ty(data)) for i, data in enumerate(inputs_numpy)
    ]

    # Generate the forward call
    if tuple_input:
        t = relax.Tuple(param_vars)
        call = op_func(t, **kwargs)
    else:
        call = op_func(*param_vars, **kwargs)

    # Forward mod
    forward_bb = relax.BlockBuilder()
    with forward_bb.function(func_name, param_vars):
        with forward_bb.dataflow():
            out = forward_bb.emit_output(call)
        forward_bb.emit_func_output(out)
    forward_mod = forward_bb.get()
    forward_ex = tvm.compile(forward_mod, target)
    forward_vm = relax.VirtualMachine(forward_ex, dev)

    # Generate weights
    # In forward process, weights represent the weight of every element of the result of the
    # forward call. The weighted result will be sum(weight * result).
    # If the result is a tuple, weights will be a list, and the weighted result will be
    # sum(i * j for i, j in zip(weights, result))
    # In the gradient process, weights is the output gradient, i.e. the gradient w.r.t. the result.
    out_ty = forward_mod[func_name].body.body.ty
    weights = _gen_weights(out_ty)

    # The inputs of the forward function are inputs_filtered below.
    def forward(*inputs):
        inputs_iter = iter(inputs)
        inputs_tvm = [
            _numpy_to_tvm(next(inputs_iter))
            if i not in ignore_grads
            else _numpy_to_tvm(inputs_numpy[i])
            for i in range(len(inputs_numpy))
        ]
        result = forward_vm[func_name](*inputs_tvm)
        result_numpy = _tvm_to_numpy(result)
        if isinstance(result_numpy, list):
            assert isinstance(weights, list)
            assert len(weights) == len(result_numpy)
            ret = 0
            for i, weight in enumerate(weights):
                ret += np.sum(weight * result_numpy[i], dtype="float64")
            return ret
        # Keep the scalar reduction precise enough for finite differences.
        return np.sum(weights * result_numpy, dtype="float64")

    # The gradient function
    assert isinstance(call.op, Op)
    op_grad_func = call.op.get_attr("FPrimalGradient")

    # The parameter Var for gradient
    grad_var = relax.Var("grad", _numpy_to_ty(weights))

    # Gradient mod
    grad_bb = relax.BlockBuilder()
    with grad_bb.function(func_name, param_vars + [grad_var]):
        with grad_bb.dataflow():
            orig = grad_bb.emit(call)
            # op_grad_func returns a list of Exprs representing the gradients
            grad_call = op_grad_func(orig, call, grad_var, grad_bb)

            # Check ignore_grads
            for i, grad in enumerate(grad_call):
                if i in ignore_grads:
                    assert _is_call_no_grad(grad), f"The {i}-th gradient should be no_grad"
                else:
                    assert not _is_call_no_grad(grad), f"The {i}-th gradient should not be no_grad"

            if tuple_input:
                # If the input is a tuple, the gradient is also a tuple.
                # The gradient tuple is the first (the only) element of grad_call.
                out = grad_bb.emit_output(grad_call[0])
            else:
                # We need to wrap the list into a relax.Tuple so as to emit it
                out = grad_bb.emit_output(relax.Tuple(grad_call))
        grad_bb.emit_func_output(out)

    grad_mod = grad_bb.get()
    grad_ex = tvm.compile(grad_mod, target)
    grad_vm = relax.VirtualMachine(grad_ex, dev)

    # tvm.runtime.Tensor inputs
    inputs_tvm = [_numpy_to_tvm(i) for i in inputs_numpy]
    weights_tvm = _numpy_to_tvm(weights)
    result_filtered = _tvm_to_numpy(grad_vm[func_name](*inputs_tvm, weights_tvm), ignore_grads)

    # Inputs contained in ignore_grads are removed
    inputs_filtered = [inputs_numpy[i] for i in range(len(inputs_numpy)) if i not in ignore_grads]

    check_numerical_grads(forward, inputs_filtered, result_filtered)


##################### Unary #####################


@pytest.mark.parametrize(
    "unary_op_func,can_be_neg",
    [
        (relax.op.abs, True),
        (relax.op.cos, True),
        (relax.op.exp, True),
        (relax.op.log, False),
        (relax.op.negative, True),
        (relax.op.sigmoid, True),
        (relax.op.sin, True),
        (relax.op.sqrt, False),
        (relax.op.tanh, True),
    ],
)
@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_unary(unary_op_func, can_be_neg):
    target = "llvm"
    (low, high) = (-1, 1) if can_be_neg else (0.1, 1)
    data_numpy = np.random.uniform(low, high, (3, 3)).astype(np.float32)
    relax_check_gradients(unary_op_func, [data_numpy], target)


##################### Binary #####################


@pytest.mark.parametrize(
    "binary_arith_op_func",
    [
        relax.op.add,
        relax.op.subtract,
        relax.op.multiply,
        relax.op.divide,
        relax.op.power,
    ],
)
@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_binary_arith(binary_arith_op_func):
    target = "llvm"
    data1_numpy = np.random.uniform(1, 2, (3, 3)).astype(np.float32)
    data2_numpy = np.random.uniform(1, 2, (3, 3)).astype(np.float32)
    relax_check_gradients(binary_arith_op_func, [data1_numpy, data2_numpy], target)


@pytest.mark.parametrize("binary_minmax_op_func", [relax.op.maximum, relax.op.minimum])
@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_binary_minmax(binary_minmax_op_func):
    target = "llvm"
    # Checking numerical gradient of min and max requires data1_numpy[i] != data2_numpy[i]
    # for all possible i.
    # If data1_numpy[i] == data2_numpy[i], the operator is not differentiable w.r.t. place i
    data1_numpy = np.random.uniform(1, 1.1, (3, 3)).astype(np.float32)
    delta = np.random.uniform(1, 1.1, (3, 3)).astype(np.float32)
    sign = np.random.randint(0, 2, (3, 3)).astype(np.float32) * 2 - 1
    data2_numpy = data1_numpy + delta * sign
    relax_check_gradients(binary_minmax_op_func, [data1_numpy, data2_numpy], target)


@pytest.mark.parametrize(
    "binary_cmp_op_func",
    [
        relax.op.equal,
        relax.op.greater,
        relax.op.greater_equal,
        relax.op.less,
        relax.op.less_equal,
        relax.op.not_equal,
    ],
)
@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_binary_cmp(binary_cmp_op_func):
    target = "llvm"
    data1_numpy = np.random.uniform(1, 2, (3, 3)).astype(np.float32)
    data2_numpy = np.random.uniform(1, 2, (3, 3)).astype(np.float32)
    relax_check_gradients(
        binary_cmp_op_func, [data1_numpy, data2_numpy], target, ignore_grads=[0, 1]
    )


##################### Create #####################


@pytest.mark.parametrize("like_op_func", [relax.op.zeros_like, relax.op.ones_like])
@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_ones_zeros_like(like_op_func):
    target = "llvm"
    data_numpy = np.random.uniform(-1, 1, (3, 3)).astype(np.float32)
    relax_check_gradients(like_op_func, [data_numpy], target, ignore_grads=[0])


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_full_like():
    target = "llvm"
    data_numpy = np.random.uniform(-1, 1, (3, 3)).astype(np.float32)
    fill_value = np.random.uniform(-1, 1, ()).astype(np.float32)
    relax_check_gradients(relax.op.full_like, [data_numpy, fill_value], target, ignore_grads=[0, 1])


@pytest.mark.parametrize("create_op_func", [relax.op.zeros, relax.op.ones])
@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_ones_zeros(create_op_func):
    target = "llvm"
    relax_check_gradients(
        create_op_func, [], target, ignore_grads=[0], shape=(3, 3), dtype="float32"
    )


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_triu():
    target = "llvm"
    data_numpy = np.random.uniform(-1, 1, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.triu, [data_numpy], target, k=0)


##################### Statistical #####################


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_sum():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.sum, [data1_numpy], target)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_sum_with_axis():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (2, 3, 4, 5)).astype(np.float32)
    relax_check_gradients(relax.op.sum, [data1_numpy], target, axis=[1, 3])


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_sum_keepdims():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.sum, [data1_numpy], target, keepdims=True, axis=1)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_mean():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.mean, [data1_numpy], target)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_mean_with_axis():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (2, 3, 4, 5)).astype(np.float32)
    relax_check_gradients(relax.op.mean, [data1_numpy], target, axis=[1, 3])


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_mean_keepdims():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.mean, [data1_numpy], target, keepdims=True, axis=1)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_variance():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.variance, [data1_numpy], target)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_variance_with_axis():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (2, 3, 4, 5)).astype(np.float32)
    relax_check_gradients(relax.op.variance, [data1_numpy], target, axis=[1, 3])


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_variance_keepdims():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.variance, [data1_numpy], target, keepdims=True, axis=1)


##################### Manipulate #####################


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_reshape():
    target = "llvm"
    data_numpy = np.random.uniform(0, 16, (2, 3, 5)).astype(np.float32)
    relax_check_gradients(relax.op.reshape, [data_numpy], target, ignore_grads=[1], shape=(5, 6))


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_reshape_infer_dim():
    target = "llvm"
    data_numpy = np.random.uniform(0, 16, (2, 3, 5)).astype(np.float32)
    relax_check_gradients(
        relax.op.reshape, [data_numpy], target, ignore_grads=[1], shape=(5, 2, 1, -1)
    )


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_permute_dims():
    target = "llvm"
    data_numpy = np.random.uniform(0, 16, (2, 3, 4, 5)).astype(np.float32)
    relax_check_gradients(relax.op.permute_dims, [data_numpy], target)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_permute_dims_with_axes():
    target = "llvm"
    data_numpy = np.random.uniform(0, 16, (2, 3, 4, 5)).astype(np.float32)
    relax_check_gradients(
        relax.op.permute_dims,
        [data_numpy],
        target,
        axes=(0, 3, 1, 2),
    )


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_concat():
    target = "llvm"
    data_numpy1 = np.random.uniform(1, 16, (3, 3)).astype(np.float32)
    data_numpy2 = np.random.uniform(1, 16, (3, 4)).astype(np.float32)
    data_numpy3 = np.random.uniform(1, 16, (3, 5)).astype(np.float32)
    relax_check_gradients(
        relax.op.concat,
        [data_numpy1, data_numpy2, data_numpy3],
        target,
        tuple_input=True,
        axis=1,
    )


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_split_indices():
    target = "llvm"
    data_numpy = np.random.uniform(1, 16, (3, 12)).astype(np.float32)
    relax_check_gradients(
        relax.op.split,
        [data_numpy],
        target,
        indices_or_sections=[3, 7],
        axis=1,
    )


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_split_section():
    target = "llvm"
    data_numpy = np.random.uniform(1, 16, (3, 12)).astype(np.float32)
    relax_check_gradients(
        relax.op.split,
        [data_numpy],
        target,
        indices_or_sections=3,
        axis=1,
    )


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_reshape():
    target = "llvm"
    data_numpy = np.random.uniform(1, 16, (3, 4)).astype(np.float32)

    relax_check_gradients(
        relax.op.reshape,
        [data_numpy],
        target,
        shape=(3, 2, 2),
        ignore_grads=[1],
    )


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_cumsum():
    target = "llvm"
    data_numpy1 = np.random.uniform(1, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(
        relax.op.cumsum,
        [data_numpy1],
        target,
        axis=1,
    )


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_cumsum_no_axis():
    target = "llvm"
    data_numpy1 = np.random.uniform(1, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(
        relax.op.cumsum,
        [data_numpy1],
        target,
    )


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_expand_dims():
    target = "llvm"
    data_numpy = np.random.uniform(1, 16, (3, 12)).astype(np.float32)
    relax_check_gradients(relax.op.expand_dims, [data_numpy], target, axis=1)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_expand_dims_list():
    target = "llvm"
    data_numpy = np.random.uniform(1, 16, (3, 12)).astype(np.float32)
    relax_check_gradients(relax.op.expand_dims, [data_numpy], target, axis=(0, 2, 3))


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_broadcast_to():
    target = "llvm"
    data_numpy = np.random.uniform(1, 16, (3, 4)).astype(np.float32)
    relax_check_gradients(
        relax.op.broadcast_to,
        [data_numpy],
        target,
        shape=(2, 3, 4),
        ignore_grads=[1],
    )


##################### Index #####################


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_take():
    target = "llvm"
    data_numpy = np.random.uniform(0, 16, size=(2, 3, 4)).astype(np.float32)
    indices = np.array([0, 1])
    relax_check_gradients(
        relax.op.take,
        [data_numpy, indices],
        target,
        axis=1,
        ignore_grads=[1],
    )


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_take_no_axis():
    target = "llvm"
    data_numpy = np.random.uniform(0, 16, size=(5,)).astype(np.float32)
    indices = np.array([1, 3])
    relax_check_gradients(
        relax.op.take,
        [data_numpy, indices],
        target,
        ignore_grads=[1],
    )


##################### Search #####################


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_where():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 1, size=(3, 3)) > 0.5
    data2_numpy = np.random.uniform(0, 16, size=(3, 3)).astype(np.float32)
    data3_numpy = np.random.uniform(0, 16, size=(3, 3)).astype(np.float32)

    relax_check_gradients(
        relax.op.where,
        [data1_numpy, data2_numpy, data3_numpy],
        target,
        ignore_grads=[0],
    )


##################### Linear Algebra #####################


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_matmul_2_2():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (2, 3)).astype(np.float32)
    data2_numpy = np.random.uniform(0, 16, (3, 4)).astype(np.float32)
    relax_check_gradients(relax.op.matmul, [data1_numpy, data2_numpy], target)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_matmul_1_1():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (4,)).astype(np.float32)
    data2_numpy = np.random.uniform(0, 16, (4,)).astype(np.float32)
    relax_check_gradients(relax.op.matmul, [data1_numpy, data2_numpy], target)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_matmul_1_4():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (4,)).astype(np.float32)
    data2_numpy = np.random.uniform(0, 16, (2, 3, 4, 5)).astype(np.float32)
    relax_check_gradients(relax.op.matmul, [data1_numpy, data2_numpy], target)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_matmul_4_1():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (2, 3, 4, 5)).astype(np.float32)
    data2_numpy = np.random.uniform(0, 16, (5,)).astype(np.float32)
    relax_check_gradients(relax.op.matmul, [data1_numpy, data2_numpy], target)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_matmul_5_4():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (2, 3, 1, 4, 5)).astype(np.float32)
    data2_numpy = np.random.uniform(0, 16, (3, 2, 5, 4)).astype(np.float32)
    relax_check_gradients(
        relax.op.matmul,
        [data1_numpy, data2_numpy],
        target,
    )


##################### Datatype #####################


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_astype():
    target = "llvm"
    data_numpy = np.random.uniform(0, 16, size=(3, 3)).astype(np.float64)
    relax_check_gradients(relax.op.astype, [data_numpy], target, dtype="float32")


##################### Neural network #####################


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_relu():
    target = "llvm"
    data1_numpy = np.random.uniform(0.2, 1, (3, 3)).astype(np.float32)
    sign = np.random.randint(0, 2, (3, 3)).astype(np.float32) * 2 - 1
    data1_numpy *= sign
    relax_check_gradients(relax.op.nn.relu, [data1_numpy], target)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_silu():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.nn.silu, [data1_numpy], target)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_softmax():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.nn.softmax, [data1_numpy], target)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_softmax_with_axis():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.nn.softmax, [data1_numpy], target, axis=1)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_log_softmax():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.nn.log_softmax, [data1_numpy], target)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_log_softmax_with_axis():
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.nn.log_softmax, [data1_numpy], target, axis=1)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_cross_entropy_with_logits():
    target = "llvm"
    data_numpy1 = np.random.uniform(1, 16, (3,)).astype(np.float32)
    data_numpy2 = np.random.uniform(1, 16, (3,)).astype(np.float32)
    relax_check_gradients(
        relax.op.nn.cross_entropy_with_logits,
        [data_numpy1, data_numpy2],
        target,
    )


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_cross_entropy_with_logits_batch():
    target = "llvm"
    data_numpy1 = np.random.uniform(1, 16, (2, 3)).astype(np.float32)
    data_numpy2 = np.random.uniform(1, 16, (2, 3)).astype(np.float32)
    relax_check_gradients(
        relax.op.nn.cross_entropy_with_logits,
        [data_numpy1, data_numpy2],
        target,
    )


@pytest.mark.parametrize(
    "nll_reduction,nll_weighted,nll_ignore_index",
    [
        ("mean", True, -1),
        ("sum", True, -1),
        ("none", True, -1),
        ("mean", True, 1),
        ("mean", True, 1),
        ("mean", False, 1),
    ],
)
@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_nll_loss(nll_reduction, nll_weighted, nll_ignore_index):
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (2, 3, 4)).astype(np.float32)
    data2_numpy = np.random.randint(0, 3, (2, 4)).astype(np.int64)
    # force a position in targets it not ignore_index, to avoid zero total weight
    data2_numpy[0][0] = 0
    # weight > 0
    data3_numpy = np.random.uniform(1, 16, (3,)).astype(np.float32)

    input = [data1_numpy, data2_numpy] + ([data3_numpy] if nll_weighted else [])
    ignore_grads = [1] + ([2] if nll_weighted else [])

    relax_check_gradients(
        relax.op.nn.nll_loss,
        input,
        target,
        ignore_grads=ignore_grads,
        reduction=nll_reduction,
        ignore_index=nll_ignore_index,
    )


@pytest.mark.parametrize(
    "nll_reduction1,nll_weighted1,nll_ignore_index1",
    [
        ("mean", True, -1),
        ("sum", True, -1),
        ("none", True, -1),
    ],
)
@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_nll_loss_no_batch(nll_reduction1, nll_weighted1, nll_ignore_index1):
    target = "llvm"
    data1_numpy = np.random.uniform(0, 16, (3,)).astype(np.float32)
    data2_numpy = np.random.randint(0, 3, ()).astype(np.int64)
    # weight > 0
    data3_numpy = np.random.uniform(1, 16, (3,)).astype(np.float32)

    input = [data1_numpy, data2_numpy] + ([data3_numpy] if nll_weighted1 else [])
    ignore_grads = [1] + ([2] if nll_weighted1 else [])

    relax_check_gradients(
        relax.op.nn.nll_loss,
        input,
        target,
        ignore_grads=ignore_grads,
        reduction=nll_reduction1,
        ignore_index=nll_ignore_index1,
    )


@pytest.mark.parametrize(
    "c2d_shape1,c2d_shape2,c2d_kwargs",
    [
        (
            (3, 2, 10, 10),
            (3, 2, 3, 3),
            {},
        ),
        (
            (3, 2, 10, 10),
            (3, 2, 1, 2),
            {},
        ),
        (
            (3, 2, 10, 10),
            (3, 2, 3, 3),
            {"strides": (2, 2), "padding": (3, 2), "dilation": (1, 1)},
        ),
        (
            (3, 2, 10, 10),
            (3, 2, 3, 3),
            {"strides": (2, 1), "padding": (2, 2), "dilation": (1, 1)},
        ),
        (
            (3, 6, 10, 10),
            (4, 3, 3, 3),
            {"groups": 2},
        ),
        (
            (3, 2, 10, 10),
            (4, 1, 3, 3),
            {"groups": 2, "strides": (2, 2), "padding": (2, 2), "dilation": (1, 1)},
        ),
    ],
)
@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_conv2d(c2d_shape1, c2d_shape2, c2d_kwargs):
    target = "llvm"
    import pytest

    # Use smaller range to reduce numerical errors in gradient check
    data1_numpy = np.random.uniform(0, 2, c2d_shape1).astype(np.float32)
    data2_numpy = np.random.uniform(0, 2, c2d_shape2).astype(np.float32)
    relax_check_gradients(
        relax.op.nn.conv2d,
        [data1_numpy, data2_numpy],
        target,
        **c2d_kwargs,
    )


pool_params = [
    (
        (3, 3),
        {},
    ),
    (
        (3, 3),
        {"strides": (2, 2), "padding": (1, 2), "dilation": (1, 1), "count_include_pad": True},
    ),
    (
        (5, 5),
        {
            "strides": (2, 2),
            "padding": (2, 1),
            "dilation": (1, 1),
            "ceil_mode": True,
            "count_include_pad": True,
        },
    ),
]


@pytest.mark.parametrize("pool_size,pool_kwargs", pool_params)
@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_max_pool2d(pool_size, pool_kwargs):
    target = "llvm"
    data_numpy = np.random.uniform(0, 3, size=(3, 2, 10, 10)).astype(np.float32)
    relax_check_gradients(
        relax.op.nn.max_pool2d,
        [data_numpy],
        target,
        pool_size=pool_size,
        **pool_kwargs,
    )


@pytest.mark.parametrize("pool_size,pool_kwargs", pool_params)
@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_avg_pool2d(pool_size, pool_kwargs):
    target = "llvm"
    data_numpy = np.random.uniform(0, 3, size=(3, 2, 10, 10)).astype(np.float32)
    relax_check_gradients(
        relax.op.nn.avg_pool2d,
        [data_numpy],
        target,
        pool_size=pool_size,
        **pool_kwargs,
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
