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
# ruff: noqa: F401
"""Provide abstraction for defining optimizers and a set of common optimizers."""

from decimal import Decimal
from typing import Optional, Union

import numpy as np  # type: ignore
import tvm_ffi

import tvm
import tvm_tirx

from ..block_builder import BlockBuilder
from ..expr import Function, TupleGetItem, Var, const
from ..expr import Tuple as RxTuple
from ..op import add, divide, multiply, sqrt, subtract
from ..type import TensorType, TupleType


# TODO(chaofan, yixin): Migrate key logics to C++
class Optimizer:
    """Relax training optimizer. This class could generate relax Functions for optimizing specified
    parameters, and store the states used in the optimization process, such as momentum.

    Parameters
    ----------
    name : str
        The name of the optimizer function. This parameter is provided by subclasses.

    Attributes
    ----------
    dtype : str
        The only dtype of the optimizer. It will be used as the dtype of the optimizer states,
        and the dtype of necessary constants, such as the learning rate. Will be set in `init()`.

    name : str
        The name of the optimizer.

    param_list : List[Var]
        The list of variables to optimize. Will be set in `init()`.

    state : tvm_ffi.Array
        `state` is an runtime Array representing the state of the optimizer. Will be set in
        `init()`.

        The states of the optimizer can store necessary information in the optimization process at
        runtime, such as the number of steps, the momentum in momentum SGD, etc.

        `opt.state` should be used as the last argument of the function that is got through
        `get_function()`, and its new value is returned as the last return value of that function.

        See examples for more details.

    Examples
    --------
    The usage of optimizers should resemble the following pattern. We will take SGD as an example.
    For detailed examples, please see the tutorial.

    .. code-block:: python

        # Construct the optimizer
        opt = relax.optimizer.SGD(0.1)

        # Initialize the parameter list, the dtype and the optimizer state
        # x is the relax Var we want to optimize
        opt.init(x)

        # The above two lines is equivalent to one line:
        opt = relax.optimizer.SGD(0.1).init(x)

        # Get the optimizer function
        # mod is an IRModule constructed earlier
        mod["SGD"] = opt.get_function()

        # Legalize and build mod
        lowered_mod = LegalizeOps()(mod)
        ex = build(lowered_mod, target="llvm")
        vm = VirtualMachine(ex, tvm_tirx.cpu())

        # Optimization process
        # param_tuple is a runtime tuple of parameters
        # param_gradient is a runtime tuple of the gradient of the parameters in param_tuple,
        # respectively
        # param_gradient can be gained by the automatic differentiation pass. Please see
        # `relax.transform.Gradient`
        param_tuple, opt.state = vm["SGD"](param_tuple, param_gradient, opt.state)
    """

    dtype: str
    name: str
    param_list: list[Var]
    state: tvm_ffi.Array

    def __init__(self, name: str) -> None:
        self.name = name
        self.param_list = None
        self.state = None
        self.dtype = None

    def init(self, params: Var | list[Var]) -> "Optimizer":
        """Set the parameters, determine the dtype, and construct the initial state for the
        optimizer.

        Parameters
        ----------
        params : Union[Var, List[Var]]
            The parameter or the list of parameters to optimize.

            Parameters should all be Vars of floating point Tensors, including float32, float64,
            float16, etc. Currently, all parameters should have the same dtype, and that dtype
            will be used as the dtype of the optimizer states.

        Returns
        -------
        self : Optimizer
            The optimizer itself.
        """
        if not isinstance(params, list):
            params = [params]
        self._set_params_and_dtype(params)
        # State should be initialized in any implementation of optimizer.
        self.state = None
        return self

    def _set_params_and_dtype(self, params: list[Var]) -> None:
        """Check params is legal and set the param_list and dtype of the optimizer."""
        params_set = set()
        dtype = None
        for x in params:
            if not isinstance(x, Var):
                raise ValueError(f"Parameter {x} is not a Var")
            if not isinstance(x.ty, TensorType):
                raise ValueError(
                    f"Optimizers only support Tensor parameters, but parameter {x.name} has "
                    f"type {x.ty}"
                )
            data_type = tvm_ffi.dtype(x.ty.dtype.dtype)
            if data_type.type_code not in (
                tvm_tirx.DataTypeCode.BFLOAT,
                tvm_tirx.DataTypeCode.FLOAT,
            ):
                raise ValueError(
                    f"Optimizers only support Tensor parameters of floating point dtype, but dtype "
                    f"of {x.name} is {x.ty.dtype}"
                )
            if dtype is None:
                dtype = x.ty.dtype
            else:
                if dtype != x.ty.dtype:
                    raise ValueError(
                        f"All parameters should have the same dtype, but parameter {x.name} "
                        f"has dtype {x.ty.dtype}, which differs from the previous dtype "
                        f"{dtype}"
                    )
            if x in params_set:
                raise ValueError(f"Parameter {x.name} appears more than once")
            params_set.add(x)
        self.param_list = params
        self.dtype = dtype

    def _check_init(self):
        """Check that the optimizer is initialized. This method should be called at the start of
        get_function().
        """
        if self.param_list is None or self.state is None or self.dtype is None:
            raise RuntimeError("Please call init() for the optimizer before calling get_function()")

    def get_function(self) -> Function:
        """Use blockbuilder to construct an optimizer function that executes updates of the
        parameters and the optimizer state.

        The optimizer function will take in a tuple of parameters, a tuple of gradients of
        parameters, and a tuple of optimizer states. It will return a tuple of updated parameters,
        and a tuple of optimizer states.

        Returns
        -------
        func : Function
            The optimizer function.

        Examples
        --------
        An example of the returned optimizer function. This function executes the stochastic
        gradient descent method with lr = 0.1.

        .. code-block:: python

            @R.function
            def SGD(
                params: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
                gradients: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
                optim_states: R.Tuple(R.Tensor((), "int64")),
            ) -> R.Tuple(
                R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
                R.Tuple(R.Tensor((), "int64")),
            ):
                with R.dataflow():
                    num_steps: R.Tensor((), "int64") = optim_states[0]
                    num_steps_new: R.Tensor((), "int64") = R.add(num_steps, R.const(1, "int64"))
                    x: R.Tensor((3, 3), "float32") = params[0]
                    x_grad: R.Tensor((3, 3), "float32") = gradients[0]
                    lv: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.01, "float32"), x_grad)
                    x_new: R.Tensor((3, 3), "float32") = R.subtract(x, lv)
                    y: R.Tensor((3,), "float32") = params[1]
                    y_grad: R.Tensor((3,), "float32") = gradients[1]
                    lv1: R.Tensor((3,), "float32") = R.multiply(R.const(0.01, "float32"), y_grad)
                    y_new: R.Tensor((3,), "float32") = R.subtract(y, lv1)
                    params_new: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")) = (
                        x_new,
                        y_new,
                    )
                    optim_states_new: R.Tuple(R.Tensor((), "int64")) = (num_steps_new,)
                    R.output(params_new, optim_states_new)
                return (params_new, optim_states_new)
        """
        self._check_init()
        raise NotImplementedError()


# TODO(chaofan, yixin): Support symbolic shapes
def _get_shape_as_int_list(var: Var) -> list[int]:
    return [int(val) for val in var.ty.shape]


# We need to subtract on hyperparameters, but do not want to introduce floating point error.
# Floating point error would lead to a few problems, such as making assert_structural_equal not
# pass in unit tests
def _high_precision_subtract(lhs: float, rhs: float) -> float:
    return float(Decimal(str(lhs)) - Decimal(str(rhs)))


class SGD(Optimizer):
    """Implements stochastic gradient descent.

    The returned function of `get_function()` is equivalent to the following numpy code:

    .. code-block:: python

        def SGD(param_tuple, grad_tuple, state_tuple):
            num_steps = state_tuple[0]
            param_tuple_new, state_tuple_new = [], []
            state_tuple_new.append(num_steps + 1)
            for i in range(len(param_tuple)):
                param = param_tuple[i]
                grad = grad_tuple[i]
                param_tuple_new.append(param - lr * (grad + weight_decay * param))
            return param_tuple_new, state_tuple_new

    Parameters
    ----------
    lr : float
        learning rate

    weight_decay : float
        weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, lr: float, weight_decay: float = 0) -> None:
        super().__init__("SGD")
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    def init(self, params: Var | list[Var]) -> "SGD":
        """Set the parameters, determine the dtype, and construct the initial state for the
        optimizer.

        The state of SGD is `(num_steps,)`.

        Parameters
        ----------
        params : Union[Var, List[Var]]
            The parameter or the list of parameters to optimize.

            Parameters should all be Vars of floating point Tensors, including float32, float64,
            float16, etc. Currently, all parameters should have the same dtype, and that dtype
            will be used as the dtype of the optimizer states.

        Returns
        -------
        self : SGD
            The SGD optimizer itself.
        """
        if not isinstance(params, list):
            params = [params]
        self._set_params_and_dtype(params)
        self.state = (
            # num_steps = 0
            tvm_tirx.runtime.tensor(np.zeros((), "int64")),
        )
        return self

    def get_function(self) -> Function:
        """Use blockbuilder to construct an optimizer function that executes updates of the
        parameters and the optimizer state. `init()` should be called before `get_function()`.

        Returns
        -------
        func : Function
            The optimizer function.
        """
        self._check_init()

        plist = self.param_list
        len_param = len(plist)
        dtype = self.dtype

        # input variables
        param_var = Var("params", TupleType([p.ty for p in plist]))
        grad_var = Var("gradients", TupleType([p.ty for p in plist]))
        state_var = Var("optim_states", TupleType([TensorType((), "int64")]))

        # constants
        lr = const(self.lr, dtype)
        weight_decay = const(self.weight_decay, dtype)
        one = const(1, "int64")

        builder = BlockBuilder()
        with builder.function(self.name, [param_var, grad_var, state_var]):
            with builder.dataflow():
                param_list_new, state_list_new = [], []

                # handle num_steps
                num_steps = builder.emit(TupleGetItem(state_var, 0), "num_steps")
                num_steps_new = builder.emit(add(num_steps, one), "num_steps_new")
                state_list_new.append(num_steps_new)

                # computation logics
                for i in range(len_param):
                    name = self.param_list[i].name
                    p = builder.emit(TupleGetItem(param_var, i), name)
                    g = builder.emit(TupleGetItem(grad_var, i), name + "_grad")
                    if self.weight_decay:
                        g = builder.emit(add(multiply(weight_decay, p), g), name + "_grad_new")
                    p_new = builder.emit(subtract(p, multiply(lr, g)), name + "_new")
                    param_list_new.append(p_new)

                # handle return values
                params_new = builder.emit_output(RxTuple(param_list_new), "params_new")
                optim_states_new = builder.emit_output(RxTuple(state_list_new), "optim_states_new")
            builder.emit_func_output((params_new, optim_states_new))
        return builder.get()[self.name]


class MomentumSGD(Optimizer):
    """Implements stochastic gradient descent with momentum. Optionally supports Nesterov
    momentum.

    The returned function of `get_function()` is equivalent to the following numpy code:

    .. code-block:: python

        def MomentumSGD(param_tuple, grad_tuple, state_tuple):
            num_steps = state_tuple[0]
            param_tuple_new, state_tuple_new = [], []
            state_tuple_new.append(num_steps + 1)

            for i in range(len(param_tuple)):
                param = param_tuple[i]
                grad = grad_tuple[i]
                velocity = state_tuple[i + 1]
                grad = param * weight_decay + grad
                velocity = momentum * velocity + grad * (1 - dampening)
                if nesterov:
                    param = param - (grad + momentum * velocity) * lr
                else:
                    param = param - velocity * lr
                param_tuple_new.append(param)
                state_tuple_new.append(velocity)

            return param_tuple_new, state_tuple_new

    Parameters
    ----------
    lr : float
        learning rate

    momentum : float
        momentum factor (default: 0)

    weight_decay : float
        weight decay (L2 penalty) (default: 0)

    dampening : float
        dampening for momentum (default: 0)

    nesterov : bool
        enables Nesterov momentum (default: False)
    """

    def __init__(
        self,
        lr: float,
        momentum: float,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ) -> None:
        super().__init__("MomentumSGD")
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.dampening = float(dampening)
        self.nesterov = nesterov

    def init(self, params: Var | list[Var]) -> "MomentumSGD":
        """Set the parameters, determine the dtype, and construct the initial state for the
        optimizer.

        The state of MomentumSGD is
        `(num_steps, velocity_of_param_0, ..., velocity_of_param_n-1)`.

        Parameters
        ----------
        params : Union[Var, List[Var]]
            The parameter or the list of parameters to optimize.

            Parameters should all be Vars of floating point Tensors, including float32, float64,
            float16, etc. Currently, all parameters should have the same dtype, and that dtype
            will be used as the dtype of the optimizer states.

        Returns
        -------
        self : MomentumSGD
            The MomentumSGD optimizer itself.
        """
        if not isinstance(params, list):
            params = [params]
        self._set_params_and_dtype(params)
        self.state = (
            # num_steps = 0
            tvm_tirx.runtime.tensor(np.zeros((), "int64")),
            # v_{param} is initialized to all zeros
            *(
                tvm_tirx.runtime.tensor(np.zeros(_get_shape_as_int_list(p), p.ty.dtype.dtype))
                for p in self.param_list
            ),
        )
        return self

    def get_function(self) -> Function:
        """Use blockbuilder to construct an optimizer function that executes updates of the
        parameters and the optimizer state. `init()` should be called before `get_function()`.

        Returns
        -------
        func : Function
            The optimizer function.
        """
        self._check_init()
        plist = self.param_list
        len_param = len(plist)
        dtype = self.dtype

        # input variables
        param_var = Var("params", TupleType([p.ty for p in plist]))
        grad_var = Var("gradients", TupleType([p.ty for p in plist]))
        state_var = Var(
            "optim_states",
            TupleType([TensorType((), "int64"), *(p.ty for p in plist)]),
        )

        # constants
        lr = const(self.lr, dtype)
        momentum = const(self.momentum, dtype)
        weight_decay = const(self.weight_decay, dtype)
        dampening_inv = const(_high_precision_subtract(1, self.dampening), dtype)
        one = const(1, "int64")

        builder = BlockBuilder()
        with builder.function(self.name, [param_var, grad_var, state_var]):
            with builder.dataflow():
                param_list_new, state_list_new = [], []

                # handle num_steps
                num_steps = builder.emit(TupleGetItem(state_var, 0), "num_steps")
                num_steps_new = builder.emit(add(num_steps, one), "num_steps_new")
                state_list_new.append(num_steps_new)

                # computation logics
                for i in range(len_param):
                    name = self.param_list[i].name
                    p = builder.emit(TupleGetItem(param_var, i), name)
                    g = builder.emit(TupleGetItem(grad_var, i), name + "_grad")
                    v = builder.emit(TupleGetItem(state_var, i + 1), name + "_v")
                    if self.weight_decay:
                        g = builder.emit(add(multiply(weight_decay, p), g), name + "_grad_new")
                    damp_g = multiply(dampening_inv, g) if self.dampening else g
                    v_new = builder.emit(add(multiply(momentum, v), damp_g), name + "_v_new")
                    g_new = (
                        builder.emit(add(g, multiply(momentum, v_new)), name + "_g_nest")
                        if self.nesterov
                        else v_new
                    )
                    p_new = builder.emit(subtract(p, multiply(lr, g_new)), name + "_new")
                    param_list_new.append(p_new)
                    state_list_new.append(v_new)

                # handle return values
                params_new = builder.emit_output(RxTuple(param_list_new), "params_new")
                optim_states_new = builder.emit_output(RxTuple(state_list_new), "optim_states_new")
            builder.emit_func_output((params_new, optim_states_new))
        return builder.get()[self.name]


class Adam(Optimizer):
    """Implements Adam optimization algorithm.

    The returned function of `get_function()` is equivalent to the following numpy code:

    .. code-block:: python

        def Adam(param_tuple, grad_tuple, state_tuple):
            num_steps = state_tuple[0]
            num_steps_new = num_steps + 1

            param_tuple_new = []
            state_tuple_new = [None] * len(state_tuple)
            state_tuple_new[0] = num_steps_new
            state_tuple_new[1] = state_tuple[1] * betas[0]
            state_tuple_new[2] = state_tuple[2] * betas[1]

            for i in range(len(param_tuple)):
                param = param_tuple[i]
                grad = grad_tuple[i]
                m = state_tuple[i + 3]
                v = state_tuple[i + 3 + len(param_tuple)]
                grad = grad + weight_decay * param
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad * grad
                m_hat = m / (1 - betas[0] ** num_steps_new)
                v_hat = v / (1 - betas[1] ** num_steps_new)
                param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
                param_tuple_new.append(param)
                state_tuple_new[i + 3] = m
                state_tuple_new[i + 3 + len(param_tuple)] = v

            return param_tuple_new, state_tuple_new

    Parameters
    ----------
    lr : float
        learning rate

    betas : Tuple[float, float]
        coefficients used for computing running averages of gradient and its square
        (default: (0.9, 0.999))

    eps : float
        term added to the denominator to improve numerical stability (default: 1e-8)

    weight_decay : float
        weight decay (L2 penalty) (default: 0)
    """

    def __init__(
        self,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
    ) -> None:
        super().__init__("Adam")
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

    def init(self, params: Var | list[Var]) -> "Adam":
        """Set the parameters, determine the dtype, and construct the initial state for the
        optimizer.

        The state of Adam is

        .. code-block:: python

            (
                num_steps,
                beta_0_prod, # beta0 ** num_steps
                beta_1_prod, # beta1 ** num_steps
                first_momentum_of_param_0, ..., first_momentum_of_param_n-1,
                second_momentum_of_param_0, ..., second_momentum_of_param_n-1
            )

        Parameters
        ----------
        params : Union[Var, List[Var]]
            The parameter or the list of parameters to optimize.

            Parameters should all be Vars of floating point Tensors, including float32, float64,
            float16, etc. Currently, all parameters should have the same dtype, and that dtype
            will be used as the dtype of the optimizer states.

        Returns
        -------
        self : Adam
            The Adam optimizer itself.
        """
        if not isinstance(params, list):
            params = [params]
        self._set_params_and_dtype(params)
        self.state = (
            # num_steps, beta_0_prod, beta_1_prod
            tvm_tirx.runtime.tensor(np.zeros((), "int64")),
            tvm_tirx.runtime.tensor(np.ones((), self.dtype)),
            tvm_tirx.runtime.tensor(np.ones((), self.dtype)),
            # first_momentum
            *(
                tvm_tirx.runtime.tensor(np.zeros(_get_shape_as_int_list(p), p.ty.dtype.dtype))
                for p in self.param_list
            ),
            # second_momentum
            *(
                tvm_tirx.runtime.tensor(np.zeros(_get_shape_as_int_list(p), p.ty.dtype.dtype))
                for p in self.param_list
            ),
        )
        return self

    def get_function(self) -> Function:
        """Use blockbuilder to construct an optimizer function that executes updates of the
        parameters and the optimizer state. `init()` should be called before `get_function()`.

        Returns
        -------
        func : Function
            The optimizer function.
        """
        self._check_init()
        plist = self.param_list
        len_param = len(plist)
        dtype = self.dtype

        # input variables
        param_var = Var("params", TupleType([p.ty for p in plist]))
        grad_var = Var("gradients", TupleType([p.ty for p in plist]))
        state_var = Var(
            "optim_states",
            TupleType(
                [
                    TensorType((), "int64"),
                    TensorType((), dtype),
                    TensorType((), dtype),
                    *(p.ty for p in plist),
                    *(p.ty for p in plist),
                ]
            ),
        )

        # constants
        lr = const(self.lr, dtype)
        beta1 = const(self.beta1, dtype)
        beta2 = const(self.beta2, dtype)
        beta1_inv = const(_high_precision_subtract(1, self.beta1), dtype)
        beta2_inv = const(_high_precision_subtract(1, self.beta2), dtype)
        eps = const(self.eps, dtype)
        weight_decay = const(self.weight_decay, dtype)
        one_int = const(1, "int64")
        one_float = const(1, dtype)

        builder = BlockBuilder()
        with builder.function(self.name, [param_var, grad_var, state_var]):
            with builder.dataflow():
                param_list_new = []
                state_list_new = [None] * (len_param * 2 + 3)  # type: List[Optional[Var]]

                # handle num_steps
                num_steps = builder.emit(TupleGetItem(state_var, 0), "num_steps")
                num_steps_new = builder.emit(add(num_steps, one_int), "num_steps_new")
                state_list_new[0] = num_steps_new
                beta1_prod = builder.emit(multiply(TupleGetItem(state_var, 1), beta1), "beta1_prod")
                beta2_prod = builder.emit(multiply(TupleGetItem(state_var, 2), beta2), "beta2_prod")
                state_list_new[1] = beta1_prod
                state_list_new[2] = beta2_prod

                # computation logics
                for i in range(len_param):
                    name = self.param_list[i].name
                    p = builder.emit(TupleGetItem(param_var, i), name)
                    g = builder.emit(TupleGetItem(grad_var, i), name + "_grad")
                    m = builder.emit(TupleGetItem(state_var, i + 3), name + "_m")
                    v = builder.emit(TupleGetItem(state_var, i + 3 + len_param), name + "_v")
                    if self.weight_decay:
                        g = builder.emit(add(multiply(weight_decay, p), g), name + "_grad_new")
                    m_new = builder.emit(
                        add(multiply(beta1, m), multiply(beta1_inv, g)), name + "_m_new"
                    )
                    v_new = builder.emit(
                        add(multiply(beta2, v), multiply(beta2_inv, multiply(g, g))),
                        name + "_v_new",
                    )
                    m_hat = builder.emit(
                        divide(m_new, subtract(one_float, state_list_new[1])), name + "_m_hat"
                    )
                    v_hat = builder.emit(
                        divide(v_new, subtract(one_float, state_list_new[2])), name + "_v_hat"
                    )
                    p_new = builder.emit(
                        subtract(p, multiply(lr, divide(m_hat, add(sqrt(v_hat), eps)))),
                        name + "_new",
                    )
                    param_list_new.append(p_new)
                    state_list_new[i + 3] = m_new
                    state_list_new[i + 3 + len_param] = v_new

                # handle return values
                params_new = builder.emit_output(RxTuple(param_list_new), "params_new")
                optim_states_new = builder.emit_output(RxTuple(state_list_new), "optim_states_new")
            builder.emit_func_output((params_new, optim_states_new))
        return builder.get()[self.name]
