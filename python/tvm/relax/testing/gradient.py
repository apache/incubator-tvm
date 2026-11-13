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
# ruff: noqa: E501

# pylint: disable=invalid-name,unnecessary-comprehension,redefined-outer-name
"""Numerical checks for Relax gradients."""

import logging

import numpy as np


def check_numerical_grads(
    function, input_values, grad_values, function_value=None, delta=1e-3, atol=1e-2, rtol=0.1
):
    """A helper function that checks that numerical gradients of a function are
    equal to gradients computed in some different way (analytical gradients).

    Numerical gradients are computed using finite difference approximation. To
    reduce the number of function evaluations, the number of points used is
    gradually increased if the error value is too high (up to 5 points).

    Parameters
    ----------
    function
        A function that takes inputs either as positional or as keyword
        arguments (either `function(*input_values)` or `function(**input_values)`
        should be correct) and returns a scalar result. Should accept numpy
        ndarrays.

    input_values : Dict[str, numpy.ndarray] or List[numpy.ndarray]
        A list of values or a dict assigning values to variables. Represents the
        point at which gradients should be computed.

    grad_values : Dict[str, numpy.ndarray] or List[numpy.ndarray]
        Gradients computed using a different method.

    function_value : float, optional
        Should be equal to `function(**input_values)`.

    delta : float, optional
        A small number used for numerical computation of partial derivatives.
        The default 1e-3 is a good choice for float32.

    atol : float, optional
        Absolute tolerance. Gets multiplied by `sqrt(n)` where n is the size of a
        gradient.

    rtol : float, optional
        Relative tolerance.
    """
    # If input_values is a list then function accepts positional arguments
    # In this case transform it to a function taking kwargs of the form {"0": ..., "1": ...}
    if not isinstance(input_values, dict):
        input_len = len(input_values)
        input_values = {str(idx): val for idx, val in enumerate(input_values)}

        def _function(_input_len=input_len, _orig_function=function, **kwargs):
            return _orig_function(*(kwargs[str(i)] for i in range(input_len)))

        function = _function

        grad_values = {str(idx): val for idx, val in enumerate(grad_values)}

    if function_value is None:
        function_value = function(**input_values)

    # a helper to modify j-th element of val by a_delta
    def modify(val, j, a_delta):
        val = val.copy()
        val.reshape(-1)[j] = val.reshape(-1)[j] + a_delta
        return val

    # numerically compute a partial derivative with respect to j-th element of the var `name`
    def derivative(x_name, j, a_delta):
        modified_values = {
            n: modify(val, j, a_delta) if n == x_name else val for n, val in input_values.items()
        }
        return (function(**modified_values) - function_value) / a_delta

    def compare_derivative(j, n_der, grad):
        der = grad.reshape(-1)[j]
        return np.abs(n_der - der) < atol + rtol * np.abs(n_der)

    for x_name, grad in grad_values.items():
        if grad.shape != input_values[x_name].shape:
            raise AssertionError(
                f"Gradient wrt '{x_name}' has unexpected shape {grad.shape}, expected {input_values[x_name].shape} "
            )

        ngrad = np.zeros_like(grad)

        wrong_positions = []

        # compute partial derivatives for each position in this variable
        for j in range(np.prod(grad.shape)):
            # forward difference approximation
            nder = derivative(x_name, j, delta)

            # if the derivative is not equal to the analytical one, try to use more
            # precise and expensive methods
            if not compare_derivative(j, nder, grad):
                # central difference approximation
                nder = (derivative(x_name, j, -delta) + nder) / 2

                if not compare_derivative(j, nder, grad):
                    # central difference approximation using h = delta/2
                    cnder2 = (
                        derivative(x_name, j, delta / 2) + derivative(x_name, j, -delta / 2)
                    ) / 2
                    # five-point derivative
                    nder = (4 * cnder2 - nder) / 3

            # if the derivatives still don't match, add this position to the
            # list of wrong positions
            if not compare_derivative(j, nder, grad):
                wrong_positions.append(np.unravel_index(j, grad.shape))

            ngrad.reshape(-1)[j] = nder

        wrong_percentage = int(100 * len(wrong_positions) / np.prod(grad.shape))

        dist = np.sqrt(np.sum((ngrad - grad) ** 2))
        grad_norm = np.sqrt(np.sum(ngrad**2))

        if not (np.isfinite(dist) and np.isfinite(grad_norm)):
            raise ValueError(
                f"NaN or infinity detected during numerical gradient checking wrt '{x_name}'\n"
                f"analytical grad = {grad}\n numerical grad = {ngrad}\n"
            )

        # we multiply atol by this number to make it more universal for different sizes
        sqrt_n = np.sqrt(float(np.prod(grad.shape)))

        if dist > atol * sqrt_n + rtol * grad_norm:
            raise AssertionError(
                f"Analytical and numerical grads wrt '{x_name}' differ too much\n"
                f"analytical grad = {grad}\n numerical grad = {ngrad}\n"
                f"{wrong_percentage}% of elements differ, first 10 of wrong positions: {wrong_positions[:10]}\n"
                "distance > atol*sqrt(n) + rtol*grad_norm\n"
                f"distance {dist} > {atol}*{sqrt_n} + {rtol}*{grad_norm}"
            )

        max_diff = np.max(np.abs(ngrad - grad))
        avg_diff = np.mean(np.abs(ngrad - grad))
        logging.info(
            "Numerical grad test wrt '%s' of shape %s passes, "
            "dist = %f, max_diff = %f, avg_diff = %f",
            x_name,
            grad.shape,
            dist,
            max_diff,
            avg_diff,
        )
