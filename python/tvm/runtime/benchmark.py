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
"""Function benchmarking for local and RPC runtime modules."""

import struct
from collections.abc import Sequence

import numpy as np
import tvm_ffi


class BenchmarkResult:
    """Runtimes from benchmarking"""

    def __init__(self, results: Sequence[float]):
        """Construct a new BenchmarkResult from a sequence of runtimes.

        Parameters
        ----------
        results : Sequence[float]
            Raw times from benchmarking

        Attributes
        ----------
        min : float
            Minimum runtime in seconds of all results.
        mean : float
            Mean runtime in seconds of all results. If :func:`time_evaluator` or
            `benchmark` is called with `number` > 0, then each result is already the mean of a
            `number` of runtimes, so this becomes the mean of means.
        median : float
            Median runtime in seconds of all results. If :func:`time_evaluator` is called
            with `number` > 0, then each result is already the mean of a `number` of runtimes, so
            this becomes the median of means.
        max : float
            Maximum runtime in seconds of all results. If :func:`time_evaluator` is called
            with `number` > 0, then each result is already the mean of a `number` of runtimes, so
            this becomes the maximum of those means.
        std : float
            Standard deviation in seconds of runtimes. If :func:`time_evaluator` is called
            with `number` > 0, then each result is already the mean of a `number` of runtimes, so
            this becomes the standard deviation of means.
        results : Sequence[float]
            The collected runtimes (in seconds). This may be a series of mean runtimes if
            :func:`time_evaluator` or `benchmark` was run with `number` > 1.
        """
        self.results = results
        self.mean = np.mean(self.results)
        self.std = np.std(self.results)
        self.median = np.median(self.results)
        self.min = np.min(self.results)
        self.max = np.max(self.results)

    def __repr__(self):
        return (
            f"BenchmarkResult(min={self.min}, mean={self.mean}, median={self.median}, "
            f"max={self.max}, std={self.std}, results={self.results})"
        )

    def __str__(self):
        return (
            f"Execution time summary:\n"
            f"{'mean (ms)':^12} {'median (ms)':^12} {'max (ms)':^12} "
            f"{'min (ms)':^12} {'std (ms)':^12}\n"
            f"{self.mean * 1000:^12.4f} {self.median * 1000:^12.4f} {self.max * 1000:^12.4f} "
            f"{self.min * 1000:^12.4f} {self.std * 1000:^12.4f}"
            "               "
        )


def time_evaluator(
    module,
    func_name,
    device,
    number=10,
    repeat=1,
    min_repeat_ms=0,
    limit_zero_time_iterations=100,
    cooldown_interval_ms=0,
    repeats_to_cooldown=1,
    cache_flush_bytes=0,
    f_preproc="",
):
    """Get an evaluator that measures time cost of running function.

    Parameters
    ----------
    module: tvm_ffi.Module
        The module containing the function to benchmark.

    func_name: str
        The name of the function in the module.

    device: Device or tvm.rpc.RemoteDevice
        The device we should run this function on.

    number: int
        The number of times to run this function for taking average.
        We call these runs as one `repeat` of measurement.

    repeat: int, optional
        The number of times to repeat the measurement.
        In total, the function will be invoked (1 + number x repeat) times,
        where the first one is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.

    min_repeat_ms: int, optional
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.

    limit_zero_time_iterations: int, optional
        The maximum number of repeats when measured time is equal to 0.
        It helps to avoid hanging during measurements.

    cooldown_interval_ms: int, optional
        The cooldown interval in milliseconds between the number of repeats defined by
        `repeats_to_cooldown`.

    repeats_to_cooldown: int, optional
        The number of repeats before the cooldown is activated.

    cache_flush_bytes: int, optional
        The number of bytes to flush from the cache before each repeat.

    f_preproc: str, optional
        The preprocess function name we want to execute before executing the time evaluator.

    Note
    ----
    The function will be invoked  (1 + number x repeat) times,
    with the first call discarded in case there is lazy initialization.

    Returns
    -------
    ftimer : function
        The function that takes same argument as func and returns a BenchmarkResult.
        The ProfileResult reports `repeat` time costs in seconds.
    """
    feval = tvm_ffi.get_global_func("runtime.ModuleTimeEvaluator")(
        module,
        func_name,
        device,
        number,
        repeat,
        min_repeat_ms,
        limit_zero_time_iterations,
        cooldown_interval_ms,
        repeats_to_cooldown,
        cache_flush_bytes,
        f_preproc,
    )

    def evaluator(*args):
        """Internal wrapped evaluator."""
        # Wrap feval so we can add more stats in future.
        blob = feval(*args)
        fmt = "@" + ("d" * repeat)
        results = struct.unpack(fmt, blob)
        return BenchmarkResult(results)

    return evaluator
