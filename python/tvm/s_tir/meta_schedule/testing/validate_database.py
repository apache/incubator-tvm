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
# ruff: noqa: F403
"""JSON Database validation script"""

import argparse
import itertools
import logging
import warnings
from collections.abc import Callable
from statistics import mean
from typing import Any

import numpy as np  # type: ignore
import tvm_ffi
from tvm_ffi import get_global_func, register_global_func

import tvm
import tvm_tirx
from tvm.s_tir import Schedule
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.testing.tune_utils import generate_input_data
from tvm.s_tir.meta_schedule.utils import remove_build_dir
from tvm.s_tir.schedule import Trace
from tvm.s_tir.tensor_intrin import *  # type: ignore # pylint: disable=wildcard-import,unused-wildcard-import
from tvm.s_tir.testing import strtobool
from tvm.support import describe
from tvm_tirx.ir import IRModule
from tvm_tirx.target import Target

DELIMITOR = "\n" + "-" * 30 + "\n"


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="The path to the work directory containing database files.",
    )
    args.add_argument(
        "--target",
        type=Target,
        required=True,
    )
    args.add_argument(
        "--baseline-target",
        type=Target,
        default='{"kind": "llvm", "num-cores": 1}',
        required=False,
        help="The baseline target to compile the original module.",
    )
    args.add_argument(
        "--top-k",
        type=int,
        default=10**9,
        required=False,
        help="The number of top-k tuning records to validate for each unique original workload.",
    )
    args.add_argument(
        "--rpc-host",
        type=str,
    )
    args.add_argument(
        "--rpc-port",
        type=int,
    )
    args.add_argument(
        "--rpc-key",
        type=str,
    )
    args.add_argument(
        "--number",
        type=int,
        default=3,
    )
    args.add_argument(
        "--repeat",
        type=int,
        default=1,
    )
    args.add_argument(
        "--min-repeat-ms",
        type=int,
        default=100,
    )
    args.add_argument(
        "--cpu-flush",
        type=lambda x: bool(strtobool(x)),
        help="example: True / False",
        required=True,
    )
    args.add_argument(
        "--input-generator-func",
        type=str,
        default="tvm.s_tir.meta_schedule.testing.default_input_generator",
    )
    args.add_argument(
        "--check-metric-func",
        type=str,
        default="tvm.s_tir.meta_schedule.testing.default_check_metric",
    )
    parsed = args.parse_args()
    parsed.target = tvm_tirx.target.Target(parsed.target)
    if parsed.rpc_host is not None and parsed.rpc_port is not None and parsed.rpc_key is not None:
        parsed.rpc_config = ms.runner.RPCConfig(
            tracker_host=parsed.rpc_host,
            tracker_port=parsed.rpc_port,
            tracker_key=parsed.rpc_key,
            session_timeout_sec=600,
        )
    else:
        parsed.rpc_config = None
        warnings.warn("RPC config is not provided, will use local runner.")
    if parsed.cpu_flush and parsed.target.kind.name != "llvm":
        warnings.warn("cpu_flush is only supported on llvm target")
    return parsed


# arg parser
ARGS = _parse_args()

# logging
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("tvm.s_tir.meta_schedule").setLevel(logging.DEBUG)
logging.getLogger("tvm.s_tir.meta_schedule.runner").setLevel(logging.WARN)


def get_device_type(target: Target) -> str:
    """Get the device type string from a target.

    Parameters
    ----------
    target : Target
        The target to get the device type from.

    Returns
    -------
    device_type : str
        The device type string.
    """
    if target.kind.name == "llvm":
        return "cpu"
    elif target.kind.name == "cuda":
        return "cuda"
    else:
        raise RuntimeError(f"Unsupported target kind for device type: {target.kind.name}")


def get_runtime_device(target: Target) -> tvm_tirx.runtime.Device:
    """Get the runtime device from a target.

    Parameters
    ----------
    target : Target
        The target to get the runtime device from.

    Returns
    -------
    device : tvm_tirx.runtime.Device
        The runtime device.
    """
    if target.kind.name == "llvm":
        return tvm_tirx.cpu()
    elif target.kind.name == "cuda":
        return tvm_tirx.cuda()
    else:
        raise RuntimeError(f"Unsupported target kind for runtime device: {target.kind.name}")


def check_and_run(func: str | Callable, *args, **kwargs) -> Any:
    """Check if the function is a string or a callable, and run it."""
    if isinstance(func, str):
        func = get_global_func(func)
    return func(*args, **kwargs)  # type: ignore


class OriginalModule:
    """Original module class for deduplication."""

    def __init__(self, mod: IRModule):
        self.mod = mod

    def __eq__(self, __o: "OriginalModule") -> bool:  # type: ignore
        return tvm_ffi.structural_equal(self.mod, __o.mod)

    def __hash__(self) -> int:
        return tvm_ffi.structural_hash(self.mod)


def initializer() -> None:
    """Initializer function to register the functions on PopenWorker."""

    @register_global_func("tvm.s_tir.meta_schedule.testing.default_check_metric")
    def default_check_metric(  # pylint: disable=unused-variable,unreachable-code
        lhs: list[tvm_tirx.runtime.Tensor], rhs: list[tvm_tirx.runtime.Tensor]
    ) -> bool:
        """Check if the outputs are equal

        Parameters
        ----------
        lhs : List[tvm_tirx.runtime.Tensor]
            The first list of Tensors to compare.

        rhs : List[tvm_tirx.runtime.Tensor]
            The second list of Tensors to compare.

        Returns
        -------
        is_equal : bool
            Whether the two lists of Tensors are equal.
        """
        assert len(lhs) == len(rhs), "Different number of outputs from two modules"
        for i in range(len(lhs)):  # pylint: disable=consider-using-enumerate
            if not np.allclose(lhs[i].numpy(), rhs[i].numpy(), rtol=1e-3, atol=2e-3):
                return False
        return True


@register_global_func("tvm.s_tir.meta_schedule.testing.default_input_generator")
def default_input_generator(  # pylint: disable=unused-variable
    mod: IRModule,
) -> list[tvm_tirx.runtime.Tensor]:
    """Default input generator function

    Parameters
    ----------
    mod : IRModule
        The IRModule to generate the input data for.

    Returns
    -------
    inputs : List[tvm_tirx.runtime.Tensor]
        The generated input data.
    """

    args_info = ms.arg_info.TensorInfo.from_function(mod["main"])
    inputs = [
        tvm_tirx.runtime.tensor(
            generate_input_data(input_shape=arg_info.shape, input_dtype=arg_info.dtype)
        )
        for arg_info in args_info
    ]
    return inputs


def to_numpy(a: list[tvm_tirx.runtime.Tensor]) -> list[np.ndarray]:
    """Convert a list of TVM Tensor to a list of numpy array

    Parameters
    ----------
    a : List[tvm_tirx.runtime.Tensor]
        The list of TVM Tensor to be converted

    Returns
    -------
    b : List[np.ndarray]
        The list of numpy array
    """
    assert a is not None, "Empty result cannot be converted to numpy"
    return [x.numpy() for x in a]


def to_tvm_tensor(a: list[np.ndarray]) -> list[tvm_tirx.runtime.Tensor]:
    """Convert a list of numpy array to a list of TVM Tensor

    Parameters
    ----------
    a : List[np.ndarray]
        The list of numpy array to be converted.

    Returns
    -------
    b : List[tvm_tirx.runtime.Tensor]
        The list of TVM Tensor.
    """
    assert a is not None, "Empty result cannot be converted to TVM Tensor"
    return [tvm_tirx.runtime.tensor(x) for x in a]


def is_failed_record(record: ms.database.TuningRecord) -> bool:
    """Check if a tuning record is failed.

    Parameters
    ----------
    record : TuningRecord
        The tuning record to check.

    Returns
    -------
    is_failed : bool
    """
    return len(record.run_secs) == 1 and record.run_secs[0] == 1e9


def print_with_counter_func(counter: int, total: int) -> Callable:
    """Print with counter

    Parameters
    ----------
    counter : int
        The counter to print with.
    total : int
        The total number of items to print with.

    Returns
    -------
    print_result : Callable
        The print result function.
    """

    def print_result(
        result: str,
        *,
        original_mod: IRModule = None,
        scheduled_mod: IRModule = None,
        inputs: list[np.ndarray] | None = None,
        original_res: list[np.ndarray] | None = None,
        scheduled_res: list[np.ndarray] | None = None,
        original_run_secs: list[float] | None = None,
        scheduled_run_secs: list[float] | None = None,
        exception: Exception | None = None,
        trace: str | None = None,
    ) -> None:
        """Print the validation result."""
        status = f"Progress {counter: 6d} / {total: 6d} (estimated) checked, result: {result:>10}, "

        if result in ["pass", "wrong answer"]:
            status += (
                f"original: {mean(original_run_secs) * 1e3: 10.3f} ms, "
                f"scheduled: {mean(scheduled_run_secs) * 1e3: 10.3f} ms"
            )

        output = [status]
        if result not in ["pass", "skip"]:
            output.extend(
                [
                    "Original IRModule:" + DELIMITOR + original_mod.script(),
                    "Scheduled IRModule:" + DELIMITOR + scheduled_mod.script(),
                    "Trace" + DELIMITOR + str(trace),
                ]
            )
            if result == "wrong answer":
                output.extend(
                    [
                        "Input:" + DELIMITOR + str(inputs),
                        "Original Result:" + DELIMITOR + str(original_res),
                        "Scheduled Result:" + DELIMITOR + str(scheduled_res),
                        "Max Diff:"
                        + DELIMITOR
                        + str(
                            [
                                np.max(np.abs(original_res[i] - scheduled_res[i]))
                                for i in range(len(original_res))
                            ]
                        )
                        + "\n",
                    ]
                )
            elif result == "exception":
                output.extend(["Exception:" + DELIMITOR + str(exception) + "\n"])
            else:
                raise ValueError(f"Unknown result: {result}")
        print("\n\n".join(output))

    return print_result


def make_alloc_arg_and_check(
    inputs: list[np.ndarray],
    original_mod: IRModule,
    scheduled_mod: IRModule,
    trace: str,
    original_res: list[np.ndarray],
    original_run_secs: list[float],
    print_result: Callable,
) -> tuple[Callable, Callable]:
    """Make alloc_arg and check functions for the given inputs and collect results.

    Parameters
    ----------
    inputs : List[np.ndarray]
        The inputs to the two modules.
    original_mod : IRModule
        The original IRModule.
    scheduled_mod : IRModule
        The scheduled IRModule.
    trace : str
        The trace of the scheduled IRModule.
    original_res : List[np.ndarray]
        The original results.
    original_run_secs : List[float]
        The original run times.
    print_result : Callable
        The print result function.

    Returns
    -------
    f_with_args_alloc_argument : Callable
        The function to allocate arguments.

    f_with_args_run_evaluator : Callable
        The function to run evaluator.
    """

    def f_with_args_alloc_argument_common(
        device: tvm_tirx.runtime.Device,
        args_info: ms.runner.rpc_runner.T_ARG_INFO_JSON_OBJ_LIST,  # pylint: disable=unused-argument
        alloc_repeat: int,
    ) -> list[ms.runner.rpc_runner.T_ARGUMENT_LIST]:
        """Allocate arguments using the given inputs.

        Parameters
        ----------
        session : RPCSession
            The RPC session.
        device : Device
            The device.
        args_info : T_ARG_INFO_JSON_OBJ_LIST
            argument information.
        alloc_repeat : int
            The number of times to repeat the allocation.

        Returns
        -------
        args_list : List[T_ARGUMENT_LIST]
            The list of argument lists.
        """
        return [
            [tvm_tirx.runtime.tensor(arg, device=device) for arg in inputs]
            for _ in range(alloc_repeat)
        ]

    def f_with_args_run_evaluator_common(
        rt_mod: tvm_ffi.Module,
        device: tvm_tirx.runtime.Device,
        evaluator_config: ms.runner.EvaluatorConfig,
        repeated_args: list[ms.runner.rpc_runner.T_ARGUMENT_LIST],
    ) -> list[float]:
        """With args function to run the evaluator

        Parameters
        ----------
        session : tvm.rpc.RPCSession
            The RPC session
        rt_mod: Module
            The runtime module
        device: Device
            The device to run the evaluator
        evaluator_config: EvaluatorConfig
            The evaluator config
        repeated_args: List[T_ARGUMENT_LIST]
            The repeated arguments

        Returns
        -------
        costs: List[float]
            The evaluator results
        """
        evaluator = tvm.runtime.time_evaluator(
            rt_mod,
            func_name=rt_mod.entry_name,
            device=device,
            number=evaluator_config.number,
            repeat=evaluator_config.repeat,
            min_repeat_ms=evaluator_config.min_repeat_ms,
            f_preproc=(
                "runtime.CacheFlushCPUArguments" if evaluator_config.enable_cpu_cache_flush else ""
            ),
        )

        repeated_costs: list[list[float]] = []
        for args in repeated_args:
            device.sync()
            profile_result = evaluator(*args)
            repeated_costs.append(profile_result.results)
        costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]

        assert len(repeated_args) == 1, "Only support one set of arguments"
        scheduled_res = [arg.numpy() for arg in repeated_args[0]]  # type: ignore
        # fetch comparison function
        passed = check_and_run(
            ARGS.check_metric_func,
            to_tvm_tensor(original_res),
            to_tvm_tensor(scheduled_res),
        )

        print_result(
            result="pass" if passed else "wrong answer",
            original_mod=original_mod,
            scheduled_mod=scheduled_mod,
            trace=trace,
            inputs=inputs,
            original_res=original_res,
            scheduled_res=scheduled_res,
            original_run_secs=original_run_secs,
            scheduled_run_secs=costs,
        )

        return costs

    def f_with_args_alloc_argument_rpc(
        rpc_session: ms.runner.rpc_runner.RPCSession,  # pylint: disable=unused-argument
        device: tvm_tirx.runtime.Device,
        args_info: ms.runner.rpc_runner.T_ARG_INFO_JSON_OBJ_LIST,
        alloc_repeat: int,
    ) -> list[ms.runner.rpc_runner.T_ARGUMENT_LIST]:
        return [[device.tensor(arg) for arg in inputs] for _ in range(alloc_repeat)]

    def f_with_args_run_evaluator_rpc(
        rpc_session: ms.runner.rpc_runner.RPCSession,  # pylint: disable=unused-argument
        rt_mod: tvm_ffi.Module,
        device: tvm_tirx.runtime.Device,
        evaluator_config: ms.runner.EvaluatorConfig,
        repeated_args: list[ms.runner.rpc_runner.T_ARGUMENT_LIST],
    ) -> list[float]:
        return f_with_args_run_evaluator_common(rt_mod, device, evaluator_config, repeated_args)

    if ARGS.rpc_config is None:
        return f_with_args_alloc_argument_common, f_with_args_run_evaluator_common
    else:
        return f_with_args_alloc_argument_rpc, f_with_args_run_evaluator_rpc


def local_build_and_run(
    mod: IRModule,
    target: Target,
    device: tvm_tirx.runtime.Device,
    inputs: list[np.ndarray],
) -> tuple[list[np.ndarray], list[float]]:
    """Build and run the module locally.

    Parameters
    ----------
    mod: IRModule
        The module to build and run
    target: Target
        The target to build the module
    device: Device
        The device to run the module
    inputs: List[np.ndarray]
        The inputs to run the module

    Returns
    -------
    res: List[np.ndarray]
        The results of running the module
    run_secs: List[float]
        The running time of running the module
    """
    # potential memory leak https://github.com/apache/tvm/issues/11096
    lib = tvm.compile(mod, target=target).jit()
    tvm_inputs = [tvm_tirx.runtime.tensor(inp, device=device) for inp in inputs]
    device.sync()
    func = tvm.runtime.time_evaluator(
        lib, lib.entry_name, device=device, number=ARGS.number, repeat=ARGS.repeat
    )
    benchmark_res = func(*tvm_inputs)
    device.sync()
    return [arg.numpy() for arg in tvm_inputs], list(benchmark_res.results)


def _check_builder_result(builder_result: ms.builder.BuilderResult) -> None:
    """Check if the builder result is defined.

    Parameters
    ----------
    builder_result: BuilderResult
        The builder result
    """
    assert builder_result.error_msg is None, "Builder failed: " + str(
        builder_result.error_msg if builder_result.error_msg else "Empty error message"
    )


def _apply_trace(mod: IRModule, trace: Trace) -> IRModule:
    """Apply the trace to the module.

    Parameters
    ----------
    mod: IRModule
        The module to apply the trace to
    trace: Trace
        The trace to apply

    Returns
    -------
    mod: IRModule
        The module with the trace applied
    """
    sch = Schedule(mod)
    trace.apply_to_schedule(sch, remove_postproc=False)
    return sch.mod


def _build_all_mods(
    mods: list[IRModule], builder: ms.builder.Builder, target: Target
) -> list[ms.builder.BuilderResult]:
    """Build all the modules.

    Parameters
    ----------
    mods: List[IRModule]
        The modules to build
    builder: Builder
        The builder to build the modules
    target: Target
        The target to build the modules

    Returns
    -------
    builder_results: List[BuilderResult]
        The builder results
    """
    builder_results = builder.build([ms.builder.BuilderInput(mod, target) for mod in mods])
    assert len(builder_results) == len(mods), (
        f"Unexpected number of build results, expected {len(mods)} got {len(builder_results)}"
    )
    return builder_results


def _run_single_mod(
    builder_result: ms.builder.BuilderResult,
    runner: ms.runner.Runner,
    dev_type: str,
) -> None:
    """Run a single module.

    Parameters
    ----------
    builder_result: BuilderResult
        The builder result
    runner: Runner
        The runner to run the module
    dev_type: str
        The device type
    """
    runner_futures = runner.run(
        # arginfo is not used in this case so we can pass an empty list
        [ms.runner.RunnerInput(builder_result.artifact_path, device_type=dev_type, args_info=[])]
    )
    assert len(runner_futures) == 1, (
        f"Unexpected number of runner futures, expected 1 got {len(runner_futures)}"
    )
    (runner_future,) = runner_futures  # pylint: disable=unbalanced-tuple-unpacking
    runner_res = runner_future.result()
    assert runner_res.error_msg is None, "Runner failed: " + (
        runner_res.error_msg if runner_res.error_msg else "Empty error message"
    )


def main():
    """Main function"""
    describe()
    with ms.Profiler() as profiler:
        # initialize
        target = ARGS.target
        dev_type = get_device_type(target)
        builder = ms.builder.LocalBuilder()
        database = ms.database.create(work_dir=ARGS.work_dir)

        # collect records
        with profiler.timeit("collect records"):
            records = database.get_all_tuning_records()
        total = len(records)
        print(
            f"Total {total} records to be validated. "
            f"Collected in {float(profiler.get()['collect records']): 3.3f} sec."
        )

        # collect unique original TIR
        with profiler.timeit("deduplicate records"):
            workloads = set()
            for record in records:
                workloads.add(OriginalModule(record.workload.mod))
        print(
            f"Total {len(workloads)} unique original TIR to validate. "
            f"Deduplicated in {float(profiler.get()['deduplicate records']): 3.3f} sec."
        )
        if ARGS.top_k < 10**9:
            print(f"Top {ARGS.top_k} records for each original TIR will be validated.")
            total = len(workloads) * ARGS.top_k
        print()

        # validate correctness
        counter = 0
        for item in workloads:
            original_mod = item.mod
            records = database.get_top_k(
                workload=database.commit_workload(original_mod), top_k=ARGS.top_k
            )
            if len(records) < ARGS.top_k:
                total -= ARGS.top_k - len(records)
            inputs = to_numpy(check_and_run(ARGS.input_generator_func, original_mod))
            original_res, original_run_secs = local_build_and_run(
                original_mod,
                target=ARGS.baseline_target,
                inputs=inputs,
                device=get_runtime_device(ARGS.baseline_target),
            )
            scheduled_mods = [_apply_trace(original_mod, record.trace) for record in records]
            builder_results = _build_all_mods(scheduled_mods, builder, target)  # type: ignore
            for i, record in enumerate(records):
                counter += 1
                print_result = print_with_counter_func(counter=counter, total=total)
                if is_failed_record(record):
                    # skip failed records where run_secs is 1e9
                    # these records are only negative samples for cost model
                    print_result(result="skip")
                    continue
                try:
                    # prepare scheduled module
                    scheduled_mod = scheduled_mods[i]
                    # check build result
                    builder_result = builder_results[i]
                    _check_builder_result(builder_result)
                    # fetch functions
                    (
                        f_with_args_alloc_argument,
                        f_with_args_run_evaluator,
                    ) = make_alloc_arg_and_check(
                        inputs,
                        original_mod,
                        scheduled_mod,
                        str(record.trace),
                        original_res=original_res,
                        original_run_secs=original_run_secs,
                        print_result=print_result,
                    )
                    # create runner
                    evaluator_config = ms.runner.EvaluatorConfig(
                        number=ARGS.number,
                        repeat=ARGS.repeat,
                        min_repeat_ms=ARGS.min_repeat_ms,
                        enable_cpu_cache_flush=ARGS.cpu_flush,
                    )
                    if ARGS.rpc_config is not None:
                        runner: ms.Runner = ms.runner.RPCRunner(  # type: ignore
                            ARGS.rpc_config,
                            evaluator_config=evaluator_config,
                            alloc_repeat=1,
                            f_alloc_argument=f_with_args_alloc_argument,
                            f_run_evaluator=f_with_args_run_evaluator,
                            initializer=initializer,
                        )
                    else:
                        runner: ms.Runner = ms.runner.LocalRunner(  # type: ignore
                            evaluator_config=evaluator_config,
                            alloc_repeat=1,
                            f_alloc_argument=f_with_args_alloc_argument,
                            f_run_evaluator=f_with_args_run_evaluator,
                            initializer=initializer,
                        )

                    # run and validate
                    _run_single_mod(builder_result, runner, dev_type)  # type: ignore
                except Exception as e:  # pylint: disable=broad-except, invalid-name
                    # validation failed with exception
                    print_result(
                        result="exception",
                        original_mod=original_mod,
                        scheduled_mod=scheduled_mod,
                        trace=str(record.trace),
                        exception=e,
                    )
                # clean up
                remove_build_dir(builder_result.artifact_path)
    print(f"Validation finished! Total time spent: {float(profiler.get()['Total']): 3.3f} sec.")


if __name__ == "__main__":
    main()
