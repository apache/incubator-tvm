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
# ruff: noqa: E501, F821
"""A script to measure GPU memory bandwidth"""

import argparse
import itertools

import numpy as np

import tvm
import tvm_tirx
from tvm import te
from tvm.s_tir.meta_schedule.runner import EvaluatorConfig, RPCConfig
from tvm.s_tir.testing.runner import local_run, rpc_run


def _parse_args() -> argparse.Namespace:
    def _parse_list_int(source: str):
        return [int(i) for i in source.split(",")]

    parser = argparse.ArgumentParser(
        prog="GPU memory bandwidth testing",
        description="""Example for host GPU:
    python -m tvm.exec.gpu_memory_bandwidth "nvidia/geforce-rtx-3090-ti" \
        --dtype "float32"
        --bx "8,16,32,64,128,256"      \
        --tx "32,64,128,256,512,1024"  \
        --vec "1,2,4" \

    Example for Android GPU: \
    python -m tvm.exec.gpu_memory_bandwidth "opencl" --target_host '{"kind": "llvm", "mtriple": "arm64-linux-android"}' \
        --rpc_host "127.0.0.1" \
        --rpc_port 9190 \
        --rpc_key "android" \
        --export_func "ndk" \
        --dtype "float32" \
        --bx "8,16,32,64,128,256"      \
        --tx "32,64,128,256,512,1024"  \
        --vec "1,2,4" \
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target",
        type=str,
        help="The target to be benchmarked",
    )
    parser.add_argument(
        "--target_host",
        type=str,
        default=None,
        help="The target host for build",
    )
    parser.add_argument(
        "--xo",
        type=int,
        default=1024,
        help="The value of `XO` in [XO, K, XI] => [XO, XI] reduction",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=64,
        help="The value of `K` in [XO, K, XI] => [XO, XI] reduction",
    )
    parser.add_argument(
        "--xi",
        type=int,
        default=4096,
        help="The value of `XI` in [XO, K, XI] -> [XO, XI] reduction",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="The data type to be used in the workload",
    )
    parser.add_argument(
        "--bx",
        type=_parse_list_int,
        default=[8, 16, 32, 64, 128, 256],
        help="The value to be used to split `XO` into [BX, _]",
    )
    parser.add_argument(
        "--tx",
        type=_parse_list_int,
        default=[32, 64, 128, 256, 512, 1024],
        help="Number of threads to be used",
    )
    parser.add_argument(
        "--vec",
        type=_parse_list_int,
        default=[1, 2, 4],
        help="Vector length to be used in vectorized load",
    )
    parser.add_argument(
        "--rpc_host",
        type=str,
        default=None,
        help="The address of RPC host (default: None, that means that RPC is not used)",
    )
    parser.add_argument(
        "--rpc_port",
        type=int,
        default=None,
        help="The port of RPC connection (default: None, that means that RPC is not used)",
    )
    parser.add_argument(
        "--rpc_key",
        type=str,
        default=None,
        help="The device key in RPC tracker (default: None, that means that RPC is not used)",
    )
    parser.add_argument(
        "--export_func",
        type=str,
        default="tar",
        help="Export function, actual only for RPC",
        choices=["tar", "ndk"],
    )
    return parser.parse_args()


def _workload(
    len_xo: int,
    len_k: int,
    len_xi: int,
    dtype: str,
):
    # pylint: disable=invalid-name
    A = te.placeholder((len_xo, len_k, len_xi), dtype=dtype, name="A")
    k = te.reduce_axis((0, len_k), "k")
    B = te.compute(
        (len_xo, len_xi),
        lambda i, j: te.sum(A[i, k, j], axis=k),
        name="B",
    )
    # pylint: enable=invalid-name
    return te.create_function([A, B])


def _schedule(
    sch: s_tir.Schedule,
    len_bx: int,
    len_tx: int,
    len_vec: int,
):
    # pylint: disable=invalid-name
    block = sch.get_sblock("B")
    xo, xi, k = sch.get_loops(block)
    bx, xo = sch.split(xo, factors=[len_bx, None])
    xi, tx, vec = sch.split(xi, factors=[None, len_tx, len_vec])
    sch.reorder(bx, xi, tx, xo, k, vec)
    bx = sch.fuse(bx, xi)
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")
    ldg = sch.cache_read(block, 0, "local")
    sch.compute_at(ldg, k, preserve_unit_loops=True)
    sch.vectorize(sch.get_loops(ldg)[-1])
    sch.decompose_reduction(block, k)
    # pylint: enable=invalid-name


def main():  # pylint: disable=too-many-locals
    """Entry point"""
    args = _parse_args()
    # pylint: disable=invalid-name
    target = tvm_tirx.target.Target(args.target)
    if args.target_host is not None:
        target = tvm_tirx.target.Target(args.target, host=args.target_host)
    dtype = args.dtype
    rpcConfig = None
    if args.rpc_host is not None and args.rpc_port is not None and args.rpc_key is not None:
        rpcConfig = RPCConfig(
            tracker_host=args.rpc_host,
            tracker_port=args.rpc_port,
            tracker_key=args.rpc_key,
            session_priority=1,
            session_timeout_sec=600,
        )

    a = np.random.uniform(-1, 1, (args.xo, args.k, args.xi)).astype(dtype)
    b = np.zeros((args.xo, args.xi), dtype=dtype)
    num_bytes = a.size * a.itemsize + b.size * b.itemsize
    print("###### Bandwidth Test ######")
    print(
        f"Workload [XO, K, XI] => [XO, XI]. "
        f"[{args.xo}, {args.k}, {args.xi}] => [{args.xo}, {args.xi}]"
    )
    print(f"Input size: {num_bytes / 1048576} MB")
    print(f"Target: {target}")

    # pylint: enable=invalid-name
    best_bandwidth = -1
    for len_bx, len_tx, len_vec in itertools.product(
        args.bx,
        args.tx,
        args.vec,
    ):
        func = _workload(
            len_xo=args.xo,
            len_k=args.k,
            len_xi=args.xi,
            dtype=dtype,
        )
        sch = tvm.s_tir.Schedule(func)
        _schedule(sch, len_bx, len_tx, len_vec)

        if rpcConfig is None:
            _, profile_result = local_run(
                tvm.compile(sch.mod, target=target),
                target.kind.name,
                [a, b],
                evaluator_config=EvaluatorConfig(
                    number=10,
                    repeat=1,
                    min_repeat_ms=100,
                    enable_cpu_cache_flush=False,
                ),
            )
        else:
            _, profile_result = rpc_run(
                tvm.compile(sch.mod, target=target),
                target.kind.name,
                [a, b],
                evaluator_config=EvaluatorConfig(
                    number=10,
                    repeat=1,
                    min_repeat_ms=100,
                    enable_cpu_cache_flush=False,
                ),
                rpc_config=rpcConfig,
                export_func=args.export_func,
            )
        bandwidth = num_bytes / profile_result.mean / (1024**3)
        bx = len_bx * args.xi // (len_tx * len_vec)  # pylint: disable=invalid-name
        mbs = num_bytes / 1024 / 1024
        print(
            f"bandwidth = {bandwidth:.3f} GB/s, bx = {bx}, tx = {len_tx}, "
            f"len_vec = {len_vec}, bytes = {mbs} MB"
        )
        if bandwidth > best_bandwidth:
            best_bandwidth = bandwidth
    print(f"peak bandwidth: {best_bandwidth:.3f} GB/s")


if __name__ == "__main__":
    main()
