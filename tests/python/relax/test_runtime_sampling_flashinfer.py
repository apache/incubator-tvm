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
import random

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm_tirx.support import utils


@pytest.mark.skip(reason="Requires FlashInfer enabled and proper setup")
def test_sampling():
    def load_module(name: str, static_modules: list[tvm_ffi.Module]):
        assert len(static_modules) > 0
        if len(static_modules) == 1:
            return static_modules[0]
        static_mod = static_modules[0]
        for mod in static_modules[1:]:
            static_mod.import_module(mod)
        temp = utils.tempdir()
        mod_path = temp.relpath(f"{name}.so")
        tvm_tirx.codegen.Executable(static_mod).export_library(mod_path)
        return tvm_tirx.runtime.load_module(mod_path)

    # Test configuration
    batch_size = 10
    vocab_size = 5
    num_iterations = 1000
    tol_atol = 0.02
    tol_rtol = 0.05  # relative tolerance

    # Probability tensor (each row sums to 1)
    probs_np = np.array([[0.1, 0.2, 0.3, 0.2, 0.2] for _ in range(batch_size)], dtype="float32")

    target = tvm_tirx.testing.run_with_gpu_lock(
        lambda: tvm_tirx.target.Target.from_device(tvm_tirx.cuda())
    )
    sampling_mod = load_module(
        "flashinfer_sampling",
        relax.backend.cuda.flashinfer.gen_sampling_module(
            target=target,
        ),
    )
    sampling_func = sampling_mod["sampling_from_probs"]

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        prob_tvm = tvm_tirx.runtime.tensor(probs_np, device=dev)
        output_tvm = tvm_tirx.runtime.empty((batch_size,), "int32", device=dev)
        counts = np.zeros((batch_size, vocab_size), dtype="int32")

        for _ in range(num_iterations):
            deterministic = False
            philox_seed = np.uint64(random.getrandbits(63))
            philox_offset = np.uint64(random.getrandbits(63) % 1000)

            sampling_func(prob_tvm, output_tvm, None, deterministic, philox_seed, philox_offset, 0)

            out = output_tvm.numpy()
            for i in range(batch_size):
                sampled_token = out[i]
                counts[i, sampled_token] += 1

        frequencies = counts / float(num_iterations)
        for row in range(batch_size):
            tvm_tirx.testing.assert_allclose(
                frequencies[row], probs_np[row], rtol=tol_rtol, atol=tol_atol
            )

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    # Run the test standalone (if not using pytest)
    test_sampling()
