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
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


def test_multinomial_from_uniform():
    bb = relax.BlockBuilder()
    prob0 = relax.Var("prob", R.Tensor((3, 5), "float32"))
    prob1 = relax.Var("prob", R.Tensor(ndim=2, dtype="float32"))
    prob2 = relax.Var("prob", R.Tensor(dtype="float32"))

    uniform_sample0 = relax.Var("u", R.Tensor((6, 1), "float32"))
    uniform_sample1 = relax.Var("u", R.Tensor(ndim=2, dtype="float32"))
    uniform_sample2 = relax.Var("u", R.Tensor(dtype="float32"))

    sample_indices0 = relax.Var("s", R.Tensor((6, 1), "int64"))
    sample_indices1 = relax.Var("s", R.Tensor((6, 1), "int32"))

    _check_inference(
        bb,
        relax.op.multinomial_from_uniform(prob0, uniform_sample0, sample_indices0),
        R.Tensor((6, 1), "int64"),
    )
    _check_inference(
        bb,
        relax.op.multinomial_from_uniform(prob0, uniform_sample0, sample_indices0, dtype="int32"),
        R.Tensor((6, 1), "int32"),
    )
    _check_inference(
        bb,
        relax.op.multinomial_from_uniform(prob1, uniform_sample1, sample_indices1),
        R.Tensor(ndim=2, dtype="int64"),
    )
    _check_inference(
        bb,
        relax.op.multinomial_from_uniform(prob1, uniform_sample1, sample_indices1, dtype="int32"),
        R.Tensor(ndim=2, dtype="int32"),
    )
    _check_inference(
        bb,
        relax.op.multinomial_from_uniform(prob2, uniform_sample2, sample_indices0),
        R.Tensor(dtype="int64"),
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
