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

import inspect

import pytest

import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


class Base:
    def test_compare(self):
        transform = relax.transform.ReorderTakeAfterMatmul()

        if inspect.isclass(self.Expected) and issubclass(self.Expected, Exception):
            with pytest.raises(self.Expected):
                transform(self.Before)
        else:
            after = transform(self.Before)
            tvm_tirx.ir.assert_structural_equal(self.Expected, after)


weight_table_size_before = T.dynamic("weight_table_size", "int64")
weight_table_size_expected = T.dynamic("weight_table_size", "int64")


class TestSimple(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([1, 16], "float32"),
            weight_table: R.Tensor([16, weight_table_size_before], "float32"),
            routing_table: R.Tensor([32], "int64"),
        ) -> R.Tensor([1, 32], "float32"):
            with R.dataflow():
                weight: R.Tensor([16, 32], "float32") = R.take(weight_table, routing_table, axis=1)
                out: R.Tensor([1, 32], "float32") = R.matmul(x, weight)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([1, 16], "float32"),
            weight_table: R.Tensor([16, weight_table_size_expected], "float32"),
            routing_table: R.Tensor([32], "int64"),
        ) -> R.Tensor([1, 32], "float32"):
            with R.dataflow():
                out_table: R.Tensor([1, weight_table_size_expected], "float32") = R.matmul(
                    x, weight_table
                )
                out: R.Tensor([1, 32], "float32") = R.take(out_table, routing_table, axis=1)
                R.output(out)
            return out


class TestVectorActivations(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight_table: R.Tensor([16, 64], "float32"),
            routing_table: R.Tensor([32], "int64"),
        ) -> R.Tensor([32], "float32"):
            weight = R.take(weight_table, routing_table, axis=1, mode="clip")
            out = R.matmul(x, weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight_table: R.Tensor([16, 64], "float32"),
            routing_table: R.Tensor([32], "int64"),
        ) -> R.Tensor([32], "float32"):
            out_table = R.matmul(x, weight_table)
            out = R.take(out_table, routing_table, axis=0, mode="clip")
            return out


class TestNoReorderVectorWeights(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([3], "float32"),
            weight_table: R.Tensor([3], "float32"),
            routing_table: R.Tensor([3], "int64"),
        ) -> R.Tensor([], "float32"):
            weight = R.take(weight_table, routing_table, axis=0, mode="clip")
            out = R.matmul(x, weight)
            return out

    Expected = Before


class TestNoReorderVectorWeightsWithMatrixActivations(Base):
    """Equal extents keep the invalid rewrite well typed, but change its result."""

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([3, 3], "float32"),
            weight_table: R.Tensor([3], "float32"),
            routing_table: R.Tensor([3], "int64"),
        ) -> R.Tensor([3], "float32"):
            weight = R.take(weight_table, routing_table, axis=0, mode="clip")
            out = R.matmul(x, weight)
            return out

    Expected = Before


batch_size_before = T.dynamic("batch_size", "int64")
weight_table_size_before = T.dynamic("weight_table_size", "int64")
batch_size_expected = T.dynamic("batch_size", "int64")
weight_table_size_expected = T.dynamic("weight_table_size", "int64")


class TestBatchedActivations(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([batch_size_before, 1, 16], "float32"),
            weight_table: R.Tensor([16, weight_table_size_before], "float32"),
            routing_table: R.Tensor([32], "int64"),
        ) -> R.Tensor([batch_size_before, 1, 32], "float32"):
            with R.dataflow():
                weight: R.Tensor([16, 32], "float32") = R.take(weight_table, routing_table, axis=1)
                out: R.Tensor([batch_size_before, 1, 32], "float32") = R.matmul(x, weight)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([batch_size_expected, 1, 16], "float32"),
            weight_table: R.Tensor([16, weight_table_size_expected], "float32"),
            routing_table: R.Tensor([32], "int64"),
        ) -> R.Tensor([batch_size_expected, 1, 32], "float32"):
            with R.dataflow():
                out_table: R.Tensor(
                    [batch_size_expected, 1, weight_table_size_expected], "float32"
                ) = R.matmul(x, weight_table)
                out: R.Tensor([batch_size_expected, 1, 32], "float32") = R.take(
                    out_table, routing_table, axis=2
                )
                R.output(out)
            return out


batch_size_before = T.dynamic("batch_size", "int64")
routing_table_size_before = T.dynamic("routing_table_size", "int64")
batch_size_expected = T.dynamic("batch_size", "int64")
routing_table_size_expected = T.dynamic("routing_table_size", "int64")


class TestStaticBatchedActivationsAndWeights(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([128, 1, 16], "float32"),
            weight_table: R.Tensor([routing_table_size_before, 16, 32], "float32"),
            routing_table: R.Tensor([128], "int64"),
        ) -> R.Tensor([128, 1, 32], "float32"):
            with R.dataflow():
                weight = R.take(weight_table, routing_table, axis=0)
                out = R.matmul(x, weight)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([128, 1, 16], "float32"),
            weight_table: R.Tensor([routing_table_size_expected, 16, 32], "float32"),
            routing_table: R.Tensor([128], "int64"),
        ) -> R.Tensor([128, 1, 32], "float32"):
            with R.dataflow():
                reordered_weight = R.permute_dims(weight_table, [1, 0, 2])
                fused_weight = R.reshape(reordered_weight, [16, routing_table_size_expected * 32])
                fused_output = R.matmul(x, fused_weight)
                reordered_output = R.reshape(
                    fused_output, [128, 1, routing_table_size_expected, 32]
                )
                tabular_output = R.take(reordered_output, routing_table, axis=2)
                out = R.einsum([tabular_output], "ijik->ijk")
                R.output(out)
            return out


batch_size_before = T.dynamic("batch_size", "int64")
routing_table_size_before = T.dynamic("routing_table_size", "int64")
batch_size_expected = T.dynamic("batch_size", "int64")
routing_table_size_expected = T.dynamic("routing_table_size", "int64")


class TestDynamicBatchedActivationsAndWeights(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([batch_size_before, 1, 16], "float32"),
            weight_table: R.Tensor([routing_table_size_before, 16, 32], "float32"),
            routing_table: R.Tensor([batch_size_before], "int64"),
        ) -> R.Tensor([batch_size_before, 1, 32], "float32"):
            with R.dataflow():
                weight = R.take(weight_table, routing_table, axis=0)
                out = R.matmul(x, weight)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([batch_size_expected, 1, 16], "float32"),
            weight_table: R.Tensor([routing_table_size_expected, 16, 32], "float32"),
            routing_table: R.Tensor([batch_size_expected], "int64"),
        ) -> R.Tensor([batch_size_expected, 1, 32], "float32"):
            with R.dataflow():
                reordered_weight = R.permute_dims(weight_table, [1, 0, 2])
                fused_weight = R.reshape(reordered_weight, [16, routing_table_size_expected * 32])
                fused_output = R.matmul(x, fused_weight)
                reordered_output = R.reshape(
                    fused_output, [batch_size_expected, 1, routing_table_size_expected, 32]
                )
                tabular_output = R.take(reordered_output, routing_table, axis=2)
                out = R.einsum([tabular_output], "ijik->ijk")
                R.output(out)
            return out


class TestPreserveTakeMode(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([1, 16], "float32"),
            weight_table: R.Tensor([16, 64], "float32"),
            routing_table: R.Tensor([32], "int64"),
        ) -> R.Tensor([1, 32], "float32"):
            with R.dataflow():
                weight = R.take(weight_table, routing_table, axis=1, mode="clip")
                out = R.matmul(x, weight)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([1, 16], "float32"),
            weight_table: R.Tensor([16, 64], "float32"),
            routing_table: R.Tensor([32], "int64"),
        ) -> R.Tensor([1, 32], "float32"):
            with R.dataflow():
                out_table = R.matmul(x, weight_table)
                out = R.take(out_table, routing_table, axis=1, mode="clip")
                R.output(out)
            return out


class TestPreserveTakeModeForBatchedWeights(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([128, 1, 16], "float32"),
            weight_table: R.Tensor([64, 16, 32], "float32"),
            routing_table: R.Tensor([128], "int64"),
        ) -> R.Tensor([128, 1, 32], "float32"):
            with R.dataflow():
                weight = R.take(weight_table, routing_table, axis=0, mode="clip")
                out = R.matmul(x, weight)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([128, 1, 16], "float32"),
            weight_table: R.Tensor([64, 16, 32], "float32"),
            routing_table: R.Tensor([128], "int64"),
        ) -> R.Tensor([128, 1, 32], "float32"):
            with R.dataflow():
                reordered_weight = R.permute_dims(weight_table, [1, 0, 2])
                fused_weight = R.reshape(reordered_weight, [16, 2048])
                fused_output = R.matmul(x, fused_weight)
                reordered_output = R.reshape(fused_output, [128, 1, 64, 32])
                tabular_output = R.take(reordered_output, routing_table, axis=2, mode="clip")
                out = R.einsum([tabular_output], "ijik->ijk")
                R.output(out)
            return out


if __name__ == "__main__":
    tvm_tirx.testing.main()
