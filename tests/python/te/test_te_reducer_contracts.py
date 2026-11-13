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

"""Tensor-expression reducer construction and scalar overloads."""

import pytest

import tvm
import tvm_tirx


def test_reduce_constructor():
    lhs, rhs = tvm_tirx.tirx.Var("lhs", "int32"), tvm_tirx.tirx.Var("rhs", "int32")
    combiner = tvm.te.CommReducer([lhs], [rhs], [lhs + rhs], [tvm_tirx.tirx.const(0, "int32")])
    with pytest.raises(TypeError, match="te.CommReducer"):
        tvm.te.Reduce(None, [1], [tvm.s_tir.IterVar((0, 1), "x", 2)], None, 0)
    x = tvm.te.Reduce(combiner, [1], [tvm.s_tir.IterVar((0, 1), "x", 2)], None, 0)
    assert isinstance(x, tvm.te.Reduce)
    assert x.combiner.same_as(combiner)
    assert x.value_index == 0


def test_reduce_has_primitive_type():
    lhs, rhs = tvm_tirx.tirx.Var("lhs", "int32"), tvm_tirx.tirx.Var("rhs", "int32")
    combiner = tvm.te.CommReducer([lhs], [rhs], [lhs + rhs], [tvm_tirx.tirx.const(0, "int32")])
    expression = tvm.te.Reduce(
        combiner,
        [1],
        [tvm.s_tir.IterVar((0, 1), "i", tvm.s_tir.IterVar.CommReduce)],
        None,
        0,
    )
    assert isinstance(expression, tvm_tirx.ir.ExprWithOp)
    assert isinstance(expression.ty, tvm_tirx.ir.PrimType)


@pytest.mark.parametrize("num_args", list(range(2, 10)))
def test_comm_reducer(num_args):
    """Handle all arguments in TE comm_reducer

    The `te.comm_reducer` API has two distinct usages.  It can reduce
    a tensor along a specified axis, similar to numpy.max, or it can
    reduce several arguments together, simililar to Python's built-in
    max().  This choice is based on the type of the second argument.

    If the `te.comm_reducer` is reducing all arguments, then all
    arguments should be used.  In the past, the introduction of new
    arguments intended for use when reducing along a tensor axis has
    failed to forward these arguments when reducing along a list of
    items.
    """
    assert tvm.te.max(*range(num_args)) == num_args - 1
