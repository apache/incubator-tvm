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

"""S-TIR iteration variables reject non-arithmetic tensor regions."""

import pytest

import tvm
import tvm_tirx


def test_tensor_region_is_not_arithmetic_operand():
    int_region = tvm_tirx.tirx.Var("tensor", tvm_tirx.tirx.TensorType([4], "int32"))[0:4]
    with pytest.raises(TypeError, match="construct a TensorLoad explicitly"):
        tvm.s_tir.IterVar((0, 4), "i", tvm.s_tir.IterVar.DataPar) + int_region
