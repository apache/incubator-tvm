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
from tvm.relax import script as R
from tvm.relax.transform import LowerAllocTensor
from tvm_tirx.script import ir as I


def test_basic():
    @I.ir_module
    class Before:
        @R.function
        def main():
            x = R.builtin.alloc_tensor(R.shape([16, 32]), "float32", 0)
            return x

    @I.ir_module
    class Expected:
        @R.function
        def main():
            storage = R.memory.alloc_storage(R.shape([2048]), 0, "global", "uint8")
            x = R.memory.alloc_tensor(storage, 0, R.shape([16, 32]), "float32")
            return x

    After = LowerAllocTensor()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


if __name__ == "__main__":
    tvm_tirx.testing.main()
