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
import numpy as np
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import te
from tvm.contrib.dlpack import to_pytorch_func


def verify_torch_dlpack():
    a = np.random.randn(1337)
    tvm_a = tvm_tirx.runtime.tensor(a)
    np.testing.assert_equal(tvm_ffi.from_dlpack(tvm_a).numpy(), a)

    try:
        import torch
        import torch.utils.dlpack

        x = torch.rand(56, 56)
        tvm_x = tvm_ffi.from_dlpack(torch.utils.dlpack.to_dlpack(x))
        np.testing.assert_equal(x.numpy(), tvm_x.numpy())
        y = tvm_ffi.from_dlpack(tvm_x)
        np.testing.assert_equal(y.numpy(), tvm_x.numpy())
        np.testing.assert_equal(torch.utils.dlpack.from_dlpack(y).numpy(), tvm_x.numpy())

        n = tvm_ffi.convert(137)
        xx = torch.rand(137, 137)
        yy = torch.rand(137, 137)
        zz2 = torch.empty(137, 137)
        zz = xx.mm(yy)
        XX = te.placeholder((n, n), name="X")
        YY = te.placeholder((n, n), name="Y")

        k = te.reduce_axis((0, n), name="k")
        ZZ = te.compute((n, n), lambda i, j: te.sum(XX[i, k] * YY[k, j], axis=k))
        # No need to speficy target_host if it's llvm
        # Otherwise you will need to specify the target and target_host
        f = tvm.compile(te.create_function([XX, YY, ZZ]))

        f_pytorch = to_pytorch_func(f)
        zz2 = torch.empty(137, 137)
        f_pytorch(xx, yy, zz2)
        tvm_tirx.testing.assert_allclose(zz.numpy(), zz2.numpy(), rtol=1e-4, atol=1e-4)

    except ImportError:
        pass


def test_torch_dlpack():
    # Run dlpack interoperability test a few times to make sure it's stable.
    for i in range(5):
        verify_torch_dlpack()


if __name__ == "__main__":
    test_torch_dlpack()
