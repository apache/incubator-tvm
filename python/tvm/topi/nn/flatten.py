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
"""TVM operator flatten compute."""

import tvm
import tvm_tirx
from tvm import te

from .. import tag


@tvm.te.tag_scope(tag=tag.INJECTIVE)
def flatten(data):
    """Flattens the input array into a 2-D array by collapsing the higher dimensions.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input array.

    Returns
    -------
    output : tvm.te.Tensor
        2-D array with collapsed higher dimensions.
    """
    ishape = data.shape
    dim = 1
    for i in range(1, len(ishape)):
        dim = dim * ishape[i]
    oshape = [ishape[0], dim]
    idxdiv = tvm_tirx.tirx.indexdiv
    idxmod = tvm_tirx.tirx.indexmod

    def unwrap(idx, shape):
        index = []
        for s in reversed(shape):
            index.append(idxmod(idx, s))
            idx = idxdiv(idx, s)
        return list(reversed(index))

    return te.compute(oshape, lambda i, j: data(i, *unwrap(j, ishape[1:])))
