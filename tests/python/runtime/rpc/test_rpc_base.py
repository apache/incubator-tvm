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

import pytest

import tvm_tirx.testing
from tvm.rpc import base


@pytest.mark.parametrize("device_key", ["16e995b6", "127.0.0.1:5555"])
def test_rpc_base_random_key(device_key):
    random.seed(0)
    key = base.random_key(device_key)
    assert key.startswith(device_key)
    res_device_key, _ = base.split_random_key(key)
    assert device_key == res_device_key
    # start with seed 0 as well, but use cmap arg(a conflict map)
    # to generate another unique random key
    random.seed(0)
    new_key = base.random_key(device_key, cmap={key})
    assert key != new_key
    assert new_key.startswith(device_key)
    res_device_key2, _ = base.split_random_key(new_key)
    assert device_key == res_device_key2


if __name__ == "__main__":
    tvm_tirx.testing.main()
