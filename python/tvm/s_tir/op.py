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
"""S-TIR operator construction helpers."""

import inspect
from collections.abc import Callable

from tvm.te import CommReducer
from tvm_tirx.tirx import Expr, Var


def comm_reducer(combiner: Callable, identity: list[Expr]) -> CommReducer:
    """
    Create a CommReducer from lambda inputs/outputs and the identities

    Parameters
    ----------
    combiner : Callable
        A binary function which takes two Expr as input to return a Expr.

    identity : List[Expr]
        A list of types of output Expr.

    Returns
    -------
    res : CommReducer
        The CommReducer.
    """
    params = inspect.signature(combiner).parameters
    num_args = len(params)
    args = []
    for name, i in zip(params.keys(), identity + identity):
        if isinstance(i, int):
            args.append(Var(name, "int32"))
        else:
            args.append(Var(name, i.ty))
    res = combiner(*args)
    if not isinstance(res, tuple):
        res = (res,)
    return CommReducer(args[: num_args // 2], args[num_args // 2 :], res, identity)
