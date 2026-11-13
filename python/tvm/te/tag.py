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
"""Tag class for TVM operators."""

import functools
import warnings


class TagScope:
    """Tag scope object to set tag for operators, working as context
    manager and decorator both. See also tag_scope.
    """

    _current = None

    @classmethod
    def get_current(cls):
        if cls._current:
            cls._current.accessed = True
        return cls._current

    def __init__(self, tag):
        self._old_scope = None
        self.tag = tag
        self.accessed = False

    def __enter__(self):
        if TagScope._current is not None:
            raise ValueError("nested op_tag is not allowed for now")
        self._old_scope = TagScope._current
        TagScope._current = self
        return self

    def __exit__(self, ptype, value, trace):
        assert self._old_scope is None
        if not self.accessed:
            warnings.warn(f"Tag '{self.tag}' declared via TagScope was not used.")
        TagScope._current = self._old_scope

    def __call__(self, fdecl):
        @functools.wraps(fdecl)
        def tagged_fdecl(*args, **kwargs):
            with self:
                return fdecl(*args, **kwargs)

        return tagged_fdecl


def tag_scope(tag):
    """The operator tag scope.

    Parameters
    ----------
    tag: str
        The tag name.

    Returns
    -------
    tag_scope: TagScope
        The tag scope object, which can be used as decorator or
        context manger.

    Example
    -------
    .. code-block:: python

        n = te.var('n')
        m = te.var('m')
        l = te.var('l')
        A = te.placeholder((n, l), name='A')
        B = te.placeholder((m, l), name='B')
        k = te.reduce_axis((0, l), name='k')

        with tvm.te.tag_scope(tag='matmul'):
            C = te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[j, k], axis=k))

        # or use tag_scope as decorator
        @tvm.te.tag_scope(tag="conv")
        def compute_relu(data):
            return te.compute(
                data.shape, lambda *i: tvm_tirx.tirx.Select(data(*i) < 0, 0.0, data(*i))
            )
    """
    return TagScope(tag)
