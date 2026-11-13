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
"""Utilities for schedulable TensorIR objects."""

from typing import TypeVar

T = TypeVar("T")


def derived_object(cls: type[T]) -> type[T]:
    """A decorator to register derived subclasses for TVM objects.

    Parameters
    ----------
    cls : type
        The derived class to be registered.

    Returns
    -------
    cls : type
        The decorated TVM object.

    Example
    -------
    .. code-block:: python

        @register_object("s_tir.meta_schedule.PyRunner")
        class _PyRunner(meta_schedule.Runner):
            def __init__(self, f_run: Callable = None):
                self.__init_handle_by_constructor__(_ffi_api.RunnerPyRunner, f_run)

        class PyRunner:
            _tvm_metadata = {
                "cls": _PyRunner,
                "methods": ["run"]
            }
            def run(self, runner_inputs):
                raise NotImplementedError

        @derived_object
        class LocalRunner(PyRunner):
            def run(self, runner_inputs):
                ...
    """

    import functools  # pylint: disable=import-outside-toplevel
    import weakref  # pylint: disable=import-outside-toplevel

    def _extract(inst: type, name: str):
        """Extract function from intrinsic class."""

        def method(*args, **kwargs):
            return getattr(inst, name)(*args, **kwargs)

        for inherit_cls, base_cls in zip(cls.__mro__, cls.__mro__[1:]):
            # extract functions that differ from the base class
            if not hasattr(base_cls, name):
                continue
            if getattr(base_cls, name) is getattr(inherit_cls, name):
                continue
            return method

        # for task scheduler return None means calling default function
        # otherwise it will trigger a RuntimeError of method not implemented
        # on the c++ side when you call the method
        return None

    assert isinstance(cls.__base__, type)
    if hasattr(cls, "_type") and cls._type == "TVMDerivedObject":  # type: ignore
        raise TypeError(
            f"Inheritance from a decorated object `{cls.__name__}` is not allowed. "
            f"Please inherit from `{cls.__name__}._cls`."
        )
    assert hasattr(cls, "_tvm_metadata"), (
        "Please use the user-facing method overriding class, i.e., PyRunner."
    )

    base = cls.__base__
    metadata = getattr(base, "_tvm_metadata")
    fields = metadata.get("fields", [])
    methods = metadata.get("methods", [])

    base_cls = metadata["cls"]
    slots = []
    if getattr(base_cls, "__dictoffset__", 0) == 0:
        slots.append("__dict__")
    if getattr(base_cls, "__weakrefoffset__", 0) == 0:
        slots.append("__weakref__")

    class TVMDerivedObject(base_cls):  # type: ignore
        """The derived object to avoid cyclic dependency."""

        __slots__ = tuple(slots)

        _cls = cls
        _type = "TVMDerivedObject"

        def __init__(self, *args, **kwargs):
            """Constructor."""
            self._inst = cls(*args, **kwargs)

            super().__init__(
                # the constructor's parameters, builder, runner, etc.
                *[getattr(self._inst, name) for name in fields],
                # the function methods, init_with_tune_context, build, run, etc.
                *[_extract(self._inst, name) for name in methods],
            )

            # for task scheduler hybrid funcs in c++ & python side
            # using weakref to avoid cyclic dependency
            self._inst._outer = weakref.ref(self)

        def __getattr__(self, name):
            import inspect  # pylint: disable=import-outside-toplevel

            try:
                # fall back to instance attribute if there is not any
                # return self._inst.__getattribute__(name)
                result = self._inst.__getattribute__(name)
            except AttributeError:
                result = super().__getattr__(name)

            if inspect.ismethod(result):

                def method(*args, **kwargs):
                    return result(*args, **kwargs)

                # set __own__ to aviod implicit deconstruction
                setattr(method, "__own__", self)
                return method

            return result

        def __setattr__(self, name, value):
            if name not in ["_inst", "key", "handle"]:
                self._inst.__setattr__(name, value)
            else:
                super().__setattr__(name, value)

    functools.update_wrapper(TVMDerivedObject.__init__, cls.__init__)  # type: ignore
    TVMDerivedObject.__name__ = cls.__name__
    TVMDerivedObject.__doc__ = cls.__doc__
    TVMDerivedObject.__module__ = cls.__module__
    for key, value in cls.__dict__.items():
        if isinstance(value, classmethod | staticmethod):
            setattr(TVMDerivedObject, key, value)
    return TVMDerivedObject
