..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _code_guide:

Code Guide and Tips
===================

.. contents::
  :depth: 2
  :local:

This is a document used to record tips in TVM codebase for reviewers and contributors.
Most of them are summarized through lessons during the contributing and process.


C++ Code Styles
---------------
- Use the Google C/C++ style.
- The public facing functions are documented in doxygen format.
- Favor concrete type declaration over ``auto`` as long as it is short.
- Favor passing by const reference (e.g. ``const Expr&``) over passing by value.
  Except when the function consumes the value by copy constructor or move,
  pass by value is better than pass by const reference in such cases.
- Favor ``const`` member function when possible.

We use ``clang-format`` to enforce the code style. Because different version
of clang-format might change by its version, it is recommended to use the same
version of the clang-format as the main one.
You can use pre-commit hooks to run formatting checks:

.. code:: bash

    # Run clang-format on all files
    pre-commit run clang-format --all-files

    # Run all linters (Python + C++ + custom checks)
    pre-commit run --all-files


clang-format is also not perfect, when necessary, you can use disble clang-format on certain code regions.

.. code :: c

   // clang-format off
   void Test() {
      // clang-format will be disabled in this region.
   }
   // clang-format on


Because clang-format may not recognize macros, it is recommended to use macro like normal function styles.


.. code :: c

   #define MACRO_IMPL { custom impl; }
   #define MACRO_FUNC(x)

   // not preferred, because clang-format might recognize it as types.
   virtual void Func1() MACRO_IMPL

   // preferred
   virtual void Func2() MACRO_IMPL;

   void Func3() {
     // preferred
     MACRO_FUNC(xyz);
   }


Python Code Styles
------------------
- The functions and classes are documented in `numpydoc <https://numpydoc.readthedocs.io/en/latest/>`_ format.
- Check your code style using ``pre-commit run --all-files``
- Stick to language features in ``python 3.10``

- For functions with early returns, prefer ``if``/``elif``/``else``
  chains for functions with parallel and short bodies to the
  conditions, such as functions that apply a simple mapping to the
  arguments.  For more procedural functions, especially where the
  final ``else`` block would be much longer than the ``if`` and
  ``elif`` blocks, prefer having the final ``else`` case unindented.

  The pylint check ``no-else-return`` is disabled to allow for this
  distinction.  See further discussion `here
  <https://github.com/apache/tvm/pull/11327>`_.

  .. code:: python

    # All cases have bodies with similar flow control.  While this could
    # be expressed as a sequence of if conditions, a reader would need to
    # inspect the body of each condition to know that only one conditional
    # body may be reached.
    def sign(x):
        if x > 0:
            return "+"
        elif x < 0:
            return "-"
        else:
            return ""

    # The initial special case is an early return for a special case,
    # followed by a more general method.  Using an else block for the
    # condition would add unnecessary indentation for the remainder of the
    # function.
    def num_unique_subsets(values):
        if len(values)==0:
            return 1

        # Longer, more general solution here
        ...

Writing Python Tests
--------------------
We use `pytest <https://docs.pytest.org/en/stable/>`_ for all Python testing. Regular tests live
under ``tests/python``, while environment-specific nightly tests live under ``tests/nightly/python``.
See :doc:`testing` for details on running tests, target parametrization,
and the target-specific marks used by CI.

If you want your test to run over a variety of targets, parametrize over ``target`` with ``@pytest.mark.parametrize``, tag GPU targets with ``@pytest.mark.gpu`` so the CI routes them to GPU nodes, and skip a target that is unavailable on the current machine with :py:func:`tvm_tirx.testing.device_enabled`. For example:

.. code:: python

  @pytest.mark.parametrize("target", ["llvm", pytest.param("cuda", marks=pytest.mark.gpu)])
  def test_mytest(target):
      if not tvm_tirx.testing.device_enabled(target):
          pytest.skip(f"{target} not enabled")
      dev = tvm_tirx.device_from_target(target)
      ...

will run ``test_mytest`` with ``target="llvm"`` and ``target="cuda"``, skipping any target whose device is not present. If you only want to test against a single target, drop the parametrization and hardcode the target. Mark GPU tests with ``@pytest.mark.gpu`` so the CI can select them, and skip when the required feature is unavailable with ``@pytest.mark.skipif``. For example, CUDA tests use:

.. code:: python

  @pytest.mark.gpu
  @pytest.mark.skipif(not tvm_tirx.testing.env.has_cuda(), reason="need cuda")
  def test_mycudatest():
    ...

The ``tvm_tirx.testing.env`` module exposes a ``has_*()`` probe for each runtime and hardware feature (e.g. ``has_cuda()``, ``has_rocm()``, ``has_vulkan()``, ``has_llvm()``). To skip a test when an optional Python package is missing, use ``pytest.importorskip("package_name")``.


Network Resources
-----------------

In CI, downloading files from the Internet is a big source of flaky test failures (e.g. remote
server can go down or be slow), so try to avoid using the network at all during tests. In some cases
this isn't a reasonable proposition (e.g. the docs tutorials which need to download models).

Prefer checked-in fixtures or generated data. If a download is unavoidable, arrange a
stable project-managed mirror with the maintainers.


Handle Integer Constant Expression
----------------------------------
We often need to handle constant integer expressions in TVM. Before we do so, the first question we want to ask is that is it really necessary to get a constant integer. If symbolic expression also works and let the logic flow, we should use symbolic expression as much as possible. So the generated code works for shapes that are not known ahead of time.

Note that in some cases we cannot know certain information, e.g. sign of symbolic variable, it is ok to make assumptions in certain cases. While adding precise support if the variable is constant.

If we do have to get constant integer expression, we should get the constant value using type ``int64_t`` instead of ``int``, to avoid potential integer overflow. We can always reconstruct an integer with the corresponding expression type via ``make_const``. The following code gives an example.

.. code:: c++

   Expr CalculateExpr(Expr value) {
     int64_t int_value = GetConstInt<int64_t>(value);
     int_value = CalculateExprInInt64(int_value);
     return make_const(value.type(), int_value);
   }
