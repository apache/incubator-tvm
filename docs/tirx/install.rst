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

Installation
============

There are two pieces:

- the **TIRx compiler** (``tvm.tirx``), which ships inside Apache TVM — this is
  the authoring, IR, lowering, and backend infrastructure. Compiling or running
  for a particular target also requires a TVM build with that target enabled;
- the optional **kernel library** (``tirx-kernels``), a collection of ready-made
  kernels and benchmark infrastructure built with TIRx.

Requirements
------------

- Python ≥ 3.10.
- For the CUDA programming guide and bundled kernels: a CUDA-enabled TVM build,
  an NVIDIA driver, and a compatible CUDA toolkit. The bundled kernels target
  Blackwell (``sm_100a``); TIRx itself also supports other target backends.

Install the TIRx compiler
-------------------------

Install the Apache TVM wheel (the TIRx compiler is the ``tvm.tirx`` module):

.. code-block:: bash

   pip install apache-tvm

The wheel is enough to inspect and author TIRx IR. Some CUDA workflows also
need NVIDIA's Python CUDA bindings, available through ``apache-tvm[cuda]``.
That extra does not enable CUDA in the TVM library and does not install a driver
or toolkit. To compile and run the CUDA examples, use a TVM build configured
with ``TVM_TIRX_USE_CUDA=ON``; see :ref:`install TVM from source <install-from-source>`.

Verify:

.. code-block:: bash

   python -c "import tvm, tvm.tirx; print(tvm.__version__)"

Install the kernel library (optional)
-------------------------------------

Install the latest ``tirx-kernels`` release from PyPI:

.. code-block:: bash

   pip install tirx-kernels

Or install a checkout for development:

.. code-block:: bash

   git clone https://github.com/mlc-ai/tirx-kernels
   cd tirx-kernels
   pip install -e .

Apache TVM and PyTorch are externally managed runtime/compiler dependencies;
installing ``tirx-kernels`` does not install them. Put a TIRx-enabled TVM on
``PYTHONPATH`` and use a CUDA build of PyTorch matching the system. Individual
correctness tests and reference baselines need additional upstream projects.
For a source checkout, install their pinned, mutually compatible revisions with:

.. code-block:: bash

   python scripts/install_reference_dependencies.py

The kernel registry and reference dependency set evolve independently of TVM.
Use ``python -m tirx_kernels.registry --format json`` for the installed kernel
list, and consult the `tirx-kernels README
<https://github.com/mlc-ai/tirx-kernels#readme>`_ and
``reference-dependencies.json`` in that repository for current optional
requirements.

Standalone CMake build
----------------------

The core can be built independently from the repository checkout. Initialize
its bundled FFI dependency, then use native CMake options (CMake 3.26 or newer)::

    git submodule update --init --recursive 3rdparty/tvm-tirx
    cmake -S 3rdparty/tvm-tirx -B 3rdparty/tvm-tirx/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DTVM_TIRX_BUILD_TESTS=ON
    cmake --build 3rdparty/tvm-tirx/build
    ctest --test-dir 3rdparty/tvm-tirx/build
    cmake --install 3rdparty/tvm-tirx/build --prefix /path/to/install

Core feature options use the ``TVM_TIRX_USE_`` prefix. SDK selectors such as
``TVM_TIRX_USE_CUDA`` and ``TVM_TIRX_USE_LLVM`` also accept an SDK path or
``llvm-config`` command. Compiler caches, linkers, and toolchain flags use
standard CMake settings.

LLVM is enabled by default and requires LLVM 15 or newer with development
headers and libraries. Install ``llvm-dev`` on Debian/Ubuntu, ``llvm-devel``
on Fedora, or ``llvmdev`` from conda-forge. Explicitly set
``TVM_TIRX_USE_LLVM=OFF`` to omit LLVM support.

CUDA defaults to ``AUTO``: a usable toolkit, including its compiler, runtime,
and driver library or stub, enables CUDA; otherwise the build continues
without CUDA.

C++ tests are off by default; enabling them supplies GTest in the test subtree.
Installation includes public headers, libraries, and CMake package targets.

With the dependencies above installed and the FFI submodule initialized,
build and install the Python core package using the same defaults::

    python -m pip install ./tirx
    python -c "import tvm_tirx"

This installs the ``apache-tvm-tirx`` distribution and its Python dependencies.
To override the defaults, pass CMake options through ``CMAKE_ARGS`` or pip's
``--config-settings=cmake.define.<OPTION>=<VALUE>``. Quote values containing
spaces inside ``CMAKE_ARGS``::

    CMAKE_ARGS='-DTVM_TIRX_USE_LLVM="llvm-config --link-static"' \
        python -m pip install -v ./tirx

The repository's native build script also accepts ``CMAKE_ARGS`` for its TVM
configuration, including parent-only options::

    mkdir -p build
    CMAKE_ARGS='-DTVM_TIRX_USE_LLVM=ON -DTVM_USE_RPC=ON' \
        python tests/scripts/task_build.py --build-dir build

The native build script applies these settings only to TVM configuration;
its separate source-FFI Python package build keeps its own defaults. Direct
``cmake`` invocations use command-line options as shown above, since CMake
itself does not read ``CMAKE_ARGS``.

A parent project can use ``add_subdirectory(path/to/tirx)`` and link to
``tvm_tirx_compiler::shared`` or ``tvm_runtime::shared``. This builds the bundled FFI
source targets and leaves tests, packaging, and installation to the parent.
TIRx registers installation rules only when it is the top-level project.
An embedding parent owns the installation rules for the libraries, public
headers, and package files it chooses to include.

Installed consumers
-------------------

An ordinary downstream compiler links TIRx and the separate FFI package
explicitly. Install both Python packages into the environment used by CMake::

    python -m pip install apache-tvm-tirx apache-tvm-ffi

Query each package's CMake directory::

    python -m tvm_tirx.config --cmakedir
    python -m tvm_ffi.config --cmakedir

Pass those directories as ``tvm_tirx_DIR`` and ``tvm_ffi_DIR`` when configuring
the consumer. Its CMakeLists.txt uses both packages directly:

.. code-block:: cmake

    find_package(tvm_tirx CONFIG REQUIRED)
    find_package(tvm_ffi CONFIG REQUIRED)
    add_executable(app main.cc)
    target_link_libraries(app PRIVATE tvm_tirx_compiler::shared tvm_ffi::shared)

The public TIRx targets are ``tvm_tirx::header``, ``tvm_tirx_compiler::shared``, and
``tvm_runtime::shared``. The package config describes the normal Python build;
it does not reconstruct build options, expose static or backend targets, or
find/link FFI implicitly. Enabled backend libraries are still installed and
loaded by the runtime. Consumers of an internally built static runtime supply
its dependencies, including OpenMP when enabled, themselves.

TIRx installs only its own public headers, libraries, and package helpers;
it does not copy FFI libraries, headers, or CMake files. For a native prefix
installation, use its ``lib/cmake/tvm_tirx`` directory (or the configured
library directory). The handwritten config queries that exact prefix, so a
separately installed TIRx wheel cannot override the selected installation.
``Python_EXECUTABLE`` selects the Python interpreter used for discovery.

Component configuration CLI
---------------------------

``tvm-tirx-config`` (also ``python -m tvm_tirx.config``) prints shell-quoted
compiler and linker arguments without initializing TVM, device runtimes, or a
device driver. The compiler library is named ``libtvm_tirx_compiler``; the
Python package remains ``tvm_tirx``. For example::

    tvm-tirx-config --version --prefix --includedir --libdir --cmakedir
    tvm-tirx-config --components
    tvm-tirx-config --cxxflags --ldflags --libs compiler
    tvm-tirx-config --ldflags --libs runtime
    tvm-tirx-config --ldflags --libs runtime cuda

Positional components select ``compiler``, ``runtime``, ``ffi``, or an enabled
device runtime (``cuda``, ``metal``, ``opencl``, ``vulkan``, ``rocm``). Dependencies
are included once, after their dependents: compiler and device runtimes depend
on runtime, which depends on FFI. Selecting compiler does not select any device
runtime. Like ``llvm-config``, omitting components or selecting ``all`` means
all available components. Select components explicitly when optional device
libraries are unwanted.

``--cflags`` and ``--cxxflags`` include the matching FFI and DLPack headers;
``--ldflags`` supplies library search directories. ``--libs`` supplies the
complete shared-library inputs, including scoped retention flags for runtime
registrations (ELF ``--push-state,--no-as-needed`` / ``--pop-state`` or Mach-O
``-needed_library``). ``--libnames`` and ``--libfiles`` print only filenames and
full paths, respectively, without retention switches. Use ``shlex.split`` when
consuming these outputs from Python, including paths containing spaces::

    flags = shlex.split(subprocess.check_output(
        ["tvm-tirx-config", "--libs", "runtime"], text=True))

``--link-shared`` is the default. ``--link-static`` reports an unavailable mode:
the distributed runtime archive alone is not a complete static dependency
closure, and the FFI package does not distribute all of its static dependencies.
The CLI never substitutes shared libraries for an explicit static request.
``--install-prefix PATH`` and ``--library-dir PATH`` query an explicit native
installation. The FFI dependency is selected from the invoking Python environment.

For ahead-of-time deployment, export an LLVM-host module with
``tvm_tirx.codegen.Executable(module).export_library("model.tar")``, extract every
object in the archive,
and link those objects into a shared library with the CLI's ``--ldflags`` and
``--libs runtime`` (plus the device component when required). CUDA exports also
contain host-format objects packing imported device code. Do not assume an
object count or filename. A fresh process can then use
``tvm_ffi.load_module("model.so")`` without importing ``tvm_tirx``. Make the
reported library directories available to the platform loader, for example
through the deployment's loader search path; runtime preloading is unnecessary.


Export and runtime APIs
-----------------------

``tvm_tirx.codegen.Executable`` owns compiler-side export, import packing, and
JIT linking. It accepts an ordinary ``tvm_ffi.Module``; compilation entry points
return this executable wrapper. Use ``executable.mod`` to inspect the underlying
FFI module and ``executable.jit()`` for a runnable module.
``tvm_tirx.codegen.load_static_library`` wraps an existing object for inclusion
in an executable's import graph.

Use ``tvm_ffi.Module`` and ``tvm_ffi.system_lib`` directly.
``tvm_tirx.runtime.load_module`` remains a convenience alias for
``tvm_ffi.load_module``. Loading no longer invokes a linker for objects or archives; export
and link explicitly before loading. Parent TVM's RPC server retains remote
object/archive linking through its registered server hook.

Benchmarking belongs to parent TVM: use
``tvm.runtime.time_evaluator(module, name, device, ...)`` and
``tvm.runtime.BenchmarkResult``. The standalone runtime and device libraries
retain the small device-timer interface but do not acquire parent benchmarking
policy or a dependency on ``libtvm_runtime_extra``.
