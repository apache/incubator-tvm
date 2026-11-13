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

.. _install-from-source:

Install from Source
===================
This page gives instructions on how to build and install the TVM package from source.

.. contents:: Table of Contents
    :local:
    :depth: 2

.. _install-dependencies:

Step 1. Install Dependencies
----------------------------

Apache TVM requires the following dependencies:

- CMake (>= 3.24.0)
- LLVM (recommended >= 15)
- Git
- A recent C++ compiler supporting C++ 20, at the minimum
    - GCC 10
    - Clang 10
    - Apple Clang 14
    - Visual Studio 2022

  Optional dependencies that use newer C++20 standard library facilities, such
  as ``std::format``, may require a newer standard library (for example GCC 13
  or newer on Linux).
- Python (>= 3.10)
- (Optional) Conda (Strongly Recommended)

System Dependencies (Non-Conda)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you are not using Conda, TVM requires several system libraries. On Ubuntu/Debian systems, install them with:

.. code:: bash

   sudo apt update
   sudo apt install zlib1g-dev libxml2-dev

For other operating systems, please refer to your package manager documentation.

The easiest way to manage dependency is via conda, which maintains a set of toolchains
including LLVM across platforms. To create the environment of those build dependencies,
one may simply use:

.. code:: bash

    # make sure to start with a fresh environment
    conda env remove -n tvm-build-venv
    # create the conda environment with build dependency
    conda create -n tvm-build-venv -c conda-forge \
        "llvmdev>=15" \
        "cmake>=3.26" \
        git \
        python=3.11
    # enter the build environment
    conda activate tvm-build-venv

.. note::
   **For Frontend Contributors (TFLite):** If you plan to run or contribute to the frontend tests (e.g., ``test_frontend_tflite.py``), you must install ``tensorflow==2.19.0``.

   .. code:: bash

       pip install tensorflow==2.19.0

   Python 3.11 is supported.

Step 2. Get Source from GitHub
------------------------------
You can also choose to clone the source repo from GitHub.

.. code:: bash

    git clone --recursive https://github.com/apache/tvm tvm

TVM pins the standalone `Apache TIRx <https://github.com/apache/tvm-tirx>`_
compiler and core runtime in ``3rdparty/tvm-tirx``. Recursive submodule
initialization also checks out its FFI dependency. After updating TVM, run
``git submodule update --init --recursive`` to restore the pinned versions.

.. note::
    It's important to use the ``--recursive`` flag when cloning the TVM repository, which will
    automatically clone the submodules. If you forget to use this flag, you can manually clone the submodules
    by running ``git submodule update --init --recursive`` in the root directory of the TVM repository.

Step 3. Configure and Build
---------------------------
Configure the build from the repository root with explicit CMake options. Enable
GPU SDKs and vendor libraries as needed:

.. code:: bash

    cd tvm
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DTVM_TIRX_USE_LLVM="llvm-config --ignore-libllvm --link-static" \
        -DTVM_HIDE_PRIVATE_SYMBOLS=ON \
        -DTVM_TIRX_USE_CUDA=OFF \
        -DTVM_TIRX_USE_METAL=OFF \
        -DTVM_TIRX_USE_VULKAN=OFF \
        -DTVM_TIRX_USE_OPENCL=OFF \
        -DTVM_USE_CUBLAS=OFF \
        -DTVM_USE_CUDNN=OFF \
        -DTVM_USE_CUTLASS=OFF

CMake stores these values in ``build/CMakeCache.txt``. To change an option,
rerun the configure command with its new ``-D`` value. Parent TVM features use
``TVM_USE_*``; core features retain ``TVM_TIRX_USE_*`` and FFI options retain
``TVM_FFI_*``. CMake also supports local ``CMakeUserPresets.json`` presets.


.. note::
    ``TVM_HIDE_PRIVATE_SYMBOLS`` is a configuration option that enables the ``-fvisibility=hidden`` flag.
    This flag helps prevent potential symbol conflicts between TVM and PyTorch. These conflicts arise due to
    the frameworks shipping LLVMs of different versions.

    `CMAKE_BUILD_TYPE <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_ controls default compilation flag:

    - ``Debug`` sets ``-O0 -g``
    - ``RelWithDebInfo`` sets ``-O2 -g -DNDEBUG`` (recommended)
    - ``Release`` sets ``-O3 -DNDEBUG``

After configuration succeeds, build with:

.. code-block:: bash

    cmake --build build --parallel $(nproc)

.. note::
    ``nproc`` may not be available on all systems, please replace it with the number of cores on your system

A successful build produces the compiler and runtime libraries under ``build/lib/``.

Apache TVM relies on the tvm-ffi package to support its python bindings.
Therefore, after we finish the build, we need to install the tvm-ffi package.

.. code-block:: bash

    pip install ./3rdparty/tvm-tirx/3rdparty/tvm-ffi


Leaving the build environment ``tvm-build-venv``, there are two ways to install the successful build into your environment:

-  Install via environment variable

.. code-block:: bash

    export TVM_HOME=/path-to-tvm
    pip install --target=$TVM_HOME/python $TVM_HOME/3rdparty/tvm-tirx/3rdparty/tvm-ffi
    export PYTHONPATH=$TVM_HOME/3rdparty/tvm-tirx/python:$TVM_HOME/python:$PYTHONPATH
    export TVM_TIRX_LIBRARY_PATH=$TVM_HOME/build/lib

- Install via pip local project

.. code-block:: bash

    conda activate your-own-env
    conda install python # make sure python is installed
    pip install "scikit-build-core>=0.11" "setuptools-scm>=8"
    pip install /path-to-tvm/3rdparty/tvm-tirx
    pip install --no-build-isolation /path-to-tvm

Step 4. Validate Installation
-----------------------------

Using a compiler infrastructure with multiple language bindings could be error-prone.
Therefore, it is highly recommended to validate Apache TVM installation before use.

**Step 1. Locate TVM Python package.** The following command can help confirm that TVM is properly installed as a python package and provide the location of the TVM python package:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm.__file__)"
    /some-path/lib/python3.11/site-packages/tvm/__init__.py

**Step 2. Confirm which TVM library is used.** When maintaining multiple build or installation of TVM, it becomes important to double check if the python package is using the proper ``libtvm`` with the following command:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm.base._LOADED_LIBS['tvm_runtime'])"
    <CDLL '/some-path/lib/python3.11/site-packages/tvm/libtvm_runtime.dylib', handle 95ada510 at 0x1030e4e50>

**Step 3. Inspect the build configuration.** For a local source build, inspect
the CMake cache to check which options were configured:

.. code-block:: bash

    cmake -LA -N build

The output includes parent ``TVM_USE_*`` options and the core's
``TVM_TIRX_USE_*`` options. Keep ``build/CMakeCache.txt`` when reporting a
configuration problem.


**Step 4. Check device detection.** Sometimes it could be helpful to understand if TVM could detect your device at all with the following commands:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm.metal().exist)"
    True # or False
    >>> python -c "import tvm; print(tvm.cuda().exist)"
    False # or True
    >>> python -c "import tvm; print(tvm.vulkan().exist)"
    False # or True

Please note that the commands above verify the presence of an actual device on the local machine for the TVM runtime (not the compiler) to execute properly. However, TVM compiler can perform compilation tasks without requiring a physical device. As long as the necessary toolchain, such as NVCC, is available, TVM supports cross-compilation even in the absence of an actual device.


Step 5. Extra Python Dependencies
---------------------------------
Building from source does not ensure the installation of all necessary Python dependencies.

Python Dependencies
~~~~~~~~~~~~~~~~~~~
The following commands can be used to install the extra Python dependencies:

* Necessary dependencies:

.. code:: bash

    pip3 install numpy cython

* If you want to use RPC Tracker

.. code:: bash

    pip3 install tornado

* If you want to use auto-tuning module

.. code:: bash

    pip3 install tornado psutil 'xgboost>=1.1.0' cloudpickle

Windows-Specific Build Notes
----------------------------

If you're building TVM on Windows, note these platform-specific considerations:

Path Conventions
~~~~~~~~~~~~~~~~
- Use forward slashes (``/``) in Python/CMake paths, not Windows backslashes
- Example: ``-DCMAKE_TOOLCHAIN_FILE=C:/toolchains/windows.cmake``.

Advanced Build Configuration
----------------------------

Ccache
~~~~~~
On supported platforms, the `Ccache compiler wrapper <https://ccache.dev/>`_ may be helpful for
reducing TVM's build time, especially when building with `cutlass <https://github.com/NVIDIA/cutlass>`_.
There are several ways to enable CCache in TVM builds:

    - Configure with ``-DTVM_USE_CCACHE=AUTO`` (the default). CCache will be used if it is found.

    - Ccache's Masquerade mode. This is typically enabled during the Ccache installation process.
      To have TVM use Ccache in masquerade, simply specify the appropriate C/C++ compiler
      paths when configuring TVM's build system.  For example:
      ``cmake -DCMAKE_CXX_COMPILER=/usr/lib/ccache/c++ ...``.

    - Ccache as CMake's C++ compiler prefix.  When configuring TVM's build system,
      set the CMake variable ``CMAKE_CXX_COMPILER_LAUNCHER`` to an appropriate value.
      E.g. ``cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ...``.


Building on Windows
~~~~~~~~~~~~~~~~~~~
TVM support build via MSVC using CMake. You will need to obtain a visual studio compiler.
The minimum required VS version is **Visual Studio Enterprise 2019** (NOTE: we test
against GitHub Actions' `windows-latest Runner <https://github.com/actions/runner-images>`_, so see that page for full details.
We recommend following :ref:`install-dependencies` to obtain necessary dependencies and
get an activated tvm-build environment. Then you can run the following command to build

.. code:: bash

    mkdir build
    cd build
    cmake ..
    cd ..

The above command generates the solution file under the build directory.
You can then run the following command to build

.. code:: bash

    cmake --build build --config Release -- /m

CUDA Configuration
..................
For CUDA support on Windows:

.. code-block:: batch

   set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
   set PATH=%CUDA_PATH%\bin;%PATH%
   cmake .. -DTVM_TIRX_USE_CUDA=ON

CMake & Compiler Setup
......................
- Specify generator: ``cmake -G "Visual Studio 16 2019" -A x64 ..``
- Ensure Python is in PATH or specify: ``-DPython_EXECUTABLE=C:\Python311\python.exe``


Building ROCm support
~~~~~~~~~~~~~~~~~~~~~

Currently, ROCm is supported only on linux, so all the instructions are written with linux in mind.

- Configure with ``-DTVM_TIRX_USE_ROCM=ON`` or set it to the ROCm SDK path.
- You need to first install HIP runtime from ROCm. Make sure the installation system has ROCm installed in it.
- Install LLVM (>= 15), and LLD, make sure ``ld.lld`` is available via command line.

.. _install-from-source-cpp-tests:

Enable C++ Tests
~~~~~~~~~~~~~~~~
We use `Google Test <https://github.com/google/googletest>`_ to drive the C++
tests in TVM. The easiest way to install GTest is from source.

.. code:: bash

    git clone https://github.com/google/googletest
    cd googletest
    mkdir build
    cd build
    cmake -DBUILD_SHARED_LIBS=ON ..
    make
    sudo make install

After installing GTest, the C++ tests can be built and started with ``./tests/scripts/task_cpp_unittest.sh``, or built via CMake with ``-DTVM_USE_GTEST=ON`` and then running ``./build/cpptest``.
