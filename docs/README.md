<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# TVM Documentation

This folder contains the source of TVM's documentation, hosted at https://tvm.apache.org/docs

## Build Locally

### With Docker (recommended)

Run these commands from the repository root. The Docker image aliases select
this checkout's CI images; their first download can take some time.

1. Build the FFI, TIRx, and TVM wheels in dependency order and run the CPU
   C++ tests, using the same commands as CI:

   ```bash
   docker/bash.sh ci_cpu bash -euo pipefail -c '
       source ./tests/scripts/task_install_build_dependencies.sh
       ./tests/scripts/task_build_cpu.sh
       sccache --show-stats
   '
   ```

2. Build the Python documentation without executing tutorials. Install the
   additional CPU documentation dependencies in the same container as the build:

   ```bash
   docker/bash.sh --env IS_LOCAL=1 \
       --env PYTHON_DOCS_ONLY=1 --env TVM_TUTORIAL_EXEC_PATTERN=none \
       ci_cpu bash -euo pipefail -c '
           tests/scripts/task_install_docs_dependencies.sh
           source tests/scripts/test_task_install_dist_wheels.sh
           tests/scripts/task_python_docs.sh
       '
   ```

   The sourced helper installs the wheels from `dist/` and sets `PYTHONPATH`
   for the docs script. Rebuild the wheels after changing TVM code;
   documentation-only edits can reuse them.

3. Serve the generated docs and visit http://localhost:8000 in your browser:

   ```bash
   python3 -m http.server --directory _docs
   ```

For the full documentation build, including GPU tutorials and non-Python APIs,
see [Helper Scripts](#helper-scripts) below.

### Native

1. [Build TVM](https://tvm.apache.org/docs/install/from_source.html) first in the repo root folder, then make it importable:

   ```bash
   export TVM_HOME=/path-to-tvm
   export TVM_LIBRARY_PATH=$TVM_HOME/build
   pip install --target=$TVM_HOME/python $TVM_HOME/3rdparty/tvm-tirx/3rdparty/tvm-ffi
   export PYTHONPATH="$TVM_HOME/3rdparty/tvm-tirx/python:$TVM_HOME/python:${PYTHONPATH:-}"
   export TVM_TIRX_LIBRARY_PATH="${TVM_HOME}/build/lib"
   ```

   `docs/conf.py` unconditionally imports `tvm` at startup, so the build will
   fail immediately if TVM is not importable.

2. Install dependencies

   ```bash
   # Pillow on Ubuntu may require libjpeg-dev from apt
   ./docker/bash.sh ci_gpu -c \
       'python3 -m pip install --quiet sphinx-book-theme==1.1.4 && python3 -m pip freeze' > frozen-requirements.txt

   pip install -r frozen-requirements.txt
   ```

3. Generate the docs

   ```bash
   # TVM_TUTORIAL_EXEC_PATTERN=none skips the tutorial execution to the build
   # work on most environments (e.g. MacOS).
   export TVM_TUTORIAL_EXEC_PATTERN=none

   cd docs
   make html
   ```

4. Run an HTTP server and visit http://localhost:8000 in your browser

   ```bash
   cd _build/html && python3 -m http.server
   ```

## Only Execute Specified Tutorials

The document build process will execute all the tutorials in the sphinx gallery.
This will cause failure in some cases when certain machines do not have necessary
environment. You can set `TVM_TUTORIAL_EXEC_PATTERN` to only execute
the path that matches the regular expression pattern.

After building GPU wheels as shown below, execute only tutorials under
`/get_started/tutorials` with:

```bash
docker/bash.sh --env IS_LOCAL=1 \
    --env PYTHON_DOCS_ONLY=1 --env TVM_TUTORIAL_EXEC_PATTERN=/get_started/tutorials \
    ci_gpu bash -euo pipefail -c '
        source tests/scripts/test_task_install_dist_wheels.sh
        tests/scripts/task_python_docs.sh
    '
```

To execute only one file, use a regular expression matching its name. Quote the
pattern so the shell preserves the backslash that escapes the dot:

```bash
docker/bash.sh --env IS_LOCAL=1 \
    --env PYTHON_DOCS_ONLY=1 --env 'TVM_TUTORIAL_EXEC_PATTERN=file_name\.py' \
    ci_gpu bash -euo pipefail -c '
        source tests/scripts/test_task_install_dist_wheels.sh
        tests/scripts/task_python_docs.sh
    '
```

## Helper Scripts

`tests/scripts/task_python_docs.sh` mirrors the CI docs pipeline. It runs a
Sphinx warning pre-check when `IS_LOCAL=0`, then builds the documentation with
`make htmldepoly`. `IS_LOCAL=1` skips the pre-check; the final build still checks
warnings. `PYTHON_DOCS_ONLY=1` skips the C++, Java, and TypeScript API docs for
local builds.

To run the full build, including tutorial execution and all API docs, use a
host with an NVIDIA GPU and Docker GPU support:

```bash
docker/bash.sh --no-gpu ci_gpu bash -euo pipefail -c '
    source ./tests/scripts/task_install_build_dependencies.sh
    ./tests/scripts/task_build_gpu.sh
    sccache --show-stats
'
docker/bash.sh --env IS_LOCAL=1 \
    --env PYTHON_DOCS_ONLY=0 --env TVM_TUTORIAL_EXEC_PATTERN=.py \
    ci_gpu bash -euo pipefail -c '
        source tests/scripts/test_task_install_dist_wheels.sh
        tests/scripts/task_python_docs.sh
    '
python3 -m http.server --directory _docs
```

Native builds use `build/` and wheels use `dist/`. Clean these directories
and run the matching platform build when switching between CPU and GPU
documentation builds.

## Define the Order of Tutorials

You can define the order of tutorials with `subsection_order` and
`within_subsection_order` in [`conf.py`](conf.py).
By default, the tutorials within one subsection are sorted by filename.

## Google Colab Integration

All the TVM tutorials can be opened and used interactively in Google Colab by
clicking the button at the top of the page. To do this, `sphinx-gallery` builds
`.ipynb` files from each tutorial, which are automatically deployed to the
[apache/tvm-site](https://github.com/apache/tvm-site/tree/asf-site) repo's
`asf-site` branch by [@tvm-bot](https://github.com/tvm-bot).

To make sure your tutorial runs correctly on Colab, any non-Python parts of
the tutorial (e.g. dependency installations) should be prefixed by an
[IPython magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html).
These will not be included in the built `HTML` file. For example, to install
Pytorch in your tutorial, add a ReStructured Text block like the following:

```python
######################################################################
# To run this tutorial, we must install PyTorch:
#
# .. code-block:: bash
#
#     %%shell
#     pip install torch
#
```

### Interactive Bash Scripts

In stock IPython, the `%%bash` magic command should be used to run shell
commands. However, this command does not give real-time output - the
tutorial's user will not see any output until the entire cell finishes
running. When running commands that take several minutes (e.g. installing
dependencies), this is annoying.

Luckily, Google Colab has the `%%shell` magic command that does the same
thing as `%%bash`, but gives output in real time. This command is specific
to Colab, and its [source code](https://github.com/googlecolab/colabtools)
is public. Thus, `%%shell` should be used instead of `%%bash` when writing
TVM tutorials.
