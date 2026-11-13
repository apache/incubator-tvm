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

Submit a Pull Request
=====================

.. contents::
  :depth: 2
  :local:

Guidelines
----------

- We recommend authors send well scoped PRs that are easy to review and revert in case there is a problem. As such, authors should avoid merging multiple unrelated changes into a single PR
- Before you submit a PR, please rebase your code on the most recent version of ``main``, you can do it by
  running

  .. code:: bash

    git remote add upstream [url to tvm repo]
    git fetch upstream
    git rebase upstream/main

- Make sure code passes lint checks

    .. code:: bash

      # Run all lint checks via pre-commit hooks
      pre-commit run --all-files

      # Run specific linters individually
      pre-commit run ruff-check --all-files    # Python lint
      pre-commit run ruff-format --all-files   # Python format
      pre-commit run clang-format --all-files  # C++ format

- Add test-cases to cover the new features or bugfix the patch introduces.
- Document the code you wrote, see more at :ref:`doc_guide`
- `Create a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_ and fix the problems reported by CI checks.
- Request code reviews from other contributors and improve your patch according
  to their reviews by ``@``-ing them in your pull request. Use relevant topic
  tags in PR titles to make the affected area clear (e.g. ``[microTVM] Add a
  cool change`` and not ``a cool change for microTVM``).
  Please see the Commit Message Guideline below on the guidelines about the tags
  in a PR/commit title and how to write good PR/commit messages.

  - To get your code reviewed quickly, we encourage you to help review others' code so they can do the favor in return.
  - Code review is a shepherding process that helps to improve contributor's code quality.
    We should treat it proactively, to improve the code as much as possible before the review.
    We highly value patches that can get in without extensive reviews.
  - The detailed guidelines and summarizes useful lessons.

- The PR can be merged after the reviewers approve the pull request.

Commit Message Guideline
------------------------

Apache TVM uses the GitHub (GH) platform for patch submission and code review
via Pull Requests (PRs). The final commit (title and body) that is merged into
the Apache TVM main tree is composed of the PR's title and body and must be kept
updated and reflecting the new changes in the code as per the reviews and
discussions.

Although these guidelines apply essentially to the PRs’ title and body messages,
because GH auto-generates the PR’s title and body from the commits on a given
branch, it’s recommended to follow these guidelines right from the beginning,
when preparing commits in general to be submitted to the Apache TVM project.
This will ease the creation of a new PR, avoiding rework, and also will help the
review.

The rules below will help to achieve uniformity that has several benefits, both
for review and for the code base maintenance as a whole, helping you to write
commit messages with a good quality suitable for the Apache TVM project,
allowing fast log searches, bisecting, and so on.

*PR/commit title*:

 - Guarantee a title exists (enforced);
 - Don’t use GitHub usernames in the title, like @username (enforced);
 - A tag must be present as a hint about what component(s) of the code
   the PRs / commits “touch” (enforced). For example [BugFix], [CI], [microTVM],
   and [TVMC]. Tags go between square brackets and appear first in the title. If
   more than one tag exist, multiple brackets should be used, like [BugFix][CI].
   The case recommended for tags, in geral, is the upper camel case. For example,
   prefer the forms [Fix], [BugFix], and [Docker] instead of [fix], [bug_fix],
   and [docker]. Acronyms should be kept as such so, for example, use [CI] and
   [TVMC] instead of [ci] and [tvmc]. Tags help reviewers to identify the PRs
   they can/want to review and also help the release folks when generating the
   release notes;
 - Use an imperative mood. Avoid titles like “Added operator X” and “Updated
   image Y in the CI”, instead use the forms “Add feature X” and “Update image Y
   in the CI” instead;
 - Observe proper use of caps at the beginning (uppercase for the first letter)
   and for acronyms, like, for instance, TVM, FVP, OpenCL. Hence instead of
   “fix tvm use of opencl library”, write it as “Fix TVM use of OpenCL library”;
 - Do not put a period at the end of the title.

*PR/commit body*:

 - Guarantee a body exists (enforced);
 - Don’t use GitHub usernames in body text, like @username (enforced);
 - Avoid “bullet” commit message bodies: “bullet” commit message bodies are not
   bad per se, but “bullet” commit messages without any description or
   explanation is likely as bad as commits without any description, rationale,
   or explanation in the body.

For minor deviations from these guidelines, the community will normally favor
reminding the contributor of this policy over reverting or blocking a commit /
PR.

Commits and PRs without a title and/or a body are not considered minor
deviations from these guidelines and hence must be avoided.

Most importantly, the contents of the commit message, especially the body,
should be written to convey the intention of the change, so it should avoid
being vague. For example, commits with a title like “Fix”, “Cleanup”, and
“Fix flaky test” and without any body text should be avoided. Also, for the
review, it will leave the reviewer wondering about what exactly was fixed or
changed and why the change is necessary, slowing the review.

Below is an example that can be used as a model:

::

 [microTVM] Zephyr: Remove zephyr_board option from build, flash, and open_transport methods

 Currently it’s necessary to pass the board type via ‘zephyr_board’ option to
 the Project API build, flash, and open_transport methods.

 However, since the board type is already configured when the project is
 created (i.e. when the generate_project method is called), it’s possible to
 avoid this redundancy by obtaining the board type from the project
 configuration files.

 This commit adds code to obtain the board type from the project CMake files,
 removing this option from build, flash, and open_transport methods, so it’s
 only necessary to specify the ‘zephyr_board’ option when calling
 generate_project.

 This commit also moves the ‘verbose’ and ‘west_cmd’ options from ‘build’
 method to ‘generate_project’, reducing further the number of required options
 when building a project, since the ‘build’ method is usually called more often
 than the ‘generate_project’.

After a new PR is created and the review starts it’s common that reviewers will
request changes. Usually the author will address the reviewers’ comments and
push additional commits on top of the initial ones. For these additional commits
there is no recommendation regarding the commit messages. However if the
additional commits render the PR title and/or body outdated then it's the
author's responsibility to keep the PR title and body in sync with new changes
in the code and updated the PR title and body accordingly (remember that the PR
title and body will be used to compose the final commit message that will land
in the main tree).

Committers will seek to fix any issues with the commit message prior to
committing but they retain the right to inform the author of the rules and
encourage them to follow them in future. Also, they retain the right to ask to
the author to update the PR title and/or body when they are not correctly
updated or fixed.

CI Environment
--------------
We use Docker images to create stable CI environments that can be deployed to multiple machines.
Follow the steps in `this issue template <https://github.com/apache/tvm/issues/new?assignees=&labels=&template=ci-image.md&title=%5BCI+Image%5D+>`_
to update a CI Docker image.

.. _pr-testing:

Testing
-------
Even though we have hooks to run unit tests automatically for each pull request, it's always recommended to run unit tests
locally beforehand to reduce reviewers' burden and speedup review process.

Docker (recommended)
^^^^^^^^^^^^^^^^^^^^
Use ``docker/bash.sh`` with the same platform build and test scripts used in CI.
Run these commands from the repository root. The CPU and GPU Docker images are
large and may take some time to download on their first use.

.. code:: bash

    # show Docker options
    docker/bash.sh --help

    # build FFI, TIRx, and TVM wheels and run CPU C++ tests
    # from the shared native build
    docker/bash.sh ci_cpu bash -euo pipefail -c '
        source ./tests/scripts/task_install_build_dependencies.sh
        ./tests/scripts/task_build_cpu.sh
        sccache --show-stats
    '

    # install the wheels from dist/ and run Python unit tests
    docker/bash.sh ci_cpu bash -euo pipefail -c '
        source tests/scripts/test_task_install_dist_wheels.sh
        tests/scripts/task_python_unittest.sh
    '

    # reuse the wheels to run a selected test
    docker/bash.sh ci_cpu bash -euo pipefail -c '
        source tests/scripts/test_task_install_dist_wheels.sh
        tests/scripts/task_python_unittest.sh \
            tests/python/s_tir/schedule/test_tir_schedule_rolling_buffer.py::test_upscale
    '

    # enter an interactive container and install the wheels for that shell
    docker/bash.sh ci_cpu
    source tests/scripts/test_task_install_dist_wheels.sh
    python3 -m pytest tests/python/s_tir/schedule/test_tir_schedule_rolling_buffer.py
    exit

Rebuild the wheels after changing TVM code. For GPU tests, use a host with an
NVIDIA GPU and Docker GPU support:

.. code:: bash

    docker/bash.sh --no-gpu ci_gpu bash -euo pipefail -c '
        source ./tests/scripts/task_install_build_dependencies.sh
        ./tests/scripts/task_build_gpu.sh
        sccache --show-stats
    '
    docker/bash.sh ci_gpu bash -euo pipefail -c '
        source tests/scripts/test_task_install_dist_wheels.sh
        tests/scripts/task_python_unittest_gpuonly.sh
    '

Native builds use ``build/`` and wheels use ``dist/``. Clean these directories
and run the matching platform build when switching test environments.
For lint checks, use ``pre-commit run --files <changed-file>...``.

We regularly update our docker images and, over time, stale images may unnecessarily consume disk
space. You can remove stale images that aren't used in the presently checked-out branch plus any
other worktrees using the following command:

.. code:: bash

    docker/clear-stale-images.sh

Consult the ``--help`` for more options.

C++ (local)
^^^^^^^^^^^

Running the C++ tests requires installation of gtest, following the instructions in
:ref:`install-from-source-cpp-tests`


.. code:: bash

  # assume you are in tvm source root
  TVM_ROOT=`pwd`

  ./tests/scripts/task_cpp_unittest.sh

Python (local)
^^^^^^^^^^^^^^
Necessary dependencies:

.. code:: bash

  pip install --user pytest pytest-xdist Cython

If you want to run all tests:

.. code:: bash

  # build tvm (see install-from-source for CMake build instructions)
  cd build && cmake .. && cmake --build . --parallel $(nproc) && cd ..

  python -m pytest -vvs -n auto tests/python

If you want to run a single test:

.. code:: bash

  # let python know where to find tvm related libraries
  export PYTHONPATH="$PWD/3rdparty/tvm-tirx/python:$PWD/python:${PYTHONPATH:-}"
  export TVM_TIRX_LIBRARY_PATH="${PWD}/build/lib"
  rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc

  python -m pytest -v 3rdparty/tvm-tirx/tests/python/tirx/transform/test_storage_rewrite.py

  # Additionally if you want to run a single test, for example test_add_one_2d inside a file.
  python -m pytest -v -k "test_add_one_2d" tests/python/relax/test_frontend_tflite.py
