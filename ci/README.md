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

# Apache TVM Continuous Integration (CI)

## Overview

TVM's Continuous Integration is responsible for verifying the code in `apache/tvm` and testing PRs
before they merge to inform TVM contributors and committers. These jobs are essential to keeping the
TVM project in a healthy state and preventing breakages. CI in TVM is broken into these pieces:
 - Lint scripts in [`tests/lint`](../tests/lint).
 - The tests themselves, all of which live underneath [`tests`](../tests).
 - Definitions of test suites, with each suite defined as a separate `task_` script in
   [`tests/scripts`](../tests/scripts).
 - GitHub automation [`ci/github`](github), wheel builds [`ci/wheel`](wheel),
   and Jenkins host commands [`ci/jenkins/scripts`](jenkins/scripts).
 - The linux test sequence (in [`Jenkinsfile`](../ci/jenkins/generated/)), which lints and builds TVM and runs test
   suites using Docker on Linux.
 - The Windows and Mac test sequences (in [`.github/actions`](../.github/actions)).
 - GitHub Actions that support the code review process (in [`.github/actions`](../.github/actions)).
 - Tools to reproduce the CI locally (in `tests/scripts`).
 - Infrastructure-as-Code that configures the cloud services that provide Jenkins for the TVM CI (in the
     [`tlc-pack/ci`](https://github.com/tlc-pack/ci) repo).

## CI Documentation Index

The CI documentation belongs with the implementation it describes. To make that concrete, the
documentation is split like so:
1. An overview of the CI is in this file.
1. User-facing documentation lives in `apache/tvm`'s `docs/contribute` sub-directory and is served on the
   [TVM docs site](https://tvm.apache.org/docs/contribute/ci.html).
2. Documentation of the tools that run TVM's various regression tests locally and the test suites
   are in this sub-directory.
3. Documentation of the cloud services and their configuration lives in the
   [`tlc-pack/ci`](https://github.com/tlc-pack/ci) repo.

## Jenkins

Jenkins runs all of the Linux-based TVM CI-enabled regression tests. This includes tests against accelerated hardware such as GPUs. It excludes those regression tests that run against hardware not available in the cloud (those tests aren't currently exercised in TVM CI). The tests run by Jenkins represent most of the merge-blocking tests (and passing Jenkins should mostly correlate with passing the remaining Windows/Mac builds).

## GitHub Actions

GitHub Actions is used to run Windows jobs, MacOS jobs, and various on-GitHub automations. These are defined in [`.github/workflows`](../.github/workflows/). These automations include bots to:
* [allow non-committers to merge approved / CI passing PRs](https://discuss.tvm.apache.org/t/rfc-allow-merging-via-pr-comments/12220)

https://github.com/apache/tvm/actions has the logs for each of these workflows. Note that when debugging these workflows changes from PRs from forked repositories won't be reflected in the PR. These should be tested in the forked repository first and linked in the PR body.

## Docker Images

Each CI job runs most of its work inside a Docker container, built from files
in the [`docker/`](../docker) folder. These
files are built nightly in Jenkins via the [tvm-docker](https://ci.tlcpack.ai/job/tvm-docker/) job.
The images for these containers are hosted in the [tlcpack Docker Hub](https://hub.docker.com/u/tlcpack)
and referenced in the [`jenkins/generated`](/ci/jenkins/generated/). These can be inspected and run
locally via standard Docker commands.

### `ci-docker-staging`

The [ci-docker-staging](https://github.com/apache/tvm/tree/ci-docker-staging)
branch is used to test updates to Docker images and `Jenkinsfile` changes. When
running a build for a normal PR from a forked repository, Jenkins uses the code
from the PR except for the `Jenkinsfile` itself, which comes from the base branch.
When branches are built, the `Jenkinsfile` in the branch is used, so a committer
with write access must push PRs to a branch in apache/tvm to properly test
`Jenkinsfile` changes. If your PR makes changes to the `Jenkinsfile`, make sure
to @ a [committer](/CONTRIBUTORS.md)
and ask them to push your PR as a branch to test the changes.

# Jenkins CI

TVM uses Jenkins for running Linux continuous integration (CI) tests on
[branches](https://ci.tlcpack.ai/job/tvm/) and
[pull requests](https://ci.tlcpack.ai/job/tvm/view/change-requests/) through a
build configuration specified in a [`Jenkinsfile`](/ci/jenkins/generated/).
Other jobs run in GitHub Actions for Windows and MacOS jobs.

## `Jenkinsfile`

The native Groovy entry points in `ci/jenkins/generated/` load shared helpers
from `common.groovy`. Edit these sources directly; no generation step is needed.
See comments in [common.groovy](jenkins/generated/common.groovy) for shared utility options and trust constraints.

The native Jenkins pipelines source
`tests/scripts/task_install_build_dependencies.sh` once to clean `dist/` and
install build backend and Z3 prerequisites. They then invoke
`tests/scripts/task_build_<platform>.sh`. Each root platform script sets the full
platform CMake options and directly builds the pinned-source FFI wheel from
`3rdparty/tvm-tirx/3rdparty/tvm-ffi`, followed by the TIRx and TVM wheels. All three wheel
builds use the same root CMake source and absolute `build/` directory, with fixed
package-specific install components. They perform normal CMake configuration
passes over the shared native build graph. The root script also runs enabled
C++ tests, and all three wheels are written to `dist/`.

Before running a consumer script, Groovy sources
`tests/scripts/test_task_install_dist_wheels.sh` in the same shell. This installs
the three transferred wheels offline without dependencies into `.local/python`
and sets its absolute path as `PYTHONPATH`; `TVM_HOME` retains checkout resources.
Test and docs scripts use that caller-provided environment. Groovy sets the
cache, parallelism and C/C++ compiler-launcher environment values;
`docker/bash.sh` forwards them to the build scripts.
