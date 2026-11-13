#!/usr/bin/env bash
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

# NOTE: allow unbound variable here
set +u

if [[ ! -z $CI_PYTEST_ADD_OPTIONS ]]; then
    export PYTEST_ADDOPTS="-s -vv $CI_PYTEST_ADD_OPTIONS $PYTEST_ADDOPTS"
else
    export PYTEST_ADDOPTS="-s -vv $PYTEST_ADDOPTS"
fi
set -ux

export TVM_PATH=`pwd`
export PYTHONPATH="${TVM_PATH}/3rdparty/tvm-tirx/python:${TVM_PATH}/python"
export TVM_TIRX_LIBRARY_PATH="${TVM_PATH}/build/lib"

export TVM_PYTEST_RESULT_DIR="${TVM_PATH}/build/pytest-results"
mkdir -p "${TVM_PYTEST_RESULT_DIR}"
pytest_errors=()

# This ensures that all pytest invocations that are run through run_pytest will
# complete and errors will be reported once Bash is done executing all scripts.
function cleanup() {
    set +x
    if [ "${#pytest_errors[@]}" -gt 0 ]; then
        echo "These pytest invocations failed, the results can be found in the Jenkins 'Tests' tab or by scrolling up through the raw logs here."
        printf '%s\n' "${pytest_errors[@]}"
        exit 1
    fi
    set -x
}
trap cleanup 0

function run_pytest() {
    ffi_type="cython"
    set -e
    local test_suite_name="$1"
    shift
    extra_args=( "$@" )
    if [ -z "${ffi_type}" -o -z "${test_suite_name}" ]; then
        echo "error: run_pytest called incorrectly: run_pytest ${test_suite_name}" "${extra_args[@]}"
        echo "usage: run_pytest <TEST_SUITE_NAME> [pytest args...]"
        exit 2
    fi

    # Allow unbound variable here.
    set +u
    if [[ -z "${TVM_SHARD_INDEX}" ]]; then
      current_shard="no-shard"
    else
      current_shard="shard-${TVM_SHARD_INDEX}"
    fi
    set -u

    has_reruns=$(python3 -m pytest --help 2>&1 | grep 'reruns=' || true)
    if [ -n "$has_reruns" ]; then
        if [[ ! "${extra_args[*]}" == *"--reruns"* ]]; then
          extra_args+=('--reruns=3')
        fi
    fi

    suite_name="${test_suite_name}-${current_shard}-${ffi_type}"

    DEFAULT_PARALLELISM=1

    if [[ ! "${extra_args[*]}" == *" -n"* ]] && [[ ! "${extra_args[*]}" == *" -dist"* ]]; then
        extra_args+=("-n=$DEFAULT_PARALLELISM")
    fi

    exit_code=0
    set +e
    python3 -m pytest \
           -o "junit_suite_name=${suite_name}" \
           "--junit-xml=${TVM_PYTEST_RESULT_DIR}/${suite_name}.xml" \
           "--junit-prefix=${ffi_type}" \
           "${extra_args[@]}" || exit_code=$?
    # Pytest will return error code -5 if no test is collected.
    if [ "$exit_code" -ne "0" ] && [ "$exit_code" -ne "5" ]; then
        pytest_errors+=("${suite_name}: $@")
    fi
    # To avoid overwriting.
    set -e
}
