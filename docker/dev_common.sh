#!/bin/bash -e
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


if [ -z "${BASH_SOURCE[0]}" ]; then
    echo "NOTE: This script must be source'd from another bash script--it cannot be run directly"
    exit 2
fi

INVOCATION_PWD="$(pwd)"


GIT_TOPLEVEL=$(cd $(dirname ${BASH_SOURCE[0]}) && git rev-parse --show-toplevel)

DOCKER_IS_ROOTLESS=$(docker info 2> /dev/null | grep 'Context: \+rootless' || true)

function lookup_image_spec() {
    img_spec=$(python3 "${GIT_TOPLEVEL}/ci/jenkins/scripts/determine_docker_images.py" \
        --lookup-only "$1" 2>/dev/null)
    if [ -n "${img_spec}" ]; then
        has_similar_docker_image=1
        docker inspect "${1}" &>/dev/null || has_similar_docker_image=0
        if [ ${has_similar_docker_image} -ne 0 ]; then
            echo "WARNING: resolved docker image through Jenkinsfile to \"${img_spec}\"" >&2
        fi
        echo "${img_spec}"
    fi
}

function run_docker() {
    docker_bash_args=( )
    while [ "x${1:0:1}" == "x-" ]; do
        docker_bash_args=( "${docker_bash_args[@]}" "$1" )
        shift
    done
    image_name="$1"  # Name of the Jenkinsfile var to find
    shift

    docker_bash_args=( "${docker_bash_args[@]}" "${image_name}" "$@" )
    "${GIT_TOPLEVEL}/docker/bash.sh" "${docker_bash_args[@]}"
}
