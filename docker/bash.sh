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

#
# Start a bash, mount REPO_MOUNT_POINT to be current directory.
#
# Usage: docker/bash.sh [-i|--interactive] [--net=host] [-t|--tty]
#          [--shell SHELL] [--mount MOUNT_DIR]
#          [--repo-mount-point REPO_MOUNT_POINT]
#          [--dry-run] [--name NAME] [--privileged]
#          <DOCKER_IMAGE_NAME> [--] [COMMAND]
#
# Usage: docker/bash.sh <CONTAINER_NAME>
#     Starts an interactive session
#
# Usage2: docker/bash.sh [-i] <CONTAINER_NAME> [COMMAND]
#     Execute command in the docker image, default non-interactive
#     With -i, execute interactively.
#

set -euo pipefail

##############################################
### Section 1: Constants and defaults      ###
##############################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

DRY_RUN=false
INTERACTIVE=false
TTY=false
USE_NET_HOST=false
USE_GPU=true
DOCKER_IMAGE_NAME=
COMMAND=bash
MOUNT_DIRS=( )
CONTAINER_NAME=
USER_SHELL=

# TODO(Lunderberg): Remove this if statement and always set to
# "${REPO_DIR}".  The consistent directory for Jenkins is currently
# necessary to allow CMake build commands to run in CI after the build
# steps.
# TODO(https://github.com/apache/tvm/issues/11952):
# Figure out a better way to keep the same path
# between build and testing stages.
if [[ -n "${JENKINS_HOME:-}" ]]; then
    REPO_MOUNT_POINT=/workspace
else
    REPO_MOUNT_POINT="${REPO_DIR}"
fi

##############################################
### Section 2: Helper functions            ###
##############################################

function show_usage() {
    cat <<EOF
Usage: docker/bash.sh [-i|--interactive] [--net=host] [-t|--tty]
          [--shell SHELL] [--cpus NUM_CPUS] [--mount MOUNT_DIR]
          [--repo-mount-point REPO_MOUNT_POINT]
          [--dry-run] [--name NAME]
          <DOCKER_IMAGE_NAME> [--] [COMMAND]


-h, --help

    Display this help message.

-i, --interactive

    Start the docker session in interactive mode.

-t, --tty

    Start the docker session with a pseudo terminal (tty).

--shell SHELL

    Specify a shell (e.g. --shell zsh) to use instead of bash
    when entering auto-interactive mode (no command or command
    is "bash").

--cpus NUM_CPUS

    Limit the number of CPU cores to be used.

--net=host

    Expose servers run into the container to the host, passing the
    "--net=host" argument through to docker.  On MacOS, this is
    instead passed as "-p 8888:8888" since the host networking driver
    isn't supported.

--mount MOUNT_DIR

    Expose MOUNT_DIR as an additional mount point inside the docker
    container.  The mount point inside the container is the same as
    the folder location outside the container.  This option can be
    specified multiple times.

--repo-mount-point REPO_MOUNT_POINT

    The directory inside the docker container at which the TVM
    repository should be mounted, and is used as the workspace inside
    the docker container.

    If unspecified, the mount location depends on the environment.  If
    running inside Jenkins, the mount location will be /workspace.
    Otherwise, the mount location of the repository will be the same
    as the external location of the repository, to maintain
    compatibility with git-worktree.

--no-gpu

    Do not use GPU device drivers even if using an CUDA Docker image

--dry-run

    Print the docker command to be run, but do not execute it.

--build-dir BUILD_DIR

    The build directory of TVM. This is appended to LD_LIBRARY_PATH

--env

    Pass an environment variable through to the container.

--name

    Set the name of the docker container, and the hostname that will
    appear inside the container.

--privileged

    Give extended privileges to this container.

DOCKER_IMAGE_NAME

    The name of the docker container to be run.  This can be an
    explicit name of a docker image (e.g. "tlcpack/ci-gpu:v0.76") or
    can be a shortcut as defined in the TVM Jenkinsfile
    (e.g. "ci_gpu").

COMMAND

    The command to be run inside the docker container.  If this is set
    to "bash", the --interactive, --tty and --net=host flags are set.
    If no command is specified, defaults to "bash".  If the command
    contains dash-prefixed arguments, the command should be preceded
    by -- to indicate arguments that are not intended for bash.sh.

Environment Variables:

TVM_DEV_DOCKER_MOUNTS

    Space-separated list of additional mount specifications.
    Each entry is either:
      /path/to/dir          - bind mount at same path inside container
      /src:/dst             - bind mount src to dst inside container
    Source paths are validated; missing sources are skipped with a warning.

EOF
}

function parse_error() {
    echo "$@" >&2
    show_usage >&2
    exit 1
}

# Detect if Docker daemon is running in rootless mode.
# Uses DOCKER_IS_ROOTLESS env var as override (for CI), otherwise auto-detects.
# Result is cached after first call.
_ROOTLESS_CACHED=
_ROOTLESS_RESULT=1
function detect_rootless() {
    if [ -n "${_ROOTLESS_CACHED}" ]; then
        return $_ROOTLESS_RESULT
    fi
    _ROOTLESS_CACHED=1
    if [ -n "${DOCKER_IS_ROOTLESS:-}" ]; then
        # env var override for CI
        _ROOTLESS_RESULT=0
        return 0
    fi
    if docker info -f '{{.SecurityOptions}}' 2>/dev/null | grep -q "rootless"; then
        _ROOTLESS_RESULT=0
        return 0
    fi
    _ROOTLESS_RESULT=1
    return 1
}

# Validate a mount source path exists, warn if missing.
# Usage: validate_mount /path/to/source
# Returns 0 if source exists, 1 if missing.
function validate_mount() {
    local src="$1"
    if [ -e "$src" ]; then
        return 0
    else
        echo "Warning: Mount source '$src' not found on host, skipping." >&2
        return 1
    fi
}


##############################################
### Section 3: Argument parsing            ###
##############################################

# Handle joined flags, such as interpreting -ih as -i -h.  Either rewrites
# the current argument if it is a joined argument, or shifts all arguments
# otherwise.  Should be called as "eval $break_joined_flag" where joined
# flags are possible.  Can't use a function definition, because it needs
# to overwrite the parent scope's behavior.
break_joined_flag='if (( ${#1} == 2 )); then shift; else set -- -"${1#-i}" "${@:2}"; fi'

DOCKER_ENV=( )
# Forward build settings selected by the caller without overriding explicit values.
BUILD_ENV_VARS=(SCCACHE_BUCKET SCCACHE_REGION CMAKE_BUILD_PARALLEL_LEVEL
                CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER)
for build_var in "${BUILD_ENV_VARS[@]}"; do
    if [ "${!build_var+x}" = "x" ]; then
        DOCKER_ENV+=( --env "$build_var=${!build_var}" )
    fi
done
DOCKER_FLAGS=( )

while (( $# )); do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;

        -i*|--interactive)
            INTERACTIVE=true
            eval $break_joined_flag
            ;;

        -t*|--tty)
            TTY=true
            if [[ "$1" == "--tty" ]]; then
                shift
            else
                # Short form: -t or combined like -th
                eval $break_joined_flag
            fi
            ;;

        --shell)
            if (( $# >= 2 )) && [[ -n "$2" ]]; then
                USER_SHELL="$2"
                shift 2
            else
                parse_error 'ERROR: --shell requires a non-empty argument'
            fi
            ;;

        --cpus)
            DOCKER_FLAGS+=(--cpus "$2")
            shift 2
            ;;

        --net=host)
            USE_NET_HOST=true
            shift
            ;;

        --net)
            DOCKER_FLAGS+=( --net "$2" )
            shift 2
            ;;

        --mount)
            if [[ -n "$2" ]]; then
                MOUNT_DIRS+=("$2")
                shift 2
            else
                parse_error 'ERROR: --mount requires a non-empty argument'
            fi
            ;;

        --mount=?*)
            MOUNT_DIRS+=("${1#*=}")
            shift
            ;;

        --name)
            if [[ -n "$2" ]]; then
                CONTAINER_NAME="$2"
                shift 2
            else
                parse_error 'ERROR: --name requires a non empty argument'
            fi
            ;;

        --privileged)
            DOCKER_FLAGS+=( "--privileged" )
            shift 1
            ;;

        --env)
            DOCKER_ENV+=( --env "$2" )
            shift 2
            ;;

        --volume)
            DOCKER_FLAGS+=( --volume "$2" )
            shift 2
            ;;

        -e)
            DOCKER_ENV+=( --env "$2" )
            shift 2
            ;;

        -v)
            DOCKER_FLAGS+=( --volume "$2" )
            shift 2
            ;;

        --dry-run)
            DRY_RUN=true
            shift
            ;;

        --no-gpu)
            USE_GPU=false
            shift
            ;;

        --repo-mount-point)
            if [[ -n "$2" ]]; then
                REPO_MOUNT_POINT="$2"
                shift 2
            else
                parse_error 'ERROR: --repo-mount-point requires a non-empty argument'
            fi
            ;;

        --repo-mount-point=?*)
            REPO_MOUNT_POINT="${1#*=}"
            shift
            ;;

        --build-dir)
            DOCKER_ENV+=( --env LD_LIBRARY_PATH=${REPO_MOUNT_POINT}/${2})
            shift 2
            ;;

        --)
            shift
            COMMAND=( "$@" )
            break
            ;;

        -*|--*)
            echo "Error: Unknown flag: $1" >&2
            echo "  If this flag is intended to be passed to the" >&2
            echo "  docker command, please add -- before the docker" >&2
            echo "  command (e.g. docker/bash.sh ci_gpu -- build -j2)" >&2
            show_usage >&2
            exit 1
            ;;

        *)
            # First positional argument is the image name, all
            # remaining below to the COMMAND.
            if [[ -z "${DOCKER_IMAGE_NAME}" ]]; then
                DOCKER_IMAGE_NAME=$1
                shift
            else
                COMMAND=( "$@" )
                break
            fi
            ;;
    esac
done

if [[ -z "${DOCKER_IMAGE_NAME}" ]]; then
    echo "Error: Missing DOCKER_IMAGE_NAME" >&2
    show_usage >&2
fi

# Smart defaults: "bash" command triggers interactive mode
if [[ ${COMMAND[@]+"${COMMAND[@]}"} = bash ]]; then
    INTERACTIVE=true
    TTY=true
fi


##############################################
### Section 4: Image resolution            ###
##############################################

# Save user-set DOCKER_IS_ROOTLESS before dev_common.sh overwrites it.
# detect_rootless() should only see the user's explicit env var, not
# the auto-detection done by dev_common.sh.
_USER_ROOTLESS="${DOCKER_IS_ROOTLESS+__set__}"
_USER_ROOTLESS_VAL="${DOCKER_IS_ROOTLESS:-}"

source "$(dirname $0)/dev_common.sh" || exit 2

# Restore: if user had set DOCKER_IS_ROOTLESS, keep their value;
# otherwise unset the one dev_common.sh created.
if [ "$_USER_ROOTLESS" = "__set__" ]; then
    DOCKER_IS_ROOTLESS="$_USER_ROOTLESS_VAL"
else
    unset DOCKER_IS_ROOTLESS
fi
unset _USER_ROOTLESS _USER_ROOTLESS_VAL

DOCKER_MOUNT=( )
DOCKER_DEVICES=( )
# If the user gave a shortcut defined in the Jenkinsfile, use it.
EXPANDED_SHORTCUT=$(lookup_image_spec "${DOCKER_IMAGE_NAME}")
if [ -n "${EXPANDED_SHORTCUT}" ]; then
    if [ "${CI+x}" == "x" ]; then
        DOCKER_IMAGE_NAME="${EXPANDED_SHORTCUT}"
    else
        python3 ci/jenkins/scripts/determine_docker_images.py "$DOCKER_IMAGE_NAME" 2> /dev/null
        DOCKER_IMAGE_NAME=$(cat ".docker-image-names/$DOCKER_IMAGE_NAME")
        if [[ "$DOCKER_IMAGE_NAME" == *"tlcpackstaging"* ]]; then
            echo "WARNING: resolved docker image to fallback tag in tlcpackstaging" >&2
        fi
    fi
fi


##############################################
### Section 5: Mount construction          ###
##############################################

# Working directories
DOCKER_FLAGS+=( --workdir "${REPO_MOUNT_POINT}" )
DOCKER_MOUNT+=( --volume "${REPO_DIR}":"${REPO_MOUNT_POINT}"
                --volume "${SCRIPT_DIR}":/docker
              )

# Expose external directories from --mount flags
for MOUNT_DIR in ${MOUNT_DIRS[@]+"${MOUNT_DIRS[@]}"}; do
    DOCKER_MOUNT+=( --volume "${MOUNT_DIR}:${MOUNT_DIR}" )
done

# Parse TVM_DEV_DOCKER_MOUNTS env var: space-separated mount specs
# Each entry is either /path (same path) or /src:/dst
if [[ -n "${TVM_DEV_DOCKER_MOUNTS:-}" ]]; then
    for mount_spec in ${TVM_DEV_DOCKER_MOUNTS}; do
        if [[ "$mount_spec" == *:* ]]; then
            # src:dst format
            mount_src="${mount_spec%%:*}"
            if validate_mount "$mount_src"; then
                DOCKER_MOUNT+=( --volume "$mount_spec" )
            fi
        else
            # Same path format
            if validate_mount "$mount_spec"; then
                DOCKER_MOUNT+=( --volume "$mount_spec:$mount_spec" )
            fi
        fi
    done
fi

# When running from a git worktree, also mount the original git dir.
if [ -f "${REPO_DIR}/.git" ]; then
    git_dir=$(cd ${REPO_DIR} && git rev-parse --git-common-dir)
    if [ "${git_dir}" != "${REPO_DIR}/.git" ]; then
        DOCKER_MOUNT+=( --volume "${git_dir}:${git_dir}" )
    fi
fi


##############################################
### Section 6: Environment construction    ###
##############################################

# CI-specific environment variables
DOCKER_ENV+=( --env CI_BUILD_HOME="${REPO_MOUNT_POINT}"
              --env CI_BUILD_USER="$(id -u -n)"
              --env CI_BUILD_UID="$(id -u)"
              --env CI_BUILD_GROUP="$(id -g -n)"
              --env CI_BUILD_GID="$(id -g)"
              --env CI_PYTEST_ADD_OPTIONS="${CI_PYTEST_ADD_OPTIONS:-}"
              --env CI_IMAGE_NAME="${DOCKER_IMAGE_NAME}"
            )

# Remove the container once it finishes running (--rm).
DOCKER_FLAGS+=(--rm)

# Share the PID namespace (--pid=host).  The process inside does not
# have pid 1 and SIGKILL is propagated to the process inside, allowing
# jenkins to kill it if needed.  This is only necessary for docker
# daemons running as root.
if ! detect_rootless; then
    DOCKER_FLAGS+=(--pid=host)
fi

# Expose services running in container to the host.
if $USE_NET_HOST; then
    if [[ $(uname) == "Darwin" ]]; then
        # Docker's host networking driver isn't supported on macOS.
        # Use default bridge network and expose port for jupyter notebook.
        DOCKER_FLAGS+=( -p 8888:8888 )
    else
        DOCKER_FLAGS+=(--net=host)
    fi
fi

# Set up interactive sessions
if ${INTERACTIVE}; then
    DOCKER_FLAGS+=( --interactive )
fi

if ${TTY}; then
    DOCKER_FLAGS+=( --tty )
fi

# Setup the docker name and the hostname inside the container
if [[ ! -z "${CONTAINER_NAME}" ]]; then
    DOCKER_FLAGS+=( --name ${CONTAINER_NAME} --hostname ${CONTAINER_NAME})
fi

# Pass any restrictions of allowed CUDA devices from the host to the
# docker container.
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    DOCKER_ENV+=( --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" )
fi

# Set TVM import path inside the docker image
if [[ "${DOCKER_IMAGE_NAME}" == *"ci"* ]]; then
    DOCKER_ENV+=( --env PYTHONPATH="${REPO_MOUNT_POINT}/3rdparty/tvm-tirx/python:${REPO_MOUNT_POINT}/python" )
    DOCKER_ENV+=( --env TVM_TIRX_LIBRARY_PATH="${REPO_MOUNT_POINT}/build/lib" )
fi


##############################################
### Section 7: GPU/device setup            ###
##############################################

DOCKER_BINARY=docker

# Use --gpus all for GPU/CUDA containers
if [[ "$USE_GPU" == "true" ]] && [[ "${DOCKER_IMAGE_NAME}" == *"gpu"* || "${DOCKER_IMAGE_NAME}" == *"cuda"* ]]; then
    DOCKER_FLAGS+=( --gpus all )
    DOCKER_ENV+=( --env NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility )
fi

# Add ROCm devices and set ROCM_ENABLED=1 which is used in the with_the_same_user script
# to add the user to the video group
if [[ "${DOCKER_IMAGE_NAME}" == *"rocm"* && -d "/dev/dri" ]]; then
    DOCKER_DEVICES+=( --device=/dev/kfd --device=/dev/dri )
    DOCKER_ENV+=( --env ROCM_ENABLED=1 )
fi


##############################################
### Section 8: Entry command               ###
##############################################

# Apply --shell override: when command is "bash" (auto-interactive)
# and a custom shell was requested via --shell <name>, use that shell
if [[ ${COMMAND[@]+"${COMMAND[@]}"} = bash ]] && [[ -n "$USER_SHELL" ]]; then
    COMMAND=( "$USER_SHELL" )
fi

# Build the entry command
if ! detect_rootless; then
    # If the docker daemon is running as root, use the TVM-provided
    # "with_the_same_user" script to update the PID.  When using rootless
    # docker, this step is unnecessary.
    COMMAND=(
        bash --login /docker/with_the_same_user
        ${COMMAND[@]+"${COMMAND[@]}"}
    )
fi


##############################################
### Section 9: Final docker run            ###
##############################################

# Print arguments.
echo "REPO_DIR: ${REPO_DIR}"
echo "DOCKER CONTAINER NAME: ${DOCKER_IMAGE_NAME}"
echo ""

echo Running \'${COMMAND[@]+"${COMMAND[@]}"}\' inside ${DOCKER_IMAGE_NAME}...

DOCKER_CMD=(${DOCKER_BINARY} run
            ${DOCKER_FLAGS[@]+"${DOCKER_FLAGS[@]}"}
            ${DOCKER_ENV[@]+"${DOCKER_ENV[@]}"}
            ${DOCKER_MOUNT[@]+"${DOCKER_MOUNT[@]}"}
            ${DOCKER_DEVICES[@]+"${DOCKER_DEVICES[@]}"}
            "${DOCKER_IMAGE_NAME}"
            ${COMMAND[@]+"${COMMAND[@]}"}
           )

if ${DRY_RUN}; then
    echo ${DOCKER_CMD[@]+"${DOCKER_CMD[@]}"}
else
    ${DOCKER_CMD[@]+"${DOCKER_CMD[@]}"}
fi
