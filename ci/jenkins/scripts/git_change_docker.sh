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

set -eux

if [ "${BRANCH_NAME}" == "main" ]; then
    changed_files=$(git diff --no-commit-id --name-only -r HEAD~1)
else
    changed_files=$(git diff --no-commit-id --name-only -r origin/main)
fi

FILES_THAT_SHOULDNT_TRIGGER_REBUILDS=(
    "docker/bash.sh"
    "docker/with_the_same_user"
    "docker/README.md"
    "docker/Dockerfile.ci_lint"
    "docker/install/ubuntu_install_clang_format.sh"
    "docker/lint.sh"
    "docker/clear-stale-images.sh"
    "README.md"
    "lint.sh"
    "clear-stale-images.sh"
)

for file in $changed_files; do
    # Certain files under docker/ don't matter for rebuilds, so ignore them
    if printf '%s\0' "${FILES_THAT_SHOULDNT_TRIGGER_REBUILDS[@]}" | grep -F -x -z -- "$file"; then
        echo "Skipping $file"
        continue
    fi
    # if grep -q "docker/"
    echo "Checking $file"
    if grep -q "docker/" <<< "$file"; then
        exit 1
    fi
done

# No docker changes
exit 0
