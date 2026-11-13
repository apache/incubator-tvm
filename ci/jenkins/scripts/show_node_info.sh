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

set -euxo pipefail

echo "===== JENKINS INFO ====="
echo "NODE_NAME=$NODE_NAME"
echo "EXECUTOR_NUMBER=$EXECUTOR_NUMBER"
echo "WORKSPACE=$WORKSPACE"
echo "BUILD_NUMBER=$BUILD_NUMBER"
echo "WORKSPACE=$WORKSPACE"

echo "===== EC2 INFO ====="
function ec2_metadata() {
    # See https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
    curl -w '\n' -fsSL "http://169.254.169.254/latest/meta-data/$1" || echo failed
}

ec2_metadata ami-id
ec2_metadata instance-id
ec2_metadata instance-type
ec2_metadata hostname
ec2_metadata public-hostname

echo "===== RUNNER INFO ====="
df --human-readable
lscpu
free
nvidia-smi 2>/dev/null || echo "CUDA not found"
