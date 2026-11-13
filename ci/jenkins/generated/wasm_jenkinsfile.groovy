// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// Reject untrusted helper edits before allocating nodes or running pipeline steps.
def helper_source
try {
  helper_source = readTrusted('ci/jenkins/generated/common.groovy')
} catch (Throwable ex) {
  echo 'Jenkinsfile changes must be pushed to an upstream branch. ' +
    'wasm/pr-head will fail for these changes; check wasm/branch status instead.'
  throw ex
}
ci = evaluate(helper_source)
ci.start()

stage('Build') {
  ci.with_node(instance: 'CPU-SPOT', retry_instance: 'CPU',
    name: 'BUILD: WASM', folder: 'tvm/build-wasm', variables: ['PLATFORM=wasm']) {
    ci.docker_init(images.ci_wasm)
    withEnv(ci.build_environment()) {
      sh(
        script: """${docker_run} ${images.ci_wasm} bash -euo pipefail -c '
          source ./tests/scripts/task_install_build_dependencies.sh
          ./tests/scripts/task_build_wasm.sh
        '""",
        label: 'Build wasm wheels and C++ tests',
      )
    }
    sh(
      script: """${docker_run} ${images.ci_wasm} bash -euo pipefail -c '
            source ./tests/scripts/test_task_install_dist_wheels.sh
            ./tests/scripts/task_web_wasm.sh
          '""",
      label: 'Run WASM lint and tests',
    )
  }
}
