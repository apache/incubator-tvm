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
    'arm/pr-head will fail for these changes; check arm/branch status instead.'
  throw ex
}
ci = evaluate(helper_source)
ci.start()

stage('Build') {
  ci.with_node(instance: 'ARM-GRAVITON3-SPOT', retry_instance: 'ARM-GRAVITON3',
    name: 'BUILD: arm', folder: 'tvm/build-arm', variables: ['PLATFORM=arm']) {
    ci.docker_init(images.ci_arm)
    withEnv(ci.build_environment()) {
      sh(
        script: """${docker_run} ${images.ci_arm} bash -euo pipefail -c '
          source ./tests/scripts/task_install_build_dependencies.sh
          ./tests/scripts/task_build_arm.sh
        '""",
        label: 'Build arm wheels and C++ tests',
      )
    }
    ci.upload_artifacts('arm', '--path "dist/*.whl"')

  }
}
