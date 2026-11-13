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

import org.jenkinsci.plugins.pipeline.modeldefinition.Utils

// Reject untrusted helper edits before allocating nodes or running pipeline steps.
def helper_source
try {
  helper_source = readTrusted('ci/jenkins/generated/common.groovy')
} catch (Throwable ex) {
  echo 'Jenkinsfile changes must be pushed to an upstream branch. ' +
    'gpu/pr-head will fail for these changes; check gpu/branch status instead.'
  throw ex
}
ci = evaluate(helper_source)
ci.start()

stage('Build') {
  ci.with_node(instance: 'CPU-SPOT', retry_instance: 'CPU',
    name: 'BUILD: GPU', folder: 'tvm/build-gpu', variables: ['PLATFORM=gpu']) {
    ci.docker_init(images.ci_gpu)
    withEnv(ci.build_environment()) {
      sh(
        script: """${docker_run} --no-gpu ${images.ci_gpu} bash -euo pipefail -c '
          source ./tests/scripts/task_install_build_dependencies.sh
          ./tests/scripts/task_build_gpu.sh
        '""",
        label: 'Build gpu',
      )
    }
    ci.upload_artifacts('gpu', '--path "dist/*.whl"')

    // Check the compiler-only GPU configuration after publishing the test build.
    sh 'rm -rf build'
    withEnv(ci.build_environment()) {
      sh(
        script: "${docker_run} --no-gpu ${images.ci_gpu} ./tests/scripts/task_build_gpu_other.sh",
        label: 'Build gpu_other',
      )
    }
  }
}

stage('Test') {
  parallel(
    'unittest: GPU': {
      ci.with_node(instance: 'GPU-SPOT', retry_instance: 'GPU',
        name: 'unittest: GPU', folder: 'tvm/ut-python-gpu',
        variables: ['PLATFORM=gpu', 'TEST_STEP_NAME=unittest: GPU']) {
        ci.docker_init(images.ci_gpu)
        ci.download_artifacts('gpu')
        sh(
          script: """${docker_run} ${images.ci_gpu} bash -euo pipefail -c '
            source ./tests/scripts/test_task_install_dist_wheels.sh
            ./tests/scripts/task_java_unittest.sh
            ./tests/scripts/task_python_unittest_gpuonly.sh
          '""",
          label: 'Run Java and Python GPU unit tests',
        )
      }
    },
    'docs: GPU': {
      ci.with_node(instance: 'GPU-SPOT', retry_instance: 'GPU',
        name: 'docs: GPU', folder: 'tvm/docs-python-gpu',
        variables: ['PLATFORM=gpu', 'TEST_STEP_NAME=docs: GPU']) {
        ci.docker_init(images.ci_gpu)
        ci.download_artifacts('gpu')
        sh(
          script: """${docker_run} ${images.ci_gpu} bash -euo pipefail -c '
            source ./tests/scripts/test_task_install_dist_wheels.sh
            ./tests/scripts/task_python_docs.sh
          '""",
          label: 'Build docs',
        )
        ci.upload_artifacts('docs', '--path docs.tgz')
        sh(
          script: "aws s3 cp --no-progress _docs s3://${s3_bucket}/${s3_prefix}/docs --recursive",
          label: 'Upload docs to S3',
        )
      }
    },
  )
}

def deploy_docs() {
  // Note: This code must stay in the Jenkinsfile to ensure that it runs
  // from a trusted context only
  sh(
    script: '''
      set -eux
      rm -rf tvm-site
      git clone -b $DOCS_DEPLOY_BRANCH --depth=1 https://github.com/apache/tvm-site
      cd tvm-site
      git status
      git checkout -B $DOCS_DEPLOY_BRANCH

      git ls-tree HEAD docs/ --name-only | grep -vP '^docs/v\\d' | xargs rm -rf
      mkdir -p docs
      tar xf ../docs.tgz -C docs
      COMMIT=$(cat docs/commit_hash)
      git add .
      git config user.name tvm-bot
      git config user.email 95660001+tvm-bot@users.noreply.github.com
      git commit -m"deploying docs (apache/tvm@$COMMIT)"
      git status
    ''',
    label: 'Unpack docs and update tvm-site'
  )

  withCredentials([string(
    credentialsId: 'docs-push-token',
    variable: 'GITHUB_TOKEN',
    )]) {
    sh(
      script: '''
        cd tvm-site
        git remote add deploy https://$GITHUB_TOKEN:x-oauth-basic@github.com/apache/tvm-site.git
        git push deploy $DOCS_DEPLOY_BRANCH || true
      ''',
      label: 'Upload docs to apache/tvm-site'
    )
  }
}

stage('Deploy') {
  if (env.BRANCH_NAME == 'main') {
    parallel('Deploy Docs': {
      if (env.DOCS_DEPLOY_ENABLED == 'yes') {
        node('CPU') {
          ws(ci.per_exec_ws('tvm/deploy-docs')) {
            timeout(time: max_time, unit: 'MINUTES') {
              ci.init_git()
              sh(
                script: "./${jenkins_scripts_root}/s3.py --action download --bucket ${s3_bucket} --prefix ${s3_prefix}/docs --path docs.tgz",
                label: 'Download docs folder from S3',
              )
              deploy_docs()
            }
          }
        }
      } else {
        Utils.markStageSkippedForConditional('Deploy Docs')
      }
    })
  }
}
