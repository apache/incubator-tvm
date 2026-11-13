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

// ------ Config ------
// Keep image names aligned with docker-images.ini. Tags are resolved in prepare.
image_platforms = [ci_arm: 'ARM', ci_cpu: 'CPU', ci_gpu: 'GPU', ci_wasm: 'CPU']
// Resolve Hub inputs through the caller account ECR cache; false keeps public names.
enable_docker_hub_cache = true
max_time = 180
// Temporarily bypass image sccache wrappers and inherited compiler launchers.
docker_run = 'docker/bash.sh --env CI --env PLATFORM --env TEST_STEP_NAME' +
  ' --env CC=/usr/bin/gcc --env CXX=/usr/bin/g++' +
  ' --env CMAKE_C_COMPILER_LAUNCHER= --env CMAKE_CXX_COMPILER_LAUNCHER='
docker_build = 'docker/build.sh'
s3_bucket = 'tvm-jenkins-artifacts-prod'
jenkins_scripts_root = 'ci/jenkins/scripts'

// ------ Runtime state ------
images = image_platforms.collectEntries { name, platform -> [(name): ''] }
source_images = [:]
built_images = [:]
upstream_revision = null
rebuild_docker_images = false
s3_prefix = "tvm/${env.BRANCH_NAME}/${env.BUILD_NUMBER}"

properties([parameters(image_platforms.collect { name, platform ->
  string(name: "${name}_param", defaultValue: '')
})])

// Shared utilities; platform Jenkinsfiles own stages, parallel groups and commands.

def per_exec_ws(folder) {
  return "workspace/exec_${env.EXECUTOR_NUMBER}/" + folder
}

def update_upstream_revision(git_ref) {
  if (upstream_revision == null) {
    upstream_revision = sh(
      script: "git log -1 ${git_ref} --format=\'%H\'",
      label: 'Determine upstream revision',
      returnStdout: true,
    ).trim()
  }
}

def merge_with_target() {
  def target = env.CHANGE_TARGET
  sh (
    script: "git fetch origin ${target}",
    label: "Fetch target branch ${target}",
  )
  update_upstream_revision("FETCH_HEAD")
  sh (
    script: "git -c user.name=TVM-Jenkins -c user.email=jenkins@tvm.apache.org merge ${upstream_revision}",
    label: "Merge to origin/${target}"
  )
}

def checkout_trusted_files() {
  // trust everything from branch builds
  if (env.BRANCH_NAME == null || !env.BRANCH_NAME.startsWith('PR-')) {
    return;
  }

  // Trust contributors listed in the pinned target revision.
  // That target must contain the matching scripts and artifacts.py registry.
  // Before the coordinated rename reaches the target, use a trusted branch;
  // never mix target scripts with untrusted PR artifact metadata.
  grep_code = sh(
    returnStatus: true,
    script: "git show '${upstream_revision}:CONTRIBUTORS.md' | grep '@${env.CHANGE_AUTHOR}'",
    label: 'Check if change is from a contributor',
  )

  if (grep_code == 1) {
    // Any scripts that run on the bare host and not inside a Docker container
    // (especially those that access secrets) should be checked out here so
    // only trusted versions are used in CI
    sh(
      script: "git checkout ${upstream_revision} ${jenkins_scripts_root}/. ci/jenkins/artifacts.py",
      label: 'Check out trusted files',
    )
  }
}

def init_git() {
  retry(5) {
    checkout scm
  }

  // Add more info about job node
  sh (
    script: './ci/jenkins/scripts/show_node_info.sh',
    label: 'Show executor node info',
  )

  // Determine merge commit to use for all stages
  if (env.CHANGE_TARGET) {
    // This is a PR build, so merge with the latest of the PR's target branch
    // (e.g. main or a release branch like v0.25.0).
    merge_with_target()
  } else {
    // This is a branch build (main or a release branch). Only set
    // upstream_revision to HEAD and skip merging to avoid a race with another
    // commit merged to the branch.
    update_upstream_revision("HEAD")
  }

  sh(
    script: """
      set -eux
      . ${jenkins_scripts_root}/retry.sh
      retry 3 timeout 5m git submodule update --init --recursive -f --jobs 0
    """,
    label: 'Update git submodules',
  )
  checkout_trusted_files()
}

// Apply only after Hub selection and job overrides; inputs stay public image names.
def docker_hub_image(image, cache) {
  if (!cache || !image) return image

  def path = image
  if (image.contains('/')) {
    def host = image.substring(0, image.indexOf('/'))
    if (host in ['docker.io', 'index.docker.io', 'registry-1.docker.io']) {
      path = image.substring(image.indexOf('/') + 1)
    } else if (host.contains('.') || host.contains(':') || host == 'localhost') {
      return image
    }
  }
  if (!path.contains('/')) path = "library/${path}"
  return "${cache}/${path}"
}

def ecr_registry(image) {
  def registry = image.tokenize('/')[0]
  return registry ==~ /[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com(\.cn)?/ ? registry : null
}

def ecr_pull(full_name) {
  def registry = ecr_registry(full_name)
  def region = registry.tokenize('.')[3]
  withEnv(["AWS_DEFAULT_REGION=${region}", "AWS_ECR_REPO=${registry}"]) {
    try {
      sh(
        script: '''
          set -eux
          aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ECR_REPO
        ''',
        label: 'Log in to ECR'
      )
      sh(
        script: """
          set -eux
          . ${jenkins_scripts_root}/retry.sh
          retry 5 docker pull ${full_name}
        """,
        label: 'Pull image from ECR'
      )
    } finally {
      sh(
        script: 'docker logout $AWS_ECR_REPO',
        label: 'Clean up login credentials'
      )
    }
  }
}

def cancel_previous_build() {
  // cancel previous build if it is not on main.
  if (env.BRANCH_NAME != 'main') {
    def buildNumber = env.BUILD_NUMBER as int
    // Milestone API allows us to cancel previous build
    // with the same milestone number
    if (buildNumber > 1) milestone(buildNumber - 1)
    milestone(buildNumber)
  }
}

def is_last_build() {
  // check whether it is last build
  try {
    return currentBuild.number == currentBuild.rawBuild.project.getLastBuild().number
  } catch (Throwable ex) {
    echo 'Error during check is_last_build ' + ex.toString()
    return false
  }
}

def should_skip_ci(pr_number) {
  if (env.BRANCH_NAME == null || !env.BRANCH_NAME.startsWith('PR-')) {
    // never skip CI on build sourced from a branch
    return false
  }
  glob_skip_ci_code = sh (
    returnStatus: true,
    script: "./${jenkins_scripts_root}/git_skip_ci_globs.py",
    label: 'Check if CI should be skipped due to changed files',
  )
  if (glob_skip_ci_code == 0) {
    return true
  }
  withCredentials([string(
    credentialsId: 'tvm-bot-jenkins-reader',
    variable: 'GITHUB_TOKEN',
    )]) {
    // Exit code of 1 means run full CI (or the script had an error, so run
    // full CI just in case). Exit code of 0 means skip CI.
    git_skip_ci_code = sh (
      returnStatus: true,
      script: "./${jenkins_scripts_root}/git_skip_ci.py --pr '${pr_number}'",
      label: 'Check if CI should be skipped',
    )
  }
  return git_skip_ci_code == 0
}

def prepare(node_type) {
  stage('Prepare') {
    node(node_type) {
      ws("workspace/exec_${env.EXECUTOR_NUMBER}/tvm/prepare") {
        init_git()

        if (env.DETERMINE_DOCKER_IMAGES == 'yes') {
          sh(
            script: "./${jenkins_scripts_root}/determine_docker_images.py ${images.keySet().join(' ')}",
            label: 'Decide whether to use tlcpack or tlcpackstaging for Docker images',
          )
          images.keySet().each { name ->
            images[name] = sh(
              script: "cat .docker-image-names/${name}",
              label: "Find docker image name for ${name}",
              returnStdout: true,
            ).trim()
          }
        }
        def cache = ''
        if (enable_docker_hub_cache) {
          def aws_account_id = sh(
            returnStdout: true,
            script: 'aws sts get-caller-identity --query Account --output text',
            label: 'Get AWS ID'
          ).trim()
          cache = "${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/docker-hub"
        }
        images.keySet().each { name ->
          source_images[name] = params["${name}_param"] ?: images[name]
          images[name] = docker_hub_image(source_images[name], cache)
        }
        sh(
          script: 'echo "Docker images being used in this build:"\n' +
            images.collect { name, image -> "echo \" ${name} = ${image}\"" }.join('\n'),
          label: 'Docker image names',
        )

        skip_ci = should_skip_ci(env.CHANGE_ID)
        rebuild_docker_images = sh (
          returnStatus: true,
          script: "./${jenkins_scripts_root}/git_change_docker.sh",
          label: 'Check for any docker changes',
        )

        if (skip_ci) {
          // Don't rebuild when skipping CI
          rebuild_docker_images = false
        }
      }
    }
  }
}

def retry_latest(ex, body) {
  echo 'Exception during SPOT run ' + ex.toString()
  if (!is_last_build()) {
    echo 'Exit since it is not last build'
    throw ex
  }
  echo 'Retry on-demand given it is last build'
  currentBuild.result = 'SUCCESS'
  // Do not catch failure of the on-demand retry.
  body()
}

// Public entry points.

// with_node(instance: 'CPU-SPOT', retry_instance: 'CPU', name: 'Build',
//           folder: 'tvm/build', variables: ['PLATFORM=cpu']) { docker_init(image); ... }
// name marks skipped stages; folder isolates executor workspaces; variables is optional.
// Skip before allocation/setup. Each attempt checks out sources, then times the body.
// Keep docker_init explicit in the body. Omit retry_instance for a single attempt;
// otherwise failures, including interruptions, retry only for the newest build.
def with_node(options, body) {
  if (skip_ci) {
    Utils.markStageSkippedForConditional(options.name)
    return
  }
  def run = { instance ->
    node(instance) {
      ws(per_exec_ws(options.folder)) {
        init_git()
        timeout(time: max_time, unit: 'MINUTES') {
          withEnv(options.variables ?: []) { body() }
        }
      }
    }
  }
  try {
    run(options.instance)
  } catch (Throwable ex) {
    if (!options.retry_instance) throw ex
    // Preserve the original newest-build retry policy, including interruptions.
    // Failures from the single on-demand attempt propagate without another retry.
    retry_latest(ex) { run(options.retry_instance) }
  }
}

// upload_artifacts('cpu', '--artifact tvm_lib --artifact cpptest')
// Or select paths: '--path "build/*.whl"'. Options pass unchanged to s3.py.
// Registry artifacts and explicit paths are mutually exclusive; repeat one-value flags.
// Upload globs are sorted/deduplicated; download_artifacts(tag) copies the whole saved prefix.
def upload_artifacts(tag, options) {
  sh(
    script: "./${jenkins_scripts_root}/s3.py --action upload --bucket ${s3_bucket} --prefix ${s3_prefix}/${tag} ${options}",
    label: 'Upload artifacts to S3',
  )
}

def download_artifacts(tag) {
  sh(
    script: "./${jenkins_scripts_root}/s3.py --action download --bucket ${s3_bucket} --prefix ${s3_prefix}/${tag}",
    label: 'Download artifacts from S3',
  )
}

def docker_init(image) {
  // Forget compiler wrappers cached before the temporary sccache bypass.
  sh 'rm -f build/CMakeCache.txt tirx/3rdparty/tvm-ffi/build/CMakeCache.txt'
  // Clear out all Docker images that aren't going to be used
  sh(
    script: """
    set -eux
    docker image ls --all
    IMAGES=\$(docker image ls --all --format '{{.Repository}}:{{.Tag}}  {{.ID}}')

    echo -e "Found images:\\n\$IMAGES"
    echo "\$IMAGES" | { grep -vE '${image}' || test \$? = 1; } | { xargs docker rmi || test \$? = 123; }

    docker image ls --all
    """,
    label: 'Clean old Docker images',
  )

  if (ecr_registry(image)) {
    ecr_pull(image)
  } else {
    sh(
      script: """
      set -eux
      . ${jenkins_scripts_root}/retry.sh
      retry 5 docker pull ${image}
      """,
      label: 'Pull docker image',
    )
  }
}

// Preparation retries Exceptions unconditionally; Errors and cancellation of the
// previous build are deliberately outside this fallback.
def start() {
  cancel_previous_build()
  try {
    prepare('CPU-SMALL-SPOT')
  } catch (Exception ex) {
    prepare('CPU-SMALL')
  }
}

// Preserve caller limits, otherwise share the node among its executors.
def build_environment() {
  def parallel_limit = env.CMAKE_BUILD_PARALLEL_LEVEL ?: sh(
    script: 'n=$(( $(getconf _NPROCESSORS_ONLN) / ${CI_NUM_EXECUTORS:-1} )); if [ "$n" -lt 1 ]; then n=1; fi; echo "$n"',
    returnStdout: true,
    label: 'Determine build parallelism',
  ).trim()
  return ["CMAKE_BUILD_PARALLEL_LEVEL=${parallel_limit}"]
}

return this
