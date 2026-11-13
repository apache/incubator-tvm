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
    'docker/pr-head will fail for these changes; check docker/branch status instead.'
  throw ex
}
ci = evaluate(helper_source)
ci.start()

def ecr_push(full_name) {
  aws_account_id = sh(
    returnStdout: true,
    script: 'aws sts get-caller-identity | grep Account | cut -f4 -d\\"',
    label: 'Get AWS ID'
  ).trim()

  def ecr_name = "${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/${full_name}"
  try {
    withEnv([
      "AWS_ACCOUNT_ID=${aws_account_id}",
      'AWS_DEFAULT_REGION=us-west-2',
      "AWS_ECR_REPO=${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com"]) {
      sh(
        script: '''
          set -eux
          aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ECR_REPO
        ''',
        label: 'Log in to ECR'
      )
      sh(
        script: """
          set -x
          . ${jenkins_scripts_root}/retry.sh
          docker tag ${full_name} \$AWS_ECR_REPO/${full_name}
          retry 5 docker push \$AWS_ECR_REPO/${full_name}
        """,
        label: 'Upload image to ECR'
      )
    }
  } finally {
    withEnv([
      "AWS_ACCOUNT_ID=${aws_account_id}",
      'AWS_DEFAULT_REGION=us-west-2',
      "AWS_ECR_REPO=${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com"]) {
      sh(
        script: 'docker logout $AWS_ECR_REPO',
        label: 'Clean up login credentials'
      )
    }
  }
  return ecr_name
}

def build_image(image_name) {
  hash = sh(
    returnStdout: true,
    script: 'git log -1 --format=\'%h\''
  ).trim()
  def full_name = "${image_name}:${env.BRANCH_NAME}-${hash}-${env.BUILD_NUMBER}".replace('/', '_')
  sh(
    script: "${docker_build} ${image_name} --spec ${full_name}",
    label: 'Build docker image'
  )
  return ecr_push(full_name)
}

def update_docker(ecr_image, hub_image) {
  if (ecr_image == null) {
    sh("image was not rebuilt, skipping")
    return
  }
  if (!ecr_image.contains("amazonaws.com")) {
    sh("echo \"Skipping '${ecr_image}' -> '${hub_image}' since it doesn\'t look like an ECR image\"")
    return
  }
  ci.docker_init(ecr_image)
  sh(
    script: """
    set -eux
    . ${jenkins_scripts_root}/retry.sh
    docker tag \
      ${ecr_image} \
      ${hub_image}
    retry 5 docker push ${hub_image}
    """,
    label: "Update ${hub_image} on Docker Hub",
  )
}

def deploy() {
  stage('Deploy') {
    if (env.BRANCH_NAME != 'main') return
    parallel(
      'Upload built Docker images': {
        if (env.DEPLOY_DOCKER_IMAGES == 'yes' && rebuild_docker_images && upstream_revision != null) {
          node('CPU') {
            ws(ci.per_exec_ws('tvm/deploy-docker')) {
              timeout(time: max_time, unit: 'MINUTES') {
                ci.init_git()
                try {
                  withCredentials([string(credentialsId: 'dockerhub-tlcpackstaging-key', variable: 'DOCKERHUB_KEY')]) {
                    sh(script: 'docker login -u tlcpackstaging -p ${DOCKERHUB_KEY}', label: 'Log in to Docker Hub')
                  }
                  def date = sh(
                    script: 'python3 -c \'import datetime; print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))\'',
                    label: 'Determine date', returnStdout: true,
                  ).trim()
                  def tag = "${date}-${upstream_revision.substring(0, 8)}"
                  images.keySet().each { name -> update_docker(built_images[name], "tlcpackstaging/${name}:${tag}") }
                } finally {
                  sh(script: 'docker logout', label: 'Clean up login credentials')
                }
              }
            }
          }
        } else {
          Utils.markStageSkippedForConditional('Upload built Docker images')
        }
      },
      'Tag tlcpackstaging to tlcpack': {
        if (env.DEPLOY_DOCKER_IMAGES == 'yes') {
          node('CPU') {
            ws(ci.per_exec_ws('tvm/tag-images')) {
              timeout(time: max_time, unit: 'MINUTES') {
                ci.init_git()
                withCredentials([string(credentialsId: 'dockerhub-tlcpack-key', variable: 'TLCPACK_TOKEN')]) {
                  try {
                    sh(script: 'echo $TLCPACK_TOKEN | docker login --username octomldriazati --password-stdin', label: 'Log in to Docker Hub')
                    source_images.each { name, image ->
                      if (image.contains('tlcpackstaging')) {
                        def tag = image.split(':')[1]
                        sh(
                          script: """
                            set -eux
                            . ${jenkins_scripts_root}/retry.sh
                            docker pull tlcpackstaging/${name}:${tag}
                            docker tag tlcpackstaging/${name}:${tag} tlcpack/${name.replace('_', '-')}:${tag}
                            retry 5 docker push tlcpack/${name.replace('_', '-')}:${tag}
                          """,
                          label: "Tag tlcpackstaging/${name} image to tlcpack",
                        )
                      }
                    }
                  } finally {
                    sh(script: 'docker logout', label: 'Clean up login credentials')
                  }
                }
              }
            }
          }
        } else {
          Utils.markStageSkippedForConditional('Tag tlcpackstaging to tlcpack')
        }
      },
    )
  }
}

if (rebuild_docker_images) {
  stage('Docker Image Build') {
    def builds = [:]
    image_platforms.each { name, label ->
      builds[name] = {
        node(label) {
          timeout(time: max_time, unit: 'MINUTES') {
            ci.init_git()
            // Test images remain unchanged until deployment publishes these builds.
            built_images[name] = build_image(name)
          }
        }
      }
    }
    parallel(builds)
  }
  deploy()
}
