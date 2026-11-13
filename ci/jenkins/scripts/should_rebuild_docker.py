#!/usr/bin/env python3
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
# ruff: noqa: E501
import argparse
import datetime
import json
import logging
import subprocess
from typing import Any

from http_utils import get

DOCKER_API_BASE = "https://hub.docker.com/v2/"
PAGE_SIZE = 25
TEST_DATA = None


def docker_api(url: str) -> dict[str, Any]:
    """
    Run a paginated fetch from the public Docker Hub API
    """
    if TEST_DATA is not None:
        return TEST_DATA[url]
    pagination = f"?page_size={PAGE_SIZE}&page=1"
    url = DOCKER_API_BASE + url + pagination
    r, headers = get(url)
    reset = headers.get("x-ratelimit-reset")
    if reset is not None:
        reset = datetime.datetime.fromtimestamp(int(reset))
        reset = reset.isoformat()
    logging.info(
        f"Docker API Rate Limit: {headers.get('x-ratelimit-remaining')} / {headers.get('x-ratelimit-limit')} (reset at {reset})"
    )
    if "results" not in r:
        raise RuntimeError(f"Error fetching data, no results found in: {r}")
    return r


def any_docker_changes_since(hash: str) -> bool:
    """
    Check the docker/ directory, return True if there have been any code changes
    since the specified hash
    """
    cmd = ["git", "diff", hash, "--", "docker/"]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True
    )
    stdout = proc.stdout.strip()
    return stdout != "", stdout


def does_commit_exist(hash: str) -> bool:
    """
    Returns True if the hash exists in the repo
    """
    cmd = ["git", "rev-parse", "-q", hash]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
    )
    print(proc.stdout)
    if proc.returncode == 0:
        return True

    if "unknown revision or path not in the working tree" in proc.stdout:
        return False

    raise RuntimeError(f"Unexpected failure when running: {cmd}")


def find_hash_for_tag(tag: dict[str, Any]) -> str:
    """
    Split the hash off of a name like <date>-<time>-<hash>
    """
    name = tag["name"]
    name_parts = name.split("-")
    if len(name_parts) != 3:
        raise RuntimeError(f"Image {name} is not using new naming scheme")
    shorthash = name_parts[2]
    return shorthash


def find_commit_in_repo(tags: list[dict[str, Any]]):
    """
    Look through all the docker tags, find the most recent one which references
    a commit that is present in the repo
    """
    for tag in tags["results"]:
        shorthash = find_hash_for_tag(tag)
        logging.info(f"Hash '{shorthash}' does not exist in repo")
        if does_commit_exist(shorthash):
            return shorthash, tag

    raise RuntimeError(f"No extant hash found in tags:\n{tags}")


def main():
    # Fetch all tlcpack images
    images = docker_api("repositories/tlcpack")

    # Ignore all non-ci images
    relevant_images = [image for image in images["results"] if image["name"].startswith("ci-")]
    image_names = [image["name"] for image in relevant_images]
    logging.info(f"Found {len(relevant_images)} images to check: {', '.join(image_names)}")

    for image in relevant_images:
        # Check the tags for the image
        tags = docker_api(f"repositories/tlcpack/{image['name']}/tags")

        # Find the hash of the most recent tag
        shorthash, tag = find_commit_in_repo(tags)
        name = tag["name"]
        logging.info(f"Looking for docker/ changes since {shorthash}")

        any_docker_changes, diff = any_docker_changes_since(shorthash)
        if any_docker_changes:
            logging.info(f"Found docker changes from {shorthash} when checking {name}")
            logging.info(diff)
            exit(2)

    logging.info("Did not find changes, no rebuild necessary")
    exit(0)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(filename)s:%(lineno)d %(levelname)s] %(message)s", level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        description="Exits 0 if Docker images don't need to be rebuilt, 1 otherwise"
    )
    parser.add_argument(
        "--testing-docker-data",
        help="(testing only) JSON data to mock response from Docker Hub API",
    )
    args = parser.parse_args()

    if args.testing_docker_data is not None:
        TEST_DATA = json.loads(args.testing_docker_data)

    main()
