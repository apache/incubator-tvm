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

import argparse
import json
import logging
import os
import re
import subprocess
from urllib import request


def parse_remote(remote: str) -> tuple[str, str]:
    """
    Get a GitHub (user, repo) pair out of a git remote
    """
    if remote.startswith("https://"):
        # Parse HTTP remote
        parts = remote.split("/")
        if len(parts) < 2:
            raise RuntimeError(f"Unable to parse remote '{remote}'")
        user, repo = parts[-2], parts[-1].replace(".git", "")
    else:
        # Parse SSH remote
        m = re.search(r":(.*)/(.*)\.git", remote)
        if m is None or len(m.groups()) != 2:
            raise RuntimeError(f"Unable to parse remote '{remote}'")
        user, repo = m.groups()

    user = os.getenv("DEBUG_USER", user)
    repo = os.getenv("DEBUG_REPO", repo)
    return user, repo


def git(command, **kwargs):
    command = ["git", *command]
    logging.info(f"Running {command}")
    proc = subprocess.run(command, stdout=subprocess.PIPE, encoding="utf-8", **kwargs)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed {command}:\nstdout:\n{proc.stdout}")
    return proc.stdout.strip()


if __name__ == "__main__":
    help = "Exits with 0 if CI should be skipped, 1 otherwise"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--pr", required=True)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument(
        "--pr-title", help="(testing) PR title to use instead of fetching from GitHub"
    )
    args = parser.parse_args()
    logging.basicConfig(
        format="[%(filename)s:%(lineno)d %(levelname)s] %(message)s", level=logging.INFO
    )

    branch = git(["rev-parse", "--abbrev-ref", "HEAD"])
    log = git(["log", "--format=%s", "-1"])

    # Check the PR's title (don't check this until everything else passes first)
    def check_pr_title():
        remote = git(["config", "--get", f"remote.{args.remote}.url"])
        user, repo = parse_remote(remote)

        if args.pr_title:
            title = args.pr_title
        else:
            req = request.Request(
                f"https://api.github.com/repos/{user}/{repo}/pulls/{args.pr}",
                headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"},
            )
            with request.urlopen(req) as response:
                pr = json.loads(response.read())
            title = pr["title"]
        logging.info(f"pr title: {title}")
        tags = [tag.strip() for tag in re.findall(r"\[(.*?)\]", title)]
        logging.info(f"Found title tags: {tags}")
        return "skip ci" in tags

    if args.pr != "null" and args.pr.strip() != "" and branch != "main" and check_pr_title():
        logging.info("PR title starts with '[skip ci]', skipping...")
        exit(0)
    else:
        logging.info(f"Not skipping CI:\nargs.pr: {args.pr}\nbranch: {branch}\ncommit: {log}")
        exit(1)
