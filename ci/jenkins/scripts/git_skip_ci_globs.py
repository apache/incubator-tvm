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
import fnmatch

from git_skip_ci import git

globs = [
    "*.md",
    ".github/*",
    ".asf.yaml",
    ".gitignore",
    "LICENSE",
    "NOTICE",
    "KEYS",
    "tests/lint/*",
    ".pre-commit-config.yaml",
]


def match_any(f: str) -> str | None:
    for glob in globs:
        if fnmatch.fnmatch(f, glob):
            return glob
    return None


if __name__ == "__main__":
    help = "Exits with code 1 if a change only touched files, indicating that CI could be skipped for this changeset"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--files", help="(testing only) comma separated list of files to check")
    args = parser.parse_args()
    print(args)
    if args.files is not None:
        diff = [x for x in args.files.split(",") if x.strip() != ""]
    else:
        diff = git(["diff", "--no-commit-id", "--name-only", "-r", "origin/main"])
        diff = diff.split("\n")
        diff = [d.strip() for d in diff]
        diff = [d for d in diff if d != ""]

    print(f"Changed files:\n{diff}")

    if len(diff) == 0:
        print("Found no changed files, skipping CI")
        exit(0)

    print(f"Checking with globs:\n{globs}")

    for file in diff:
        match = match_any(file)
        if match is None:
            print(f"{file} did not match any globs, running CI")
            exit(1)
        else:
            print(f"{file} matched glob {match}")

    print("All files matched a glob, skipping CI")
    exit(0)
