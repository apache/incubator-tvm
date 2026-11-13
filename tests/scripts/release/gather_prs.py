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
# ruff: noqa: E402, F401

import argparse
import csv
import os
import pickle
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(REPO_ROOT / "ci" / "github"))

from git_utils import GitHubRepo, git

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]


PRS_QUERY = """
query ($owner: String!, $name: String!, $after: String, $pageSize: Int!) {
  repository(owner: $owner, name: $name) {
    defaultBranchRef {
      name
      target {
        ... on Commit {
          oid
          history(after: $after, first: $pageSize) {
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              oid
              committedDate
              associatedPullRequests(first: 1) {
                nodes {
                  number
                  additions
                  changedFiles
                  deletions
                  author {
                    login
                  }
                  title
                  body
                }
              }
            }
          }
        }
      }
    }
  }
}
"""


def append_and_save(items, file):
    if not file.exists():
        data = []
    else:
        with open(file, "rb") as f:
            data = pickle.load(f)

    data += items
    with open(file, "wb") as f:
        pickle.dump(data, f)


def fetch_pr_data(args, cache):
    github = GitHubRepo(user=user, repo=repo, token=GITHUB_TOKEN)

    if args.from_commit is None or args.to_commit is None:
        print("--from-commit and --to-commit must be specified if --skip-query is not used")
        exit(1)

    i = 0
    page_size = 80
    cursor = f"{args.from_commit} {i}"

    while True:
        try:
            r = github.graphql(
                query=PRS_QUERY,
                variables={
                    "owner": user,
                    "name": repo,
                    "after": cursor,
                    "pageSize": page_size,
                },
            )
        except RuntimeError as e:
            print(f"{e}\nPlease check enviroment variable GITHUB_TOKEN whether is valid.")
            exit(1)
        data = r["data"]["repository"]["defaultBranchRef"]["target"]["history"]
        if not data["pageInfo"]["hasNextPage"]:
            break
        cursor = data["pageInfo"]["endCursor"]
        results = data["nodes"]

        to_add = []
        stop = False
        for r in results:
            if r["oid"] == args.to_commit:
                print(f"Found {r['oid']}, stopping")
                stop = True
                break
            else:
                to_add.append(r)

        oids = [r["oid"] for r in to_add]
        print(oids)
        append_and_save(to_add, cache)
        if stop:
            break
        print(i)
        i += page_size


def write_csv(
    filename: str, data: list[dict[str, Any]], threshold_filter: Callable[[dict[str, Any]], bool]
) -> None:
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, quotechar='"')
        writer.writerow(
            (
                "category",
                "subject",
                "date",
                "url",
                "author",
                "pr_title_tags",
                "pr_title",
                "additions",
                "deletions",
                "additions+deletions>threshold",
                "changed_files",
            )
        )
        for item in data:
            nodes = item["associatedPullRequests"]["nodes"]
            if len(nodes) == 0:
                continue
            pr = nodes[0]
            tags = [tag.strip() for tag in re.findall(r"\[(.*?)\]", pr["title"])]
            actual_tags = []
            for t in tags:
                items = [x.strip() for x in t.split(",")]
                actual_tags += items
            tags = actual_tags
            tags = [t.lower() for t in tags]
            category = tags[0] if len(tags) > 0 else ""
            author = pr["author"] if pr["author"] else "ghost"
            author = author.get("login", "") if isinstance(author, dict) else author
            writer.writerow(
                (
                    category,
                    "n/a",
                    item["committedDate"],
                    f"https://github.com/apache/tvm/pull/{pr['number']}",
                    author,
                    "/".join(tags),
                    pr["title"].replace(",", " "),
                    pr["additions"],
                    pr["deletions"],
                    1 if threshold_filter(pr) else 0,
                    pr["changedFiles"],
                )
            )
    print(f"{filename} generated!")


if __name__ == "__main__":
    help = "List out commits with attached PRs since a certain commit"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--from-commit", help="commit to start checking PRs from")
    parser.add_argument("--to-commit", help="commit to stop checking PRs from")
    parser.add_argument(
        "--threshold", default=0, help="sum of additions + deletions to consider large, such as 150"
    )
    parser.add_argument(
        "--skip-query", action="store_true", help="don't query GitHub and instead use cache file"
    )
    args = parser.parse_args()
    user = "apache"
    repo = "tvm"
    threshold = int(args.threshold)

    cache = Path("out.pkl")
    if not args.skip_query:
        fetch_pr_data(args, cache)

    with open(cache, "rb") as f:
        data = pickle.load(f)

    print(f"Found {len(data)} PRs")

    write_csv(
        filename="out_pr_gathered.csv",
        data=data,
        threshold_filter=lambda pr: pr["additions"] + pr["deletions"] > threshold,
    )
