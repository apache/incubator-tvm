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
# ruff: noqa: E501, RUF005

import base64
import json
import logging
import os
import re
import subprocess
from typing import Any
from urllib import error, request

DRY_RUN = object()


def compress_query(query: str) -> str:
    query = query.replace("\n", "")
    query = re.sub("\s+", " ", query)
    return query


def post(url: str, body: Any | None = None, auth: tuple[str, str] | None = None):
    logging.info(f"Requesting POST to {url} with {body}")
    headers = {}
    req = request.Request(url, headers=headers, method="POST")
    if auth is not None:
        auth_str = base64.b64encode(f"{auth[0]}:{auth[1]}".encode())
        req.add_header("Authorization", f"Basic {auth_str.decode()}")

    if body is None:
        body = ""

    req.add_header("Content-Type", "application/json; charset=utf-8")
    data = json.dumps(body)
    data = data.encode("utf-8")
    req.add_header("Content-Length", len(data))

    with request.urlopen(req, data) as response:
        return response.read()


def dry_run_token(is_dry_run: bool) -> Any:
    if is_dry_run:
        return DRY_RUN
    return os.environ["GITHUB_TOKEN"]


class GitHubRepo:
    GRAPHQL_URL = "https://api.github.com/graphql"

    def __init__(self, user, repo, token, test_data=None):
        self.token = token
        self.user = user
        self.repo = repo
        self.test_data = test_data
        self.num_calls = 0
        self.base = f"https://api.github.com/repos/{user}/{repo}/"

    def headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
        }

    def dry_run(self) -> bool:
        return self.token == DRY_RUN

    def graphql(self, query: str, variables: dict[str, str] | None = None) -> dict[str, Any]:
        query = compress_query(query)
        if variables is None:
            variables = {}

        response = self._request(
            self.GRAPHQL_URL,
            {"query": query, "variables": variables},
            method="POST",
        )
        if self.dry_run():
            return self.testing_response("POST", self.GRAPHQL_URL)

        if "data" not in response:
            msg = f"Error fetching data with query:\n{query}\n\nvariables:\n{variables}\n\nerror:\n{json.dumps(response, indent=2)}"
            raise RuntimeError(msg)
        return response

    def testing_response(self, method: str, url: str) -> Any:
        self.num_calls += 1
        key = f"[{self.num_calls}] {method} - {url}"
        if self.test_data is not None and key in self.test_data:
            return self.test_data[key]
        logging.info(f"Unknown URL in dry run: {key}")
        return {}

    def _request(self, full_url: str, body: dict[str, Any], method: str) -> dict[str, Any]:
        if self.dry_run():
            logging.info(f"Dry run, would have requested a {method} to {full_url} with {body}")
            return self.testing_response(method, full_url)

        logging.info(f"Requesting {method} to {full_url} with {body}")
        req = request.Request(full_url, headers=self.headers(), method=method.upper())
        req.add_header("Content-Type", "application/json; charset=utf-8")
        data = json.dumps(body)
        data = data.encode("utf-8")
        req.add_header("Content-Length", len(data))

        try:
            with request.urlopen(req, data) as response:
                content = response.read()
        except error.HTTPError as e:
            msg = str(e)
            error_data = e.read().decode()
            raise RuntimeError(f"Error response: {msg}\n{error_data}")

        logging.info(f"Got response from {full_url}: {content}")
        try:
            response = json.loads(content)
        except json.decoder.JSONDecodeError:
            return content

        return response

    def put(self, url: str, data: dict[str, Any]) -> dict[str, Any]:
        return self._request(self.base + url, data, method="PUT")

    def patch(self, url: str, data: dict[str, Any]) -> dict[str, Any]:
        return self._request(self.base + url, data, method="PATCH")

    def post(self, url: str, data: dict[str, Any]) -> dict[str, Any]:
        return self._request(self.base + url, data, method="POST")

    def get(self, url: str) -> dict[str, Any]:
        if self.dry_run():
            logging.info(f"Dry run, would have requested a GET to {url}")
            return self.testing_response("GET", url)
        url = self.base + url
        logging.info(f"Requesting GET to {url}")
        req = request.Request(url, headers=self.headers())
        with request.urlopen(req) as response:
            response = json.loads(response.read())
        return response

    def delete(self, url: str) -> dict[str, Any]:
        if self.dry_run():
            logging.info(f"Dry run, would have requested a DELETE to {url}")
            return self.testing_response("DELETE", url)
        url = self.base + url
        logging.info(f"Requesting DELETE to {url}")
        req = request.Request(url, headers=self.headers(), method="DELETE")
        with request.urlopen(req) as response:
            response = json.loads(response.read())
        return response


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
    command = ["git"] + command
    logging.info(f"Running {command}")
    proc = subprocess.run(command, stdout=subprocess.PIPE, encoding="utf-8", **kwargs)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed {command}:\nstdout:\n{proc.stdout}")
    return proc.stdout.strip()


def find_ccs(body: str) -> list[str]:
    matches = re.findall(r"(cc( @[-A-Za-z0-9]+)+)", body, flags=re.MULTILINE)
    matches = [full for full, last in matches]

    reviewers = []
    for match in matches:
        if match.startswith("cc "):
            match = match.replace("cc ", "")
        users = [x.strip() for x in match.split("@")]
        reviewers += users

    reviewers = set(x for x in reviewers if x != "")
    return list(reviewers)
