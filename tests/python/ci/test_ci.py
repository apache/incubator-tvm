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
# ruff: noqa: RUF005
"""Test various CI scripts and GitHub Actions workflows"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

JENKINS_SCRIPT_ROOT = Path(__file__).resolve().parents[3] / "ci" / "jenkins" / "scripts"


class TempGit:
    """
    A wrapper to run commands in a directory (specifically for use in CI tests)
    """

    def __init__(self, cwd):
        self.cwd = cwd
        # Jenkins git is too old and doesn't have 'git init --initial-branch',
        # so init and checkout need to be separate steps
        self.run("init", stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        self.run("checkout", "-b", "main", stderr=subprocess.PIPE)
        self.run("remote", "add", "origin", "https://github.com/apache/tvm.git")

    def run(self, *args, **kwargs):
        """
        Run a git command based on *args
        """
        proc = subprocess.run(
            ["git"] + list(args), encoding="utf-8", cwd=self.cwd, check=False, **kwargs
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git command failed: '{args}'")

        return proc


def run_script(command: list[Any], check: bool = True, **kwargs):
    """
    Wrapper to run a script and print its output if there was an error
    """
    command = [str(c) for c in command]
    kwargs_to_send = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "encoding": "utf-8",
    }
    env = kwargs.pop("env", None)
    if env is not None:
        kwargs_to_send["env"] = {**os.environ, **env}
    kwargs_to_send.update(kwargs)
    proc = subprocess.run(
        command,
        check=False,
        **kwargs_to_send,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"Process failed:\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}")

    return proc


@pytest.fixture
def jenkins_script_root():
    return JENKINS_SCRIPT_ROOT


def parameterize_named(**kwargs):
    keys = next(iter(kwargs.values())).keys()
    return pytest.mark.parametrize(
        ",".join(keys), [tuple(d.values()) for d in kwargs.values()], ids=kwargs.keys()
    )


@parameterize_named(
    dont_skip_main=dict(
        commands=[],
        should_skip=False,
        pr_title="[skip ci] test",
        why="ci should not be skipped on main",
    ),
    dont_skip_main_with_commit=dict(
        commands=[
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
        ],
        should_skip=False,
        pr_title="[skip ci] test",
        why="ci should not be skipped on main",
    ),
    skip_on_new_branch=dict(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
        ],
        should_skip=True,
        pr_title="[skip ci] test",
        why="ci should be skipped on a branch with [skip ci] in the last commit",
    ),
    no_skip_in_pr_title=dict(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
        ],
        should_skip=False,
        pr_title="[no skip ci] test",
        why="ci should not be skipped on a branch with "
        "[skip ci] in the last commit but not the PR title",
    ),
    skip_in_pr_title=dict(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
            ["commit", "--allow-empty", "--message", "commit 2"],
        ],
        should_skip=True,
        pr_title="[skip ci] test",
        why="ci should be skipped with [skip ci] in the PR title",
    ),
    skip_in_pr_title_many_commits=dict(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "commit 1"],
            ["commit", "--allow-empty", "--message", "commit 2"],
            ["commit", "--allow-empty", "--message", "commit 3"],
            ["commit", "--allow-empty", "--message", "commit 4"],
        ],
        should_skip=True,
        pr_title="[skip ci] test",
        why="ci should be skipped with [skip ci] in the PR title",
    ),
    skip_anywhere_in_title=dict(
        commands=[
            ["checkout", "-b", "some_new_branch"],
        ],
        should_skip=True,
        pr_title="[something][skip ci] test",
        why="skip ci tag should work anywhere in title",
    ),
)
def test_skip_ci(tmpdir_factory, commands, should_skip, pr_title, why, jenkins_script_root):
    """
    Test that CI is skipped when it should be
    """
    skip_ci_script = jenkins_script_root / "git_skip_ci.py"

    git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))

    git.run("config", "user.name", "ci")
    git.run("config", "user.email", "email@example.com")
    git.run("commit", "--allow-empty", "--message", "base commit")
    for command in commands:
        git.run(*command)
    pr_number = "1234"
    proc = run_script(
        [skip_ci_script, "--pr", pr_number, "--pr-title", pr_title],
        cwd=git.cwd,
        check=False,
    )
    expected = 0 if should_skip else 1
    if proc.returncode != expected:
        raise RuntimeError(
            f"Unexpected return code {proc.returncode} "
            f"(expected {expected}) in {why}:\n{proc.stdout}"
        )


@parameterize_named(
    no_file=dict(files=[], should_skip=True),
    readme=dict(files=["README.md"], should_skip=True),
    c_file=dict(files=["test.c"], should_skip=False),
    c_and_readme=dict(files=["test.c", "README.md"], should_skip=False),
    src_file_and_readme=dict(
        files=["src/autotvm/feature_visitor.cc", "README.md"], should_skip=False
    ),
    yaml_and_readme=dict(files=[".asf.yaml", "docs/README.md"], should_skip=True),
)
def test_skip_globs(tmpdir_factory, files, should_skip, jenkins_script_root):
    """
    Test that CI is skipped if only certain files are edited
    """
    script = jenkins_script_root / "git_skip_ci_globs.py"

    git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))

    proc = run_script(
        [
            script,
            "--files",
            ",".join(files),
        ],
        check=False,
        cwd=git.cwd,
    )

    if should_skip:
        assert proc.returncode == 0
    else:
        assert proc.returncode == 1


def assert_in(needle: str, haystack: str):
    """
    Check that 'needle' is in 'haystack'
    """
    if needle not in haystack:
        raise AssertionError(f"item not found:\n{needle}\nin:\n{haystack}")


@parameterize_named(
    use_tlcpack=dict(
        images=["ci_arm", "ci_cpu"],
        expected={
            "ci_arm": "tlcpack/ci-arm:abc-abc-123",
            "ci_cpu": "tlcpack/ci-cpu:abc-abc-234",
        },
    ),
    use_staging=dict(
        images=["ci_arm2"],
        expected={
            "ci_arm2": "tlcpackstaging/ci_arm2:abc-abc-123",
        },
    ),
)
def test_determine_docker_images(tmpdir_factory, images, expected, jenkins_script_root):
    """Test script to decide whether to use tlcpack or tlcpackstaging for images"""
    script = jenkins_script_root / "determine_docker_images.py"

    git_dir = tmpdir_factory.mktemp("tmp_git_dir")

    docker_data = {
        "repositories/tlcpack/ci-arm/tags/abc-abc-123": {},
        "repositories/tlcpack/ci-cpu/tags/abc-abc-234": {},
    }

    images_data = {
        "ci_arm": "tlcpack/ci-arm:abc-abc-123",
        "ci_cpu": "tlcpack/ci-cpu:abc-abc-234",
        "ci_arm2": "tlcpack/ci-arm2:abc-abc-123",
    }

    run_script(
        [
            script,
            "--testing-docker-data",
            json.dumps(docker_data),
            "--testing-images-data",
            json.dumps(images_data),
            "--base-dir",
            git_dir,
        ]
        + images,
        cwd=git_dir,
    )

    for expected_filename, expected_image in expected.items():
        with open(Path(git_dir) / expected_filename) as f:
            actual_image = f.read()

        assert actual_image == expected_image


@parameterize_named(
    invalid_name=dict(
        changed_files=[],
        name="abc",
        check="Image abc is not using new naming scheme",
        expected_code=1,
    ),
    no_hash=dict(
        changed_files=[], name="123-123-abc", check="No extant hash found", expected_code=1
    ),
    no_changes=dict(
        changed_files=[["test.txt"]],
        name=None,
        check="Did not find changes, no rebuild necessary",
        expected_code=0,
    ),
    docker_changes=dict(
        changed_files=[["test.txt"], ["docker/test.txt"]],
        name=None,
        check="Found docker changes",
        expected_code=2,
    ),
)
def test_should_rebuild_docker(
    tmpdir_factory, changed_files, name, check, expected_code, jenkins_script_root
):
    """
    Check that the Docker images are built when necessary
    """
    tag_script = jenkins_script_root / "should_rebuild_docker.py"

    git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
    git.run("config", "user.name", "ci")
    git.run("config", "user.email", "email@example.com")

    git_path = Path(git.cwd)
    for i, commits in enumerate(changed_files):
        for filename in commits:
            path = git_path / filename
            path.parent.mkdir(exist_ok=True, parents=True)
            path.touch()
            git.run("add", filename)

        git.run("commit", "-m", f"message {i}")

    if name is None:
        ref = "HEAD"
        if len(changed_files) > 1:
            ref = f"HEAD~{len(changed_files) - 1}"
        proc = git.run("rev-parse", ref, stdout=subprocess.PIPE)
        last_hash = proc.stdout.strip()
        name = f"123-123-{last_hash}"

    docker_data = {
        "repositories/tlcpack": {
            "results": [
                {
                    "name": "ci-something",
                },
                {
                    "name": "something-else",
                },
            ],
        },
        "repositories/tlcpack/ci-something/tags": {
            "results": [{"name": name}, {"name": name + "old"}],
        },
    }

    proc = run_script(
        [
            tag_script,
            "--testing-docker-data",
            json.dumps(docker_data),
        ],
        stderr=subprocess.STDOUT,
        cwd=git.cwd,
        check=False,
    )

    assert_in(check, proc.stdout)
    assert proc.returncode == expected_code


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
