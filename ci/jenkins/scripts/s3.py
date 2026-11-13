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
import glob
import importlib.util
import logging
import re
import subprocess
from enum import Enum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

ARTIFACTS_PY = REPO_ROOT / "ci" / "jenkins" / "artifacts.py"


def load_artifacts():
    """Load the ARTIFACTS dict from ci/jenkins/artifacts.py."""
    spec = importlib.util.spec_from_file_location("artifacts", ARTIFACTS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ARTIFACTS


def resolve_artifacts(artifact_names: list[str]) -> list[str]:
    """Resolve a list of artifact names to a flat list of file paths."""
    artifacts = load_artifacts()
    items = []
    for name in artifact_names:
        if name not in artifacts:
            known = list(artifacts.keys())
            logging.error(f"Unknown artifact '{name}'. Known artifacts: {known}")
            raise SystemExit(1)
        items.extend(artifacts[name])
    return items


def resolve_upload_items(items: list[str]) -> list[str]:
    """Expand local artifact patterns once, preserving literal path handling."""
    paths = set()
    for item in items:
        path = Path(item)
        if glob.has_magic(item):
            matches = glob.glob(item, root_dir=REPO_ROOT, recursive=True)
            if not matches:
                logging.warning("No artifacts matched pattern: %s", item)
            paths.update(Path(match).as_posix() for match in matches)
        else:
            # Keep missing literal paths for the existing optional-artifact warning.
            paths.add(path.as_posix())
    directories = {Path(path) for path in paths if Path(path).is_dir()}
    # Recursive directory copies already include descendants selected by other
    # entries or overlapping patterns. Transfer each destination only once.
    return sorted(
        path for path in paths if not any(parent in directories for parent in Path(path).parents)
    )


RETRY_SCRIPT = REPO_ROOT / "ci" / "jenkins" / "scripts" / "retry.sh"
S3_DOWNLOAD_REGEX = re.compile(r"download: s3://.* to (.*)")


class Action(Enum):
    UPLOAD = 1
    DOWNLOAD = 2


def show_md5(item: str) -> None:
    if not Path(item).is_dir():
        subprocess.run(["md5sum", item], check=True)


def parse_output_files(stdout: str) -> list[str]:
    """
    Grab the list of downloaded files from the output of 'aws s3 cp'. Lines look
    like:

        download: s3://some/prefix/a_file.txt to a_file.txt
    """
    files = []
    for line in stdout.split("\n"):
        line = line.strip()
        if line == "":
            continue
        m = S3_DOWNLOAD_REGEX.match(line)
        if m:
            files.append(m.groups()[0])

    return files


def chmod(files: list[str]) -> None:
    """
    S3 has no concept of file permissions so add them back in here to every file
    """
    # Add execute bit for downloads
    to_chmod = [str(f) for f in files]
    logging.info(f"Adding execute bit for files: {to_chmod}")
    if len(to_chmod) > 0:
        subprocess.run(["chmod", "+x", *to_chmod], check=True)


def s3(source: str, destination: str, recursive: bool) -> list[str]:
    """
    Send or download the source to the destination in S3
    """
    # Retry each transfer independently; retain retry.sh's bounded random backoff.
    command = [
        "bash",
        "-c",
        '. "$1"; shift; retry 3 "$@"',
        "s3",
        str(RETRY_SCRIPT),
        "aws",
        "s3",
        "cp",
        "--no-progress",
    ]
    if recursive:
        command.append("--recursive")
    command.extend([source, destination])
    logging.info("Copying %s to %s", source, destination)
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
        output = []
        for line in process.stdout:
            print(line, end="", flush=True)
            output.append(line)
        stdout = "".join(output)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command, stdout)
    return stdout


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(filename)s:%(lineno)d %(levelname)s] %(message)s", level=logging.INFO
    )
    help = "Uploads or downloads files from S3"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--action", help="either 'upload' or 'download'", required=True)
    parser.add_argument("--bucket", help="s3 bucket", required=True)
    parser.add_argument(
        "--prefix", help="s3 bucket + tag (e.g. s3://tvm-ci-prod/PR-1234/cpu", required=True
    )
    parser.add_argument(
        "--path",
        dest="paths",
        action="append",
        help="repository-relative file, folder or upload glob pattern (repeatable)",
        metavar="PATH",
    )
    parser.add_argument(
        "--artifact",
        "--bundle",
        dest="artifacts",
        action="append",
        help="artifact name from ci/jenkins/artifacts.py (repeatable)",
        metavar="NAME",
    )
    # Legacy --items accepts several paths and retains the last occurrence.
    parser.add_argument("--items", nargs="+", help="files and folders to transfer")

    args = parser.parse_args()
    if args.artifacts is not None and (args.paths is not None or args.items is not None):
        parser.error("--artifact/--bundle and --path/--items are mutually exclusive")

    logging.info(args)

    if Path.cwd() != REPO_ROOT:
        logging.error(f"s3.py can only be executed from the repo root, instead was in {Path.cwd()}")
        exit(1)

    prefix = args.prefix.strip("/")
    s3_path = f"s3://{args.bucket}/{prefix}"
    logging.info(f"Using s3 path: {s3_path}")

    if args.action == "upload":
        action = Action.UPLOAD
    elif args.action == "download":
        action = Action.DOWNLOAD
    else:
        logging.error(f"Unsupported action: {args.action}")
        exit(1)

    if args.artifacts is not None:
        items = resolve_artifacts(args.artifacts)
    elif args.paths is not None or args.items is not None:
        items = (args.paths or []) + (args.items or [])
    else:
        if args.action == "upload":
            logging.error("Cannot upload without --path or --artifact")
            exit(1)
        else:
            # Download the whole prefix
            items = ["."]

    if action == Action.UPLOAD:
        items = resolve_upload_items(items)

    for item in items:
        if action == Action.DOWNLOAD:
            source = s3_path
            recursive = True
            if item != ".":
                source = s3_path + "/" + item
                recursive = False
            try:
                stdout = s3(source=source, destination=item, recursive=recursive)
            except Exception:
                # Optional artifacts (e.g. per-backend device runtime DSOs) may not
                # exist in S3 when the build config didn't produce them. Skip silently.
                logging.warning(f"Download failed for {item}, skipping (may be optional)")
                continue
            files = parse_output_files(stdout)
            chmod(files)
            for file in files:
                # Show md5 after downloading
                show_md5(file)
        elif action == Action.UPLOAD:
            if not Path(item).exists():
                logging.warning(f"The path doesn't exist: {item}")
                continue
            show_md5(item)
            if Path(item).is_dir():
                if len(list(Path(item).glob("**/*"))) == 0:
                    raise RuntimeError(f"Cannot upload empty folder with name: {item}")
            s3(item, s3_path + "/" + item, recursive=Path(item).is_dir())
