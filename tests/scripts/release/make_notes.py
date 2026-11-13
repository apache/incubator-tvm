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
import csv
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

# Tag dictionary used to create a mapping relation to categorize PRs owning same tag.
TAG_DICT = {
    "metaschedule": "MetaSchedule",
    "cuda": "cuda & cutlass & tensorrt",
    "cutlass": "cuda & cutlass & tensorrt",
    "tensorrt": "cuda & cutlass & tensorrt",
    "hexagon": "Hexagon",
    "metal": "Metal",
    "vulkan": "Vulkan",
    "clml": "OpenCL & CLML",
    "opencl": "OpenCL & CLML",
    "openclml": "OpenCL & CLML",
    "adreno": "Adreno",
    "acl": "ArmComputeLibrary",
    "rocm": "ROCm",
    "crt": "CRT",
    "web": "web",
    "wasm": "web",
    "runtime": "Runtime",
    "aot": "AOT",
    "arith": "Symbolic Analysis",
    "sym": "Symbolic Analysis",
    "byoc": "BYOC",
    "community": "Community",
    "tensorir": "TIR",
    "tirx": "TIR",
    "tensorflow": "Frontend",
    "tflite": "Frontend",
    "pytorch": "Frontend",
    "torch": "Frontend",
    "keras": "Frontend",
    "frontend": "Frontend",
    "onnx": "Frontend",
    "roofline": "Misc",
    "rpc": "Misc",
    "tophub": "Misc",
    "ux": "Misc",
    "APP": "Misc",
    "docker": "Docker",
    "doc": "Docs",
    "docs": "Docs",
    "llvm": "LLVM",
    "sve": "LLVM",
    "ci": "CI",
    "test": "CI",
    "tests": "CI",
    "testing": "CI",
    "unittest": "CI",
    "bugfix": "BugFix",
    "fix": "BugFix",
    "bug": "BugFix",
    "hotfix": "BugFix",
    "qnn": "Relay",
    "quantization": "Relay",
    "relax": "Relax",
    "unity": "Relax",
    "transform": "Relax",
    "kvcache": "Relax",
    "s_tir": "S-TIR",
    "disco": "Disco",
    "tvmscript": "TVMScript",
    "tvmscripts": "TVMScript",
    "tvmc": "TVMC",
    "topi": "TOPI",
}


def strip_header(title: str, header: str) -> str:
    pos = title.lower().find(header.lower())
    if pos == -1:
        return title

    return title[0:pos] + title[pos + len(header) :].strip()


def sprint(*args):
    print(*args, file=sys.stderr)


def create_pr_dict(cache: Path):
    with open(cache, "rb") as f:
        data = pickle.load(f)

    sprint(data[1])
    pr_dict = {}
    for item in data:
        prs = item["associatedPullRequests"]["nodes"]
        if len(prs) != 1:
            continue

        pr = prs[0]
        pr_dict[pr["number"]] = pr
    return pr_dict


def categorize_csv_file(csv_path: str):
    headings = defaultdict(lambda: defaultdict(list))
    sprint("Opening CSV")
    with open(csv_path) as f:
        input_file = csv.DictReader(f)

        i = 0
        blank_cate_set = {"Misc"}
        for row in input_file:
            # print(row)
            tags = row["pr_title_tags"].split("/")
            tags = ["misc"] if len(tags) == 0 else tags

            categories = map(lambda t: TAG_DICT.get(t.lower(), "Misc"), tags)
            categories = list(categories)
            categories = list(set(categories) - blank_cate_set)
            category = "Misc" if len(categories) == 0 else categories[0]

            subject = row["subject"].strip()
            pr_number = row["url"].split("/")[-1]

            if category == "" or subject == "":
                sprint(f"Skipping {i}th pr with number: {pr_number}, row: {row}")
                continue

            headings[category][subject].append(pr_number)
            i += 1
            # if i > 30:
            #     break
    return headings


if __name__ == "__main__":
    help = "List out commits with attached PRs since a certain commit"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument(
        "--notes", required=True, help="csv or markdown file of categorized PRs in order"
    )
    parser.add_argument(
        "--is-pr-with-link",
        required=False,
        help="exported pr number with hyper-link for forum format",
    )
    parser.add_argument(
        "--convert-with-link",
        required=False,
        help="make PR number in markdown file owning hyper-link",
    )
    args = parser.parse_args()
    user = "apache"
    repo = "tvm"

    if args.convert_with_link:
        with open(args.notes) as f:
            lines = f.readlines()
        formated = []
        for line in lines:
            match = re.search(r"#\d+", line)
            if match:
                pr_num_str = match.group()
                pr_num_int = pr_num_str.replace("#", "")
                pr_number_str = f"[#{pr_num_int}](https://github.com/apache/tvm/pull/{pr_num_int})"
                line = line.replace(pr_num_str, pr_number_str)
            formated.append(line)
        result = "".join(formated)
        print(result)
        exit(0)

    # 1. Create PR dict from cache file
    cache = Path("out.pkl")
    if not cache.exists():
        sprint("run gather_prs.py first to generate out.pkl")
        exit(1)
    pr_dict = create_pr_dict(cache)

    # 2. Categorize csv file as dict by category and subject (sub-category)
    headings = categorize_csv_file(args.notes)

    # 3. Summarize and sort all categories
    def sorter(x):
        if x == "Misc":
            return 10
        return 0

    keys = list(headings.keys())
    keys = list(sorted(keys))
    keys = list(sorted(keys, key=sorter))

    # 4. Generate markdown by loop categorized csv file dict
    def pr_title(number, heading):
        # print(f"number:{number}, heading:{heading}, len(pr_dict):{len(pr_dict)}")
        try:
            title = pr_dict[int(number)]["title"]
            title = strip_header(title, heading)
        except Exception:
            sprint("The out.pkl file is not match with csv file.")
            exit(1)
        return title

    output = ""
    for key in keys:
        value = headings[key]
        if key == "DO NOT INCLUDE":
            continue
        value = dict(value)
        output += f"### {key}\n"
        misc = []
        misc += value.get("n/a", [])
        misc += value.get("Misc", [])
        for pr_number in misc:
            if args.is_pr_with_link:
                pr_number_str = f"[#{pr_number}](https://github.com/apache/tvm/pull/{pr_number})"
            else:
                pr_number_str = f"#{pr_number}"
            pr_str = f" * {pr_number_str} - {pr_title(pr_number, '[' + key + ']')}\n"
            output += pr_str

        for subheading, pr_numbers in value.items():
            if subheading == "DO NOT INCLUDE":
                continue
            if subheading == "n/a" or subheading == "Misc":
                continue
            else:
                output += f" * {subheading} - " + ", ".join([f"#{n}" for n in pr_numbers]) + "\n"
            # print(value)

        output += "\n"

    # 5. Print markdown-format output
    print(output)
