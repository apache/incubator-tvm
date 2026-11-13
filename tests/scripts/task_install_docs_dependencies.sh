#!/usr/bin/env bash
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

set -euxo pipefail

# Additional documentation dependencies for the CPU CI image.
uv pip install \
    'Sphinx==8.1.3' \
    'sphinx_autodoc_annotation~=1.0' \
    'sphinx-gallery==0.20.0' \
    'sphinx-book-theme==1.1.4' \
    'pydata-sphinx-theme==0.15.4' \
    'autodocsumm==0.2.14' \
    'image==1.5.33' \
    'matplotlib==3.10.8' \
    'commonmark==0.9.1' \
    'docutils==0.21.2' \
    'Pillow==12.1.1'
