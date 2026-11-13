<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->
# tests/cpp

This folder contains some unit tests for C++ utilities in the codebase.

In principle we aim to do most compiler related tests in the python to
bring more development velocity, and only use this folder for low-level unit-tests.
All tests should finish fast and not dependent on a presence of an accelerator device.

Tests are grouped by subsystem in `ir/`, `runtime/`, `support/`, `sym/`,
`target/`, and `tirx/`. CMake discovers `.cc` files in these folders recursively.
Configure and build out of source in `build/`, then run `build/cpptest`.
