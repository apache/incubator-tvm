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

TVM_ROOT=$(realpath $(shell dirname $(firstword $(MAKEFILE_LIST))))/../

INCLUDE_FLAGS = -I$(TVM_ROOT) -I$(TVM_ROOT)/include -I$(TVM_ROOT)/3rdparty/tvm-tirx/include\
	-I$(TVM_ROOT)/3rdparty/tvm-tirx/3rdparty/tvm-ffi/include\
	-I$(TVM_ROOT)/3rdparty/tvm-tirx/3rdparty/tvm-ffi/3rdparty/dlpack/include\
	-I$(TVM_ROOT)/3rdparty/tvm-tirx/3rdparty/compiler-rt

.PHONY: clean all rmtypedep preparetest

all: dist/wasm/tvmjs_runtime.wasm dist/wasm/tvmjs_runtime.wasi.js src/tvmjs_runtime_wasi.js

EMCC = emcc

EMCC_CFLAGS = $(INCLUDE_FLAGS) -O3 -std=c++17 -Wno-ignored-attributes

EMCC_LDFLAGS = --no-entry -s WASM_BIGINT=1 -s ALLOW_MEMORY_GROWTH=1 -s STANDALONE_WASM=1\
 -s ERROR_ON_UNDEFINED_SYMBOLS=0 --pre-js emcc/preload.js\
 -s ASYNCIFY=1

dist/wasm/%.bc: emcc/%.cc
	@mkdir -p $(@D)
	$(EMCC) $(EMCC_CFLAGS) -c -MM -MT dist/wasm/$*.bc $< >dist/wasm/$*.d
	$(EMCC) $(EMCC_CFLAGS) -emit-llvm -c -o dist/wasm/$*.bc $<


dist/wasm/tvmjs_runtime.wasm: dist/wasm/wasm_runtime.bc dist/wasm/tvmjs_support.bc dist/wasm/webgpu_runtime.bc
	@mkdir -p $(@D)
	$(EMCC) $(EMCC_CFLAGS) -o dist/wasm/tvmjs_runtime.js $+ $(EMCC_LDFLAGS)

dist/wasm/tvmjs_runtime.wasi.js: dist/wasm/tvmjs_runtime.wasm emcc/decorate_as_wasi.py
	python3 emcc/decorate_as_wasi.py dist/wasm/tvmjs_runtime.js $@ cjs

src/tvmjs_runtime_wasi.js: dist/wasm/tvmjs_runtime.wasm emcc/decorate_as_wasi.py
	python3 emcc/decorate_as_wasi.py dist/wasm/tvmjs_runtime.js $@ es

clean:
	@rm -rf dist/wasm lib src/tvmjs_runtime_wasi.js

-include dist/wasm/*.d
