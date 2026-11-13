/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
const path = require("path");
const fs = require("fs");
const assert = require("assert");
const tvmjs = require("../../dist/tvmjs.bundle")

const wasmPath = tvmjs.wasmPath();
const wasmSource = fs.readFileSync(path.join(wasmPath, "tvmjs_runtime.wasm"));

let tvm = new tvmjs.Instance(
  new WebAssembly.Module(wasmSource),
  tvmjs.createPolyfillWASI());

test("object", () => {
  tvm.withNewScope(() => {
    let data = [1, 2, 3, 4, 5, 6];
    let a = tvm.empty([2, 3], "float32").copyFrom(data);

    let t = tvm.makeTVMArray([]);
    let b = tvm.makeTVMArray([a, t]);
    // assert b instanceof tvmjs.TVMArray
    assert(b instanceof tvmjs.TVMArray);
    assert(b.size() == 2);

    let t1 = b.get(1);
    assert(t1.getHandle() == t.getHandle());
  });
});

test("shape cache does not invalidate caller-owned tuples", () => {
  tvm.beginScope();
  const disposedTuple = tvm.makeShapeTuple([987654321, -1]);
  disposedTuple.dispose();
  const cachedTuple = tvm.makeShapeTuple([987654321, -1]);
  assert.doesNotThrow(() => cachedTuple.typeKey());

  const evictedTuple = tvm.makeShapeTuple([987654321, 0]);
  for (let i = 1; i <= 256; ++i) {
    tvm.makeShapeTuple([987654321, i]);
  }
  assert.doesNotThrow(() => evictedTuple.typeKey());

  tvm.endScope();
  assert.throws(() => cachedTuple.getHandle(), /already been disposed/);
  assert.throws(() => evictedTuple.getHandle(), /already been disposed/);
});
