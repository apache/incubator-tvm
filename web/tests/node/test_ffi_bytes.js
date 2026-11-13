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
jest.mock("../../src/tvmjs_runtime_wasi", () => ({
  __esModule: true,
  default: function MockWASI() {},
}));
const { Instance } = require("../../src/runtime");
const { createPolyfillWASI } = require("../../dist/tvmjs.bundle");

const wasmSource = fs.readFileSync(
  path.join(__dirname, "../../dist/wasm/tvmjs_runtime.wasm")
);

let tvm;

beforeAll(() => {
  tvm = new Instance(
    new WebAssembly.Module(wasmSource),
    createPolyfillWASI()
  );
});

afterAll(() => {
  tvm.dispose();
});

function makeBytes(length) {
  const bytes = new Uint8Array(length);
  for (let i = 0; i < bytes.length; ++i) bytes[i] = i % 251;
  return bytes;
}

function expectExactBytes(actual, expected) {
  expect(actual).toBeInstanceOf(Uint8Array);
  expect(actual).toEqual(expected);
}


test("PackedFunc large callback return supports reentrant calls", () => {
  tvm.withNewScope(() => {
    const echo = tvm.getGlobalFunc("testing.echo");
    const input = makeBytes(64 * 1024 + 17);
    const reentrantEcho = tvm.toPackedFunc((value) => {
      const innerResult = echo(value);
      expectExactBytes(innerResult, input);
      return innerResult;
    });

    expectExactBytes(reentrantEcho(input), input);
    expectExactBytes(reentrantEcho(input), input);
  });
});


test("PackedFunc reports non-ASCII callback errors and remains reusable", () => {
  tvm.withNewScope(() => {
    const message = "é".repeat(16);
    const reportedMessage = Array.from(
      new TextEncoder().encode(message),
      (byte) => String.fromCharCode(byte)
    ).join("");
    const throwing = tvm.toPackedFunc(() => {
      throw new Error(message);
    });

    expect(() => throwing()).toThrow(reportedMessage);

    const echo = tvm.getGlobalFunc("testing.echo");
    expect(echo(7)).toBe(7);
  });
});

test("PackedFunc non-byte arguments do not allocate byte-source descriptors", () => {
  expect(tvm.captureWasmByteSources([1, "x", new Uint8Array(16)]))
    .toBeUndefined();
});

test("PackedFunc captures Wasm-backed byte sources automatically", () => {
  const ptr = tvm.exports.TVMWasmAllocSpace(32);
  if (ptr === 0) {
    throw new Error("Cannot allocate the Wasm-backed input");
  }
  try {
    const input = new Uint8Array(tvm.memory.memory.buffer, ptr, 32);
    const expected = makeBytes(input.length);
    input.set(expected);
    const capture = tvm.captureWasmByteSources;
    tvm.captureWasmByteSources = function(args) {
      const sources = capture.call(this, args);
      this.memory.memory.grow(1);
      return sources;
    };
    try {
      tvm.withNewScope(() => {
        const echo = tvm.getGlobalFunc("testing.echo");
        expectExactBytes(echo(input), expected);
      });
      expect(input.byteLength).toBe(0);
    } finally {
      tvm.captureWasmByteSources = capture;
    }
  } finally {
    tvm.exports.TVMWasmFreeSpace(ptr);
  }
});

test("Asyncify keeps transient byte arguments alive across every rewind", async () => {
  if (!tvm.asyncifyEnabled()) {
    throw new Error("The test runtime must be built with Asyncify enabled");
  }

  tvm.beginScope();
  try {
    const input = makeBytes(64 * 1024 + 17);
    const heldPointers = [];
    tvm.registerAsyncifyFunc("tvmjs.testing.clobber_transient_bytes", async () => {
      await Promise.resolve();
      const ptr = tvm.exports.TVMWasmAllocSpace(input.byteLength);
      if (ptr === 0) {
        throw new Error("Cannot allocate the clobber buffer");
      }
      heldPointers.push(ptr);
      tvm.memory.storeRawBytes(ptr, new Uint8Array(input.byteLength).fill(0xa5));
    }, true);

    const callback = tvm.getGlobalFunc("tvmjs.testing.clobber_transient_bytes");
    const check = tvm.wrapAsyncifyPackedFunc(
      tvm.getGlobalFunc("tvmjs.testing.check_byte_array_across_callbacks")
    );
    try {
      const pending = check(input, callback);
      check.dispose();
      expect(await pending).toBe(0);
      expect(heldPointers).toHaveLength(2);
      expect(check._tvmPackedCell.getHandle(false)).toBe(0);
      await expect(check(input, callback)).rejects.toThrow(
        "Asyncify packed function has already been disposed"
      );
    } finally {
      for (const ptr of heldPointers) {
        tvm.exports.TVMWasmFreeSpace(ptr);
      }
    }
  } finally {
    tvm.endScope();
  }
});

test("Asyncify rejects overlapping calls from different wrappers", async () => {
  tvm.beginScope();
  let release;
  try {
    const input = makeBytes(64 * 1024 + 17);
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    tvm.registerAsyncifyFunc("tvmjs.testing.wait_for_concurrency_test", async () => {
      await gate;
    }, true);

    const callback = tvm.getGlobalFunc("tvmjs.testing.wait_for_concurrency_test");
    const first = tvm.wrapAsyncifyPackedFunc(
      tvm.getGlobalFunc("tvmjs.testing.check_byte_array_across_callbacks")
    );
    const second = tvm.wrapAsyncifyPackedFunc(
      tvm.getGlobalFunc("tvmjs.testing.check_byte_array_across_callbacks")
    );

    const pending = first(input, callback);
    await expect(second(input, callback)).rejects.toThrow(
      "Another Asyncify packed function is already running"
    );
    release();
    expect(await pending).toBe(0);
  } finally {
    if (release !== undefined) {
      release();
    }
    tvm.endScope();
  }
});

test("Asyncify call retains the function across alias disposal", async () => {
  tvm.beginScope();
  let release;
  try {
    const input = makeBytes(64 * 1024 + 17);
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    tvm.registerAsyncifyFunc("tvmjs.testing.wait_for_alias_disposal", async () => {
      await gate;
    }, true);

    const callback = tvm.getGlobalFunc("tvmjs.testing.wait_for_alias_disposal");
    const original = tvm.getGlobalFunc(
      "tvmjs.testing.check_byte_array_across_callbacks"
    );
    const active = tvm.wrapAsyncifyPackedFunc(original);
    const alias = tvm.wrapAsyncifyPackedFunc(original);

    const pending = active(input, callback);
    alias.dispose();
    expect(original._tvmPackedCell.getHandle(false)).toBe(0);
    release();
    expect(await pending).toBe(0);
  } finally {
    if (release !== undefined) {
      release();
    }
    tvm.endScope();
  }
});

test("Asyncify clears call state when frame cleanup throws", async () => {
  tvm.beginScope();
  const releasePackedCall = tvm.releasePackedCall;
  try {
    const original = tvm.getGlobalFunc("testing.echo");
    const wrapped = tvm.wrapAsyncifyPackedFunc(original);
    tvm.releasePackedCall = function(frame) {
      releasePackedCall.call(this, frame);
      throw new Error("expected frame cleanup failure");
    };

    const pending = wrapped(1);
    wrapped.dispose();
    await expect(pending).rejects.toThrow("expected frame cleanup failure");
    expect(original._tvmPackedCell.getHandle(false)).toBe(0);
    expect(tvm.asyncifyCallInProgress).toBe(false);
  } finally {
    tvm.releasePackedCall = releasePackedCall;
    tvm.endScope();
  }
});

test("Asyncify does not consume an owned byte result before completion", async () => {
  if (!tvm.asyncifyEnabled()) {
    throw new Error("The test runtime must be built with Asyncify enabled");
  }

  tvm.beginScope();
  try {
    const input = makeBytes(64 * 1024 + 17);
    const heldPointers = [];
    tvm.registerAsyncifyFunc("tvmjs.testing.clobber_owned_result", async () => {
      await Promise.resolve();
      const ptr = tvm.exports.TVMWasmAllocSpace(input.byteLength);
      if (ptr === 0) {
        throw new Error("Cannot allocate the clobber buffer");
      }
      heldPointers.push(ptr);
      tvm.memory.storeRawBytes(ptr, new Uint8Array(input.byteLength).fill(0x5a));
    }, true);

    const callback = tvm.getGlobalFunc("tvmjs.testing.clobber_owned_result");
    const returnBytes = tvm.wrapAsyncifyPackedFunc(
      tvm.getGlobalFunc("tvmjs.testing.return_bytes_across_callbacks")
    );
    try {
      expectExactBytes(await returnBytes(input, callback), input);
      expect(heldPointers).toHaveLength(2);
    } finally {
      for (const ptr of heldPointers) {
        tvm.exports.TVMWasmFreeSpace(ptr);
      }
    }
  } finally {
    tvm.endScope();
  }
});
