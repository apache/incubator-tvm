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
const { CachedCallStack, Memory } = require("../../src/memory");

test("loadU64 reads unsigned values and rejects unsafe integers", () => {
  const wasmMemory = new WebAssembly.Memory({ initial: 1 });
  const memory = new Memory(wasmMemory);
  const words = new Uint32Array(wasmMemory.buffer);

  words.set([0x80000001, 1], 2);
  expect(memory.loadU64(8)).toBe(0x180000001);

  words.set([0, 0x200000], 2);
  expect(() => memory.loadU64(8)).toThrow("Cannot represent uint64 value");
});

const CACHE_LIMIT_BYTES = 64 * 1024;

function createMockStack(allocImpl) {
  const wasmMemory = new WebAssembly.Memory({ initial: 1 });
  const memory = {
    wasm32: true,
    memory: wasmMemory,
    sizeofPtr: () => 4,
    storeRawBytes: jest.fn(),
  };
  const allocations = [];
  const frees = [];
  let nextPtr = 1024;
  const alloc = jest.fn((size) => {
    const ptr = allocImpl === undefined
      ? (nextPtr += Math.max(size, 1)) - Math.max(size, 1)
      : allocImpl(size, allocations.length);
    allocations.push([ptr, size]);
    return ptr;
  });
  const free = jest.fn((ptr) => frees.push(ptr));
  return {
    memory,
    allocations,
    frees,
    alloc,
    free,
    stack: new CachedCallStack(memory, alloc, free),
  };
}

function loadU32(bytes, offset) {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    .getUint32(offset, true);
}

test("loadRawBytes returns an owned copy", () => {
  const wasmMemory = new WebAssembly.Memory({ initial: 1 });
  const memory = new Memory(wasmMemory);
  const source = new Uint8Array(wasmMemory.buffer, 8, 4);
  source.set([1, 2, 3, 4]);

  const result = memory.loadRawBytes(8, 4);

  expect(Array.from(result)).toEqual([1, 2, 3, 4]);
  expect(result.buffer).not.toBe(wasmMemory.buffer);

  result[0] = 10;
  source[1] = 20;
  expect(Array.from(result)).toEqual([10, 2, 3, 4]);
  expect(Array.from(source)).toEqual([1, 20, 3, 4]);
});

test("loadRawBytes preserves the requested length at the end of memory", () => {
  const wasmMemory = new WebAssembly.Memory({ initial: 1 });
  const memory = new Memory(wasmMemory);
  const source = new Uint8Array(wasmMemory.buffer);
  source.set([5, 6], source.length - 2);

  const result = memory.loadRawBytes(source.length - 2, 4);

  expect(Array.from(result)).toEqual([5, 6, 0, 0]);
});

test("viewRawBytes returns a borrowed Wasm memory view", () => {
  const wasmMemory = new WebAssembly.Memory({ initial: 1 });
  const memory = new Memory(wasmMemory);
  const source = new Uint8Array(wasmMemory.buffer, 8, 4);
  source.set([1, 2, 3, 4]);

  const result = memory.viewRawBytes(8, 4);

  expect(Array.from(result)).toEqual([1, 2, 3, 4]);
  expect(result.buffer).toBe(wasmMemory.buffer);
  source[0] = 10;
  result[1] = 20;
  expect(Array.from(result)).toEqual([10, 20, 3, 4]);
  expect(Array.from(source)).toEqual([10, 20, 3, 4]);
});

test("viewRawBytes refreshes its backing view after memory growth", () => {
  const wasmMemory = new WebAssembly.Memory({ initial: 1, maximum: 2 });
  const memory = new Memory(wasmMemory);
  const oldBuffer = wasmMemory.buffer;

  wasmMemory.grow(1);
  const source = new Uint8Array(wasmMemory.buffer, 65536, 4);
  source.set([5, 6, 7, 8]);
  const result = memory.viewRawBytes(65536, 4);

  expect(result.buffer).toBe(wasmMemory.buffer);
  expect(result.buffer).not.toBe(oldBuffer);
  expect(Array.from(result)).toEqual([5, 6, 7, 8]);
});

test("viewRawBytes supports shared Wasm memory", () => {
  const wasmMemory = new WebAssembly.Memory({
    initial: 1,
    maximum: 2,
    shared: true,
  });
  const memory = new Memory(wasmMemory);
  const source = new Uint8Array(wasmMemory.buffer, 16, 4);
  source.set([1, 2, 3, 4]);

  const result = memory.viewRawBytes(16, 4);

  expect(result.buffer).toBe(wasmMemory.buffer);
  expect(result.buffer).toBeInstanceOf(SharedArrayBuffer);
  expect(Array.from(result)).toEqual([1, 2, 3, 4]);
});

test.each([
  [-1, 1, "pointer"],
  [0.5, 1, "pointer"],
  [Number.MAX_SAFE_INTEGER + 1, 1, "pointer"],
  [0, -1, "byte length"],
  [0, 0.5, "byte length"],
  [0, Number.MAX_SAFE_INTEGER + 1, "byte length"],
  [65536, 1, "exceeds memory size"],
  [65535, 2, "exceeds memory size"],
  [Number.MAX_SAFE_INTEGER, 1, "exceeds memory size"],
])("viewRawBytes rejects invalid range (%p, %p)", (ptr, nbytes, message) => {
  const memory = new Memory(new WebAssembly.Memory({ initial: 1 }));
  expect(() => memory.viewRawBytes(ptr, nbytes)).toThrow(message);
});

test("CachedCallStack commits a view of its cached bytes", () => {
  const memory = {
    wasm32: true,
    sizeofPtr: () => 4,
    storeRawBytes: jest.fn(),
  };
  const stack = new CachedCallStack(memory, () => 1024, () => {});
  const offset = stack.allocRawBytes(4);
  stack.storeRawBytes(offset, new Uint8Array([1, 2, 3, 4]));

  stack.commitToWasmMemory(4);

  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
  const [ptr, bytes] = memory.storeRawBytes.mock.calls[0];
  expect(ptr).toBe(1024);
  expect(Array.from(bytes)).toEqual([1, 2, 3, 4]);
  expect(bytes.buffer).toBe(stack.buffer);
});

test("CachedCallStack keeps an exactly 64 KiB projected byte argument inline", () => {
  const { stack, memory, allocations } = createMockStack();
  const argsOffset = stack.allocRawBytes(16);
  const bytes = new Uint8Array(CACHE_LIMIT_BYTES - 24);
  bytes[0] = 3;
  bytes[bytes.length - 1] = 9;

  stack.allocThenSetArgBytes(argsOffset + 8, bytes);
  stack.commitToWasmMemory();

  expect(stack.buffer.byteLength).toBe(CACHE_LIMIT_BYTES);
  expect(allocations.map(([, size]) => size)).toEqual([128, CACHE_LIMIT_BYTES]);
  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
  const committed = memory.storeRawBytes.mock.calls[0][1];
  const headerPtr = loadU32(committed, argsOffset + 8);
  const headerOffset = headerPtr - allocations[1][0];
  const dataPtr = loadU32(committed, headerOffset);
  const dataOffset = dataPtr - allocations[1][0];
  expect(loadU32(committed, headerOffset + 4)).toBe(bytes.length);
  expect(committed[dataOffset]).toBe(3);
  expect(committed[dataOffset + bytes.length - 1]).toBe(9);
});

test("CachedCallStack stores an over-limit byte argument in one transient allocation", () => {
  const { stack, memory, allocations, frees } = createMockStack();
  const argsOffset = stack.allocRawBytes(16);
  const bytes = new Uint8Array(CACHE_LIMIT_BYTES - 23);
  bytes.set([1, 2, 3, 4]);
  bytes[bytes.length - 1] = 5;

  stack.allocThenSetArgBytes(argsOffset + 8, bytes);
  stack.commitToWasmMemory();

  expect(stack.buffer.byteLength).toBe(128);
  expect(allocations.map(([, size]) => size)).toEqual([128, bytes.length]);
  expect(memory.storeRawBytes).toHaveBeenCalledTimes(2);
  const [payloadPtr, payload] = memory.storeRawBytes.mock.calls[0];
  const [basePtr, committed] = memory.storeRawBytes.mock.calls[1];
  expect(payloadPtr).toBe(allocations[1][0]);
  expect(payload).toBe(bytes);
  const headerPtr = loadU32(committed, argsOffset + 8);
  const headerOffset = headerPtr - basePtr;
  expect(loadU32(committed, headerOffset)).toBe(payloadPtr);
  expect(loadU32(committed, headerOffset + 4)).toBe(bytes.length);

  stack.reset();
  expect(frees).toEqual([payloadPtr]);
  stack.reset();
  expect(frees).toEqual([payloadPtr]);
});

test("CachedCallStack considers byte arguments collectively when applying the cap", () => {
  const { stack, memory, allocations, frees } = createMockStack();
  const argsOffset = stack.allocRawBytes(32);
  const first = new Uint8Array(40000).fill(7);
  const second = new Uint8Array(30000).fill(11);

  stack.allocThenSetArgBytes(argsOffset + 8, first);
  stack.allocThenSetArgBytes(argsOffset + 24, second);
  stack.commitToWasmMemory();

  expect(stack.buffer.byteLength).toBeLessThanOrEqual(CACHE_LIMIT_BYTES);
  expect(memory.storeRawBytes).toHaveBeenCalledTimes(2);
  expect(memory.storeRawBytes.mock.calls[0][1]).toBe(second);
  const transient = allocations.find(([, size]) => size === second.length);
  expect(transient).toBeDefined();
  stack.reset();
  expect(frees).toContain(transient[0]);
});


test("CachedCallStack frees transient bytes after failed and repeated calls", () => {
  const { stack, memory, allocations, frees } = createMockStack();
  const bytes = new Uint8Array(CACHE_LIMIT_BYTES);
  memory.storeRawBytes.mockImplementationOnce(() => {
    throw new Error("copy failed");
  });

  const argsOffset = stack.allocRawBytes(16);
  expect(() => stack.allocThenSetArgBytes(argsOffset + 8, bytes))
    .toThrow("copy failed");
  const failedPtr = allocations[1][0];
  expect(frees).toEqual([failedPtr]);
  stack.reset();
  expect(frees).toEqual([failedPtr]);

  for (let i = 0; i < 2; ++i) {
    const offset = stack.allocRawBytes(16);
    stack.allocThenSetArgBytes(offset + 8, bytes);
    stack.commitToWasmMemory();
    const transientPtr = allocations[2 + i][0];
    stack.reset();
    expect(frees).toContain(transientPtr);
  }
});

test("CachedCallStack reset drops uncommitted fixups and dispose drains allocations", () => {
  const { stack, allocations, frees } = createMockStack();
  const argsOffset = stack.allocRawBytes(16);
  stack.allocThenSetArgBytes(argsOffset + 8, new Uint8Array(CACHE_LIMIT_BYTES));
  const transientPtr = allocations[1][0];

  stack.reset();
  expect(frees).toEqual([transientPtr]);
  stack.dispose();
  expect(frees).toContain(allocations[0][0]);
  stack.dispose();
  expect(frees.filter((ptr) => ptr === allocations[0][0])).toHaveLength(1);
});

test("CachedCallStack uses a null data pointer for an out-of-line empty array", () => {
  const { stack, memory, alloc } = createMockStack();
  stack.allocRawBytes(CACHE_LIMIT_BYTES);
  stack.allocThenSetArgBytes(8, new Uint8Array(0));
  stack.commitToWasmMemory();

  expect(alloc).not.toHaveBeenCalledWith(0);
  const [basePtr, committed] = memory.storeRawBytes.mock.calls[0];
  const headerOffset = loadU32(committed, 8) - basePtr;
  expect(loadU32(committed, headerOffset)).toBe(0);
  expect(loadU32(committed, headerOffset + 4)).toBe(0);
});

test("CachedCallStack rejects sizes that cannot be passed to the Wasm allocator", () => {
  const { stack } = createMockStack();
  expect(() => stack.allocRawBytes(-1)).toThrow(RangeError);
  expect(() => stack.allocRawBytes(1.5)).toThrow(RangeError);
  expect(() => stack.allocRawBytes(0x7ffffff9)).toThrow(RangeError);
  expect(() => stack.allocRawBytes(0x7fffffff)).toThrow(RangeError);
  expect(() => stack.allocRawBytes(0x100000000)).toThrow(RangeError);
});

test("CachedCallStack survives a failed Wasm allocation", () => {
  // Initial allocation fails: nothing to free.
  const free = jest.fn();
  expect(() => new CachedCallStack(
    { wasm32: true, sizeofPtr: () => 4, storeRawBytes: jest.fn() },
    () => 0,
    free
  )).toThrow("Cannot allocate 128 bytes");
  expect(free).not.toHaveBeenCalled();

  // Growth fails: the old allocation stays usable.
  let mock = createMockStack((size, index) => index === 0 ? 1024 : 0);
  expect(() => mock.stack.allocRawBytes(256)).toThrow("Cannot allocate 256 bytes");
  expect(mock.stack.buffer.byteLength).toBe(128);
  expect(mock.frees).toEqual([]);
  expect(mock.stack.allocRawBytes(8)).toBe(0);
  mock.stack.dispose();
  expect(mock.frees).toEqual([1024]);

  // Transient allocation fails: no copy, no dangling temp entry.
  mock = createMockStack((size, index) => index === 0 ? 1024 : 0);
  const argsOffset = mock.stack.allocRawBytes(16);
  expect(() => mock.stack.allocThenSetArgBytes(
    argsOffset + 8,
    new Uint8Array(CACHE_LIMIT_BYTES)
  )).toThrow(`Cannot allocate ${CACHE_LIMIT_BYTES} bytes`);
  expect(mock.memory.storeRawBytes).not.toHaveBeenCalled();
  expect(mock.stack.tempArgs).toHaveLength(0);
  expect(mock.frees).toEqual([]);
  mock.stack.dispose();
  expect(mock.frees).toEqual([1024]);
});

test("CachedCallStack rebuilds a Wasm-backed input after allocation grows memory", () => {
  const wasmMemory = new WebAssembly.Memory({ initial: 2 });
  const memory = new Memory(wasmMemory);
  const frees = [];
  let allocCount = 0;
  const alloc = (size) => {
    if (allocCount++ === 0) return 1024;
    wasmMemory.grow(2);
    return 150000;
  };
  const stack = new CachedCallStack(memory, alloc, (ptr) => frees.push(ptr));
  const source = new Uint8Array(wasmMemory.buffer, 4096, CACHE_LIMIT_BYTES);
  for (let i = 0; i < source.length; ++i) source[i] = i % 251;
  const sourceLength = source.length;
  const wasmSource = {
    buffer: source.buffer,
    byteOffset: source.byteOffset,
    byteLength: source.byteLength,
  };
  const argsOffset = stack.allocRawBytes(16);

  stack.allocThenSetArgBytes(argsOffset + 8, source, wasmSource);

  const result = memory.loadRawBytes(150000, sourceLength);
  expect(result[0]).toBe(0);
  expect(result[250]).toBe(250);
  expect(result[251]).toBe(0);
  expect(result[result.length - 1]).toBe((result.length - 1) % 251);
  expect(source.byteLength).toBe(0);
  stack.reset();
  expect(frees).toContain(150000);
});
