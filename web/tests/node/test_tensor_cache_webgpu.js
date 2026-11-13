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
const fs = require("fs");
const path = require("path");
const tvmjs = require("../../dist/tvmjs.bundle");

global.GPUBufferUsage = {
  MAP_READ: 1 << 0,
  COPY_DST: 1 << 1,
  COPY_SRC: 1 << 2,
  STORAGE: 1 << 3,
  UNIFORM: 1 << 4,
};

const wasmSource = fs.readFileSync(
  path.join(__dirname, "../../dist/wasm/tvmjs_runtime.wasm"),
);

function createInstance() {
  return new tvmjs.Instance(
    new WebAssembly.Module(wasmSource),
    tvmjs.createPolyfillWASI(),
  );
}

function createDeferred() {
  let resolve;
  const promise = new Promise((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

function createMockGPUDevice({
  detachWriteSources = false,
  lost = new Promise(() => {}),
} = {}) {
  const buffers = [];
  const writes = [];
  const queue = {
    submit: jest.fn(),
    onSubmittedWorkDone: jest.fn(() => Promise.resolve()),
    writeBuffer: jest.fn(
      (buffer, bufferOffset, data, dataOffset = 0, size = data.byteLength) => {
        const source = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        const snapshot = source.slice(dataOffset, dataOffset + size);
        if (bufferOffset + snapshot.byteLength > buffer.contents.byteLength) {
          throw new RangeError(
            `write of ${snapshot.byteLength} bytes at ${bufferOffset} exceeds ` +
            `${buffer.contents.byteLength}-byte mock buffer`,
          );
        }
        new Uint8Array(buffer.contents).set(snapshot, bufferOffset);
        writes.push({ buffer, bufferOffset, data, dataOffset, size, snapshot });
        if (detachWriteSources) {
          structuredClone(data.buffer, { transfer: [data.buffer] });
        }
      },
    ),
  };
  const device = {
    queue,
    lost,
    addEventListener: jest.fn(),
    pushErrorScope: jest.fn(),
    popErrorScope: jest.fn(() => Promise.resolve(null)),
    createBuffer: jest.fn((descriptor) => {
      const buffer = {
        size: descriptor.size,
        contents: new ArrayBuffer(descriptor.size),
        destroy: jest.fn(),
      };
      buffers.push(buffer);
      return buffer;
    }),
    destroy: jest.fn(),
  };
  return { device, buffers, writes };
}

function createArtifactCache(manifest, shard) {
  return {
    hasAllKeys: async () => true,
    addToCache: async () => {},
    deleteInCache: async () => {},
    fetchWithCache: async (_url, storeType) => {
      return storeType === "json" ? manifest : shard;
    },
  };
}

test("an external owner can defer disposal after device loss", async () => {
  const lost = createDeferred();
  const tvm = createInstance();
  const dispose = jest.spyOn(tvm, "dispose");
  const gpu = createMockGPUDevice({ lost: lost.promise });
  const log = jest.spyOn(console, "error").mockImplementation(() => {});
  try {
    tvm.initWebGPU(gpu.device);
    tvm.setDeviceLostAutoDispose(false);

    lost.resolve({ reason: "unknown", message: "test device loss" });
    await lost.promise;
    await Promise.resolve();

    expect(dispose).not.toHaveBeenCalled();

    tvm.dispose();
    expect(dispose).toHaveBeenCalledTimes(1);
  } finally {
    if (dispose.mock.calls.length === 0) {
      tvm.dispose();
    }
    dispose.mockRestore();
    log.mockRestore();
  }
});

test("WebGPU tensor cache uploads pass-through records and decodes BF16 in place", async () => {
  const tvm = createInstance();
  const gpu = createMockGPUDevice();
  tvm.initWebGPU(gpu.device);

  const records = [
    {
      name: "test.direct.raw_uint32",
      shape: [2],
      dtype: "uint32",
      format: "raw",
      byteOffset: 0,
      nbytes: 8,
    },
    {
      name: "test.direct.passthrough_float16",
      shape: [2],
      dtype: "float16",
      format: "f32-to-bf16",
      byteOffset: 8,
      nbytes: 4,
    },
    {
      name: "test.direct.raw_float32",
      shape: [1],
      dtype: "float32",
      format: "raw",
      byteOffset: 12,
      nbytes: 4,
    },
    {
      name: "test.decode.packed_bf16",
      shape: [2],
      dtype: "float32",
      format: "f32-to-bf16",
      byteOffset: 16,
      nbytes: 4,
    },
  ];
  const shard = Uint8Array.from([
    1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16,
    0x80, 0x3f, 0x00, 0xc0,
  ]).buffer;
  const manifest = {
    metadata: {},
    records: [{
      dataPath: "params.bin",
      format: "raw-shard",
      nbytes: shard.byteLength,
      records,
    }],
  };

  const originalDecode = tvm.ctx.arrayDecodeStorage;
  const originalInplace = tvm.ctx.arrayDecodeBF16ToF32Inplace;
  const decode = jest.fn((...args) => originalDecode(...args));
  const inplace = jest.fn((...args) => originalInplace(...args));
  tvm.ctx.arrayDecodeStorage = decode;
  tvm.ctx.arrayDecodeBF16ToF32Inplace = inplace;
  try {
    await tvm.fetchTensorCache(
      "https://example.test/model/",
      tvm.webgpu(),
      { artifactCache: createArtifactCache(manifest, shard) },
    );

    expect(decode).not.toHaveBeenCalled();
    expect(inplace).toHaveBeenCalledTimes(1);
    expect(gpu.writes.map((write) => Array.from(write.snapshot))).toEqual([
      [1, 2, 3, 4, 5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
      [0, 0, 128, 63, 0, 0, 0, 192],
    ]);
  } finally {
    tvm.ctx.arrayDecodeStorage = originalDecode;
    tvm.ctx.arrayDecodeBF16ToF32Inplace = originalInplace;
    tvm.tensorCacheClear();
    tvm.dispose();
  }
});

test("direct upload snapshots a borrowed shard view before synchronization", async () => {
  const tvm = createInstance();
  const gpu = createMockGPUDevice({ detachWriteSources: true });
  tvm.initWebGPU(gpu.device);

  const shard = new Uint8Array([1, 2, 3, 4]).buffer;
  const manifest = {
    metadata: {},
    records: [{
      dataPath: "params.bin",
      format: "raw-shard",
      nbytes: shard.byteLength,
      records: [{
        name: "test.direct.release_before_sync",
        shape: [1],
        dtype: "uint32",
        format: "raw",
        byteOffset: 0,
        nbytes: 4,
      }],
    }],
  };

  try {
    await tvm.fetchTensorCache(
      "https://example.test/model/",
      tvm.webgpu(),
      { artifactCache: createArtifactCache(manifest, shard) },
    );

    expect(gpu.writes).toHaveLength(1);
    expect(gpu.writes[0].data.buffer).toBe(shard);
    expect(gpu.writes[0].data.byteLength).toBe(0);
    expect(Array.from(gpu.writes[0].snapshot)).toEqual([1, 2, 3, 4]);
  } finally {
    tvm.tensorCacheClear();
    tvm.dispose();
  }
});

test("direct tensor-cache upload rejects a record with the wrong size", async () => {
  const log = jest.spyOn(console, "log").mockImplementation(() => {});
  const tvm = createInstance();
  const gpu = createMockGPUDevice();
  tvm.initWebGPU(gpu.device);
  const shard = new Uint8Array([1, 2, 3, 4]).buffer;
  const manifest = {
    metadata: {},
    records: [{
      dataPath: "params.bin",
      format: "raw-shard",
      nbytes: shard.byteLength,
      records: [{
        name: "test.direct.invalid_size",
        shape: [2],
        dtype: "uint32",
        format: "raw",
        byteOffset: 0,
        nbytes: 4,
      }],
    }],
  };

  try {
    await expect(tvm.fetchTensorCache(
      "https://example.test/model/",
      tvm.webgpu(),
      { artifactCache: createArtifactCache(manifest, shard) },
    )).rejects.toThrow("nbytes=8");
    expect(gpu.writes).toHaveLength(0);
    expect(gpu.buffers).toHaveLength(1);
    expect(gpu.buffers[0].destroy).toHaveBeenCalledTimes(1);
  } finally {
    tvm.tensorCacheClear();
    tvm.dispose();
    log.mockRestore();
  }
});

test("CPU tensor-cache loading keeps the decoder path", async () => {
  const tvm = createInstance();
  const shard = new Uint8Array([
    1, 2, 3, 4,
    5, 6, 7, 8,
  ]).buffer;
  const manifest = {
    metadata: {},
    records: [{
      dataPath: "params.bin",
      format: "raw-shard",
      nbytes: shard.byteLength,
      records: [
        {
          name: "test.cpu.raw",
          shape: [4],
          dtype: "uint8",
          format: "raw",
          byteOffset: 0,
          nbytes: 4,
        },
        {
          name: "test.cpu.passthrough",
          shape: [4],
          dtype: "uint8",
          format: "f32-to-bf16",
          byteOffset: 4,
          nbytes: 4,
        },
      ],
    }],
  };
  const originalDecode = tvm.ctx.arrayDecodeStorage;
  const decode = jest.fn((...args) => originalDecode(...args));
  tvm.ctx.arrayDecodeStorage = decode;

  try {
    await tvm.fetchTensorCache(
      "https://example.test/model/",
      tvm.cpu(),
      { artifactCache: createArtifactCache(manifest, shard) },
    );

    expect(decode).toHaveBeenCalledTimes(2);
    tvm.withNewScope(() => {
      expect(Array.from(tvm.tensorCacheGet("test.cpu.raw").toRawBytes()))
        .toEqual([1, 2, 3, 4]);
      expect(Array.from(tvm.tensorCacheGet("test.cpu.passthrough").toRawBytes()))
        .toEqual([5, 6, 7, 8]);
    });
  } finally {
    tvm.ctx.arrayDecodeStorage = originalDecode;
    tvm.tensorCacheClear();
    tvm.dispose();
  }
});

test("copyFromRawBytes uses packed tensor size and tensor byte offset", () => {
  const tvm = createInstance();
  const gpu = createMockGPUDevice();
  tvm.initWebGPU(gpu.device);

  tvm.withNewScope(() => {
    const packed = tvm.empty([3], "uint4", tvm.webgpu());
    expect(() => packed.copyFromRawBytes(new Uint8Array(3))).toThrow("nbytes=2");

    const base = tvm.empty([2], "uint32", tvm.webgpu());
    const view = tvm.ctx.tensorCreateView(
      base,
      tvm.ctx.makeShapeTuple(new tvmjs.Scalar(1, "int")),
      "uint32",
      new tvmjs.Scalar(4, "int"),
    );
    view.copyFromRawBytes(new Uint8Array([21, 22, 23, 24]));

    expect(gpu.writes).toHaveLength(1);
    expect(gpu.writes[0].bufferOffset).toBe(4);
    expect(Array.from(gpu.writes[0].snapshot)).toEqual([21, 22, 23, 24]);
  });
  tvm.dispose();
});

test("Tensor parses byte offsets as checked unsigned 64-bit values", () => {
  const tvm = createInstance();

  tvm.withNewScope(() => {
    const tensor = tvm.empty([1], "uint32", tvm.cpu());
    // DLTensor::byte_offset starts at byte 32 in the wasm32 ABI.
    const byteOffsetPtr = tensor.dltensor + 32;
    const words = new Uint32Array(tvm.memory.memory.buffer);
    const wordOffset = byteOffsetPtr >>> 2;
    const original = [words[wordOffset], words[wordOffset + 1]];

    try {
      words.set([0x80000001, 1], wordOffset);
      const parsed = new tvmjs.Tensor(
        tensor.dltensor,
        tensor.lib,
        tensor.ctx,
        true,
      );
      expect(parsed.byteOffset).toBe(0x180000001);

      words.set([0, 0x200000], wordOffset);
      expect(() => new tvmjs.Tensor(
        tensor.dltensor,
        tensor.lib,
        tensor.ctx,
        true,
      )).toThrow("Cannot represent uint64 value");
    } finally {
      words.set(original, wordOffset);
    }
  });
  tvm.dispose();
});
