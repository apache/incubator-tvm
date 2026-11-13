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
const { WebGPUContext } = require("../../src/webgpu");

global.GPUBufferUsage = {
  MAP_READ: 1 << 0,
  COPY_DST: 1 << 1,
  COPY_SRC: 1 << 2,
  STORAGE: 1 << 3,
  UNIFORM: 1 << 4,
};
global.GPUMapMode = {
  READ: 1,
};
global.GPUShaderStage = {
  COMPUTE: 1,
};

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function createMockDevice({
  mapAsync = () => Promise.resolve(),
  onSubmittedWorkDone = () => Promise.resolve(),
} = {}) {
  const events = [];
  // Like `events`, but also records compute pass boundaries.
  const trace = [];
  const encoders = [];

  const queue = {
    submit: jest.fn(() => events.push("submit")),
    writeBuffer: jest.fn(() => events.push("writeBuffer")),
    onSubmittedWorkDone: jest.fn(onSubmittedWorkDone),
  };

  const device = {
    queue,
    createCommandEncoder: jest.fn(() => {
      const commands = [];
      const passes = [];
      const encoderId = encoders.length;
      // WebGPU locks a command encoder while one of its passes is open:
      // recording a copy, opening a second pass, or calling finish() in that
      // state is a validation error. The mock throws so tests catch it.
      let openPass = null;
      const assertUnlocked = (what) => {
        if (openPass !== null) {
          throw new Error("mock WebGPU: " + what + " while a compute pass is open");
        }
      };
      const encoder = {
        commands,
        passes,
        beginComputePass: jest.fn(() => {
          assertUnlocked("beginComputePass");
          const pass = {
            ended: false,
            dispatchCount: 0,
            setPipeline: jest.fn(),
            setBindGroup: jest.fn(),
            dispatchWorkgroups: jest.fn(() => {
              if (pass.ended) {
                throw new Error("mock WebGPU: dispatch on an ended compute pass");
              }
              pass.dispatchCount += 1;
              commands.push("compute");
              events.push("compute");
              trace.push("dispatch");
            }),
            end: jest.fn(() => {
              if (pass.ended) {
                throw new Error("mock WebGPU: compute pass ended twice");
              }
              pass.ended = true;
              openPass = null;
              trace.push("endPass");
            }),
          };
          openPass = pass;
          passes.push(pass);
          trace.push("beginPass");
          return pass;
        }),
        copyBufferToBuffer: jest.fn(() => {
          assertUnlocked("copyBufferToBuffer");
          commands.push("copy");
          events.push("copy");
          trace.push("copy");
        }),
        finish: jest.fn(() => {
          assertUnlocked("finish");
          const commandBuffer = { encoderId, commands: commands.slice() };
          events.push("finish");
          trace.push("finish");
          return commandBuffer;
        }),
      };
      encoders.push(encoder);
      return encoder;
    }),
    createBuffer: jest.fn((descriptor) => {
      const mappedData = new ArrayBuffer(descriptor.size);
      return {
        size: descriptor.size,
        destroy: jest.fn(() => events.push("destroy")),
        mapAsync: jest.fn(mapAsync),
        getMappedRange: jest.fn(() => mappedData),
        unmap: jest.fn(),
      };
    }),
    createBindGroupLayout: jest.fn(() => ({})),
    createPipelineLayout: jest.fn(() => ({})),
    createShaderModule: jest.fn(() => ({})),
    createComputePipeline: jest.fn(() => ({})),
    createBindGroup: jest.fn(() => ({})),
    pushErrorScope: jest.fn(),
    popErrorScope: jest.fn(() => Promise.resolve(null)),
    destroy: jest.fn(),
  };

  return { device, queue, events, trace, encoders };
}

function createContext(deviceOptions) {
  const gpu = createMockDevice(deviceOptions);
  const memory = {
    loadRawBytes: jest.fn(),
    viewRawBytes: jest.fn(),
    storeRawBytes: jest.fn(),
  };
  const context = new WebGPUContext(memory, gpu.device);
  const allocate = context.getDeviceAPI("deviceAllocDataSpace");

  return {
    ...gpu,
    context,
    memory,
    source: allocate(64),
    destination: allocate(64),
  };
}

test("compute dispatches and GPU copies share one submission", async () => {
  const { context, device, queue, encoders, source, destination } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const shader = context.createShader(
    {
      name: "main",
      arg_types: [],
      launch_param_tags: [],
    },
    "@compute @workgroup_size(1) fn main() {}"
  );

  shader();
  copyWithinGPU(source, 0, destination, 0, 16);
  copyWithinGPU(destination, 16, source, 32, 16);

  expect(device.createCommandEncoder).toHaveBeenCalledTimes(1);
  expect(queue.submit).not.toHaveBeenCalled();
  expect(encoders[0].commands).toEqual(["compute", "copy", "copy"]);

  await context.sync();

  expect(encoders[0].finish).toHaveBeenCalledTimes(1);
  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(queue.submit.mock.calls[0][0]).toEqual([
    {
      encoderId: 0,
      commands: encoders[0].commands,
    },
  ]);
  expect(queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);

  await context.sync();
  expect(queue.submit).toHaveBeenCalledTimes(1);
});

test("a host write flushes pending GPU copies before writeBuffer", () => {
  const { context, queue, events, source, destination } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const rawBytes = new Uint8Array([1, 2, 3, 4]);

  copyWithinGPU(source, 0, destination, 0, rawBytes.length);
  expect(queue.submit).not.toHaveBeenCalled();

  context.copyRawBytesToBuffer(rawBytes, destination, 4, rawBytes.length);

  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(queue.writeBuffer).toHaveBeenCalledTimes(1);
  expect(events).toEqual(["copy", "finish", "submit", "writeBuffer"]);
});

test("an aligned CPU to GPU copy writes the requested bytes", () => {
  const { context, device, queue, memory, destination } = createContext();
  const copyToGPU = context.getDeviceAPI("deviceCopyToGPU");
  const wasmMemory = new WebAssembly.Memory({ initial: 1 });
  const rawBytes = new Uint8Array(wasmMemory.buffer, 128, 8);
  rawBytes.set([1, 2, 3, 4, 5, 6, 7, 8]);
  memory.viewRawBytes.mockReturnValue(rawBytes);

  copyToGPU(128, destination, 12, rawBytes.length);

  expect(memory.viewRawBytes).toHaveBeenCalledWith(128, rawBytes.length);
  expect(memory.loadRawBytes).not.toHaveBeenCalled();
  expect(queue.writeBuffer.mock.calls[0][2].buffer).toBe(wasmMemory.buffer);
  expect(queue.writeBuffer).toHaveBeenCalledWith(
    device.createBuffer.mock.results[1].value,
    12,
    rawBytes,
    0,
    rawBytes.length
  );
});

test("an unaligned CPU to GPU copy pads the write to four bytes", () => {
  const { context, device, queue, memory, destination } = createContext();
  const copyToGPU = context.getDeviceAPI("deviceCopyToGPU");
  const rawBytes = new Uint8Array([1, 2, 3]);
  memory.viewRawBytes.mockReturnValue(rawBytes);

  copyToGPU(256, destination, 4, rawBytes.length);

  expect(memory.viewRawBytes).toHaveBeenCalledWith(256, rawBytes.length);
  expect(memory.loadRawBytes).not.toHaveBeenCalled();
  expect(queue.writeBuffer).toHaveBeenCalledTimes(1);
  const [buffer, toOffset, data, dataOffset, nbytes] =
    queue.writeBuffer.mock.calls[0];
  expect(buffer).toBe(device.createBuffer.mock.results[1].value);
  expect(toOffset).toBe(4);
  expect(Array.from(data)).toEqual([1, 2, 3, 0]);
  expect(dataOffset).toBe(0);
  expect(nbytes).toBe(4);
});

test("a non-four-byte GPU allocation is rounded up for padded writes", () => {
  const { context, device } = createContext();
  const allocate = context.getDeviceAPI("deviceAllocDataSpace");

  allocate(3);

  expect(device.createBuffer).toHaveBeenLastCalledWith({
    size: 4,
    usage: GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  expect(context.currAllocatedBytes).toBe(64 + 64 + 4);
});

test.each([
  [-1, "destination offset"],
  [0.5, "destination offset"],
  [2, "destination offset"],
])("a CPU to GPU copy rejects invalid offset %p", (offset, message) => {
  const { context, memory, destination } = createContext();
  const copyToGPU = context.getDeviceAPI("deviceCopyToGPU");

  expect(() => copyToGPU(128, destination, offset, 4)).toThrow(message);
  expect(memory.viewRawBytes).not.toHaveBeenCalled();
});

test.each([
  [-1, "source offset"],
  [0.5, "source offset"],
  [2, "source offset"],
])("a GPU readback rejects invalid offset %p", (offset, message) => {
  const { context, source } = createContext();
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");

  expect(() => copyFromGPU(source, offset, 128, 4)).toThrow(message);
});

test("an unaligned GPU readback copies four bytes and stores the logical bytes", async () => {
  const {
    context,
    device,
    memory,
    source,
  } = createContext();
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");
  device.createBuffer.mockImplementationOnce((descriptor) => {
    const mappedData = new Uint8Array([1, 2, 3, 99]).buffer;
    return {
      size: descriptor.size,
      destroy: jest.fn(),
      mapAsync: jest.fn(() => Promise.resolve()),
      getMappedRange: jest.fn(() => mappedData),
      unmap: jest.fn(),
    };
  });

  copyFromGPU(source, 0, 128, 3);
  await context.sync();

  const copyEncoder = device.createCommandEncoder.mock.results[0].value;
  expect(copyEncoder.copyBufferToBuffer).toHaveBeenCalledWith(
    device.createBuffer.mock.results[0].value,
    0,
    device.createBuffer.mock.results[2].value,
    0,
    4,
  );
  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
  expect(memory.storeRawBytes.mock.calls[0][0]).toBe(128);
  expect(Array.from(memory.storeRawBytes.mock.calls[0][1])).toEqual([1, 2, 3]);
});

test("a GPU readback flushes pending copies before its own submission", async () => {
  const {
    context,
    queue,
    events,
    encoders,
    memory,
    source,
    destination,
  } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");

  copyWithinGPU(source, 0, destination, 0, 16);
  copyFromGPU(destination, 0, 128, 16);

  expect(queue.submit).toHaveBeenCalledTimes(2);
  expect(encoders).toHaveLength(2);
  expect(encoders[0].commands).toEqual(["copy"]);
  expect(encoders[1].commands).toEqual(["copy"]);
  expect(events).toEqual(["copy", "finish", "submit", "copy", "finish", "submit"]);

  await context.sync();

  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
  expect(memory.storeRawBytes.mock.calls[0][0]).toBe(128);
  expect(memory.storeRawBytes.mock.calls[0][1]).toHaveLength(16);
  expect(queue.onSubmittedWorkDone).not.toHaveBeenCalled();
});

test("buffer deallocation flushes pending copies before destroy", () => {
  const { context, queue, events, source, destination } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const free = context.getDeviceAPI("deviceFreeDataSpace");

  copyWithinGPU(source, 0, destination, 0, 16);
  free(source);

  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(events).toEqual(["copy", "finish", "submit", "destroy"]);
});

test("drawing flushes pending copies first", () => {
  const { context, queue, events, source, destination } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const canvasRenderManager = {
    draw: jest.fn(() => events.push("draw")),
  };
  context.canvasRenderManager = canvasRenderManager;

  copyWithinGPU(source, 0, destination, 0, 16);
  context.drawImageFromBuffer(destination, 2, 2);

  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(canvasRenderManager.draw).toHaveBeenCalledTimes(1);
  expect(events).toEqual(["copy", "finish", "submit", "draw"]);
});

test("sync awaits a readback and a later batched GPU copy", async () => {
  const readback = createDeferred();
  const queueDone = createDeferred();
  const {
    context,
    queue,
    memory,
    source,
    destination,
  } = createContext({
    mapAsync: () => readback.promise,
    onSubmittedWorkDone: () => queueDone.promise,
  });
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");

  copyFromGPU(source, 0, 128, 16);
  copyWithinGPU(source, 0, destination, 0, 16);

  let syncResolved = false;
  const syncPromise = context.sync().then(() => {
    syncResolved = true;
  });

  expect(queue.submit).toHaveBeenCalledTimes(2);
  expect(queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);

  queueDone.resolve();
  await Promise.resolve();
  expect(syncResolved).toBe(false);
  expect(memory.storeRawBytes).not.toHaveBeenCalled();

  readback.resolve();
  await syncPromise;

  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
  expect(memory.storeRawBytes.mock.calls[0][0]).toBe(128);
  expect(memory.storeRawBytes.mock.calls[0][1]).toHaveLength(16);
});

test("a host write after a readback makes sync wait for the queue", async () => {
  const queueDone = createDeferred();
  const {
    context,
    queue,
    memory,
    source,
    destination,
  } = createContext({
    onSubmittedWorkDone: () => queueDone.promise,
  });
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");

  copyFromGPU(source, 0, 128, 16);
  context.copyRawBytesToBuffer(
    new Uint8Array([1, 2, 3, 4]),
    destination,
    0,
    4
  );

  let syncResolved = false;
  const syncPromise = context.sync().then(() => {
    syncResolved = true;
  });
  expect(queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);

  await Promise.resolve();
  expect(syncResolved).toBe(false);

  queueDone.resolve();
  await syncPromise;

  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
});

test("sync propagates a pending readback failure", async () => {
  const readError = new Error("mapAsync failed");
  const {
    context,
    queue,
    source,
    destination,
  } = createContext({
    mapAsync: () => Promise.reject(readError),
  });
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");

  copyFromGPU(source, 0, 128, 16);
  copyWithinGPU(source, 0, destination, 0, 16);

  await expect(context.sync()).rejects.toBe(readError);
  expect(queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);
});

function createNoArgShader(context, name = "main") {
  return context.createShader(
    {
      name,
      arg_types: [],
      launch_param_tags: [],
    },
    "@compute @workgroup_size(1) fn " + name + "() {}"
  );
}

test("consecutive dispatches share one compute pass", async () => {
  const { context, queue, trace, encoders } = createContext();
  const shader = createNoArgShader(context);

  shader();
  shader();
  shader();

  expect(encoders).toHaveLength(1);
  expect(encoders[0].beginComputePass).toHaveBeenCalledTimes(1);
  expect(encoders[0].passes[0].dispatchCount).toBe(3);
  expect(encoders[0].passes[0].ended).toBe(false);
  expect(queue.submit).not.toHaveBeenCalled();

  await context.sync();

  expect(trace).toEqual([
    "beginPass", "dispatch", "dispatch", "dispatch", "endPass", "finish",
  ]);
  expect(queue.submit).toHaveBeenCalledTimes(1);

  // Dispatches after a flush start a new encoder and a new pass.
  shader();
  shader();
  await context.sync();

  expect(encoders).toHaveLength(2);
  expect(encoders[1].passes).toHaveLength(1);
  expect(encoders[1].passes[0].dispatchCount).toBe(2);
  expect(queue.submit).toHaveBeenCalledTimes(2);
});

test("a GPU copy ends the open compute pass and the next dispatch opens a new one", async () => {
  const { context, trace, encoders, source, destination } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const shader = createNoArgShader(context);

  shader();
  shader();
  copyWithinGPU(source, 0, destination, 0, 16);
  copyWithinGPU(destination, 16, source, 32, 16);
  shader();
  await context.sync();

  expect(encoders).toHaveLength(1);
  expect(encoders[0].beginComputePass).toHaveBeenCalledTimes(2);
  expect(trace).toEqual([
    "beginPass", "dispatch", "dispatch", "endPass",
    "copy", "copy",
    "beginPass", "dispatch", "endPass",
    "finish",
  ]);
});

test.each([
  ["buffer deallocation", ({ context, source }) =>
    context.getDeviceAPI("deviceFreeDataSpace")(source)],
  ["host write", ({ context, destination }) =>
    context.copyRawBytesToBuffer(new Uint8Array([1, 2, 3, 4]), destination, 0, 4)],
  ["GPU readback", ({ context, source }) =>
    context.getDeviceAPI("deviceCopyFromGPU")(source, 0, 128, 16)],
  ["dispose", ({ context }) => context.dispose()],
])("%s ends the open compute pass before finishing the encoder", (_name, flushPoint) => {
  const gpu = createContext();
  createNoArgShader(gpu.context)();
  // The mock throws if the encoder is finished while its pass is open.
  flushPoint(gpu);
  expect(gpu.trace.slice(0, 4)).toEqual(["beginPass", "dispatch", "endPass", "finish"]);
});

test("a launch that throws leaves the shared pass usable and closable", async () => {
  const { context, queue, trace } = createContext();
  const shader = createNoArgShader(context);

  // Wrong argument count: the launch throws after the pass has been opened.
  expect(() => shader(1)).toThrow();
  shader();
  await context.sync();

  expect(trace).toEqual(["beginPass", "dispatch", "endPass", "finish"]);
  expect(queue.submit).toHaveBeenCalledTimes(1);
});
