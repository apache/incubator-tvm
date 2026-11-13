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
/**
 * Classes to manipulate Wasm memories.
 */
import { Pointer, PtrOffset, SizeOf } from "./ctypes";
import { Disposable } from "./types";
import { StringToUint8Array } from "./support";

import * as ctypes from "./ctypes";

const CALL_STACK_CACHE_LIMIT_BYTES = 64 * 1024;
// TVMWasmAllocSpace takes a signed int and rounds `(size + 7) / 8` in C++.
// Keep the input at the largest aligned value that cannot overflow that sum.
const MAX_WASM_ALLOCATION_BYTES = 0x7ffffff8;

export interface WasmByteArraySource {
  buffer: ArrayBuffer | SharedArrayBuffer;
  byteOffset: number;
  byteLength: number;
}

class WasmAllocation implements Disposable {
  private ptr: Pointer;
  private freeSpace: ctypes.FTVMWasmFreeSpace;

  constructor(ptr: Pointer, freeSpace: ctypes.FTVMWasmFreeSpace) {
    this.ptr = ptr;
    this.freeSpace = freeSpace;
  }

  dispose(): void {
    if (this.ptr !== 0) {
      this.freeSpace(this.ptr);
      this.ptr = 0;
    }
  }
}

/**
 * Wasm Memory wrapper to perform JS side raw memory access.
 */
export class Memory {
  memory: WebAssembly.Memory;
  wasm32 = true;
  private buffer: ArrayBuffer | SharedArrayBuffer;
  private viewU8: Uint8Array;
  private viewU16: Uint16Array;
  private viewI32: Int32Array;
  private viewU32: Uint32Array;
  private viewF32: Float32Array;
  private viewF64: Float64Array;

  constructor(memory: WebAssembly.Memory) {
    this.memory = memory;
    this.buffer = this.memory.buffer;
    this.viewU8 = new Uint8Array(this.buffer);
    this.viewU16 = new Uint16Array(this.buffer);
    this.viewI32 = new Int32Array(this.buffer);
    this.viewU32 = new Uint32Array(this.buffer);
    this.viewF32 = new Float32Array(this.buffer);
    this.viewF64 = new Float64Array(this.buffer);
  }

  loadU8(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    return this.viewU8[ptr >> 0];
  }

  loadU16(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    return this.viewU16[ptr >> 1];
  }

  loadU32(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    return this.viewU32[ptr >> 2];
  }

  loadI32(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    return this.viewI32[ptr >> 2];
  }

  loadI64(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    const base = ptr >> 2;
    // assumes little endian, for now truncate high.
    return this.viewI32[base];
  }

  loadU64(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    // WebAssembly is little-endian. Reject values JavaScript cannot represent exactly.
    const base = ptr >> 2;
    const value = this.viewU32[base] + this.viewU32[base + 1] * 0x100000000;
    if (!Number.isSafeInteger(value)) {
      throw new Error("Cannot represent uint64 value as a JavaScript number");
    }
    return value;
  }

  loadF32(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    return this.viewF32[ptr >> 2];
  }

  loadF64(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    return this.viewF64[ptr >> 3];
  }

  loadPointer(ptr: Pointer): Pointer {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    if (this.wasm32) {
      return this.loadU32(ptr);
    } else {
      return this.loadI64(ptr);
    }
  }
  loadUSize(ptr: Pointer): Pointer {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    if (this.wasm32) {
      return this.loadU32(ptr);
    } else {
      return this.loadI64(ptr);
    }
  }
  sizeofPtr(): number {
    return this.wasm32 ? SizeOf.I32 : SizeOf.I64;
  }
  /**
   * Load raw bytes from ptr.
   * @param ptr The head address
   * @param numBytes The number
   */
  loadRawBytes(ptr: Pointer, numBytes: number): Uint8Array {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    const result = new Uint8Array(numBytes);
    result.set(this.viewU8.subarray(ptr, ptr + numBytes));
    return result;
  }
  /**
   * Return a borrowed view of raw bytes in Wasm memory.
   *
   * The returned view aliases the current WebAssembly.Memory buffer and must
   * not be retained across a call that can grow the memory.
   *
   * @param ptr The head address.
   * @param numBytes The number of bytes.
   */
  viewRawBytes(ptr: Pointer, numBytes: number): Uint8Array {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    if (!Number.isSafeInteger(ptr) || ptr < 0) {
      throw new Error(`Invalid Wasm memory pointer: ${ptr}`);
    }
    if (!Number.isSafeInteger(numBytes) || numBytes < 0) {
      throw new Error(`Invalid Wasm memory byte length: ${numBytes}`);
    }
    const end = ptr + numBytes;
    if (!Number.isSafeInteger(end) || end > this.viewU8.byteLength) {
      throw new Error(
        `Wasm memory range [${ptr}, ${end}) exceeds memory size ` +
        `${this.viewU8.byteLength}`,
      );
    }
    return this.viewU8.subarray(ptr, end);
  }
  /**
   * Load null-terminated C-string from ptr.
   * @param ptr The head address
   */
  loadCString(ptr: Pointer): string {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    // NOTE: the views are still valid for read.
    const ret = [];
    let ch = 1;
    while (ch != 0) {
      ch = this.viewU8[ptr];
      if (ch != 0) {
        ret.push(String.fromCharCode(ch));
      }
      ++ptr;
    }
    return ret.join("");
  }
  /**
   * Store raw bytes to the ptr.
   * @param ptr The head address.
   * @param bytes The bytes content.
   */
  storeRawBytes(ptr: Pointer, bytes: Uint8Array): void {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    this.viewU8.set(bytes, ptr);
  }
  // the following functions are related to TVM FFI
  /**
   * Load the object type index from the object handle.
   * @param objectHandle The handle of the object.
   * @returns The object type index.
   */
  loadObjectTypeIndex(objectHandle: Pointer): number {
    // The object layout is [ref_counter (i64), type_index (i32), ...].
    return this.loadI32(objectHandle + SizeOf.I64);
  }
  /**
   * Load the type key from the type info pointer.
   * @param typeInfoPtr The pointer to the type info.
   * @returns The type key.
   */
  loadTypeInfoTypeKey(typeInfoPtr: Pointer): string {
    const typeKeyPtr = typeInfoPtr + 2 * SizeOf.I32;
    return this.loadByteArrayAsString(typeKeyPtr);
  }
  /**
   * Load small string from value pointer.
   * @param ffiAnyPtr The pointer to the value.
   * @returns The small string.
   */
  loadSmallStr(ffiAnyPtr: Pointer): string {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    const sizePtr = ffiAnyPtr + SizeOf.I32;
    const length = this.loadU32(sizePtr);
    const dataPtr = ffiAnyPtr + SizeOf.I32 + SizeOf.I32;
    const ret = [];
    for (let i = 0; i < length; i++) {
      ret.push(String.fromCharCode(this.viewU8[dataPtr + i]));
    }
    return ret.join("");
  }
  /**
   * Load small bytes from value pointer.
   * @param ffiAnyPtr
   */
  loadSmallBytes(ffiAnyPtr: Pointer): Uint8Array {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    const sizePtr = ffiAnyPtr + SizeOf.I32;
    const length = this.loadU32(sizePtr);
    const dataPtr = ffiAnyPtr + SizeOf.I32 + SizeOf.I32;
    const result = new Uint8Array(length);
    result.set(this.viewU8.subarray(dataPtr, dataPtr + length));
    return result;
  }
  /**
   * Load bytearray as string from ptr.
   * @param byteArrayPtr The head address of the bytearray.
   */
  loadByteArrayAsString(byteArrayPtr: Pointer): string {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    const ptr = this.loadPointer(byteArrayPtr);
    const length = this.loadUSize(byteArrayPtr + this.sizeofPtr());
    // NOTE: the views are still valid for read.
    const ret = [];
    for (let i = 0; i < length; i++) {
      ret.push(String.fromCharCode(this.viewU8[ptr + i]));
    }
    return ret.join("");
  }
  /**
   * Load bytearray as bytes from ptr.
   * @param byteArrayPtr The head address of the bytearray.
   */
  loadByteArrayAsBytes(byteArrayPtr: Pointer): Uint8Array {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    const ptr = this.loadPointer(byteArrayPtr);
    const length = this.loadUSize(byteArrayPtr + this.sizeofPtr());
    const result = new Uint8Array(length);
    result.set(this.viewU8.subarray(ptr, ptr + length));
    return result;
  }
  // private functions
  /**
   * Update memory view after the memory growth.
   */
  private updateViews(): void {
    this.buffer = this.memory.buffer;
    this.viewU8 = new Uint8Array(this.buffer);
    this.viewU16 = new Uint16Array(this.buffer);
    this.viewI32 = new Int32Array(this.buffer);
    this.viewU32 = new Uint32Array(this.buffer);
    this.viewF32 = new Float32Array(this.buffer);
    this.viewF64 = new Float64Array(this.buffer);
  }
}

/**
 * Auxiliary call stack for the FFI calls.
 *
 * Lifecyle of a call stack.
 * - Calls into allocXX to allocate space, mixed with storeXXX to store data.
 * - Calls into ptrFromOffset, no further allocation(as ptrFromOffset can change),
 *   can still call into storeXX
 * - Calls into commitToWasmMemory once.
 * - reset.
 */
export class CachedCallStack implements Disposable {
  /** List of temporay arguments that can be disposed during reset. */
  tempArgs: Array<Disposable> = [];

  private memory: Memory;
  private cAllocSpace: ctypes.FTVMWasmAllocSpace;
  private cFreeSpace: ctypes.FTVMWasmFreeSpace;

  private buffer: ArrayBuffer;
  private viewU8: Uint8Array;
  private viewI32: Int32Array;
  private viewU32: Uint32Array;
  private viewF64: Float64Array;

  private stackTop: PtrOffset = 0;
  private basePtr: Pointer = 0;

  private addressToSetTargetValue: Array<[PtrOffset, PtrOffset]> = [];

  constructor(
    memory: Memory,
    allocSpace: ctypes.FTVMWasmAllocSpace,
    freeSpace: ctypes.FTVMWasmFreeSpace
  ) {
    const initCallStackSize = 128;
    this.memory = memory;
    this.cAllocSpace = allocSpace;
    this.cFreeSpace = freeSpace;
    this.buffer = new ArrayBuffer(initCallStackSize);
    this.basePtr = this.cAllocSpace(initCallStackSize);
    if (this.basePtr === 0) {
      throw new Error(`Cannot allocate ${initCallStackSize} bytes in Wasm memory`);
    }
    this.viewU8 = new Uint8Array(this.buffer);
    this.viewI32 = new Int32Array(this.buffer);
    this.viewU32 = new Uint32Array(this.buffer);
    this.viewF64 = new Float64Array(this.buffer);
    this.updateViews();
  }

  dispose(): void {
    this.reset();
    if (this.basePtr != 0) {
      this.cFreeSpace(this.basePtr);
      this.basePtr = 0;
    }
  }
  /**
   * Rest the call stack so that it can be reused again.
   */
  reset(): void {
    this.stackTop = 0;
    this.addressToSetTargetValue.length = 0;
    while (this.tempArgs.length != 0) {
      (this.tempArgs.pop() as Disposable).dispose();
    }
  }

  /**
   * Commit all the cached data to WasmMemory.
   * This function can only be called once.
   * No further store function should be called.
   *
   * @param nbytes Number of bytes to be stored.
   */
  commitToWasmMemory(nbytes: number = this.stackTop): void {
    // commit all pointer values.
    while (this.addressToSetTargetValue.length != 0) {
      const [targetOffset, valueOffset] = this.addressToSetTargetValue.pop() as [
        number,
        number
      ];
      this.storePtr(targetOffset, this.ptrFromOffset(valueOffset));
    }
    this.memory.storeRawBytes(this.basePtr, this.viewU8.subarray(0, nbytes));
  }

  /**
   * Allocate space by number of bytes
   * @param nbytes Number of bytes.
    * Note: This function always allocate space that aligns to 64bit.
   */
  allocRawBytes(nbytes: number): PtrOffset {
    // always aligns to 64bit
    nbytes = this.alignToEight(nbytes);

    const requiredSize = this.checkedAdd(this.stackTop, nbytes);

    if (requiredSize > this.buffer.byteLength) {
      const doubledSize = Math.min(
        this.buffer.byteLength * 2,
        MAX_WASM_ALLOCATION_BYTES
      );
      let newSize = Math.max(doubledSize, requiredSize);
      if (this.buffer.byteLength < CALL_STACK_CACHE_LIMIT_BYTES &&
          requiredSize <= CALL_STACK_CACHE_LIMIT_BYTES) {
        newSize = Math.min(newSize, CALL_STACK_CACHE_LIMIT_BYTES);
      } else if (this.buffer.byteLength <= CALL_STACK_CACHE_LIMIT_BYTES &&
                 requiredSize > CALL_STACK_CACHE_LIMIT_BYTES) {
        // Headers and non-byte arguments may cross the soft cache cap. Grow
        // only to the requested size instead of doubling past the cap.
        newSize = requiredSize;
      }
      const newBuffer = new ArrayBuffer(newSize);
      new Uint8Array(newBuffer).set(this.viewU8);
      const newBasePtr = this.cAllocSpace(newSize);
      if (newBasePtr === 0) {
        throw new Error(`Cannot allocate ${newSize} bytes in Wasm memory`);
      }
      this.cFreeSpace(this.basePtr);
      this.basePtr = newBasePtr;
      this.buffer = newBuffer;
      this.updateViews();
    }
    const retOffset = this.stackTop;
    this.stackTop = requiredSize;
    return retOffset;
  }

  /**
   * Allocate space for pointers.
   * @param count Number of pointers.
   * @returns The allocated pointer array.
   */
  allocPtrArray(count: number): PtrOffset {
    return this.allocRawBytes(this.memory.sizeofPtr() * count);
  }

  /**
   * Get the real pointer from offset values.
   * Note that the returned value becomes obsolete if alloc is called on the stack.
   * @param offset The allocated offset.
   */
  ptrFromOffset(offset: PtrOffset): Pointer {
    return this.basePtr + offset;
  }

  // Store APIs
  storePtr(offset: PtrOffset, value: Pointer): void {
    if (this.memory.wasm32) {
      this.storeU32(offset, value);
    } else {
      this.storeI64(offset, value);
    }
  }

  storeUSize(offset: PtrOffset, value: Pointer): void {
    if (this.memory.wasm32) {
      this.storeU32(offset, value);
    } else {
      this.storeI64(offset, value);
    }
  }

  storeI32(offset: PtrOffset, value: number): void {
    this.viewI32[offset >> 2] = value;
  }

  storeU32(offset: PtrOffset, value: number): void {
    this.viewU32[offset >> 2] = value;
  }

  storeI64(offset: PtrOffset, value: number): void {
    // For now, just store as 32bit
    // NOTE: wasm always uses little endian.
    const low = value & 0xffffffff;
    const base = offset >> 2;
    this.viewI32[base] = low;
    // sign extend
    this.viewI32[base + 1] = value < 0 ? -1 : 0;
  }

  storeF64(offset: PtrOffset, value: number): void {
    this.viewF64[offset >> 3] = value;
  }

  storeRawBytes(offset: PtrOffset, bytes: Uint8Array): void {
    this.viewU8.set(bytes, offset);
  }

  /**
   * Allocate a byte array for a string and return the offset of the byte array.
   * @param data The string to allocate.
   * @returns The offset of the byte array.
   */
  allocByteArrayForString(data: string): PtrOffset {
    const dataUint8: Uint8Array = StringToUint8Array(data);
    // Note: size of size_t equals sizeof ptr.
    const headerOffset = this.allocRawBytes(this.memory.sizeofPtr() * 2);
    const dataOffset = this.allocRawBytes(dataUint8.length);
    this.storeUSize(headerOffset + this.memory.sizeofPtr(), data.length);
    this.storeRawBytes(dataOffset, dataUint8);
    this.addressToSetTargetValue.push([headerOffset, dataOffset]);
    return headerOffset;
  }
  /**
   * Allocate then set C-String pointer to the offset.
   * This function will call into allocBytes to allocate necessary data.
   * The address won't be set immediately(because the possible change of basePtr)
   * and will be filled when we commit the data.
   *
   * @param offset The offset to set ot data pointer.
   * @param data The string content.
   */
  allocThenSetArgString(offset: PtrOffset, data: string): void {
    const dataUint8: Uint8Array = StringToUint8Array(data);
    const strOffset = this.allocRawBytes(dataUint8.length);
    this.storeRawBytes(strOffset, dataUint8);
    this.addressToSetTargetValue.push([offset, strOffset]);
  }
  /**
   * Allocate then set the argument location with a TVMByteArray.
   * Allocate new temporary space for bytes.
   *
   * @param offset The offset to set ot data pointer.
   * @param data The string content.
   */
  allocThenSetArgBytes(
    offset: PtrOffset,
    data: Uint8Array,
    wasmSource?: WasmByteArraySource
  ): void {
    if (wasmSource === undefined && data.buffer === this.memory.memory.buffer) {
      wasmSource = {
        buffer: data.buffer,
        byteOffset: data.byteOffset,
        byteLength: data.byteLength,
      };
    }
    // Note: size of size_t equals sizeof ptr.
    const dataLength = wasmSource?.byteLength ?? data.length;
    const headerSize = this.alignToEight(this.memory.sizeofPtr() * 2);
    const inlineDataSize = this.alignToEight(dataLength);
    const cacheSizeWithHeader = this.checkedAdd(this.stackTop, headerSize);
    const keepDataInline = cacheSizeWithHeader <= CALL_STACK_CACHE_LIMIT_BYTES &&
      inlineDataSize <= CALL_STACK_CACHE_LIMIT_BYTES - cacheSizeWithHeader;
    const headerOffset = this.allocRawBytes(this.memory.sizeofPtr() * 2);
    this.storeUSize(headerOffset + this.memory.sizeofPtr(), dataLength);

    this.addressToSetTargetValue.push([offset, headerOffset]);
    if (keepDataInline) {
      const dataOffset = this.allocRawBytes(dataLength);
      this.storeRawBytes(dataOffset, this.resolveByteSource(data, wasmSource));
      this.addressToSetTargetValue.push([headerOffset, dataOffset]);
    } else {
      const dataPtr = dataLength === 0 ? 0 : this.allocTransientBytes(data, wasmSource);
      this.storePtr(headerOffset, dataPtr);
    }
  }

  /** Allocate bytes directly in Wasm memory for the duration of one FFI call. */
  private allocTransientBytes(
    data: Uint8Array,
    wasmSource?: WasmByteArraySource
  ): Pointer {
    const dataLength = wasmSource?.byteLength ?? data.length;
    this.checkAllocationSize(dataLength);

    const dataPtr = this.cAllocSpace(dataLength);
    if (dataPtr === 0) {
      throw new Error(`Cannot allocate ${dataLength} bytes in Wasm memory`);
    }

    const allocation = new WasmAllocation(dataPtr, this.cFreeSpace);
    this.tempArgs.push(allocation);
    try {
      this.memory.storeRawBytes(dataPtr, this.resolveByteSource(data, wasmSource));
    } catch (err) {
      this.tempArgs.pop();
      allocation.dispose();
      throw err;
    }
    return dataPtr;
  }

  private resolveByteSource(
    data: Uint8Array,
    wasmSource?: WasmByteArraySource
  ): Uint8Array {
    if (wasmSource !== undefined && wasmSource.buffer !== this.memory.memory.buffer) {
      return new Uint8Array(
        this.memory.memory.buffer,
        wasmSource.byteOffset,
        wasmSource.byteLength
      );
    }
    return data;
  }

  private alignToEight(nbytes: number): number {
    this.checkAllocationSize(nbytes);
    const remainder = nbytes % 8;
    const aligned = remainder === 0 ? nbytes : nbytes + 8 - remainder;
    this.checkAllocationSize(aligned);
    return aligned;
  }

  private checkedAdd(lhs: number, rhs: number): number {
    const result = lhs + rhs;
    this.checkAllocationSize(result);
    return result;
  }

  private checkAllocationSize(nbytes: number): void {
    if (!Number.isSafeInteger(nbytes) || nbytes < 0 || nbytes > MAX_WASM_ALLOCATION_BYTES) {
      throw new RangeError(
        `Wasm allocation size must be an integer between 0 and ${MAX_WASM_ALLOCATION_BYTES}, ` +
        `but received ${nbytes}`
      );
    }
  }

  /**
   * Update internal cache views.
   */
  private updateViews(): void {
    this.viewU8 = new Uint8Array(this.buffer);
    this.viewI32 = new Int32Array(this.buffer);
    this.viewU32 = new Uint32Array(this.buffer);
    this.viewF64 = new Float64Array(this.buffer);
  }
}
