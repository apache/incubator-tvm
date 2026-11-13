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
// Helper tools to enable asynctify handling
// Thie following code is used to support wrapping of
// functins that can have async await calls in the backend runtime
// reference
// - https://kripken.github.io/blog/wasm/2019/07/16/asyncify.html
// - https://github.com/GoogleChromeLabs/asyncify
import { assert, isPromise } from "./support";

/**
 * enums to check the current state of asynctify
 */
const enum AsyncifyStateKind {
  None = 0,
  Unwinding = 1,
  Rewinding = 2
}

/** The start location of asynctify stack data */
const ASYNCIFY_DATA_ADDR = 16;
/** The data start of stack rewind/unwind */
const ASYNCIFY_DATA_START = ASYNCIFY_DATA_ADDR + 8;
/** The data end of stack rewind/unwind */
const ASYNCIFY_DATA_END = 1024;

/** Hold asynctify handler instance that runtime can use */
export class AsyncifyHandler {
  /** exports from wasm */
  private exports: Record<string, Function>;
  /** current state kind */
  private state: AsyncifyStateKind = AsyncifyStateKind.None;
  /** The stored value before unwind */
  private storedPromiseBeforeUnwind : Promise<any> = null;
  // NOTE: asynctify do not work with exceptions
  // this implementation here is mainly for possible future compact
  /** The stored value that is resolved */
  private storedValueBeforeRewind: any = null;
  /** The stored exception */
  private storedExceptionBeforeRewind: any = null;

  constructor(exports: Record<string, Function>, memory: WebAssembly.Memory) {
    this.exports = exports;
    this.initMemory(memory);
  }

  // NOTE: wrapImport and wrapExport are closely related to each other
  // We mark the logical jump pt in comments to increase the readability
  /**
   * Whether the wasm enables asynctify
   * @returns Whether the wasm enables asynctify
   */
  enabled(): boolean {
    return this.exports.asyncify_stop_rewind !== undefined;
  }

  /**
   * Get the current asynctify state
   *
   * @returns The current asynctify state
   */
  isNormalStackState(): boolean {
    return this.state == AsyncifyStateKind.None;
  }

  /**
   * Get the current asynctify state
   *
   * @returns The current asynctify state
   */
  getState(): AsyncifyStateKind {
    return this.state;
  }

  /**
   * Wrap a function that can be used as import of the wasm asynctify layer
   *
   * @param func The input import function
   * @returns The wrapped function that can be registered to the system
   */
  wrapImport(func: (...args: Array<any>) => any): (...args: Array<any>) => any {
    return (...args: any) => {
      // this is being called second time
      // where we are rewinding the stack
      if (this.getState() == AsyncifyStateKind.Rewinding) {
        // JUMP-PT-REWIND: rewind will jump to this pt
        // while rewinding the stack
        this.stopRewind();
        // the value has been resolved
        if (this.storedValueBeforeRewind !== null) {
          assert(this.storedExceptionBeforeRewind === null);
          const result = this.storedValueBeforeRewind;
          this.storedValueBeforeRewind = null;
          return result;
        } else {
          assert(this.storedValueBeforeRewind === null);
          const error = this.storedExceptionBeforeRewind;
          this.storedExceptionBeforeRewind = null;
          throw error;
        }
      }
      // this function is being called for the first time
      assert(this.getState() == AsyncifyStateKind.None);

      // call the function
      const value = func(...args);
      // if the value is promise
      // we need to unwind the stack
      // so the caller will be able to evaluate the promise
      if (isPromise(value)) {
        // The next code step is JUMP-PT-UNWIND in wrapExport
        // The value will be passed to that pt through storedPromiseBeforeUnwind
        // getState() == Unwinding and we will enter the while loop in wrapExport
        this.startUnwind();
        assert(this.storedPromiseBeforeUnwind == null);
        this.storedPromiseBeforeUnwind = value;
        return undefined;
      } else {
        // The next code step is JUMP-PT-UNWIND in wrapExport
        // normal value, we don't have to do anything
        // getState() == None and we will exit while loop there
        return value;
      }
    };
  }

  /**
   * Warp an exported asynctify function so it can return promise
   *
   * @param func The input function
   * @returns The wrapped async function
   */
  wrapExport(
    func: (...args: Array<any>) => any,
    onSuspend?: () => void
  ): (...args: Array<any>) => Promise<any> {
    return async (...args: Array<any>) => {
      assert(this.getState() == AsyncifyStateKind.None);

      // call the original function
      let result = func(...args);

      // JUMP-PT-UNWIND
      // after calling the function
      // the caller may hit a unwinding point depending on
      // the if (isPromise(value)) condition in wrapImport
      while (this.getState() == AsyncifyStateKind.Unwinding) {
        this.stopUnwind();
        // The stack is unwound and no other JavaScript has run yet, so the
        // caller can still take references before the await below.
        if (onSuspend !== undefined) {
          onSuspend();
        }
        // try to resolve the promise that the internal requested
        // we then store it into the temp value in storedValueBeforeRewind
        // which then get passed onto the function(see wrapImport)
        // that can return the value
        const storedPromiseBeforeUnwind = this.storedPromiseBeforeUnwind;
        this.storedPromiseBeforeUnwind = null;
        assert(this.storedExceptionBeforeRewind === null);
        assert(this.storedValueBeforeRewind == null);

        try {
          this.storedValueBeforeRewind = await storedPromiseBeforeUnwind;
        } catch (error) {
          // the store exception
          this.storedExceptionBeforeRewind = error;
        }
        assert(!isPromise(this.storedValueBeforeRewind));
        // because we called asynctify_stop_unwind,the state is now none
        assert(this.getState() == AsyncifyStateKind.None);

        // re-enter the function, jump to JUMP-PT-REWIND in wrapImport
        // the value will be passed to that point via storedValueBeforeRewind
        //
        // NOTE: we guarantee that if exception is throw the asynctify state
        // will already be at None, this is because we will goto JUMP-PT-REWIND
        // which will call aynctify_stop_rewind
        this.startRewind();
        result = func(...args);
      }
      return result;
    };
  }

  private startRewind() : void {
    if (this.exports.asyncify_start_rewind === undefined) {
      throw Error("Asynctify is not enabled, please compile with -s ASYNCIFY=1 in emcc");
    }
    this.exports.asyncify_start_rewind(ASYNCIFY_DATA_ADDR);
    this.state = AsyncifyStateKind.Rewinding;
  }

  private stopRewind() : void {
    if (this.exports.asyncify_stop_rewind === undefined) {
      throw Error("Asynctify is not enabled, please compile with -s ASYNCIFY=1 in emcc");
    }
    this.exports.asyncify_stop_rewind();
    this.state = AsyncifyStateKind.None;
  }

  private startUnwind() : void {
    if (this.exports.asyncify_start_unwind === undefined) {
      throw Error("Asynctify is not enabled, please compile with -s ASYNCIFY=1 in emcc");
    }
    this.exports.asyncify_start_unwind(ASYNCIFY_DATA_ADDR);
    this.state = AsyncifyStateKind.Unwinding;
  }

  private stopUnwind() : void {
    if (this.exports.asyncify_stop_unwind === undefined) {
      throw Error("Asynctify is not enabled, please compile with -s ASYNCIFY=1 in emcc");
    }
    this.exports.asyncify_stop_unwind();
    this.state = AsyncifyStateKind.None;
  }
  /**
   * Initialize the wasm memory to setup necessary meta-data
   * for asynctify handling
   * @param memory The memory ti
   */
  private initMemory(memory: WebAssembly.Memory): void {
    // Set the meta-data at address ASYNCTIFY_DATA_ADDR
    new Int32Array(memory.buffer, ASYNCIFY_DATA_ADDR, 2).set(
      [ASYNCIFY_DATA_START, ASYNCIFY_DATA_END]
    );
  }
}
