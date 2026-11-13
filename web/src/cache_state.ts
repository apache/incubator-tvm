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
 * Caching utilities for the TVM web runtime.
 *
 * Provides a generic LRUCache and a CacheState container that manages
 * domain-specific caches used by the WebGPU runtime.
 */
import { Disposable } from "./types";

/**
 * A generic LRU (Least Recently Used) cache with bounded size.
 *
 * Entries are evicted in insertion order when the cache exceeds `maxSize`.
 * Uses a Map to maintain insertion order for O(1) LRU eviction.
 *
 * @typeParam K - Cache key type.
 * @typeParam V - Cache value type.
 */
export class LRUCache<K, V> {
  private cache: Map<K, V> = new Map();
  private readonly maxSize: number;
  /** Optional callback invoked when an entry is evicted. */
  private readonly onEvict?: (key: K, value: V) => void;

  constructor(maxSize: number, onEvict?: (key: K, value: V) => void) {
    this.maxSize = maxSize;
    this.onEvict = onEvict;
  }

  /**
   * Get a value from the cache, constructing it via `constructor` on miss.
   *
   * On hit: moves the entry to most-recently-used position and returns it.
   * On miss: calls `constructor()` to create the value, inserts it, and
   * returns it. If the cache is full, the least-recently-used entry is
   * evicted first.
   *
   * @param key The cache key.
   * @param constructor Factory function called on cache miss to produce the value.
   * @returns The cached or newly constructed value.
   */
  get(key: K, constructor: () => V): V {
    const existing = this.cache.get(key);
    if (existing !== undefined) {
      // Move to most-recently-used position
      this.cache.delete(key);
      this.cache.set(key, existing);
      return existing;
    }
    // Evict LRU entry if at capacity
    if (this.cache.size >= this.maxSize) {
      const oldest = this.cache.keys().next().value;
      if (oldest !== undefined) {
        if (this.onEvict) {
          this.onEvict(oldest, this.cache.get(oldest)!);
        }
        this.cache.delete(oldest);
      }
    }
    const value = constructor();
    this.cache.set(key, value);
    return value;
  }

  /**
   * Check whether eviction would be needed for a new entry.
   *
   * Useful when the caller needs to perform side effects before eviction
   * (e.g. flushing pending GPU commands before destroying an evicted buffer).
   *
   * @param key The key to check.
   * @returns true if inserting `key` would trigger eviction of another entry.
   */
  needEviction(key: K): boolean {
    if (this.cache.has(key)) return false;
    return this.cache.size >= this.maxSize;
  }

  /**
   * Clear all cached entries.
   *
   * Does not dispose values — the caller is responsible for cleanup
   * (e.g. destroying GPU buffers) before calling invalidate.
   */
  invalidate(): void {
    this.cache.clear();
  }

  /** Number of entries currently in the cache. */
  get size(): number {
    return this.cache.size;
  }

  /** Iterate over all cached values (for disposal). */
  values(): IterableIterator<V> {
    return this.cache.values();
  }
}

/**
 * CacheState manages domain-specific caches for the WebGPU runtime.
 *
 * Currently contains:
 * - **shapeCache**: Caches TVM ShapeTuple objects keyed by dimension string.
 *   - Why: `makeShapeTuple()` is called on every tensor operation, crossing
 *     the JS→WASM FFI boundary each time. During LLM decode, the same shapes
 *     repeat every token (e.g. [1,32,128]), so caching avoids thousands of
 *     redundant FFI round-trips.
 *   - Invalidation: Cache entries may be evicted, but returned shape tuples
 *     hold independent references and remain valid for their caller's scope.
 *
 * - **bindGroupCache**: Caches GPUBindGroup objects for kernel launches.
 *   - Why: creating a bind group is a round trip to the GPU process, and
 *     during LLM decode nearly every launch rebinds the buffers it bound one
 *     token earlier.
 *   - Key: exact string of the shader uid, the buffer uids in binding order
 *     and the uniform-buffer uid; uids are never reused, unlike GPU pointers.
 *   - Invalidation: Never. Bind groups are immutable and an entry is only
 *     reachable with the ids of live buffers; stale entries leave through LRU.
 *
 * Future additions (follow-up PR):
 * - **uniformCache**: Caches GPU uniform buffers keyed by content hash.
 *   - Why: Many dispatches use identical scalar arguments (matrix dims, etc.).
 *     Reusing the buffer avoids `createBuffer` + `writeBuffer` overhead.
 *   - Invalidation: Must invalidate on any GPU buffer deallocation, since
 *     buffer pointers can be reused by the allocator, making cached entries
 *     that reference the old buffer stale.
 */
export class CacheState {
  /**
   * Cache for TVM ShapeTuple objects.
   *
   * Key: comma-separated dimension string, e.g. "1,32,128"
   * Value: TVM ShapeTuple object (Disposable)
   *
   * Eviction releases only the cache's reference. Shape tuples returned to
   * callers have independent references and normal scope-managed lifetimes.
   */
  readonly shapeCache: LRUCache<string, Disposable>;

  /**
   * Cache for the bind groups of WebGPU kernel launches (see class comment).
   * The size must exceed the number of launches in one repeating unit of
   * work (a few hundred per LLM decode step), or every lookup misses.
   */
  readonly bindGroupCache: LRUCache<string, GPUBindGroup>;

  private nextUid = 0;

  constructor(shapeCacheSize: number = 256, bindGroupCacheSize: number = 2048) {
    this.shapeCache = new LRUCache<string, Disposable>(
      shapeCacheSize,
      (_key, value) => value.dispose()
    );
    this.bindGroupCache = new LRUCache<string, GPUBindGroup>(bindGroupCacheSize);
  }

  /**
   * Compute the cache key for a shape tuple.
   *
   * @param shape Array of dimension values.
   * @returns String key suitable for shapeCache lookup.
   */
  static computeShapeKey(shape: Array<number>): string {
    return shape.toString();
  }

  /**
   * Allocate an id for an object that takes part in cache keys.
   *
   * @returns An id that this CacheState has not returned before.
   */
  allocUid(): number {
    return this.nextUid++;
  }

  /**
   * Compute the cache key for the bind group of a kernel launch.
   *
   * A bind group is fully determined by its layout and the resource bound at
   * each binding. For a kernel launch:
   * - the layout, and the size of the uniform binding, are fixed per shader,
   *   so `shaderUid` stands for both;
   * - every buffer argument is bound whole (offset 0, full size), so its
   *   buffer's unique id stands for the binding;
   * - the last binding is the uniform buffer holding the POD arguments.
   *
   * POD argument values are not part of the key: they live in the uniform
   * buffer's contents, not in the bind group.
   *
   * @param shaderUid Unique id of the shader.
   * @param bufferUids Unique ids of the buffer arguments, in binding order.
   * @param uniformUid Unique id of the uniform buffer.
   * @returns String key suitable for bindGroupCache lookup.
   */
  static computeBindGroupKey(
    shaderUid: number,
    bufferUids: Array<number>,
    uniformUid: number
  ): string {
    return shaderUid + ":" + bufferUids.join(",") + ":" + uniformUid;
  }

  /**
   * Dispose all cached objects and clear all caches.
   */
  dispose(): void {
    for (const obj of this.shapeCache.values()) {
      obj.dispose();
    }
    this.shapeCache.invalidate();
    this.bindGroupCache.invalidate();
  }
}
