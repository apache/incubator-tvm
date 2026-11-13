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

  constructor(shapeCacheSize: number = 256) {
    this.shapeCache = new LRUCache<string, Disposable>(
      shapeCacheSize,
      (_key, value) => value.dispose()
    );
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
   * Dispose all cached objects and clear all caches.
   */
  dispose(): void {
    for (const obj of this.shapeCache.values()) {
      obj.dispose();
    }
    this.shapeCache.invalidate();
  }
}
