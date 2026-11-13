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

import { OPFSStore, type OPFSAccessMode } from "./opfs_store";

export type { OPFSAccessMode } from "./opfs_store";

export interface TensorCacheEntry {
  name: string;
  shape: Array<number>;
  dtype: string;
  format: "f32-to-bf16" | "raw";
  byteOffset: number;
  nbytes: number;
}

export interface TensorShardEntry {
  dataPath: string;
  format: "raw-shard";
  nbytes: number;
  records: Array<TensorCacheEntry>;
}

/**
 * Return a borrowed view of one tensor record within a shard.
 *
 * The returned view aliases the shard and is only valid for as long as the
 * shard data remains alive. Tensor-cache decoding consumes it synchronously.
 */
export function getTensorCacheRecordBytes(
  shardData: ArrayBuffer | Uint8Array,
  record: Pick<TensorCacheEntry, "byteOffset" | "nbytes">,
): Uint8Array {
  const shardBytes =
    shardData instanceof Uint8Array ? shardData : new Uint8Array(shardData);
  const { byteOffset, nbytes } = record;
  if (!Number.isSafeInteger(byteOffset) || byteOffset < 0) {
    throw new Error(`Invalid tensor-cache byteOffset: ${byteOffset}`);
  }
  if (!Number.isSafeInteger(nbytes) || nbytes < 0) {
    throw new Error(`Invalid tensor-cache nbytes: ${nbytes}`);
  }
  const endOffset = byteOffset + nbytes;
  if (!Number.isSafeInteger(endOffset) || endOffset > shardBytes.byteLength) {
    throw new Error(
      `Tensor-cache record range [${byteOffset}, ${endOffset}) exceeds ` +
      `shard size ${shardBytes.byteLength}`,
    );
  }
  return shardBytes.subarray(byteOffset, endOffset);
}

/**
 *   Common Interface for the artifact cache
 */
export interface ArtifactCacheTemplate {
  /**
   * Retrieve data object that corresponds to `url` from cache. If data object does not exist in
   * cache, fetch the data and then add to cache.
   *
    * @param url The url to the data to be cached.
    * @param storetype This field is required so that `ArtifactIndexedDBCache` can store the
   * actual data object (see `addToCache()`), while `ArtifactCache` which uses the Cache API can
   * return the actual data object rather than the request. There are two options:
   * 1. "json": returns equivalent to `fetch(url).json()`
   * 2. "arraybuffer": returns equivalent to `fetch(url).arraybuffer()`
    * @param signal An optional AbortSignal allowing user to abort the fetching before its completion.
   * @return The data object (i.e. users do not need to call `.json()` or `.arraybuffer()`).
   *
    * Note: This is an async function.
   */
  fetchWithCache(url: string, storetype?: string, signal?: AbortSignal): Promise<any>;

  /**
   * Fetch data from url and add into cache. If already exists in cache, should return instantly.
   *
    * @param url The url to the data to be cached.
    * @param storetype Only applies to `ArtifactIndexedDBCache`. Since `indexedDB` stores the actual
    * @param signal An optional AbortSignal to abort data retrival.
   * data rather than a request, we specify `storagetype`. There are two options:
   * 1. "json": IndexedDB stores `fetch(url).json()`
   * 2. "arraybuffer": IndexedDB stores `fetch(url).arrayBuffer()`
   *
    * Note: This is an async function.
   */
  addToCache(url: string, storetype?: string, signal?: AbortSignal): Promise<void>;

  /**
   * check if cache has all keys in Cache
   *
    * Note: This is an async function.
   */
  hasAllKeys(keys: string[]): Promise<boolean>;

  /**
   * Delete url in cache if url exists
   *
    * Note: This is an async function.
   */
  deleteInCache(url: string): Promise<void>;
}

export type ArtifactCacheType = "cache" | "indexeddb" | "cross-origin" | "opfs";

export interface TensorCacheAccessOptions {
  cacheScope?: string;
  cacheType?: ArtifactCacheType;
  artifactCache?: ArtifactCacheTemplate;
  opfsAccessMode?: OPFSAccessMode;
}

type StoreType = string | undefined;
type RequestLike = string | URL | Request | { url?: string };

interface CrossOriginHashDescriptor {
  algorithm: string;
  value: string;
}

interface CrossOriginStorageHandle {
  getFile(): Promise<Blob>;
  createWritable(): Promise<CrossOriginStorageWritable>;
}

interface CrossOriginStorageGetFileHandleOptions {
  create?: boolean;
  origins?: string[] | string | undefined;
}

interface CrossOriginStorageWritable {
  write(data: Blob): Promise<void>;
  close(): Promise<void>;
}

interface CrossOriginStorageAPI {
  getFileHandle(
    descriptor: CrossOriginHashDescriptor,
    options?: CrossOriginStorageGetFileHandleOptions,
  ): Promise<CrossOriginStorageHandle>;
}

declare global {
  interface Navigator {
    crossOriginStorage?: CrossOriginStorageAPI;
  }
  interface WorkerNavigator {
    crossOriginStorage?: CrossOriginStorageAPI;
  }
}

const HASH_ALGORITHM = "SHA-256";
const DEFAULT_FETCH_OPTIONS: RequestInit = { method: "GET" };
const COS_HASH_META_CACHE = "tvmjs-cos-hash-meta";
let crossOriginFallbackWarningLogged = false;

const GLOBAL_HASH_CACHE = new Map<
  string,
  CrossOriginHashDescriptor
>();

class CrossOriginStorage {
  private hashCache: Map<string, CrossOriginHashDescriptor>;

  constructor() {
    this.hashCache = GLOBAL_HASH_CACHE;
  }

  static isAvailable(): boolean {
    if (typeof navigator === "undefined") {
      return false;
    }
    // Check for `getFileHandle()` itself, so that an implementation exposing
    // `navigator.crossOriginStorage` without the method falls back to the
    // default cache.
    return typeof navigator.crossOriginStorage?.getFileHandle === "function";
  }

  async match(request: RequestLike): Promise<Response | undefined> {
    const url = this.normalizeRequest(request);
    const hash = await this.resolveHashDescriptor(url);
    if (!hash) {
      return undefined;
    }
    try {
      const api = this.getApi();
      if (!api) {
        return undefined;
      }
      const handle = await api.getFileHandle(hash);
      if (!handle) {
        return undefined;
      }
      const blob = await handle.getFile();
      return new Response(blob);
    } catch {
      return undefined;
    }
  }

  async put(request: RequestLike, response: Response): Promise<void> {
    const url = this.normalizeRequest(request);
    const blob = await response.blob();
    const hash = await this.getBlobHash(blob);
    const api = this.getApi();
    if (!api) {
      throw new Error("Cross-origin storage API unavailable.");
    }
    const handle = await api.getFileHandle(hash, { create: true, origins: "*" /* All origins */ });
    if (!handle) {
      throw new Error("Cross-origin storage API returned no handle.");
    }
    const writableStream = await handle.createWritable();
    await writableStream.write(blob);
    await writableStream.close();
    this.hashCache.set(url, hash);
    await this.persistHashEntry(url, hash);
  }

  async delete(_request: RequestLike): Promise<void> {
    // Cross-origin storage extension currently has no delete API.
    return;
  }

  private getApi(): CrossOriginStorageAPI | undefined {
    if (!CrossOriginStorage.isAvailable()) {
      return undefined;
    }
    return navigator.crossOriginStorage;
  }

  private normalizeRequest(request: RequestLike): string {
    if (typeof request === "string") {
      return request;
    }
    if (request instanceof URL) {
      return request.href;
    }
    if (request instanceof Request) {
      return request.url;
    }
    if (request && typeof request.url === "string") {
      return request.url;
    }
    throw new Error("CrossOriginStorage: Unsupported request type.");
  }

  private async persistHashEntry(
    url: string,
    hash: CrossOriginHashDescriptor,
  ): Promise<void> {
    try {
      if (typeof caches === "undefined") {
        return;
      }
      const store = await caches.open(COS_HASH_META_CACHE);
      await store.put(url, new Response(JSON.stringify(hash)));
    } catch {
      // best-effort: ignore storage errors
    }
  }

  private async loadPersistedHashEntry(
    url: string,
  ): Promise<CrossOriginHashDescriptor | null> {
    try {
      if (typeof caches === "undefined") {
        return null;
      }
      const store = await caches.open(COS_HASH_META_CACHE);
      const response = await store.match(url);
      if (!response) {
        return null;
      }
      return JSON.parse(await response.text()) as CrossOriginHashDescriptor;
    } catch {
      return null;
    }
  }

  private async resolveHashDescriptor(
    url: string,
  ): Promise<CrossOriginHashDescriptor | null> {
    const cached = this.hashCache.get(url);
    if (cached) {
      return cached;
    }
    // Check persistent store before falling back to network-based hash extraction.
    // This covers non-LFS files (JSON configs, tokenizers) and non-HuggingFace URLs
    // (e.g. GitHub raw .wasm files) whose hashes were computed from blob content on a
    // previous visit and persisted to the Cache API.
    const persisted = await this.loadPersistedHashEntry(url);
    if (persisted) {
      this.hashCache.set(url, persisted);
      return persisted;
    }
    const hashValue = await this.getFileHash(url);
    if (!hashValue) {
      return null;
    }
    const descriptor: CrossOriginHashDescriptor = {
      algorithm: HASH_ALGORITHM,
      value: hashValue,
    };
    this.hashCache.set(url, descriptor);
    // Persist pointer-derived hashes so subsequent visits skip the LFS pointer
    // network request (especially important for models with many shards).
    await this.persistHashEntry(url, descriptor);
    return descriptor;
  }

  private async getFileHash(url: string): Promise<string | null> {
    if (/\/resolve\//.test(url)) {
      const pointerHash = await this.extractHashFromPointer(url);
      if (pointerHash) {
        return pointerHash;
      }
    }
    return null;
  }

  private async extractHashFromPointer(url: string): Promise<string | null> {
    const rawUrl = url.replace(/\/resolve\//, "/raw/");
    try {
      const text = await fetch(rawUrl).then((res) => res.text());
      if (!text.includes("oid sha256:")) {
        return null;
      }
      const match = text.match(/oid sha256:([A-Fa-f0-9]+)/);
      return match ? match[1] : null;
    } catch {
      return null;
    }
  }

  private async getBlobHash(blob: Blob): Promise<CrossOriginHashDescriptor> {
    const arrayBuffer = await blob.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest(HASH_ALGORITHM, arrayBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
    return {
      algorithm: HASH_ALGORITHM,
      value: hashHex,
    };
  }
}


/**
 * Cache to store model related data, implemented with the Cache API.
 */
export class ArtifactCache implements ArtifactCacheTemplate {
  private scope: string;
  private cache?: Cache;

  constructor(scope: string) {
    this.scope = scope;
  }

  /**
   * Convert the Response object to the expected storetype instead
   */
  async responseTostoretype(response: Response, storetype?: string): Promise<any> {
    if (storetype === undefined) {
      return response;
    } else if (storetype.toLowerCase() === "json") {
      return await response.json();
    } else if (storetype.toLowerCase() === "arraybuffer") {
      return await response.arrayBuffer();
    } else {
      console.error("Unknown storage type " + storetype + ", returning raw response");
      return response;
    }
  }

  /**
   * fetch the corresponding url object in response or stored object format
   * @param url url
   * @param storetype the storage type for indexedDB
   * @param signal an optional abort signal to abort fetching
   * @returns response in json, arraybuffer or pure response format
   */
  async fetchWithCache(url: string, storetype?: string, signal?: AbortSignal): Promise<any> {
    await this.addToCache(url, storetype, signal);
    const result = await this.cache.match(new Request(url));
    if (result === undefined) {
      // Already called `addToCache()`, should expect the request in cache.
      throw Error("Cannot fetch " + url);
    }
    return await this.responseTostoretype(result, storetype);
  }

  async addToCache(url: string, storetype?: string, signal?: AbortSignal) {
    const request = new Request(url, signal ? { signal } : undefined);
    if (this.cache === undefined) {
      this.cache = await caches.open(this.scope);
    }
    const result = await this.cache.match(request);
    if (result === undefined) {
      await this.cache.add(request);
    }
  }

  /**
   * Determine if all keys exist in the cache
   * @param keys the url key list of the strings
   * @returns boolean value indicate if all keys are in cache
   */
  async hasAllKeys(keys: string[]) {
    if (this.cache === undefined) {
      this.cache = await caches.open(this.scope);
    }
    return this.cache.keys()
      .then(requests => requests.map(request => request.url))
      .then(cacheKeys => keys.every(key => cacheKeys.indexOf(key) !== -1))
      .catch(() => false);
  }

  /**
   * Delete the corresponding url object in cache
   * @param url the corresponding url object to be deleted
   */
  async deleteInCache(url: string) {
    if (this.cache === undefined) {
      this.cache = await caches.open(this.scope);
    }
    await this.cache.delete(url);
  }
}

/**
 * Cache by IndexedDB to support caching model data
 */
export class ArtifactIndexedDBCache implements ArtifactCacheTemplate {
  private dbName?: string;
  private dbVersion = 1;
  private db: IDBDatabase | undefined;

  constructor(dbName: string) {
    this.dbName = dbName;
  }

  /**
   * Init the indexed DB database if it is not initialized.
   */
  private async initDB() {
    if (this.db != null) {
      return; // the db is already inialized
    }
    return new Promise<void>((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);
      request.onupgradeneeded = (event) => {
        this.db = (event.target as IDBOpenDBRequest).result;
        if (!this.db.objectStoreNames.contains('urls')) {
          this.db.createObjectStore('urls', { keyPath: 'url' });
        }
      };
      request.onsuccess = (event) => {
        this.db = (event.target as IDBOpenDBRequest).result;
        resolve();
      };
      request.onerror = (event) => {
        console.error("Database error: ", (event.target as IDBOpenDBRequest).error);
        reject((event.target as IDBOpenDBRequest).error);
      };
    });
  }

  /**
   * Check if current url object is in indexedDB or not
   * @param url the url link
   * @returns boolean indicate if url object in indexedDB
   */
  private async isUrlInDB(url: string): Promise<boolean> {
    return new Promise<boolean>((resolve, reject) => {
      const transaction = this.db?.transaction(['urls'], 'readonly');
      if (transaction === undefined) {
        return false;
      }
      const store = transaction.objectStore('urls');
      const request = store.get(url);
      request.onsuccess = () => {
        resolve(request.result !== undefined);
      };
      request.onerror = (event) => {
        reject((event.target as IDBRequest).error);
      };
    });
  }

  async asyncGetHelper(url: string): Promise<any> {
    return new Promise((resolve, reject) => {
      let result: any;
      const transaction = this.db?.transaction(['urls'], 'readonly');
      if (transaction === undefined) {
        return false;
      }
      transaction.oncomplete = () => resolve(result);
      transaction.onerror = () => reject(transaction.error);
      const objectStore = transaction.objectStore('urls');
      const getRequest = objectStore.get(url);
      getRequest.onsuccess = () => {
        result = getRequest.result;
      }
    })
  }

  async fetchWithCache(url: string, storetype?: string, signal?: AbortSignal): Promise<any> {
    await this.addToCache(url, storetype, signal);
    let result = await this.asyncGetHelper(url);
    if (result === null) {
      // previously null data in cache or somehow failed to add to cache, delete and retry
      await this.deleteInCache(url);
      await this.addToCache(url, storetype);
      result = await this.asyncGetHelper(url);
    }
    if (result != null && typeof result === "object" && "data" in result) {
      // `storetype` not used here because the data stored in indexedDB is already in that type
      return result.data;
    }
    throw Error("ArtifactIndexedDBCache failed to fetch: " + url);
  }

  async addToIndexedDB(url: string, response: any, storetype?: string) {
    await this.initDB();
    let data: any;
    // IndexedDB, unlike the Cache API, stores the actual data object, so we convert reponse here.
    if (storetype != undefined) {
      if (storetype.toLowerCase() === "json") {
        data = await response.json();
      } else if (storetype.toLocaleLowerCase() === "arraybuffer") {
        data = await response.arrayBuffer();
      } else {
        throw Error("Unsupported storetyp for IndexedDB: " + storetype);
      }
    }
    return new Promise<void>((resolve, reject) => {
      const transaction = this.db?.transaction(['urls'], 'readwrite');
      if (transaction === undefined) {
        return;
      }
      const store = transaction.objectStore('urls');
      const request = store.add({ data, url }); // Index DB follows a {value, key} format, instead of {key, value} format!
      request.onsuccess = () => resolve();
      request.onerror = (event) => reject((event.target as IDBRequest).error);
    });
  }

  async addToCache(url: string, storetype?: string, signal?: AbortSignal): Promise<void> {
    await this.initDB(); // await the initDB process
    // If already cached, nothing to do
    const isInDB = await this.isUrlInDB(url);
    if (isInDB) {
      return;
    }
    try {
      const response = await fetch(url, signal ? { signal } : undefined);
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const response_copy = response.clone();
      await this.addToIndexedDB(url, response_copy, storetype);
    } catch (error) {
      throw Error("Failed to store " + url + " with error: " + error);
    }
  }

  async hasAllKeys(keys: string[]): Promise<boolean> {
    await this.initDB(); // Ensure the DB is initialized
    if (!this.db) {
      throw new Error('Database is not initialized');
    }
    return new Promise<boolean>((resolve, reject) => {
      const transaction = this.db.transaction(['urls'], 'readonly');
      const store = transaction.objectStore('urls');
      const promises = keys.map(key => {
        return new Promise<boolean>((resolve) => {
          const request = store.get(key);
          request.onsuccess = () => {
            if (request.result === undefined) {
              resolve(false); // Key not found, resolve with false
            } else {
              resolve(true); // Key found, resolve with true
            }
          };
          request.onerror = () => {
            resolve(false); // On error, resolve as if the key was not found
          };
        });
      });
      Promise.all(promises).then(results => {
        const allExist = results.every(exists => exists);
        resolve(allExist);
      }).catch(error => {
        reject(error); // Reject the main promise if any of the promises are rejected
      });
    });
  }

  async deleteInCache(url: string) {
    await this.initDB(); // Make sure the DB is initialized
    const transaction = this.db?.transaction(['urls'], 'readwrite');
    if (transaction === undefined) {
      return;
    }
    const store = transaction.objectStore('urls');
    const request = store.delete(url);
    // Await completion of the delete request
    await new Promise<void>((resolve, reject) => {
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
    return;
  }
}

/**
 * Cache by Origin Private File System (OPFS).
 */
export class ArtifactOPFSCache implements ArtifactCacheTemplate {
  private readonly store: OPFSStore;

  constructor(scope: string, accessMode: OPFSAccessMode = "async") {
    this.store = new OPFSStore(scope, accessMode);
  }

  static isAvailable(): boolean {
    return OPFSStore.isAvailable();
  }

  async fetchWithCache(
    url: string,
    storetype?: string,
    signal?: AbortSignal,
  ): Promise<any> {
    // Try the cache first to avoid a redundant OPFS lookup on a hit.
    const cached = await this.readFromCache(url, storetype);
    if (cached !== undefined) {
      return cached;
    }
    await this.addToCache(url, storetype, signal);
    const fetched = await this.readFromCache(url, storetype);
    if (fetched === undefined) {
      throw new Error("ArtifactOPFSCache failed to fetch: " + url);
    }
    return fetched;
  }

  private async readFromCache(
    url: string,
    storetype?: string,
  ): Promise<any> {
    if (storetype?.toLowerCase() === "arraybuffer") {
      return this.store.readArrayBuffer(url);
    }
    const cachedResponse = await this.store.read(url);
    if (cachedResponse === undefined) {
      return undefined;
    }
    return this.responseToStoreType(cachedResponse, storetype);
  }

  async addToCache(
    url: string,
    _storetype?: string,
    signal?: AbortSignal,
  ): Promise<void> {
    if (await this.store.has(url)) {
      return;
    }
    const request = new Request(
      url,
      signal ? { ...DEFAULT_FETCH_OPTIONS, signal } : DEFAULT_FETCH_OPTIONS,
    );
    const response = await fetch(request);
    if (!response.ok) {
      throw new Error(
        `ArtifactOPFSCache: Unable to fetch ${url}, received status ${response.status}`,
      );
    }
    await this.store.write(url, response);
  }

  async hasAllKeys(keys: string[]): Promise<boolean> {
    const results = await Promise.all(
      keys.map(async (key) => await this.store.has(key)),
    );
    return results.every((result) => result);
  }

  async deleteInCache(url: string): Promise<void> {
    await this.store.remove(url);
  }

  private async responseToStoreType(
    response: Response,
    storetype?: StoreType,
  ): Promise<any> {
    if (storetype === undefined) {
      return response;
    }
    const format = storetype.toLowerCase();
    if (format === "json") {
      return response.json();
    }
    if (format === "arraybuffer") {
      return response.arrayBuffer();
    }
    return response;
  }
}

/**
 * Cache by cross-origin storage extension.
 */
export class ArtifactCrossOriginStorageCache implements ArtifactCacheTemplate {
  private storage: CrossOriginStorage;

  constructor(
    _scope: string,
    storage: CrossOriginStorage = new CrossOriginStorage(),
  ) {
    this.storage = storage;
  }

  async fetchWithCache(
    url: string,
    storetype?: StoreType,
    signal?: AbortSignal,
  ): Promise<any> {
    const cachedResponse = await this.storage.match(url);
    if (cachedResponse !== undefined) {
      return this.responseToStoreType(cachedResponse, storetype);
    }
    await this.addToCache(url, storetype, signal);
    const hydrated = await this.storage.match(url);
    if (hydrated === undefined) {
      throw new Error(`ArtifactCrossOriginStorageCache: failed to hydrate ${url}`);
    }
    return this.responseToStoreType(hydrated, storetype);
  }

  async addToCache(
    url: string,
    _storetype?: StoreType,
    signal?: AbortSignal,
  ): Promise<void> {
    const existing = await this.storage.match(url);
    if (existing !== undefined) {
      return;
    }
    const request = new Request(
      url,
      signal ? { ...DEFAULT_FETCH_OPTIONS, signal } : DEFAULT_FETCH_OPTIONS,
    );
    const response = await fetch(request);
    if (!response.ok) {
      throw new Error(
        `ArtifactCrossOriginStorageCache: Unable to fetch ${url}, received status ${response.status}`,
      );
    }
    await this.storage.put(url, response.clone());
  }

  async hasAllKeys(keys: string[]): Promise<boolean> {
    const results = await Promise.all(
      keys.map(async (key) => {
        const cached = await this.storage.match(key);
        return cached !== undefined;
      }),
    );
    return results.every((result) => result);
  }

  async deleteInCache(url: string): Promise<void> {
    await this.storage.delete(url);
  }

  private async responseToStoreType(
    response: Response,
    storetype?: StoreType,
  ): Promise<any> {
    if (storetype === undefined) {
      return response;
    }
    const format = storetype.toLowerCase();
    if (format === "json") {
      return response.json();
    }
    if (format === "arraybuffer") {
      return response.arrayBuffer();
    }
    return response;
  }
}

function normalizeCacheType(cacheType?: string): ArtifactCacheType {
  if (cacheType === undefined) {
    return "cache";
  }
  const normalized = cacheType.toLowerCase();
  if (normalized === "cache") {
    return "cache";
  }
  if (normalized === "indexeddb") {
    return "indexeddb";
  }
  if (normalized === "cross-origin") {
    return "cross-origin";
  }
  if (normalized === "opfs") {
    return "opfs";
  }
  console.error("Unsupported cacheType: " + cacheType + ", using default ArtifactCache.");
  return "cache";
}

function isTensorCacheAccessOptions(
  value: string | TensorCacheAccessOptions | undefined,
): value is TensorCacheAccessOptions {
  return typeof value === "object" && value !== null;
}

function normalizeCacheAccessOptions(
  cacheScopeOrOptions: string | TensorCacheAccessOptions | undefined,
  cacheType?: string,
): TensorCacheAccessOptions {
  if (isTensorCacheAccessOptions(cacheScopeOrOptions)) {
    return cacheScopeOrOptions;
  }
  return {
    cacheScope: cacheScopeOrOptions,
    cacheType: normalizeCacheType(cacheType),
  };
}

export function createArtifactCache(
  scope: string,
  options: TensorCacheAccessOptions = {},
): ArtifactCacheTemplate {
  if (options.artifactCache !== undefined) {
    return options.artifactCache;
  }
  const cacheType = normalizeCacheType(options.cacheType);
  if (cacheType === "indexeddb") {
    return new ArtifactIndexedDBCache(scope);
  }
  if (cacheType === "cross-origin") {
    if (CrossOriginStorage.isAvailable()) {
      return new ArtifactCrossOriginStorageCache(scope);
    }
    if (!crossOriginFallbackWarningLogged) {
      console.warn(
        "Cross-origin storage backend is unavailable; falling back to ArtifactCache.",
      );
      crossOriginFallbackWarningLogged = true;
    }
  }
  if (cacheType === "opfs") {
    return new ArtifactOPFSCache(scope, options.opfsAccessMode);
  }
  return new ArtifactCache(scope);
}


/**
 * Function to check if NDarray is in Cache or not
 *
 * @param tensorCacheUrl The cache url which links to the Tensor
 * @param cacheScope The scope identifier of the cache
 * @param cacheType The type of the cache: "cache", "indexedDB", "cross-origin", or "opfs"
 * @returns the result if the cache has Tensor
 */
export async function hasTensorInCache(
  tensorCacheUrl: string,
  options?: TensorCacheAccessOptions,
): Promise<boolean>;
export async function hasTensorInCache(
  tensorCacheUrl: string,
  cacheScope?: string,
  cacheType?: string,
): Promise<boolean>;
export async function hasTensorInCache(
  tensorCacheUrl: string,
  cacheScopeOrOptions: string | TensorCacheAccessOptions = "tvmjs",
  cacheType = "cache",
): Promise<boolean> {
  const options = normalizeCacheAccessOptions(cacheScopeOrOptions, cacheType);
  const cacheScope = options.cacheScope ?? "tvmjs";
  const artifactCache = createArtifactCache(cacheScope, options);
  const jsonUrl = new URL("tensor-cache.json", tensorCacheUrl).href;
  const hasJsonUrlInCache = await artifactCache.hasAllKeys([jsonUrl]);
  if (!hasJsonUrlInCache) {
    return false;
  }
  const list = (await artifactCache.fetchWithCache(
    jsonUrl,
    "json",
  ))["records"] as Array<TensorShardEntry>;
  return await artifactCache.hasAllKeys(list.map(key => new URL(key.dataPath, tensorCacheUrl).href));
}


/**
 * Given cacheUrl, search up items to delete based on cacheUrl/tensor-cache.json
 *
 * @param cacheUrl The cacheUrl for the items
 * @param cacheScope The scope identifier of the cache
 * @param cacheType The type of the cache: "cache", "indexedDB", "cross-origin", or "opfs"
 */
export async function deleteTensorCache(
  cacheUrl: string,
  options?: TensorCacheAccessOptions,
): Promise<void>;
export async function deleteTensorCache(
  cacheUrl: string,
  cacheScope?: string,
  cacheType?: string,
): Promise<void>;
export async function deleteTensorCache(
  cacheUrl: string,
  cacheScopeOrOptions: string | TensorCacheAccessOptions = "tvmjs",
  cacheType = "cache",
): Promise<void> {
  const options = normalizeCacheAccessOptions(cacheScopeOrOptions, cacheType);
  const cacheScope = options.cacheScope ?? "tvmjs";
  const artifactCache = createArtifactCache(cacheScope, options);
  if (artifactCache instanceof ArtifactCrossOriginStorageCache) {
    // Cross-origin storage extension does not currently support programmatic deletion.
    return;
  }
  const jsonUrl = new URL("tensor-cache.json", cacheUrl).href;
  const list = await artifactCache.fetchWithCache(jsonUrl, "json");
  const arrayentry = list["records"] as Array<TensorShardEntry>;
  const processShard = async (i: number) => {
    const dataUrl = new URL(arrayentry[i].dataPath, cacheUrl).href;
    await artifactCache.deleteInCache(dataUrl);
  }
  await Promise.all(arrayentry.map((_, index) => processShard(index)));
}
