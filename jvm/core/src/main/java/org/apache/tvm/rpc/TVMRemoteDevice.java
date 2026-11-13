/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.tvm.rpc;

import org.apache.tvm.Function;
import org.apache.tvm.TVMObject;
import org.apache.tvm.TVMType;
import org.apache.tvm.TVMValue;
import org.apache.tvm.TVMValueNull;

/**
 * A session-owned remote device, distinct from a local Device.
 * Its native object retains the session and an ordinary endpoint device type and
 * index. The endpoint descriptor has no session identity by itself. Allocation,
 * queries and synchronization use the owning session and fail after its closure.
 */
public class TVMRemoteDevice extends TVMObject {
  public TVMRemoteDevice(long handle, int typeIndex) {
    super(handle, typeIndex);
  }

  public boolean exist() {
    return RPC.getApi("DeviceGetAttr").call(this, "exist").asLong() != 0;
  }

  public long maxThreadsPerBlock() {
    return RPC.getApi("DeviceGetAttr").call(this, "max_threads_per_block").asLong();
  }

  public long warpSize() {
    return RPC.getApi("DeviceGetAttr").call(this, "warp_size").asLong();
  }

  public void sync() {
    RPC.getApi("DeviceSync").call(this, new TVMValueNull());
  }

  /** Allocate remote storage; shape metadata is constructed locally. */
  public RemoteTensor empty(long[] dimensions, TVMType dtype) {
    Function shapeConstructor = Function.getFunction("ffi.Shape");
    for (long dim : dimensions) {
      shapeConstructor.pushArg(dim);
    }
    TVMValue shape = shapeConstructor.invoke();
    try {
      return (RemoteTensor) RPC.getApi("Empty").call(
          shape, dtype.toString(), this, new TVMValueNull());
    } finally {
      shape.release();
    }
  }
}
