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

import org.apache.tvm.TVMObject;
import org.apache.tvm.TVMValueNull;
import org.apache.tvm.TensorBase;

/**
 * Remote storage whose opaque address cannot be used as a local Tensor.
 * The native object retains the allocation or returned tensor handle and its
 * session. Releasing this reference releases one owner; shared views retain the
 * backing allocation. Transfers use explicit local or same-session destinations.
 */
public class RemoteTensor extends TVMObject {
  public RemoteTensor(long handle, int typeIndex) {
    super(handle, typeIndex);
  }

  public RemoteTensor copyFrom(TensorBase source) {
    RPC.getApi("CopyFromLocal").call(source, this);
    return this;
  }

  public TensorBase copyTo(TensorBase destination) {
    RPC.getApi("CopyToLocal").call(this, destination);
    return destination;
  }

  /** Copy within the owning session. Synchronize the device before observing results. */
  public RemoteTensor copyTo(RemoteTensor destination) {
    RPC.getApi("CopyFromTo").call(this, destination, new TVMValueNull());
    return destination;
  }
}
