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
import org.apache.tvm.TVMValue;

/**
 * RPC Client.
 */
public class Client {
  /**
   * Connect to RPC Server.
   * @param url The url of the host.
   * @param port The port to connect to.
   * @param key Additional key to match server.
   * @return The connected session.
   */
  public static RPCSession connect(String url, int port, String key) {
    Function doConnect = RPC.getApi("Connect");
    if (doConnect == null) {
      throw new RuntimeException("Please compile with TVM_USE_RPC=1");
    }
    TVMValue sess = doConnect.pushArg(url).pushArg(port).pushArg(key).invoke();
    return new RPCSession(sess.asModule());
  }

  /**
   * Connect to RPC Server.
   * @param url The url of the host.
   * @param port The port to connect to.
   * @return The connected session.
   */
  public static RPCSession connect(String url, int port) {
    return connect(url, port, "");
  }
}
