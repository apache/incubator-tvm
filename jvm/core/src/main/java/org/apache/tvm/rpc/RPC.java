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

import java.util.HashMap;
import java.util.Map;
import org.apache.tvm.Function;

public class RPC {
  public static final int RPC_TRACKER_MAGIC = 0x2f271;
  public static final int RPC_MAGIC = 0xff271;
  public static final int RPC_CODE_MISMATCH = RPC_MAGIC + 2;

  public static final String TIMEOUT_ARG = "-timeout=";

  public class TrackerCode {
    public static final int PUT = 3;
    public static final int UPDATE_INFO = 5;
    public static final int GET_PENDING_MATCHKEYS = 7;
    public static final int SUCCESS = 0;
  }

  private static ThreadLocal<Map<String, Function>> apiFuncs =
      new ThreadLocal<Map<String, Function>>() {
        @Override
        protected Map<String, Function> initialValue() {
          return new HashMap<String, Function>();
        }
      };

  /**
   * Get internal function starts with namespace tvm.rpc.
   * @param name function name.
   * @return the function, null if not exists.
   */
  static Function getApi(String name) {
    Function func = apiFuncs.get().get(name);
    if (func == null) {
      func = Function.getFunction("rpc." + name);
      if (func == null) {
        return null;
      }
      apiFuncs.get().put(name, func);
    }
    return func;
  }
}
