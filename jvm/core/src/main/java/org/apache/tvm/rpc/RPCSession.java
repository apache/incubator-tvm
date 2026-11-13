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

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import org.apache.tvm.Device;
import org.apache.tvm.Function;
import org.apache.tvm.Module;

/**
 * RPC Client session module.
 * Do not directly create the object, use Client.connect.
 */
public class RPCSession {
  private final Module session;
  private final Map<String, Function> remoteFuncs = new HashMap<String, Function>();

  RPCSession(Module sess) {
    session = sess;
  }

  /**
   * Get function from the session.
   * @param name The name of the function.
   * @return The result function.
   */
  public Function getFunction(String name) {
    return session.getFunction(name);
  }

  /**
   * Construct a remote device.
   * @param devType device type.
   * @param devId device id.
   * @return The session-owned remote device.
   */
  public TVMRemoteDevice device(String devType, int devId) {
    Device dev = new Device(devType, devId);
    return device(dev.deviceType, devId);
  }

  /**
   * Construct a remote device.
   * @param devType device type.
   * @return The session-owned remote device.
   */
  public TVMRemoteDevice device(String devType) {
    return device(devType, 0);
  }

  /**
   * Construct a remote device.
   * @param devType device type.
   * @param devId device id.
   * @return The session-owned remote device.
   */
  public TVMRemoteDevice device(int devType, int devId) {
    return (TVMRemoteDevice) RPC.getApi("RemoteDevice").call(session, new Device(devType, devId));
  }

  /**
   * Construct a remote device.
   * @param devType device type.
   * @return The session-owned remote device.
   */
  public TVMRemoteDevice device(int devType) {
    return device(devType, 0);
  }

  /**
   * Construct remote CPU device.
   * @param devId device id.
   * @return Remote CPU device.
   */
  public TVMRemoteDevice cpu(int devId) {
    return device("cpu", devId);
  }

  /**
   * Construct remote CPU device.
   * @return Remote CPU device.
   */
  public TVMRemoteDevice cpu() {
    return cpu(0);
  }

  /**
   * Construct remote CUDA GPU device.
   * @param devId device id.
   * @return Remote CUDA GPU device.
   */
  public TVMRemoteDevice cuda(int devId) {
    return device("cuda", devId);
  }

  /**
   * Construct remote CUDA GPU device.
   * @return Remote CUDA GPU device.
   */
  public TVMRemoteDevice cuda() {
    return cuda(0);
  }

  /**
   * Construct remote OpenCL device.
   * @param devId device id.
   * @return Remote OpenCL device.
   */
  public TVMRemoteDevice cl(int devId) {
    return device("opencl", devId);
  }

  /**
   * Construct remote OpenCL device.
   * @return Remote OpenCL device.
   */
  public TVMRemoteDevice cl() {
    return cl(0);
  }

  /**
   * Construct remote OpenCL device.
   * @param devId device id.
   * @return Remote OpenCL device.
   */
  public TVMRemoteDevice vulkan(int devId) {
    return device("vulkan", devId);
  }

  /**
   * Construct remote OpenCL device.
   * @return Remote OpenCL device.
   */
  public TVMRemoteDevice vulkan() {
    return vulkan(0);
  }

  /**
   * Construct remote Metal device.
   * @param devId device id.
   * @return Remote metal device.
   */
  public TVMRemoteDevice metal(int devId) {
    return device("metal", devId);
  }

  /**
   * Construct remote Metal device.
   * @return Remote metal device.
   */
  public TVMRemoteDevice metal() {
    return metal(0);
  }

  /**
   * Upload binary to remote runtime temp folder.
   * @param data The binary in local to upload.
   * @param target The path in remote, cannot be null.
   */
  public void upload(byte[] data, String target) {
    if (target == null) {
      throw new IllegalArgumentException("Please specify the upload target");
    }
    final String funcName = "upload";
    Function remoteFunc = remoteFuncs.get(funcName);
    if (remoteFunc == null) {
      remoteFunc = getFunction("tvm.rpc.server.upload");
      remoteFuncs.put(funcName, remoteFunc);
    }
    remoteFunc.pushArg(target).pushArg(data).invoke();
  }

  /**
   * Upload file to remote runtime temp folder.
   * @param data The file in local to upload.
   * @param target The path in remote.
   * @throws java.io.IOException for network failure.
   */
  public void upload(File data, String target) throws IOException {
    byte[] blob = getBytesFromFile(data);
    upload(blob, target);
  }

  /**
   * Upload file to remote runtime temp folder.
   * @param data The file in local to upload.
   * @throws java.io.IOException for network failure.
   */
  public void upload(File data) throws IOException {
    upload(data, data.getName());
  }

  /**
   * Download file from remote temp folder.
   * @param path The relative location to remote temp folder.
   * @return The result blob from the file.
   */
  public byte[] download(String path) {
    final String name = "download";
    Function func = remoteFuncs.get(name);
    if (func == null) {
      func = getFunction("tvm.rpc.server.download");
      remoteFuncs.put(name, func);
    }
    return func.pushArg(path).invoke().asBytes();
  }

  /**
   * Load a remote module, the file need to be uploaded first.
   * @param path The relative location to remote temp folder.
   * @return The remote module containing remote function.
   */
  public Module loadModule(String path) {
    return RPC.getApi("LoadRemoteModule").pushArg(session).pushArg(path).invoke().asModule();
  }

  private static byte[] getBytesFromFile(File file) throws IOException {
    // Get the size of the file
    long length = file.length();

    if (length > Integer.MAX_VALUE) {
      throw new IOException("File " + file.getName() + " is too large!");
    }

    // cannot create an array using a long type.
    byte[] bytes = new byte[(int) length];

    // Read in the bytes
    int offset = 0;
    int numRead = 0;

    InputStream is = new FileInputStream(file);
    try {
      while (
          offset < bytes.length && (numRead = is.read(bytes, offset, bytes.length - offset)) >= 0) {
        offset += numRead;
      }
    } finally {
      is.close();
    }

    // Ensure all the bytes have been read in
    if (offset < bytes.length) {
      throw new IOException("Could not completely read file " + file.getName());
    }
    return bytes;
  }
}
