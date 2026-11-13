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

package org.apache.tvm;

import java.util.HashMap;
import java.util.Map;

public class Device {
  /**
   * Provides the same information as the C++ enums DLDeviceType and
   * TVMDeviceExtType.
   */
  static final int kDLCPU = 1;
  static final int kDLCUDA = 2;
  static final int kDLCUDAHost = 3;
  static final int kDLOpenCL = 4;
  static final int kDLVulkan = 7;
  static final int kDLMetal = 8;
  static final int kDLVPI = 9;
  static final int kDLROCM = 10;
  static final int kDLROCMHost = 11;
  static final int kDLExtDev = 12;
  static final int kDLCUDAManaged = 13;
  static final int kDLOneAPI = 14;
  static final int kDLWebGPU = 15;
  static final int kDLHexagon = 16;

  private static final Map<Integer, String> DEVICE_TYPE_TO_NAME = new HashMap<Integer, String>();
  private static final Map<String, Integer> DEVICE_NAME_TO_TYPE = new HashMap<String, Integer>();

  static {
    DEVICE_TYPE_TO_NAME.put(kDLCPU, "cpu");
    DEVICE_TYPE_TO_NAME.put(kDLCUDA, "cuda");
    DEVICE_TYPE_TO_NAME.put(kDLOpenCL, "opencl");
    DEVICE_TYPE_TO_NAME.put(kDLVulkan, "vulkan");
    DEVICE_TYPE_TO_NAME.put(kDLMetal, "metal");
    DEVICE_TYPE_TO_NAME.put(kDLVPI, "vpi");
    DEVICE_TYPE_TO_NAME.put(kDLHexagon, "hexagon");

    DEVICE_NAME_TO_TYPE.put("cpu", kDLCPU);
    DEVICE_NAME_TO_TYPE.put("cuda", kDLCUDA);
    DEVICE_NAME_TO_TYPE.put("cl", kDLOpenCL);
    DEVICE_NAME_TO_TYPE.put("opencl", kDLOpenCL);
    DEVICE_NAME_TO_TYPE.put("vulkan", kDLVulkan);
    DEVICE_NAME_TO_TYPE.put("metal", kDLMetal);
    DEVICE_NAME_TO_TYPE.put("vpi", kDLVPI);
    DEVICE_NAME_TO_TYPE.put("hexagon", kDLHexagon);
  }

  /**
   * Construct a CPU device.
   * @param devId The device id
   * @return The created device
   */
  public static Device cpu(int devId) {
    return new Device(kDLCPU, devId);
  }

  public static Device cpu() {
    return cpu(0);
  }

  /**
   * Construct a CUDA GPU device.
   * @param devId The device id
   * @return The created device
   */
  public static Device cuda(int devId) {
    return new Device(kDLCUDA, devId);
  }

  public static Device cuda() {
    return cuda(0);
  }

  /**
   * Construct a OpenCL device.
   * @param devId The device id
   * @return The created device
   */
  public static Device opencl(int devId) {
    return new Device(kDLOpenCL, devId);
  }

  public static Device opencl() {
    return opencl(0);
  }

  /**
   * Construct a Vulkan device.
   * @param devId The device id
   * @return The created device
   */
  public static Device vulkan(int devId) {
    return new Device(kDLVulkan, devId);
  }

  public static Device vulkan() {
    return vulkan(0);
  }

  /**
   * Construct a metal device.
   * @param devId The device id
   * @return The created device
   */
  public static Device metal(int devId) {
    return new Device(kDLMetal, devId);
  }

  public static Device metal() {
    return metal(0);
  }

  /**
   * Construct a VPI simulated device.
   * @param devId The device id
   * @return The created device
   */
  public static Device vpi(int devId) {
    return new Device(kDLVPI, devId);
  }

  public static Device vpi() {
    return vpi(0);
  }

  /**
   * Construct a Hexagon device.
   * @param devId The device id
   * @return The created device
   */
  public static Device hexagon(int devId) {
    return new Device(kDLHexagon, devId);
  }

  public static Device hexagon() {
    return hexagon(0);
  }

  public final int deviceType;
  public final int deviceId;

  public Device(int deviceType, int deviceId) {
    this.deviceType = deviceType;
    this.deviceId = deviceId;
  }

  public Device(String deviceType, int deviceId) {
    this(DEVICE_NAME_TO_TYPE.get(deviceType), deviceId);
  }

  /**
   * Whether this device exists.
   * @return true if exists.
   */
  public boolean exist() {
    TVMValue ret = APIInternal.get("runtime.GetDeviceAttr")
                       .pushArg(deviceType)
                       .pushArg(deviceId)
                       .pushArg("exist")
                       .invoke();
    return ((TVMValueLong) ret).value != 0;
  }

  /**
   * Maximum number of threads on each block.
   * @return the maximum thread number.
   */
  public long maxThreadsPerBlock() {
    TVMValue ret = APIInternal.get("runtime.GetDeviceAttr")
                       .pushArg(deviceType)
                       .pushArg(deviceId)
                       .pushArg("max_threads_per_block")
                       .invoke();
    return ((TVMValueLong) ret).value;
  }

  /**
   * Number of threads that executes in concurrent.
   * @return the thread number.
   */
  public long warpSize() {
    TVMValue ret = APIInternal.get("runtime.GetDeviceAttr")
                       .pushArg(deviceType)
                       .pushArg(deviceId)
                       .pushArg("warp_size")
                       .invoke();
    return ret.asLong();
  }

  /**
   * Synchronize until jobs finished at the device.
   */
  public void sync() {
    Base.checkCall(Base._LIB.tvmSynchronize(deviceType, deviceId));
  }

  @Override
  public int hashCode() {
    return (deviceType << 16) | deviceId;
  }

  @Override
  public boolean equals(Object other) {
    if (other != null && other instanceof Device) {
      Device obj = (Device) other;
      return deviceId == obj.deviceId && deviceType == obj.deviceType;
    }
    return false;
  }

  @Override
  public String toString() {
    return String.format("%s(%d)", DEVICE_TYPE_TO_NAME.get(deviceType), deviceId);
  }
}
