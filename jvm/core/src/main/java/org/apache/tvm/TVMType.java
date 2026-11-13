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

public class TVMType {
  public static final int INT = 0;
  public static final int UINT = 1;
  public static final int FLOAT = 2;
  public static final int HANDLE = 4;

  public final int typeCode;
  public final int bits;
  public final int numOfBytes;
  public final int lanes;

  /**
   * TVMType constructor.
   * @param typeStr type name, e.g., "float32", "float64", "uint8", etc.
   * @param lanes Tensor lanes.
   */
  public TVMType(String typeStr, int lanes) {
    this.lanes = lanes;
    int bitsTemp = 0;
    if (typeStr.startsWith("int")) {
      typeCode = INT;
      bitsTemp = Integer.parseInt(typeStr.substring(3));
    } else if (typeStr.startsWith("uint")) {
      typeCode = UINT;
      bitsTemp = Integer.parseInt(typeStr.substring(4));
    } else if (typeStr.startsWith("float")) {
      typeCode = FLOAT;
      bitsTemp = Integer.parseInt(typeStr.substring(5));
    } else if (typeStr.startsWith("handle")) {
      typeCode = HANDLE;
      bitsTemp = 64;
    } else {
      throw new IllegalArgumentException("Do not know how to handle type " + typeStr);
    }
    bits = (bitsTemp == 0) ? 32 : bitsTemp;
    if ((bits & (bits - 1)) != 0 || bits < 8) {
      throw new IllegalArgumentException("Do not know how to handle type " + typeStr);
    }
    numOfBytes = bits / 8;
  }

  public TVMType(String typeStr) {
    this(typeStr, 1);
  }

  @Override
  public int hashCode() {
    return (typeCode << 16) | (bits << 8) | lanes;
  }

  @Override
  public boolean equals(Object other) {
    if (other != null && other instanceof TVMType) {
      TVMType otherInst = (TVMType) other;
      return (bits == otherInst.bits) && (typeCode == otherInst.typeCode)
          && (lanes == otherInst.lanes);
    }
    return false;
  }

  @Override
  public String toString() {
    String typeCodeStr;
    switch (typeCode) {
      case INT:
        typeCodeStr = "int";
        break;
      case UINT:
        typeCodeStr = "uint";
        break;
      case FLOAT:
        typeCodeStr = "float";
        break;
      case HANDLE:
        typeCodeStr = "handle";
        break;
      default:
        typeCodeStr = "Unknown";
        break;
    }
    String str = typeCodeStr + bits;
    if (lanes != 1) {
      str += "x" + lanes;
    }
    return str;
  }
}
