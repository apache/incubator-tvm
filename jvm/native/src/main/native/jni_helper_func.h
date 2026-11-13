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
/*!
 * \file jni_helper_func.h
 * \brief Helper functions for operating JVM objects
 */
#include <jni.h>

#ifndef TVM4J_JNI_MAIN_NATIVE_JNI_HELPER_FUNC_H_
#define TVM4J_JNI_MAIN_NATIVE_JNI_HELPER_FUNC_H_

// Helper functions for RefXXX getter & setter
jlong getLongField(JNIEnv* env, jobject obj) {
  jclass refClass = env->FindClass("org/apache/tvm/Base$RefLong");
  jfieldID refFid = env->GetFieldID(refClass, "value", "J");
  jlong ret = env->GetLongField(obj, refFid);
  env->DeleteLocalRef(refClass);
  return ret;
}

jint getIntField(JNIEnv* env, jobject obj) {
  jclass refClass = env->FindClass("org/apache/tvm/Base$RefInt");
  jfieldID refFid = env->GetFieldID(refClass, "value", "I");
  jint ret = env->GetIntField(obj, refFid);
  env->DeleteLocalRef(refClass);
  return ret;
}

void setIntField(JNIEnv* env, jobject obj, jint value) {
  jclass refClass = env->FindClass("org/apache/tvm/Base$RefInt");
  jfieldID refFid = env->GetFieldID(refClass, "value", "I");
  env->SetIntField(obj, refFid, value);
  env->DeleteLocalRef(refClass);
}

void setLongField(JNIEnv* env, jobject obj, jlong value) {
  jclass refClass = env->FindClass("org/apache/tvm/Base$RefLong");
  jfieldID refFid = env->GetFieldID(refClass, "value", "J");
  env->SetLongField(obj, refFid, value);
  env->DeleteLocalRef(refClass);
}

void setStringField(JNIEnv* env, jobject obj, const char* value) {
  jclass refClass = env->FindClass("org/apache/tvm/Base$RefString");
  jfieldID refFid = env->GetFieldID(refClass, "value", "Ljava/lang/String;");
  env->SetObjectField(obj, refFid, env->NewStringUTF(value));
  env->DeleteLocalRef(refClass);
}

// Helper functions for TVMValue
jlong getTVMValueLongField(JNIEnv* env, jobject obj,
                           const char* clsname = "org/apache/tvm/TVMValueLong") {
  jclass cls = env->FindClass(clsname);
  jfieldID fid = env->GetFieldID(cls, "value", "J");
  jlong ret = env->GetLongField(obj, fid);
  env->DeleteLocalRef(cls);
  return ret;
}

jdouble getTVMValueDoubleField(JNIEnv* env, jobject obj) {
  jclass cls = env->FindClass("org/apache/tvm/TVMValueDouble");
  jfieldID fid = env->GetFieldID(cls, "value", "D");
  jdouble ret = env->GetDoubleField(obj, fid);
  env->DeleteLocalRef(cls);
  return ret;
}

jstring getTVMValueStringField(JNIEnv* env, jobject obj) {
  jclass cls = env->FindClass("org/apache/tvm/TVMValueString");
  jfieldID fid = env->GetFieldID(cls, "value", "Ljava/lang/String;");
  jstring ret = static_cast<jstring>(env->GetObjectField(obj, fid));
  env->DeleteLocalRef(cls);
  return ret;
}

jobject newTVMValueHandle(JNIEnv* env, jlong value) {
  jclass cls = env->FindClass("org/apache/tvm/TVMValueHandle");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(J)V");
  jobject object = env->NewObject(cls, constructor, value);
  env->DeleteLocalRef(cls);
  return object;
}

jobject newTVMValueLong(JNIEnv* env, jlong value) {
  jclass cls = env->FindClass("org/apache/tvm/TVMValueLong");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(J)V");
  jobject object = env->NewObject(cls, constructor, value);
  env->DeleteLocalRef(cls);
  return object;
}

jobject newTVMValueDouble(JNIEnv* env, jdouble value) {
  jclass cls = env->FindClass("org/apache/tvm/TVMValueDouble");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(D)V");
  jobject object = env->NewObject(cls, constructor, value);
  env->DeleteLocalRef(cls);
  return object;
}

jobject newTVMValueString(JNIEnv* env, const TVMFFIByteArray* value) {
  jstring jvalue = env->NewStringUTF(value->data);
  jclass cls = env->FindClass("org/apache/tvm/TVMValueString");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;)V");
  jobject object = env->NewObject(cls, constructor, jvalue);
  env->DeleteLocalRef(cls);
  env->DeleteLocalRef(jvalue);
  return object;
}

jobject newTVMValueBytes(JNIEnv* env, const TVMFFIByteArray* arr) {
  jbyteArray jarr = env->NewByteArray(arr->size);
  env->SetByteArrayRegion(jarr, 0, arr->size,
                          reinterpret_cast<jbyte*>(const_cast<char*>(arr->data)));
  jclass cls = env->FindClass("org/apache/tvm/TVMValueBytes");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "([B)V");
  jobject object = env->NewObject(cls, constructor, jarr);
  env->DeleteLocalRef(cls);
  env->DeleteLocalRef(jarr);
  return object;
}

jobject newModule(JNIEnv* env, jlong value) {
  jclass cls = env->FindClass("org/apache/tvm/Module");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(J)V");
  jobject object = env->NewObject(cls, constructor, value);
  env->DeleteLocalRef(cls);
  return object;
}

jobject newFunction(JNIEnv* env, jlong value) {
  jclass cls = env->FindClass("org/apache/tvm/Function");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(J)V");
  jobject object = env->NewObject(cls, constructor, value);
  env->DeleteLocalRef(cls);
  return object;
}

jobject newTensor(JNIEnv* env, jlong handle, jboolean isview) {
  jclass cls = env->FindClass("org/apache/tvm/TensorBase");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(JZ)V");
  jobject object = env->NewObject(cls, constructor, handle, isview);
  env->DeleteLocalRef(cls);
  return object;
}

jobject newTVMNull(JNIEnv* env) {
  jclass cls = env->FindClass("org/apache/tvm/TVMValueNull");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "()V");
  jobject object = env->NewObject(cls, constructor);
  env->DeleteLocalRef(cls);
  return object;
}

jobject newTVMObject(JNIEnv* env, jlong handle, jint type_index) {
  const char* class_name = "org/apache/tvm/TVMObject";
  std::string type_key = reinterpret_cast<tvm::ffi::Object*>(handle)->GetTypeKey();
  if (type_key == "rpc.RemoteDevice") {
    class_name = "org/apache/tvm/rpc/TVMRemoteDevice";
  } else if (type_key == "rpc.RemoteTensor") {
    class_name = "org/apache/tvm/rpc/RemoteTensor";
  }
  jclass cls = env->FindClass(class_name);
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(JI)V");
  jobject object = env->NewObject(cls, constructor, handle, type_index);
  env->DeleteLocalRef(cls);
  return object;
}

void fromJavaDType(JNIEnv* env, jobject jdtype, DLDataType* dtype) {
  jclass tvmTypeClass = env->FindClass("org/apache/tvm/DLDataType");
  dtype->code = (uint8_t)(env->GetIntField(jdtype, env->GetFieldID(tvmTypeClass, "typeCode", "I")));
  dtype->bits = (uint8_t)(env->GetIntField(jdtype, env->GetFieldID(tvmTypeClass, "bits", "I")));
  dtype->lanes = (uint16_t)(env->GetIntField(jdtype, env->GetFieldID(tvmTypeClass, "lanes", "I")));
  env->DeleteLocalRef(tvmTypeClass);
}

void fromJavaDevice(JNIEnv* env, jobject jdev, DLDevice* dev) {
  jclass deviceClass = env->FindClass("org/apache/tvm/Device");
  dev->device_type = static_cast<DLDeviceType>(
      env->GetIntField(jdev, env->GetFieldID(deviceClass, "deviceType", "I")));
  dev->device_id =
      static_cast<int>(env->GetIntField(jdev, env->GetFieldID(deviceClass, "deviceId", "I")));
  env->DeleteLocalRef(deviceClass);
}

jobject tvmRetValueToJava(JNIEnv* env, TVMFFIAny value) {
  using tvm::ffi::TypeIndex;
  switch (value.type_index) {
    case TypeIndex::kTVMFFINone: {
      return newTVMNull(env);
    }
    case TypeIndex::kTVMFFIBool: {
      // use long for now to represent bool
      return newTVMValueLong(env, static_cast<jlong>(value.v_int64));
    }
    case TypeIndex::kTVMFFIInt: {
      return newTVMValueLong(env, static_cast<jlong>(value.v_int64));
    }
    case TypeIndex::kTVMFFIFloat: {
      return newTVMValueDouble(env, static_cast<jdouble>(value.v_float64));
    }
    case TypeIndex::kTVMFFIOpaquePtr: {
      return newTVMValueHandle(env, reinterpret_cast<jlong>(value.v_ptr));
    }
    case TypeIndex::kTVMFFIModule: {
      return newModule(env, reinterpret_cast<jlong>(value.v_obj));
    }
    case TypeIndex::kTVMFFIFunction: {
      return newFunction(env, reinterpret_cast<jlong>(value.v_obj));
    }
    case TypeIndex::kTVMFFIDLTensorPtr: {
      return newTensor(env, reinterpret_cast<jlong>(value.v_ptr), true);
    }
    case TypeIndex::kTVMFFITensor: {
      return newTensor(env, reinterpret_cast<jlong>(value.v_obj), false);
    }
    case TypeIndex::kTVMFFISmallStr: {
      TVMFFIByteArray arr = TVMFFISmallBytesGetContentByteArray(&value);
      return newTVMValueString(env, &arr);
    }
    case TypeIndex::kTVMFFIStr: {
      return newTVMValueString(env, TVMFFIBytesGetByteArrayPtr(value.v_obj));
    }
    case TypeIndex::kTVMFFISmallBytes: {
      TVMFFIByteArray arr = TVMFFISmallBytesGetContentByteArray(&value);
      return newTVMValueBytes(env, &arr);
    }
    case TypeIndex::kTVMFFIBytes: {
      jobject ret = newTVMValueBytes(env, TVMFFIBytesGetByteArrayPtr(value.v_obj));
      TVMFFIObjectDecRef(value.v_obj);
      return ret;
    }
    default: {
      if (value.type_index >= TypeIndex::kTVMFFIStaticObjectBegin) {
        return newTVMObject(env, reinterpret_cast<jlong>(value.v_obj), value.type_index);
      }
      TVM_FFI_THROW(RuntimeError) << "Do NOT know how to handle return type_index "
                                  << value.type_index;
      TVM_FFI_UNREACHABLE();
    }
  }
}

#endif  // TVM4J_JNI_MAIN_NATIVE_JNI_HELPER_FUNC_H_
