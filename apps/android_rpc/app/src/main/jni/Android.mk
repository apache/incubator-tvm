# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

LOCAL_PATH := $(call my-dir)
MY_PATH := $(LOCAL_PATH)

include $(CLEAR_VARS)

LOCAL_PATH := $(MY_PATH)
ROOT_PATH := $(MY_PATH)/../../../../../..

ifndef config
	ifneq ("$(wildcard ./config.mk)","")
	  config ?= config.mk
	else
	  config ?= make/config.mk
	endif
endif

include $(config)

LOCAL_SRC_FILES := org_apache_tvm_native_c_api.cc
LOCAL_LDFLAGS := -L$(SYSROOT)/usr/lib/ -llog

LOCAL_C_INCLUDES := $(ROOT_PATH)/include $(ROOT_PATH)/3rdparty/tvm-tirx/include \
                    $(ROOT_PATH)/3rdparty/tvm-tirx/3rdparty/tvm-ffi/include \
                    $(ROOT_PATH)/3rdparty/tvm-tirx/3rdparty/tvm-ffi/3rdparty/dlpack/include

ifeq ($(USE_OPENCL), 1)
    LOCAL_C_INCLUDES += $(OPENCL_SDK)/include
endif

LOCAL_MODULE = tvm4j_runtime_packed

LOCAL_CPP_FEATURES += exceptions
LOCAL_LDLIBS += -latomic
LOCAL_ARM_MODE := arm

ifdef ADD_C_INCLUDES
	LOCAL_C_INCLUDES += $(ADD_C_INCLUDES)
endif

ifdef ADD_LDLIBS
	LOCAL_LDLIBS += $(ADD_LDLIBS)
endif

include $(BUILD_SHARED_LIBRARY)
