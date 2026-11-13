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

if(TVM_USE_AMX)
    file(GLOB AMX_RUNTIME_CONFIG CONFIGURE_DEPENDS src/runtime/extra/contrib/amx/amx_config.cc)
    add_library(tvm_amx_objs OBJECT ${AMX_RUNTIME_CONFIG})
    tvm_configure_target(tvm_amx_objs)
    target_link_libraries(tvm_amx_objs PRIVATE tvm_runtime_extra_defs)
    target_compile_options(tvm_amx_objs PRIVATE -march=sapphirerapids)
    target_link_libraries(tvm_runtime_extra PRIVATE tvm_amx_objs)
    message(STATUS "Build with Intel AMX support...")
endif()
