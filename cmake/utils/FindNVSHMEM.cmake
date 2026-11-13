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

#######################################################
# Enhanced version of find NVSHMEM.
#
# Usage:
#   find_nvshmem(${TVM_USE_NVSHMEM})
#
# - When TVM_USE_NVSHMEM=ON, use auto search
# - When TVM_USE_NVSHMEM=/path/to/installed/nvshmem, use the installed nvshmem path.
#   Can be useful when nvshmem is installed at specified location.
#
# Provide variables:
#
# - NVSHMEM_FOUND
# - NVSHMEM_INCLUDE_DIR
# - NVSHMEM_LIB_DIR
#

macro(find_nvshmem use_nvshmem)
  set(__use_nvshmem ${use_nvshmem})
  if(IS_DIRECTORY ${__use_nvshmem})
    set(__nvshmem_path ${__use_nvshmem})
    message(STATUS "Custom NVSHMEM PATH=" ${__use_nvshmem})
   elseif(IS_DIRECTORY $ENV{NVSHMEM_HOME})
     set(__nvshmem_path $ENV{NVSHMEM_HOME})
   else()
     set(__nvshmem_path "")
   endif()

   find_package(NVSHMEM HINTS ${__nvshmem_path}/lib/cmake/nvshmem/)

  if(NVSHMEM_FOUND)
    message(STATUS "NVSHMEM_INCLUDE_DIR=" ${NVSHMEM_INCLUDE_DIR})
    message(STATUS "NVSHMEM_LIB_DIR=" ${NVSHMEM_LIB_DIR})
  endif(NVSHMEM_FOUND)
endmacro(find_nvshmem)
