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

/*
 * \file src/runtime/contrib/amx/amx_config.cc
 * \brief extraction of AMX configuration on x86 platforms
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#ifdef __linux__
#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <tvm/runtime/logging.h>
#include <unistd.h>

#endif

namespace tvm {
namespace runtime {

#ifdef __linux__
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

typedef struct __tile_config {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[8]; /* Colum size of each tmm register in bytes */
  uint16_t reserved_1[8];
  uint8_t rows[8]; /* Row size of each tmm reg in bytes */
  uint8_t reserved_2[8];
} __tilecfg;

typedef union __union_tile_config {
  __tilecfg s;
  uint8_t a[64];
} __tilecfg_u;

void init_tile_config(__tilecfg_u* dst, uint16_t cols, uint8_t rows) {
  dst->s.palette_id = 1;
  dst->s.start_row = 0;

  for (int i = 0; i < 14; i++) dst->s.reserved_0[i] = 0;

  for (int i = 0; i < 8; i++) {
    dst->s.colsb[i] = cols;
    dst->s.rows[i] = rows;
    dst->s.reserved_1[i] = 0;
    dst->s.reserved_2[i] = 0;
  }

  _tile_loadconfig(dst->a);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("runtime.amx_tileconfig", [](ffi::PackedArgs args, ffi::Any* rv) {
    int rows = args[0].cast<int>();
    int cols = args[1].cast<int>();
    LOG(INFO) << "rows: " << rows << ", cols:" << cols;
    // -----------Config for AMX tile resgister----------------------
    __tilecfg_u cfg;
    init_tile_config(&cfg, cols, rows);

    *rv = 1;
    return;
  });
}

// register a global packed function in c++，to init the system for AMX config
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("runtime.amx_init", [](ffi::PackedArgs args, ffi::Any* rv) {
    // -----------Detect and request for AMX control----------------------
    uint64_t bitmask = 0;
    int64_t status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) {
      *rv = 0;
      TVM_FFI_THROW(InternalError) << "errno:" << errno << ", " << strerror(errno);
      TVM_FFI_THROW(InternalError)
          << "status[0]: " << status << ", bitmask: " << bitmask
          << ", XFEATURE_XTILEDATA setup is failed, TMUL feature is not allowed.";
      return;
    }
    if (bitmask & XFEATURE_MASK_XTILEDATA) {
      *rv = 1;
      return;
    }  // TILE_DATA feature was not detected

    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    // if XFEATURE_XTILEDATA setup is failed, TMUL usage is not allowed
    if (0 != status) {
      *rv = 0;
      TVM_FFI_THROW(InternalError) << "errno:" << errno << ", " << strerror(errno);
      TVM_FFI_THROW(InternalError)
          << "status[1]: " << status << ", bitmask: " << bitmask
          << ", XFEATURE_XTILEDATA setup is failed, TMUL usage is not allowed.";
      return;
    }

    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    // if XFEATURE_XTILEDATA setup is failed, can't use TMUL
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) {
      *rv = 0;
      TVM_FFI_THROW(InternalError) << "errno:" << errno << ", " << strerror(errno);
      TVM_FFI_THROW(InternalError) << "status[2]: " << status << ", bitmask: " << bitmask
                                   << ", XFEATURE_XTILEDATA setup is failed, can't use TMUL.";
      return;
    }

    // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
    *rv = 1;
    return;
  });
}

#endif
}  // namespace runtime
}  // namespace tvm
