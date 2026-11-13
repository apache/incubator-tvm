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
 * \file src/s_tir/backend/adreno/texture_utils.h
 * \brief Texture utilities
 */
#ifndef TVM_S_TIR_BACKEND_ADRENO_TEXTURE_UTILS_H_
#define TVM_S_TIR_BACKEND_ADRENO_TEXTURE_UTILS_H_

#include <tvm/ffi/error.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>

#include <cstddef>
#include <cstdint>

namespace tvm {
namespace texture {

/*! \brief Structure to represent flattened texture shape */
template <typename T>
struct Texture2DShape {
  T width;
  T height;
  T depth;
  T channel;
};

/*!
 * \param shape_rank Rank N of the Nd-shape
 * \param convention Storage scope convention to use for flattening
 * \return The axis separator that defines the Nd shape partitioning in 2d
 */
inline size_t DefaultTextureLayoutSeparator(size_t shape_rank, runtime::MemoryScope convention) {
  static const auto scope_global_texture = runtime::MemoryScope::Get("global.texture");
  static const auto scope_global_texture_weight =
      runtime::MemoryScope::Get("global.texture-weight");
  static const auto scope_global_texture_nhwc = runtime::MemoryScope::Get("global.texture-nhwc");
  // Texture activation:
  // e.g. [N,C,H,W,c] -> Texture2d[N*C*H, W, c]
  // Texture weight:
  // e.g. [O,I,H,W,c] -> Texture2d[O, I*H*W, c]
  size_t separator = 0;
  if (convention == scope_global_texture) {
    separator = shape_rank - 2;
  } else if (convention == scope_global_texture_weight) {
    separator = 1;
  } else if (convention == scope_global_texture_nhwc) {
    separator = 2;
  } else {
    TVM_FFI_THROW(InternalError) << "Encountered unknown texture lowering convention: "
                                 << convention.name();
  }
  return separator;
}

/*!
 * \param shape Nd shape
 * \param rank Number of dimensions N of the Nd shape
 * \param axis The axis separator that splits the Nd axes into two sets
 * \return Width and height of the 2d shape
 */
template <typename T, typename S>
Texture2DShape<T> ApplyTexture2DFlattening(const S& shape, size_t rank, size_t axis) {
  TVM_FFI_ICHECK(axis < rank)
      << "Number of axes to flatten into rows must be less than shape rank for 2d flattening";
  Texture2DShape<T> texture{1, 1, 1, shape[rank - 1]};
  for (size_t i = 0; i < rank - 1; i++) {
    if (i < (axis - 1)) {
      texture.depth *= shape[i];
    } else if (i < axis) {
      texture.height *= shape[i];
    } else {
      texture.width *= shape[i];
    }
  }
  return texture;
}

inline bool IsTextureStorage(runtime::MemoryScope scope) {
  static ffi::reflection::TypeAttrColumn state_column(ffi::type_attr::kEnumState);
  auto state =
      state_column[runtime::MemoryScopeObj::_GetOrAllocRuntimeTypeIndex()].cast<ffi::EnumState>();
  // This attribute is supplied by the owning OpenCL backend registrations.
  if (auto column = state->attrs.Get("opencl.texture_storage")) {
    if (auto value = column->Get(scope)) return value->cast<bool>();
  }
  return false;
}

/*!
 * \brief Returns the physical backing memory size required for given specification
 * \param shape shape of tensor
 * \param bits dtype bits
 * \param lanes vectorization lanes
 * \param mem_scope the memory scope info
 * \param image_row_align image rowwise alignment size
 * \return returns the backing memory size
 */
template <typename T>
size_t GetTextureMemorySize(T shape, int bits, int lanes, runtime::MemoryScope mem_scope,
                            int image_row_align) {
  size_t axis = texture::DefaultTextureLayoutSeparator(shape.size(), mem_scope);
  auto tshape = ApplyTexture2DFlattening<int64_t>(shape, shape.size(), axis);

  auto pack_size = shape[shape.size() - 1];
  auto pixel_size = (bits * lanes + 7) / 8;
  auto row_size = tshape.width * pixel_size * pack_size;
  size_t row_pitch = (row_size + (image_row_align - 1)) & ~(image_row_align - 1);
  return row_pitch * tshape.height * tshape.depth;
}

}  // namespace texture
}  // namespace tvm
#endif  // TVM_S_TIR_BACKEND_ADRENO_TEXTURE_UTILS_H_
