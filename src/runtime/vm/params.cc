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
 * \file params.cc
 * \brief VM parameter dictionary serialization.
 */
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/tensor.h>
#include <tvm/support/bytes_io.h>
#include <tvm/support/io.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

namespace {
constexpr uint64_t kTVMTensorListMagic = 0xF7E58D4F05049CB7;

struct SimpleBinaryFileStream : public support::Stream {
 public:
  using Stream::Read;
  using Stream::Write;

  SimpleBinaryFileStream(const std::string& path, std::string mode) {
    const char* fname = path.c_str();

    TVM_FFI_ICHECK(mode == "wb" || mode == "rb") << "Only allowed modes are 'wb' and 'rb'";
    read_ = mode == "rb";
    fp_ = std::fopen(fname, mode.c_str());
    TVM_FFI_ICHECK(fp_ != nullptr) << "Unable to open file " << path;
  }
  virtual ~SimpleBinaryFileStream(void) { this->Close(); }
  virtual size_t Read(void* ptr, size_t size) {
    TVM_FFI_ICHECK(read_) << "File opened in write-mode, cannot read.";
    TVM_FFI_ICHECK(fp_ != nullptr) << "File is closed";
    return std::fread(ptr, 1, size, fp_);
  }
  virtual size_t Write(const void* ptr, size_t size) {
    TVM_FFI_ICHECK(!read_) << "File opened in read-mode, cannot write.";
    TVM_FFI_ICHECK(fp_ != nullptr) << "File is closed";
    size_t nwrite = std::fwrite(ptr, 1, size, fp_);
    int err = std::ferror(fp_);

    TVM_FFI_ICHECK_EQ(err, 0) << "SimpleBinaryFileStream.Write incomplete: " << std::strerror(err);
    return nwrite;
  }
  inline void Close(void) {
    if (fp_ != nullptr) {
      std::fclose(fp_);
      fp_ = nullptr;
    }
  }

 private:
  std::FILE* fp_ = nullptr;
  bool read_;
};  // class SimpleBinaryFileStream

}  // namespace

ffi::Map<ffi::String, ffi::Tensor> LoadParams(support::Stream* strm) {
  ffi::Map<ffi::String, ffi::Tensor> params;
  uint64_t header, reserved;
  TVM_FFI_ICHECK(strm->Read(&header)) << "Invalid parameters file format";
  TVM_FFI_ICHECK(header == kTVMTensorListMagic) << "Invalid parameters file format";
  TVM_FFI_ICHECK(strm->Read(&reserved)) << "Invalid parameters file format";

  std::vector<std::string> names;
  TVM_FFI_ICHECK(strm->Read(&names)) << "Invalid parameters file format";
  uint64_t sz;
  strm->Read(&sz);
  size_t size = static_cast<size_t>(sz);
  TVM_FFI_ICHECK(size == names.size()) << "Invalid parameters file format";
  for (size_t i = 0; i < size; ++i) {
    // The data_entry is allocated on device, Tensor.load always load the array into CPU.
    ffi::Tensor temp = LoadTensor(strm);
    params.Set(names[i], temp);
  }
  return params;
}

ffi::Map<ffi::String, ffi::Tensor> LoadParams(const std::string& param_blob) {
  support::BytesInStream strm(param_blob);
  return LoadParams(&strm);
}

void SaveParams(support::Stream* strm, const ffi::Map<ffi::String, ffi::Tensor>& params) {
  for (const auto& entry : params) {
    TVM_FFI_ICHECK(entry.second.defined());
  }
  std::vector<std::string> names;
  std::vector<const DLTensor*> arrays;
  for (auto& p : params) {
    names.push_back(p.first);
    arrays.push_back(p.second.GetDLTensorPtr());
  }

  uint64_t header = kTVMTensorListMagic, reserved = 0;
  strm->Write(header);
  strm->Write(reserved);
  strm->Write(names);
  {
    uint64_t sz = static_cast<uint64_t>(arrays.size());
    strm->Write(sz);
    for (size_t i = 0; i < sz; ++i) {
      tvm::runtime::SaveTensor(arrays[i], strm);
    }
  }
}

std::string SaveParams(const ffi::Map<ffi::String, ffi::Tensor>& params) {
  std::string result;
  support::BytesOutStream strm(&result);
  SaveParams(&strm, params);
  return result;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("vm.SaveParams",
           [](const ffi::Map<ffi::String, ffi::Tensor>& params) {
             std::string s = ::tvm::runtime::vm::SaveParams(params);
             return ffi::Bytes(std::move(s));
           })
      .def("vm.SaveParamsToFile",
           [](const ffi::Map<ffi::String, ffi::Tensor>& params, const ffi::String& path) {
             for (const auto& entry : params) {
               TVM_FFI_ICHECK(entry.second.defined());
             }
             SimpleBinaryFileStream strm(path, "wb");
             SaveParams(&strm, params);
           })
      .def("vm.LoadParams", [](const ffi::Bytes& s) { return ::tvm::runtime::vm::LoadParams(s); })
      .def("vm.LoadParamsFromFile", [](const ffi::String& path) {
        SimpleBinaryFileStream strm(path, "rb");
        return LoadParams(&strm);
      });
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
