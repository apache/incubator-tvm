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
 * \file rpc_env.cc
 * \brief Server environment of the RPC.
 */
#include "rpc_env.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/logging.h>

#ifndef _WIN32
#include <sys/wait.h>
#endif

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace {
int Execute(std::string cmd, std::string* err_msg) {
  std::array<char, 128> buffer;
  cmd += " 2>&1";
#ifdef _WIN32
  FILE* fd = _popen(cmd.c_str(), "r");
#else
  FILE* fd = popen(cmd.c_str(), "r");
#endif
  while (fgets(buffer.data(), buffer.size(), fd) != nullptr) {
    *err_msg += buffer.data();
  }
#ifdef _WIN32
  int status = _pclose(fd);
  return status != 3 ? status : 255;
#else
  int status = pclose(fd);
  return WIFEXITED(status) ? WEXITSTATUS(status) : 255;
#endif
}

std::string GenerateUntarCommand(const std::string& tar_file, const std::string& output_dir) {
  std::string untar_cmd;
  untar_cmd.reserve(512);
#if defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)
  untar_cmd += "tar -C ";
  untar_cmd += output_dir;
  untar_cmd += " -zxf ";
  untar_cmd += tar_file;
#elif defined(_WIN32)
  untar_cmd += "python -m tarfile -e ";
  untar_cmd += tar_file;
  untar_cmd += " ";
  untar_cmd += output_dir;
#endif
  return untar_cmd;
}

}  // Anonymous namespace

namespace tvm {
namespace runtime {

RPCEnv::RPCEnv(const std::string& wd) {
  std::error_code ec;
  if (!wd.empty()) {
    base_ = wd + "/.cache";
    std::filesystem::create_directories(base_, ec);
    if (ec) {
      LOG(WARNING) << "Failed to create directory " << base_ << " : " << ec.message();
    }
  } else {
#if defined(ANDROID) || defined(__ANDROID__)
    std::string pkg_name;
    if (std::ifstream cmdline("/proc/self/cmdline"); cmdline) {
      std::getline(cmdline, pkg_name, '\0');
    }
    std::string android_base_ = "/data/data/" + pkg_name + "/cache";
    // Check if application data directory exist. If not exist, usually means we run tvm_rpc from
    // adb shell terminal.
    if (!std::filesystem::is_directory(android_base_)) {
      // Tmp directory is always writable for 'shell' user.
      android_base_ = "/data/local/tmp";
    }
    base_ = android_base_ + "/rpc";
#elif !defined(_WIN32)
    base_ = std::filesystem::current_path(ec).string() + "/rpc";
    if (ec) {
      base_ = "./rpc";
    }
#else
    base_ = "./rpc";
#endif
    std::filesystem::create_directories(base_, ec);
    if (ec) {
      LOG(WARNING) << "Failed to create directory " << base_ << " : " << ec.message();
    }
  }
  std::filesystem::permissions(base_, std::filesystem::perms::all,
                               std::filesystem::perm_options::replace, ec);
  if (ec) {
    LOG(WARNING) << "Failed to grant permissions to " << base_ << " : " << ec.message();
  }

  ffi::Function::SetGlobal(
      "tvm.rpc.server.workpath",
      ffi::Function::FromTyped([this](const std::string& path) { return this->GetPath(path); }));

  ffi::Function::SetGlobal("tvm.rpc.server.listdir",
                           ffi::Function::FromTyped([this](const std::string& path) {
                             std::string dir = this->GetPath(path);
                             std::ostringstream os;
                             for (auto d : ListDir(dir)) {
                               os << d << ",";
                             }
                             return os.str();
                           }));

  ffi::Function::SetGlobal("tvm.rpc.server.load_module",
                           ffi::Function::FromTyped([this](const std::string& path) {
                             std::string file_name = this->GetPath(path);
                             file_name = BuildSharedLibrary(file_name);
                             LOG(INFO) << "Load module from " << file_name << " ...";
                             return ffi::Module::LoadFromFile(file_name);
                           }));

  ffi::Function::SetGlobal("tvm.rpc.server.download_linked_module",
                           ffi::Function::FromTyped([this](const std::string& path) {
                             std::string file_name = this->GetPath(path);
                             file_name = BuildSharedLibrary(file_name);
                             std::string bin;

                             std::ifstream fs(file_name, std::ios::in | std::ios::binary);
                             TVM_FFI_ICHECK(!fs.fail()) << "Cannot open " << file_name;
                             fs.seekg(0, std::ios::end);
                             size_t size = static_cast<size_t>(fs.tellg());
                             fs.seekg(0, std::ios::beg);
                             bin.resize(size);
                             fs.read(bin.data(), size);
                             LOG(INFO) << "Send linked module " << file_name << " to client";
                             return ffi::Bytes(bin);
                           }));
}

/*!
 * \brief GetPath To get the work path from packed function
 * \param file_name The file name
 * \return The full path of file.
 */
std::string RPCEnv::GetPath(const std::string& file_name) const {
  // we assume file_name starts with "/" means file_name is the exact path
  // and does not create /.rpc/
  return !file_name.empty() && file_name[0] == '/' ? file_name : base_ + "/" + file_name;
}

/*!
 * \brief Remove The RPC Environment cleanup function
 */
void RPCEnv::CleanUp() const {
  std::error_code ec;
  std::filesystem::remove_all(base_, ec);
  if (ec) {
    LOG(WARNING) << "Cleanup " << base_ << " failed: " << ec.message();
  }
}

/*!
 * \brief ListDir get the list of files in a directory
 * \param dirname The root directory name
 * \return vector Files in directory.
 */
std::vector<std::string> ListDir(const std::string& dirname) {
  std::error_code ec;
  std::vector<std::string> vec;
  auto iter = std::filesystem::directory_iterator(dirname, ec);
  if (ec) {
    if (ec == std::errc::no_such_file_or_directory) return vec;
    TVM_FFI_THROW(InternalError) << "ListDir " << dirname << " error: " << ec.message();
  }
  for (const auto& entry : iter) {
    vec.push_back(entry.path().generic_string());
  }
  return vec;
}

#if defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)
/*!
 * \brief LinuxShared Creates a linux shared library
 * \param output The output file name
 * \param files The files for building
 * \param options The compiler options
 * \param cc The compiler
 */
void LinuxShared(const std::string output, const std::vector<std::string>& files,
                 std::string options = "", std::string cc = "g++") {
  std::string cmd = cc;
  cmd += " -shared -fPIC ";
  cmd += " -o " + output;
  for (auto f = files.begin(); f != files.end(); ++f) {
    cmd += " " + *f;
  }
  cmd += " " + options;
  std::string err_msg;
  auto executed_status = Execute(cmd, &err_msg);
  if (executed_status) {
    TVM_FFI_THROW(InternalError) << err_msg;
  }
}
#endif

#ifdef _WIN32
/*!
 * \brief WindowsShared Creates a Windows shared library
 * \param output The output file name
 * \param files The files for building
 * \param options The compiler options
 * \param cc The compiler
 */
void WindowsShared(const std::string& output, const std::vector<std::string>& files,
                   const std::string& options = "", const std::string& cc = "clang") {
  std::string cmd = cc;
  cmd += " -O2 -flto=full -fuse-ld=lld-link -shared ";
  cmd += " -o " + output;
  for (const auto& file : files) {
    cmd += " " + file;
  }
  cmd += " " + options;
  std::string err_msg;
  const auto executed_status = Execute(cmd, &err_msg);
  if (executed_status) {
    TVM_FFI_THROW(InternalError) << err_msg;
  }
}
#endif

/*!
 * \brief CreateShared Creates a shared library
 * \param output The output file name
 * \param files The files for building
 */
void CreateShared(const std::string& output, const std::vector<std::string>& files) {
#if defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)
  LinuxShared(output, files);
#elif defined(_WIN32)
  WindowsShared(output, files);
#else
  TVM_FFI_THROW(InternalError) << "Operating system not supported";
#endif
}

std::string BuildSharedLibrary(std::string file) {
  if (file.ends_with(".so") || file.ends_with(".dll") || file.ends_with(".dylib")) {
    return file;
  }

  std::string file_name = file + ".so";
  if (file.ends_with(".o")) {
    CreateShared(file_name, {file});
  } else if (file.ends_with(".tar")) {
    const std::string tmp_dir = "./rpc/tmp/";
    std::error_code ec;
    std::filesystem::create_directories(tmp_dir, ec);
    if (ec) {
      LOG(WARNING) << "Failed to create directory " << tmp_dir << " : " << ec.message();
    }
    std::filesystem::permissions(tmp_dir, std::filesystem::perms::all,
                                 std::filesystem::perm_options::replace, ec);
    if (ec) {
      LOG(WARNING) << "Failed to grant permissions to " << tmp_dir << " : " << ec.message();
    }

    const std::string cmd = GenerateUntarCommand(file, tmp_dir);

    std::string err_msg;
    const int executed_status = Execute(cmd, &err_msg);
    if (executed_status) {
      TVM_FFI_THROW(InternalError) << err_msg;
    }
    CreateShared(file_name, ListDir(tmp_dir));
    std::filesystem::remove_all(tmp_dir, ec);
    if (ec) {
      LOG(WARNING) << "Remove " << tmp_dir << " failed: " << ec.message();
    }
  } else {
    file_name = file;
  }
  return file_name;
}

}  // namespace runtime
}  // namespace tvm
