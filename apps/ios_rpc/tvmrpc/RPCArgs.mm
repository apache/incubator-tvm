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

#import "RPCArgs.h"

#import <Foundation/Foundation.h>

#import "../../../src/support/socket.h"

#import <string>

using std::string;

namespace {
bool IsNumber(const std::string& str) {
  return !str.empty() && str.find_first_not_of("0123456789") == std::string::npos;
}
}  // namespace

const char* kUsage =
    "\n"
    "iOS tvmrpc application supported flags:\n"
    "--host_url      The tracker/proxy address, Default=0.0.0.0\n"
    "--host_port     The tracker/proxy port, Default=9190\n"
    "--key           The key used to identify the device type in tracker. Default=\"\"\n"
    "--custom_addr   Custom IP Address to Report to RPC Tracker. Default=\"\"\n"
    "--immediate_connect   No UI interconnection, connect to tracker immediately. Default=False\n"
    "--verbose       Allow to print status info to std out. Default=False\n"
    "--server_mode   Server mode. Can be \"standalone\", \"proxy\" or \"tracker\". "
    "Default=standalone \n"
    "\n";

struct RPCArgs_cpp {
  string host_url = "0.0.0.0";
  int host_port = 9190;

  string key;
  string custom_addr = "null";

  bool immediate_connect = false;
  bool verbose = false;
  RPCServerMode server_mode = RPCServerMode_Tracker;

  operator RPCArgs() const {
    return RPCArgs{.host_url = host_url.c_str(),
                   .host_port = host_port,
                   .key = key.c_str(),
                   .custom_addr = custom_addr.c_str(),
                   .verbose = verbose,
                   .immediate_connect = immediate_connect,
                   .server_mode = server_mode};
  };

  RPCArgs_cpp& operator=(const RPCArgs& args) {
    host_url = args.host_url;
    host_port = args.host_port;
    key = args.key;
    custom_addr = args.custom_addr;
    verbose = args.verbose;
    immediate_connect = args.immediate_connect;
    server_mode = args.server_mode;
    return *this;
  }
};

struct RPCArgs_cpp g_rpc_args;

static void restore_from_cache() {
  NSUserDefaults* defaults = [NSUserDefaults standardUserDefaults];

  auto get_string_from_cache = [defaults](const char* key) {
    NSString* ns_key = [NSString stringWithUTF8String:key];
    NSString* ns_val = [defaults stringForKey:ns_key];
    return std::string(ns_val != nil ? [ns_val UTF8String] : "");
  };

  auto get_int_from_cache = [defaults](const char* key) {
    NSString* ns_key = [NSString stringWithUTF8String:key];
    return static_cast<int>([defaults integerForKey:ns_key]);
  };

  g_rpc_args.host_url = get_string_from_cache("RPCArgs_url");
  g_rpc_args.host_port = get_int_from_cache("RPCArgs_port");
  g_rpc_args.key = get_string_from_cache("RPCArgs_key");
}

static void update_in_cache() {
  NSUserDefaults* defaults = [NSUserDefaults standardUserDefaults];

  [defaults setObject:[NSString stringWithUTF8String:g_rpc_args.host_url.c_str()]
               forKey:@"RPCArgs_url"];
  [defaults setInteger:g_rpc_args.host_port forKey:@"RPCArgs_port"];
  [defaults setObject:[NSString stringWithUTF8String:g_rpc_args.key.c_str()] forKey:@"RPCArgs_key"];
}

string GetCmdOption(int argc, char* argv[], string option, bool key = false) {
  string cmd;
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg.find(option) == 0) {
      if (key) {
        cmd = argv[i];
        return cmd;
      }
      // We assume "=" is the end of option.
      TVM_FFI_ICHECK_EQ(*option.rbegin(), '=');
      cmd = arg.substr(arg.find('=') + 1);
      return cmd;
    }
  }
  return cmd;
}

void update_rpc_args(int argc, char* argv[]) {
  restore_from_cache();
  RPCArgs_cpp& args = g_rpc_args;

  using tvm::support::ValidateIP;
  constexpr int MAX_PORT_NUM = 65535;

  const string immediate_connect = GetCmdOption(argc, argv, "--immediate_connect", true);
  args.immediate_connect = !immediate_connect.empty();

  const string verbose = GetCmdOption(argc, argv, "--verbose", true);
  args.verbose = !verbose.empty();

  const string server_mode = GetCmdOption(argc, argv, "--server_mode=", false);
  if (!server_mode.empty()) {
    if (server_mode == "tracker") {
      args.server_mode = RPCServerMode_Tracker;
    } else if (server_mode == "proxy") {
      args.server_mode = RPCServerMode_Proxy;
    } else if (server_mode == "standalone") {
      args.server_mode = RPCServerMode_Standalone;
    } else {
      LOG(WARNING) << "Wrong server_mode value.";
      LOG(INFO) << kUsage;
      exit(1);
    }
  }

  const string host_url = GetCmdOption(argc, argv, "--host_url=");
  if (!host_url.empty()) {
    if (!ValidateIP(host_url)) {
      LOG(WARNING) << "Wrong tracker address format.";
      LOG(INFO) << kUsage;
      exit(1);
    }
    args.host_url = host_url;
  }

  const string host_port = GetCmdOption(argc, argv, "--host_port=");
  if (!host_port.empty()) {
    if (!IsNumber(host_port) || stoi(host_port) > MAX_PORT_NUM) {
      LOG(WARNING) << "Wrong trackerport number.";
      LOG(INFO) << kUsage;
      exit(1);
    }
    args.host_port = stoi(host_port);
  }

  const string key = GetCmdOption(argc, argv, "--key=");
  if (!key.empty()) {
    args.key = key;
  }

  const string custom_addr = GetCmdOption(argc, argv, "--custom_addr=");
  if (!custom_addr.empty()) {
    if (!ValidateIP(custom_addr)) {
      LOG(WARNING) << "Wrong custom address format.";
      LOG(INFO) << kUsage;
      exit(1);
    }
    args.custom_addr = '"' + custom_addr + '"';
  }

  update_in_cache();
}

RPCArgs get_current_rpc_args(void) { return g_rpc_args; }

void set_current_rpc_args(RPCArgs args) {
  g_rpc_args = args;
  update_in_cache();
}
