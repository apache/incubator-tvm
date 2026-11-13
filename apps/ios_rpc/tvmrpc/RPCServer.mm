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
 * \file ViewController.mm
 */

#import "RPCServer.h"

#include <tvm/ffi/function.h>

#include <random>
#include <string>

// To get device WiFi IP
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>

// TVM internal header to access Magic keys like kRPCMagic and others
#include "../../../src/runtime/rpc/rpc_endpoint.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Message handling function for event driven server.
 *
 * \param in_bytes The incoming bytes.
 * \param event_flag  1: read_available, 2: write_avaiable.
 * \return State flag.
 *     1: continue running, no need to write,
 *     2: need to write
 *     0: shutdown
 */
using FEventHandler = ffi::Function;

/*!
 * \brief Create a server event handler.
 *
 * \param outputStream The output stream used to send outputs.
 * \param name The name of the server.
 * \param remote_key The remote key
 * \return The event handler.
 */
FEventHandler CreateServerEventHandler(NSOutputStream* outputStream, std::string name,
                                       std::string remote_key) {
  auto event_handler_factory = tvm::ffi::Function::GetGlobal("rpc.CreateEventDrivenServer");
  TVM_FFI_ICHECK(event_handler_factory.has_value())
      << "You are using tvm_runtime module built without RPC support. "
      << "Please rebuild it with TVM_USE_RPC flag.";

  ffi::Function writer_func([outputStream](ffi::PackedArgs args, ffi::Any* rv) {
    TVMByteArray* data = args[0].ptr<TVMByteArray>();
    int64_t nbytes = [outputStream write:reinterpret_cast<const uint8_t*>(data->data)
                               maxLength:data->size];
    if (nbytes < 0) {
      NSLog(@"%@", [outputStream streamError].localizedDescription);
      throw tvm::Error("Stream error");
    }
    *rv = nbytes;
  });

  return (*event_handler_factory)(writer_func, name, remote_key);
}

/*!
 * \brief Helper function to query real IP of device in WiFi network
 * \return string with IPv4 in format "192.168.0.1" or "unknown" if cannot detect
 */
static std::string getWiFiAddress() {
  std::string address = "unknown";
  ifaddrs* interfaces = nullptr;

  int success = getifaddrs(&interfaces);
  if (success == 0) {
    ifaddrs* temp_addr = interfaces;
    while (temp_addr != NULL) {
      if (temp_addr->ifa_addr->sa_family == AF_INET) {
        // Check if interface is en0 which is the wifi connection on the iPhone
        if (std::string(temp_addr->ifa_name) == "en0") {
          address = inet_ntoa(((sockaddr_in*)temp_addr->ifa_addr)->sin_addr);
        }
      }
      temp_addr = temp_addr->ifa_next;
    }
  }

  freeifaddrs(interfaces);
  return address;
}

}  // namespace runtime
}  // namespace tvm

// Base class for any type of RPC servicing
@interface RPCServerBase : RPCServer

/*!
 * Methods should be implemented in inherited classes
 */
- (bool)onReadHandler;   // return true - continue feeding, false - stop, try to drain output buffer
- (bool)onWriteHandler;  // return true - continue draining, false - no data to write
- (void)onEndEncountered;  // called on disconnect or session decided that it's shutdown time
- (void)open;              // Initiate listening objects like i/o streams and other resources
- (void)close;             // Deinitialize resources opend in "open" method
@end

@implementation RPCServerBase {
  // Worker thread
  NSThread* worker_thread_;
  // Trigger to continue RunLoop processing inside worker_thread_
  BOOL shouldKeepRunning;
  // Input socket stream
 @protected
  NSInputStream* inputStream_;
  // Output socket stream
  NSOutputStream* outputStream_;
  // Temporal buffer with data to send
  std::string sendBuffer_;
  // Temporal receive buffer
  std::string recvBuffer_;
  // Requested data size to accumulate in recvBuffer_ before continue processing
  int requiredToRecv_;
}

/*!
 * Start internal worker thread with RunLoop and submit corresponding open handlers into it
 * Not blocking
 */
- (void)start {
  worker_thread_ = [[NSThread alloc] initWithBlock:^{
    @autoreleasepool {
      [self notifyState:RPCServerStatus_Launched];
      [self open];
      shouldKeepRunning = YES;
      while (shouldKeepRunning && [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode
                                                           beforeDate:[NSDate distantFuture]]);
      [self notifyState:RPCServerStatus_Stopped];
    }
  }];
  [worker_thread_ start];
}

/*!
 * Send message to workel thread runloop to finish processing
 * Not blocking
 */
- (void)stop {
  if (worker_thread_ == nil) return;

  [self performSelector:@selector(stop_) onThread:worker_thread_ withObject:nil waitUntilDone:NO];
  worker_thread_ = nil;  // TODO: is it valid? may be better to do that inside NSThread?
}

- (void)stop_ {
  [self close];
  shouldKeepRunning = NO;
}

/*!
 * Base implementation to setup i/o streams
 * Will connect to host and port specified in corresponding properties
 */
- (void)open {
  CFReadStreamRef readStream;
  CFWriteStreamRef writeStream;
  CFStreamCreatePairWithSocketToHost(NULL, (__bridge CFStringRef)self.host, self.port, &readStream,
                                     &writeStream);
  inputStream_ = (__bridge NSInputStream*)readStream;
  outputStream_ = (__bridge NSOutputStream*)writeStream;
  [inputStream_ setDelegate:self];
  [outputStream_ setDelegate:self];
  [inputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ open];
  [inputStream_ open];
}

/*!
 * Base implementation to setup i/o streams
 * Will assign i/o streams to provided socket connection.
 */
- (void)openWithSocket:(CFSocketNativeHandle)sock {
  CFReadStreamRef readStream;
  CFWriteStreamRef writeStream;
  CFStreamCreatePairWithSocket(NULL, sock, &readStream, &writeStream);
  inputStream_ = (__bridge NSInputStream*)readStream;
  outputStream_ = (__bridge NSOutputStream*)writeStream;
  [inputStream_ setDelegate:self];
  [outputStream_ setDelegate:self];
  [inputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ open];
  [inputStream_ open];
}

/*!
 * Close i/o streams associated with connection
 */
- (void)close {
  [inputStream_ close];
  [outputStream_ close];
  [inputStream_ removeFromRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ removeFromRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [inputStream_ setDelegate:nil];
  [outputStream_ setDelegate:nil];
  inputStream_ = nil;
  outputStream_ = nil;
}

/// Unimplemented stubs
- (bool)onReadHandler {
  return false;
}
- (bool)onWriteHandler {
  return false;
}
- (void)onEndEncountered {
}

/*!
 * Try to read data from stream and call processing handler
 */
- (void)tryToRead {
  const int kBufferSize = 4 << 10;  // 4kB buffer
  const int prev_size = recvBuffer_.size();
  recvBuffer_.resize(kBufferSize);
  size_t nbytes = [inputStream_ read:(uint8_t*)recvBuffer_.data() + prev_size
                           maxLength:recvBuffer_.size() - prev_size];
  recvBuffer_.resize(nbytes + prev_size);

  // feed while it accept or requested particulat buffer size
  while (!recvBuffer_.empty() && requiredToRecv_ <= recvBuffer_.size() && [self onReadHandler]);
}

/*!
 * Try to write remaining data to stream and call processing handler
 */
- (void)tryToWrite {
  if (!sendBuffer_.empty()) {
    size_t nbytes = [outputStream_ write:(uint8_t*)sendBuffer_.data() maxLength:sendBuffer_.size()];
    sendBuffer_.erase(0, nbytes);
  }
  // call write handler while it want be called and space is available
  while (sendBuffer_.empty() && [outputStream_ hasSpaceAvailable] && [self onWriteHandler]);
}

/*!
 * Main event handler of socket stream events
 */
- (void)stream:(NSStream*)strm handleEvent:(NSStreamEvent)event {
  std::string buffer;
  switch (event) {
    case NSStreamEventOpenCompleted: {
      // Nothing
      break;
    }
    case NSStreamEventHasBytesAvailable:
      if (strm == inputStream_) {
        [self tryToRead];
        if ([outputStream_ hasSpaceAvailable]) [self tryToWrite];
      }
      break;
    case NSStreamEventHasSpaceAvailable: {
      if (strm == outputStream_) {
        [self tryToWrite];
        if ([inputStream_ hasBytesAvailable]) [self tryToRead];
      }
      break;
    }
    case NSStreamEventErrorOccurred: {
      [self notifyError:[strm streamError].localizedDescription];
      break;
    }
    case NSStreamEventEndEncountered: {
      [self onEndEncountered];
      break;
    }
    default: {
      NSLog(@"Unknown event");
    }
  }
}

#pragma mark - Helpers

/*!
 * Set buffer to send into stream. Try to send immediately or submit to lazy sending
 * Non blocking operation
 */
- (void)toSend:(NSData*)data {
  sendBuffer_.append(static_cast<const char*>(data.bytes), data.length);

  // try to flush buffer
  NSInteger sent_size = [outputStream_ write:(uint8_t*)sendBuffer_.data()
                                   maxLength:sendBuffer_.size()];
  sendBuffer_.erase(0, sent_size);
}

/*!
 * Set buffer to send  in packet format [size, data]. Behaviour is same as for toSend.
 */
- (void)toSendPacked:(NSData*)data {
  uint32_t packet_size = data.length;
  [self toSend:[NSData dataWithBytes:&packet_size length:sizeof(packet_size)]];
  [self toSend:data];
}

/*!
 */
- (NSData*)requestInputDataWithSize:(NSInteger)size {
  if (recvBuffer_.size() < size) {
    requiredToRecv_ = size;
    return nil;
  }
  NSData* res = [NSData dataWithBytes:recvBuffer_.data() length:size];
  recvBuffer_.erase(0, size);
  return res;
}

/*!
 */
- (NSData*)requestInputDataPacked {
  uint32_t size;
  if (recvBuffer_.size() < sizeof(size)) {
    requiredToRecv_ = sizeof(size);
    return nil;
  }
  size = *(uint32_t*)recvBuffer_.data();
  if (recvBuffer_.size() < sizeof(size) + size) {
    requiredToRecv_ = sizeof(size) + size;
    return nil;
  }
  NSData* res = [NSData dataWithBytes:recvBuffer_.data() + sizeof(size) length:size];
  recvBuffer_.erase(0, sizeof(size) + size);
  return res;
};

#pragma mark - Notifiers

/*!
 * Notify external listener about error.
 * Also print error message to std out in case of Verbose mode
 */
- (void)notifyError:(NSString*)msg {
  // Duplicate error message in std output. Host launcher script may listen it.
  if (self.verbose) NSLog(@"[IOS-RPC] ERROR: %@", msg);
  if (self.delegate) [self.delegate onError:msg];
}

/*!
 * Notify external listener about server state changes.
 * Also print information to std out in case of Verbose mode
 */
- (void)notifyState:(RPCServerStatus)state {
  // Duplicate sattus changing in std output. Host launcher script may listen it.
  if (self.verbose) NSLog(@"[IOS-RPC] STATE: %d", state);
  if (self.delegate != nil) [self.delegate onStatusChanged:state];
}

@end

@interface RPCServerProxy : RPCServerBase
@end

typedef enum {
  RPCServerProxyState_Idle,
  RPCServerProxyState_HandshakeToSend,
  RPCServerProxyState_HandshakeToRecv,
  RPCServerProxyState_Processing,
} RPCServerProxyState;

@implementation RPCServerProxy {
  /// Original TVM RPC event handler
  tvm::runtime::FEventHandler handler_;
 @protected
  /// Sate of Proxy client implementation
  RPCServerProxyState state_;
}

- (instancetype)init {
  if (self = [super init]) {
    handler_ = nullptr;
    state_ = RPCServerProxyState_Idle;
  }
  return self;
}

/*!
 * Implement matching of internat state on state available for outside users
 */
- (void)setState:(RPCServerProxyState)new_state {
  // Send Connected notification because Proxy doesn't response until client connected.
  if (new_state == RPCServerProxyState_HandshakeToRecv)
    [self notifyState:RPCServerStatus_Connected];
  if (new_state == RPCServerProxyState_Idle) [self notifyState:RPCServerStatus_Disconnected];
  if (state_ == RPCServerProxyState_HandshakeToRecv && new_state == RPCServerProxyState_Processing)
    [self notifyState:RPCServerStatus_RPCSessionStarted];
  if (state_ == RPCServerProxyState_Processing && new_state == RPCServerProxyState_Idle)
    [self notifyState:RPCServerStatus_RPCSessionStarted];

  state_ = new_state;
}

- (bool)onWriteHandler {
  switch (state_) {
    case RPCServerProxyState_HandshakeToSend: {
      // Send together kRPCMagic and server descriptor because of Proxy
      int32_t code = tvm::runtime::kRPCMagic;
      [self toSend:[NSData dataWithBytes:&code length:sizeof(code)]];

      std::string full_key = std::string("server:") + self.key.UTF8String;
      [self toSendPacked:[NSData dataWithBytes:full_key.data() length:full_key.size()]];

      self.state = RPCServerProxyState_HandshakeToRecv;
      return TRUE;
    }
    case RPCServerProxyState_Processing: {
      try {
        TVMByteArray dummy{nullptr, 0};
        int flag = handler_(dummy, 2);
        if (flag == 0) {
          [self onEndEncountered];
        }
        return flag == 2;
      } catch (const tvm::Error& e) {
        [self close];
      }
      break;
    }
    default:
      // Nothing
      break;
  }
  return FALSE;
}

- (bool)onReadHandler {
  switch (state_) {
    case RPCServerProxyState_HandshakeToRecv: {
      int32_t code = tvm::runtime::kRPCMagic;
      NSData* data = [self requestInputDataWithSize:sizeof(code)];
      if (data == nil) return FALSE;

      if (*(int32_t*)data.bytes != tvm::runtime::kRPCMagic) {
        [self notifyError:@"Wrong response, is not RPC client."];
        [self close];
        return FALSE;
        break;
      }

      handler_ = tvm::runtime::CreateServerEventHandler(outputStream_, "iphone", "%toinit");

      self.state = RPCServerProxyState_Processing;
      return TRUE;
      break;
    }
    case RPCServerProxyState_Processing: {
      int flag = 1;
      if ([outputStream_ hasSpaceAvailable]) {
        flag |= 2;
      }
      // always try to write
      try {
        TVMByteArray arr{recvBuffer_.data(), recvBuffer_.size()};
        flag = handler_(arr, flag);
        recvBuffer_.clear();
        if (flag == 0) {
          [self onEndEncountered];
        }
        return flag == 1;
      } catch (const tvm::Error& e) {
        [self close];
      }
      break;
    }
    default:
      // Nothing
      break;
  }
  return FALSE;
}

- (void)onEndEncountered {
  // Automatic reconnection when session is finished.
  [self close];
  [self open];
}

- (void)open {
  [super open];
  self.state = RPCServerProxyState_HandshakeToSend;
}

- (void)close {
  [super close];
  handler_ = nullptr;
  self.state = RPCServerProxyState_Idle;
}

@end

@interface RPCServerStandalone : RPCServerProxy
@property(readonly) int rpc_port;
@end

@implementation RPCServerStandalone {
  // Socket to listen incoming connections
  CFSocketRef socket_;
  /// Current socket connection handler
  CFSocketNativeHandle connection_;
  /// Port range to try bind to socket
  int port_range_start;
  int port_range_end;
}

- (instancetype)init {
  if (self = [super init]) {
    connection_ = 0;
    port_range_start = 9090;
    port_range_end = 9099;
  }
  return self;
}

- (void)setState:(RPCServerProxyState)new_state {
  if (state_ == RPCServerProxyState_Idle && new_state == RPCServerProxyState_HandshakeToSend) {
    self.actual_port = _rpc_port;
    self.device_addr = [NSString stringWithUTF8String:tvm::runtime::getWiFiAddress().c_str()];
    if (self.verbose) {
      // Notify host runner script with actual address
      NSLog(@"[IOS-RPC] IP: %s", tvm::runtime::getWiFiAddress().c_str());
      NSLog(@"[IOS-RPC] PORT: %d", _rpc_port);
    }
    [self notifyState:RPCServerStatus_Connected];
  }
  if (new_state == RPCServerProxyState_Idle) [self notifyState:RPCServerStatus_Disconnected];
  if (state_ == RPCServerProxyState_HandshakeToRecv && new_state == RPCServerProxyState_Processing)
    [self notifyState:RPCServerStatus_RPCSessionStarted];
  if (state_ == RPCServerProxyState_Processing && new_state == RPCServerProxyState_HandshakeToSend)
    [self notifyState:RPCServerStatus_RPCSessionFinished];

  state_ = new_state;
}

- (void)handleConnect:(CFSocketNativeHandle)hdl {
  connection_ = hdl;
  [super openWithSocket:connection_];
  self.state = RPCServerProxyState_HandshakeToSend;
}

static void handleConnect(CFSocketRef socket, CFSocketCallBackType type, CFDataRef address,
                          const void* data, void* info) {
  RPCServerStandalone* it = (__bridge RPCServerStandalone*)(info);
  [it handleConnect:*static_cast<const CFSocketNativeHandle*>(data)];
}

- (void)open {
  CFSocketContext ctx{};
  ctx.info = (__bridge void*)self;

  socket_ = CFSocketCreate(kCFAllocatorDefault, PF_INET, SOCK_STREAM, IPPROTO_TCP,
                           kCFSocketAcceptCallBack, handleConnect, &ctx);
  self->_rpc_port = 0;

  // Try to bind with range
  for (int port = port_range_start; port < port_range_end; port++) {
    struct sockaddr_in sin;
    memset(&sin, 0, sizeof(sin));
    sin.sin_len = sizeof(sin);
    sin.sin_family = AF_INET;
    sin.sin_port = htons(port);
    sin.sin_addr.s_addr = INADDR_ANY;

    CFDataRef sincfd = CFDataCreate(kCFAllocatorDefault, (UInt8*)&sin, sizeof(sin));
    CFSocketError res = CFSocketSetAddress(socket_, sincfd);
    CFRelease(sincfd);
    if (res == kCFSocketSuccess) {
      self->_rpc_port = port;
      break;
    }
  }
  if (self->_rpc_port == 0) {
    @throw
        [NSException exceptionWithName:@"SocketError"
                                reason:[NSString stringWithFormat:@"Unable bind socket to port"
                                                                   "in range [%d, %d]",
                                                                  port_range_start, port_range_end]
                              userInfo:nil];
  }

  CFRunLoopSourceRef socketsource = CFSocketCreateRunLoopSource(kCFAllocatorDefault, socket_, 0);
  CFRunLoopAddSource(CFRunLoopGetCurrent(), socketsource, kCFRunLoopDefaultMode);

  self.state = RPCServerProxyState_HandshakeToSend;
}

- (void)closeSocket {
  CFSocketInvalidate(socket_);
}

- (void)close {
  [super close];
  close(connection_);
}

- (void)onEndEncountered {
  [self close];
  [self notifyState:RPCServerStatus_RPCSessionFinished];
}

@end

@interface RPCServerTracker : RPCServerBase <RPCServerEventListener>
@end

typedef enum {
  RPCServerTracker_Idle,
  RPCServerTracker_HandshakeToSend,
  RPCServerTracker_HandshakeToRecv,
  RPCServerTracker_ServerInfoToSend,
  RPCServerTracker_ServerInfoToRecv,
  RPCServerTracker_ReportResToSend,
  RPCServerTracker_ReportResToRecv,
  RPCServerTracker_UpdateKeyToSend,
  RPCServerTracker_UpdateKeyToRecv,
  RPCServerTracker_WaitConnection
} RPCServerTrackerState;

@implementation RPCServerTracker {
  RPCServerTrackerState state_;
  RPCServerStandalone* rpc_server_;
}

- (void)setState:(RPCServerTrackerState)new_state {
  if (state_ == RPCServerTracker_ReportResToRecv && new_state == RPCServerTracker_WaitConnection)
    [self notifyState:RPCServerStatus_Connected];
  if (state_ == RPCServerTracker_WaitConnection && new_state == RPCServerTracker_Idle)
    [self notifyState:RPCServerStatus_Disconnected];

  state_ = new_state;
}

- (bool)onWriteHandler {
  switch (state_) {
    case RPCServerTracker_HandshakeToSend: {
      int32_t code = tvm::runtime::kRPCTrackerMagic;
      [self toSend:[NSData dataWithBytes:&code length:sizeof(code)]];
      self.state = RPCServerTracker_HandshakeToRecv;
      return TRUE;
      break;
    }
    case RPCServerTracker_ServerInfoToSend: {
      std::ostringstream ss;
      ss << "[" << static_cast<int>(tvm::runtime::TrackerCode::kUpdateInfo)
         << ", {\"key\": \"server:" << self.key.UTF8String << "\", \"addr\": ["
         << self.custom_addr.UTF8String << ", \"" << self.port << "\"]}]";
      std::string data_s = ss.str();
      [self toSendPacked:[NSData dataWithBytes:data_s.data() length:data_s.length()]];
      self.state = RPCServerTracker_ServerInfoToRecv;
      return TRUE;
      break;
    }
    case RPCServerTracker_ReportResToSend: {
      std::mt19937 gen(std::random_device{}());
      std::uniform_real_distribution<float> dis(0.0, 1.0);

      std::string address_to_report = "null";
      if (self.custom_addr != nil && self.custom_addr.length != 0) {
        address_to_report = self.custom_addr.UTF8String;
      }

      std::string matchkey = std::string(self.key.UTF8String) + ":" + std::to_string(dis(gen));
      std::ostringstream ss;
      ss << "[" << static_cast<int>(tvm::runtime::TrackerCode::kPut) << ", \""
         << self.key.UTF8String << "\", [" << rpc_server_.rpc_port << ", \"" << matchkey << "\"], "
         << address_to_report << "]";

      std::string data_s = ss.str();
      [self toSendPacked:[NSData dataWithBytes:data_s.data() length:data_s.length()]];
      self.state = RPCServerTracker_ReportResToRecv;
      return TRUE;
      break;
    }
    default:
      // Nothing
      break;
  }
  return FALSE;
}

- (bool)onReadHandler {
  static const std::string resp_OK =
      std::to_string(static_cast<int>(tvm::runtime::TrackerCode::kSuccess));
  switch (state_) {
    case RPCServerTracker_HandshakeToRecv: {
      NSData* data = [self requestInputDataWithSize:sizeof(int)];
      if (data == nil) return FALSE;

      if (*(int*)data.bytes != tvm::runtime::kRPCTrackerMagic) {
        [self notifyError:@"Wrong response, is not RPC Tracker."];
        [self close];
        return FALSE;
        break;
      }
      self.state = RPCServerTracker_ServerInfoToSend;
      return TRUE;
      break;
    }
    case RPCServerTracker_ServerInfoToRecv: {
      NSData* data = [self requestInputDataPacked];
      if (data == nil) return FALSE;

      if (std::string((char*)data.bytes, data.length) != resp_OK) {
        [self notifyError:@"Failed to Update info on tracker. Response is not OK."];
        [self close];
        return FALSE;
        break;
      }
      self.state = RPCServerTracker_ReportResToSend;
      return TRUE;
      break;
    }
    case RPCServerTracker_ReportResToRecv: {
      NSData* data = [self requestInputDataPacked];
      if (data == nil) return FALSE;

      if (std::string((char*)data.bytes, data.length) != resp_OK) {
        [self notifyError:@"Failed to Put server into tracker. Response is not OK."];
        [self close];
        return FALSE;
        break;
      }
      self.state = RPCServerTracker_WaitConnection;
      return TRUE;
      break;
    }
    default:
      // Nothing
      break;
  }
  return FALSE;
}

- (void)onEndEncountered {
  [self close];
}

- (void)close {
  [rpc_server_ close];
  [rpc_server_ closeSocket];
  [super close];
  self.state = RPCServerTracker_Idle;
}

- (void)open {
  // Start internal Standalone RPC server at first
  rpc_server_ = [[RPCServerStandalone alloc] init];
  rpc_server_.key = self.key;
  rpc_server_.delegate = self;
  [rpc_server_ open];

  [super open];
  self.state = RPCServerTracker_HandshakeToSend;
}

- (void)onError:(NSString*)msg {
  // transfer error form internal rpc_server_ to real delegate
  [self notifyError:msg];
}

- (void)onStatusChanged:(RPCServerStatus)status {
  if (status == RPCServerStatus_RPCSessionFinished) {
    [self notifyState:status];
    self.state = RPCServerTracker_ReportResToSend;
    [self tryToWrite];
  }
}
@end

@implementation RPCServer

+ (instancetype)serverWithMode:(RPCServerMode)mode {
  if (mode == RPCServerMode_Standalone) return [[RPCServerStandalone alloc] init];
  if (mode == RPCServerMode_Proxy) return [[RPCServerProxy alloc] init];
  if (mode == RPCServerMode_Tracker) return [[RPCServerTracker alloc] init];
  return nil;
}

/// Unimplemented stubs
- (void)start {
}
- (void)stop {
}

@end
