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
 * \file socket.h
 * \brief this file aims to provide a wrapper of sockets
 * \author Tianqi Chen
 */
#ifndef TVM_SUPPORT_SOCKET_H_
#define TVM_SUPPORT_SOCKET_H_

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <winsock2.h>
#include <ws2tcpip.h>

#ifdef _MSC_VER
#pragma comment(lib, "Ws2_32.lib")
#endif
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <tvm/runtime/logging.h>
#include <tvm/support/io.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "errno_handling.h"
#include "ssize.h"

#if defined(_WIN32)
static inline int poll(struct pollfd* pfd, int nfds, int timeout) {
  return WSAPoll(pfd, nfds, timeout);
}
#else
#include <sys/poll.h>
#endif  // defined(_WIN32)

namespace tvm {
namespace support {

/*!
 * \brief Get current host name.
 * \return The hostname.
 */
inline std::string GetHostName() {
  std::string buf;
  buf.resize(256);
  TVM_FFI_ICHECK_NE(gethostname(&buf[0], 256), -1);
  return std::string(buf.c_str());
}

/*!
 * \brief ValidateIP validates an ip address.
 * \param ip The ip address in string format localhost or x.x.x.x format
 * \return result of operation.
 */
inline bool ValidateIP(std::string ip) {
  if (ip == "localhost") {
    return true;
  }
  struct sockaddr_in sa_ipv4;
  struct sockaddr_in6 sa_ipv6;
  bool is_ipv4 = inet_pton(AF_INET, ip.c_str(), &(sa_ipv4.sin_addr));
  bool is_ipv6 = inet_pton(AF_INET6, ip.c_str(), &(sa_ipv6.sin6_addr));
  return is_ipv4 || is_ipv6;
}

/*!
 * \brief Common data structure for network address.
 */
struct SockAddr {
  sockaddr_storage addr;
  SockAddr() {}
  /*!
   * \brief construct address by url and port
   * \param url The url of the address
   * \param port The port of the address.
   */
  SockAddr(const char* url, int port) { this->Set(url, port); }

  /*!
   * \brief SockAddr Get the socket address from tracker.
   * \param tracker The url containing the ip and port number. Format is ('192.169.1.100', 9090)
   * \return SockAddr parsed from url.
   */
  explicit SockAddr(const std::string& url) {
    size_t sep = url.find(",");
    std::string host = url.substr(2, sep - 3);
    std::string port = url.substr(sep + 1, url.length() - 1);
    TVM_FFI_ICHECK(ValidateIP(host)) << "Url address is not valid " << url;
    if (host == "localhost") {
      host = "127.0.0.1";
    }
    this->Set(host.c_str(), std::stoi(port));
  }

  /*!
   * \brief set the address
   * \param host the url of the address
   * \param port the port of address
   */
  void Set(const char* host, int port) {
    addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = PF_UNSPEC;
    hints.ai_flags = AI_PASSIVE;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    int sig = getaddrinfo(host, nullptr, &hints, &res);
    TVM_FFI_ICHECK(sig == 0 && res != nullptr) << "cannot obtain address of " << host;
    switch (res->ai_family) {
      case AF_INET: {
        sockaddr_in* addr4 = reinterpret_cast<sockaddr_in*>(&addr);
        memcpy(addr4, res->ai_addr, res->ai_addrlen);
        addr4->sin_port = htons(port);
        addr4->sin_family = AF_INET;
      } break;
      case AF_INET6: {
        sockaddr_in6* addr6 = reinterpret_cast<sockaddr_in6*>(&addr);
        memcpy(addr6, res->ai_addr, res->ai_addrlen);
        addr6->sin6_port = htons(port);
        addr6->sin6_family = AF_INET6;
      } break;
      default:
        TVM_FFI_ICHECK(false) << "cannot decode address";
    }
    freeaddrinfo(res);
  }
  /*! \brief return port of the address */
  int port() const {
    return ntohs((addr.ss_family == AF_INET6)
                     ? reinterpret_cast<const sockaddr_in6*>(&addr)->sin6_port
                     : reinterpret_cast<const sockaddr_in*>(&addr)->sin_port);
  }
  /*! \brief return the ip address family */
  int ss_family() const { return addr.ss_family; }
  /*! \return a string representation of the address */
  std::string AsString() const {
    std::string buf;
    buf.resize(256);

    const void* sinx_addr = nullptr;
    if (addr.ss_family == AF_INET6) {
      const in6_addr& addr6 = reinterpret_cast<const sockaddr_in6*>(&addr)->sin6_addr;
      sinx_addr = reinterpret_cast<const void*>(&addr6);
    } else if (addr.ss_family == AF_INET) {
      const in_addr& addr4 = reinterpret_cast<const sockaddr_in*>(&addr)->sin_addr;
      sinx_addr = reinterpret_cast<const void*>(&addr4);
    } else {
      TVM_FFI_ICHECK(false) << "illegal address";
    }

#ifdef _WIN32
    const char* s = inet_ntop(addr.ss_family, (PVOID)sinx_addr,  // NOLINT(*)
                              &buf[0], buf.length());
#else
    const char* s =
        inet_ntop(addr.ss_family, sinx_addr, &buf[0], static_cast<socklen_t>(buf.length()));
#endif
    TVM_FFI_ICHECK(s != nullptr) << "cannot decode address";
    std::ostringstream os;
    os << s << ":" << port();
    return os.str();
  }
};
/*!
 * \brief base class containing common operations of TCP and UDP sockets
 */
class Socket {
 public:
#if defined(_WIN32)
  using sock_size_t = int;
  using SockType = SOCKET;
#else
  using SockType = int;
  using sock_size_t = size_t;
  static constexpr int INVALID_SOCKET = -1;
#endif
  /*! \brief the file descriptor of socket */
  SockType sockfd;
  /*!
   * \brief set this socket to use non-blocking mode
   * \param non_block whether set it to be non-block, if it is false
   *        it will set it back to block mode
   */
  void SetNonBlock(bool non_block) {
#ifdef _WIN32
    u_long mode = non_block ? 1 : 0;
    if (ioctlsocket(sockfd, FIONBIO, &mode) != NO_ERROR) {
      Socket::Error("SetNonBlock");
    }
#else
    int flag = fcntl(sockfd, F_GETFL, 0);
    if (flag == -1) {
      Socket::Error("SetNonBlock-1");
    }
    if (non_block) {
      flag |= O_NONBLOCK;
    } else {
      flag &= ~O_NONBLOCK;
    }
    if (fcntl(sockfd, F_SETFL, flag) == -1) {
      Socket::Error("SetNonBlock-2");
    }
#endif
  }
  /*!
   * \brief bind the socket to an address
   * \param addr The address to be binded
   */
  void Bind(const SockAddr& addr) {
    if (bind(sockfd, reinterpret_cast<const sockaddr*>(&addr.addr),
             (addr.addr.ss_family == AF_INET6 ? sizeof(sockaddr_in6) : sizeof(sockaddr_in))) ==
        -1) {
      Socket::Error("Bind");
    }
  }
  /*!
   * \brief try bind the socket to host, from start_port to end_port
   * \param host host address to bind the socket
   * \param start_port starting port number to try
   * \param end_port ending port number to try
   * \return the port successfully bind to, return -1 if failed to bind any port
   */
  inline int TryBindHost(std::string host, int start_port, int end_port) {
    for (int port = start_port; port < end_port; ++port) {
      SockAddr addr(host.c_str(), port);
      if (bind(sockfd, reinterpret_cast<sockaddr*>(&addr.addr),
               (addr.addr.ss_family == AF_INET6 ? sizeof(sockaddr_in6) : sizeof(sockaddr_in))) ==
          0) {
        return port;
      } else {
#if defined(_WIN32)
        if (WSAGetLastError() != WSAEADDRINUSE) {
          Socket::Error("TryBindHost");
        }
#else
        if (errno != EADDRINUSE) {
          Socket::Error("TryBindHost");
        }
#endif
        LOG(WARNING) << "Bind failed to " << host << ":" << port;
      }
    }
    return -1;
  }
  /*! \brief get last error code if any */
  int GetSockError() const {
    int error = 0;
    socklen_t len = sizeof(error);
    if (getsockopt(sockfd, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&error), &len) != 0) {
      Error("GetSockError");
    }
    return error;
  }
  /*! \brief check if anything bad happens */
  bool BadSocket() const {
    if (IsClosed()) return true;
    int err = GetSockError();
    if (err == EBADF || err == EINTR) return true;
    return false;
  }
  /*! \brief check if socket is already closed */
  bool IsClosed() const { return sockfd == INVALID_SOCKET; }
  /*! \brief close the socket */
  void Close() {
    if (sockfd != INVALID_SOCKET) {
#ifdef _WIN32
      closesocket(sockfd);
#else
      close(sockfd);
#endif
      sockfd = INVALID_SOCKET;
    } else {
      Error("Socket::Close double close the socket or close without create");
    }
  }
  /*!
   * \return last error of socket operation
   */
  static int GetLastErrorCode() {
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
  }
  /*! \return whether last error was would block */
  static bool LastErrorWouldBlock() {
    int errsv = GetLastErrorCode();
#ifdef _WIN32
    return errsv == WSAEWOULDBLOCK;
#else
    return errsv == EAGAIN || errsv == EWOULDBLOCK;
#endif
  }
  /*!
   * \brief start up the socket module
   *   call this before using the sockets
   */
  static void Startup() {
#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) == -1) {
      Socket::Error("Startup");
    }
    if (LOBYTE(wsa_data.wVersion) != 2 || HIBYTE(wsa_data.wVersion) != 2) {
      WSACleanup();
      TVM_FFI_THROW(InternalError) << "Could not find a usable version of Winsock.dll";
    }
#endif
  }
  /*!
   * \brief shutdown the socket module after use, all sockets need to be closed
   */
  static void Finalize() {
#ifdef _WIN32
    WSACleanup();
#endif
  }
  /*!
   * \brief Report an socket error.
   * \param msg The error message.
   */
  static void Error(const char* msg) {
    int errsv = GetLastErrorCode();
#ifdef _WIN32
    TVM_FFI_THROW(InternalError) << "Socket " << msg << " Error:WSAError-code=" << errsv;
#else
    TVM_FFI_THROW(InternalError) << "Socket " << msg << " Error:" << strerror(errsv);
#endif
  }

 protected:
  explicit Socket(SockType sockfd) : sockfd(sockfd) {}
};

/*!
 * \brief a wrapper of TCP socket that hopefully be cross platform
 */
class TCPSocket : public Socket, public Stream {
 public:
  TCPSocket() : Socket(INVALID_SOCKET) {}
  /*!
   * \brief construct a TCP socket from existing descriptor
   * \param sockfd The descriptor
   */
  explicit TCPSocket(SockType sockfd) : Socket(sockfd) {}
  /*!
   * \brief enable/disable TCP keepalive
   * \param keepalive whether to set the keep alive option on
   */
  void SetKeepAlive(bool keepalive) {
    int opt = static_cast<int>(keepalive);
    if (setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, reinterpret_cast<char*>(&opt), sizeof(opt)) <
        0) {
      Socket::Error("SetKeepAlive");
    }
  }
  /*!
   * \brief create the socket, call this before using socket
   * \param af domain
   */
  void Create(int af = PF_INET) {
    sockfd = socket(af, SOCK_STREAM, 0);
    if (sockfd == INVALID_SOCKET) {
      Socket::Error("Create");
    }
  }
  /*!
   * \brief perform listen of the socket
   * \param backlog backlog parameter
   */
  void Listen(int backlog = 16) { listen(sockfd, backlog); }
  /*!
   * \brief get a new connection
   * \return The accepted socket connection.
   */
  TCPSocket Accept() {
    SockType newfd =
        RetryCallOnEINTR([&]() { return accept(sockfd, nullptr, nullptr); }, GetLastErrorCode);
    if (newfd == INVALID_SOCKET) {
      Socket::Error("Accept");
    }
    return TCPSocket(newfd);
  }
  /*!
   * \brief get a new connection
   * \param addr client address from which connection accepted
   * \return The accepted socket connection.
   */
  TCPSocket Accept(SockAddr* addr) {
    socklen_t addrlen = sizeof(addr->addr);
    SockType newfd = RetryCallOnEINTR(
        [&]() { return accept(sockfd, reinterpret_cast<sockaddr*>(&addr->addr), &addrlen); },
        GetLastErrorCode);
    if (newfd == INVALID_SOCKET) {
      Socket::Error("Accept");
    }
    return TCPSocket(newfd);
  }
  /*!
   * \brief decide whether the socket is at OOB mark
   * \return 1 if at mark, 0 if not, -1 if an error occurred
   */
  int AtMark() const {
#ifdef _WIN32
    unsigned long atmark;  // NOLINT(*)
    if (ioctlsocket(sockfd, SIOCATMARK, &atmark) != NO_ERROR) return -1;
#else
    int atmark;
    if (ioctl(sockfd, SIOCATMARK, &atmark) == -1) return -1;
#endif
    return static_cast<int>(atmark);
  }
  /*!
   * \brief connect to an address
   * \param addr the address to connect to
   * \return whether connect is successful
   */
  bool Connect(const SockAddr& addr) {
    return connect(
               sockfd, reinterpret_cast<const sockaddr*>(&addr.addr),
               (addr.addr.ss_family == AF_INET6 ? sizeof(sockaddr_in6) : sizeof(sockaddr_in))) == 0;
  }
  /*!
   * \brief send data using the socket
   * \param buf_ the pointer to the buffer
   * \param len the size of the buffer
   * \param flag extra flags
   * \return size of data actually sent
   *         return -1 if error occurs
   */
  ssize_t Send(const void* buf_, size_t len, int flag = 0) {
    const char* buf = reinterpret_cast<const char*>(buf_);
    return RetryCallOnEINTR(
        [&]() { return send(sockfd, buf, static_cast<sock_size_t>(len), flag); }, GetLastErrorCode);
  }
  /*!
   * \brief receive data using the socket
   * \param buf_ the pointer to the buffer
   * \param len the size of the buffer
   * \param flags extra flags
   * \return size of data actually received
   *         return -1 if error occurs
   */
  ssize_t Recv(void* buf_, size_t len, int flags = 0) {
    char* buf = reinterpret_cast<char*>(buf_);
    return RetryCallOnEINTR(
        [&]() { return recv(sockfd, buf, static_cast<sock_size_t>(len), flags); },
        GetLastErrorCode);
  }
  /*!
   * \brief perform block write that will attempt to send all data out
   *    can still return smaller than request when error occurs
   * \param buf_ the pointer to the buffer
   * \param len the size of the buffer
   * \return size of data actually sent
   */
  size_t SendAll(const void* buf_, size_t len) {
    const char* buf = reinterpret_cast<const char*>(buf_);
    size_t ndone = 0;
    while (ndone < len) {
      ssize_t ret = RetryCallOnEINTR(
          [&]() { return send(sockfd, buf, static_cast<ssize_t>(len - ndone), 0); },
          GetLastErrorCode);
      if (ret == -1) {
        if (LastErrorWouldBlock()) return ndone;
        Socket::Error("SendAll");
      }
      buf += ret;
      ndone += ret;
    }
    return ndone;
  }
  /*!
   * \brief perform block read that will attempt to read all data
   *    can still return smaller than request when error occurs
   * \param buf_ the buffer pointer
   * \param len length of data to recv
   * \return size of data actually sent
   */
  size_t RecvAll(void* buf_, size_t len) {
    char* buf = reinterpret_cast<char*>(buf_);
    size_t ndone = 0;
    while (ndone < len) {
      ssize_t ret = RetryCallOnEINTR(
          [&]() { return recv(sockfd, buf, static_cast<sock_size_t>(len - ndone), MSG_WAITALL); },
          GetLastErrorCode);
      if (ret == -1) {
        if (LastErrorWouldBlock()) {
          TVM_FFI_THROW(InternalError) << "would block";
        }
        Socket::Error("RecvAll");
      }
      if (ret == 0) return ndone;
      buf += ret;
      ndone += ret;
    }
    return ndone;
  }
  /*!
   * \brief Send the data to remote.
   * \param data The data to be sent.
   */
  void SendBytes(std::string data) {
    int datalen = data.length();
    TVM_FFI_ICHECK_EQ(SendAll(&datalen, sizeof(datalen)), sizeof(datalen));
    TVM_FFI_ICHECK_EQ(SendAll(data.c_str(), datalen), datalen);
  }
  /*!
   * \brief Receive the data to remote.
   * \return The data received.
   */
  std::string RecvBytes() {
    int datalen = 0;
    TVM_FFI_ICHECK_EQ(RecvAll(&datalen, sizeof(datalen)), sizeof(datalen));
    std::string data;
    data.resize(datalen);
    TVM_FFI_ICHECK_EQ(RecvAll(&data[0], datalen), datalen);
    return data;
  }

  size_t Read(void* data, size_t size) final { return RecvAll(data, size); }

  size_t Write(const void* data, size_t size) final { return SendAll(data, size); }
};

/*! \brief helper data structure to perform poll */
struct PollHelper {
 public:
  /*!
   * \brief add file descriptor to watch for read
   * \param fd file descriptor to be watched
   */
  inline void WatchRead(TCPSocket::SockType fd) {
    auto& pfd = fds[fd];
    pfd.fd = fd;
    pfd.events |= POLLIN;
  }
  /*!
   * \brief add file descriptor to watch for write
   * \param fd file descriptor to be watched
   */
  inline void WatchWrite(TCPSocket::SockType fd) {
    auto& pfd = fds[fd];
    pfd.fd = fd;
    pfd.events |= POLLOUT;
  }
  /*!
   * \brief add file descriptor to watch for exception
   * \param fd file descriptor to be watched
   */
  inline void WatchException(TCPSocket::SockType fd) {
    auto& pfd = fds[fd];
    pfd.fd = fd;
    pfd.events |= POLLPRI;
  }
  /*!
   * \brief Check if the descriptor is ready for read
   * \param fd file descriptor to check status
   */
  inline bool CheckRead(TCPSocket::SockType fd) const {
    const auto& pfd = fds.find(fd);
    return pfd != fds.end() && ((pfd->second.events & POLLIN) != 0);
  }
  /*!
   * \brief Check if the descriptor is ready for write
   * \param fd file descriptor to check status
   */
  inline bool CheckWrite(TCPSocket::SockType fd) const {
    const auto& pfd = fds.find(fd);
    return pfd != fds.end() && ((pfd->second.events & POLLOUT) != 0);
  }
  /*!
   * \brief Check if the descriptor has any exception
   * \param fd file descriptor to check status
   */
  inline bool CheckExcept(TCPSocket::SockType fd) const {
    const auto& pfd = fds.find(fd);
    return pfd != fds.end() && ((pfd->second.events & POLLPRI) != 0);
  }
  /*!
   * \brief wait for exception event on a single descriptor
   * \param fd the file descriptor to wait the event for
   * \param timeout the timeout counter, can be negative, which means wait until the event happen
   * \return 1 if success, 0 if timeout, and -1 if error occurs
   */
  inline static int WaitExcept(TCPSocket::SockType fd, long timeout = -1) {  // NOLINT(*)
    pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLPRI;
    return poll(&pfd, 1, timeout);
  }

  /*!
   * \brief perform poll on the set defined, read, write, exception
   * \param timeout specify timeout in milliseconds(ms) if negative, means poll will block
   * \return
   */
  inline void Poll(long timeout = -1) {  // NOLINT(*)
    std::vector<pollfd> fdset;
    fdset.reserve(fds.size());
    for (auto kv : fds) {
      fdset.push_back(kv.second);
    }
    int ret = poll(fdset.data(), fdset.size(), timeout);
    if (ret == -1) {
      Socket::Error("Poll");
    } else {
      for (auto& pfd : fdset) {
        auto revents = pfd.revents & pfd.events;
        if (!revents) {
          fds.erase(pfd.fd);
        } else {
          fds[pfd.fd].events = revents;
        }
      }
    }
  }

  std::unordered_map<TCPSocket::SockType, pollfd> fds;
};

}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_SOCKET_H_
