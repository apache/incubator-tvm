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
 * \file session.h
 * \brief This file serves as the entry point of Disco and defines key data structures and
 * interfaces.
 *
 * Disco is a distributed runtime that consists of a controler and a cluster of workers. The
 * controler is responsible for managing the workers by broadcasting commands to all the workers
 * together, and the workers are responsible for executing the commands and. The controler and
 * workers communicate with each other through a bi-directional channel.
 *
 * Different from a generic system, Disco is designed to as "single-program-multiple-data" (SPMD)
 * runtime, which means that all the workers execute the same instruction at the same time, but the
 * data they are working on may be different. For example, in data parallelism, each worker may
 * work on a different batches of the data, but they all execute the same set of instructions.
 * Therefore, imagine there is a virtual machine that executes the program, the structures of
 * workers' register files could be considered as "identical" (single program) although the values
 * may differ (multiple data).
 *
 *
 * **DRef.** Following the design above, consider the program in SPMD in a virtual ISA, then each
 * worker is a virtual machine instance to execute the ISA maintaining its own register file.
 * The controler denotes each of their register files with a unique integer "register id",
 * and the workers use this id to refer to the register file that resides on itself.
 * DRef is a control-side object backed by such a register id. The data it contains is not assumed
 * to be directly accessible by the controler, with an exception for worker-0, which is a special
 * worker that is always co-located with the controler.
 *
 * **Worker-0.** Worker-0 is a special worker that is always co-located with the controler.
 * It is assumed that the controler can synchronize with and access the registers of worker-0.
 * The Disco session provides multiple APIs to interact specifically with the worker-0.
 * To shared data with other workers, a common paradigm in Disco is to copy data from the
 * controler-side Tensor to the worker-0, and then copy it to other workers using primitives on
 * the data plane, for example, `broadcast` and `send`.
 *
 * **Control plane.** The controler broadcasts commands to all the workers as control signals.
 * For example, the control may ask all workers to load a library or call a function respectively.
 * Common control signals include: shutdown, retrievel a global ffi::Function, call packed function,
 * etc. The controler is assumed to keep a message channel to each worker to implement the broadcast
 * behavior, and the message channel may vary depends on usecases.
 *
 * **Data plane.** The data channel is usually used to exchange data between workers, especially for
 * tensor data which is usually large. For example, performing an allreduce operator for sharded
 * matrix multiplication, or broadcasting for an input tensor. For efficiency, the data channel is
 * usually backed by NCCL on NVIDIA GPUs, RCCL on AMD GPUs, or MPI on CPUs.
 *
 * **Session.** A Disco session is a primary interface to interact with the Disco runtime, serving
 * as a global context that manages the control and workers. It could be implemented as a
 * multi-threaded with a pool of workers for single-node multi-gpu scenarios, or TCP sockets for
 * workloads that span over a cluster of nodes.
 *
 * **Channel.** Disco channel is a bi-directional communication channel between the controler and
 * workers for exchanging control signals. It is no different from a generic RPC channel, but
 * adopts TVM's ffi::Function calling convention to support polymorphic and variadic arguments.
 */
#ifndef TVM_RUNTIME_DISCO_SESSION_H_
#define TVM_RUNTIME_DISCO_SESSION_H_

#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/extra_base.h>
#include <tvm/runtime/tensor.h>

#include <mutex>
#include <queue>
#include <string>
#include <utility>

namespace tvm {
namespace runtime {

/*!
 * \brief Static FFI type index for `runtime::disco::DRef`.
 *
 * Allocated within the [kTVMFFIDynObjectBegin - 16, kTVMFFIDynObjectBegin)
 * custom-static slot range. The sibling constant `kRuntimeRPCObjectRef`
 * lives in `src/runtime/rpc/rpc_session.h` and uses `... - 13`; values must
 * remain disjoint across this small reserved block.
 */
constexpr int32_t kRuntimeDiscoDRef = TVMFFITypeIndex::kTVMFFIDynObjectBegin - 14;

static_assert(kRuntimeDiscoDRef >= TVMFFITypeIndex::kTVMFFIStaticObjectEnd &&
                  kRuntimeDiscoDRef < TVMFFITypeIndex::kTVMFFIDynObjectBegin,
              "kRuntimeDiscoDRef must live in the static custom-index slot range");

/*!
 * \brief All possible kinds of Disco commands.
 */
enum class DiscoAction : int32_t {
  kShutDown = 0,
  kKillReg = 1,
  kGetGlobalFunc = 2,
  kCallPacked = 3,
  kSyncWorker = 4,
  kCopyFromWorker0 = 5,
  kCopyToWorker0 = 6,
  kDebugGetFromRemote = 7,
  kDebugSetRegister = 8,
};

/*! \brief Converts the enum class `DiscoAction` to string */
inline std::string DiscoAction2String(DiscoAction action) {
  switch (action) {
    case DiscoAction::kShutDown:
      return "kShutDown";
    case DiscoAction::kKillReg:
      return "kKillReg";
    case DiscoAction::kGetGlobalFunc:
      return "kGetGlobalFunc";
    case DiscoAction::kCallPacked:
      return "kCallPacked";
    case DiscoAction::kSyncWorker:
      return "kSyncWorker";
    case DiscoAction::kCopyFromWorker0:
      return "kCopyFromWorker0";
    case DiscoAction::kCopyToWorker0:
      return "kCopyToWorker0";
    case DiscoAction::kDebugGetFromRemote:
      return "kDebugGetFromRemote";
    case DiscoAction::kDebugSetRegister:
      return "kDebugSetRegister";
  }
  TVM_FFI_THROW(ValueError) << "Unknown DiscoAction: " << static_cast<int>(action);
}

class SessionObj;

/*!
 * \brief An object that exists on all workers.
 *
 * The controler assigns a unique "register id" to each object, and the worker uses this id to
 * refer to the object residing on itself.
 */
class DRefObj : public ffi::Object {
 public:
  /*!\ brief Send dellocation command for `reg_id` */
  inline ~DRefObj();
  /*!
   * \brief Get the value of a DRef from a remote worker.
   * \param worker_id The id of the worker to be fetched from.
   * \return The value of the register.
   */
  inline ffi::Any DebugGetFromRemote(int worker_id);
  /*!
   * \brief Copy from the Tensor provided to a remote worker.
   * \param worker_id The id of the worker to be copied to.
   * \param source The Tensor to be copied.
   */
  inline void DebugCopyFrom(int worker_id, ffi::AnyView source);

  static constexpr const uint32_t _type_index = kRuntimeDiscoDRef;
  static const constexpr bool _type_final = true;
  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_STATIC("runtime.disco.DRef", DRefObj, ffi::Object);

  /*! \brief The id of the register */
  int64_t reg_id;
  /*! \brief Back-pointer to the host controler session */
  ffi::ObjectRef session{nullptr};

 private:
  inline SessionObj* GetSession();
};

/*!
 * \brief Managed reference to DRefObj.
 * \sa DRefObj
 * \note No public constructor is provided as it is not supposed to be directly created by users.
 */
class DRef : public ffi::ObjectRef {
 public:
  explicit DRef(ffi::ObjectPtr<DRefObj> data) : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DRef, ffi::ObjectRef, DRefObj);
};

/*!
 * \brief A Disco interactive session. It allows users to interact with the Disco command queue with
 * various ffi::Function calling convention.
 */
class SessionObj : public ffi::Object {
 public:
  virtual ~SessionObj() = default;
  /*!
   * \brief Call a ffi::Function on workers providing variadic arguments.
   * \tparam Args In the variadic arguments, the supported types include:
   * - integers and floating point numbers;
   * - DataType;
   * - Device;
   * - std::string;
   * - DRef.
   * Examples of unsupported types:
   * - Tensor, DLTensor;
   * - TVM Objects, including ffi::Function, Module and String;
   * \param func The function to be called.
   * \param args The variadic arguments.
   * \return The return value of function call
   */
  template <typename... Args>
  TVM_FFI_INLINE DRef CallPacked(const DRef& func, Args&&... args);
  /*!
   * \brief Call packed function on each worker using a packed sequence. The calling convention:
   * The first element must be DiscoAction::kCallPacked,
   * The second element must be 0, which will later be updated by the session to return reg_id
   * The thirtd element is the function to be called.
   */
  TVM_RUNTIME_EXTRA_DLL virtual DRef CallWithPacked(const ffi::PackedArgs& args) = 0;
  /*! \brief Get the number of workers in the session. */
  TVM_RUNTIME_EXTRA_DLL virtual int64_t GetNumWorkers() = 0;
  /*! \brief Get a global functions on workers. */
  TVM_RUNTIME_EXTRA_DLL virtual DRef GetGlobalFunc(const std::string& name) = 0;
  /*!
   * \brief Copy an Tensor from worker-0 to the controler-side Tensor
   * \param host_array The array to be copied to worker-0
   * \param remote_array The Tensor on worker-0
   */
  TVM_RUNTIME_EXTRA_DLL virtual void CopyFromWorker0(const ffi::Tensor& host_array,
                                                     const DRef& remote_array) = 0;
  /*!
   * \brief Copy the controler-side Tensor to worker-0
   * \param host_array The array to be copied to worker-0
   * \param remote_array The Tensor on worker-0
   */
  TVM_RUNTIME_EXTRA_DLL virtual void CopyToWorker0(const ffi::Tensor& host_array,
                                                   const DRef& remote_array) = 0;
  /*!
   * \brief Synchrnoize the controler with a worker, and it will wait until worker finishes
   * executing this instruction.
   * \param worker_id The id of the worker to be synced with.
   * \note This function is usually used for worker-0, because it is the only worker that is
   * assumed to collocate with the controler. Syncing with other workers may not be supported.
   */
  TVM_RUNTIME_EXTRA_DLL virtual void SyncWorker(int worker_id) = 0;
  /*! \brief Signal all the workers to shutdown */
  TVM_RUNTIME_EXTRA_DLL virtual void Shutdown() = 0;
  /*!
   * \brief Initialize the data plane between workers.
   * \param ccl The name of the communication backend, e.g., nccl, rccl, mpi.
   * \param device_ids The device ids of the workers.
   */
  TVM_RUNTIME_EXTRA_DLL virtual void InitCCL(ffi::String ccl, ffi::Shape device_ids) = 0;
  /*!
   * \brief Get the value of a register from a remote worker.
   * \param reg_id The id of the register to be fetched.
   * \param worker_id The id of the worker to be fetched from.
   * \return The value of the register.
   */
  TVM_RUNTIME_EXTRA_DLL virtual ffi::Any DebugGetFromRemote(int64_t reg_id, int worker_id) = 0;
  /*!
   * \brief Set the value of a register on a remote worker.
   * \param reg_id The id of the register to be set.
   * \param value The value to be set.
   * \param worker_id The id of the worker to be set.
   */
  TVM_RUNTIME_EXTRA_DLL virtual void DebugSetRegister(int64_t reg_id, ffi::AnyView value,
                                                      int worker_id) = 0;

  struct FFI;
  friend struct SessionObj::FFI;
  friend class DRefObj;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("runtime.disco.Session", SessionObj, ffi::Object);

 protected:
  /*! \brief Deallocate a register id, kill it on all workers, and append it to `free_regs_`. */
  virtual void DeallocReg(int reg_id) = 0;
};

/*!
 * \brief Managed reference to SessionObj
 * \sa SessionObj
 */
class Session : public ffi::ObjectRef {
 public:
  /*!
   * \brief Create a session backed by a thread pool of workers
   * \param num_workers The number of workers.
   * \param num_groups The number of worker groups.
   */
  TVM_RUNTIME_EXTRA_DLL static Session ThreadedSession(int num_workers, int num_groups);
  /*!
   * \brief Create a session backed by pipe-based multiprocessing
   * \param num_workers The number of workers.
   * \param num_groups The number of worker groups.
   * \param process_pool_creator The name of a global function that takes `num_workers` as an input,
   * and returns a ffi::Function, which takes an integer `worker_id` as the input and returns None.
   * When `worker-id` is 0, it shuts down the process pool; Otherwise, it retursn a tuple
   * (read_fd, writefd) used to communicate with the corresponding worker.
   * \param entrypoint The entrypoint of DiscoWorker main worker function.
   * \note Worker-0 is always co-located with the controler as a separate thread, and therefore
   * worker-0 does not exist in the process pool.
   */
  TVM_RUNTIME_EXTRA_DLL static Session ProcessSession(int num_workers, int num_groups,
                                                      ffi::String process_pool_creator,
                                                      ffi::String entrypoint);

  explicit Session(ffi::UnsafeInit tag, ffi::ObjectPtr<SessionObj> n) : ffi::ObjectRef(tag) {
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Session, ffi::ObjectRef, SessionObj);
};

/*!
 * \brief A bi-directional channel for controler-worker communication.
 * This channel is primarily used to transfer control messages but not data.
 */
class DiscoChannel {
 public:
  virtual ~DiscoChannel() = default;
  /*! \brief Send a packed sequence to the receiver */
  virtual void Send(const ffi::PackedArgs& args) = 0;
  /*! \brief Receive a packed sequence from worker */
  virtual ffi::PackedArgs Recv() = 0;
  /*! \brief Reply a packed sequence to the sender */
  virtual void Reply(const ffi::PackedArgs& args) = 0;
  /*! \brief Receive a reply from the worker */
  virtual ffi::PackedArgs RecvReply() = 0;
};

/*!
 * \brief A special communication channel between controler and worker-0,
 * assuming they are always collocated in the same process.
 */
class WorkerZeroData {
 public:
  /*!
   * \brief The host-side arrays to passed to worker-0 for special uses, for example,
   * copy-to-worker0 and copy-from-worker0
   */
  std::queue<ffi::Tensor> host_arrays;
  /*! \brief The mutex that guards `host_arrays` */
  std::mutex queue_mutex_;
};

// Implementation details

inline SessionObj* DRefObj::GetSession() {
  return const_cast<SessionObj*>(static_cast<const SessionObj*>(session.get()));
}

DRefObj::~DRefObj() {
  if (this->session.defined()) {
    GetSession()->DeallocReg(reg_id);
  }
}

ffi::Any DRefObj::DebugGetFromRemote(int worker_id) {
  return GetSession()->DebugGetFromRemote(this->reg_id, worker_id);
}

void DRefObj::DebugCopyFrom(int worker_id, ffi::AnyView value) {
  return GetSession()->DebugSetRegister(this->reg_id, value, worker_id);
}

template <typename... Args>
DRef SessionObj::CallPacked(const DRef& func, Args&&... args) {
  constexpr int offset = 3;
  constexpr int kNumArgs = offset + sizeof...(Args);
  ffi::AnyView packed_args[kNumArgs];
  ffi::PackedArgs::Fill(packed_args,
                        /*.0=*/static_cast<int>(DiscoAction::kCallPacked),  // action
                        /*.1=*/0,     // reg_id, which will be updated by this->CallWithPacked
                        /*.2=*/func,  // the function to be called
                        std::forward<Args>(args)...);
  return this->CallWithPacked(ffi::PackedArgs(packed_args, kNumArgs));
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DISCO_SESSION_H_
