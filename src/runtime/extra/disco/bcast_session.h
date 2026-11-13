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
#ifndef TVM_RUNTIME_DISCO_BCAST_SESSION_H_
#define TVM_RUNTIME_DISCO_BCAST_SESSION_H_

#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/disco/session.h>

#include <string>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief A Disco interactive session. It allows users to interact with the Disco command queue with
 * various ffi::Function calling convention.
 */
class BcastSessionObj : public SessionObj {
 public:
  virtual ~BcastSessionObj() = default;

  DRef GetGlobalFunc(const std::string& name) override;
  void CopyFromWorker0(const ffi::Tensor& host_array, const DRef& remote_array) override;
  void CopyToWorker0(const ffi::Tensor& host_array, const DRef& remote_array) override;
  void SyncWorker(int worker_id) override;
  void Shutdown() override;
  void InitCCL(ffi::String ccl, ffi::Shape device_ids) override;
  ffi::Any DebugGetFromRemote(int64_t reg_id, int worker_id) override = 0;
  void DebugSetRegister(int64_t reg_id, ffi::AnyView value, int worker_id) override = 0;

 protected:
  /*! \brief Deallocate a register id, kill it on all workers, and append it to `free_regs_`. */
  void DeallocReg(int reg_id) override;
  /*! \brief Call packed function on each worker using a packed sequence */
  DRef CallWithPacked(const ffi::PackedArgs& args) override;
  /*! \brief Allocate a register id, either from `free_regs_` or by incrementing `reg_count_` */
  virtual int AllocateReg();
  /*!
   * \brief Append an controler-side Tensor to a special queue used to communicate with
   worker-0.
   * \param host_array The array to be appended to worker-0
   */
  virtual void AppendHostTensor(const ffi::Tensor& host_array);
  /*!
   * \brief Broadcast a command to all workers via TVM's ffi::Function calling convention.
   * As part of the calling convention, The first argument in the packed sequence must be
   * the action, and the second argument must be the register id.
   * \param ffi::PackedArgs The input arguments in TVM's ffi::Function calling convention
   */
  virtual void BroadcastPacked(const ffi::PackedArgs& args) = 0;

  /*!
   * \brief Send a packed sequence to a worker. This function is usually called by the controler to
   * communicate with worker-0, because the worker-0 is assumed to be always collocated with the
   * controler. Sending to other workers may not be supported.
   * \param worker_id The worker id to send the packed sequence to.
   * \param args The packed sequence to send.
   */
  virtual void SendPacked(int worker_id, const ffi::PackedArgs& args) = 0;

  /*!
   * \brief Receive a packed sequence from a worker. This function is usually called by the
   * controler to communicate with worker-0, because the worker-0 is assumed to be always
   collocated
   * with the controler. Receiving from other workers may not be supported.
   * \return The packed sequence received.
   */
  virtual ffi::PackedArgs RecvReplyPacked(int worker_id) = 0;

  /*! \brief A side channel to communicate with worker-0 */
  WorkerZeroData worker_zero_data_;
  /*! \brief Number of registers used, including those in `free_regs_` */
  int reg_count_ = 1;
  /*! \brief The regsiter ids that have been deallocated */
  std::vector<int64_t> free_regs_;

  struct Internal;
  friend struct Internal;
  friend class SocketSessionObj;
  friend class RemoteSocketSession;
};

/*!
 * \brief Managed reference to BcastSessionObj.
 */
class BcastSession : public Session {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BcastSession, Session, BcastSessionObj);
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DISCO_BCAST_SESSION_H_
