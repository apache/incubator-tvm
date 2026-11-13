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
 * \file src/runtime/timer.cc
 * \brief Parent runtime timing and benchmarking policy.
 */

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>
#include <tvm/runtime/timing.h>

#include <chrono>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace tvm {
namespace runtime {

class DefaultTimerObj : public TimerObj {
 public:
  virtual void Start() {
    DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    start_ = std::chrono::high_resolution_clock::now();
  }
  virtual void Stop() {
    DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    duration_ = std::chrono::high_resolution_clock::now() - start_;
  }
  virtual int64_t SyncAndGetElapsedNanos() { return duration_.count(); }
  virtual ~DefaultTimerObj() {}

  explicit DefaultTimerObj(Device dev) : device_(dev) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.DefaultTimer", DefaultTimerObj, TimerObj);

 private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::duration<int64_t, std::nano> duration_;
  Device device_;
};

static Timer DefaultTimer(Device dev) {
  return Timer(ffi::UnsafeInit{}, ffi::make_object<DefaultTimerObj>(dev));
}

class CPUTimerObj : public TimerObj {
 public:
  virtual void Start() { start_ = std::chrono::high_resolution_clock::now(); }
  virtual void Stop() { duration_ = std::chrono::high_resolution_clock::now() - start_; }
  virtual int64_t SyncAndGetElapsedNanos() { return duration_.count(); }
  virtual ~CPUTimerObj() {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.CPUTimer", CPUTimerObj, TimerObj);

 private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::duration<int64_t, std::nano> duration_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.timer.cpu", [](Device dev) {
    return Timer(ffi::UnsafeInit{}, ffi::make_object<CPUTimerObj>());
  });
}

Timer StartTimer(Device dev) {
  // Function-local statics: thread-safe lazy init (C++11 magic statics),
  // visible only to this function.
  static std::set<DLDeviceType> seen_devices;
  static std::mutex seen_devices_lock;
  auto f = tvm::ffi::Function::GetGlobal(std::string("runtime.timer.") +
                                         DLDeviceType2Str(dev.device_type));
  if (!f.has_value()) {
    {
      std::lock_guard<std::mutex> lock(seen_devices_lock);
      if (seen_devices.find(dev.device_type) == seen_devices.end()) {
        LOG(WARNING)
            << "No timer implementation for " << DLDeviceType2Str(dev.device_type)
            << ", using default timer instead. It may be inaccurate or have extra overhead.";
        seen_devices.insert(dev.device_type);
      }
    }
    Timer t = DefaultTimer(dev);
    t->Start();
    return t;
  } else {
    Timer t = f->operator()(dev).cast<Timer>();
    t->Start();
    return t;
  }
}

ffi::Function WrapTimeEvaluator(ffi::Function pf, Device dev, int number, int repeat,
                                int min_repeat_ms, int limit_zero_time_iterations,
                                int cooldown_interval_ms, int repeats_to_cooldown,
                                int cache_flush_bytes, ffi::Function f_preproc) {
  TVM_FFI_ICHECK(pf != nullptr);

  auto ftimer = [pf, dev, number, repeat, min_repeat_ms, limit_zero_time_iterations,
                 cooldown_interval_ms, repeats_to_cooldown, cache_flush_bytes,
                 f_preproc](const ffi::AnyView* args, int num_args, ffi::Any* rv) mutable {
    ffi::Any temp;
    std::ostringstream os;
    // skip first time call, to activate lazy compilation components.
    pf.CallPacked(args, num_args, &temp);

    // allocate two large arrays to flush L2 cache
    ffi::Optional<ffi::Tensor> arr1, arr2;
    if (cache_flush_bytes > 0) {
      arr1 = tvm::runtime::Empty({cache_flush_bytes / 4}, {kDLInt, 32, 1}, dev);
      arr2 = tvm::runtime::Empty({cache_flush_bytes / 4}, {kDLInt, 32, 1}, dev);
    }

    DeviceAPI::Get(dev)->StreamSync(dev, nullptr);

    for (int i = 0; i < repeat; ++i) {
      if (f_preproc != nullptr) {
        f_preproc.CallPacked(args, num_args, &temp);
      }
      double duration_ms = 0.0;
      int absolute_zero_times = 0;
      do {
        if (duration_ms > 0.0) {
          const double golden_ratio = 1.618;
          number = static_cast<int>(
              std::max((min_repeat_ms / (duration_ms / number) + 1), number * golden_ratio));
        }
        if (cache_flush_bytes > 0) {
          tvm::runtime::CopyFromTo(arr2.value().GetDLTensorPtr(), arr1.value().GetDLTensorPtr());
        }
        DeviceAPI::Get(dev)->StreamSync(dev, nullptr);
        // start timing
        Timer t = StartTimer(dev);
        for (int j = 0; j < number; ++j) {
          pf.CallPacked(args, num_args, &temp);
        }
        t->Stop();
        int64_t t_nanos = t->SyncAndGetElapsedNanos();
        if (t_nanos == 0) absolute_zero_times++;
        duration_ms = t_nanos / 1e6;
      } while (duration_ms < min_repeat_ms && absolute_zero_times < limit_zero_time_iterations);

      double speed = duration_ms / 1e3 / number;
      os.write(reinterpret_cast<char*>(&speed), sizeof(speed));

      if (cooldown_interval_ms > 0 && (i % repeats_to_cooldown) == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cooldown_interval_ms));
      }
    }

    std::string blob = os.str();
    // return the time.
    *rv = ffi::Bytes(std::move(blob));
  };
  return ffi::Function::FromPacked(ftimer);
}

/*!
 * \brief Flush the cache.
 * \param addr The address of data we want to flush
 * \param len The length of data
 */
/*
 * When we are in the tuning of TVM, we will make TVM occupy
 * the cache fully and doesn't flush it during iteration.
 * This has problems then in e2e testing, since arrays that
 * we assume exist in cache (ie. weights) are evicted during e2e runs,
 * which leads to lower performance.
 */
inline void CPUCacheFlushImpl(const char* addr, unsigned int len) {
#if (defined(_M_X64) || defined(__x86_64__) || defined(__aarch64__))

#if defined(__aarch64__)
  size_t ctr_el0 = 0;
  asm volatile("mrs %0, ctr_el0" : "=r"(ctr_el0));
  const size_t cache_line = 4 << ((ctr_el0 >> 16) & 15);
#else
  const size_t cache_line = 64;
#endif

  if (addr == nullptr || len <= 0) {
    return;
  }

  for (uintptr_t uptr = (uintptr_t)addr & ~(cache_line - 1); uptr < (uintptr_t)addr + len;
       uptr += cache_line) {
#if defined(__aarch64__)
    asm volatile("dc civac, %0\n\t" : : "r"(reinterpret_cast<const void*>(uptr)) : "memory");
#else
    _mm_clflush(reinterpret_cast<const void*>(uptr));
#endif
  }

#if defined(__aarch64__)
  asm volatile("dmb ishst" : : : "memory");
#endif

#endif
}

inline void CPUCacheFlush(int begin_index, const ffi::PackedArgs& args) {
  for (int i = begin_index; i < args.size(); i++) {
    CPUCacheFlushImpl(static_cast<char*>((args[i].cast<DLTensor*>()->data)),
                      ffi::GetDataSize(*(args[i].cast<DLTensor*>())));
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.LocalTimeEvaluator",
           [](ffi::Optional<ffi::Module> opt_mod, std::string name, int device_type, int device_id,
              int number, int repeat, int min_repeat_ms, int limit_zero_time_iterations,
              int cooldown_interval_ms, int repeats_to_cooldown, int cache_flush_bytes,
              std::string f_preproc_name) {
             Device dev{static_cast<DLDeviceType>(device_type), device_id};
             auto pf = opt_mod.has_value() ? opt_mod.value()->GetFunction(name)
                                           : ffi::Function::GetGlobal(name);
             TVM_FFI_ICHECK(pf.has_value()) << "Cannot find " << name;
             ffi::Function f_preproc;
             if (!f_preproc_name.empty()) {
               f_preproc = ffi::Function::GetGlobalRequired(f_preproc_name);
             }
             return WrapTimeEvaluator(*pf, dev, number, repeat, min_repeat_ms,
                                      limit_zero_time_iterations, cooldown_interval_ms,
                                      repeats_to_cooldown, cache_flush_bytes, f_preproc);
           })
      .def("runtime.ModuleTimeEvaluator",
           [](ffi::Module mod, std::string name, ffi::AnyView device, int number, int repeat,
              int min_repeat_ms, int limit_zero_time_iterations, int cooldown_interval_ms,
              int repeats_to_cooldown, int cache_flush_bytes, std::string f_preproc_name) {
             if (std::string(mod->kind()) == "rpc") {
               return ffi::Function::GetGlobalRequired("rpc.ModuleTimeEvaluator")(
                          mod, name, device, number, repeat, min_repeat_ms,
                          limit_zero_time_iterations, cooldown_interval_ms, repeats_to_cooldown,
                          cache_flush_bytes, f_preproc_name)
                   .cast<ffi::Function>();
             }
             auto dev = device.cast<DLDevice>();
             return ffi::Function::GetGlobalRequired("runtime.LocalTimeEvaluator")(
                        mod, name, static_cast<int>(dev.device_type), dev.device_id, number, repeat,
                        min_repeat_ms, limit_zero_time_iterations, cooldown_interval_ms,
                        repeats_to_cooldown, cache_flush_bytes, f_preproc_name)
                 .cast<ffi::Function>();
           })
      .def_packed("runtime.CacheFlushCPUArguments",
                  [](ffi::PackedArgs args, ffi::Any* rv) { CPUCacheFlush(1, args); });
}

}  // namespace runtime
}  // namespace tvm
