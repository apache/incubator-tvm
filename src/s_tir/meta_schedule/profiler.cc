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
#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <chrono>

#include "./utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/**************** Profiler ****************/

ffi::Map<ffi::String, FloatImm> ProfilerObj::Get() const {
  ffi::Map<ffi::String, FloatImm> ret;
  for (const auto& kv : stats_sec) {
    ret.Set(kv.first, FloatImm(PrimType::Float(64), kv.second));
  }
  return ret;
}

ffi::String ProfilerObj::Table() const {
  TVM_FFI_CHECK(!stats_sec.empty(), ValueError)
      << "The stats are empty. Please run the profiler first.";
  TVM_FFI_CHECK(stats_sec.count("Total"), ValueError)
      << "The total time is not recorded. This method should be called only after "
         "exiting the profiler's with scope.";
  double total = stats_sec.at("Total");
  struct Entry {
    ffi::String name;
    double minutes;
    double percentage;
    bool operator<(const Entry& other) const { return percentage > other.percentage; }
  };
  std::vector<Entry> table_entry;
  for (const auto& kv : stats_sec) {
    table_entry.push_back(Entry{kv.first, kv.second / 60.0, kv.second / total * 100.0});
  }
  std::sort(table_entry.begin(), table_entry.end());
  support::TablePrinter p;
  p.Row() << "ID"
          << "Name"
          << "Time (min)"
          << "Percentage";
  p.Separator();
  for (int i = 0, n = table_entry.size(); i < n; ++i) {
    if (i == 0) {
      p.Row() << table_entry[i].name << table_entry[i].minutes << table_entry[i].percentage;
    } else {
      p.Row() << i << table_entry[i].name << table_entry[i].minutes << table_entry[i].percentage;
    }
  }
  p.Separator();
  return p.AsStr();
}

Profiler::Profiler() {
  ffi::ObjectPtr<ProfilerObj> n = ffi::make_object<ProfilerObj>();
  n->stats_sec.clear();
  n->total_timer = nullptr;
  data_ = n;
}

ffi::Function ProfilerTimedScope(ffi::String name) {
  if (ffi::Optional<Profiler> opt_profiler = Profiler::Current()) {
    return ffi::TypedFunction<void()>([profiler = opt_profiler.value(),                  //
                                       tik = std::chrono::high_resolution_clock::now(),  //
                                       name = std::move(name)]() {
      auto tok = std::chrono::high_resolution_clock::now();
      double duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(tok - tik).count() / 1e9;
      profiler->stats_sec[name] += duration;
    });
  }
  return nullptr;
}

ScopedTimer Profiler::TimedScope(ffi::String name) { return ScopedTimer(ProfilerTimedScope(name)); }

/**************** Context Manager ****************/

std::vector<Profiler>* ThreadLocalProfilers() {
  static thread_local std::vector<Profiler> profilers;
  return &profilers;
}

void Profiler::EnterWithScope() {
  ThreadLocalProfilers()->push_back(*this);
  (*this)->total_timer = ProfilerTimedScope("Total");
}

void Profiler::ExitWithScope() {
  ThreadLocalProfilers()->pop_back();
  if ((*this)->total_timer != nullptr) {
    (*this)->total_timer();
    (*this)->total_timer = nullptr;
  }
}

ffi::Optional<Profiler> Profiler::Current() {
  std::vector<Profiler>* profilers = ThreadLocalProfilers();
  if (profilers->empty()) {
    return std::nullopt;
  } else {
    return profilers->back();
  }
}

TVM_FFI_STATIC_INIT_BLOCK() { ProfilerObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.meta_schedule.Profiler", []() -> Profiler { return Profiler(); })
      .def_method("s_tir.meta_schedule.ProfilerEnterWithScope", &Profiler::EnterWithScope)
      .def_method("s_tir.meta_schedule.ProfilerExitWithScope", &Profiler::ExitWithScope)
      .def("s_tir.meta_schedule.ProfilerCurrent", Profiler::Current)
      .def_method("s_tir.meta_schedule.ProfilerGet", &ProfilerObj::Get)
      .def_method("s_tir.meta_schedule.ProfilerTable", &ProfilerObj::Table)
      .def("s_tir.meta_schedule.ProfilerTimedScope", ProfilerTimedScope);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
