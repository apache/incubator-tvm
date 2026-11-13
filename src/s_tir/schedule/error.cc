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
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/script/printer/printer.h>

#include "./utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

const ScheduleErrorContextObj* GetScheduleErrorContext(const ffi::Error& error) {
  if (auto context = error.extra_context()) {
    if (const auto* payload = context->as<ScheduleErrorContextObj>()) {
      return payload;
    }
    if (const auto* visit_context = context->as<ffi::VisitErrorContextObj>()) {
      if (visit_context->prev_error_context) {
        return visit_context->prev_error_context.value().as<ScheduleErrorContextObj>();
      }
    }
  }
  return nullptr;
}

ffi::String ScheduleErrorContextObj::RenderReport(const ffi::String& primitive) const {
  IRModule mod = this->mod();
  std::ostringstream os;

  // get locations of interest
  ffi::Array<ffi::ObjectRef> locs = LocationsOfInterest();
  std::unordered_map<ffi::ObjectRef, ffi::String, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      loc_obj_to_name;
  int n_locs = locs.size();
  std::string msg = DetailRenderTemplate();
  PrinterConfig cfg;
  if (n_locs > 0) {
    for (int i = 0; i < n_locs; ++i) {
      std::string name = locs[i]->GetTypeKey() + '#' + std::to_string(i);
      std::string src = "{" + std::to_string(i) + "}";
      for (size_t pos; (pos = msg.find(src)) != std::string::npos;) {
        msg.replace(pos, src.length(), name);
      }
      cfg->obj_to_annotate.Set(locs[i], name);
      cfg->obj_to_underline.push_back(locs[i]);
    }
  }
  os << "ScheduleError: An error occurred in the schedule primitive '" << primitive
     << "'.\n\nThe IR with diagnostic is:\n"
     << tvm::Script(mod, cfg) << std::endl;

  // print error message
  os << "Error message: " << msg;
  return os.str();
}

TVM_FFI_STATIC_INIT_BLOCK() { ffi::reflection::ObjectDef<ScheduleErrorContextObj>(); }

}  // namespace s_tir
}  // namespace tvm
