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
#ifndef SRC_RELAX_SCRIPT_PRINTER_UTILS_H_
#define SRC_RELAX_SCRIPT_PRINTER_UTILS_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/global_info.h>
#include <tvm/relax/type.h>
#include <tvm/script/printer/support.h>

#include <optional>

namespace tvm {
namespace script {
namespace printer {
namespace details {

ffi::Optional<ffi::String> GlobalInfoSelector(DocTranslatorObj* d, const GlobalInfo& info);
ExprDoc RelaxShapeDim(DocTranslatorObj* d, const PrimExpr& dim);
ffi::Array<StmtDoc> RelaxSeqBody(DocTranslatorObj* d, const relax::SeqExprObj* seq,
                                 ffi::Optional<IdDoc> destination = std::nullopt,
                                 ffi::Optional<ExprDoc> annotation = std::nullopt,
                                 const ffi::Object* destination_object = nullptr);

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // SRC_RELAX_SCRIPT_PRINTER_UTILS_H_
