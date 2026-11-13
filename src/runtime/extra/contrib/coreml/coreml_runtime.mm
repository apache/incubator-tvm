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
 * \file coreml_runtime.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <tvm/support/bytes_io.h>
#include "coreml_runtime.h"

namespace tvm {
namespace runtime {

void CoreMLModel::Invoke() {
  id<MLFeatureProvider> input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:input_dict_
                                                                                  error:nil];
  output_ = [model_ predictionFromFeatures:input error:nil];
}

void CoreMLModel::SetInput(const std::string& key, DLTensor* data_in) {
  int64_t size = 1;
  NSMutableArray* shape = [[NSMutableArray alloc] init];
  for (int64_t i = 0; i < data_in->ndim; ++i) {
    size *= data_in->shape[i];
    [shape addObject:[NSNumber numberWithInteger:data_in->shape[i]]];
  }

  DLDataType dtype = data_in->dtype;
  MLMultiArrayDataType dataType;
  if (dtype == DLDataType{kDLFloat, 64, 1}) {
    dataType = MLMultiArrayDataTypeDouble;
    size *= sizeof(double);
  } else if (dtype == DLDataType{kDLFloat, 32, 1}) {
    dataType = MLMultiArrayDataTypeFloat32;
    size *= sizeof(float);
  } else if (dtype == DLDataType{kDLInt, 32, 1}) {
    dataType = MLMultiArrayDataTypeInt32;
    size *= sizeof(int);
  } else {
    LOG(FATAL) << "unsupported data type " << dtype;
    return;
  }

  MLMultiArray* dest = [[MLMultiArray alloc] initWithShape:shape dataType:dataType error:nil];

  TVM_FFI_ICHECK(ffi::IsContiguous(*data_in));
  memcpy(dest.dataPointer, data_in->data, size);

  NSString* nsKey = [NSString stringWithUTF8String:key.c_str()];
  [input_dict_ setObject:dest forKey:nsKey];
}

ffi::Tensor CoreMLModel::GetOutput(int index) const {
  MLModelDescription* model_desc = model_.modelDescription;
  NSString* metadata = [model_desc metadata][MLModelDescriptionKey];
  NSData* data = [metadata dataUsingEncoding:NSUTF8StringEncoding];
  NSDictionary* json = [NSJSONSerialization JSONObjectWithData:data
                                                       options:NSJSONReadingAllowFragments
                                                         error:nil];
  NSString* name = json[@"outputs"][index];
  MLFeatureDescription* output_desc = model_desc.outputDescriptionsByName[name];
  MLMultiArrayConstraint* data_desc = output_desc.multiArrayConstraint;
  std::vector<int64_t> shape;
  int64_t size = 1;
  for (int64_t i = 0; i < data_desc.shape.count; ++i) {
    int n = data_desc.shape[i].intValue;
    size *= n;
    shape.push_back(n);
  }

  DLDataType dtype = DLDataType{kDLOpaqueHandle, 0, 0};
  if (data_desc.dataType == MLMultiArrayDataTypeDouble) {
    dtype = DLDataType{kDLFloat, 64, 1};
    size *= sizeof(double);
  } else if (data_desc.dataType == MLMultiArrayDataTypeFloat32) {
    dtype = DLDataType{kDLFloat, 32, 1};
    size *= sizeof(float);
  } else if (data_desc.dataType == MLMultiArrayDataTypeInt32) {
    dtype = DLDataType{kDLInt, 32, 1};
    size *= sizeof(int);
  } else {
    LOG(FATAL) << "unexpected data type " << data_desc.dataType;
  }
  MLMultiArray* src = [output_ featureValueForName:name].multiArrayValue;
  Device cpu_dev = {
      .device_type = kDLCPU,
      .device_id = 0,
  };
  ffi::Tensor ret = tvm::runtime::Empty(shape, dtype, cpu_dev);
  tvm::runtime::CopyFromBytes(ret, src.dataPointer, size);

  return ret;
}

int CoreMLModel::GetNumOutputs() const {
  MLModelDescription* model_desc = model_.modelDescription;
  return [[model_desc outputDescriptionsByName] count];
}

void CoreMLRuntime::Init(const std::string& symbol, const std::string& _model_path) {
  symbol_ = symbol;

  NSString* model_path = [NSString stringWithUTF8String:(_model_path).c_str()];
  if (![model_path hasPrefix:@"/"]) {
    // find models in the bundle's framework
    NSBundle* bundle = [NSBundle mainBundle];
    NSString* base = [[bundle privateFrameworksPath] stringByAppendingPathComponent:@"tvm"];
    model_path = [base stringByAppendingPathComponent:model_path];
  }

  NSURL* url = [NSURL fileURLWithPath:model_path];
  model_ = std::unique_ptr<CoreMLModel>(new CoreMLModel(url));
}

ffi::Optional<ffi::Function> CoreMLRuntime::GetFunction(const ffi::String& name) {
  // Return member functions during query.
  if (name == "invoke" || name == "run") {
    return ffi::Function([this](ffi::PackedArgs args, ffi::Any* rv) { model_->Invoke(); });
  } else if (name == "set_input") {
    return ffi::Function([this](ffi::PackedArgs args, ffi::Any* rv) {
      model_->SetInput(args[0].cast<std::string>(), args[1].cast<DLTensor*>());
    });
  } else if (name == "get_output") {
    return ffi::Function([this](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = model_->GetOutput(args[0].cast<int>());
    });
  } else if (name == "get_num_outputs") {
    return ffi::Function(
        [this](ffi::PackedArgs args, ffi::Any* rv) { *rv = model_->GetNumOutputs(); });
  } else if (name == symbol_) {
    // Return the packedfunc which executes the subgraph.
    return ffi::Function([this](ffi::PackedArgs args, ffi::Any* rv) {
      MLModelDescription* model_desc = [model_->model_ modelDescription];
      NSString* metadata = [model_desc metadata][MLModelDescriptionKey];
      NSData* data = [metadata dataUsingEncoding:NSUTF8StringEncoding];
      NSDictionary* json = [NSJSONSerialization JSONObjectWithData:data
                                                           options:NSJSONReadingAllowFragments
                                                             error:nil];
      NSArray<NSString*>* input_names = json[@"inputs"];

      // Copy input tensors to corresponding data entries.
      for (auto i = 0; i < args.size() - 1; ++i) {
        model_->SetInput([input_names[i] UTF8String], args[i].cast<DLTensor*>());
      }

      // Execute the subgraph.
      model_->Invoke();

      // TODO: Support multiple outputs.
      ffi::Tensor out = model_->GetOutput(0);
      tvm::runtime::CopyFromTo(out, args[args.size() - 1].cast<DLTensor*>());
      *rv = out;
    });
  } else {
    return std::nullopt;
  }
}

ffi::Module CoreMLRuntimeCreate(const std::string& symbol, const std::string& model_path) {
  auto exec = ffi::make_object<CoreMLRuntime>();
  exec->Init(symbol, model_path);
  return ffi::Module(exec);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tvm.coreml_runtime.create", [](ffi::PackedArgs args, ffi::Any* rv) {
    *rv = CoreMLRuntimeCreate(args[0].cast<std::string>(), args[1].cast<std::string>());
  });
}

ffi::Bytes CoreMLRuntime::SaveToBytes() const {
  std::string result;
  support::BytesOutStream stream(&result);
  NSURL* url = model_->url_;
  NSFileWrapper* dirWrapper = [[[NSFileWrapper alloc] initWithURL:url options:0
                                                            error:nil] autorelease];
  NSData* dirData = [dirWrapper serializedRepresentation];
  stream.Write(symbol_);
  stream.Write((uint64_t)[dirData length]);
  stream.Write([dirData bytes], [dirData length]);
  DLOG(INFO) << "Save " << symbol_ << " (" << [dirData length] << " bytes)";
  return ffi::Bytes(std::move(result));
}

/*!
 * \brief Load a CoreML module from stream.
 *
 * \param strm The binary stream to load json.
 *
 * \return The created CoreML module.
 */
ffi::Module CoreMLRuntimeLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);

  NSString* tempBaseDir = NSTemporaryDirectory();
  if (tempBaseDir == nil) tempBaseDir = @"/tmp";

  NSString* templateStr = [tempBaseDir stringByAppendingPathComponent:@"tvm.XXXXXX"];
  const char* fsTemplate = [templateStr fileSystemRepresentation];
  NSMutableData* bufferData = [NSMutableData dataWithBytes:fsTemplate
                                                    length:strlen(fsTemplate) + 1];
  char* buffer = (char*)[bufferData mutableBytes];
  char* result = mkdtemp(buffer);
  NSString* tempDir = [NSString stringWithUTF8String:result];

  std::string symbol;
  stream.Read(&symbol);
  uint64_t length;
  stream.Read(&length);
  void* ptr = new char[length];
  stream.Read(ptr, length);
  NSData* data = [[NSData alloc] initWithBytesNoCopy:ptr length:length];
  NSFileWrapper* dirWrapper =
      [[[NSFileWrapper alloc] initWithSerializedRepresentation:data] autorelease];
  NSString* dirname = [NSString stringWithUTF8String:(symbol + ".mlmodelc").c_str()];
  NSString* model_path = [tempDir stringByAppendingPathComponent:dirname];
  NSURL* url = [NSURL fileURLWithPath:model_path];
  BOOL res = [dirWrapper writeToURL:url options:0 originalContentsURL:nil error:nil];
  TVM_FFI_ICHECK(res) << "Failed to create model directory " << [model_path UTF8String];

  auto exec = ffi::make_object<CoreMLRuntime>();
  exec->Init(symbol, [model_path UTF8String]);
  return ffi::Module(exec);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ffi.Module.load_from_bytes.coreml", CoreMLRuntimeLoadFromBytes);
}

}  // namespace runtime
}  // namespace tvm
