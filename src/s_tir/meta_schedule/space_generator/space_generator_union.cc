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

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*! \brief The union of design space generators. */
class SpaceGeneratorUnionObj : public SpaceGeneratorObj {
 public:
  /*! \brief The array of design space generators unioned, could be recursive. */
  ffi::Array<SpaceGenerator> space_generators;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpaceGeneratorUnionObj>().def_ro("space_generators",
                                                     &SpaceGeneratorUnionObj::space_generators);
  }

  void InitializeWithTuneContext(const TuneContext& context) final {
    SpaceGeneratorObj::InitializeWithTuneContext(context);
    for (const SpaceGenerator& space_generator : space_generators) {
      space_generator->InitializeWithTuneContext(context);
    }
  }

  ffi::Array<s_tir::Schedule> GenerateDesignSpace(const IRModule& mod) final {
    ffi::Array<s_tir::Schedule> design_spaces;
    for (const SpaceGenerator& space_generator : space_generators) {
      // Generate partial design spaces from each design space generator.
      ffi::Array<s_tir::Schedule> partial = space_generator->GenerateDesignSpace(mod);
      // Merge the partial design spaces.
      design_spaces.insert(design_spaces.end(), partial.begin(), partial.end());
    }
    return design_spaces;
  }

  SpaceGenerator Clone() const final {
    ffi::ObjectPtr<SpaceGeneratorUnionObj> n = ffi::make_object<SpaceGeneratorUnionObj>(*this);
    n->space_generators = ffi::Array<SpaceGenerator>();
    for (const SpaceGenerator& space_generator : this->space_generators) {
      n->space_generators.push_back(space_generator->Clone());
    }
    CloneRules(this, n.get());
    return SpaceGenerator(n);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.SpaceGeneratorUnion",
                                    SpaceGeneratorUnionObj, SpaceGeneratorObj);
};

/*!
 * \brief Create a design space generator as union of given design space generators.
 * \param space_generators Array of the design space generators to be unioned.
 * \return The design space generator created.
 */
SpaceGenerator SpaceGenerator::SpaceGeneratorUnion(
    ffi::Array<SpaceGenerator> space_generators, ffi::Optional<ffi::Array<ScheduleRule>> sch_rules,
    ffi::Optional<ffi::Array<Postproc>> postprocs,
    ffi::Optional<ffi::Map<Mutator, FloatImm>> mutator_probs) {
  ffi::ObjectPtr<SpaceGeneratorUnionObj> n = ffi::make_object<SpaceGeneratorUnionObj>();
  n->sch_rules = std::move(sch_rules);
  n->postprocs = std::move(postprocs);
  n->mutator_probs = std::move(mutator_probs);
  n->space_generators = std::move(space_generators);
  return SpaceGenerator(n);
}

TVM_FFI_STATIC_INIT_BLOCK() { SpaceGeneratorUnionObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.SpaceGeneratorSpaceGeneratorUnion",
                        SpaceGenerator::SpaceGeneratorUnion);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
