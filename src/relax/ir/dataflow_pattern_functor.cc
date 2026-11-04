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
 * \file src/tvm/relax/dataflow_matcher.cc
 * \brief The dataflow pattern matcher for Relax.
 */

#include <tvm/relax/dataflow_pattern_functor.h>

namespace tvm {
namespace relax {

// DFPatternVisitor

void DFPatternVisitor::VisitDFPattern(const DFPattern& pattern) {
  if (this->visited_.count(pattern.get()) == 0) {
    visited_.insert(pattern.get());
    DFPatternFunctor::VisitDFPattern(pattern);
  }
}

void DFPatternVisitor::VisitDFPattern_(const OrPatternObj* op) {
  VisitDFPattern(op->left);
  VisitDFPattern(op->right);
}

void DFPatternVisitor::VisitDFPattern_(const AndPatternObj* op) {
  VisitDFPattern(op->left);
  VisitDFPattern(op->right);
}

void DFPatternVisitor::VisitDFPattern_(const NotPatternObj* op) { VisitDFPattern(op->reject); }

void DFPatternVisitor::VisitDFPattern_(const AttrPatternObj* op) { VisitDFPattern(op->pattern); }

void DFPatternVisitor::VisitDFPattern_(const CallPatternObj* op) {
  VisitDFPattern(op->op);
  if (op->args.defined()) {
    for (auto arg : op->args) {
      VisitDFPattern(arg);
    }
  }
}

void DFPatternVisitor::VisitDFPattern_(const DataTypePatternObj* op) {
  VisitDFPattern(op->pattern);
}

void DFPatternVisitor::VisitDFPattern_(const ExprPatternObj* op) {}

void DFPatternVisitor::VisitDFPattern_(const FunctionPatternObj* op) {
  if (op->params.defined()) {
    for (auto param : op->params) {
      VisitDFPattern(param);
    }
  }
  VisitDFPattern(op->body);
}

void DFPatternVisitor::VisitDFPattern_(const ShapePatternObj* op) { VisitDFPattern(op->pattern); }

void DFPatternVisitor::VisitDFPattern_(const TupleGetItemPatternObj* op) {
  VisitDFPattern(op->tuple);
}

void DFPatternVisitor::VisitDFPattern_(const TuplePatternObj* op) {
  if (op->fields.defined()) {
    for (auto field : op->fields) {
      VisitDFPattern(field);
    }
  }
}

void DFPatternVisitor::VisitDFPattern_(const UnorderedTuplePatternObj* op) {
  if (op->fields.defined()) {
    for (auto field : op->fields) {
      VisitDFPattern(field);
    }
  }
}

void DFPatternVisitor::VisitDFPattern_(const TypePatternObj* op) { VisitDFPattern(op->pattern); }

// leaf nodes.
void DFPatternVisitor::VisitDFPattern_(const PrimArrPatternObj* op) {}
void DFPatternVisitor::VisitDFPattern_(const VarPatternObj* op) {}
void DFPatternVisitor::VisitDFPattern_(const ConstantPatternObj* op) {}
void DFPatternVisitor::VisitDFPattern_(const DataflowVarPatternObj* op) {}
void DFPatternVisitor::VisitDFPattern_(const GlobalVarPatternObj* op) {}
void DFPatternVisitor::VisitDFPattern_(const ExternFuncPatternObj* op) {}
void DFPatternVisitor::VisitDFPattern_(const WildcardPatternObj* op) {}

}  // namespace relax
}  // namespace tvm
