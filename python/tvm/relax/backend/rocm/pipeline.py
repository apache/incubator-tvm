# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The Relax ROCm backend compilation pipeline and other passes."""

import tvm
import tvm_tirx
from tvm import relax


def library_dispatch_passes(target: tvm_tirx.target.Target):  # pylint: disable=unused-argument
    """The default library dispatch passes for ROCm backend."""
    return [
        relax.backend.DispatchSampling(target=target),
        relax.backend.DispatchSortScan(target=target),
    ]


def legalize_passes(target: tvm_tirx.target.Target):  # pylint: disable=unused-argument
    """The default legalization passes for ROCm backend."""
    from tvm.s_tir import dlight as dl  # pylint: disable=import-outside-toplevel

    return [
        tvm.relax.transform.LegalizeOps(target=target),
        tvm.relax.transform.AnnotateTIROpPattern(),
        tvm.relax.transform.FoldConstant(),
        tvm.relax.transform.FuseOps(),
        tvm.relax.transform.FuseTIR(),
        dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
            target=target,
        ),
    ]


def dataflow_lower_passes(target: tvm_tirx.target.Target):  # pylint: disable=unused-argument
    """The default dataflow lowering passes for ROCm backend."""
    return [
        relax.transform.RewriteDataflowReshape(),
        relax.transform.ToNonDataflow(),
        relax.transform.RemovePurityChecking(),
        relax.transform.CallTIRRewrite(),
    ]


def finalize_passes(target: tvm_tirx.target.Target):  # pylint: disable=unused-argument
    """The default finalization passes for ROCm backend."""
    return [
        relax.transform.StaticPlanBlockMemory(),
        relax.transform.LowerAllocTensor(),
        relax.transform.KillAfterLastUse(),
        relax.transform.LowerRuntimeBuiltin(),
        relax.transform.ComputePrimValue(),
        relax.transform.VMShapeLower(),
        relax.transform.AttachGlobalSymbol(),
    ]


def get_default_pipeline(target: tvm_tirx.target.Target):
    """Return the default compilation pipeline for ROCm."""

    @tvm_tirx.ir.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm_tirx.ir.IRModule, _ctx: tvm_tirx.ir.transform.PassContext):
        seq = tvm_tirx.ir.transform.Sequential(
            library_dispatch_passes(target)
            + legalize_passes(target)
            + dataflow_lower_passes(target)
            + finalize_passes(target)
        )
        mod = seq(mod)
        return mod

    return _pipeline
