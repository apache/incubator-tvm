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
"""The Relax Adreno GPU backend compilation pipeline and other passes."""

import tvm
import tvm_tirx
from tvm import relax
from tvm.relax.transform.legalize_ops import adreno as legalize_adreno


def library_dispatch_passes(target: tvm_tirx.target.Target):  # pylint: disable=unused-argument
    """The default library dispatch passes for Adreno GPU backend."""
    if "clml" in target.keys:
        return [
            relax.backend.adreno.clml.OpenCLMLOffLoadForLLM(target),
            relax.backend.adreno.clml.OpenCLMLOffLoad(),
        ]
    else:
        return []


def legalize_passes(target: tvm_tirx.target.Target):  # pylint: disable=unused-argument
    """The default legalization passes for Adreno GPU backend."""
    desired_layouts = {"relax.nn.conv2d": ["NCHW4c", "OIHW4o", "NCHW4c"]}
    skip_ops = [
        "relax.nn.conv2d",
        "relax.nn.max_pool2d",
        "relax.nn.adaptive_avg_pool2d",
    ]
    pass_list = []

    pass_list.extend(
        [
            tvm_tirx.tirx.transform.BindTarget(target),
            relax.transform.DecomposeOpsForInference(),
        ]
    )
    if "texture" in target.keys:
        pass_list.extend(
            [
                relax.transform.ConvertLayout(desired_layouts),
                relax.transform.Normalize(),
                relax.transform.FoldConstant(),
                relax.transform.LegalizeOps(skip_ops=skip_ops, target=target),
                relax.transform.AnnotateTIROpPattern(),
                relax.backend.adreno.transform.AnnotateCustomMemoryScope(target),
            ]
        )
    pass_list.extend([tvm.relax.transform.LegalizeOps(target=target)])
    if "texture" in target.keys:
        pass_list.extend(
            [
                relax.transform.LegalizeOps(
                    {"relax.nn.conv2d": legalize_adreno.conv2d_NCHWc_OIHWo}, target=target
                )
            ]
        )

    pass_list.extend(
        [
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FoldConstant(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
            relax.transform.DeadCodeElimination(),
        ]
    )
    if "texture" in target.keys:
        pass_list.extend(
            [
                relax.backend.adreno.transform.FoldVDeviceScopeChange(),
                relax.transform.DeadCodeElimination(),
                relax.transform.SpecializeFunctionBasedOnCallSite(),
            ]
        )
    from tvm.s_tir import dlight as dl  # pylint: disable=import-outside-toplevel

    pass_list.extend([relax.transform.Normalize()])
    pass_list.extend(
        [
            dl.ApplyDefaultSchedule(
                dl.adreno.Conv2d(), dl.adreno.LayoutTransform(), dl.adreno.Pool2D(), target=target
            )
        ]
    )
    pass_list.extend(
        [
            dl.ApplyDefaultSchedule(
                dl.gpu.Reduction(), dl.gpu.GeneralReduction(), dl.gpu.Fallback(), target=target
            )
        ]
    )
    return pass_list


def dataflow_lower_passes(target: tvm_tirx.target.Target):  # pylint: disable=unused-argument
    """The default dataflow lowering passes for Adreno GPU backend."""
    return relax.backend.gpu_generic.dataflow_lower_passes(target)


def finalize_passes(target: tvm_tirx.target.Target):  # pylint: disable=unused-argument
    """The default finalization passes for Adreno GPU backend."""
    return relax.backend.gpu_generic.finalize_passes(target)


def get_default_pipeline(target: tvm_tirx.target.Target):
    """Return the default compilation pipeline for Adreno GPU."""

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
