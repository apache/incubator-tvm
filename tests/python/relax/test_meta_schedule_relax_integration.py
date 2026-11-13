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
# ruff: noqa: E501, F401, F841
"""Integration test for MetaSchedule"""

import tempfile

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.runtime.vm import VirtualMachine
from tvm.s_tir import meta_schedule as ms
from tvm_tirx.runtime import cpu as tvm_cpu
from tvm_tirx.runtime import tensor as tvm_tensor
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


# fmt: off
@I.ir_module
class Module0:
    @R.function
    def main(data: R.Tensor((1, 8, 8, 4), dtype="int32")) -> R.Tensor((1, 8, 8, 4), dtype="int32"):
        cls = Module0
        with R.dataflow():
            c = R.const([[[[-171701247],[-1719837685],[1801664104],[-634316588]],[[920159370],[-132073802],[2142531563],[1465185701]],[[-1505608067],[1737948828],[1581089391],[-1986167320]]],[[[-1449581822],[35714587],[496324563],[-1430879015]],[[-1615680873],[1198514997],[1494683955],[1567376558]],[[1319924884],[-380548171],[296785437],[-1546305981]]],[[[-398644701],[-2004794585],[-1850413687],[2072643657]],[[847950121],[-544212073],[-199532669],[-343273682]],[[953721562],[-1930209358],[1573600108],[-577689853]]]], "int32")
            lv: R.Tensor((1, 8, 8, 4), dtype="int32") = R.nn.conv2d(data, c, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=4, data_layout="NHWC", kernel_layout="HWOI", out_layout="NHWC", out_dtype="int32")
            b = R.const([[[[1, 1, 1, 1]]]], "int32")
            lv1: R.Tensor((1, 8, 8, 4), dtype="int32") = R.add(lv, b)
            c1 = R.const([[[[2042349344],[-2076067063],[1528163722],[-1156452837]],[[-2097172051],[1137787079],[-601389657],[1907495997]],[[987801941],[1073738593],[-1410339796],[-689755358]]],[[[90351522],[-44886952],[-1914103775],[-691553659]],[[-1288505112],[-1376578817],[-2067933148],[-1413101824]],[[1261422027],[-156976862],[-1185734459],[1608778622]]],[[[-664209483],[1907479806],[1838595152],[464942526]],[[877953160],[415131837],[-2010736511],[1218242769]],[[-1440127632],[112931],[521745784],[-1931145893]]]], "int32")
            lv2: R.Tensor((1, 8, 8, 4), dtype="int32") = R.nn.conv2d(lv1, c1, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=4, data_layout="NHWC", kernel_layout="HWOI", out_layout="NHWC", out_dtype="int32")
            c2 = R.const([[[[687940110],[-910571705],[-901609800],[-500525928]],[[506872399],[1070176297],[-305936110],[1625439784]],[[-1565626954],[-1705688881],[-866370805],[-1750740826]]],[[[300497007],[-626864803],[390295545],[222549121]],[[319224543],[-2003064970],[657992492],[2014175448]],[[653278589],[-768810984],[-294555581],[-1197167662]]],[[[1703154671],[-1540759805],[-568817430],[-1729755444]],[[-275458074],[2078945571],[1683298006],[-1029327874]],[[1315093181],[159010501],[875694807],[-223655381]]]], "int32")
            lv3: R.Tensor((1, 8, 8, 4), dtype="int32") = R.nn.conv2d(lv2, c2, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=4, data_layout="NHWC", kernel_layout="HWOI", out_layout="NHWC", out_dtype="int32")
            gv: R.Tensor((1, 8, 8, 4), dtype="int32") = lv3
            R.output(gv)
        return gv

# fmt: on


def test_extracting_tasks():
    target = {"kind": "llvm", "mcpu": "core-avx2", "num-cores": 1}

    relax_mod = Module0
    relax_mod = relax.transform.LegalizeOps()(relax_mod)
    relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
    relax_mod = relax.transform.FuseOps()(relax_mod)
    relax_mod = relax.transform.FoldConstant()(relax_mod)
    relax_mod = relax.transform.FuseTIR()(relax_mod)

    relax_expectation = {
        "structural": 2,  # The relax constants do not reach the tirx at the lowering.
        "ignore-tensor": 2,
        "anchor-block": 1,
    }
    for module_equality, count in relax_expectation.items():
        extracted_tasks = ms.relax_integration.extract_tasks(
            relax_mod,
            target,
            {},
            module_equality=module_equality,
        )
        assert len(extracted_tasks) == count


def test_compile_relax_with_database():
    """End-to-end test: tune with MetaSchedule then compile_relax with the database.

    Verifies that the pipeline ordering in compile_relax is correct: tasks are
    extracted and tuned against fused-TIR keys, and compile_relax produces those
    same keys (by running LegalizeOps + FuseOps + FuseTIR before applying the
    database), so the scheduled kernels are actually picked up.
    """
    pytest.importorskip("cloudpickle")  # needed by meta_schedule popen workers

    target = tvm_tirx.target.Target({"kind": "llvm", "num-cores": 1})

    # Prepare the fused module whose TIR keys will populate the database.
    fused_mod = Module0
    fused_mod = relax.transform.LegalizeOps()(fused_mod)
    fused_mod = relax.transform.AnnotateTIROpPattern()(fused_mod)
    fused_mod = relax.transform.FuseOps()(fused_mod)
    fused_mod = relax.transform.FoldConstant()(fused_mod)
    fused_mod = relax.transform.FuseTIR()(fused_mod)

    with tempfile.TemporaryDirectory() as work_dir:
        database = ms.relax_integration.tune_relax(
            fused_mod,
            params={},
            target=target,
            work_dir=work_dir,
            max_trials_global=4,
        )
        # compile_relax takes the raw module and builds the fused-TIR pipeline
        # internally; the database keys must therefore match the ones above.
        exe = ms.relax_integration.compile_relax(
            database=database,
            mod=Module0,
            target=target,
            params=None,
        )

    dev = tvm_cpu()
    vm = VirtualMachine(exe.jit(), dev)
    data = tvm_tensor(np.zeros((1, 8, 8, 4), dtype="int32"), device=dev)
    result = vm["main"](data)
    assert result.numpy().shape == (1, 8, 8, 4)


if __name__ == "__main__":
    tvm_tirx.testing.main()
