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

# pylint: disable=invalid-name, missing-function-docstring, not-callable
# pylint: disable=import-outside-toplevel, unused-argument, use-list-literal
# mypy: ignore-errors
"""PyTorch Dynamo backend of Relax."""

import functools

import tvm_ffi

import tvm
import tvm_tirx
from tvm.relax import build as relax_build

from .fx_translator import from_fx


def device_from_inputs(example_inputs):
    for x in example_inputs:
        if hasattr(x, "device"):
            return x.device
    return None


def relax_dynamo(pipeline: tvm_tirx.ir.transform.Pass | None = None):
    """A helper function to create a relax backend.

    Parameters
    ----------
    pipeline : Optional[tvm_tirx.ir.transform.Pass]
        The pipeline to be applied to the relax module before sent to build.

    Returns
    -------
    backend : Callable[[torch.fx.GraphModule, List[torch.Tensor]], Callable]
        The relax dynamo backend.
    """

    def _relax_backend(graph_module, example_inputs):
        import torch  # type: ignore[import]

        assert isinstance(graph_module, torch.fx.GraphModule)

        def to_torch_tensor(nd_tensor):
            """A helper function to transfer a Tensor to torch.tensor."""
            if isinstance(nd_tensor, torch.Tensor):
                # tvm-ffi #517 (Recursive DLPack container conversion) auto-converts
                # ffi::Tensor items returned in containers back to torch.Tensor when
                # the call site passed torch.Tensor inputs.
                return nd_tensor
            if isinstance(nd_tensor, tvm_tirx.runtime.Tensor):
                return torch.from_numpy(nd_tensor.numpy())
            elif isinstance(nd_tensor, tvm_ffi.Array):
                return tuple(to_torch_tensor(x) for x in nd_tensor)
            else:
                raise ValueError(f"Unsupported type {type(nd_tensor)}")

        graph_module.graph.eliminate_dead_code()

        device = device_from_inputs(example_inputs)

        assert len(example_inputs)

        fake_inputs = []
        if isinstance(example_inputs[0], torch._subclasses.fake_tensor.FakeTensor):
            # Fake tensors
            fake_inputs = example_inputs
        else:
            # Real tensors
            for node in graph_module.graph.nodes:
                if node.op != "placeholder":
                    continue
                if "grapharg" not in node.meta:
                    continue
                fake_tensor = node.meta["grapharg"].fake_tensor
                if fake_tensor is None:
                    continue
                fake_inputs.append(fake_tensor)

        input_info = []
        shape_vars = {}
        for tensor in fake_inputs:
            shape = []
            for s in tensor.shape:
                if isinstance(s, torch.SymInt):
                    if str(s) not in shape_vars:
                        shape_vars[str(s)] = tvm_tirx.tirx.Var(str(s), "int64")
                    shape.append(shape_vars[str(s)])
                else:
                    shape.append(s)
            input_info.append((shape, tensor.dtype))

        mod = from_fx(graph_module, input_info)

        if device.type == "cuda":
            dev = tvm_tirx.cuda(device.index)
            target = tvm_tirx.target.Target("cuda")
        else:
            dev = tvm_tirx.cpu(0)
            target = tvm_tirx.target.Target(llvm_target())

        # invoke optimization pipeline.
        if pipeline is None:
            # get default pipeline
            seq = tvm.relax.get_pipeline()
        elif isinstance(pipeline, str):
            # lookup by name
            seq = tvm.relax.get_pipeline(pipeline)
        else:
            seq = pipeline

        mod = mod.with_attr("target", target)
        mod = seq(mod)

        ex = relax_build(mod, target=target)

        vm = tvm.relax.VirtualMachine(ex.mod, device=dev)

        def exec_tvm(*i_args):
            args = [a.contiguous() for a in i_args if isinstance(a, torch.Tensor)]
            vm_args = list()
            for arg in args:
                if arg.requires_grad:
                    arg = arg.detach()
                if isinstance(arg, torch._subclasses.fake_tensor.FakeTensor):
                    # Materialize a real (eager) Tensor
                    arg = torch.randn(arg.shape, dtype=arg.dtype, device=device)
                vm_args.append(arg)
            outputs = vm["main"](*vm_args)
            return to_torch_tensor(outputs)

        return exec_tvm

    return _relax_backend


def dynamo_capture_subgraphs(model, *params, **kwargs) -> tvm_tirx.IRModule:
    """Capture subgraphs of the PyTorch model using torch.compile into an IRModule.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be captured.

    params : List[torch.Tensor]
        The parameters of the PyTorch model.

    keep_params_as_input : bool
        Whether to keep model parameters as input variables of the captured Relax functions.

    Returns
    -------
    output : ImporterOutput
        The output of translation, including the translated IRModule.
        If `keep_params_as_input` is true, the functions in the IRModule have an
        attribute "params" that contains the weights of the input model. The
        weights can be detached by `relax.frontend.detach_params`.
    """
    import torch  # type: ignore[import]
    from torch import _dynamo as dynamo  # type: ignore[import]
    from torch import fx  # type: ignore[import]

    keep_params_as_input = "keep_params_as_input" in kwargs and kwargs["keep_params_as_input"]
    kwargs.pop("keep_params_as_input", None)
    mod = tvm_tirx.IRModule()

    def _capture(graph_module: fx.GraphModule, example_inputs):
        assert isinstance(graph_module, torch.fx.GraphModule)
        input_info = [(tuple(tensor.shape), str(tensor.dtype)) for tensor in example_inputs]
        mod_ = from_fx(
            graph_module,
            input_info,
            keep_params_as_input=keep_params_as_input,
            unwrap_unit_return_tuple=True,
        )
        new_name = f"subgraph_{len(mod.get_global_vars())}"
        mod[new_name] = mod_["main"].with_attr("global_symbol", new_name)
        return graph_module.forward

    dynamo.reset()
    compiled_model = torch.compile(model, backend=_capture)

    with torch.no_grad():
        compiled_model(*params, **kwargs)

    return mod


@functools.lru_cache(None)
def llvm_target():
    import platform
    import subprocess

    AVX512_TARGET = {"kind": "llvm", "mcpu": "skylake-avx512"}
    AVX2_TARGET = {"kind": "llvm", "mcpu": "core-avx2"}
    DEFAULT_TARGET = "llvm"

    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
            if "avx512" in cpuinfo:
                return AVX512_TARGET
            return AVX2_TARGET
        except FileNotFoundError:
            pass
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                cpu_features = result.stdout.lower()
                if "avx512" in cpu_features:
                    return AVX512_TARGET
                if "avx2" in cpu_features:
                    return AVX2_TARGET
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

        if platform.machine() == "arm64":
            return DEFAULT_TARGET

    # Default fallback
    return DEFAULT_TARGET
