# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Relax script pyfunc."""

import pytest
import torch

import tvm_tirx
import tvm_tirx.testing
from tvm.relax import BasePyModule
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

n = T.dynamic("n", "int32")


@R.py_module
class TestPyFuncModule(BasePyModule):
    """Test module with Python functions using @T.pyfunc decorator."""

    @I.pyfunc
    def pytorch_processor(x: torch.Tensor) -> torch.Tensor:
        """Python function that processes PyTorch tensors."""
        return torch.nn.functional.relu(x) * 2.0

    @I.pyfunc
    def pytorch_adder(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Python function that adds two PyTorch tensors."""
        return x + y

    @I.pyfunc
    def pytorch_complex_ops(x: torch.Tensor) -> torch.Tensor:
        """Complex PyTorch operations."""
        result = torch.nn.functional.softmax(x, dim=0)
        result = torch.nn.functional.dropout(result, p=0.1, training=False)
        return result * 10.0

    @Ts.function
    def simple_tir_func(
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})

        for i in T.grid(n):
            with Ts.sblock("copy"):
                vi = Ts.axis.remap("S", [i])
                B[vi] = A[vi]


class TestTVMScriptPyFunc:
    def test_pyfunc_decorator_creates_pyfuncs_attribute(self):
        module = TestPyFuncModule

        assert hasattr(module, "__pyfuncs__"), "Module should have pyfuncs attribute"

        pyfuncs = module.__pyfuncs__
        assert isinstance(pyfuncs, dict), "pyfuncs should be a dictionary"

        expected_functions = ["pytorch_processor", "pytorch_adder", "pytorch_complex_ops"]
        for func_name in expected_functions:
            assert func_name in pyfuncs, f"Function {func_name} should be in pyfuncs"

    def test_pyfunc_functions_are_callable(self):
        """Test that Python functions in pyfuncs are callable."""
        module = TestPyFuncModule
        pyfuncs = module.__pyfuncs__

        # Test pytorch_processor
        processor_func = pyfuncs["pytorch_processor"]
        assert callable(processor_func), "pytorch_processor should be callable"

        # Test pytorch_adder
        adder_func = pyfuncs["pytorch_adder"]
        assert callable(adder_func), "pytorch_adder should be callable"

        # Test pytorch_complex_ops
        complex_func = pyfuncs["pytorch_complex_ops"]
        assert callable(complex_func), "pytorch_complex_ops should be callable"

    def test_pyfunc_functions_execute_correctly(self):
        """Test that Python functions execute correctly."""
        module = TestPyFuncModule
        pyfuncs = module.__pyfuncs__

        # Create test data
        x = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0], dtype=torch.float32)
        y = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)

        # Test pytorch_processor
        processor_func = pyfuncs["pytorch_processor"]
        processor_result = processor_func(x)

        assert isinstance(processor_result, torch.Tensor)
        expected = torch.nn.functional.relu(x) * 2.0
        assert torch.allclose(processor_result, expected, atol=1e-5)

        # Test pytorch_adder
        adder_func = pyfuncs["pytorch_adder"]
        adder_result = adder_func(x, y)

        assert isinstance(adder_result, torch.Tensor)
        expected = x + y
        assert torch.allclose(adder_result, expected, atol=1e-5)

        # Test pytorch_complex_ops
        complex_func = pyfuncs["pytorch_complex_ops"]
        complex_result = complex_func(x)

        assert isinstance(complex_result, torch.Tensor)
        # Note: dropout is non-deterministic, so we just check shape and type
        assert complex_result.shape == x.shape
        assert complex_result.dtype == x.dtype

    def test_pyfunc_module_has_functions_attribute(self):
        """Test that the module has functions attribute for IRModule operations."""
        module = TestPyFuncModule

        # Check if functions attribute exists
        assert hasattr(module, "functions"), "Module should have functions attribute"

        functions = module.functions
        # TVM IRModule.functions is not a standard dict, but has dict-like behavior
        assert hasattr(functions, "__getitem__"), "functions should support dict-like access"
        assert hasattr(functions, "__iter__"), "functions should be iterable"

    def test_pyfunc_module_script_method(self):
        """Test that the module has script() method for TVMScript output."""
        module = TestPyFuncModule

        # Check if script method exists
        assert hasattr(module, "script"), "Module should have script method"

        # Test script method execution
        script_output = module.script()
        assert isinstance(script_output, str), "script() should return a string"
        assert len(script_output) > 0, "script() should return non-empty string"

    def test_pyfunc_module_inheritance_flag(self):
        """Test that the module has BasePyModule inheritance flag."""
        module = TestPyFuncModule

        # Check if inheritance flag exists (this might not be set in all implementations)
        if hasattr(module, "_base_py_module_inherited"):
            assert module._base_py_module_inherited, "Inheritance flag should be True"
        else:
            # Alternative: check if the module supports Python functions
            assert hasattr(module, "__pyfuncs__"), "Module should support Python functions"

        # Check if original class is preserved (this might not be set in all implementations)
        if hasattr(module, "_original_class"):
            assert module._original_class is not None, "Original class should be preserved"
        else:
            # Alternative: check if module is callable (ModuleFactory)
            assert hasattr(module, "__call__"), "Module should be callable (ModuleFactory)"

    def test_pyfunc_module_creation_and_execution(self):
        module = TestPyFuncModule

        assert hasattr(module, "__call__"), "Module should be callable"

        device = tvm_tirx.cpu(0)
        instance = module(device)

        assert isinstance(instance, BasePyModule), "Instance should be BasePyModule"
        assert hasattr(instance, "pyfuncs"), "Instance should have pyfuncs"

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = instance.pytorch_processor(x)

        assert isinstance(result, torch.Tensor)
        expected = torch.nn.functional.relu(x) * 2.0
        assert torch.allclose(result, expected, atol=1e-5)

    def test_pyfunc_module_creation_and_execution_gpu(self):
        module = TestPyFuncModule

        if tvm_tirx.cuda().exist:

            def run_and_check():
                device = tvm_tirx.cuda(0)
                instance = module(device)

                assert isinstance(instance, BasePyModule), "Instance should be BasePyModule"
                assert hasattr(instance, "pyfuncs"), "Instance should have pyfuncs"
                assert "cuda" in str(instance.target)

                x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda")
                result = instance.pytorch_processor(x)

                assert isinstance(result, torch.Tensor)
                assert result.device.type == "cuda"
                expected = torch.nn.functional.relu(x) * 2.0
                assert torch.allclose(result, expected, atol=1e-5)

            tvm_tirx.testing.run_with_gpu_lock(run_and_check)
        else:
            pytest.skip("CUDA not available")

    def test_pyfunc_with_tir_integration(self):
        """Test that Python functions can work with TIR functions."""
        module = TestPyFuncModule

        # Create instance
        device = tvm_tirx.cpu(0)
        instance = module(device)

        # Test TIR function execution
        n = 5
        input_tensor = torch.randn(n, dtype=torch.float32)

        # Call TIR function - it needs 3 arguments: input, output, and size
        # But call_tir handles the output buffer creation, so we only pass input and size
        # Note: TIR functions expect TVM types, not Python types
        result = instance.call_tir(
            instance.simple_tir_func,
            [input_tensor],  # Only pass input tensor, let call_tir handle the rest
            R.Tensor((n,), "float32"),
        )

        # Verify result
        assert isinstance(result, torch.Tensor)
        assert result.shape == (n,)
        assert torch.allclose(result, input_tensor, atol=1e-5)

    def test_pyfunc_decorator_preserves_function_signatures(self):
        """Test that @T.pyfunc decorator preserves function signatures."""
        module = TestPyFuncModule
        pyfuncs = module.__pyfuncs__

        # Check function signatures
        import inspect

        # pytorch_processor signature
        processor_func = pyfuncs["pytorch_processor"]
        sig = inspect.signature(processor_func)
        params = list(sig.parameters.keys())
        assert len(params) == 1, "pytorch_processor should have 1 parameter"
        assert params[0] == "x", "First parameter should be 'x'"

        # pytorch_adder signature
        adder_func = pyfuncs["pytorch_adder"]
        sig = inspect.signature(adder_func)
        params = list(sig.parameters.keys())
        assert len(params) == 2, "pytorch_adder should have 2 parameters"
        assert params[0] == "x", "First parameter should be 'x'"
        assert params[1] == "y", "Second parameter should be 'y'"


if __name__ == "__main__":
    tvm_tirx.testing.main()
