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
"""
Test DLPack integration between PyTorch and TVM.

This test verifies:
1. DLPack conversion from PyTorch to TVM
2. DLPack conversion from TVM to PyTorch
3. Data integrity preservation during conversion
4. Functionality equivalence between DLPack and numpy fallback
"""

import pytest
import torch
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm.relax import BasePyModule
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T


class TestDLPackIntegration:
    def test_dlpack_pytorch_to_tvm_conversion(self):
        pytorch_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

        tvm_tensor = tvm_ffi.from_dlpack(pytorch_tensor)

        assert isinstance(tvm_tensor, tvm_tirx.runtime.Tensor)
        assert tvm_tensor.shape == pytorch_tensor.shape
        assert str(tvm_tensor.dtype) == str(pytorch_tensor.dtype).replace("torch.", "")

        tvm_numpy = tvm_tensor.numpy()
        pytorch_numpy = pytorch_tensor.numpy()
        tvm_tirx.testing.assert_allclose(tvm_numpy, pytorch_numpy, atol=1e-5)

    def test_dlpack_pytorch_to_tvm_conversion_gpu(self):
        if tvm_tirx.cuda().exist:

            def run_and_check():
                pytorch_tensor = torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device="cuda"
                )

                tvm_tensor = tvm_ffi.from_dlpack(pytorch_tensor)

                assert isinstance(tvm_tensor, tvm_tirx.runtime.Tensor)
                assert tvm_tensor.shape == pytorch_tensor.shape
                assert str(tvm_tensor.dtype) == str(pytorch_tensor.dtype).replace("torch.", "")
                assert str(tvm_tensor.device) == "cuda:0"

                # Move to CPU for numpy conversion
                tvm_numpy = tvm_tensor.numpy()
                pytorch_numpy = pytorch_tensor.cpu().numpy()
                tvm_tirx.testing.assert_allclose(tvm_numpy, pytorch_numpy, atol=1e-5)

            tvm_tirx.testing.run_with_gpu_lock(run_and_check)
        else:
            pytest.skip("CUDA not available")

    def test_dlpack_tvm_to_pytorch_conversion(self):
        import numpy as np

        data = np.array([1.0, 2.0, 3.0, 5.0], dtype="float32")
        tvm_tensor = tvm_tirx.runtime.tensor(data)

        pytorch_tensor = torch.from_dlpack(tvm_tensor)

        assert isinstance(pytorch_tensor, torch.Tensor)
        assert pytorch_tensor.shape == tvm_tensor.shape
        assert pytorch_tensor.dtype == torch.float32

        tvm_numpy = tvm_tensor.numpy()
        pytorch_numpy = pytorch_tensor.numpy()
        tvm_tirx.testing.assert_allclose(tvm_numpy, pytorch_numpy, atol=1e-5)

    def test_dlpack_tvm_to_pytorch_conversion_gpu(self):
        if tvm_tirx.cuda().exist:
            import numpy as np

            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")

            def run_and_check():
                tvm_tensor = tvm_tirx.runtime.tensor(data, device=tvm_tirx.cuda(0))

                pytorch_tensor = torch.from_dlpack(tvm_tensor)

                assert isinstance(pytorch_tensor, torch.Tensor)
                assert pytorch_tensor.shape == tvm_tensor.shape
                assert pytorch_tensor.dtype == torch.float32
                assert pytorch_tensor.device.type == "cuda"

                tvm_numpy = tvm_tensor.numpy()
                pytorch_numpy = pytorch_tensor.cpu().numpy()
                tvm_tirx.testing.assert_allclose(tvm_numpy, pytorch_numpy, atol=1e-5)

            tvm_tirx.testing.run_with_gpu_lock(run_and_check)
        else:
            pytest.skip("CUDA not available")

    def test_dlpack_roundtrip_conversion(self):
        """Test roundtrip conversion: PyTorch -> TVM -> PyTorch."""
        # Create PyTorch tensor
        original_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

        # Convert to TVM
        tvm_tensor = tvm_ffi.from_dlpack(original_tensor)

        # Convert back to PyTorch
        result_tensor = torch.from_dlpack(tvm_tensor)

        # Verify roundtrip integrity
        assert torch.allclose(original_tensor, result_tensor, atol=1e-5)
        assert original_tensor.dtype == result_tensor.dtype
        assert original_tensor.shape == result_tensor.shape

    def test_dlpack_different_data_types(self):
        """Test DLPack conversion with different data types."""
        test_types = [
            (torch.float32, "float32"),
            (torch.float64, "float64"),
            (torch.int32, "int32"),
            (torch.int64, "int64"),
        ]

        for torch_dtype, tvm_dtype in test_types:
            # Create PyTorch tensor
            pytorch_tensor = torch.tensor([1, 2, 3], dtype=torch_dtype)

            # Convert to TVM
            tvm_tensor = tvm_ffi.from_dlpack(pytorch_tensor)

            # Convert back to PyTorch
            result_tensor = torch.from_dlpack(tvm_tensor)

            # Verify conversion
            assert torch.allclose(pytorch_tensor, result_tensor, atol=1e-5)
            assert pytorch_tensor.dtype == result_tensor.dtype

    def test_dlpack_different_shapes(self):
        """Test DLPack conversion with different tensor shapes."""
        test_shapes = [
            (1,),
            (2, 3),
            (4, 5, 6),
            (1, 1, 1, 1),
        ]

        for shape in test_shapes:
            # Create PyTorch tensor
            pytorch_tensor = torch.randn(shape, dtype=torch.float32)

            # Convert to TVM
            tvm_tensor = tvm_ffi.from_dlpack(pytorch_tensor)

            # Convert back to PyTorch
            result_tensor = torch.from_dlpack(tvm_tensor)

            # Verify conversion
            assert torch.allclose(pytorch_tensor, result_tensor, atol=1e-5)
            assert pytorch_tensor.shape == result_tensor.shape

    def test_dlpack_functionality_verification(self):
        """Test that DLPack and numpy conversions produce identical results."""
        # Create large PyTorch tensor
        size = 1000000
        pytorch_tensor = torch.randn(size, dtype=torch.float32)

        # Test DLPack conversion
        tvm_tensor_dlpack = tvm_ffi.from_dlpack(pytorch_tensor)

        # Test numpy conversion
        numpy_array = pytorch_tensor.detach().cpu().numpy()
        tvm_tensor_numpy = tvm_tirx.runtime.tensor(numpy_array)

        # Verify both methods produce same result
        result_dlpack = torch.from_dlpack(tvm_tensor_dlpack)
        result_numpy = torch.from_numpy(tvm_tensor_numpy.numpy())
        assert torch.allclose(result_dlpack, result_numpy, atol=1e-5)

        # Verify data integrity
        assert torch.allclose(result_dlpack, pytorch_tensor, atol=1e-5)
        assert result_dlpack.shape == pytorch_tensor.shape
        assert result_dlpack.dtype == pytorch_tensor.dtype

    def test_dlpack_with_base_py_module(self):
        """Test DLPack conversion within BasePyModule context."""

        # Create a simple IRModule
        @Ts.function
        def identity_func(A: T.Tensor((3,), "float32"), B: T.Tensor((3,), "float32")):
            for i in T.grid(3):
                B[i] = A[i]

        @Ts.function
        def constant_func(B: T.Tensor((2,), "float32")):
            for i in T.grid(2):
                B[i] = T.float32(5.0)

        ir_mod = tvm_tirx.IRModule({"identity_func": identity_func, "constant_func": constant_func})
        device = tvm_tirx.cpu(0)
        py_mod = BasePyModule(ir_mod, device)

        # Create PyTorch tensor
        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        # Call TIR function (this will trigger DLPack conversion)
        result = py_mod.call_tir(identity_func, [input_tensor], R.Tensor((3,), "float32"))

        # Verify result
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, input_tensor, atol=1e-5)

        # Preserve the output-only call_tir path, where there are no input tensors
        # from which to infer or transfer the output.
        result = py_mod.call_tir(constant_func, [], R.Tensor((2,), "float32"))
        torch.testing.assert_close(result, torch.full((2,), 5.0))

    def test_dlpack_device_consistency(self):
        """Test DLPack conversion maintains device consistency."""
        # Test CPU tensor
        cpu_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        cpu_tvm = tvm_ffi.from_dlpack(cpu_tensor)
        cpu_result = torch.from_dlpack(cpu_tvm)

        assert cpu_result.device.type == "cpu"
        assert torch.allclose(cpu_tensor, cpu_result, atol=1e-5)

        # Note: GPU testing would require CUDA/OpenCL setup
        # This is a basic test that CPU works correctly

    def test_dlpack_memory_sharing(self):
        """Test that DLPack conversion shares memory when possible."""
        # Create PyTorch tensor
        pytorch_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

        # Convert to TVM
        tvm_tensor = tvm_ffi.from_dlpack(pytorch_tensor)

        # Modify the original tensor
        pytorch_tensor[0] = 10.0

        # Convert back to PyTorch
        result_tensor = torch.from_dlpack(tvm_tensor)

        # The result should reflect the modification (memory sharing)
        assert result_tensor[0] == 10.0
        assert torch.allclose(pytorch_tensor, result_tensor, atol=1e-5)

    def test_dlpack_batch_operations(self):
        """Test DLPack conversion with batch operations."""
        # Create batch of tensors
        batch_size = 10
        pytorch_tensors = [torch.randn(5, dtype=torch.float32) for _ in range(batch_size)]

        # Convert all to TVM
        tvm_tensors = [tvm_ffi.from_dlpack(t) for t in pytorch_tensors]

        # Convert all back to PyTorch
        result_tensors = [torch.from_dlpack(t) for t in tvm_tensors]

        # Verify all conversions
        for i in range(batch_size):
            assert torch.allclose(pytorch_tensors[i], result_tensors[i], atol=1e-5)

    def test_dlpack_edge_cases(self):
        """Test DLPack conversion with edge cases."""
        # Empty tensor
        empty_tensor = torch.tensor([], dtype=torch.float32)
        empty_tvm = tvm_ffi.from_dlpack(empty_tensor)
        empty_result = torch.from_dlpack(empty_tvm)

        assert empty_result.shape == empty_tensor.shape
        assert empty_result.dtype == empty_tensor.dtype

        # Single element tensor
        single_tensor = torch.tensor([42.0], dtype=torch.float32)
        single_tvm = tvm_ffi.from_dlpack(single_tensor)
        single_result = torch.from_dlpack(single_tvm)

        assert single_result.shape == single_tensor.shape
        assert single_result[0] == 42.0


if __name__ == "__main__":
    pytest.main([__file__])
