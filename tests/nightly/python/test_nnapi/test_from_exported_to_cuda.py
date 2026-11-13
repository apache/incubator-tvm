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
# ruff: noqa: E501, F401

import numpy as np
import pytest
import torch
from torch import nn
from torch.export import export
from torch.nn import Softmax, Upsample
from torch.nn import functional as F

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.testing import device_enabled


def assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev):
    """
    This util ensures that a torch module can successfully be exported to TVM
    using torch.export and that the resuling IR program gives the same result
    as PyTorch when ran on CUDA.
    """
    raw_data_for_tvm = raw_data.copy()  # In case the data is modified
    torch_data = torch.from_numpy(raw_data)
    example_args = (torch_data,)

    with torch.no_grad():
        exported_program = export(torch_module, example_args)
        mod_from_torch = from_exported_program(exported_program, keep_params_as_input=True)

    tvm_mod, tvm_params = relax.frontend.detach_params(mod_from_torch)

    relax_pipeline = relax.get_default_pipeline(tvm_tirx.target.Target.from_device(tvm_tirx.cuda()))
    ex = relax.build(tvm_mod, target=target, relax_pipeline=relax_pipeline)
    pytorch_out = torch_module(torch_data)

    def run_and_check():
        vm = relax.VirtualMachine(ex, dev)
        gpu_data = tvm_tirx.runtime.tensor(raw_data_for_tvm, dev)
        gpu_params = [tvm_tirx.runtime.tensor(p, dev) for p in tvm_params["main"]]
        gpu_out = vm["main"](gpu_data, *gpu_params)

        if isinstance(pytorch_out, tuple):
            for i in range(len(pytorch_out)):
                actual = gpu_out[i].numpy()
                desired = pytorch_out[i].detach().numpy()
                tvm_tirx.testing.assert_allclose(
                    actual=actual, desired=desired, rtol=1e-5, atol=1e-5
                )
        else:
            actual = gpu_out[0].numpy()
            desired = pytorch_out.detach().numpy()
            tvm_tirx.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-5)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_index_tensor():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class IndexModel0(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[torch.tensor([0])]

    torch_module = IndexModel0().eval()
    raw_data = np.random.rand(3, 3).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

    class IndexModel1(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[torch.tensor([[0]])]

    torch_module = IndexModel1().eval()
    raw_data = np.random.rand(2, 3).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

    class IndexTensorModel2(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[torch.tensor([0, 2])]

    torch_module = IndexTensorModel2().eval()
    raw_data = np.random.rand(3, 4).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

    class IndexTensorModel3(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[[[[0, 2], [1, 3]]]]

    torch_module = IndexTensorModel3().eval()
    raw_data = np.random.rand(5, 5, 5).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

    class IndexTensorModel4(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[[[1, 4]]]

    torch_module = IndexTensorModel4().eval()
    raw_data = np.random.rand(5, 5, 5).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

    class IndexTensorModel5(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[[[[1, 2, 4]]]]

    torch_module = IndexTensorModel5().eval()
    raw_data = np.random.rand(5, 5, 5).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

    class IndexTensorModel6(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[[[0, 1], [0, 1]]]

    torch_module = IndexTensorModel6().eval()
    raw_data = np.random.rand(5, 5, 5, 5).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

    class IndexTensorModel7(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[[[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 0]]]

    torch_module = IndexTensorModel7().eval()
    raw_data = np.random.rand(5, 5, 5, 5).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

    class IndexTensorModel8(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[[[[0, 1], [2, 3]], [[2, 3], [3, 4]], [[2, 4], [1, 2]], [[0, 4], [0, 3]]]]

    torch_module = IndexTensorModel8().eval()
    raw_data = np.random.rand(5, 5, 5, 5).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_full():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.full((2, 3), 3.141592)

    torch_module = FullModel().eval()
    raw_data = np.random.rand(3, 3).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_full_like():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class FullLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.fill_value = 7.0

        def forward(self, x):
            return torch.full_like(x, self.fill_value)

    torch_module = FullLike().eval()
    raw_data = np.random.rand(2, 3).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_ones():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.ones((2, 3))

    torch_module = FullModel().eval()
    raw_data = np.random.rand(1, 1).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_sort():
    target = "cuda"
    dev = tvm_tirx.cuda()

    raw_data = np.array([[4, 1, 13], [-30, 1, 3], [4, 0, 10]]).astype("float32")

    # Test values
    class SortModelValues(nn.Module):
        def forward(self, x):
            A, _ = torch.sort(x, dim=0, descending=True)
            B, _ = torch.sort(x, dim=1, descending=False)
            return A + B

    torch_module = SortModelValues().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

    # Test indices
    class SortModelIndices(nn.Module):
        def forward(self, x):
            _, A = torch.sort(x, dim=0, descending=True)
            _, B = torch.sort(x, dim=1, descending=False)
            return A + B

    torch_module = SortModelIndices().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_tensor_clamp():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class ClampBothTensor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("min_val", torch.tensor(-1.0))
            self.register_buffer("max_val", torch.tensor(1.0))

        def forward(self, x):
            return x.clamp(min=self.min_val, max=self.max_val)

    class ClampBothInt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.min_val = -1
            self.max_val = 1

        def forward(self, x):
            return x.clamp(min=self.min_val, max=self.max_val)

    class ClampMinOnlyTensor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("min_val", torch.tensor(0.0))

        def forward(self, x):
            return x.clamp(min=self.min_val)

    class ClampMinOnlyInt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.min_val = 0

        def forward(self, x):
            return x.clamp(min=self.min_val)

    class ClampMaxOnlyTensor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("max_val", torch.tensor(0.5))

        def forward(self, x):
            return x.clamp(max=self.max_val)

    class ClampMaxOnlyInt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.max_val = 0.5

        def forward(self, x):
            return x.clamp(max=self.max_val)

    class ClampDifferentValues(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.min_val = -2
            self.max_val = 2

        def forward(self, x):
            return x.clamp(min=self.min_val, max=self.max_val)

    # Create random data with values outside our clamp ranges
    raw_data = np.random.uniform(-3.0, 3.0, (2, 3, 4, 5)).astype(np.float32)

    torch_module0 = ClampBothTensor().eval()
    torch_module1 = ClampBothInt().eval()
    torch_module2 = ClampMinOnlyTensor().eval()
    torch_module3 = ClampMinOnlyInt().eval()
    torch_module4 = ClampMaxOnlyTensor().eval()
    torch_module5 = ClampMaxOnlyInt().eval()
    torch_module6 = ClampDifferentValues().eval()

    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module1, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module2, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module3, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module4, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module5, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module6, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_tensor_expand_as():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class ExpandAs0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.template = torch.ones((1, 1, 1, 1))

        def forward(self, x):
            return self.template.expand_as(x)

    class ExpandAs1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.template = torch.ones((2, 1, 4, 1))

        def forward(self, x):
            return self.template.expand_as(x)

    class ExpandAs2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.template = torch.ones((2, 1, 1, 10))

        def forward(self, x):
            return self.template.expand_as(x)

    class ExpandAs3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.template = torch.ones((2, 3, 1, 1))

        def forward(self, x):
            return self.template.expand_as(x)

    raw_data = np.random.randn(2, 3, 4, 10).astype(np.float32)

    torch_module0 = ExpandAs0().eval()
    torch_module1 = ExpandAs1().eval()
    torch_module2 = ExpandAs2().eval()
    torch_module3 = ExpandAs3().eval()

    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module1, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module2, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module3, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_copy_():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class CopyTester(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.register_buffer("buffer", torch.zeros(size))

        def forward(self, x):
            self.buffer.copy_(x)

            return x * 3 + self.buffer * 5

    size = (2, 2)
    raw_data = np.random.rand(*size).astype(np.float32)
    torch_module = CopyTester(size).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_upsample_with_size():
    """
    The Upsample module can be used with the size arugment or the scale
    factor argument but not both. This tests the former.
    """
    target = "cuda"
    dev = tvm_tirx.cuda()

    batch_size = 1
    channels = 3
    height, width = 8, 8

    torch_module = Upsample(size=(64, 64), mode="nearest", recompute_scale_factor=None)

    raw_data = np.random.rand(batch_size, channels, height, width).astype("float32")

    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_detach_no_change():
    target = "cuda"
    dev = tvm_tirx.cuda()

    # In TVM, detach() is just identity
    class DetachTester(nn.Module):
        def forward(self, x):
            detached = x.detach()
            return detached

    raw_data = np.ones((2, 2)).astype(np.float32)
    torch_module = DetachTester().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_upsample_with_scale_factor():
    """
    The Upsample module can be used with the size arugment or the scale
    factor argument but not both. This tests the latter.
    """
    target = "cuda"
    dev = tvm_tirx.cuda()

    batch_size = 2
    channels = 3
    height, width = 32, 32

    torch_module = Upsample(
        size=None, scale_factor=7, mode="nearest", align_corners=None, recompute_scale_factor=True
    )

    raw_data = np.random.rand(batch_size, channels, height, width).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_linalg_vector_norm():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class VectorNorm0(torch.nn.Module):
        def forward(self, x):
            return torch.linalg.vector_norm(x, ord=1, dim=-1)

    class VectorNorm1(torch.nn.Module):
        def forward(self, x):
            return torch.linalg.vector_norm(x, ord=2, dim=2)

    class VectorNorm2(torch.nn.Module):
        def forward(self, x):
            return torch.linalg.vector_norm(x, ord=1, dim=-1)

    class VectorNorm3(torch.nn.Module):
        def forward(self, x):
            return torch.linalg.vector_norm(x, ord=2, dim=2)

    raw_data = np.random.randn(2, 3, 4, 10).astype(np.float32)

    torch_module0 = VectorNorm0().eval()
    torch_module1 = VectorNorm1().eval()
    torch_module2 = VectorNorm2().eval()
    torch_module3 = VectorNorm3().eval()

    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module1, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module2, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module3, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_batch_norm_prog():
    target = "cuda"
    dev = tvm_tirx.cuda()

    # Default args, in a pytorch program (to ensure output is in proper type and format)
    raw_data = np.random.randn(2, 3, 2, 2).astype(np.float32)

    class BatchNormWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(3)

        def forward(self, x):
            x = self.bn(x)
            x = x + 1
            return x

    torch_module = BatchNormWrapper().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_split_size():
    target = "cuda"
    dev = tvm_tirx.cuda()

    # Test split using the split_size argument such that it is not a divisor
    # of the dimension to split (the last tensor will be smaller)
    batch = 2
    channels = 7
    height, width = 2, 2
    split_size = 3  # last tensor will have just 1 element
    dim = 1  # split across channels
    raw_data = np.random.rand(batch, channels, height, width).astype("float32")

    class SplitModelSplitSize(nn.Module):
        def __init__(self, split_size, dim):
            super().__init__()
            self.split_size = split_size
            self.dim = dim

        def forward(self, x):
            return torch.split(x, split_size_or_sections=self.split_size, dim=self.dim)

    torch_module = SplitModelSplitSize(split_size=split_size, dim=dim).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_split_sections_list():
    target = "cuda"
    dev = tvm_tirx.cuda()

    # Test split using a list of section sizes
    batch = 3
    channels = 2
    height = 10
    width = 5
    sections = [3, 2, 5]
    dim = 2  # split across height
    raw_data = np.random.rand(batch, channels, height, width).astype("float32")

    class SplitModelSectionsList(nn.Module):
        def __init__(self, split_size, dim):
            super().__init__()
            self.split_size = split_size
            self.dim = dim

        def forward(self, x):
            return torch.split(x, split_size_or_sections=self.split_size, dim=self.dim)

    torch_module = SplitModelSectionsList(split_size=sections, dim=dim).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_batch_norm0():
    target = "cuda"
    dev = tvm_tirx.cuda()

    # Eval, no momentum, no affine, no running stats
    raw_data = np.random.randn(8, 3, 4, 4).astype(np.float32)
    torch_module = nn.BatchNorm2d(
        3, eps=1e-02, momentum=0.0, affine=False, track_running_stats=False, device=None, dtype=None
    ).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_batch_norm1():
    target = "cuda"
    dev = tvm_tirx.cuda()

    # Eval, with momentum, no affine, with running stats
    raw_data = np.random.randn(1, 4, 2, 2).astype(np.float32)
    torch_module = nn.BatchNorm2d(
        4, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True, device=None, dtype=None
    ).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_batch_norm2():
    target = "cuda"
    dev = tvm_tirx.cuda()

    # Eval, with momentum, affine, no running stats
    raw_data = np.random.randn(3, 4, 2, 2).astype(np.float32)
    torch_module = nn.BatchNorm2d(
        4, eps=1e-05, momentum=0.2, affine=True, track_running_stats=False
    ).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_batch_norm3():
    target = "cuda"
    dev = tvm_tirx.cuda()

    # Eval, no momentum, affine, with running stats
    raw_data = np.random.randn(1, 2, 2, 2).astype(np.float32)
    torch_module = nn.BatchNorm2d(
        2, eps=1e-05, momentum=0.0, affine=True, track_running_stats=True
    ).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_chunk_even():
    target = "cuda"
    dev = tvm_tirx.cuda()

    # Chunks is a divisor of the dimension size
    batch = 6
    channels = 2
    height = 3
    width = 4
    chunks = 3
    dim = 0
    raw_data = np.random.rand(batch, channels, height, width).astype("float32")

    class ChunkModel(nn.Module):
        def __init__(self, chunks, dim):
            super().__init__()
            self.chunks = chunks
            self.dim = dim

        def forward(self, x):
            return x.chunk(self.chunks, dim=self.dim)

    torch_module = ChunkModel(chunks=chunks, dim=dim).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_chunk_uneven():
    target = "cuda"
    dev = tvm_tirx.cuda()

    # Chunks is not a divisor of the dimension size
    batch = 2
    channels = 5
    height = 4
    width = 5
    chunks = 2
    dim = 1
    raw_data = np.random.rand(batch, channels, height, width).astype("float32")

    class ChunkModel(nn.Module):
        def __init__(self, chunks, dim):
            super().__init__()
            self.chunks = chunks
            self.dim = dim

        def forward(self, x):
            return x.chunk(self.chunks, dim=self.dim)

    torch_module = ChunkModel(chunks=chunks, dim=dim).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_chunk_too_many():
    target = "cuda"
    dev = tvm_tirx.cuda()

    # If user asks for more chunks than the size of the dim, pytorch simply splits in sections of size 1
    batch = 1
    channels = 3
    height = 2
    width = 2
    chunks = 99
    dim = 1
    raw_data = np.random.rand(batch, channels, height, width).astype("float32")

    class ChunkModel(nn.Module):
        def __init__(self, chunks, dim):
            super().__init__()
            self.chunks = chunks
            self.dim = dim

        def forward(self, x):
            return x.chunk(self.chunks, dim=self.dim)

    torch_module = ChunkModel(chunks=chunks, dim=dim).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_arange():
    target = "cuda"
    dev = tvm_tirx.cuda()

    # arange.default
    raw_data = np.array([0, 0, 0, 0, 0])

    class ArangeDefaultModel(nn.Module):
        def forward(self, x):
            return x + torch.arange(5)

    torch_module = ArangeDefaultModel().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

    # arange.start
    raw_data = np.array([0, 0, 0])

    class ArangeStartModel(nn.Module):
        def forward(self, x):
            return x + torch.arange(1, 4)

    torch_module = ArangeStartModel().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

    # arange.start_step
    raw_data = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    class ArangeStartStopModel(nn.Module):
        def forward(self, x):
            return x + torch.arange(1, 2.5, 0.5, dtype=torch.float32)

    torch_module = ArangeStartStopModel().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_index_select():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class IndexSelectModel(nn.Module):
        def forward(self, x):
            indices = torch.tensor([0, 2])
            return torch.index_select(x, 0, indices)

    raw_data = np.random.rand(3, 4).astype("float32")
    torch_module = IndexSelectModel().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_stack():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class StackModel(nn.Module):
        def forward(self, x):
            val1 = x[1, 4]
            val2 = x[3, 2]
            val3 = x[5, 6]
            z = torch.stack([val1, val2, val3])
            return z

    torch_module = StackModel().eval()
    raw_data = np.random.rand(10, 10, 10).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_sum():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class SumModel(nn.Module):
        def forward(self, x):
            new_vec = x[1, 4]
            return new_vec.sum()

    torch_module = SumModel().eval()
    raw_data = np.random.rand(10, 10, 10).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_mul():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class MulModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.y = torch.tensor(np.random.rand(2, 3).astype("float32"))

        def forward(self, x):
            return x.mul(self.y)

    torch_module = MulModule().eval()
    raw_data = np.random.rand(2, 3).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_concat():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class ConcatFour(nn.Module):
        def __init__(self, dim=0):
            super().__init__()
            self.dim = dim
            self.x2 = torch.randn(2, 3)
            self.x3 = torch.randn(2, 3)
            self.x4 = torch.randn(2, 3)

        def forward(self, x):
            return torch.cat((x, self.x2, self.x3, self.x4), dim=self.dim)

    torch_module = ConcatFour().eval()
    raw_data = np.random.rand(2, 3).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_leakyrelu_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class LeakyReLUModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.LeakyReLU(negative_slope=0.1)

        def forward(self, x):
            return self.act(x)

    raw_data = np.random.randn(2, 3).astype(np.float32)
    torch_module = LeakyReLUModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_log_softmax_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class LogSoftmaxModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.logsoftmax = nn.LogSoftmax(dim=1)

        def forward(self, x):
            return self.logsoftmax(x)

    raw_data = np.random.randn(4, 5).astype(np.float32)
    torch_module = LogSoftmaxModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_softmax_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class SoftmaxModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            return self.softmax(x)

    raw_data = np.random.randn(4, 5).astype(np.float32)
    torch_module = SoftmaxModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_adaptive_avg_pool2d_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class AdaptiveAvgPool2dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            return self.pool(x)

    raw_data = np.random.randn(2, 3, 8, 8).astype(np.float32)
    torch_module = AdaptiveAvgPool2dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_avg_pool2d_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class AvgPool2dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=2)

        def forward(self, x):
            return self.pool(x)

    raw_data = np.random.randn(2, 3, 8, 8).astype(np.float32)
    torch_module = AvgPool2dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_conv1d_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class Conv1dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(in_channels=3, out_channels=4, kernel_size=3)

        def forward(self, x):
            return self.conv(x)

    raw_data = np.random.randn(2, 3, 10).astype(np.float32)
    torch_module = Conv1dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_conv2d_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class Conv2dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)

        def forward(self, x):
            return self.conv(x)

    raw_data = np.random.randn(2, 3, 10, 10).astype(np.float32)
    torch_module = Conv2dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_conv3d_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class Conv3dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(in_channels=2, out_channels=3, kernel_size=3)

        def forward(self, x):
            return self.conv(x)

    raw_data = np.random.randn(1, 2, 8, 8, 8).astype(np.float32)
    torch_module = Conv3dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_group_norm_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class GroupNormModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.gn = nn.GroupNorm(num_groups=1, num_channels=4)

        def forward(self, x):
            return self.gn(x)

    raw_data = np.random.randn(2, 4, 8, 8).astype(np.float32)
    torch_module = GroupNormModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_layer_norm_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class LayerNormModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(normalized_shape=8)

        def forward(self, x):
            return self.ln(x)

    raw_data = np.random.randn(2, 4, 8).astype(np.float32)
    torch_module = LayerNormModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_linear_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class LinearModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    raw_data = np.random.randn(4, 10).astype(np.float32)
    torch_module = LinearModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_max_pool2d_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class MaxPool2dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=2)

        def forward(self, x):
            return self.pool(x)

    raw_data = np.random.randn(2, 3, 8, 8).astype(np.float32)
    torch_module = MaxPool2dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_embedding_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class EmbeddingModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(num_embeddings=10, embedding_dim=3)

        def forward(self, x):
            return self.embed(x)

    raw_data = np.random.randint(0, 10, (2, 4)).astype(np.int64)
    torch_module = EmbeddingModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_flatten_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class FlattenModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()

        def forward(self, x):
            return self.flatten(x)

    raw_data = np.random.randn(2, 3, 4, 5).astype(np.float32)
    torch_module = FlattenModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_numel():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class NumelModule(nn.Module):
        def forward(self, x):
            return torch.tensor(x.numel())

    raw_data = np.random.randn(2, 3, 4).astype(np.float32)
    torch_module = NumelModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_size():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class SizeModule(nn.Module):
        def forward(self, x):
            return torch.tensor(x.size(0))

    raw_data = np.random.randn(5, 4).astype(np.float32)
    torch_module = SizeModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_tensor():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class TensorModule(nn.Module):
        def forward(self, x):
            return torch.tensor([1, 2, 3])

    raw_data = np.zeros((1,)).astype(np.float32)
    torch_module = TensorModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_type():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class TypeModule(nn.Module):
        def forward(self, x):
            return x.type(torch.float16)

    raw_data = np.random.randn(2, 3).astype(np.float32)
    torch_module = TypeModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_float():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class FloatModule(nn.Module):
        def forward(self, x):
            return x.float()

    raw_data = np.random.randn(2, 3).astype(np.float32)
    torch_module = FloatModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_half():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class HalfModule(nn.Module):
        def forward(self, x):
            return x.half()

    raw_data = np.random.randn(2, 3).astype(np.float32)
    torch_module = HalfModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_getattr():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class GetAttrModule(nn.Module):
        def forward(self, x):
            # Use getattr to call the ndimension method.
            return torch.tensor(getattr(x, "ndimension")())

    raw_data = np.random.randn(2, 3, 4).astype(np.float32)
    torch_module = GetAttrModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_sym_size_int():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class SymSizeIntModule(nn.Module):
        def forward(self, x):
            return torch.tensor(x.shape[1])

    raw_data = np.random.randn(2, 3, 4).astype(np.float32)
    torch_module = SymSizeIntModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_interpolate():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class InterpolateModule(nn.Module):
        def forward(self, x):
            # Upsample to a fixed size.
            return F.interpolate(x, size=(16, 16), mode="nearest")

    raw_data = np.random.randn(2, 3, 8, 8).astype(np.float32)
    torch_module = InterpolateModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_cross_entropy_module():
    target = "cuda"
    dev = tvm_tirx.cuda()

    class CrossEntropyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.criterion = nn.CrossEntropyLoss()
            self.target = torch.tensor([0, 1, 2, 1])

        def forward(self, x):
            return self.criterion(x, self.target)

    raw_data = np.random.randn(4, 3).astype(np.float32)
    torch_module = CrossEntropyModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


if __name__ == "__main__":
    tvm_tirx.testing.main()
