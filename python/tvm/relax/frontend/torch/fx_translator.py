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

# pylint: disable=invalid-name, inconsistent-return-statements, unidiomatic-typecheck
# pylint: disable=import-outside-toplevel
"""PyTorch FX frontend of Relax."""

from collections.abc import Callable
from functools import partial, reduce

import tvm_tirx
from tvm import relax

from .base_fx_graph_translator import BaseFXGraphImporter


class TorchFXImporter(BaseFXGraphImporter):
    """An importer from PyTorch FX to Relax."""

    import torch  # type: ignore
    from torch import fx

    def __init__(self, default_image_layout: str = "NCHW") -> None:
        import torch  # type: ignore

        super().__init__()
        self.named_modules: dict[str, torch.Module] = None
        self.default_image_layout = default_image_layout

    ########## Utilities ##########

    def _fetch_attr(self, model, target: str):
        import torch  # type: ignore

        target_atoms = target.split(".")
        attr_itr = model
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced non existing target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)
        if isinstance(attr_itr, torch.Tensor):
            # Its possible for the resulting tensor to be a parameter.
            # If so, return the parameter instead.
            if attr_itr in self.params:
                return self.params[attr_itr]
            return self._convert_torch_tensor_to_relax(attr_itr)
        return attr_itr

    ########## Unary Ops ##########

    def _reciprocal(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(relax.op.divide(relax.const(1.0, x.ty.dtype), x))

    def _leakyrelu_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        alpha = module.negative_slope
        return self.block_builder.emit(relax.op.nn.leakyrelu(x, alpha))

    def _softplus_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        beta = module.beta
        threshold = module.threshold
        return self.block_builder.emit(relax.op.nn.softplus(x, beta, threshold))

    def _log2(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(
            relax.op.divide(relax.op.log(x), relax.const(0.6931471805599453, x.ty.dtype))
        )

    def _log10(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(
            relax.op.divide(relax.op.log(x), relax.const(2.302585092994046, x.ty.dtype))
        )

    def _log1p(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        one = relax.const(1, x.ty.dtype)
        return self.block_builder.emit(relax.op.log(relax.op.add(x, one)))

    def _sqrt(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dtype = x.ty.dtype

        # Check if input is integer type and convert to float32 if needed
        if dtype in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]:
            x = self.block_builder.emit(relax.op.astype(x, "float32"))

        return self.block_builder.emit(relax.op.sqrt(x))

    def _rsqrt(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dtype = x.ty.dtype

        # Check if input is integer type and convert to float32 if needed
        if dtype in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]:
            x = self.block_builder.emit(relax.op.astype(x, "float32"))

        return self.block_builder.emit(relax.op.rsqrt(x))

    def _log_softmax_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        dim = module.dim
        assert dim is not None
        return self.block_builder.emit(relax.op.nn.log_softmax(x, dim))

    def _prelu_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        alpha_tensor = module.weight.numpy()
        alpha = relax.const(alpha_tensor, dtype="float32")
        axis = 0 if len(x.ty.shape) == 1 else 1  # Extract Channel size
        return self.block_builder.emit(relax.op.nn.prelu(x, alpha, axis))

    def _softmax_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        dim = module.dim
        assert dim is not None
        return self.block_builder.emit(relax.op.nn.softmax(x, dim))

    def _inplace_tril_triu(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            x = self.env[node.args[0]]
            k = node.args[1] if len(node.args) > 1 else 0
            assert isinstance(k, int)

            mutated = self.block_builder.emit(op(x, k))
            self.env[node.args[0]] = mutated
            return mutated

        return convert

    ########## Binary Ops ##############

    def _binary_op_inplace(self, relax_op: Callable, intrinsic_op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            def promote_binary_op_args(lhs, rhs):
                if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
                    return lhs, rhs
                elif isinstance(lhs, relax.Expr):
                    assert isinstance(lhs.ty, relax.TensorType)
                    return lhs, relax.const(rhs, lhs.ty.dtype)
                elif isinstance(rhs, relax.Expr):
                    assert isinstance(rhs.ty, relax.TensorType)
                    return relax.const(lhs, rhs.ty.dtype), rhs
                else:
                    assert False

            def call_binary_op(op, lhs, rhs):
                lhs, rhs = promote_binary_op_args(lhs, rhs)
                return self.block_builder.emit(op(lhs, rhs))

            lhs, rhs = self.retrieve_args(node)

            if (isinstance(lhs, tvm_tirx.ir.Var) and isinstance(lhs.ty, relax.TensorType)) or (
                isinstance(rhs, tvm_tirx.ir.Var) and isinstance(rhs.ty, relax.TensorType)
            ):
                output = call_binary_op(relax_op, lhs, rhs)
                self.env[node.args[0]] = output
                return output

            elif isinstance(lhs, tvm_tirx.ir.GenericConst):
                output = call_binary_op(relax_op, lhs, relax.const(rhs, dtype=lhs.ty.dtype))
                self.env[node.args[0]] = output
                return output

            elif isinstance(rhs, tvm_tirx.ir.GenericConst):
                output = call_binary_op(relax_op, relax.const(lhs, dtype=rhs.ty.dtype), rhs)
                self.env[node.args[0]] = output
                return output

            output = intrinsic_op(lhs, rhs)
            self.env[node.args[0]] = output
            return output

        return convert

    ########## Neural Network ##########

    def _adaptive_avg_pool1d_module(self, node: fx.Node) -> relax.Var:
        module = self.named_modules[node.target]
        x = self.env[node.args[0]]
        output_size = module.output_size
        # Expand to 3D by adding batch dim if input is 2D
        x_ndim = x.ty.ndim
        if x_ndim == 2:
            x = relax.op.expand_dims(x, axis=0)
        result = self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool1d(x, output_size, layout="NCW")  # (N, C, L)
        )
        # Remove added batch dim from result
        if x_ndim == 2:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _adaptive_avg_pool2d_module(self, node: fx.Node) -> relax.Var:
        module = self.named_modules[node.target]
        x = self.env[node.args[0]]
        output_size = module.output_size
        # Expand to 4D by adding batch dim if input is 3D
        x_ndim = x.ty.ndim
        if x_ndim == 3:
            x = relax.op.expand_dims(x, axis=0)
        result = self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool2d(x, output_size, layout="NCHW")
        )
        # Remove added batch dim from result
        if x_ndim == 3:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _adaptive_avg_pool3d_module(self, node: fx.Node) -> relax.Var:
        module = self.named_modules[node.target]
        x = self.env[node.args[0]]
        output_size = module.output_size
        # Expand to 5D by adding batch dim if input is 4D
        x_ndim = x.ty.ndim
        if x_ndim == 4:
            x = relax.op.expand_dims(x, axis=0)
        result = self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool3d(x, output_size, layout="NCDHW")  # (N, C, D, H, W)
        )
        # Remove added batch dim from result
        if x_ndim == 4:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _avg_pool1d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        ceil_mode = module.ceil_mode
        return self._avg_pool1d_impl(x, kernel_size, stride, padding, ceil_mode)

    def _avg_pool2d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        ceil_mode = module.ceil_mode
        return self._avg_pool2d_impl(x, kernel_size, stride, padding, ceil_mode)

    def _avg_pool3d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        ceil_mode = module.ceil_mode
        return self._avg_pool3d_impl(x, kernel_size, stride, padding, ceil_mode)

    def _batch_norm_2d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params[module.bias]
        running_mean = self._convert_torch_tensor_to_relax(module.running_mean)
        running_var = self._convert_torch_tensor_to_relax(module.running_var)
        eps = module.eps

        res_tuple = self.block_builder.emit(
            relax.op.nn.batch_norm(
                x,
                weight,
                bias,
                running_mean,
                running_var,
                axis=1,
                epsilon=eps,
            )
        )

        return self.block_builder.emit(relax.TupleGetItem(res_tuple, 0))

    def _instance_norm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]

        if module.affine:
            weight = self.params[module.weight]
            bias = self.params[module.bias]
        else:
            import numpy as np

            dtype = x.ty.dtype
            channel = int(self.shape_of(x)[1])
            weight = relax.const(np.ones(channel), dtype=dtype)
            bias = relax.const(np.zeros(channel), dtype=dtype)

        eps = module.eps
        channel_axis = 1
        dim = len(self.shape_of(x))

        return self.block_builder.emit(
            relax.op.nn.instance_norm(
                x,
                weight,
                bias,
                channel_axis=channel_axis,
                axes=list(range(2, dim)),
                epsilon=eps,
            )
        )

    def _conv_transpose1d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params.get(module.bias, None)

        return self._conv_transpose1d_impl(
            x,
            weight,
            bias=bias,
            strides=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            output_padding=module.output_padding,
        )

    def _conv_transpose2d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params.get(module.bias, None)

        return self._conv_transpose2d_impl(
            x,
            weight,
            bias=bias,
            strides=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            output_padding=module.output_padding,
        )

    def _conv1d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params.get(module.bias, None)

        return self._conv1d_impl(
            x,
            weight,
            bias=bias,
            strides=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    def _conv2d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params.get(module.bias, None)

        return self._conv2d_impl(
            x,
            weight,
            bias=bias,
            strides=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    def _conv3d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params.get(module.bias, None)

        return self._conv3d_impl(
            x,
            weight,
            bias=bias,
            strides=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    def _cross_entropy(self, node: fx.Node) -> relax.Expr:
        preds = self.env[node.args[0]]
        targets = self.env[node.args[1]]
        weights = self.env.get(node.kwargs["weight"], None)
        reduction = node.kwargs["reduction"]
        ignore_index = node.kwargs["ignore_index"]
        return self._cross_entropy_loss(preds, targets, weights, reduction, ignore_index)

    def _cross_entropy_module(self, node: fx.Node) -> relax.Expr:
        preds = self.env[node.args[0]]
        targets = self.env[node.args[1]]
        module = self.named_modules[node.target]

        weights = module.weight
        if weights is not None:
            if weights in self.params:
                weights = self.params[weights]
            else:
                weights = relax.const(weights.numpy(), preds.ty.dtype)

        reduction = module.reduction
        ignore_index = module.ignore_index

        return self._cross_entropy_loss(
            preds,
            targets,
            weights,
            reduction,
            ignore_index,
        )

    def _embedding_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        return self._embedding_impl(x, weight)

    def _group_norm_module(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        num_groups = module.num_groups
        if module.affine:
            gamma = self.params[module.weight]
            beta = self.params[module.bias]
        else:
            gamma = relax.const(torch.ones_like(module.num_channels), x.ty.dtype)
            beta = relax.const(torch.zeros_like(module.num_channels), x.ty.dtype)
        eps = module.eps

        dim = len(self.shape_of(x))
        return self.block_builder.emit(
            relax.op.nn.group_norm(
                x,
                gamma,
                beta,
                num_groups=num_groups,
                channel_axis=1,
                axes=list(range(2, dim)),
                epsilon=eps,
            )
        )

    def _interpolate(self, node: fx.Node) -> relax.Var:
        # torch.nn.functional.interpolate(
        #   input, size=None, scale_factor=None, mode='nearest', align_corners=None,
        #   recompute_scale_factor=None, antialias=False)
        data = self.env[node.args[0]]
        size = (
            node.args[1]
            if len(node.args) > 1
            else (node.kwargs["size"] if "size" in node.kwargs else None)
        )
        scale_factor = (
            node.args[2]
            if len(node.args) > 2
            else (node.kwargs["scale_factor"] if "scale_factor" in node.kwargs else None)
        )
        method = (
            node.args[3]
            if len(node.args) > 3
            else (node.kwargs["mode"] if "mode" in node.kwargs else "nearest")
        )
        align_corners = (
            node.args[4]
            if len(node.args) > 4
            else (node.kwargs["align_corners"] if "align_corners" in node.kwargs else None)
        )
        recompute_scale_factor = (
            node.args[5]
            if len(node.args) > 5
            else (
                node.kwargs["recompute_scale_factor"]
                if "recompute_scale_factor" in node.kwargs
                else None
            )
        )
        antialias = (
            node.args[6]
            if len(node.args) > 6
            else (node.kwargs["antialias"] if "antialias" in node.kwargs else False)
        )

        assert recompute_scale_factor is None
        assert antialias is False

        if size is None:
            shape = self.shape_of(data)
            assert isinstance(shape, relax.ShapeExpr)
            # Determine spatial dimension indices based on layout
            # NCHW: spatial dims are [2, 3, ...] (skip batch and channel)
            # NHWC: spatial dims are [1, 2, ...] (skip batch, before channel)
            if self.default_image_layout in ("NHWC", "NDHWC"):
                spatial_start = 1
                spatial_end = len(shape) - 1
            else:  # NCHW or other layouts
                spatial_start = 2
                spatial_end = len(shape)

            if isinstance(scale_factor, tuple):
                assert len(scale_factor) == spatial_end - spatial_start
                size = tuple(
                    int(shape[i].value * scale_factor[i - spatial_start])
                    for i in range(spatial_start, spatial_end)
                )
            else:
                size = tuple(
                    int(shape[i].value * scale_factor) for i in range(spatial_start, spatial_end)
                )

        if method.startswith("nearest"):
            method = "nearest_neighbor"
        elif method.startswith("bi"):
            method = method[2:]
        elif method.startswith("tri"):
            method = method[3:]

        if method == "nearest_neighbor":
            coord_trans = "asymmetric"
        elif align_corners is True:
            coord_trans = "align_corners"
        else:
            coord_trans = "half_pixel"

        if data.ty.ndim == 5:
            if self.default_image_layout == "NDHWC":
                layout_3d = "NDHWC"
            else:
                layout_3d = "NCDHW"

            return self.block_builder.emit(
                relax.op.image.resize3d(
                    data,
                    size,
                    layout=layout_3d,
                    method=method,
                    coordinate_transformation_mode=coord_trans,
                )
            )
        else:
            return self.block_builder.emit(
                relax.op.image.resize2d(
                    data,
                    size,
                    layout=self.default_image_layout,
                    method=method,
                    coordinate_transformation_mode=coord_trans,
                )
            )

    def _linear_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params.get(module.bias, None)
        return self.block_builder.emit(relax.op.linear(x, weight, bias, "float32"))

    def _max_pool1d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        dilation = module.dilation
        ceil_mode = module.ceil_mode

        return self._max_pool1d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)

    def _max_pool2d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        dilation = module.dilation
        ceil_mode = module.ceil_mode

        return self._max_pool2d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)

    def _max_pool3d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        dilation = module.dilation
        ceil_mode = module.ceil_mode

        return self._max_pool3d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)

    def _pixel_shuffle_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        upscale_factor = module.upscale_factor

        return self.block_builder.emit(relax.op.nn.pixel_shuffle(x, upscale_factor))

    ########## Linear Interpolation ##########

    def _lerp(self, node: fx.Node) -> relax.Var:
        start = self.env[node.args[0]]
        end = self.env[node.args[1]]
        weight = self.env[node.args[2]]
        return self.block_builder.emit(
            relax.op.add(start, relax.op.multiply(weight, relax.op.subtract(end, start)))
        )

    ########## Manipulation ##########

    def _chunk(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        chunks = node.args[1]
        dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        return self.block_builder.emit(relax.op.split(x, chunks, dim))

    def _flatten_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        start_dim = module.start_dim
        end_dim = module.end_dim
        return self._flatten_impl(x, start_dim, end_dim)

    def _narrow(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1]
        start = node.args[2]
        length = node.args[3]
        return self.block_builder.emit(relax.op.strided_slice(x, [dim], [start], [length]))

    def _numel(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        shape = self.shape_of(x)
        return relax.const(reduce(lambda x, y: x * y, [s.value for s in shape]), "int32")

    def _select(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1]
        index = relax.const(node.args[2], "int64")
        return self.block_builder.emit(relax.op.take(x, index, dim))

    def _size(self, node: fx.Node) -> relax.Expr:
        x = self.env[node.args[0]]
        shape = self.shape_of(x)
        if len(node.args) == 1:
            assert isinstance(shape, relax.ShapeExpr)
            return shape
        assert len(node.args) == 2
        idx = node.args[1]
        return self.shape_of(x)[idx].value

    ########## Creation ##########

    def _inplace_copy(self, node: fx.Node) -> relax.Var:
        dest = self.env[node.args[0]]
        src = self.env[node.args[1]]

        if src.ty.dtype != dest.ty.dtype:
            src = self.block_builder.emit(relax.op.astype(src, dest.ty.dtype.dtype))

        dest_shape = self.shape_of(dest)
        src_shape = self.shape_of(src)
        if dest_shape != src_shape:
            src = self.block_builder.emit(relax.op.broadcast_to(src, dest_shape))

        self.env[node.args[0]] = src
        return src

    def _masked_scatter(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        mask = self.env[node.args[1]]
        source = self.env[node.args[2]]
        ndim = len(mask.ty.shape)
        if ndim == 1:
            index = self.block_builder.emit(relax.op.cumsum(mask, 0, dtype="int32"))
            index = self.block_builder.emit(relax.op.subtract(index, relax.const(1, "int32")))
            gathered_source = self.block_builder.emit(relax.op.take(source, index, axis=0))
        else:
            f_mask = self.block_builder.emit(relax.op.reshape(mask, [-1]))
            index = self.block_builder.emit(relax.op.cumsum(f_mask, 0, dtype="int32"))
            index = self.block_builder.emit(relax.op.subtract(index, relax.const(1, "int32")))
            source_shape = [-1] + [s for idx, s in enumerate(source.ty.shape) if idx >= ndim]
            f_source = self.block_builder.emit(relax.op.reshape(source, source_shape))
            gathered_source = self.block_builder.emit(relax.op.take(f_source, index, axis=0))
            gathered_source = self.block_builder.emit(relax.op.reshape(gathered_source, x.ty.shape))
        if ndim != len(x.ty.shape):
            mask = self.block_builder.emit(relax.op.broadcast_to(mask, x.ty.shape))
        return self.block_builder.emit(relax.op.where(mask, gathered_source, x))

    def _one_hot(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        num_classes = node.args[1] if len(node.args) > 1 else node.kwargs.get("num_classes")
        if num_classes is None:
            raise ValueError("num_classes not found in node.args or node.kwargs")
        # torch only rejects a non-positive num_classes when the model runs, and neither
        # fx tracing nor export runs it, so the invalid value reaches this converter.
        # num_classes is a static attribute of relax.op.one_hot, so it has to be rejected
        # here rather than by the C++ builder, whose `depth > 0` check never mentions it.
        if isinstance(num_classes, int) and num_classes <= 0:
            raise ValueError(
                f"one_hot num_classes must be a positive integer, but got {num_classes}. "
                "Inferring the depth from the input (torch's num_classes=-1) is not "
                "supported because the resulting depth is data dependent."
            )
        on_value = node.args[2] if len(node.args) > 2 else node.kwargs.get("on_value", 1)
        off_value = node.args[3] if len(node.args) > 3 else node.kwargs.get("off_value", 0)
        axis = node.args[4] if len(node.args) > 4 else node.kwargs.get("axis", -1)
        on_value = relax.prim_value(on_value)
        off_value = relax.prim_value(off_value)
        return self.block_builder.emit(relax.op.one_hot(x, on_value, off_value, num_classes, axis))

    def _tensor(self, node: fx.Node) -> relax.Var:
        dtype = node.kwargs.get("dtype", None)
        if isinstance(node.args[0], float):
            return relax.const(node.args[0], dtype if dtype is not None else "float32")
        elif isinstance(node.args[0], int):
            return relax.const(node.args[0], dtype if dtype is not None else "int64")
        raise ValueError("torch.tensor with value not a float or int is not accepted")

    ########## DataType ##########

    def _float(self, node: fx.Node) -> relax.Var:
        return self.block_builder.emit(relax.op.astype(self.env[node.args[0]], "float32"))

    def _half(self, node: fx.Node) -> relax.Var:
        return self.block_builder.emit(relax.op.astype(self.env[node.args[0]], "float16"))

    def _is_floating_point(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        return relax.const(
            x.ty.dtype.dtype in ["float16", "float32", "float64", "bfloat16"], "bool"
        )

    def _type(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dtype = TorchFXImporter._convert_data_type(node.args[1], self.env)
        return self.block_builder.emit(relax.op.astype(x, dtype))

    ########## Others ##########

    def _getattr(self, node: fx.Node) -> relax.Var:
        if isinstance(self.env[node.args[0]], relax.Expr):
            if node.args[1] == "dtype":
                return self.env[node.args[0]].ty.dtype.dtype
            elif node.args[1] == "shape":
                return self.shape_of(self.env[node.args[0]])
        return getattr(self.env[node.args[0]], node.args[1])

    def create_input_vars(self, input_info: list[tuple[tuple[int], str]]) -> list[relax.Var]:
        inputs = list()
        for idx, (shape, dtype) in enumerate(input_info):
            inputs.append(
                relax.Var(f"inp_{idx}", relax.TensorType(shape, self._convert_data_type(dtype)))
            )
        return inputs

    def create_convert_map(
        self,
    ) -> dict[torch.nn.Module | str, Callable[[fx.Node], relax.Var]]:
        import operator

        import torch  # type: ignore
        from torch import nn

        return {
            ## call_module
            # unary
            nn.CELU: self._celu,
            nn.Dropout: lambda node: self.env[node.args[0]],
            nn.ELU: self._elu,
            nn.GELU: self._gelu,
            nn.Hardsigmoid: self._hardsigmoid,
            nn.Hardswish: self._hardswish,
            nn.Hardtanh: self._hardtanh,
            nn.Identity: lambda node: self.env[node.args[0]],
            nn.LeakyReLU: self._leakyrelu_module,
            nn.LogSoftmax: self._log_softmax_module,
            nn.PReLU: self._prelu_module,
            nn.ReLU: self._unary_op(relax.op.nn.relu),
            nn.ReLU6: self._unary_op(relax.op.nn.relu6),
            nn.Sigmoid: self._unary_op(relax.op.sigmoid),
            nn.SELU: self._unary_op(relax.op.nn.selu),
            nn.SiLU: self._unary_op(relax.op.nn.silu),
            nn.Softmax: self._softmax_module,
            nn.Softplus: self._softplus_module,
            nn.Tanh: self._unary_op(relax.op.tanh),
            # neural network
            nn.AdaptiveAvgPool1d: self._adaptive_avg_pool1d_module,
            nn.AdaptiveAvgPool2d: self._adaptive_avg_pool2d_module,
            nn.AdaptiveAvgPool3d: self._adaptive_avg_pool3d_module,
            nn.AvgPool1d: self._avg_pool1d_module,
            nn.AvgPool2d: self._avg_pool2d_module,
            nn.AvgPool3d: self._avg_pool3d_module,
            nn.BatchNorm2d: self._batch_norm_2d_module,
            nn.InstanceNorm1d: self._instance_norm,
            nn.InstanceNorm2d: self._instance_norm,
            nn.InstanceNorm3d: self._instance_norm,
            nn.Conv1d: self._conv1d_module,
            nn.Conv2d: self._conv2d_module,
            nn.Conv3d: self._conv3d_module,
            nn.ConvTranspose1d: self._conv_transpose1d_module,
            nn.ConvTranspose2d: self._conv_transpose2d_module,
            nn.CrossEntropyLoss: self._cross_entropy_module,
            nn.GroupNorm: self._group_norm_module,
            nn.LayerNorm: self._layer_norm_module,
            nn.Linear: self._linear_module,
            nn.MaxPool1d: self._max_pool1d_module,
            nn.MaxPool2d: self._max_pool2d_module,
            nn.MaxPool3d: self._max_pool3d_module,
            nn.modules.sparse.Embedding: self._embedding_module,
            nn.PixelShuffle: self._pixel_shuffle_module,
            # tensor manipulation
            nn.Flatten: self._flatten_module,
            ## call_function and call_method
            # unary
            "abs": self._unary_op(relax.op.abs),
            "acos": self._unary_op(relax.op.acos),
            "acosh": self._unary_op(relax.op.acosh),
            "asin": self._unary_op(relax.op.asin),
            "asinh": self._unary_op(relax.op.asinh),
            "atan": self._unary_op(relax.op.atan),
            "atanh": self._unary_op(relax.op.atanh),
            "bitwise_not": self._unary_op(relax.op.bitwise_not),
            "ceil": self._unary_op(relax.op.ceil),
            "celu": self._celu,
            "clamp": self._clamp,
            "cos": self._unary_op(relax.op.cos),
            "cosh": self._unary_op(relax.op.cosh),
            "dropout": lambda node: self.env[node.args[0]],
            "elu": self._elu,
            "erf": self._unary_op(relax.op.erf),
            "exp": self._unary_op(relax.op.exp),
            "floor": self._unary_op(relax.op.floor),
            "gelu": self._gelu,
            "hardsigmoid": self._hardsigmoid,
            "hardswish": self._hardswish,
            "hardtanh": self._hardtanh,
            "isfinite": self._unary_op(relax.op.isfinite),
            "isinf": self._unary_op(relax.op.isinf),
            "isin": self._isin,
            "isnan": self._unary_op(relax.op.isnan),
            "leaky_relu": self._leakyrelu,
            "log": self._unary_op(relax.op.log),
            "log2": self._log2,
            "log10": self._log10,
            "log1p": self._log1p,
            "logical_and": self._logical_and,
            "logical_not": self._logical_not,
            "logical_or": self._logical_or,
            "logical_xor": self._logical_xor,
            "log_softmax": self._log_softmax,
            "neg": self._unary_op(relax.op.negative),
            "pad": self._pad,
            "pixel_shuffle": self._pixel_shuffle,
            "prelu": self._prelu,
            "reciprocal": self._reciprocal,
            "relu": self._unary_op(relax.op.nn.relu),
            "relu6": self._unary_op(relax.op.nn.relu6),
            "round": self._round,
            "rsqrt": self._rsqrt,
            "selu": self._unary_op(relax.op.nn.selu),
            "sigmoid": self._unary_op(relax.op.sigmoid),
            "sign": self._unary_op(relax.op.sign),
            "silu": self._unary_op(relax.op.nn.silu),
            "sin": self._unary_op(relax.op.sin),
            "sinh": self._unary_op(relax.op.sinh),
            "softmax": self._softmax,
            "softplus": self._softplus,
            "sqrt": self._sqrt,
            "square": self._unary_op(relax.op.square),
            "tan": self._unary_op(relax.op.tan),
            "tanh": self._unary_op(relax.op.tanh),
            "tril_": self._inplace_tril_triu(relax.op.tril),
            "tril": self._tril_triu(relax.op.tril),
            "triu_": self._inplace_tril_triu(relax.op.triu),
            "triu": self._tril_triu(relax.op.triu),
            "trunc": self._unary_op(relax.op.trunc),
            # binary
            "add": self._binary_op(relax.op.add, operator.add),
            "and_": self._binary_op(relax.op.bitwise_and, operator.and_),
            "atan2": self._binary_op(relax.op.atan2, torch.atan2),
            "bitwise_or_": self._binary_op_inplace(relax.op.bitwise_or, operator.or_),
            "bitwise_or": self._binary_op(relax.op.bitwise_or, operator.or_),
            "div": self._div,
            "eq": self._binary_op(relax.op.equal, operator.eq),
            "floordiv": self._binary_op(relax.op.floor_divide, operator.floordiv),
            "fmod": self._fmod,
            "ge": self._binary_op(relax.op.greater_equal, operator.ge),
            "gt": self._binary_op(relax.op.greater, operator.gt),
            "iadd": self._binary_op(relax.op.add, operator.add),
            "le": self._binary_op(relax.op.less_equal, operator.le),
            "lshift": self._binary_op(relax.op.left_shift, operator.lshift),
            "lt": self._binary_op(relax.op.less, operator.lt),
            "matmul": self._binary_op(
                partial(relax.op.linear_algebra.matmul, out_dtype="float32"), operator.matmul
            ),
            "max": self._binary_op(relax.op.maximum, max),
            "min": self._binary_op(relax.op.minimum, min),
            "mod": self._binary_op(relax.op.floor_mod, operator.mod),
            "mul": self._binary_op(relax.op.multiply, operator.mul),
            "ne": self._binary_op(relax.op.not_equal, operator.ne),
            "outer": lambda node: self.block_builder.emit(
                relax.op.outer(self.env[node.args[0]], self.env[node.args[1]])
            ),
            "pow": self._pow,
            "or_": self._binary_op(relax.op.bitwise_or, operator.or_),
            "rshift": self._binary_op(relax.op.right_shift, operator.rshift),
            "rsub": self._rsub,
            "sub": self._binary_op(relax.op.subtract, operator.sub),
            "truediv": self._binary_op(relax.op.divide, operator.truediv),
            "xor": self._binary_op(relax.op.bitwise_xor, operator.xor),
            # neural network
            "adaptive_avg_pool1d": self._adaptive_avg_pool1d,
            "adaptive_avg_pool2d": self._adaptive_avg_pool2d,
            "adaptive_avg_pool3d": self._adaptive_avg_pool3d,
            "addmm": self._addmm,
            "avg_pool1d": self._avg_pool1d,
            "avg_pool2d": self._avg_pool2d,
            "avg_pool3d": self._avg_pool3d,
            "baddbmm": self._baddbmm,
            "bmm": self._binary_op(
                partial(relax.op.linear_algebra.matmul, out_dtype="float32"), operator.matmul
            ),
            "conv_transpose1d": self._conv_transpose1d,
            "conv_transpose2d": self._conv_transpose2d,
            "conv1d": self._conv1d,
            "conv2d": self._conv2d,
            "conv3d": self._conv3d,
            "cross_entropy": self._cross_entropy,
            "diagonal": self._diagonal,
            "einsum": self._einsum,
            "interpolate": self._interpolate,
            "layer_norm": self._layer_norm,
            "linear": self._linear,
            "max_pool1d": self._max_pool1d,
            "max_pool2d": self._max_pool2d,
            "max_pool3d": self._max_pool3d,
            "scaled_dot_product_attention": self._scaled_dot_product_attention,
            "stochastic_depth": lambda node: self.env[node.args[0]],
            "unbind": self._unbind,
            # linear interpolation
            "lerp": self._lerp,
            # statistical
            "mean": self._mean,
            "norm": self._norm,
            "prod": self._prod,
            "std": self._std,
            "sum": self._sum,
            "var": self._var,
            # search
            "argmax": self._argmax_argmin(relax.op.argmax),
            "argmin": self._argmax_argmin(relax.op.argmin),
            "where": self._where,
            "bucketize": self._bucketize,
            "searchsorted": self._searchsorted,
            # tensor manipulation
            "argsort": self._argsort,
            "broadcast_to": self._broadcast_to,
            "cat": self._cat,
            "chunk": self._chunk,
            "concat": self._cat,
            "contiguous": lambda node: self.env[node.args[0]],
            "cumprod": self._cumprod,
            "cumsum": self._cumsum,
            "expand": self._expand,
            "expand_as.default": self._expand_as,
            "flatten": self._flatten,
            "flip": self._flip,
            "gather": self._gather,
            "index_put_": self._index_put,
            "meshgrid": self._meshgrid,
            "narrow": self._narrow,
            "numel": self._numel,
            "permute": self._permute,
            "repeat": self._repeat,
            "roll": self._roll,
            "reshape": self._reshape,
            "scatter": self._scatter,
            "select": self._select,
            "size": self._size,
            "slice_scatter": self._slice_scatter,
            "sort": self._sort,
            "split": self._split,
            "squeeze": self._squeeze,
            "stack": self._stack,
            "take": self._take,
            "tile": self._tile,
            "topk": self._topk,
            "transpose": self._transpose,
            "unsqueeze": lambda node: self.block_builder.emit(
                relax.op.expand_dims(self.env[node.args[0]], node.args[1])
            ),
            "view": self._reshape,
            # tensor creation
            "arange": self._arange,
            "clone": lambda node: self.env[node.args[0]],
            "empty": self._empty,
            "empty_like": self._empty_like,
            "eye": self._eye,
            "fill": self._fill,
            "fill_": self._inplace_fill,
            "full": self._full,
            "index_select": self._index_select,
            "linspace": self._linspace,
            "masked_fill_": self._inplace_masked_fill,
            "masked_fill": self._masked_fill,
            "masked_scatter": self._masked_scatter,
            "new_ones": self._new_ones,
            "new_zeros": self._new_zeros,
            "ones": self._ones,
            "one_hot": self._one_hot,
            "ones_like": lambda node: self.block_builder.emit(
                relax.op.ones_like(self.env[node.args[0]])
            ),
            "tensor": self._tensor,
            "zero_": self._zeros_inplace,
            "zeros_like": self._zeros_like,
            "copy_": self._inplace_copy,
            # datatype
            "astype": self._type,
            "float": self._float,
            "half": self._half,
            "is_floating_point": self._is_floating_point,
            "to": self._to,
            "type": self._type,
            "type_as": self._type_as,
            # other
            "getattr": self._getattr,
            "getitem": self._getitem,
            "sym_size.int": self._sym_size_int,
            "item": self._item,
        }

    def from_fx(
        self,
        model,
        input_info: list[tuple[tuple[int], str]],
        keep_params_as_input: bool,
        unwrap_unit_return_tuple: bool,
        no_bind_return_tuple: bool,
        custom_convert_map: dict | None = None,
    ) -> tvm_tirx.IRModule:
        """Convert a PyTorch FX GraphModule to a Relax program."""
        from torch import fx

        if custom_convert_map:
            custom_ops = set(custom_convert_map.keys())
            self.update_convert_map(custom_convert_map)
        else:
            custom_ops = set()
        self.named_modules = dict(model.named_modules())

        graph: fx.Graph = model.graph

        # Create input variables.
        inputs = self.create_input_vars(input_info)

        # Initialize the block builder with a function and a dataflow block.
        func_name = "main"
        self.block_builder = relax.BlockBuilder()
        params = []
        if keep_params_as_input:
            func_attrs = {"num_input": len(inputs)}
            for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                shape = param.data.shape
                dtype = self._convert_data_type(str(param.data.dtype))
                inputs.append(relax.Var(name, relax.TensorType(shape, dtype)))
                self.params[param] = inputs[-1]
                params.append(tvm_tirx.runtime.tensor(param.data.cpu().numpy()))
        else:
            func_attrs = None

        # Find all the missing function types
        self._check_unsupported_func_type(graph.nodes)

        sym_vars = {
            v.name: v for shape, _ in input_info for v in shape if isinstance(v, tvm_tirx.ir.Var)
        }

        with self.block_builder.function(name=func_name, params=inputs.copy(), attrs=func_attrs):
            output = None
            with self.block_builder.dataflow():
                # Translate model parameters.
                for _, param in model.named_parameters():
                    shape = param.data.shape
                    dtype = self._convert_data_type(str(param.data.dtype))
                    if dtype in ("float32", "float16"):
                        if not keep_params_as_input:
                            self.params[param] = self._convert_torch_tensor_to_relax(param)
                    else:
                        raise ValueError(f"Unsupported data type for model parameters: {dtype}")
                # Translate the model.
                for node in graph.nodes:
                    if node.op == "placeholder":
                        if "grapharg" in node.meta and node.meta["grapharg"].fake_tensor is None:
                            # Sym input: bind to the matching shape var if referenced
                            if node.name in sym_vars:
                                self.env[node] = sym_vars[node.name]
                            continue

                        assert len(inputs) > 0, "Provided inputs is less than actual inputs"
                        self.env[node] = inputs.pop(0)
                    elif node.op == "output":
                        args = self.retrieve_args(node)
                        assert len(args) == 1

                        # return tuple
                        if isinstance(args[0], tuple | list | relax.Tuple):
                            # unit tuple
                            if unwrap_unit_return_tuple and len(args[0]) == 1:
                                output = self.block_builder.emit_output(args[0][0])
                            elif no_bind_return_tuple:
                                output = []
                                for ret in args[0]:
                                    output.append(self.block_builder.emit_output(ret))

                        if output is None:
                            output = self.block_builder.emit_output(args[0])
                        break
                    elif node.op == "get_attr":
                        self.env[node] = self._fetch_attr(model, node.target)
                    elif node.op == "call_module":
                        module = self.named_modules[node.target]
                        assert type(module) in self.convert_map, (
                            f"Unsupported module type {type(module)}"
                        )
                        self.env[node] = self.convert_map[type(module)](node)
                    elif node.op == "call_function":
                        func_name = node.target.__name__
                        if func_name in custom_ops:
                            self.env[node] = self.convert_map[func_name](node, self)
                        else:
                            self.env[node] = self.convert_map[func_name](node)
                    elif node.op == "call_method":
                        assert node.target in self.convert_map, (
                            f"Unsupported function target {node.target}"
                        )
                        self.env[node] = self.convert_map[node.target](node)
                    else:
                        raise ValueError(f"Unsupported op {node.op}")
            assert output is not None
            self.block_builder.emit_func_output(output)

        mod = self.block_builder.get()
        if keep_params_as_input:
            mod["main"] = mod["main"].with_attr("params", params)
        return mod


def from_fx(
    model,
    input_info: list[tuple[tuple[int], str]],
    *,
    keep_params_as_input: bool = False,
    unwrap_unit_return_tuple: bool = False,
    no_bind_return_tuple: bool = False,
    custom_convert_map: dict | None = None,
    default_image_layout: str = "NCHW",
) -> tvm_tirx.IRModule:
    """Convert a PyTorch FX GraphModule to a Relax program

    Parameters
    ----------
    model : fx.GraphModule
        The PyTorch FX GraphModule to convert.

    input_info : List[Tuple[Tuple[int], str]]
        A list of shapes and data types of input tensors.

    keep_params_as_input : bool
        Whether to keep model parameters as input variables.

    unwrap_unit_return_tuple : bool
        A boolean flag indicating if to the return value when it is an unit tuple.
        When the return value is not a unit tuple, no unwrap will take place.

    no_bind_return_tuple : bool
        A boolean flag indicating whether to bind the return tuple as a relax var.
        If the flag is true and the return value is a tuple, it will not bind it to a var.

    custom_convert_map : Dictionary of str to Relax op
        A custom op conversion map in the same format as TorchFXImporter.convert_map

    default_image_layout : str
        The default layout for image operations (e.g., "NCHW" or "NHWC").
        Default is "NCHW" which is the standard PyTorch layout.

    Returns
    -------
    output : tvm_tirx.IRModule
        The import result IRModule, with the function "main" containing the
        translated logic.
        If `keep_params_as_input` is true, the "main" function have an attribute
        "params" that contains the weights of the input model. The weights
        can be detached by `relax.frontend.detach_params`.

    Examples
    --------
    Users can use the FX tracer or dynamo.export() to extract
    a fx.GraphModule from a PyTorch model. The following codes show
    how to convert a PyTorch model to a Relax program.

    .. code-block:: python

        # Import the importer.
        import numpy as np
        import torch
        from tvm.relax.frontend.torch_fx import from_fx
        from torch import _dynamo as dynamo

        # Define the module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(in_features=10, out_features=7, bias=True)

            def forward(self, input):
                return self.linear(input)

        # Instantiate the model and create the input info dict.
        torch_model = MyModule()
        input_info = [((128, 10), "float32")]
        input_tensors = [
            torch.astensor(np.random.randn(*shape).astype(dtype))
            for shape, dtype in input_info
        ]

        # Use FX tracer to trace the PyTorch model.
        graph_module = fx.symbolic_trace(torch_model)

        # Use the dynamo.export() to export the PyTorch model to FX.
        try:
            graph_module = dynamo.export(torch_model, *input_tensors)
        except Exception:
            raise RuntimeError("Failed to export the PyTorch model to FX.")

        # Use the importer to import the PyTorch model to Relax.
        mod: tvm_tirx.IRModule = from_fx(graph_module, input_info)

        # Print out the imported model.
        print(mod.script())

    Notes
    -----
    For a given PyTorch model, to lookup the names of the model inputs in
    FX, one can use

    .. code-block:: python

        fx.symbolic_trace(model).graph.print_tabular()

    to print out the tabular representation of the PyTorch module, and then
    check the placeholder rows in the beginning of the tabular.
    """
    return TorchFXImporter(default_image_layout=default_image_layout).from_fx(
        model,
        input_info,
        keep_params_as_input,
        unwrap_unit_return_tuple,
        no_bind_return_tuple,
        custom_convert_map=custom_convert_map,
    )
