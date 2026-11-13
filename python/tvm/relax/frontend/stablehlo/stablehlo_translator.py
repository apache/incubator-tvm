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
# pylint: disable=import-outside-toplevel, unused-argument

"""StableHLO frontend of Relax."""

from collections.abc import Callable
from typing import Any

import tvm_tirx
from tvm import relax
from tvm_tirx import tirx


def _is_tensor_var(value) -> bool:
    """Whether ``value`` belongs to Relax's tensor domain."""
    return isinstance(value, tvm_tirx.ir.Var) and isinstance(value.ty, relax.TensorType)


class StableHLOImporter:
    """An importer from StableHLO to Relax."""

    from jaxlib import mlir
    from jaxlib.mlir.dialects import stablehlo

    def __init__(self) -> None:
        from jaxlib import mlir

        self._nodes: dict[str | mlir.ir.Operation, relax.Expr] = {}
        self.block_builder: relax.BlockBuilder = None
        self.create_convert_map()

    @staticmethod
    def _convert_data_type(input_type):
        """converts the data type from mlir to tvm."""
        from jaxlib import mlir

        if mlir.ir.ShapedType.isinstance(input_type):
            input_type = mlir.ir.ShapedType(input_type).element_type

        input_type = str(input_type)
        if input_type == "f16":
            return "float16"
        elif input_type in ["f32", "F32Type"]:
            return "float32"
        elif input_type in ["f64", "F64Type"]:
            return "float64"
        elif input_type == "i1":
            return "bool"
        elif input_type == "i8":
            return "int8"
        elif input_type == "i16":
            return "int16"
        elif input_type == "i32":
            return "int32"
        elif input_type == "i64":
            return "int64"
        elif input_type == "ui8":
            return "uint8"
        elif input_type == "ui16":
            return "uint16"
        elif input_type == "ui32":
            return "uint32"
        elif input_type == "ui64":
            return "uint64"
        else:
            raise NotImplementedError(f"input_type {input_type} is not handled yet")

    def _attr2value(self, node) -> Any | list[Any]:
        import numpy as np
        from jaxlib import mlir

        if mlir.ir.IntegerAttr.isinstance(node):
            int_attr = mlir.ir.IntegerAttr(node)
            return int_attr.value
        if mlir.ir.FloatAttr.isinstance(node):
            float_attr = mlir.ir.FloatAttr(node)
            return float_attr.value
        if mlir.ir.DenseIntElementsAttr.isinstance(node):
            dense_attr = mlir.ir.DenseIntElementsAttr(node)
        elif mlir.ir.DenseFPElementsAttr.isinstance(node):
            dense_attr = mlir.ir.DenseFPElementsAttr(node)
        else:
            raise ValueError("Unsupported Attribute type: " + str(type(node)))
        ret = []
        for val in dense_attr:
            ret.append(val)
        shape = self.get_shape(node.type)
        dtype = self._convert_data_type(node.type)
        return np.asarray(ret, dtype).reshape(shape).tolist()

    def retrieve_operands(self, node):
        return self._retrieve_operands(node.operands)

    def _retrieve_operands(self, node):
        from jaxlib import mlir

        # the operand is one of the inputs of FuncOp
        if isinstance(node, mlir.ir.Operation):
            return self._nodes[node]
        if isinstance(node, tuple):
            return tuple(self._retrieve_operands(x) for x in node)
        if isinstance(node, list | mlir.ir.OpOperandList):
            return [self._retrieve_operands(x) for x in node]
        if isinstance(node, dict):
            return {self._retrieve_operands(k): self._retrieve_operands(v) for k, v in node.items()}
        if isinstance(node, mlir.ir.Value):
            if isinstance(node.owner, mlir.ir.Block):
                block_arg = mlir.ir.BlockArgument(node)
                return self._nodes["arg" + str(block_arg.arg_number)]
            return self._retrieve_operands(node.owner)
        return node

    def get_shape(self, inpt_type) -> list[Any]:
        """Get the shape from Type like tensor<?x?xf32>"""
        from jaxlib import mlir

        shape_type = inpt_type
        if isinstance(shape_type, mlir.ir.Type):
            shape_type = mlir.ir.ShapedType(shape_type)
        ret = []
        for i in range(shape_type.rank):
            # get_dim_size
            if shape_type.is_dynamic_dim(i):
                n = tirx.Var("n", "int64")
                ret.append(n)
            else:
                ret.append(shape_type.get_dim_size(i))

        return ret

    @staticmethod
    def _promote_binary_op_args(lhs, rhs):
        if not isinstance(lhs, relax.Expr) and not isinstance(rhs, relax.Expr):
            msg = "Both the lhs and the rhs are not expressions."
            raise AssertionError(msg)
        if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
            return lhs, rhs
        if isinstance(lhs, relax.Expr):
            assert isinstance(lhs.ty, relax.TensorType)
            return lhs, relax.const(rhs, lhs.ty.dtype)
        assert isinstance(rhs.ty, relax.TensorType)
        return relax.const(lhs, rhs.ty.dtype), rhs

    def _call_binary_op(self, op, lhs, rhs):
        lhs, rhs = StableHLOImporter._promote_binary_op_args(lhs, rhs)
        return self.block_builder.emit(op(lhs, rhs))

    def _add(self, node: mlir.ir.Operation) -> relax.Expr:
        lhs, rhs = self.retrieve_operands(node)
        if _is_tensor_var(lhs) or _is_tensor_var(rhs):
            return self._call_binary_op(relax.op.add, lhs, rhs)
        return lhs + rhs

    def _maximum(self, node: mlir.ir.Operation) -> relax.Expr:
        lhs, rhs = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.maximum(lhs, rhs))

    def _minimum(self, node: mlir.ir.Operation) -> relax.Expr:
        lhs, rhs = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.minimum(lhs, rhs))

    def _divide(self, node: mlir.ir.Operation) -> relax.Expr:
        lhs, rhs = self.retrieve_operands(node)
        if _is_tensor_var(lhs) or _is_tensor_var(rhs):
            return self._call_binary_op(relax.op.divide, lhs, rhs)
        return lhs / rhs

    def _multiply(self, node: mlir.ir.Operation) -> relax.Expr:
        lhs, rhs = self.retrieve_operands(node)
        if _is_tensor_var(lhs) or _is_tensor_var(rhs):
            return self._call_binary_op(relax.op.multiply, lhs, rhs)
        return lhs * rhs

    def _subtract(self, node: mlir.ir.Operation) -> relax.Expr:
        lhs, rhs = self.retrieve_operands(node)
        if _is_tensor_var(lhs) or _is_tensor_var(rhs):
            return self._call_binary_op(relax.op.subtract, lhs, rhs)
        return lhs - rhs

    def _broadcast_in_dim(self, node: mlir.ir.Operation) -> relax.Expr:
        operands = self.retrieve_operands(node)
        data = operands[0]
        # broadcast_dims = self._attr2value(node.attributes["broadcast_dimensions"])
        shape = self.get_shape(node.result.type)
        # scalar
        if len(shape) == 0:
            return data
        return self.block_builder.emit(relax.op.broadcast_to(data, shape))

    def _const(self, node: mlir.ir.Operation) -> relax.Expr:
        const_value = self._attr2value(node.attributes["value"])
        dtype = self._convert_data_type(node.result.type)
        return relax.const(const_value, dtype)

    def _dot_general(self, node: mlir.ir.Operation) -> relax.Expr:
        lhs, rhs = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.matmul(lhs, rhs))

    def _convolution(self, node) -> relax.Expr:
        from jaxlib import mlir

        x, weight = self.retrieve_operands(node)
        shaped_type = mlir.ir.ShapedType(node.result.type)
        out_dtype = self._convert_data_type(shaped_type.element_type)
        strides = self._attr2value(node.attributes["window_strides"])
        padding = self._attr2value(node.attributes["padding"])
        lhs_dilation = self._attr2value(node.attributes["lhs_dilation"])
        rhs_dilation = self._attr2value(node.attributes["rhs_dilation"])
        if len(lhs_dilation) > 0:
            lhs_dilation = lhs_dilation[0]
        if len(rhs_dilation) > 0:
            rhs_dilation = rhs_dilation[0]
        dilation = (lhs_dilation, rhs_dilation)
        groups = self._attr2value(node.attributes["batch_group_count"])
        conv2d = relax.op.nn.conv2d(
            x,
            weight,
            strides=strides,
            padding=padding[0],
            dilation=dilation,
            groups=groups,
            data_layout="NHWC",
            kernel_layout="HWIO",
            out_dtype=out_dtype,
        )

        return self.block_builder.emit(conv2d)

    def _reshape(self, node: mlir.ir.Operation) -> relax.Expr:
        data = self.retrieve_operands(node)
        if isinstance(data, list):
            assert len(data) == 1
            data = data[0]
        new_shape = self.get_shape(node.result.type)
        return self.block_builder.emit(relax.op.reshape(data, new_shape))

    def _reduce(self, node: mlir.ir.Operation) -> relax.Expr:
        data = self.retrieve_operands(node)
        dimensions = self._attr2value(node.attributes["dimensions"])
        if node.body is not None:
            reducer_op = node.body.blocks[0].operations[0].OPERATION_NAME
            assert reducer_op == "stablehlo.add", f"reducer {reducer_op} in reduce is not supported"
        return self.block_builder.emit(relax.op.sum(data[0], axis=dimensions))

    def _reduce_window(self, node: mlir.ir.Operation) -> relax.Expr:
        operands = self.retrieve_operands(node)
        window_dimensions = self._attr2value(node.attributes["window_dimensions"])
        window_dilations = self._attr2value(node.attributes["window_dilations"])

        if node.body is not None:
            reducer_op = node.body.blocks[0].operations[0].OPERATION_NAME
            assert reducer_op == "stablehlo.maximum", (
                f"the reducer {reducer_op} in reduce_window is not supported"
            )

        pool_size = []
        for i, window_dim in enumerate(window_dimensions):
            if window_dim == 0:
                pool_size.append(0)
            else:
                dilated_window_size = (window_dim - 1) * window_dilations[i] + 1
                pool_size.append(dilated_window_size)
        strides = self._attr2value(node.attributes["window_strides"])
        # padding = self._attr2value(node.attributes["padding"])

        # TODO (yongwww): Infer the layout automatically
        layout = "NHWC"

        ret = self.block_builder.emit(
            relax.op.nn.max_pool2d(
                operands[0],
                pool_size=pool_size[1:3],  # HW
                strides=strides[1:3],
                padding=[1, 1],
                dilation=window_dilations[1:3],
                layout=layout,
            )
        )
        return ret

    def _rsqrt(self, node: mlir.ir.Operation) -> relax.Expr:
        data = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.rsqrt(data[0]))

    def _sin(self, node: mlir.ir.Operation) -> relax.Expr:
        data = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.sin(data[0]))

    def _sinh(self, node: mlir.ir.Operation) -> relax.Expr:
        data = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.sinh(data[0]))

    def _cos(self, node: mlir.ir.Operation) -> relax.Expr:
        data = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.cos(data[0]))

    def _cosh(self, node: mlir.ir.Operation) -> relax.Expr:
        data = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.cosh(data[0]))

    def _sqrt(self, node: mlir.ir.Operation) -> relax.Expr:
        data = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.sqrt(data[0]))

    def _round(self, node: mlir.ir.Operation) -> relax.Expr:
        data = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.round(data[0]))

    def _exp(self, node: mlir.ir.Operation) -> relax.Expr:
        data = self.retrieve_operands(node)
        return self.block_builder.emit(relax.op.exp(data[0]))

    def _return(self, node: mlir.ir.Operation) -> relax.Expr:
        outputs = self.retrieve_operands(node)
        return self.block_builder.emit_output(self.nodes[outputs])

    def create_convert_map(self):
        from jaxlib import mlir

        self.convert_map: dict[str, Callable[[mlir.ir.Operation], relax.Var]] = {
            "stablehlo.add": self._add,
            "stablehlo.broadcast_in_dim": self._broadcast_in_dim,
            "stablehlo.constant": self._const,
            "stablehlo.convolution": self._convolution,
            "stablehlo.cosine": self._cos,
            "stablehlo.cosh": self._cosh,
            "stablehlo.divide": self._divide,
            "stablehlo.dot_general": self._dot_general,
            "stablehlo.exponential": self._exp,
            "stablehlo.maximum": self._maximum,
            "stablehlo.minimum": self._minimum,
            "stablehlo.multiply": self._multiply,
            "stablehlo.reshape": self._reshape,
            "stablehlo.reduce": self._reduce,
            "stablehlo.reduce_window": self._reduce_window,
            "stablehlo.round_nearest_afz": self._round,
            "stablehlo.rsqrt": self._rsqrt,
            "stablehlo.sine": self._sin,
            "chlo.sinh": self._sinh,
            "stablehlo.sqrt": self._sqrt,
            "stablehlo.subtract": self._subtract,
            "func.return": self._return,
            "stablehlo.return": self._return,
        }

    def from_stablehlo(self, model, input_info: list[tuple[tuple[int], str]]) -> tvm_tirx.IRModule:
        """Convert a StableHLO Module to a Relax program.

        Parameters
        ----------
        model : mlir.ir.Module
            The StableHLO Module to convert.

        input_info : List[Tuple[Tuple[int], str]]
            A list of shapes and data types of input tensors.

        Returns
        -------
        output : tvm_tirx.IRModule
            The result IRModule with entry function "main"
        """
        from jaxlib import mlir
        from jaxlib.mlir.dialects import stablehlo

        assert isinstance(model, mlir.ir.Module)
        block: mlir.ir.Block = model.body.operations[0].regions[0].blocks[0]

        # inputs of the function
        inputs = []
        for idx, arg in enumerate(block.arguments.types):
            arg_shape = mlir.ir.ShapedType(arg)
            ipt_shape = self.get_shape(arg_shape)
            ipt_dtype = self._convert_data_type(arg_shape.element_type)
            ipt_name = "arg" + str(idx)
            ipt_var = relax.Var(f"arg{idx}", relax.TensorType(ipt_shape, ipt_dtype))
            self._nodes[ipt_name] = ipt_var
            inputs.append(ipt_var)

        # TODO (yongwww): Handle mlir.ir.Module with multiple functions
        # Initialize the block builder with a function and a dataflow block.
        # Raise error if the input stablehlo op is impure
        func_name = "main"
        self.block_builder = relax.BlockBuilder()

        with self.block_builder.function(name=func_name, params=inputs.copy()):
            output = None
            with self.block_builder.dataflow():
                block = model.body.operations[0].regions[0].blocks[0]
                for operation in block.operations:
                    if isinstance(operation, mlir.dialects.func.ReturnOp | stablehlo.ReturnOp):
                        operation = operation.operands[0].owner
                        # TODO (yongwww): handle multiple outputs
                        output = self.block_builder.emit_output(self._nodes[operation])
                        break

                    if isinstance(operation, mlir.ir.OpView):
                        op_name = operation.operation.name
                        assert op_name in self.convert_map, f"Unsupported operation {op_name}"
                        self._nodes[operation] = self.convert_map[op_name](operation)
                    else:
                        raise ValueError(f"Unsupported op {operation}")
            assert output is not None
            self.block_builder.emit_func_output(output)

        mod = self.block_builder.get()
        return mod


def from_stablehlo(
    stablehlo_module,
    input_info: list[tuple[tuple[int], str]] | None = None,
) -> tvm_tirx.IRModule:
    """Convert a StableHLO Module to a Relax program

    Parameters
    ----------
    stablehlo_module : Union[str, mlir.ir.Module]
        The StableHLO Module to convert.

    input_info : List[Tuple[Tuple[int], str]]
        A list of shapes and data types of input tensors.

    Returns
    -------
    output : tvm_tirx.IRModule
        The result IRModule with entry function "main"
    """
    from jax._src.interpreters import mlir as jax_mlir

    if isinstance(stablehlo_module, str):
        # TODO (yongwww): support the serialized bytecode format of StableHLO
        # model using stablehlo.deserialize_portable_artifact(ir) if the python
        # binding is ready
        context = jax_mlir.make_ir_context()
        stablehlo_module = jax_mlir.ir.Module.parse(stablehlo_module, context)
    return StableHLOImporter().from_stablehlo(stablehlo_module, input_info)
