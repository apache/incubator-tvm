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
"""Operation class for computation declaration."""

import inspect

# pylint: disable=invalid-name
from numbers import Integral as _Integral

import tvm_ffi
from tvm_ffi import Array

import tvm
import tvm_tirx.sym._ffi_api
import tvm_tirx.tirx
import tvm_tirx.tirx._ffi_api
from tvm_tirx.ir import is_prim_expr
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from . import _ffi_api
from . import tag as _tag
from . import tensor as _tensor
from .reduction import _iter_var_type


def placeholder(shape, dtype=None, name="placeholder"):
    """Construct an empty tensor object.

    Parameters
    ----------
    shape: Tuple of Expr
        The shape of the tensor

    dtype: str, optional
        The data type of the tensor

    name: str, optional
        The name hint of the tensor

    Returns
    -------
    tensor: Tensor
        The created tensor
    """
    dtype = "float32" if dtype is None else dtype
    return _ffi_api.Placeholder(shape, dtype, name)


def compute(shape, fcompute, name="compute", tag="", attrs=None, varargs_names=None):
    """Construct a new tensor by computing over the shape domain.

    The compute rule is result[axis] = fcompute(axis)

    Parameters
    ----------
    shape: Tuple of Expr
        The shape of the tensor

    fcompute: lambda function of indices-> value
        Specifies the input source expression

    name: str, optional
        The name hint of the tensor

    tag: str, optional
        Additional tag information about the compute.

    attrs: dict, optional
        The additional auxiliary attributes about the compute.

    varargs_names: list, optional
        The names to use for each of the varargs. If not supplied, the varargs
        will be called i1, i2, ...

    Returns
    -------
    tensor: Tensor
        The created tensor
    """
    if _tag.TagScope.get_current() is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.get_current().tag
    shape = (shape,) if tvm_tirx.ir.is_prim_expr(shape) else shape
    # for python3
    shape = tuple([int(s) if isinstance(s, float) else s for s in shape])
    out_ndim = len(shape)

    argspec = inspect.getfullargspec(fcompute)
    if len(argspec.args) == 0 and argspec.varargs is None:
        arg_names = [f"i{i}" for i in range(out_ndim)]
    elif argspec.varargs is not None:
        # if there is a varargs, it takes the remaining dimensions of out_ndim
        num_remaining_args = out_ndim - len(argspec.args)
        if varargs_names is not None:
            if len(varargs_names) != num_remaining_args:
                raise RuntimeError(
                    f"Number of varargs ({num_remaining_args}) does not match number"
                    f"of varargs_names ({len(varargs_names)})"
                )
            arg_names = argspec.args + varargs_names
        else:
            arg_names = argspec.args + [f"i{i}" for i in range(out_ndim - len(argspec.args))]
    else:
        arg_names = argspec.args
        # if there are fewer args than out dimensions, the remaining dimensions
        # are implicitly broadcast
        out_ndim = len(arg_names)
    assert argspec.varkw is None, "Variable keyword arguments not supported in fcompute"
    assert argspec.defaults is None, "Default arguments not supported in fcompute"
    assert len(argspec.kwonlyargs) == 0, "Keyword arguments are not supported in fcompute"

    if out_ndim != len(arg_names):
        raise ValueError(
            "Number of args to fcompute does not match dimension, "
            f"args={len(arg_names)}, dimension={out_ndim}"
        )

    dim_var = [_iter_var_type()((0, s), x, 0) for x, s in zip(arg_names, shape[:out_ndim])]
    body = fcompute(*[v.var for v in dim_var])

    if not isinstance(body, list | tuple):
        body = [body]
    body = tvm_ffi.convert(body)
    op_node = _ffi_api.ComputeOp(name, tag, attrs, dim_var, body)

    num = op_node.num_outputs
    outputs = tuple(op_node.output(i) for i in range(num))
    return outputs[0] if num == 1 else outputs


def scan(init, update, state_placeholder, inputs=None, name="scan", tag="", attrs=None):
    """Construct new tensors by scanning over axis.

    Parameters
    ----------
    init: Tensor or list of Tensor
        The initial condition of first init.shape[0] timestamps

    update: Tensor or list of Tensor
        The update rule of the scan given by symbolic tensor.

    state_placeholder: Tensor or list of Tensor
        The placeholder variables used by update.

    inputs: Tensor or list of Tensor, optional
        The list of inputs to the scan. This is not required, but can
        be useful for the compiler to detect scan body faster.

    name: str, optional
        The name hint of the tensor

    tag: str, optional
        Additonal tag information about the compute.

    attrs: dict, optional
        The additional auxiliary attributes about the compute.

    Returns
    -------
    tensor: Tensor or list of Tensors
        The created tensor or tuple of tensors contains multiple outputs.

    Example
    -------
    .. code-block:: python

      # The following code is equivalent to numpy.cumsum
      m = te.var("m")
      n = te.var("n")
      X = te.placeholder((m, n), name="X")
      s_state = te.placeholder((m, n))
      s_init = te.compute((1, n), lambda _, i: X[0, i])
      s_update = te.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
      res = tvm.te.scan(s_init, s_update, s_state, X)
    """
    if _tag.TagScope.get_current() is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.get_current().tag
    if isinstance(init, _tensor.Tensor):
        init = [init]
    if isinstance(update, _tensor.Tensor):
        update = [update]
    if isinstance(state_placeholder, _tensor.Tensor):
        state_placeholder = [state_placeholder]
    if isinstance(inputs, _tensor.Tensor):
        inputs = [inputs]
    if inputs is None:
        inputs = []
    if len(init) != len(update) or len(init) != len(state_placeholder):
        raise ValueError("init, update, state_placeholder must have same length")
    axis = _iter_var_type()((init[0].shape[0], update[0].shape[0]), f"{name}.idx", 3)
    op = _ffi_api.ScanOp(name, tag, attrs, axis, init, update, state_placeholder, inputs)
    res = [op.output(i) for i in range(len(update))]
    return res[0] if len(res) == 1 else res


def extern(
    shape,
    inputs,
    fcompute,
    name="extern",
    dtype=None,
    in_tensors=None,
    out_tensors=None,
    tag="",
    attrs=None,
):
    """Compute several tensors via an extern function.

    Parameters
    ----------
    shape: tuple or list of tuples.
        The shape of the outputs.

    inputs: list of Tensor
        The inputs

    fcompute: lambda function of inputs, outputs-> stmt
        Specifies the IR statement or sequence of statements to do the computation.
        See the following note for function signature of fcompute

        .. note::
             **Parameters**

             - **ins** (list of :any:`tvm_tirx.ir.Var`) - Placeholder for each inputs
             - **outs** (list of :any:`tvm_tirx.ir.Var`) - Placeholder for each outputs

             **Returns**

             - **stmt** (:any:`tvm_tirx.ir.Stmt` or a sequence of statements) -
               The body that carries out array computation.

    name: str, optional
        The name hint of the tensor

    dtype: str or list of str, optional
        The data types of outputs,
        by default dtype will be same as inputs.

    in_tensors: tvm_tirx.ir.Var or list of tvm_tirx.ir.Var, optional
        Input tensors.

    out_tensors: tvm_tirx.ir.Var or list of tvm_tirx.ir.Var, optional
        Output tensors.


    tag: str, optional
        Additonal tag information about the compute.

    attrs: dict, optional
        The additional auxiliary attributes about the compute.

    Returns
    -------
    tensor: Tensor or list of Tensors
        The created tensor or tuple of tensors contains multiple outputs.

    Example
    -------
    In the code below, C is generated by calling external PackedFunc
    `tvm.contrib.cblas.matmul`

    .. code-block:: python

        A = te.placeholder((n, l), name="A")
        B = te.placeholder((l, m), name="B")
        C = te.extern((n, m), [A, B],
                       lambda ins, outs: tvm_tirx.tirx.call_packed(
                          "tvm.contrib.cblas.matmul",
                            ins[0], ins[1], outs[0], 0, 0), name="C")
    """
    if _tag.TagScope.get_current() is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.get_current().tag
    shape = (shape,) if is_prim_expr(shape) or isinstance(shape, _Integral) else shape
    if shape == () or is_prim_expr(shape[0]) or isinstance(shape[0], _Integral):
        shape = [shape]
    if in_tensors is not None:
        in_tensors = [in_tensors] if not isinstance(in_tensors, list) else in_tensors
        if len(inputs) != len(in_tensors):
            raise RuntimeError(
                f"Number of inputs and in_tensors mismatch: {len(inputs)} vs {len(in_tensors)}."
            )
    if out_tensors is not None:
        out_tensors = [out_tensors] if not isinstance(out_tensors, list) else out_tensors
        if len(shape) != len(out_tensors):
            raise RuntimeError(
                f"Number of outputs and out_tensors mismatch: {len(shape)} vs {len(out_tensors)}."
            )
    input_placeholders = in_tensors or []
    output_placeholders = out_tensors or []
    types = set()
    for t in inputs:
        if not isinstance(t, _tensor.Tensor):
            raise ValueError("expect inputs to be tensor")
        if in_tensors is None:
            input_placeholders.append(
                tvm_tirx.tirx.Var(
                    t.op.name,
                    tvm_tirx.tirx.TensorType(
                        t.shape,
                        t.dtype,
                        layout=tvm_tirx.tirx.TileLayout(
                            tvm_tirx.tirx.layout.S[tuple(t.shape)]
                            + tvm_tirx.tirx.Var("elem_offset", "int32")
                        ),
                    ),
                )
            )
        types.add(t.dtype)

    if out_tensors is None:
        if dtype is None:
            if len(types) != 1:
                raise ValueError("Cannot infer output type, please provide dtype argument")
            infered_type = types.pop()
            dtype = [infered_type for _ in shape]
        if isinstance(dtype, str):
            dtype = [dtype]

        for shp, dt in zip(shape, dtype):
            output_placeholders.append(
                tvm_tirx.tirx.Var(
                    name,
                    tvm_tirx.tirx.TensorType(
                        shp,
                        "float32" if dt is None else dt,
                        layout=tvm_tirx.tirx.TileLayout(
                            tvm_tirx.tirx.layout.S[tuple(shp)]
                            + tvm_tirx.tirx.Var("elem_offset", "int32")
                        ),
                    ),
                )
            )
    body = fcompute(input_placeholders, output_placeholders)
    if tvm_tirx.ir.is_prim_expr(body):
        body = tvm_tirx.ir.Evaluate(body)
    if not isinstance(body, tvm_tirx.ir.Stmt | list | tuple | Array):
        raise ValueError(
            f"Function '{fcompute.__name__}' should return Expr, Stmt, or a sequence of Stmt, "
            "but it returned "
            f"'{type(body)}'"
        )

    op = _ffi_api.ExternOp(name, tag, attrs, inputs, input_placeholders, output_placeholders, body)
    res = [op.output(i) for i in range(len(output_placeholders))]
    return res[0] if len(res) == 1 else res


def extern_function(
    input_tensors: list[_tensor.Tensor], function: tvm_tirx.tirx.Function, **kwargs
):
    """Compute tensors via a schedulable TIR Function

    Parameters
    ----------
    input_tensors: list of Tensor
        Input tensors that map to the corresponding function input params.

    function: Function
        The TIR Function

    Returns
    -------
    tensor: Tensor or list of Tensors
        The created tensor or tuple of tensors if it contains multiple outputs.

    Example
    -------
    In the code below, a TVMScript defined TIR Function is inlined into
    a TE ExternOp. Applying te.create_function on this

    .. code-block:: python

        A = te.placeholder((128, 128), name="A")
        B = te.placeholder((128, 128), name="B")

        @Ts.function
        def before_split(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


            for i, j in T.grid(128, 128):
                with Ts.sblock("B"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] * 2.0

        C = te.extern_function([A, B], func)
    """

    # Preserve the function parameter order while selecting TensorType annotations.
    dt_access_map = tvm.s_tir._ffi_api.DomainTouchedAccessMap(function)
    ordered_tensors = [param for param in function.params if tvm_tirx.tirx.is_tensor_var(param)]
    in_tensors = [buf for buf in ordered_tensors if len(dt_access_map[buf][0])]
    out_tensors = [buf for buf in ordered_tensors if len(dt_access_map[buf][1])]
    assert in_tensors, "Function has no input tensors"
    assert out_tensors, "Function has no output tensors"

    outputs = []
    inplace = []
    input_tir_tensors = in_tensors
    for obuf in out_tensors:
        if obuf in in_tensors:
            inplace.append(obuf)
        else:
            outputs.append(obuf)

    if not outputs:
        iobuf = inplace.pop()
        input_tir_tensors.remove(iobuf)
        outputs = [iobuf]

    assert len(input_tir_tensors) == len(input_tensors), (
        "The number of provided input input_tensors does not match the number of ",
        "input tensors in the function",
    )
    for tensor, tir_tensor in zip(input_tensors, input_tir_tensors):
        # TODO(csullivan): Can a stronger comparison between Tensor<>TensorVar be made?
        assert len(tensor.shape) == len(tir_tensor.shape)
        for d1, d2 in zip(tensor.shape, tir_tensor.shape):
            assert d1 == d2, (
                "The input input_tensors provided do not match the input tensors in the ",
                "function. Please check that the order of input te.Input_Tensors and the ",
                "order of the function variables in the params list agree.",
            )
    output = extern(
        [buf.shape for buf in outputs],
        input_tensors,
        lambda ins, outs: function.body,
        in_tensors=input_tir_tensors,
        out_tensors=outputs,
        **kwargs,
    )
    return output


def var(name="tindex", dtype="int32", loc: Location = UNKNOWN_LOC):
    """Create a new variable with specified name and dtype

    Parameters
    ----------
    name : str
        The name

    dtype : str
        The data type

    loc : Location, optional
        The location of this variable in the source.

    Returns
    -------
    var : tirx.Var
        The result symbolic variable.
    """
    return tvm_tirx.tirx.Var(name, dtype, loc)


def const(value, dtype="int32", loc: Location = UNKNOWN_LOC):
    """Create a new constant with specified value and dtype

    Parameters
    ----------
    value : Union[bool, int, float, numpy.ndarray, tvm_tirx.runtime.Tensor]
        The constant value.

    dtype : str
        The data type

    loc : Location, optional
        The location of this variable in the source.

    Returns
    -------
    const : Expr
        The result constant expr.
    """
    return tvm_tirx.tirx.const(value, dtype, loc)


def thread_axis(dom=None, tag="", name="", loc: Location = UNKNOWN_LOC):
    """Create a new IterVar to represent thread index.

    Parameters
    ----------
    dom : Range or str
        The domain of iteration
        When str is passed, dom is set to None and str is used as tag

    tag : str, optional
        The thread tag

    name : str, optional
        The name of the var.

    loc : Location, optional
        The location of this variable in the source.

    Returns
    -------
    axis : IterVar
        The thread itervar.
    """
    if isinstance(dom, str):
        tag, dom = dom, None
    if not tag:
        raise ValueError("tag must be given as Positional or keyword argument")
    name = name if name else tag
    return _iter_var_type()(dom, name, 1, tag, loc)


def reduce_axis(dom, name="rv", thread_tag="", loc: Location = UNKNOWN_LOC):
    """Create a new IterVar for reduction.

    Parameters
    ----------
    dom : Range
        The domain of iteration.

    name : str
        The name of the variable.

    thread_tag : Optional[str]
        The name of the thread_tag.

    loc : Location, optional
        The location of this variable in the source.

    Returns
    -------
    axis : IterVar
        An iteration variable representing the value.
    """
    return _iter_var_type()(dom, name, 2, thread_tag, loc)


def create_function(
    ops: list[_tensor.Tensor | tvm_tirx.tirx.Var], index_dtype_override: str | None = None
) -> tvm_tirx.tirx.Function:
    """Create a TensorIR Function from tensor expression

    Parameters
    ----------
    ops : List[Union[_tensor.Tensor, tvm_tirx.tirx.Var]]
        The source expression.

    Example
    -------
    We define a matmul kernel using following code:

    .. code-block:: python

        import tvm, tvm_tirx
        from tvm import te
        from tvm.te import create_function
        import tvm_tirx.script

        A = te.placeholder((128, 128), name="A")
        B = te.placeholder((128, 128), name="B")
        k = te.reduce_axis((0, 128), "k")
        C = te.compute((128, 128), lambda x, y: te.sum(A[x, k] * B[y, k], axis=k), name="C")
        func = create_function([A, B, C])
        print(func.script())

    If we want to use TensorIR schedule to do transformations on such kernel,
    we need to use `create_function([A, B, C])` to create a schedulable Function.
    The generated function looks like:

    .. code-block:: python

        @Ts.function
        def tir_matmul(
            A: T.Tensor((128, 128)), B: T.Tensor((128, 128)), C: T.Tensor((128, 128))
        ) -> None:

            for i, j, k in T.grid(128, 128, 128):
                with Ts.sblock():
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                    with Ts.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] += A[vi, vk] * B[vj, vk]

    Returns
    -------
    func : tirx.Function
        The created function.
    """
    if not isinstance(ops, list | tuple | Array):
        ops = [ops]
    return _ffi_api.CreateFunction(ops, index_dtype_override)
