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
# ruff: noqa: E402, E501

"""
.. _import_model:

Importing Models from ML Frameworks
====================================
Apache TVM supports importing models from popular ML frameworks including PyTorch, ONNX,
and TensorFlow Lite. This tutorial walks through each import path with a minimal working
example and explains the key parameters. The PyTorch section additionally demonstrates
how to handle unsupported operators via a custom converter map.

For end-to-end optimization and deployment after importing, see :ref:`optimize_model`.

.. note::

    The ONNX section requires the ``onnx`` package. The TFLite section requires
    ``tensorflow`` and ``tflite``. Sections whose dependencies are missing are skipped
    automatically.

.. contents:: Table of Contents
    :local:
    :depth: 2
"""

######################################################################
# Importing from PyTorch (Recommended)
# -------------------------------------
# TVM's PyTorch frontend is the most feature-complete. The recommended entry point is
# :py:func:`~tvm.relax.frontend.torch.from_exported_program`, which works with PyTorch's
# ``torch.export`` API.
#
# We start by defining a small CNN model for demonstration. No pretrained weights are
# needed — we only care about the graph structure.

import numpy as np
import torch
from torch import nn
from torch.export import export

import tvm
import tvm_tirx
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


torch_model = SimpleCNN().eval()
example_args = (torch.randn(1, 3, 32, 32),)

######################################################################
# Basic import
# ~~~~~~~~~~~~
# The standard workflow is: ``torch.export.export()`` → ``from_exported_program()`` →
# ``detach_params()``.

with torch.no_grad():
    exported_program = export(torch_model, example_args)
    mod = from_exported_program(
        exported_program,
        keep_params_as_input=True,
        unwrap_unit_return_tuple=True,
    )

mod, params = relax.frontend.detach_params(mod)
mod.show()

######################################################################
# Key parameters
# ~~~~~~~~~~~~~~
# ``from_exported_program`` accepts several parameters that control how the model is
# translated:
#
# - **keep_params_as_input** (bool, default ``False``): When ``True``, model weights become
#   function parameters, separated via ``relax.frontend.detach_params()``. When ``False``,
#   weights are embedded as constants inside the IRModule. Use ``True`` when you want to
#   manage weights independently (e.g., for weight sharing or quantization).
#
# - **unwrap_unit_return_tuple** (bool, default ``False``): PyTorch ``export`` always wraps
#   the return value in a tuple. Set ``True`` to unwrap single-element return tuples for a
#   cleaner Relax function signature.
#
# - **run_ep_decomposition** (bool, default ``True``): Runs PyTorch's built-in operator
#   decomposition before translation. This breaks high-level ops (e.g., ``batch_norm``) into
#   lower-level primitives, which generally improves TVM's coverage and optimization
#   opportunities. Set ``False`` if you want to preserve the original op granularity.

######################################################################
# Handling unsupported operators with ``custom_convert_map``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# When TVM encounters a PyTorch operator it does not recognize, it raises an error
# indicating the unsupported operator name. You can extend the frontend by providing a
# **custom converter map** — a dictionary mapping operator names to your own conversion
# functions.
#
# A custom converter function receives two arguments:
#
# - **node** (``torch.fx.Node``): The FX graph node being converted, carrying operator
#   info and references to input nodes.
# - **importer** (``ExportedProgramImporter``): The importer instance, giving access to:
#
#   - ``importer.env``: Dict mapping FX nodes to their converted Relax expressions.
#   - ``importer.block_builder``: The Relax ``BlockBuilder`` for emitting operations.
#   - ``importer.retrieve_args(node)``: Helper to look up converted args.
#
# The function must return a ``relax.Var`` — the Relax expression for this node's output.
# Here is an example that maps an operator to ``relax.op.sigmoid``:

from tvm.relax.frontend.torch.exported_program_translator import ExportedProgramImporter


def convert_sigmoid(node: torch.fx.Node, importer: ExportedProgramImporter) -> relax.Var:
    """Custom converter: map an op to relax.op.sigmoid."""
    args = importer.retrieve_args(node)
    return importer.block_builder.emit(relax.op.sigmoid(args[0]))


######################################################################
# To use the custom converter, pass it via the ``custom_convert_map`` parameter. The key
# is the ATen operator name in ``"op_name.variant"`` format (e.g., ``"sigmoid.default"``):
#
# .. code-block:: python
#
#    mod = from_exported_program(
#        exported_program,
#        custom_convert_map={"sigmoid.default": convert_sigmoid},
#    )
#
# .. note::
#
#    To find the correct operator name, check the error message TVM raises when encountering
#    the unsupported op — it includes the exact ATen name. You can also inspect the exported
#    program's graph via ``print(exported_program.graph_module.graph)`` to see all operator
#    names.

######################################################################
# Alternative PyTorch import methods
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Besides ``from_exported_program``, TVM also provides:
#
# - :py:func:`~tvm.relax.frontend.torch.from_fx`: Works with ``torch.fx.GraphModule``
#   from ``torch.fx.symbolic_trace()``. Requires explicit ``input_info`` (shapes and dtypes).
#   Use this when ``torch.export`` fails on certain Python control flow patterns.
#
# - :py:func:`~tvm.relax.frontend.torch.relax_dynamo`: A ``torch.compile`` backend that
#   compiles and executes the model through TVM in one step. Useful for integrating TVM
#   into an existing PyTorch training or inference loop.
#
# - :py:func:`~tvm.relax.frontend.torch.dynamo_capture_subgraphs`: Captures subgraphs from
#   a PyTorch model into an IRModule via ``torch.compile``. Each subgraph becomes a separate
#   function in the IRModule.
#
# For most use cases, ``from_exported_program`` is the recommended path.

######################################################################
# Verifying the imported model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# After importing, it is good practice to verify that TVM produces the same output as the
# original framework. We compile with the minimal ``"zero"`` pipeline (no tuning) and
# compare. The same approach applies to models imported via the ONNX and TFLite frontends
# shown below.

mod_compiled = relax.get_pipeline("zero")(mod)
exec_module = tvm.compile(mod_compiled, target="llvm")
dev = tvm_tirx.cpu()
vm = relax.VirtualMachine(exec_module, dev)

# Run inference
input_data = np.random.rand(1, 3, 32, 32).astype("float32")
tvm_input = tvm_tirx.runtime.tensor(input_data, dev)
tvm_params = [tvm_tirx.runtime.tensor(p, dev) for p in params["main"]]
tvm_out = vm["main"](tvm_input, *tvm_params).numpy()

# Compare with PyTorch
with torch.no_grad():
    pt_out = torch_model(torch.from_numpy(input_data)).numpy()

np.testing.assert_allclose(tvm_out, pt_out, rtol=1e-5, atol=1e-5)
print("PyTorch vs TVM outputs match!")

######################################################################
# Importing from ONNX
# --------------------
# TVM can import ONNX models via :py:func:`~tvm.relax.frontend.onnx.from_onnx`. The
# function accepts an ``onnx.ModelProto`` object, so you need to load the model with
# ``onnx.load()`` first.
#
# Here we export the same CNN model to ONNX format and then import it into TVM.

try:
    import onnx
    import onnxscript  # noqa: F401  # required by torch.onnx.export

    HAS_ONNX = True
except ImportError:
    onnx = None  # type: ignore[assignment]
    HAS_ONNX = False

if HAS_ONNX:
    from tvm.relax.frontend.onnx import from_onnx

    # Export the PyTorch model to ONNX
    dummy_input = torch.randn(1, 3, 32, 32)
    onnx_path = "simple_cnn.onnx"
    torch.onnx.export(torch_model, dummy_input, onnx_path, input_names=["input"])

    # Load and import into TVM
    onnx_model = onnx.load(onnx_path)
    mod_onnx = from_onnx(onnx_model, keep_params_in_input=True)
    mod_onnx, params_onnx = relax.frontend.detach_params(mod_onnx)
    mod_onnx.show()

######################################################################
# If you already have an ``.onnx`` file on disk, the workflow is even simpler:
#
# .. code-block:: python
#
#    import onnx
#    from tvm.relax.frontend.onnx import from_onnx
#
#    onnx_model = onnx.load("my_model.onnx")
#    mod = from_onnx(onnx_model)
#

######################################################################
# Key parameters
# ~~~~~~~~~~~~~~
# - **shape_dict** (dict, optional): Maps input names to shapes. Auto-inferred from the
#   model if not provided. Useful when the ONNX model has dynamic dimensions that you
#   want to fix to concrete sizes:
#
#   .. code-block:: python
#
#      mod = from_onnx(onnx_model, shape_dict={"input": [1, 3, 224, 224]})
#
# - **dtype_dict** (str or dict, default ``"float32"``): Input dtypes. A single string
#   applies to all inputs, or use a dict to set per-input dtypes:
#
#   .. code-block:: python
#
#      mod = from_onnx(onnx_model, dtype_dict={"input": "float16"})
#
# - **keep_params_in_input** (bool, default ``False``): Same semantics as PyTorch — whether
#   model weights are function parameters or embedded constants.
#
# - **opset** (int, optional): Override the opset version auto-detected from the model.
#   Each ONNX op may have different semantics across opset versions; TVM's converter
#   selects the appropriate implementation automatically. You rarely need to set this
#   unless the model metadata is incorrect.

######################################################################
# Importing from TensorFlow Lite
# -------------------------------
# TVM can import TFLite flat buffer models via
# :py:func:`~tvm.relax.frontend.tflite.from_tflite`. The function expects a TFLite
# ``Model`` object parsed from flat buffer bytes via ``GetRootAsModel``.
#
# .. note::
#
#    The ``tflite`` Python package has changed its module layout across versions.
#    Older versions use ``tflite.Model.Model.GetRootAsModel``, while newer versions use
#    ``tflite.Model.GetRootAsModel``. The code below handles both.
#
# Below we create a minimal TFLite model from TensorFlow and import it.

try:
    import tensorflow as tf
    import tflite
    import tflite.Model

    HAS_TFLITE = True
except ImportError:
    HAS_TFLITE = False

if HAS_TFLITE:
    from tvm.relax.frontend.tflite import from_tflite

    # Define a simple TF module and convert to TFLite.
    # We use plain TF ops (not keras layers) to avoid variable-handling ops
    # that some TFLite converter versions do not support cleanly.
    class TFModule(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(1, 784), dtype=tf.float32),
                tf.TensorSpec(shape=(784, 10), dtype=tf.float32),
            ]
        )
        def forward(self, x, weight):
            return tf.matmul(x, weight) + 0.1

    tf_module = TFModule()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tf_module.forward.get_concrete_function()], tf_module
    )
    tflite_buf = converter.convert()

    # Parse and import into TVM (API differs between tflite package versions)
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(tflite_buf, 0)
    mod_tflite = from_tflite(tflite_model)
    mod_tflite.show()

######################################################################
# Loading from a ``.tflite`` file
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If you already have a ``.tflite`` file on disk, load the raw bytes and parse them:
#
# .. code-block:: python
#
#    import tflite
#    import tflite.Model
#    from tvm.relax.frontend.tflite import from_tflite
#
#    with open("my_model.tflite", "rb") as f:
#        tflite_buf = f.read()
#
#    if hasattr(tflite.Model, "Model"):
#        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_buf, 0)
#    else:
#        tflite_model = tflite.Model.GetRootAsModel(tflite_buf, 0)
#    mod = from_tflite(tflite_model)

######################################################################
# Key parameters
# ~~~~~~~~~~~~~~
# - **shape_dict** / **dtype_dict** (optional): Override input shapes and dtypes. If not
#   provided, they are inferred from the TFLite model metadata.
#
# - **op_converter** (class, optional): A custom operator converter class. Subclass
#   ``OperatorConverter`` and override its ``convert_map`` dictionary to add or replace
#   operator conversions. For example, to add a hypothetical ``CUSTOM_RELU`` op:
#
#   .. code-block:: python
#
#      from tvm.relax.frontend.tflite.tflite_frontend import OperatorConverter
#
#      class MyConverter(OperatorConverter):
#          def __init__(self, model, subgraph, exp_tab, ctx):
#              super().__init__(model, subgraph, exp_tab, ctx)
#              self.convert_map["CUSTOM_RELU"] = self._convert_custom_relu
#
#          def _convert_custom_relu(self, op):
#              # implement your conversion logic here
#              ...
#
#      mod = from_tflite(tflite_model, op_converter=MyConverter)

######################################################################
# Summary
# -------
#
# +---------------------+----------------------------+-------------------------------+-----------------------------+
# | Aspect              | PyTorch                    | ONNX                          | TFLite                      |
# +=====================+============================+===============================+=============================+
# | Entry function      | ``from_exported_program``  | ``from_onnx``                 | ``from_tflite``             |
# +---------------------+----------------------------+-------------------------------+-----------------------------+
# | Input               | ``ExportedProgram``        | ``onnx.ModelProto``           | TFLite ``Model`` object     |
# +---------------------+----------------------------+-------------------------------+-----------------------------+
# | Custom extension    | ``custom_convert_map``     | —                             | ``op_converter`` class      |
# +---------------------+----------------------------+-------------------------------+-----------------------------+
#
# **Which to use?** Pick the frontend that matches your model format:
#
# - Have a PyTorch model? Use ``from_exported_program`` — it has the broadest operator coverage.
# - Have an ``.onnx`` file? Use ``from_onnx``.
# - Have a ``.tflite`` file? Use ``from_tflite``.
#
# The verification workflow (compile → run → compare) demonstrated in the PyTorch section
# above applies equally to ONNX and TFLite imports.
#
# For the full list of supported operators, see the converter map in each frontend's source:
# PyTorch uses ``create_convert_map()`` in ``exported_program_translator.py``, ONNX uses
# ``_get_convert_map()`` in ``onnx_frontend.py``, and TFLite uses ``convert_map`` in
# ``OperatorConverter`` in ``tflite_frontend.py``.
#
# After importing, refer to :ref:`optimize_model` for optimization and deployment.
