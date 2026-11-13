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
"""IR Builder Frame for Relax dialect"""

from tvm_ffi import register_object as _register_object

from tvm_tirx.script.ir_builder.base import IRBuilderFrame


@_register_object("script.ir_builder.relax.RelaxFrame")
class RelaxFrame(IRBuilderFrame):
    """The base ir_builder frame for the relax dialect."""


@_register_object("script.ir_builder.relax.SeqExprFrame")
class SeqExprFrame(RelaxFrame): ...


@_register_object("script.ir_builder.relax.FunctionFrame")
class FunctionFrame(SeqExprFrame):
    """Native function frame retaining signature and finalized results."""


@_register_object("script.ir_builder.relax.BindingBlockFrame")
class BindingBlockFrame(RelaxFrame):
    """The ir_builder frame for relax binding blocks."""


@_register_object("script.ir_builder.relax.IfFrame")
class IfFrame(RelaxFrame): ...


@_register_object("script.ir_builder.relax.ThenFrame")
class ThenFrame(SeqExprFrame): ...


@_register_object("script.ir_builder.relax.ElseFrame")
class ElseFrame(SeqExprFrame): ...
