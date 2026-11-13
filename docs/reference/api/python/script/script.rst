..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

tvm_tirx.script
---------------

tvm_tirx.script
***************
.. automodule:: tvm_tirx.script
   :members:
   :imported-members:
   :exclude-members: Any

tvm.relax.script
****************

Operator attributes re-exported by this namespace are documented in :mod:`tvm.relax.op`.
The shared ``NoneType`` class is documented in :mod:`tvm_tirx.ir`.

.. automodule:: tvm.relax.script
   :members:
   :exclude-members: ExternFunc, NoneType, ShapeExpr, TupleGetItem, Range, function, macro,
      ArgmaxArgminAttrs, ArgsortAttrs, AstypeAttrs, CallInplacePackedAttrs, CallTIRInplaceAttrs,
      CallTIRWithGradAttrs, ConcatAttrs, EinsumAttrs, ExpandDimsAttrs, FlipAttrs,
      GatherElementsAttrs, GatherNDAttrs, HintOnDeviceAttrs, IndexPutAttrs, InitAttrs,
      LayoutTransformAttrs, MatmulAttrs, MeshgridAttrs, MultinomialFromUniformAttrs,
      OneHotAttrs, PermuteDimsAttrs, QuantizeAttrs, RepeatAttrs, ReverseSequenceAttrs,
      ScanopAttrs, ScatterElementsAttrs, ScatterNDAttrs, SliceScatterAttrs, SortAttrs,
      SplitAttrs, SqueezeAttrs, StackAttrs, StatisticalAttrs, StridedSliceAttrs, TakeAttrs,
      TileAttrs, TopKAttrs, ToVDeviceAttrs, TriluAttrs, WrapParamAttrs

.. autofunction:: tvm.relax.script.function

.. autofunction:: tvm.relax.script.macro

tvm_tirx.script.tirx
********************

Operator attributes re-exported by this namespace are documented in :mod:`tvm_tirx.tirx`.
Shared IR classes and builder infrastructure are documented in :mod:`tvm_tirx.ir` and
:mod:`tvm_tirx.script.ir_builder`.
The canonical ``MemoryScope`` class is documented in :mod:`tvm_tirx.runtime`.

.. automodule:: tvm_tirx.script.tirx
   :members:
   :exclude-members: AlreadyEmitted, Bind, DataTypeImm, FuncType, GenericConst, IRBuilder,
      IRModuleFrame, MemoryScope, MissingType, NoneType, Op, PrimType, StringImm, StringType, TensorLoad, TensorRegion,
      TensorRegionType, Type, Range, meta_var, Var, Call, CommReducer, Reduce, FloatImm, IntImm, Cast, Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, LShift, RShift, BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, Min, Max, EQ, NE, LT, LE, GT, GE, And, Or, Not, Select, Ramp, Broadcast, Shuffle, CallEffectKind, IterVar, ComposeLayout, DtypeConstructor, ThreadScope, Iter, Layout, LetAnnotation, LocalVectorAnnotation, TileLayout, CallFFIKernelAttr, TensorMapEncodeTiledAttr, function, jit, inline, macro

.. autofunction:: tvm_tirx.script.tirx.function

.. autofunction:: tvm_tirx.script.tirx.jit

.. autofunction:: tvm_tirx.script.tirx.inline

.. autofunction:: tvm_tirx.script.tirx.macro
