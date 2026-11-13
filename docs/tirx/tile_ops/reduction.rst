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

Reduction Instructions
======================

CUDA reduction algorithms belong to the caller. Use scalar or packed
arithmetic, warp shuffle instructions, shared storage, and explicit
synchronization as appropriate for the kernel's decomposition.

Trainium exposes the native ``T.trn.tile.tensorreduce`` instruction and the
fused ``tensorscalar_reduce`` and ``activation_reduce`` instructions. Their
axes and opcode are static qualifiers; tensors and optional reduction
workspace are ordinary Call arguments.

See :doc:`../tile_ops` for the complete contract.
