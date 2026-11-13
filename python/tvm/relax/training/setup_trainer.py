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
# pylint: disable=not-callable, unused-argument
"""Setup Trainer Pass."""

import tvm_tirx
from tvm_tirx.ir import IntImm
from tvm_tirx.ir.module import IRModule

from ..analysis import check_well_formed
from ..expr import Tuple
from ..training.utils import AppendLoss
from ..transform import DecomposeOpsForInference, DecomposeOpsForTraining, Gradient, LegalizeOps
from ..type import TensorType
from .loss import Loss
from .optimizer import Optimizer


@tvm_tirx.ir.transform.module_pass(opt_level=0, name="SetupTrainer")
class SetupTrainer:
    """Transform a backbone module to a complete, legalized trainer module.

    The provided backbone module should contain at least a function named `backbone`, and has two
    int attributes `param_num` and `state_num`, as follows:

    .. code-block:: python

        @I.ir_module
        class Backbone:
            I.module_attrs({"param_num": 1, "state_num": 1})
            @R.function
            def backbone(input_instances, parameters, states):
                # Predicts the result
                # Should contain only one DataflowBlock
                ...
                return backbone_result, updated_states

    Here each of input_instances, parameters, states, backbone_result and updated_states can
    denote a number of parameters. The length of parameters and the length of states is specified
    by param_num and state_num respectively.

    `states` denote the states that we need to maintain as the training process proceeds, such as
    the running mean and the running var of the batch norm operator. The updated states is returned
    in `updated_states`. States can be empty if there is no state that needs to be updated.

    The transformed module will at least contain the functions and attributes listed below:

    .. code-block:: python

        @I.ir_module
        class Module:
            I.module_attrs({"input_num": 1, "param_num": 1, "state_num": 1, "optim_states": ...})

            @R.function
            def backbone(input_instances, parameters, states):
                # Predicts the result. It is provided in the input module.
                ...
                return backbone_result, updated_states

            @R.function
            def backbone_loss(input_instances, parameters, states, targets):
                # Runs like backbone and then computes the loss between the result and targets.
                ...
                return loss, updated_states

            @R.function
            def backbone_loss_adjoint(input_instances, parameters, states, targets):
                # Runs like backbone_loss and then calculates the gradient of parameters.
                ...
                return (loss, updated_states), gradient_of_params

            @R.function
            def optimizer(params, gradients, optim_states):
                # Update parameters and optimizer states with the gradient computed
                ...
                return (updated_params, updated_optim_states)

    The transformed module contains an attribute `optim_states` as the initial optimizer states.

    Then the transformed module will be legalized by `relax.transform.LegalizeOps()` to lower
    relax operators into TIR functions.

    Parameters
    ----------
    loss : Loss
        The loss function. It will be appended to the backbone function using
        relax.transform.AppendLoss.

    optimizer : Optimizer
        The optimizer. It will be put as the `optimizer` function of the transformed module.

    loss_args : List[TensorType]
        The arguments to call the loss function.

    legalize : bool
        Whether to legalize the module. Default: True.
    """

    BACKBONE_FUNC: str = "backbone"
    BACKBONE_LOSS_FUNC: str = "backbone_loss"
    ADJOINT_FUNC: str = "backbone_loss_adjoint"
    OPTIMIZER_FUNC: str = "optimizer"

    PARAM_NUM_ATTR_KEY: str = "param_num"
    STATE_NUM_ATTR_KEY: str = "state_num"

    def __init__(
        self, loss: Loss, optimizer: Optimizer, loss_args: list[TensorType], legalize=True
    ):
        self._loss = loss
        self._optimizer = optimizer
        self._loss_args = loss_args
        self._legalize = legalize

    def _check_well_formed(self, mod: IRModule):
        check_well_formed(mod)
        try:
            func = mod[self.BACKBONE_FUNC]
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"SetupTrainer: The backbone module does not contain a function named "
                f"{self.BACKBONE_FUNC}"
            ) from exc

        # Check function attrs
        if self.PARAM_NUM_ATTR_KEY not in mod.attrs or not isinstance(
            mod.attrs[self.PARAM_NUM_ATTR_KEY], IntImm | int
        ):
            raise ValueError(
                f"SetupTrainer: The backbone module should has an integer attribute named "
                f"{self.PARAM_NUM_ATTR_KEY}"
            )
        if self.STATE_NUM_ATTR_KEY not in mod.attrs or not isinstance(
            mod.attrs[self.STATE_NUM_ATTR_KEY], IntImm | int
        ):
            raise ValueError(
                f"SetupTrainer: The backbone module should has an integer attribute named "
                f"{self.STATE_NUM_ATTR_KEY}"
            )

        nparam = int(mod.attrs[self.PARAM_NUM_ATTR_KEY])
        nstate = int(mod.attrs[self.STATE_NUM_ATTR_KEY])

        # Check parameters and return values
        if len(func.params) < nparam + nstate:
            raise ValueError(
                "SetupTrainer: The number of parameters of the predict function should be no less "
                "than the number of parameters and states"
            )

        if nstate > 0:
            if not isinstance(func.body.body, Tuple) or len(func.body.body) <= nstate:
                raise ValueError(
                    "SetupTrainer: When model state exists, the predict function should return a "
                    "tuple of length more than the number of states"
                )

    def transform_module(self, mod: IRModule, ctx: tvm_tirx.ir.transform.PassContext) -> IRModule:
        """Transform the backbone module into a trainer module."""
        self._check_well_formed(mod)

        mod = AppendLoss(
            self.BACKBONE_FUNC,
            self._loss(*self._loss_args),  # type: ignore
            self._loss.num_backbone_outputs,
            self.BACKBONE_LOSS_FUNC,
        )(mod)

        # Decompose batch_norm operator, which behaves differently in inference and training stages
        mod = DecomposeOpsForInference(self.BACKBONE_FUNC)(mod)
        mod = DecomposeOpsForTraining(self.BACKBONE_LOSS_FUNC)(mod)

        # Gradient pass.
        param_num = int(mod.attrs[self.PARAM_NUM_ATTR_KEY])
        state_num = int(mod.attrs[self.STATE_NUM_ATTR_KEY])
        input_num = len(mod[self.BACKBONE_FUNC].params) - param_num - state_num
        params = mod[self.BACKBONE_LOSS_FUNC].params[input_num : input_num + param_num]
        mod = Gradient(self.BACKBONE_LOSS_FUNC, require_grads=params, target_index=0)(mod)

        # Add optimizer function.
        self._optimizer.init(params)
        # Need the global symbol to match the function's name
        mod[self.OPTIMIZER_FUNC] = self._optimizer.get_function().with_attr(
            "global_symbol", self.OPTIMIZER_FUNC
        )

        # Module attrs
        mod = mod.with_attrs(
            {
                "input_num": input_num,
                "optim_state": self._optimizer.state,
            }
        )

        if self._legalize:
            mod = LegalizeOps()(mod)

        return mod
