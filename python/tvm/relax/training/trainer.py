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
# pylint: disable=invalid-name
"""Unified Trainer API for relax training."""

import numpy as np  # type: ignore

import tvm_tirx
from tvm import relax
from tvm_tirx.ir.module import IRModule
from tvm_tirx.runtime._tensor import Tensor


class Trainer:
    r"""Unified wrapper for relax training. It accepts the IRModule (that is the result of
    SetupTrainer) and the relax VM (that contains the built result of the IRModule), and helps run
    the VM. It maintains the parameters, the model states and the optimizer states internally.

    Parameters
    ----------
    train_mod : tvm_tirx.IRModule
        The IRModule that will be run. Should be the result of a backbone module being transformed
        by the SetupTrainer pass.

    vm : tvm.relax.VirtualMachine
        The relax virtual machine that contains the built result of train_mod. Considering the
        complexity and flexibility of building, we require user build the train_mod outside of
        trainer and pass the result vm.

    device : tvm_tirx.runtime.Device
        The device to place the parameters and states in.

    zero_init_param_state : bool
        If true, all parameters and states will be inited to zero. It requires all parameters and
        states have static shape.

    Examples
    --------
    .. code-block:: python

        setup_trainer = SetupTrainer(
            MSELoss(reduction="sum"),
            SGD(0.001),
            [pred_ty, target_ty],
        )
        train_mod = setup_trainer(Backbone)
        ex = tvm.compile(train_mod, target)
        vm = relax.VirtualMachine(ex, dev)

        trainer = training.Trainer(train_mod, vm, dev, False)

        trainer.xaiver_uniform_init_params()
        trainer.predict(input_instances)
        trainer.update([input_instances], [labels])
    """

    BACKBONE_FUNC: str = "backbone"
    BACKBONE_LOSS_FUNC: str = "backbone_loss"
    ADJOINT_FUNC: str = "backbone_loss_adjoint"
    OPTIMIZER_FUNC: str = "optimizer"

    def __init__(
        self,
        train_mod: IRModule,
        vm: relax.VirtualMachine,
        device: tvm_tirx.runtime.Device,
        zero_init_param_state: bool = True,
    ) -> None:
        self.mod = train_mod.without_attr("optim_state")
        self.vm = vm
        self.device = device

        self._optim_state = [d.copyto(device) for d in train_mod.attrs["optim_state"]]

        self._input_num = int(train_mod.attrs["input_num"])
        self._param_num = int(train_mod.attrs["param_num"])
        self._state_num = int(train_mod.attrs["state_num"])

        # are used to initialize params and states
        self._param_vars = train_mod[self.ADJOINT_FUNC].params[
            self._input_num : self._input_num + self._param_num
        ]
        self._state_vars = train_mod[self.ADJOINT_FUNC].params[
            (self._input_num + self._param_num) : (
                self._input_num + self._param_num + self._state_num
            )
        ]

        self._params: list[Tensor | None] = [None] * self._param_num
        self._param_name_to_pos: dict[str, int] = {
            p.name: i for i, p in enumerate(self._param_vars)
        }

        self._states: list[Tensor | None] = [None] * self._state_num
        self._state_name_to_pos: dict[str, int] = {
            s.name: i for i, s in enumerate(self._state_vars)
        }

        if zero_init_param_state:
            self.zero_init_params()
            self.zero_init_states()

    @staticmethod
    def _get_shape_list(expr):
        return [int(dim) for dim in expr.ty.shape]

    def xaiver_uniform_init_params(self):
        """Xaiver uniformly initialize parameters using the method described in `Understanding the
        difficulty of training deep feedforward neural networks` - Glorot, X. & Bengio, Y.
        (2010).

        Requires all parameters have static shapes.
        """
        self._params = []
        for p in self._param_vars:
            shape, dtype = self._get_shape_list(p), p.ty.dtype
            self._params.append(
                tvm_tirx.runtime.tensor(
                    (np.sqrt(6.0 / np.sum(shape)) * np.random.uniform(-1.0, 1.0, shape)).astype(
                        dtype
                    ),
                    self.device,
                )
            )

    def zero_init_params(self):
        """Zero initialize all parameters. Requires all parameters have static shapes."""
        self._params = [
            tvm_tirx.runtime.tensor(
                np.zeros(self._get_shape_list(p), p.ty.dtype.dtype), self.device
            )
            for p in self._param_vars
        ]

    def zero_init_states(self):
        """Zero initialize all states. Requires all states have static shapes."""
        self._states = [
            tvm_tirx.runtime.tensor(
                np.zeros(self._get_shape_list(s), s.ty.dtype.dtype), self.device
            )
            for s in self._state_vars
        ]

    def load_params(
        self,
        params: list[np.ndarray | Tensor] | dict[str, np.ndarray | Tensor],
    ):
        """Load parameters from a dict or a list.

        Will convert parameters into tvm_tirx.runtime.Tensor in self.device.

        Parameters
        ----------
        params : List[Union[np.ndarray, Tensor]], Dict[str, Union[np.ndarray, Tensor]]
            The numerical value of the parameters.

            If params is a list, its length should be param_num. The value of parameters at the
            corresponding index will be updated.

            If params is a dict, it should map variable name to value. The name should be the same
            as the parameter name in the backbone function. The values of the corresponding
            parameters will be updated.
        """
        if isinstance(params, list):
            if len(params) != self._param_num:
                raise ValueError(
                    f"The length of extern parameters is {len(params)}, which does not "
                    f"match the number of parameters {self._param_num}"
                )
            self._params = [tvm_tirx.runtime.tensor(v, self.device) for v in params]
        elif isinstance(params, dict):
            for key, val in params.items():
                if key not in self._param_name_to_pos:
                    raise ValueError(f"Parameter {key} is not found in the model")
                self._params[self._param_name_to_pos[key]] = tvm_tirx.runtime.tensor(
                    val, self.device
                )
        else:
            raise ValueError("The type of extern_params should be either list or dict")

    def load_states(
        self,
        states: list[np.ndarray | Tensor] | dict[str, np.ndarray | Tensor],
    ):
        """Load model states from a dict or a list. Will convert states into tvm_tirx.runtime.Tensor
        in self.device.

        Parameters
        ----------
        states : List[Union[np.ndarray, Tensor]], Dict[str, Union[np.ndarray, Tensor]]
            The numerical value of the model states.

            If states is a list, its length should be state_num. The value of states at the
            corresponding index will be updated.

            If params is a dict, it should map variable name to value. The name should be the same
            as the state name in the backbone function. The values of the corresponding states will
            be updated.
        """
        if isinstance(states, list):
            if len(states) != self._state_num:
                raise ValueError(
                    f"The length of extern states is {len(states)}, which does not match "
                    f"the number of model states {self._state_num}"
                )
            self._states = [tvm_tirx.runtime.tensor(v, self.device) for v in states]
        elif isinstance(states, dict):
            for key, val in states.items():
                if key not in self._param_name_to_pos:
                    raise ValueError(f"Parameter {key} is not found in the model")
                self._states[self._param_name_to_pos[key]] = tvm_tirx.runtime.tensor(
                    val, self.device
                )
        else:
            raise ValueError("The type of extern_states should be either list or dict")

    def export_params(self) -> dict[str, Tensor]:
        """Export parameters to a dict (parameter name -> Tensor).

        Returns
        -------
        exported_dict : Dict[str, Tensor]
            The exported dictionary of parameters.
        """
        return {key: self._params[pos] for key, pos in self._param_name_to_pos.items()}

    def export_states(self) -> dict[str, Tensor]:
        """Export model states to a dict (parameter name -> Tensor).

        Returns
        -------
        exported_dict : Dict[str, Tensor]
            The exported dictionary of model states.
        """
        return {key: self._states[pos] for key, pos in self._state_name_to_pos.items()}

    def _check_inited(self):
        """Check that all parameters and model states are initialized."""
        idx_not_inited_param = next((i for i, p in enumerate(self._params) if p is None), -1)
        if idx_not_inited_param != -1:
            raise RuntimeError(
                f"The {idx_not_inited_param}-th parameter is not initialized before training or "
                "inference."
            )

        idx_not_inited_state = next((i for i, s in enumerate(self._states) if s is None), -1)
        if idx_not_inited_state != -1:
            raise RuntimeError(
                f"The {idx_not_inited_state}-th model state is not initialized before training or "
                "inference."
            )

    def predict(self, *input_instances: np.ndarray | Tensor) -> Tensor:
        """Call the `backbone` function and return the prediction result of the backbone.

        Parameters
        ----------
        *input_instances : Union[np.ndarray, Tensor]
            The values corresponding to the input_instances part of the backbone function.
            Parameters and model states are not needed to provide.

        Returns
        -------
        output : Tensor
            The result of the backbone function. If the backbone contains model states, the updated
            states WILL NOT be returned.
        """
        self._check_inited()
        if len(input_instances) != self._input_num:
            raise ValueError("The length of the input does not match the backbone")
        all_inputs: list[Tensor] = (
            [tvm_tirx.runtime.tensor(i, self.device) for i in input_instances]
            + self._params
            + self._states
        )
        res = self.vm[self.BACKBONE_FUNC](*all_inputs)

        # remove the states part, if they exist
        if self._state_num != 0:
            res = res[: -self._state_num]
            if len(res) == 1:
                res = res[0]
        return res

    def update(
        self,
        input_instances: np.ndarray | Tensor | list[np.ndarray | Tensor],
        targets: np.ndarray | Tensor | list[np.ndarray | Tensor],
    ) -> Tensor:
        """Update parameters and model states. It will calculate the gradients of parameters
        and update them using the `optimizer` function.

        Parameters, model states and optimizer states are provided in the function, so you do not
        need to provied them.

        Parameters
        ----------
        input_instances : Union[np.ndarray, Tensor, List[Union[np.ndarray, Tensor]]]
            The values corresponding to the input_instances part of the backbone function.
            Parameters and model states are not needed to provide.

            If there are more than one input instances, you can provide a list.

        targets : Union[np.ndarray, Tensor, List[Union[np.ndarray, Tensor]]]
            The values corresponding to the targets part of the backbone function.

            If there are more than one targets, you can provide a list.

        Returns
        -------
        loss : Tensor
            The loss stored in tvm_tirx.runtime.Tensor.
        """
        self._check_inited()

        if not isinstance(input_instances, list):
            input_instances = [input_instances]

        if not isinstance(targets, list):
            targets = [targets]

        if len(input_instances) != self._input_num:
            raise ValueError("The length of the input does not match the backbone")

        all_inputs: list[Tensor] = (
            [tvm_tirx.runtime.tensor(i, self.device) for i in input_instances]
            + self._params
            + self._states
            + [tvm_tirx.runtime.tensor(i, self.device) for i in targets]
        )
        ret, grads = self.vm[self.ADJOINT_FUNC](*all_inputs)

        # update model states
        if self._state_num != 0:
            self._states = list(ret[1:])
            ret = ret[0]

        # update params
        new_params, self._optim_state = self.vm[self.OPTIMIZER_FUNC](
            self._params, grads, self._optim_state
        )
        self._params = list(new_params)

        return ret
