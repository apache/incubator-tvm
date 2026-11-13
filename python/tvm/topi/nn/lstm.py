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
# ruff: noqa: E731
"""General LSTM implementation using TE scan."""

from tvm import te
from tvm.topi import tag
from tvm_tirx import tirx


def lstm(
    Xs,
    Wi,
    Wh,
    Bi=None,
    Bh=None,
    h_init=None,
    c_init=None,
    proj=None,
    p_i=None,
    p_f=None,
    p_o=None,
    f_act=tirx.sigmoid,
    g_act=tirx.tanh,
    h_act=tirx.tanh,
    reverse=False,
    weight_layout: str = "IFGO",
):
    """General LSTM implemented using TE scan.

    Parameters
    ----------
    Xs : te.Tensor
        Input sequence with shape `(seq_len, batch_size, in_dim)`
    Wi : te.Tensor
        Input weight matrix with shape `(4 * hidden_dim, in_dim)`. The weights are packed according
        to `weight_layout`.
    Wh : te.Tensor
        Hidden weight matrix with shape `(4 * hidden_dim, hidden_dim or proj_dim)`. Packed as `Wh`.
    Bi : te.Tensor, optional
        Input bias with shape `(4 * hidden_dim,)`, by default None. Packed as `Wh`.
    Bh : te.Tensor, optional
        Hidden bias with shape as `Bi`, by default None. Packed as `Wh`.
    h_init : te.Tensor, optional
        Initial hidden state with shape `(batch_size, hidden_dim or proj_dim)`, zero if None
    c_init : te.Tensor, optional
        Initial cell state with same shape as `h_init`, zero if None
    proj : te.Tensor, optional
        Projection matrix with shape `(proj_dim, hidden_dim)`, by default None
    p_i, p_f, p_o : te.Tensor, optional
        Peephole LSTM matrices with shape `(batch_size, hidden_dim)`, by default None
    f_act, g_act, h_act : F, optional
        Gate activation functions
    reverse : bool, optional
        Whether to process `Xs` in reverse, by default False
    weight_layout : str, optional
        The packed weight layout for gates, by default "IFGO". Note: I = input, F = forget,
        G = cell, O = output.

    Returns
    -------
    result : te.Tensor, te.Tensor
        Tuple of hidden states (with shape `(seq_len, batch_size, hidden_dim or proj_dim)`), and
        cell states (with shape `(seq_len, batch_size, hidden_dim)`).
    """
    assert len(weight_layout) == 4 and sorted(weight_layout) == sorted("IFGO"), (
        f'given weight layout "{weight_layout}" is not a permutation of "IFGO"'
    )

    i_gate_idx = weight_layout.find("I")
    f_gate_idx = weight_layout.find("F")
    g_gate_idx = weight_layout.find("G")
    o_gate_idx = weight_layout.find("O")

    seq_len, batch_size, in_dim = Xs.shape
    assert Wi.shape[0] % 4 == 0, (
        f"dim 0 of input weight should be 4 * hidden_dim, but {Wi.shape[0]} is not divisible by 4"
    )
    hidden_dim = Wi.shape[0] // 4
    proj_dim = hidden_dim
    if proj is not None:
        proj_dim = proj.shape[0]

    # te.scan uses up 1 element for the initial value
    scan_len = seq_len + 1

    # precompute input-hidden matmul outside the scan
    ki = te.reduce_axis((0, in_dim), name="ki2h")
    Xi2h = te.compute(
        (seq_len * batch_size, 4 * hidden_dim),
        lambda tb, ij: te.sum(Xs[(tb // batch_size), tb % batch_size, ki] * Wi[ij, ki], axis=ki),
        name="Xi2h",
    )
    if Bi is not None:
        Xi2h = te.compute(
            Xi2h.shape, lambda tb, ij: Xi2h[tb, ij] + Bi[ij], name="Xi2h_bias", tag=tag.INJECTIVE
        )

    h_state = te.placeholder((scan_len, batch_size, proj_dim), name="h_state")
    c_state = te.placeholder((scan_len, batch_size, hidden_dim), name="c_state")
    h_init = te.compute(
        (1, batch_size, proj_dim),
        lambda _, b, i: h_init[b, i] if h_init is not None else 0.0,
        name="h_init",
    )
    c_init = te.compute(
        (1, batch_size, hidden_dim),
        lambda _, b, i: c_init[b, i] if c_init is not None else 0.0,
        name="c_init",
    )

    # begin scan computations, first the (batched) hidden-hidden dense
    kh = te.reduce_axis((0, proj_dim), name="kh2h")
    s_h2h = te.compute(
        (scan_len, batch_size, 4, hidden_dim),
        lambda t, b, i, j: te.sum(h_state[t - 1, b, kh] * Wh[i * hidden_dim + j, kh], axis=kh),
        name="s_h2h",
    )
    if Bh is not None:
        s_h2h = te.compute(
            s_h2h.shape,
            lambda t, b, i, j: s_h2h[t, b, i, j] + Bh[i * hidden_dim + j],
            name="s_h2h_bias",
            tag=tag.INJECTIVE,
        )

    # helper to reverse time if scanning backwards
    get_x_t = lambda t: seq_len - t if reverse else t - 1

    gates = te.compute(
        (scan_len, batch_size, 4, hidden_dim),
        lambda t, b, i, j: (
            Xi2h[get_x_t(t) * batch_size + b, i * hidden_dim + j] + s_h2h[t, b, i, j]
        ),
        name="gates",
        tag=tag.INJECTIVE,
    )

    # helper to correctly read each gate dense from the batched output
    read_gate = lambda t, b, j, idx: gates[t, b, idx, j]

    gate_shape = (scan_len, batch_size, hidden_dim)

    # compute the activated gates (and do some extra stuff if peephole weights are present)
    if p_i is not None and p_f is not None:
        i_gate = te.compute(
            gate_shape,
            lambda t, b, j: f_act(
                read_gate(t, b, j, i_gate_idx) + p_i[b, j] * c_state[t - 1, b, j]
            ),
            name="i_gate_p",
            tag=tag.INJECTIVE,
        )
        f_gate = te.compute(
            gate_shape,
            lambda t, b, j: f_act(
                read_gate(t, b, j, f_gate_idx) + p_f[b, j] * c_state[t - 1, b, j]
            ),
            name="f_gate_p",
            tag=tag.INJECTIVE,
        )
    else:
        i_gate = te.compute(
            gate_shape,
            lambda *i: f_act(read_gate(*i, i_gate_idx)),
            name="i_gate",
            tag=tag.INJECTIVE,
        )
        f_gate = te.compute(
            gate_shape,
            lambda *i: f_act(read_gate(*i, f_gate_idx)),
            name="f_gate",
            tag=tag.INJECTIVE,
        )

    g_gate = te.compute(
        gate_shape, lambda *i: g_act(read_gate(*i, g_gate_idx)), name="g_gate", tag=tag.INJECTIVE
    )

    next_c = te.compute(
        gate_shape,
        lambda t, b, j: f_gate[t, b, j] * c_state[t - 1, b, j] + i_gate[t, b, j] * g_gate[t, b, j],
        name="next_c",
    )

    if p_o is not None:
        o_gate = te.compute(
            gate_shape,
            lambda t, b, j: f_act(read_gate(t, b, j, o_gate_idx) + p_o[b, j] * next_c[t, b, j]),
            name="o_gate_p",
            tag=tag.INJECTIVE,
        )
    else:
        o_gate = te.compute(
            gate_shape,
            lambda *i: f_act(read_gate(*i, o_gate_idx)),
            name="o_gate",
            tag=tag.INJECTIVE,
        )

    next_h = te.compute(gate_shape, lambda *i: o_gate(*i) * h_act(next_c(*i)), name="next_h")

    # project hidden state back to proj_dim if projection matrix is present
    if proj is not None:
        kr = te.reduce_axis((0, hidden_dim), name="kh2p")
        next_h = te.compute(
            (scan_len, batch_size, proj_dim),
            lambda t, b, j: te.sum(next_h[t, b, kr] * proj[j, kr], axis=kr),
            name="next_h_proj",
        )

    scan_h, scan_c = te.scan(
        [h_init, c_init], [next_h, next_c], [h_state, c_state], name="lstm_scan"
    )

    # drop the initial values, TODO(@altanh): is there a better way?
    scan_h = te.compute(
        (seq_len, batch_size, proj_dim), lambda t, b, j: scan_h[t + 1, b, j], name="hidden_states"
    )
    scan_c = te.compute(
        (seq_len, batch_size, hidden_dim), lambda t, b, j: scan_c[t + 1, b, j], name="cell_states"
    )

    return scan_h, scan_c
