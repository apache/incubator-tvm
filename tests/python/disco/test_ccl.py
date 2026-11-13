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
# pylint: disable=missing-docstring
"""Tests for NCCL/RCCL"""

import tempfile

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax as rx
from tvm.relax import script as R
from tvm.runtime import disco as di
from tvm.runtime.vm import VirtualMachine
from tvm.s_tir import dlight as dl

if di is None:
    pytest.skip("disco runtime is not available", allow_module_level=True)

_all_session_kinds = [di.ThreadedSession, di.ProcessSession]
_compiled_ccl = tvm_ffi.get_global_func("runtime.disco.compiled_ccl", allow_missing=True)
if _compiled_ccl is None:
    pytest.skip("Disco CCL is not enabled in this TVM build", allow_module_level=True)
_ccl = [_compiled_ccl()]


def create_device_target(ccl):
    if ccl == "nccl":
        dev = tvm_tirx.cuda(0)
    else:
        dev = tvm_tirx.rocm(0)
    target = tvm_tirx.target.Target.from_device(dev)
    return (dev, target)


def _run_with_ccl_session(session_kind, ccl, devices, func, *, num_groups=1):
    def run_and_check():
        sess = session_kind(num_workers=len(devices), num_groups=num_groups)
        try:
            sess.init_ccl(ccl, *devices)
            return func(sess)
        finally:
            sess.shutdown()

    return tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_init(session_kind, ccl):
    devices = [0, 1]

    def run_test(_sess):
        pass

    _run_with_ccl_session(session_kind, ccl, devices, run_test)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_allreduce(session_kind, ccl):
    devices = [0, 1]
    array_1 = np.arange(12, dtype="float32").reshape(3, 4)
    array_2 = np.arange(start=1, stop=-11, step=-1, dtype="float32").reshape(3, 4)

    def run_test(sess):
        d_array = sess.empty((3, 4), "float32")
        d_array.debug_copy_from(0, array_1)
        d_array.debug_copy_from(1, array_2)
        for op, np_op in [  # pylint: disable=invalid-name
            ("sum", np.add),
            ("prod", np.multiply),
            ("min", np.minimum),
            ("max", np.maximum),
            ("avg", lambda a, b: (a + b) * 0.5),
        ]:
            dst_array = sess.empty((3, 4), "float32")
            sess.allreduce(d_array, dst_array, op=op)
            result = dst_array.debug_get_from_remote(0).numpy()
            expected = np_op(array_1, array_2)
            np.testing.assert_equal(result, expected)

    _run_with_ccl_session(session_kind, ccl, devices, run_test)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_group_allreduce(session_kind, ccl):
    devices = [0, 1, 2, 3]
    array_1 = np.arange(12, dtype="float32").reshape(3, 4)
    array_2 = np.arange(start=1, stop=-11, step=-1, dtype="float32").reshape(3, 4)
    array_3 = np.arange(30, dtype="float32").reshape(5, 6)
    array_4 = np.arange(start=1, stop=-29, step=-1, dtype="float32").reshape(5, 6)

    def run_test(sess):
        d_array_1 = sess.empty((3, 4), "float32")
        d_array_2 = sess.empty((5, 6), "float32")
        d_array_1.debug_copy_from(0, array_1)
        d_array_1.debug_copy_from(1, array_2)
        d_array_2.debug_copy_from(2, array_3)
        d_array_2.debug_copy_from(3, array_4)
        for op, np_op in [  # pylint: disable=invalid-name
            ("sum", np.add),
            ("prod", np.multiply),
            ("min", np.minimum),
            ("max", np.maximum),
            ("avg", lambda a, b: (a + b) * 0.5),
        ]:
            dst_array_1 = sess.empty((3, 4), "float32")
            dst_array_2 = sess.empty((5, 6), "float32")
            sess.allreduce(d_array_1, dst_array_1, op=op, in_group=True)
            sess.allreduce(d_array_2, dst_array_2, op=op, in_group=True)
            result_1 = dst_array_1.debug_get_from_remote(0).numpy()
            result_2 = dst_array_2.debug_get_from_remote(2).numpy()
            expected_1 = np_op(array_1, array_2)
            expected_2 = np_op(array_3, array_4)
            np.testing.assert_equal(result_1, expected_1)
            np.testing.assert_equal(result_2, expected_2)

    _run_with_ccl_session(session_kind, ccl, devices, run_test, num_groups=2)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_allgather(session_kind, ccl):
    devices = [0, 1]
    array = np.arange(36, dtype="float32")

    def run_test(sess):
        d_src = sess.empty((3, 3, 2), "float32")
        d_dst = sess.empty((3, 4, 3), "float32")
        d_src.debug_copy_from(0, array[:18])
        d_src.debug_copy_from(1, array[18:])
        sess.allgather(d_src, d_dst)
        np.testing.assert_equal(
            d_dst.debug_get_from_remote(0).numpy(),
            array.reshape(3, 4, 3),
        )
        np.testing.assert_equal(
            d_dst.debug_get_from_remote(1).numpy(),
            array.reshape(3, 4, 3),
        )

    _run_with_ccl_session(session_kind, ccl, devices, run_test)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_group_allgather(session_kind, ccl):
    devices = [0, 1, 2, 3]
    array_1 = np.arange(36, dtype="float32")
    array_2 = np.arange(48, dtype="float32")

    def run_test(sess):
        d_src_1 = sess.empty((3, 3, 2), "float32")
        d_dst_1 = sess.empty((3, 4, 3), "float32")
        d_src_2 = sess.empty((2, 4, 3), "float32")
        d_dst_2 = sess.empty((2, 6, 4), "float32")
        d_src_1.debug_copy_from(0, array_1[:18])
        d_src_1.debug_copy_from(1, array_1[18:])
        d_src_2.debug_copy_from(2, array_2[:24])
        d_src_2.debug_copy_from(3, array_2[24:])
        sess.allgather(d_src_1, d_dst_1, in_group=True)
        sess.allgather(d_src_2, d_dst_2, in_group=True)
        np.testing.assert_equal(
            d_dst_1.debug_get_from_remote(0).numpy(),
            array_1.reshape(3, 4, 3),
        )
        np.testing.assert_equal(
            d_dst_1.debug_get_from_remote(1).numpy(),
            array_1.reshape(3, 4, 3),
        )
        np.testing.assert_equal(
            d_dst_2.debug_get_from_remote(2).numpy(),
            array_2.reshape(2, 6, 4),
        )
        np.testing.assert_equal(
            d_dst_2.debug_get_from_remote(3).numpy(),
            array_2.reshape(2, 6, 4),
        )

    _run_with_ccl_session(session_kind, ccl, devices, run_test, num_groups=2)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
@pytest.mark.parametrize("use_explicit_output", [True, False])
def test_broadcast(session_kind, ccl, use_explicit_output):
    devices = [0, 1]
    array = np.arange(12, dtype="float32").reshape(3, 4)

    def run_test(sess):
        if use_explicit_output:
            src_array = sess.empty((3, 4), "float32", worker0_only=True)
            src_array.debug_copy_from(0, array)
            dst_array = sess.empty((3, 4), "float32")
            sess.broadcast_from_worker0(src_array, dst_array)
        else:
            dst_array = sess.broadcast(array)

        result = dst_array.debug_get_from_remote(1).numpy()
        np.testing.assert_equal(result, array)

    _run_with_ccl_session(session_kind, ccl, devices, run_test)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_group_broadcast(session_kind, ccl):
    devices = [0, 1, 2, 3]
    array_1 = np.arange(12, dtype="float32").reshape(3, 4)
    array_2 = np.multiply(array_1, -1)

    def run_test(sess):
        src_array = sess.empty((3, 4), "float32", worker0_only=True, in_group=True)
        src_array.debug_copy_from(0, array_1)
        src_array.debug_copy_from(2, array_2)
        dst_array = sess.empty((3, 4), "float32")
        sess.broadcast_from_worker0(src_array, dst_array)

        result_1 = dst_array.debug_get_from_remote(1).numpy()
        np.testing.assert_equal(result_1, array_1)

        result_3 = dst_array.debug_get_from_remote(3).numpy()
        np.testing.assert_equal(result_3, array_2)

    _run_with_ccl_session(session_kind, ccl, devices, run_test, num_groups=2)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
@pytest.mark.parametrize("use_explicit_output", [True, False])
def test_scatter(session_kind, ccl, use_explicit_output, capfd):
    devices = [0, 1]
    array = np.arange(36, dtype="float32").reshape(2, 6, 3)

    def run_test(sess):
        if use_explicit_output:
            d_src = sess.empty((2, 6, 3), "float32", worker0_only=True)
            d_dst = sess.empty((6, 3), "float32")
            d_src.debug_copy_from(0, array)
            sess.scatter_from_worker0(d_src, d_dst)
        else:
            d_dst = sess.scatter(array)

        np.testing.assert_equal(
            d_dst.debug_get_from_remote(0).numpy(),
            array[0, :, :],
        )
        np.testing.assert_equal(
            d_dst.debug_get_from_remote(1).numpy(),
            array[1, :, :],
        )

        captured = capfd.readouterr()
        assert not captured.err, (
            "No warning messages should be generated from disco.Session.scatter_from_worker0"
        )

    _run_with_ccl_session(session_kind, ccl, devices, run_test)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_group_scatter(session_kind, ccl, capfd):
    devices = [0, 1, 2, 3]
    array_1 = np.arange(36, dtype="float32").reshape(2, 6, 3)
    array_2 = np.multiply(array_1, -1)

    def run_test(sess):
        d_src = sess.empty((2, 6, 3), "float32", worker0_only=True, in_group=True)
        d_src.debug_copy_from(0, array_1)
        d_src.debug_copy_from(2, array_2)
        d_dst = sess.empty((6, 3), "float32")
        sess.scatter_from_worker0(d_src, d_dst)

        np.testing.assert_equal(
            d_dst.debug_get_from_remote(0).numpy(),
            array_1[0, :, :],
        )
        np.testing.assert_equal(
            d_dst.debug_get_from_remote(1).numpy(),
            array_1[1, :, :],
        )
        np.testing.assert_equal(
            d_dst.debug_get_from_remote(2).numpy(),
            array_2[0, :, :],
        )
        np.testing.assert_equal(
            d_dst.debug_get_from_remote(3).numpy(),
            array_2[1, :, :],
        )

        captured = capfd.readouterr()
        assert not captured.err, (
            "No warning messages should be generated from disco.Session.scatter_from_worker0"
        )

    _run_with_ccl_session(session_kind, ccl, devices, run_test, num_groups=2)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_scatter_with_implicit_reshape(session_kind, ccl, capfd):
    """Scatter may perform an implicit reshape

    Scattering elements to the workers requires the total number of
    elements to be divisible by the number of workers.  It does not
    necessarily correspond to scattering across the outermost
    dimension.  Here, the number of workers (2) and the outermost
    dimension (3) are not divisible, but the scatter may still be
    performed.

    This is only allowed when the caller explicitly uses the
    `sess.scatter_from_worker0` method, and is not allowed in
    `sess.scatter` method.  Because the `sess.scatter` method may
    perform an allocation on the disco workers, it requires that the
    scatter occur across the outermost dimension.

    """
    devices = [0, 1]
    array = np.arange(36, dtype="float32").reshape(3, 4, 3)

    def run_test(sess):
        d_src = sess.empty((3, 4, 3), "float32", worker0_only=True)
        d_dst = sess.empty((3, 3, 2), "float32")
        d_src.debug_copy_from(0, array)
        sess.scatter_from_worker0(d_src, d_dst)

        np.testing.assert_equal(
            d_dst.debug_get_from_remote(0).numpy(),
            array.flat[:18].reshape(3, 3, 2),
        )
        np.testing.assert_equal(
            d_dst.debug_get_from_remote(1).numpy(),
            array.flat[18:].reshape(3, 3, 2),
        )

        captured = capfd.readouterr()
        assert not captured.err, (
            "No warning messages should be generated from disco.Session.scatter_from_worker0"
        )

    _run_with_ccl_session(session_kind, ccl, devices, run_test)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_gather(session_kind, ccl, capfd):
    devices = [0, 1]
    array = np.arange(36, dtype="float32")

    def run_test(sess):
        d_src = sess.empty((3, 3, 2), "float32")
        d_dst = sess.empty((3, 4, 3), "float32", worker0_only=True)
        d_src.debug_copy_from(0, array[:18])
        d_src.debug_copy_from(1, array[18:])
        sess.gather_to_worker0(d_src, d_dst)
        np.testing.assert_equal(
            d_dst.debug_get_from_remote(0).numpy(),
            array.reshape(3, 4, 3),
        )

        captured = capfd.readouterr()
        assert not captured.err, (
            "No warning messages should be generated from disco.Session.gather_to_worker0"
        )

    _run_with_ccl_session(session_kind, ccl, devices, run_test)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_group_gather(session_kind, ccl, capfd):
    devices = [0, 1, 2, 3]
    array_1 = np.arange(36, dtype="float32")
    array_2 = np.multiply(array_1, -1)

    def run_test(sess):
        d_src = sess.empty((3, 3, 2), "float32")
        d_dst = sess.empty((3, 4, 3), "float32", worker0_only=True, in_group=True)
        d_src.debug_copy_from(0, array_1[:18])
        d_src.debug_copy_from(1, array_1[18:])
        d_src.debug_copy_from(2, array_2[:18])
        d_src.debug_copy_from(3, array_2[18:])
        sess.gather_to_worker0(d_src, d_dst)
        np.testing.assert_equal(
            d_dst.debug_get_from_remote(0).numpy(),
            array_1.reshape(3, 4, 3),
        )
        np.testing.assert_equal(
            d_dst.debug_get_from_remote(2).numpy(),
            array_2.reshape(3, 4, 3),
        )

        captured = capfd.readouterr()
        assert not captured.err, (
            "No warning messages should be generated from disco.Session.gather_to_worker0"
        )

    _run_with_ccl_session(session_kind, ccl, devices, run_test, num_groups=2)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_send_to_next_group_receive_from_prev_group(session_kind, ccl):
    devices = [0, 1, 2, 3]
    array_1 = np.arange(12, dtype="float32").reshape(3, 4)
    array_2 = np.arange(start=1, stop=-11, step=-1, dtype="float32").reshape(3, 4)

    def run_test(sess):
        d_array = sess.empty((3, 4), "float32")
        d_array.debug_copy_from(0, array_1)
        d_array.debug_copy_from(1, array_2)
        sess.get_global_func(
            "runtime.disco." + ccl + ".test_send_to_next_group_recv_from_prev_group"
        )(d_array)

        result_1 = d_array.debug_get_from_remote(2).numpy()
        result_2 = d_array.debug_get_from_remote(3).numpy()
        np.testing.assert_equal(result_1, array_1)
        np.testing.assert_equal(result_2, array_2)

    _run_with_ccl_session(session_kind, ccl, devices, run_test, num_groups=2)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_worker2_send_to_worker0(session_kind, ccl):
    devices = [0, 1, 2, 3]
    array = np.arange(start=1, stop=-11, step=-1, dtype="float32").reshape(3, 4)

    def run_test(sess):
        d_array = sess.empty((3, 4), "float32")
        d_array.debug_copy_from(2, array)
        sess.get_global_func("runtime.disco." + ccl + ".test_worker2_sends_to_worker0")(d_array)

        result = d_array.debug_get_from_remote(0).numpy()
        np.testing.assert_equal(result, array)

    _run_with_ccl_session(session_kind, ccl, devices, run_test, num_groups=2)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_mlp(session_kind, ccl):  # pylint: disable=too-many-locals
    devices = [0, 1]

    # pylint: disable=invalid-name
    @tvm_tirx.script.ir_module
    class MLP:  # pylint: disable=too-few-public-methods
        @R.function
        def main(
            x: R.Tensor((128, 128), "float32"),
            W1: R.Tensor((128, 128), "float32"),
            W2: R.Tensor((128, 128), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                lv0: R.Tensor((128, 128), "float32") = R.matmul(x, W1)
                lv1: R.Tensor((128, 128), "float32") = R.nn.gelu(lv0)
                lv2: R.Tensor((128, 128), "float32") = R.matmul(lv1, W2)
                R.output(lv2)
            return lv2

    @tvm_tirx.script.ir_module
    class ShardedMLP:  # pylint: disable=too-few-public-methods
        @R.function
        def main(
            x: R.Tensor((128, 128), "float32"),
            W1: R.Tensor((128, 64), "float32"),  # shard along axis 1
            W2: R.Tensor((64, 128), "float32"),  # shard along axis 0
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                broadcast_x: R.Tensor((128, 128), "float32") = R.ccl.broadcast_from_worker0(x)
                lv0: R.Tensor((128, 64), "float32") = R.matmul(broadcast_x, W1)
                lv1: R.Tensor((128, 64), "float32") = R.nn.gelu(lv0)
                lv2: R.Tensor((128, 128), "float32") = R.matmul(lv1, W2)
                lv3: R.Tensor((128, 128), "float32") = R.ccl.allreduce(lv2, "sum")
                R.output(lv3)
            return lv3

    # pylint: enable=invalid-name
    dev, target = create_device_target(ccl)

    def relax_build(mod, target):
        mod = rx.get_pipeline("zero", target=target)(mod)  # pylint: disable=no-value-for-parameter
        mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
            target=target,
        )(mod)
        return tvm.compile(mod, target=target)

    # pylint: disable=invalid-name
    X = np.random.randn(128, 128).astype("float32")
    W1 = np.random.randn(128, 128).astype("float32")
    W2 = np.random.randn(128, 128).astype("float32")
    expected_ex = relax_build(MLP, target)
    sharded_ex = relax_build(ShardedMLP, target)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.so"
        sharded_ex.export_library(path)

        def run_test(sess):
            Y_expected = VirtualMachine(expected_ex, device=dev)["main"](
                tvm_tirx.runtime.tensor(X, device=dev),
                tvm_tirx.runtime.tensor(W1, device=dev),
                tvm_tirx.runtime.tensor(W2, device=dev),
            ).numpy()

            mod = sess.load_vm_module(path)
            d_X = sess.empty((128, 128), "float32")
            d_W1 = sess.empty((128, 64), "float32")
            d_W2 = sess.empty((64, 128), "float32")

            d_X.debug_copy_from(0, X)
            d_W1.debug_copy_from(0, W1[:, :64])
            d_W1.debug_copy_from(1, W1[:, 64:])
            d_W2.debug_copy_from(0, W2[:64, :])
            d_W2.debug_copy_from(1, W2[64:, :])
            d_Y = mod["main"](d_X, d_W1, d_W2)
            Y_result = tvm_tirx.runtime.empty((128, 128), "float32", device=dev)
            sess.copy_from_worker_0(Y_result, d_Y)
            sess.sync_worker_0()
            Y_result = Y_result.numpy()
            tvm_tirx.testing.assert_allclose(Y_result, Y_expected, rtol=1e-4, atol=1e-4)

        _run_with_ccl_session(session_kind, ccl, devices, run_test)
    # pylint: enable=invalid-name


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_attention(session_kind, ccl):  # pylint: disable=too-many-locals,too-many-statements
    devices = [0, 1]

    # pylint: disable=invalid-name
    @tvm_tirx.script.ir_module
    class Attention:  # pylint: disable=too-few-public-methods
        @R.function
        def main(  # pylint: disable=too-many-locals
            x: R.Tensor((1, 10, 128), "float32"),
            Wq: R.Tensor((128, 512), "float32"),
            Wk: R.Tensor((128, 512), "float32"),
            Wv: R.Tensor((128, 512), "float32"),
            Wo: R.Tensor((512, 128), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                # q
                lv0: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wq)
                lv1: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv0, [1, 10, 8, 64])
                lv2: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv1, [0, 2, 1, 3])
                # k
                lv3: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wk)
                lv4: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv3, [1, 10, 8, 64])
                lv5: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv4, [0, 2, 1, 3])
                # v
                lv6: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wv)
                lv7: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv6, [1, 10, 8, 64])
                lv8: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv7, [0, 2, 1, 3])
                # softmax(q @ k / sqrt(dk))
                lv9: R.Tensor((1, 8, 64, 10), "float32") = R.permute_dims(lv5, [0, 1, 3, 2])
                lv10: R.Tensor((1, 8, 10, 10), "float32") = R.matmul(lv2, lv9)
                lv11: R.Tensor((1, 8, 10, 10), "float32") = R.multiply(
                    lv10, R.const(1 / 8, "float32")
                )
                lv12: R.Tensor((1, 8, 10, 10), "float32") = R.nn.softmax(lv11, axis=-1)
                # attn_weight @ v
                lv13: R.Tensor((1, 8, 10, 64), "float32") = R.matmul(lv12, lv8)
                lv14: R.Tensor((1, 10, 8, 64), "float32") = R.permute_dims(lv13, [0, 2, 1, 3])
                lv15: R.Tensor((1, 10, 512), "float32") = R.reshape(lv14, [1, 10, 512])
                # attn_output @ o
                lv16: R.Tensor((1, 10, 128), "float32") = R.matmul(lv15, Wo)
                R.output(lv16)
            return lv16

    @tvm_tirx.script.ir_module
    class ShardedAttention:  # pylint: disable=too-few-public-methods
        @R.function
        def main(  # pylint: disable=too-many-locals
            x: R.Tensor((1, 10, 128), "float32"),
            Wq: R.Tensor((128, 256), "float32"),  # shard along axis 1
            Wk: R.Tensor((128, 256), "float32"),  # shard along axis 1
            Wv: R.Tensor((128, 256), "float32"),  # shard along axis 1
            Wo: R.Tensor((256, 128), "float32"),  # shard along axis 0
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                broadcast_x: R.Tensor((1, 10, 128), "float32") = R.ccl.broadcast_from_worker0(x)
                # q
                lv0: R.Tensor((1, 10, 256), "float32") = R.matmul(broadcast_x, Wq)
                lv1: R.Tensor((1, 10, 4, 64), "float32") = R.reshape(lv0, [1, 10, 4, 64])
                lv2: R.Tensor((1, 4, 10, 64), "float32") = R.permute_dims(lv1, [0, 2, 1, 3])
                # k
                lv3: R.Tensor((1, 10, 256), "float32") = R.matmul(broadcast_x, Wk)
                lv4: R.Tensor((1, 10, 4, 64), "float32") = R.reshape(lv3, [1, 10, 4, 64])
                lv5: R.Tensor((1, 4, 10, 64), "float32") = R.permute_dims(lv4, [0, 2, 1, 3])
                # v
                lv6: R.Tensor((1, 10, 256), "float32") = R.matmul(broadcast_x, Wv)
                lv7: R.Tensor((1, 10, 4, 64), "float32") = R.reshape(lv6, [1, 10, 4, 64])
                lv8: R.Tensor((1, 4, 10, 64), "float32") = R.permute_dims(lv7, [0, 2, 1, 3])
                # softmax(q @ k / sqrt(dk))
                lv9: R.Tensor((1, 4, 64, 10), "float32") = R.permute_dims(lv5, [0, 1, 3, 2])
                lv10: R.Tensor((1, 4, 10, 10), "float32") = R.matmul(lv2, lv9)
                lv11: R.Tensor((1, 4, 10, 10), "float32") = R.multiply(
                    lv10, R.const(1 / 8, "float32")
                )
                lv12: R.Tensor((1, 4, 10, 10), "float32") = R.nn.softmax(lv11, axis=-1)
                # attn_weight @ v
                lv13: R.Tensor((1, 4, 10, 64), "float32") = R.matmul(lv12, lv8)
                lv14: R.Tensor((1, 10, 4, 64), "float32") = R.permute_dims(lv13, [0, 2, 1, 3])
                lv15: R.Tensor((1, 10, 256), "float32") = R.reshape(lv14, [1, 10, 256])
                # attn_output @ o
                lv16: R.Tensor((1, 10, 128), "float32") = R.matmul(lv15, Wo)
                lv17: R.Tensor((1, 10, 128), "float32") = R.ccl.allreduce(lv16, "sum")
                R.output(lv17)
            return lv17

    # pylint: enable=invalid-name
    dev, target = create_device_target(ccl)

    def relax_build(mod, target):
        mod = rx.get_pipeline("zero", target=target)(mod)  # pylint: disable=no-value-for-parameter
        mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
            target=target,
        )(mod)
        return tvm.compile(mod, target=target)

    # pylint: disable=invalid-name
    X = np.random.randn(1, 10, 128).astype("float32")
    Wq = np.random.randn(128, 512).astype("float32")
    Wk = np.random.randn(128, 512).astype("float32")
    Wv = np.random.randn(128, 512).astype("float32")
    Wo = np.random.randn(512, 128).astype("float32")
    expected_ex = relax_build(Attention, target)
    sharded_ex = relax_build(ShardedAttention, target)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.so"
        sharded_ex.export_library(path)

        def run_test(sess):
            Y_expected = VirtualMachine(expected_ex, device=dev)["main"](
                tvm_tirx.runtime.tensor(X, device=dev),
                tvm_tirx.runtime.tensor(Wq, device=dev),
                tvm_tirx.runtime.tensor(Wk, device=dev),
                tvm_tirx.runtime.tensor(Wv, device=dev),
                tvm_tirx.runtime.tensor(Wo, device=dev),
            ).numpy()

            mod = sess.load_vm_module(path)
            d_X = sess.empty((1, 10, 128), "float32")
            d_Wq = sess.empty((128, 256), "float32")
            d_Wk = sess.empty((128, 256), "float32")
            d_Wv = sess.empty((128, 256), "float32")
            d_Wo = sess.empty((256, 128), "float32")

            d_X.debug_copy_from(0, X)
            d_Wq.debug_copy_from(0, Wq[:, :256])
            d_Wq.debug_copy_from(1, Wq[:, 256:])
            d_Wk.debug_copy_from(0, Wk[:, :256])
            d_Wk.debug_copy_from(1, Wk[:, 256:])
            d_Wv.debug_copy_from(0, Wv[:, :256])
            d_Wv.debug_copy_from(1, Wv[:, 256:])
            d_Wo.debug_copy_from(0, Wo[:256, :])
            d_Wo.debug_copy_from(1, Wo[256:, :])
            d_Y = mod["main"](d_X, d_Wq, d_Wk, d_Wv, d_Wo)
            Y_result = tvm_tirx.runtime.empty((1, 10, 128), "float32", device=dev)
            sess.copy_from_worker_0(Y_result, d_Y)
            sess.sync_worker_0()
            Y_result = Y_result.numpy()
            tvm_tirx.testing.assert_allclose(Y_result, Y_expected, rtol=1e-3, atol=1e-3)

        _run_with_ccl_session(session_kind, ccl, devices, run_test)
    # pylint: enable=invalid-name


if __name__ == "__main__":
    tvm_tirx.testing.main()
