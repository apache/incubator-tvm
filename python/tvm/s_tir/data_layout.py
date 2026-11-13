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
"""Data layout."""

import tvm_ffi

from . import _ffi_api


@tvm_ffi.register_object("s_tir.SLayout")
class SLayout(tvm_ffi.Object):
    """SLayout is composed of upper cases, lower cases and numbers,
    where upper case indicates a primal axis and
    the corresponding lower case with factor size indicates the subordinate axis.
    For example, NCHW16c can describe a 5-D tensor of
    [batch_size, channel, height, width, channel_block].
    Here subordinate axis channel_block=16 is the factor size of the primal axis C (channel).

    See Also
    --------
    slayout : Declare a layout
    """

    def __len__(self):
        return _ffi_api.SLayoutNdim(self)  # type: ignore

    def __contains__(self, axis):
        # Note: We do a weaker check for packed axis assuming layout is valid
        return not any(bkt in axis for bkt in "[]") and axis in self.name

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("SLayout index out of range")
        return _ffi_api.SLayoutGetItem(self, index)  # type: ignore

    def index_of(self, axis):
        """Get the index of an axis

        Parameters
        ----------
        axis : str
            The axis name, needs to be [a-z,A-Z] or a packed axis

        Returns
        -------
        index : int
            The index of the axis, -1 if not found.
        """
        return _ffi_api.SLayoutIndexOf(self, axis)  # type: ignore

    def factor_of(self, axis):
        """Get the factor size of the subordinate axis.

        Parameters
        ----------
        axis : str
            The axis name, need to be [a-z,A-Z]

        Returns
        -------
        factor : int
            the size of the subordinate-axis of axis (if axis is a primal-axis),
            or the size of axis itself (if axis is a subordinate-axis).
            Return -1 if axis is not in the layout.
        """
        return _ffi_api.SLayoutFactorOf(self, axis)  # type: ignore


@tvm_ffi.register_object("s_tir.SBijectiveLayout")
class SBijectiveLayout(tvm_ffi.Object):
    """Bijective mapping for two layouts (src-layout and dst-layout).
    It provides shape and index conversion between each other.

    Do not construct directly, use :any:`sbijective_layout` instead.
    See the documentation of :any:`sbijective_layout` for more details.

    Parameters
    ----------
    src_layout : str or SLayout
        source layout.

    dst_layout : str or SLayout
        destination layout.

    See Also
    --------
    sbijective_layout : Declare a layout
    """

    def forward_index(self, index):
        """Given the indices of the src-layout, infer the dst index.

        Parameters
        ----------
        index: Array of Expr
            The indices in src-layout.

        Returns
        -------
        dst_index: Array of Expr
            The inferred indices in dst-layout.
        """
        return _ffi_api.SBijectiveLayoutForwardIndex(self, index)  # type: ignore

    def backward_index(self, index):
        """Given the indices of the dst-layout, infer the src index.

        Parameters
        ----------
        index: Array of Expr
            The indices in dst-layout.

        Returns
        -------
        src_index: Array of Expr
            The inferred indices in src-layout.
        """
        return _ffi_api.SBijectiveLayoutBackwardIndex(self, index)  # type: ignore

    def forward_shape(self, shape):
        """Given the shape of the src-layout, infer the dst shape.

        Parameters
        ----------
        shape: Array of Expr
            The shape in src-layout.

        Returns
        -------
        dst_shape: Array of Expr
            The inferred shape in dst-layout.
        """
        return _ffi_api.SBijectiveLayoutForwardShape(self, shape)  # type: ignore

    def backward_shape(self, shape):
        """Given the shape of the dst-layout, infer the src shape.

        Parameters
        ----------
        shape: Array of Expr
            The shape in dst-layout.

        Returns
        -------
        src_shape: Array of Expr
            The inferred shape in src-layout.
        """
        return _ffi_api.SBijectiveLayoutBackwardShape(self, shape)  # type: ignore


def slayout(layout_str: str, dtype: str = "int32") -> SLayout | None:
    """Create a layout node from a string.

    Parameters
    ----------
    layout_str : str
        A layout representation is composed of upper cases, lower cases and numbers,
        where upper case indicates a primal axis and
        the corresponding lower case with factor size indicates the subordinate axis.
        For example, NCHW16c can describe a 5-D tensor of
        [batch_size, channel, height, width, channel_block].
        Here subordinate axis channel_block=16 is the factor size of
        the primal axis C (channel).

    dtype : str
        The dtype of generated axes vars in the returned layout.
        It is required to be integer type.

    Returns
    -------
    layout : SLayout or None
        The created layout, or None for an empty string or "__undef__".
    """
    return _ffi_api.SLayout(layout_str, dtype)  # type: ignore


def sbijective_layout(
    src_layout: str | SLayout | None, dst_layout: str | SLayout | None
) -> SBijectiveLayout | None:
    """Create a bijective layout mapping.

    Parameters
    ----------
    src_layout : str or SLayout
        source layout.

    dst_layout : str or SLayout
        destination layout.

    Returns
    -------
    sbijective_layout : SBijectiveLayout or None
        The created bijective layout, or None when the layouts are not convertible.
    """
    if isinstance(src_layout, str):
        src_layout = slayout(src_layout)
    if isinstance(dst_layout, str):
        dst_layout = slayout(dst_layout)
    return _ffi_api.SBijectiveLayout(src_layout, dst_layout)  # type: ignore
