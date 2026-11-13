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
"""We extract one feature vector per TensorStoreObj statement in a TIR Stmt,
so we call this feature as "per-store" feature.
"""

from tvm_ffi import register_object

from .. import _ffi_api
from .feature_extractor import FeatureExtractor


@register_object("s_tir.meta_schedule.PerStoreFeature")
class PerStoreFeature(FeatureExtractor):
    """PerStoreFeature extracts one feature vector per TensorStoreObj

    Parameters
    ----------
    tensors_per_store : int
        The number of tensors in each TensorStore; Pad or truncate if necessary.
    arith_intensity_curve_num_samples : int
        The number of samples used in the arithmetic intensity curve.
    cache_line_bytes : int
        The number of bytes in a cache line.
    extract_workload : bool
        Whether to extract features in the workload in tuning context or not.
    """

    tensors_per_store: int
    """The number of tensors in each TensorStore; Pad or truncate if necessary."""
    arith_intensity_curve_num_samples: int  # pylint: disable=invalid-name
    """The number of samples used in the arithmetic intensity curve."""
    cache_line_bytes: int
    """The number of bytes in a cache line."""
    extract_workload: bool
    """Whether to extract features in the workload in tuning context or not."""
    feature_vector_length: int
    """Length of the feature vector."""

    def __init__(
        self,
        tensors_per_store: int = 5,
        arith_intensity_curve_num_samples: int = 10,
        cache_line_bytes: int = 64,
        extract_workload: bool = False,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.FeatureExtractorPerStoreFeature,  # type: ignore # pylint: disable=no-member
            tensors_per_store,
            arith_intensity_curve_num_samples,
            cache_line_bytes,
            extract_workload,
        )
