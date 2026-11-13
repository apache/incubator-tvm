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
import json

import pytest
import tvm_ffi
from tvm_ffi import Shape

import tvm_tirx
import tvm_tirx.testing
from tvm.relax.frontend.nn.llm.kv_cache import AttnKind, RopeMode

reserved_nseq = 4
maximum_total_seq_length = 128
prefill_chunk_size = 64
page_size = 16
num_layers = 4
num_qo_heads = 32
num_kv_heads = 4
head_dim = 64
rope_scale = 1.0
rope_theta = 1e4
device = tvm_tirx.cpu()


def _nop(*args):
    return None


def create_kv_cache(
    *,
    dtype="float16",
    head_dim_value=head_dim,
    v_head_dim_value=None,
    page_size_value=page_size,
    num_layers_value=num_layers,
    rope_mode=RopeMode.NORMAL,
    attn_kind=AttnKind.MHA,
    support_sliding_window=False,
    reserved_nseq_value=reserved_nseq,
    maximum_total_seq_length_value=maximum_total_seq_length,
    prefill_chunk_size_value=prefill_chunk_size,
    rope_ext_factors=None,
):
    fcreate = tvm_ffi.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    dummy_func = tvm_ffi.convert(_nop)
    return fcreate(
        tvm_ffi.Shape(
            [
                reserved_nseq_value,
                maximum_total_seq_length_value,
                prefill_chunk_size_value,
                page_size_value,
                int(support_sliding_window),
            ]
        ),
        tvm_ffi.Shape([0, num_layers_value]),
        num_qo_heads,
        num_kv_heads,
        head_dim_value,
        head_dim_value if v_head_dim_value is None else v_head_dim_value,
        tvm_ffi.Shape([int(attn_kind) for _ in range(num_layers_value)]),
        False,  # enable_kv_transfer
        int(rope_mode),
        rope_scale,
        rope_theta,
        rope_ext_factors,
        tvm_tirx.runtime.empty((), dtype, device=device),
        dummy_func,  # f_transpose_append_mha
        None,  # f_transpose_append_mla
        [],  # f_attention_prefill_ragged
        [],  # f_attention_prefill
        [],  # f_attention_decode
        [],  # f_attention_prefill_sliding_window
        [],  # f_attention_decode_sliding_window
        [],  # f_attention_prefill_with_tree_mask_paged_kv
        [],  # f_attention_prefill_with_tree_mask
        [],  # f_mla_prefill
        [dummy_func],  # f_merge_inplace
        dummy_func,  # f_split_rotary
        dummy_func,  # f_copy_single_page
        dummy_func,  # f_debug_get_kv
        dummy_func,  # f_compact_copy
    )


def append_tokens(kv_cache, seq_id=0, append_length=page_size + 1):
    fadd_sequence = tvm_ffi.get_global_func("vm.builtin.kv_state_add_sequence")
    fbegin_forward = tvm_ffi.get_global_func("vm.builtin.kv_state_begin_forward")
    fend_forward = tvm_ffi.get_global_func("vm.builtin.kv_state_end_forward")
    fadd_sequence(kv_cache, seq_id)
    fbegin_forward(kv_cache, Shape([seq_id]), Shape([append_length]), None)
    fend_forward(kv_cache)


def test_checkpoint_metadata_reports_layout_pages_and_groups():
    fget_checkpoint_metadata = tvm_ffi.get_global_func(
        "vm.builtin.attention_kv_cache_get_checkpoint_metadata"
    )
    fget_layout_hash = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_get_layout_hash")
    fexport_page_group = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_export_page_group")
    fprepare_import = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_prepare_import")
    fimport_page_group = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_import_page_group")
    ffinish_import = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_finish_import")
    fget_sequence_length = tvm_ffi.get_global_func(
        "vm.builtin.attention_kv_cache_get_sequence_length"
    )

    kv_cache = create_kv_cache()
    append_tokens(kv_cache)
    metadata_json = fget_checkpoint_metadata(kv_cache, 0)
    metadata = json.loads(metadata_json)

    assert metadata["format_version"] == 1
    assert metadata["cache_type"] == "relax.vm.PagedAttentionKVCache"
    assert metadata["layout_hash"] == fget_layout_hash(kv_cache)
    assert metadata["seq_id"] == 0
    assert metadata["seq_length"] == page_size + 1
    assert metadata["page_size"] == page_size
    assert metadata["dtype"] == "float16"
    assert metadata["layer_begin"] == 0
    assert metadata["layer_end"] == num_layers
    assert metadata["num_kv_heads"] == num_kv_heads
    assert metadata["qk_head_dim"] == head_dim
    assert metadata["v_head_dim"] == head_dim
    assert metadata["reserved_num_seqs"] == reserved_nseq
    assert metadata["attn_kinds"] == ["mha"] * num_layers
    assert (
        metadata["page_group_layout"] == "1,num_logical_pages,2,num_kv_heads,page_size,qk_head_dim"
    )
    assert "blocks" not in metadata
    assert len(metadata["logical_pages"]) == 2
    assert metadata["logical_pages"][0]["start_pos"] == 0
    assert metadata["logical_pages"][0]["length"] == page_size
    assert metadata["logical_pages"][1]["start_pos"] == page_size
    assert metadata["logical_pages"][1]["length"] == 1
    assert "page_id" not in metadata["logical_pages"][0]
    assert "block_index" not in metadata["logical_pages"][0]
    assert len(metadata["groups"]) == num_layers
    assert metadata["groups"][0]["layer_begin"] == 0
    assert metadata["groups"][0]["layer_end"] == 1
    assert metadata["groups"][0]["num_logical_pages"] == 2
    assert metadata["groups"][0]["dtype"] == "float16"
    assert metadata["groups"][0]["shape"] == [1, 2, 2, num_kv_heads, page_size, head_dim]

    exported_groups = []
    for group_metadata in metadata["groups"]:
        group = tvm_tirx.runtime.empty(tuple(group_metadata["shape"]), "float16", device=device)
        fexport_page_group(kv_cache, 0, group_metadata["group_index"], group)
        exported_groups.append(group)

    import_cache = create_kv_cache(
        reserved_nseq_value=reserved_nseq * 2,
        maximum_total_seq_length_value=maximum_total_seq_length * 2,
        prefill_chunk_size_value=prefill_chunk_size // 2,
    )
    fprepare_import(import_cache, 0, metadata_json)
    with pytest.raises(RuntimeError, match="until checkpoint import is finished"):
        fget_sequence_length(import_cache, 0)
    for group_metadata, group in zip(metadata["groups"], exported_groups):
        fimport_page_group(import_cache, 0, group_metadata["group_index"], group)
    ffinish_import(import_cache, 0)
    assert fget_sequence_length(import_cache, 0) == page_size + 1


def test_checkpoint_layout_hash_is_stable_and_layout_sensitive():
    fget_layout_hash = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_get_layout_hash")

    kv_cache = create_kv_cache()
    same_layout = create_kv_cache()
    different_page_size = create_kv_cache(page_size_value=page_size * 2)
    different_num_layers = create_kv_cache(num_layers_value=2)
    different_head_dim = create_kv_cache(head_dim_value=128)
    different_dtype = create_kv_cache(dtype="float32")
    different_rope = create_kv_cache(rope_mode=RopeMode.NONE)
    different_operational_limits = create_kv_cache(
        reserved_nseq_value=reserved_nseq * 2,
        maximum_total_seq_length_value=maximum_total_seq_length * 2,
        prefill_chunk_size_value=prefill_chunk_size // 2,
    )

    layout_hash = fget_layout_hash(kv_cache)
    assert layout_hash == fget_layout_hash(kv_cache)
    assert layout_hash == fget_layout_hash(same_layout)
    assert layout_hash != fget_layout_hash(different_page_size)
    assert layout_hash != fget_layout_hash(different_num_layers)
    assert layout_hash != fget_layout_hash(different_head_dim)
    assert layout_hash != fget_layout_hash(different_dtype)
    assert layout_hash != fget_layout_hash(different_rope)
    assert layout_hash == fget_layout_hash(different_operational_limits)


def test_checkpoint_metadata_rejects_unsupported_layouts_and_sequence_ids():
    fget_checkpoint_metadata = tvm_ffi.get_global_func(
        "vm.builtin.attention_kv_cache_get_checkpoint_metadata"
    )
    fget_layout_hash = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_get_layout_hash")
    fexport_page_group = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_export_page_group")

    sliding_cache = create_kv_cache(support_sliding_window=True)
    append_tokens(sliding_cache)
    mla_cache = create_kv_cache(attn_kind=AttnKind.MLA)
    asymmetric_cache = create_kv_cache(v_head_dim_value=head_dim // 2)
    rope_ext_cache = create_kv_cache(
        rope_ext_factors=tvm_tirx.runtime.empty((head_dim // 2,), "float32", device=device)
    )
    dst = tvm_tirx.runtime.empty(
        (1, 1, 2, num_kv_heads, page_size, head_dim), "float16", device=device
    )

    with pytest.raises(RuntimeError, match="sliding-window"):
        fget_layout_hash(sliding_cache)
    with pytest.raises(RuntimeError, match="sliding-window"):
        fget_checkpoint_metadata(sliding_cache, 0)
    with pytest.raises(RuntimeError, match="sliding-window"):
        fexport_page_group(sliding_cache, 0, 0, dst)
    with pytest.raises(RuntimeError, match="sequence id 0"):
        fget_checkpoint_metadata(create_kv_cache(), 1)
    with pytest.raises(RuntimeError, match="full-context MHA/GQA"):
        fget_layout_hash(mla_cache)
    with pytest.raises(RuntimeError, match="full-context MHA/GQA"):
        fexport_page_group(mla_cache, 0, 0, dst)
    with pytest.raises(RuntimeError, match="qk_head_dim to equal v_head_dim"):
        fget_layout_hash(asymmetric_cache)
    with pytest.raises(RuntimeError, match="RoPE extension factors"):
        fget_layout_hash(rope_ext_cache)

    tree_cache = create_kv_cache()
    tvm_ffi.get_global_func("vm.builtin.kv_state_add_sequence")(tree_cache, 0)
    tvm_ffi.get_global_func("vm.builtin.kv_state_begin_forward")(
        tree_cache, Shape([0]), Shape([2]), Shape([-1, 0])
    )
    with pytest.raises(RuntimeError, match="committed token-chain state"):
        fexport_page_group(tree_cache, 0, 0, dst)


def test_checkpoint_export_page_group_validates_group_shape():
    fget_checkpoint_metadata = tvm_ffi.get_global_func(
        "vm.builtin.attention_kv_cache_get_checkpoint_metadata"
    )
    fexport_page_group = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_export_page_group")

    kv_cache = create_kv_cache()
    append_tokens(kv_cache)
    metadata = json.loads(fget_checkpoint_metadata(kv_cache, 0))
    shape = metadata["groups"][0]["shape"]

    with pytest.raises(RuntimeError, match="group id"):
        fexport_page_group(
            kv_cache,
            0,
            num_layers,
            tvm_tirx.runtime.empty(tuple(shape), "float16", device=device),
        )

    bad_shape = shape.copy()
    bad_shape[-1] += 1
    with pytest.raises(RuntimeError, match="ExportPageGroup expects"):
        fexport_page_group(
            kv_cache,
            0,
            0,
            tvm_tirx.runtime.empty(tuple(bad_shape), "float16", device=device),
        )

    with pytest.raises(RuntimeError, match="dtype mismatches"):
        fexport_page_group(
            kv_cache,
            0,
            0,
            tvm_tirx.runtime.empty(tuple(shape), "float32", device=device),
        )


def test_checkpoint_prepare_import_validates_metadata():
    fget_checkpoint_metadata = tvm_ffi.get_global_func(
        "vm.builtin.attention_kv_cache_get_checkpoint_metadata"
    )
    fprepare_import = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_prepare_import")

    kv_cache = create_kv_cache()
    append_tokens(kv_cache)
    metadata_json = fget_checkpoint_metadata(kv_cache, 0)
    metadata = json.loads(metadata_json)

    with pytest.raises(RuntimeError, match="sequence id 0"):
        fprepare_import(create_kv_cache(), 1, metadata_json)

    with pytest.raises(RuntimeError, match="dtype"):
        fprepare_import(create_kv_cache(dtype="float32"), 0, metadata_json)

    bad_length = json.loads(metadata_json)
    bad_length["seq_length"] = page_size * 2 + 1
    with pytest.raises(RuntimeError, match="sequence length"):
        fprepare_import(create_kv_cache(), 0, json.dumps(bad_length))

    bad_group = json.loads(metadata_json)
    bad_group["groups"][0]["shape"][-1] += 1
    with pytest.raises(RuntimeError, match="shape"):
        fprepare_import(create_kv_cache(), 0, json.dumps(bad_group))

    bad_nbytes = json.loads(metadata_json)
    bad_nbytes["groups"][0]["nbytes"] += 1
    with pytest.raises(RuntimeError, match="nbytes"):
        fprepare_import(create_kv_cache(), 0, json.dumps(bad_nbytes))

    bad_version = json.loads(metadata_json)
    bad_version["format_version"] += 1
    with pytest.raises(RuntimeError, match="format_version"):
        fprepare_import(create_kv_cache(), 0, json.dumps(bad_version))

    with pytest.raises(RuntimeError, match="only has 1 pages"):
        fprepare_import(
            create_kv_cache(maximum_total_seq_length_value=0),
            0,
            metadata_json,
        )

    metadata["layout_hash"] = "bad-layout-hash"
    with pytest.raises(RuntimeError, match="layout hash mismatch"):
        fprepare_import(create_kv_cache(), 0, json.dumps(metadata))


def test_checkpoint_import_page_group_validates_group_shape():
    fget_checkpoint_metadata = tvm_ffi.get_global_func(
        "vm.builtin.attention_kv_cache_get_checkpoint_metadata"
    )
    fprepare_import = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_prepare_import")
    fimport_page_group = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_import_page_group")

    kv_cache = create_kv_cache()
    append_tokens(kv_cache)
    metadata_json = fget_checkpoint_metadata(kv_cache, 0)
    metadata = json.loads(metadata_json)
    shape = metadata["groups"][0]["shape"]

    import_cache = create_kv_cache()
    fprepare_import(import_cache, 0, metadata_json)

    with pytest.raises(RuntimeError, match="group id"):
        fimport_page_group(
            import_cache,
            0,
            num_layers,
            tvm_tirx.runtime.empty(tuple(shape), "float16", device=device),
        )

    bad_shape = shape.copy()
    bad_shape[-1] += 1
    with pytest.raises(RuntimeError, match="ImportPageGroup expects"):
        fimport_page_group(
            import_cache,
            0,
            0,
            tvm_tirx.runtime.empty(tuple(bad_shape), "float16", device=device),
        )

    with pytest.raises(RuntimeError, match="dtype mismatches"):
        fimport_page_group(
            import_cache,
            0,
            0,
            tvm_tirx.runtime.empty(tuple(shape), "float32", device=device),
        )


def test_checkpoint_import_requires_all_groups_and_explicit_finish():
    fget_checkpoint_metadata = tvm_ffi.get_global_func(
        "vm.builtin.attention_kv_cache_get_checkpoint_metadata"
    )
    fexport_page_group = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_export_page_group")
    fprepare_import = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_prepare_import")
    fimport_page_group = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_import_page_group")
    ffinish_import = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_finish_import")
    fbegin_forward = tvm_ffi.get_global_func("vm.builtin.kv_state_begin_forward")

    source_cache = create_kv_cache()
    append_tokens(source_cache)
    metadata_json = fget_checkpoint_metadata(source_cache, 0)
    metadata = json.loads(metadata_json)
    groups = []
    for group_metadata in metadata["groups"]:
        group = tvm_tirx.runtime.empty(tuple(group_metadata["shape"]), "float16", device=device)
        fexport_page_group(source_cache, 0, group_metadata["group_index"], group)
        groups.append(group)

    import_cache = create_kv_cache()
    fprepare_import(import_cache, 0, metadata_json)
    fimport_page_group(import_cache, 0, 0, groups[0])

    with pytest.raises(RuntimeError, match="already imported"):
        fimport_page_group(import_cache, 0, 0, groups[0])
    with pytest.raises(RuntimeError, match="missing group 1"):
        ffinish_import(import_cache, 0)
    with pytest.raises(RuntimeError, match="before checkpoint import is finished"):
        fbegin_forward(import_cache, Shape([0]), Shape([1]), None)

    for group_id in range(1, num_layers):
        fimport_page_group(import_cache, 0, group_id, groups[group_id])
    ffinish_import(import_cache, 0)
    with pytest.raises(RuntimeError, match="has not been prepared"):
        ffinish_import(import_cache, 0)


if __name__ == "__main__":
    tvm_tirx.testing.main()
