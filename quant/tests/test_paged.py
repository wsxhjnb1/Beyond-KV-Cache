"""Paged cache layout roundtrip tests for 4-bit per-token K/V."""

import math
import os
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import quant.beyond_cute as beyond_cute  # noqa: E402
from quant.beyond_cute import (  # noqa: E402
    CODE_LAYOUT_HEAD_WORD_TOKEN,
    CODE_LAYOUT_HND_TOKEN_WORD,
    PagedLayout,
    _packed_readback_launch_policy,
    _packed_readback_use_flat_grid,
    _packed_writer_groups_per_program,
    _packed_writer_num_warps,
    _try_write_tokens_to_paged_cache_triton,
    _write_tokens_to_paged_cache_reference,
    dequantize_k,
    dequantize_paged_kv_to_dense,
    dequantize_v,
    quantize_k,
    quantize_v,
    read_request_from_paged_cache,
    write_tokens_to_paged_cache,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _assert_quant_roundtrip(actual, expected):
    if DEVICE == "cuda":
        diff = (actual.float() - expected.float()).abs()
        # CUDA fdiv/comparison can place rare tie values on the other side of
        # a bucket boundary by one 4-bit code. Keep that tolerance bounded.
        assert float(diff.max().item()) <= 0.35
        assert int((diff > 0).sum().item()) <= max(4, actual.numel() // 2048)
    else:
        torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)


def test_packed_writer_uses_one_warp():
    assert _packed_writer_num_warps() == 1


@pytest.mark.parametrize(
    ("num_heads", "write_dq", "last_independent_token"),
    [
        (4, False, 306),
        (8, False, 163),
        (4, True, 367),
        (8, True, 189),
    ],
)
def test_packed_writer_independent_group_boundaries(num_heads, write_dq, last_independent_token):
    assert _packed_writer_groups_per_program(last_independent_token, num_heads, write_dq) == 1
    assert _packed_writer_groups_per_program(last_independent_token + 1, num_heads, write_dq) == 4


def test_packed_writer_production_mode_fails_closed(monkeypatch):
    layout = PagedLayout.build_hnd(
        block_size=64,
        num_kv_heads=4,
        head_dim=128,
        bits=4,
        k_group_size=32,
        v_group_size=32,
    )
    cache = torch.zeros((1, layout.block_i32), dtype=torch.int32)
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    slots = torch.zeros((1,), dtype=torch.int32)
    key = torch.zeros((1, 4, 128), dtype=torch.float16)
    value = torch.zeros_like(key)
    monkeypatch.setattr(
        beyond_cute,
        "_try_write_tokens_to_paged_cache_triton",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(RuntimeError, match="fused Triton writer"):
        write_tokens_to_paged_cache(
            cache,
            layout,
            block_table,
            slots,
            key,
            value,
            require_fast=True,
        )


def _make_cache(num_blocks: int, layout: PagedLayout, device) -> torch.Tensor:
    return torch.zeros((num_blocks, layout.block_i32), dtype=torch.int32, device=device)


def _single_request_block_table(block_ids, pad_blocks: int, device):
    bt = torch.full((1, pad_blocks), -1, dtype=torch.int32, device=device)
    bt[0, : len(block_ids)] = torch.tensor(block_ids, dtype=torch.int32, device=device)
    return bt


def _slot_mapping(block_ids, block_size: int, seq_len: int, device):
    slots = torch.empty(seq_len, dtype=torch.int32, device=device)
    for bi, bid in enumerate(block_ids):
        start = bi * block_size
        end = min(seq_len, start + block_size)
        slots[start:end] = bid * block_size + torch.arange(
            end - start, dtype=torch.int32, device=device
        )
    return slots


@pytest.mark.parametrize("code_layout", [CODE_LAYOUT_HEAD_WORD_TOKEN, CODE_LAYOUT_HND_TOKEN_WORD])
@pytest.mark.parametrize("block_size", [16, 64])
def test_paged_roundtrip_single_request(block_size, code_layout):
    torch.manual_seed(0)
    H_kv, D = 2, 96
    k_group_size = v_group_size = 32
    layout = PagedLayout.build(
        block_size=block_size,
        num_kv_heads=H_kv,
        head_dim=D,
        bits=4,
        k_group_size=k_group_size,
        v_group_size=v_group_size,
        code_layout=code_layout,
    )
    num_blocks = 8
    cache = _make_cache(num_blocks, layout, DEVICE)
    seq_len = block_size + 7
    block_ids = [3, 1]
    block_table = _single_request_block_table(block_ids, pad_blocks=4, device=DEVICE)
    slot_mapping = _slot_mapping(block_ids, block_size, seq_len, DEVICE)

    k = torch.randn(seq_len, H_kv, D, dtype=torch.float16, device=DEVICE)
    v = torch.randn(seq_len, H_kv, D, dtype=torch.float16, device=DEVICE)
    write_tokens_to_paged_cache(cache, layout, block_table, slot_mapping, k, v)

    k_read, v_read = read_request_from_paged_cache(
        cache,
        layout,
        torch.tensor(block_ids, dtype=torch.int32, device=DEVICE),
        seq_len,
        out_dtype=torch.float16,
    )
    k_oracle = dequantize_k(quantize_k(k.unsqueeze(0), 4, k_group_size))
    v_oracle = dequantize_v(quantize_v(v.unsqueeze(0), 4, v_group_size))
    _assert_quant_roundtrip(k_read, k_oracle)
    _assert_quant_roundtrip(v_read, v_oracle)
    _assert_quant_roundtrip(k_read, k_oracle)
    _assert_quant_roundtrip(v_read, v_oracle)


def test_paged_to_attention_end_to_end():
    import torch.nn.functional as F

    torch.manual_seed(123)
    H_kv, H_q, D = 2, 4, 96
    block_size = 32
    k_group_size = v_group_size = 32
    layout = PagedLayout.build(
        block_size=block_size,
        num_kv_heads=H_kv,
        head_dim=D,
        bits=4,
        k_group_size=k_group_size,
        v_group_size=v_group_size,
    )
    cache = _make_cache(8, layout, DEVICE)
    seq_len = 3 * block_size + 5
    block_ids = [3, 1, 4, 6]
    block_table = _single_request_block_table(block_ids, pad_blocks=6, device=DEVICE)
    slots = _slot_mapping(block_ids, block_size, seq_len, DEVICE)

    k = torch.randn(seq_len, H_kv, D, dtype=torch.float16, device=DEVICE)
    v = torch.randn(seq_len, H_kv, D, dtype=torch.float16, device=DEVICE)
    write_tokens_to_paged_cache(cache, layout, block_table, slots, k, v)
    k_read, v_read = read_request_from_paged_cache(
        cache,
        layout,
        torch.tensor(block_ids, dtype=torch.int32, device=DEVICE),
        seq_len,
        out_dtype=torch.float16,
    )
    k_oracle = dequantize_k(quantize_k(k.unsqueeze(0), 4, k_group_size))
    v_oracle = dequantize_v(quantize_v(v.unsqueeze(0), 4, v_group_size))

    q = torch.randn(1, seq_len, H_q, D, dtype=torch.float16, device=DEVICE)

    def run_sdpa(qv, kv, vv):
        repeat = H_q // H_kv
        kv = kv.repeat_interleave(repeat, dim=2)
        vv = vv.repeat_interleave(repeat, dim=2)
        return (
            F.scaled_dot_product_attention(
                qv.transpose(1, 2).contiguous(),
                kv.transpose(1, 2).contiguous(),
                vv.transpose(1, 2).contiguous(),
                is_causal=True,
            )
            .transpose(1, 2)
            .contiguous()
        )

    attn_tol = 1e-2 if DEVICE == "cuda" else 3e-3
    torch.testing.assert_close(
        run_sdpa(q, k_read, v_read).float(),
        run_sdpa(q, k_oracle, v_oracle).float(),
        atol=attn_tol,
        rtol=attn_tol,
    )


def test_paged_roundtrip_uses_grouped_qpoints():
    torch.manual_seed(321)
    H_kv, D = 2, 64
    group_size = 32
    seq_len = 19
    block_size = 16
    layout = PagedLayout.build(
        block_size=block_size,
        num_kv_heads=H_kv,
        head_dim=D,
        bits=4,
        k_group_size=group_size,
        v_group_size=group_size,
    )
    cache = _make_cache(4, layout, DEVICE)
    block_ids = [0, 1]
    block_table = _single_request_block_table(block_ids, pad_blocks=2, device=DEVICE)
    slots = _slot_mapping(block_ids, block_size, seq_len, DEVICE)
    num_tables = H_kv * (D // group_size)
    base = torch.linspace(0.0, 1.0, 16, dtype=torch.float32, device=DEVICE)
    table = torch.arange(num_tables, dtype=torch.float32, device=DEVICE)
    q_points_k = base.unsqueeze(0).pow(1.0 + 0.04 * table.unsqueeze(1))
    q_points_v = base.unsqueeze(0).pow(0.75 + 0.03 * table.unsqueeze(1))
    q_points_k[:, 0], q_points_k[:, -1] = 0.0, 1.0
    q_points_v[:, 0], q_points_v[:, -1] = 0.0, 1.0
    thresholds_k = (q_points_k[:, :-1] + q_points_k[:, 1:]) / 2.0
    thresholds_v = (q_points_v[:, :-1] + q_points_v[:, 1:]) / 2.0

    k = torch.randn(seq_len, H_kv, D, dtype=torch.float16, device=DEVICE)
    v = torch.randn(seq_len, H_kv, D, dtype=torch.float16, device=DEVICE)
    write_tokens_to_paged_cache(
        cache,
        layout,
        block_table,
        slots,
        k,
        v,
        q_points_k=q_points_k,
        q_points_v=q_points_v,
        thresholds_k=thresholds_k,
        thresholds_v=thresholds_v,
    )
    k_read, v_read = read_request_from_paged_cache(
        cache,
        layout,
        torch.tensor(block_ids, dtype=torch.int32, device=DEVICE),
        seq_len,
        q_points_k=q_points_k,
        q_points_v=q_points_v,
        out_dtype=torch.float16,
    )
    k_oracle = dequantize_k(
        quantize_k(
            k.unsqueeze(0),
            4,
            group_size,
            q_points=q_points_k,
            thresholds=thresholds_k,
        ),
        q_points=q_points_k,
    )
    v_oracle = dequantize_v(
        quantize_v(
            v.unsqueeze(0),
            4,
            group_size,
            q_points=q_points_v,
            thresholds=thresholds_v,
        ),
        q_points=q_points_v,
    )

    _assert_quant_roundtrip(k_read, k_oracle)
    _assert_quant_roundtrip(v_read, v_oracle)


def test_paged_production_code_layout_roundtrip():
    torch.manual_seed(17)

    H_kv, D = 2, 64
    group_size = 32
    block_size = 16
    seq_len = 23
    layout = PagedLayout.build(
        block_size=block_size,
        num_kv_heads=H_kv,
        head_dim=D,
        bits=4,
        k_group_size=group_size,
        v_group_size=group_size,
    )
    assert layout.code_layout == "head_word_token"

    cache = _make_cache(4, layout, DEVICE)
    block_ids = [2, 0]
    block_table = _single_request_block_table(block_ids, pad_blocks=3, device=DEVICE)
    slots = _slot_mapping(block_ids, block_size, seq_len, DEVICE)
    k = torch.randn(seq_len, H_kv, D, dtype=torch.float16, device=DEVICE)
    v = torch.randn(seq_len, H_kv, D, dtype=torch.float16, device=DEVICE)
    write_tokens_to_paged_cache(cache, layout, block_table, slots, k, v)
    k_read, v_read = read_request_from_paged_cache(
        cache,
        layout,
        torch.tensor(block_ids, dtype=torch.int32, device=DEVICE),
        seq_len,
        out_dtype=torch.float16,
    )
    _assert_quant_roundtrip(k_read, dequantize_k(quantize_k(k.unsqueeze(0), 4, group_size)))
    _assert_quant_roundtrip(v_read, dequantize_v(quantize_v(v.unsqueeze(0), 4, group_size)))


def test_paged_hnd_layout_is_explicit_abi_v2():
    common = dict(
        block_size=64,
        num_kv_heads=8,
        head_dim=128,
        bits=4,
        k_group_size=32,
        v_group_size=32,
    )
    legacy = PagedLayout.build(**common)
    hnd = PagedLayout.build_hnd(**common)

    assert legacy.code_layout == CODE_LAYOUT_HEAD_WORD_TOKEN
    assert legacy.abi_version == 1
    assert hnd.code_layout == CODE_LAYOUT_HND_TOKEN_WORD
    assert hnd.abi_version == 2
    assert hnd.block_i32 == legacy.block_i32
    assert hnd.v_code_off == legacy.v_code_off


@pytest.mark.skipif(
    not torch.cuda.is_available() or not beyond_cute._TRITON_OK,
    reason="requires CUDA Triton",
)
@pytest.mark.parametrize("pairwise", [False, True])
@pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_size", [64, 128])
def test_fused_hnd_paged_dequant_matches_request_oracle(
    model_dtype, block_size, pairwise, monkeypatch
):
    monkeypatch.setattr(
        beyond_cute,
        "_PACKED_READBACK_PAIRWISE_MIN_TOKEN_HEADS",
        0 if pairwise else sys.maxsize,
    )
    torch.manual_seed(913)
    device = torch.device("cuda")
    num_heads, head_dim = 2, 128
    lengths = [137, 255]
    block_rows = [[3, 1, 5], [4, 2, 6, 7]] if block_size == 64 else [[3, 1], [4, 2]]
    layout = PagedLayout.build_hnd(
        block_size=block_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        bits=4,
        k_group_size=32,
        v_group_size=32,
    )
    cache = _make_cache(8 if block_size == 64 else 6, layout, device)
    table_width = max(len(row) for row in block_rows)
    block_table = torch.full((2, table_width), -1, dtype=torch.int32, device=device)
    for request, row in enumerate(block_rows):
        block_table[request, : len(row)] = torch.tensor(
            row,
            dtype=torch.int32,
            device=device,
        )
    slots = torch.cat(
        [_slot_mapping(row, block_size, length, device) for row, length in zip(block_rows, lengths)]
    )
    total_tokens = sum(lengths)
    key = torch.randn(
        total_tokens,
        num_heads,
        head_dim,
        dtype=model_dtype,
        device=device,
    )
    value = torch.randn_like(key)
    tables = num_heads * 4
    base = torch.linspace(0.0, 1.0, 16, device=device, dtype=torch.float32)
    powers = torch.arange(tables, device=device, dtype=torch.float32).unsqueeze(1)
    q_points_k = base.unsqueeze(0).pow(0.8 + 0.025 * powers).to(torch.float16)
    q_points_v = base.unsqueeze(0).pow(1.2 + 0.025 * powers).to(torch.float16)
    thresholds_k = ((q_points_k[:, :-1].float() + q_points_k[:, 1:].float()) / 2.0).to(
        torch.float16
    )
    thresholds_v = ((q_points_v[:, :-1].float() + q_points_v[:, 1:].float()) / 2.0).to(
        torch.float16
    )
    write_tokens_to_paged_cache(
        cache,
        layout,
        block_table,
        slots,
        key,
        value,
        q_points_k=q_points_k,
        q_points_v=q_points_v,
        thresholds_k=thresholds_k,
        thresholds_v=thresholds_v,
    )
    starts = torch.tensor(
        [0, lengths[0], total_tokens],
        dtype=torch.int32,
        device=device,
    )
    out_k = torch.empty_like(key)
    out_v = torch.empty_like(value)
    dequantize_paged_kv_to_dense(
        cache,
        layout,
        block_table,
        torch.tensor(lengths, dtype=torch.int32, device=device),
        starts,
        out_k,
        out_v,
        max_seq_len=max(lengths),
        q_points_k=q_points_k,
        q_points_v=q_points_v,
    )

    for request, (start, end) in enumerate(zip(starts.tolist(), starts.tolist()[1:])):
        request_pages = math.ceil(lengths[request] / block_size)
        expected_k, expected_v = read_request_from_paged_cache(
            cache,
            layout,
            block_table[request, :request_pages],
            lengths[request],
            q_points_k=q_points_k,
            q_points_v=q_points_v,
            out_dtype=model_dtype,
            stat_dtype=model_dtype,
        )
        torch.testing.assert_close(out_k[start:end], expected_k.squeeze(0), atol=0, rtol=0)
        torch.testing.assert_close(out_v[start:end], expected_v.squeeze(0), atol=0, rtol=0)


def test_packed_readback_launch_policy_switches_only_at_saturated_grid():
    assert _packed_readback_launch_policy(
        model_dtype=torch.float16,
        batch_size=1,
        max_seq_bucket=8192,
        num_heads=4,
    ) == (False, 8, 4)
    assert _packed_readback_launch_policy(
        model_dtype=torch.float16,
        batch_size=1,
        max_seq_bucket=16384,
        num_heads=4,
    ) == (True, 8, 2)
    assert _packed_readback_launch_policy(
        model_dtype=torch.bfloat16,
        batch_size=1,
        max_seq_bucket=8192,
        num_heads=8,
    ) == (True, 4, 1)
    assert _packed_readback_use_flat_grid(
        model_dtype=torch.bfloat16,
        batch_size=1,
        max_seq_bucket=16384,
        num_heads=8,
        tokens_per_program=4,
    )
    assert not _packed_readback_use_flat_grid(
        model_dtype=torch.bfloat16,
        batch_size=1,
        max_seq_bucket=32768,
        num_heads=8,
        tokens_per_program=4,
    )
    assert not _packed_readback_use_flat_grid(
        model_dtype=torch.float16,
        batch_size=1,
        max_seq_bucket=16384,
        num_heads=8,
        tokens_per_program=8,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not beyond_cute._TRITON_OK,
    reason="requires CUDA Triton",
)
@pytest.mark.parametrize(("num_tokens", "grouped_tables"), [(8, False), (513, True)])
@pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("code_layout", [CODE_LAYOUT_HEAD_WORD_TOKEN, CODE_LAYOUT_HND_TOKEN_WORD])
def test_triton_packed_writer_matches_reference_bytes(
    num_tokens,
    grouped_tables,
    model_dtype,
    code_layout,
):
    torch.manual_seed(2708 + num_tokens)
    device = torch.device("cuda")
    num_heads = 2
    head_dim = 128
    block_size = 64
    num_blocks = 12
    layout = PagedLayout.build(
        block_size=block_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        bits=4,
        k_group_size=32,
        v_group_size=32,
        code_layout=code_layout,
    )
    cache_seed = torch.randint(
        -(1 << 31),
        1 << 31,
        (num_blocks, layout.block_i32),
        dtype=torch.int32,
        device=device,
    )
    cache_triton = cache_seed.clone()
    cache_reference = cache_seed.clone()
    key = torch.randn(
        num_tokens,
        num_heads,
        head_dim,
        dtype=model_dtype,
        device=device,
    )
    value = torch.randn_like(key)
    slots = torch.randperm(num_blocks * block_size, device=device, dtype=torch.int64)[
        :num_tokens
    ].to(torch.int32)
    slots[::17] = -1

    q_points_k = q_points_v = thresholds_k = thresholds_v = None
    if grouped_tables:
        num_tables = num_heads * (head_dim // 32)
        base = torch.linspace(0.0, 1.0, 16, dtype=torch.float32, device=device)
        table = torch.arange(num_tables, dtype=torch.float32, device=device)
        q_points_k = base.unsqueeze(0).pow(0.75 + 0.03 * table.unsqueeze(1))
        q_points_v = base.unsqueeze(0).pow(1.10 + 0.02 * table.unsqueeze(1))
        q_points_k[:, 0], q_points_k[:, -1] = 0.0, 1.0
        q_points_v[:, 0], q_points_v[:, -1] = 0.0, 1.0
        thresholds_k = (q_points_k[:, :-1] + q_points_k[:, 1:]) / 2.0
        thresholds_v = (q_points_v[:, :-1] + q_points_v[:, 1:]) / 2.0
        q_points_k = q_points_k.to(torch.float16).contiguous()
        q_points_v = q_points_v.to(torch.float16).contiguous()
        thresholds_k = thresholds_k.to(torch.float16).contiguous()
        thresholds_v = thresholds_v.to(torch.float16).contiguous()

    assert _try_write_tokens_to_paged_cache_triton(
        cache_triton,
        layout,
        slots,
        key,
        value,
        q_points_k=q_points_k,
        q_points_v=q_points_v,
        thresholds_k=thresholds_k,
        thresholds_v=thresholds_v,
    )
    valid = slots >= 0
    _write_tokens_to_paged_cache_reference(
        cache_reference,
        layout,
        torch.empty(0, device=device),
        slots[valid].contiguous(),
        key[valid].contiguous(),
        value[valid].contiguous(),
        q_points_k=q_points_k,
        q_points_v=q_points_v,
        thresholds_k=thresholds_k,
        thresholds_v=thresholds_v,
    )
    torch.testing.assert_close(cache_triton, cache_reference, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not beyond_cute._TRITON_OK,
    reason="requires CUDA Triton",
)
@pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16])
def test_triton_writer_emits_exact_nonuniform_reconstruction_in_place(model_dtype):
    torch.manual_seed(3008)
    device = torch.device("cuda")
    num_tokens = 33
    num_heads = 2
    layout = PagedLayout.build_hnd(
        block_size=128,
        num_kv_heads=num_heads,
        head_dim=128,
        bits=4,
        k_group_size=32,
        v_group_size=32,
    )
    num_tables = num_heads * 4
    base = torch.linspace(0.0, 1.0, 16, dtype=torch.float32, device=device)
    table = torch.arange(num_tables, dtype=torch.float32, device=device)
    q_points_k_f32 = base.unsqueeze(0).pow(0.70 + 0.03 * table.unsqueeze(1))
    q_points_v_f32 = base.unsqueeze(0).pow(1.25 + 0.02 * table.unsqueeze(1))
    q_points_k = q_points_k_f32.to(torch.float16).contiguous()
    q_points_v = q_points_v_f32.to(torch.float16).contiguous()
    thresholds_k = (
        ((q_points_k_f32[:, :-1] + q_points_k_f32[:, 1:]) * 0.5).to(torch.float16).contiguous()
    )
    thresholds_v = (
        ((q_points_v_f32[:, :-1] + q_points_v_f32[:, 1:]) * 0.5).to(torch.float16).contiguous()
    )
    raw_key = torch.randn(num_tokens, num_heads, 128, dtype=model_dtype, device=device)
    raw_value = torch.randn_like(raw_key)
    slots = torch.arange(num_tokens, dtype=torch.int32, device=device)
    slots[7] = -1

    cache_reference = torch.zeros((1, layout.block_i32), dtype=torch.int32, device=device)
    key_reference = torch.empty_like(raw_key)
    value_reference = torch.empty_like(raw_value)
    _write_tokens_to_paged_cache_reference(
        cache_reference,
        layout,
        torch.empty(0, device=device),
        slots,
        raw_key,
        raw_value,
        q_points_k=q_points_k,
        q_points_v=q_points_v,
        thresholds_k=thresholds_k,
        thresholds_v=thresholds_v,
        k_dq_out=key_reference,
        v_dq_out=value_reference,
    )

    cache_triton = torch.zeros_like(cache_reference)
    key_in_place = raw_key.clone()
    value_in_place = raw_value.clone()
    assert _try_write_tokens_to_paged_cache_triton(
        cache_triton,
        layout,
        slots,
        key_in_place,
        value_in_place,
        q_points_k=q_points_k,
        q_points_v=q_points_v,
        thresholds_k=thresholds_k,
        thresholds_v=thresholds_v,
        k_dq_out=key_in_place,
        v_dq_out=value_in_place,
    )

    torch.testing.assert_close(cache_triton, cache_reference, rtol=0, atol=0)
    torch.testing.assert_close(key_in_place, key_reference, rtol=0, atol=0)
    torch.testing.assert_close(value_in_place, value_reference, rtol=0, atol=0)
    assert not torch.equal(key_in_place, raw_key)
    assert not torch.equal(value_in_place, raw_value)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not beyond_cute._TRITON_OK,
    reason="requires CUDA Triton",
)
@pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16])
def test_triton_packed_writer_keeps_right_bucketize_at_exact_thresholds(model_dtype):
    device = torch.device("cuda")
    layout = PagedLayout.build(
        block_size=64,
        num_kv_heads=1,
        head_dim=128,
        bits=4,
        k_group_size=32,
        v_group_size=32,
    )
    cache_triton = torch.zeros((1, layout.block_i32), dtype=torch.int32, device=device)
    cache_reference = torch.zeros_like(cache_triton)
    thresholds = (torch.arange(1, 16, dtype=torch.float32, device=device) / 16).to(torch.float16)
    q_points = (torch.arange(16, dtype=torch.float32, device=device) / 15).to(torch.float16)
    group = torch.zeros(32, dtype=model_dtype, device=device)
    group[1] = 1
    group[2:17] = thresholds.to(model_dtype)
    key = group.repeat(4).view(1, 1, 128)
    value = key.clone()
    slots = torch.zeros(1, dtype=torch.int32, device=device)

    assert _try_write_tokens_to_paged_cache_triton(
        cache_triton,
        layout,
        slots,
        key,
        value,
        q_points_k=q_points,
        q_points_v=q_points,
        thresholds_k=thresholds,
        thresholds_v=thresholds,
    )
    _write_tokens_to_paged_cache_reference(
        cache_reference,
        layout,
        torch.empty(0, device=device),
        slots,
        key,
        value,
        q_points_k=q_points,
        q_points_v=q_points,
        thresholds_k=thresholds,
        thresholds_v=thresholds,
    )

    torch.testing.assert_close(cache_triton, cache_reference, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not beyond_cute._TRITON_OK,
    reason="requires CUDA Triton",
)
def test_divide_free_writer_matches_div_rn_at_rounding_boundary():
    """Guard the FP32 division boundary that a plain affine compare misses."""
    device = torch.device("cuda")
    layout = PagedLayout.build_hnd(
        block_size=128,
        num_kv_heads=1,
        head_dim=128,
        bits=4,
        k_group_size=32,
        v_group_size=32,
    )
    cache_triton = torch.zeros((1, layout.block_i32), dtype=torch.int32, device=device)
    cache_reference = torch.zeros_like(cache_triton)

    # For this representable FP16 group, div.rn(delta, scale) is the FP32
    # predecessor of threshold, while the rounded product scale * threshold
    # equals delta.  Comparing delta >= scale * threshold would therefore
    # assign code 12 instead of the reference code 11.
    group = torch.zeros(32, dtype=torch.float16, device=device)
    group[0] = -18080.0
    group[1] = 1677.0
    group[2] = -1.5751953125
    threshold = torch.tensor(0.9150390625, dtype=torch.float16, device=device)
    thresholds = torch.tensor(
        [
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.70,
            0.9150390625,
            0.94,
            0.96,
            0.98,
        ],
        dtype=torch.float16,
        device=device,
    )
    delta = group[2].float() - group.float().amin()
    scale = group.float().amax() - group.float().amin()
    assert bool((torch.div(delta, scale) < threshold.float()).item())
    assert bool((delta >= scale * threshold.float()).item())

    key = group.repeat(4).view(1, 1, 128)
    value = key.clone()
    slots = torch.zeros(1, dtype=torch.int32, device=device)
    q_points = torch.linspace(0.0, 1.0, 16, dtype=torch.float16, device=device)
    assert _try_write_tokens_to_paged_cache_triton(
        cache_triton,
        layout,
        slots,
        key,
        value,
        q_points_k=q_points,
        q_points_v=q_points,
        thresholds_k=thresholds,
        thresholds_v=thresholds,
    )
    _write_tokens_to_paged_cache_reference(
        cache_reference,
        layout,
        torch.empty(0, device=device),
        slots,
        key,
        value,
        q_points_k=q_points,
        q_points_v=q_points,
        thresholds_k=thresholds,
        thresholds_v=thresholds,
    )

    torch.testing.assert_close(cache_triton, cache_reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reference_writer_masks_slots_without_shortening_dq_outputs():
    torch.manual_seed(827)
    device = torch.device("cuda")
    layout = PagedLayout.build(
        block_size=64,
        num_kv_heads=2,
        head_dim=128,
        bits=4,
        k_group_size=32,
        v_group_size=32,
    )
    cache_seed = torch.randint(
        -(1 << 31),
        1 << 31,
        (2, layout.block_i32),
        dtype=torch.int32,
        device=device,
    )
    cache = cache_seed.clone()
    key = torch.randn(3, 2, 128, dtype=torch.float16, device=device)
    value = torch.randn_like(key)
    slots = torch.tensor([0, -1, 65], dtype=torch.int32, device=device)
    key_dq = torch.empty_like(key)
    value_dq = torch.empty_like(value)

    _write_tokens_to_paged_cache_reference(
        cache,
        layout,
        torch.empty(0, device=device),
        slots,
        key,
        value,
        k_dq_out=key_dq,
        v_dq_out=value_dq,
    )

    expected = cache_seed.clone()
    _write_tokens_to_paged_cache_reference(
        expected,
        layout,
        torch.empty(0, device=device),
        slots[[0, 2]],
        key[[0, 2]],
        value[[0, 2]],
    )
    assert key_dq.shape == key.shape
    assert value_dq.shape == value.shape
    assert torch.isfinite(key_dq).all()
    assert torch.isfinite(value_dq).all()
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


def test_paged_layout_rejects_removed_real_modes():
    with pytest.raises(ValueError, match="4-bit"):
        PagedLayout.build(
            block_size=32,
            num_kv_heads=1,
            head_dim=96,
            bits=5,
            k_group_size=32,
            v_group_size=32,
        )
    with pytest.raises(ValueError, match="group_size"):
        PagedLayout.build(
            block_size=32,
            num_kv_heads=1,
            head_dim=96,
            bits=4,
            k_group_size=24,
            v_group_size=32,
        )
