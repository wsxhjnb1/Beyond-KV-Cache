import math
import types

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("vllm")

from beyond.runtime.vllm import packed_backend as mod  # noqa: E402
from quant.beyond_cute import (  # noqa: E402
    PagedLayout,
    prepare_qpoints_and_thresholds,
    read_request_from_paged_cache,
    write_tokens_to_paged_cache,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Beyond page-table backend parity test requires CUDA",
)


def _make_impl(num_heads: int, num_kv_heads: int, head_size: int):
    impl = mod.BeyondPackedImpl.__new__(mod.BeyondPackedImpl)
    impl.num_heads = num_heads
    impl.num_kv_heads = num_kv_heads
    impl.head_size = head_size
    impl.scale = 1.0 / math.sqrt(head_size)
    impl._q_points_loaded = False
    return impl


def _sdpa_from_cache(
    *,
    query: torch.Tensor,
    cache: torch.Tensor,
    layout: PagedLayout,
    block_table: torch.Tensor,
    q_starts: list[int],
    seq_lens: list[int],
    req_idx: int,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    q_start = q_starts[req_idx]
    q_end = q_starts[req_idx + 1]
    q_len = q_end - q_start
    seq_len = seq_lens[req_idx]
    n_blocks = (seq_len + layout.block_size - 1) // layout.block_size
    block_ids = block_table[req_idx, :n_blocks].contiguous()
    k_ctx, v_ctx = read_request_from_paged_cache(
        cache,
        layout,
        block_ids,
        seq_len,
        out_dtype=query.dtype,
        stat_dtype=query.dtype,
    )
    repeat = num_heads // num_kv_heads
    k_ctx = k_ctx[0].repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
    v_ctx = v_ctx[0].repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
    q_req = query[q_start:q_end].transpose(0, 1).unsqueeze(0)
    attn_mask = None
    is_causal = q_len == seq_len
    if q_len > 1 and q_len != seq_len:
        q_pos = torch.arange(seq_len - q_len, seq_len, device=query.device)
        k_pos = torch.arange(seq_len, device=query.device)
        attn_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    out = F.scaled_dot_product_attention(
        q_req,
        k_ctx,
        v_ctx,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)


def _sdpa_from_fake(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    q_points, thresholds = prepare_qpoints_and_thresholds(
        num_bits=4,
        device=query.device,
        q_points=None,
    )
    key_qdq = mod._fake_quant_token_4d(
        key.unsqueeze(0),
        q_points,
        thresholds,
        32,
    )[0]
    value_qdq = mod._fake_quant_token_4d(
        value.unsqueeze(0),
        q_points,
        thresholds,
        32,
    )[0]
    repeat = num_heads // num_kv_heads
    k_ctx = key_qdq.repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
    v_ctx = value_qdq.repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
    q_req = query.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(
        q_req,
        k_ctx,
        v_ctx,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)


def test_production_prefill_uses_official_flash_attention_over_rounded_kv(monkeypatch):
    monkeypatch.setenv("BEYOND_RUNTIME_AUDIT", "1")
    mod.reset_runtime_audit()

    torch.manual_seed(123)
    device = torch.device("cuda")
    num_heads = 4
    num_kv_heads = 2
    head_size = 32
    block_size = 64
    lengths = [3, 2]
    q_starts = [0, 3, 5]
    total = q_starts[-1]
    layout = PagedLayout.build(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        bits=4,
        k_group_size=32,
        v_group_size=32,
    )
    cache = torch.zeros((2, layout.block_i32), device=device, dtype=torch.int32)
    block_table = torch.tensor([[0], [1]], device=device, dtype=torch.int32)
    slots = torch.tensor([0, 1, 2, block_size, block_size + 1], device=device)
    query = torch.randn(total, num_heads, head_size, device=device, dtype=torch.float16)
    key = torch.randn(total, num_kv_heads, head_size, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    raw_key = key.clone()
    raw_value = value.clone()
    output = torch.empty_like(query)
    observed = {}

    def fake_flash_attn_varlen_func(*, q, k, v, out, **kwargs):
        observed.update(kwargs)
        observed["k"] = k.clone()
        observed["v"] = v.clone()
        for req_idx, (start, end) in enumerate(zip(q_starts, q_starts[1:])):
            repeat = num_heads // num_kv_heads
            q_req = q[start:end].transpose(0, 1).unsqueeze(0)
            k_req = k[start:end].repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
            v_req = v[start:end].repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
            ref = F.scaled_dot_product_attention(
                q_req,
                k_req,
                v_req,
                dropout_p=0.0,
                is_causal=True,
                scale=kwargs["softmax_scale"],
            )
            out[start:end].copy_(ref.squeeze(0).transpose(0, 1))
        return out

    monkeypatch.setattr(mod, "flash_attn_varlen_func", fake_flash_attn_varlen_func)
    mod._run_production_prefill_forward(
        query=query,
        key=key,
        value=value,
        output=output,
        cache=cache,
        layout=layout,
        block_table=block_table,
        slot_mapping=slots,
        query_start_loc=torch.tensor(q_starts, device=device, dtype=torch.int32),
        q_starts=q_starts,
        seq_lens=lengths,
        prefill_reqs=[0, 1],
        q_points_k=None,
        q_points_v=None,
        thresholds_k=None,
        thresholds_v=None,
        softmax_scale=1.0 / math.sqrt(head_size),
        causal=True,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        logits_soft_cap=0.0,
        fa_version=4,
        sinks=None,
    )

    assert observed["fa_version"] == 4
    assert observed["causal"] is True
    assert observed["max_seqlen_q"] == 3
    q_points, thresholds = prepare_qpoints_and_thresholds(
        num_bits=4,
        device=device,
        q_points=None,
    )
    rounded_key = mod._fake_quant_token_4d(raw_key.unsqueeze(0), q_points, thresholds, 32)[0]
    rounded_value = mod._fake_quant_token_4d(raw_value.unsqueeze(0), q_points, thresholds, 32)[0]
    torch.testing.assert_close(observed["k"], rounded_key, rtol=0.0, atol=3e-3)
    torch.testing.assert_close(observed["v"], rounded_value, rtol=0.0, atol=3e-3)
    assert not torch.equal(observed["k"], raw_key)
    assert not torch.equal(observed["v"], raw_value)
    audit = mod.runtime_audit_snapshot()
    assert audit["production_prefill_calls"] == 1
    assert audit["rounded_prefill_attention_calls"] == 1
    for counter in (
        "production_chunked_prefill_calls",
        "page_table_exact_calls",
        "prefix_tail_piecewise_calls",
        "prefix_tail_dense_decode_calls",
        "sm103_nonuniform_decode_launches",
    ):
        assert audit[counter] == 0
    for req_idx, (start, end) in enumerate(zip(q_starts, q_starts[1:])):
        k_ctx, v_ctx = read_request_from_paged_cache(
            cache,
            layout,
            block_table[req_idx],
            end - start,
            out_dtype=torch.float16,
            stat_dtype=torch.float16,
        )
        key_ref = mod._fake_quant_token_4d(
            raw_key[start:end].unsqueeze(0), q_points, thresholds, 32
        )
        value_ref = mod._fake_quant_token_4d(
            raw_value[start:end].unsqueeze(0), q_points, thresholds, 32
        )
        torch.testing.assert_close(k_ctx, key_ref, rtol=0.0, atol=3e-3)
        torch.testing.assert_close(v_ctx, value_ref, rtol=0.0, atol=3e-3)


def test_exact_page_table_backend_handles_prefill_and_decode(monkeypatch):
    monkeypatch.delenv("BEYOND_QUANT_CONFIG", raising=False)
    monkeypatch.setenv("BEYOND_VLLM_PAGE_TABLE_EXACT", "1")

    def fail_fast_decode(*_args, **_kwargs):
        raise AssertionError("FA4/CuTe fast decode should not run in exact mode")

    monkeypatch.setattr(
        mod.BeyondPackedImpl,
        "_run_sm103_nonuniform_dense_decode",
        fail_fast_decode,
    )

    torch.manual_seed(1234)
    device = torch.device("cuda")
    num_heads = 4
    num_kv_heads = 2
    head_size = 32
    block_size = 64
    # The production backend uses the head-major token-word ABI required by
    # the SM103 TMA loader.  Keep the test's pre-populated cache and reference
    # reader on that same physical layout.
    layout = PagedLayout.build_hnd(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        bits=4,
        k_group_size=32,
        v_group_size=32,
    )
    kv_cache = torch.zeros((2, layout.block_i32), device=device, dtype=torch.int32)
    block_table = torch.tensor([[0], [1]], device=device, dtype=torch.int32)

    prefill_len = 5
    decode_ctx_len = 3
    query_prefill = torch.randn(
        prefill_len,
        num_heads,
        head_size,
        device=device,
        dtype=torch.float16,
    )
    key_prefill = torch.randn(
        prefill_len,
        num_kv_heads,
        head_size,
        device=device,
        dtype=torch.float16,
    )
    value_prefill = torch.randn_like(key_prefill)
    query_decode = torch.randn(
        1,
        num_heads,
        head_size,
        device=device,
        dtype=torch.float16,
    )
    key_decode_ctx = torch.randn(
        decode_ctx_len,
        num_kv_heads,
        head_size,
        device=device,
        dtype=torch.float16,
    )
    value_decode_ctx = torch.randn_like(key_decode_ctx)
    key_decode = torch.randn(
        1,
        num_kv_heads,
        head_size,
        device=device,
        dtype=torch.float16,
    )
    value_decode = torch.randn_like(key_decode)

    ctx_slots = torch.arange(
        block_size,
        block_size + decode_ctx_len,
        device=device,
        dtype=torch.int32,
    )
    write_tokens_to_paged_cache(
        kv_cache,
        layout,
        block_table,
        ctx_slots,
        key_decode_ctx,
        value_decode_ctx,
    )

    query = torch.cat([query_prefill, query_decode], dim=0)
    key = torch.cat([key_prefill, key_decode], dim=0)
    value = torch.cat([value_prefill, value_decode], dim=0)
    slot_mapping = torch.tensor(
        [0, 1, 2, 3, 4, block_size + decode_ctx_len],
        device=device,
        dtype=torch.int32,
    )
    metadata = types.SimpleNamespace(
        num_actual_tokens=query.shape[0],
        query_start_loc=torch.tensor(
            [0, prefill_len, prefill_len + 1],
            device=device,
            dtype=torch.int32,
        ),
        seq_lens=torch.tensor(
            [prefill_len, decode_ctx_len + 1],
            device=device,
            dtype=torch.int32,
        ),
        block_table=block_table,
        slot_mapping=slot_mapping,
        max_query_len=prefill_len,
        max_seq_len=prefill_len,
    )
    impl = _make_impl(num_heads, num_kv_heads, head_size)
    layer = types.SimpleNamespace(layer_name="model.layers.0.self_attn")

    out = impl.forward(layer, query, key, value, kv_cache, metadata)

    q_starts = [0, prefill_len, prefill_len + 1]
    seq_lens = [prefill_len, decode_ctx_len + 1]
    ref = torch.empty_like(out)
    ref[:prefill_len].copy_(
        _sdpa_from_cache(
            query=query,
            cache=kv_cache,
            layout=layout,
            block_table=block_table,
            q_starts=q_starts,
            seq_lens=seq_lens,
            req_idx=0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            scale=impl.scale,
        )
    )
    ref[prefill_len:].copy_(
        _sdpa_from_cache(
            query=query,
            cache=kv_cache,
            layout=layout,
            block_table=block_table,
            q_starts=q_starts,
            seq_lens=seq_lens,
            req_idx=1,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            scale=impl.scale,
        )
    )
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)

    fake_ref = torch.cat(
        [
            _sdpa_from_fake(
                query=query_prefill,
                key=key_prefill,
                value=value_prefill,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                scale=impl.scale,
                is_causal=True,
            ),
            _sdpa_from_fake(
                query=query_decode,
                key=torch.cat([key_decode_ctx, key_decode], dim=0),
                value=torch.cat([value_decode_ctx, value_decode], dim=0),
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                scale=impl.scale,
                is_causal=False,
            ),
        ],
        dim=0,
    )
    torch.testing.assert_close(out, fake_ref, rtol=0.0, atol=3e-3)
