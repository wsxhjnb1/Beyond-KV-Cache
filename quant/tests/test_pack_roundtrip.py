"""Roundtrip tests for the real Beyond packed path."""

import pytest
import torch

from beyond.quantization.layers import UnifiedQuantLayer
from quant.beyond_cute import (
    dequantize_k,
    dequantize_v,
    pack_codes,
    quantize_k,
    quantize_v,
    unpack_codes,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("group_size,words", [(32, 4)])
@pytest.mark.parametrize("shape_prefix", [(), (3,), (2, 4)])
def test_pack_unpack_roundtrip_4bit(group_size, words, shape_prefix):
    torch.manual_seed(0)
    shape = (*shape_prefix, group_size)
    codes = torch.randint(0, 16, shape, dtype=torch.uint8, device=DEVICE)
    packed = pack_codes(codes, 4)
    assert packed.shape == (*shape_prefix, words)
    recovered = unpack_codes(packed, 4, group_size)
    assert recovered.shape == codes.shape
    assert torch.equal(recovered, codes)


def test_pack_axis_first_removed():
    codes = torch.zeros(32, dtype=torch.uint8, device=DEVICE)
    with pytest.raises(TypeError, match="pack_axis_first"):
        pack_codes(codes, 4, pack_axis_first=True)


@pytest.mark.parametrize("group_size,words", [(32, 4)])
def test_pack_layout_boundary_values(group_size, words):
    zeros = torch.zeros(group_size, dtype=torch.uint8, device=DEVICE)
    assert torch.all(pack_codes(zeros, 4) == 0)

    maxes = torch.full((group_size,), 15, dtype=torch.uint8, device=DEVICE)
    packed_max = pack_codes(maxes, 4)
    assert packed_max.numel() == words
    popcnt = sum(int(bin(int(w.item())).count("1")) for w in packed_max.flatten())
    assert popcnt == group_size * 4


def test_pack_rejects_removed_group_size():
    codes = torch.zeros(24, dtype=torch.uint8, device=DEVICE)
    with pytest.raises(ValueError, match="group_size"):
        pack_codes(codes, 4)


@pytest.mark.parametrize("group_size", [32])
def test_k_matches_unified_quant_layer_per_token(group_size):
    torch.manual_seed(5)
    B, S, H, D = 2, 13, 2, 96
    x = torch.randn(B, S, H, D, dtype=torch.float16, device=DEVICE)
    pk = quantize_k(x, 4, group_size)
    y_ours = dequantize_k(pk)

    layer = UnifiedQuantLayer(
        num_bits=4,
        group_size=group_size,
        grouping_dim="token",
        quant_width=H * D,
    ).to(DEVICE)
    with torch.no_grad():
        y_ref = layer(x.reshape(B, S, H * D)).reshape(B, S, H, D)

    torch.testing.assert_close(
        y_ours.to(torch.float32),
        y_ref.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("group_size", [32])
def test_v_matches_unified_quant_layer_per_token(group_size):
    torch.manual_seed(13)
    B, S, H, D = 2, 17, 2, 96
    x = torch.randn(B, S, H, D, dtype=torch.float16, device=DEVICE)
    pv = quantize_v(x, 4, group_size)
    y_ours = dequantize_v(pv)

    layer = UnifiedQuantLayer(
        num_bits=4,
        group_size=group_size,
        grouping_dim="token",
        quant_width=H * D,
    ).to(DEVICE)
    with torch.no_grad():
        y_ref = layer(x.reshape(B, S, H * D)).reshape(B, S, H, D)

    torch.testing.assert_close(
        y_ours.to(torch.float32),
        y_ref.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("group_size,words", [(32, 4)])
def test_quantize_v_token_last_matches_token_major(group_size, words):
    torch.manual_seed(16)
    B, S, H, D = 2, 11, 2, 96
    x = torch.randn(B, S, H, D, dtype=torch.float16, device=DEVICE)
    pv = quantize_v(x, 4, group_size)
    pv_tl = quantize_v(x, 4, group_size, pack_token_last=True)
    assert pv_tl.pack_token_last is True
    assert pv_tl.packed.shape == (B, H, D // group_size, words, S)
    assert pv_tl.scale.shape == (B, H, D // group_size, S)
    assert torch.equal(
        pv_tl.packed.permute(0, 4, 1, 2, 3).contiguous(),
        pv.packed,
    )
    assert torch.equal(dequantize_v(pv), dequantize_v(pv_tl))


@pytest.mark.parametrize("group_size", [32])
def test_k_has_no_sequence_tail_residual(group_size):
    torch.manual_seed(7)
    B, S, H, D = 1, 19, 2, 96
    x = torch.randn(B, S, H, D, dtype=torch.float16, device=DEVICE)
    pk = quantize_k(x, 4, group_size)
    assert not hasattr(pk, "residual")
    y = dequantize_k(pk)
    assert y.shape == x.shape


@pytest.mark.parametrize("group_size", [32])
def test_packed_size_matches_4bit_budget(group_size):
    torch.manual_seed(0)
    B, S, H, D = 1, 19, 2, 96
    x = torch.randn(B, S, H, D, dtype=torch.float16, device=DEVICE)
    pk = quantize_k(x, 4, group_size)
    expected_code_bytes = B * S * H * D * 4 // 8
    actual_code_bytes = pk.packed.element_size() * pk.packed.numel()
    assert actual_code_bytes == expected_code_bytes
