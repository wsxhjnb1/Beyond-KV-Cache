"""4-bit packed KV-quant attention for Blackwell GPUs.

This module consolidates the real packed cache layout, tokenwise 4-bit
compression/decompression helpers, and paged-cache I/O primitives.

Layout summary
--------------
* K/V cache grouping: along head_dim per token. Each (token, head, sub_group)
  owns an independent (min, scale). There is no fp16 sidecar tail path.
* Bit packing: real compression supports only 4-bit groups of 32 values.
  These occupy 4 uint32 words.
* q_points LUT: default is uniform ``i/(L-1)``; when ``BEYOND_QUANT_CONFIG``
  is set the backend loads per-(layer, proj) LUTs and passes them through
  as ``q_points_k`` / ``q_points_v``.
* Nonuniform real decode uses the fake-quant-equivalent representation:
  learned thresholds choose the 4-bit bucket during compression and decode
  reconstructs with the learned q_points LUT.
* Production Sq=1 4-bit/G32 decode is implemented by
  ``quant.fa4_cute.beyond_mixed_decode_runtime``. This module owns only the
  packed-cache representation, cache I/O, and numerical reference helpers.

Public API
----------
* ``PackedK`` / ``PackedV``: dataclass holding bit-packed codes + stats
* ``quantize_k`` / ``quantize_v``: pack fp16 -> PackedK/V (numerical oracle)
* ``dequantize_k`` / ``dequantize_v``: torch dequant helpers
* ``PagedLayout``: per-block byte layout for vLLM cache
* ``write_tokens_to_paged_cache``: quantize and write K/V cache tokens
* ``read_request_from_paged_cache``: full K/V read
"""

from dataclasses import dataclass
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import gdc_launch_dependents

    _TRITON_OK = True
except Exception:
    triton = None
    tl = None
    gdc_launch_dependents = None
    _TRITON_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PACK_CHUNK = 32
SUPPORTED_BITS = (4,)
SUPPORTED_GROUP_SIZES = (32,)
CODE_LAYOUT_HEAD_WORD_TOKEN = "head_word_token"
CODE_LAYOUT_HND_TOKEN_WORD = "hnd_token_word"
SUPPORTED_CODE_LAYOUTS = (CODE_LAYOUT_HEAD_WORD_TOKEN, CODE_LAYOUT_HND_TOKEN_WORD)
_UNIFORM_QPOINTS_CACHE: dict[tuple[int, str], torch.Tensor] = {}
_THRESHOLDS_CACHE: dict[tuple[int, str, int], torch.Tensor] = {}
_ARANGE_CACHE: dict[tuple[str, int], torch.Tensor] = {}
_V_WRITE_COMPONENTS_CACHE: dict[
    tuple[str, int, int, int, int, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_CACHE_FP16_VIEW_CACHE: dict = {}
_PACKED_READBACK_PAIRWISE_MIN_TOKEN_HEADS = 65536


# ===========================================================================
# Section 1: pack / unpack / quant / dequant (torch reference + vectorized)
# ===========================================================================


def make_uniform_q_points(num_bits: int) -> torch.Tensor:
    """Default uniform LUT matching ``BaseQuantLayer.__init__`` exactly.

    We replicate the Python-side ``[i / (2^bits - 1) for i in range(2^bits)]``
    list then cast to fp32, because ``torch.linspace`` can differ by 1 ULP at
    some entries which in turn shifts the fp32 thresholds and produces a
    handful of 1-bucket differences at ties.
    """
    if int(num_bits) != 4:
        raise ValueError("Beyond inference keeps only 4-bit quantization")
    levels = 1 << num_bits
    denom = (1 << num_bits) - 1
    q_init = [i / denom for i in range(levels)]
    return torch.tensor(q_init, dtype=torch.float32)


def make_uniform_thresholds(q_points: torch.Tensor) -> torch.Tensor:
    # Must match BaseQuantLayer: (q[:-1] + q[1:]) / 2.0 (division, not 0.5*).
    return (q_points[:-1] + q_points[1:]) / 2.0


def _device_cache_key(device: torch.device) -> str:
    if device.type == "cuda":
        idx = torch.cuda.current_device() if device.index is None else device.index
        return f"cuda:{idx}"
    return str(device)


def _tensor_on_device(t: torch.Tensor, dev_key: str) -> bool:
    return _device_cache_key(t.device) == dev_key


def _cached_arange(n: int, device: torch.device) -> torch.Tensor:
    key = (_device_cache_key(device), int(n))
    cached = _ARANGE_CACHE.get(key)
    if cached is None:
        cached = torch.arange(n, device=device, dtype=torch.long)
        _ARANGE_CACHE[key] = cached
    return cached


def prepare_qpoints_and_thresholds(
    *,
    num_bits: int,
    device: torch.device,
    q_points: Optional[torch.Tensor],
    thresholds: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous deployment tables on ``device``.

    The hot fused paths repeatedly request identical LUTs. Cache both the
    default uniform q_points and thresholds derived from long-lived q_points
    tensors to avoid tiny per-call allocations and pointwise launches. New
    deployment tables remain 16-bit in HBM (FP16 or BF16); legacy FP32 inputs
    stay FP32 for fake-QDQ compatibility.
    """
    if int(num_bits) != 4:
        raise ValueError("Beyond inference keeps only 4-bit quantization")
    uniform_qpoints = q_points is None
    dev_key = _device_cache_key(device)

    if q_points is None:
        key = (num_bits, dev_key)
        qp = _UNIFORM_QPOINTS_CACHE.get(key)
        if qp is None:
            qp = (
                make_uniform_q_points(num_bits)
                .to(
                    device=device,
                    dtype=torch.float16,
                )
                .contiguous()
            )
            _UNIFORM_QPOINTS_CACHE[key] = qp
    else:
        storage_dtype = (
            q_points.dtype
            if q_points.dtype in (torch.float16, torch.bfloat16)
            else torch.float32
        )
        if (
            q_points.dtype == storage_dtype
            and q_points.is_contiguous()
            and _tensor_on_device(q_points, dev_key)
        ):
            qp = q_points
        else:
            qp = q_points.to(device=device, dtype=storage_dtype).contiguous()

    if thresholds is not None:
        if (
            thresholds.dtype == qp.dtype
            and thresholds.is_contiguous()
            and _tensor_on_device(thresholds, dev_key)
        ):
            thr = thresholds
        else:
            thr = thresholds.to(device=device, dtype=qp.dtype).contiguous()
        expected_thr_shape = tuple(qp.shape[:-1]) + (int(qp.shape[-1]) - 1,)
        if tuple(thr.shape) != expected_thr_shape:
            raise ValueError(
                f"thresholds shape {tuple(thr.shape)} != expected {expected_thr_shape}"
            )
        return qp, thr

    if qp.dim() != 1:
        raise ValueError("thresholds are required for grouped q_points")

    thr_key = (qp.data_ptr(), dev_key, qp.numel())
    thr = _THRESHOLDS_CACHE.get(thr_key)
    if thr is None:
        if uniform_qpoints:
            # Match training/export exactly: thresholds are rounded from the
            # FP32 master midpoints, not recomputed from already-rounded FP16
            # codepoints.
            threshold_master = make_uniform_thresholds(make_uniform_q_points(num_bits))
        else:
            threshold_master = make_uniform_thresholds(qp.to(torch.float32))
        thr = threshold_master.to(device=device, dtype=qp.dtype).contiguous()
        _THRESHOLDS_CACHE[thr_key] = thr
    return qp, thr


def _prepare_qpoints_for_decode(
    *,
    num_bits: int,
    device: torch.device,
    q_points: Optional[torch.Tensor],
) -> torch.Tensor:
    if int(num_bits) != 4:
        raise ValueError("Beyond inference keeps only 4-bit quantization")
    dev_key = _device_cache_key(device)
    if q_points is None:
        key = (num_bits, dev_key)
        qp = _UNIFORM_QPOINTS_CACHE.get(key)
        if qp is None:
            qp = (
                make_uniform_q_points(num_bits)
                .to(
                    device=device,
                    dtype=torch.float16,
                )
                .contiguous()
            )
            _UNIFORM_QPOINTS_CACHE[key] = qp
        return qp
    storage_dtype = (
        q_points.dtype
        if q_points.dtype in (torch.float16, torch.bfloat16)
        else torch.float32
    )
    if (
        q_points.dtype == storage_dtype
        and q_points.is_contiguous()
        and _tensor_on_device(q_points, dev_key)
    ):
        return q_points
    return q_points.to(device=device, dtype=storage_dtype).contiguous()


def _prepare_qpoints_and_thresholds(
    *,
    num_bits: int,
    device: torch.device,
    q_points: Optional[torch.Tensor],
    thresholds: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return prepare_qpoints_and_thresholds(
        num_bits=num_bits,
        device=device,
        q_points=q_points,
        thresholds=thresholds,
    )


def _bucketize(x_norm: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """Match UnifiedQuantLayer's bucketize semantics (right=True)."""
    thresholds = thresholds.to(x_norm.device, dtype=torch.float32)
    if thresholds.dim() == 1:
        return torch.bucketize(x_norm, thresholds, right=True)
    if thresholds.dim() != 2 or x_norm.dim() != 5:
        raise ValueError("thresholds must be shared (15,) or per-(head, group) tables")
    _, _, H, G_h, _ = x_norm.shape
    expected_tables = int(H) * int(G_h)
    if int(thresholds.shape[0]) != expected_tables:
        raise ValueError(
            f"thresholds table count {thresholds.shape[0]} != expected {expected_tables}"
        )
    thr = thresholds.reshape(H, G_h, thresholds.shape[-1]).view(1, 1, H, G_h, 1, -1)
    return (x_norm.unsqueeze(-1) >= thr).sum(dim=-1)


def _lookup_qpoints_for_codes(codes: torch.Tensor, q_points: torch.Tensor) -> torch.Tensor:
    q_points = q_points.to(codes.device, dtype=torch.float32)
    codes_l = codes.to(torch.long)
    if q_points.dim() == 1:
        return q_points[codes_l]
    if q_points.dim() != 2 or codes.dim() != 5:
        raise ValueError("q_points must be shared or per-(head, group) tables")
    B, S, H, G_h, G = codes.shape
    expected_tables = int(H) * int(G_h)
    if int(q_points.shape[0]) != expected_tables:
        raise ValueError(f"q_points table count {q_points.shape[0]} != expected {expected_tables}")
    tables = q_points.reshape(H, G_h, q_points.shape[-1]).view(
        1,
        1,
        H,
        G_h,
        1,
        -1,
    )
    tables = tables.expand(B, S, H, G_h, G, q_points.shape[-1])
    return torch.gather(tables, dim=-1, index=codes_l.unsqueeze(-1)).squeeze(-1)


@dataclass
class PackedK:
    """K cache packed state.

    Shapes (with H = num_kv_heads, D = head_dim, S = seqlen, G = group_size,
    W = 4-bit words per group):
      codes:  uint8  (B, S, H, D // G, G)
      packed: uint32 (B, S, H, D // G, W), or
              uint32 (B, H, D // G, W, S) when pack_token_last=True.
      mn:     fp16/bf16 (B, S, H, D // G), or token-last variant.
      scale:  fp16/bf16 same shape as mn.
    """

    # Fast fused paths do not materialize per-element uint8 codes; callers
    # should consume ``packed`` + ``mn`` + ``scale`` instead.
    codes: torch.Tensor
    packed: torch.Tensor
    mn: torch.Tensor
    scale: torch.Tensor
    bits: int
    group_size: int
    pack_token_last: bool = False


@dataclass
class PackedV:
    """V cache packed state.

    Shapes (G_h = head_dim // group_size, G = group_size):
      codes:  uint8 (B, S, H, G_h, G)
      packed: uint32 (B, S, H, G_h, W) OR
              uint32 (B, H, G_h, W, S) (pack_token_last=True;
              token is the innermost axis for coalesced Sq=1 decode V reads).
      mn:     fp16  (B, S, H, G_h) OR (B, H, G_h, S) when pack_token_last=True
      scale:  fp16  same shape as mn
    """

    # Fast fused paths do not materialize per-element uint8 codes; callers
    # should consume ``packed`` + ``mn`` + ``scale`` instead.
    codes: torch.Tensor
    packed: torch.Tensor
    mn: torch.Tensor
    scale: torch.Tensor
    bits: int
    group_size: int
    pack_token_last: bool = False


# ---------------------------------------------------------------------------
# Bit packing: 4-bit groups of 32 codes <-> 4 uint32 words
# ---------------------------------------------------------------------------


def _validate_real_quant_params(num_bits: int, group_size: int) -> None:
    if int(num_bits) != 4:
        raise ValueError("real packed Beyond supports only 4-bit codes")
    if int(group_size) not in SUPPORTED_GROUP_SIZES:
        raise ValueError(f"group_size {group_size} not in {SUPPORTED_GROUP_SIZES}")


def _pack_words_per_group(num_bits: int, group_size: int) -> int:
    _validate_real_quant_params(num_bits, group_size)
    total_bits = int(num_bits) * int(group_size)
    if total_bits % 32 != 0:
        raise ValueError(f"{num_bits}-bit group_size={group_size} is not uint32-aligned")
    return total_bits // 32


def pack_codes(
    codes: torch.Tensor,
    bits: int,
    pack_axis_first: bool = False,
) -> torch.Tensor:
    """Pack a flat ``(..., group_size)`` uint8 tensor into uint32 words.

    The real packed path is intentionally narrow: only 4-bit groups of 32
    values are accepted. ``pack_axis_first`` is kept in the signature for old
    callers but no longer changes the layout because the K path is per-token
    now.
    """
    if pack_axis_first:
        raise TypeError("pack_axis_first was removed from real 4-bit packing")
    if codes.dtype != torch.uint8:
        raise TypeError("codes must be uint8")
    group_size = int(codes.shape[-1])
    words = _pack_words_per_group(bits, group_size)
    codes_c = codes.to(torch.int64)
    out_shape = (*codes.shape[:-1], words)
    out = torch.zeros(out_shape, dtype=torch.int64, device=codes.device)
    mask32 = (1 << 32) - 1
    for w in range(words):
        lo_bit = 32 * w
        hi_bit = 32 * (w + 1)
        acc = torch.zeros(codes_c.shape[:-1], dtype=torch.int64, device=codes.device)
        for i in range(group_size):
            bit_start = i * bits
            bit_end = bit_start + bits
            if bit_end <= lo_bit or bit_start >= hi_bit:
                continue
            shift = bit_start - lo_bit  # may be negative
            if shift >= 0:
                acc = acc | (codes_c[..., i] << shift)
            else:
                acc = acc | (codes_c[..., i] >> (-shift))
        out[..., w] = acc & mask32
    return out.to(torch.uint32)


def unpack_codes(
    packed: torch.Tensor,
    bits: int,
    num_codes: int,
    pack_axis_first: bool = False,
) -> torch.Tensor:
    if pack_axis_first:
        raise TypeError("pack_axis_first was removed from real 4-bit packing")
    words = _pack_words_per_group(bits, num_codes)
    if packed.shape[-1] != words:
        raise ValueError(f"packed shape tail {packed.shape[-1]} != words_per_group {words}")
    packed_nc = packed
    packed_i = packed_nc.to(torch.int64) & 0xFFFFFFFF
    mask = (1 << bits) - 1
    out = torch.zeros(
        (*packed_nc.shape[:-1], num_codes),
        dtype=torch.int64,
        device=packed.device,
    )
    for i in range(num_codes):
        bit_start = i * bits
        bit_end = bit_start + bits
        w_lo = bit_start // 32
        w_hi = (bit_end - 1) // 32
        shift_lo = bit_start - 32 * w_lo
        lo = (packed_i[..., w_lo] >> shift_lo) & mask
        if w_hi == w_lo:
            val = lo
        else:
            spill = bit_end - 32 * w_hi  # bits taken from next word
            hi = packed_i[..., w_hi] & ((1 << spill) - 1)
            val = lo | (hi << (bits - spill))
        out[..., i] = val
    return out.to(torch.uint8)


# ---------------------------------------------------------------------------
# K quantization (grouping along head_dim, per token)
# ---------------------------------------------------------------------------


def quantize_k(
    x: torch.Tensor,
    num_bits: int,
    group_size: int,
    q_points: Optional[torch.Tensor] = None,
    thresholds: Optional[torch.Tensor] = None,
    pack_axis_first: bool = False,
    stat_dtype: Optional[torch.dtype] = None,
) -> PackedK:
    """Quantize K in (B, S, H, D) layout, grouping along D per token."""
    if pack_axis_first:
        raise TypeError("pack_axis_first was removed; K is per-token")
    _validate_real_quant_params(num_bits, group_size)
    if x.dim() != 4:
        raise ValueError("expected x.shape=(B, S, H, D)")
    B, S, H, D = x.shape
    if D % group_size != 0:
        raise ValueError(f"head_dim {D} not divisible by group_size {group_size}")
    q_points, thresholds = prepare_qpoints_and_thresholds(
        num_bits=num_bits,
        device=x.device,
        q_points=q_points,
        thresholds=thresholds,
    )
    q_points = q_points.to(x.device, dtype=torch.float32)
    thresholds = thresholds.to(x.device, dtype=torch.float32)
    stat_dtype = _stat_dtype_from_model_dtype(stat_dtype or x.dtype)

    G = group_size
    G_h = D // G
    x_f = x.to(torch.float32).reshape(B, S, H, G_h, G)

    mn = x_f.amin(dim=-1)  # (B, S, H, G_h)
    mx = x_f.amax(dim=-1)
    scale = (mx - mn).clamp_min(1e-6)
    x_norm = ((x_f - mn.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp(0.0, 1.0)
    codes = _bucketize(x_norm, thresholds).to(torch.uint8)  # (B, S, H, G_h, G)
    packed = pack_codes(codes, num_bits)

    return PackedK(
        codes=codes,
        packed=packed,
        mn=mn.to(stat_dtype),
        scale=scale.to(stat_dtype),
        bits=num_bits,
        group_size=group_size,
        pack_token_last=False,
    )


def dequantize_k(
    packed: PackedK,
    q_points: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return reconstructed K in the packed stats dtype."""
    q_points = _prepare_qpoints_for_decode(
        num_bits=packed.bits,
        device=packed.mn.device,
        q_points=q_points,
    )
    q_points = q_points.to(packed.mn.device, dtype=torch.float32)

    if packed.pack_token_last:
        packed_nc = packed.packed.permute(0, 4, 1, 2, 3).contiguous()
        mn = packed.mn.permute(0, 3, 1, 2).contiguous()
        scale = packed.scale.permute(0, 3, 1, 2).contiguous()
    else:
        packed_nc = packed.packed
        mn = packed.mn
        scale = packed.scale

    G = packed.group_size
    codes = unpack_codes(packed_nc, packed.bits, num_codes=G)  # (B, S, H, G_h, G)
    vals = _lookup_qpoints_for_codes(codes, q_points)
    mn = mn.to(torch.float32).unsqueeze(-1)
    scale = scale.to(torch.float32).unsqueeze(-1)
    xq = vals * scale + mn
    B, S, H, G_h, _ = codes.shape
    return xq.reshape(B, S, H, G_h * G).to(packed.mn.dtype)


# ---------------------------------------------------------------------------
# V quantization (grouping along head_dim, per token)
# ---------------------------------------------------------------------------


def quantize_v(
    x: torch.Tensor,
    num_bits: int,
    group_size: int,
    q_points: Optional[torch.Tensor] = None,
    thresholds: Optional[torch.Tensor] = None,
    pack_token_last: bool = False,
    stat_dtype: Optional[torch.dtype] = None,
) -> PackedV:
    """Quantize V in (B, S, H, D) layout, grouping along D.

    ``pack_token_last`` stores packed words as
    ``(B, H, G_h, num_chunks, bits, S)`` so Sq=1 decode warps mapped across
    tokens can issue coalesced V-code loads.
    """
    _validate_real_quant_params(num_bits, group_size)
    if x.dim() != 4:
        raise ValueError("expected x.shape=(B, S, H, D)")
    B, S, H, D = x.shape
    if D % group_size != 0:
        raise ValueError(f"head_dim {D} not divisible by group_size {group_size}")
    q_points, thresholds = prepare_qpoints_and_thresholds(
        num_bits=num_bits,
        device=x.device,
        q_points=q_points,
        thresholds=thresholds,
    )
    q_points = q_points.to(x.device, dtype=torch.float32)
    thresholds = thresholds.to(x.device, dtype=torch.float32)
    stat_dtype = _stat_dtype_from_model_dtype(stat_dtype or x.dtype)

    G = group_size
    G_h = D // G
    x_f = x.to(torch.float32).reshape(B, S, H, G_h, G)

    mn = x_f.amin(dim=-1)  # (B, S, H, G_h)
    mx = x_f.amax(dim=-1)
    scale = (mx - mn).clamp_min(1e-6)
    x_norm = ((x_f - mn.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp(0.0, 1.0)
    codes = _bucketize(x_norm, thresholds).to(torch.uint8)  # (B, S, H, G_h, G)

    packed = pack_codes(codes, num_bits)
    if pack_token_last:
        # Put token last so decode threads that map to consecutive tokens read
        # contiguous words for a fixed (head, group, word).
        packed = packed.permute(0, 2, 3, 4, 1).contiguous()
        mn_out = mn.to(stat_dtype).permute(0, 2, 3, 1).contiguous()
        scale_out = scale.to(stat_dtype).permute(0, 2, 3, 1).contiguous()
    else:
        mn_out = mn.to(stat_dtype)
        scale_out = scale.to(stat_dtype)
    return PackedV(
        codes=codes,
        packed=packed,
        mn=mn_out,
        scale=scale_out,
        bits=num_bits,
        group_size=group_size,
        pack_token_last=pack_token_last,
    )


def dequantize_v(
    packed: PackedV,
    q_points: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return fp16 (B, S, H, D) reconstructed V."""
    q_points = _prepare_qpoints_for_decode(
        num_bits=packed.bits,
        device=packed.mn.device,
        q_points=q_points,
    )
    q_points = q_points.to(packed.mn.device, dtype=torch.float32)

    # Normalize to legacy layout for per-group unpack (ref path, not perf).
    if packed.pack_token_last:
        packed_nc = packed.packed.permute(0, 4, 1, 2, 3).contiguous()
        mn = packed.mn.permute(0, 3, 1, 2).contiguous()
        scale = packed.scale.permute(0, 3, 1, 2).contiguous()
    else:
        packed_nc = packed.packed
        mn = packed.mn
        scale = packed.scale

    G = packed.group_size
    codes = unpack_codes(packed_nc, packed.bits, num_codes=G)  # (B, S, H, G_h, G)
    vals = _lookup_qpoints_for_codes(codes, q_points)
    mn = mn.to(torch.float32).unsqueeze(-1)
    scale = scale.to(torch.float32).unsqueeze(-1)
    xq = vals * scale + mn
    B, S, H, G_h, _ = codes.shape
    return xq.reshape(B, S, H, G_h * G).to(packed.mn.dtype)


# ---------------------------------------------------------------------------
# Reference end-to-end roundtrip matching UnifiedQuantLayer exactly
# ---------------------------------------------------------------------------


# ===========================================================================
# Section 2: paged KV cache layout + write/read helpers
# ===========================================================================


@dataclass(frozen=True)
class PagedLayout:
    """Offsets (in fp16 halves) into a paged block buffer.

    We address blocks as fp16 arrays internally because all tensors are a
    multiple of 2 bytes. Scale/min pairs are physically interleaved in one
    int32 word per (head, group, token), while the half-count fields retain
    the logical size accounting used to place the code/stat regions.
    """

    block_size: int
    num_kv_heads: int
    head_dim: int
    bits: int
    k_group_size: int
    v_group_size: int

    # Sub-region element counts (fp16 halves)
    k_code_halves: int  # codes stored as uint32, occupies 2 halves per word
    k_scale_halves: int
    k_min_halves: int
    v_code_halves: int
    v_scale_halves: int
    v_min_halves: int

    # Offsets in halves
    k_code_off: int
    k_scale_off: int
    k_min_off: int
    v_code_off: int
    v_scale_off: int
    v_min_off: int

    block_halves: int  # total halves per block
    block_i32: int  # == block_halves // 2

    # Derived
    k_groups_per_block: int
    num_chunks_k: int
    num_chunks_v: int
    v_groups_per_token: int
    code_layout: str = CODE_LAYOUT_HEAD_WORD_TOKEN
    abi_version: int = 1
    stats_interleaved: bool = True
    v_stats_group_major: bool = True

    @classmethod
    def build(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        bits: int,
        k_group_size: int,
        v_group_size: int,
        code_layout: str = CODE_LAYOUT_HEAD_WORD_TOKEN,
    ) -> "PagedLayout":
        _validate_real_quant_params(bits, k_group_size)
        _validate_real_quant_params(bits, v_group_size)
        if code_layout not in SUPPORTED_CODE_LAYOUTS:
            raise ValueError(
                f"unsupported code_layout {code_layout!r}; expected one of {SUPPORTED_CODE_LAYOUTS}"
            )
        if head_dim % k_group_size != 0:
            raise ValueError(f"head_dim {head_dim} not divisible by k_group_size {k_group_size}")
        if head_dim % v_group_size != 0:
            raise ValueError(f"head_dim {head_dim} not divisible by v_group_size {v_group_size}")

        k_groups_per_token = head_dim // k_group_size
        v_groups_per_token = head_dim // v_group_size
        k_groups_per_block = k_groups_per_token
        num_chunks_k = _pack_words_per_group(bits, k_group_size)
        num_chunks_v = _pack_words_per_group(bits, v_group_size)

        # ABI v1 stores codes and statistics group/word-major within a head.
        # ABI v2 makes both code regions and K statistics token-major for TMA.
        # V statistics deliberately remain group-major in both ABIs because
        # the PV converter consumes one G32 table across all 128 page tokens.
        # Region sizes and K/stats/V/stats offsets are identical across ABIs.
        k_code_words = num_kv_heads * block_size * k_groups_per_token * num_chunks_k
        k_code_halves = k_code_words * 2  # uint32 == 2 fp16 halves

        k_stat_halves_one = num_kv_heads * block_size * k_groups_per_token
        k_scale_halves = k_stat_halves_one
        k_min_halves = k_stat_halves_one

        # V codes use the same head-partitioned token-major order as K.
        v_code_words = num_kv_heads * block_size * v_groups_per_token * num_chunks_v
        v_code_halves = v_code_words * 2

        # V stats: (H_kv, G_h, block_size) fp16 each.
        v_stat_halves_one = num_kv_heads * block_size * v_groups_per_token
        v_scale_halves = v_stat_halves_one
        v_min_halves = v_stat_halves_one

        # Offsets
        k_code_off = 0
        k_scale_off = k_code_off + k_code_halves
        k_min_off = k_scale_off + k_scale_halves
        v_code_off = k_min_off + k_min_halves
        v_scale_off = v_code_off + v_code_halves
        v_min_off = v_scale_off + v_scale_halves
        block_halves = v_min_off + v_min_halves
        if block_halves % 2 != 0:
            raise RuntimeError("block_halves not even -- off by one bug")
        block_i32 = block_halves // 2

        return cls(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bits=bits,
            k_group_size=k_group_size,
            v_group_size=v_group_size,
            k_code_halves=k_code_halves,
            k_scale_halves=k_scale_halves,
            k_min_halves=k_min_halves,
            v_code_halves=v_code_halves,
            v_scale_halves=v_scale_halves,
            v_min_halves=v_min_halves,
            k_code_off=k_code_off,
            k_scale_off=k_scale_off,
            k_min_off=k_min_off,
            v_code_off=v_code_off,
            v_scale_off=v_scale_off,
            v_min_off=v_min_off,
            block_halves=block_halves,
            block_i32=block_i32,
            k_groups_per_block=k_groups_per_block,
            num_chunks_k=num_chunks_k,
            num_chunks_v=num_chunks_v,
            v_groups_per_token=v_groups_per_token,
            code_layout=code_layout,
            abi_version=2 if code_layout == CODE_LAYOUT_HND_TOKEN_WORD else 1,
            stats_interleaved=True,
            v_stats_group_major=True,
        )

    @classmethod
    def build_hnd(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        bits: int,
        k_group_size: int,
        v_group_size: int,
    ) -> "PagedLayout":
        """Build ABI v2 with contiguous packed-D words inside each HND token."""
        return cls.build(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bits=bits,
            k_group_size=k_group_size,
            v_group_size=v_group_size,
            code_layout=CODE_LAYOUT_HND_TOKEN_WORD,
        )


def _stat_dtype_from_model_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.bfloat16:
        return torch.bfloat16
    if dtype == torch.float16:
        return torch.float16
    raise TypeError(f"Beyond stats require fp16/bf16 model dtype, got {dtype}")


def _stat_dtype_from_tensor(t: torch.Tensor) -> torch.dtype:
    return _stat_dtype_from_model_dtype(t.dtype)


def _is_supported_model_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.bfloat16)


def _check_same_model_dtype(*tensors: torch.Tensor) -> torch.dtype:
    if not tensors:
        return torch.float16
    dtype = tensors[0].dtype
    _stat_dtype_from_model_dtype(dtype)
    for t in tensors[1:]:
        if t.dtype != dtype:
            raise TypeError(f"Beyond tensors must share dtype {dtype}, got {t.dtype}")
    return dtype


def _view_block_as_stat(
    cache: torch.Tensor,
    stat_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """(num_blocks, block_i32) int32 -> 16-bit mn/scale stat view."""
    if cache.dtype != torch.int32:
        raise TypeError(f"cache must be int32, got {cache.dtype}")
    stat_dtype = _stat_dtype_from_model_dtype(stat_dtype)
    nb, bi = cache.shape
    key = (cache.data_ptr(), tuple(cache.shape), tuple(cache.stride()), stat_dtype)
    cached = _CACHE_FP16_VIEW_CACHE.get(key)
    if cached is not None:
        return cached[1]
    if len(_CACHE_FP16_VIEW_CACHE) > 128:
        _CACHE_FP16_VIEW_CACHE.clear()
    view = cache.view(stat_dtype).view(nb, bi * 2)
    _CACHE_FP16_VIEW_CACHE[key] = (cache, view)
    return view


def _view_block_as_half(cache: torch.Tensor) -> torch.Tensor:
    """Raw fp16 view used when copying packed code words as 16-bit halves."""
    return _view_block_as_stat(cache, torch.float16)


def _block_halves(
    cache_half: torch.Tensor, layout: PagedLayout, block_idx: torch.Tensor
) -> torch.Tensor:
    """(num_blocks, block_halves) fp16, gather by block_idx -> (len(block_idx), block_halves)."""
    return cache_half.index_select(0, block_idx.to(torch.long))


def _v_write_components(
    layout: PagedLayout,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (
        _device_cache_key(device),
        layout.num_kv_heads,
        layout.v_groups_per_token,
        layout.num_chunks_v,
        layout.bits,
        layout.block_size,
    )
    cached = _V_WRITE_COMPONENTS_CACHE.get(key)
    if cached is not None:
        return cached

    v_codes_per_tok = layout.num_kv_heads * layout.v_groups_per_token * layout.num_chunks_v
    words_per_head = layout.v_groups_per_token * layout.num_chunks_v
    words_per_group = layout.num_chunks_v
    arange_c = _cached_arange(v_codes_per_tok, device)
    h_c = arange_c // words_per_head
    rem_c = arange_c - h_c * words_per_head
    g_c = rem_c // words_per_group
    word_c = rem_c - g_c * words_per_group

    pitch_s = layout.num_kv_heads * layout.v_groups_per_token
    arange_s = _cached_arange(pitch_s, device)
    h_s = arange_s // layout.v_groups_per_token
    g_s = arange_s - h_s * layout.v_groups_per_token

    cached = (h_c, g_c, word_c, h_s, g_s)
    _V_WRITE_COMPONENTS_CACHE[key] = cached
    return cached


if _TRITON_OK:

    @triton.jit
    def _packed_div_rn_threshold_ge_tl(
        delta,
        scale,
        threshold,
    ):
        """Compare ``div.rn(delta, scale) >= threshold`` without division.

        Deployment thresholds originate in FP16/BF16, so their FP32
        significands are even.  Round-to-nearest therefore changes from the
        predecessor to ``threshold`` at their exact midpoint.  Compare the
        affine-domain operands against that midpoint with an FMA product
        residual so the multiplication rounding cannot move the boundary.
        """
        threshold_bits = threshold.to(tl.uint32, bitcast=True)
        predecessor = (threshold_bits - 1).to(tl.float32, bitcast=True)
        half_ulp = (threshold - predecessor) * 0.5
        residual = tl.fma(-scale, threshold, delta)
        return residual >= -(scale * half_ulp)

    @triton.jit
    def _packed_threshold_code_4bit_divide_free_tl(
        delta,
        scale,
        thresholds,
        BLOCK_G: tl.constexpr,
    ):
        """Exact divide-free bucketize for one normalized 4-bit group."""
        threshold = tl.load(thresholds + 7).to(tl.float32)
        codes = tl.where(
            _packed_div_rn_threshold_ge_tl(delta, scale, threshold), 8, 0
        ).to(tl.int32)
        probe = codes + 3
        threshold = tl.load(thresholds + probe).to(tl.float32)
        codes += tl.where(
            _packed_div_rn_threshold_ge_tl(delta, scale, threshold), 4, 0
        )
        probe = codes + 1
        threshold = tl.load(thresholds + probe).to(tl.float32)
        codes += tl.where(
            _packed_div_rn_threshold_ge_tl(delta, scale, threshold), 2, 0
        )
        probe = codes
        threshold = tl.load(thresholds + probe).to(tl.float32)
        codes += tl.where(
            _packed_div_rn_threshold_ge_tl(delta, scale, threshold), 1, 0
        )
        return codes

    @triton.jit
    def _write_packed_kv_cache_group_head128_kernel(
        key,
        value,
        cache_i32,
        cache_stat,
        slot_mapping,
        thresholds_k,
        thresholds_v,
        q_points_k,
        q_points_v,
        k_dq_out,
        v_dq_out,
        key_stride_t: tl.constexpr,
        key_stride_h: tl.constexpr,
        key_stride_d: tl.constexpr,
        value_stride_t: tl.constexpr,
        value_stride_h: tl.constexpr,
        value_stride_d: tl.constexpr,
        k_dq_stride_t: tl.constexpr,
        k_dq_stride_h: tl.constexpr,
        k_dq_stride_d: tl.constexpr,
        v_dq_stride_t: tl.constexpr,
        v_dq_stride_h: tl.constexpr,
        v_dq_stride_d: tl.constexpr,
        slot_stride: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_I32: tl.constexpr,
        K_CODE_I32: tl.constexpr,
        K_STATS_I32: tl.constexpr,
        V_CODE_I32: tl.constexpr,
        V_STATS_I32: tl.constexpr,
        HND_TOKEN_WORD: tl.constexpr,
        GROUPED_K: tl.constexpr,
        GROUPED_V: tl.constexpr,
        WRITE_DQ: tl.constexpr,
        STAT_BF16: tl.constexpr,
        SIGNAL_PDL: tl.constexpr,
        GROUPS_PER_PROGRAM: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        """Quantize, pack, and scatter one token/head K/V pair."""
        token = tl.program_id(0)
        head_group = tl.program_id(1)
        if GROUPS_PER_PROGRAM == 1:
            head = head_group // 4
            group_base = head_group % 4
        else:
            head = head_group
            group_base = 0
        slot = tl.load(slot_mapping + token * slot_stride).to(tl.int64)
        # The following CuTe attention launch may use programmatic dependent
        # launch.  Signal as soon as every writer CTA has started; its consumer
        # performs griddepcontrol.wait before the first cache read, so these
        # quantization/stores can overlap launch latency without weakening the
        # consumer's read-after-write dependency.
        if SIGNAL_PDL:
            gdc_launch_dependents()
        if not WRITE_DQ:
            if slot < 0:
                return

        block = slot // BLOCK_SIZE
        block_offset = slot % BLOCK_SIZE
        offs = tl.arange(0, BLOCK_G)
        nibble_shift = (offs % 8) * 4

        for group_offset in tl.static_range(0, GROUPS_PER_PROGRAM):
            group = group_base + group_offset
            table = head * 4 + group
            d = group * 32 + offs

            src_k = key + token * key_stride_t + head * key_stride_h + d * key_stride_d
            vals_k = tl.load(src_k).to(tl.float32)
            mn_k = tl.min(vals_k, axis=0)
            mx_k = tl.max(vals_k, axis=0)
            scale_k = tl.maximum(mx_k - mn_k, 1.0e-6)
            delta_k = vals_k - mn_k
            threshold_base_k = thresholds_k + (table * 15 if GROUPED_K else 0)
            codes_k = _packed_threshold_code_4bit_divide_free_tl(
                delta_k,
                scale_k,
                threshold_base_k,
                BLOCK_G,
            )
            codes_k = tl.where(
                (mx_k > mn_k) & (vals_k == mx_k), 15, codes_k
            )
            codes_k = tl.where(vals_k == mn_k, 0, codes_k)

            src_v = value + token * value_stride_t + head * value_stride_h + d * value_stride_d
            vals_v = tl.load(src_v).to(tl.float32)
            mn_v = tl.min(vals_v, axis=0)
            mx_v = tl.max(vals_v, axis=0)
            scale_v = tl.maximum(mx_v - mn_v, 1.0e-6)
            delta_v = vals_v - mn_v
            threshold_base_v = thresholds_v + (table * 15 if GROUPED_V else 0)
            codes_v = _packed_threshold_code_4bit_divide_free_tl(
                delta_v,
                scale_v,
                threshold_base_v,
                BLOCK_G,
            )
            codes_v = tl.where(
                (mx_v > mn_v) & (vals_v == mx_v), 15, codes_v
            )
            codes_v = tl.where(vals_v == mn_v, 0, codes_v)

            if WRITE_DQ:
                # Cache statistics are physically stored in the model dtype.
                # Round them before reconstruction so this direct output is
                # bit-identical to a later packed-cache dequantization.
                if STAT_BF16:
                    stored_scale_k = scale_k.to(tl.bfloat16).to(tl.float32)
                    stored_mn_k = mn_k.to(tl.bfloat16).to(tl.float32)
                    stored_scale_v = scale_v.to(tl.bfloat16).to(tl.float32)
                    stored_mn_v = mn_v.to(tl.bfloat16).to(tl.float32)
                else:
                    stored_scale_k = scale_k.to(tl.float16).to(tl.float32)
                    stored_mn_k = mn_k.to(tl.float16).to(tl.float32)
                    stored_scale_v = scale_v.to(tl.float16).to(tl.float32)
                    stored_mn_v = mn_v.to(tl.float16).to(tl.float32)
                qpoint_base_k = table * 16 if GROUPED_K else 0
                qpoint_base_v = table * 16 if GROUPED_V else 0
                normalized_k = tl.load(q_points_k + qpoint_base_k + codes_k).to(
                    tl.float32
                )
                normalized_v = tl.load(q_points_v + qpoint_base_v + codes_v).to(
                    tl.float32
                )
                reconstructed_k = normalized_k * stored_scale_k + stored_mn_k
                reconstructed_v = normalized_v * stored_scale_v + stored_mn_v
                tl.store(
                    k_dq_out
                    + token * k_dq_stride_t
                    + head * k_dq_stride_h
                    + d * k_dq_stride_d,
                    reconstructed_k,
                )
                tl.store(
                    v_dq_out
                    + token * v_dq_stride_t
                    + head * v_dq_stride_h
                    + d * v_dq_stride_d,
                    reconstructed_v,
                )

            for word in tl.static_range(0, 4):
                word_lanes = (offs // 8) == word
                packed_k = tl.sum(
                    tl.where(word_lanes, codes_k << nibble_shift, 0),
                    axis=0,
                )
                packed_v = tl.sum(
                    tl.where(word_lanes, codes_v << nibble_shift, 0),
                    axis=0,
                )
                packed_word = group * 4 + word
                if HND_TOKEN_WORD:
                    k_code_index = (
                        block * BLOCK_I32
                        + K_CODE_I32
                        + head * (BLOCK_SIZE * 16)
                        + block_offset * 16
                        + packed_word
                    )
                    v_code_index = (
                        block * BLOCK_I32
                        + V_CODE_I32
                        + head * (BLOCK_SIZE * 16)
                        + block_offset * 16
                        + packed_word
                    )
                else:
                    k_code_index = (
                        block * BLOCK_I32
                        + K_CODE_I32
                        + head * (16 * BLOCK_SIZE)
                        + packed_word * BLOCK_SIZE
                        + block_offset
                    )
                    v_code_index = (
                        block * BLOCK_I32
                        + V_CODE_I32
                        + head * (16 * BLOCK_SIZE)
                        + packed_word * BLOCK_SIZE
                        + block_offset
                    )
                if WRITE_DQ:
                    tl.store(cache_i32 + k_code_index, packed_k, mask=slot >= 0)
                    tl.store(cache_i32 + v_code_index, packed_v, mask=slot >= 0)
                else:
                    tl.store(cache_i32 + k_code_index, packed_k)
                    tl.store(cache_i32 + v_code_index, packed_v)

            if HND_TOKEN_WORD:
                k_stat_index = (
                    block * BLOCK_I32
                    + K_STATS_I32
                    + head * (BLOCK_SIZE * 4)
                    + block_offset * 4
                    + group
                )
            else:
                k_stat_index = (
                    block * BLOCK_I32
                    + K_STATS_I32
                    + head * (4 * BLOCK_SIZE)
                    + group * BLOCK_SIZE
                    + block_offset
                )
            v_stat_index = (
                block * BLOCK_I32
                + V_STATS_I32
                + head * (4 * BLOCK_SIZE)
                + group * BLOCK_SIZE
                + block_offset
            )
            # The physical ABI stores one interleaved 16-bit
            # (scale, minimum) pair in each int32 stats word.  Stats use the
            # model dtype so BF16 deployments retain BF16's exponent range.
            if WRITE_DQ:
                tl.store(cache_stat + 2 * k_stat_index, scale_k, mask=slot >= 0)
                tl.store(cache_stat + 2 * k_stat_index + 1, mn_k, mask=slot >= 0)
                tl.store(cache_stat + 2 * v_stat_index, scale_v, mask=slot >= 0)
                tl.store(cache_stat + 2 * v_stat_index + 1, mn_v, mask=slot >= 0)
            else:
                tl.store(cache_stat + 2 * k_stat_index, scale_k)
                tl.store(cache_stat + 2 * k_stat_index + 1, mn_k)
                tl.store(cache_stat + 2 * v_stat_index, scale_v)
                tl.store(cache_stat + 2 * v_stat_index + 1, mn_v)

    @triton.jit(do_not_specialize=("max_seq_log2",))
    def _dequantize_paged_kv_hnd_head128_kernel(
        cache_i32,
        cache_stat,
        block_table,
        seq_lens,
        seq_start_loc,
        q_points_k,
        q_points_v,
        out_k,
        out_v,
        max_seq_log2,
        block_table_stride_b: tl.constexpr,
        block_table_stride_p: tl.constexpr,
        out_k_stride_t: tl.constexpr,
        out_k_stride_h: tl.constexpr,
        out_k_stride_d: tl.constexpr,
        out_v_stride_t: tl.constexpr,
        out_v_stride_h: tl.constexpr,
        out_v_stride_d: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_I32: tl.constexpr,
        K_CODE_I32: tl.constexpr,
        K_STATS_I32: tl.constexpr,
        V_CODE_I32: tl.constexpr,
        V_STATS_I32: tl.constexpr,
        GROUPED_K: tl.constexpr,
        GROUPED_V: tl.constexpr,
        OUTPUT_BF16: tl.constexpr,
        TOKENS_PER_PROGRAM: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        FLAT_GRID: tl.constexpr,
    ):
        """Materialize packed HND K/V for FA4 chunked-prefill attention."""
        lane = tl.arange(0, TOKENS_PER_PROGRAM * HEAD_DIM)
        token_offset = lane // HEAD_DIM
        d = lane % HEAD_DIM
        if FLAT_GRID:
            request_position = tl.program_id(0) * TOKENS_PER_PROGRAM + token_offset
            request = request_position >> max_seq_log2
            position = request_position & ((1 << max_seq_log2) - 1)
            head = tl.program_id(1)
        else:
            request_head = tl.program_id(0)
            request = request_head // NUM_HEADS
            head = request_head % NUM_HEADS
            position = tl.program_id(1) * TOKENS_PER_PROGRAM + token_offset
        seq_len = tl.load(seq_lens + request).to(tl.int32)
        valid = position < seq_len

        logical_page = position // BLOCK_SIZE
        physical_page = tl.load(
            block_table
            + request * block_table_stride_b
            + logical_page * block_table_stride_p,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid = valid & (physical_page >= 0)
        block_offset = position % BLOCK_SIZE
        group = d // 32
        table = head * 4 + group
        packed_word = group * 4 + (d % 32) // 8
        nibble_shift = (d % 8) * 4

        code_base = (
            physical_page * BLOCK_I32
            + head * (BLOCK_SIZE * 16)
            + block_offset * 16
            + packed_word
        )
        packed_k = tl.load(
            cache_i32 + K_CODE_I32 + code_base,
            mask=valid,
            other=0,
        ).to(tl.uint32)
        packed_v = tl.load(
            cache_i32 + V_CODE_I32 + code_base,
            mask=valid,
            other=0,
        ).to(tl.uint32)
        code_k = ((packed_k >> nibble_shift) & 0xF).to(tl.int32)
        code_v = ((packed_v >> nibble_shift) & 0xF).to(tl.int32)

        k_stat_index = (
            physical_page * BLOCK_I32
            + K_STATS_I32
            + head * (BLOCK_SIZE * 4)
            + block_offset * 4
            + group
        )
        v_stat_index = (
            physical_page * BLOCK_I32
            + V_STATS_I32
            + head * (4 * BLOCK_SIZE)
            + group * BLOCK_SIZE
            + block_offset
        )
        scale_k = tl.load(
            cache_stat + 2 * k_stat_index,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        minimum_k = tl.load(
            cache_stat + 2 * k_stat_index + 1,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        scale_v = tl.load(
            cache_stat + 2 * v_stat_index,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        minimum_v = tl.load(
            cache_stat + 2 * v_stat_index + 1,
            mask=valid,
            other=0.0,
        ).to(tl.float32)

        qpoint_base_k = table * 16 if GROUPED_K else 0
        qpoint_base_v = table * 16 if GROUPED_V else 0
        normalized_k = tl.load(q_points_k + qpoint_base_k + code_k).to(tl.float32)
        normalized_v = tl.load(q_points_v + qpoint_base_v + code_v).to(tl.float32)
        value_k = normalized_k * scale_k + minimum_k
        value_v = normalized_v * scale_v + minimum_v

        out_token = tl.load(seq_start_loc + request).to(tl.int64) + position
        out_k_ptr = (
            out_k
            + out_token * out_k_stride_t
            + head * out_k_stride_h
            + d * out_k_stride_d
        )
        out_v_ptr = (
            out_v
            + out_token * out_v_stride_t
            + head * out_v_stride_h
            + d * out_v_stride_d
        )
        tl.store(out_k_ptr, value_k, mask=valid)
        tl.store(out_v_ptr, value_v, mask=valid)


    @triton.jit(do_not_specialize=("max_seq_log2",))
    def _dequantize_paged_kv_hnd_head128_pairwise_kernel(
        cache_i32,
        cache_stat,
        block_table,
        seq_lens,
        seq_start_loc,
        q_points_k,
        q_points_v,
        out_k,
        out_v,
        max_seq_log2,
        block_table_stride_b: tl.constexpr,
        block_table_stride_p: tl.constexpr,
        out_k_stride_t: tl.constexpr,
        out_k_stride_h: tl.constexpr,
        out_k_stride_d: tl.constexpr,
        out_v_stride_t: tl.constexpr,
        out_v_stride_h: tl.constexpr,
        out_v_stride_d: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_I32: tl.constexpr,
        K_CODE_I32: tl.constexpr,
        K_STATS_I32: tl.constexpr,
        V_CODE_I32: tl.constexpr,
        V_STATS_I32: tl.constexpr,
        GROUPED_K: tl.constexpr,
        GROUPED_V: tl.constexpr,
        OUTPUT_BF16: tl.constexpr,
        TOKENS_PER_PROGRAM: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        FLAT_GRID: tl.constexpr,
    ):
        """Materialize two adjacent 4-bit values per logical Triton lane.

        Both values share their packed word, page metadata, and G32 affine
        statistics.  K and V still perform independent, data-dependent lookups
        in their exact 16-entry nonuniform codebooks.
        """
        pairs_per_token: tl.constexpr = HEAD_DIM // 2
        lane = tl.arange(0, TOKENS_PER_PROGRAM * pairs_per_token)
        token_offset = lane // pairs_per_token
        pair = lane % pairs_per_token
        d0 = pair * 2
        if FLAT_GRID:
            request_position = tl.program_id(0) * TOKENS_PER_PROGRAM + token_offset
            request = request_position >> max_seq_log2
            position = request_position & ((1 << max_seq_log2) - 1)
            head = tl.program_id(1)
        else:
            request_head = tl.program_id(0)
            request = request_head // NUM_HEADS
            head = request_head % NUM_HEADS
            position = tl.program_id(1) * TOKENS_PER_PROGRAM + token_offset
        seq_len = tl.load(seq_lens + request).to(tl.int32)
        valid = position < seq_len

        logical_page = position // BLOCK_SIZE
        physical_page = tl.load(
            block_table
            + request * block_table_stride_b
            + logical_page * block_table_stride_p,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid = valid & (physical_page >= 0)
        block_offset = position % BLOCK_SIZE
        group = d0 // 32
        table = head * 4 + group
        packed_word = group * 4 + (d0 % 32) // 8
        nibble_shift = (d0 % 8) * 4

        code_base = (
            physical_page * BLOCK_I32
            + head * (BLOCK_SIZE * 16)
            + block_offset * 16
            + packed_word
        )
        packed_k = tl.load(
            cache_i32 + K_CODE_I32 + code_base,
            mask=valid,
            other=0,
        ).to(tl.uint32)
        packed_v = tl.load(
            cache_i32 + V_CODE_I32 + code_base,
            mask=valid,
            other=0,
        ).to(tl.uint32)
        code_k0 = ((packed_k >> nibble_shift) & 0xF).to(tl.int32)
        code_k1 = ((packed_k >> (nibble_shift + 4)) & 0xF).to(tl.int32)
        code_v0 = ((packed_v >> nibble_shift) & 0xF).to(tl.int32)
        code_v1 = ((packed_v >> (nibble_shift + 4)) & 0xF).to(tl.int32)

        k_stat_index = (
            physical_page * BLOCK_I32
            + K_STATS_I32
            + head * (BLOCK_SIZE * 4)
            + block_offset * 4
            + group
        )
        v_stat_index = (
            physical_page * BLOCK_I32
            + V_STATS_I32
            + head * (4 * BLOCK_SIZE)
            + group * BLOCK_SIZE
            + block_offset
        )
        scale_k = tl.load(
            cache_stat + 2 * k_stat_index,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        minimum_k = tl.load(
            cache_stat + 2 * k_stat_index + 1,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        scale_v = tl.load(
            cache_stat + 2 * v_stat_index,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        minimum_v = tl.load(
            cache_stat + 2 * v_stat_index + 1,
            mask=valid,
            other=0.0,
        ).to(tl.float32)

        qpoint_base_k = table * 16 if GROUPED_K else 0
        qpoint_base_v = table * 16 if GROUPED_V else 0
        normalized_k0 = tl.load(q_points_k + qpoint_base_k + code_k0).to(tl.float32)
        normalized_k1 = tl.load(q_points_k + qpoint_base_k + code_k1).to(tl.float32)
        normalized_v0 = tl.load(q_points_v + qpoint_base_v + code_v0).to(tl.float32)
        normalized_v1 = tl.load(q_points_v + qpoint_base_v + code_v1).to(tl.float32)
        value_k0 = normalized_k0 * scale_k + minimum_k
        value_k1 = normalized_k1 * scale_k + minimum_k
        value_v0 = normalized_v0 * scale_v + minimum_v
        value_v1 = normalized_v1 * scale_v + minimum_v

        out_token = tl.load(seq_start_loc + request).to(tl.int64) + position
        out_k_ptr = (
            out_k
            + out_token * out_k_stride_t
            + head * out_k_stride_h
            + pair * out_k_stride_d
        )
        out_v_ptr = (
            out_v
            + out_token * out_v_stride_t
            + head * out_v_stride_h
            + pair * out_v_stride_d
        )
        if OUTPUT_BF16:
            k0_bits = value_k0.to(tl.bfloat16).to(tl.uint16, bitcast=True)
            k1_bits = value_k1.to(tl.bfloat16).to(tl.uint16, bitcast=True)
            v0_bits = value_v0.to(tl.bfloat16).to(tl.uint16, bitcast=True)
            v1_bits = value_v1.to(tl.bfloat16).to(tl.uint16, bitcast=True)
        else:
            k0_bits = value_k0.to(tl.float16).to(tl.uint16, bitcast=True)
            k1_bits = value_k1.to(tl.float16).to(tl.uint16, bitcast=True)
            v0_bits = value_v0.to(tl.float16).to(tl.uint16, bitcast=True)
            v1_bits = value_v1.to(tl.float16).to(tl.uint16, bitcast=True)
        packed_k_out = k0_bits.to(tl.uint32) | (k1_bits.to(tl.uint32) << 16)
        packed_v_out = v0_bits.to(tl.uint32) | (v1_bits.to(tl.uint32) << 16)
        tl.store(out_k_ptr, packed_k_out, mask=valid)
        tl.store(out_v_ptr, packed_v_out, mask=valid)


def _packed_writer_groups_per_program(
    total_tokens: int,
    num_heads: int,
    write_dq: bool = False,
) -> int:
    """Keep independent G32 CTAs until their larger grid stops paying off."""
    if int(num_heads) == 4:
        independent_group_limit = 367 if bool(write_dq) else 306
    elif int(num_heads) == 8:
        independent_group_limit = 189 if bool(write_dq) else 163
    else:
        return 4
    if int(total_tokens) <= independent_group_limit:
        return 1
    return 4


def _packed_writer_num_warps() -> int:
    """Use the one warp that owns the G32 reduction and packed stores."""
    return 1


def _packed_readback_launch_policy(
    *,
    model_dtype: torch.dtype,
    batch_size: int,
    max_seq_bucket: int,
    num_heads: int,
) -> tuple[bool, int, int]:
    """Choose the byte-exact scalar or adjacent-pair readback mapping.

    The pairwise mapping amortizes page, packed-word, and affine-stat loads and
    writes two rounded 16-bit values with one 32-bit store.  Its smaller
    one/two-warp programs win once the launched token-head grid is large enough
    to cover B300, while the four-warp scalar mapping retains lower latency for
    shallow grids.
    """
    token_heads = int(batch_size) * int(max_seq_bucket) * int(num_heads)
    if token_heads < int(_PACKED_READBACK_PAIRWISE_MIN_TOKEN_HEADS):
        return False, 8, 4
    if model_dtype == torch.float16:
        return True, 8, 2
    if model_dtype == torch.bfloat16:
        return True, 4, 1
    raise ValueError("packed readback supports only FP16 or BF16")


def _packed_readback_use_flat_grid(
    *,
    model_dtype: torch.dtype,
    batch_size: int,
    max_seq_bucket: int,
    num_heads: int,
    tokens_per_program: int,
) -> bool:
    """Use 2D only where the B300 gate beats the polymorphic flat mapping.

    Both mappings are length-polymorphic.  The flattened launch is preferable
    for shallow grids; the 2D launch wins once token-head parallelism is large
    enough.  BF16 Hkv=4 needs a larger grid than Hkv=8 before that crossover.
    CUDA also limits grid Y to 65535, which makes the flat launch mandatory for
    the BF16 pairwise 256k bucket.
    """
    max_seq_bucket = int(max_seq_bucket)
    tokens_per_program = int(tokens_per_program)
    if max_seq_bucket // tokens_per_program > 65535:
        return True
    token_heads = int(batch_size) * max_seq_bucket * int(num_heads)
    if model_dtype == torch.float16 and int(num_heads) in (4, 8):
        return token_heads < 131072
    if model_dtype == torch.bfloat16 and int(num_heads) == 8:
        return token_heads < 262144
    if model_dtype == torch.bfloat16 and int(num_heads) == 4:
        return token_heads < 524288
    return True


def _try_write_tokens_to_paged_cache_triton(
    cache: torch.Tensor,
    layout: PagedLayout,
    slot_mapping: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_points_k: Optional[torch.Tensor],
    q_points_v: Optional[torch.Tensor],
    thresholds_k: Optional[torch.Tensor],
    thresholds_v: Optional[torch.Tensor],
    k_dq_out: Optional[torch.Tensor] = None,
    v_dq_out: Optional[torch.Tensor] = None,
    signal_pdl_dependents: bool = False,
) -> bool:
    """Launch the production writer, optionally emitting reconstructed K/V."""
    write_dq = k_dq_out is not None and v_dq_out is not None
    if (
        not _TRITON_OK
        or not cache.is_cuda
        or not k.is_cuda
        or not v.is_cuda
        or not slot_mapping.is_cuda
        or cache.dtype != torch.int32
        or cache.dim() != 2
        or not cache.is_contiguous()
        or not _is_supported_model_dtype(k.dtype)
        or v.dtype != k.dtype
        or k.dim() != 3
        or tuple(k.shape) != tuple(v.shape)
        or slot_mapping.dim() != 1
        or slot_mapping.dtype not in (torch.int32, torch.int64)
        or int(slot_mapping.numel()) != int(k.shape[0])
        or layout.bits != 4
        or layout.k_group_size != 32
        or layout.v_group_size != 32
        or layout.head_dim != 128
        or layout.k_groups_per_block != 4
        or layout.v_groups_per_token != 4
        or layout.num_chunks_k != 4
        or layout.num_chunks_v != 4
        or layout.code_layout not in SUPPORTED_CODE_LAYOUTS
        or not layout.stats_interleaved
        or int(cache.shape[1]) != int(layout.block_i32)
        or ((k_dq_out is None) != (v_dq_out is None))
    ):
        return False

    if write_dq and (
        k_dq_out.device != k.device
        or v_dq_out.device != v.device
        or k_dq_out.dtype != k.dtype
        or v_dq_out.dtype != v.dtype
        or tuple(k_dq_out.shape) != tuple(k.shape)
        or tuple(v_dq_out.shape) != tuple(v.shape)
    ):
        return False

    total_tokens, num_heads, _ = k.shape
    if int(num_heads) != int(layout.num_kv_heads):
        return False
    if int(total_tokens) == 0:
        return True

    q_points_k, thresholds_k = _prepare_qpoints_and_thresholds(
        num_bits=layout.bits,
        device=k.device,
        q_points=q_points_k,
        thresholds=thresholds_k,
    )
    q_points_v, thresholds_v = _prepare_qpoints_and_thresholds(
        num_bits=layout.bits,
        device=v.device,
        q_points=q_points_v,
        thresholds=thresholds_v,
    )
    expected_tables = int(num_heads) * 4
    grouped_k = q_points_k.dim() == 2
    grouped_v = q_points_v.dim() == 2
    if (
        not q_points_k.is_cuda
        or not q_points_v.is_cuda
        or not thresholds_k.is_cuda
        or not thresholds_v.is_cuda
        or not q_points_k.is_contiguous()
        or not q_points_v.is_contiguous()
        or not thresholds_k.is_contiguous()
        or not thresholds_v.is_contiguous()
        or tuple(q_points_k.shape) not in {(16,), (expected_tables, 16)}
        or tuple(q_points_v.shape) not in {(16,), (expected_tables, 16)}
        or tuple(thresholds_k.shape) not in {(15,), (expected_tables, 15)}
        or tuple(thresholds_v.shape) not in {(15,), (expected_tables, 15)}
    ):
        return False

    groups_per_program = _packed_writer_groups_per_program(
        int(total_tokens), int(num_heads), bool(write_dq)
    )
    cache_i32 = cache.reshape(-1)
    cache_stat = cache.view(k.dtype).reshape(-1)
    k_dq_target = k if k_dq_out is None else k_dq_out
    v_dq_target = v if v_dq_out is None else v_dq_out
    writer_heads = int(num_heads) * (4 if groups_per_program == 1 else 1)
    _write_packed_kv_cache_group_head128_kernel[(int(total_tokens), writer_heads)](
        k,
        v,
        cache_i32,
        cache_stat,
        slot_mapping,
        thresholds_k,
        thresholds_v,
        q_points_k,
        q_points_v,
        k_dq_target,
        v_dq_target,
        int(k.stride(0)),
        int(k.stride(1)),
        int(k.stride(2)),
        int(v.stride(0)),
        int(v.stride(1)),
        int(v.stride(2)),
        int(k_dq_target.stride(0)),
        int(k_dq_target.stride(1)),
        int(k_dq_target.stride(2)),
        int(v_dq_target.stride(0)),
        int(v_dq_target.stride(1)),
        int(v_dq_target.stride(2)),
        int(slot_mapping.stride(0)),
        BLOCK_SIZE=int(layout.block_size),
        BLOCK_I32=int(layout.block_i32),
        K_CODE_I32=int(layout.k_code_off // 2),
        K_STATS_I32=int(layout.k_scale_off // 2),
        V_CODE_I32=int(layout.v_code_off // 2),
        V_STATS_I32=int(layout.v_scale_off // 2),
        HND_TOKEN_WORD=layout.code_layout == CODE_LAYOUT_HND_TOKEN_WORD,
        GROUPED_K=bool(grouped_k),
        GROUPED_V=bool(grouped_v),
        WRITE_DQ=bool(write_dq),
        STAT_BF16=k.dtype == torch.bfloat16,
        SIGNAL_PDL=bool(signal_pdl_dependents),
        GROUPS_PER_PROGRAM=groups_per_program,
        BLOCK_G=32,
        num_warps=_packed_writer_num_warps(),
    )
    return True


def dequantize_paged_kv_to_dense(
    cache: torch.Tensor,
    layout: PagedLayout,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_start_loc: torch.Tensor,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
    *,
    max_seq_len: int,
    q_points_k: Optional[torch.Tensor] = None,
    q_points_v: Optional[torch.Tensor] = None,
) -> None:
    """Materialize an HND block-64/128 packed batch into dense varlen K/V.

    This is the bounded chunked-prefill bridge: cache storage remains packed
    and nonuniform, while official FA4 consumes a temporary dense varlen view.
    """
    if not _TRITON_OK or not cache.is_cuda:
        raise RuntimeError("packed chunked-prefill dequantization requires CUDA Triton")
    if (
        cache.dtype != torch.int32
        or cache.dim() != 2
        or not cache.is_contiguous()
        or block_table.device != cache.device
        or seq_lens.device != cache.device
        or seq_start_loc.device != cache.device
        or block_table.dtype != torch.int32
        or seq_lens.dtype != torch.int32
        or seq_start_loc.dtype != torch.int32
        or block_table.dim() != 2
        or seq_lens.dim() != 1
        or seq_start_loc.dim() != 1
        or int(seq_start_loc.numel()) != int(seq_lens.numel()) + 1
        or layout.bits != 4
        or layout.block_size not in (64, 128)
        or layout.head_dim != 128
        or layout.k_group_size != 32
        or layout.v_group_size != 32
        or layout.code_layout != CODE_LAYOUT_HND_TOKEN_WORD
        or not layout.stats_interleaved
        or int(cache.shape[1]) != int(layout.block_i32)
    ):
        raise ValueError("unsupported packed chunked-prefill cache ABI")
    if (
        out_k.device != cache.device
        or out_v.device != cache.device
        or out_k.dtype not in (torch.float16, torch.bfloat16)
        or out_v.dtype != out_k.dtype
        or out_k.dim() != 3
        or tuple(out_v.shape) != tuple(out_k.shape)
        or tuple(out_k.shape[1:]) != (layout.num_kv_heads, 128)
        or not out_k.is_contiguous()
        or not out_v.is_contiguous()
    ):
        raise ValueError("dense chunked-prefill outputs must be contiguous FP16/BF16 HND tensors")
    max_seq_len = int(max_seq_len)
    if max_seq_len <= 0:
        return
    if int(block_table.shape[0]) != int(seq_lens.numel()):
        raise ValueError("block table and sequence-length batch dimensions differ")
    if int(block_table.shape[1]) * int(layout.block_size) < max_seq_len:
        raise ValueError("block table cannot cover max_seq_len")

    q_points_k = _prepare_qpoints_for_decode(
        num_bits=layout.bits,
        device=cache.device,
        q_points=q_points_k,
    )
    q_points_v = _prepare_qpoints_for_decode(
        num_bits=layout.bits,
        device=cache.device,
        q_points=q_points_v,
    )
    expected_tables = int(layout.num_kv_heads) * 4
    if tuple(q_points_k.shape) not in {(16,), (expected_tables, 16)}:
        raise ValueError("K q_points do not match the packed HND tables")
    if tuple(q_points_v.shape) not in {(16,), (expected_tables, 16)}:
        raise ValueError("V q_points do not match the packed HND tables")

    max_seq_bucket = max(128, 1 << (max_seq_len - 1).bit_length())
    pairwise, tokens_per_program, num_warps = _packed_readback_launch_policy(
        model_dtype=out_k.dtype,
        batch_size=int(seq_lens.numel()),
        max_seq_bucket=max_seq_bucket,
        num_heads=int(layout.num_kv_heads),
    )
    flat_grid = _packed_readback_use_flat_grid(
        model_dtype=out_k.dtype,
        batch_size=int(seq_lens.numel()),
        max_seq_bucket=max_seq_bucket,
        num_heads=int(layout.num_kv_heads),
        tokens_per_program=tokens_per_program,
    )
    cache_i32 = cache.reshape(-1)
    cache_stat = cache.view(out_k.dtype).reshape(-1)
    readback_kernel = (
        _dequantize_paged_kv_hnd_head128_pairwise_kernel
        if pairwise
        else _dequantize_paged_kv_hnd_head128_kernel
    )
    readback_out_k = out_k.view(torch.int32).reshape(-1) if pairwise else out_k
    readback_out_v = out_v.view(torch.int32).reshape(-1) if pairwise else out_v
    readback_out_k_strides = (
        (int(out_k.stride(0)) // 2, int(out_k.stride(1)) // 2, 1)
        if pairwise
        else (int(out_k.stride(0)), int(out_k.stride(1)), int(out_k.stride(2)))
    )
    readback_out_v_strides = (
        (int(out_v.stride(0)) // 2, int(out_v.stride(1)) // 2, 1)
        if pairwise
        else (int(out_v.stride(0)), int(out_v.stride(1)), int(out_v.stride(2)))
    )
    grid = (
        (
            int(seq_lens.numel()) * max_seq_bucket // tokens_per_program,
            int(layout.num_kv_heads),
        )
        if flat_grid
        else (
            int(seq_lens.numel()) * int(layout.num_kv_heads),
            max_seq_bucket // tokens_per_program,
        )
    )
    readback_kernel[grid](
        cache_i32,
        cache_stat,
        block_table,
        seq_lens,
        seq_start_loc,
        q_points_k,
        q_points_v,
        readback_out_k,
        readback_out_v,
        max_seq_bucket.bit_length() - 1 if flat_grid else 0,
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        *readback_out_k_strides,
        *readback_out_v_strides,
        BLOCK_SIZE=int(layout.block_size),
        BLOCK_I32=int(layout.block_i32),
        K_CODE_I32=int(layout.k_code_off // 2),
        K_STATS_I32=int(layout.k_scale_off // 2),
        V_CODE_I32=int(layout.v_code_off // 2),
        V_STATS_I32=int(layout.v_scale_off // 2),
        GROUPED_K=bool(q_points_k.dim() == 2),
        GROUPED_V=bool(q_points_v.dim() == 2),
        OUTPUT_BF16=bool(out_k.dtype == torch.bfloat16),
        TOKENS_PER_PROGRAM=tokens_per_program,
        HEAD_DIM=128,
        NUM_HEADS=int(layout.num_kv_heads),
        FLAT_GRID=bool(flat_grid),
        num_warps=num_warps,
    )


def write_v_pv_to_cache(
    cache: torch.Tensor,
    layout: PagedLayout,
    slot_mapping: torch.Tensor,
    pv: PackedV,
) -> None:
    """Write a pre-computed ``PackedV`` into the cache (no extra quantize)."""
    if pv.packed.dim() == 5:
        if pv.pack_token_last:
            B = pv.packed.shape[0]
            S = pv.packed.shape[-1]
            packed = (
                pv.packed.permute(0, 4, 1, 2, 3).contiguous().reshape(B * S, *pv.packed.shape[1:-1])
            )
            scale = pv.scale.permute(0, 3, 1, 2).contiguous().reshape(B * S, *pv.scale.shape[1:-1])
            mn = pv.mn.permute(0, 3, 1, 2).contiguous().reshape(B * S, *pv.mn.shape[1:-1])
        else:
            B, S = pv.packed.shape[:2]
            packed = pv.packed.reshape(B * S, *pv.packed.shape[2:])
            scale = pv.scale.reshape(B * S, *pv.scale.shape[2:])
            mn = pv.mn.reshape(B * S, *pv.mn.shape[2:])
    else:
        if pv.pack_token_last:
            raise ValueError("flat PackedV must use legacy token-major layout")
        packed = pv.packed
        scale = pv.scale
        mn = pv.mn
    N = int(packed.shape[0])
    if N == 0:
        return
    if slot_mapping.numel() != N:
        raise ValueError(f"slot_mapping has {slot_mapping.numel()} tokens, PackedV has {N}")
    device = cache.device
    H_kv = layout.num_kv_heads
    BS = layout.block_size

    cache_i32_flat = cache.reshape(-1)

    slot_blk = (slot_mapping // BS).to(torch.long)
    slot_off = (slot_mapping % BS).to(torch.long)

    v_codes_per_tok = H_kv * layout.v_groups_per_token * layout.num_chunks_v
    pitch_s = H_kv * layout.v_groups_per_token

    v_codes_flat = packed.reshape(N, v_codes_per_tok).view(torch.int32)
    v_scale_flat = scale.reshape(N, pitch_s)
    v_min_flat = mn.reshape(N, pitch_s)
    v_stats_flat = (
        torch.stack((v_scale_flat, v_min_flat), dim=-1)
        .contiguous()
        .view(torch.int32)
        .reshape(N, pitch_s)
    )

    base_blk_i32 = slot_blk * layout.block_i32

    h_c, g_c, word_c, h_s, g_s = _v_write_components(layout, device)
    word_idx_c = g_c * layout.num_chunks_v + word_c
    if layout.code_layout == CODE_LAYOUT_HND_TOKEN_WORD:
        idx_c = (
            base_blk_i32[:, None]
            + layout.v_code_off // 2
            + h_c[None, :] * (BS * layout.v_groups_per_token * layout.num_chunks_v)
            + slot_off[:, None] * (layout.v_groups_per_token * layout.num_chunks_v)
            + word_idx_c[None, :]
        ).reshape(-1)
    else:
        idx_c = (
            base_blk_i32[:, None]
            + layout.v_code_off // 2
            + h_c[None, :] * (layout.v_groups_per_token * layout.num_chunks_v * BS)
            + word_idx_c[None, :] * BS
            + slot_off[:, None]
        ).reshape(-1)
    cache_i32_flat.index_copy_(0, idx_c, v_codes_flat.reshape(-1))

    idx_stats = (
        base_blk_i32[:, None]
        + layout.v_scale_off // 2
        + h_s[None, :] * (layout.v_groups_per_token * BS)
        + g_s[None, :] * BS
        + slot_off[:, None]
    ).reshape(-1)
    cache_i32_flat.index_copy_(0, idx_stats, v_stats_flat.reshape(-1))


def _write_flat_k_pk_to_cache(
    cache: torch.Tensor,
    layout: PagedLayout,
    slot_mapping: torch.Tensor,
    pk: PackedK,
) -> None:
    """Write a flat batch of per-token PackedK rows using vLLM slot IDs."""
    if pk.packed.shape[0] != 1:
        raise ValueError(f"PackedK batch must be 1, got {pk.packed.shape[0]}")
    if pk.bits != layout.bits or pk.group_size != layout.k_group_size:
        raise ValueError("PackedK layout does not match paged layout")
    if pk.pack_token_last:
        packed = pk.packed.permute(0, 4, 1, 2, 3).contiguous()
        scale = pk.scale.permute(0, 3, 1, 2).contiguous()
        mn = pk.mn.permute(0, 3, 1, 2).contiguous()
    else:
        packed = pk.packed
        scale = pk.scale
        mn = pk.mn

    N = int(packed.shape[1])
    if N == 0:
        return
    if slot_mapping.numel() != N:
        raise ValueError(f"slot_mapping has {slot_mapping.numel()} tokens, PackedK has {N}")

    H_kv = layout.num_kv_heads
    G_h = layout.head_dim // layout.k_group_size
    if packed.shape[2:] != (H_kv, G_h, layout.num_chunks_k):
        raise ValueError(
            f"PackedK shape {tuple(pk.packed.shape)} incompatible with "
            f"layout token shape ({H_kv}, {G_h}, {layout.num_chunks_k})"
        )

    device = cache.device
    BS = layout.block_size
    cache_i32_flat = cache.reshape(-1)

    slot_blk = (slot_mapping // BS).to(torch.long)
    slot_off = (slot_mapping % BS).to(torch.long)
    base_blk_i32 = slot_blk * layout.block_i32

    k_codes_per_tok = H_kv * G_h * layout.num_chunks_k
    pitch_s = H_kv * G_h
    k_codes_flat = packed.reshape(N, k_codes_per_tok).view(torch.int32)
    k_scale_flat = scale.reshape(N, pitch_s)
    k_min_flat = mn.reshape(N, pitch_s)
    k_stats_flat = (
        torch.stack((k_scale_flat, k_min_flat), dim=-1)
        .contiguous()
        .view(torch.int32)
        .reshape(N, pitch_s)
    )

    arange_c = _cached_arange(k_codes_per_tok, device)
    h_c = arange_c // (G_h * layout.num_chunks_k)
    rem_c = arange_c - h_c * (G_h * layout.num_chunks_k)
    g_c = rem_c // layout.num_chunks_k
    word_c = rem_c - g_c * layout.num_chunks_k
    arange_s = _cached_arange(pitch_s, device)
    h_s = arange_s // G_h
    g_s = arange_s - h_s * G_h

    word_idx_c = g_c * layout.num_chunks_k + word_c
    if layout.code_layout == CODE_LAYOUT_HND_TOKEN_WORD:
        idx_c = (
            base_blk_i32[:, None]
            + layout.k_code_off // 2
            + h_c[None, :] * (BS * G_h * layout.num_chunks_k)
            + slot_off[:, None] * (G_h * layout.num_chunks_k)
            + word_idx_c[None, :]
        ).reshape(-1)
        idx_stats = (
            base_blk_i32[:, None]
            + layout.k_scale_off // 2
            + h_s[None, :] * (BS * G_h)
            + slot_off[:, None] * G_h
            + g_s[None, :]
        ).reshape(-1)
    else:
        idx_c = (
            base_blk_i32[:, None]
            + layout.k_code_off // 2
            + h_c[None, :] * (G_h * layout.num_chunks_k * BS)
            + word_idx_c[None, :] * BS
            + slot_off[:, None]
        ).reshape(-1)
        idx_stats = (
            base_blk_i32[:, None]
            + layout.k_scale_off // 2
            + h_s[None, :] * (G_h * BS)
            + g_s[None, :] * BS
            + slot_off[:, None]
        ).reshape(-1)
    cache_i32_flat.index_copy_(0, idx_c, k_codes_flat.reshape(-1))
    cache_i32_flat.index_copy_(0, idx_stats, k_stats_flat.reshape(-1))


def _write_tokens_to_paged_cache_reference(
    cache: torch.Tensor,  # (num_blocks, block_i32) int32
    layout: PagedLayout,
    block_table: torch.Tensor,  # (num_reqs, max_blocks) int32
    slot_mapping: torch.Tensor,  # (total_tokens,) int32, flat slot index
    k: torch.Tensor,  # (total_tokens, H_kv, D) fp16/bf16
    v: torch.Tensor,  # (total_tokens, H_kv, D) fp16/bf16
    *,
    q_points_k: Optional[torch.Tensor] = None,
    q_points_v: Optional[torch.Tensor] = None,
    thresholds_k: Optional[torch.Tensor] = None,
    thresholds_v: Optional[torch.Tensor] = None,
    k_dq_out: Optional[torch.Tensor] = None,
    v_dq_out: Optional[torch.Tensor] = None,
) -> None:
    """Numerical-oracle implementation of packed K/V cache writeback."""
    del block_table
    total_tokens, H_kv, D = k.shape
    assert v.shape == (total_tokens, H_kv, D), "K/V shape mismatch"
    assert k.dtype == v.dtype
    assert H_kv == layout.num_kv_heads and D == layout.head_dim
    if total_tokens == 0:
        return

    valid = slot_mapping >= 0
    has_valid = bool(valid.any().item())
    if not has_valid and k_dq_out is None and v_dq_out is None:
        return

    bits = layout.bits
    G_k = layout.k_group_size
    G_v = layout.v_group_size

    q_points_k, thresholds_k = _prepare_qpoints_and_thresholds(
        num_bits=bits,
        device=k.device,
        q_points=q_points_k,
        thresholds=thresholds_k,
    )
    q_points_v, thresholds_v = _prepare_qpoints_and_thresholds(
        num_bits=bits,
        device=v.device,
        q_points=q_points_v,
        thresholds=thresholds_v,
    )

    stat_dtype = _stat_dtype_from_tensor(k)
    pk = quantize_k(
        k.unsqueeze(0),
        bits,
        G_k,
        q_points=q_points_k,
        thresholds=thresholds_k,
        stat_dtype=stat_dtype,
    )
    pv = quantize_v(
        v.unsqueeze(0),
        bits,
        G_v,
        q_points=q_points_v,
        thresholds=thresholds_v,
        stat_dtype=stat_dtype,
    )
    if has_valid:
        if bool(valid.all().item()):
            write_slots = slot_mapping
            write_pk = pk
            write_pv = pv
        else:
            write_slots = slot_mapping[valid].contiguous()
            write_pk = PackedK(
                codes=pk.codes[:, valid].contiguous(),
                packed=pk.packed.view(torch.int32)[:, valid].contiguous().view(torch.uint32),
                mn=pk.mn[:, valid].contiguous(),
                scale=pk.scale[:, valid].contiguous(),
                bits=pk.bits,
                group_size=pk.group_size,
                pack_token_last=False,
            )
            write_pv = PackedV(
                codes=pv.codes[:, valid].contiguous(),
                packed=pv.packed.view(torch.int32)[:, valid].contiguous().view(torch.uint32),
                mn=pv.mn[:, valid].contiguous(),
                scale=pv.scale[:, valid].contiguous(),
                bits=pv.bits,
                group_size=pv.group_size,
                pack_token_last=False,
            )
        _write_flat_k_pk_to_cache(cache, layout, write_slots, write_pk)
        write_v_pv_to_cache(cache, layout, write_slots, write_pv)
    if k_dq_out is not None or v_dq_out is not None:
        k_dq, v_dq = (
            dequantize_k(pk, q_points=q_points_k).squeeze(0),
            dequantize_v(
                pv,
                q_points=q_points_v,
            ).squeeze(0),
        )
        if k_dq_out is not None:
            k_dq_out.copy_(k_dq)
        if v_dq_out is not None:
            v_dq_out.copy_(v_dq)


def write_tokens_to_paged_cache(
    cache: torch.Tensor,  # (num_blocks, block_i32) int32
    layout: PagedLayout,
    block_table: torch.Tensor,  # (num_reqs, max_blocks) int32
    slot_mapping: torch.Tensor,  # (total_tokens,) int32, flat slot index
    k: torch.Tensor,  # (total_tokens, H_kv, D) fp16/bf16
    v: torch.Tensor,  # (total_tokens, H_kv, D) fp16/bf16
    *,
    q_points_k: Optional[torch.Tensor] = None,
    q_points_v: Optional[torch.Tensor] = None,
    thresholds_k: Optional[torch.Tensor] = None,
    thresholds_v: Optional[torch.Tensor] = None,
    k_dq_out: Optional[torch.Tensor] = None,
    v_dq_out: Optional[torch.Tensor] = None,
    signal_pdl_dependents: bool = False,
    require_fast: bool = False,
) -> None:
    """Quantize, pack, and scatter K/V tokens into the physical page cache.

    The fixed FP16/BF16 head-dim-128 production ABI uses one Triton kernel,
    including the optional reconstructed-output path used by rounded-K/V
    prefill. Reference and unit-test callers may retain the numerical oracle;
    production callers set ``require_fast=True`` and fail closed instead of
    silently changing the measured execution path.
    """
    if (
        _try_write_tokens_to_paged_cache_triton(
            cache,
            layout,
            slot_mapping,
            k,
            v,
            q_points_k=q_points_k,
            q_points_v=q_points_v,
            thresholds_k=thresholds_k,
            thresholds_v=thresholds_v,
            k_dq_out=k_dq_out,
            v_dq_out=v_dq_out,
            signal_pdl_dependents=signal_pdl_dependents,
        )
    ):
        return
    if require_fast:
        raise RuntimeError(
            "Beyond production cache writes require the fused Triton writer; "
            "the current device, dtype, layout, or shape is unsupported."
        )
    _write_tokens_to_paged_cache_reference(
        cache,
        layout,
        block_table,
        slot_mapping,
        k,
        v,
        q_points_k=q_points_k,
        q_points_v=q_points_v,
        thresholds_k=thresholds_k,
        thresholds_v=thresholds_v,
        k_dq_out=k_dq_out,
        v_dq_out=v_dq_out,
    )


def read_request_from_paged_cache(
    cache: torch.Tensor,  # (num_blocks, block_i32) int32
    layout: PagedLayout,
    block_ids: torch.Tensor,  # (num_full_blocks,) int32 — all full blocks for this req
    seq_len: int,  # tokens in this request (<= len(block_ids)*BS)
    *,
    q_points_k: Optional[torch.Tensor] = None,
    q_points_v: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.float16,
    stat_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize a single request's full K/V context from paged cache."""
    device = cache.device
    H_kv = layout.num_kv_heads
    D = layout.head_dim
    G_k = layout.k_group_size
    G_v = layout.v_group_size
    BS = layout.block_size
    bits = layout.bits
    num_chunks_k = layout.num_chunks_k
    num_chunks_v = layout.num_chunks_v
    G_h_k = D // G_k
    G_h_v = layout.v_groups_per_token

    stat_dtype = _stat_dtype_from_model_dtype(stat_dtype or out_dtype)
    gathered_code = _block_halves(
        _view_block_as_half(cache),
        layout,
        block_ids,
    )  # (NB, block_halves)
    gathered_i32 = cache.index_select(0, block_ids.to(torch.long))
    NB = gathered_code.shape[0]
    assert NB * BS >= seq_len, "not enough blocks for seq_len"
    assert layout.stats_interleaved

    # --- K codes & stats --------------------------------------------
    k_stats_flat = (
        gathered_i32[
            :,
            layout.k_scale_off // 2 : layout.k_scale_off // 2 + layout.k_scale_halves,
        ]
        .contiguous()
        .view(stat_dtype)
    )
    k_code_region = gathered_code[:, layout.k_code_off : layout.k_code_off + layout.k_code_halves]
    if layout.code_layout == CODE_LAYOUT_HND_TOKEN_WORD:
        k_code_u32 = k_code_region.view(torch.int32).view(
            NB,
            H_kv,
            BS,
            G_h_k,
            num_chunks_k,
        )
        k_stats = k_stats_flat.view(NB, H_kv, BS, G_h_k, 2)
        k_packed = (
            k_code_u32.permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(NB * BS, H_kv, G_h_k, num_chunks_k)[:seq_len]
            .unsqueeze(0)
        )
        k_scale = (
            k_stats[..., 0]
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(NB * BS, H_kv, G_h_k)[:seq_len]
            .unsqueeze(0)
        )
        k_min = (
            k_stats[..., 1]
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(NB * BS, H_kv, G_h_k)[:seq_len]
            .unsqueeze(0)
        )
    else:
        k_code_u32 = k_code_region.view(torch.int32).view(
            NB,
            H_kv,
            G_h_k,
            num_chunks_k,
            BS,
        )
        k_stats = k_stats_flat.view(NB, H_kv, G_h_k, BS, 2)
        k_packed = (
            k_code_u32.permute(0, 4, 1, 2, 3)
            .contiguous()
            .reshape(NB * BS, H_kv, G_h_k, num_chunks_k)[:seq_len]
            .unsqueeze(0)
        )
        k_scale = (
            k_stats[..., 0]
            .permute(0, 3, 1, 2)
            .contiguous()
            .reshape(NB * BS, H_kv, G_h_k)[:seq_len]
            .unsqueeze(0)
        )
        k_min = (
            k_stats[..., 1]
            .permute(0, 3, 1, 2)
            .contiguous()
            .reshape(NB * BS, H_kv, G_h_k)[:seq_len]
            .unsqueeze(0)
        )

    k_codes = unpack_codes(k_packed, bits, num_codes=G_k)
    k_qp = _prepare_qpoints_for_decode(
        num_bits=bits,
        device=device,
        q_points=q_points_k,
    )
    k_qp = k_qp.to(device, dtype=torch.float32)
    k_vals = _lookup_qpoints_for_codes(k_codes, k_qp)
    k_full = (
        (k_vals * k_scale.to(torch.float32).unsqueeze(-1) + k_min.to(torch.float32).unsqueeze(-1))
        .reshape(1, seq_len, H_kv, G_h_k * G_k)
        .to(out_dtype)
    )

    # --- V codes & stats ------------------------------------------
    v_code_region = gathered_code[:, layout.v_code_off : layout.v_code_off + layout.v_code_halves]
    if layout.code_layout == CODE_LAYOUT_HND_TOKEN_WORD:
        v_code_u32 = v_code_region.view(torch.int32).view(
            NB,
            H_kv,
            BS,
            G_h_v,
            num_chunks_v,
        )
        v_code_all = (
            v_code_u32.permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(NB * BS, H_kv, G_h_v, num_chunks_v)[:seq_len]
        )
    else:
        v_code_u32 = v_code_region.view(torch.int32).view(
            NB,
            H_kv,
            G_h_v,
            num_chunks_v,
            BS,
        )
        v_code_all = (
            v_code_u32.permute(0, 4, 1, 2, 3)
            .contiguous()
            .reshape(NB * BS, H_kv, G_h_v, num_chunks_v)[:seq_len]
        )
    v_packed = v_code_all.unsqueeze(0)

    v_stats_flat = (
        gathered_i32[
            :,
            layout.v_scale_off // 2 : layout.v_scale_off // 2 + layout.v_scale_halves,
        ]
        .contiguous()
        .view(stat_dtype)
    )
    v_stats = v_stats_flat.view(NB, H_kv, G_h_v, BS, 2)
    v_scale = (
        v_stats[..., 0]
        .permute(0, 3, 1, 2)
        .contiguous()
        .reshape(NB * BS, H_kv, G_h_v)[:seq_len]
        .unsqueeze(0)
    )
    v_min = (
        v_stats[..., 1]
        .permute(0, 3, 1, 2)
        .contiguous()
        .reshape(NB * BS, H_kv, G_h_v)[:seq_len]
        .unsqueeze(0)
    )

    v_codes = unpack_codes(v_packed, bits, num_codes=G_v)  # (1, S, H_kv, G_h, G_v)
    v_qp = _prepare_qpoints_for_decode(
        num_bits=bits,
        device=device,
        q_points=q_points_v,
    )
    v_qp = v_qp.to(device, dtype=torch.float32)
    v_vals = _lookup_qpoints_for_codes(v_codes, v_qp)
    v_full = v_vals * v_scale.to(torch.float32).unsqueeze(-1) + v_min.to(torch.float32).unsqueeze(
        -1
    )
    v_full = v_full.reshape(1, seq_len, H_kv, G_h_v * G_v).to(out_dtype)

    return k_full, v_full
