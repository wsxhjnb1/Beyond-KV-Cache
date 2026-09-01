"""Triton fast path for hard-forward STE quantization."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_BLOCK_SIZE = 1024
_BACKWARD_BLOCK_SIZE = 512
_NORM_FORWARD_TILE_ELEMS = 2048
_NORM_FORWARD_MAX_GROUPS_PER_BLOCK = 64
_MAX_Q_POINTS = 64


@triton.jit
def _quant_group_forward_kernel(
    x,
    q_points,
    thresholds,
    out,
    bins,
    n_elements: tl.constexpr,
    group_size: tl.constexpr,
    NUM_TABLES: tl.constexpr,
    NUM_Q: tl.constexpr,
    NUM_T: tl.constexpr,
    NUM_SEARCH_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    table = (offsets // group_size) % NUM_TABLES

    lo = tl.full((BLOCK_SIZE,), 0, dtype=tl.int32)
    hi = tl.full((BLOCK_SIZE,), NUM_T, dtype=tl.int32)
    for _ in tl.static_range(0, NUM_SEARCH_STEPS):
        mid = (lo + hi) // 2
        th = tl.load(
            thresholds + table * NUM_T + mid,
            mask=(mid < NUM_T) & mask,
            other=float("inf"),
        ).to(tl.float32)
        go_right = vals >= th
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)

    codes = tl.minimum(lo, NUM_Q - 1)
    qv = tl.load(q_points + table * NUM_Q + codes, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + offsets, qv, mask=mask)
    tl.store(bins + offsets, codes, mask=mask)


@triton.jit
def _quant_group_norm_forward_kernel(
    x,
    q_points,
    thresholds,
    out,
    bins,
    mins,
    scales,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    NUM_TABLES: tl.constexpr,
    NUM_Q: tl.constexpr,
    NUM_T: tl.constexpr,
    NUM_SEARCH_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    group_base = tl.program_id(0) * GROUPS_PER_BLOCK
    group_offsets = group_base + tl.arange(0, GROUPS_PER_BLOCK)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = (group_offsets[:, None] < num_groups) & (offs[None, :] < group_size)
    elem_offsets = group_offsets[:, None] * group_size + offs[None, :]
    vals = tl.load(x + elem_offsets, mask=mask, other=0.0).to(tl.float32)

    vals_for_min = tl.where(mask, vals, float("inf"))
    vals_for_max = tl.where(mask, vals, -float("inf"))
    mn = tl.min(vals_for_min, axis=1)
    mx = tl.max(vals_for_max, axis=1)
    scale = tl.maximum(mx - mn, 1.0e-6)
    norm = tl.minimum(tl.maximum((vals - mn[:, None]) / scale[:, None], 0.0), 1.0)
    table = group_offsets % NUM_TABLES

    lo = tl.full((GROUPS_PER_BLOCK, BLOCK_SIZE), 0, dtype=tl.int32)
    hi = tl.full((GROUPS_PER_BLOCK, BLOCK_SIZE), NUM_T, dtype=tl.int32)
    for _ in tl.static_range(0, NUM_SEARCH_STEPS):
        mid = (lo + hi) // 2
        th = tl.load(
            thresholds + table[:, None] * NUM_T + mid,
            mask=(mid < NUM_T) & mask,
            other=float("inf"),
        ).to(tl.float32)
        go_right = norm >= th
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)

    codes = tl.minimum(lo, NUM_Q - 1)
    qv = tl.load(q_points + table[:, None] * NUM_Q + codes, mask=mask, other=0.0).to(tl.float32)
    tl.store(bins + elem_offsets, codes, mask=mask)
    tl.store(out + elem_offsets, qv * scale[:, None] + mn[:, None], mask=mask)
    tl.store(mins + group_offsets, mn, mask=group_offsets < num_groups)
    tl.store(scales + group_offsets, scale, mask=group_offsets < num_groups)


@triton.jit
def _quant_group_backward_ste_stats_kernel(
    x,
    grad_out,
    bins,
    thresholds,
    grad_q,
    sum_left,
    sum_right,
    num_outer: tl.constexpr,
    eps: tl.constexpr,
    group_size: tl.constexpr,
    NUM_TABLES: tl.constexpr,
    NUM_Q: tl.constexpr,
    NUM_T: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    table_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offs = tl.arange(0, BLOCK_SIZE)
    row_delta = offs // group_size
    chan = offs - row_delta * group_size
    row = block_idx * ROWS_PER_BLOCK + row_delta
    offsets = (row * NUM_TABLES + table_idx) * group_size + chan
    mask = (row_delta < ROWS_PER_BLOCK) & (row < num_outer)
    vals = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    gout = tl.load(grad_out + offsets, mask=mask, other=0.0).to(tl.float32)
    codes = tl.load(bins + offsets, mask=mask, other=0).to(tl.int32)

    for q_idx in tl.static_range(0, NUM_Q):
        q_mask = mask & (codes == q_idx)
        q_sum = tl.sum(tl.where(q_mask, gout, 0.0), axis=0)
        tl.atomic_add(
            grad_q + table_idx * NUM_Q + q_idx,
            q_sum,
            sem="relaxed",
        )

    left_idx = codes
    left_valid = mask & (left_idx < NUM_T)
    left_th = tl.load(
        thresholds + table_idx * NUM_T + left_idx,
        mask=left_valid,
        other=0.0,
    ).to(tl.float32)
    left_diff = vals - left_th
    left_mask = left_valid & (left_diff >= -eps) & (left_diff < 0.0)
    tl.atomic_add(
        sum_left + table_idx * NUM_T + left_idx,
        tl.where(left_mask, gout, 0.0),
        mask=left_mask,
        sem="relaxed",
    )

    right_idx = codes - 1
    right_valid = mask & (codes > 0)
    right_th = tl.load(
        thresholds + table_idx * NUM_T + right_idx,
        mask=right_valid,
        other=0.0,
    ).to(tl.float32)
    right_diff = vals - right_th
    right_mask = right_valid & (right_diff >= 0.0) & (right_diff <= eps)
    tl.atomic_add(
        sum_right + table_idx * NUM_T + right_idx,
        tl.where(right_mask, gout, 0.0),
        mask=right_mask,
        sem="relaxed",
    )


@triton.jit
def _quant_group_norm_backward_ste_sparse_t_stats_kernel(
    x,
    grad_out,
    mins,
    scales,
    bins,
    thresholds,
    grad_q,
    sum_left,
    sum_right,
    num_outer: tl.constexpr,
    eps: tl.constexpr,
    group_size: tl.constexpr,
    NUM_TABLES: tl.constexpr,
    NUM_Q: tl.constexpr,
    NUM_T: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    table_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offs = tl.arange(0, BLOCK_SIZE)
    row_delta = offs // group_size
    chan = offs - row_delta * group_size
    row = block_idx * ROWS_PER_BLOCK + row_delta
    offsets = (row * NUM_TABLES + table_idx) * group_size + chan
    mask = (row_delta < ROWS_PER_BLOCK) & (row < num_outer)
    raw_vals = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    mn = tl.load(mins + row * NUM_TABLES + table_idx, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scales + row * NUM_TABLES + table_idx, mask=mask, other=0.0).to(tl.float32)
    vals = tl.minimum(tl.maximum((raw_vals - mn) / scale, 0.0), 1.0)
    gout = tl.load(grad_out + offsets, mask=mask, other=0.0).to(tl.float32) * scale
    codes = tl.load(bins + offsets, mask=mask, other=0).to(tl.int32)

    for q_idx in tl.static_range(0, NUM_Q):
        q_mask = mask & (codes == q_idx)
        q_sum = tl.sum(tl.where(q_mask, gout, 0.0), axis=0)
        tl.atomic_add(
            grad_q + table_idx * NUM_Q + q_idx,
            q_sum,
            sem="relaxed",
        )

    left_idx = codes
    left_valid = mask & (left_idx < NUM_T)
    left_th = tl.load(
        thresholds + table_idx * NUM_T + left_idx,
        mask=left_valid,
        other=0.0,
    ).to(tl.float32)
    left_diff = vals - left_th
    left_mask = left_valid & (left_diff >= -eps) & (left_diff < 0.0)
    tl.atomic_add(
        sum_left + table_idx * NUM_T + left_idx,
        tl.where(left_mask, gout, 0.0),
        mask=left_mask,
        sem="relaxed",
    )

    right_idx = codes - 1
    right_valid = mask & (codes > 0)
    right_th = tl.load(
        thresholds + table_idx * NUM_T + right_idx,
        mask=right_valid,
        other=0.0,
    ).to(tl.float32)
    right_diff = vals - right_th
    right_mask = right_valid & (right_diff >= 0.0) & (right_diff <= eps)
    tl.atomic_add(
        sum_right + table_idx * NUM_T + right_idx,
        tl.where(right_mask, gout, 0.0),
        mask=right_mask,
        sem="relaxed",
    )


@triton.jit
def _zero_group_backward_buffers_kernel(
    grad_q,
    sum_left,
    sum_right,
    total_q: tl.constexpr,
    total_t: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    zero = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(grad_q + offs, zero, mask=offs < total_q)
    tl.store(sum_left + offs, zero, mask=offs < total_t)
    tl.store(sum_right + offs, zero, mask=offs < total_t)


def _can_use_group_triton(
    x_norm: torch.Tensor,
    q_points: torch.Tensor,
    thresholds: torch.Tensor,
    *,
    input_dtypes=(torch.float32,),
) -> bool:
    return (
        _TRITON_AVAILABLE
        and torch.is_tensor(x_norm)
        and torch.is_tensor(q_points)
        and torch.is_tensor(thresholds)
        and x_norm.is_cuda
        and q_points.is_cuda
        and thresholds.is_cuda
        and x_norm.dtype in input_dtypes
        and q_points.dtype == torch.float32
        and thresholds.dtype == torch.float32
        and x_norm.device == q_points.device == thresholds.device
        and x_norm.dim() >= 2
        and q_points.dim() == 2
        and thresholds.dim() == 2
        and int(x_norm.shape[-2]) == int(q_points.shape[0])
        and int(q_points.shape[0]) == int(thresholds.shape[0])
        and int(q_points.shape[1]) == int(thresholds.shape[1]) + 1
        and 1 <= int(q_points.shape[1]) <= _MAX_Q_POINTS
        and 1 <= int(q_points.shape[0]) <= 512
        and 1 <= int(x_norm.shape[-1]) <= _BLOCK_SIZE
    )


def triton_group_quant_forward(
    x_norm: torch.Tensor,
    q_points: torch.Tensor,
    thresholds: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not _can_use_group_triton(x_norm, q_points, thresholds):
        return None

    x_contig = x_norm.contiguous()
    q_contig = q_points.contiguous()
    t_contig = thresholds.contiguous()
    out = torch.empty_like(x_contig)
    bins = torch.empty(x_contig.shape, device=x_contig.device, dtype=torch.uint8)
    n_elements = x_contig.numel()
    if n_elements == 0:
        return out.view_as(x_norm), bins.view_as(x_norm)

    num_q = int(q_contig.shape[1])
    num_t = int(t_contig.shape[1])
    search_steps = max(1, math.ceil(math.log2(max(1, num_t + 1))))
    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
    _quant_group_forward_kernel[grid](
        x_contig,
        q_contig,
        t_contig,
        out,
        bins,
        n_elements,
        int(x_contig.shape[-1]),
        NUM_TABLES=int(q_contig.shape[0]),
        NUM_Q=num_q,
        NUM_T=num_t,
        NUM_SEARCH_STEPS=search_steps,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return out.view_as(x_norm), bins.view_as(x_norm)


def triton_group_quant_norm_forward(
    x: torch.Tensor,
    q_points: torch.Tensor,
    thresholds: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not _can_use_group_triton(
        x,
        q_points,
        thresholds,
        input_dtypes=(torch.float32, torch.float16, torch.bfloat16),
    ):
        return None

    x_contig = x.contiguous()
    q_contig = q_points.contiguous()
    t_contig = thresholds.contiguous()
    out = torch.empty_like(x_contig)
    bins = torch.empty(x_contig.shape, device=x_contig.device, dtype=torch.uint8)
    scale_shape = list(x_contig.shape)
    scale_shape[-1] = 1
    mins = torch.empty(scale_shape, device=x_contig.device, dtype=torch.float32)
    scales = torch.empty(scale_shape, device=x_contig.device, dtype=torch.float32)
    n_elements = x_contig.numel()
    if n_elements == 0:
        return out.view_as(x), bins.view_as(x), mins, scales

    group_size = int(x_contig.shape[-1])
    num_groups = n_elements // group_size
    num_q = int(q_contig.shape[1])
    num_t = int(t_contig.shape[1])
    block_size = triton.next_power_of_2(group_size)
    groups_per_block = max(
        1,
        min(_NORM_FORWARD_MAX_GROUPS_PER_BLOCK, _NORM_FORWARD_TILE_ELEMS // block_size),
    )
    search_steps = max(1, math.ceil(math.log2(max(1, num_t + 1))))
    num_warps = max(1, min(8, (block_size * groups_per_block) // 32))
    grid = (triton.cdiv(num_groups, groups_per_block),)
    _quant_group_norm_forward_kernel[grid](
        x_contig,
        q_contig,
        t_contig,
        out,
        bins,
        mins,
        scales,
        num_groups,
        group_size,
        NUM_TABLES=int(q_contig.shape[0]),
        NUM_Q=num_q,
        NUM_T=num_t,
        NUM_SEARCH_STEPS=search_steps,
        BLOCK_SIZE=block_size,
        GROUPS_PER_BLOCK=groups_per_block,
        num_warps=num_warps,
    )
    return out.view_as(x), bins.view_as(x), mins, scales


def triton_group_quant_backward_ste(
    grad_output: torch.Tensor,
    x_norm: torch.Tensor,
    q_points: torch.Tensor,
    thresholds: torch.Tensor,
    bin_indices: torch.Tensor,
    eps: float,
    *,
    return_grad_x: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not _can_use_group_triton(x_norm, q_points, thresholds):
        return None
    if not (
        torch.is_tensor(grad_output) and grad_output.is_cuda and grad_output.device == x_norm.device
    ):
        return None
    if not (
        torch.is_tensor(bin_indices) and bin_indices.is_cuda and bin_indices.device == x_norm.device
    ):
        return None

    x_contig = x_norm.contiguous()
    grad_contig = grad_output.to(torch.float32).contiguous()
    bins_contig = bin_indices.to(torch.uint8).contiguous()
    q_contig = q_points.contiguous()
    t_contig = thresholds.contiguous()
    n_elements = x_contig.numel()

    grad_q = torch.empty_like(q_contig)
    side_sums = torch.empty(
        (2,) + tuple(t_contig.shape),
        device=t_contig.device,
        dtype=t_contig.dtype,
    )
    sum_left, sum_right = side_sums.unbind(0)
    zero_block = 1024
    zero_grid = (triton.cdiv(max(q_contig.numel(), t_contig.numel()), zero_block),)
    _zero_group_backward_buffers_kernel[zero_grid](
        grad_q,
        sum_left,
        sum_right,
        q_contig.numel(),
        t_contig.numel(),
        BLOCK_SIZE=zero_block,
    )

    if n_elements > 0:
        group_size = int(x_contig.shape[-1])
        num_tables = int(q_contig.shape[0])
        block_size = max(_BACKWARD_BLOCK_SIZE, group_size)
        rows_per_block = max(1, block_size // group_size)
        num_outer = n_elements // (num_tables * group_size)
        grid = (num_tables, triton.cdiv(num_outer, rows_per_block))
        _quant_group_backward_ste_stats_kernel[grid](
            x_contig,
            grad_contig,
            bins_contig,
            t_contig,
            grad_q,
            sum_left,
            sum_right,
            num_outer,
            float(eps),
            group_size,
            NUM_TABLES=num_tables,
            NUM_Q=int(q_contig.shape[1]),
            NUM_T=int(t_contig.shape[1]),
            ROWS_PER_BLOCK=rows_per_block,
            BLOCK_SIZE=block_size,
        )

    grad_x = grad_output.to(x_norm.dtype) if return_grad_x else None
    return grad_x, grad_q, sum_left, sum_right


def triton_group_quant_norm_backward_ste(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    q_points: torch.Tensor,
    thresholds: torch.Tensor,
    bin_indices: torch.Tensor,
    mins: torch.Tensor,
    scales: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not _can_use_group_triton(
        x,
        q_points,
        thresholds,
        input_dtypes=(torch.float32, torch.float16, torch.bfloat16),
    ):
        return None
    if not (
        torch.is_tensor(grad_output) and grad_output.is_cuda and grad_output.device == x.device
    ):
        return None
    if not (
        torch.is_tensor(bin_indices) and bin_indices.is_cuda and bin_indices.device == x.device
    ):
        return None
    if not (
        torch.is_tensor(mins)
        and mins.is_cuda
        and mins.device == x.device
        and mins.dtype == torch.float32
        and mins.shape[:-1] == x.shape[:-1]
        and int(mins.shape[-1]) == 1
        and torch.is_tensor(scales)
        and scales.is_cuda
        and scales.device == x.device
        and scales.dtype == torch.float32
        and scales.shape[:-1] == x.shape[:-1]
        and int(scales.shape[-1]) == 1
    ):
        return None

    x_contig = x.contiguous()
    grad_contig = grad_output.contiguous()
    bins_contig = bin_indices.to(torch.uint8).contiguous()
    mins_contig = mins.contiguous()
    scale_contig = scales.contiguous()
    q_contig = q_points.contiguous()
    t_contig = thresholds.contiguous()
    n_elements = x_contig.numel()

    grad_q = torch.empty_like(q_contig)
    side_sums = torch.empty(
        (2,) + tuple(t_contig.shape),
        device=t_contig.device,
        dtype=t_contig.dtype,
    )
    sum_left, sum_right = side_sums.unbind(0)
    zero_block = 1024
    zero_grid = (triton.cdiv(max(q_contig.numel(), t_contig.numel()), zero_block),)
    _zero_group_backward_buffers_kernel[zero_grid](
        grad_q,
        sum_left,
        sum_right,
        q_contig.numel(),
        t_contig.numel(),
        BLOCK_SIZE=zero_block,
    )

    if n_elements > 0:
        group_size = int(x_contig.shape[-1])
        num_tables = int(q_contig.shape[0])
        block_size = max(_BACKWARD_BLOCK_SIZE, group_size)
        rows_per_block = max(1, block_size // group_size)
        num_outer = n_elements // (num_tables * group_size)
        grid = (num_tables, triton.cdiv(num_outer, rows_per_block))
        _quant_group_norm_backward_ste_sparse_t_stats_kernel[grid](
            x_contig,
            grad_contig,
            mins_contig,
            scale_contig,
            bins_contig,
            t_contig,
            grad_q,
            sum_left,
            sum_right,
            num_outer,
            float(eps),
            group_size,
            NUM_TABLES=num_tables,
            NUM_Q=int(q_contig.shape[1]),
            NUM_T=int(t_contig.shape[1]),
            ROWS_PER_BLOCK=rows_per_block,
            BLOCK_SIZE=block_size,
        )

    return grad_q, sum_left, sum_right
