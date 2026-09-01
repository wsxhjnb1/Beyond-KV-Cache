"""Graph-safe production launcher for the SM103 non-uniform KV decode kernel.

The CUDA kernel lives in :mod:`beyond_mixed_decode_sm100`; this module owns
the PyTorch/CuTe pointer bridge, compile cache, packed-cache ABI checks, and
workspace contract needed by vLLM.  Compilation is fail-closed during CUDA
Graph capture: every launch shape must be warmed before capture begins.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from quant.beyond_cute import CODE_LAYOUT_HND_TOKEN_WORD, PagedLayout
from quant.fa4_cute.beyond_decode_policy import (
    conversion_stage_cap,
    packed_kv_stage_cap,
    scheduled_pages_per_split,
    select_decode_launch_plan,
    select_direct_smem_store,
    select_reduction_kind,
    select_task_major_grid,
)
from quant.fa4_cute.beyond_mixed_decode_sm100 import (
    MixedInputFusedMultiHeadAttentionDecode,
)

_DEFAULT_PAGE_SIZE = 128
_HEAD_DIM = 128
_GROUP_SIZE = 32
_COMPILED: dict["DecodeRuntimeSpec", object] = {}
_COMPILE_LOCK = threading.RLock()


@dataclass(frozen=True)
class DecodeRuntimeSpec:
    device_index: int
    model_dtype: str
    qpoint_dtype: str
    batch_size: int
    heads_q: int
    heads_kv: int
    max_seq_len: int
    page_size: int
    num_splits: int
    physical_pages: int
    page_table_stride: int
    packed_page_stride_i32: int
    q_batch_stride: int
    kv_stage_cap: int
    cvt_stage_cap: int
    reduction_kind: str
    cluster_reduction: bool
    task_major_grid: bool
    direct_smem_store: bool
    wait_for_pdl_writer: bool

    @property
    def pages_per_split(self) -> int:
        return scheduled_pages_per_split(
            tiles=self.max_seq_len // self.page_size,
            splits=self.num_splits,
            adaptive_split_base=0,
            adaptive_extra_tasks=0,
        )


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise TypeError(f"expected fp16 or bf16, got {dtype}")


def _cutlass_dtype(name: str):
    if name == "fp16":
        return cutlass.Float16
    if name == "bf16":
        return cutlass.BFloat16
    raise ValueError(f"unsupported dtype name {name!r}")


def prepare_runtime_qpoints(
    q_points_k: torch.Tensor,
    q_points_v: torch.Tensor,
    *,
    heads_kv: int,
    storage_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the lane-major K and code-major V tables expected by the kernel.

    Input projection tables use the deployment shape ``(Hkv * 4, 16)`` (or
    its explicit ``(Hkv, 4, 16)`` form).  K and V remain distinct allocations
    and may contain completely independent non-uniform values.
    """
    if storage_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("runtime qpoints must use 16-bit fp16 or bf16 storage")
    expected = int(heads_kv) * (_HEAD_DIM // _GROUP_SIZE)

    def grouped(table: torch.Tensor, label: str) -> torch.Tensor:
        if table.numel() != expected * 16:
            raise ValueError(
                f"{label} qpoints require {expected} independent 16-point tables, "
                f"got {table.numel()} values"
            )
        if table.ndim == 2 and tuple(table.shape) != (expected, 16):
            raise ValueError(
                f"{label} qpoints must have shape ({expected}, 16), got "
                f"{tuple(table.shape)}"
            )
        if table.ndim not in (2, 3):
            raise ValueError(f"{label} qpoints must be rank 2 or 3")
        return table.reshape(int(heads_kv), _HEAD_DIM // _GROUP_SIZE, 16)

    grouped_k = grouped(q_points_k, "K")
    grouped_v = grouped(q_points_v, "V")
    k_lane_major = (
        grouped_k.permute(0, 2, 1)
        .to(device=q_points_k.device, dtype=storage_dtype)
        .contiguous()
    )
    v_code_major = (
        grouped_v.to(device=q_points_v.device, dtype=storage_dtype).contiguous()
    )
    return k_lane_major, v_code_major


def make_runtime_spec(
    *,
    q: torch.Tensor,
    cache: torch.Tensor,
    layout: PagedLayout,
    block_table: torch.Tensor,
    heads_q: int,
    max_seq_len: int,
    num_splits: int = 0,
    qpoint_dtype: torch.dtype = torch.float16,
    wait_for_pdl_writer: bool = False,
) -> DecodeRuntimeSpec:
    """Validate a launch bucket and freeze its measured SM103 policy."""
    if not q.is_cuda or not cache.is_cuda or not block_table.is_cuda:
        raise ValueError("q, packed cache, and block_table must be CUDA tensors")
    if torch.cuda.get_device_capability(q.device) != (10, 3):
        raise RuntimeError("the production non-uniform launcher requires B300 SM103")
    model_dtype = _dtype_name(q.dtype)
    qpoint_dtype_name = _dtype_name(qpoint_dtype)
    page_size = int(layout.block_size)
    if page_size not in (64, _DEFAULT_PAGE_SIZE):
        raise ValueError("packed cache block_size must be 64 or 128")
    if layout.code_layout != CODE_LAYOUT_HND_TOKEN_WORD or layout.abi_version != 2:
        raise ValueError("packed cache must use HND token-major ABI v2")
    if not layout.v_stats_group_major:
        raise ValueError("packed cache V statistics must retain PV group-major order")
    if (
        layout.bits != 4
        or layout.k_group_size != _GROUP_SIZE
        or layout.v_group_size != _GROUP_SIZE
        or layout.head_dim != _HEAD_DIM
    ):
        raise ValueError("launcher requires independent K/V Int4-G32-D128 cache")
    if cache.dtype != torch.int32 or cache.ndim != 2 or not cache.is_contiguous():
        raise ValueError("cache must be contiguous (physical_pages, block_i32) int32")
    if int(cache.shape[1]) != int(layout.block_i32):
        raise ValueError("cache row stride does not match PagedLayout.block_i32")
    if block_table.dtype != torch.int32 or block_table.ndim != 2:
        raise ValueError("block_table must be a rank-2 int32 CUDA tensor")
    if block_table.stride(1) != 1:
        raise ValueError("block_table must be contiguous within each request row")
    if max_seq_len <= 0 or max_seq_len % page_size:
        raise ValueError(
            f"max_seq_len must be a positive multiple of page_size={page_size}"
        )
    logical_pages = max_seq_len // page_size
    if logical_pages > int(block_table.shape[1]):
        raise ValueError("launch bucket exceeds the block-table capacity")
    if q.ndim != 3 or tuple(q.shape[1:]) != (int(heads_q), _HEAD_DIM):
        raise ValueError("q must have shape (B, Hq, D128)")
    batch_size = int(q.shape[0])
    if batch_size <= 0:
        raise ValueError("q decode batch must be non-empty")
    if int(q.stride(2)) != 1 or int(q.stride(1)) != _HEAD_DIM:
        raise ValueError("q must be contiguous within each attention head")
    q_batch_stride = int(q.stride(0))
    if q_batch_stride < int(heads_q) * _HEAD_DIM:
        raise ValueError("q batch stride overlaps adjacent decode rows")
    if int(heads_q) % int(layout.num_kv_heads):
        raise ValueError("heads_q must be divisible by heads_kv")

    tiles = max_seq_len // page_size
    cluster_reduction = False
    auto_plan = None
    if num_splits <= 0:
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
        auto_plan = select_decode_launch_plan(
            tiles=tiles,
            batch=batch_size,
            heads_kv=int(layout.num_kv_heads),
            sm_count=sm_count,
            model_dtype=model_dtype,
        )
        num_splits = auto_plan.splits
        # M64 uses independent half-warp head ownership. Keep DSM disabled
        # unless a complete backend gate, including the current-token writer,
        # demonstrates a win rather than only a standalone kernel speedup.
        cluster_reduction = bool(
            auto_plan.cluster_reduction and page_size == _DEFAULT_PAGE_SIZE
        )
    if not 1 <= int(num_splits) <= 32:
        raise ValueError("num_splits must be in [1, 32]")
    pages_per_split = scheduled_pages_per_split(
        tiles=tiles,
        splits=int(num_splits),
        adaptive_split_base=0,
        adaptive_extra_tasks=0,
    )
    reduction_kind = os.environ.get(
        "BEYOND_SM103_REDUCTION_KIND", "auto"
    ).strip().lower()
    if reduction_kind == "auto":
        reduction_kind = (
            auto_plan.reduction_kind
            if auto_plan is not None
            else select_reduction_kind(
                batch=batch_size,
                splits=int(num_splits),
                pages_per_split=pages_per_split,
            )
        )
    elif reduction_kind not in {
        "parallel",
        "cta4",
        "warp_parallel",
        "warp_parallel2",
    }:
        raise ValueError(
            "BEYOND_SM103_REDUCTION_KIND must be auto, parallel, cta4, "
            "warp_parallel, or warp_parallel2"
        )
    return DecodeRuntimeSpec(
        device_index=q.device.index if q.device.index is not None else torch.cuda.current_device(),
        model_dtype=model_dtype,
        qpoint_dtype=qpoint_dtype_name,
        batch_size=batch_size,
        heads_q=int(heads_q),
        heads_kv=int(layout.num_kv_heads),
        max_seq_len=int(max_seq_len),
        page_size=page_size,
        num_splits=int(num_splits),
        physical_pages=int(cache.shape[0]),
        page_table_stride=int(block_table.stride(0)),
        packed_page_stride_i32=int(layout.block_i32),
        q_batch_stride=q_batch_stride,
        kv_stage_cap=(
            auto_plan.kv_stage_cap
            if auto_plan is not None
            else packed_kv_stage_cap(pages_per_split=pages_per_split)
        ),
        cvt_stage_cap=(
            auto_plan.cvt_stage_cap
            if auto_plan is not None
            else conversion_stage_cap(
                splits=int(num_splits),
                pages_per_split=pages_per_split,
                model_dtype=model_dtype,
            )
        ),
        reduction_kind=reduction_kind,
        cluster_reduction=cluster_reduction,
        task_major_grid=(
            False
            if cluster_reduction
            else (
                auto_plan.task_major_grid
                if auto_plan is not None
                else select_task_major_grid(
                    batch=batch_size,
                    splits=int(num_splits),
                    pages_per_split=pages_per_split,
                )
            )
        ),
        direct_smem_store=(
            auto_plan.direct_smem_store
            if auto_plan is not None
            else select_direct_smem_store(
                batch=batch_size,
                heads_kv=int(layout.num_kv_heads),
                splits=int(num_splits),
                pages_per_split=pages_per_split,
            )
        ),
        wait_for_pdl_writer=bool(wait_for_pdl_writer),
    )


def allocate_runtime_workspace(
    spec: DecodeRuntimeSpec,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Allocate the stable split workspace before CUDA Graph capture."""
    model_dtype = torch.float16 if spec.model_dtype == "fp16" else torch.bfloat16
    partial_shape = (
        spec.num_splits,
        spec.batch_size,
        spec.heads_q,
        1,
        _HEAD_DIM,
    )
    stats_shape = partial_shape[:-1]
    return {
        "o_partial": torch.empty(partial_shape, dtype=model_dtype, device=device),
        "m_partial": torch.empty(stats_shape, dtype=torch.float32, device=device),
        "l_partial": torch.empty(stats_shape, dtype=torch.float32, device=device),
    }


def _as_cute(tensor: torch.Tensor, element_type):
    result = from_dlpack(tensor, assumed_align=16)
    result.element_type = element_type
    return result


def _packed_cache_cute_views(
    cache: torch.Tensor,
    layout: PagedLayout,
    model_dtype,
    torch_model_dtype: torch.dtype,
):
    flat = cache.reshape(-1)

    def code_region(offset_halves: int):
        return _as_cute(flat[offset_halves // 2 :], cutlass.Int4)

    def stat_region(offset_halves: int):
        return _as_cute(
            flat[offset_halves // 2 :].view(torch_model_dtype),
            model_dtype,
        )

    return (
        code_region(layout.k_code_off),
        code_region(layout.v_code_off),
        stat_region(layout.k_scale_off),
        stat_region(layout.v_scale_off),
    )


def _validate_launch_tensors(
    *,
    spec: DecodeRuntimeSpec,
    q: torch.Tensor,
    out: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    k_qpoints_lane_major: torch.Tensor,
    v_qpoints_code_major: torch.Tensor,
    workspace: dict[str, torch.Tensor],
) -> None:
    expected_shape = (spec.batch_size, spec.heads_q, _HEAD_DIM)
    expected_q = spec.batch_size * spec.heads_q * _HEAD_DIM
    if (
        tuple(q.shape) != expected_shape
        or int(q.stride(2)) != 1
        or int(q.stride(1)) != _HEAD_DIM
        or int(q.stride(0)) != spec.q_batch_stride
    ):
        raise ValueError("q shape or fused-QKV batch stride changed after compilation")
    if out.dtype != q.dtype or not out.is_contiguous() or out.numel() != expected_q:
        raise ValueError("out must match the contiguous dense q bucket")
    if block_table.dtype != torch.int32 or block_table.stride(1) != 1:
        raise ValueError("block_table must retain the compiled int32 row layout")
    if int(block_table.stride(0)) != spec.page_table_stride:
        raise ValueError("block_table row stride changed after compilation")
    if seq_lens.dtype != torch.int32 or not seq_lens.is_contiguous():
        raise ValueError("seq_lens must be contiguous int32")
    if seq_lens.numel() < spec.batch_size:
        raise ValueError("seq_lens is shorter than the decode batch")
    expected_k = (spec.heads_kv, 16, _HEAD_DIM // _GROUP_SIZE)
    expected_v = (spec.heads_kv, _HEAD_DIM // _GROUP_SIZE, 16)
    if tuple(k_qpoints_lane_major.shape) != expected_k:
        raise ValueError(f"K runtime qpoints must have shape {expected_k}")
    if tuple(v_qpoints_code_major.shape) != expected_v:
        raise ValueError(f"V runtime qpoints must have shape {expected_v}")
    if (
        not k_qpoints_lane_major.is_contiguous()
        or not v_qpoints_code_major.is_contiguous()
        or k_qpoints_lane_major.dtype != v_qpoints_code_major.dtype
        or _dtype_name(k_qpoints_lane_major.dtype) != spec.qpoint_dtype
    ):
        raise ValueError("K/V qpoints must be contiguous and share the compiled 16-bit dtype")
    partial = workspace.get("o_partial")
    m_partial = workspace.get("m_partial")
    l_partial = workspace.get("l_partial")
    partial_numel = spec.num_splits * expected_q
    stats_numel = spec.num_splits * spec.batch_size * spec.heads_q
    if (
        partial is None
        or partial.dtype != q.dtype
        or not partial.is_contiguous()
        or partial.numel() < partial_numel
    ):
        raise ValueError("o_partial workspace is missing or undersized")
    for name, value in (("m_partial", m_partial), ("l_partial", l_partial)):
        if (
            value is None
            or value.dtype != torch.float32
            or not value.is_contiguous()
            or value.numel() < stats_numel
        ):
            raise ValueError(f"{name} workspace is missing or undersized")


def _compile_or_get(
    *,
    spec: DecodeRuntimeSpec,
    q_cute,
    k_cute,
    v_cute,
    k_stats_cute,
    v_stats_cute,
    k_qpoints_cute,
    v_qpoints_cute,
    page_table_cute,
    seq_lens_cute,
    out_cute,
    o_partial_cute,
    m_partial_cute,
    l_partial_cute,
    softmax_scale: float,
    stream,
):
    compiled = _COMPILED.get(spec)
    if compiled is not None:
        return compiled
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "SM103 non-uniform decode bucket was not precompiled before CUDA Graph capture"
        )
    with _COMPILE_LOCK:
        compiled = _COMPILED.get(spec)
        if compiled is not None:
            return compiled
        grouped_heads = spec.heads_q // spec.heads_kv
        grouped_head_tile = min(math.ceil(grouped_heads / 8) * 8, 32)
        model_dtype = _cutlass_dtype(spec.model_dtype)
        qpoint_dtype = _cutlass_dtype(spec.qpoint_dtype)
        fmha = MixedInputFusedMultiHeadAttentionDecode(
            headdim=_HEAD_DIM,
            block_scaledim=_GROUP_SIZE,
            heads_per_kv=grouped_heads,
            grouped_head_tile=grouped_head_tile,
            page_size=spec.page_size,
            convert_warpgroups=2,
            lut_value_dtype=qpoint_dtype,
            kv_stage_cap=spec.kv_stage_cap,
            cvt_stage_cap=spec.cvt_stage_cap,
            reduction_kind=spec.reduction_kind,
            single_split_direct=spec.num_splits == 1,
            reduction_splits=spec.num_splits,
            packed_page_stride_i32=spec.packed_page_stride_i32,
            physical_page_capacity=spec.physical_pages,
            page_table_stride=spec.page_table_stride,
            q_batch_stride=spec.q_batch_stride,
            task_major_grid=spec.task_major_grid,
            cluster_reduction=spec.cluster_reduction,
            direct_smem_store=spec.direct_smem_store,
            wait_for_pdl_writer=spec.wait_for_pdl_writer,
        )
        problem_shape = (
            spec.batch_size,
            spec.heads_q,
            spec.heads_kv,
            spec.max_seq_len,
            _HEAD_DIM,
        )
        fmha.can_implement(
            problem_shape,
            spec.num_splits,
            model_dtype,
            cutlass.Int4,
            model_dtype,
            cutlass.Float32,
        )
        compiled = cute.compile(
            fmha,
            problem_shape,
            spec.num_splits,
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            k_stats_cute.iterator,
            v_stats_cute.iterator,
            k_qpoints_cute.iterator,
            v_qpoints_cute.iterator,
            page_table_cute.iterator,
            seq_lens_cute.iterator,
            out_cute.iterator,
            o_partial_cute.iterator,
            m_partial_cute.iterator,
            l_partial_cute.iterator,
            float(softmax_scale),
            1.0,
            stream,
            options="--opt-level 2",
        )
        _COMPILED[spec] = compiled
        return compiled


def run_nonuniform_decode(
    *,
    spec: DecodeRuntimeSpec,
    q: torch.Tensor,
    cache: torch.Tensor,
    layout: PagedLayout,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    k_qpoints_lane_major: torch.Tensor,
    v_qpoints_code_major: torch.Tensor,
    out: torch.Tensor,
    workspace: dict[str, torch.Tensor],
    softmax_scale: float,
) -> torch.Tensor:
    """Launch one pre-warmed dense single-token decode bucket."""
    _validate_launch_tensors(
        spec=spec,
        q=q,
        out=out,
        block_table=block_table,
        seq_lens=seq_lens,
        k_qpoints_lane_major=k_qpoints_lane_major,
        v_qpoints_code_major=v_qpoints_code_major,
        workspace=workspace,
    )
    if int(cache.shape[0]) != spec.physical_pages or int(cache.shape[1]) != spec.packed_page_stride_i32:
        raise ValueError("packed cache allocation changed after compilation")
    torch_model_dtype = q.dtype
    model_dtype = _cutlass_dtype(spec.model_dtype)
    qpoint_dtype = _cutlass_dtype(spec.qpoint_dtype)
    q_cute = _as_cute(q, model_dtype)
    out_cute = _as_cute(out, model_dtype)
    k_cute, v_cute, k_stats_cute, v_stats_cute = _packed_cache_cute_views(
        cache,
        layout,
        model_dtype,
        torch_model_dtype,
    )
    k_qpoints_cute = _as_cute(k_qpoints_lane_major, qpoint_dtype)
    v_qpoints_cute = _as_cute(v_qpoints_code_major, qpoint_dtype)
    page_table_cute = _as_cute(block_table, cutlass.Int32)
    seq_lens_cute = _as_cute(seq_lens, cutlass.Int32)
    o_partial_cute = _as_cute(workspace["o_partial"], model_dtype)
    m_partial_cute = _as_cute(workspace["m_partial"], cutlass.Float32)
    l_partial_cute = _as_cute(workspace["l_partial"], cutlass.Float32)
    torch_stream = torch.cuda.current_stream(q.device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled = _compile_or_get(
        spec=spec,
        q_cute=q_cute,
        k_cute=k_cute,
        v_cute=v_cute,
        k_stats_cute=k_stats_cute,
        v_stats_cute=v_stats_cute,
        k_qpoints_cute=k_qpoints_cute,
        v_qpoints_cute=v_qpoints_cute,
        page_table_cute=page_table_cute,
        seq_lens_cute=seq_lens_cute,
        out_cute=out_cute,
        o_partial_cute=o_partial_cute,
        m_partial_cute=m_partial_cute,
        l_partial_cute=l_partial_cute,
        softmax_scale=softmax_scale,
        stream=stream,
    )
    problem_shape = (
        spec.batch_size,
        spec.heads_q,
        spec.heads_kv,
        spec.max_seq_len,
        _HEAD_DIM,
    )
    compiled(
        problem_shape,
        spec.num_splits,
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        k_stats_cute.iterator,
        v_stats_cute.iterator,
        k_qpoints_cute.iterator,
        v_qpoints_cute.iterator,
        page_table_cute.iterator,
        seq_lens_cute.iterator,
        out_cute.iterator,
        o_partial_cute.iterator,
        m_partial_cute.iterator,
        l_partial_cute.iterator,
        float(softmax_scale),
        1.0,
        stream,
    )
    return out


def clear_compiled_runtime_cache() -> None:
    """Drop Python references to compiled launchers between isolated runs."""
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("cannot clear the decode compile cache during CUDA Graph capture")
    with _COMPILE_LOCK:
        _COMPILED.clear()
