"""vLLM v1 attention backend with 4-bit packed KV cache (Blackwell).

Scope
-----
* Production prefill uses the packed writer followed by the official FA4
  attention implementation.
* Production single-token decode has one SM103 non-uniform CuTe kernel family.
* Exact page-table readback plus PyTorch SDPA is an explicit numerical oracle.
* K and V are quantized per token along head_dim.
* Real packed compression supports only 4-bit groups of 32 values.

Wire-up
-------
Call ``register_backend()`` once before ``LLM(...)``, then pass
``attention_backend="CUSTOM"``. The KV cache is
allocated as an ``(num_blocks, block_i32)`` int32 buffer; per-block layout
is defined in :mod:`quant.beyond_cute.PagedLayout`.
"""

from __future__ import annotations

import functools
import gc
import hashlib
import json
import os
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as tl_cuda_libdevice

    _TRITON_OK = True
except Exception:
    triton = None
    tl = None
    tl_cuda_libdevice = None
    _TRITON_OK = False

# ---- project imports ----
_REPO = str(Path(__file__).resolve().parents[4])
for _p in (_REPO,):
    if _p not in sys.path:
        sys.path.insert(0, _p)


from beyond.common.compile_cache import configure_compile_cache  # noqa: E402
from beyond.runtime.vllm.config import (  # noqa: E402
    FAKE_QUANT_SUPPORTED_BITS as _FAKE_QUANT_SUPPORTED_BITS,
)
from beyond.runtime.vllm.config import (
    env_bits as _env_bits,
)
from beyond.runtime.vllm.config import (
    env_bool as _env_bool,
)
from beyond.runtime.vllm.config import (
    env_int as _env_int,
)
from beyond.runtime.vllm.config import (
    validate_real_packed_layout as _validate_real_packed_layout,
)

configure_compile_cache(_REPO)

from beyond.quantization.runtime_config import (  # noqa: E402
    get_config_from_env,
)
from beyond.quantization.table_precision import (  # noqa: E402
    TABLE_PRECISION_ABI,
    TABLE_STORAGE_DTYPE_NAME,
)
from quant.beyond_cute import (  # noqa: E402
    CODE_LAYOUT_HND_TOKEN_WORD,
    PagedLayout,
    dequantize_paged_kv_to_dense,
    read_request_from_paged_cache,
    write_tokens_to_paged_cache,
)
from quant.beyond_cute import (  # noqa: E402
    _packed_readback_launch_policy as _select_packed_readback_launch_policy,
)
from quant.beyond_cute import (  # noqa: E402
    _packed_readback_use_flat_grid as _select_packed_readback_use_flat_grid,
)
from quant.beyond_cute import (
    _packed_writer_groups_per_program as _select_packed_writer_groups_per_program,
)

# ---- vllm imports ----
# Block flash_attn.ops import failure pattern seen in beyond_backend.py.
sys.modules.setdefault("flash_attn", None)

from vllm.v1.attention.backend import AttentionCGSupport  # noqa: E402
from vllm.v1.attention.backends.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec  # noqa: E402
from vllm.vllm_flash_attn import flash_attn_varlen_func  # noqa: E402

# ---------------------------------------------------------------------------
# Tunables (env-driven; real packed inference validates the active 4-bit/G32 layout)
# ---------------------------------------------------------------------------


BEYOND_BITS = _env_bits("BEYOND_BITS")
BEYOND_K_GROUP_SIZE = _env_int("BEYOND_K_GROUP_SIZE", 32)
BEYOND_V_GROUP_SIZE = _env_int("BEYOND_V_GROUP_SIZE", 32)
BEYOND_PACKED_BLOCK_SIZE = 128
BEYOND_PACKED_BLOCK_SIZES = (64, BEYOND_PACKED_BLOCK_SIZE)
BEYOND_ENABLE_PIECEWISE_GRAPH = _env_int("BEYOND_ENABLE_PIECEWISE_GRAPH", 0)
BEYOND_ENABLE_CUDAGRAPH_DECODE = _env_int("BEYOND_ENABLE_CUDAGRAPH_DECODE", 0)
BEYOND_PREWARM_FULL_DECODE_GRAPH = _env_int(
    "BEYOND_PREWARM_FULL_DECODE_GRAPH",
    1,
)
BEYOND_FULL_DECODE_MAX_BATCH = _env_int("BEYOND_FULL_DECODE_MAX_BATCH", 0)
BEYOND_MOE_FULL_GRAPH_MAX_MODEL_LEN = _env_int(
    "BEYOND_MOE_FULL_GRAPH_MAX_MODEL_LEN",
    131072,
)
BEYOND_MOE_FULL_GRAPH_PROFILE_SEQ_LEN = _env_int(
    "BEYOND_MOE_FULL_GRAPH_PROFILE_SEQ_LEN",
    128,
)
BEYOND_CUDAGRAPH_WORKSPACE_RESERVE_MB = _env_int(
    "BEYOND_CUDAGRAPH_WORKSPACE_RESERVE_MB",
    0,
)
BEYOND_CUDAGRAPH_KV_SAFETY_MARGIN_MB = _env_int(
    "BEYOND_CUDAGRAPH_KV_SAFETY_MARGIN_MB",
    0,
)
BEYOND_INDUCTOR_ACTIVATION_RESERVE_MB = _env_int(
    "BEYOND_INDUCTOR_ACTIVATION_RESERVE_MB",
    0,
)


if _TRITON_OK:

    @triton.jit(do_not_specialize=("num_reqs", "num_tokens"))
    def _beyond_compute_slot_mapping_kernel(
        num_reqs,
        num_tokens,
        max_num_tokens,
        query_start_loc_ptr,
        positions_ptr,
        block_table_ptr,
        block_table_stride,
        block_size,
        slot_mapping_ptr,
        TOTAL_CP_WORLD_SIZE: tl.constexpr,
        TOTAL_CP_RANK: tl.constexpr,
        CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
        PAD_ID: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """vLLM slot mapping without a batch-size-specialized grid query."""
        req_idx = tl.program_id(0)
        if req_idx == num_reqs:
            for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(
                    slot_mapping_ptr + offsets,
                    PAD_ID,
                    mask=offsets < max_num_tokens,
                )
            return

        start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
        end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)
        virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
        row_offset = req_idx * block_table_stride
        for i in range(start_idx, end_idx, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < end_idx
            pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
            block_indices = pos // virtual_block_size
            block_numbers = tl.load(
                block_table_ptr + row_offset + block_indices
            ).to(tl.int64)
            virtual_block_offsets = pos - block_indices * virtual_block_size
            is_local = (
                virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
            ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
            local_block_offsets = (
                virtual_block_offsets
                // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
            ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
                virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
            )
            slot_ids = block_numbers * block_size + local_block_offsets
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)
            tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


def _beyond_vllm_page_table_exact_enabled() -> bool:
    return _env_bool("BEYOND_VLLM_PAGE_TABLE_EXACT", False)


def _decode_graph_past_cap(max_model_len: int, block_table, block_size: int) -> int:
    max_model_len = int(max_model_len or 0)
    if max_model_len <= 1 or block_table is None:
        return 0
    try:
        block_table_blocks = int(block_table.shape[1])
    except (AttributeError, IndexError, TypeError, ValueError):
        return 0
    cap = min(max_model_len - 1, block_table_blocks * int(block_size))
    cap = max(0, int(cap))
    return cap


def _beyond_full_decode_graph_overrides(
    *,
    enabled: bool,
    max_query_len: int | None,
    max_model_len: int,
    block_table,
    block_size: int,
) -> tuple[bool, int]:
    """Return stable full-decode graph plan overrides for capture and replay."""
    force_full_decode = bool(enabled and int(max_query_len or 0) == 1)
    if not force_full_decode:
        return False, 0
    graph_past_cap = _decode_graph_past_cap(
        int(max_model_len or 0),
        block_table,
        int(block_size),
    )
    return True, int(graph_past_cap)


def _beyond_has_pending_prompt_tokens(input_batch, num_reqs: int) -> bool:
    """Whether any active request still has prompt tokens left to compute."""
    computed = input_batch.num_computed_tokens_cpu[: int(num_reqs)]
    prompt = input_batch.num_prompt_tokens[: int(num_reqs)]
    return any(int(done) < int(total) for done, total in zip(computed, prompt))


def _beyond_moe_requires_padded_full_decode(vllm_config) -> bool:
    """Whether FULL decode must avoid the one-token model shape.

    FlashInfer's TRT-LLM MoE path on SM103 rejects the ``M=1`` dummy run used
    for FULL CUDA Graph capture.  The same model runs correctly when a single
    request is represented by the next real graph bucket (normally batch 2),
    with the inactive row masked by vLLM metadata. Dense models do not have this
    restriction and retain their exact batch-1 graph.
    """
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return False
    try:
        is_moe = getattr(model_config, "is_moe", False)
        if callable(is_moe):
            is_moe = is_moe()
        return bool(is_moe)
    except Exception:
        architectures = getattr(model_config, "architectures", ()) or ()
        return any("moe" in str(arch).lower() for arch in architectures)


def _beyond_long_context_moe_uses_bounded_full_graph_profile(vllm_config) -> bool:
    """Whether native long-context MoE needs a bounded FULL-graph dummy row.

    The SM103 attention launcher is compiled against the entire block-table
    capacity, so bounding the dummy sequence does not reduce replay capacity.
    It only avoids making vLLM's synthetic FULL-graph warmup traverse a large
    unpopulated page-table prefix.  Qwen's native 256k configuration completes
    reliably with a 128-token dummy while retaining FULL decode graphs.
    """
    if not _beyond_moe_requires_padded_full_decode(vllm_config):
        return False
    limit = _as_positive_int(BEYOND_MOE_FULL_GRAPH_MAX_MODEL_LEN)
    if limit is None:
        return False
    model_config = getattr(vllm_config, "model_config", None)
    max_model_len = _as_positive_int(getattr(model_config, "max_model_len", None))
    return max_model_len is not None and int(max_model_len) > int(limit)


def _beyond_select_full_decode_graph_keys(
    full_keys,
    *,
    full_decode_batch_cap: int,
    pad_single_token: bool,
):
    """Select stable FULL graph keys without capturing an unsafe MoE M=1."""
    eligible_keys = {
        key
        for key in full_keys
        if int(getattr(key, "num_tokens", 0)) <= int(full_decode_batch_cap)
    }
    if pad_single_token:
        safe_keys = {
            key
            for key in eligible_keys
            if int(getattr(key, "num_tokens", 0)) >= 2
        }
        # Fail closed when the configured capture roster contains only M=1:
        # PIECEWISE/NONE is slower but cannot trip the TRT-LLM MoE capture bug.
        return safe_keys
    return eligible_keys


def _beyond_dense_decode_batch_eligible(
    *,
    num_reqs: int,
    num_actual_tokens: int,
    prefill_reqs: list[int],
    chunked_prefill_reqs: list[int],
    decode_reqs: list[int],
    q_indices_dense: bool,
) -> bool:
    """Accept ordinary decode and a dense one-token cached-prefix tail.

    vLLM deliberately keeps the final prompt token on its prefill metadata
    path after a prefix-cache hit.  The attention operation is nevertheless
    identical to single-token decode when every active request contributes one
    contiguous query.  Keeping that case on the dense packed path avoids a
    dynamic cache readback without misclassifying mixed/chunked prefill.
    """
    num_reqs = int(num_reqs)
    return bool(
        num_reqs > 0
        and int(num_actual_tokens) >= num_reqs
        and not prefill_reqs
        and not chunked_prefill_reqs
        and len(decode_reqs) == num_reqs
        and decode_reqs == list(range(num_reqs))
        and q_indices_dense
    )


def _cuda_is_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _sm103_nonuniform_decode_mode() -> str:
    return os.environ.get("BEYOND_SM103_NONUNIFORM_DECODE", "auto").strip().lower()


def _sm103_nonuniform_decode_requested(device: torch.device) -> bool:
    mode = _sm103_nonuniform_decode_mode()
    if mode in {"0", "false", "no", "off", "disable", "disabled", "none", ""}:
        return False
    if not torch.cuda.is_available():
        return False
    supported = torch.cuda.get_device_capability(device) == (10, 3)
    if mode == "force" and not supported:
        raise RuntimeError("BEYOND_SM103_NONUNIFORM_DECODE=force requires B300 SM103")
    return supported


def _sm103_decode_length_bucket(
    total_seq_len: int,
    table_capacity: int,
    block_size: int = BEYOND_PACKED_BLOCK_SIZE,
) -> int:
    """Use power-of-two graph/JIT buckets without exceeding the page table."""
    total_seq_len = max(1, int(total_seq_len))
    table_capacity = int(table_capacity)
    block_size = int(block_size)
    if block_size not in BEYOND_PACKED_BLOCK_SIZES:
        raise ValueError("SM103 decode block_size must be 64 or 128")
    if table_capacity <= 0 or table_capacity % block_size:
        raise ValueError(
            "block-table token capacity must be a positive multiple of block_size"
        )
    if total_seq_len > table_capacity:
        raise ValueError(
            f"decode sequence length {total_seq_len} exceeds block-table capacity "
            f"{table_capacity}"
        )
    bucket = max(block_size, 1 << (total_seq_len - 1).bit_length())
    return min(bucket, table_capacity)


def _sm103_decode_compile_bucket(
    total_seq_len: int,
    table_capacity: int,
    *,
    cuda_graph_enabled: bool,
    block_size: int = BEYOND_PACKED_BLOCK_SIZE,
) -> int:
    """Choose a replay-safe graph bucket or a smaller eager JIT bucket."""
    eager_bucket = _sm103_decode_length_bucket(
        total_seq_len,
        table_capacity,
        block_size,
    )
    return int(table_capacity) if cuda_graph_enabled else eager_bucket


def _get_sm103_nonuniform_workspace(spec, device: torch.device) -> dict[str, torch.Tensor]:
    workspace = _SM103_NONUNIFORM_WORKSPACE.get(spec)
    if workspace is not None:
        return workspace
    if _cuda_is_capturing():
        raise RuntimeError(
            "SM103 non-uniform decode workspace was not allocated before CUDA Graph capture"
        )
    from quant.fa4_cute.beyond_mixed_decode_runtime import (
        allocate_runtime_workspace,
    )

    workspace = allocate_runtime_workspace(spec, device=device)
    _SM103_NONUNIFORM_WORKSPACE[spec] = workspace
    return workspace


@functools.lru_cache(maxsize=None)
def _paged_layout_cached(
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    bits: int,
    k_group_size: int,
    v_group_size: int,
) -> PagedLayout:
    return PagedLayout.build_hnd(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        bits=bits,
        k_group_size=k_group_size,
        v_group_size=v_group_size,
    )


def _paged_layout(block_size: int, num_kv_heads: int, head_size: int) -> PagedLayout:
    return _paged_layout_cached(
        block_size,
        num_kv_heads,
        head_size,
        BEYOND_BITS,
        BEYOND_K_GROUP_SIZE,
        BEYOND_V_GROUP_SIZE,
    )


def _beyond_runner_packed_layout(runner, num_kv_heads: int) -> PagedLayout | None:
    """Return the serving layout selected by vLLM's cache configuration."""
    cache_config = getattr(runner, "cache_config", None)
    block_size = int(getattr(cache_config, "block_size", 0) or 0)
    if block_size not in BEYOND_PACKED_BLOCK_SIZES:
        return None
    return _paged_layout(block_size, int(num_kv_heads), 128)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class BeyondPackedSpec(FullAttentionSpec):
    """KV cache spec carrying our packed layout.

    ``page_size_padded`` holds the int32-bytes-per-block (== block_i32 * 4)
    so vLLM's allocator sizes the pool correctly.
    """

    @property
    def real_page_size_bytes(self) -> int:
        # Our cache is a flat (num_blocks, block_i32) int32 buffer; each block
        # holds bit-packed K codes + K stats + V codes + V stats.
        layout = _paged_layout(self.block_size, self.num_kv_heads, self.head_size)
        return layout.block_i32 * 4

# ---------------------------------------------------------------------------
# Impl
# ---------------------------------------------------------------------------


_live_impls: "weakref.WeakSet" = weakref.WeakSet()
_HBM_QTABLE_CACHE: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
_DECODE_SLOT_WORKSPACE: dict[tuple[object, ...], torch.Tensor] = {}
_CHUNKED_PREFILL_DENSE_WORKSPACE: dict[tuple[object, ...], dict[str, object]] = {}
_SM103_NONUNIFORM_WORKSPACE: dict[object, dict[str, torch.Tensor]] = {}
_BLOCK_SIZE_CACHE: dict[tuple[int, int, int, int, int, int], int] = {}
_AOT_CUBLAS_WARMED_DEVICES: set[int] = set()
_FAKE_QUANT_QTABLE_DEBUGGED: set[tuple[str, str]] = set()
_FAKE_QUANT_DEBUGGED: set[tuple[int, str]] = set()
_FAKE_QUANT_VERIFY_CALL_COUNT: dict[int, int] = {}
_ACTIVE_CONFIG_IDENTITY: str | None = None
_ACTIVE_CONFIG_STRICT: bool | None = None
_FAKE_QUANT_QTABLE_CACHE_EPOCH = 0
_FAKE_QUANT_LAYER_QTABLE_CACHE_ATTR = "_beyond_fake_quant_cache_side_qtables"
_BEYOND_INDEX_CACHE_ATTRS = (
    "_beyond_chunked_seq_start_gpu_i32",
    "_beyond_dense_decode_slot_mapping_gpu_i32",
)
_RUNTIME_AUDIT_COUNTS = {
    "production_prefill_calls": 0,
    "production_chunked_prefill_calls": 0,
    "rounded_prefill_attention_calls": 0,
    "mixed_prefill_decode_split_calls": 0,
    "page_table_exact_calls": 0,
    "prefix_tail_piecewise_calls": 0,
    "prefix_tail_dense_decode_calls": 0,
    "sm103_nonuniform_decode_launches": 0,
}
_RUNTIME_AUDIT_MIXED_PLANS: dict[tuple[object, ...], int] = {}
_RUNTIME_AUDIT_CHUNKED_PLANS: dict[tuple[object, ...], int] = {}


def reset_runtime_audit() -> None:
    """Reset opt-in runtime diagnostics."""
    for key in _RUNTIME_AUDIT_COUNTS:
        _RUNTIME_AUDIT_COUNTS[key] = 0
    _RUNTIME_AUDIT_MIXED_PLANS.clear()
    _RUNTIME_AUDIT_CHUNKED_PLANS.clear()


def runtime_audit_snapshot() -> dict[str, object]:
    """Return process-local proof that prefill/decode production paths ran."""
    counts: dict[str, object] = {
        key: int(value) for key, value in _RUNTIME_AUDIT_COUNTS.items()
    }
    counts["mixed_plan_shapes"] = [
        {
            "prefill_reqs": list(signature[0]),
            "chunked_prefill_reqs": list(signature[1]),
            "decode_reqs": list(signature[2]),
            "q_lens": list(signature[3]),
            "seq_lens": list(signature[4]),
            "calls": int(call_count),
        }
        for signature, call_count in sorted(
            _RUNTIME_AUDIT_MIXED_PLANS.items(),
            key=lambda item: repr(item[0]),
        )
    ]
    counts["chunked_plan_shapes"] = [
        {
            "q_lens": list(signature[0]),
            "context_lens": list(signature[1]),
            "seq_lens": list(signature[2]),
            "calls": int(call_count),
        }
        for signature, call_count in sorted(
            _RUNTIME_AUDIT_CHUNKED_PLANS.items(),
            key=lambda item: repr(item[0]),
        )
    ]
    if _env_bool("BEYOND_SM103_GRAPH_DEBUG", False):
        torch.cuda.synchronize()
        metadata: list[dict[str, object]] = []
        dump_path = os.environ.get("BEYOND_SM103_GRAPH_DUMP_PATH", "").strip()
        dump_batch = _env_int("BEYOND_SM103_GRAPH_DUMP_BATCH", 2)
        dump_label = os.environ.get(
            "BEYOND_SM103_GRAPH_DUMP_LABEL", ".layers.0."
        ).strip()
        dumped = False
        for impl in list(_live_impls):
            buffers_by_signature = getattr(
                impl, "_sm103_graph_debug_buffers", {}
            )
            context_by_signature = getattr(
                impl, "_sm103_graph_debug_context", {}
            )
            for signature, buffers in buffers_by_signature.items():
                context = context_by_signature.get(signature)
                if context is None:
                    continue
                q = buffers["query"]
                out = buffers["output"]
                block_table = buffers["block_table"]
                batch = int(q.shape[0])
                lengths = [
                    int(x) for x in buffers["seq_lens"][:batch].cpu().tolist()
                ]
                slot_values = [
                    int(x)
                    for x in buffers["slot_mapping"][:batch].cpu().tolist()
                ]
                page_size = int(context["layout"].block_size)
                derived_slots: list[int] = []
                block_rows: list[list[int]] = []
                for req, length in enumerate(lengths):
                    if length <= 0:
                        derived_slots.append(-1)
                        block_rows.append([])
                        continue
                    logical_page = (length - 1) // page_size
                    pages = block_table[req, : logical_page + 1].cpu().tolist()
                    block_rows.append([int(x) for x in pages])
                    physical_page = int(pages[-1])
                    derived_slots.append(
                        physical_page * page_size
                        + (length - 1) % page_size
                    )
                metadata.append(
                    {
                        "layer": str(context["label"]),
                        "batch": batch,
                        "page_size": page_size,
                        "seq_lens": lengths,
                        "slot_mapping": slot_values,
                        "derived_slots": derived_slots,
                        "block_table_prefix": block_rows,
                        "q_finite": bool(torch.isfinite(q).all().item()),
                        "out_finite": bool(torch.isfinite(out).all().item()),
                    }
                )
                if (
                    dump_path
                    and not dumped
                    and batch == int(dump_batch)
                    and dump_label in str(context["label"])
                ):
                    used_ids = [
                        page
                        for row in block_rows
                        for page in row
                        if int(page) >= 0
                    ]
                    if used_ids:
                        unique_ids = torch.tensor(
                            sorted(set(used_ids)),
                            dtype=torch.long,
                            device=q.device,
                        )
                        cache = context["cache"]
                        compact_cache = cache.index_select(0, unique_ids)
                        id_map = {
                            int(page): index
                            for index, page in enumerate(unique_ids.cpu().tolist())
                        }
                        compact_table = torch.full_like(block_table, -1)
                        for req, row in enumerate(block_rows):
                            for logical_page, physical_page in enumerate(row):
                                compact_table[req, logical_page] = id_map[
                                    int(physical_page)
                                ]
                        torch.save(
                            {
                                "query": q.detach().cpu(),
                                "key": buffers["key"].detach().cpu(),
                                "value": buffers["value"].detach().cpu(),
                                "captured_output": out.detach().cpu(),
                                "slot_mapping": buffers[
                                    "slot_mapping"
                                ].detach().cpu(),
                                "cache": compact_cache.detach().cpu(),
                                "block_table": compact_table.detach().cpu(),
                                "seq_lens": buffers["seq_lens"].detach().cpu(),
                                "k_qpoints_lane_major": context[
                                    "runtime_k"
                                ].detach().cpu(),
                                "v_qpoints_code_major": context[
                                    "runtime_v"
                                ].detach().cpu(),
                                "layout": context["layout"],
                                "heads_q": int(q.shape[1]),
                                "max_seq_len": int(context["max_seq_len"]),
                                "num_splits": int(context["num_splits"]),
                                "softmax_scale": float(context["softmax_scale"]),
                                "debug_label": str(context["label"]),
                            },
                            dump_path,
                        )
                        dumped = True
        metadata.sort(key=lambda item: (str(item["layer"]), int(item["batch"])))
        counts["sm103_graph_debug"] = metadata
        counts["sm103_graph_dump"] = dump_path if dumped else None
    return counts


def _runtime_audit_event(name: str) -> None:
    if _env_bool("BEYOND_RUNTIME_AUDIT", False):
        _RUNTIME_AUDIT_COUNTS[name] += 1


def _runtime_audit_mixed_plan(
    *,
    prefill_reqs: list[int],
    chunked_prefill_reqs: list[int],
    decode_reqs: list[int],
    q_starts: list[int],
    seq_lens: list[int],
) -> None:
    if not _env_bool("BEYOND_RUNTIME_AUDIT", False):
        return
    q_lens = tuple(
        int(q_starts[i + 1]) - int(q_starts[i])
        for i in range(len(q_starts) - 1)
    )
    signature = (
        tuple(int(x) for x in prefill_reqs),
        tuple(int(x) for x in chunked_prefill_reqs),
        tuple(int(x) for x in decode_reqs),
        q_lens,
        tuple(int(x) for x in seq_lens),
    )
    _RUNTIME_AUDIT_MIXED_PLANS[signature] = (
        _RUNTIME_AUDIT_MIXED_PLANS.get(signature, 0) + 1
    )


def _runtime_audit_chunked_plan(
    *,
    q_starts: list[int],
    seq_lens: list[int],
) -> None:
    if not _env_bool("BEYOND_RUNTIME_AUDIT", False):
        return
    q_lens = tuple(
        int(q_starts[i + 1]) - int(q_starts[i])
        for i in range(len(q_starts) - 1)
    )
    active_seq_lens = tuple(int(x) for x in seq_lens[: len(q_lens)])
    context_lens = tuple(
        seq_len - q_len for seq_len, q_len in zip(active_seq_lens, q_lens)
    )
    signature = (q_lens, context_lens, active_seq_lens)
    _RUNTIME_AUDIT_CHUNKED_PLANS[signature] = (
        _RUNTIME_AUDIT_CHUNKED_PLANS.get(signature, 0) + 1
    )


def refresh_env_tunables() -> None:
    """Refresh module-level tunables after callers update ``BEYOND_*`` env vars."""
    global BEYOND_BITS, BEYOND_K_GROUP_SIZE, BEYOND_V_GROUP_SIZE
    global BEYOND_ENABLE_PIECEWISE_GRAPH, BEYOND_ENABLE_CUDAGRAPH_DECODE
    global BEYOND_PREWARM_FULL_DECODE_GRAPH
    global BEYOND_FULL_DECODE_MAX_BATCH
    global BEYOND_MOE_FULL_GRAPH_MAX_MODEL_LEN
    global BEYOND_MOE_FULL_GRAPH_PROFILE_SEQ_LEN
    global BEYOND_CUDAGRAPH_WORKSPACE_RESERVE_MB
    global BEYOND_CUDAGRAPH_KV_SAFETY_MARGIN_MB
    global BEYOND_INDUCTOR_ACTIVATION_RESERVE_MB
    global _ACTIVE_CONFIG_IDENTITY, _ACTIVE_CONFIG_STRICT
    global _FAKE_QUANT_QTABLE_CACHE_EPOCH

    old_layout = (BEYOND_BITS, BEYOND_K_GROUP_SIZE, BEYOND_V_GROUP_SIZE)

    BEYOND_BITS = _env_bits("BEYOND_BITS")
    BEYOND_K_GROUP_SIZE = _env_int("BEYOND_K_GROUP_SIZE", 32)
    BEYOND_V_GROUP_SIZE = _env_int("BEYOND_V_GROUP_SIZE", 32)
    BEYOND_ENABLE_PIECEWISE_GRAPH = _env_int("BEYOND_ENABLE_PIECEWISE_GRAPH", 0)
    BEYOND_ENABLE_CUDAGRAPH_DECODE = _env_int("BEYOND_ENABLE_CUDAGRAPH_DECODE", 0)
    BEYOND_PREWARM_FULL_DECODE_GRAPH = _env_int(
        "BEYOND_PREWARM_FULL_DECODE_GRAPH",
        1,
    )
    BEYOND_FULL_DECODE_MAX_BATCH = _env_int("BEYOND_FULL_DECODE_MAX_BATCH", 0)
    BEYOND_MOE_FULL_GRAPH_MAX_MODEL_LEN = _env_int(
        "BEYOND_MOE_FULL_GRAPH_MAX_MODEL_LEN",
        131072,
    )
    BEYOND_MOE_FULL_GRAPH_PROFILE_SEQ_LEN = _env_int(
        "BEYOND_MOE_FULL_GRAPH_PROFILE_SEQ_LEN",
        128,
    )
    BEYOND_CUDAGRAPH_WORKSPACE_RESERVE_MB = _env_int(
        "BEYOND_CUDAGRAPH_WORKSPACE_RESERVE_MB",
        0,
    )
    BEYOND_CUDAGRAPH_KV_SAFETY_MARGIN_MB = _env_int(
        "BEYOND_CUDAGRAPH_KV_SAFETY_MARGIN_MB",
        0,
    )
    BEYOND_INDUCTOR_ACTIVATION_RESERVE_MB = _env_int(
        "BEYOND_INDUCTOR_ACTIVATION_RESERVE_MB",
        0,
    )

    new_layout = (BEYOND_BITS, BEYOND_K_GROUP_SIZE, BEYOND_V_GROUP_SIZE)
    cfg = get_config_from_env(force_reload=True)
    config_identity = None if cfg is None else cfg.identity
    config_strict = _env_bool("BEYOND_QUANT_CONFIG_STRICT", True)
    if (
        new_layout != old_layout
        or config_identity != _ACTIVE_CONFIG_IDENTITY
        or config_strict != _ACTIVE_CONFIG_STRICT
    ):
        _paged_layout_cached.cache_clear()
        _BLOCK_SIZE_CACHE.clear()
        _HBM_QTABLE_CACHE.clear()
        _FAKE_QUANT_QTABLE_DEBUGGED.clear()
        _FAKE_QUANT_DEBUGGED.clear()
        # Per-layer fake-QDQ qtables are cached on the layer itself.  Advancing
        # the epoch invalidates those entries without retaining strong global
        # references to model layers.
        _FAKE_QUANT_QTABLE_CACHE_EPOCH += 1
    _ACTIVE_CONFIG_IDENTITY = config_identity
    _ACTIVE_CONFIG_STRICT = config_strict
    _cached_beyond_tensor_parallel_rank_world.cache_clear()


def _clear_beyond_tensor_workspaces() -> None:
    """Drop graph-profile workspaces before vLLM allocates the real KV cache."""
    _DECODE_SLOT_WORKSPACE.clear()
    _CHUNKED_PREFILL_DENSE_WORKSPACE.clear()
    _SM103_NONUNIFORM_WORKSPACE.clear()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def _clear_beyond_index_caches(attn_metadata) -> None:
    for attr_name in _BEYOND_INDEX_CACHE_ATTRS:
        for suffix in ("", "_signature"):
            try:
                delattr(attn_metadata, attr_name + suffix)
            except AttributeError:
                pass


def _beyond_fake_quant_enabled() -> bool:
    return os.environ.get("BEYOND_FAKE_QUANT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _beyond_fake_quant_debug_enabled() -> bool:
    return os.environ.get("BEYOND_FAKE_QUANT_DEBUG", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _beyond_fake_quant_verify_oracle_enabled() -> bool:
    """Return whether the debug-only same-input semantic oracle is active."""
    return os.environ.get("BEYOND_FAKE_QUANT_VERIFY_ORACLE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _beyond_fake_quant_debug(message: str, *, once_key: str | None = None) -> None:
    if not _beyond_fake_quant_debug_enabled():
        return
    if _torch_is_compiling():
        return
    if once_key is not None:
        debug_key = (os.getpid(), once_key)
        if debug_key in _FAKE_QUANT_DEBUGGED:
            return
        _FAKE_QUANT_DEBUGGED.add(debug_key)
    line = f"[beyond] fake_quant_debug pid={os.getpid()} {message}"
    path = os.environ.get("BEYOND_FAKE_QUANT_DEBUG_FILE")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return
        except Exception:
            pass
    print(line, flush=True)


def _beyond_fake_quant_triton_enabled() -> bool:
    value = os.environ.get("BEYOND_FAKE_QUANT_TRITON", "1")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _beyond_fake_quant_fused_kv_enabled() -> bool:
    value = os.environ.get("BEYOND_FAKE_QUANT_FUSED_KV", "1")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _beyond_fake_quant_direct_cache_enabled() -> bool:
    """Enable fused QDQ-to-paged-cache scatter for the supported dense layout."""
    value = os.environ.get("BEYOND_FAKE_QUANT_DIRECT_CACHE", "1")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _beyond_fake_quant_fused_kv_max_tokens() -> int:
    # Non-positive means no cap.  B300 microbench keeps fused ahead from decode
    # to 32k-token chunks; keep an escape hatch for unusual prefill regimes.
    return _env_int("BEYOND_FAKE_QUANT_FUSED_KV_MAX_TOKENS", 0)


def _torch_is_compiling() -> bool:
    try:
        compiler = getattr(torch, "compiler", None)
        is_compiling = getattr(compiler, "is_compiling", None)
        if is_compiling is not None and bool(is_compiling()):
            return True
    except Exception:
        pass
    try:
        dynamo = getattr(torch, "_dynamo", None)
        is_compiling = getattr(dynamo, "is_compiling", None)
        return bool(is_compiling is not None and is_compiling())
    except Exception:
        return False


def _beyond_tensor_parallel_rank_world() -> tuple[int, int]:
    """Best-effort tensor-parallel rank discovery inside vLLM workers."""
    try:
        from vllm.distributed import parallel_state as ps

        rank_fn = getattr(ps, "get_tensor_model_parallel_rank", None)
        world_fn = getattr(ps, "get_tensor_model_parallel_world_size", None)
        if rank_fn is not None and world_fn is not None:
            return int(rank_fn()), max(1, int(world_fn()))
    except Exception:
        pass

    try:
        dist = getattr(torch, "distributed", None)
        if dist is not None and dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), max(1, int(dist.get_world_size()))
    except Exception:
        pass

    try:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    except ValueError:
        rank = 0
    try:
        world = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world = 1
    return rank, max(1, world)


@functools.lru_cache(maxsize=1)
def _cached_beyond_tensor_parallel_rank_world() -> tuple[int, int]:
    """Cache the worker's immutable tensor-parallel topology."""
    return _beyond_tensor_parallel_rank_world()


def _local_group_qtable_count(x: torch.Tensor, group_size: int) -> int | None:
    if x.dim() != 3:
        return None
    G = int(group_size)
    if G <= 0:
        return None
    H = int(x.shape[1])
    D = int(x.shape[2])
    if D % G != 0:
        return None
    return H * (D // G)


def _slice_group_qtables_for_tensor_parallel(
    q_points: torch.Tensor,
    thresholds: torch.Tensor,
    local_num_tables: int | None,
    *,
    label: str,
    tp_rank: int | None = None,
    tp_world: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice full-model rank-2 grouped qtables to this worker's local KV heads."""
    if q_points.dim() != 2 or thresholds.dim() != 2 or local_num_tables is None:
        return q_points, thresholds

    local_num_tables = int(local_num_tables)
    full_num_tables = int(q_points.shape[0])
    if full_num_tables == local_num_tables:
        return q_points, thresholds

    if tp_rank is None or tp_world is None:
        tp_rank, tp_world = _beyond_tensor_parallel_rank_world()
    tp_rank = int(tp_rank)
    tp_world = max(1, int(tp_world))

    if full_num_tables == local_num_tables * tp_world:
        start = tp_rank * local_num_tables
        end = start + local_num_tables
        if start < 0 or end > full_num_tables:
            raise ValueError(
                f"{label} per-group qtables cannot be sliced for "
                f"tp_rank={tp_rank}, tp_world={tp_world}: "
                f"tables={full_num_tables}, local_tables={local_num_tables}"
            )
        _beyond_fake_quant_debug(
            f"slice_group_qtables label={label} tables={full_num_tables} "
            f"local_tables={local_num_tables} tp_rank={tp_rank} "
            f"tp_world={tp_world} start={start} end={end}",
            once_key=f"slice_group_qtables:{label}:{tp_rank}:{tp_world}",
        )
        return q_points[start:end], thresholds[start:end]

    raise ValueError(
        f"{label} per-group qtable count {full_num_tables} does not match "
        f"local H * head_dim/group_size = {local_num_tables}; "
        f"tp_rank={tp_rank}, tp_world={tp_world}. For tensor-parallel "
        "fake-quant inference, grouped qtables must be either already local "
        "or full-model tables divisible across tensor-parallel ranks."
    )


def _validate_fake_quant_per_token_config(layer) -> None:
    """Reject configs whose learned tables were not trained for token grouping."""
    if int(BEYOND_BITS) not in _FAKE_QUANT_SUPPORTED_BITS:
        raise ValueError(f"Beyond fakequant bits must be in {_FAKE_QUANT_SUPPORTED_BITS}")
    cfg = get_config_from_env()
    if cfg is None or layer is None or not hasattr(layer, "layer_name"):
        return
    strict = _env_bool("BEYOND_QUANT_CONFIG_STRICT", True)
    expected_groups = {
        "k_proj": int(BEYOND_K_GROUP_SIZE),
        "v_proj": int(BEYOND_V_GROUP_SIZE),
    }
    for proj_type, expected_group_size in expected_groups.items():
        entry = cfg.for_attention_layer(layer.layer_name, proj_type)
        if entry is None:
            if strict:
                raise ValueError(
                    f"Quant config {cfg.source_path!r} is missing {proj_type} "
                    f"for attention layer {layer.layer_name!r}"
                )
            continue
        if str(entry.grouping_dim).lower() != "token":
            raise ValueError(
                "Fake-quant per-token K/V requires token-grouped "
                f"config for {proj_type}; got grouping_dim={entry.grouping_dim!r} "
                f"at layer {layer.layer_name!r}"
            )
        if int(entry.num_bits) != int(BEYOND_BITS):
            raise ValueError(
                f"Fake-quant runtime uses {BEYOND_BITS}-bit but config has "
                f"num_bits={entry.num_bits} for {proj_type} "
                f"at layer {layer.layer_name!r}"
            )
        if strict and entry.group_size is None:
            raise ValueError(
                f"Quant config entry {layer.layer_name!r}.{proj_type} lacks group_size metadata"
            )
        if strict and str(entry.table_axis).lower() != "group":
            raise ValueError(
                f"Quant config entry {layer.layer_name!r}.{proj_type} uses "
                f"table_axis={entry.table_axis!r}; Beyond learned runs require "
                "per-group qtables"
            )
        if entry.group_size is not None and int(entry.group_size) != expected_group_size:
            raise ValueError(
                f"Quant config entry {layer.layer_name!r}.{proj_type} was trained "
                f"with group_size={entry.group_size}, but runtime uses "
                f"group_size={expected_group_size}"
            )
        if not _beyond_fake_quant_enabled():
            if entry.table_storage_dtype != TABLE_STORAGE_DTYPE_NAME:
                raise ValueError(
                    f"Real packed deployment requires {TABLE_STORAGE_DTYPE_NAME} "
                    f"q_points/thresholds, but {layer.layer_name!r}.{proj_type} "
                    f"uses {entry.table_storage_dtype!r}"
                )
            if entry.table_precision_abi != TABLE_PRECISION_ABI:
                raise ValueError(
                    f"Real packed deployment requires table_precision_abi="
                    f"{TABLE_PRECISION_ABI!r}, but {layer.layer_name!r}.{proj_type} "
                    f"uses {entry.table_precision_abi!r}"
                )


@torch.compiler.disable
def _fake_quant_token_4d(
    x: torch.Tensor,
    q_points: torch.Tensor,
    thresholds: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Training-equivalent per-token QDQ for (B, S, H, D), grouping along D.

    This function is the semantic oracle, not the production Triton fast path.
    Keep it outside TorchDynamo: Inductor rewrites the normalize/bucketize
    sequence and can move thousands of values across learned thresholds.  The
    explicit graph break preserves the eager training hard-forward contract in
    ``python-compiled`` parity runs while Triton remains compiled independently.
    """
    if x.dim() != 4:
        raise ValueError("expected tensor in (B, S, H, D) layout")
    B, S, H, D = x.shape
    G = int(group_size)
    if G <= 0:
        raise ValueError(f"group_size must be positive, got {G}")
    if D % G != 0:
        raise ValueError(f"head_dim {D} not divisible by group_size {G}")
    q_points = q_points.to(device=x.device, dtype=torch.float32)
    thresholds = thresholds.to(device=x.device, dtype=torch.float32)
    if q_points.dim() not in (1, 2):
        raise ValueError(
            "Fake-quant expects a 1-D shared qtable or 2-D per-group qtables; "
            f"got q_points={tuple(q_points.shape)}"
        )
    bits = _fake_quant_bits_from_qpoints(q_points)
    if bits is None:
        raise ValueError(
            "Fake-quant qtable levels must be a supported power of two "
            f"(2..64); got {int(q_points.shape[-1])} levels"
        )
    expected_threshold_shape = tuple(q_points.shape[:-1]) + (int(q_points.shape[-1]) - 1,)
    if tuple(thresholds.shape) != expected_threshold_shape:
        raise ValueError(
            f"thresholds shape {tuple(thresholds.shape)} does not match "
            f"expected {expected_threshold_shape}"
        )

    groups_per_head = D // G
    x_g = x.to(torch.float32).reshape(B, S, H, groups_per_head, G)
    table_x = x_g.reshape(B, S, H * groups_per_head, G)
    mn = table_x.amin(dim=-1)
    mx = table_x.amax(dim=-1)
    scale = (mx - mn).clamp_min(1e-6)
    x_norm = ((table_x - mn.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp(0.0, 1.0)
    if q_points.dim() == 1:
        codes = torch.bucketize(x_norm, thresholds, right=True).clamp(
            0,
            int(q_points.numel()) - 1,
        )
        vals = q_points[codes.to(torch.long)]
    else:
        num_tables = H * groups_per_head
        if int(q_points.shape[0]) != num_tables:
            raise ValueError(
                f"per-group qtable count {q_points.shape[0]} does not match "
                f"H * head_dim/group_size = {H} * {groups_per_head} = {num_tables}"
            )
        vals_by_table = []
        for table_idx in range(num_tables):
            codes = torch.bucketize(
                x_norm[:, :, table_idx, :].contiguous(),
                thresholds[table_idx],
                right=True,
            ).clamp_(0, int(q_points.shape[-1]) - 1)
            vals_by_table.append(q_points[table_idx][codes.to(torch.long)])
        vals = torch.stack(vals_by_table, dim=2)
    x_qdq = vals * scale.unsqueeze(-1) + mn.unsqueeze(-1)
    return x_qdq.reshape(B, S, H, groups_per_head, G).reshape(B, S, H, D).to(x.dtype)


def _fake_quant_bits_from_qpoints(q_points: torch.Tensor) -> int | None:
    if q_points.dim() not in (1, 2):
        return None
    levels = int(q_points.shape[-1])
    if levels < 2 or levels > 64 or levels & (levels - 1):
        return None
    bits = levels.bit_length() - 1
    return bits if bits in _FAKE_QUANT_SUPPORTED_BITS else None


def _fake_quant_head128_num_warps(num_tokens: int, levels: int) -> int:
    """Select the measured launch shape for the fused head-dim-128 kernel."""
    threshold_work = max(0, int(num_tokens)) * max(0, int(levels) - 1)
    return 1 if threshold_work >= 7168 else 4


if _TRITON_OK:

    @triton.jit
    def _fake_quant_token_3d_kernel(
        x,
        out,
        q_points,
        thresholds,
        H: tl.constexpr,
        D: tl.constexpr,
        x_stride_t: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_d: tl.constexpr,
        out_stride_t: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_d: tl.constexpr,
        group_size: tl.constexpr,
        NUM_Q: tl.constexpr,
        NUM_T: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        group = tl.program_id(2)
        offs = tl.arange(0, BLOCK_G)
        d = group * group_size + offs
        mask = (offs < group_size) & (d < D)
        src = x + token * x_stride_t + head * x_stride_h + d * x_stride_d
        vals = tl.load(src, mask=mask, other=0.0).to(tl.float32)
        vals_min = tl.where(mask, vals, float("inf"))
        vals_max = tl.where(mask, vals, -float("inf"))
        mn = tl.min(vals_min, axis=0)
        mx = tl.max(vals_max, axis=0)
        scale = tl.maximum(mx - mn, 1.0e-6)
        # PyTorch's CUDA division uses correctly rounded fp32 division.  The
        # ordinary Triton ``/`` may use an approximate reciprocal; values that
        # land exactly on a learned threshold can then select the adjacent
        # code.  div_rn keeps the production path aligned with the hard-QDQ
        # oracle used during training.
        xn = tl.minimum(tl.maximum(tl.div_rn(vals - mn, scale), 0.0), 1.0)
        has_range = mx > mn
        xn = tl.where(has_range & (vals == mx), 1.0, xn)
        xn = tl.where(vals == mn, 0.0, xn)
        codes = _threshold_code_tl(xn, thresholds, BLOCK_G, NUM_T=NUM_T)
        qv = tl.load(q_points + codes).to(tl.float32)
        qdq = tl_cuda_libdevice.add_rn(tl_cuda_libdevice.mul_rn(qv, scale), mn)
        dst = out + token * out_stride_t + head * out_stride_h + d * out_stride_d
        tl.store(dst, qdq, mask=mask)

    @triton.jit
    def _fake_quant_token_3d_group_kernel(
        x,
        out,
        q_points,
        thresholds,
        H: tl.constexpr,
        D: tl.constexpr,
        x_stride_t: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_d: tl.constexpr,
        out_stride_t: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_d: tl.constexpr,
        groups_per_head: tl.constexpr,
        group_size: tl.constexpr,
        NUM_Q: tl.constexpr,
        NUM_T: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        group = tl.program_id(2)
        table = head * groups_per_head + group
        offs = tl.arange(0, BLOCK_G)
        d = group * group_size + offs
        mask = (offs < group_size) & (d < D)
        src = x + token * x_stride_t + head * x_stride_h + d * x_stride_d
        vals = tl.load(src, mask=mask, other=0.0).to(tl.float32)
        vals_min = tl.where(mask, vals, float("inf"))
        vals_max = tl.where(mask, vals, -float("inf"))
        mn = tl.min(vals_min, axis=0)
        mx = tl.max(vals_max, axis=0)
        scale = tl.maximum(mx - mn, 1.0e-6)
        xn = tl.minimum(tl.maximum(tl.div_rn(vals - mn, scale), 0.0), 1.0)
        has_range = mx > mn
        xn = tl.where(has_range & (vals == mx), 1.0, xn)
        xn = tl.where(vals == mn, 0.0, xn)
        codes = _threshold_code_tl(
            xn,
            thresholds + table * NUM_T,
            BLOCK_G,
            NUM_T=NUM_T,
        )
        qv = tl.load(q_points + table * NUM_Q + codes).to(tl.float32)
        qdq = tl_cuda_libdevice.add_rn(tl_cuda_libdevice.mul_rn(qv, scale), mn)
        dst = out + token * out_stride_t + head * out_stride_h + d * out_stride_d
        tl.store(dst, qdq, mask=mask)

    @triton.jit
    def _fake_quant_kv_token_3d_kernel(
        key,
        value,
        key_out,
        value_out,
        q_points_k,
        thresholds_k,
        q_points_v,
        thresholds_v,
        H: tl.constexpr,
        D: tl.constexpr,
        key_stride_t: tl.constexpr,
        key_stride_h: tl.constexpr,
        key_stride_d: tl.constexpr,
        value_stride_t: tl.constexpr,
        value_stride_h: tl.constexpr,
        value_stride_d: tl.constexpr,
        key_out_stride_t: tl.constexpr,
        key_out_stride_h: tl.constexpr,
        key_out_stride_d: tl.constexpr,
        value_out_stride_t: tl.constexpr,
        value_out_stride_h: tl.constexpr,
        value_out_stride_d: tl.constexpr,
        group_size: tl.constexpr,
        NUM_Q: tl.constexpr,
        NUM_T: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        group = tl.program_id(2)
        offs = tl.arange(0, BLOCK_G)
        d = group * group_size + offs
        mask = (offs < group_size) & (d < D)

        src_k = key + token * key_stride_t + head * key_stride_h + d * key_stride_d
        vals_k = tl.load(src_k, mask=mask, other=0.0).to(tl.float32)
        vals_k_min = tl.where(mask, vals_k, float("inf"))
        vals_k_max = tl.where(mask, vals_k, -float("inf"))
        mn_k = tl.min(vals_k_min, axis=0)
        mx_k = tl.max(vals_k_max, axis=0)
        scale_k = tl.maximum(mx_k - mn_k, 1.0e-6)
        xn_k = tl.minimum(tl.maximum(tl.div_rn(vals_k - mn_k, scale_k), 0.0), 1.0)
        has_range_k = mx_k > mn_k
        xn_k = tl.where(has_range_k & (vals_k == mx_k), 1.0, xn_k)
        xn_k = tl.where(vals_k == mn_k, 0.0, xn_k)
        codes_k = _threshold_code_tl(
            xn_k,
            thresholds_k,
            BLOCK_G,
            NUM_T=NUM_T,
        )
        qv_k = tl.load(q_points_k + codes_k).to(tl.float32)
        qdq_k = tl_cuda_libdevice.add_rn(tl_cuda_libdevice.mul_rn(qv_k, scale_k), mn_k)
        dst_k = key_out + token * key_out_stride_t + head * key_out_stride_h + d * key_out_stride_d
        tl.store(dst_k, qdq_k, mask=mask)

        src_v = value + token * value_stride_t + head * value_stride_h + d * value_stride_d
        vals_v = tl.load(src_v, mask=mask, other=0.0).to(tl.float32)
        vals_v_min = tl.where(mask, vals_v, float("inf"))
        vals_v_max = tl.where(mask, vals_v, -float("inf"))
        mn_v = tl.min(vals_v_min, axis=0)
        mx_v = tl.max(vals_v_max, axis=0)
        scale_v = tl.maximum(mx_v - mn_v, 1.0e-6)
        xn_v = tl.minimum(tl.maximum(tl.div_rn(vals_v - mn_v, scale_v), 0.0), 1.0)
        has_range_v = mx_v > mn_v
        xn_v = tl.where(has_range_v & (vals_v == mx_v), 1.0, xn_v)
        xn_v = tl.where(vals_v == mn_v, 0.0, xn_v)
        codes_v = _threshold_code_tl(
            xn_v,
            thresholds_v,
            BLOCK_G,
            NUM_T=NUM_T,
        )
        qv_v = tl.load(q_points_v + codes_v).to(tl.float32)
        qdq_v = tl_cuda_libdevice.add_rn(tl_cuda_libdevice.mul_rn(qv_v, scale_v), mn_v)
        dst_v = (
            value_out
            + token * value_out_stride_t
            + head * value_out_stride_h
            + d * value_out_stride_d
        )
        tl.store(dst_v, qdq_v, mask=mask)

    @triton.jit
    def _fake_quant_kv_token_3d_group_kernel(
        key,
        value,
        key_out,
        value_out,
        q_points_k,
        thresholds_k,
        q_points_v,
        thresholds_v,
        H: tl.constexpr,
        D: tl.constexpr,
        key_stride_t: tl.constexpr,
        key_stride_h: tl.constexpr,
        key_stride_d: tl.constexpr,
        value_stride_t: tl.constexpr,
        value_stride_h: tl.constexpr,
        value_stride_d: tl.constexpr,
        key_out_stride_t: tl.constexpr,
        key_out_stride_h: tl.constexpr,
        key_out_stride_d: tl.constexpr,
        value_out_stride_t: tl.constexpr,
        value_out_stride_h: tl.constexpr,
        value_out_stride_d: tl.constexpr,
        groups_per_head: tl.constexpr,
        group_size: tl.constexpr,
        NUM_Q: tl.constexpr,
        NUM_T: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        group = tl.program_id(2)
        table = head * groups_per_head + group
        offs = tl.arange(0, BLOCK_G)
        d = group * group_size + offs
        mask = (offs < group_size) & (d < D)

        src_k = key + token * key_stride_t + head * key_stride_h + d * key_stride_d
        vals_k = tl.load(src_k, mask=mask, other=0.0).to(tl.float32)
        vals_k_min = tl.where(mask, vals_k, float("inf"))
        vals_k_max = tl.where(mask, vals_k, -float("inf"))
        mn_k = tl.min(vals_k_min, axis=0)
        mx_k = tl.max(vals_k_max, axis=0)
        scale_k = tl.maximum(mx_k - mn_k, 1.0e-6)
        xn_k = tl.minimum(tl.maximum(tl.div_rn(vals_k - mn_k, scale_k), 0.0), 1.0)
        has_range_k = mx_k > mn_k
        xn_k = tl.where(has_range_k & (vals_k == mx_k), 1.0, xn_k)
        xn_k = tl.where(vals_k == mn_k, 0.0, xn_k)
        codes_k = _threshold_code_tl(
            xn_k,
            thresholds_k + table * NUM_T,
            BLOCK_G,
            NUM_T=NUM_T,
        )
        qv_k = tl.load(q_points_k + table * NUM_Q + codes_k).to(tl.float32)
        qdq_k = tl_cuda_libdevice.add_rn(tl_cuda_libdevice.mul_rn(qv_k, scale_k), mn_k)
        dst_k = key_out + token * key_out_stride_t + head * key_out_stride_h + d * key_out_stride_d
        tl.store(dst_k, qdq_k, mask=mask)

        src_v = value + token * value_stride_t + head * value_stride_h + d * value_stride_d
        vals_v = tl.load(src_v, mask=mask, other=0.0).to(tl.float32)
        vals_v_min = tl.where(mask, vals_v, float("inf"))
        vals_v_max = tl.where(mask, vals_v, -float("inf"))
        mn_v = tl.min(vals_v_min, axis=0)
        mx_v = tl.max(vals_v_max, axis=0)
        scale_v = tl.maximum(mx_v - mn_v, 1.0e-6)
        xn_v = tl.minimum(tl.maximum(tl.div_rn(vals_v - mn_v, scale_v), 0.0), 1.0)
        has_range_v = mx_v > mn_v
        xn_v = tl.where(has_range_v & (vals_v == mx_v), 1.0, xn_v)
        xn_v = tl.where(vals_v == mn_v, 0.0, xn_v)
        codes_v = _threshold_code_tl(
            xn_v,
            thresholds_v + table * NUM_T,
            BLOCK_G,
            NUM_T=NUM_T,
        )
        qv_v = tl.load(q_points_v + table * NUM_Q + codes_v).to(tl.float32)
        qdq_v = tl_cuda_libdevice.add_rn(tl_cuda_libdevice.mul_rn(qv_v, scale_v), mn_v)
        dst_v = (
            value_out
            + token * value_out_stride_t
            + head * value_out_stride_h
            + d * value_out_stride_d
        )
        tl.store(dst_v, qdq_v, mask=mask)

    @triton.jit
    def _fake_quant_kv_token_3d_group_head128_kernel(
        key,
        value,
        key_out,
        value_out,
        q_points_k,
        thresholds_k,
        q_points_v,
        thresholds_v,
        key_stride_t: tl.constexpr,
        key_stride_h: tl.constexpr,
        key_stride_d: tl.constexpr,
        value_stride_t: tl.constexpr,
        value_stride_h: tl.constexpr,
        value_stride_d: tl.constexpr,
        key_out_stride_t: tl.constexpr,
        key_out_stride_h: tl.constexpr,
        key_out_stride_d: tl.constexpr,
        value_out_stride_t: tl.constexpr,
        value_out_stride_h: tl.constexpr,
        value_out_stride_d: tl.constexpr,
        NUM_Q: tl.constexpr,
        NUM_T: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        offs = tl.arange(0, BLOCK_G)

        for group in tl.static_range(0, 4):
            table = head * 4 + group
            d = group * 32 + offs

            src_k = key + token * key_stride_t + head * key_stride_h + d * key_stride_d
            vals_k = tl.load(src_k).to(tl.float32)
            mn_k = tl.min(vals_k, axis=0)
            mx_k = tl.max(vals_k, axis=0)
            scale_k = tl.maximum(mx_k - mn_k, 1.0e-6)
            xn_k = tl.minimum(tl.maximum(tl.div_rn(vals_k - mn_k, scale_k), 0.0), 1.0)
            has_range_k = mx_k > mn_k
            xn_k = tl.where(has_range_k & (vals_k == mx_k), 1.0, xn_k)
            xn_k = tl.where(vals_k == mn_k, 0.0, xn_k)
            codes_k = _threshold_code_tl(
                xn_k,
                thresholds_k + table * NUM_T,
                BLOCK_G,
                NUM_T=NUM_T,
            )
            qv_k = tl.load(q_points_k + table * NUM_Q + codes_k).to(tl.float32)
            qdq_k = tl_cuda_libdevice.add_rn(tl_cuda_libdevice.mul_rn(qv_k, scale_k), mn_k)
            dst_k = (
                key_out + token * key_out_stride_t + head * key_out_stride_h + d * key_out_stride_d
            )
            tl.store(dst_k, qdq_k)

            src_v = value + token * value_stride_t + head * value_stride_h + d * value_stride_d
            vals_v = tl.load(src_v).to(tl.float32)
            mn_v = tl.min(vals_v, axis=0)
            mx_v = tl.max(vals_v, axis=0)
            scale_v = tl.maximum(mx_v - mn_v, 1.0e-6)
            xn_v = tl.minimum(tl.maximum(tl.div_rn(vals_v - mn_v, scale_v), 0.0), 1.0)
            has_range_v = mx_v > mn_v
            xn_v = tl.where(has_range_v & (vals_v == mx_v), 1.0, xn_v)
            xn_v = tl.where(vals_v == mn_v, 0.0, xn_v)
            codes_v = _threshold_code_tl(
                xn_v,
                thresholds_v + table * NUM_T,
                BLOCK_G,
                NUM_T=NUM_T,
            )
            qv_v = tl.load(q_points_v + table * NUM_Q + codes_v).to(tl.float32)
            qdq_v = tl_cuda_libdevice.add_rn(tl_cuda_libdevice.mul_rn(qv_v, scale_v), mn_v)
            dst_v = (
                value_out
                + token * value_out_stride_t
                + head * value_out_stride_h
                + d * value_out_stride_d
            )
            tl.store(dst_v, qdq_v)

    @triton.jit
    def _fake_quant_kv_cache_group_head128_kernel(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        q_points_k,
        thresholds_k,
        q_points_v,
        thresholds_v,
        key_stride_t: tl.constexpr,
        key_stride_h: tl.constexpr,
        key_stride_d: tl.constexpr,
        value_stride_t: tl.constexpr,
        value_stride_h: tl.constexpr,
        value_stride_d: tl.constexpr,
        key_cache_stride_b: tl.constexpr,
        key_cache_stride_s: tl.constexpr,
        key_cache_stride_h: tl.constexpr,
        key_cache_stride_d: tl.constexpr,
        value_cache_stride_b: tl.constexpr,
        value_cache_stride_s: tl.constexpr,
        value_cache_stride_h: tl.constexpr,
        value_cache_stride_d: tl.constexpr,
        slot_stride: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        NUM_Q: tl.constexpr,
        NUM_T: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        """Quantize K/V and scatter directly into a dense FlashAttention cache."""
        token = tl.program_id(0)
        head = tl.program_id(1)
        slot = tl.load(slot_mapping + token * slot_stride).to(tl.int64)
        if slot < 0:
            return

        block = slot // BLOCK_SIZE
        block_offset = slot % BLOCK_SIZE
        offs = tl.arange(0, BLOCK_G)
        for group in tl.static_range(0, 4):
            table = head * 4 + group
            d = group * 32 + offs

            src_k = key + token * key_stride_t + head * key_stride_h + d * key_stride_d
            vals_k = tl.load(src_k).to(tl.float32)
            mn_k = tl.min(vals_k, axis=0)
            mx_k = tl.max(vals_k, axis=0)
            scale_k = tl.maximum(mx_k - mn_k, 1.0e-6)
            xn_k = tl.minimum(tl.maximum(tl.div_rn(vals_k - mn_k, scale_k), 0.0), 1.0)
            has_range_k = mx_k > mn_k
            xn_k = tl.where(has_range_k & (vals_k == mx_k), 1.0, xn_k)
            xn_k = tl.where(vals_k == mn_k, 0.0, xn_k)
            codes_k = _threshold_code_tl(
                xn_k,
                thresholds_k + table * NUM_T,
                BLOCK_G,
                NUM_T=NUM_T,
            )
            qv_k = tl.load(q_points_k + table * NUM_Q + codes_k).to(tl.float32)
            qdq_k = tl_cuda_libdevice.add_rn(tl_cuda_libdevice.mul_rn(qv_k, scale_k), mn_k)
            dst_k = (
                key_cache
                + block * key_cache_stride_b
                + block_offset * key_cache_stride_s
                + head * key_cache_stride_h
                + d * key_cache_stride_d
            )
            tl.store(dst_k, qdq_k)

            src_v = value + token * value_stride_t + head * value_stride_h + d * value_stride_d
            vals_v = tl.load(src_v).to(tl.float32)
            mn_v = tl.min(vals_v, axis=0)
            mx_v = tl.max(vals_v, axis=0)
            scale_v = tl.maximum(mx_v - mn_v, 1.0e-6)
            xn_v = tl.minimum(tl.maximum(tl.div_rn(vals_v - mn_v, scale_v), 0.0), 1.0)
            has_range_v = mx_v > mn_v
            xn_v = tl.where(has_range_v & (vals_v == mx_v), 1.0, xn_v)
            xn_v = tl.where(vals_v == mn_v, 0.0, xn_v)
            codes_v = _threshold_code_tl(
                xn_v,
                thresholds_v + table * NUM_T,
                BLOCK_G,
                NUM_T=NUM_T,
            )
            qv_v = tl.load(q_points_v + table * NUM_Q + codes_v).to(tl.float32)
            qdq_v = tl_cuda_libdevice.add_rn(tl_cuda_libdevice.mul_rn(qv_v, scale_v), mn_v)
            dst_v = (
                value_cache
                + block * value_cache_stride_b
                + block_offset * value_cache_stride_s
                + head * value_cache_stride_h
                + d * value_cache_stride_d
            )
            tl.store(dst_v, qdq_v)


def _fake_quant_token_3d(
    x: torch.Tensor,
    q_points: torch.Tensor,
    thresholds: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Fast per-token QDQ for token-major (T, H, D) tensors."""
    bits = _fake_quant_bits_from_qpoints(q_points)
    levels = 0 if bits is None else 1 << bits
    if (
        _TRITON_OK
        and _beyond_fake_quant_triton_enabled()
        and x.is_cuda
        and q_points.is_cuda
        and thresholds.is_cuda
        and bits is not None
        and q_points.dim() == 2
        and thresholds.dim() == 2
        and x.dim() == 3
        and int(group_size) > 0
        and int(x.shape[2]) % int(group_size) == 0
        and int(q_points.shape[1]) == levels
        and int(thresholds.shape[1]) == levels - 1
    ):
        N, H, D = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
        G = int(group_size)
        groups_per_head = D // G
        if int(q_points.shape[0]) == H * groups_per_head:
            _beyond_fake_quant_debug(
                "qdq_path implementation=triton_unfused_grouped "
                f"bits={bits} tokens={N} heads={H} head_dim={D} group_size={G}",
                once_key=(f"qdq_path:triton_unfused_grouped:{bits}:{H}:{D}:{G}"),
            )
            out = torch.empty_like(x)
            grid = (N, H, groups_per_head)
            _fake_quant_token_3d_group_kernel[grid](
                x,
                out,
                q_points.contiguous(),
                thresholds.contiguous(),
                H,
                D,
                int(x.stride(0)),
                int(x.stride(1)),
                int(x.stride(2)),
                int(out.stride(0)),
                int(out.stride(1)),
                int(out.stride(2)),
                groups_per_head,
                G,
                NUM_Q=levels,
                NUM_T=levels - 1,
                BLOCK_G=triton.next_power_of_2(G),
            )
            return out
    if (
        _TRITON_OK
        and _beyond_fake_quant_triton_enabled()
        and x.is_cuda
        and q_points.is_cuda
        and thresholds.is_cuda
        and bits is not None
        and q_points.dim() == 1
        and thresholds.dim() == 1
        and int(q_points.shape[0]) == levels
        and int(thresholds.shape[0]) == levels - 1
        and x.dim() == 3
        and int(group_size) > 0
        and int(x.shape[2]) % int(group_size) == 0
    ):
        N, H, D = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
        _beyond_fake_quant_debug(
            "qdq_path implementation=triton_unfused_shared "
            f"bits={bits} tokens={N} heads={H} head_dim={D} "
            f"group_size={int(group_size)}",
            once_key=(f"qdq_path:triton_unfused_shared:{bits}:{H}:{D}:{int(group_size)}"),
        )
        out = torch.empty_like(x)
        G = int(group_size)
        grid = (N, H, triton.cdiv(D, G))
        _fake_quant_token_3d_kernel[grid](
            x,
            out,
            q_points.contiguous(),
            thresholds.contiguous(),
            H,
            D,
            int(x.stride(0)),
            int(x.stride(1)),
            int(x.stride(2)),
            int(out.stride(0)),
            int(out.stride(1)),
            int(out.stride(2)),
            G,
            NUM_Q=levels,
            NUM_T=levels - 1,
            BLOCK_G=triton.next_power_of_2(G),
        )
        return out
    _beyond_fake_quant_debug(
        "qdq_path implementation=python_reference "
        f"bits={bits} tokens={int(x.shape[0])} heads={int(x.shape[1])} "
        f"head_dim={int(x.shape[2])} group_size={int(group_size)} "
        f"triton_requested={_beyond_fake_quant_triton_enabled()}",
        once_key=(
            "qdq_path:python_reference:"
            f"{bits}:{int(x.shape[1])}:{int(x.shape[2])}:{int(group_size)}"
        ),
    )
    return _fake_quant_token_4d(
        x.unsqueeze(0),
        q_points,
        thresholds,
        group_size,
    )[0]


def _fake_quant_kv_token_3d(
    key: torch.Tensor,
    value: torch.Tensor,
    q_points_k: torch.Tensor,
    thresholds_k: torch.Tensor,
    q_points_v: torch.Tensor,
    thresholds_v: torch.Tensor,
    group_size_k: int,
    group_size_v: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fused K/V fake-QDQ for cache-side decode and short prefill chunks."""
    if (
        not _TRITON_OK
        or not _beyond_fake_quant_triton_enabled()
        or not _beyond_fake_quant_fused_kv_enabled()
        or not key.is_cuda
        or not value.is_cuda
        or not q_points_k.is_cuda
        or not thresholds_k.is_cuda
        or not q_points_v.is_cuda
        or not thresholds_v.is_cuda
        or key.dim() != 3
        or value.dim() != 3
        or tuple(key.shape) != tuple(value.shape)
    ):
        return None
    bits_k = _fake_quant_bits_from_qpoints(q_points_k)
    bits_v = _fake_quant_bits_from_qpoints(q_points_v)
    if bits_k is None:
        return None
    if bits_v is None or bits_v != bits_k:
        return None
    levels = 1 << bits_k

    N, H, D = (int(key.shape[0]), int(key.shape[1]), int(key.shape[2]))
    G = int(group_size_k)
    if G <= 0 or int(group_size_v) != G or D % G != 0:
        return None
    max_tokens = _beyond_fake_quant_fused_kv_max_tokens()
    if max_tokens > 0 and N > max_tokens:
        return None
    groups_per_head = D // G

    key_out = torch.empty_like(key)
    value_out = torch.empty_like(value)
    grid = (N, H, groups_per_head)

    if (
        q_points_k.dim() == 2
        and thresholds_k.dim() == 2
        and q_points_v.dim() == 2
        and thresholds_v.dim() == 2
        and int(q_points_k.shape[0]) == H * groups_per_head
        and int(thresholds_k.shape[0]) == H * groups_per_head
        and int(q_points_v.shape[0]) == H * groups_per_head
        and int(thresholds_v.shape[0]) == H * groups_per_head
        and int(q_points_k.shape[1]) == levels
        and int(thresholds_k.shape[1]) == levels - 1
        and int(q_points_v.shape[1]) == levels
        and int(thresholds_v.shape[1]) == levels - 1
    ):
        if D == 128 and G == 32:
            _beyond_fake_quant_debug(
                "qdq_path implementation=triton_fused_grouped_head128 "
                f"bits={bits_k} tokens={N} heads={H} head_dim={D} "
                f"group_size={G}",
                once_key=(f"qdq_path:triton_fused_grouped_head128:{bits_k}:{H}:{G}"),
            )
            _fake_quant_kv_token_3d_group_head128_kernel[(N, H)](
                key,
                value,
                key_out,
                value_out,
                q_points_k.contiguous(),
                thresholds_k.contiguous(),
                q_points_v.contiguous(),
                thresholds_v.contiguous(),
                int(key.stride(0)),
                int(key.stride(1)),
                int(key.stride(2)),
                int(value.stride(0)),
                int(value.stride(1)),
                int(value.stride(2)),
                int(key_out.stride(0)),
                int(key_out.stride(1)),
                int(key_out.stride(2)),
                int(value_out.stride(0)),
                int(value_out.stride(1)),
                int(value_out.stride(2)),
                NUM_Q=levels,
                NUM_T=levels - 1,
                BLOCK_G=32,
                num_warps=_fake_quant_head128_num_warps(N, levels),
            )
            return key_out, value_out
        _beyond_fake_quant_debug(
            "qdq_path implementation=triton_fused_grouped "
            f"bits={bits_k} tokens={N} heads={H} head_dim={D} group_size={G}",
            once_key=(f"qdq_path:triton_fused_grouped:{bits_k}:{H}:{D}:{G}"),
        )
        _fake_quant_kv_token_3d_group_kernel[grid](
            key,
            value,
            key_out,
            value_out,
            q_points_k.contiguous(),
            thresholds_k.contiguous(),
            q_points_v.contiguous(),
            thresholds_v.contiguous(),
            H,
            D,
            int(key.stride(0)),
            int(key.stride(1)),
            int(key.stride(2)),
            int(value.stride(0)),
            int(value.stride(1)),
            int(value.stride(2)),
            int(key_out.stride(0)),
            int(key_out.stride(1)),
            int(key_out.stride(2)),
            int(value_out.stride(0)),
            int(value_out.stride(1)),
            int(value_out.stride(2)),
            groups_per_head,
            G,
            NUM_Q=levels,
            NUM_T=levels - 1,
            BLOCK_G=triton.next_power_of_2(G),
        )
        return key_out, value_out

    if (
        q_points_k.dim() == 1
        and thresholds_k.dim() == 1
        and q_points_v.dim() == 1
        and thresholds_v.dim() == 1
        and int(q_points_k.shape[0]) == levels
        and int(thresholds_k.shape[0]) == levels - 1
        and int(q_points_v.shape[0]) == levels
        and int(thresholds_v.shape[0]) == levels - 1
    ):
        _beyond_fake_quant_debug(
            "qdq_path implementation=triton_fused_shared "
            f"bits={bits_k} tokens={N} heads={H} head_dim={D} group_size={G}",
            once_key=(f"qdq_path:triton_fused_shared:{bits_k}:{H}:{D}:{G}"),
        )
        _fake_quant_kv_token_3d_kernel[grid](
            key,
            value,
            key_out,
            value_out,
            q_points_k.contiguous(),
            thresholds_k.contiguous(),
            q_points_v.contiguous(),
            thresholds_v.contiguous(),
            H,
            D,
            int(key.stride(0)),
            int(key.stride(1)),
            int(key.stride(2)),
            int(value.stride(0)),
            int(value.stride(1)),
            int(value.stride(2)),
            int(key_out.stride(0)),
            int(key_out.stride(1)),
            int(key_out.stride(2)),
            int(value_out.stride(0)),
            int(value_out.stride(1)),
            int(value_out.stride(2)),
            G,
            NUM_Q=levels,
            NUM_T=levels - 1,
            BLOCK_G=triton.next_power_of_2(G),
        )
        return key_out, value_out

    return None


def _force_cleanup_profiling_kv_cache(runner) -> None:
    cleanup = getattr(runner, "_cleanup_profiling_kv_cache", None)
    if cleanup is not None:
        try:
            cleanup()
        except Exception:
            pass
    if hasattr(runner, "kv_caches"):
        try:
            runner.kv_caches.clear()
        except Exception:
            runner.kv_caches = []
    for attr in ("cross_layers_kv_cache", "cross_layers_attn_backend"):
        if hasattr(runner, attr):
            try:
                setattr(runner, attr, None)
            except Exception:
                pass
    static_ctx = getattr(runner.compilation_config, "static_forward_context", {})
    for layer in getattr(static_ctx, "values", lambda: [])():
        if hasattr(layer, "kv_cache"):
            try:
                layer.kv_cache = []
            except Exception:
                pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def _as_positive_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None


def _device_cache_key(device: torch.device) -> str:
    if device.type == "cuda":
        idx = torch.cuda.current_device() if device.index is None else device.index
        return f"cuda:{idx}"
    return str(device)


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (int(x) - 1).bit_length()


def _beyond_workspace_batch_bucket(num_reqs: int, max_cap: int | None = None) -> int:
    bucket = _next_power_of_2(max(1, int(num_reqs)))
    if max_cap is not None and int(max_cap) > 0:
        return min(int(max_cap), bucket)
    return bucket


def _prepare_fake_qpoints_and_thresholds(
    *,
    num_bits: int,
    device: torch.device,
    q_points: torch.Tensor | None,
    thresholds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_bits = int(num_bits)
    if num_bits not in _FAKE_QUANT_SUPPORTED_BITS:
        raise ValueError(f"fakequant bits must be in {_FAKE_QUANT_SUPPORTED_BITS}, got {num_bits}")
    levels = 1 << num_bits
    if q_points is None:
        denom = levels - 1
        q_points_master = torch.tensor(
            [idx / denom for idx in range(levels)],
            dtype=torch.float32,
            device=device,
        )
        thresholds_master = (q_points_master[:-1] + q_points_master[1:]) / 2.0
        q_points = q_points_master.to(torch.float16)
        if thresholds is None:
            thresholds = thresholds_master.to(torch.float16)
    else:
        storage_dtype = (
            q_points.dtype
            if q_points.dtype in (torch.float16, torch.bfloat16)
            else torch.float32
        )
        q_points = q_points.to(device=device, dtype=storage_dtype).contiguous()
    if q_points.dim() not in (1, 2) or int(q_points.shape[-1]) != levels:
        raise ValueError(
            f"fakequant expected {levels} q_points for {num_bits}-bit, "
            f"got shape {tuple(q_points.shape)}"
        )
    if thresholds is None:
        if q_points.dim() != 1:
            raise ValueError("thresholds are required for per-group q_points")
        thresholds = ((q_points.to(torch.float32)[:-1] + q_points.to(torch.float32)[1:]) / 2.0).to(
            q_points.dtype
        )
    else:
        thresholds = thresholds.to(device=device, dtype=q_points.dtype).contiguous()
    expected_shape = tuple(q_points.shape[:-1]) + (levels - 1,)
    if tuple(thresholds.shape) != expected_shape:
        raise ValueError(
            f"fakequant thresholds shape {tuple(thresholds.shape)} does not match "
            f"expected {expected_shape}"
        )
    return q_points.contiguous(), thresholds.contiguous()


def _quant_cache_identity(cfg) -> str:
    return "uniform" if cfg is None else str(cfg.identity)


def _prime_quant_hbm_cache(device: torch.device) -> None:
    """Preload all configured q_points/thresholds to device memory.

    This runs once per worker/device so the hot attention path only grabs
    cached tensor references instead of constructing thresholds on demand.
    """
    cfg = get_config_from_env()
    # Always prime the default uniform table once so the no-config path also
    # starts with GPU-resident LUT/threshold tensors.
    config_identity = _quant_cache_identity(cfg)
    qp, thr = _prepare_fake_qpoints_and_thresholds(
        num_bits=BEYOND_BITS,
        device=device,
        q_points=None,
    )
    _HBM_QTABLE_CACHE.setdefault(
        (_device_cache_key(device), config_identity, f"uniform:{BEYOND_BITS}"),
        (qp, thr),
    )
    if cfg is None:
        return
    for layer_key, entry in cfg.items():
        if int(entry.num_bits) != int(BEYOND_BITS):
            raise ValueError(
                f"{layer_key} has num_bits={entry.num_bits}, but runtime BEYOND_BITS={BEYOND_BITS}"
            )
        cache_key = (_device_cache_key(device), config_identity, layer_key)
        if cache_key not in _HBM_QTABLE_CACHE:
            _HBM_QTABLE_CACHE[cache_key] = _prepare_fake_qpoints_and_thresholds(
                num_bits=entry.num_bits,
                device=device,
                q_points=entry.q_points,
                thresholds=entry.thresholds,
            )


def _resolve_qtables_for_layer(
    layer,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    bool,
    bool,
]:
    cfg = get_config_from_env()
    dev_key = _device_cache_key(device)
    config_identity = _quant_cache_identity(cfg)
    q_points_k = thresholds_k = q_points_v = thresholds_v = None
    uniform_qpoints_k = True
    uniform_qpoints_v = True

    if cfg is not None and layer is not None and hasattr(layer, "layer_name"):
        base = layer.layer_name
        if base.endswith(".attn"):
            base = base[: -len(".attn")]
        k_cfg = cfg.for_attention_layer(layer.layer_name, "k_proj")
        v_cfg = cfg.for_attention_layer(layer.layer_name, "v_proj")
        if k_cfg is not None:
            if int(k_cfg.num_bits) != int(BEYOND_BITS):
                raise ValueError(
                    f"{base}.k_proj has num_bits={k_cfg.num_bits}, but runtime "
                    f"BEYOND_BITS={BEYOND_BITS}"
                )
            uniform_qpoints_k = False
            cache_key = (dev_key, config_identity, f"{base}.k_proj")
            if cache_key not in _HBM_QTABLE_CACHE:
                _HBM_QTABLE_CACHE[cache_key] = _prepare_fake_qpoints_and_thresholds(
                    num_bits=k_cfg.num_bits,
                    device=device,
                    q_points=k_cfg.q_points,
                    thresholds=k_cfg.thresholds,
                )
            q_points_k, thresholds_k = _HBM_QTABLE_CACHE[cache_key]
        if v_cfg is not None:
            if int(v_cfg.num_bits) != int(BEYOND_BITS):
                raise ValueError(
                    f"{base}.v_proj has num_bits={v_cfg.num_bits}, but runtime "
                    f"BEYOND_BITS={BEYOND_BITS}"
                )
            uniform_qpoints_v = False
            cache_key = (dev_key, config_identity, f"{base}.v_proj")
            if cache_key not in _HBM_QTABLE_CACHE:
                _HBM_QTABLE_CACHE[cache_key] = _prepare_fake_qpoints_and_thresholds(
                    num_bits=v_cfg.num_bits,
                    device=device,
                    q_points=v_cfg.q_points,
                    thresholds=v_cfg.thresholds,
                )
            q_points_v, thresholds_v = _HBM_QTABLE_CACHE[cache_key]

    uniform_key = (dev_key, config_identity, f"uniform:{BEYOND_BITS}")
    uniform_entry = _HBM_QTABLE_CACHE.get(uniform_key)
    if uniform_entry is None:
        uniform_entry = _prepare_fake_qpoints_and_thresholds(
            num_bits=BEYOND_BITS,
            device=device,
            q_points=None,
        )
        _HBM_QTABLE_CACHE[uniform_key] = uniform_entry
    if q_points_k is None or thresholds_k is None:
        q_points_k, thresholds_k = uniform_entry
    if q_points_v is None or thresholds_v is None:
        q_points_v, thresholds_v = uniform_entry

    if (
        _env_int("BEYOND_FAKE_QUANT_DEBUG", 0) > 0
        and layer is not None
        and not _torch_is_compiling()
    ):
        layer_name = str(getattr(layer, "layer_name", "<unknown>"))
        debug_key = (_device_cache_key(device), layer_name)
        if debug_key not in _FAKE_QUANT_QTABLE_DEBUGGED:
            _FAKE_QUANT_QTABLE_DEBUGGED.add(debug_key)
            cfg_path = os.environ.get("BEYOND_QUANT_CONFIG") or "<uniform>"
            _beyond_fake_quant_debug(
                f"qtables layer={layer_name} "
                f"k={'uniform' if uniform_qpoints_k else 'config'} "
                f"v={'uniform' if uniform_qpoints_v else 'config'} "
                f"bits={BEYOND_BITS} config={cfg_path}",
                once_key=f"qtables:{layer_name}",
            )

    return (
        q_points_k,
        thresholds_k,
        q_points_v,
        thresholds_v,
        uniform_qpoints_k,
        uniform_qpoints_v,
    )


def _get_decode_slot_workspace(
    device: torch.device,
    slot_mapping_capacity: int,
) -> torch.Tensor:
    key = (
        _device_cache_key(device),
        slot_mapping_capacity,
    )
    workspace = _DECODE_SLOT_WORKSPACE.get(key)
    if workspace is None:
        workspace = torch.empty(
            max(1, int(slot_mapping_capacity)),
            dtype=torch.int32,
            device=device,
        )
        _DECODE_SLOT_WORKSPACE[key] = workspace
    return workspace


def _get_chunked_prefill_dense_workspace(
    device: torch.device,
    total_tokens: int,
    H_kv: int,
    D: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return a bounded reusable dense bridge for official FA4 prefill."""
    total_tokens = int(total_tokens)
    if total_tokens <= 0:
        return None
    capacity = 1 << (total_tokens - 1).bit_length()
    element_size = 2 if dtype in (torch.float16, torch.bfloat16) else 0
    required_bytes = 2 * capacity * int(H_kv) * int(D) * element_size
    cap_mb = _env_int("BEYOND_CHUNKED_PREFILL_DENSE_MAX_MB", 1024)
    if element_size == 0 or cap_mb <= 0 or required_bytes > cap_mb * 1024 * 1024:
        return None
    key = (_device_cache_key(device), int(H_kv), int(D), str(dtype))
    workspace = _CHUNKED_PREFILL_DENSE_WORKSPACE.get(key)
    if workspace is None or int(workspace["capacity"]) < capacity:
        workspace = {
            "capacity": capacity,
            "k": torch.empty((capacity, H_kv, D), device=device, dtype=dtype),
            "v": torch.empty((capacity, H_kv, D), device=device, dtype=dtype),
        }
        _CHUNKED_PREFILL_DENSE_WORKSPACE[key] = workspace
    return workspace["k"][:total_tokens], workspace["v"][:total_tokens]


def _chunked_seq_start_loc(
    attn_metadata,
    seq_lens: list[int],
    device: torch.device,
) -> torch.Tensor:
    signature = tuple(int(length) for length in seq_lens)
    cached = getattr(attn_metadata, "_beyond_chunked_seq_start_gpu_i32", None)
    if (
        cached is not None
        and getattr(
            attn_metadata,
            "_beyond_chunked_seq_start_gpu_i32_signature",
            None,
        )
        == signature
    ):
        return cached
    starts = [0]
    for length in signature:
        starts.append(starts[-1] + int(length))
    cached = torch.tensor(starts, dtype=torch.int32, device=device)
    attn_metadata._beyond_chunked_seq_start_gpu_i32 = cached
    attn_metadata._beyond_chunked_seq_start_gpu_i32_signature = signature
    return cached


if _TRITON_OK:

    @triton.jit
    def _derive_dense_decode_slot_mapping_kernel(
        seq_lens,
        block_table,
        slot_mapping_out,
        seq_lens_stride: tl.constexpr,
        block_table_stride_b: tl.constexpr,
        block_table_stride_p: tl.constexpr,
        BLOCK_TABLE_WIDTH: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Derive one decode slot per request from graph-stable metadata.

        vLLM fills dummy CUDA-graph slot mappings with ``-1``.  The packed
        decode path therefore reconstructs the current-token slot from the
        same live ``seq_lens`` and ``block_table`` tensors consumed by the
        attention kernel instead of depending on the dummy-capture value.
        """
        req = tl.program_id(0)
        seq_len = tl.load(seq_lens + req * seq_lens_stride).to(tl.int64)
        position = seq_len - 1
        logical_page = position // BLOCK_SIZE
        valid = (seq_len > 0) & (logical_page < BLOCK_TABLE_WIDTH)
        physical_page = tl.load(
            block_table
            + req * block_table_stride_b
            + logical_page * block_table_stride_p,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        slot = physical_page * BLOCK_SIZE + position % BLOCK_SIZE
        slot = tl.where(valid & (physical_page >= 0), slot, -1)
        tl.store(slot_mapping_out + req, slot.to(tl.int32))

    @triton.jit
    def _threshold_code_tl(
        xn,
        thresholds,
        BLOCK_D: tl.constexpr,
        NUM_T: tl.constexpr,
    ):
        # Config loading enforces strictly increasing thresholds.  Six-bit
        # tables otherwise execute 63 comparisons per value; an upper-bound
        # search preserves bucketize(right=True) equality semantics in six.
        if NUM_T == 63:
            lo = tl.full((BLOCK_D,), 0, dtype=tl.int32)
            hi = tl.full((BLOCK_D,), NUM_T, dtype=tl.int32)
            for _ in tl.static_range(0, 6):
                mid = (lo + hi) // 2
                th = tl.load(thresholds + mid).to(tl.float32)
                move_right = xn >= th
                lo = tl.where(move_right, mid + 1, lo)
                hi = tl.where(move_right, hi, mid)
            return lo
        codes = tl.full((BLOCK_D,), 0, dtype=tl.int32)
        for idx in tl.static_range(0, NUM_T):
            th = tl.load(thresholds + idx).to(tl.float32)
            codes += tl.where(xn >= th, 1, 0)
        return codes


def _run_decode_scatter_output_gpu(
    *,
    out_f: torch.Tensor,
    output: torch.Tensor,
    decode_q_indices: torch.Tensor,
    num_decode: int,
    H_q: int,
    D: int,
) -> None:
    if int(num_decode) <= 0:
        return
    indices = decode_q_indices[: int(num_decode)].to(
        device=output.device,
        dtype=torch.long,
    )
    output.index_copy_(0, indices, out_f[: int(num_decode)])


def _page_table_context_block_ids(
    block_table: torch.Tensor,
    req_idx: int,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    num_blocks = (int(seq_len) + int(block_size) - 1) // int(block_size)
    if num_blocks <= 0:
        return block_table[req_idx, :0].contiguous()
    return block_table[req_idx, :num_blocks].contiguous()


def _page_table_decode_attention(
    *,
    query: torch.Tensor,
    cache: torch.Tensor,
    layout: PagedLayout,
    block_table: torch.Tensor,
    q_starts: list[int],
    seq_lens: list[int],
    output: torch.Tensor,
    q_points_k: torch.Tensor | None,
    q_points_v: torch.Tensor | None,
    softmax_scale: float,
    req_indices: list[int],
    H_q: int,
    H_kv: int,
    D: int,
) -> None:
    """Reference page-table attention used by the vLLM fake-parity path."""
    repeat = int(H_q) // int(H_kv)
    for req_idx in req_indices:
        q_start = int(q_starts[req_idx])
        q_end = int(q_starts[req_idx + 1])
        q_len = q_end - q_start
        if q_len <= 0:
            continue
        seq_len = int(seq_lens[req_idx])
        if seq_len <= 0:
            output[q_start:q_end].zero_()
            continue
        block_ids = _page_table_context_block_ids(
            block_table,
            int(req_idx),
            int(seq_len),
            int(layout.block_size),
        )
        k_ctx, v_ctx = read_request_from_paged_cache(
            cache,
            layout,
            block_ids,
            int(seq_len),
            q_points_k=q_points_k,
            q_points_v=q_points_v,
            out_dtype=query.dtype,
            stat_dtype=query.dtype,
        )
        k_ctx = k_ctx[0].repeat_interleave(repeat, dim=1)
        v_ctx = v_ctx[0].repeat_interleave(repeat, dim=1)
        q_req = query[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_req = k_ctx.transpose(0, 1).unsqueeze(0)
        v_req = v_ctx.transpose(0, 1).unsqueeze(0)

        attn_mask = None
        is_causal = False
        if q_len == seq_len:
            is_causal = True
        elif q_len > 1:
            first_q_pos = seq_len - q_len
            q_pos = torch.arange(
                first_q_pos,
                seq_len,
                device=query.device,
                dtype=torch.int32,
            )
            k_pos = torch.arange(
                seq_len,
                device=query.device,
                dtype=torch.int32,
            )
            attn_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)

        out_req = F.scaled_dot_product_attention(
            q_req,
            k_req,
            v_req,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=float(softmax_scale),
        )
        output[q_start:q_end].copy_(out_req.squeeze(0).transpose(0, 1))


def _run_production_prefill_forward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    cache: torch.Tensor,
    layout: PagedLayout,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    q_starts: list[int],
    seq_lens: list[int],
    prefill_reqs: list[int],
    q_points_k: torch.Tensor | None,
    q_points_v: torch.Tensor | None,
    thresholds_k: torch.Tensor | None,
    thresholds_v: torch.Tensor | None,
    softmax_scale: float,
    causal: bool,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    fa_version: int,
    sinks: torch.Tensor | None,
) -> None:
    """Run homogeneous prefill with official FA4 over rounded K/V.

    The packed writer quantizes K and V into their independent nonuniform
    codebooks, commits the physical cache, and reconstructs the selected
    codepoints in-place. FA4 therefore consumes exactly the values represented
    by the cache rather than the raw projection outputs.
    """
    num_reqs = len(q_starts) - 1
    expected_reqs = list(range(num_reqs))
    if prefill_reqs != expected_reqs:
        raise NotImplementedError(
            "production packed prefill requires a homogeneous all-prefill batch"
        )
    if len(seq_lens) != num_reqs:
        raise ValueError("prefill sequence-length metadata is incomplete")

    query_lens = [int(q_starts[i + 1]) - int(q_starts[i]) for i in range(num_reqs)]
    if any(q_len != int(seq_lens[i]) for i, q_len in enumerate(query_lens)):
        raise NotImplementedError(
            "the rounded full-prefill helper requires each query to cover its "
            "complete sequence"
        )
    if not query_lens or max(query_lens) <= 0:
        return

    slots = slot_mapping.reshape(-1).to(device=query.device, dtype=torch.int32)
    if int(slots.numel()) != int(key.shape[0]):
        raise ValueError(
            "slot_mapping must contain exactly one entry per prefill K/V token; "
            f"got slots={int(slots.numel())}, tokens={int(key.shape[0])}"
        )
    # K/V are private attention inputs after RoPE. Reconstructing into the same
    # storage avoids a second dense workspace and makes FA4 depend directly on
    # the fused writer without a packed-cache readback pass.
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
        k_dq_out=key,
        v_dq_out=value,
        require_fast=True,
    )

    cu_seqlens = query_start_loc[: num_reqs + 1].contiguous()
    flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        out=output,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max(query_lens),
        cu_seqlens_k=cu_seqlens,
        max_seqlen_k=max(int(length) for length in seq_lens),
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        alibi_slopes=alibi_slopes,
        window_size=list(sliding_window),
        softcap=float(logits_soft_cap),
        fa_version=int(fa_version),
        s_aux=sinks,
    )

    _runtime_audit_event("rounded_prefill_attention_calls")
    _runtime_audit_event("production_prefill_calls")


def _run_production_chunked_prefill_forward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    cache: torch.Tensor,
    layout: PagedLayout,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_start_loc: torch.Tensor,
    seq_lens_gpu: torch.Tensor,
    q_starts: list[int],
    seq_lens: list[int],
    chunked_prefill_reqs: list[int],
    q_points_k: torch.Tensor | None,
    q_points_v: torch.Tensor | None,
    thresholds_k: torch.Tensor | None,
    thresholds_v: torch.Tensor | None,
    softmax_scale: float,
    causal: bool,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    fa_version: int,
    sinks: torch.Tensor | None,
) -> bool:
    """Serve a homogeneous chunk from packed cache through official FA4.

    The current chunk is first committed to the exact nonuniform cache ABI.
    A fused Triton reader then materializes one bounded dense varlen view for
    FA4, preserving the same hard-quantized K/V semantics as page-table exact.
    """
    num_reqs = len(q_starts) - 1
    if chunked_prefill_reqs != list(range(num_reqs)):
        return False
    query_lens = [int(q_starts[i + 1]) - int(q_starts[i]) for i in range(num_reqs)]
    if (
        not query_lens
        or any(length <= 1 for length in query_lens)
        or any(int(seq_lens[i]) <= query_lens[i] for i in range(num_reqs))
    ):
        return False
    total_k = sum(int(length) for length in seq_lens)
    workspace = _get_chunked_prefill_dense_workspace(
        query.device,
        total_k,
        layout.num_kv_heads,
        layout.head_dim,
        query.dtype,
    )
    if workspace is None:
        return False
    dense_k, dense_v = workspace

    slots = slot_mapping.reshape(-1).to(device=query.device, dtype=torch.int32)
    if int(slots.numel()) != int(key.shape[0]):
        raise ValueError(
            "slot_mapping must contain one entry per chunked-prefill K/V token"
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
        require_fast=True,
    )
    dequantize_paged_kv_to_dense(
        cache,
        layout,
        block_table,
        seq_lens=seq_lens_gpu,
        seq_start_loc=seq_start_loc,
        out_k=dense_k,
        out_v=dense_v,
        max_seq_len=max(int(length) for length in seq_lens),
        q_points_k=q_points_k,
        q_points_v=q_points_v,
    )
    flash_attn_varlen_func(
        q=query,
        k=dense_k,
        v=dense_v,
        out=output,
        cu_seqlens_q=query_start_loc[: num_reqs + 1].contiguous(),
        max_seqlen_q=max(query_lens),
        cu_seqlens_k=seq_start_loc,
        max_seqlen_k=max(int(length) for length in seq_lens),
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        alibi_slopes=alibi_slopes,
        window_size=list(sliding_window),
        softcap=float(logits_soft_cap),
        fa_version=int(fa_version),
        s_aux=sinks,
    )
    _runtime_audit_event("rounded_prefill_attention_calls")
    _runtime_audit_event("production_chunked_prefill_calls")
    _runtime_audit_chunked_plan(q_starts=q_starts, seq_lens=seq_lens)
    return True


def _run_page_table_exact_forward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    cache: torch.Tensor,
    layout: PagedLayout,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    q_starts: list[int],
    seq_lens: list[int],
    req_indices: list[int],
    q_points_k: torch.Tensor | None,
    q_points_v: torch.Tensor | None,
    thresholds_k: torch.Tensor | None,
    thresholds_v: torch.Tensor | None,
    softmax_scale: float,
    H_q: int,
    H_kv: int,
    D: int,
) -> None:
    if _cuda_is_capturing():
        raise RuntimeError(
            "BEYOND_VLLM_PAGE_TABLE_EXACT=1 uses dynamic page-table readback and "
            "does not support CUDA graph capture; disable vLLM graphs or set "
            "BEYOND_VLLM_PAGE_TABLE_EXACT=0 for the production SM103 path."
        )

    slots = slot_mapping.reshape(-1).to(device=query.device, dtype=torch.int32)
    if int(slots.numel()) != int(key.shape[0]):
        raise ValueError(
            "slot_mapping must contain exactly one entry per current K/V token "
            f"for exact page-table forward; got slots={int(slots.numel())}, "
            f"tokens={int(key.shape[0])}"
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

    _page_table_decode_attention(
        query=query,
        cache=cache,
        layout=layout,
        block_table=block_table,
        q_starts=q_starts,
        seq_lens=seq_lens,
        output=output,
        q_points_k=q_points_k,
        q_points_v=q_points_v,
        softmax_scale=float(softmax_scale),
        req_indices=req_indices,
        H_q=H_q,
        H_kv=H_kv,
        D=D,
    )
    _runtime_audit_event("page_table_exact_calls")


class BeyondPackedImpl(FlashAttentionImpl):
    """Minimal packed-KV attention forward.

    Piggybacks on FlashAttentionImpl for the constructor (scale, num_heads,
    etc.) but completely replaces ``forward``.
    """

    can_return_lse_for_decode: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _live_impls.add(self)
        # Preloaded during worker/impl init when possible; layer-specific
        # lookup still happens lazily on first forward because it needs the
        # concrete Attention layer name.
        self._q_points_k: Optional[torch.Tensor] = None
        self._q_points_v: Optional[torch.Tensor] = None
        self._thresholds_k: Optional[torch.Tensor] = None
        self._thresholds_v: Optional[torch.Tensor] = None
        self._uniform_qpoints_k: bool = True
        self._uniform_qpoints_v: bool = True
        self._q_points_loaded: bool = False
        self._sm103_runtime_qpoints_signature: tuple[object, ...] | None = None
        self._sm103_runtime_qpoints_k: Optional[torch.Tensor] = None
        self._sm103_runtime_qpoints_v: Optional[torch.Tensor] = None
        self._sm103_graph_debug_buffers: dict[
            tuple[object, ...], dict[str, torch.Tensor]
        ] = {}
        self._sm103_graph_debug_context: dict[tuple[object, ...], dict[str, object]] = {}
        if torch.cuda.is_available():
            _prime_quant_hbm_cache(torch.device("cuda", torch.cuda.current_device()))

    def _ensure_q_points(self, layer, device: torch.device) -> None:
        """Resolve trained per-layer q_points (if ``BEYOND_QUANT_CONFIG`` set)
        and cache as GPU tensors on this impl. Idempotent. Robust to
        instances built via ``__new__`` (unit tests) that skip __init__."""
        if getattr(self, "_q_points_loaded", False):
            return
        (
            self._q_points_k,
            self._thresholds_k,
            self._q_points_v,
            self._thresholds_v,
            self._uniform_qpoints_k,
            self._uniform_qpoints_v,
        ) = _resolve_qtables_for_layer(layer, device)
        self._q_points_loaded = True

    def _ensure_tp_local_q_points(
        self,
        layer,
        device: torch.device,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Resolve and retain the exact TP-local K/V nonuniform tables."""
        self._ensure_q_points(layer, device)
        q_points_k = self._q_points_k
        q_points_v = self._q_points_v
        thresholds_k = self._thresholds_k
        thresholds_v = self._thresholds_v
        layer_name = str(getattr(layer, "layer_name", "<unknown>"))
        if q_points_k is not None and q_points_k.dim() == 2:
            q_points_k, thresholds_k = _slice_group_qtables_for_tensor_parallel(
                q_points_k,
                thresholds_k,
                int(self.num_kv_heads) * (int(self.head_size) // BEYOND_K_GROUP_SIZE),
                label=f"{layer_name}.k_proj",
            )
            self._q_points_k = q_points_k
            self._thresholds_k = thresholds_k
        if q_points_v is not None and q_points_v.dim() == 2:
            q_points_v, thresholds_v = _slice_group_qtables_for_tensor_parallel(
                q_points_v,
                thresholds_v,
                int(self.num_kv_heads) * (int(self.head_size) // BEYOND_V_GROUP_SIZE),
                label=f"{layer_name}.v_proj",
            )
            self._q_points_v = q_points_v
            self._thresholds_v = thresholds_v
        return q_points_k, thresholds_k, q_points_v, thresholds_v

    def _ensure_sm103_runtime_qpoints(
        self,
        q_points_k: torch.Tensor,
        q_points_v: torch.Tensor,
        *,
        heads_kv: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signature = (
            int(q_points_k.data_ptr()),
            tuple(q_points_k.shape),
            q_points_k.dtype,
            int(q_points_v.data_ptr()),
            tuple(q_points_v.shape),
            q_points_v.dtype,
            int(heads_kv),
        )
        if (
            getattr(self, "_sm103_runtime_qpoints_signature", None) == signature
            and getattr(self, "_sm103_runtime_qpoints_k", None) is not None
            and getattr(self, "_sm103_runtime_qpoints_v", None) is not None
        ):
            return self._sm103_runtime_qpoints_k, self._sm103_runtime_qpoints_v
        if _cuda_is_capturing():
            raise RuntimeError(
                "SM103 K/V runtime qpoints were not prepared before CUDA Graph capture"
            )
        from quant.fa4_cute.beyond_mixed_decode_runtime import (
            prepare_runtime_qpoints,
        )

        runtime_k, runtime_v = prepare_runtime_qpoints(
            q_points_k,
            q_points_v,
            heads_kv=int(heads_kv),
            storage_dtype=torch.float16,
        )
        self._sm103_runtime_qpoints_signature = signature
        self._sm103_runtime_qpoints_k = runtime_k
        self._sm103_runtime_qpoints_v = runtime_v
        return runtime_k, runtime_v

    def _retain_sm103_cudagraph_buffers(self, *tensors: torch.Tensor) -> None:
        """Keep direct-CuTe capture addresses alive for the graph lifetime.

        This backend launches CuTe directly from Python, outside a torch custom
        op. CUDA Graph records the raw addresses, but torch's graph wrapper does
        not discover these Tensor objects as inputs or outputs. Retaining one
        tiny decode buffer set per captured shape prevents allocator reuse.
        """
        if not _cuda_is_capturing():
            return
        signature = tuple(int(tensor.data_ptr()) for tensor in tensors)
        keepalive = getattr(self, "_sm103_cudagraph_keepalive", None)
        if keepalive is None:
            keepalive = {}
            self._sm103_cudagraph_keepalive = keepalive
        keepalive.setdefault(signature, tensors)

    def _get_sm103_graph_debug_buffers(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[tuple[object, ...], dict[str, torch.Tensor]]:
        """Return per-impl, per-graph buffers that preserve the last replay."""
        signature = (
            int(query.shape[0]),
            int(block_table.shape[1]),
            tuple(query.shape),
            tuple(key.shape),
            str(query.dtype),
        )
        buffers_by_signature = getattr(self, "_sm103_graph_debug_buffers", None)
        if buffers_by_signature is None:
            buffers_by_signature = {}
            self._sm103_graph_debug_buffers = buffers_by_signature
        buffers = buffers_by_signature.get(signature)
        if buffers is not None:
            return signature, buffers
        if _cuda_is_capturing():
            raise RuntimeError(
                "SM103 graph debug buffers were not prepared before CUDA Graph capture"
            )
        buffers = {
            "query": torch.empty_like(query),
            "key": torch.empty_like(key),
            "value": torch.empty_like(value),
            "output": torch.empty_like(output),
            "seq_lens": torch.empty(
                int(query.shape[0]), dtype=torch.int32, device=query.device
            ),
            "slot_mapping": torch.empty(
                int(query.shape[0]), dtype=torch.int32, device=query.device
            ),
            "block_table": torch.empty(
                (int(query.shape[0]), int(block_table.shape[1])),
                dtype=block_table.dtype,
                device=block_table.device,
            ),
        }
        buffers_by_signature[signature] = buffers
        return signature, buffers

    def _run_sm103_nonuniform_dense_decode(
        self,
        *,
        debug_label: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        cache: torch.Tensor,
        layout: PagedLayout,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
        q_points_k: torch.Tensor,
        q_points_v: torch.Tensor,
        thresholds_k: torch.Tensor,
        thresholds_v: torch.Tensor,
        max_past_len: int,
    ) -> torch.Tensor:
        """Write the new token and launch one exact-page SM103 decode bucket."""
        from quant.fa4_cute.beyond_mixed_decode_runtime import (
            allocate_runtime_workspace,
            make_runtime_spec,
            run_nonuniform_decode,
        )

        batch_size = int(query.shape[0])
        if int(block_table.shape[0]) < batch_size:
            raise ValueError("dense decode block_table is shorter than the query batch")
        table_capacity = int(block_table.shape[1]) * int(layout.block_size)
        # A vLLM FULL decode graph is keyed by batch size, not by sequence
        # length.  Its dummy capture metadata commonly reports one 128-token
        # page, while later replays may address the entire block table.  Freeze
        # graph kernels at the full table capacity so replay cannot silently
        # truncate longer contexts.  Eager execution retains smaller JIT
        # buckets for latency.
        max_seq_len = _sm103_decode_compile_bucket(
            int(max_past_len) + 1,
            table_capacity,
            cuda_graph_enabled=BEYOND_ENABLE_CUDAGRAPH_DECODE,
            block_size=int(layout.block_size),
        )
        runtime_k, runtime_v = self._ensure_sm103_runtime_qpoints(
            q_points_k,
            q_points_v,
            heads_kv=int(layout.num_kv_heads),
        )
        graph_debug_key = None
        graph_debug = None
        if _env_bool("BEYOND_SM103_GRAPH_DEBUG", False):
            graph_debug_key, graph_debug = self._get_sm103_graph_debug_buffers(
                query,
                key,
                value,
                output,
                block_table,
            )
            graph_debug["query"].copy_(query)
            graph_debug["key"].copy_(key)
            graph_debug["value"].copy_(value)
            graph_debug["seq_lens"].copy_(seq_lens[:batch_size])
            graph_debug["slot_mapping"].copy_(slot_mapping[:batch_size])
            graph_debug["block_table"].copy_(block_table[:batch_size])
        # The attention kernel consumes the same physical packed allocation;
        # only the current token must be quantized/scattered before it reads.
        writer_pdl = _env_bool("BEYOND_SM103_WRITER_PDL", True)
        write_tokens_to_paged_cache(
            cache,
            layout,
            block_table,
            slot_mapping[:batch_size],
            key[:batch_size],
            value[:batch_size],
            q_points_k=q_points_k,
            q_points_v=q_points_v,
            thresholds_k=thresholds_k,
            thresholds_v=thresholds_v,
            signal_pdl_dependents=writer_pdl,
            require_fast=True,
        )
        spec = make_runtime_spec(
            q=query,
            cache=cache,
            layout=layout,
            block_table=block_table,
            heads_q=int(self.num_heads),
            max_seq_len=max_seq_len,
            num_splits=0,
            qpoint_dtype=runtime_k.dtype,
            wait_for_pdl_writer=writer_pdl,
        )
        workspace = _get_sm103_nonuniform_workspace(spec, query.device)
        run_nonuniform_decode(
            spec=spec,
            q=query,
            cache=cache,
            layout=layout,
            block_table=block_table,
            seq_lens=seq_lens,
            k_qpoints_lane_major=runtime_k,
            v_qpoints_code_major=runtime_v,
            out=output,
            workspace=workspace,
            softmax_scale=float(self.scale),
        )
        if graph_debug is not None:
            graph_debug["output"].copy_(output)
            context_by_signature = getattr(
                self, "_sm103_graph_debug_context", None
            )
            if context_by_signature is None:
                context_by_signature = {}
                self._sm103_graph_debug_context = context_by_signature
            context_by_signature[graph_debug_key] = {
                "label": debug_label,
                "cache": cache,
                "layout": layout,
                "runtime_k": runtime_k,
                "runtime_v": runtime_v,
                "max_seq_len": int(spec.max_seq_len),
                "num_splits": int(spec.num_splits),
                "softmax_scale": float(self.scale),
            }
        if _env_bool("BEYOND_SM103_FINITE_GUARD", False) and not _cuda_is_capturing():
            torch.cuda.current_stream(query.device).synchronize()
            if not bool(torch.isfinite(output).all().item()):
                lengths = [int(x) for x in seq_lens[:batch_size].cpu().tolist()]
                page_size = int(layout.block_size)
                max_pages = max(
                    (length + page_size - 1) // page_size for length in lengths
                )
                compact_replay = None
                if _env_bool("BEYOND_SM103_COMPACT_REPLAY", False):
                    used_ids = torch.cat(
                        [
                            block_table[
                                batch_idx,
                                : (length + page_size - 1) // page_size,
                            ]
                            for batch_idx, length in enumerate(lengths)
                        ]
                    )
                    unique_ids = torch.unique(used_ids, sorted=True)
                    compact_cache = cache.index_select(0, unique_ids.to(torch.long))
                    id_map = {
                        int(block_id): compact_id
                        for compact_id, block_id in enumerate(unique_ids.cpu().tolist())
                    }
                    compact_table_cpu = torch.full(
                        (batch_size, int(block_table.shape[1])),
                        -1,
                        dtype=torch.int32,
                    )
                    table_cpu = block_table[:batch_size, :max_pages].cpu()
                    for batch_idx, length in enumerate(lengths):
                        for page_idx in range(
                            (length + page_size - 1) // page_size
                        ):
                            compact_table_cpu[batch_idx, page_idx] = id_map[
                                int(table_cpu[batch_idx, page_idx])
                            ]
                    compact_table = compact_table_cpu.to(query.device)
                    compact_spec = make_runtime_spec(
                        q=query,
                        cache=compact_cache,
                        layout=layout,
                        block_table=compact_table,
                        heads_q=int(self.num_heads),
                        max_seq_len=spec.max_seq_len,
                        num_splits=spec.num_splits,
                        qpoint_dtype=runtime_k.dtype,
                    )
                    compact_workspace = allocate_runtime_workspace(
                        compact_spec,
                        device=query.device,
                    )
                    compact_out = torch.empty_like(output)
                    run_nonuniform_decode(
                        spec=compact_spec,
                        q=query,
                        cache=compact_cache,
                        layout=layout,
                        block_table=compact_table,
                        seq_lens=seq_lens,
                        k_qpoints_lane_major=runtime_k,
                        v_qpoints_code_major=runtime_v,
                        out=compact_out,
                        workspace=compact_workspace,
                        softmax_scale=float(self.scale),
                    )
                    torch.cuda.current_stream(query.device).synchronize()
                    compact_replay = {
                        "physical_pages": int(compact_cache.shape[0]),
                        "output_finite": bool(
                            torch.isfinite(compact_out).all().item()
                        ),
                        "output_nonfinite": int(
                            (~torch.isfinite(compact_out)).sum().item()
                        ),
                    }
                    dump_path = os.environ.get("BEYOND_SM103_DUMP_PATH", "").strip()
                    if dump_path:
                        torch.save(
                            {
                                "query": query.detach().cpu(),
                                "cache": compact_cache.detach().cpu(),
                                "block_table": compact_table.detach().cpu(),
                                "seq_lens": seq_lens[:batch_size].detach().cpu(),
                                "k_qpoints_lane_major": runtime_k.detach().cpu(),
                                "v_qpoints_code_major": runtime_v.detach().cpu(),
                                "layout": layout,
                                "heads_q": int(self.num_heads),
                                "max_seq_len": int(spec.max_seq_len),
                                "num_splits": int(spec.num_splits),
                                "softmax_scale": float(self.scale),
                            },
                            dump_path,
                        )
                        compact_replay["dump_path"] = dump_path
                active_workspace = []
                cache_readback = []
                for batch_idx, length in enumerate(lengths):
                    active_splits = min(
                        spec.num_splits,
                        (length + page_size - 1) // page_size,
                    )
                    block_ids = block_table[
                        batch_idx, : (length + page_size - 1) // page_size
                    ]
                    dense_k, dense_v = read_request_from_paged_cache(
                        cache,
                        layout,
                        block_ids,
                        length,
                        q_points_k=q_points_k,
                        q_points_v=q_points_v,
                        out_dtype=query.dtype,
                        stat_dtype=query.dtype,
                    )
                    cache_readback.append(
                        {
                            "batch": batch_idx,
                            "k_finite": bool(torch.isfinite(dense_k).all().item()),
                            "v_finite": bool(torch.isfinite(dense_v).all().item()),
                            "k_max_abs": float(dense_k.float().abs().max().item()),
                            "v_max_abs": float(dense_v.float().abs().max().item()),
                        }
                    )
                    active_workspace.append(
                        {
                            "batch": batch_idx,
                            "splits": active_splits,
                            "o_finite": bool(
                                torch.isfinite(
                                    workspace["o_partial"][:active_splits, batch_idx]
                                ).all().item()
                            ),
                            "m_finite": bool(
                                torch.isfinite(
                                    workspace["m_partial"][:active_splits, batch_idx]
                                ).all().item()
                            ),
                            "l_finite": bool(
                                torch.isfinite(
                                    workspace["l_partial"][:active_splits, batch_idx]
                                ).all().item()
                            ),
                        }
                    )
                raise RuntimeError(
                    "SM103 non-uniform decode produced non-finite output: "
                    + json.dumps(
                        {
                            "layer": debug_label,
                            "spec": {
                                "batch": spec.batch_size,
                                "max_seq_len": spec.max_seq_len,
                                "num_splits": spec.num_splits,
                                "reduction": spec.reduction_kind,
                                "physical_pages": spec.physical_pages,
                                "page_table_stride": spec.page_table_stride,
                                "q_batch_stride": spec.q_batch_stride,
                            },
                            "seq_lens": lengths,
                            "block_table": block_table[
                                :batch_size, :max_pages
                            ].cpu().tolist(),
                            "slot_mapping": slot_mapping[:batch_size].cpu().tolist(),
                            "input_finite": {
                                "q": bool(torch.isfinite(query).all().item()),
                                "k": bool(torch.isfinite(key).all().item()),
                                "v": bool(torch.isfinite(value).all().item()),
                                "qpoints_k": bool(
                                    torch.isfinite(runtime_k).all().item()
                                ),
                                "qpoints_v": bool(
                                    torch.isfinite(runtime_v).all().item()
                                ),
                            },
                            "active_workspace": active_workspace,
                            "cache_readback": cache_readback,
                            "compact_replay": compact_replay,
                            "output_nonfinite": int(
                                (~torch.isfinite(output)).sum().item()
                            ),
                        },
                        sort_keys=True,
                    )
                )
        _runtime_audit_event("sm103_nonuniform_decode_launches")
        return output

    def _try_run_split_mixed_forward(
        self,
        *,
        layer_name: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        cache: torch.Tensor,
        layout: PagedLayout,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_gpu: torch.Tensor,
        q_starts: list[int],
        seq_lens: list[int],
        prefill_reqs: list[int],
        chunked_prefill_reqs: list[int],
        decode_reqs: list[int],
        attn_metadata,
        q_points_k: torch.Tensor | None,
        q_points_v: torch.Tensor | None,
        thresholds_k: torch.Tensor | None,
        thresholds_v: torch.Tensor | None,
        exact_q_points_k: torch.Tensor | None,
        exact_q_points_v: torch.Tensor | None,
        exact_thresholds_k: torch.Tensor | None,
        exact_thresholds_v: torch.Tensor | None,
    ) -> bool:
        """Split vLLM's common decode/chunked/full-prefill request order.

        V1 schedules running decodes first, followed by chunked prefills and new
        prefills.  Those request groups occupy contiguous token ranges, so each
        production kernel can consume a view without gather/scatter buffers.
        This removes the dynamic page-table oracle from normal mixed serving.
        """
        num_reqs = len(q_starts) - 1
        ordered_reqs = decode_reqs + chunked_prefill_reqs + prefill_reqs
        if ordered_reqs != list(range(num_reqs)):
            return False
        if not chunked_prefill_reqs and not (prefill_reqs and decode_reqs):
            return False
        if int(q_starts[-1]) != int(query.shape[0]):
            return False
        if not block_table.is_contiguous():
            return False

        num_decode = len(decode_reqs)
        if decode_reqs:
            grouped_heads = int(self.num_heads) // int(self.num_kv_heads)
            if (
                any(int(q_starts[i + 1]) - int(q_starts[i]) != 1 for i in decode_reqs)
                or not _sm103_nonuniform_decode_requested(query.device)
                or int(self.head_size) != 128
                or grouped_heads not in (4, 8)
                or int(layout.block_size) not in BEYOND_PACKED_BLOCK_SIZES
                or layout.code_layout != CODE_LAYOUT_HND_TOKEN_WORD
                or q_points_k is None
                or q_points_v is None
                or thresholds_k is None
                or thresholds_v is None
                or q_points_k.dim() != 2
                or q_points_v.dim() != 2
            ):
                return False

        starts_cache = getattr(
            attn_metadata,
            "_beyond_split_query_start_loc_gpu_i32",
            None,
        )
        if starts_cache is None:
            starts_cache = {}
            attn_metadata._beyond_split_query_start_loc_gpu_i32 = starts_cache

        def group_views(reqs: list[int]):
            req_start = int(reqs[0])
            req_end = int(reqs[-1]) + 1
            if reqs != list(range(req_start, req_end)):
                raise RuntimeError("Beyond split mixed groups must be contiguous")
            token_start = int(q_starts[req_start])
            token_end = int(q_starts[req_end])
            relative_q_starts = tuple(
                int(q_starts[i]) - token_start for i in range(req_start, req_end + 1)
            )
            cache_key = (req_start, req_end, token_start, relative_q_starts)
            relative_gpu = starts_cache.get(cache_key)
            if relative_gpu is None:
                relative_gpu = torch.tensor(
                    relative_q_starts,
                    dtype=torch.int32,
                    device=query.device,
                )
                starts_cache[cache_key] = relative_gpu
            return (
                req_start,
                req_end,
                token_start,
                token_end,
                list(relative_q_starts),
                relative_gpu,
            )

        # Chunked-prefill can decline if its bounded dense bridge is too large;
        # run it first so a fallback never follows partially computed full prefill.
        if chunked_prefill_reqs:
            (
                req_start,
                req_end,
                token_start,
                token_end,
                relative_q_starts,
                relative_gpu,
            ) = group_views(chunked_prefill_reqs)
            chunk_seq_lens = [int(x) for x in seq_lens[req_start:req_end]]
            seq_start_loc = _chunked_seq_start_loc(
                attn_metadata,
                chunk_seq_lens,
                query.device,
            )
            if not _run_production_chunked_prefill_forward(
                query=query[token_start:token_end],
                key=key[token_start:token_end],
                value=value[token_start:token_end],
                output=output[token_start:token_end],
                cache=cache,
                layout=layout,
                block_table=block_table[req_start:req_end],
                slot_mapping=slot_mapping[token_start:token_end],
                query_start_loc=relative_gpu,
                seq_start_loc=seq_start_loc,
                seq_lens_gpu=seq_lens_gpu[req_start:req_end],
                q_starts=relative_q_starts,
                seq_lens=chunk_seq_lens,
                chunked_prefill_reqs=list(range(req_end - req_start)),
                q_points_k=exact_q_points_k,
                q_points_v=exact_q_points_v,
                thresholds_k=exact_thresholds_k,
                thresholds_v=exact_thresholds_v,
                softmax_scale=self.scale,
                causal=bool(getattr(attn_metadata, "causal", True)),
                alibi_slopes=self.alibi_slopes,
                sliding_window=self.sliding_window,
                logits_soft_cap=self.logits_soft_cap,
                fa_version=int(self.vllm_flash_attn_version),
                sinks=self.sinks,
            ):
                return False

        if prefill_reqs:
            (
                req_start,
                req_end,
                token_start,
                token_end,
                relative_q_starts,
                relative_gpu,
            ) = group_views(prefill_reqs)
            _run_production_prefill_forward(
                query=query[token_start:token_end],
                key=key[token_start:token_end],
                value=value[token_start:token_end],
                output=output[token_start:token_end],
                cache=cache,
                layout=layout,
                block_table=block_table[req_start:req_end],
                slot_mapping=slot_mapping[token_start:token_end],
                query_start_loc=relative_gpu,
                q_starts=relative_q_starts,
                seq_lens=[int(x) for x in seq_lens[req_start:req_end]],
                prefill_reqs=list(range(req_end - req_start)),
                q_points_k=exact_q_points_k,
                q_points_v=exact_q_points_v,
                thresholds_k=exact_thresholds_k,
                thresholds_v=exact_thresholds_v,
                softmax_scale=self.scale,
                causal=bool(getattr(attn_metadata, "causal", True)),
                alibi_slopes=self.alibi_slopes,
                sliding_window=self.sliding_window,
                logits_soft_cap=self.logits_soft_cap,
                fa_version=int(self.vllm_flash_attn_version),
                sinks=self.sinks,
            )

        if decode_reqs:
            self._run_sm103_nonuniform_dense_decode(
                debug_label=layer_name,
                query=query[:num_decode],
                key=key[:num_decode],
                value=value[:num_decode],
                output=output[:num_decode],
                cache=cache,
                layout=layout,
                block_table=block_table[:num_decode],
                seq_lens=seq_lens_gpu[:num_decode],
                slot_mapping=slot_mapping[:num_decode],
                q_points_k=q_points_k,
                q_points_v=q_points_v,
                thresholds_k=thresholds_k,
                thresholds_v=thresholds_v,
                max_past_len=max(int(x) for x in seq_lens[:num_decode]) - 1,
            )

        _runtime_audit_event("mixed_prefill_decode_split_calls")
        _runtime_audit_mixed_plan(
            prefill_reqs=prefill_reqs,
            chunked_prefill_reqs=chunked_prefill_reqs,
            decode_reqs=decode_reqs,
            q_starts=q_starts,
            seq_lens=seq_lens,
        )
        return True

    def forward(
        self,
        layer,
        query: torch.Tensor,  # (num_tokens, num_heads, head_size)
        key: torch.Tensor,  # (num_tokens, num_kv_heads, head_size)
        value: torch.Tensor,  # (num_tokens, num_kv_heads, head_size)
        kv_cache: torch.Tensor,  # (num_blocks, block_i32) int32
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Packed-KV forward for the page-table real-compression path."""
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "BeyondPacked backend does not support fused output quantization."
            )
        if attn_metadata is None:
            if output is None:
                return torch.zeros_like(query)
            return output.fill_(0)

        num_actual = attn_metadata.num_actual_tokens
        q = query[:num_actual]
        k = key[:num_actual]
        v = value[:num_actual]
        device = q.device
        if (
            q.dtype not in (torch.float16, torch.bfloat16)
            or k.dtype != q.dtype
            or v.dtype != q.dtype
        ):
            raise RuntimeError(
                "BeyondPacked CUSTOM backend supports fp16/bf16 Q/K/V "
                "tensors with a shared dtype; "
                f"got q={q.dtype}, k={k.dtype}, v={v.dtype}."
            )
        H_q = self.num_heads
        H_kv = self.num_kv_heads
        D = self.head_size
        bits = BEYOND_BITS
        G_k = BEYOND_K_GROUP_SIZE
        G_v = BEYOND_V_GROUP_SIZE
        _validate_real_packed_layout(bits, G_k, G_v)
        _validate_fake_quant_per_token_config(layer)
        (
            q_points_k,
            thresholds_k,
            q_points_v,
            thresholds_v,
        ) = self._ensure_tp_local_q_points(layer, device)
        layer_name = str(getattr(layer, "layer_name", "<unknown>"))
        if H_q % H_kv != 0:
            raise ValueError(f"H_q={H_q} not multiple of H_kv={H_kv}")
        if H_q // H_kv > 8:
            raise NotImplementedError("packed path requires H_q / H_kv <= 8")

        block_size = _infer_block_size(kv_cache, H_kv, D)
        layout = _paged_layout(block_size, H_kv, D)
        if kv_cache.shape[1] != layout.block_i32:
            raise RuntimeError(f"kv_cache block stride {kv_cache.shape[1]} != layout.block_i32")

        q_starts = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        slot_mapping = attn_metadata.slot_mapping[:num_actual]
        num_reqs = q_starts.shape[0] - 1

        if output is None:
            output = torch.empty(
                (num_actual, H_q, D),
                dtype=q.dtype,
                device=device,
            )

        q_starts_cpu = getattr(attn_metadata, "_beyond_query_start_loc_list", None)
        if q_starts_cpu is None:
            q_starts_cpu_t = getattr(
                attn_metadata,
                "_beyond_query_start_loc_cpu",
                None,
            )
            if q_starts_cpu_t is None:
                if _cuda_is_capturing():
                    raise RuntimeError(
                        "Beyond cudagraph requires metadata-builder "
                        "query_start_loc CPU mirror/list."
                    )
                q_starts_cpu_t = q_starts.cpu()
            q_starts_cpu = q_starts_cpu_t[: num_reqs + 1].tolist()
            attn_metadata._beyond_query_start_loc_list = q_starts_cpu
        seq_lens_cpu = getattr(attn_metadata, "_beyond_seq_lens_list", None)
        if seq_lens_cpu is None:
            seq_lens_cpu_t = getattr(attn_metadata, "_beyond_seq_lens_cpu", None)
            if seq_lens_cpu_t is None:
                if _cuda_is_capturing():
                    raise RuntimeError(
                        "Beyond cudagraph requires metadata-builder seq_lens CPU mirror/list."
                    )
                seq_lens_cpu_t = seq_lens.cpu()
            seq_lens_cpu = seq_lens_cpu_t[:num_reqs].tolist()
            attn_metadata._beyond_seq_lens_list = seq_lens_cpu
        _ensure_beyond_plan(
            attn_metadata,
            num_reqs=num_reqs,
            num_actual_tokens=num_actual,
            q_starts=q_starts_cpu,
            seq_lens=seq_lens_cpu,
        )

        prefill_reqs: list[int] = list(getattr(attn_metadata, "_beyond_prefill_req_list", []))
        chunked_prefill_reqs: list[int] = list(
            getattr(attn_metadata, "_beyond_chunked_prefill_req_list", [])
        )
        decode_reqs: list[int] = list(getattr(attn_metadata, "_beyond_decode_req_list", []))
        single_token_decode_batch = bool(
            getattr(attn_metadata, "_beyond_single_token_decode_batch", False)
        )

        active_reqs = prefill_reqs + chunked_prefill_reqs + decode_reqs
        if not active_reqs:
            return output

        exact_q_points_k = None if self._uniform_qpoints_k else q_points_k
        exact_q_points_v = None if self._uniform_qpoints_v else q_points_v
        exact_thresholds_k = None if self._uniform_qpoints_k else thresholds_k
        exact_thresholds_v = None if self._uniform_qpoints_v else thresholds_v
        if _beyond_vllm_page_table_exact_enabled():
            _run_page_table_exact_forward(
                query=q,
                key=k,
                value=v,
                output=output,
                cache=kv_cache,
                layout=layout,
                block_table=block_table,
                slot_mapping=slot_mapping,
                q_starts=q_starts_cpu,
                seq_lens=seq_lens_cpu,
                req_indices=active_reqs,
                q_points_k=exact_q_points_k,
                q_points_v=exact_q_points_v,
                thresholds_k=exact_thresholds_k,
                thresholds_v=exact_thresholds_v,
                softmax_scale=self.scale,
                H_q=H_q,
                H_kv=H_kv,
                D=D,
            )
            return output

        # Homogeneous chunked prefill uses the bounded packed-cache bridge.
        # Mixed batches split into production prefill and SM103 decode groups.
        # Unsupported scheduler shapes fail closed instead of silently running
        # the numerical oracle.
        if chunked_prefill_reqs and not prefill_reqs and not decode_reqs:
            if self.vllm_flash_attn_version is None:
                raise RuntimeError("vLLM FlashAttention version was not initialized")
            seq_start_loc = _chunked_seq_start_loc(
                attn_metadata,
                seq_lens_cpu,
                device,
            )
            if _run_production_chunked_prefill_forward(
                query=q,
                key=k,
                value=v,
                output=output[:num_actual],
                cache=kv_cache,
                layout=layout,
                block_table=block_table,
                slot_mapping=slot_mapping,
                query_start_loc=q_starts,
                seq_start_loc=seq_start_loc,
                seq_lens_gpu=seq_lens,
                q_starts=q_starts_cpu,
                seq_lens=seq_lens_cpu,
                chunked_prefill_reqs=chunked_prefill_reqs,
                q_points_k=exact_q_points_k,
                q_points_v=exact_q_points_v,
                thresholds_k=exact_thresholds_k,
                thresholds_v=exact_thresholds_v,
                softmax_scale=self.scale,
                causal=bool(getattr(attn_metadata, "causal", True)),
                alibi_slopes=self.alibi_slopes,
                sliding_window=self.sliding_window,
                logits_soft_cap=self.logits_soft_cap,
                fa_version=self.vllm_flash_attn_version,
                sinks=self.sinks,
            ):
                return output
        if chunked_prefill_reqs or (prefill_reqs and decode_reqs):
            if self.vllm_flash_attn_version is None:
                raise RuntimeError("vLLM FlashAttention version was not initialized")
            if self._try_run_split_mixed_forward(
                layer_name=layer_name,
                query=q,
                key=k,
                value=v,
                output=output[:num_actual],
                cache=kv_cache,
                layout=layout,
                block_table=block_table,
                slot_mapping=slot_mapping,
                query_start_loc=q_starts,
                seq_lens_gpu=seq_lens,
                q_starts=q_starts_cpu,
                seq_lens=seq_lens_cpu,
                prefill_reqs=prefill_reqs,
                chunked_prefill_reqs=chunked_prefill_reqs,
                decode_reqs=decode_reqs,
                attn_metadata=attn_metadata,
                q_points_k=q_points_k,
                q_points_v=q_points_v,
                thresholds_k=thresholds_k,
                thresholds_v=thresholds_v,
                exact_q_points_k=exact_q_points_k,
                exact_q_points_v=exact_q_points_v,
                exact_thresholds_k=exact_thresholds_k,
                exact_thresholds_v=exact_thresholds_v,
            ):
                return output
            _runtime_audit_mixed_plan(
                prefill_reqs=prefill_reqs,
                chunked_prefill_reqs=chunked_prefill_reqs,
                decode_reqs=decode_reqs,
                q_starts=q_starts_cpu,
                seq_lens=seq_lens_cpu,
            )
            raise NotImplementedError(
                "Beyond production attention could not represent this mixed or "
                "chunked-prefill scheduler plan. Set "
                "BEYOND_VLLM_PAGE_TABLE_EXACT=1 only to run the explicit eager "
                "reference oracle."
            )

        if prefill_reqs:
            if self.vllm_flash_attn_version is None:
                raise RuntimeError("vLLM FlashAttention version was not initialized")
            _run_production_prefill_forward(
                query=q,
                key=k,
                value=v,
                output=output[:num_actual],
                cache=kv_cache,
                layout=layout,
                block_table=block_table,
                slot_mapping=slot_mapping,
                query_start_loc=q_starts,
                q_starts=q_starts_cpu,
                seq_lens=seq_lens_cpu,
                prefill_reqs=prefill_reqs,
                q_points_k=exact_q_points_k,
                q_points_v=exact_q_points_v,
                thresholds_k=exact_thresholds_k,
                thresholds_v=exact_thresholds_v,
                softmax_scale=self.scale,
                causal=bool(getattr(attn_metadata, "causal", True)),
                alibi_slopes=self.alibi_slopes,
                sliding_window=self.sliding_window,
                logits_soft_cap=self.logits_soft_cap,
                fa_version=self.vllm_flash_attn_version,
                sinks=self.sinks,
            )
            return output

        if D not in (64, 96, 128):
            raise NotImplementedError(
                "default paged decode currently requires head_size in (64, 96, 128)"
            )

        num_decode = len(decode_reqs)
        decode_full_dense = (
            _beyond_dense_decode_batch_eligible(
                num_reqs=num_reqs,
                num_actual_tokens=num_actual,
                prefill_reqs=prefill_reqs,
                chunked_prefill_reqs=chunked_prefill_reqs,
                decode_reqs=decode_reqs,
                q_indices_dense=bool(
                    getattr(attn_metadata, "_beyond_q_indices_dense", False)
                ),
            )
            and block_table.is_contiguous()
        )
        if decode_full_dense and not single_token_decode_batch:
            _runtime_audit_event("prefix_tail_dense_decode_calls")
        slot_workspace = _get_decode_slot_workspace(device, int(slot_mapping.numel()))
        seq_lens_decode = seq_lens
        block_table_decode = block_table
        slot_mapping_decode = slot_mapping
        q_decode = q
        k_decode = k
        v_decode = v
        max_past_len = int(getattr(attn_metadata, "_beyond_max_past_len_aligned", 0))
        if decode_full_dense and seq_lens_cpu:
            max_past_len = max(max_past_len, max(int(x) for x in seq_lens_cpu) - 1)
        if slot_mapping_decode.dtype != torch.int32 or not slot_mapping_decode.is_contiguous():
            slot_mapping_i32 = slot_workspace[: slot_mapping_decode.numel()]
            slot_mapping_i32.copy_(slot_mapping_decode, non_blocking=True)
            slot_mapping_decode = slot_mapping_i32

        grouped_heads = H_q // H_kv
        sm103_requested = _sm103_nonuniform_decode_requested(device)
        sm103_dense_eligible = (
            sm103_requested
            and decode_full_dense
            and D == 128
            and grouped_heads in (4, 8)
            and int(layout.block_size) in BEYOND_PACKED_BLOCK_SIZES
            and layout.code_layout == CODE_LAYOUT_HND_TOKEN_WORD
            and q_points_k is not None
            and q_points_v is not None
            and thresholds_k is not None
            and thresholds_v is not None
            and q_points_k.dim() == 2
            and q_points_v.dim() == 2
        )
        if sm103_dense_eligible:
            if BEYOND_ENABLE_CUDAGRAPH_DECODE:
                # FULL-graph capture deliberately receives an all--1 dummy
                # slot mapping from vLLM.  Reconstruct the live decode slots
                # once at the start of the captured attention stack.  Every
                # layer consumes the same metadata, so caching the output on
                # that metadata removes one redundant kernel per later layer.
                # The derivation itself remains in the CUDA graph and runs on
                # every replay, so changing sequence lengths/page tables never
                # reuses stale slot values.
                slot_mapping_signature = (
                    int(num_decode),
                    int(layout.block_size),
                    int(seq_lens_decode.data_ptr()),
                    int(block_table_decode.data_ptr()),
                    int(seq_lens_decode.stride(0)),
                    int(block_table_decode.stride(0)),
                    int(block_table_decode.stride(1)),
                    int(block_table_decode.shape[1]),
                )
                slot_mapping_decode = getattr(
                    attn_metadata,
                    "_beyond_dense_decode_slot_mapping_gpu_i32",
                    None,
                )
                if (
                    slot_mapping_decode is None
                    or getattr(
                        attn_metadata,
                        "_beyond_dense_decode_slot_mapping_gpu_i32_signature",
                        None,
                    )
                    != slot_mapping_signature
                ):
                    slot_mapping_decode = slot_workspace[:num_decode]
                    _derive_dense_decode_slot_mapping_kernel[(num_decode,)](
                        seq_lens_decode,
                        block_table_decode,
                        slot_mapping_decode,
                        int(seq_lens_decode.stride(0)),
                        int(block_table_decode.stride(0)),
                        int(block_table_decode.stride(1)),
                        BLOCK_TABLE_WIDTH=int(block_table_decode.shape[1]),
                        BLOCK_SIZE=int(layout.block_size),
                        num_warps=1,
                    )
                    attn_metadata._beyond_dense_decode_slot_mapping_gpu_i32 = (
                        slot_mapping_decode
                    )
                    attn_metadata._beyond_dense_decode_slot_mapping_gpu_i32_signature = (
                        slot_mapping_signature
                    )
            self._retain_sm103_cudagraph_buffers(
                q_decode,
                k_decode,
                v_decode,
                output[:num_decode],
                seq_lens_decode,
                slot_mapping_decode,
                block_table_decode,
                kv_cache,
            )
            return self._run_sm103_nonuniform_dense_decode(
                debug_label=layer_name,
                query=q_decode,
                key=k_decode,
                value=v_decode,
                output=output[:num_decode],
                cache=kv_cache,
                layout=layout,
                block_table=block_table_decode,
                seq_lens=seq_lens_decode,
                slot_mapping=slot_mapping_decode,
                q_points_k=q_points_k,
                q_points_v=q_points_v,
                thresholds_k=thresholds_k,
                thresholds_v=thresholds_v,
                max_past_len=max_past_len,
            )
        raise NotImplementedError(
            "Beyond packed decode exposes one production fast path: dense SM103 "
            "non-uniform 4-bit/G32 attention with D=128, GQA4/GQA8, block "
            "64/128, and the HND packed-cache ABI. "
            f"Got sm103_requested={sm103_requested}, dense={decode_full_dense}, "
            f"D={D}, grouped_heads={grouped_heads}, block_size={layout.block_size}, "
            f"code_layout={layout.code_layout!r}. Set "
            "BEYOND_VLLM_PAGE_TABLE_EXACT=1 only for the explicit reference oracle."
        )


def _fake_quant_current_attn_metadata(layer):
    layer_name = getattr(layer, "layer_name", None)
    if layer_name is None:
        return None
    forward_context = None
    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if is_forward_context_available():
            forward_context = get_forward_context()
    except Exception:
        forward_context = None

    if forward_context is not None:
        attn_metadata = getattr(forward_context, "attn_metadata", None)
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata.get(layer_name)
        if attn_metadata is not None:
            return attn_metadata

    try:
        from vllm.model_executor.layers.attention.attention import (
            get_attention_context,
        )

        attn_metadata, _attn_layer, _kv_cache, _slot_mapping = get_attention_context(layer_name)
        return attn_metadata
    except Exception as exc:
        _beyond_fake_quant_debug(
            f"metadata fallback failed layer={layer_name} error={type(exc).__name__}: {exc}",
            once_key=f"metadata_fallback_failed:{layer_name}",
        )
        return None


def _fake_quant_tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash exact tensor storage bytes after materializing them on the CPU."""
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _fake_quant_exact_comparison(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[bool, int, float]:
    equal = bool(torch.equal(actual, expected))
    if equal:
        return True, 0, 0.0
    mismatch = int(torch.count_nonzero(actual != expected).item())
    maximum = float((actual.to(torch.float32) - expected.to(torch.float32)).abs().max().item())
    return False, mismatch, maximum


@torch.compiler.disable
def _fake_quant_verify_triton_against_oracle(
    *,
    layer_name: str,
    key_raw: torch.Tensor,
    value_raw: torch.Tensor,
    key_triton: torch.Tensor,
    value_triton: torch.Tensor,
    q_points_k: torch.Tensor,
    thresholds_k: torch.Tensor,
    q_points_v: torch.Tensor,
    thresholds_v: torch.Tensor,
    group_size_k: int,
    group_size_v: int,
) -> None:
    """Fail closed when a production Triton cache update differs from Python.

    This verifier is deliberately debug-only and compiler-disabled.  It sees
    the exact raw K/V tensors consumed by the production cache update, runs the
    training-equivalent hard-forward oracle on those same tensors, and records
    byte hashes plus exact mismatch statistics before the cache write occurs.
    """
    key_oracle = _fake_quant_token_4d(
        key_raw.unsqueeze(0),
        q_points_k,
        thresholds_k,
        int(group_size_k),
    )[0]
    value_oracle = _fake_quant_token_4d(
        value_raw.unsqueeze(0),
        q_points_v,
        thresholds_v,
        int(group_size_v),
    )[0]
    key_equal, key_mismatch, key_max_abs = _fake_quant_exact_comparison(key_triton, key_oracle)
    value_equal, value_mismatch, value_max_abs = _fake_quant_exact_comparison(
        value_triton, value_oracle
    )

    pid = os.getpid()
    call_index = _FAKE_QUANT_VERIFY_CALL_COUNT.get(pid, 0)
    _FAKE_QUANT_VERIFY_CALL_COUNT[pid] = call_index + 1
    payload = {
        "call_index": call_index,
        "layer": str(layer_name),
        "tokens": int(key_raw.shape[0]),
        "key_shape": list(key_raw.shape),
        "value_shape": list(value_raw.shape),
        "dtype": str(key_raw.dtype),
        "raw_k_sha256": _fake_quant_tensor_sha256(key_raw),
        "raw_v_sha256": _fake_quant_tensor_sha256(value_raw),
        "triton_k_sha256": _fake_quant_tensor_sha256(key_triton),
        "triton_v_sha256": _fake_quant_tensor_sha256(value_triton),
        "oracle_k_sha256": _fake_quant_tensor_sha256(key_oracle),
        "oracle_v_sha256": _fake_quant_tensor_sha256(value_oracle),
        "key_equal": key_equal,
        "value_equal": value_equal,
        "key_mismatch_count": key_mismatch,
        "value_mismatch_count": value_mismatch,
        "key_max_abs": key_max_abs,
        "value_max_abs": value_max_abs,
        "equal": bool(key_equal and value_equal),
    }
    _beyond_fake_quant_debug(
        "oracle_verify " + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    if not payload["equal"]:
        raise RuntimeError(
            "Beyond fake-QDQ Triton cache update failed the same-input Python "
            f"oracle at layer={layer_name} call_index={call_index}: "
            f"K mismatches={key_mismatch}, V mismatches={value_mismatch}"
        )


def _fake_quant_metadata_cpu_list(
    attn_metadata,
    *,
    list_attr: str,
    cpu_attr: str,
    tensor: torch.Tensor,
    length: int,
) -> list[int]:
    values = getattr(attn_metadata, list_attr, None)
    if values is not None:
        return list(values[:length])

    cpu_tensor = getattr(attn_metadata, cpu_attr, None)
    if cpu_tensor is None:
        if _cuda_is_capturing():
            raise RuntimeError(
                "Beyond fake-quant cache patch needs CPU attention metadata "
                "when CUDA graph capture is active."
            )
        cpu_tensor = tensor[:length].detach().cpu()
    values = cpu_tensor[:length].tolist()
    setattr(attn_metadata, list_attr, values)
    return values


def _fake_quant_cache_side_qtables(
    layer,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return validated, TP-local qtables for the fake-QDQ cache hot path.

    Config validation, HBM-table resolution, and tensor-parallel slicing are
    invariant across tokens for one attention layer.  Cache their successful
    result on the layer, keyed by every input that can change the selected
    tables.  The cache is deliberately populated only after all validation and
    slicing succeeds, so an invalid config is never memoized.
    """
    device = key.device
    G_k = int(BEYOND_K_GROUP_SIZE)
    G_v = int(BEYOND_V_GROUP_SIZE)
    cached = (
        getattr(layer, _FAKE_QUANT_LAYER_QTABLE_CACHE_ATTR, None) if layer is not None else None
    )
    tp_rank, tp_world = _cached_beyond_tensor_parallel_rank_world()
    cached_signature = cached[0] if isinstance(cached, tuple) and len(cached) == 2 else None
    if (
        isinstance(cached_signature, tuple)
        and len(cached_signature) == 15
        and cached_signature[0] == int(_FAKE_QUANT_QTABLE_CACHE_EPOCH)
        and cached_signature[1] == _ACTIVE_CONFIG_IDENTITY
        and cached_signature[2] == bool(_ACTIVE_CONFIG_STRICT)
        and cached_signature[3] == int(BEYOND_BITS)
        and cached_signature[4] == G_k
        and cached_signature[5] == G_v
        and cached_signature[6] == device
        and cached_signature[7] == value.device
        and cached_signature[8] == int(tp_rank)
        and cached_signature[9] == int(tp_world)
        and cached_signature[10] == key.shape[1:]
        and cached_signature[11] == value.shape[1:]
    ):
        return cached[1]

    local_k_tables = _local_group_qtable_count(key, G_k)
    local_v_tables = _local_group_qtable_count(value, G_v)
    layer_name = str(getattr(layer, "layer_name", "<unknown>"))
    signature = (
        int(_FAKE_QUANT_QTABLE_CACHE_EPOCH),
        _ACTIVE_CONFIG_IDENTITY,
        bool(_ACTIVE_CONFIG_STRICT),
        int(BEYOND_BITS),
        G_k,
        G_v,
        device,
        value.device,
        int(tp_rank),
        int(tp_world),
        key.shape[1:],
        value.shape[1:],
        local_k_tables,
        local_v_tables,
        layer_name,
    )
    _validate_fake_quant_per_token_config(layer)
    (
        q_points_k,
        thresholds_k,
        q_points_v,
        thresholds_v,
        _uniform_qpoints_k,
        _uniform_qpoints_v,
    ) = _resolve_qtables_for_layer(layer, device)
    q_points_k, thresholds_k = _slice_group_qtables_for_tensor_parallel(
        q_points_k,
        thresholds_k,
        local_k_tables,
        label=f"{layer_name}.k_proj",
        tp_rank=tp_rank,
        tp_world=tp_world,
    )
    q_points_v, thresholds_v = _slice_group_qtables_for_tensor_parallel(
        q_points_v,
        thresholds_v,
        local_v_tables,
        label=f"{layer_name}.v_proj",
        tp_rank=tp_rank,
        tp_world=tp_world,
    )
    result = (q_points_k, thresholds_k, q_points_v, thresholds_v)
    if layer is not None:
        try:
            setattr(
                layer,
                _FAKE_QUANT_LAYER_QTABLE_CACHE_ATTR,
                (signature, result),
            )
        except (AttributeError, RuntimeError, TypeError):
            # Exotic layer wrappers may forbid new attributes.  Keep the
            # original semantics and simply resolve again on their next call.
            pass
    return result


def _fake_quant_num_actual_tokens(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    attn_metadata: FlashAttentionMetadata | None,
) -> int:
    num_actual = int(slot_mapping.numel())
    if attn_metadata is not None:
        num_actual = min(
            num_actual,
            int(getattr(attn_metadata, "num_actual_tokens", num_actual)),
        )
    return min(num_actual, int(key.shape[0]), int(value.shape[0]))


def _fake_quant_direct_cache_update(
    attn_impl,
    layer,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    attn_metadata: FlashAttentionMetadata | None,
) -> bool:
    """Fuse learned hard-QDQ and the dense paged-cache scatter when possible.

    Unsupported cache layouts and debug-oracle runs deliberately fall back to
    the allocation-based reference path.  The supported FlashAttention layout
    eliminates two dense K/V temporaries plus the stock scatter launch.
    """
    if (
        not _TRITON_OK
        or not _beyond_fake_quant_triton_enabled()
        or not _beyond_fake_quant_fused_kv_enabled()
        or not _beyond_fake_quant_direct_cache_enabled()
        or _beyond_fake_quant_verify_oracle_enabled()
        or not key.is_cuda
        or not value.is_cuda
        or not kv_cache.is_cuda
        or not slot_mapping.is_cuda
        or key.dim() != 3
        or value.dim() != 3
        or tuple(key.shape) != tuple(value.shape)
        or slot_mapping.dim() != 1
        or slot_mapping.dtype not in (torch.int32, torch.int64)
        or str(getattr(attn_impl, "kv_cache_dtype", "auto")) != "auto"
    ):
        return False

    attn_type = getattr(attn_impl, "attn_type", None)
    attn_type_name = str(getattr(attn_type, "name", attn_type)).upper()
    if attn_type_name in {"ENCODER", "ENCODER_ONLY"}:
        return False

    num_actual = _fake_quant_num_actual_tokens(key, value, slot_mapping, attn_metadata)
    if num_actual <= 0:
        return True
    max_tokens = _beyond_fake_quant_fused_kv_max_tokens()
    if max_tokens > 0 and num_actual > max_tokens:
        return False

    N = int(num_actual)
    H = int(key.shape[1])
    D = int(key.shape[2])
    G_k = int(BEYOND_K_GROUP_SIZE)
    G_v = int(BEYOND_V_GROUP_SIZE)
    if D != 128 or G_k != 32 or G_v != 32:
        return False

    (
        q_points_k,
        thresholds_k,
        q_points_v,
        thresholds_v,
    ) = _fake_quant_cache_side_qtables(layer, key, value)
    bits_k = _fake_quant_bits_from_qpoints(q_points_k)
    bits_v = _fake_quant_bits_from_qpoints(q_points_v)
    if bits_k is None or bits_v != bits_k:
        return False
    levels = 1 << bits_k
    expected_tables = H * 4
    if not all(
        tensor.is_cuda and tensor.is_contiguous()
        for tensor in (q_points_k, thresholds_k, q_points_v, thresholds_v)
    ):
        return False
    if (
        tuple(q_points_k.shape) != (expected_tables, levels)
        or tuple(thresholds_k.shape) != (expected_tables, levels - 1)
        or tuple(q_points_v.shape) != (expected_tables, levels)
        or tuple(thresholds_v.shape) != (expected_tables, levels - 1)
    ):
        return False

    if int(kv_cache.shape[0]) != 2:
        return False
    key_cache, value_cache = kv_cache.unbind(0)
    if (
        key_cache.dim() != 4
        or value_cache.dim() != 4
        or key_cache.device != key.device
        or value_cache.device != value.device
        or key_cache.dtype != key.dtype
        or value_cache.dtype != value.dtype
        or tuple(key_cache.shape[2:]) != (H, D)
        or tuple(value_cache.shape[2:]) != (H, D)
        or int(key_cache.shape[1]) != int(value_cache.shape[1])
    ):
        return False

    key_actual = key[:N]
    value_actual = value[:N]
    slot_actual = slot_mapping[:N]
    block_size = int(key_cache.shape[1])
    _fake_quant_kv_cache_group_head128_kernel[(N, H)](
        key_actual,
        value_actual,
        key_cache,
        value_cache,
        slot_actual,
        q_points_k,
        thresholds_k,
        q_points_v,
        thresholds_v,
        int(key_actual.stride(0)),
        int(key_actual.stride(1)),
        int(key_actual.stride(2)),
        int(value_actual.stride(0)),
        int(value_actual.stride(1)),
        int(value_actual.stride(2)),
        int(key_cache.stride(0)),
        int(key_cache.stride(1)),
        int(key_cache.stride(2)),
        int(key_cache.stride(3)),
        int(value_cache.stride(0)),
        int(value_cache.stride(1)),
        int(value_cache.stride(2)),
        int(value_cache.stride(3)),
        int(slot_actual.stride(0)),
        BLOCK_SIZE=block_size,
        NUM_Q=levels,
        NUM_T=levels - 1,
        BLOCK_G=32,
        num_warps=_fake_quant_head128_num_warps(N, levels),
    )
    _beyond_fake_quant_debug(
        "qdq_path implementation=triton_direct_dense_cache_grouped_head128 "
        f"bits={bits_k} tokens={N} heads={H} head_dim={D} group_size=32",
        once_key=f"qdq_path:triton_direct_dense_cache:{bits_k}:{H}:{D}",
    )
    return True


def _fake_quant_cache_side_qdq(
    layer,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    attn_metadata: FlashAttentionMetadata | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_actual = _fake_quant_num_actual_tokens(key, value, slot_mapping, attn_metadata)
    if num_actual <= 0:
        return key[:0], value[:0], slot_mapping[:0]

    (
        q_points_k,
        thresholds_k,
        q_points_v,
        thresholds_v,
    ) = _fake_quant_cache_side_qtables(layer, key, value)

    key_actual = key[:num_actual]
    value_actual = value[:num_actual]
    G_k = int(BEYOND_K_GROUP_SIZE)
    G_v = int(BEYOND_V_GROUP_SIZE)
    layer_name = str(getattr(layer, "layer_name", "<unknown>"))
    fused_work = _fake_quant_kv_token_3d(
        key_actual,
        value_actual,
        q_points_k,
        thresholds_k,
        q_points_v,
        thresholds_v,
        G_k,
        G_v,
    )
    if fused_work is None:
        key_work = _fake_quant_token_3d(
            key_actual,
            q_points_k,
            thresholds_k,
            G_k,
        )
        value_work = _fake_quant_token_3d(
            value_actual,
            q_points_v,
            thresholds_v,
            G_v,
        )
    else:
        key_work, value_work = fused_work
    if _beyond_fake_quant_verify_oracle_enabled() and _beyond_fake_quant_triton_enabled():
        _fake_quant_verify_triton_against_oracle(
            layer_name=layer_name,
            key_raw=key_actual,
            value_raw=value_actual,
            key_triton=key_work,
            value_triton=value_work,
            q_points_k=q_points_k,
            thresholds_k=thresholds_k,
            q_points_v=q_points_v,
            thresholds_v=thresholds_v,
            group_size_k=G_k,
            group_size_v=G_v,
        )
    return key_work, value_work, slot_mapping[:num_actual].to(torch.long)


def patch_flash_attention_fake_quant_cache_update() -> None:
    """Inject fake K/V QDQ into stock vLLM FlashAttention cache writes."""
    if getattr(FlashAttentionImpl.do_kv_cache_update, "_beyond_fake_quant_patched", False):
        _beyond_fake_quant_debug(
            "patch already installed",
            once_key="patch:already_installed",
        )
        return

    orig_do_kv_cache_update = FlashAttentionImpl.do_kv_cache_update

    def _patched_do_kv_cache_update(
        self,
        layer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        layer_name = str(getattr(layer, "layer_name", "<unknown>"))
        _beyond_fake_quant_debug(
            "cache_update enter "
            f"layer={layer_name} "
            f"enabled={_beyond_fake_quant_enabled()} "
            f"key_shape={None if key is None else tuple(key.shape)} "
            f"value_shape={None if value is None else tuple(value.shape)} "
            f"slot_tokens={None if slot_mapping is None else int(slot_mapping.numel())}",
            once_key=f"enter:{layer_name}",
        )
        if (
            not _beyond_fake_quant_enabled()
            or key is None
            or value is None
            or kv_cache is None
            or slot_mapping is None
        ):
            return orig_do_kv_cache_update(
                self,
                layer,
                key,
                value,
                kv_cache,
                slot_mapping,
            )

        attn_metadata = _fake_quant_current_attn_metadata(layer)
        if attn_metadata is None:
            _beyond_fake_quant_debug(
                f"cache_update per_token_without_metadata layer={layer_name}",
                once_key=f"per_token_without_metadata:{layer_name}",
            )

        if _fake_quant_direct_cache_update(
            self,
            layer,
            key,
            value,
            kv_cache,
            slot_mapping,
            attn_metadata,
        ):
            return None

        key_qdq, value_qdq, slot_mapping_qdq = _fake_quant_cache_side_qdq(
            layer,
            key,
            value,
            kv_cache,
            slot_mapping,
            attn_metadata,
        )
        return orig_do_kv_cache_update(
            self,
            layer,
            key_qdq,
            value_qdq,
            kv_cache,
            slot_mapping_qdq,
        )

    _patched_do_kv_cache_update._beyond_fake_quant_patched = True
    _patched_do_kv_cache_update._beyond_orig_do_kv_cache_update = orig_do_kv_cache_update
    FlashAttentionImpl.do_kv_cache_update = _patched_do_kv_cache_update
    _beyond_fake_quant_debug("patch installed", once_key="patch:installed")


def patch_flash_attention_fake_quant() -> None:
    get_config_from_env()
    patch_flash_attention_fake_quant_cache_update()


def _infer_block_size(kv_cache: torch.Tensor, H_kv: int, D: int) -> int:
    """Recover block_size from (num_blocks, block_i32) given our layout."""
    block_i32 = kv_cache.shape[1]
    key = (
        int(block_i32),
        int(H_kv),
        int(D),
        int(BEYOND_BITS),
        int(BEYOND_K_GROUP_SIZE),
        int(BEYOND_V_GROUP_SIZE),
    )
    cached = _BLOCK_SIZE_CACHE.get(key)
    if cached is not None:
        return cached
    for bs in (16, 32, 64, 128, 256, 512):
        try:
            lay = _paged_layout(bs, H_kv, D)
        except ValueError:
            continue
        if lay.block_i32 == block_i32:
            _BLOCK_SIZE_CACHE[key] = bs
            return bs
    raise RuntimeError(
        f"cannot infer block_size: kv_cache.shape[1]={block_i32}, "
        f"H_kv={H_kv}, D={D}, bits={BEYOND_BITS}, "
        f"k_group_size={BEYOND_K_GROUP_SIZE}, v_group_size={BEYOND_V_GROUP_SIZE}"
    )


def _beyond_prefilling_list(is_prefilling, num_reqs: int) -> list[bool] | None:
    if is_prefilling is None:
        return None
    if isinstance(is_prefilling, torch.Tensor):
        if is_prefilling.device.type != "cpu":
            is_prefilling = is_prefilling.detach().cpu()
        return [bool(x) for x in is_prefilling[:num_reqs].tolist()]
    return [bool(x) for x in is_prefilling[:num_reqs]]


def _beyond_int_or(value, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _attach_beyond_plan(
    attn_metadata,
    *,
    num_reqs: int,
    num_actual_tokens: int,
    q_starts: list[int],
    seq_lens: list[int],
    max_query_len: int | None,
    max_seq_len: int | None,
    is_prefilling_list: list[bool] | None,
    force_full_decode_capture: bool = False,
    graph_past_cap: int = 0,
) -> None:
    """Attach the packed-backend request plan built from vLLM metadata.

    The forward path consumes these fields directly so request classification
    stays in the metadata builder, matching the shape of the native FA backend.
    """
    num_reqs = int(num_reqs)
    q_starts_signature = tuple(int(x) for x in q_starts[: num_reqs + 1])
    seq_lens_signature = tuple(int(x) for x in seq_lens[:num_reqs])
    is_prefilling_signature = (
        None
        if is_prefilling_list is None
        else tuple(bool(x) for x in is_prefilling_list[:num_reqs])
    )
    plan_request_signature = (
        num_reqs,
        int(num_actual_tokens),
        q_starts_signature,
        seq_lens_signature,
    )
    plan_signature = (
        plan_request_signature,
        _beyond_int_or(max_query_len, 0),
        _beyond_int_or(max_seq_len, 0),
        is_prefilling_signature,
        bool(force_full_decode_capture),
        int(graph_past_cap),
    )
    if getattr(attn_metadata, "_beyond_plan_signature", None) != plan_signature:
        _clear_beyond_index_caches(attn_metadata)
    attn_metadata._beyond_plan_request_signature = plan_request_signature
    attn_metadata._beyond_plan_signature = plan_signature
    q_lens = [int(q_starts[i + 1]) - int(q_starts[i]) for i in range(num_reqs)]
    actual_max_query_len = max(q_lens) if q_lens else 0
    actual_max_seq_len = max(int(seq_lens[i]) for i in range(num_reqs)) if num_reqs > 0 else 0
    num_actual_tokens = int(num_actual_tokens)
    max_query_len_cap = max(
        actual_max_query_len,
        _beyond_int_or(max_query_len, actual_max_query_len),
    )
    max_seq_len_cap = max(
        actual_max_seq_len,
        _beyond_int_or(max_seq_len, actual_max_seq_len),
    )
    explicit_model_cap = _as_positive_int(os.environ.get("BEYOND_MODEL_MAX_LEN_CAP"))
    explicit_token_cap = _as_positive_int(os.environ.get("BEYOND_SCHEDULER_TOKEN_CAP"))
    explicit_block_cap = _as_positive_int(os.environ.get("BEYOND_BLOCK_SIZE_CAP"))
    explicit_seq_cap = _as_positive_int(BEYOND_FULL_DECODE_MAX_BATCH)
    graph_shapes = (
        bool(force_full_decode_capture)
        or _env_int("BEYOND_ENABLE_PIECEWISE_GRAPH", BEYOND_ENABLE_PIECEWISE_GRAPH) > 0
        or _env_int(
            "BEYOND_ENABLE_CUDAGRAPH_DECODE",
            BEYOND_ENABLE_CUDAGRAPH_DECODE,
        )
        > 0
        or explicit_model_cap is not None
        or explicit_token_cap is not None
        or explicit_block_cap is not None
        or explicit_seq_cap is not None
    )
    stable_model_len_cap = max(max_seq_len_cap, int(explicit_model_cap or 0))
    stable_token_cap = max(num_actual_tokens, int(explicit_token_cap or 0))
    stable_block_size = int(explicit_block_cap or BEYOND_K_GROUP_SIZE)
    stable_seq_cap = int(explicit_seq_cap or max(1, num_reqs))
    stable_prefill_len = max(1, stable_model_len_cap - stable_block_size)
    stable_chunked_B_cap = _beyond_workspace_batch_bucket(
        (stable_token_cap + stable_prefill_len - 1) // stable_prefill_len,
        stable_seq_cap,
    )
    q_indices_dense = q_starts[: num_reqs + 1] == list(range(num_reqs + 1))

    if force_full_decode_capture:
        prefill_req_list: list[int] = []
        chunked_req_list: list[int] = []
        decode_req_list = list(range(num_reqs))
        single_token_decode = True
        q_indices_dense = True
        max_past_len_aligned = int(graph_past_cap)
    else:
        any_prefill = any(is_prefilling_list) if is_prefilling_list is not None else False
        single_token_decode = (
            max_query_len == 1
            and num_reqs > 0
            and int(num_actual_tokens) >= int(q_starts[num_reqs])
            and all(q_len == 1 for q_len in q_lens)
            # A padded FULL CUDA-graph batch represents inactive rows with a
            # zero sequence length.  Keep those rows in the dense decode
            # bucket: the SM103 kernel emits zero and the graph-safe slot
            # derivation maps their cache write to -1.  Length one remains a
            # genuine first-token prefill and must not be reclassified.
            and all(
                int(seq_lens[i]) == 0 or int(seq_lens[i]) > 1
                for i in range(num_reqs)
            )
            and not any_prefill
        )
        prefill_req_list = []
        chunked_req_list = []
        decode_req_list = []
        if single_token_decode:
            decode_req_list = list(range(num_reqs))
        else:
            for i, q_len in enumerate(q_lens):
                if q_len <= 0:
                    continue
                ctx_len = int(seq_lens[i]) - q_len
                if ctx_len == 0:
                    prefill_req_list.append(i)
                elif q_len == 1:
                    decode_req_list.append(i)
                else:
                    chunked_req_list.append(i)
        max_past_len_aligned = (
            max(int(seq_lens[i]) - 1 for i in decode_req_list) if decode_req_list else 0
        )
        if single_token_decode and int(graph_past_cap) > 0:
            max_past_len_aligned = max(
                int(max_past_len_aligned),
                int(graph_past_cap),
            )

    prefill_q_lens = [q_lens[i] for i in prefill_req_list]
    prefill_total_q = sum(prefill_q_lens)
    prefill_max_q = max(prefill_q_lens) if prefill_q_lens else 0
    if prefill_req_list and graph_shapes:
        prefill_total_q_cap = _next_power_of_2(
            max(prefill_total_q, num_actual_tokens),
        )
        prefill_max_q_cap = _next_power_of_2(max(prefill_max_q, max_query_len_cap))
    elif prefill_req_list:
        prefill_total_q_cap = max(1, prefill_total_q)
        prefill_max_q_cap = max(1, prefill_max_q)
    else:
        prefill_total_q_cap = 0
        prefill_max_q_cap = 0

    chunked_q_lens: list[int] = []
    chunked_new_lens: list[int] = []
    chunked_starts: list[int] = []
    for i in chunked_req_list:
        q_len = q_lens[i]
        ctx_len = int(seq_lens[i]) - q_len
        chunked_q_lens.append(q_len)
        chunked_new_lens.append(q_len)
        chunked_starts.append(ctx_len)

    chunked_write_lens = list(chunked_new_lens)
    chunked_k_lens = [start + new_len for start, new_len in zip(chunked_starts, chunked_new_lens)]
    chunked_q_bucket = max(chunked_q_lens) if chunked_q_lens else 0
    chunked_write_bucket = max(chunked_write_lens) if chunked_write_lens else 0
    chunked_total_q = sum(chunked_q_lens)
    chunked_total_k = sum(chunked_k_lens)
    chunked_max_k = max(chunked_k_lens) if chunked_k_lens else 0
    if chunked_req_list:
        if graph_shapes:
            stable_chunked_B_cap = max(
                len(chunked_req_list),
                int(stable_chunked_B_cap),
            )
            stable_chunked_B_cap = min(int(stable_seq_cap), stable_chunked_B_cap)
            stable_chunked_k_cap = max(chunked_max_k, stable_model_len_cap)
            chunked_q_bucket_cap = _next_power_of_2(
                max(chunked_q_bucket, max_query_len_cap),
            )
            chunked_write_bucket_cap = _next_power_of_2(
                max(chunked_write_bucket, max_query_len_cap),
            )
            chunked_total_q_cap = _next_power_of_2(
                max(chunked_total_q, num_actual_tokens),
            )
            chunked_max_k_cap = _next_power_of_2(stable_chunked_k_cap)
            chunked_total_k_cap = _next_power_of_2(
                max(
                    chunked_total_k,
                    stable_chunked_B_cap * chunked_max_k_cap,
                ),
            )
        else:
            chunked_q_bucket_cap = max(1, chunked_q_bucket)
            chunked_write_bucket_cap = max(1, chunked_write_bucket)
            chunked_total_q_cap = max(1, chunked_total_q)
            chunked_max_k_cap = max(1, chunked_max_k)
            chunked_total_k_cap = max(1, chunked_total_k)
    else:
        chunked_q_bucket_cap = 0
        chunked_write_bucket_cap = 0
        chunked_total_q_cap = 0
        chunked_max_k_cap = 0
        chunked_total_k_cap = 0

    attn_metadata._beyond_is_prefilling_list = is_prefilling_list
    attn_metadata._beyond_single_token_decode_batch = single_token_decode
    attn_metadata._beyond_q_indices_dense = q_indices_dense

    attn_metadata._beyond_prefill_req_list = prefill_req_list
    attn_metadata._beyond_prefill_q_len_list = prefill_q_lens
    attn_metadata._beyond_prefill_total_q = prefill_total_q
    attn_metadata._beyond_prefill_max_q_len = prefill_max_q
    attn_metadata._beyond_prefill_total_q_cap = prefill_total_q_cap
    attn_metadata._beyond_prefill_max_q_len_cap = prefill_max_q_cap

    attn_metadata._beyond_chunked_prefill_req_list = chunked_req_list
    attn_metadata._beyond_chunked_q_len_list = chunked_q_lens
    attn_metadata._beyond_chunked_new_len_list = chunked_new_lens
    attn_metadata._beyond_chunked_start_list = chunked_starts
    attn_metadata._beyond_chunked_q_bucket = chunked_q_bucket
    attn_metadata._beyond_chunked_write_bucket = chunked_write_bucket
    attn_metadata._beyond_chunked_total_q = chunked_total_q
    attn_metadata._beyond_chunked_total_k = chunked_total_k
    attn_metadata._beyond_chunked_max_k = chunked_max_k
    attn_metadata._beyond_chunked_q_bucket_cap = chunked_q_bucket_cap
    attn_metadata._beyond_chunked_write_bucket_cap = chunked_write_bucket_cap
    attn_metadata._beyond_chunked_total_q_cap = chunked_total_q_cap
    attn_metadata._beyond_chunked_total_k_cap = chunked_total_k_cap
    attn_metadata._beyond_chunked_max_k_cap = chunked_max_k_cap

    attn_metadata._beyond_decode_req_list = decode_req_list
    attn_metadata._beyond_max_past_len_aligned = int(max_past_len_aligned)


def _ensure_beyond_plan(
    attn_metadata,
    *,
    num_reqs: int,
    num_actual_tokens: int,
    q_starts: list[int],
    seq_lens: list[int],
) -> None:
    if hasattr(attn_metadata, "_beyond_prefill_req_list"):
        current_signature = (
            int(num_reqs),
            int(num_actual_tokens),
            tuple(int(x) for x in q_starts[: int(num_reqs) + 1]),
            tuple(int(x) for x in seq_lens[: int(num_reqs)]),
        )
        if getattr(attn_metadata, "_beyond_plan_request_signature", None) == current_signature:
            return
        _clear_beyond_index_caches(attn_metadata)
    _attach_beyond_plan(
        attn_metadata,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        q_starts=q_starts,
        seq_lens=seq_lens,
        max_query_len=getattr(attn_metadata, "max_query_len", None),
        max_seq_len=getattr(attn_metadata, "max_seq_len", None),
        is_prefilling_list=getattr(attn_metadata, "_beyond_is_prefilling_list", None),
    )


def _beyond_attention_cg_support():
    """Select the attention graph boundary, with an opt-in diagnostic escape hatch."""
    if _beyond_vllm_page_table_exact_enabled():
        return AttentionCGSupport.NEVER
    mode = os.environ.get(
        "BEYOND_ATTENTION_CG_SUPPORT",
        "uniform_single_token_decode",
    ).strip().lower()
    if mode in {"never", "none", "off", "0"}:
        return AttentionCGSupport.NEVER
    if mode in {"uniform_single_token_decode", "decode", "auto", "1"}:
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    if mode in {"uniform_batch", "uniform"}:
        return AttentionCGSupport.UNIFORM_BATCH
    raise ValueError(
        "BEYOND_ATTENTION_CG_SUPPORT must be one of "
        "{uniform_single_token_decode, uniform_batch, never}; "
        f"got {mode!r}"
    )


class BeyondPackedMetadataBuilder(FlashAttentionMetadataBuilder):
    """FlashAttention metadata plus CPU copies needed by the packed backend."""

    # CUDA graphs are enabled according to the explicit attention capture
    # policy. Dynamic page-table readback remains an eager diagnostic path.
    _cudagraph_support = _beyond_attention_cg_support()

    def build_for_cudagraph_capture(self, common_attn_metadata):
        attn_metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        num_reqs = common_attn_metadata.num_reqs
        attn_metadata._beyond_query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        attn_metadata._beyond_seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        q_starts = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1].tolist()
        attn_metadata._beyond_query_start_loc_list = q_starts
        seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = common_attn_metadata.seq_lens.detach().cpu()
        seq_lens = seq_lens_cpu[:num_reqs].tolist()
        attn_metadata._beyond_seq_lens_list = seq_lens
        attn_metadata._beyond_num_logits_indices = common_attn_metadata.num_logits_indices
        attn_metadata._beyond_block_table_cpu = None
        attn_metadata._beyond_block_table_cpu_pending = False
        attn_metadata._beyond_block_key_rows = None
        is_prefilling_list = _beyond_prefilling_list(
            common_attn_metadata.is_prefilling,
            num_reqs,
        )
        max_query_len = int(getattr(common_attn_metadata, "max_query_len", 1))
        force_full_decode_capture, graph_past_cap = _beyond_full_decode_graph_overrides(
            enabled=_env_int(
                "BEYOND_ENABLE_CUDAGRAPH_DECODE",
                BEYOND_ENABLE_CUDAGRAPH_DECODE,
            )
            > 0
            and max_query_len == 1,
            max_query_len=max_query_len,
            max_model_len=int(getattr(self.model_config, "max_model_len", 0)),
            block_table=attn_metadata.block_table,
            block_size=int(self.block_size),
        )
        if force_full_decode_capture:
            is_prefilling_list = [False] * num_reqs
        _attach_beyond_plan(
            attn_metadata,
            num_reqs=num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            q_starts=q_starts,
            seq_lens=seq_lens,
            max_query_len=max_query_len,
            max_seq_len=getattr(common_attn_metadata, "max_seq_len", None),
            is_prefilling_list=is_prefilling_list,
            force_full_decode_capture=force_full_decode_capture,
            graph_past_cap=graph_past_cap,
        )
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        attn_metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build=fast_build,
        )
        attn_metadata._beyond_query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        attn_metadata._beyond_seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        num_reqs = common_attn_metadata.num_reqs
        q_starts_list = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1].tolist()
        seq_lens_list = common_attn_metadata._seq_lens_cpu[:num_reqs].tolist()
        attn_metadata._beyond_query_start_loc_list = q_starts_list
        attn_metadata._beyond_seq_lens_list = seq_lens_list
        attn_metadata._beyond_num_logits_indices = common_attn_metadata.num_logits_indices
        attn_metadata._beyond_block_table_cpu = None
        attn_metadata._beyond_block_table_cpu_pending = False
        attn_metadata._beyond_block_key_rows = None
        max_query_len = getattr(
            attn_metadata,
            "max_query_len",
            getattr(common_attn_metadata, "max_query_len", None),
        )
        is_prefilling_list = _beyond_prefilling_list(
            common_attn_metadata.is_prefilling,
            num_reqs,
        )
        has_prefill = any(is_prefilling_list) if is_prefilling_list is not None else False
        force_full_decode_capture, graph_past_cap = _beyond_full_decode_graph_overrides(
            enabled=_env_int(
                "BEYOND_ENABLE_CUDAGRAPH_DECODE",
                BEYOND_ENABLE_CUDAGRAPH_DECODE,
            )
            > 0
            and not has_prefill,
            max_query_len=max_query_len,
            max_model_len=int(getattr(self.model_config, "max_model_len", 0)),
            block_table=attn_metadata.block_table,
            block_size=int(self.block_size),
        )
        if force_full_decode_capture:
            is_prefilling_list = [False] * num_reqs
        _attach_beyond_plan(
            attn_metadata,
            num_reqs=num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            q_starts=q_starts_list,
            seq_lens=seq_lens_list,
            max_query_len=max_query_len,
            max_seq_len=getattr(common_attn_metadata, "max_seq_len", None),
            is_prefilling_list=is_prefilling_list,
            force_full_decode_capture=force_full_decode_capture,
            graph_past_cap=graph_past_cap,
        )
        return attn_metadata


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class BeyondPackedBackend(FlashAttentionBackend):
    forward_includes_kv_cache_update = True  # we do the KV write ourselves
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_name() -> str:
        # Must match ``AttentionBackendEnum.CUSTOM.name`` -- vLLM does
        # ``AttentionBackendEnum[self.attn_backend.get_name()]`` during
        # Attention.__init__ and we register under the CUSTOM slot.
        return "CUSTOM"

    @classmethod
    def get_supported_kernel_block_sizes(cls) -> list[int]:
        """Expose the exact physical page sizes implemented by SM103 kernels."""
        return list(BEYOND_PACKED_BLOCK_SIZES)

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        # The generic vLLM implementation treats multiples of a kernel block
        # size as supported for hybrid-cache backends.  These are exact
        # physical layouts, so arbitrary multiples cannot be silently accepted
        # as logical sub-pages.
        return block_size is None or int(block_size) in BEYOND_PACKED_BLOCK_SIZES

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        del default_block_size
        return BEYOND_PACKED_BLOCK_SIZE

    @staticmethod
    def get_impl_cls():
        return BeyondPackedImpl

    @staticmethod
    def get_builder_cls():
        return BeyondPackedMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        layout = _paged_layout(block_size, num_kv_heads, head_size)
        return (num_blocks, layout.block_i32)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return (0, 1) if not include_num_layers_dimension else (0, 1, 2)


# ---------------------------------------------------------------------------
# Registration hook (manual; call from application code before LLM() init).
# ---------------------------------------------------------------------------


def _beyond_runner_attention_shape(runner) -> tuple[int, int, int, int]:
    model_config = getattr(runner, "model_config", None)
    if model_config is None:
        model_config = getattr(getattr(runner, "vllm_config", None), "model_config", None)
    parallel_config = getattr(getattr(runner, "vllm_config", None), "parallel_config", None)
    H_q = H_kv = D = None
    if model_config is not None:
        get_heads = getattr(model_config, "get_num_attention_heads", None)
        get_kv_heads = getattr(model_config, "get_num_kv_heads", None)
        get_head_size = getattr(model_config, "get_head_size", None)
        try:
            if get_heads is not None and parallel_config is not None:
                H_q = int(get_heads(parallel_config))
        except Exception:
            H_q = None
        try:
            if get_kv_heads is not None and parallel_config is not None:
                H_kv = int(get_kv_heads(parallel_config))
        except Exception:
            H_kv = None
        try:
            if get_head_size is not None:
                D = int(get_head_size())
        except Exception:
            D = None
        hf_config = _beyond_runner_hf_config(runner)
        if H_q is None:
            H_q = _as_positive_int(getattr(hf_config, "num_attention_heads", None))
        if H_kv is None:
            H_kv = _as_positive_int(getattr(hf_config, "num_key_value_heads", None))
        if D is None:
            D = _as_positive_int(getattr(hf_config, "head_dim", None))
            if D is None:
                hidden = _as_positive_int(getattr(hf_config, "hidden_size", None))
                if hidden is not None and H_q:
                    D = max(1, hidden // H_q)
    H_q = int(H_q or 32)
    H_kv = int(H_kv or H_q)
    D = int(D or 128)
    dtype = getattr(runner, "dtype", None)
    if dtype is None and model_config is not None:
        dtype = getattr(model_config, "dtype", None)
    try:
        dtype_bytes = torch.empty((), dtype=dtype).element_size()
    except Exception:
        dtype_bytes = 2
    return H_q, H_kv, D, int(dtype_bytes)


def _beyond_runner_hf_config(runner):
    model_config = getattr(runner, "model_config", None)
    if model_config is None:
        model_config = getattr(
            getattr(runner, "vllm_config", None),
            "model_config",
            None,
        )
    if model_config is None:
        return None
    # Multimodal Hugging Face configs keep the language-model dimensions under
    # ``text_config``.  vLLM resolves that object once as ``hf_text_config``;
    # using the outer vision-language config here silently falls back to
    # generic widths and underestimates graph/workspace memory for Ministral.
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if hf_text_config is not None:
        return hf_text_config
    hf_config = getattr(model_config, "hf_config", None)
    nested_text_config = getattr(hf_config, "text_config", None)
    return nested_text_config if nested_text_config is not None else hf_config


def _beyond_runner_num_hidden_layers(runner) -> int | None:
    hf_config = _beyond_runner_hf_config(runner)
    return _as_positive_int(getattr(hf_config, "num_hidden_layers", None))


def _beyond_runner_hidden_size(runner, H_q: int, D: int) -> int:
    hf_config = _beyond_runner_hf_config(runner)
    hidden = _as_positive_int(getattr(hf_config, "hidden_size", None))
    if hidden is None:
        hidden = max(1, int(H_q) * int(D))
    return int(hidden)


def _beyond_runner_intermediate_size(runner, H_q: int, D: int) -> int:
    hf_config = _beyond_runner_hf_config(runner)
    intermediate = _as_positive_int(getattr(hf_config, "intermediate_size", None))
    if intermediate is not None:
        return int(intermediate)
    hidden = _beyond_runner_hidden_size(runner, H_q, D)
    return int(4 * hidden)


def _beyond_runner_activation_peak_width(runner, H_q: int, H_kv: int, D: int) -> int:
    """Peak row width for uncaptured Inductor temporaries.

    vLLM profiles the CUDA graph pool before the real KV cache is allocated,
    but later uncaptured mixed-prefill shapes can still allocate Inductor
    buffers in the compiled transformer body.  Account the structural live set
    of a gated MLP instead of multiplying by a hand-tuned buffer count:
    fused gate/up projection, activation/mul temporaries, and hidden/residual
    scratch.  Compare with attention projection scratch for non-MLP-heavy
    architectures.
    """
    hidden = _beyond_runner_hidden_size(runner, H_q, D)
    intermediate = _beyond_runner_intermediate_size(runner, H_q, D)
    gate_up_width = 2 * int(intermediate)
    # Inductor can keep the fused gate/up result live while separately
    # materializing activation, mul, and down-proj input staging buffers.
    elementwise_stage_width = 3 * int(intermediate)
    hidden_live_width = 2 * int(hidden)
    mlp_peak_width = gate_up_width + elementwise_stage_width + hidden_live_width
    qkv_width = (int(H_q) + 2 * int(H_kv)) * int(D)
    qkv_peak_width = qkv_width + hidden_live_width
    # The compiled Llama segment keeps residual/RMS buffers live across the
    # gated MLP and the following QKV projection.  Count that structural live
    # set directly so a long uncaptured mixed-prefill step still has room for
    # the Inductor temp allocator after the real KV cache is created.
    compiled_segment_width = 3 * int(intermediate) + qkv_width + 5 * int(hidden)
    return max(
        int(mlp_peak_width),
        int(qkv_peak_width),
        int(compiled_segment_width),
        int(hidden),
    )


def _beyond_runner_largest_temp_width(runner, H_q: int, H_kv: int, D: int) -> int:
    """Largest single uncaptured Inductor allocation in one scheduler step."""
    hidden = _beyond_runner_hidden_size(runner, H_q, D)
    intermediate = _beyond_runner_intermediate_size(runner, H_q, D)
    gate_up_width = 2 * int(intermediate)
    qkv_width = (int(H_q) + 2 * int(H_kv)) * int(D)
    return max(gate_up_width, int(intermediate), qkv_width, int(hidden))


@dataclass(frozen=True)
class _BeyondGraphWorkspaceReserve:
    workspace_bytes: int = 0
    activation_bytes: int = 0
    allocator_slack_bytes: int = 0
    largest_temp_bytes: int = 0
    intermediate_temp_bytes: int = 0
    next_temp_guard_bytes: int = 0
    contiguous_headroom_bytes: int = 0
    tail_guard_bytes: int = 0
    total_bytes: int = 0
    override_bytes: int = 0


def _beyond_graph_workspace_reserve_bytes(runner) -> int:
    return _beyond_graph_workspace_reserve(runner).total_bytes


def _beyond_fast_cudagraph_pool_estimate_bytes(runner) -> int:
    requested = _as_positive_int(
        getattr(runner.compilation_config, "max_cudagraph_capture_size", None)
    )
    if requested is None or (
        BEYOND_ENABLE_PIECEWISE_GRAPH <= 0 and BEYOND_ENABLE_CUDAGRAPH_DECODE <= 0
    ):
        return 0
    H_q, H_kv, D, dtype_bytes = _beyond_runner_attention_shape(runner)
    max_num_tokens = _as_positive_int(
        getattr(runner.scheduler_config, "max_num_batched_tokens", None)
    )
    if max_num_tokens is None:
        max_num_tokens = requested
    graph_tokens = max(1, min(int(requested), int(max_num_tokens)))
    graph_width = max(
        _beyond_runner_activation_peak_width(runner, H_q, H_kv, D),
        _beyond_runner_largest_temp_width(runner, H_q, H_kv, D),
    )
    capture_sizes = (
        getattr(
            runner.compilation_config,
            "cudagraph_capture_sizes",
            None,
        )
        or ()
    )
    normalized_capture_sizes: tuple[int, ...] = ()
    try:
        normalized_capture_sizes = tuple(
            sorted({int(x) for x in capture_sizes if int(x) > 0})
        )
    except Exception:
        normalized_capture_sizes = ()
    full_decode_cap = _as_positive_int(BEYOND_FULL_DECODE_MAX_BATCH)
    if full_decode_cap is None:
        full_decode_cap = _as_positive_int(
            getattr(runner.scheduler_config, "max_num_seqs", None)
        )
    if full_decode_cap is None:
        full_decode_cap = int(graph_tokens)
    full_capture_sizes = tuple(
        size
        for size in normalized_capture_sizes
        if int(size) <= int(full_decode_cap)
    )
    graph_count = len(full_capture_sizes)
    if graph_count <= 0:
        graph_count = 1

    # FULL graphs retain model activations for every captured batch, so their
    # pool scales with the *sum* of capture sizes rather than only the largest
    # size.  That distinction is material for vLLM's 51-key roster through
    # B512 and especially for dense high-batch rosters.  Three QKV-width rows
    # per decoder layer cover the captured input/projection/output live set.
    # Use the sum of reachable FULL-decode capture sizes so large rosters are
    # not underestimated by a max-size-only bound.
    # PIECEWISE may capture token batches above the serving concurrency (for
    # example B256 at max_num_seqs=128), but those graphs do not retain the
    # expensive whole-model live set.  Charge the structural term only for
    # sizes reachable by FULL decode.
    captured_token_rows = sum(full_capture_sizes) if full_capture_sizes else graph_tokens
    num_hidden_layers = _beyond_runner_num_hidden_layers(runner)
    structural_graph_bytes = 0
    if num_hidden_layers is not None:
        qkv_width = (int(H_q) + 2 * int(H_kv)) * int(D)
        structural_graph_bytes = (
            3
            * int(captured_token_rows)
            * int(num_hidden_layers)
            * int(qkv_width)
            * int(dtype_bytes)
        )

    base_bytes = int(graph_tokens) * int(graph_width) * int(dtype_bytes)
    # Small FULL-decode rosters use a conservative 128-MiB driver allowance and
    # 128-MiB rounding. Larger rosters retain a 512-MiB floor and 256-MiB
    # rounding quantum.
    small_full_roster = int(graph_count) <= 4
    driver_floor_bytes = (128 if small_full_roster else 512) << 20
    per_graph_driver_bytes = max(
        int(driver_floor_bytes), int(graph_count) * (4 << 20)
    )
    floor_bytes = int(driver_floor_bytes)
    estimate = max(
        max(int(base_bytes), int(structural_graph_bytes))
        + int(per_graph_driver_bytes),
        floor_bytes,
    )
    quantum = (128 if small_full_roster else 256) << 20
    return ((int(estimate) + quantum - 1) // quantum) * quantum


def _beyond_full_decode_profile_seq_lens(runner, desc) -> int | None:
    num_tokens = _as_positive_int(getattr(desc, "num_tokens", None))
    if num_tokens is None:
        return None
    max_model_len = _as_positive_int(getattr(runner, "max_model_len", None))
    if max_model_len is None:
        model_config = getattr(getattr(runner, "vllm_config", None), "model_config", None)
        max_model_len = _as_positive_int(getattr(model_config, "max_model_len", None))
    max_num_tokens = _as_positive_int(getattr(runner, "max_num_tokens", None))
    if max_num_tokens is None:
        scheduler_config = getattr(runner, "scheduler_config", None)
        max_num_tokens = _as_positive_int(getattr(scheduler_config, "max_num_batched_tokens", None))
    if max_model_len is None or max_num_tokens is None:
        return None
    profile_seq_lens = max(
        1,
        min(int(max_model_len), int(max_num_tokens) // int(num_tokens)),
    )
    if _beyond_long_context_moe_uses_bounded_full_graph_profile(
        getattr(runner, "vllm_config", runner)
    ):
        bounded_profile = _as_positive_int(BEYOND_MOE_FULL_GRAPH_PROFILE_SEQ_LEN)
        if bounded_profile is not None:
            profile_seq_lens = min(profile_seq_lens, int(bounded_profile))
    return int(profile_seq_lens)


def _beyond_is_global_first_rank() -> bool:
    try:
        from vllm.distributed.parallel_state import is_global_first_rank

        return bool(is_global_first_rank())
    except Exception:
        return True


def _beyond_full_decode_desc_key(
    desc,
    profile_seq_lens: int | None,
) -> tuple[int, int, int, int]:
    return (
        int(getattr(desc, "num_tokens", 1)),
        int(bool(getattr(desc, "uniform", False))),
        int(getattr(desc, "num_active_loras", 0)),
        int(profile_seq_lens or 0),
    )


def _beyond_full_decode_prewarmed(runner, desc, profile_seq_lens: int | None) -> bool:
    warmed = getattr(runner, "_beyond_full_decode_prewarmed", None)
    return warmed is not None and _beyond_full_decode_desc_key(desc, profile_seq_lens) in warmed


def _beyond_mark_full_decode_prewarmed(runner, desc, profile_seq_lens: int | None) -> None:
    warmed = getattr(runner, "_beyond_full_decode_prewarmed", None)
    if warmed is None:
        warmed = set()
        setattr(runner, "_beyond_full_decode_prewarmed", warmed)
    warmed.add(_beyond_full_decode_desc_key(desc, profile_seq_lens))


def _beyond_full_decode_allow_microbatching(runner, desc) -> bool:
    try:
        from vllm.v1.worker.ubatch_utils import check_ubatch_thresholds
    except Exception:
        return False
    return bool(
        getattr(getattr(runner, "parallel_config", None), "use_ubatching", False)
        and bool(getattr(desc, "uniform", False))
        and check_ubatch_thresholds(
            config=runner.vllm_config.parallel_config,
            num_tokens=int(getattr(desc, "num_tokens", 1)),
            uniform_decode=bool(getattr(desc, "uniform", False)),
        )
    )


def _beyond_packed_writer_prewarm_token_counts(
    *,
    max_tokens: int,
    num_kv_heads: int,
    write_dq: bool,
) -> tuple[tuple[int, int], ...]:
    """Smallest token count selecting each reachable writer CTA policy."""
    max_tokens = int(max_tokens)
    num_kv_heads = int(num_kv_heads)
    if max_tokens <= 0 or num_kv_heads not in (4, 8):
        return ()
    cases: list[tuple[int, int]] = []
    for target_groups in (1, 4):
        lo = 1
        hi = max_tokens + 1
        while lo < hi:
            mid = (lo + hi) // 2
            selected = _select_packed_writer_groups_per_program(
                mid,
                num_kv_heads,
                bool(write_dq),
            )
            if selected < target_groups:
                lo = mid + 1
            else:
                hi = mid
        if lo <= max_tokens and _select_packed_writer_groups_per_program(
            lo,
            num_kv_heads,
            bool(write_dq),
        ) == target_groups:
            cases.append((lo, target_groups))
    return tuple(cases)


def _beyond_packed_readback_prewarm_cases(
    *,
    model_dtype: torch.dtype,
    max_model_len: int,
    max_num_seqs: int,
    max_dense_tokens: int,
    num_kv_heads: int,
) -> tuple[tuple[int, int, int, bool, int, int, bool], ...]:
    """Choose one minimal launch for each reachable readback binary.

    Rows are ``(batch, actual_seq_len, bucket, pairwise, tokens_per_program,
    num_warps, flat_grid)``.  The 2D mapping deliberately omits the length
    bucket from its compile key, so all buckets with the same launch policy
    share one binary.
    """
    max_model_len = int(max_model_len)
    max_num_seqs = int(max_num_seqs)
    max_dense_tokens = int(max_dense_tokens)
    num_kv_heads = int(num_kv_heads)
    if (
        model_dtype not in (torch.float16, torch.bfloat16)
        or max_model_len <= 0
        or max_num_seqs <= 0
        or max_dense_tokens <= 0
        or num_kv_heads not in (4, 8)
    ):
        return ()

    max_bucket = max(128, 1 << (max_model_len - 1).bit_length())
    best_by_signature: dict[
        tuple[bool, int, int, bool],
        tuple[int, int, int, bool, int, int, bool],
    ] = {}
    bucket = 128
    while bucket <= max_bucket:
        actual_seq_len = 1 if bucket == 128 else bucket // 2 + 1
        if actual_seq_len > max_model_len:
            break
        max_batch = min(max_num_seqs, max_dense_tokens // actual_seq_len)
        for batch_size in range(1, max_batch + 1):
            pairwise, tokens_per_program, num_warps = (
                _select_packed_readback_launch_policy(
                    model_dtype=model_dtype,
                    batch_size=batch_size,
                    max_seq_bucket=bucket,
                    num_heads=num_kv_heads,
                )
            )
            flat_grid = _select_packed_readback_use_flat_grid(
                model_dtype=model_dtype,
                batch_size=batch_size,
                max_seq_bucket=bucket,
                num_heads=num_kv_heads,
                tokens_per_program=tokens_per_program,
            )
            signature = (
                bool(pairwise),
                int(tokens_per_program),
                int(num_warps),
                bool(flat_grid),
            )
            candidate = (
                batch_size,
                actual_seq_len,
                bucket,
                bool(pairwise),
                int(tokens_per_program),
                int(num_warps),
                bool(flat_grid),
            )
            previous = best_by_signature.get(signature)
            candidate_cost = batch_size * bucket
            previous_cost = previous[0] * previous[2] if previous is not None else None
            if previous is None or candidate_cost < previous_cost:
                best_by_signature[signature] = candidate
        bucket *= 2
    return tuple(
        sorted(
            best_by_signature.values(),
            key=lambda item: (item[0] * item[2], item[2], item[0]),
        )
    )


def _beyond_chunked_prefill_fa4_prewarm_cases(
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    num_sms: int,
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Return minimal FA4 SplitKV shapes reachable from chunked prefill.

    Rows are ``(query_len, key_len, num_splits, q_stage,
    combine_log_max_splits)``.  FA4's forward compile key is independent of
    exact Q/K lengths and split count; only one/two Q stages and whether
    SplitKV is active matter.  Its combine kernel additionally buckets the
    maximum split count by ``ceil(log2(num_splits))`` with a floor of five for
    head dimension 128.  Cover those signatures directly so a long or
    prefix-cached first request never pays CuTe compilation in the serving
    window.
    """
    max_model_len = int(max_model_len)
    max_num_batched_tokens = int(max_num_batched_tokens)
    num_heads = int(num_heads)
    num_kv_heads = int(num_kv_heads)
    num_sms = int(num_sms)
    if (
        max_model_len <= 512
        or max_num_batched_tokens < 2
        or num_heads <= 0
        or num_kv_heads not in (4, 8)
        or num_heads % num_kv_heads
        or num_sms <= num_kv_heads
    ):
        return ()

    q_heads_per_kv = num_heads // num_kv_heads
    max_query_len = min(max_model_len, max_num_batched_tokens)
    base_key_len = min(max_model_len, 513)
    max_auto_splits = min(
        128,
        num_sms // num_kv_heads,
        (max_model_len + 127) // 128,
    )
    if max_auto_splits <= 1:
        return ()

    cases: list[tuple[int, int, int, int, int]] = [
        (2, base_key_len, 2, 1, 5)
    ]
    stage_two_query = 128 // q_heads_per_kv + 1
    if max_query_len >= stage_two_query:
        cases.append((stage_two_query, base_key_len, 2, 2, 5))

    max_combine_log = max(5, (max_auto_splits - 1).bit_length())
    for combine_log in range(6, max_combine_log + 1):
        num_splits = (1 << (combine_log - 1)) + 1
        key_len = (num_splits - 1) * 128 + 1
        if key_len > max_model_len:
            break
        cases.append((2, key_len, num_splits, 1, combine_log))
    return tuple(cases)


def _beyond_packed_prefill_prewarm_shapes(
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_model_len: int,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[tuple[int, int, int], ...]:
    """Return minimal homogeneous shapes covering prefill JIT variants.

    Each tuple is ``(total_tokens, per_request_tokens, groups_per_program)``.
    The packed writer specializes on one versus four G32 groups per CTA, while
    SM10x FA4 specializes on whether packed-GQA query length needs one or two
    stages.  The two decisions are independent, so select the smallest set of
    reachable shapes whose union covers both binary families.
    """
    max_num_seqs = int(max_num_seqs)
    max_num_batched_tokens = int(max_num_batched_tokens)
    max_model_len = int(max_model_len)
    num_heads = int(num_heads)
    num_kv_heads = int(num_kv_heads)
    if (
        max_num_seqs <= 0
        or max_num_batched_tokens <= 0
        or max_model_len <= 0
        or num_heads <= 0
        or num_kv_heads not in (4, 8)
        or num_heads % num_kv_heads
    ):
        return ()

    token_cap = min(max_num_batched_tokens, max_num_seqs * max_model_len)
    if token_cap <= 0:
        return ()

    q_heads_per_kv = num_heads // num_kv_heads
    max_query_len = min(max_model_len, max_num_batched_tokens)
    stage_one_max = min(max_query_len, 128 // q_heads_per_kv)
    stage_ranges = {1: (1, stage_one_max)}
    if stage_one_max < max_query_len:
        stage_ranges[2] = (stage_one_max + 1, max_query_len)

    def _smallest_pair_shape(
        groups_per_program: int,
        q_stage: int,
    ) -> tuple[int, int, int] | None:
        q_min, q_max = stage_ranges[q_stage]
        if q_min > q_max:
            return None
        best: tuple[int, int, int] | None = None
        for query_len in range(q_min, q_max + 1):
            max_reqs = min(max_num_seqs, token_cap // query_len)
            if max_reqs <= 0:
                continue
            lo_reqs = 1
            hi_reqs = max_reqs + 1
            while lo_reqs < hi_reqs:
                mid_reqs = (lo_reqs + hi_reqs) // 2
                selected = _select_packed_writer_groups_per_program(
                    query_len * mid_reqs,
                    num_kv_heads,
                    True,
                )
                if selected < groups_per_program:
                    lo_reqs = mid_reqs + 1
                else:
                    hi_reqs = mid_reqs
            if lo_reqs > max_reqs:
                continue
            total_tokens = query_len * lo_reqs
            if (
                _select_packed_writer_groups_per_program(
                    total_tokens,
                    num_kv_heads,
                    True,
                )
                != groups_per_program
            ):
                continue
            candidate = (total_tokens, query_len, groups_per_program)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        return best

    candidates: list[tuple[int, int, int, int]] = []
    for groups_per_program in (1, 4):
        for q_stage in stage_ranges:
            shape = _smallest_pair_shape(groups_per_program, q_stage)
            if shape is not None:
                candidates.append((*shape, q_stage))
    if not candidates:
        return ()

    uncovered = {
        *(('writer', item[2]) for item in candidates),
        *(('fa4_q_stage', item[3]) for item in candidates),
    }
    selected: list[tuple[int, int, int]] = []
    remaining = list(candidates)
    while uncovered:
        best = min(
            remaining,
            key=lambda item: (
                -len(
                    {
                        ('writer', item[2]),
                        ('fa4_q_stage', item[3]),
                    }
                    & uncovered
                ),
                item[0],
                item[1],
            ),
        )
        selected.append(best[:3])
        uncovered.difference_update(
            {('writer', best[2]), ('fa4_q_stage', best[3])}
        )
        remaining.remove(best)
    return tuple(selected)


def _beyond_prewarm_slot_mapping(runner) -> None:
    """Compile every slot-mapping specialization reachable after startup.

    In addition to vLLM's prompt slot mapper, cold chunked-prefill eventually
    crosses a one-token boundary that reconstructs the live decode slot from
    graph-stable sequence lengths and the block table.  FULL graph capture may
    cover only a padded MoE batch, so explicitly compile the B1 reconstruction
    here rather than charging its Triton JIT to the first real request.
    """
    if getattr(runner, "_beyond_slot_mapping_prewarmed", False):
        return
    input_batch = getattr(runner, "input_batch", None)
    block_table = getattr(input_batch, "block_table", None)
    if block_table is None or not torch.cuda.is_available():
        return
    device = torch.device(getattr(runner, "device", "cuda"))
    for num_tokens in (1, 2):
        query_start_loc = torch.tensor(
            [0, num_tokens], dtype=torch.int32, device=device
        )
        positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        block_table.compute_slot_mapping(1, query_start_loc, positions)
    physical_tables = getattr(block_table, "block_tables", None)
    if physical_tables is None:
        physical_tables = (block_table,)
    dense_decode_signatures: set[tuple[int, int, int, int]] = set()
    if _TRITON_OK:
        seq_lens = torch.ones(1, dtype=torch.int32, device=device)
        slot_mapping = torch.empty(1, dtype=torch.int32, device=device)
        for physical_table in physical_tables:
            table = getattr(
                getattr(physical_table, "block_table", None),
                "gpu",
                None,
            )
            if table is None or int(table.shape[1]) <= 0:
                continue
            signature = (
                int(table.shape[1]),
                int(table.stride(0)),
                int(table.stride(1)),
                int(physical_table.block_size),
            )
            if signature in dense_decode_signatures:
                continue
            dense_decode_signatures.add(signature)
            _derive_dense_decode_slot_mapping_kernel[(1,)](
                seq_lens,
                table,
                slot_mapping,
                int(seq_lens.stride(0)),
                int(table.stride(0)),
                int(table.stride(1)),
                BLOCK_TABLE_WIDTH=int(table.shape[1]),
                BLOCK_SIZE=int(physical_table.block_size),
                num_warps=1,
            )
    torch.cuda.synchronize(device)
    runner._beyond_slot_mapping_prewarmed = True


def _patch_vllm_slot_mapping() -> None:
    """Use one shape-polymorphic slot-mapping binary for Beyond requests."""
    if not _TRITON_OK:
        return
    try:
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID
        from vllm.v1.worker.block_table import BlockTable
    except Exception:
        return
    if getattr(BlockTable.compute_slot_mapping, "_beyond_shape_polymorphic", False):
        return

    original_compute_slot_mapping = BlockTable.compute_slot_mapping

    def _compute_slot_mapping_shape_polymorphic(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        num_reqs = int(num_reqs)
        num_tokens = int(positions.shape[0])
        if num_tokens == 1:
            return original_compute_slot_mapping(
                self,
                num_reqs,
                query_start_loc,
                positions,
            )
        total_cp_world_size = int(self.pcp_world_size) * int(self.dcp_world_size)
        total_cp_rank = (
            int(self.pcp_rank) * int(self.dcp_world_size) + int(self.dcp_rank)
        )
        _beyond_compute_slot_mapping_kernel[(num_reqs + 1,)](
            num_reqs,
            num_tokens,
            int(self.max_num_batched_tokens),
            query_start_loc,
            positions,
            self.block_table.gpu,
            int(self.block_table.gpu.stride(0)),
            int(self.block_size),
            self.slot_mapping.gpu,
            TOTAL_CP_WORLD_SIZE=total_cp_world_size,
            TOTAL_CP_RANK=total_cp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=int(self.cp_kv_cache_interleave_size),
            PAD_ID=int(PAD_SLOT_ID),
            BLOCK_SIZE=1024,
        )

    _compute_slot_mapping_shape_polymorphic._beyond_shape_polymorphic = True
    BlockTable.compute_slot_mapping = _compute_slot_mapping_shape_polymorphic


def _beyond_prewarm_chunked_prefill_kernels(
    *,
    runner,
    first_impl: BeyondPackedImpl,
    dtype: torch.dtype,
    layout: PagedLayout,
    q_points_k: torch.Tensor,
    thresholds_k: torch.Tensor,
    q_points_v: torch.Tensor,
    thresholds_v: torch.Tensor,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int, int, bool, int, int, bool], ...],
    tuple[tuple[int, int, int, int, int], ...],
]:
    """Compile reachable no-DQ writer, readback, and FA4 SplitKV binaries."""
    scheduler_config = getattr(runner, "scheduler_config", None)
    model_config = getattr(runner, "model_config", None)
    cache_config = getattr(runner, "cache_config", None)
    max_num_seqs = int(getattr(scheduler_config, "max_num_seqs", 0) or 0)
    max_num_batched_tokens = int(
        getattr(scheduler_config, "max_num_batched_tokens", 0) or 0
    )
    max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
    chunked_enabled = bool(
        getattr(scheduler_config, "enable_chunked_prefill", False)
    )
    prefix_enabled = bool(getattr(cache_config, "enable_prefix_caching", False))
    if (
        not chunked_enabled
        or max_num_seqs <= 0
        or max_num_batched_tokens <= 0
        or max_model_len <= 0
        or (max_model_len <= max_num_batched_tokens and not prefix_enabled)
    ):
        return (), (), ()

    device = q_points_k.device
    num_heads = int(first_impl.num_heads)
    num_kv_heads = int(first_impl.num_kv_heads)
    live_hybrid_table = getattr(
        getattr(runner, "input_batch", None),
        "block_table",
        None,
    )
    live_tables = getattr(live_hybrid_table, "block_tables", None)
    if not live_tables:
        return (), (), ()
    live_block_table = live_tables[0].block_table.gpu

    writer_cases = _beyond_packed_writer_prewarm_token_counts(
        max_tokens=max_num_batched_tokens,
        num_kv_heads=num_kv_heads,
        write_dq=False,
    )
    for total_tokens, groups_per_program in writer_cases:
        qv = torch.zeros(
            (total_tokens, num_heads + 2 * num_kv_heads, 128),
            dtype=dtype,
            device=device,
        )
        key = torch.zeros(
            (total_tokens, num_kv_heads, 128), dtype=dtype, device=device
        )
        value = qv[:, num_heads + num_kv_heads :]
        cache = torch.empty(
            (
                (total_tokens + int(layout.block_size) - 1)
                // int(layout.block_size),
                layout.block_i32,
            ),
            dtype=torch.int32,
            device=device,
        )
        for slot_dtype in (torch.int32, torch.int64):
            # vLLM's real scheduler owns an int64 slot buffer, while graph
            # profiling and direct packed helpers use int32.  Triton includes
            # pointer element type in its compile key, so cover both ABIs.
            slots = torch.arange(
                total_tokens, dtype=slot_dtype, device=device
            )
            for signal_pdl in (False, True):
                # Chunked/full prefill launches without PDL; production decode
                # signals its dependent CuTe consumer.  The signal instruction
                # is also a compile-time branch, so both are serving binaries.
                if _beyond_is_global_first_rank():
                    print(
                        "[beyond] prewarming packed writer "
                        f"(tokens={total_tokens}, "
                        f"groups_per_program={groups_per_program}, "
                        f"slots={str(slot_dtype).removeprefix('torch.')}, "
                        f"signal_pdl={signal_pdl})",
                        file=sys.stderr,
                    )
                write_tokens_to_paged_cache(
                    cache,
                    layout,
                    live_block_table[:1],
                    slots,
                    key,
                    value,
                    q_points_k=q_points_k,
                    q_points_v=q_points_v,
                    thresholds_k=thresholds_k,
                    thresholds_v=thresholds_v,
                    signal_pdl_dependents=signal_pdl,
                    require_fast=True,
                )

    dense_cap_mb = _env_int("BEYOND_CHUNKED_PREFILL_DENSE_MAX_MB", 1024)
    max_dense_tokens = (
        dense_cap_mb * 1024 * 1024 // (4 * num_kv_heads * 128)
        if dense_cap_mb > 0
        else 0
    )
    readback_cases = _beyond_packed_readback_prewarm_cases(
        model_dtype=dtype,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_dense_tokens=max_dense_tokens,
        num_kv_heads=num_kv_heads,
    )
    cache = torch.zeros(
        (1, layout.block_i32),
        dtype=torch.int32,
        device=device,
    )
    out_k = torch.empty((1, num_kv_heads, 128), dtype=dtype, device=device)
    out_v = torch.empty_like(out_k)
    for (
        batch_size,
        actual_seq_len,
        max_seq_bucket,
        pairwise,
        tokens_per_program,
        num_warps,
        flat_grid,
    ) in readback_cases:
        block_table = live_block_table[:batch_size]
        if (
            int(block_table.shape[0]) != batch_size
            or int(block_table.shape[1]) * int(layout.block_size) < actual_seq_len
        ):
            continue
        seq_start_loc = torch.zeros(
            batch_size + 1,
            dtype=torch.int32,
            device=device,
        )
        aligned_seq_lens = torch.zeros(
            batch_size, dtype=torch.int32, device=device
        )
        unaligned_storage = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device
        )
        for seq_lens, alignment in (
            (aligned_seq_lens, "aligned"),
            (unaligned_storage[1:], "generic"),
        ):
            # Attention metadata may expose a four-byte-offset view.  Triton
            # emits a separate pointer-alignment specialization, so compile
            # both without copying the serving metadata on every chunk.
            if _beyond_is_global_first_rank():
                print(
                    "[beyond] prewarming chunked-prefill readback "
                    f"(batch={batch_size}, bucket={max_seq_bucket}, "
                    f"pairwise={pairwise}, tokens_per_program={tokens_per_program}, "
                    f"num_warps={num_warps}, flat_grid={flat_grid}, "
                    f"seq_lens={alignment})",
                    file=sys.stderr,
                )
            dequantize_paged_kv_to_dense(
                cache,
                layout,
                block_table,
                seq_lens,
                seq_start_loc,
                out_k,
                out_v,
                max_seq_len=actual_seq_len,
                q_points_k=q_points_k,
                q_points_v=q_points_v,
            )

    fa4_cases: tuple[tuple[int, int, int, int, int], ...] = ()
    if int(first_impl.vllm_flash_attn_version) == 4:
        num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
        fa4_cases = _beyond_chunked_prefill_fa4_prewarm_cases(
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_sms=num_sms,
        )
    for query_len, key_len, num_splits, q_stage, combine_log in fa4_cases:
        query = torch.zeros(
            (query_len, num_heads, 128), dtype=dtype, device=device
        )
        key = torch.zeros(
            (key_len, num_kv_heads, 128), dtype=dtype, device=device
        )
        value = torch.zeros_like(key)
        output = torch.empty_like(query)
        cu_seqlens_q = torch.tensor(
            [0, query_len], dtype=torch.int32, device=device
        )
        cu_seqlens_k = torch.tensor(
            [0, key_len], dtype=torch.int32, device=device
        )
        if _beyond_is_global_first_rank():
            print(
                "[beyond] prewarming chunked-prefill FA4 SplitKV "
                f"(query={query_len}, key={key_len}, splits={num_splits}, "
                f"q_stage={q_stage}, combine_log={combine_log})",
                file=sys.stderr,
            )
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=query_len,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=key_len,
            softmax_scale=float(first_impl.scale),
            causal=True,
            alibi_slopes=first_impl.alibi_slopes,
            window_size=list(first_impl.sliding_window),
            softcap=float(first_impl.logits_soft_cap),
            num_splits=num_splits,
            fa_version=int(first_impl.vllm_flash_attn_version),
            s_aux=first_impl.sinks,
        )
    return writer_cases, readback_cases, fa4_cases


def _beyond_prewarm_packed_prefill_kernels(runner) -> None:
    """Compile reachable rounded-prefill writer and FA4 binaries directly."""
    if getattr(runner, "_beyond_packed_prefill_kernels_prewarmed", False):
        return

    model = getattr(runner, "model", None)
    if model is None:
        return
    layers = []
    for layer in model.modules():
        impl = getattr(layer, "impl", None)
        if isinstance(impl, BeyondPackedImpl):
            layers.append((layer, impl))
    if not layers:
        return

    first_impl = layers[0][1]
    num_heads = int(first_impl.num_heads)
    num_kv_heads = int(first_impl.num_kv_heads)
    if (
        int(first_impl.head_size) != 128
        or num_kv_heads not in (4, 8)
        or any(
            int(impl.num_heads) != num_heads
            or int(impl.num_kv_heads) != num_kv_heads
            or int(impl.head_size) != 128
            for _, impl in layers
        )
    ):
        return

    device = torch.device(getattr(runner, "device", "cuda"))
    first_tables = None
    for layer, impl in layers:
        _validate_fake_quant_per_token_config(layer)
        tables = impl._ensure_tp_local_q_points(layer, device)
        if first_tables is None:
            first_tables = tables
    if first_tables is None:
        return
    q_points_k, thresholds_k, q_points_v, thresholds_v = first_tables
    expected_tables = num_kv_heads * 4
    if (
        q_points_k is None
        or thresholds_k is None
        or q_points_v is None
        or thresholds_v is None
        or tuple(q_points_k.shape) != (expected_tables, 16)
        or tuple(thresholds_k.shape) != (expected_tables, 15)
        or tuple(q_points_v.shape) != (expected_tables, 16)
        or tuple(thresholds_v.shape) != (expected_tables, 15)
    ):
        raise RuntimeError(
            "rounded-prefill prewarm requires independent grouped nonuniform "
            "K/V qpoints and thresholds for the active model"
        )

    scheduler_config = getattr(runner, "scheduler_config", None)
    model_config = getattr(runner, "model_config", None)
    shapes = _beyond_packed_prefill_prewarm_shapes(
        max_num_seqs=int(getattr(scheduler_config, "max_num_seqs", 0) or 0),
        max_num_batched_tokens=int(
            getattr(scheduler_config, "max_num_batched_tokens", 0) or 0
        ),
        max_model_len=int(getattr(model_config, "max_model_len", 0) or 0),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    if not shapes:
        return

    dtype = getattr(model_config, "dtype", None)
    if dtype not in (torch.float16, torch.bfloat16):
        return
    layout = _beyond_runner_packed_layout(runner, num_kv_heads)
    if layout is None:
        return
    block_size = int(layout.block_size)
    for total_tokens, query_len, groups_per_program in shapes:
        num_reqs = total_tokens // query_len
        blocks_per_req = (query_len + block_size - 1) // block_size
        num_blocks = num_reqs * blocks_per_req
        qv = torch.zeros(
            (total_tokens, num_heads + 2 * num_kv_heads, 128),
            dtype=dtype,
            device=device,
        )
        query = qv[:, :num_heads]
        key = torch.zeros(
            (total_tokens, num_kv_heads, 128), dtype=dtype, device=device
        )
        value = qv[:, num_heads + num_kv_heads :]
        output = torch.empty_like(query)
        cache = torch.empty(
            (num_blocks, layout.block_i32),
            dtype=torch.int32,
            device=device,
        )
        block_ids = torch.arange(num_blocks, dtype=torch.int32, device=device)
        block_ids = block_ids.reshape(num_reqs, blocks_per_req)
        token_positions = torch.arange(query_len, dtype=torch.int32, device=device)
        slots = (
            block_ids[:, :1] * block_size + token_positions[None, :]
        ).reshape(-1)
        query_start_loc = torch.arange(
            0,
            total_tokens + 1,
            query_len,
            dtype=torch.int32,
            device=device,
        )
        if _beyond_is_global_first_rank():
            print(
                "[beyond] prewarming rounded-prefill kernels "
                f"(tokens={total_tokens}, query_len={query_len}, "
                f"groups_per_program={groups_per_program}, "
                f"block_size={block_size})",
                file=sys.stderr,
            )
        _run_production_prefill_forward(
            query=query,
            key=key,
            value=value,
            output=output,
            cache=cache,
            layout=layout,
            block_table=block_ids,
            slot_mapping=slots,
            query_start_loc=query_start_loc,
            q_starts=list(range(0, total_tokens + 1, query_len)),
            seq_lens=[query_len] * num_reqs,
            prefill_reqs=list(range(num_reqs)),
            q_points_k=q_points_k,
            q_points_v=q_points_v,
            thresholds_k=thresholds_k,
            thresholds_v=thresholds_v,
            softmax_scale=float(first_impl.scale),
            causal=True,
            alibi_slopes=first_impl.alibi_slopes,
            sliding_window=first_impl.sliding_window,
            logits_soft_cap=float(first_impl.logits_soft_cap),
            fa_version=int(first_impl.vllm_flash_attn_version),
            sinks=first_impl.sinks,
        )
    (
        chunked_writer_cases,
        readback_cases,
        chunked_fa4_cases,
    ) = _beyond_prewarm_chunked_prefill_kernels(
        runner=runner,
        first_impl=first_impl,
        dtype=dtype,
        layout=layout,
        q_points_k=q_points_k,
        thresholds_k=thresholds_k,
        q_points_v=q_points_v,
        thresholds_v=thresholds_v,
    )
    _beyond_prewarm_slot_mapping(runner)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    runner._beyond_packed_prefill_kernels_prewarmed = {
        "block_size": block_size,
        "rounded_prefill": tuple(shapes),
        "chunked_writer": tuple(chunked_writer_cases),
        "readback": tuple(readback_cases),
        "chunked_fa4": tuple(chunked_fa4_cases),
    }


def _beyond_prewarm_full_decode_desc(runner, desc) -> bool:
    if (
        _env_int(
            "BEYOND_PREWARM_FULL_DECODE_GRAPH",
            BEYOND_PREWARM_FULL_DECODE_GRAPH,
        )
        <= 0
    ):
        return False
    if not bool(getattr(desc, "uniform", False)):
        return False
    try:
        from vllm.config import CUDAGraphMode
    except Exception:
        return False
    profile_seq_lens = _beyond_full_decode_profile_seq_lens(runner, desc)
    if _beyond_full_decode_prewarmed(runner, desc, profile_seq_lens):
        return True

    num_tokens = int(getattr(desc, "num_tokens", 1))
    allow_microbatching = _beyond_full_decode_allow_microbatching(runner, desc)
    if _beyond_is_global_first_rank():
        print(
            "[beyond] prewarming FULL decode CUDA graph kernel "
            f"before capture (batch={num_tokens}, "
            f"profile_seq_lens={int(profile_seq_lens or 0)})",
            file=sys.stderr,
        )
    runner._dummy_run(
        num_tokens,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        force_attention=True,
        uniform_decode=True,
        allow_microbatching=allow_microbatching,
        skip_eplb=True,
        remove_lora=False,
        num_active_loras=int(getattr(desc, "num_active_loras", 0)),
        profile_seq_lens=profile_seq_lens,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _beyond_mark_full_decode_prewarmed(runner, desc, profile_seq_lens)
    return True


def _beyond_prewarm_full_decode_graphs(runner) -> None:
    if (
        _env_int(
            "BEYOND_PREWARM_FULL_DECODE_GRAPH",
            BEYOND_PREWARM_FULL_DECODE_GRAPH,
        )
        <= 0
    ):
        return
    if (
        _env_int(
            "BEYOND_ENABLE_CUDAGRAPH_DECODE",
            BEYOND_ENABLE_CUDAGRAPH_DECODE,
        )
        <= 0
    ):
        return
    dispatcher = getattr(runner, "cudagraph_dispatcher", None)
    if dispatcher is None:
        return
    try:
        from vllm.config import CUDAGraphMode
    except Exception:
        return
    for mode, descs in dispatcher.get_capture_descs():
        if mode != CUDAGraphMode.FULL:
            continue
        for desc in descs:
            _beyond_prewarm_full_decode_desc(runner, desc)


def _beyond_graph_workspace_reserve(runner) -> _BeyondGraphWorkspaceReserve:
    override_mb = _as_positive_int(BEYOND_CUDAGRAPH_WORKSPACE_RESERVE_MB)
    if override_mb is not None:
        override_bytes = int(override_mb) << 20
        return _BeyondGraphWorkspaceReserve(
            total_bytes=override_bytes,
            override_bytes=override_bytes,
        )
    if BEYOND_ENABLE_PIECEWISE_GRAPH <= 0:
        return _BeyondGraphWorkspaceReserve()
    requested = _as_positive_int(
        getattr(runner.compilation_config, "max_cudagraph_capture_size", None)
    )
    if requested is None:
        return _BeyondGraphWorkspaceReserve()
    decode_B = int(_as_positive_int(BEYOND_FULL_DECODE_MAX_BATCH) or 0)
    if decode_B <= 0:
        max_num_seqs = _as_positive_int(getattr(runner.scheduler_config, "max_num_seqs", None))
        if max_num_seqs is None:
            max_num_seqs = 1
        decode_B = min(requested, max_num_seqs, 16)
    H_q, H_kv, D, dtype_bytes = _beyond_runner_attention_shape(runner)
    max_num_tokens = _as_positive_int(
        getattr(runner.scheduler_config, "max_num_batched_tokens", None)
    )
    if max_num_tokens is None:
        max_num_tokens = requested
    scheduler_token_cap = max(1, int(max_num_tokens))
    workspace_bytes = 0

    activation_override_mb = _as_positive_int(BEYOND_INDUCTOR_ACTIVATION_RESERVE_MB)
    if activation_override_mb is not None:
        activation_bytes = int(activation_override_mb) << 20
    else:
        activation_width = _beyond_runner_activation_peak_width(runner, H_q, H_kv, D)
        # Reserve the model-shaped activation peak for uncaptured decode/model
        # work. CUDA graph pool memory is covered by the fast CUSTOM estimate.
        activation_bytes = int(scheduler_token_cap) * int(activation_width) * int(dtype_bytes)

    largest_temp_bytes = (
        int(scheduler_token_cap)
        * int(_beyond_runner_largest_temp_width(runner, H_q, H_kv, D))
        * int(dtype_bytes)
    )
    hidden_pre_temp_bytes = (
        int(scheduler_token_cap)
        * int(_beyond_runner_hidden_size(runner, H_q, D))
        * int(dtype_bytes)
    )
    intermediate_temp_bytes = (
        int(scheduler_token_cap)
        * int(_beyond_runner_intermediate_size(runner, H_q, D))
        * int(dtype_bytes)
    )
    next_temp_guard_bytes = max(
        int(hidden_pre_temp_bytes),
        int(intermediate_temp_bytes),
    )
    # The compiled Llama body can keep the gate/up temp live while allocating
    # a second intermediate-sized MLP temp, then immediately request another
    # hidden/intermediate temp for the down-proj / following layer.  Keep that
    # chain as explicit post-KV headroom so the next contiguous request is not
    # squeezed out by KV block rounding.
    contiguous_headroom_bytes = (
        int(largest_temp_bytes) + int(intermediate_temp_bytes) + int(next_temp_guard_bytes)
    )
    # Even after the live MLP temp chain is covered, vLLM may round KV cache
    # blocks up to the remaining budget.  Keep one more largest model-shaped
    # allocation outside the chain; this is intentionally at least one reserve
    # quantum so the tail guard cannot be swallowed by the later alignment.
    tail_guard_bytes = max(
        int(largest_temp_bytes),
        int(next_temp_guard_bytes),
        256 << 20,
    )
    # The CUSTOM path uses a fast graph-pool estimate instead of vLLM's exact
    # two-capture profiler.  This reserve covers Beyond workspaces plus
    # uncaptured Inductor temporaries after the real KV cache is created.
    # Leave a small allocator margin that scales with the estimated peak so we
    # do not need to keep hand-tuning per-model MB values.
    slack_mb = _as_positive_int(BEYOND_CUDAGRAPH_KV_SAFETY_MARGIN_MB)
    if slack_mb is not None:
        allocator_slack_bytes = int(slack_mb) << 20
    else:
        subtotal = int(workspace_bytes) + int(activation_bytes)
        allocator_fragmentation_bytes = max(
            512 << 20,
            subtotal // 10,
            max(0, int(decode_B) - 64) * (16 << 20),
        )
        allocator_slack_bytes = (
            int(allocator_fragmentation_bytes)
            + int(largest_temp_bytes)
            + int(contiguous_headroom_bytes)
            + int(tail_guard_bytes)
        )

    total_bytes = int(workspace_bytes) + int(activation_bytes) + int(allocator_slack_bytes)
    if total_bytes > 0 and largest_temp_bytes > 0:
        # KV cache allocation happens in coarse block units while Inductor
        # later asks for one contiguous temp buffer.  Round the reserve up by a
        # model-shaped temp quantum so the final free segment is not a few MiB
        # smaller than the next MLP/QKV allocation.
        reserve_quantum = max(256 << 20, int(largest_temp_bytes))
        aligned_total = (
            (int(total_bytes) + reserve_quantum - 1) // reserve_quantum * reserve_quantum
        )
        allocator_slack_bytes += int(aligned_total) - int(total_bytes)
        total_bytes = int(aligned_total)
    if total_bytes > 0:
        total_bytes = max(total_bytes, 512 << 20)
    return _BeyondGraphWorkspaceReserve(
        workspace_bytes=int(workspace_bytes),
        activation_bytes=int(activation_bytes),
        allocator_slack_bytes=int(allocator_slack_bytes),
        largest_temp_bytes=int(largest_temp_bytes),
        intermediate_temp_bytes=int(intermediate_temp_bytes),
        next_temp_guard_bytes=int(next_temp_guard_bytes),
        contiguous_headroom_bytes=int(contiguous_headroom_bytes),
        tail_guard_bytes=int(tail_guard_bytes),
        total_bytes=int(total_bytes),
    )


def _is_beyond_custom_backend(value) -> bool:
    if value is None:
        return False
    name = getattr(value, "name", None)
    if name is not None and str(name).upper() == "CUSTOM":
        return True
    value_str = str(value).upper()
    return value_str == "CUSTOM" or value_str.endswith(".CUSTOM")


def _beyond_default_seq_cap(
    max_num_seqs: int,
    *,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    block_size: int | None = None,
) -> int:
    explicit_cap = _as_positive_int(os.environ.get("BEYOND_MAX_NUM_SEQS_CAP"))
    if explicit_cap is not None:
        return int(explicit_cap)
    token_cap = _as_positive_int(max_num_batched_tokens)
    model_len = _as_positive_int(max_model_len)
    block = _as_positive_int(block_size) or BEYOND_K_GROUP_SIZE
    if token_cap is None or model_len is None:
        return max(1, min(int(max_num_seqs), 64))

    long_prefill_len = max(1, int(model_len) - int(block))
    long_prefill_batch = (int(token_cap) + int(long_prefill_len) - 1) // int(long_prefill_len)
    mid_prefill_len = max(int(block), (int(model_len) + 1) // 2)
    mid_prefill_batch = (int(token_cap) + int(mid_prefill_len) - 1) // int(mid_prefill_len)
    derived_cap = max(
        1,
        min(
            64,
            max(
                32,
                4 * int(long_prefill_batch),
                4 * int(mid_prefill_batch),
            ),
        ),
    )
    return max(1, min(int(max_num_seqs), int(derived_cap)))


def _beyond_set_default_env_int(name: str, value: int | None) -> None:
    if value is None:
        return
    value = _as_positive_int(value)
    if value is None:
        return
    if _as_positive_int(os.environ.get(name)) is None:
        os.environ[name] = str(int(value))


def _beyond_apply_moe_dtype_policy(vllm_config) -> bool:
    """Route FP16 MoE models around vLLM's BF16-only TRT-LLM backend.

    vLLM 0.21's unquantized MoE oracle currently accepts the TRT-LLM backend
    during config selection for an FP16 Qwen model, then rejects the weights
    during layout conversion because that backend requires BF16.  FlashInfer
    CUTLASS is the next native SM103 backend in vLLM's priority order. Respect
    every explicit user selection and leave BF16 on the TRT-LLM path.
    """
    model_config = getattr(vllm_config, "model_config", None)
    kernel_config = getattr(vllm_config, "kernel_config", None)
    if model_config is None or kernel_config is None:
        return False
    if getattr(model_config, "dtype", None) != torch.float16:
        return False
    if not _beyond_moe_requires_padded_full_decode(vllm_config):
        return False
    current = str(getattr(kernel_config, "moe_backend", "auto"))
    current = current.strip().lower().replace("-", "_")
    if current != "auto":
        return False
    kernel_config.moe_backend = "flashinfer_cutlass"
    print(
        "[beyond] CUSTOM FP16 MoE uses FlashInfer CUTLASS; "
        "the TRT-LLM unquantized backend requires BF16 weights",
        file=sys.stderr,
    )
    return True


def _beyond_apply_scheduler_policy(vllm_config, *, explicit_max_num_seqs: bool) -> None:
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if scheduler_config is None:
        return
    current_max_seqs = _as_positive_int(getattr(scheduler_config, "max_num_seqs", None))
    if current_max_seqs is None:
        return
    token_cap = _as_positive_int(getattr(scheduler_config, "max_num_batched_tokens", None))
    cache_config = getattr(vllm_config, "cache_config", None)
    block_size = _as_positive_int(getattr(cache_config, "block_size", None))
    model_config = getattr(vllm_config, "model_config", None)
    max_model_len = _as_positive_int(getattr(model_config, "max_model_len", None))
    explicit_env_cap = _as_positive_int(os.environ.get("BEYOND_MAX_NUM_SEQS_CAP"))
    if explicit_max_num_seqs and explicit_env_cap is None:
        # ``max_num_seqs`` is both a serving-capacity contract and the upper
        # FULL CUDA-graph batch.  The packed cache has materially more room
        # than dense KV, so silently replacing a caller's explicit value with
        # the conservative startup default can discard most of that capacity.
        # Keep automatic vLLM defaults bounded, while respecting an explicit
        # engine/CLI setting unless the caller also supplies an explicit
        # Beyond cap.
        seq_cap = int(current_max_seqs)
    else:
        seq_cap = _beyond_default_seq_cap(
            int(current_max_seqs),
            max_num_batched_tokens=token_cap,
            max_model_len=max_model_len,
            block_size=block_size,
        )
    effective_max_seqs = int(current_max_seqs)
    if int(current_max_seqs) > int(seq_cap):
        scheduler_config.max_num_seqs = int(seq_cap)
        effective_max_seqs = int(seq_cap)
        source = "explicit" if explicit_max_num_seqs else "vLLM default"
        print(
            "[beyond] CUSTOM scheduler max_num_seqs="
            f"{effective_max_seqs} "
            f"({source}={current_max_seqs}, "
            f"max_num_batched_tokens={token_cap}, "
            f"max_model_len={max_model_len}; override with "
            "BEYOND_MAX_NUM_SEQS_CAP)",
            file=sys.stderr,
        )

    _beyond_set_default_env_int("BEYOND_SCHEDULER_TOKEN_CAP", token_cap)
    _beyond_set_default_env_int("BEYOND_BLOCK_SIZE_CAP", block_size)
    _beyond_set_default_env_int("BEYOND_MODEL_MAX_LEN_CAP", max_model_len)

    # These globals are read by the decode graph dispatch/profiling patches.
    # Keep the default tied to the final scheduler sequence cap so CUSTOM
    # callers get the original full-decode graph shape unless they override it.
    global BEYOND_FULL_DECODE_MAX_BATCH
    if _as_positive_int(BEYOND_FULL_DECODE_MAX_BATCH) is None:
        BEYOND_FULL_DECODE_MAX_BATCH = int(effective_max_seqs)
        _beyond_set_default_env_int(
            "BEYOND_FULL_DECODE_MAX_BATCH",
            int(effective_max_seqs),
        )


def _beyond_apply_high_concurrency_cudagraph_policy(
    vllm_config,
    *,
    explicit_max_num_seqs: bool,
    explicit_cudagraph_roster: bool,
) -> bool:
    """Add sparse FULL-graph anchors above vLLM's default B512 ceiling.

    vLLM 0.21 stops its automatic capture roster at B512 even when a
    throughput caller explicitly requests more concurrent sequences.  In that
    case B513+ decode falls back out of the FULL graph.  Keep vLLM's dense
    low-batch roster intact and add only 128-request anchors through the
    validated B1024 limit.  Explicit graph settings and non-throughput engines
    remain entirely caller-controlled.
    """
    if not explicit_max_num_seqs or explicit_cudagraph_roster:
        return False
    performance_mode = str(getattr(vllm_config, "performance_mode", "balanced"))
    if not performance_mode.strip().lower().endswith("throughput"):
        return False

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if scheduler_config is None or compilation_config is None:
        return False
    max_num_seqs = _as_positive_int(
        getattr(scheduler_config, "max_num_seqs", None)
    )
    capture_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
    if max_num_seqs is None or not capture_sizes:
        return False

    selected_max = min(int(max_num_seqs), 1024)
    if selected_max <= 512:
        return False
    normalized = sorted({int(size) for size in capture_sizes if int(size) > 0})
    if not normalized or normalized[-1] >= selected_max:
        return False

    first_anchor = ((max(512, normalized[-1]) // 128) + 1) * 128
    high_anchors = list(range(first_anchor, selected_max + 1, 128))
    if not high_anchors or high_anchors[-1] != selected_max:
        high_anchors.append(selected_max)
    selected = sorted(set(normalized).union(high_anchors))
    compilation_config.cudagraph_capture_sizes = selected
    compilation_config.max_cudagraph_capture_size = selected[-1]
    print(
        "[beyond] extending throughput CUDA graphs above B512 with sparse "
        f"anchors {high_anchors}",
        file=sys.stderr,
    )
    return True


def _beyond_prefill_priority_enabled(scheduler) -> bool:
    """Select prefill-first scheduling only for explicit throughput service."""
    configured = os.environ.get("BEYOND_PREFILL_PRIORITY_SCHEDULER")
    if configured is not None:
        return _env_bool("BEYOND_PREFILL_PRIORITY_SCHEDULER", False)
    vllm_config = getattr(scheduler, "vllm_config", None)
    performance_mode = str(
        getattr(vllm_config, "performance_mode", "balanced")
    ).strip().lower()
    return performance_mode == "throughput"


def _beyond_can_prioritize_waiting_prefill(scheduler) -> bool:
    """Return whether one scheduler step may defer decode-only requests."""
    if not _beyond_prefill_priority_enabled(scheduler):
        return False
    cache_config = getattr(scheduler, "cache_config", None)
    if bool(getattr(cache_config, "enable_prefix_caching", False)):
        # A cache hit already reduces the prompt to a tiny suffix.  Deferring
        # decodes cannot remove a large chunked-prefill readback in that case,
        # and the stock scheduler is measurably faster for the steady-state
        # repeated-prefix workload.
        return False
    if not scheduler.running or not (scheduler.waiting or scheduler.skipped_waiting):
        return False
    remaining_slots = int(scheduler.max_num_running_reqs) - len(scheduler.running)
    if remaining_slots <= 0:
        return False
    for request in scheduler.running:
        pending = (
            int(request.num_tokens_with_spec)
            + int(request.num_output_placeholders)
            - int(request.num_computed_tokens)
        )
        if pending > 1 or request.spec_token_ids or request.has_encoder_inputs:
            return False
    return True


def _patch_vllm_beyond_prefill_priority_scheduler() -> None:
    """Optionally schedule waiting prefills before decode-only requests.

    This is an E2E throughput candidate.  It preserves vLLM's scheduler body
    and state transitions, but temporarily withholds decode-only requests for
    one step so a waiting prompt can use the full token budget instead of
    creating a nearly-full chunk plus an expensive tiny tail.
    """
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception:
        return
    if getattr(Scheduler.schedule, "_beyond_prefill_priority_patched", False):
        return

    original_schedule = Scheduler.schedule

    def _schedule_with_prefill_priority(self):
        if not _beyond_can_prioritize_waiting_prefill(self):
            return original_schedule(self)

        hidden_running = self.running
        original_max_running = int(self.max_num_running_reqs)
        self.running = []
        self.max_num_running_reqs = original_max_running - len(hidden_running)
        try:
            output = original_schedule(self)
            newly_running = self.running
        finally:
            # Preserve FCFS order among pre-existing decodes. Newly admitted
            # prefills follow them in persistent scheduler state, while only
            # the prefills are present in this step's SchedulerOutput.
            self.running = hidden_running + self.running
            self.max_num_running_reqs = original_max_running
        if len(self.running) > original_max_running:
            raise RuntimeError("Beyond prefill-priority scheduler exceeded max_num_seqs")
        if newly_running and _beyond_is_global_first_rank() and not getattr(
            self, "_beyond_prefill_priority_reported", False
        ):
            print(
                "[beyond] throughput scheduler is giving waiting prefills the "
                "full token budget before decode-only requests",
                file=sys.stderr,
            )
            self._beyond_prefill_priority_reported = True
        return output

    _schedule_with_prefill_priority._beyond_prefill_priority_patched = True
    Scheduler.schedule = _schedule_with_prefill_priority


def _patch_vllm_beyond_scheduler_policy() -> None:
    try:
        from vllm.engine.arg_utils import EngineArgs
    except Exception:
        return
    if getattr(EngineArgs.create_engine_config, "_beyond_policy_patched", False):
        return

    _orig_create_engine_config = EngineArgs.create_engine_config

    def _create_engine_config_with_beyond_policy(self, *args, **kwargs):
        explicit_max_num_seqs = getattr(self, "max_num_seqs", None) is not None
        raw_compilation_config = getattr(self, "compilation_config", None)
        explicit_cudagraph_roster = bool(
            getattr(self, "cudagraph_capture_sizes", None) is not None
            or getattr(self, "max_cudagraph_capture_size", None) is not None
            or (
                isinstance(raw_compilation_config, dict)
                and (
                    raw_compilation_config.get("cudagraph_capture_sizes") is not None
                    or raw_compilation_config.get("max_cudagraph_capture_size") is not None
                )
            )
            or (
                raw_compilation_config is not None
                and not isinstance(raw_compilation_config, dict)
                and (
                    getattr(
                        raw_compilation_config,
                        "cudagraph_capture_sizes",
                        None,
                    )
                    is not None
                    or getattr(
                        raw_compilation_config,
                        "max_cudagraph_capture_size",
                        None,
                    )
                    is not None
                )
            )
        )
        vllm_config = _orig_create_engine_config(self, *args, **kwargs)
        if _is_beyond_custom_backend(
            getattr(self, "attention_backend", None)
        ) or _is_beyond_custom_backend(os.environ.get("VLLM_ATTENTION_BACKEND")):
            _beyond_apply_moe_dtype_policy(vllm_config)
            _beyond_apply_scheduler_policy(
                vllm_config,
                explicit_max_num_seqs=explicit_max_num_seqs,
            )
            _beyond_apply_high_concurrency_cudagraph_policy(
                vllm_config,
                explicit_max_num_seqs=explicit_max_num_seqs,
                explicit_cudagraph_roster=explicit_cudagraph_roster,
            )
        return vllm_config

    _create_engine_config_with_beyond_policy._beyond_policy_patched = True
    EngineArgs.create_engine_config = _create_engine_config_with_beyond_policy
    _patch_vllm_beyond_prefill_priority_scheduler()


def _warm_cuda_libraries_for_aot_replay() -> None:
    """Initialize CUDA libraries before replaying a deserialized AOT graph.

    On Blackwell TP workers, a freshly deserialized AOT function can reach an
    inductor extern-kernel matmul before the process has created cuBLAS handles
    on the worker's CUDA device.  Fresh compilation happens to initialize that
    state earlier; make the loaded-AOT path explicit as well.
    """
    if _env_int("BEYOND_ENABLE_AOT_CUBLAS_WARMUP", 1) <= 0:
        return
    if not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    if device in _AOT_CUBLAS_WARMED_DEVICES:
        return
    with torch.inference_mode():
        a = torch.empty((1, 1), device=device, dtype=torch.float16)
        b = torch.empty((1, 1), device=device, dtype=torch.float16)
        torch.mm(a, b)
        if torch.cuda.is_bf16_supported():
            a_bf16 = torch.empty((1, 1), device=device, dtype=torch.bfloat16)
            b_bf16 = torch.empty((1, 1), device=device, dtype=torch.bfloat16)
            torch.mm(a_bf16, b_bf16)
    _AOT_CUBLAS_WARMED_DEVICES.add(device)


def _patch_vllm_copy_and_call_static_buffers() -> None:
    try:
        import vllm.compilation.backends as backends
    except Exception:
        return
    if getattr(backends.make_copy_and_call, "_beyond_static_buffer_patched", False):
        return

    def _make_copy_and_call_with_fresh_buffers(
        sym_tensor_indices: list[int],
        input_buffers: list[torch.Tensor | None],
        callable_fn,
    ):
        def copy_and_call(*args):
            list_args = list(args)
            for i, index in enumerate(sym_tensor_indices):
                runtime_tensor = list_args[index]
                if not torch.is_tensor(runtime_tensor):
                    continue
                runtime_shape = runtime_tensor.shape[0]
                static_tensor = input_buffers[i]
                needs_new_buffer = (
                    static_tensor is None
                    or not torch.is_tensor(static_tensor)
                    or static_tensor.device != runtime_tensor.device
                    or static_tensor.dtype != runtime_tensor.dtype
                    or static_tensor.dim() != runtime_tensor.dim()
                    or static_tensor.shape[0] < runtime_shape
                    or tuple(static_tensor.shape[1:]) != tuple(runtime_tensor.shape[1:])
                    or tuple(static_tensor.stride()[1:]) != tuple(runtime_tensor.stride()[1:])
                )
                if needs_new_buffer:
                    static_tensor = torch.empty_strided(
                        tuple(runtime_tensor.shape),
                        tuple(runtime_tensor.stride()),
                        device=runtime_tensor.device,
                        dtype=runtime_tensor.dtype,
                    )
                    input_buffers[i] = static_tensor
                static_slice = static_tensor[:runtime_shape]
                static_slice.copy_(runtime_tensor)
                list_args[index] = static_slice
            return callable_fn(*list_args)

        return copy_and_call

    _make_copy_and_call_with_fresh_buffers._beyond_static_buffer_patched = True
    backends.make_copy_and_call = _make_copy_and_call_with_fresh_buffers


def _patch_vllm_aot_replay_warmup() -> None:
    try:
        import vllm.compilation.decorators as decorators
    except Exception:
        return
    if getattr(
        decorators._try_load_aot_compiled_fn,
        "_beyond_aot_cublas_warmup_patched",
        False,
    ):
        return

    _orig_try_load_aot_compiled_fn = decorators._try_load_aot_compiled_fn

    def _try_load_aot_compiled_fn_with_beyond_warmup(*args, **kwargs):
        loaded_fn = _orig_try_load_aot_compiled_fn(*args, **kwargs)
        if loaded_fn is not None:
            _warm_cuda_libraries_for_aot_replay()
        return loaded_fn

    _try_load_aot_compiled_fn_with_beyond_warmup._beyond_aot_cublas_warmup_patched = True
    decorators._try_load_aot_compiled_fn = _try_load_aot_compiled_fn_with_beyond_warmup


def _patch_vllm_decode_graph_buckets() -> None:
    """Constrain CUSTOM full-decode graphs by batch bucket when requested."""
    try:
        from vllm.config import CUDAGraphMode
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        return

    _patch_vllm_slot_mapping()

    if not getattr(
        GPUModelRunner._determine_batch_execution_and_padding,
        "_beyond_prefix_tail_patched",
        False,
    ):
        _orig_determine_batch_execution_and_padding = (
            GPUModelRunner._determine_batch_execution_and_padding
        )

        def _determine_batch_execution_and_padding_with_prefix_boundary(
            self,
            num_tokens,
            num_reqs,
            num_scheduled_tokens_np,
            max_num_scheduled_tokens,
            use_cascade_attn,
            allow_microbatching=True,
            force_eager=False,
            force_uniform_decode=None,
            force_has_lora=None,
            force_num_active_loras=None,
            num_encoder_reqs=0,
        ):
            # vLLM's generic uniform-decode predicate only sees one scheduled
            # token per request. A chunked/prefix-cached prompt can have the
            # same shape for its final prompt token, but it still requires the
            # prefill metadata path. Keep precisely that boundary piecewise;
            # subsequent generated tokens retain FULL CUDA Graph dispatch.
            if force_uniform_decode is None and int(max_num_scheduled_tokens) == 1:
                if _beyond_has_pending_prompt_tokens(self.input_batch, int(num_reqs)):
                    force_uniform_decode = False
                    _runtime_audit_event("prefix_tail_piecewise_calls")
            return _orig_determine_batch_execution_and_padding(
                self,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=use_cascade_attn,
                allow_microbatching=allow_microbatching,
                force_eager=force_eager,
                force_uniform_decode=force_uniform_decode,
                force_has_lora=force_has_lora,
                force_num_active_loras=force_num_active_loras,
                num_encoder_reqs=num_encoder_reqs,
            )

        _determine_batch_execution_and_padding_with_prefix_boundary._beyond_prefix_tail_patched = (
            True
        )
        GPUModelRunner._determine_batch_execution_and_padding = (
            _determine_batch_execution_and_padding_with_prefix_boundary
        )

    if not getattr(GPUModelRunner.capture_model, "_beyond_prewarm_patched", False):
        _orig_capture_model = GPUModelRunner.capture_model

        def _capture_model_with_beyond_prewarm(self, *args, **kwargs):
            _beyond_prewarm_packed_prefill_kernels(self)
            _beyond_prewarm_full_decode_graphs(self)
            graph_size = _orig_capture_model(self, *args, **kwargs)
            return graph_size

        _capture_model_with_beyond_prewarm._beyond_prewarm_patched = True
        GPUModelRunner.capture_model = _capture_model_with_beyond_prewarm

    if not getattr(
        GPUModelRunner._warmup_and_capture,
        "_beyond_full_decode_profile_patched",
        False,
    ):
        _orig_warmup_and_capture = GPUModelRunner._warmup_and_capture

        def _warmup_and_capture_with_beyond_full_decode_profile(
            self,
            desc,
            cudagraph_runtime_mode,
            profile_seq_lens=None,
            allow_microbatching=False,
            num_warmups=None,
        ):
            if (
                cudagraph_runtime_mode == CUDAGraphMode.FULL
                and bool(getattr(desc, "uniform", False))
                and _env_int(
                    "BEYOND_ENABLE_CUDAGRAPH_DECODE",
                    BEYOND_ENABLE_CUDAGRAPH_DECODE,
                )
                > 0
            ):
                beyond_profile_seq_lens = _beyond_full_decode_profile_seq_lens(
                    self,
                    desc,
                )
                if beyond_profile_seq_lens is not None:
                    profile_seq_lens = int(beyond_profile_seq_lens)
                    if not _beyond_full_decode_prewarmed(
                        self,
                        desc,
                        profile_seq_lens,
                    ):
                        _beyond_prewarm_full_decode_desc(self, desc)
            return _orig_warmup_and_capture(
                self,
                desc,
                cudagraph_runtime_mode,
                profile_seq_lens=profile_seq_lens,
                allow_microbatching=allow_microbatching,
                num_warmups=num_warmups,
            )

        _warmup_and_capture_with_beyond_full_decode_profile._beyond_full_decode_profile_patched = (
            True
        )
        GPUModelRunner._warmup_and_capture = _warmup_and_capture_with_beyond_full_decode_profile

    if not getattr(
        GPUModelRunner.profile_cudagraph_memory,
        "_beyond_workspace_reserve_patched",
        False,
    ):

        def _profile_cudagraph_memory_with_beyond_reserve(self, *args, **kwargs):
            _force_cleanup_profiling_kv_cache(self)
            _clear_beyond_tensor_workspaces()
            graph_pool_bytes = _beyond_fast_cudagraph_pool_estimate_bytes(self)
            reserve_estimate = _beyond_graph_workspace_reserve(self)
            reserve = int(reserve_estimate.total_bytes)
            estimate = int(graph_pool_bytes) + int(reserve)
            if estimate > 0:
                if not getattr(self, "_beyond_workspace_reserve_logged", False):
                    print(
                        "[beyond] fast graph memory reserve "
                        f"{estimate / (1 << 30):.2f} GiB "
                        f"(graph_pool={graph_pool_bytes / (1 << 30):.2f}, "
                        f"workspace={reserve_estimate.workspace_bytes / (1 << 30):.2f}, "
                        "activation="
                        f"{reserve_estimate.activation_bytes / (1 << 30):.2f}, "
                        "largest_temp="
                        f"{reserve_estimate.largest_temp_bytes / (1 << 30):.2f}, "
                        "intermediate_temp="
                        f"{reserve_estimate.intermediate_temp_bytes / (1 << 30):.2f}, "
                        "next_guard="
                        f"{reserve_estimate.next_temp_guard_bytes / (1 << 30):.2f}, "
                        "headroom="
                        f"{reserve_estimate.contiguous_headroom_bytes / (1 << 30):.2f}, "
                        "tail_guard="
                        f"{reserve_estimate.tail_guard_bytes / (1 << 30):.2f}, "
                        "slack="
                        f"{reserve_estimate.allocator_slack_bytes / (1 << 30):.2f})",
                        file=sys.stderr,
                    )
                    self._beyond_workspace_reserve_logged = True
            return estimate

        _profile_cudagraph_memory_with_beyond_reserve._beyond_workspace_reserve_patched = True
        GPUModelRunner.profile_cudagraph_memory = _profile_cudagraph_memory_with_beyond_reserve

    if not getattr(CudagraphDispatcher.dispatch, "_beyond_full_batch_patched", False):
        _orig_dispatch = CudagraphDispatcher.dispatch

        def _dispatch_with_full_batch_cap(
            self,
            num_tokens: int,
            uniform_decode: bool = False,
            has_lora: bool = False,
            num_active_loras: int = 0,
            valid_modes=None,
            invalid_modes=None,
        ):
            mode, batch_desc = _orig_dispatch(
                self,
                num_tokens,
                uniform_decode=uniform_decode,
                has_lora=has_lora,
                num_active_loras=num_active_loras,
                valid_modes=valid_modes,
                invalid_modes=invalid_modes,
            )
            full_decode_batch_cap = _as_positive_int(BEYOND_FULL_DECODE_MAX_BATCH)
            if (
                full_decode_batch_cap is None
                or mode == CUDAGraphMode.FULL
                or not uniform_decode
                or has_lora
                or int(num_tokens) > int(full_decode_batch_cap)
            ):
                return mode, batch_desc
            allowed_modes = set(valid_modes or CUDAGraphMode.valid_runtime_modes())
            if invalid_modes:
                allowed_modes -= set(invalid_modes)
            if CUDAGraphMode.FULL not in allowed_modes:
                return mode, batch_desc
            candidates = [
                key
                for key in self.cudagraph_keys.get(CUDAGraphMode.FULL, ())
                if key.uniform
                and not key.has_lora
                and int(getattr(key, "num_tokens", 0)) >= int(num_tokens)
                and int(getattr(key, "num_tokens", 0)) <= int(full_decode_batch_cap)
            ]
            if not candidates:
                return mode, batch_desc
            return CUDAGraphMode.FULL, min(
                candidates,
                key=lambda key: int(getattr(key, "num_tokens", 0)),
            )

        _dispatch_with_full_batch_cap._beyond_full_batch_patched = True
        CudagraphDispatcher.dispatch = _dispatch_with_full_batch_cap

    if not getattr(
        CudagraphDispatcher.initialize_cudagraph_keys,
        "_beyond_full_batch_patched",
        False,
    ):
        _orig_init_keys = CudagraphDispatcher.initialize_cudagraph_keys

        def _init_keys_with_full_batch_cap(
            self,
            cudagraph_mode,
            uniform_decode_query_len=1,
        ):
            _orig_init_keys(self, cudagraph_mode, uniform_decode_query_len)
            full_decode_batch_cap = _as_positive_int(BEYOND_FULL_DECODE_MAX_BATCH)
            if full_decode_batch_cap is None:
                return
            full_keys = self.cudagraph_keys.get(CUDAGraphMode.FULL)
            if full_keys:
                # Dense models retain every exact batch.  SM103 TRT-LLM MoE
                # cannot capture M=1, so Qwen routes one active request to the
                # next safe graph bucket while retaining exact B>=2 buckets.
                self.cudagraph_keys[CUDAGraphMode.FULL] = (
                    _beyond_select_full_decode_graph_keys(
                        full_keys,
                        full_decode_batch_cap=int(full_decode_batch_cap),
                        pad_single_token=_beyond_moe_requires_padded_full_decode(
                            self.vllm_config
                        ),
                    )
                )
                if (
                    self.cudagraph_keys[CUDAGraphMode.FULL]
                    and _beyond_long_context_moe_uses_bounded_full_graph_profile(
                        self.vllm_config
                    )
                    and _beyond_is_global_first_rank()
                ):
                    print(
                        "[beyond] long-context MoE keeps FULL CUDA graphs with "
                        f"a {int(BEYOND_MOE_FULL_GRAPH_PROFILE_SEQ_LEN):,}-token "
                        "dummy profile; packed attention retains native "
                        "block-table capacity",
                        file=sys.stderr,
                    )

        _init_keys_with_full_batch_cap._beyond_full_batch_patched = True
        CudagraphDispatcher.initialize_cudagraph_keys = _init_keys_with_full_batch_cap


def register_backend() -> None:
    """Register the CUSTOM backend before constructing a vLLM ``LLM``."""
    from beyond.runtime.vllm.registration import register_backend as register

    register()
