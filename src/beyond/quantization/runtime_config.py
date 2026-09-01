"""Load a trained per-layer quant config and expose per-layer tensors.

Training (``analyze/discover.py``) writes ``.pt`` config dicts like::

    {
      "model.layers.0.self_attn.k_proj": {
        "type": "UnifiedQuantLayer_5bit_token_group_tables",
        "proj_type": "k_proj",
        "num_bits": 5,
        "grouping_dim": "token",
        "table_axis": "group",
        "quant_points": Tensor[num_tables, 32],
        "thresholds":   Tensor[num_tables, 31]
      },
      "model.layers.0.self_attn.v_proj": { ... "grouping_dim": "token" ... },
      ...
    }

This module reads such a file once and exposes per-(layer_base, proj) tensors.
Current training writes grouped qtables shaped
``(kv_heads * head_dim / group_size, 2^bits)`` by default.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import torch

from beyond.common.hashing import sha256_file
from beyond.quantization.table_precision import (
    LEGACY_TABLE_PRECISION_ABI,
    TABLE_COMPUTE_DTYPE_NAME,
    TABLE_PRECISION_ABI,
    storage_dtype_name,
    torch_dtype_for_storage,
)


def _config_file_identity(path: str) -> str:
    return f"{os.path.abspath(path)}:{sha256_file(path)}"


def _source_model_from_config(raw: dict, path: str) -> Optional[str]:
    for key in ("__metadata__", "metadata"):
        metadata = raw.get(key)
        if isinstance(metadata, dict):
            value = metadata.get("base_model") or metadata.get("model_id")
            if value:
                return str(value)
    direct = raw.get("base_model")
    if isinstance(direct, str) and direct:
        return direct

    metrics_path = os.path.join(os.path.dirname(os.path.dirname(path)), "metrics.json")
    try:
        with open(metrics_path, "r", encoding="utf-8") as fh:
            metrics = json.load(fh)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    value = metrics.get("base_model") if isinstance(metrics, dict) else None
    return str(value) if value else None


def _normalize_model_id(value: str) -> str:
    return str(value).strip().replace("\\", "/").rstrip("/").lower()


def _model_ids_match(expected: str, actual: str) -> bool:
    expected_norm = _normalize_model_id(expected)
    actual_norm = _normalize_model_id(actual)
    if expected_norm == actual_norm:
        return True
    return expected_norm.rsplit("/", 1)[-1] == actual_norm.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class LayerQuantConfig:
    """Per-projection config. CPU tensors retain their serialized storage dtype."""

    num_bits: int
    grouping_dim: str  # always "token"
    q_points: torch.Tensor  # (2^bits,) or (num_tables, 2^bits)
    thresholds: torch.Tensor  # (2^bits - 1,) or (num_tables, 2^bits - 1)
    table_storage_dtype: str
    table_precision_abi: str
    table_compute_dtype: str = TABLE_COMPUTE_DTYPE_NAME
    table_axis: str = "group"
    group_size: Optional[int] = None


class QuantConfig:
    """Loads a full-model config keyed by ``"{base}.{proj_type}"`` where
    ``base = layer.layer_name`` without the trailing ``.attn`` segment, e.g.
    ``"model.layers.0.self_attn"`` and ``proj_type in {'k_proj', 'v_proj'}``.
    """

    def __init__(
        self,
        per_layer: dict[str, LayerQuantConfig],
        expected_bits: Optional[int],
        *,
        source_path: str,
        source_identity: str,
        source_model: Optional[str],
    ):
        self._d = per_layer
        self.expected_bits = expected_bits
        self.source_path = source_path
        self.identity = source_identity
        self.source_model = source_model

    @classmethod
    def load(
        cls,
        path: str,
        *,
        expected_bits: Optional[int] = None,
        expected_model: Optional[str] = None,
        require_model_metadata: bool = False,
    ) -> "QuantConfig":
        path = os.path.abspath(path)
        source_identity = _config_file_identity(path)
        if str(path).endswith(".pt"):
            try:
                raw = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                raw = torch.load(path, map_location="cpu")
        else:
            with open(path, "r") as f:
                raw = json.load(f)
        metadata = raw.get("__metadata__") if isinstance(raw, dict) else None
        if isinstance(metadata, dict):
            artifact_type = str(metadata.get("artifact_type") or "")
            if (
                artifact_type == "beyond_uncompressed_kv_adapter_config"
                or metadata.get("compressed_kv_cache") is False
                or metadata.get("hard_forward") is False
            ):
                raise ValueError(
                    f"{path}: uncompressed continuous KV adapter artifacts cannot "
                    "be loaded as hard-QDQ compressed quant configs"
                )
        per_layer: dict[str, LayerQuantConfig] = {}
        for key, entry in _iter_projection_entries(raw):
            qp_raw = torch.as_tensor(entry["quant_points"])
            th_raw = torch.as_tensor(entry["thresholds"])
            declared_storage_dtype = entry.get("table_storage_dtype")
            if declared_storage_dtype is None:
                q_storage_dtype = storage_dtype_name(qp_raw)
                t_storage_dtype = storage_dtype_name(th_raw)
                if q_storage_dtype != t_storage_dtype:
                    raise ValueError(
                        f"{key}: legacy q_points/thresholds storage dtypes differ: "
                        f"{q_storage_dtype} vs {t_storage_dtype}"
                    )
                table_storage_dtype = q_storage_dtype
            else:
                table_storage_dtype = str(declared_storage_dtype).strip().lower()
            storage_dtype = torch_dtype_for_storage(table_storage_dtype)
            qp = qp_raw.to(dtype=storage_dtype).contiguous()
            th = th_raw.to(dtype=storage_dtype).contiguous()
            declared_precision_abi = entry.get("table_precision_abi")
            if declared_precision_abi is None:
                if table_storage_dtype == "float16":
                    raise ValueError(
                        f"{key}: float16 tables require an explicit table_precision_abi"
                    )
                table_precision_abi = LEGACY_TABLE_PRECISION_ABI
            else:
                table_precision_abi = str(declared_precision_abi).strip()
            declared_compute_dtype = entry.get("table_compute_dtype")
            table_compute_dtype = (
                str(declared_compute_dtype or TABLE_COMPUTE_DTYPE_NAME).strip().lower()
            )
            if table_compute_dtype != TABLE_COMPUTE_DTYPE_NAME:
                raise ValueError(
                    f"{key}: table_compute_dtype must be {TABLE_COMPUTE_DTYPE_NAME!r}, "
                    f"got {table_compute_dtype!r}"
                )
            if table_storage_dtype == "float16" and table_precision_abi != TABLE_PRECISION_ABI:
                raise ValueError(
                    f"{key}: float16 tables require table_precision_abi={TABLE_PRECISION_ABI!r}"
                )
            if table_precision_abi == TABLE_PRECISION_ABI and table_storage_dtype != "float16":
                raise ValueError(f"{key}: {TABLE_PRECISION_ABI!r} requires float16 table storage")
            if table_precision_abi == TABLE_PRECISION_ABI:
                if declared_storage_dtype is None:
                    raise ValueError(
                        f"{key}: {TABLE_PRECISION_ABI!r} requires explicit "
                        "table_storage_dtype metadata"
                    )
                if declared_compute_dtype is None:
                    raise ValueError(
                        f"{key}: {TABLE_PRECISION_ABI!r} requires explicit "
                        "table_compute_dtype metadata"
                    )
            elif table_precision_abi != LEGACY_TABLE_PRECISION_ABI:
                raise ValueError(f"{key}: unsupported table_precision_abi={table_precision_abi!r}")
            num_bits = int(entry.get("num_bits") or _infer_num_bits(entry))
            grouping_dim = entry.get("grouping_dim") or _infer_grouping_dim(entry)
            grouping_dim = str(grouping_dim).lower()
            if expected_bits is not None and num_bits != expected_bits:
                raise ValueError(
                    f"{key}: num_bits={num_bits} doesn't match expected {expected_bits}"
                )
            if grouping_dim != "token":
                raise ValueError(
                    f"{key}: grouping_dim={grouping_dim!r} is unsupported; "
                    "K/V quantization configs must be token-grouped"
                )
            expected_levels = 1 << num_bits
            if qp.dim() not in (1, 2):
                raise ValueError(
                    f"{key}: q_points must be rank 1 or rank 2, got shape {tuple(qp.shape)}"
                )
            if int(qp.shape[-1]) != expected_levels:
                raise ValueError(
                    f"{key}: q_points levels {qp.shape[-1]} != 2^{num_bits} = {expected_levels}"
                )
            expected_threshold_shape = tuple(qp.shape[:-1]) + (expected_levels - 1,)
            if tuple(th.shape) != expected_threshold_shape:
                raise ValueError(
                    f"{key}: thresholds shape {tuple(th.shape)} does not match "
                    f"expected {expected_threshold_shape}"
                )
            table_axis = str(entry.get("table_axis") or "").lower()
            if not table_axis:
                table_axis = "group" if qp.dim() == 2 else "shared"
            if table_axis in {"per_group", "groups"}:
                table_axis = "group"
            if table_axis not in {"shared", "group"}:
                raise ValueError(f"{key}: unsupported table_axis={table_axis!r}")
            if table_axis == "shared" and qp.dim() != 1:
                raise ValueError(
                    f"{key}: table_axis='shared' requires rank-1 q_points, "
                    f"got shape {tuple(qp.shape)}"
                )
            if table_axis == "group" and qp.dim() != 2:
                raise ValueError(
                    f"{key}: table_axis='group' requires rank-2 q_points, "
                    f"got shape {tuple(qp.shape)}"
                )
            group_size_raw = entry.get("group_size")
            group_size = None if group_size_raw in (None, "") else int(group_size_raw)
            if group_size is not None and group_size <= 0:
                raise ValueError(f"{key}: group_size must be positive, got {group_size}")
            if not torch.isfinite(qp).all():
                raise ValueError(f"{key}: q_points contain NaN or Inf")
            if not torch.isfinite(th).all():
                raise ValueError(f"{key}: thresholds contain NaN or Inf")
            if bool(((th < 0.0) | (th > 1.0)).any()):
                raise ValueError(f"{key}: thresholds must lie in normalized range [0, 1]")
            if int(qp.shape[-1]) > 1 and not bool((qp[..., 1:] > qp[..., :-1]).all()):
                raise ValueError(f"{key}: q_points must be strictly increasing")
            if int(th.shape[-1]) > 1 and not bool((th[..., 1:] > th[..., :-1]).all()):
                raise ValueError(f"{key}: thresholds must be strictly increasing")
            per_layer[key] = LayerQuantConfig(
                num_bits=num_bits,
                grouping_dim=grouping_dim,
                q_points=qp,
                thresholds=th,
                table_storage_dtype=table_storage_dtype,
                table_precision_abi=table_precision_abi,
                table_compute_dtype=table_compute_dtype,
                table_axis=table_axis,
                group_size=group_size,
            )
        if not per_layer:
            raise ValueError(f"{path}: quant config contains no K/V projection entries")
        source_model = _source_model_from_config(raw, path)
        config = cls(
            per_layer=per_layer,
            expected_bits=expected_bits,
            source_path=path,
            source_identity=source_identity,
            source_model=source_model,
        )
        config.validate_model(
            expected_model,
            require_metadata=bool(require_model_metadata),
        )
        return config

    def for_attention_layer(
        self,
        layer_name: str,
        proj_type: str,
    ) -> Optional[LayerQuantConfig]:
        """``layer_name = 'model.layers.0.self_attn.attn'`` -> strip trailing
        ``.attn``; proj_type in {'k_proj', 'v_proj'}."""
        base = layer_name[: -len(".attn")] if layer_name.endswith(".attn") else layer_name
        key = f"{base}.{proj_type}"
        for candidate in _projection_key_candidates(key):
            entry = self._d.get(candidate)
            if entry is not None:
                return entry
        return None

    def __len__(self):
        return len(self._d)

    def items(self):
        return self._d.items()

    def validate_model(
        self,
        expected_model: Optional[str],
        *,
        require_metadata: bool = False,
    ) -> None:
        if not expected_model:
            return
        if not self.source_model:
            if require_metadata:
                raise ValueError(
                    f"{self.source_path}: cannot verify model identity for "
                    f"{expected_model!r}; add base_model metadata or keep the "
                    "config beside its run metrics.json"
                )
            return
        if not _model_ids_match(expected_model, self.source_model):
            raise ValueError(
                f"{self.source_path}: quant config model {self.source_model!r} "
                f"does not match requested model {expected_model!r}"
            )


def _infer_num_bits(entry: dict) -> int:
    # Fallback if num_bits missing: derive from q_points length.
    q = entry["quant_points"]
    if torch.is_tensor(q):
        levels = int(q.shape[-1])
    else:
        first = q[0] if isinstance(q, list) and q and isinstance(q[0], list) else q
        levels = len(first)
    return int.bit_length(levels - 1)


def _iter_projection_entries(raw: dict):
    """Yield flat k_proj/v_proj entries, including discover fused-QKV exports."""
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        if "quant_points" in entry and "thresholds" in entry:
            yield key, entry
            continue

        has_fused = False
        if key.endswith(".query_key_value"):
            base = key[: -len(".query_key_value")]
        elif key.endswith(".qkv_proj"):
            base = key[: -len(".qkv_proj")]
        else:
            base = key
        for nested_name, proj_type in (("k", "k_proj"), ("v", "v_proj")):
            nested = entry.get(nested_name)
            if not isinstance(nested, dict):
                continue
            if "quant_points" not in nested or "thresholds" not in nested:
                continue
            has_fused = True
            flat_entry = dict(nested)
            flat_entry.setdefault("proj_type", proj_type)
            flat_entry.setdefault(
                "type",
                f"{entry.get('type', 'QuantizedQKVLinear')}.{nested_name}",
            )
            yield f"{base}.{proj_type}", flat_entry
        if has_fused:
            continue


def _projection_key_candidates(key: str):
    """Yield equivalent training/vLLM projection keys.

    Mistral3 checkpoints save the text tower under
    ``model.language_model.layers.*`` in Transformers, while vLLM serves the
    same tower as ``language_model.model.layers.*``.  Also accept direct
    text-only ``model.layers.*`` configs for local experiments.
    """

    seen: set[str] = set()

    def add(candidate: str):
        if candidate not in seen:
            seen.add(candidate)
            return candidate
        return None

    first = add(key)
    if first is not None:
        yield first

    replacements = (
        ("language_model.model.", "model.language_model."),
        ("model.language_model.", "language_model.model."),
    )
    for old, new in replacements:
        if key.startswith(old):
            candidate = add(f"{new}{key[len(old) :]}")
            if candidate is not None:
                yield candidate

    if key.startswith("language_model.model."):
        candidate = add(f"model.{key[len('language_model.model.') :]}")
        if candidate is not None:
            yield candidate
    elif key.startswith("model."):
        candidate = add(f"language_model.model.{key[len('model.') :]}")
        if candidate is not None:
            yield candidate


def _infer_grouping_dim(entry: dict) -> str:
    # Legacy configs may omit grouping_dim; current configs are always token-grouped.
    t = str(entry.get("type", "")).lower()
    if "token" in t or "pertoken" in t:
        return "token"
    return "token"


# ---- Module-level singleton loaded lazily from BEYOND_QUANT_CONFIG env. -----

_cached: Optional[QuantConfig] = None
_cached_path: Optional[str] = None
_cached_expected_bits: Optional[int] = None
_cached_expected_model: Optional[str] = None
_cached_strict: Optional[bool] = None


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


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "disable",
        "disabled",
        "none",
    }


def get_config_from_env(*, force_reload: bool = False) -> Optional[QuantConfig]:
    """Return a loaded ``QuantConfig`` (or None) based on ``BEYOND_QUANT_CONFIG``."""
    global _cached, _cached_path, _cached_expected_bits
    global _cached_expected_model, _cached_strict
    path = os.environ.get("BEYOND_QUANT_CONFIG")
    if not path:
        _cached = None
        _cached_path = None
        _cached_expected_bits = None
        _cached_expected_model = None
        _cached_strict = None
        return None
    path = os.path.abspath(path)
    expected_bits_env = os.environ.get("BEYOND_BITS")
    expected_bits = int(expected_bits_env) if expected_bits_env else None
    expected_model = os.environ.get("BEYOND_MODEL_ID")
    strict = _env_enabled("BEYOND_QUANT_CONFIG_STRICT", True)
    if (
        not force_reload
        and _cached is not None
        and _cached_path == path
        and _cached_expected_bits == expected_bits
        and _cached_expected_model == expected_model
        and _cached_strict == strict
    ):
        return _cached
    if _torch_is_compiling():
        raise RuntimeError(
            "BEYOND_QUANT_CONFIG must be loaded before torch.compile tracing. "
            "Call get_config_from_env() during fake-quant patch/worker setup."
        )
    _cached = QuantConfig.load(
        path,
        expected_bits=expected_bits,
        expected_model=expected_model if strict else None,
        require_model_metadata=bool(strict and expected_model),
    )
    _cached_path = path
    _cached_expected_bits = expected_bits
    _cached_expected_model = expected_model
    _cached_strict = strict
    return _cached
