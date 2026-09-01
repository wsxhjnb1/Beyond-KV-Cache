"""Precision contract for learned quantization tables.

The model and activations may run in BF16, but learned codepoints and decision
thresholds use FP32 master parameters during optimization.  Hard-QDQ forwards
see the exact values representable by the deployment FP16 table ABI.
"""

from __future__ import annotations

import torch

TABLE_PRECISION_ABI = "beyond.ieee-fp16-tables.v1"
LEGACY_TABLE_PRECISION_ABI = "beyond.legacy-fp32-tables.v0"
TABLE_STORAGE_DTYPE_NAME = "float16"
TABLE_COMPUTE_DTYPE_NAME = "float32"
TABLE_STORAGE_DTYPE = torch.float16
TABLE_MASTER_DTYPE = torch.float32

# The retained deployment tables stay inside roughly [-0.11, 1.11].  One
# binary16 ULP in [1, 2) is 2^-10.  Two such ULPs avoid the round-to-nearest
# tie case in which both endpoints of one rounding cell map to the same even
# FP16 value.  Export validation remains the final fail-closed guard if a
# future table leaves that range or otherwise violates the contract.
FP16_STRICT_MIN_GAP = 2.0**-9


def fp16_ste_roundtrip(master: torch.Tensor) -> torch.Tensor:
    """Expose FP16-representable values while preserving FP32 master gradients."""
    if master.dtype != TABLE_MASTER_DTYPE:
        raise TypeError(f"quant-table master must be float32, got {master.dtype}")
    rounded = master.to(TABLE_STORAGE_DTYPE).to(TABLE_MASTER_DTYPE)
    if not torch.is_grad_enabled() or not master.requires_grad:
        return rounded
    return master + (rounded - master).detach()


def cast_table_for_storage(table: torch.Tensor) -> torch.Tensor:
    """Return a detached, contiguous CPU tensor in the deployment dtype."""
    return table.detach().to(device="cpu", dtype=TABLE_STORAGE_DTYPE).contiguous()


def _strictly_increasing(table: torch.Tensor) -> bool:
    return int(table.shape[-1]) <= 1 or bool((table[..., 1:] > table[..., :-1]).all())


def validate_fp16_storage_tables(
    quant_points: torch.Tensor,
    thresholds: torch.Tensor,
    *,
    label: str = "quant table",
) -> None:
    """Fail closed unless an exported table exactly satisfies the FP16 ABI."""
    if quant_points.dtype != TABLE_STORAGE_DTYPE:
        raise TypeError(f"{label}: quant_points must be float16, got {quant_points.dtype}")
    if thresholds.dtype != TABLE_STORAGE_DTYPE:
        raise TypeError(f"{label}: thresholds must be float16, got {thresholds.dtype}")
    expected_threshold_shape = tuple(quant_points.shape[:-1]) + (int(quant_points.shape[-1]) - 1,)
    if tuple(thresholds.shape) != expected_threshold_shape:
        raise ValueError(
            f"{label}: thresholds shape {tuple(thresholds.shape)} does not match "
            f"{expected_threshold_shape}"
        )
    if not bool(torch.isfinite(quant_points).all()):
        raise ValueError(f"{label}: quant_points contain NaN or Inf")
    if not bool(torch.isfinite(thresholds).all()):
        raise ValueError(f"{label}: thresholds contain NaN or Inf")
    if not _strictly_increasing(quant_points):
        raise ValueError(f"{label}: quant_points collapse after float16 rounding")
    if not _strictly_increasing(thresholds):
        raise ValueError(f"{label}: thresholds collapse after float16 rounding")


def storage_dtype_name(tensor: torch.Tensor) -> str:
    """Return the supported serialized dtype name for a table tensor."""
    if tensor.dtype == torch.float16:
        return "float16"
    if tensor.dtype == torch.float32:
        return "float32"
    raise TypeError(f"unsupported quant-table storage dtype: {tensor.dtype}")


def torch_dtype_for_storage(name: str) -> torch.dtype:
    """Resolve a serialized table dtype without accepting ambiguous aliases."""
    normalized = str(name).strip().lower()
    if normalized == "float16":
        return torch.float16
    if normalized == "float32":
        return torch.float32
    raise ValueError(f"unsupported quant-table storage dtype: {name!r}")
