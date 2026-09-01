"""Dependency-light environment and packed-layout contracts for Beyond vLLM."""

from __future__ import annotations

import os

FAKE_QUANT_SUPPORTED_BITS = (1, 2, 3, 4, 5, 6)
REAL_PACKED_GROUP_SIZES = (32,)


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back on invalid input."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    """Read the project's permissive boolean environment syntax."""
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if value in {"0", "false", "no", "off", "disable", "disabled", "none"}:
        return False
    return bool(default)


def env_bits(name: str) -> int:
    """Read a fake-QDQ bit width and reject unsupported values."""
    bits = env_int(name, 4)
    if bits not in FAKE_QUANT_SUPPORTED_BITS:
        raise ValueError(
            f"Beyond fakequant supports bits in {FAKE_QUANT_SUPPORTED_BITS}, got {bits}"
        )
    return bits


def validate_real_packed_layout(
    bits: int,
    k_group_size: int,
    v_group_size: int,
) -> None:
    """Validate the only layout implemented by the compressed backend."""
    if int(bits) != 4:
        raise NotImplementedError("BeyondPacked real backend supports only 4-bit")
    if int(k_group_size) not in REAL_PACKED_GROUP_SIZES:
        raise NotImplementedError(
            f"BeyondPacked real backend k_group_size must be 32, got {k_group_size}"
        )
    if int(v_group_size) not in REAL_PACKED_GROUP_SIZES:
        raise NotImplementedError(
            f"BeyondPacked real backend v_group_size must be 32, got {v_group_size}"
        )
