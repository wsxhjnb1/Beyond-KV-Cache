"""Shared compile-cache defaults for FA4, CuTe DSL, and CUDA JIT.

These environment variables must be set before importing FlashAttention-4 or
CuTe DSL modules because their cache policy is captured at import time.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return value.strip().lower() not in {"", "0", "false", "off", "no"}


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def _slug(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower()).strip("_")
    return value or "unknown"


def _cute_dsl_arch(major: int, minor: int) -> str:
    """Return the CuTe DSL compilation arch for a CUDA capability."""
    arch_map = {
        (9, 0): "sm_90a",
        (10, 0): "sm_100a",
        (10, 3): "sm_103a",
        (11, 0): "sm_110a",
        (12, 0): "sm_120a",
        (12, 1): "sm_121a",
    }
    return arch_map.get((int(major), int(minor)), f"sm_{int(major)}{int(minor)}")


def _decode_gpu_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _parse_compute_capability(value) -> tuple[int, int] | None:
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    text = _decode_gpu_string(value).strip()
    match = re.search(r"(\d+)\D+(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _cuda_visible_tokens() -> list[str] | None:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    stripped = value.strip()
    if stripped.lower() in {"", "-1", "none", "void", "nodevfiles"}:
        return []
    return [item.strip() for item in stripped.split(",") if item.strip()]


def _matches_cuda_visible(index: int, uuid: str, tokens: list[str] | None) -> bool:
    if tokens is None:
        return True
    if not tokens:
        return False
    for token in tokens:
        if token.isdigit() and int(token) == int(index):
            return True
        if uuid and (token == uuid or uuid.startswith(token) or token.startswith(uuid)):
            return True
    return False


def _device_tuple(major: int, minor: int, name: str) -> tuple[str, str, str]:
    major_i, minor_i = int(major), int(minor)
    return f"sm_{major_i}{minor_i}", _cute_dsl_arch(major_i, minor_i), _slug(name)


def _summarize_devices(
    devices: list[tuple[str, str, str]],
) -> Tuple[str | None, str | None, str | None]:
    if not devices:
        return None, None, None
    flash_arches = {item[0] for item in devices}
    cute_arches = {item[1] for item in devices}
    flash_arch = devices[0][0] if len(flash_arches) == 1 else None
    cute_arch = devices[0][1] if len(cute_arches) == 1 else None
    unique_device_tags = []
    for _item_flash_arch, item_cute_arch, item_name in devices:
        tag = f"{item_cute_arch}_{item_name}"
        if tag not in unique_device_tags:
            unique_device_tags.append(tag)
    if len(unique_device_tags) == 1:
        device_tag = unique_device_tags[0]
    else:
        device_tag = "mixed_" + "__".join(unique_device_tags)
    return flash_arch, cute_arch, device_tag


def _detect_cuda_target_with_nvml() -> list[tuple[str, str, str]]:
    try:
        import pynvml
    except Exception:
        return []

    devices: list[tuple[str, str, str]] = []
    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        visible_tokens = _cuda_visible_tokens()
        get_cc = getattr(pynvml, "nvmlDeviceGetCudaComputeCapability", None)
        if get_cc is None:
            return []
        for idx in range(int(pynvml.nvmlDeviceGetCount())):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            uuid = _decode_gpu_string(pynvml.nvmlDeviceGetUUID(handle))
            if not _matches_cuda_visible(idx, uuid, visible_tokens):
                continue
            parsed_cc = _parse_compute_capability(get_cc(handle))
            if parsed_cc is None:
                continue
            name = _decode_gpu_string(pynvml.nvmlDeviceGetName(handle))
            devices.append(_device_tuple(parsed_cc[0], parsed_cc[1], name))
    except Exception:
        return []
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    return devices


def _detect_cuda_target_with_nvidia_smi() -> list[tuple[str, str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,compute_cap,name",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return []

    devices: list[tuple[str, str, str]] = []
    visible_tokens = _cuda_visible_tokens()
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        uuid, compute_cap, name = parts[1], parts[2], parts[3]
        if not _matches_cuda_visible(index, uuid, visible_tokens):
            continue
        parsed_cc = _parse_compute_capability(compute_cap)
        if parsed_cc is None:
            continue
        devices.append(_device_tuple(parsed_cc[0], parsed_cc[1], name))
    return devices


def _detect_cuda_target_with_torch() -> list[tuple[str, str, str]]:
    if not _env_flag("BEYOND_ALLOW_TORCH_CUDA_DETECT"):
        return []
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        devices = []
        for idx in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(idx)
            devices.append(_device_tuple(major, minor, torch.cuda.get_device_name(idx)))
        return devices
    except Exception:
        return []


def _detect_cuda_target() -> Tuple[str | None, str | None, str | None]:
    """Return FA4 arch, CuTe arch, and model tag without initializing CUDA."""
    for detector in (
        _detect_cuda_target_with_nvml,
        _detect_cuda_target_with_nvidia_smi,
        _detect_cuda_target_with_torch,
    ):
        devices = detector()
        if devices:
            return _summarize_devices(devices)
    return None, None, None


def _source_code_tag(repo_root: Path) -> str | None:
    override = os.environ.get("BEYOND_COMPILE_CACHE_CODE_TAG")
    if override:
        return _slug(override)
    if _env_flag("BEYOND_COMPILE_CACHE_NO_CODE_TAG"):
        return None
    digest = hashlib.sha1()
    paths = [
        "src/beyond/common/compile_cache.py",
        "quant/beyond_cute.py",
        "quant/fa4_cute/beyond_decode_policy.py",
        "quant/fa4_cute/beyond_mixed_decode_runtime.py",
        "quant/fa4_cute/beyond_mixed_decode_sm100.py",
        "src/beyond/quantization/table_precision.py",
        "src/beyond/runtime/vllm/config.py",
        "src/beyond/runtime/vllm/packed_backend.py",
        "src/beyond/runtime/vllm/registration.py",
        "src/beyond/inference/cli.py",
    ]
    found = False
    for relative in paths:
        path = repo_root / relative
        try:
            data = path.read_bytes()
        except OSError:
            continue
        found = True
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    if not found:
        return None
    return f"code_{digest.hexdigest()[:12]}"


def configure_compile_cache(
    repo_root: str | os.PathLike[str] | None = None,
    *,
    verbose: bool | None = None,
) -> Dict[str, str]:
    """Set stable on-disk cache locations for runtime-compiled kernels.

    Existing user-provided environment variables are preserved. Set
    ``BEYOND_DISABLE_COMPILE_CACHE=1`` to leave the environment untouched.
    ``BEYOND_COMPILE_CACHE_DIR`` can be used to place model-tagged cache
    directories under a custom root. Set ``BEYOND_COMPILE_CACHE_EXACT=1`` if
    you need that directory to be used literally.
    """

    if _env_flag("BEYOND_DISABLE_COMPILE_CACHE"):
        return {}

    detected_flash_arch, detected_cute_arch, detected_device_tag = _detect_cuda_target()
    device_tag = os.environ.get("BEYOND_COMPILE_CACHE_DEVICE_TAG") or detected_device_tag
    root = Path(repo_root).expanduser().resolve() if repo_root else _repo_root_from_here()
    code_tag = _source_code_tag(root)
    cache_tag = device_tag
    if cache_tag and code_tag:
        cache_tag = f"{cache_tag}_{code_tag}"
    elif code_tag:
        cache_tag = code_tag

    base_root = os.environ.get("BEYOND_COMPILE_CACHE_DIR")
    if base_root:
        cache_root = Path(base_root).expanduser().resolve()
        if cache_tag and not _env_flag("BEYOND_COMPILE_CACHE_EXACT"):
            cache_root = cache_root / cache_tag
        defaults = {
            "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": cache_root / "flash_attention_cute_dsl",
            "CUTE_DSL_CACHE_DIR": cache_root / "cute_dsl",
            "CUDA_CACHE_PATH": cache_root / "cuda",
            "TRITON_CACHE_DIR": cache_root / "triton",
            "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
            "VLLM_CACHE_ROOT": cache_root / "vllm",
        }
    else:
        if cache_tag:
            cache_root = root / ".compile_cache" / cache_tag
            defaults = {
                "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": cache_root / "flash_attention_cute_dsl",
                "CUTE_DSL_CACHE_DIR": cache_root / "cute_dsl",
                "CUDA_CACHE_PATH": cache_root / "cuda",
                "TRITON_CACHE_DIR": cache_root / "triton",
                "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
                "VLLM_CACHE_ROOT": cache_root / "vllm",
            }
        else:
            defaults = {
                "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": root / ".flash_attention_cute_dsl_cache",
                "CUTE_DSL_CACHE_DIR": root / ".cute_dsl_cache",
                "CUDA_CACHE_PATH": root / ".cuda_cache",
                "TRITON_CACHE_DIR": root / ".triton_cache",
                "TORCHINDUCTOR_CACHE_DIR": root / ".torchinductor_cache",
                "VLLM_CACHE_ROOT": root / ".vllm_cache",
            }

    applied: Dict[str, str] = {}

    if detected_flash_arch:
        os.environ.setdefault("FLASH_ATTENTION_ARCH", detected_flash_arch)
        applied["FLASH_ATTENTION_ARCH"] = os.environ["FLASH_ATTENTION_ARCH"]
    if detected_cute_arch:
        os.environ.setdefault("CUTE_DSL_ARCH", detected_cute_arch)
        applied["CUTE_DSL_ARCH"] = os.environ["CUTE_DSL_ARCH"]
    if device_tag:
        os.environ.setdefault("BEYOND_COMPILE_CACHE_DEVICE_TAG", device_tag)
        applied["BEYOND_COMPILE_CACHE_DEVICE_TAG"] = os.environ["BEYOND_COMPILE_CACHE_DEVICE_TAG"]
    if code_tag:
        os.environ.setdefault("BEYOND_COMPILE_CACHE_CODE_TAG", code_tag)
        applied["BEYOND_COMPILE_CACHE_CODE_TAG"] = os.environ["BEYOND_COMPILE_CACHE_CODE_TAG"]

    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    applied["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = os.environ[
        "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"
    ]

    for key, path in defaults.items():
        value = os.environ.setdefault(key, str(path))
        applied[key] = value
        try:
            Path(value).expanduser().mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024 * 1024 * 1024))
    applied["CUDA_CACHE_MAXSIZE"] = os.environ["CUDA_CACHE_MAXSIZE"]

    if verbose is None:
        verbose = _env_flag("BEYOND_COMPILE_CACHE_VERBOSE")
    if verbose:
        print(
            "Compile cache enabled: "
            + ", ".join(f"{key}={value}" for key, value in sorted(applied.items())),
            flush=True,
        )

    return applied
