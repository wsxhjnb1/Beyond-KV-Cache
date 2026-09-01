"""
Model utilities shared between different evaluation and training scripts.

This module contains common functionality for:
- Model type detection and configuration
- Model and tokenizer loading with fallback configurations
- Discover HF dataset loading
- Perplexity calculation
"""

import functools
import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from tqdm import tqdm

# Imported lazily so model/quantization-only commands do not initialize the
# Arrow/Pandas dataset stack at module import time.
load_dataset = None
# Model factories are also lazy: importing data/curriculum helpers must not
# initialize Transformers, scikit-learn, or optional multimodal backends.
AutoTokenizer = None
AutoModelForCausalLM = None
AutoModelForImageTextToText = None
MistralCommonBackend = None
Mistral3ForConditionalGeneration = None


def _require_datasets():
    global load_dataset
    if load_dataset is None:
        try:
            from datasets import load_dataset as hf_load_dataset
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The 'datasets' package is required for dataset loading. "
                "Install it with `pip install datasets` or "
                "`pip install -r requirements.txt`."
            ) from exc
        load_dataset = hf_load_dataset


def _require_transformers():
    global AutoTokenizer, AutoModelForCausalLM
    global AutoModelForImageTextToText, MistralCommonBackend
    global Mistral3ForConditionalGeneration
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        try:
            from transformers import AutoModelForCausalLM as HFAutoModelForCausalLM
            from transformers import AutoTokenizer as HFAutoTokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The 'transformers' package is required for model/tokenizer loading. "
                "Install it with `pip install transformers` or "
                "`pip install -r requirements.txt`."
            ) from exc
        if AutoTokenizer is None:
            AutoTokenizer = HFAutoTokenizer
        if AutoModelForCausalLM is None:
            AutoModelForCausalLM = HFAutoModelForCausalLM
    if MistralCommonBackend is None:
        try:
            from transformers import MistralCommonBackend as HFMistralCommonBackend
        except (ImportError, AttributeError):
            pass
        else:
            MistralCommonBackend = HFMistralCommonBackend
    if Mistral3ForConditionalGeneration is None:
        try:
            from transformers import (
                Mistral3ForConditionalGeneration as HFMistral3ForConditionalGeneration,
            )
        except (ImportError, AttributeError):
            pass
        else:
            Mistral3ForConditionalGeneration = HFMistral3ForConditionalGeneration
    if AutoModelForImageTextToText is None:
        try:
            from transformers import AutoModelForImageTextToText as HFAutoModelForImageTextToText
        except (ImportError, AttributeError):
            pass
        else:
            AutoModelForImageTextToText = HFAutoModelForImageTextToText
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ModuleNotFoundError(
            "The 'transformers' package is required for model/tokenizer loading. "
            "Install it with `pip install transformers` or `pip install -r requirements.txt`."
        )


def _ensure_all_tied_weights_keys():
    """Provide a compatibility shim for older model implementations."""
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return

    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def _get_all_tied_weights_keys(self):
        if "all_tied_weights_keys" in self.__dict__:
            return self.__dict__["all_tied_weights_keys"]
        keys = getattr(self, "_tied_weights_keys", None)
        if not keys:
            return {}
        return {key: None for key in keys}

    def _set_all_tied_weights_keys(self, value):
        self.__dict__["all_tied_weights_keys"] = value

    PreTrainedModel.all_tied_weights_keys = property(
        _get_all_tied_weights_keys,
        _set_all_tied_weights_keys,
    )


@functools.lru_cache(maxsize=None)
def _flash_attn_distribution_names():
    """Return normalized distributions that provide the flash_attn namespace."""
    try:
        distributions = importlib.metadata.packages_distributions().get("flash_attn", [])
    except Exception:
        return set()
    return {name.replace("_", "-").lower() for name in distributions}


@functools.lru_cache(maxsize=None)
def _flash_attention_is_available(required_major=2):
    """Check whether the requested FlashAttention implementation is available."""
    if not torch.cuda.is_available():
        return False

    distributions = _flash_attn_distribution_names()
    required_major = int(required_major)

    if required_major >= 4:
        try:
            cute_spec = importlib.util.find_spec("flash_attn.cute")
        except (ImportError, ModuleNotFoundError, ValueError):
            cute_spec = None
        if "flash-attn-4" not in distributions or cute_spec is None:
            return False
        return _flash_attention_4_smoke_test_passes()

    if "flash-attn" not in distributions:
        return False

    try:
        import flash_attn  # noqa: F401
    except Exception as exc:
        print(f"flash-attn unavailable; will try other attention implementations: ({exc})")
        return False

    try:
        version = importlib.metadata.version("flash-attn")
    except Exception:
        version = getattr(flash_attn, "__version__", "0")

    matched = re.match(r"^\s*(\d+)", str(version))
    major = int(matched.group(1)) if matched else 0
    return major >= required_major


@functools.lru_cache(maxsize=1)
def _flash_attention_4_smoke_test_passes():
    """Return whether flash-attn-4 can compile and run a minimal forward."""
    try:
        from flash_attn.cute import flash_attn_func

        with torch.inference_mode():
            q = torch.randn(1, 2, 1, 64, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 2, 1, 64, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, 2, 1, 64, device="cuda", dtype=torch.bfloat16)
            flash_attn_func(q, k, v, causal=True)
            torch.cuda.synchronize()
        return True
    except Exception as exc:
        print(f"flash-attn-4 unavailable; will try other attention implementations: ({exc})")
        return False
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@functools.lru_cache(maxsize=1)
def _flash_attention_2_is_available():
    """Backward-compatible helper for historical callsites."""
    return _flash_attention_is_available(2)


def _select_attn_implementation_candidates(requested_impl):
    """Return ordered candidate list for attention implementation fallback."""
    if requested_impl is None or requested_impl == "" or requested_impl == "auto":
        return ["flash_attention_4", "flash_attention_2", "sdpa", "eager"]
    if requested_impl == "flash_attention_4":
        return ["flash_attention_4", "flash_attention_2", "sdpa", "eager"]
    if requested_impl == "flash_attention_2":
        return ["flash_attention_2", "sdpa", "eager"]
    if requested_impl == "sdpa":
        return ["sdpa", "eager"]
    if requested_impl == "eager":
        return ["eager"]
    return [str(requested_impl)]


def _filter_attn_implementation_candidates(candidates):
    filtered = []
    for impl in candidates:
        if impl in {"flash_attention_4", "flash_attention_2"}:
            required_major = 4 if impl == "flash_attention_4" else 2
            if not _flash_attention_is_available(required_major):
                continue
        filtered.append(impl)
    return filtered or ["sdpa", "eager"]


def _normalize_model_config(model_config):
    """Normalize loader kwargs across Transformers versions."""
    if not isinstance(model_config, dict):
        return model_config

    normalized = dict(model_config)
    if "torch_dtype" in normalized and "dtype" not in normalized:
        normalized["dtype"] = normalized.pop("torch_dtype")

    requested_impl = normalized.get("attn_implementation")
    if requested_impl is not None:
        candidates = _select_attn_implementation_candidates(str(requested_impl))
        filtered_candidates = _filter_attn_implementation_candidates(candidates)
        normalized["_attn_implementation_candidates"] = filtered_candidates
        normalized["attn_implementation"] = filtered_candidates[0]

    return normalized


def _display_model_config(model_config):
    if not isinstance(model_config, dict):
        return model_config
    display = dict(model_config)
    if "config" in display:
        display["config"] = type(display["config"]).__name__
    return display


MINISTRAL3_14B_INSTRUCT_MODEL_ID = "mistralai/Ministral-3-14B-Instruct-2512-BF16"
LLAMA2_7B_BASE_MODEL_ID = "meta-llama/Llama-2-7b-hf"
QWEN3_8B_BASE_MODEL_ID = "Qwen/Qwen3-8B-Base"
QWEN3_30B_A3B_INSTRUCT_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

SUPPORTED_DISCOVER_MODELS = (
    MINISTRAL3_14B_INSTRUCT_MODEL_ID,
    LLAMA2_7B_BASE_MODEL_ID,
    "meta-llama/Llama-3.1-8B-Instruct",
    QWEN3_8B_BASE_MODEL_ID,
    QWEN3_30B_A3B_INSTRUCT_MODEL_ID,
)

_MODEL_ALIASES = {
    "ministral-3-14b": MINISTRAL3_14B_INSTRUCT_MODEL_ID,
    "ministral-3-14b-instruct": MINISTRAL3_14B_INSTRUCT_MODEL_ID,
    "ministral-3-14b-instruct-2512": MINISTRAL3_14B_INSTRUCT_MODEL_ID,
    "ministral-3-14b-instruct-2512-bf16": MINISTRAL3_14B_INSTRUCT_MODEL_ID,
    "ministral3-14b": MINISTRAL3_14B_INSTRUCT_MODEL_ID,
    "ministral3-14b-instruct": MINISTRAL3_14B_INSTRUCT_MODEL_ID,
    "ministral3-14b-instruct-2512": MINISTRAL3_14B_INSTRUCT_MODEL_ID,
    "mistral-3-14b-instruct": MINISTRAL3_14B_INSTRUCT_MODEL_ID,
    "mistralai/Ministral-3-14B-Instruct-2512": MINISTRAL3_14B_INSTRUCT_MODEL_ID,
    "mistralai/Ministral-3-14B-Instruct-2512-BF16": MINISTRAL3_14B_INSTRUCT_MODEL_ID,
    "llama-2-7b": LLAMA2_7B_BASE_MODEL_ID,
    "llama-2-7b-hf": LLAMA2_7B_BASE_MODEL_ID,
    "Llama-2-7b-hf": LLAMA2_7B_BASE_MODEL_ID,
    LLAMA2_7B_BASE_MODEL_ID: LLAMA2_7B_BASE_MODEL_ID,
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-8b-ins": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3-8b-base": QWEN3_8B_BASE_MODEL_ID,
    "Qwen3-8B-Base": QWEN3_8B_BASE_MODEL_ID,
    QWEN3_8B_BASE_MODEL_ID: QWEN3_8B_BASE_MODEL_ID,
    "qwen3-30b-a3b": QWEN3_30B_A3B_INSTRUCT_MODEL_ID,
    "qwen3-30b-a3b-instruct": QWEN3_30B_A3B_INSTRUCT_MODEL_ID,
    "qwen3-30b-a3b-instruct-2507": QWEN3_30B_A3B_INSTRUCT_MODEL_ID,
    "Qwen3-30B-A3B": QWEN3_30B_A3B_INSTRUCT_MODEL_ID,
    "Qwen3-30B-A3B-Instruct": QWEN3_30B_A3B_INSTRUCT_MODEL_ID,
    "Qwen3-30B-A3B-Instruct-2507": QWEN3_30B_A3B_INSTRUCT_MODEL_ID,
    "Qwen/Qwen3-30B-A3B": QWEN3_30B_A3B_INSTRUCT_MODEL_ID,
    "Qwen/Qwen3-30B-A3B-Instruct": QWEN3_30B_A3B_INSTRUCT_MODEL_ID,
    QWEN3_30B_A3B_INSTRUCT_MODEL_ID: QWEN3_30B_A3B_INSTRUCT_MODEL_ID,
}
_MODEL_ALIASES.update({key.lower(): value for key, value in list(_MODEL_ALIASES.items())})


def canonical_model_id(model_id: str) -> str:
    """Resolve the short names we use in scripts to their HF model IDs."""
    key = str(model_id).strip()
    canonical = _MODEL_ALIASES.get(key) or _MODEL_ALIASES.get(key.lower())
    if canonical is None:
        supported = ", ".join(SUPPORTED_DISCOVER_MODELS)
        raise ValueError(f"Unsupported model '{model_id}'. Supported discover models: {supported}.")
    return canonical


def detect_model_type(model_id):
    """Detect the small set of model families discover currently trains."""
    canonical = canonical_model_id(model_id)
    if canonical == MINISTRAL3_14B_INSTRUCT_MODEL_ID:
        return "mistral3"
    if canonical in {LLAMA2_7B_BASE_MODEL_ID, "meta-llama/Llama-3.1-8B-Instruct"}:
        return "llama"
    if canonical in {QWEN3_8B_BASE_MODEL_ID, QWEN3_30B_A3B_INSTRUCT_MODEL_ID}:
        return "qwen"
    raise AssertionError(f"Unhandled supported model: {canonical}")


def get_model_dtype(model_id, model_type):
    """All supported discover targets are trained in bf16."""
    canonical_model_id(model_id)
    if model_type not in {"mistral3", "llama", "qwen"}:
        raise ValueError(f"Unsupported model type: {model_type}")
    return torch.bfloat16


DISCOVER_EXPERTS_IMPLEMENTATION_CHOICES = ("auto", "eager", "grouped_mm", "batched_mm", "deepgemm")
DISCOVER_SAFE_EXPERTS_IMPLEMENTATIONS = {
    # Qwen3-MoE defaults to Transformers grouped_mm, which currently hits a
    # CUTLASS launch failure on B300/torch 2.10 while FA4 attention itself works.
    "qwen": "eager",
}


def _resolve_experts_implementation(model_type, experts_implementation="auto"):
    requested = "auto" if experts_implementation is None else str(experts_implementation).strip()
    if not requested:
        requested = "auto"
    if requested not in DISCOVER_EXPERTS_IMPLEMENTATION_CHOICES:
        raise ValueError(
            f"Unsupported experts implementation: {experts_implementation}. "
            f"Expected one of {DISCOVER_EXPERTS_IMPLEMENTATION_CHOICES}."
        )
    if requested == "auto":
        return DISCOVER_SAFE_EXPERTS_IMPLEMENTATIONS.get(model_type)
    return requested


def get_model_config(
    model_type, base_model, attn_implementation="auto", experts_implementation="auto"
):
    """Return the minimal loader config for the supported training targets."""
    base_model = canonical_model_id(base_model)
    model_dtype = get_model_dtype(base_model, model_type)
    config = {
        "dtype": model_dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if model_type in {"mistral3", "llama", "qwen"}:
        config["attn_implementation"] = attn_implementation
    resolved_experts_implementation = None
    if base_model == QWEN3_30B_A3B_INSTRUCT_MODEL_ID:
        resolved_experts_implementation = _resolve_experts_implementation(
            model_type,
            experts_implementation=experts_implementation,
        )
    if resolved_experts_implementation is not None:
        config["experts_implementation"] = resolved_experts_implementation

    config = _normalize_model_config(config)
    print(f"Using dtype: {model_dtype} for model: {base_model}")
    return config


def load_tokenizer(model_id, model_type):
    """Load tokenizer for the supported discover models."""
    model_id = canonical_model_id(model_id)
    if model_type not in {"mistral3", "llama", "qwen"}:
        raise ValueError(f"Unsupported model type: {model_type}")

    if model_type == "mistral3":
        if MistralCommonBackend is None:
            try:
                from transformers import MistralCommonBackend as HFMistralCommonBackend
            except (ImportError, AttributeError):
                pass
            else:
                globals()["MistralCommonBackend"] = HFMistralCommonBackend
        if MistralCommonBackend is None:
            raise ModuleNotFoundError(
                "Transformers MistralCommonBackend is required for "
                f"{model_id}. Install mistral-common>=1.8.6 and a recent "
                "Transformers build."
            )
        tokenizer = MistralCommonBackend.from_pretrained(model_id)
        if not getattr(tokenizer, "name_or_path", None):
            tokenizer.name_or_path = model_id
    else:
        _require_transformers()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"Successfully loaded tokenizer for {model_id}")

    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "right"
    return tokenizer


def load_model(base_model, model_config=None):
    """Load one of the supported discover training targets."""
    _require_transformers()
    base_model = canonical_model_id(base_model)
    model_type = detect_model_type(base_model)

    if model_config is None:
        model_config = get_model_config(model_type, base_model)
    else:
        model_config = _normalize_model_config(model_config)

    _ensure_all_tied_weights_keys()
    print(f"Detected model type: {model_type}")
    print(f"Using configuration: {_display_model_config(model_config)}")

    model_cls = AutoModelForCausalLM
    if model_type == "mistral3":
        model_cls = Mistral3ForConditionalGeneration or AutoModelForImageTextToText
        if model_cls is None:
            raise ModuleNotFoundError(
                "Transformers Mistral3ForConditionalGeneration is required for "
                f"{base_model}. Upgrade transformers or install the pinned requirements."
            )

    attn_candidates = None
    if isinstance(model_config, dict):
        attn_candidates = model_config.pop("_attn_implementation_candidates", None)

    def _load_with_candidates():
        if isinstance(attn_candidates, Sequence) and len(attn_candidates) > 0:
            last_error = None
            for impl in attn_candidates:
                candidate_config = dict(model_config)
                if impl is not None:
                    candidate_config["attn_implementation"] = impl
                print(f"Trying attention implementation: {impl}")
                try:
                    return model_cls.from_pretrained(base_model, **candidate_config)
                except Exception as exc:
                    last_error = exc
                    continue
            if last_error is not None:
                raise last_error
        return model_cls.from_pretrained(base_model, **model_config)

    try:
        model = _load_with_candidates()
        print(f"Successfully loaded {base_model}")
    except Exception as exc:
        fallback_config = {
            "dtype": model_config.get("dtype", torch.bfloat16)
            if isinstance(model_config, dict)
            else torch.bfloat16,
            "device_map": model_config.get("device_map", "auto")
            if isinstance(model_config, dict)
            else "auto",
            "trust_remote_code": True,
        }
        print(f"Error loading model with primary config: {exc}")
        print(f"Trying fallback configuration: {fallback_config}")
        model = model_cls.from_pretrained(base_model, **fallback_config)
        print(f"Successfully loaded {base_model} with fallback configuration")

    if hasattr(model.config, "max_length"):
        try:
            delattr(model.config, "max_length")
        except Exception:
            model.config.max_length = None

    model.config.use_cache = False
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = False
    for submodule in (
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
    ):
        submodule_config = getattr(submodule, "config", None)
        if submodule_config is not None and hasattr(submodule_config, "use_cache"):
            submodule_config.use_cache = False
    return model


def validate_discover_megatron_parallel_request(
    world_size,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    allow_linear_only_tensor_parallel=False,
):
    """Validate the Megatron process-grid sizes used by discover."""

    world_size = max(1, int(world_size))
    tp = max(1, int(tensor_model_parallel_size))
    pp = max(1, int(pipeline_model_parallel_size))
    parallel_world = tp * pp
    if world_size < parallel_world:
        raise ValueError(
            "WORLD_SIZE must be at least tensor_model_parallel_size * "
            f"pipeline_model_parallel_size ({world_size} < {tp} * {pp})."
        )
    if world_size % parallel_world != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must be divisible by tensor_model_parallel_size * "
            f"pipeline_model_parallel_size={parallel_world}."
        )


def _build_megatron_parallel_config(tp, pp, model_dtype):
    if tp <= 1 and pp <= 1:
        return None
    try:
        from megatron.core.model_parallel_config import ModelParallelConfig
    except Exception as exc:
        raise RuntimeError(
            "Megatron runtime is enabled but megatron-core is not importable. "
            "Install megatron-core from requirements.txt."
        ) from exc

    try:
        return ModelParallelConfig(
            tensor_model_parallel_size=max(1, int(tp)),
            pipeline_model_parallel_size=max(1, int(pp)),
            perform_initialization=False,
            bf16=(model_dtype == torch.bfloat16),
            fp16=(model_dtype == torch.float16),
            params_dtype=model_dtype,
        )
    except Exception:
        # Fallback for older/restricted constructors.
        return None


class _MegatronLinearTensorOutput(nn.Module):
    """Adapter that converts Megatron linear tuple outputs to a plain tensor.

    Megatron ColumnParallelLinear/RowParallelLinear return ``(output, bias)``
    to support fused bias handling. The model code in this project expects a
    tensor output, so we unwrap and apply deferred bias when provided.
    """

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self._beyond_megatron_linear_type = None
        self._beyond_megatron_tensor_parallel_size = 1
        self._beyond_megatron_output_is_parallel = False
        self._beyond_megatron_input_is_parallel = False

    def forward(self, *args, **kwargs):
        output = self.linear(*args, **kwargs)
        if isinstance(output, tuple):
            if len(output) == 0:
                return output
            primary_output = output[0]
            if len(output) > 1 and output[1] is not None:
                bias = output[1]
                if torch.is_tensor(primary_output) and torch.is_tensor(bias):
                    return primary_output + bias
            return primary_output
        if isinstance(output, list):
            return output[0]
        return output


def _build_megatron_linear(
    module,
    linear_type,
    tensor_parallel_rank,
    tensor_parallel_size,
    mp_config,
    init_method=None,
):
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    if init_method is None:
        init_method = torch.nn.init.xavier_uniform_

    in_features = int(getattr(module, "in_features", 0))
    out_features = int(getattr(module, "out_features", 0))
    use_bias = getattr(module, "bias", None) is not None

    if linear_type == "column":
        wrapped = ColumnParallelLinear(
            input_size=in_features,
            output_size=out_features,
            config=mp_config,
            init_method=init_method,
            bias=use_bias,
            gather_output=False,
            skip_bias_add=False,
        )
        if in_features <= 0 or out_features <= 0:
            return wrapped

        partition = max(1, int(out_features // tensor_parallel_size))
        start = tensor_parallel_rank * partition
        end = (tensor_parallel_rank + 1) * partition
        if out_features % tensor_parallel_size == 0 and wrapped.weight.numel() > 0:
            with torch.no_grad():
                wrapped.weight.data.copy_(module.weight.data[start:end, :])
            if use_bias and wrapped.bias is not None:
                with torch.no_grad():
                    bias = module.bias.data
                    if wrapped.bias.shape == bias.shape:
                        wrapped.bias.data.copy_(bias)
                    elif wrapped.bias.numel() == (end - start):
                        wrapped.bias.data.copy_(bias[start:end])
        return wrapped

    wrapped = RowParallelLinear(
        input_size=in_features,
        output_size=out_features,
        config=mp_config,
        init_method=init_method,
        bias=use_bias,
        input_is_parallel=True,
        skip_bias_add=False,
    )
    if in_features <= 0 or out_features <= 0:
        return wrapped

    partition = max(1, int(in_features // tensor_parallel_size))
    start = tensor_parallel_rank * partition
    end = (tensor_parallel_rank + 1) * partition
    if in_features % tensor_parallel_size == 0 and wrapped.weight.numel() > 0:
        with torch.no_grad():
            wrapped.weight.data.copy_(module.weight.data[:, start:end])
        if use_bias and wrapped.bias is not None and wrapped.bias.shape == module.bias.data.shape:
            with torch.no_grad():
                wrapped.bias.data.copy_(module.bias.data)
    return wrapped


def _infer_megatron_linear_type(module_name):
    lname = (module_name or "").lower()
    if not lname:
        return None

    if (
        "query_key_value" in lname
        or "qkv" in lname
        or lname.endswith("q_proj")
        or lname.endswith("k_proj")
        or lname.endswith("v_proj")
        or lname.endswith("gate_proj")
        or lname.endswith("up_proj")
        or lname.endswith("w1")
        or lname.endswith("w3")
        or lname.endswith("fc1")
        or "dense_h_to_4h" in lname
    ):
        return "column"

    if (
        lname.endswith("o_proj")
        or lname.endswith("down_proj")
        or lname.endswith("w2")
        or lname.endswith("fc2")
        or "dense_4h_to_h" in lname
        or ".out_proj" in lname
    ):
        return "row"

    return None


def _direct_child_map(model):
    children_by_parent = {}
    for module_name, _module in model.named_modules():
        if not module_name:
            continue
        parent_name, child_name = (
            module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
        )
        children_by_parent.setdefault(parent_name, set()).add(child_name)
    return children_by_parent


def _remove_incomplete_parallel_groups(replacements, model, log_fn=print):
    """Avoid half-sharding attention or MLP blocks.

    Column-parallel projections that feed an elementwise op or attention head
    partition must be paired with the matching row-parallel output projection.
    If one member is not eligible, leave the whole local block untouched.
    """

    replacement_map = dict(replacements)
    children_by_parent = _direct_child_map(model)
    common_groups = (
        ("attention", ("q_proj", "k_proj", "v_proj", "o_proj")),
        ("mlp", ("gate_proj", "up_proj", "down_proj")),
        ("mlp", ("w1", "w3", "w2")),
        ("mlp", ("fc1", "fc2")),
        ("mlp", ("dense_h_to_4h", "dense_4h_to_h")),
    )

    disabled = set()
    for parent_name, child_names in children_by_parent.items():
        for group_kind, group_children in common_groups:
            present = [child for child in group_children if child in child_names]
            if not present:
                continue
            full_names = [f"{parent_name}.{child}" if parent_name else child for child in present]
            selected = [name for name in full_names if name in replacement_map]
            if selected and len(selected) != len(present):
                disabled.update(full_names)
                if log_fn:
                    missing = [name for name in full_names if name not in replacement_map]
                    log_fn(
                        f"Skip Megatron TP for {group_kind} block '{parent_name or '<root>'}': "
                        f"incomplete shardable projection set; missing {missing}."
                    )

    if not disabled:
        return replacements
    return [(name, linear_type) for name, linear_type in replacements if name not in disabled]


def _divide_module_int_attr(module, attr_name, tp):
    value = getattr(module, attr_name, None)
    if not isinstance(value, int) or value <= 0:
        return False
    if value % tp != 0:
        raise ValueError(f"{attr_name}={value} is not divisible by TP={tp}")
    setattr(module, attr_name, value // tp)
    return True


def _patch_attention_modules_for_megatron_tp(model, tp, log_fn=print):
    """Patch HF attention metadata after q/k/v/o are true Megatron TP linears."""

    if tp <= 1:
        return 0

    patched = 0
    for module_name, module in model.named_modules():
        q_proj = getattr(module, "q_proj", None)
        k_proj = getattr(module, "k_proj", None)
        v_proj = getattr(module, "v_proj", None)
        o_proj = getattr(module, "o_proj", None)
        fused_qkv = getattr(module, "query_key_value", None)

        separate_tp = all(
            getattr(proj, "_beyond_megatron_output_is_parallel", False)
            for proj in (q_proj, k_proj, v_proj)
        ) and getattr(o_proj, "_beyond_megatron_input_is_parallel", False)
        fused_tp = getattr(fused_qkv, "_beyond_megatron_output_is_parallel", False) and any(
            getattr(getattr(module, out_name, None), "_beyond_megatron_input_is_parallel", False)
            for out_name in ("o_proj", "out_proj", "dense")
        )
        if not separate_tp and not fused_tp:
            continue
        if getattr(module, "_beyond_megatron_tp_patched", False):
            continue

        original_attrs = {}
        for attr_name in (
            "num_heads",
            "num_attention_heads",
            "n_heads",
            "n_head",
            "num_key_value_heads",
            "num_kv_heads",
            "n_kv_heads",
            "hidden_size",
        ):
            if hasattr(module, attr_name):
                original_attrs[attr_name] = getattr(module, attr_name)

        try:
            for attr_name in ("num_heads", "num_attention_heads", "n_heads", "n_head"):
                _divide_module_int_attr(module, attr_name, tp)
            for attr_name in ("num_key_value_heads", "num_kv_heads", "n_kv_heads"):
                _divide_module_int_attr(module, attr_name, tp)
            for attr_name in ("hidden_size",):
                _divide_module_int_attr(module, attr_name, tp)

            q_heads = None
            kv_heads = None
            for attr_name in ("num_heads", "num_attention_heads", "n_heads", "n_head"):
                value = getattr(module, attr_name, None)
                if isinstance(value, int) and value > 0:
                    q_heads = value
                    break
            for attr_name in ("num_key_value_heads", "num_kv_heads", "n_kv_heads"):
                value = getattr(module, attr_name, None)
                if isinstance(value, int) and value > 0:
                    kv_heads = value
                    break
            if q_heads is not None and kv_heads is not None and kv_heads > 0:
                groups = max(1, q_heads // kv_heads)
                if hasattr(module, "num_key_value_groups"):
                    module.num_key_value_groups = groups

            module._beyond_megatron_tp_patched = True
            module._beyond_megatron_tp_original_attrs = original_attrs
            module._beyond_megatron_tensor_parallel_size = tp
            patched += 1
        except Exception as exc:
            for attr_name, value in original_attrs.items():
                try:
                    setattr(module, attr_name, value)
                except Exception:
                    pass
            if log_fn:
                log_fn(
                    f"Failed to patch attention metadata for Megatron TP at {module_name}: {exc}. "
                    "The module was left unmodified."
                )

    if patched and log_fn:
        log_fn(f"Patched {patched} attention modules for Megatron tensor-parallel local heads.")
    return patched


def apply_megatron_parallel_wrappers(
    model,
    tensor_model_parallel_size=1,
    tensor_model_parallel_rank=0,
    pipeline_model_parallel_size=1,
    allow_linear_only_tensor_parallel=False,
    log_fn=print,
):
    """Replace supported HF Linear layers with Megatron tensor-parallel operators."""
    tp = max(1, int(tensor_model_parallel_size))
    pp = max(1, int(pipeline_model_parallel_size))
    if tp <= 1:
        return model
    if pp > 1 and log_fn:
        log_fn(
            "Pipeline parallelism is initialized for process-group compatibility, "
            "but discover only rewrites tensor-parallel layers inside this process."
        )
    if log_fn:
        log_fn(
            "Using Megatron tensor-parallel linear rewrites: column projections keep "
            "local shards and row projections consume local shards."
        )

    model_dtype = None
    for p in model.parameters():
        model_dtype = p.dtype
        break
    if model_dtype is None:
        model_dtype = torch.bfloat16

    try:
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    except Exception as exc:
        if log_fn:
            log_fn(
                "Megatron linear wrapper import failed; continuing with non-sharded linear modules. "
                f"Reason: {exc}"
            )
        return model

    mp_config = _build_megatron_parallel_config(tp, pp, model_dtype)
    if mp_config is None:
        if log_fn:
            log_fn(
                "Megatron ModelParallelConfig build failed; skipping Megatron linear conversion."
            )
        return model

    replacements = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        linear_type = _infer_megatron_linear_type(module_name)
        if linear_type is None:
            continue
        if not isinstance(module, nn.Linear):
            continue

        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if in_features is None or out_features is None:
            continue

        if linear_type == "column" and (out_features % tp != 0):
            if log_fn:
                log_fn(
                    f"Skip Megatron wrapping for {module_name}: out_features {out_features} not divisible by TP={tp}."
                )
            continue
        if linear_type == "row" and (in_features % tp != 0):
            if log_fn:
                log_fn(
                    f"Skip Megatron wrapping for {module_name}: in_features {in_features} not divisible by TP={tp}."
                )
            continue

        replacements.append((module_name, linear_type))

    replacements = _remove_incomplete_parallel_groups(replacements, model, log_fn=log_fn)

    if not replacements:
        if log_fn:
            log_fn("No Megatron-compatible linear layers found for TP wrapping.")
        return model

    for module_name, linear_type in replacements:
        module = model.get_submodule(module_name)
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
            continue

        try:
            parent_name, attr_name = (
                module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
            )
            wrapped = _build_megatron_linear(
                module,
                linear_type,
                tensor_model_parallel_rank,
                tp,
                mp_config,
            )
            wrapped = _MegatronLinearTensorOutput(wrapped)
            wrapped._beyond_megatron_linear_type = linear_type
            wrapped._beyond_megatron_tensor_parallel_size = tp
            wrapped._beyond_megatron_output_is_parallel = linear_type == "column"
            wrapped._beyond_megatron_input_is_parallel = linear_type == "row"
            if parent_name:
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, attr_name, wrapped)
            else:
                model = wrapped
            if log_fn:
                log_fn(f"Wrapped {module_name} as Megatron {linear_type} linear.")
        except Exception as exc:
            if log_fn:
                log_fn(
                    f"Failed Megatron wrap for {module_name} ({linear_type}); keeping original module. "
                    f"Reason: {exc}"
                )
            continue

    _patch_attention_modules_for_megatron_tp(model, tp, log_fn=log_fn)
    return model


DISCOVER_DATASET_NAME = "pile"
DISCOVER_DATASET_CHOICES = (
    "pile",
    "fineweb",
    "openwebmath",
    "fineweb2_cmn_hani",
    "fineweb2_multilingual_equal",
    "hotpotqa_distractor",
    "hotpot_2wiki_equal_input",
    "ultrachat_200k",
)
DISCOVER_REFERENCE_TOKENS = 50_000_000
DISCOVER_VALIDATION_CHUNKS = 150
DISCOVER_EVAL_CHUNKS = 150
DISCOVER_CHAT_WRAP_PILE_DEFAULT = True
DISCOVER_CHAT_WRAP_PILE_PROMPT = "Continue the following text."
DISCOVER_DATA_SEED = 42
# Preserves the historical seed-42 validation split while making it invariant
# to the optimizer/model RNG seed.
DISCOVER_EVAL_SEED = 10_000_042
DISCOVER_TEST_SEED_OFFSET = 10_000_000
DISCOVER_DEFAULT_REFERENCE_TOKENS = DISCOVER_REFERENCE_TOKENS
DISCOVER_DEFAULT_SEPARATOR = "\n\n"
DISCOVER_TRAIN_SEQLEN = int(os.getenv("DISCOVER_TRAIN_SEQLEN", "8192"))
if DISCOVER_TRAIN_SEQLEN <= 0:
    raise ValueError("DISCOVER_TRAIN_SEQLEN must be a positive integer")
DISCOVER_LENGTH_SCHEDULE_NAME = f"main_{DISCOVER_TRAIN_SEQLEN // 1024}k"
_TOKEN_LOG_INTERVAL = 1_000_000
IGNORE_INDEX = -100
_SUPERVISED_CATEGORIES = {"chat_instruction", "function_tool", "agent_trajectories"}
_PACKED_CATEGORIES = {"ordinary_lm", "long_stem", "code"}
_ASSISTANT_ROLES = {"assistant", "model", "bot", "gpt", "chatgpt"}


def discover_default_chat_wrap_pile(model_id) -> bool:
    """Return the default Pile wrapping mode for a supported Discover model."""
    if canonical_model_id(model_id) in {LLAMA2_7B_BASE_MODEL_ID, QWEN3_8B_BASE_MODEL_ID}:
        return False
    return DISCOVER_CHAT_WRAP_PILE_DEFAULT


def discover_split_seeds(
    data_seed: int = DISCOVER_DATA_SEED,
    eval_seed: int = DISCOVER_EVAL_SEED,
    stage_index: int = 0,
) -> Dict[str, int]:
    """Return deterministic train/validation/eval seeds independent of optimizer RNG."""

    stage_index = max(0, int(stage_index))
    stage_offset = stage_index * 10_000
    return {
        "train": int(data_seed) + (stage_index + 1) * 10_000,
        "validation": int(eval_seed) + stage_offset,
        "eval": int(eval_seed) + DISCOVER_TEST_SEED_OFFSET + stage_offset,
    }


_TOOL_CALL_FIELD_HINTS = (
    "tool_call",
    "tool_calls",
    "function_call",
    "function_calls",
    "arguments",
)
_ASSISTANT_TARGET_FIELD_HINTS = (
    "answer",
    "assistant",
    "completion",
    "response",
    "output",
    "final",
    "summary",
)
_TOOL_CONTEXT_FIELD_HINTS = (
    "tool_output",
    "tool_outputs",
    "tool_response",
    "tool_responses",
    "tool_result",
    "tool_results",
    "function_output",
    "function_response",
    "function_result",
    "api_output",
    "api_response",
    "api_result",
    "observation",
    "observations",
)
_CONTEXT_FIELD_HINTS = (
    "abstract",
    "article",
    "context",
    "contexts",
    "document",
    "documents",
    "full_text",
    "query",
    "prompt",
    "instruction",
    "input",
    "paragraph",
    "paragraphs",
    "passage",
    "passages",
    "report",
    "tool",
    "tools",
    "schema",
    "observation",
    "observations",
    "user",
    "system",
    "developer",
)
_ROLE_PREFIX_RE = re.compile(
    r"^\s*(?:###\s*)?"
    r"(system|developer|user|human|assistant|model|bot|gpt|chatgpt|tool|function|observation|final answer)"
    r"\s*:\s*(.*)$",
    re.IGNORECASE,
)
_TEMPLATE_ROLE_RE = re.compile(
    r"^\s*(?:<\|im_start\|>|<\|start_header_id\|>)\s*"
    r"(system|developer|user|assistant|tool)"
    r"\s*(?:<\|end_header_id\|>)?\s*(.*)$",
    re.IGNORECASE,
)
_ROLE_NORMALISATION = {
    "human": "user",
    "function": "tool",
    "observation": "tool",
    "final answer": "assistant",
}


def _allocate_stage_tokens(
    total_tokens: int, stage_specs: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    allocated = []
    used = 0
    for index, spec in enumerate(stage_specs):
        item = dict(spec)
        if index == len(stage_specs) - 1:
            token_budget = max(0, total_tokens - used)
        else:
            token_budget = int(round(total_tokens * float(spec["fraction"])))
            used += token_budget
        if token_budget < int(item["seqlen"]):
            continue
        item["token_budget"] = token_budget
        allocated.append(item)
    return allocated


def build_length_curriculum(
    model_path: str, total_tokens: int = DISCOVER_REFERENCE_TOKENS
) -> List[Dict[str, Any]]:
    """Return the token-budget length schedule for the current supported model."""
    stage_specs = (
        {"name": DISCOVER_LENGTH_SCHEDULE_NAME, "seqlen": DISCOVER_TRAIN_SEQLEN, "fraction": 1.0},
    )
    return _allocate_stage_tokens(int(total_tokens), stage_specs)


def _source(
    category: str,
    target_tokens: int,
    path: str,
    *,
    split: str = "train",
    revision: Optional[str] = None,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    fields: Sequence[str] = ("text",),
    trust_remote_code: bool = False,
    requires_auth: bool = False,
    label: Optional[str] = None,
    format_name: Optional[str] = None,
    separator: Optional[str] = DISCOVER_DEFAULT_SEPARATOR,
    doc_chunk_mode: Optional[str] = None,
    min_doc_tokens: Optional[int] = None,
    require_doc_tokens_gt_seqlen: bool = False,
    min_supervised_tokens: Optional[int] = None,
    target_unit: str = "tokens",
    allow_chat_wrap: bool = True,
    chat_wrap: bool = False,
    chat_user_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "category": category,
        "target_tokens": target_tokens,
        "path": path,
        "split": split,
        "revision": revision,
        "name": name,
        "data_dir": data_dir,
        "fields": tuple(fields),
        "trust_remote_code": trust_remote_code,
        "requires_auth": requires_auth,
        "label": label or path,
        "format": format_name,
        "separator": separator,
        "doc_chunk_mode": doc_chunk_mode,
        "min_doc_tokens": min_doc_tokens,
        "require_doc_tokens_gt_seqlen": bool(require_doc_tokens_gt_seqlen),
        "min_supervised_tokens": min_supervised_tokens,
        "target_unit": str(target_unit),
        "allow_chat_wrap": bool(allow_chat_wrap),
        "chat_wrap": bool(chat_wrap),
        "chat_user_prompt": chat_user_prompt,
    }


def _source_uses_truncated_long_docs(source: Dict[str, Any]) -> bool:
    return str(source.get("doc_chunk_mode") or "").lower() in {
        "truncate_long_docs",
        "truncate_long_doc",
        "truncate",
    }


def _source_min_doc_tokens(source: Dict[str, Any], seqlen: int) -> int:
    configured = source.get("min_doc_tokens")
    seqlen_threshold = (
        int(seqlen) + 1 if source.get("require_doc_tokens_gt_seqlen") else int(seqlen)
    )
    if configured is None:
        return seqlen_threshold
    return max(int(configured), seqlen_threshold)


def _source_min_supervised_tokens(source: Dict[str, Any]) -> int:
    configured = source.get("min_supervised_tokens")
    if configured is None:
        return 0
    return max(0, int(configured))


def _source_target_unit(source: Dict[str, Any]) -> str:
    target_unit = str(source.get("target_unit", "tokens")).strip().lower()
    if target_unit not in {"tokens", "supervised_tokens"}:
        raise ValueError(
            f"Unsupported target_unit={target_unit!r} for "
            f"{source.get('label', source.get('path', 'dataset source'))}."
        )
    return target_unit


def _source_uses_chat_wrap(source: Dict[str, Any]) -> bool:
    return bool(source.get("chat_wrap", False))


def _source_min_supervised_tokens_for_seqlen(source: Dict[str, Any], seqlen: int) -> int:
    min_supervised_tokens = _source_min_supervised_tokens(source)
    if _source_uses_chat_wrap(source) and _source_uses_truncated_long_docs(source):
        min_supervised_tokens = max(min_supervised_tokens, int(seqlen))
    return min_supervised_tokens


DISCOVER_PILE_SOURCES: Sequence[Dict[str, Any]] = (
    # The original EleutherAI/pile loader points at the-eye archive URLs that
    # are not reliably available. This HF-hosted deduplicated mirror preserves
    # the Pile text format and works with datasets streaming.
    _source(
        "ordinary_lm",
        100,
        "EleutherAI/the_pile_deduplicated",
        fields=("text",),
        label="The Pile deduplicated docs longer than target seqlen",
        doc_chunk_mode="truncate_long_docs",
        require_doc_tokens_gt_seqlen=True,
    ),
)


DISCOVER_FINEWEB_SOURCES: Sequence[Dict[str, Any]] = (
    _source(
        "ordinary_lm",
        100,
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        fields=("text",),
        label="FineWeb sample-10BT plain web text",
    ),
)


DISCOVER_OPENWEBMATH_SOURCES: Sequence[Dict[str, Any]] = (
    _source(
        "ordinary_lm",
        100,
        "open-web-math/open-web-math",
        fields=("text",),
        label="OpenWebMath mathematical web text",
    ),
)


DISCOVER_FINEWEB2_CMN_HANI_SOURCES: Sequence[Dict[str, Any]] = (
    _source(
        "ordinary_lm",
        100,
        "HuggingFaceFW/fineweb-2",
        name="cmn_Hani",
        fields=("text",),
        label="FineWeb2 Mandarin Chinese (cmn_Hani)",
    ),
)


# The four weights are equal and collection is quotaed by supervised tokens,
# not by examples or masked chat-prefix/context tokens. FineWeb2 has no
# eng_Latn builder configuration, so the English component uses the official
# FineWeb sample-10BT configuration while the other three use FineWeb2.
DISCOVER_FINEWEB2_MULTILINGUAL_EQUAL_SOURCES: Sequence[Dict[str, Any]] = (
    _source(
        "ordinary_lm",
        25,
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        fields=("text",),
        label="FineWeb English equal-supervised-token multilingual component",
        target_unit="supervised_tokens",
        allow_chat_wrap=False,
    ),
    *(
        _source(
            "ordinary_lm",
            25,
            "HuggingFaceFW/fineweb-2",
            name=config_name,
            fields=("text",),
            label=f"FineWeb2 equal-supervised-token multilingual ({config_name})",
            target_unit="supervised_tokens",
            allow_chat_wrap=False,
        )
        for config_name in ("cmn_Hani", "spa_Latn", "arb_Arab")
    ),
)


DISCOVER_HOTPOTQA_DISTRACTOR_SOURCES: Sequence[Dict[str, Any]] = (
    _source(
        "chat_instruction",
        100,
        "hotpotqa/hotpot_qa",
        name="distractor",
        fields=("context", "question", "answer"),
        label="HotpotQA distractor context+question to answer",
        format_name="hotpotqa_distractor",
        min_supervised_tokens=1,
    ),
)


DISCOVER_HOTPOT_2WIKI_EQUAL_INPUT_SOURCES: Sequence[Dict[str, Any]] = (
    _source(
        "chat_instruction",
        50,
        "hotpotqa/hotpot_qa",
        revision="1908d6afbbead072334abe2965f91bd2709910ab",
        name="distractor",
        fields=("context", "question", "answer"),
        label="HotpotQA equal-input-token multihop QA component",
        format_name="multihop_qa",
        min_supervised_tokens=1,
        target_unit="tokens",
        allow_chat_wrap=False,
    ),
    _source(
        "chat_instruction",
        50,
        "framolfese/2WikiMultihopQA",
        revision="fe713bfbd1afbca1a65246741a75890405d56a3a",
        fields=("context", "question", "answer"),
        label="2WikiMultihopQA equal-input-token multihop QA component",
        format_name="multihop_qa",
        min_supervised_tokens=1,
        target_unit="tokens",
        allow_chat_wrap=False,
    ),
)


DISCOVER_ULTRACHAT_200K_SOURCES: Sequence[Dict[str, Any]] = (
    _source(
        "chat_instruction",
        100,
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        fields=("messages",),
        label="UltraChat 200k train_sft assistant-only targets",
        format_name="assistant_messages",
        min_supervised_tokens=1,
    ),
)


DISCOVER_DATASET_SOURCES: Dict[str, Sequence[Dict[str, Any]]] = {
    "pile": DISCOVER_PILE_SOURCES,
    "fineweb": DISCOVER_FINEWEB_SOURCES,
    "openwebmath": DISCOVER_OPENWEBMATH_SOURCES,
    "fineweb2_cmn_hani": DISCOVER_FINEWEB2_CMN_HANI_SOURCES,
    "fineweb2_multilingual_equal": DISCOVER_FINEWEB2_MULTILINGUAL_EQUAL_SOURCES,
    "hotpotqa_distractor": DISCOVER_HOTPOTQA_DISTRACTOR_SOURCES,
    "hotpot_2wiki_equal_input": DISCOVER_HOTPOT_2WIKI_EQUAL_INPUT_SOURCES,
    "ultrachat_200k": DISCOVER_ULTRACHAT_200K_SOURCES,
}


def _with_chat_wrapped_pile_sources(
    sources: Sequence[Dict[str, Any]],
    *,
    chat_wrap_pile: bool,
    chat_user_prompt: Optional[str],
) -> Sequence[Dict[str, Any]]:
    prompt = str(chat_user_prompt or DISCOVER_CHAT_WRAP_PILE_PROMPT).strip()
    if not prompt:
        prompt = DISCOVER_CHAT_WRAP_PILE_PROMPT

    prepared = []
    for source in sources:
        item = dict(source)
        if (
            chat_wrap_pile
            and item.get("category") == "ordinary_lm"
            and item.get("allow_chat_wrap", True)
        ):
            item["chat_wrap"] = True
            item["chat_user_prompt"] = prompt
            item["label"] = f"{item.get('label', item['path'])} (chat-wrapped)"
        else:
            item["chat_wrap"] = False
            item["chat_user_prompt"] = ""
        prepared.append(item)
    return tuple(prepared)


def get_discover_sources(
    dataset_name: str = DISCOVER_DATASET_NAME,
    *,
    chat_wrap_pile: bool = DISCOVER_CHAT_WRAP_PILE_DEFAULT,
    chat_user_prompt: Optional[str] = DISCOVER_CHAT_WRAP_PILE_PROMPT,
) -> Sequence[Dict[str, Any]]:
    try:
        sources = DISCOVER_DATASET_SOURCES[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported discover dataset: {dataset_name}. "
            f"The discover training path supports {DISCOVER_DATASET_CHOICES}."
        ) from exc
    return _with_chat_wrapped_pile_sources(
        sources,
        chat_wrap_pile=bool(chat_wrap_pile),
        chat_user_prompt=chat_user_prompt,
    )


def _should_pack_document_chunks(sources: Sequence[Dict[str, Any]]) -> bool:
    return not all(_source_uses_truncated_long_docs(source) for source in sources)


DISCOVER_SOURCES: Sequence[Dict[str, Any]] = get_discover_sources(DISCOVER_DATASET_NAME)


class _TokenizedTextWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class StreamingTokenChunkDataset(IterableDataset):
    """Stream HF documents into fixed-length causal-LM chunks without materializing them."""

    def __init__(
        self,
        tokenizer,
        seqlen: int,
        sources: Sequence[Dict[str, Any]],
        seed: int,
        *,
        num_samples: int,
        pad_token_id: int = 0,
        data_rank: int = 0,
        data_world_size: int = 1,
        exclude_fingerprints: Optional[Sequence[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.seqlen = max(1, int(seqlen))
        self.sources = [dict(source) for source in sources]
        self.seed = int(seed)
        self.num_samples = max(0, int(num_samples))
        self.pad_token_id = int(pad_token_id)
        self.data_rank = max(0, int(data_rank))
        self.data_world_size = max(1, int(data_world_size))
        self.exclude_fingerprints = set(exclude_fingerprints or [])

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        worker = get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        worker_count = int(worker.num_workers) if worker is not None else 1
        worker_count = max(1, worker_count)
        worker_target = self.num_samples // worker_count
        if worker_id < (self.num_samples % worker_count):
            worker_target += 1
        if worker_target <= 0:
            return

        shard_world = self.data_world_size * worker_count
        shard_rank = self.data_rank * worker_count + worker_id
        default_separator_ids = _encode_ids(self.tokenizer, DISCOVER_DEFAULT_SEPARATOR)
        emitted_for_worker = 0

        source_weights = [max(1, int(source.get("target_tokens", 1))) for source in self.sources]
        total_weight = max(1, sum(source_weights))
        raw_targets = [worker_target * weight / total_weight for weight in source_weights]
        source_targets = [int(target) for target in raw_targets]
        remaining = worker_target - sum(source_targets)
        for index, _fraction in sorted(
            enumerate(target - int(target) for target in raw_targets),
            key=lambda item: item[1],
            reverse=True,
        )[:remaining]:
            source_targets[index] += 1

        for source_index, source in enumerate(self.sources):
            source_worker_target = (
                source_targets[source_index] if source_index < len(source_targets) else 0
            )
            if source_worker_target <= 0:
                continue
            label = source.get("label", source["path"])
            category = source.get("category")
            if category not in (_PACKED_CATEGORIES | _SUPERVISED_CATEGORIES):
                raise ValueError(
                    f"Unsupported streaming training category {category!r} for {label}."
                )

            source_separator = source.get("separator")
            if isinstance(source_separator, str):
                source_separator_ids = _encode_ids(self.tokenizer, source_separator)
            else:
                source_separator_ids = default_separator_ids
            truncate_long_docs = _source_uses_truncated_long_docs(source)
            min_doc_tokens = (
                _source_min_doc_tokens(source, self.seqlen) if truncate_long_docs else self.seqlen
            )
            min_supervised_tokens = _source_min_supervised_tokens_for_seqlen(source, self.seqlen)

            source_emitted = 0
            source_cycle = 0
            while source_emitted < source_worker_target:
                token_buffer: List[int] = []
                label_buffer: List[int] = []
                source_chunk_index = 0
                emitted_this_cycle = 0
                source_seed = self.seed + source_index * 17 + source_cycle * 104_729
                source_cycle += 1
                try:
                    dataset = _load_hf_source_dataset(source, source_seed, shuffle=False)
                except TypeError:
                    dataset = _load_hf_source_dataset(source, source_seed)
                dataset, source_sharded = _shard_hf_iterable_dataset(
                    dataset,
                    shard_world,
                    shard_rank,
                    label,
                )
                dataset = _shuffle_hf_iterable_dataset(dataset, source_seed)
                fallback_sample_shard = shard_world > 1 and not source_sharded
                fallback_chunk_shard = False
                for sample_index, sample in enumerate(dataset):
                    if fallback_sample_shard and (sample_index % shard_world) != shard_rank:
                        continue
                    segments = _sample_to_segments(sample, source, self.tokenizer)
                    if not segments:
                        continue
                    if self.exclude_fingerprints:
                        sample_fingerprint = _segments_fingerprint(segments)
                        if sample_fingerprint in self.exclude_fingerprints:
                            continue
                    doc_ids: List[int] = []
                    doc_labels: List[int] = []
                    last_nonempty_has_loss = False
                    try:
                        for text, has_loss in segments:
                            ids = _encode_ids(self.tokenizer, text)
                            if not ids:
                                continue
                            last_nonempty_has_loss = bool(has_loss)
                            doc_ids.extend(ids)
                            if has_loss:
                                doc_labels.extend(ids)
                            else:
                                doc_labels.extend([IGNORE_INDEX] * len(ids))
                    except Exception as exc:
                        print(f"Warning: streaming tokenization skipped for {label}: {exc}")
                        continue

                    if not doc_ids:
                        continue
                    if min_supervised_tokens > 0:
                        supervised_token_count = sum(
                            1 for label_id in doc_labels if label_id != IGNORE_INDEX
                        )
                        if supervised_token_count < min_supervised_tokens:
                            continue
                    if truncate_long_docs:
                        if len(doc_ids) < min_doc_tokens:
                            continue
                        window_ids = doc_ids[: self.seqlen]
                        window_labels = doc_labels[: self.seqlen]
                        if all(label_id == IGNORE_INDEX for label_id in window_labels):
                            if fallback_chunk_shard:
                                source_chunk_index += 1
                            continue
                        should_emit = (
                            source_chunk_index % shard_world == shard_rank
                            if fallback_chunk_shard
                            else True
                        )
                        if should_emit:
                            input_ids = torch.tensor(window_ids, dtype=torch.long)
                            labels = torch.tensor(window_labels, dtype=torch.long)
                            yield {
                                "input_ids": input_ids,
                                "labels": labels,
                                "attention_mask_all_ones": True,
                                "pad_token_id": self.pad_token_id,
                            }
                            emitted_for_worker += 1
                            source_emitted += 1
                            emitted_this_cycle += 1
                            if (
                                emitted_for_worker >= worker_target
                                or source_emitted >= source_worker_target
                            ):
                                break
                        if fallback_chunk_shard:
                            source_chunk_index += 1
                        continue
                    if source_separator_ids:
                        doc_ids.extend(source_separator_ids)
                        if category in _PACKED_CATEGORIES or last_nonempty_has_loss:
                            doc_labels.extend(source_separator_ids)
                        else:
                            doc_labels.extend([IGNORE_INDEX] * len(source_separator_ids))
                    token_buffer.extend(doc_ids)
                    label_buffer.extend(doc_labels)

                    while len(token_buffer) >= self.seqlen:
                        window_ids = token_buffer[: self.seqlen]
                        window_labels = label_buffer[: self.seqlen]
                        should_emit = (
                            source_chunk_index % shard_world == shard_rank
                            if fallback_chunk_shard
                            else True
                        )
                        if should_emit and any(
                            label_id != IGNORE_INDEX for label_id in window_labels
                        ):
                            input_ids = torch.tensor(window_ids, dtype=torch.long)
                            labels = torch.tensor(window_labels, dtype=torch.long)
                            yield {
                                "input_ids": input_ids,
                                "labels": labels,
                                "attention_mask_all_ones": True,
                                "pad_token_id": self.pad_token_id,
                            }
                            emitted_for_worker += 1
                            source_emitted += 1
                            emitted_this_cycle += 1
                            if (
                                emitted_for_worker >= worker_target
                                or source_emitted >= source_worker_target
                            ):
                                break
                        del token_buffer[: self.seqlen]
                        del label_buffer[: self.seqlen]
                        if fallback_chunk_shard:
                            source_chunk_index += 1
                    if (
                        emitted_for_worker >= worker_target
                        or source_emitted >= source_worker_target
                    ):
                        break

                if emitted_for_worker >= worker_target:
                    return
                if source_emitted >= source_worker_target:
                    break
                if emitted_this_cycle == 0:
                    raise RuntimeError(
                        f"Streaming HF source {label} could not emit supervised chunks for "
                        f"rank shard {shard_rank}/{shard_world}."
                    )

        if emitted_for_worker >= worker_target:
            return

        raise RuntimeError(
            f"Streaming HF sources exhausted after emitting {emitted_for_worker}/{worker_target} "
            f"samples for rank shard {shard_rank}/{shard_world}."
        )


def _hf_auth_token():
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _stream_shuffle_buffer() -> int:
    try:
        return int(os.environ.get("DISCOVER_STREAM_SHUFFLE_BUFFER", "10000"))
    except ValueError:
        return 10_000


def _shuffle_hf_iterable_dataset(dataset, seed: int):
    shuffle_buffer = _stream_shuffle_buffer()
    try:
        if shuffle_buffer > 0:
            return dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
        return dataset
    except Exception:
        return dataset


def _shard_hf_iterable_dataset(dataset, num_shards: int, index: int, label: str):
    num_shards = max(1, int(num_shards))
    index = int(index)
    if num_shards <= 1:
        return dataset, False
    if str(os.environ.get("DISCOVER_USE_HF_DATASET_SHARD", "0")).strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return dataset, False
    shard_fn = getattr(dataset, "shard", None)
    if not callable(shard_fn):
        return dataset, False
    try:
        return shard_fn(num_shards=num_shards, index=index), True
    except TypeError:
        try:
            return shard_fn(num_shards, index), True
        except Exception as exc:
            print(
                f"Warning: failed to shard streaming HF source {label}: {exc}; falling back to chunk sharding."
            )
            return dataset, False
    except Exception as exc:
        print(
            f"Warning: failed to shard streaming HF source {label}: {exc}; falling back to chunk sharding."
        )
        return dataset, False


def _load_hf_source_dataset(source: Dict[str, Any], seed: int, *, shuffle: bool = True):
    _require_datasets()
    kwargs: Dict[str, Any] = {
        "split": source["split"],
        "streaming": True,
    }
    if source.get("data_dir"):
        kwargs["data_dir"] = source["data_dir"]
    if source.get("revision"):
        kwargs["revision"] = source["revision"]
    if source.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True

    token = _hf_auth_token()
    if token:
        kwargs["token"] = token
    elif source.get("requires_auth"):
        # Allows cached `huggingface-cli login` credentials to be used for gated HF datasets.
        kwargs["token"] = True

    try:
        if source.get("name"):
            dataset = load_dataset(source["path"], source["name"], **kwargs)
        else:
            dataset = load_dataset(source["path"], **kwargs)
    except Exception as exc:
        label = source.get("label", source["path"])
        auth_hint = (
            " Set HF_TOKEN or run `huggingface-cli login` if this is a gated dataset."
            if source.get("requires_auth")
            else ""
        )
        raise RuntimeError(f"Failed to load HF source {label}: {exc}.{auth_hint}") from exc

    if shuffle:
        return _shuffle_hf_iterable_dataset(dataset, seed)
    return dataset


def _normalise_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _looks_like_messages(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(item, dict) and "role" in item and "content" in item for item in value[:8]
        )
    )


def _format_messages(messages: Sequence[Dict[str, Any]], tokenizer) -> str:
    normalised = [
        {
            "role": str(message.get("role", "user")),
            "content": _normalise_message_content(message.get("content")),
        }
        for message in messages
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                normalised,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    return "\n".join(
        f"{message['role']}: {message['content']}" for message in normalised if message["content"]
    )


def _chat_generation_prefix(tokenizer, user_prompt: Optional[str]):
    prompt = str(user_prompt or DISCOVER_CHAT_WRAP_PILE_PROMPT).strip()
    if not prompt:
        prompt = DISCOVER_CHAT_WRAP_PILE_PROMPT
    user_message = {"role": "user", "content": prompt}
    sentinel = "<|beyond_target_text|>"
    if hasattr(tokenizer, "apply_chat_template"):
        if tokenizer.__class__.__name__ == "MistralCommonBackend":
            try:
                prefix_ids = tokenizer.apply_chat_template(
                    [user_message],
                    add_generation_prompt=True,
                    return_dict=False,
                )
                if prefix_ids:
                    return [int(token_id) for token_id in prefix_ids]
            except Exception:
                pass

        # Render a real assistant turn so templates that require channel/message
        # headers put the target text in the right state.
        try:
            rendered = tokenizer.apply_chat_template(
                [
                    user_message,
                    {"role": "assistant", "content": sentinel},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            rendered = str(rendered)
            if sentinel in rendered:
                prefix = rendered.split(sentinel, 1)[0]
                if prefix:
                    return prefix
        except Exception:
            pass

        messages = [user_message]
        try:
            prefix = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if prefix:
                return str(prefix)
        except TypeError:
            try:
                prefix = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                if prefix:
                    return f"{prefix}\nassistant:\n"
            except Exception:
                pass
        except Exception:
            pass
    return ""


def _maybe_chat_wrap_plain_lm_segments(
    segments: List[tuple],
    source: Dict[str, Any],
    tokenizer,
) -> List[tuple]:
    if not segments or not _source_uses_chat_wrap(source):
        return segments
    prefix = _chat_generation_prefix(tokenizer, source.get("chat_user_prompt"))
    if not prefix:
        return segments
    return [(prefix, False), *segments]


def _field_is_target(field_name: str) -> bool:
    name = str(field_name).lower()
    if any(hint in name for hint in _TOOL_CALL_FIELD_HINTS):
        return True
    if _field_is_context(field_name):
        return False
    return any(hint in name for hint in _ASSISTANT_TARGET_FIELD_HINTS)


def _field_is_context(field_name: str) -> bool:
    name = str(field_name).lower()
    if any(hint in name for hint in _TOOL_CALL_FIELD_HINTS):
        return False
    if any(hint in name for hint in _TOOL_CONTEXT_FIELD_HINTS):
        return True
    return any(hint in name for hint in _CONTEXT_FIELD_HINTS)


def _segment_text(value: Any, tokenizer) -> str:
    text = _value_to_text(value, tokenizer).strip()
    return text


def _normalise_role(role: str) -> str:
    role = str(role).strip().lower()
    return _ROLE_NORMALISATION.get(role, role)


def _strip_template_markers(text: str) -> str:
    return (
        text.replace("<|im_end|>", "")
        .replace("<|eot_id|>", "")
        .replace("<|end_of_turn|>", "")
        .strip()
    )


def _conversation_text_to_segments(text: str) -> List[tuple]:
    lines = text.splitlines()
    segments: List[tuple] = []
    current_role = "context"
    current_lines: List[str] = []
    saw_role_marker = False

    def flush_current():
        nonlocal current_lines
        content = _strip_template_markers("\n".join(current_lines))
        if content:
            target = current_role in _ASSISTANT_ROLES
            segments.append((f"{current_role}:\n", False))
            segments.append((content, target))
            segments.append(("\n", False))
        current_lines = []

    for line in lines:
        marker = _ROLE_PREFIX_RE.match(line)
        if marker is None:
            marker = _TEMPLATE_ROLE_RE.match(line)
        if marker is not None:
            flush_current()
            saw_role_marker = True
            current_role = _normalise_role(marker.group(1))
            remainder = _strip_template_markers(marker.group(2))
            current_lines = [remainder] if remainder else []
        else:
            current_lines.append(line)

    flush_current()
    if not saw_role_marker:
        return []
    segments.append(("\n", False))
    return segments


def _message_to_segments(message: Dict[str, Any], tokenizer) -> List[tuple]:
    role = str(message.get("role", message.get("from", ""))).lower()
    if not role and "speaker" in message:
        role = str(message.get("speaker", "")).lower()
    is_target = role in _ASSISTANT_ROLES
    display_role = role or ("assistant" if is_target else "context")
    segments: List[tuple] = [(f"{display_role}:\n", False)]

    content = message.get("content", message.get("value", message.get("text", None)))
    content_text = _normalise_message_content(content).strip()
    if content_text:
        segments.append((content_text, is_target))
        segments.append(("\n", False))

    for key in ("tool_calls", "tool_call", "function_call"):
        if key in message and message[key] not in (None, "", [], {}):
            call_text = json.dumps(message[key], ensure_ascii=False, sort_keys=True)
            segments.append((f"{key}:\n", False))
            segments.append((call_text, is_target))
            segments.append(("\n", False))

    # Preserve any assistant-side final/answer/output fields that appear outside
    # the normal content slot; tool/user observations remain context.
    for key, value in message.items():
        if key in {
            "role",
            "from",
            "speaker",
            "content",
            "value",
            "text",
            "tool_calls",
            "tool_call",
            "function_call",
        }:
            continue
        if value in (None, "", [], {}):
            continue
        if is_target and _field_is_target(key):
            key_target = True
        elif _field_is_context(key):
            key_target = False
        else:
            continue
        value_text = _segment_text(value, tokenizer)
        if value_text:
            segments.append((f"{key}:\n", False))
            segments.append((value_text, key_target))
            segments.append(("\n", False))

    segments.append(("\n", False))
    return segments


def _looks_like_dialogue_list(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    checked = value[:8]
    return all(
        isinstance(item, dict)
        and (
            "role" in item
            or "from" in item
            or "speaker" in item
            or "content" in item
            or "value" in item
        )
        for item in checked
    )


def _dialogue_to_segments(messages: Sequence[Dict[str, Any]], tokenizer) -> List[tuple]:
    segments: List[tuple] = []
    for message in messages:
        if isinstance(message, dict):
            segments.extend(_message_to_segments(message, tokenizer))
        else:
            text = _segment_text(message, tokenizer)
            if text:
                segments.append((f"{text}\n\n", False))
    return segments


def _value_to_segments(value: Any, tokenizer, default_target: bool) -> List[tuple]:
    if value in (None, "", [], {}):
        return []
    if _looks_like_dialogue_list(value):
        return _dialogue_to_segments(value, tokenizer)
    if isinstance(value, str) and not default_target:
        parsed_segments = _conversation_text_to_segments(value)
        if parsed_segments:
            return parsed_segments
    if isinstance(value, list):
        segments: List[tuple] = []
        for item in value:
            segments.extend(_value_to_segments(item, tokenizer, default_target))
        return segments
    if isinstance(value, dict):
        if "role" in value or "from" in value or "speaker" in value:
            return _message_to_segments(value, tokenizer)
        segments = []
        for key, item in value.items():
            if item in (None, "", [], {}):
                continue
            if _field_is_target(key):
                target = True
            elif _field_is_context(key):
                target = False
            else:
                target = default_target
            text = _segment_text(item, tokenizer)
            if text:
                segments.append((f"{key}:\n", False))
                segments.append((text, target))
                segments.append(("\n\n", False))
        return segments
    text = _segment_text(value, tokenizer)
    return [(text, default_target), ("\n\n", False)] if text else []


def _value_to_text(value: Any, tokenizer) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if _looks_like_messages(value):
        return _format_messages(value, tokenizer)
    if isinstance(value, list):
        parts = [_value_to_text(item, tokenizer) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if "role" in value and "content" in value:
            return (
                f"{value.get('role', 'user')}: {_normalise_message_content(value.get('content'))}"
            )
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _sample_to_text(sample: Dict[str, Any], source: Dict[str, Any], tokenizer) -> str:
    blocks = []
    plain_fields = {"text", "content", "sample", "messages", "conversation", "trajectory"}
    for field in source.get("fields", ()):
        if field not in sample:
            continue
        text = _value_to_text(sample[field], tokenizer).strip()
        if not text:
            continue
        blocks.append(text if field in plain_fields else f"{field}:\n{text}")

    if blocks:
        return "\n\n".join(blocks)

    fallback = {
        key: value
        for key, value in sample.items()
        if value not in (None, "", [], {})
        and not key.endswith("_moderation")
        and key
        not in {"id", "url", "file_path", "timestamp", "language_score", "score", "int_score"}
    }
    return _value_to_text(fallback, tokenizer).strip()


def _multihop_qa_context_blocks(context: Any) -> List[str]:
    """Normalize HotpotQA-style and pair-list multihop-QA contexts."""

    context_rows = []
    if isinstance(context, dict):
        titles = list(context.get("title") or [])
        sentence_groups = list(context.get("sentences") or [])
        for index in range(max(len(titles), len(sentence_groups))):
            title = titles[index] if index < len(titles) else f"Passage {index + 1}"
            sentences = sentence_groups[index] if index < len(sentence_groups) else []
            context_rows.append((title, sentences))
    elif isinstance(context, (list, tuple)):
        for index, row in enumerate(context):
            if isinstance(row, dict):
                title = row.get("title", f"Passage {index + 1}")
                sentences = row.get("sentences", row.get("sentence", []))
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                title, sentences = row[0], row[1]
            else:
                title, sentences = f"Passage {index + 1}", row
            context_rows.append((title, sentences))

    context_blocks = []
    for index, (raw_title, sentences) in enumerate(context_rows):
        title = str(raw_title).strip() or f"Passage {index + 1}"
        if isinstance(sentences, (list, tuple)):
            passage = " ".join(
                str(sentence).strip() for sentence in sentences if str(sentence).strip()
            )
        else:
            passage = str(sentences).strip()
        context_blocks.append(f"[{title}]\n{passage}" if passage else f"[{title}]")
    return context_blocks


def _multihop_qa_segments(sample: Dict[str, Any]) -> List[tuple]:
    """Format multihop-QA rows with context/question masked and answer supervised."""

    context = sample.get("context") or {}
    context_text = "\n\n".join(_multihop_qa_context_blocks(context)).strip()
    question = _normalise_message_content(sample.get("question")).strip()
    answer = _normalise_message_content(sample.get("answer")).strip()
    if not answer or (not context_text and not question):
        return []

    segments: List[tuple] = []
    if context_text:
        segments.extend([("context:\n", False), (context_text, False), ("\n\n", False)])
    if question:
        segments.extend([("question:\n", False), (question, False), ("\n\n", False)])
    segments.extend([("assistant:\n", False), (answer, True), ("\n", False)])
    return segments


def _hotpotqa_distractor_segments(sample: Dict[str, Any]) -> List[tuple]:
    """Backward-compatible alias for the original HotpotQA formatter."""

    return _multihop_qa_segments(sample)


def _sample_to_segments(sample: Dict[str, Any], source: Dict[str, Any], tokenizer) -> List[tuple]:
    format_name = source.get("format")
    if format_name == "hotpotqa_distractor":
        return _hotpotqa_distractor_segments(sample)
    if format_name == "multihop_qa":
        return _multihop_qa_segments(sample)
    if source.get("category") not in _SUPERVISED_CATEGORIES:
        text = _sample_to_text(sample, source, tokenizer)
        segments = [(text, True)] if text else []
        return _maybe_chat_wrap_plain_lm_segments(segments, source, tokenizer)

    segments: List[tuple] = []
    for field in source.get("fields", ()):
        if field not in sample:
            continue
        value = sample[field]
        if value in (None, "", [], {}):
            continue
        if field in {"messages", "conversation", "conversations"}:
            field_segments = _value_to_segments(value, tokenizer, default_target=False)
        elif _field_is_target(field):
            field_segments = _value_to_segments(value, tokenizer, default_target=True)
        else:
            field_segments = _value_to_segments(value, tokenizer, default_target=False)
        if field_segments:
            if field not in {"messages", "conversation", "conversations", "sample", "trajectory"}:
                segments.append((f"{field}:\n", False))
            segments.extend(field_segments)

    if not segments:
        fallback = {
            key: value
            for key, value in sample.items()
            if value not in (None, "", [], {})
            and not key.endswith("_moderation")
            and key
            not in {"id", "url", "file_path", "timestamp", "language_score", "score", "int_score"}
        }
        segments = _value_to_segments(fallback, tokenizer, default_target=False)

    return [(text, target) for text, target in segments if text]


def _encode_ids(tokenizer, text: str) -> List[int]:
    if torch.is_tensor(text):
        return [int(token_id) for token_id in text.detach().cpu().flatten().tolist()]
    if isinstance(text, (list, tuple)) and not isinstance(text, (str, bytes)):
        return [int(token_id) for token_id in text]
    try:
        ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    except TypeError:
        ids = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _tokenizer_eos_ids(tokenizer) -> List[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token:
        ids = _encode_ids(tokenizer, eos_token)
        if ids:
            return ids
    return _encode_ids(tokenizer, "\n\n")


def _sample_token_count(sample_tuple) -> int:
    if len(sample_tuple) >= 3:
        attention_mask = sample_tuple[2]
        try:
            if torch.is_tensor(attention_mask):
                return int(attention_mask.to(dtype=torch.long).sum().item())
            return sum(1 for value in attention_mask if int(value) != 0)
        except Exception:
            pass
    input_ids = sample_tuple[0]
    if torch.is_tensor(input_ids):
        return int(input_ids.numel())
    return len(input_ids)


def _samples_token_count(samples: Sequence[Any]) -> int:
    return sum(_sample_token_count(sample) for sample in samples)


def _tokenizer_pad_id(tokenizer) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_token_id, int):
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        return int(eos_token_id)
    return 0


def _to_1d_long_tensor(value) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(dtype=torch.long).reshape(-1).cpu()
    return torch.tensor(list(value), dtype=torch.long).reshape(-1)


def _make_packed_sample(input_ids, labels, attention_mask, position_ids=None):
    if position_ids is not None:
        return (
            torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            torch.tensor(labels, dtype=torch.long).unsqueeze(0),
            torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0),
            torch.tensor(position_ids, dtype=torch.long).unsqueeze(0),
        )
    return (
        torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0),
    )


def pack_samples_to_seqlen(
    samples: Sequence[Any],
    seqlen: int,
    *,
    pad_token_id: int = 0,
    drop_last: bool = False,
) -> List[Any]:
    """Pack variable-length samples into fixed-length chunks.

    Document boundaries are represented by separator tokens already inserted by
    the data formatter.  We intentionally keep a 2D attention mask here; dense
    block-causal masks are not practical for 32K/128K eval.
    """

    seqlen = max(1, int(seqlen))
    pad_token_id = int(pad_token_id)
    packed: List[Any] = []
    cur_ids: List[int] = []
    cur_labels: List[int] = []
    cur_attention: List[int] = []

    def flush_current(force: bool = False) -> None:
        if not cur_ids:
            return
        if len(cur_ids) < seqlen:
            if drop_last and not force:
                cur_ids.clear()
                cur_labels.clear()
                cur_attention.clear()
                return
            pad_len = seqlen - len(cur_ids)
            cur_ids.extend([pad_token_id] * pad_len)
            cur_labels.extend([IGNORE_INDEX] * pad_len)
            cur_attention.extend([0] * pad_len)
        packed.append(_make_packed_sample(cur_ids, cur_labels, cur_attention))
        cur_ids.clear()
        cur_labels.clear()
        cur_attention.clear()

    for sample in samples:
        if not sample:
            continue
        input_ids = _to_1d_long_tensor(sample[0]).tolist()
        labels = _to_1d_long_tensor(sample[1]).tolist()
        if len(sample) >= 3:
            attention = _to_1d_long_tensor(sample[2]).tolist()
            if len(attention) < len(input_ids):
                attention.extend([1] * (len(input_ids) - len(attention)))
            elif len(attention) > len(input_ids):
                attention = attention[: len(input_ids)]
            keep = [idx for idx, value in enumerate(attention) if int(value) != 0]
            input_ids = [input_ids[idx] for idx in keep]
            labels = [labels[idx] for idx in keep if idx < len(labels)]
        if len(labels) < len(input_ids):
            labels.extend([IGNORE_INDEX] * (len(input_ids) - len(labels)))
        elif len(labels) > len(input_ids):
            labels = labels[: len(input_ids)]

        offset = 0
        while offset < len(input_ids):
            if len(cur_ids) >= seqlen:
                flush_current(force=True)
            space = seqlen - len(cur_ids)
            take = min(space, len(input_ids) - offset)
            if take <= 0:
                flush_current(force=True)
                continue
            cur_ids.extend(int(v) for v in input_ids[offset : offset + take])
            cur_labels.extend(int(v) for v in labels[offset : offset + take])
            cur_attention.extend([1] * take)
            offset += take
            if len(cur_ids) >= seqlen:
                flush_current(force=True)

    flush_current(force=False)
    return packed


def _segments_fingerprint(segments: Sequence[tuple]) -> str:
    payload = "\n".join(str(text) for text, _target in segments)
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def _truncate_to_supervised_budget(
    input_ids: Sequence[int],
    labels: Sequence[int],
    supervised_budget: int,
) -> tuple:
    """Keep context plus at most ``supervised_budget`` non-masked target tokens."""

    supervised_budget = max(0, int(supervised_budget))
    if supervised_budget == 0:
        return [], []
    seen = 0
    cutoff = len(input_ids)
    for index, label_id in enumerate(labels):
        if int(label_id) == IGNORE_INDEX:
            continue
        seen += 1
        if seen >= supervised_budget:
            cutoff = index + 1
            break
    return list(input_ids[:cutoff]), list(labels[:cutoff])


def _source_targets_for_token_budget(
    sources: Sequence[Dict[str, Any]], total_tokens: int
) -> List[Dict[str, Any]]:
    base_total = sum(int(source["target_tokens"]) for source in sources)
    scaled_sources = []
    allocated = 0
    for index, source in enumerate(sources):
        scaled = dict(source)
        if index == len(sources) - 1:
            target = max(1, total_tokens - allocated)
        else:
            target = max(1, int(round(total_tokens * int(source["target_tokens"]) / base_total)))
            allocated += target
        scaled["target_tokens"] = target
        scaled_sources.append(scaled)
    return scaled_sources


def _heldout_stage_specs_from_curriculum(
    curriculum: Sequence[Dict[str, Any]],
    total_chunks: int,
) -> List[Dict[str, Any]]:
    """Build held-out budgets with an exact global sequence/chunk count."""
    if not curriculum:
        return []
    train_total = sum(int(stage["token_budget"]) for stage in curriculum)
    if train_total <= 0:
        raise ValueError("Curriculum token budget must be positive.")

    chunk_budget = max(0, int(total_chunks))
    if chunk_budget == 0:
        return []

    weighted = []
    for index, stage in enumerate(curriculum):
        exact = chunk_budget * int(stage["token_budget"]) / train_total
        base_chunks = int(math.floor(exact))
        weighted.append((index, stage, exact, base_chunks))

    allocated_chunks = sum(item[3] for item in weighted)
    remainder = chunk_budget - allocated_chunks
    if remainder > 0:
        by_fraction = sorted(
            weighted,
            key=lambda item: (item[2] - item[3], int(item[1]["token_budget"])),
            reverse=True,
        )
        increments = {index: 0 for index, _stage, _exact, _base in weighted}
        for index, _stage, _exact, _base in by_fraction[:remainder]:
            increments[index] += 1
    else:
        increments = {index: 0 for index, _stage, _exact, _base in weighted}

    heldout_stages = []
    for index, stage, _exact, base_chunks in weighted:
        stage_chunks = base_chunks + increments[index]
        if stage_chunks <= 0:
            continue
        seqlen = int(stage["seqlen"])
        heldout_stages.append(
            {
                "name": stage["name"],
                "seqlen": seqlen,
                "sample_budget": int(stage_chunks),
                "token_budget": int(stage_chunks) * seqlen,
                "train_token_fraction": int(stage["token_budget"]) / train_total,
            }
        )
    return heldout_stages


def _collect_hf_token_chunks(
    tokenizer,
    seqlen: int,
    sources: Sequence[Dict[str, Any]],
    seed: int,
    *,
    return_chunks: bool,
    exclude_fingerprints: Optional[set] = None,
    record_fingerprints: Optional[set] = None,
    dedupe_recorded: bool = False,
    pack_documents_to_seqlen: bool = False,
    shuffle_chunks: bool = False,
):
    chunks = []
    token_buffer: List[int] = []
    label_buffer: List[int] = []
    flat_tokens: List[int] = []
    total_tokens = 0
    supervised_tokens = 0
    next_log = _TOKEN_LOG_INTERVAL
    default_separator_ids = _encode_ids(tokenizer, DISCOVER_DEFAULT_SEPARATOR)
    pad_token_id = _tokenizer_pad_id(tokenizer)
    pack_ids: List[int] = []
    pack_labels: List[int] = []
    pack_attention: List[int] = []

    def flush_pack(force: bool = False):
        if not pack_ids:
            return
        if len(pack_ids) < seqlen:
            if not force:
                return
            pad_len = seqlen - len(pack_ids)
            pack_ids.extend([pad_token_id] * pad_len)
            pack_labels.extend([IGNORE_INDEX] * pad_len)
            pack_attention.extend([0] * pad_len)
        if any(label_id != IGNORE_INDEX for label_id in pack_labels):
            chunks.append(_make_packed_sample(pack_ids, pack_labels, pack_attention))
        pack_ids.clear()
        pack_labels.clear()
        pack_attention.clear()

    def append_doc_to_pack(doc_ids: Sequence[int], doc_labels: Sequence[int]):
        offset = 0
        doc_len = len(doc_ids)
        while offset < doc_len:
            if len(pack_ids) >= seqlen:
                flush_pack(force=True)
            space = seqlen - len(pack_ids)
            take = min(space, doc_len - offset)
            if take <= 0:
                flush_pack(force=True)
                continue
            pack_ids.extend(int(v) for v in doc_ids[offset : offset + take])
            pack_labels.extend(int(v) for v in doc_labels[offset : offset + take])
            pack_attention.extend([1] * take)
            offset += take
            if len(pack_ids) >= seqlen:
                flush_pack(force=True)

    for source_index, source in enumerate(sources):
        label = source.get("label", source["path"])
        category = source.get("category")
        target_tokens = int(source["target_tokens"])
        target_unit = _source_target_unit(source)
        source_seed = seed + source_index * 17
        dataset = _load_hf_source_dataset(source, source_seed)
        source_tokens = 0
        source_supervised_tokens = 0
        source_docs = 0
        source_scanned_docs = 0
        source_skipped_seen = 0
        source_skipped_short = 0
        source_skipped_low_target = 0
        source_docs_ge_half_seqlen = 0
        source_docs_ge_seqlen = 0
        source_max_doc_tokens = 0
        source_separator = source.get("separator")
        if isinstance(source_separator, str):
            source_separator_ids = _encode_ids(tokenizer, source_separator)
        else:
            source_separator_ids = default_separator_ids
        truncate_long_docs = _source_uses_truncated_long_docs(source)
        min_doc_tokens = _source_min_doc_tokens(source, seqlen) if truncate_long_docs else seqlen
        min_supervised_tokens = _source_min_supervised_tokens_for_seqlen(source, seqlen)
        source_target_chunks = (
            max(1, (target_tokens + seqlen - 1) // seqlen) if truncate_long_docs else None
        )
        doc_threshold_desc = (
            f"> {seqlen:,}"
            if truncate_long_docs and source.get("require_doc_tokens_gt_seqlen")
            else f">= {min_doc_tokens:,}"
        )
        mode_desc = (
            f", mode=truncate docs {doc_threshold_desc} tokens to {seqlen:,}"
            if truncate_long_docs
            else ""
        )
        print(f"  {source['category']}: {label} -> {target_tokens:,} {target_unit}{mode_desc}")

        progress = tqdm(
            total=None,
            desc=f"    scan {label}",
            unit="doc",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=True,
        )

        def refresh_progress(force: bool = False):
            if progress is None:
                return
            accepted = (
                f"{source_docs}/{source_target_chunks}"
                if source_target_chunks is not None
                else f"{source_docs}"
            )
            progress.set_postfix(
                {
                    "accepted": accepted,
                    "tok": f"{source_tokens / 1_000_000:.2f}M",
                    "sup": f"{source_supervised_tokens / 1_000_000:.2f}M",
                    "quota": (
                        f"{(source_supervised_tokens if target_unit == 'supervised_tokens' else source_tokens) / 1_000_000:.2f}M/"
                        f"{target_tokens / 1_000_000:.2f}M"
                    ),
                    "short": source_skipped_short,
                    "low_tgt": source_skipped_low_target,
                    "seen": source_skipped_seen,
                    "max": source_max_doc_tokens,
                },
                refresh=force,
            )

        try:
            for sample in dataset:
                source_scanned_docs += 1
                progress.update(1)
                segments = _sample_to_segments(sample, source, tokenizer)
                if not segments:
                    if source_scanned_docs % 100 == 0:
                        refresh_progress()
                    continue
                sample_fingerprint = _segments_fingerprint(segments)
                if exclude_fingerprints is not None and sample_fingerprint in exclude_fingerprints:
                    source_skipped_seen += 1
                    if source_scanned_docs % 100 == 0:
                        refresh_progress()
                    continue
                if (
                    dedupe_recorded
                    and record_fingerprints is not None
                    and sample_fingerprint in record_fingerprints
                ):
                    source_skipped_seen += 1
                    if source_scanned_docs % 100 == 0:
                        refresh_progress()
                    continue

                doc_ids: List[int] = []
                doc_labels: List[int] = []
                last_nonempty_has_loss = False
                try:
                    for text, has_loss in segments:
                        ids = _encode_ids(tokenizer, text)
                        if not ids:
                            continue
                        last_nonempty_has_loss = bool(has_loss)
                        doc_ids.extend(ids)
                        if has_loss:
                            doc_labels.extend(ids)
                        else:
                            doc_labels.extend([IGNORE_INDEX] * len(ids))
                except Exception as exc:
                    print(f"Warning: tokenization skipped for {label}: {exc}")
                    if source_scanned_docs % 100 == 0:
                        refresh_progress()
                    continue

                if not doc_ids:
                    if source_scanned_docs % 100 == 0:
                        refresh_progress()
                    continue
                raw_doc_token_len = len(doc_ids)
                raw_doc_supervised_tokens = sum(
                    1 for label_id in doc_labels if label_id != IGNORE_INDEX
                )
                if min_supervised_tokens > 0 and raw_doc_supervised_tokens < min_supervised_tokens:
                    source_skipped_low_target += 1
                    source_max_doc_tokens = max(source_max_doc_tokens, raw_doc_token_len)
                    if source_scanned_docs % 100 == 0:
                        refresh_progress()
                    continue
                if truncate_long_docs:
                    if raw_doc_token_len < min_doc_tokens:
                        source_skipped_short += 1
                        source_max_doc_tokens = max(source_max_doc_tokens, raw_doc_token_len)
                        if source_scanned_docs % 100 == 0:
                            refresh_progress()
                        continue
                    doc_ids = doc_ids[:seqlen]
                    doc_labels = doc_labels[:seqlen]
                else:
                    if source_separator_ids:
                        doc_ids.extend(source_separator_ids)
                        if category in _PACKED_CATEGORIES:
                            doc_labels.extend(source_separator_ids)
                        elif last_nonempty_has_loss:
                            doc_labels.extend(source_separator_ids)
                        else:
                            doc_labels.extend([IGNORE_INDEX] * len(source_separator_ids))
                    source_progress = (
                        source_supervised_tokens
                        if target_unit == "supervised_tokens"
                        else source_tokens
                    )
                    remaining = target_tokens - source_progress
                    if remaining <= 0:
                        break
                    if target_unit == "supervised_tokens":
                        doc_ids, doc_labels = _truncate_to_supervised_budget(
                            doc_ids,
                            doc_labels,
                            remaining,
                        )
                        raw_doc_token_len = min(raw_doc_token_len, len(doc_ids))
                    elif len(doc_ids) > remaining:
                        doc_ids = doc_ids[:remaining]
                        doc_labels = doc_labels[:remaining]
                        raw_doc_token_len = min(raw_doc_token_len, len(doc_ids))
                if not doc_ids:
                    break
                doc_token_len = len(doc_ids)
                doc_supervised_tokens = sum(
                    1 for label_id in doc_labels if label_id != IGNORE_INDEX
                )

                source_docs += 1
                source_docs_ge_half_seqlen += int(raw_doc_token_len >= max(1, seqlen // 2))
                source_docs_ge_seqlen += int(raw_doc_token_len >= seqlen)
                source_max_doc_tokens = max(source_max_doc_tokens, raw_doc_token_len)
                source_tokens += doc_token_len
                source_supervised_tokens += doc_supervised_tokens
                total_tokens += doc_token_len
                supervised_tokens += doc_supervised_tokens
                if record_fingerprints is not None:
                    record_fingerprints.add(sample_fingerprint)

                if return_chunks:
                    if truncate_long_docs:
                        chunks.append(_make_packed_sample(doc_ids, doc_labels, [1] * doc_token_len))
                    elif pack_documents_to_seqlen:
                        append_doc_to_pack(doc_ids, doc_labels)
                    elif category in _PACKED_CATEGORIES:
                        token_buffer.extend(doc_ids)
                        label_buffer.extend(doc_labels)
                        while len(token_buffer) >= seqlen:
                            chunk = torch.tensor(token_buffer[:seqlen], dtype=torch.long).unsqueeze(
                                0
                            )
                            labels = torch.tensor(
                                label_buffer[:seqlen], dtype=torch.long
                            ).unsqueeze(0)
                            if torch.any(labels != IGNORE_INDEX):
                                chunks.append((chunk, labels))
                            del token_buffer[:seqlen]
                            del label_buffer[:seqlen]
                    elif category in _SUPERVISED_CATEGORIES:
                        for start in range(0, len(doc_ids), seqlen):
                            window_ids = doc_ids[start : start + seqlen]
                            window_labels = doc_labels[start : start + seqlen]
                            if not window_ids:
                                continue
                            if all(label_id == IGNORE_INDEX for label_id in window_labels):
                                continue
                            chunk = torch.tensor(window_ids, dtype=torch.long).unsqueeze(0)
                            labels = torch.tensor(window_labels, dtype=torch.long).unsqueeze(0)
                            attention_mask = torch.ones_like(chunk)
                            chunks.append((chunk, labels, attention_mask))
                    else:
                        raise ValueError(f"Unsupported discover source category: {category}")
                else:
                    flat_tokens.extend(doc_ids)

                if total_tokens >= next_log:
                    progress.write(f"    collected {total_tokens:,} tokens")
                    next_log += _TOKEN_LOG_INTERVAL

                refresh_progress(force=True)

                if truncate_long_docs:
                    if source_target_chunks is not None and source_docs >= source_target_chunks:
                        break
                elif (
                    source_supervised_tokens >= target_tokens
                    if target_unit == "supervised_tokens"
                    else source_tokens >= target_tokens
                ):
                    break
        finally:
            refresh_progress(force=True)
            progress.close()

        if truncate_long_docs:
            if source_target_chunks is not None and source_docs < source_target_chunks:
                raise RuntimeError(
                    f"HF source {label} exhausted after {source_docs:,}/{source_target_chunks:,} "
                    f"accepted docs {doc_threshold_desc} tokens "
                    f"({source_tokens:,}/{target_tokens:,} requested tokens)."
                )
        elif (
            source_supervised_tokens < target_tokens
            if target_unit == "supervised_tokens"
            else source_tokens < target_tokens
        ):
            collected_target = (
                source_supervised_tokens if target_unit == "supervised_tokens" else source_tokens
            )
            raise RuntimeError(
                f"HF source {label} exhausted at {collected_target:,}/{target_tokens:,} "
                f"{target_unit}."
            )
        if source.get("category") in _SUPERVISED_CATEGORIES and source_supervised_tokens == 0:
            print(
                f"    warning: no supervised target tokens found for {label}; check dataset schema."
            )
        print(
            f"    done: {source_docs:,} accepted docs from {source_scanned_docs:,} scanned docs, "
            f"{source_tokens:,} tokens, "
            f"{source_supervised_tokens:,} supervised target tokens"
        )
        if source_docs:
            avg_doc_tokens = source_tokens / source_docs
            print(
                f"    length: avg={avg_doc_tokens:,.0f} tokens/doc, "
                f"max={source_max_doc_tokens:,}, "
                f">={seqlen // 2:,}={source_docs_ge_half_seqlen:,}, "
                f">={seqlen:,}={source_docs_ge_seqlen:,}"
            )
        if source_skipped_seen:
            print(f"    skipped {source_skipped_seen:,} previously seen/duplicate docs")
        if source_skipped_short:
            print(
                f"    skipped {source_skipped_short:,} docs shorter than {min_doc_tokens:,} tokens"
            )
        if source_skipped_low_target:
            print(
                f"    skipped {source_skipped_low_target:,} docs with fewer than "
                f"{min_supervised_tokens:,} supervised target tokens"
            )

    if return_chunks:
        if pack_documents_to_seqlen:
            flush_pack(force=True)
        if shuffle_chunks and chunks:
            random.Random(seed + 104_729).shuffle(chunks)
            print(f"Shuffled {len(chunks):,} training chunks/windows with seed {seed + 104_729}.")
        print(
            f"Built {len(chunks):,} training chunks/windows "
            f"({total_tokens:,} source tokens, {supervised_tokens:,} supervised target tokens)."
        )
        return chunks

    usable_tokens = (len(flat_tokens) // seqlen) * seqlen
    if usable_tokens == 0:
        raise RuntimeError("Validation mixture did not produce enough tokens for one sequence.")
    input_ids = torch.tensor(flat_tokens[:usable_tokens], dtype=torch.long).unsqueeze(0)
    print(
        f"Built validation tensor with {usable_tokens:,} tokens ({usable_tokens // seqlen:,} chunks)."
    )
    return _TokenizedTextWrapper(input_ids)


def get_training_dataset(
    seed,
    seqlen,
    model_path,
    dataset_name=DISCOVER_DATASET_NAME,
    load_train=True,
    train_tokens=DISCOVER_REFERENCE_TOKENS,
    return_eval_stages=False,
    chat_wrap_pile=DISCOVER_CHAT_WRAP_PILE_DEFAULT,
    chat_user_prompt=DISCOVER_CHAT_WRAP_PILE_PROMPT,
    data_seed=DISCOVER_DATA_SEED,
    eval_seed=DISCOVER_EVAL_SEED,
):
    """Build a single-length discover HF mixture used for QAT or eval.

    ``seed`` is retained for call-site compatibility; data selection is governed
    by ``data_seed`` and ``eval_seed`` so optimizer RNG changes cannot move the
    held-out examples.
    """
    discover_sources = get_discover_sources(
        dataset_name,
        chat_wrap_pile=chat_wrap_pile,
        chat_user_prompt=chat_user_prompt,
    )

    model_type_local = detect_model_type(model_path)
    tokenizer_local = load_tokenizer(model_path, model_type_local)
    split_seeds = discover_split_seeds(data_seed=data_seed, eval_seed=eval_seed)

    trainloader = []
    train_fingerprints = set()
    if load_train:
        train_sources = _source_targets_for_token_budget(discover_sources, int(train_tokens))
        print(
            f"Loading discover HF {dataset_name} training mixture "
            f"({int(train_tokens):,} tokens, seqlen={seqlen})."
        )
        trainloader = _collect_hf_token_chunks(
            tokenizer_local,
            seqlen,
            train_sources,
            split_seeds["train"],
            return_chunks=True,
            record_fingerprints=train_fingerprints,
            shuffle_chunks=True,
        )

    heldout_chunks = DISCOVER_VALIDATION_CHUNKS if return_eval_stages else DISCOVER_EVAL_CHUNKS
    heldout_role = "validation" if return_eval_stages else "eval"
    val_tokens = seqlen * heldout_chunks
    val_sources = _source_targets_for_token_budget(discover_sources, val_tokens)
    print(
        f"Loading discover HF {dataset_name} held-out {heldout_role} mixture "
        f"({heldout_chunks:,} chunks, {val_tokens:,} tokens, "
        f"excluding {len(train_fingerprints):,} train/val fingerprints)."
    )
    if return_eval_stages:
        eval_fingerprints = set()
        eval_samples = _collect_hf_token_chunks(
            tokenizer_local,
            seqlen,
            val_sources,
            split_seeds["validation"],
            return_chunks=True,
            exclude_fingerprints=train_fingerprints,
            record_fingerprints=eval_fingerprints,
            dedupe_recorded=True,
            pack_documents_to_seqlen=_should_pack_document_chunks(val_sources),
        )
        eval_stages = [
            {
                "name": f"single_{seqlen // 1024}k_val",
                "seqlen": int(seqlen),
                "sample_budget": int(heldout_chunks),
                "token_budget": int(val_tokens),
                "samples": eval_samples,
                "effective_tokens": _samples_token_count(eval_samples),
                "train_token_fraction": 1.0,
                "excluded_train_fingerprints": len(train_fingerprints),
                "eval_fingerprints": len(eval_fingerprints),
                "heldout_from_train": True,
            }
        ]
        return trainloader, eval_stages

    testenc = _collect_hf_token_chunks(
        tokenizer_local,
        seqlen,
        val_sources,
        split_seeds["eval"],
        return_chunks=False,
        exclude_fingerprints=train_fingerprints,
    )

    return trainloader, testenc


def get_training_curriculum_dataset(
    seed,
    model_path,
    dataset_name=DISCOVER_DATASET_NAME,
    train_tokens=DISCOVER_REFERENCE_TOKENS,
    load_train=True,
    validation_chunks=DISCOVER_VALIDATION_CHUNKS,
    eval_chunks=DISCOVER_EVAL_CHUNKS,
    chat_wrap_pile=DISCOVER_CHAT_WRAP_PILE_DEFAULT,
    chat_user_prompt=DISCOVER_CHAT_WRAP_PILE_PROMPT,
    stream_train=True,
    materialized_train_chunks_per_stage=None,
    data_seed=DISCOVER_DATA_SEED,
    eval_seed=DISCOVER_EVAL_SEED,
):
    """Build train metadata/chunks and held-out eval chunks for the current length schedule.

    ``seed`` is retained for call-site compatibility. Training data and held-out
    splits use the independently supplied ``data_seed`` and ``eval_seed``.
    """
    discover_sources = get_discover_sources(
        dataset_name,
        chat_wrap_pile=chat_wrap_pile,
        chat_user_prompt=chat_user_prompt,
    )

    model_type_local = detect_model_type(model_path)
    tokenizer_local = load_tokenizer(model_path, model_type_local)

    stages = []
    curriculum = build_length_curriculum(model_path, int(train_tokens))
    if load_train and stream_train:
        print(
            f"Preparing discover HF {dataset_name} {DISCOVER_TRAIN_SEQLEN // 1024}K streaming training schedule "
            f"({int(train_tokens):,} reference tokens for held-out sizing)."
        )
        for stage_index, stage in enumerate(curriculum):
            stage_sources = _source_targets_for_token_budget(
                discover_sources, int(stage["token_budget"])
            )
            stage_seed = discover_split_seeds(data_seed, eval_seed, stage_index)["train"]
            print(
                f"Stage {stage_index + 1}/{len(curriculum)}: {stage['name']} "
                f"seqlen={stage['seqlen']:,}, reference budget={stage['token_budget']:,} tokens"
            )
            stages.append(
                {
                    **stage,
                    "samples": [],
                    "streaming_train": True,
                    "sources": stage_sources,
                    "effective_tokens": int(stage["token_budget"]),
                    "collection_seed": stage_seed,
                }
            )
    elif load_train:
        requested_chunks = max(1, int(materialized_train_chunks_per_stage or 0))
        print(
            f"Preparing discover HF {dataset_name} {DISCOVER_TRAIN_SEQLEN // 1024}K materialized training chunks "
            f"({requested_chunks:,} chunks/stage, held-out fingerprints excluded before training)."
        )

    eval_stage_specs = _heldout_stage_specs_from_curriculum(curriculum, int(validation_chunks))
    eval_stages = []
    print(
        f"Loading discover HF {dataset_name} held-out validation schedule "
        f"({int(validation_chunks):,} chunks, "
        f"{sum(int(stage['token_budget']) for stage in eval_stage_specs):,} requested tokens, "
        "streamed independently from training)."
    )
    eval_fingerprints = set()
    for stage_index, stage in enumerate(eval_stage_specs):
        eval_sources = _source_targets_for_token_budget(
            discover_sources, int(stage["token_budget"])
        )
        validation_seed = discover_split_seeds(data_seed, eval_seed, stage_index)["validation"]
        fingerprints_before = set(eval_fingerprints)
        print(
            f"Eval stage {stage_index + 1}/{len(eval_stage_specs)}: {stage['name']} "
            f"seqlen={stage['seqlen']:,}, chunks={stage['sample_budget']:,}, "
            f"budget={stage['token_budget']:,} tokens"
        )
        samples = _collect_hf_token_chunks(
            tokenizer_local,
            int(stage["seqlen"]),
            eval_sources,
            validation_seed,
            return_chunks=True,
            record_fingerprints=eval_fingerprints,
            dedupe_recorded=True,
            pack_documents_to_seqlen=_should_pack_document_chunks(eval_sources),
        )
        stage_fingerprints = sorted(eval_fingerprints - fingerprints_before)
        eval_stages.append(
            {
                **stage,
                "name": f"{stage['name']}_val",
                "samples": samples,
                "effective_tokens": _samples_token_count(samples),
                "excluded_train_fingerprints": 0,
                "eval_fingerprints": len(eval_fingerprints),
                "fingerprints": stage_fingerprints,
                "heldout_from_train": False,
                "collection_seed": validation_seed,
            }
        )

    test_stages = []
    test_stage_specs = _heldout_stage_specs_from_curriculum(curriculum, int(eval_chunks))
    print(
        f"Loading discover HF {dataset_name} held-out test schedule "
        f"({int(eval_chunks):,} chunks, "
        f"{sum(int(stage['token_budget']) for stage in test_stage_specs):,} requested tokens, "
        "excluding validation fingerprints)."
    )
    test_fingerprints = set()
    for stage_index, stage in enumerate(test_stage_specs):
        test_sources = _source_targets_for_token_budget(
            discover_sources, int(stage["token_budget"])
        )
        test_seed = discover_split_seeds(data_seed, eval_seed, stage_index)["eval"]
        fingerprints_before = set(test_fingerprints)
        print(
            f"Test stage {stage_index + 1}/{len(test_stage_specs)}: {stage['name']} "
            f"seqlen={stage['seqlen']:,}, chunks={stage['sample_budget']:,}, "
            f"budget={stage['token_budget']:,} tokens"
        )
        samples = _collect_hf_token_chunks(
            tokenizer_local,
            int(stage["seqlen"]),
            test_sources,
            test_seed,
            return_chunks=True,
            exclude_fingerprints=eval_fingerprints,
            record_fingerprints=test_fingerprints,
            dedupe_recorded=True,
            pack_documents_to_seqlen=_should_pack_document_chunks(test_sources),
        )
        stage_fingerprints = sorted(test_fingerprints - fingerprints_before)
        test_stages.append(
            {
                **stage,
                "name": f"{stage['name']}_test",
                "samples": samples,
                "effective_tokens": _samples_token_count(samples),
                "excluded_eval_fingerprints": len(eval_fingerprints),
                "test_fingerprints": len(test_fingerprints),
                "fingerprints": stage_fingerprints,
                "heldout_from_train": False,
                "heldout_from_eval": True,
                "collection_seed": test_seed,
            }
        )

    heldout_fingerprints = sorted(eval_fingerprints | test_fingerprints)
    if load_train and not stream_train:
        requested_chunks = max(1, int(materialized_train_chunks_per_stage or 0))
        for stage_index, stage in enumerate(curriculum):
            materialized_tokens = requested_chunks * int(stage["seqlen"])
            stage_sources = _source_targets_for_token_budget(discover_sources, materialized_tokens)
            stage_seed = discover_split_seeds(data_seed, eval_seed, stage_index)["train"]
            train_fingerprints = set()
            print(
                f"Train stage {stage_index + 1}/{len(curriculum)}: {stage['name']} "
                f"seqlen={stage['seqlen']:,}, materialized_chunks={requested_chunks:,}, "
                f"budget={materialized_tokens:,} tokens"
            )
            samples = _collect_hf_token_chunks(
                tokenizer_local,
                int(stage["seqlen"]),
                stage_sources,
                stage_seed,
                return_chunks=True,
                exclude_fingerprints=set(heldout_fingerprints),
                record_fingerprints=train_fingerprints,
                dedupe_recorded=True,
                pack_documents_to_seqlen=_should_pack_document_chunks(stage_sources),
                shuffle_chunks=True,
            )
            stages.append(
                {
                    **stage,
                    "samples": samples,
                    "streaming_train": False,
                    "sources": stage_sources,
                    "effective_tokens": _samples_token_count(samples),
                    "collection_seed": stage_seed,
                    "materialized_train_chunks": len(samples),
                    "target_materialized_train_chunks": requested_chunks,
                    "train_fingerprints": len(train_fingerprints),
                    "fingerprints": sorted(train_fingerprints),
                }
            )

    for stage in stages:
        stage["exclude_fingerprints"] = heldout_fingerprints
        stage["excluded_heldout_fingerprints"] = len(heldout_fingerprints)

    return stages, eval_stages, test_stages


class TextDataset(Dataset):
    """Text dataset class for fixed or example-preserving token chunks."""

    def __init__(self, trainloader, pad_token_id: int = 0):
        self.input_ids = []
        self.labels = []
        self.attention_masks = []
        self.position_ids = []
        self.pad_token_id = int(pad_token_id)

        for sample in trainloader:
            if len(sample) == 4:
                inp, tar, attention_mask, position_ids = sample
            elif len(sample) == 3:
                inp, tar, attention_mask = sample
                position_ids = None
            else:
                inp, tar = sample
                attention_mask = torch.ones_like(inp)
                position_ids = None
            self.input_ids.append(inp.squeeze(0))
            self.labels.append(tar.squeeze(0))
            self.attention_masks.append(attention_mask.squeeze(0))
            self.position_ids.append(None if position_ids is None else position_ids.squeeze(0))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
            "pad_token_id": torch.tensor(self.pad_token_id, dtype=torch.long),
        }
        if self.position_ids[idx] is not None:
            item["position_ids"] = self.position_ids[idx]
        return item


def _item_flag_true(item: Dict[str, Any], key: str) -> bool:
    value = item.get(key, False)
    if torch.is_tensor(value):
        return bool(value.detach().cpu().item())
    return bool(value)


def collate_text_batch(batch):
    """Pad example-preserving chunks to the longest sequence in the batch."""
    pad_value = batch[0].get("pad_token_id", 0)
    pad_token_id = int(pad_value.item() if torch.is_tensor(pad_value) else pad_value)
    lengths = [int(item["input_ids"].numel()) for item in batch]
    max_len = max(lengths)
    batch_size = len(batch)
    has_position_ids = any("position_ids" in item for item in batch)
    all_same_len = all(length == max_len for length in lengths)
    all_attention_ones = all(_item_flag_true(item, "attention_mask_all_ones") for item in batch)

    if all_same_len:
        input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
        labels = torch.stack([item["labels"] for item in batch], dim=0)
        attention_mask = None
        if not all_attention_ones:
            masks = []
            for item in batch:
                mask = item.get("attention_mask")
                if mask is None:
                    mask = torch.ones_like(item["input_ids"], dtype=torch.long)
                masks.append(mask)
            attention_mask = torch.stack(masks, dim=0)
        collated = {
            "input_ids": input_ids,
            "labels": labels,
        }
        if all_attention_ones:
            collated["attention_mask_all_ones"] = torch.tensor(True)
        else:
            collated["attention_mask"] = attention_mask
        if has_position_ids:
            positions = []
            for item, length in zip(batch, lengths):
                if "position_ids" in item:
                    positions.append(item["position_ids"])
                else:
                    positions.append(torch.arange(length, dtype=torch.long))
            collated["position_ids"] = torch.stack(positions, dim=0)
        return collated

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    position_ids = (
        torch.zeros((batch_size, max_len), dtype=torch.long) if has_position_ids else None
    )

    for row, item in enumerate(batch):
        length = lengths[row]
        input_ids[row, :length] = item["input_ids"]
        labels[row, :length] = item["labels"]
        mask = item.get("attention_mask")
        if mask is None:
            attention_mask[row, :length] = 1
        else:
            attention_mask[row, :length] = mask
        if has_position_ids and "position_ids" in item:
            position_ids[row, :length] = item["position_ids"]
        elif has_position_ids:
            position_ids[row, :length] = torch.arange(length, dtype=torch.long)

    collated = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if has_position_ids:
        collated["position_ids"] = position_ids
    return collated


@torch.no_grad()
def _extract_model_logits(outputs):
    """Extract logits from different model output formats."""
    if torch.is_tensor(outputs):
        return outputs

    if isinstance(outputs, dict):
        if "logits" in outputs:
            return _extract_model_logits(outputs["logits"])
        raise TypeError(
            "Unsupported model output type for logits extraction: dict without 'logits' key."
        )

    if hasattr(outputs, "logits"):
        return _extract_model_logits(getattr(outputs, "logits"))

    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 0:
            raise ValueError("Model output tuple/list is empty; no logits found.")

        candidate = outputs[0]
        if not isinstance(candidate, (tuple, list, dict)) and torch.is_tensor(candidate):
            return candidate

        for item in outputs:
            if torch.is_tensor(item):
                return item
            try:
                extracted = _extract_model_logits(item)
            except TypeError:
                continue
            except ValueError:
                continue
            if torch.is_tensor(extracted):
                return extracted

        raise TypeError("Unable to find tensor logits in model output tuple/list.")

    raise TypeError(f"Unsupported model output type for logits extraction: {type(outputs)}")


@torch.no_grad()
def calculate_perplexity(model, testenc, delay=0, distributed=False, rank=0, world_size=1):
    """Calculate perplexity for model evaluation"""
    if not distributed or rank == 0:
        print("Evaluating ...")
    model_for_attrs = getattr(model, "module", model)
    model_seqlen = getattr(model, "seqlen", getattr(model_for_attrs, "seqlen", None))
    if model_seqlen is None or model_seqlen <= 0:
        raise ValueError("Model must define a positive seqlen for perplexity evaluation.")
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device
    # Handle both dictionary and object-based tokenizer returns
    if isinstance(testenc, dict):
        testenc = testenc["input_ids"]
    else:
        testenc = testenc.input_ids
    nsamples = testenc.numel() // model_seqlen
    # Parallel eval needs identical work on each rank so all-gathered PPL
    # reduction stays aligned across distributed processes.
    sample_indices = range(nsamples)

    use_tqdm = (not distributed) or rank == 0
    iterator = tqdm(sample_indices, desc="Calculating perplexity") if use_tqdm else sample_indices

    total_nll = torch.zeros((), device=model_device)
    total_tokens = torch.zeros((), device=model_device)

    for i in iterator:
        input_ids = testenc[:, (i * model_seqlen) : ((i + 1) * model_seqlen)].to(model_device)
        outputs = model(input_ids)
        logits = _extract_model_logits(outputs)

        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        total_nll += loss.float() * model_seqlen
        total_tokens += model_seqlen

        if use_tqdm:
            current_ppl = torch.exp(total_nll / total_tokens)
            iterator.set_postfix({"Current perplexity": f"{current_ppl.item():.4f}"})

        if delay > 0:
            time.sleep(delay)

    if distributed and world_size > 1:
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(total_nll, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_tokens, op=torch.distributed.ReduceOp.SUM)

    ppl = torch.exp(total_nll / total_tokens)
    return ppl.item()


def validate_model_setup(model, tokenizer, model_type):
    """Validate that the model and tokenizer are properly loaded and configured"""
    print("\n=== Model Setup Validation ===")
    print(f"Model type: {model_type}")
    print(f"Model class: {type(model).__name__}")
    print(f"Tokenizer class: {type(tokenizer).__name__}")

    # Test tokenizer
    try:
        test_text = "Hello, how are you?"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"Tokenizer test: ✓ (input_ids shape: {tokens['input_ids'].shape})")

        # Get the device of the model
        model_device = next(model.parameters()).device
        print(f"Model device: {model_device}")

        # Convert tokenizer output to dict if it's not already (handles TokenizerOutput objects)
        if not isinstance(tokens, dict):
            tokens = dict(tokens)

        # Move tokens to the same device as the model
        tokens = {k: v.to(model_device) for k, v in tokens.items()}

        # Test model forward pass
        with torch.no_grad():
            outputs = model(**tokens)
            logits = _extract_model_logits(outputs)
            print(f"Model forward pass: ✓ (logits shape: {logits.shape})")

    except Exception as e:
        print(f"Validation error: {e}")
        return False

    print("=== Validation Complete ===\n")
    return True


def find_key_cache_stats_path(
    root_dir: str,
    model_name: str,
    bits: Optional[int],
    env_var: str = "KEY_CACHE_STATS_PATH",
) -> Optional[str]:
    """Find key cache stats file under logs/ by model name and bit-width."""
    env_path = os.environ.get(env_var)
    if env_path:
        if os.path.exists(env_path):
            return env_path
        print(f"Warning: {env_var} set but file not found: {env_path}")

    logs_dir = os.path.join(root_dir, "logs")
    pattern = os.path.join(logs_dir, "*", "quant_configs", "key_cache_stats.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    token_bits = f"{bits}bit" if bits is not None else None

    def matches(path: str) -> bool:
        run_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if "one_group" not in run_dir:
            return False
        if model_name and model_name not in run_dir:
            return False
        if token_bits and token_bits not in run_dir:
            return False
        return True

    filtered = [path for path in candidates if matches(path)]
    if not filtered:
        return None
    return max(filtered, key=os.path.getmtime)


def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
