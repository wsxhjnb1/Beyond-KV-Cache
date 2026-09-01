import json
import os
from typing import Callable, Dict, List, Optional

import torch

from beyond.quantization.table_precision import (
    FP16_STRICT_MIN_GAP,
    TABLE_COMPUTE_DTYPE_NAME,
    TABLE_PRECISION_ABI,
    TABLE_STORAGE_DTYPE_NAME,
    cast_table_for_storage,
    validate_fp16_storage_tables,
)

EXPERIMENT_CONTROL_SPECS = {
    "beyond_nll": {
        "training_objective": "causal_lm_nll",
        "quantizer_parameterization": "nonuniform_points_thresholds",
        "available": True,
        "compressed_kv_cache": True,
    },
    "uniform_affine_nll": {
        "training_objective": "causal_lm_nll",
        "quantizer_parameterization": "uniform_affine_endpoints",
        "available": True,
        "compressed_kv_cache": True,
    },
    "nonuniform_reconstruction": {
        "training_objective": "kv_reconstruction_mse",
        "quantizer_parameterization": "nonuniform_points_thresholds",
        "available": True,
        "compressed_kv_cache": True,
    },
    "fisher_reconstruction": {
        "training_objective": "fisher_weighted_kv_reconstruction",
        "quantizer_parameterization": "nonuniform_points_thresholds",
        "available": False,
        "compressed_kv_cache": True,
        "unavailable_reason": (
            "Fisher-weighted reconstruction is not implemented: the current training "
            "path does not produce a validated per-activation Fisher target."
        ),
    },
    "uncompressed_affine_nll": {
        "training_objective": "causal_lm_nll",
        "quantizer_parameterization": "full_precision_group_affine",
        "available": False,
        "compressed_kv_cache": False,
        "unavailable_reason": (
            "The full-precision affine adaptation control is not implemented until its "
            "export and evaluation path can guarantee uncompressed K/V cache storage."
        ),
    },
    "uncompressed_matched_nll": {
        "training_objective": "causal_lm_nll",
        "quantizer_parameterization": "full_precision_chebyshev_residual",
        "available": True,
        "compressed_kv_cache": False,
        "hard_forward_qdq": False,
        "parameter_budget": "matched_to_nonuniform_levels_plus_thresholds",
    },
    "uncompressed_bucket_residual_nll": {
        "training_objective": "causal_lm_nll",
        "quantizer_parameterization": "full_precision_bucket_residual",
        "available": True,
        "compressed_kv_cache": False,
        "hard_forward_qdq": False,
        "hard_bucket_assignment": True,
        "parameter_budget": "matched_to_nonuniform_levels_plus_thresholds",
    },
}
EXPERIMENT_CONTROL_CHOICES = tuple(EXPERIMENT_CONTROL_SPECS)


def resolve_experiment_control(name, *, require_available=True):
    key = str(name or "").strip().lower()
    if key not in EXPERIMENT_CONTROL_SPECS:
        raise ValueError(
            f"Unknown experiment control {name!r}; expected one of "
            f"{', '.join(EXPERIMENT_CONTROL_CHOICES)}"
        )
    spec = dict(EXPERIMENT_CONTROL_SPECS[key])
    spec["name"] = key
    if require_available and not spec.get("available", False):
        raise ValueError(
            f"Experiment control {key!r} is fail-closed: "
            f"{spec.get('unavailable_reason', 'implementation unavailable')}"
        )
    return spec


@torch.no_grad()
def project_quant_points(
    q_points,
    min_val=None,
    max_val=None,
    min_gap=FP16_STRICT_MIN_GAP,
):
    """Project codepoints so they stay ordered after deployment FP16 rounding."""
    if q_points is None:
        return
    n = int(q_points.shape[-1]) if q_points.dim() > 0 else int(q_points.numel())
    if n <= 1:
        return
    sorted_q, _ = torch.sort(q_points, dim=-1)
    if min_val is not None:
        sorted_q = sorted_q.clamp(min=float(min_val))
    if max_val is not None:
        sorted_q = sorted_q.clamp(max=float(max_val))
    idx_shape = [1] * sorted_q.dim()
    idx_shape[-1] = n
    idx = torch.arange(n, device=q_points.device, dtype=sorted_q.dtype).view(idx_shape)
    v = sorted_q - idx * min_gap
    v, _ = torch.cummax(v, dim=-1)
    sorted_q = v + idx * min_gap
    if max_val is not None:
        sorted_q = sorted_q.clamp(max=float(max_val))
        rev_idx = (n - 1) - idx
        v2 = sorted_q + rev_idx * min_gap
        v2_flipped = torch.flip(v2, dims=[-1])
        v2_cummin, _ = torch.cummin(v2_flipped, dim=-1)
        v2 = torch.flip(v2_cummin, dims=[-1])
        sorted_q = v2 - rev_idx * min_gap
        if min_val is not None:
            sorted_q = sorted_q.clamp(min=float(min_val))
    q_points.copy_(sorted_q)


@torch.no_grad()
def project_thresholds(
    thresholds,
    min_val=0.0,
    max_val=1.0,
    min_gap=FP16_STRICT_MIN_GAP,
):
    """Project thresholds for order and distinct deployment FP16 values."""
    thresholds.clamp_(min_val, max_val)
    n = int(thresholds.shape[-1]) if thresholds.dim() > 0 else int(thresholds.numel())
    if n <= 1:
        return
    sorted_t, _ = torch.sort(thresholds, dim=-1)
    idx_shape = [1] * sorted_t.dim()
    idx_shape[-1] = n
    idx = torch.arange(n, device=thresholds.device, dtype=sorted_t.dtype).view(idx_shape)
    values = sorted_t - idx * min_gap
    values, _ = torch.cummax(values, dim=-1)
    sorted_t = (values + idx * min_gap).clamp_(max=max_val)
    reverse_idx = (n - 1) - idx
    values = sorted_t + reverse_idx * min_gap
    values = torch.flip(values, dims=[-1])
    values, _ = torch.cummin(values, dim=-1)
    sorted_t = torch.flip(values, dims=[-1]) - reverse_idx * min_gap
    thresholds.copy_(sorted_t.clamp_(min_val, max_val))


def _default_projection_resolver(parent_module, proj_type, spec):
    if parent_module is None or not isinstance(proj_type, str):
        return None

    candidates = [
        proj_type,
        f"{proj_type}_proj",
        f"{proj_type}_linear",
        f"{proj_type}_fc",
        f"{proj_type}weight",
        f"{proj_type}dense",
        "dense",
        "linear",
    ]
    for candidate in candidates:
        if candidate and hasattr(parent_module, candidate):
            return getattr(parent_module, candidate)

    normalized = proj_type.replace("_", "").lower()
    for child_name, child_module in parent_module.named_children():
        child_key = child_name.replace("_", "").lower()
        if child_key == normalized:
            return child_module
        if child_key in (f"{normalized}proj", f"{normalized}linear"):
            return child_module
        if normalized in child_key and child_key.startswith(normalized):
            return child_module
    return None


def _is_vision_module_name(name: str) -> bool:
    parts = {part.lower() for part in str(name).split(".")}
    return bool(parts.intersection({"vision", "visual", "vision_tower", "vision_model"}))


_FUSED_QKV_PROJ_TYPES = frozenset({"query_key_value", "qkv_proj"})


def find_kv_proj_layers(model, patterns=None, include_vision=False):
    """Return KV projection layer specs with grouping metadata.

    - For separate projections: returns entries for `k_proj` and `v_proj`.
      Both are always token grouped.
    - For fused projections (e.g., `query_key_value` or `qkv_proj`): returns
      two entries targeting the same module with `kv_slice` set to `k` and `v`.
    """
    if patterns is None:
        patterns = ("k_proj", "v_proj", "query_key_value", "qkv_proj")

    pattern_set = set(patterns)
    layers = []
    seen = set()
    for name, _module in model.named_modules():
        if not name:
            continue
        if not include_vision and _is_vision_module_name(name):
            continue

        leaf = name.rsplit(".", 1)[-1]
        if leaf not in pattern_set:
            continue
        if name in seen:
            continue
        seen.add(name)

        if leaf == "k_proj":
            layers.append(
                {
                    "name": name,
                    "proj_type": "k_proj",
                    "grouping_dim": "token",
                }
            )
        elif leaf == "v_proj":
            layers.append(
                {
                    "name": name,
                    "proj_type": "v_proj",
                    "grouping_dim": "token",
                }
            )
        elif leaf in _FUSED_QKV_PROJ_TYPES:
            # Fused QKV: create separate targets for K and V slices
            layers.append(
                {
                    "name": name,
                    "proj_type": leaf,
                    "grouping_dim": "token",
                    "kv_slice": "k",
                }
            )
            layers.append(
                {
                    "name": name,
                    "proj_type": leaf,
                    "grouping_dim": "token",
                    "kv_slice": "v",
                }
            )
    return layers


def _is_meta_device(device):
    return device is not None and getattr(device, "type", None) == "meta"


def _get_module_device(module):
    if hasattr(module, "weight") and hasattr(module.weight, "device"):
        device = module.weight.device
        return None if _is_meta_device(device) else device
    for param in module.parameters():
        device = param.device
        return None if _is_meta_device(device) else device
    return None


def _unwrap_projection_module(module):
    current = module
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        inner = getattr(current, "linear", None)
        if inner is None:
            return current
        current = inner
    return module


def _get_projection_out_features(module):
    for candidate in (module, _unwrap_projection_module(module)):
        if candidate is None:
            continue
        for attr_name in (
            "output_size_per_partition",
            "out_features_per_partition",
            "local_out_features",
            "out_features",
            "output_size",
        ):
            value = getattr(candidate, attr_name, None)
            if isinstance(value, int) and value > 0:
                return value
        weight = getattr(candidate, "weight", None)
        if torch.is_tensor(weight) and weight.dim() >= 2:
            return int(weight.shape[0])
    return None


def _first_positive_int(module, attr_names):
    if module is None:
        return None
    sources = (module, getattr(module, "config", None))
    for source in sources:
        if source is None:
            continue
        for attr_name in attr_names:
            value = getattr(source, attr_name, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def _infer_qkv_splits(parent_module, proj_module):
    """Infer fused QKV split sizes for MHA/GQA projections.

    Megatron TP patching updates attention head counts to their local values, so
    the same metadata path works for both full and tensor-parallel fused outputs.
    """

    out_features = _get_projection_out_features(proj_module)
    head_dim = _first_positive_int(parent_module, ("head_dim", "hidden_size_per_attention_head"))
    q_heads = _first_positive_int(
        parent_module,
        ("num_heads", "num_attention_heads", "n_heads", "n_head"),
    )
    kv_heads = _first_positive_int(
        parent_module,
        ("num_key_value_heads", "num_kv_heads", "n_kv_heads"),
    )

    if head_dim is not None and q_heads is not None:
        if kv_heads is None:
            kv_heads = q_heads
        q_dim = int(q_heads) * int(head_dim)
        k_dim = int(kv_heads) * int(head_dim)
        v_dim = k_dim
        if out_features is None or q_dim + k_dim + v_dim == int(out_features):
            return (q_dim, k_dim, v_dim)

    if out_features is not None and int(out_features) % 3 == 0:
        part = int(out_features) // 3
        return (part, part, part)
    return None


def build_kv_quant_targets(
    model,
    quant_layer_factory: Callable[[Dict, torch.nn.Module], torch.nn.Module],
    layer_specs: Optional[List[Dict]] = None,
    patterns=None,
    log_fn=print,
    projection_resolver=None,
):
    """Create quantizer targets without applying them to the model."""
    if layer_specs is None:
        layer_specs = find_kv_proj_layers(model, patterns=patterns)
    if projection_resolver is None:
        projection_resolver = _default_projection_resolver

    targets = []
    for spec in layer_specs:
        spec = dict(spec)
        name = spec.get("name")
        proj_type = spec.get("proj_type")
        grouping_dim = spec.get("grouping_dim")
        kv_slice = spec.get("kv_slice")
        parent_name = name.rsplit(".", 1)[0] if name else None
        target_attr = name.rsplit(".", 1)[1] if name and "." in name else proj_type
        try:
            if not name or not parent_name or not proj_type:
                raise ValueError("Missing quantization target metadata")
            parent_module = model.get_submodule(parent_name)
            proj_module = projection_resolver(parent_module, proj_type, spec)
            if proj_module is None:
                raise AttributeError(f"Failed to resolve projection module {proj_type}")
            head_dim = _first_positive_int(
                parent_module, ("head_dim", "hidden_size_per_attention_head")
            )
            if head_dim is not None:
                spec["head_dim"] = int(head_dim)
            if "quant_width" not in spec:
                quant_width = None
                if proj_type in _FUSED_QKV_PROJ_TYPES:
                    qkv_splits = _infer_qkv_splits(parent_module, proj_module)
                    if qkv_splits is not None:
                        _, k_dim, v_dim = qkv_splits
                        if kv_slice == "k":
                            quant_width = k_dim
                        elif kv_slice == "v":
                            quant_width = v_dim
                else:
                    quant_width = _get_projection_out_features(proj_module)
                if quant_width is not None:
                    spec["quant_width"] = int(quant_width)
            quant_module = quant_layer_factory(spec, proj_module)
            target_device = _get_module_device(proj_module)
            if target_device is not None:
                quant_module.to(target_device)
            targets.append(
                {
                    "name": name,
                    "parent_name": parent_name,
                    "proj_type": proj_type,
                    "grouping_dim": grouping_dim,
                    "kv_slice": kv_slice,
                    "quant_width": spec.get("quant_width"),
                    "head_dim": spec.get("head_dim"),
                    "target_attr": target_attr,
                    "proj_module": proj_module,
                    "quant_module": quant_module,
                }
            )
        except (AttributeError, ValueError) as exc:
            if log_fn:
                log_fn(f"Warning: Failed to prepare quantizer for {name}: {exc}")

    return targets


def apply_kv_quantization_targets(
    model,
    quant_targets,
    quantized_linear_cls,
    type_fn: Optional[Callable[[Dict], str]] = None,
    log_fn=print,
    qkv_quantized_linear_cls=None,
    projection_resolver=None,
):
    """Wrap projection layers with quantizers using prepared targets.

    If multiple targets exist for the same fused QKV module (with `kv_slice`
    values), they are combined into a single `QuantizedQKVLinear` wrapper that
    applies K/V quantizers separately.
    """
    if projection_resolver is None:
        projection_resolver = _default_projection_resolver

    # Group targets by (parent_name, proj_type)
    grouped = {}
    for t in quant_targets:
        key = (t.get("parent_name"), t.get("proj_type"))
        grouped.setdefault(key, []).append(t)

    quantized_layer_names = []
    for (parent_name, proj_type), targets in grouped.items():
        try:
            if not parent_name or not proj_type:
                raise ValueError("Missing quantization target metadata")
            parent_module = model.get_submodule(parent_name)
            target_entry = targets[0] if targets else {}
            original_proj = target_entry.get("proj_module")
            if original_proj is None:
                original_proj = projection_resolver(parent_module, proj_type, target_entry)
            if original_proj is None:
                raise AttributeError(f"Failed to resolve projection module {proj_type}")

            # Fused QKV path: expect two targets with kv_slice 'k' and 'v'
            if proj_type in _FUSED_QKV_PROJ_TYPES:
                if qkv_quantized_linear_cls is None:
                    raise ValueError("qkv_quantized_linear_cls is required for fused QKV modules")

                k_q = None
                v_q = None
                for t in targets:
                    kv_slice = t.get("kv_slice")
                    qm = t.get("quant_module")
                    if qm is None:
                        continue
                    target_device = _get_module_device(original_proj)
                    if target_device is not None:
                        qm.to(target_device)
                    if kv_slice == "k":
                        k_q = qm
                    elif kv_slice == "v":
                        v_q = qm

                qkv_splits = _infer_qkv_splits(parent_module, original_proj)
                try:
                    wrapped = qkv_quantized_linear_cls(
                        original_proj,
                        k_quantizer=k_q,
                        v_quantizer=v_q,
                        qkv_splits=qkv_splits,
                    )
                except TypeError:
                    wrapped = qkv_quantized_linear_cls(
                        original_proj,
                        k_quantizer=k_q,
                        v_quantizer=v_q,
                    )
                target_attr = target_entry.get("target_attr") or proj_type
                setattr(parent_module, target_attr, wrapped)

                # Type annotation string
                if type_fn is None:
                    quantized_layer_names.append(f"{parent_name}.{proj_type}")
                else:
                    # Compose combined type string for fused layer
                    type_str = "QuantizedQKVLinear"
                    quantized_layer_names.append((f"{parent_name}.{proj_type}", type_str))
            else:
                # Separate K/V path: one target per module
                for t in targets:
                    layer_name = t.get("name")
                    quant_module = t.get("quant_module")
                    if not layer_name or quant_module is None:
                        continue
                    target_device = _get_module_device(original_proj)
                    if target_device is not None:
                        quant_module.to(target_device)
                    wrapped_module = quantized_linear_cls(original_proj, quant_module)
                    target_attr = t.get("target_attr") or proj_type
                    setattr(parent_module, target_attr, wrapped_module)
                    if type_fn is None:
                        quantized_layer_names.append(layer_name)
                    else:
                        quantized_layer_names.append((layer_name, type_fn(t)))
        except (AttributeError, ValueError) as exc:
            if log_fn:
                # Derive a display name
                if targets and targets[0].get("name"):
                    disp = targets[0].get("name")
                else:
                    disp = f"{parent_name}.{proj_type}"
                log_fn(f"Warning: Failed to add quantization for {disp}: {exc}")
            continue

    return quantized_layer_names


def apply_kv_quantization(
    model,
    quant_layer_factory: Callable[[Dict, torch.nn.Module], torch.nn.Module],
    quantized_linear_cls,
    layer_specs: Optional[List[Dict]] = None,
    patterns=None,
    type_fn: Optional[Callable[[Dict], str]] = None,
    log_fn=print,
):
    """Build and apply KV quantizers in one step."""
    quant_targets = build_kv_quant_targets(
        model,
        quant_layer_factory,
        layer_specs=layer_specs,
        patterns=patterns,
        log_fn=log_fn,
    )
    return apply_kv_quantization_targets(
        model,
        quant_targets,
        quantized_linear_cls,
        type_fn=type_fn,
        log_fn=log_fn,
    )


def generate_quantization_points(num_bits):
    """Generate evenly spaced quantization points and thresholds."""
    n_levels = 2**num_bits
    q_points = torch.linspace(0.0, 1.0, n_levels).tolist()
    thresholds = []
    for i in range(n_levels - 1):
        thresholds.append((q_points[i] + q_points[i + 1]) / 2.0)
    return q_points, thresholds


def _compute_thresholds_from_points(quant_points):
    if torch.is_tensor(quant_points):
        return (quant_points[..., :-1] + quant_points[..., 1:]) / 2.0
    q = torch.as_tensor(quant_points, dtype=torch.float32)
    thresholds = (q[..., :-1] + q[..., 1:]) / 2.0
    return thresholds.tolist()


@torch.no_grad()
def set_quant_params(quant_module, quant_points, thresholds=None):
    materialize = getattr(quant_module, "materialize_quant_tables", None)
    if callable(materialize):
        target_q, target_t = materialize()
    elif hasattr(quant_module, "q_points"):
        target_q = quant_module.q_points
        target_t = getattr(quant_module, "thresholds", None)
    else:
        return
    q_points_tensor = torch.as_tensor(
        quant_points,
        device=target_q.device,
        dtype=target_q.dtype,
    )
    if q_points_tensor.dim() == 1 and target_q.dim() == 2:
        q_points_tensor = q_points_tensor.unsqueeze(0).expand(target_q.shape[0], -1).contiguous()
    if q_points_tensor.dim() not in (1, 2):
        raise ValueError(
            "Each K/V quant config must contain a 1-D shared quant_points table "
            "or a 2-D per-group table; "
            f"got shape {tuple(q_points_tensor.shape)}"
        )
    if q_points_tensor.shape != target_q.shape:
        raise ValueError(
            f"quant_points shape {tuple(q_points_tensor.shape)} does not match "
            f"quantizer shape {tuple(target_q.shape)}"
        )
    thresholds_tensor = None
    if target_t is not None:
        if thresholds is None or len(thresholds) == 0:
            thresholds = _compute_thresholds_from_points(q_points_tensor)
        thresholds_tensor = torch.as_tensor(
            thresholds,
            device=target_t.device,
            dtype=target_t.dtype,
        )
        if thresholds_tensor.dim() == 1 and target_t.dim() == 2:
            thresholds_tensor = (
                thresholds_tensor.unsqueeze(0).expand(target_t.shape[0], -1).contiguous()
            )
        if thresholds_tensor.dim() not in (1, 2):
            raise ValueError(
                "Each K/V quant config must contain a 1-D shared thresholds table "
                "or a 2-D per-group table; "
                f"got shape {tuple(thresholds_tensor.shape)}"
            )
        if thresholds_tensor.shape != target_t.shape:
            raise ValueError(
                f"thresholds shape {tuple(thresholds_tensor.shape)} does not match "
                f"quantizer shape {tuple(target_t.shape)}"
            )
    setter = getattr(quant_module, "set_quant_tables", None)
    if callable(setter):
        setter(q_points_tensor, thresholds_tensor)
    else:
        quant_module.q_points.data.copy_(q_points_tensor)
        if target_t is not None:
            quant_module.thresholds.data.copy_(thresholds_tensor)


def build_unified_quant_layer(
    spec: Dict,
    quant_layer_cls,
    num_bits: int,
    group_size: int,
    one_group: bool,
    quant_config: Optional[Dict] = None,
    quant_only: bool = False,
    force_num_bits: bool = False,
    quant_table_axis: str = "group",
):
    """Create a UnifiedQuantLayer and optionally initialize from config."""
    layer_name = spec.get("name")
    grouping_dim = "token"
    quant_width = spec.get("quant_width")

    num_bits_for_init = num_bits
    grouping_dim_for_init = grouping_dim
    use_config = False
    config_layer = None
    table_axis_for_init = str(quant_table_axis or "group").lower()
    if table_axis_for_init in {"per_group", "group"}:
        table_axis_for_init = "group"
    elif table_axis_for_init != "shared":
        raise ValueError(f"Unsupported quant_table_axis: {quant_table_axis}")
    num_tables_for_init = 1
    if quant_config and layer_name in quant_config and not quant_only:
        config_layer = quant_config[layer_name]
        config_num_bits = config_layer.get("num_bits", num_bits)
        if not force_num_bits or config_num_bits == num_bits:
            num_bits_for_init = config_num_bits
            grouping_dim_for_init = "token"
            use_config = True
            config_q = torch.as_tensor(config_layer.get("quant_points", []))
            if config_q.dim() == 2:
                table_axis_for_init = "group"
                num_tables_for_init = int(config_q.shape[0])

    if table_axis_for_init == "group" and num_tables_for_init <= 1:
        if one_group:
            raise ValueError("Grouped quant tables are not supported with one_group.")
        if quant_width is None:
            raise ValueError(f"quant_width is required for grouped quant tables on {layer_name}")
        quant_width = int(quant_width)
        if group_size <= 0 or quant_width % int(group_size) != 0:
            raise ValueError(
                f"quant_width {quant_width} must be divisible by group_size {group_size}"
            )
        num_tables_for_init = quant_width // int(group_size)

    quant_module = quant_layer_cls(
        num_bits=num_bits_for_init,
        group_size=group_size,
        grouping_dim=grouping_dim_for_init,
        one_group=one_group,
        table_axis=table_axis_for_init,
        num_tables=num_tables_for_init,
        quant_width=quant_width,
    )

    if quant_only or not use_config:
        q_points_list, thresholds_list = generate_quantization_points(num_bits_for_init)
        set_quant_params(quant_module, q_points_list, thresholds_list)
    else:
        quant_points = config_layer.get("quant_points", [])
        thresholds = config_layer.get("thresholds")
        if thresholds is None:
            thresholds = _compute_thresholds_from_points(quant_points)
        set_quant_params(quant_module, quant_points, thresholds)

    return quant_module


def load_quant_config(path: str, allow_missing: bool = False):
    if not path:
        if allow_missing:
            return {}
        raise ValueError("quant_config path is required")
    try:
        if str(path).endswith(".pt"):
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(path, map_location="cpu")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if allow_missing:
            return {}
        raise


def extract_quant_config(
    model,
    quantized_layer_names,
    num_bits,
    quantized_linear_cls,
    log_fn=print,
    metadata=None,
):
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("quant config metadata must be a dictionary")
    quant_points_config = {"__metadata__": dict(metadata)} if metadata is not None else {}

    def _dump_quant_module(quant_module, *, quant_type_str="", proj_type="", default_grouping=None):
        cfg = {
            "type": quant_type_str,
            "proj_type": proj_type,
            "num_bits": num_bits,
            "grouping_dim": getattr(quant_module, "grouping_dim", default_grouping),
            "table_axis": getattr(quant_module, "table_axis", "group"),
            "num_tables": int(getattr(quant_module, "num_tables", 1)),
            "group_size": int(getattr(quant_module, "group_size", 0) or 0),
        }
        parameterization_fn = getattr(quant_module, "quantizer_parameterization", None)
        cfg["quantizer_parameterization"] = (
            parameterization_fn()
            if callable(parameterization_fn)
            else "nonuniform_points_thresholds"
        )
        is_uncompressed_adapter = cfg["quantizer_parameterization"] in {
            "full_precision_chebyshev_residual",
            "full_precision_bucket_residual",
        }
        cfg["hard_forward"] = not is_uncompressed_adapter
        cfg["compressed_kv_cache"] = not is_uncompressed_adapter
        if not is_uncompressed_adapter:
            cfg.update(
                {
                    "table_precision_abi": TABLE_PRECISION_ABI,
                    "table_storage_dtype": TABLE_STORAGE_DTYPE_NAME,
                    "table_compute_dtype": TABLE_COMPUTE_DTYPE_NAME,
                }
            )
        export_adapter_state = getattr(quant_module, "export_adapter_state", None)
        materialize = getattr(quant_module, "materialize_quant_tables", None)
        if callable(export_adapter_state):
            with torch.no_grad():
                adapter_state = export_adapter_state()
            if not isinstance(adapter_state, dict):
                raise TypeError("export_adapter_state() must return a dictionary")
            cfg.update(adapter_state)
        elif callable(materialize):
            with torch.no_grad():
                q_values, threshold_values = materialize()
                quant_points, _ = torch.sort(q_values.detach().cpu(), dim=-1)
                thresholds_list = threshold_values.detach().cpu().contiguous()
                quant_points = cast_table_for_storage(quant_points)
                thresholds_list = cast_table_for_storage(thresholds_list)
                validate_fp16_storage_tables(
                    quant_points,
                    thresholds_list,
                    label=f"{proj_type or quant_type_str or 'quantizer'} export",
                )
            cfg.update(
                {
                    "quant_points": quant_points.contiguous(),
                    "thresholds": thresholds_list,
                }
            )
        elif hasattr(quant_module, "q_points"):
            with torch.no_grad():
                q_values = quant_module.q_points.detach().cpu()
                quant_points, _ = torch.sort(q_values, dim=-1)
                if hasattr(quant_module, "thresholds"):
                    thresholds_list = quant_module.thresholds.detach().cpu().contiguous()
                else:
                    thresholds_list = _compute_thresholds_from_points(quant_points)
                quant_points = cast_table_for_storage(quant_points)
                thresholds_list = cast_table_for_storage(thresholds_list)
                validate_fp16_storage_tables(
                    quant_points,
                    thresholds_list,
                    label=f"{proj_type or quant_type_str or 'quantizer'} export",
                )
            cfg.update(
                {
                    "quant_points": quant_points.contiguous(),
                    "thresholds": thresholds_list,
                }
            )
        return cfg

    for entry in quantized_layer_names:
        # entry may be either a string (layer_name) or (layer_name, type_str)
        if isinstance(entry, tuple):
            layer_name, quant_type_str = entry
        else:
            layer_name = entry
            quant_type_str = ""

        parent_name = layer_name.rsplit(".", 1)[0]
        target_proj_name = layer_name.split(".")[-1]
        try:
            parent_module = model.get_submodule(parent_name)
            wrapped_module = getattr(parent_module, target_proj_name)

            # Fused QKV wrapper support (duck-typing)
            if hasattr(wrapped_module, "k_quantizer") and hasattr(wrapped_module, "v_quantizer"):
                layer_config = {
                    "type": quant_type_str or "QuantizedQKVLinear",
                    "proj_type": target_proj_name,
                }

                def _dump_quant(qm, default_grouping=None):
                    if qm is None:
                        return None
                    return _dump_quant_module(
                        qm,
                        quant_type_str=quant_type_str or "QuantizedQKVLinear",
                        proj_type=target_proj_name,
                        default_grouping=default_grouping,
                    )

                layer_config["k"] = _dump_quant(
                    wrapped_module.k_quantizer, default_grouping="token"
                )
                layer_config["v"] = _dump_quant(
                    wrapped_module.v_quantizer, default_grouping="token"
                )
                quant_points_config[layer_name] = layer_config
                continue

            # Non-fused path (QuantizedLinear)
            if not isinstance(wrapped_module, quantized_linear_cls):
                continue
            quant_module = wrapped_module.quantizer

            layer_config = _dump_quant_module(
                quant_module,
                quant_type_str=quant_type_str,
                proj_type=target_proj_name,
                default_grouping="token",
            )

            quant_points_config[layer_name] = layer_config
        except Exception as exc:
            if log_fn:
                log_fn(f"Error extracting quantization config for {layer_name}: {exc}")
            continue
    return quant_points_config


def save_quant_config(
    quant_config_dir,
    model,
    quantized_layer_names,
    num_bits,
    quantized_linear_cls,
    step_or_epoch_label,
    log_fn=print,
    metadata=None,
):
    quant_points_config = extract_quant_config(
        model,
        quantized_layer_names,
        num_bits,
        quantized_linear_cls,
        log_fn=log_fn,
        metadata=metadata,
    )
    quant_config_path = os.path.join(quant_config_dir, f"quant_config_{step_or_epoch_label}.pt")
    os.makedirs(quant_config_dir, exist_ok=True)
    try:
        torch.save(quant_points_config, quant_config_path)
    except Exception as exc:
        if log_fn:
            log_fn(f"Error saving quantization config: {exc}")
    return quant_config_path


def apply_quant_config(model, quant_config, quantized_linear_cls, log_fn=print):
    applied = 0
    for layer_name, layer_config in quant_config.items():
        if str(layer_name).startswith("__") or not isinstance(layer_config, dict):
            continue
        if not (
            "quant_points" in layer_config
            or "adapter_coefficients" in layer_config
            or "bucket_offsets" in layer_config
            or isinstance(layer_config.get("k"), dict)
            or isinstance(layer_config.get("v"), dict)
        ):
            continue
        parent_name = layer_name.rsplit(".", 1)[0]
        target_proj_name = layer_name.split(".")[-1]
        try:
            parent_module = model.get_submodule(parent_name)
            wrapped_module = getattr(parent_module, target_proj_name)

            # Fused QKV path (duck-typing)
            if hasattr(wrapped_module, "k_quantizer") and hasattr(wrapped_module, "v_quantizer"):
                # layer_config expected to contain optional 'k'/'v' sub-configs
                def _apply(qm, cfg):
                    nonlocal applied
                    if qm is None or cfg is None:
                        return
                    if "quant_points" in cfg:
                        set_quant_params(qm, cfg["quant_points"], cfg.get("thresholds"))
                    elif "adapter_coefficients" in cfg:
                        setter = getattr(qm, "set_adapter_state", None)
                        if not callable(setter):
                            raise TypeError(
                                "Target module cannot load full-precision adapter state"
                            )
                        setter(cfg["adapter_coefficients"])
                    elif "bucket_offsets" in cfg:
                        setter = getattr(qm, "set_bucket_residual_state", None)
                        if not callable(setter):
                            raise TypeError(
                                "Target module cannot load bucket-residual adapter state"
                            )
                        setter(cfg["bucket_offsets"], cfg.get("bucket_thresholds"))
                    else:
                        return
                    applied += 1

                _apply(wrapped_module.k_quantizer, layer_config.get("k"))
                _apply(wrapped_module.v_quantizer, layer_config.get("v"))
                continue

            # Non-fused path
            if not isinstance(wrapped_module, quantized_linear_cls):
                continue
            quant_module = wrapped_module.quantizer
            if "quant_points" in layer_config:
                set_quant_params(
                    quant_module,
                    layer_config["quant_points"],
                    layer_config.get("thresholds"),
                )
            elif "adapter_coefficients" in layer_config:
                setter = getattr(quant_module, "set_adapter_state", None)
                if not callable(setter):
                    raise TypeError("Target module cannot load full-precision adapter state")
                setter(layer_config["adapter_coefficients"])
            elif "bucket_offsets" in layer_config:
                setter = getattr(quant_module, "set_bucket_residual_state", None)
                if not callable(setter):
                    raise TypeError("Target module cannot load bucket-residual adapter state")
                setter(
                    layer_config["bucket_offsets"],
                    layer_config.get("bucket_thresholds"),
                )
            else:
                continue
            applied += 1
        except Exception as exc:
            if log_fn:
                log_fn(f"Error applying quant config for {layer_name}: {exc}")
            continue
    if log_fn:
        log_fn(f"Applied quant config to {applied} layers.")
    return applied
