"""Tests for QuantConfig loader + layer name matching."""

import json
import os
import tempfile

import pytest
import torch

from beyond.quantization.runtime_config import QuantConfig  # noqa: E402
from beyond.quantization.table_precision import (  # noqa: E402
    LEGACY_TABLE_PRECISION_ABI,
    TABLE_PRECISION_ABI,
)


def _make_layer_entry(num_bits, grouping_dim, proj_type, layer_idx):
    L = 1 << num_bits
    # Strictly increasing, non-uniform, and contained in the normalized range.
    qp = [(i / (L - 1)) ** 1.1 for i in range(L)]
    th = [(qp[i] + qp[i + 1]) / 2.0 for i in range(L - 1)]
    return {
        f"model.layers.{layer_idx}.self_attn.{proj_type}": {
            "type": f"UnifiedQuantLayer_{num_bits}bit_{grouping_dim}",
            "proj_type": proj_type,
            "num_bits": num_bits,
            "group_size": 32,
            "grouping_dim": grouping_dim,
            "quant_points": qp,
            "thresholds": th,
        }
    }


@pytest.mark.parametrize("bits", [5, 6])
def test_quant_config_loads_and_matches_layer(bits):
    data = {}
    for li in range(3):
        data.update(_make_layer_entry(bits, "token", "k_proj", li))
        data.update(_make_layer_entry(bits, "token", "v_proj", li))

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(data, tf)
        path = tf.name
    try:
        cfg = QuantConfig.load(path, expected_bits=bits)
        assert len(cfg) == 6
        # The Attention layer_name looks like "model.layers.0.self_attn.attn"
        layer_name = "model.layers.1.self_attn.attn"
        k = cfg.for_attention_layer(layer_name, "k_proj")
        v = cfg.for_attention_layer(layer_name, "v_proj")
        assert k is not None and v is not None
        assert k.num_bits == bits and v.num_bits == bits
        assert k.grouping_dim == "token"
        assert v.grouping_dim == "token"
        assert k.q_points.shape == (1 << bits,)
        assert k.thresholds.shape == ((1 << bits) - 1,)
        assert torch.is_floating_point(k.q_points)
        assert k.q_points.dtype == torch.float32
        assert k.table_storage_dtype == "float32"
        assert k.table_precision_abi == LEGACY_TABLE_PRECISION_ABI
    finally:
        os.unlink(path)


def test_quant_config_matches_ministral3_hf_text_tower_names_to_vllm_names():
    data = {
        "model.language_model.layers.0.self_attn.k_proj": next(
            iter(_make_layer_entry(4, "token", "k_proj", 0).values())
        ),
        "model.language_model.layers.0.self_attn.v_proj": next(
            iter(_make_layer_entry(4, "token", "v_proj", 0).values())
        ),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(data, tf)
        path = tf.name
    try:
        cfg = QuantConfig.load(path, expected_bits=4)
        assert len(cfg) == 2
        layer_name = "language_model.model.layers.0.self_attn.attn"
        assert cfg.for_attention_layer(layer_name, "k_proj").grouping_dim == "token"
        assert cfg.for_attention_layer(layer_name, "v_proj").grouping_dim == "token"
    finally:
        os.unlink(path)


def test_quant_config_matches_ministral3_text_only_names_to_vllm_wrapper_names():
    data = _make_layer_entry(4, "token", "k_proj", 0)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(data, tf)
        path = tf.name
    try:
        cfg = QuantConfig.load(path, expected_bits=4)
        layer_name = "language_model.model.layers.0.self_attn.attn"
        assert cfg.for_attention_layer(layer_name, "k_proj").grouping_dim == "token"
    finally:
        os.unlink(path)


def test_quant_config_expected_bits_mismatch_raises():
    data = _make_layer_entry(5, "token", "k_proj", 0)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(data, tf)
        path = tf.name
    try:
        with pytest.raises(ValueError, match="num_bits"):
            QuantConfig.load(path, expected_bits=6)
    finally:
        os.unlink(path)


def test_quant_config_rejects_sequence_grouping():
    data = _make_layer_entry(5, "sequence", "k_proj", 0)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(data, tf)
        path = tf.name
    try:
        with pytest.raises(ValueError, match="token-grouped"):
            QuantConfig.load(path, expected_bits=5)
    finally:
        os.unlink(path)


def test_quant_config_loads_discover_fused_qkv_export():
    k_entry = next(iter(_make_layer_entry(4, "token", "k_proj", 0).values()))
    v_entry = next(iter(_make_layer_entry(4, "token", "v_proj", 0).values()))
    data = {
        "model.layers.0.self_attn.query_key_value": {
            "type": "QuantizedQKVLinear",
            "proj_type": "query_key_value",
            "k": k_entry,
            "v": v_entry,
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(data, tf)
        path = tf.name
    try:
        cfg = QuantConfig.load(path, expected_bits=4)
        assert len(cfg) == 2
        layer_name = "model.layers.0.self_attn.attn"
        assert cfg.for_attention_layer(layer_name, "k_proj").grouping_dim == "token"
        assert cfg.for_attention_layer(layer_name, "v_proj").grouping_dim == "token"
    finally:
        os.unlink(path)


def test_quant_config_loads_discover_fused_qkv_proj_export():
    k_entry = next(iter(_make_layer_entry(4, "token", "k_proj", 0).values()))
    v_entry = next(iter(_make_layer_entry(4, "token", "v_proj", 0).values()))
    data = {
        "model.layers.0.self_attn.qkv_proj": {
            "type": "QuantizedQKVLinear",
            "proj_type": "qkv_proj",
            "k": k_entry,
            "v": v_entry,
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(data, tf)
        path = tf.name
    try:
        cfg = QuantConfig.load(path, expected_bits=4)
        assert len(cfg) == 2
        layer_name = "model.layers.0.self_attn.attn"
        assert cfg.for_attention_layer(layer_name, "k_proj").grouping_dim == "token"
        assert cfg.for_attention_layer(layer_name, "v_proj").grouping_dim == "token"
    finally:
        os.unlink(path)


def test_quant_config_loads_pt_grouped_token_tables():
    q_points = torch.stack(
        [
            torch.linspace(0.0, 1.0, 16),
            torch.linspace(0.0, 1.0, 16) ** 0.5,
        ],
    )
    thresholds = (q_points[..., :-1] + q_points[..., 1:]) / 2.0
    data = {
        "model.layers.0.self_attn.v_proj": {
            "type": "UnifiedQuantLayer_4bit_token",
            "proj_type": "v_proj",
            "num_bits": 4,
            "grouping_dim": "token",
            "quant_points": q_points,
            "thresholds": thresholds,
        }
    }
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
        path = tf.name
    torch.save(data, path)
    try:
        cfg = QuantConfig.load(path, expected_bits=4)
        entry = cfg.for_attention_layer("model.layers.0.self_attn.attn", "v_proj")
        assert entry is not None
        assert entry.table_axis == "group"
        assert entry.q_points.shape == (2, 16)
        assert entry.thresholds.shape == (2, 15)
    finally:
        os.unlink(path)


def test_quant_config_preserves_versioned_fp16_deployment_tables():
    q_points = torch.stack(
        [
            torch.linspace(0.0, 1.0, 16),
            torch.linspace(0.0, 1.0, 16) ** 0.5,
        ],
    ).to(torch.float16)
    thresholds = (
        (q_points.to(torch.float32)[..., :-1] + q_points.to(torch.float32)[..., 1:]) / 2.0
    ).to(torch.float16)
    data = {
        "model.layers.0.self_attn.v_proj": {
            "type": "UnifiedQuantLayer_4bit_token",
            "proj_type": "v_proj",
            "num_bits": 4,
            "grouping_dim": "token",
            "table_storage_dtype": "float16",
            "table_compute_dtype": "float32",
            "table_precision_abi": TABLE_PRECISION_ABI,
            "quant_points": q_points,
            "thresholds": thresholds,
        }
    }
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
        path = tf.name
    torch.save(data, path)
    try:
        cfg = QuantConfig.load(path, expected_bits=4)
        entry = cfg.for_attention_layer("model.layers.0.self_attn.attn", "v_proj")
        assert entry is not None
        assert entry.q_points.dtype == torch.float16
        assert entry.thresholds.dtype == torch.float16
        assert entry.table_storage_dtype == "float16"
        assert entry.table_compute_dtype == "float32"
        assert entry.table_precision_abi == TABLE_PRECISION_ABI
    finally:
        os.unlink(path)


def test_quant_config_rejects_fp16_tables_with_wrong_abi():
    data = _make_layer_entry(4, "token", "v_proj", 0)
    entry = next(iter(data.values()))
    entry["table_storage_dtype"] = "float16"
    entry["table_precision_abi"] = "wrong"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(data, tf)
        path = tf.name
    try:
        with pytest.raises(ValueError, match="float16 tables require"):
            QuantConfig.load(path, expected_bits=4)
    finally:
        os.unlink(path)


def test_quant_config_rejects_unversioned_fp16_tables():
    data = _make_layer_entry(4, "token", "v_proj", 0)
    entry = next(iter(data.values()))
    entry["table_storage_dtype"] = "float16"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(data, tf)
        path = tf.name
    try:
        with pytest.raises(ValueError, match="explicit table_precision_abi"):
            QuantConfig.load(path, expected_bits=4)
    finally:
        os.unlink(path)


def test_quant_config_requires_complete_current_abi_metadata():
    data = _make_layer_entry(4, "token", "v_proj", 0)
    entry = next(iter(data.values()))
    entry["table_storage_dtype"] = "float16"
    entry["table_precision_abi"] = TABLE_PRECISION_ABI
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(data, tf)
        path = tf.name
    try:
        with pytest.raises(ValueError, match="explicit table_compute_dtype"):
            QuantConfig.load(path, expected_bits=4)
    finally:
        os.unlink(path)


def test_quant_config_rejects_bad_grouped_threshold_shape():
    q_points = torch.stack(
        [
            torch.linspace(0.0, 1.0, 16),
            torch.linspace(0.0, 1.0, 16) ** 0.5,
        ],
    )
    data = {
        "model.layers.0.self_attn.v_proj": {
            "type": "UnifiedQuantLayer_4bit_token",
            "proj_type": "v_proj",
            "num_bits": 4,
            "grouping_dim": "token",
            "quant_points": q_points,
            "thresholds": torch.zeros(15),
        }
    }
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
        path = tf.name
    torch.save(data, path)
    try:
        with pytest.raises(ValueError, match="thresholds shape"):
            QuantConfig.load(path, expected_bits=4)
    finally:
        os.unlink(path)


def test_env_loader_caches_by_path(monkeypatch):
    data = _make_layer_entry(5, "token", "k_proj", 0)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(data, tf)
        path = tf.name
    try:
        monkeypatch.setenv("BEYOND_QUANT_CONFIG", path)
        monkeypatch.setenv("BEYOND_BITS", "5")
        monkeypatch.delenv("BEYOND_MODEL_ID", raising=False)
        import beyond.quantization.runtime_config as qc

        qc._cached = None
        qc._cached_path = None
        cfg1 = qc.get_config_from_env()
        cfg2 = qc.get_config_from_env()
        assert cfg1 is cfg2  # cached
        assert cfg1 is not None and len(cfg1) == 1
    finally:
        os.unlink(path)


def test_quant_config_model_identity_uses_run_metrics_sidecar(tmp_path):
    run_dir = tmp_path / "run"
    quant_dir = run_dir / "quant_configs"
    quant_dir.mkdir(parents=True)
    config_path = quant_dir / "quant_config_final.json"
    config_path.write_text(
        json.dumps(_make_layer_entry(4, "token", "k_proj", 0)),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"base_model": "meta-llama/Llama-3.1-8B-Instruct"}),
        encoding="utf-8",
    )

    cfg = QuantConfig.load(
        str(config_path),
        expected_bits=4,
        expected_model="meta-llama/Llama-3.1-8B-Instruct",
        require_model_metadata=True,
    )
    assert cfg.source_model == "meta-llama/Llama-3.1-8B-Instruct"
    with pytest.raises(ValueError, match="does not match requested model"):
        QuantConfig.load(
            str(config_path),
            expected_bits=4,
            expected_model="Qwen/Qwen3-8B-Base",
            require_model_metadata=True,
        )


def test_quant_config_model_identity_prefers_embedded_metadata(tmp_path):
    config_path = tmp_path / "quant_config_final.json"
    payload = _make_layer_entry(4, "token", "k_proj", 0)
    payload["__metadata__"] = {
        "artifact_type": "beyond_kv_quant_config",
        "base_model": "Qwen/Qwen3-8B-Base",
        "model_revision": "deadbeef",
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = QuantConfig.load(
        str(config_path),
        expected_bits=4,
        expected_model="Qwen/Qwen3-8B-Base",
        require_model_metadata=True,
    )

    assert cfg.source_model == "Qwen/Qwen3-8B-Base"
    assert len(cfg) == 1


def test_quant_config_rejects_uncompressed_adapter_artifact(tmp_path):
    config_path = tmp_path / "adapter_config.pt"
    torch.save(
        {
            "__metadata__": {
                "artifact_type": "beyond_uncompressed_kv_adapter_config",
                "base_model": "Qwen/Qwen3-8B-Base",
                "compressed_kv_cache": False,
                "hard_forward": False,
            },
            "model.layers.0.self_attn.k_proj": {
                "num_bits": 4,
                "grouping_dim": "token",
                "table_axis": "group",
                "group_size": 32,
                "adapter_coefficients": torch.zeros(4, 31),
            },
        },
        config_path,
    )

    with pytest.raises(ValueError, match="uncompressed continuous KV adapter"):
        QuantConfig.load(str(config_path), expected_bits=4)


def test_env_loader_force_reload_changes_identity_for_same_path(monkeypatch, tmp_path):
    path = tmp_path / "quant_config.json"
    data = _make_layer_entry(4, "token", "k_proj", 0)
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("BEYOND_QUANT_CONFIG", str(path))
    monkeypatch.setenv("BEYOND_BITS", "4")
    monkeypatch.delenv("BEYOND_MODEL_ID", raising=False)
    import beyond.quantization.runtime_config as qc

    qc._cached = None
    qc._cached_path = None
    cfg1 = qc.get_config_from_env()
    key = next(iter(data))
    data[key]["quant_points"][1] += 1.0e-3
    path.write_text(json.dumps(data), encoding="utf-8")

    assert qc.get_config_from_env() is cfg1
    cfg2 = qc.get_config_from_env(force_reload=True)
    assert cfg2 is not cfg1
    assert cfg2.identity != cfg1.identity


def test_quant_config_rejects_nonmonotonic_tables(tmp_path):
    data = _make_layer_entry(4, "token", "k_proj", 0)
    key = next(iter(data))
    data[key]["thresholds"][2] = data[key]["thresholds"][1]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="strictly increasing"):
        QuantConfig.load(str(path), expected_bits=4)


def test_prime_quant_hbm_cache_preloads_qpoints_and_thresholds(monkeypatch):
    data = {}
    data.update(_make_layer_entry(4, "token", "k_proj", 0))
    data.update(_make_layer_entry(4, "token", "v_proj", 0))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(data, tf)
        path = tf.name
    try:
        monkeypatch.setenv("BEYOND_QUANT_CONFIG", path)
        monkeypatch.setenv("BEYOND_BITS", "4")
        monkeypatch.delenv("BEYOND_MODEL_ID", raising=False)
        import beyond.quantization.runtime_config as qc

        pytest.importorskip("vllm")
        from beyond.runtime.vllm import packed_backend as backend

        qc._cached = None
        qc._cached_path = None
        backend._HBM_QTABLE_CACHE.clear()
        backend._prime_quant_hbm_cache(torch.device("cpu"))

        identity = qc.get_config_from_env().identity
        k_key = ("cpu", identity, "model.layers.0.self_attn.k_proj")
        v_key = ("cpu", identity, "model.layers.0.self_attn.v_proj")
        assert k_key in backend._HBM_QTABLE_CACHE
        assert v_key in backend._HBM_QTABLE_CACHE

        k_qp, k_th = backend._HBM_QTABLE_CACHE[k_key]
        v_qp, v_th = backend._HBM_QTABLE_CACHE[v_key]
        assert k_qp.device.type == "cpu"
        assert v_qp.device.type == "cpu"
        assert k_qp.shape == (16,)
        assert v_qp.shape == (16,)
        assert k_th.shape == (15,)
        assert v_th.shape == (15,)
    finally:
        os.unlink(path)
