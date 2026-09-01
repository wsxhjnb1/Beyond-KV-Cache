from __future__ import annotations

import pytest
import torch

from beyond.quantization.layers import UnifiedQuantLayer
from beyond.quantization.table_precision import (
    FP16_STRICT_MIN_GAP,
    TABLE_PRECISION_ABI,
    cast_table_for_storage,
    fp16_ste_roundtrip,
    validate_fp16_storage_tables,
)
from beyond.quantization.training import project_quant_points, project_thresholds


def test_fp16_shadow_uses_deployment_values_and_fp32_master_gradients():
    master = torch.tensor([0.123456, 0.654321], dtype=torch.float32, requires_grad=True)

    effective = fp16_ste_roundtrip(master)

    assert effective.dtype == torch.float32
    assert torch.equal(effective.detach(), master.detach().to(torch.float16).to(torch.float32))
    effective.sum().backward()
    assert torch.equal(master.grad, torch.ones_like(master))


def test_fp16_shadow_rejects_half_precision_master_parameters():
    master = torch.ones(2, dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(TypeError, match="master must be float32"):
        fp16_ste_roundtrip(master)


def test_quant_layer_keeps_fp32_masters_but_materializes_fp16_lattice_values():
    layer = UnifiedQuantLayer(
        num_bits=4,
        group_size=4,
        grouping_dim="token",
        quant_width=4,
    )

    q_points, thresholds = layer.materialize_quant_tables()

    assert layer.q_points.dtype == torch.float32
    assert layer.thresholds.dtype == torch.float32
    assert layer.table_storage_dtype == "float16"
    assert torch.equal(q_points.detach(), q_points.detach().to(torch.float16).to(torch.float32))
    assert torch.equal(thresholds.detach(), thresholds.detach().to(torch.float16).to(torch.float32))
    (q_points.sum() + thresholds.sum()).backward()
    assert torch.equal(layer.q_points.grad, torch.ones_like(layer.q_points))
    assert torch.equal(layer.thresholds.grad, torch.ones_like(layer.thresholds))


def test_default_projection_keeps_tables_distinct_after_fp16_rounding():
    quant_points = torch.tensor([[0.5, 0.500001, 0.500002]], dtype=torch.float32)
    thresholds = torch.tensor([[0.75, 0.750001]], dtype=torch.float32)

    project_quant_points(quant_points)
    project_thresholds(thresholds)

    q_storage = cast_table_for_storage(quant_points)
    t_storage = cast_table_for_storage(thresholds)
    assert bool((q_storage[..., 1:] > q_storage[..., :-1]).all())
    assert bool((t_storage[..., 1:] > t_storage[..., :-1]).all())
    assert float((quant_points[..., 1:] - quant_points[..., :-1]).min()) >= (
        FP16_STRICT_MIN_GAP - 1e-7
    )


def test_fp16_storage_validation_rejects_collapsed_tables():
    q_points = torch.tensor([[0.0, 0.5, 0.5]], dtype=torch.float16)
    thresholds = torch.tensor([[0.25, 0.75]], dtype=torch.float16)

    with pytest.raises(ValueError, match="collapse"):
        validate_fp16_storage_tables(q_points, thresholds, label="test")


def test_fp16_storage_validation_accepts_versioned_ordered_tables():
    q_points = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float16)
    thresholds = torch.tensor([[0.25, 0.75]], dtype=torch.float16)

    validate_fp16_storage_tables(q_points, thresholds, label=TABLE_PRECISION_ABI)
