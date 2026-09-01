from __future__ import annotations

import torch

from quant.beyond_cute import prepare_qpoints_and_thresholds


def test_uniform_runtime_tables_use_fp16_storage():
    q_points, thresholds = prepare_qpoints_and_thresholds(
        num_bits=4,
        device=torch.device("cpu"),
        q_points=None,
    )

    assert q_points.dtype == torch.float16
    assert thresholds.dtype == torch.float16
    assert q_points.is_contiguous()
    assert thresholds.is_contiguous()


def test_versioned_fp16_runtime_tables_remain_fp16():
    q_points = torch.linspace(0.0, 1.0, 16, dtype=torch.float16)
    thresholds = torch.linspace(0.01, 0.99, 15, dtype=torch.float16)

    prepared_q, prepared_t = prepare_qpoints_and_thresholds(
        num_bits=4,
        device=torch.device("cpu"),
        q_points=q_points,
        thresholds=thresholds,
    )

    assert prepared_q.dtype == torch.float16
    assert prepared_t.dtype == torch.float16
    assert torch.equal(prepared_q, q_points)
    assert torch.equal(prepared_t, thresholds)


def test_legacy_fp32_runtime_tables_remain_available_for_reproduction():
    q_points = torch.linspace(0.0, 1.0, 16, dtype=torch.float32)
    thresholds = torch.linspace(0.01, 0.99, 15, dtype=torch.float32)

    prepared_q, prepared_t = prepare_qpoints_and_thresholds(
        num_bits=4,
        device=torch.device("cpu"),
        q_points=q_points,
        thresholds=thresholds,
    )

    assert prepared_q.dtype == torch.float32
    assert prepared_t.dtype == torch.float32
