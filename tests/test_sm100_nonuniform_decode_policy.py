from __future__ import annotations

from pathlib import Path

import pytest

from quant.fa4_cute.beyond_decode_policy import (
    conversion_stage_cap,
    modeled_splits,
    packed_kv_stage_cap,
    scheduled_pages_per_split,
    select_cluster_splits,
    select_decode_launch_plan,
    select_direct_smem_store,
    select_reduction_kind,
    select_task_major_grid,
)


def test_sm100_online_softmax_retains_running_max() -> None:
    """Guard the long-context online-softmax invariant in the CuTe kernel."""
    source = (
        Path(__file__).resolve().parents[1] / "quant" / "fa4_cute" / "beyond_mixed_decode_sm100.py"
    ).read_text(encoding="utf-8")
    assert "tSrM_lane = cute.arch.fmax(tSrM[i], tSrM_prev[i])" in source


@pytest.mark.parametrize(
    (
        "tiles",
        "splits",
        "adaptive_split_base",
        "adaptive_extra_tasks",
        "expected",
    ),
    [
        (24, 3, 0, 0, 8),
        (24, 4, 3, 4, 8),
        (25, 4, 3, 1, 9),
        (8, 8, 0, 0, 1),
    ],
)
def test_scheduled_pages_per_split_reports_largest_cta_chunk(
    tiles: int,
    splits: int,
    adaptive_split_base: int,
    adaptive_extra_tasks: int,
    expected: int,
) -> None:
    assert (
        scheduled_pages_per_split(
            tiles=tiles,
            splits=splits,
            adaptive_split_base=adaptive_split_base,
            adaptive_extra_tasks=adaptive_extra_tasks,
        )
        == expected
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tiles": 0, "splits": 1},
        {"tiles": 8, "splits": 0},
        {
            "tiles": 8,
            "splits": 3,
            "adaptive_split_base": 0,
            "adaptive_extra_tasks": 1,
        },
        {
            "tiles": 8,
            "splits": 4,
            "adaptive_split_base": 2,
            "adaptive_extra_tasks": 1,
        },
        {
            "tiles": 8,
            "splits": 2,
            "adaptive_split_base": 1,
            "adaptive_extra_tasks": 0,
        },
    ],
)
def test_scheduled_pages_per_split_rejects_inconsistent_schedule(
    kwargs: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        scheduled_pages_per_split(**kwargs)


@pytest.mark.parametrize(
    ("tiles", "batch", "heads_kv", "expected"),
    [
        (24, 5, 8, 3),
        (24, 7, 8, 2),
        (24, 12, 8, 3),
        (1024, 5, 8, 11),
        (1024, 7, 8, 13),
        (1024, 16, 8, 8),
        (1024, 8, 4, 9),
    ],
)
def test_modeled_splits_matches_frozen_b300_frontier(
    tiles: int, batch: int, heads_kv: int, expected: int
) -> None:
    assert (
        modeled_splits(
            tiles=tiles,
            batch=batch,
            heads_kv=heads_kv,
            sm_count=148,
        )
        == expected
    )


def test_modeled_splits_never_exceeds_tiles_or_cap() -> None:
    assert modeled_splits(tiles=3, batch=1, heads_kv=4, sm_count=148) <= 3
    assert (
        modeled_splits(
            tiles=1024,
            batch=1,
            heads_kv=4,
            sm_count=148,
            max_splits=7,
        )
        <= 7
    )


@pytest.mark.parametrize("name", ["tiles", "batch", "heads_kv", "sm_count", "max_splits"])
def test_modeled_splits_rejects_nonpositive_inputs(name: str) -> None:
    kwargs = {
        "tiles": 8,
        "batch": 1,
        "heads_kv": 4,
        "sm_count": 148,
        "max_splits": 32,
    }
    kwargs[name] = 0
    with pytest.raises(ValueError, match=name):
        modeled_splits(**kwargs)


@pytest.mark.parametrize(
    ("tiles", "batch", "heads_kv", "expected"),
    [
        (32, 1, 4, 16),
        (128, 1, 4, 16),
        (32, 1, 8, 9),
        (32, 2, 4, 9),
        (32, 4, 8, 4),
        (32, 8, 4, 4),
        (40, 2, 4, 10),
        (48, 5, 4, 6),
        (48, 6, 4, 5),
        (48, 7, 4, 4),
        (40, 4, 8, 4),
        (48, 4, 8, 4),
        (28, 10, 4, 3),
        (28, 14, 4, 2),
        (12, 10, 8, 3),
        (96, 14, 8, 2),
        (48, 2, 8, 6),
        (28, 5, 4, 6),
        (40, 5, 4, 6),
        (12, 6, 4, 4),
        (40, 6, 4, 5),
        (28, 7, 4, 4),
        (40, 7, 4, 4),
        (192, 4, 8, 4),
        (24, 4, 8, 4),
        (28, 4, 8, 4),
        (192, 5, 8, 3),
        (24, 5, 8, 3),
        (40, 5, 8, 3),
        (96, 10, 4, 3),
        (12, 7, 8, 2),
        (24, 7, 8, 2),
        (96, 7, 8, 2),
        (28, 2, 4, 10),
        (192, 10, 8, 3),
        (28, 10, 8, 3),
        (48, 10, 8, 3),
        (128, 1, 8, 0),
        (64, 1, 4, 16),
    ],
)
def test_cluster_splits_match_fixed_clock_b300_frontier(
    tiles: int, batch: int, heads_kv: int, expected: int
) -> None:
    assert (
        select_cluster_splits(
            tiles=tiles,
            batch=batch,
            heads_kv=heads_kv,
        )
        == expected
    )


@pytest.mark.parametrize("name", ["tiles", "batch", "heads_kv"])
def test_cluster_splits_reject_nonpositive_inputs(name: str) -> None:
    kwargs = {"tiles": 32, "batch": 1, "heads_kv": 4}
    kwargs[name] = 0
    with pytest.raises(ValueError, match=name):
        select_cluster_splits(**kwargs)


@pytest.mark.parametrize(
    ("batch", "splits", "expected"),
    [
        (16, 1, "parallel"),
        (1, 8, "parallel"),
        (2, 12, "parallel"),
        (3, 8, "warp_parallel"),
        (5, 7, "warp_parallel"),
        (3, 4, "warp_parallel2"),
        (16, 2, "warp_parallel2"),
    ],
)
def test_reduction_kind_matches_fixed_clock_frontier(
    batch: int, splits: int, expected: str
) -> None:
    assert select_reduction_kind(batch=batch, splits=splits) == expected


def test_reduction_kind_selects_cta4_at_long_context_boundary() -> None:
    assert select_reduction_kind(batch=1, splits=16, pages_per_split=8) == "cta4"
    assert select_reduction_kind(batch=2, splits=16, pages_per_split=2) == "parallel"


@pytest.mark.parametrize("name", ["batch", "splits"])
def test_reduction_kind_rejects_nonpositive_inputs(name: str) -> None:
    kwargs = {"batch": 8, "splits": 4}
    kwargs[name] = 0
    with pytest.raises(ValueError, match=name):
        select_reduction_kind(**kwargs)


@pytest.mark.parametrize(
    ("splits", "pages_per_split", "expected"),
    [
        (1, 1, 1),
        (1, 2, 3),
        (1, 8, 1),
        (1, 12, 3),
        (1, 13, 1),
        (2, 8, 1),
        (4, 4, 1),
    ],
)
def test_conversion_stage_cap_uses_measured_direct_window(
    splits: int, pages_per_split: int, expected: int
) -> None:
    assert conversion_stage_cap(splits=splits, pages_per_split=pages_per_split) == expected


def test_conversion_stage_cap_uses_one_stage_for_eight_page_direct() -> None:
    assert conversion_stage_cap(splits=1, pages_per_split=8, model_dtype="bf16") == 1


@pytest.mark.parametrize(
    ("batch", "heads_kv", "splits", "pages_per_split", "expected"),
    [
        (1, 4, 1, 1, True),
        (16, 4, 1, 1, True),
        (16, 8, 1, 4, True),
        (16, 8, 1, 8, True),
        (8, 8, 1, 8, False),
        (16, 8, 1, 32, False),
        (16, 8, 2, 4, False),
    ],
)
def test_direct_smem_store_matches_supported_policy(
    batch: int,
    heads_kv: int,
    splits: int,
    pages_per_split: int,
    expected: bool,
) -> None:
    assert (
        select_direct_smem_store(
            batch=batch,
            heads_kv=heads_kv,
            splits=splits,
            pages_per_split=pages_per_split,
        )
        is expected
    )


@pytest.mark.parametrize("name", ["batch", "heads_kv", "splits", "pages_per_split"])
def test_direct_smem_store_rejects_nonpositive_inputs(name: str) -> None:
    kwargs = {
        "batch": 16,
        "heads_kv": 8,
        "splits": 1,
        "pages_per_split": 8,
    }
    kwargs[name] = 0
    with pytest.raises(ValueError, match=name):
        select_direct_smem_store(**kwargs)
    assert conversion_stage_cap(splits=1, pages_per_split=8, model_dtype="fp16") == 1


@pytest.mark.parametrize(
    ("batch", "splits", "pages_per_split", "expected"),
    [
        (1, 16, 8, True),
        (2, 16, 8, True),
        (1, 8, 2, False),
        (2, 8, 4, False),
        (4, 4, 2, False),
        (3, 16, 8, False),
        (2, 16, 7, False),
        (2, 15, 8, False),
    ],
)
def test_task_major_grid_matches_fresh_fixed_clock_frontier(
    batch: int, splits: int, pages_per_split: int, expected: bool
) -> None:
    assert (
        select_task_major_grid(
            batch=batch,
            splits=splits,
            pages_per_split=pages_per_split,
        )
        is expected
    )


@pytest.mark.parametrize("name", ["splits", "pages_per_split"])
def test_conversion_stage_cap_rejects_nonpositive_inputs(name: str) -> None:
    kwargs = {"splits": 1, "pages_per_split": 8}
    kwargs[name] = 0
    with pytest.raises(ValueError, match=name):
        conversion_stage_cap(**kwargs)


@pytest.mark.parametrize(
    ("pages_per_split", "expected"),
    [(1, 1), (12, 1), (32, 1), (33, 2), (64, 2), (256, 2)],
)
def test_packed_kv_stage_cap_uses_measured_crossover(pages_per_split: int, expected: int) -> None:
    assert packed_kv_stage_cap(pages_per_split=pages_per_split) == expected


def test_packed_kv_stage_cap_rejects_nonpositive_input() -> None:
    with pytest.raises(ValueError, match="pages_per_split"):
        packed_kv_stage_cap(pages_per_split=0)


@pytest.mark.parametrize(
    (
        "tiles",
        "batch",
        "heads_kv",
        "expected",
    ),
    [
        (16, 1, 8, (8, False, "parallel", 1, 1, False)),
        (16, 3, 8, (4, True, "warp_parallel2", 1, 1, False)),
        (32, 3, 8, (5, True, "warp_parallel", 1, 1, False)),
        (64, 3, 8, (6, False, "cta4", 1, 4, False)),
        (16, 5, 8, (3, True, "warp_parallel2", 1, 1, False)),
        (32, 5, 8, (3, True, "warp_parallel2", 1, 1, False)),
        (128, 5, 8, (3, True, "warp_parallel2", 2, 1, False)),
        (16, 8, 8, (2, True, "warp_parallel2", 1, 1, False)),
        (256, 8, 8, (2, False, "warp_parallel2", 2, 1, False)),
        (8, 12, 8, (1, False, "parallel", 1, 1, True)),
        (16, 12, 8, (1, False, "parallel", 1, 4, True)),
        (64, 12, 8, (3, False, "warp_parallel2", 2, 1, False)),
        (8, 16, 8, (1, False, "parallel", 1, 1, True)),
        (256, 16, 8, (1, False, "parallel", 2, 1, False)),
        (64, 1, 4, (16, True, "parallel", 1, 1, False)),
        (256, 1, 4, (32, False, "cta4", 1, 1, False)),
        (16, 3, 4, (10, False, "parallel", 1, 1, False)),
        (32, 3, 4, (8, True, "warp_parallel", 1, 1, False)),
        (64, 3, 4, (8, True, "warp_parallel", 1, 1, False)),
        (128, 3, 4, (12, False, "cta4", 1, 2, False)),
        (16, 5, 4, (6, True, "warp_parallel", 1, 1, False)),
        (40, 2, 4, (10, True, "parallel", 1, 1, False)),
        (48, 5, 4, (6, True, "warp_parallel", 1, 1, False)),
        (48, 6, 4, (5, True, "warp_parallel", 1, 1, False)),
        (48, 7, 4, (4, True, "warp_parallel2", 1, 1, False)),
        (40, 4, 8, (4, True, "warp_parallel2", 1, 1, False)),
        (48, 4, 8, (4, True, "warp_parallel2", 1, 1, False)),
        (28, 10, 4, (3, True, "warp_parallel2", 1, 1, False)),
        (28, 14, 4, (2, True, "warp_parallel2", 1, 1, False)),
        (12, 10, 8, (3, True, "warp_parallel2", 1, 1, False)),
        (96, 14, 8, (2, True, "warp_parallel2", 2, 1, False)),
        (48, 2, 8, (6, True, "warp_parallel", 1, 1, False)),
        (28, 5, 4, (6, True, "warp_parallel", 1, 1, False)),
        (40, 5, 4, (6, True, "warp_parallel", 1, 1, False)),
        (12, 6, 4, (4, True, "warp_parallel2", 1, 1, False)),
        (40, 6, 4, (5, True, "warp_parallel", 1, 1, False)),
        (28, 7, 4, (4, True, "warp_parallel2", 1, 1, False)),
        (40, 7, 4, (4, True, "warp_parallel2", 1, 1, False)),
        (192, 4, 8, (4, True, "warp_parallel2", 2, 1, False)),
        (24, 4, 8, (4, True, "warp_parallel2", 1, 1, False)),
        (28, 4, 8, (4, True, "warp_parallel2", 1, 1, False)),
        (192, 5, 8, (3, True, "warp_parallel2", 2, 1, False)),
        (24, 5, 8, (3, True, "warp_parallel2", 1, 1, False)),
        (40, 5, 8, (3, True, "warp_parallel2", 1, 1, False)),
        (96, 10, 4, (3, True, "warp_parallel2", 1, 1, False)),
        (12, 7, 8, (2, True, "warp_parallel2", 1, 1, False)),
        (24, 7, 8, (2, True, "warp_parallel2", 1, 1, False)),
        (96, 7, 8, (2, True, "warp_parallel2", 2, 1, False)),
        (28, 2, 4, (10, True, "parallel", 1, 1, False)),
        (192, 10, 8, (3, True, "warp_parallel2", 2, 1, False)),
        (28, 10, 8, (3, True, "warp_parallel2", 1, 1, False)),
        (48, 10, 8, (3, True, "warp_parallel2", 1, 1, False)),
        (64, 5, 4, (6, True, "warp_parallel", 2, 1, False)),
        (32, 8, 4, (4, True, "warp_parallel2", 2, 1, False)),
        (256, 8, 4, (4, False, "warp_parallel2", 2, 1, False)),
        (32, 12, 4, (3, False, "cta4", 1, 2, False)),
        (16, 16, 4, (2, True, "warp_parallel2", 1, 1, False)),
        (256, 16, 4, (2, False, "warp_parallel2", 2, 1, False)),
    ],
)
def test_decode_launch_plan_matches_five_process_chosen_retests(
    tiles: int,
    batch: int,
    heads_kv: int,
    expected: tuple[int, bool, str, int, int, bool],
) -> None:
    plan = select_decode_launch_plan(
        tiles=tiles,
        batch=batch,
        heads_kv=heads_kv,
        sm_count=148,
        model_dtype="bf16",
    )
    assert (
        plan.splits,
        plan.cluster_reduction,
        plan.reduction_kind,
        plan.kv_stage_cap,
        plan.cvt_stage_cap,
        plan.direct_smem_store,
    ) == expected


def test_decode_launch_plan_falls_back_off_measured_b300_topology() -> None:
    plan = select_decode_launch_plan(
        tiles=16,
        batch=3,
        heads_kv=8,
        sm_count=120,
        model_dtype="bf16",
    )
    assert not plan.cluster_reduction
    assert plan.splits == modeled_splits(
        tiles=16,
        batch=3,
        heads_kv=8,
        sm_count=120,
    )
