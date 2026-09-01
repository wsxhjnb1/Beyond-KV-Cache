"""Pure launch-policy helpers for the SM100 non-uniform decode kernel."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeLaunchPlan:
    """Concrete B300 launch configuration for one decode bucket."""

    splits: int
    pages_per_split: int
    reduction_kind: str
    kv_stage_cap: int
    cvt_stage_cap: int
    cluster_reduction: bool
    task_major_grid: bool
    direct_smem_store: bool


# Shape-specific SM103 launch policies use (batch * Hkv, block128 tiles) as the
# topology key. Supported shapes compile an N8 TCGEN05 tile, so equal keys use
# the same CTA geometry even when the number of valid query heads differs.
# The analytical fallback remains responsible for every unlisted shape.
_B300_CLUSTER_SPLITS = {
    (4, 32): 16,
    (4, 64): 16,
    (4, 128): 16,
    (8, 28): 10,
    (8, 32): 9,
    (8, 40): 10,
    (12, 32): 8,
    (12, 64): 8,
    (16, 48): 6,
    (20, 16): 6,
    (20, 28): 6,
    (20, 40): 6,
    (20, 48): 6,
    (20, 64): 6,
    (24, 12): 4,
    (24, 16): 4,
    (24, 32): 5,
    (24, 40): 5,
    (24, 48): 5,
    (28, 28): 4,
    (28, 40): 4,
    (28, 48): 4,
    (32, 24): 4,
    (32, 28): 4,
    (32, 32): 4,
    (32, 40): 4,
    (32, 48): 4,
    (32, 192): 4,
    (40, 16): 3,
    (40, 24): 3,
    (40, 28): 3,
    (40, 32): 3,
    (40, 40): 3,
    (40, 96): 3,
    (40, 128): 3,
    (40, 192): 3,
    (56, 12): 2,
    (56, 24): 2,
    (56, 28): 2,
    (56, 96): 2,
    (64, 16): 2,
    (80, 12): 3,
    (80, 28): 3,
    (80, 48): 3,
    (80, 192): 3,
    (112, 96): 2,
}

_B300_FIXED_SPLITS = {
    (4, 256): 32,
    (8, 16): 8,
    (12, 16): 10,
    (12, 128): 12,
    (24, 64): 6,
    (32, 256): 4,
    (48, 32): 3,
    (64, 256): 2,
    (96, 16): 1,
    (96, 64): 3,
    (128, 256): 1,
}

_B300_REDUCER_OVERRIDES = {
    (4, 256): "cta4",
    (12, 128): "cta4",
    (24, 64): "cta4",
    (48, 32): "cta4",
}

_B300_KV_STAGE_OVERRIDES = {
    (20, 64): 2,
    (32, 32): 2,
    (96, 64): 2,
}

_B300_CVT_STAGE_OVERRIDES = {
    (12, 128): 2,
    (24, 64): 4,
    (48, 32): 2,
    (96, 16): 4,
}

_B300_DIRECT_STORE_OVERRIDES = {
    (96, 8),
    (96, 16),
}


def scheduled_pages_per_split(
    *,
    tiles: int,
    splits: int,
    adaptive_split_base: int = 0,
    adaptive_extra_tasks: int = 0,
) -> int:
    """Return the largest page count assigned to any decode CTA.

    A fixed schedule divides every KV task into ``splits`` chunks.  The exact
    wave scheduler instead assigns ``adaptive_split_base`` chunks to most KV
    tasks and one extra chunk to ``adaptive_extra_tasks`` of them.  Shared-
    memory staging must be sized for the smaller split count because it owns
    the largest chunk.
    """
    values = {
        "tiles": tiles,
        "splits": splits,
        "adaptive_split_base": adaptive_split_base,
        "adaptive_extra_tasks": adaptive_extra_tasks,
    }
    invalid = [
        name
        for name, value in values.items()
        if value < 0 or (name in {"tiles", "splits"} and value == 0)
    ]
    if invalid:
        raise ValueError(
            "scheduled-page inputs must be positive/nonnegative as appropriate: "
            + ", ".join(invalid)
        )
    if adaptive_extra_tasks:
        if adaptive_split_base <= 0:
            raise ValueError(
                "adaptive_split_base must be positive when adaptive tasks exist"
            )
        if splits != adaptive_split_base + 1:
            raise ValueError(
                "splits must equal adaptive_split_base + 1 for an adaptive schedule"
            )
        schedule_splits = adaptive_split_base
    else:
        if adaptive_split_base:
            raise ValueError(
                "adaptive_split_base requires nonzero adaptive_extra_tasks"
            )
        schedule_splits = splits
    return (tiles + schedule_splits - 1) // schedule_splits


def modeled_splits(
    *,
    tiles: int,
    batch: int,
    heads_kv: int,
    sm_count: int,
    max_splits: int = 32,
) -> int:
    """Choose split-KV from resident waves, page work, and reducer cost.

    The model is intentionally hardware-only: codebook values do not affect
    execution time, so Llama and Ministral share the Hkv8 policy while Qwen
    uses the same formula with Hkv4. Constants are frozen from the B300
    GQA4 B5/B7/B12/B16 tuning frontier.
    """
    values = {
        "tiles": tiles,
        "batch": batch,
        "heads_kv": heads_kv,
        "sm_count": sm_count,
        "max_splits": max_splits,
    }
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise ValueError(f"launch-policy inputs must be positive: {', '.join(invalid)}")

    grid_per_split = batch * heads_kv
    candidates = range(1, min(max_splits, tiles) + 1)

    def score(splits: int) -> tuple[float, int]:
        resident_waves = (grid_per_split * splits + sm_count - 1) // sm_count
        pages_per_split = (tiles + splits - 1) // splits
        modeled_cost = resident_waves * (pages_per_split + 2)
        modeled_cost += 0.05 * batch * splits
        return modeled_cost, splits

    return min(candidates, key=score)


def select_cluster_splits(*, tiles: int, batch: int, heads_kv: int) -> int:
    """Return the measured single-cluster split count, or zero.

    A full DSM cluster removes the global split workspace round-trip and the
    dependent reducer launch.  It is only selected on fixed-clock B300
    topology points where the complete cluster remains resident and was the
    fastest retained BeYond path in repeated FP16 and BF16 trials.  Whether a
    row also beats the dense TRTLLM-Gen reference is reported separately; it
    is not a reason to retain a slower BeYond launch.  All unmeasured shapes
    keep the independent reducer path instead of extrapolating across a
    residency cliff.
    """
    values = {"tiles": tiles, "batch": batch, "heads_kv": heads_kv}
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise ValueError(
            f"cluster-policy inputs must be positive: {', '.join(invalid)}"
        )

    tasks = batch * heads_kv
    return _B300_CLUSTER_SPLITS.get((tasks, tiles), 0)


def select_reduction_kind(
    *, batch: int, splits: int, pages_per_split: int | None = None
) -> str:
    """Choose the independent split-output reducer measured on B300.

    Four heads per 512-thread CTA win at the measured 16-split/eight-page long
    context boundary.  Elsewhere the dedicated weight warp scales best for
    batch-one and more than eight splits.  A single output warp wins for split
    counts five through eight; two output warps hide the D128 vector tail for
    split counts two through four.  Split one bypasses this kernel, but returns
    ``parallel`` so callers always receive a valid concrete implementation.
    """
    values = {"batch": batch, "splits": splits}
    if pages_per_split is not None:
        values["pages_per_split"] = pages_per_split
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise ValueError(f"reduction-policy inputs must be positive: {', '.join(invalid)}")
    if splits == 16 and pages_per_split == 8:
        return "cta4"
    if splits == 1 or batch == 1 or splits > 8:
        return "parallel"
    if splits >= 5:
        return "warp_parallel"
    return "warp_parallel2"


def conversion_stage_cap(
    *, splits: int, pages_per_split: int, model_dtype: str | None = None
) -> int:
    """Choose the independent K/V conversion depth for B300 D128 decode.

    Fixed-clock sweeps across GQA4/GQA8 and FP16/BF16 found a narrow direct
    path window where three TMEM stages hide conversion latency.  The
    eight-page B16/S1K production point is consistently faster with one stage
    for both model dtypes; multi-split and longer rows also remain one-stage.
    """
    values = {"splits": splits, "pages_per_split": pages_per_split}
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise ValueError(f"conversion-stage inputs must be positive: {', '.join(invalid)}")
    if model_dtype not in (None, "fp16", "bf16"):
        raise ValueError(f"unsupported model_dtype: {model_dtype}")
    if splits == 1 and pages_per_split == 8:
        return 1
    if splits == 1 and 2 <= pages_per_split <= 12:
        return 3
    return 1


def select_direct_smem_store(
    *, batch: int, heads_kv: int, splits: int, pages_per_split: int
) -> bool:
    """Select the supported direct shared-memory epilogue on SM103.

    The epilogue bypasses the single-split register transpose by publishing O
    through the existing swizzled shared-memory tile.  Fixed-clock FP16/BF16
    The policy covers the one-page Hkv4 path and the four/eight-page Hkv8 B16
    path. Other shapes retain the register epilogue.
    """
    values = {
        "batch": batch,
        "heads_kv": heads_kv,
        "splits": splits,
        "pages_per_split": pages_per_split,
    }
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise ValueError(
            f"direct-store policy inputs must be positive: {', '.join(invalid)}"
        )
    if splits != 1:
        return False
    if heads_kv == 4 and pages_per_split == 1:
        return True
    return heads_kv == 8 and batch == 16 and pages_per_split in {4, 8}


def select_task_major_grid(
    *, batch: int, splits: int, pages_per_split: int
) -> bool:
    """Select the task-major CTA traversal on SM103.

    Task-major order is enabled only for the supported 16-split/eight-page
    long-context geometry at B1/Hkv8 and B2/Hkv4. Other geometries remain
    split-major, and unlisted batch counts fail closed.
    """
    values = {
        "batch": batch,
        "splits": splits,
        "pages_per_split": pages_per_split,
    }
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise ValueError(f"grid-policy inputs must be positive: {', '.join(invalid)}")
    return batch in {1, 2} and splits == 16 and pages_per_split == 8


def packed_kv_stage_cap(*, pages_per_split: int) -> int:
    """Choose packed K/V shared-memory depth after removing read-side fences.

    Fixed-clock GQA4 and GQA8 sweeps place the exact crossover between 32 and
    33 compute pages per split.  One stage avoids buffer-management overhead
    through page 32; two stages overlap TMA with conversion beyond that point.
    """
    if pages_per_split <= 0:
        raise ValueError("packed-KV stage input must be positive: pages_per_split")
    return 1 if pages_per_split <= 32 else 2


def select_decode_launch_plan(
    *,
    tiles: int,
    batch: int,
    heads_kv: int,
    sm_count: int,
    model_dtype: str,
) -> DecodeLaunchPlan:
    """Select the fastest retained B300 plan, with analytical fallback.

    Exact entries come only from five-fresh-process, 1000-replay FP16 and BF16
    retests. Shapes absent from the finite table use the resident-wave model
    and conservative stage/reducer helpers.
    """
    values = {
        "tiles": tiles,
        "batch": batch,
        "heads_kv": heads_kv,
        "sm_count": sm_count,
    }
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise ValueError(f"decode-plan inputs must be positive: {', '.join(invalid)}")
    if model_dtype not in {"fp16", "bf16"}:
        raise ValueError(f"unsupported model_dtype: {model_dtype}")

    tasks = batch * heads_kv
    key = (tasks, tiles)
    use_b300_table = sm_count == 148
    cluster_splits = (
        select_cluster_splits(tiles=tiles, batch=batch, heads_kv=heads_kv)
        if use_b300_table
        else 0
    )
    if cluster_splits:
        splits = cluster_splits
        cluster_reduction = True
    else:
        splits = (
            _B300_FIXED_SPLITS[key]
            if use_b300_table and key in _B300_FIXED_SPLITS
            else modeled_splits(
                tiles=tiles,
                batch=batch,
                heads_kv=heads_kv,
                sm_count=sm_count,
            )
        )
        cluster_reduction = False

    pages_per_split = scheduled_pages_per_split(tiles=tiles, splits=splits)
    reduction_kind = select_reduction_kind(
        batch=batch,
        splits=splits,
        pages_per_split=pages_per_split,
    )
    kv_stage_cap = packed_kv_stage_cap(pages_per_split=pages_per_split)
    cvt_stage_cap = conversion_stage_cap(
        splits=splits,
        pages_per_split=pages_per_split,
        model_dtype=model_dtype,
    )
    if use_b300_table:
        reduction_kind = _B300_REDUCER_OVERRIDES.get(key, reduction_kind)
        kv_stage_cap = _B300_KV_STAGE_OVERRIDES.get(key, kv_stage_cap)
        cvt_stage_cap = _B300_CVT_STAGE_OVERRIDES.get(key, cvt_stage_cap)

    direct_smem_store = (
        not cluster_reduction
        and (
            select_direct_smem_store(
                batch=batch,
                heads_kv=heads_kv,
                splits=splits,
                pages_per_split=pages_per_split,
            )
            or (use_b300_table and key in _B300_DIRECT_STORE_OVERRIDES)
        )
    )
    task_major_grid = (
        not cluster_reduction
        and select_task_major_grid(
            batch=batch,
            splits=splits,
            pages_per_split=pages_per_split,
        )
    )
    return DecodeLaunchPlan(
        splits=splits,
        pages_per_split=pages_per_split,
        reduction_kind=reduction_kind,
        kv_stage_cap=kv_stage_cap,
        cvt_stage_cap=cvt_stage_cap,
        cluster_reduction=cluster_reduction,
        task_major_grid=task_major_grid,
        direct_smem_store=direct_smem_store,
    )
