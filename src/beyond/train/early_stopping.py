"""Pure validation early-stopping state transitions.

The trainer stops after ``patience`` consecutive checks whose perplexity is
strictly worse than the best finite perplexity seen so far. Equality and invalid
values reset the worse-than-best streak.

Keeping this transition free of torch/model state makes the exact comparison
contract cheap to unit-test and reuse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationEarlyStopDecision:
    """Result of incorporating one validation perplexity observation."""

    comparison: str
    best_value: float | None
    worse_streak: int
    is_new_best: bool
    should_stop: bool


def update_validation_early_stopping(
    *,
    current_value: float | None,
    best_value: float | None,
    worse_streak: int,
    patience: int,
) -> ValidationEarlyStopDecision:
    """Apply one strict worse-than-best early-stopping transition.

    ``patience <= 0`` disables stopping but does not disable best-value and
    streak bookkeeping.  A non-finite current value is never treated as
    evidence of degradation because it cannot be ordered reliably.
    """

    if isinstance(worse_streak, bool) or int(worse_streak) < 0:
        raise ValueError("worse_streak must be a non-negative integer")
    if isinstance(patience, bool):
        raise ValueError("patience must be an integer")
    worse_streak = int(worse_streak)
    patience = int(patience)

    current_is_finite = (
        current_value is not None
        and not isinstance(current_value, bool)
        and math.isfinite(float(current_value))
    )
    best_is_finite = (
        best_value is not None
        and not isinstance(best_value, bool)
        and math.isfinite(float(best_value))
    )
    if not current_is_finite:
        return ValidationEarlyStopDecision(
            comparison="invalid",
            best_value=float(best_value) if best_is_finite else None,
            worse_streak=0,
            is_new_best=False,
            should_stop=False,
        )

    current = float(current_value)
    if not best_is_finite or current < float(best_value):
        return ValidationEarlyStopDecision(
            comparison="initial" if not best_is_finite else "improved",
            best_value=current,
            worse_streak=0,
            is_new_best=True,
            should_stop=False,
        )

    if current > float(best_value):
        updated_streak = worse_streak + 1
        return ValidationEarlyStopDecision(
            comparison="worse_than_best",
            best_value=float(best_value),
            worse_streak=updated_streak,
            is_new_best=False,
            should_stop=patience > 0 and updated_streak >= patience,
        )

    return ValidationEarlyStopDecision(
        comparison="equal_to_best",
        best_value=float(best_value),
        worse_streak=0,
        is_new_best=False,
        should_stop=False,
    )
