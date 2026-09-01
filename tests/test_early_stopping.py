import math

import pytest

from beyond.train.early_stopping import update_validation_early_stopping


def _run(values, *, patience=3):
    best = None
    streak = 0
    decisions = []
    for value in values:
        decision = update_validation_early_stopping(
            current_value=value,
            best_value=best,
            worse_streak=streak,
            patience=patience,
        )
        decisions.append(decision)
        best = decision.best_value
        streak = decision.worse_streak
    return decisions


def test_three_consecutive_checks_worse_than_global_best_trigger_stop():
    decisions = _run([5.0, 4.0, 4.1, 4.2, 4.15])

    assert [item.comparison for item in decisions] == [
        "initial",
        "improved",
        "worse_than_best",
        "worse_than_best",
        "worse_than_best",
    ]
    assert [item.worse_streak for item in decisions] == [0, 0, 1, 2, 3]
    assert decisions[-1].should_stop is True
    # The final value improved over the immediately preceding value.  The
    # stopping is nevertheless triggered because all three checks are worse
    # than the all-time best.
    assert 4.15 < 4.2


def test_improvement_equality_and_invalid_values_reset_the_streak():
    decisions = _run([5.0, 5.1, 5.0, 5.2, math.nan, 4.9, 5.0])

    assert [item.comparison for item in decisions] == [
        "initial",
        "worse_than_best",
        "equal_to_best",
        "worse_than_best",
        "invalid",
        "improved",
        "worse_than_best",
    ]
    assert [item.worse_streak for item in decisions] == [0, 1, 0, 1, 0, 0, 1]
    assert not any(item.should_stop for item in decisions)


def test_nonpositive_patience_disables_stopping_but_keeps_tracking():
    decisions = _run([1.0, 2.0, 3.0, 4.0], patience=0)

    assert decisions[-1].worse_streak == 3
    assert decisions[-1].should_stop is False


@pytest.mark.parametrize("streak", [-1, True])
def test_invalid_streak_is_rejected(streak):
    with pytest.raises(ValueError, match="worse_streak"):
        update_validation_early_stopping(
            current_value=1.0,
            best_value=None,
            worse_streak=streak,
            patience=3,
        )
