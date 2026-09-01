import pytest
import torch
from torch.utils.checkpoint import checkpoint

from beyond.quantization import layers as quant_layers_module
from beyond.quantization.layers import (
    CustomQuantFunction,
    NormalizedGroupQuantFunction,
    UnifiedQuantLayer,
    UniformAffineQuantLayer,
    clear_threshold_side_sums_,
    finalize_deferred_threshold_grads,
    threshold_side_activity_roster,
)


def _run_custom(parts, mode):
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    layer.bandwidth = 0.1
    x = torch.tensor([[[[0.45, 0.55]]], [[[0.45, 0.55]]]])
    upstream = torch.tensor([[[[-2.0, -3.0]]], [[[3.0, 4.0]]]])
    for start, stop in parts:
        out = CustomQuantFunction.apply(
            x[start:stop],
            layer.q_points,
            layer.thresholds,
            layer.bandwidth,
            layer,
        )
        (out * upstream[start:stop]).sum().backward()
    assert layer.thresholds.grad is None
    finalize_deferred_threshold_grads([layer], mode=mode)
    return layer.q_points.grad, layer.thresholds.grad


@pytest.mark.parametrize(
    ("mode", "expected_threshold"),
    [("half_wave", -1.0), ("raw", -2.0)],
)
def test_custom_quant_accumulates_before_combining(mode, expected_threshold):
    full_q, full_t = _run_custom([(0, 2)], mode)
    split_q, split_t = _run_custom([(0, 1), (1, 2)], mode)
    torch.testing.assert_close(full_q, torch.tensor([[1.0, 1.0]]), rtol=0, atol=0)
    torch.testing.assert_close(split_q, full_q, rtol=0, atol=0)
    torch.testing.assert_close(
        full_t,
        torch.tensor([[expected_threshold]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(split_t, full_t, rtol=0, atol=0)


def _run_shared(parts):
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        one_group=True,
        table_axis="shared",
    )
    layer.bandwidth = 0.1
    x = torch.tensor([[[0.45, 0.55]], [[0.45, 0.55]]])
    upstream = torch.tensor([[[-2.0, -3.0]], [[3.0, 4.0]]])
    for start, stop in parts:
        out = CustomQuantFunction.apply(
            x[start:stop],
            layer.q_points,
            layer.thresholds,
            layer.bandwidth,
            layer,
        )
        (out * upstream[start:stop]).sum().backward()
    finalize_deferred_threshold_grads([layer], mode="half_wave")
    return layer.thresholds.grad


def test_shared_quant_accumulates_before_combining():
    full = _run_shared([(0, 2)])
    split = _run_shared([(0, 1), (1, 2)])
    torch.testing.assert_close(full, torch.tensor([-1.0]), rtol=0, atol=0)
    torch.testing.assert_close(split, full, rtol=0, atol=0)


def _run_normalized(parts):
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=4,
        grouping_dim="token",
        quant_width=4,
    )
    layer.bandwidth = 0.1
    x = torch.tensor([[[0.0, 0.45, 0.55, 1.0]], [[0.0, 0.45, 0.55, 1.0]]])
    upstream = torch.tensor([[[0.0, -2.0, -3.0, 0.0]], [[0.0, 3.0, 4.0, 0.0]]])
    for start, stop in parts:
        out = NormalizedGroupQuantFunction.apply(
            x[start:stop],
            layer.q_points,
            layer.thresholds,
            layer.bandwidth,
            4,
            layer,
        )
        (out * upstream[start:stop]).sum().backward()
    finalize_deferred_threshold_grads([layer], mode="half_wave")
    return layer.thresholds.grad


def test_normalized_quant_accumulates_before_combining():
    full = _run_normalized([(0, 2)])
    split = _run_normalized([(0, 1), (1, 2)])
    torch.testing.assert_close(full, torch.tensor([[-1.0]]), rtol=0, atol=0)
    torch.testing.assert_close(split, full, rtol=0, atol=0)


def test_internal_side_outputs_use_packed_accumulation_branch():
    class PackedCountingLayer(UnifiedQuantLayer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.packed_accumulations = 0

        def accumulate_threshold_side_sums_packed_(self, side_sums):
            self.packed_accumulations += 1
            return super().accumulate_threshold_side_sums_packed_(side_sums)

    layer = PackedCountingLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    x = torch.tensor([[[[0.45, 0.55]]]])
    upstream = torch.tensor([[[[-2.0, -3.0]]]])
    out = CustomQuantFunction.apply(
        x,
        layer.q_points,
        layer.thresholds,
        0.1,
        layer,
    )
    (out * upstream).sum().backward()

    assert layer.packed_accumulations == 1
    left, right = layer.threshold_side_sums()
    torch.testing.assert_close(left, torch.tensor([[-2.0]]), rtol=0, atol=0)
    torch.testing.assert_close(right, torch.tensor([[-3.0]]), rtol=0, atol=0)


def test_packed_and_legacy_side_accumulation_have_microbatch_parity():
    packed_layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    legacy_layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    microbatches = (
        torch.tensor([[[-2.0]], [[3.0]]], dtype=torch.float64),
        torch.tensor([[[5.0]], [[-7.0]]], dtype=torch.float64),
    )
    for packed in microbatches:
        packed_layer.accumulate_threshold_side_sums_packed_(packed)
        # Clones force the backward-compatible, non-adjacent tensor branch.
        legacy_layer.accumulate_threshold_side_sums_(
            packed[0].clone(),
            packed[1].clone(),
        )

    assert packed_layer._threshold_side_sum_updates == 2
    assert legacy_layer._threshold_side_sum_updates == 2
    packed_sides = packed_layer.threshold_side_sums()
    legacy_sides = legacy_layer.threshold_side_sums()
    torch.testing.assert_close(packed_sides[0], legacy_sides[0], rtol=0, atol=0)
    torch.testing.assert_close(packed_sides[1], legacy_sides[1], rtol=0, atol=0)

    finalize_deferred_threshold_grads([packed_layer], mode="half_wave")
    finalize_deferred_threshold_grads([legacy_layer], mode="half_wave")
    torch.testing.assert_close(
        packed_layer.thresholds.grad,
        legacy_layer.thresholds.grad,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("half_wave", -0.5), ("raw", -1.0)],
)
def test_dp_average_happens_on_raw_sides(mode, expected):
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    layer.accumulate_threshold_side_sums_(
        torch.tensor([[-2.0]]),
        torch.tensor([[-3.0]]),
    )

    def fake_dp_average(packed):
        packed.add_(torch.tensor([[[3.0]], [[4.0]]])).div_(2.0)

    finalize_deferred_threshold_grads(
        [layer],
        mode=mode,
        data_reduce=fake_dp_average,
        include_all_enabled=True,
    )
    assert not layer.has_pending_threshold_side_sums()
    torch.testing.assert_close(
        layer.thresholds.grad,
        torch.tensor([[expected]]),
        rtol=0,
        atol=0,
    )


def test_activity_and_raw_reductions_precede_single_global_combine(monkeypatch):
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    layer.accumulate_threshold_side_sums_(
        torch.tensor([[-2.0]]),
        torch.tensor([[-3.0]]),
    )
    events = []

    original_combine = quant_layers_module.combine_threshold_side_grads

    def combine_spy(g_left, g_right, mode=None):
        events.append("combine")
        return original_combine(g_left, g_right, mode=mode)

    monkeypatch.setattr(
        quant_layers_module,
        "combine_threshold_side_grads",
        combine_spy,
    )

    def fake_activity_sum(activity):
        events.append("activity_sum")

    def fake_tensor_sum(packed):
        events.append("tensor_sum")
        packed.add_(torch.tensor([[[3.0]], [[4.0]]]))

    def fake_data_average(packed):
        events.append("data_average")
        packed.div_(2.0)

    finalize_deferred_threshold_grads(
        [layer],
        mode="half_wave",
        tensor_reduce=fake_tensor_sum,
        data_reduce=fake_data_average,
        activity_reduce=fake_activity_sum,
        include_all_enabled=True,
    )
    assert events == ["activity_sum", "tensor_sum", "data_average", "combine"]
    torch.testing.assert_close(
        layer.thresholds.grad,
        torch.tensor([[-0.5]]),
        rtol=0,
        atol=0,
    )


def test_enabled_zero_contributor_keeps_roster_without_rectification():
    contributing = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    zero_contributor = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    contributing.accumulate_threshold_side_sums_(
        torch.tensor([[-2.0]]),
        torch.tensor([[-3.0]]),
    )
    packed_shapes = []

    def record_bucket(packed):
        packed_shapes.append(tuple(packed.shape))

    finalized = finalize_deferred_threshold_grads(
        [contributing, zero_contributor],
        mode="half_wave",
        data_reduce=record_bucket,
        include_all_enabled=True,
    )
    assert finalized == 1
    assert packed_shapes == [(4, 1, 1)]
    torch.testing.assert_close(
        contributing.thresholds.grad,
        torch.tensor([[2.0]]),
        rtol=0,
        atol=0,
    )
    assert zero_contributor.thresholds.grad is None


def test_remote_only_activity_rectifies_once_on_every_replica():
    local_idle = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    remote_active = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )

    def fake_activity_sum(activity):
        activity[1] = 1

    def fake_side_average(packed):
        packed[2].fill_(-2.0)
        packed[3].fill_(-3.0)

    finalized = finalize_deferred_threshold_grads(
        [local_idle, remote_active],
        mode="half_wave",
        data_reduce=fake_side_average,
        activity_reduce=fake_activity_sum,
        include_all_enabled=True,
    )
    assert finalized == 1
    assert local_idle.thresholds.grad is None
    torch.testing.assert_close(
        remote_active.thresholds.grad,
        torch.tensor([[2.0]]),
        rtol=0,
        atol=0,
    )


def test_fixed_activity_roster_deduplicates_and_keeps_enabled_idle_modules():
    active = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    idle = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    frozen = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    frozen.thresholds.requires_grad_(False)
    active.accumulate_threshold_side_sums_(
        torch.tensor([[-1.0]]),
        torch.tensor([[0.0]]),
    )

    active_only = threshold_side_activity_roster([active, active, idle, frozen])
    assert [(entry[0], entry[3]) for entry in active_only] == [(active, True)]

    fixed = threshold_side_activity_roster(
        [active, active, idle, frozen],
        include_all_enabled=True,
    )
    assert [(entry[0], entry[3]) for entry in fixed] == [
        (active, True),
        (idle, False),
    ]
    assert all(tuple(left.shape) == (1, 1) for _module, left, _right, _pending in fixed)


def test_prereduced_global_activity_skips_activity_reducer():
    local_idle = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    remote_active = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )

    def fake_side_average(packed):
        packed[2].fill_(-2.0)
        packed[3].fill_(-3.0)

    finalized = finalize_deferred_threshold_grads(
        [local_idle, remote_active],
        mode="half_wave",
        data_reduce=fake_side_average,
        global_activity={id(local_idle): False, id(remote_active): True},
        include_all_enabled=True,
    )
    assert finalized == 1
    assert local_idle.thresholds.grad is None
    torch.testing.assert_close(
        remote_active.thresholds.grad,
        torch.tensor([[2.0]]),
        rtol=0,
        atol=0,
    )


def test_prereduced_global_activity_is_complete_and_mutually_exclusive():
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    layer.accumulate_threshold_side_sums_(
        torch.tensor([[-1.0]]),
        torch.tensor([[0.0]]),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        finalize_deferred_threshold_grads(
            [layer],
            activity_reduce=lambda activity: activity,
            global_activity={id(layer): True},
            include_all_enabled=True,
        )
    with pytest.raises(ValueError, match="missing deferred-side module ids"):
        finalize_deferred_threshold_grads(
            [layer],
            global_activity={},
            include_all_enabled=True,
        )
    assert layer.has_pending_threshold_side_sums()


@pytest.mark.parametrize("mode", ["half_wave", "raw"])
def test_one_rectification_per_active_module_per_optimizer_window(monkeypatch, mode):
    active = UnifiedQuantLayer(
        num_bits=1,
        group_size=4,
        grouping_dim="token",
        quant_width=4,
    )
    idle = UnifiedQuantLayer(
        num_bits=1,
        group_size=4,
        grouping_dim="token",
        quant_width=4,
    )
    active.bandwidth = 0.1
    calls = []
    original_combine = quant_layers_module.combine_threshold_side_grads

    def combine_spy(g_left, g_right, mode=None):
        calls.append(mode)
        return original_combine(g_left, g_right, mode=mode)

    monkeypatch.setattr(
        quant_layers_module,
        "combine_threshold_side_grads",
        combine_spy,
    )

    def backward_once(use_checkpoint=None):
        x = torch.tensor(
            [[[0.0, 0.45, 0.55, 1.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        if use_checkpoint is None:
            out = active(x)
        else:
            out = checkpoint(active, x, use_reentrant=use_checkpoint)
        upstream = torch.tensor([[[0.0, -2.0, -3.0, 0.0]]])
        (out * upstream).sum().backward()

    backward_once()
    backward_once(use_checkpoint=False)
    backward_once(use_checkpoint=True)
    assert active._threshold_side_sum_updates == 3
    assert calls == []

    finalized = finalize_deferred_threshold_grads(
        [active, active, idle],
        mode=mode,
        include_all_enabled=True,
    )
    assert finalized == 1
    assert calls == [mode]
    assert idle.thresholds.grad is None

    # Re-entering the boundary without a new backward is idempotent even when
    # the fixed distributed roster includes every enabled module.
    finalized = finalize_deferred_threshold_grads(
        [active, idle],
        mode=mode,
        include_all_enabled=True,
    )
    assert finalized == 0
    assert calls == [mode]

    active.zero_grad(set_to_none=True)
    backward_once()
    finalized = finalize_deferred_threshold_grads(
        [active, idle],
        mode=mode,
        include_all_enabled=True,
    )
    assert finalized == 1
    assert calls == [mode, mode]


def test_pending_zero_side_sum_is_rectified_once_but_idle_is_not():
    pending_zero = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    idle = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    pending_zero.accumulate_threshold_side_sums_(
        torch.zeros_like(pending_zero.thresholds),
        torch.zeros_like(pending_zero.thresholds),
    )
    assert (
        finalize_deferred_threshold_grads(
            [pending_zero, idle],
            mode="half_wave",
            include_all_enabled=True,
        )
        == 1
    )
    torch.testing.assert_close(
        pending_zero.thresholds.grad,
        torch.zeros_like(pending_zero.thresholds),
        rtol=0,
        atol=0,
    )
    assert idle.thresholds.grad is None


def test_idle_threshold_keeps_adam_state_and_parameter_unchanged():
    active = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    idle = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    optimizer = torch.optim.Adam([active.thresholds, idle.thresholds], lr=0.01)

    for layer in (active, idle):
        layer.accumulate_threshold_side_sums_(
            torch.tensor([[-1.0]]),
            torch.tensor([[0.0]]),
        )
    assert (
        finalize_deferred_threshold_grads(
            [active, idle],
            mode="half_wave",
            include_all_enabled=True,
        )
        == 2
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    idle_value = idle.thresholds.detach().clone()
    idle_step = optimizer.state[idle.thresholds]["step"].detach().clone()
    active.accumulate_threshold_side_sums_(
        torch.tensor([[-1.0]]),
        torch.tensor([[0.0]]),
    )
    assert (
        finalize_deferred_threshold_grads(
            [active, idle],
            mode="half_wave",
            include_all_enabled=True,
        )
        == 1
    )
    assert idle.thresholds.grad is None
    optimizer.step()

    torch.testing.assert_close(idle.thresholds, idle_value, rtol=0, atol=0)
    torch.testing.assert_close(
        optimizer.state[idle.thresholds]["step"],
        idle_step,
        rtol=0,
        atol=0,
    )


def test_shared_deferred_update_parameter_fails_before_double_rectification():
    first = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    second = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    second.thresholds = first.thresholds
    first.accumulate_threshold_side_sums_(
        torch.tensor([[-1.0]]),
        torch.tensor([[0.0]]),
    )

    with pytest.raises(RuntimeError, match="one quantizer owner"):
        finalize_deferred_threshold_grads([first, second], mode="half_wave")


def test_partial_accumulation_scale_applies_after_global_half_wave():
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    layer.bandwidth = 0.1
    x = torch.tensor([[[[0.45, 0.55]]], [[[0.45, 0.55]]]])
    upstream = torch.tensor([[[[-2.0, -3.0]]], [[[3.0, 4.0]]]]) / 4.0
    for index in range(2):
        out = CustomQuantFunction.apply(
            x[index : index + 1],
            layer.q_points,
            layer.thresholds,
            layer.bandwidth,
            layer,
        )
        (out * upstream[index : index + 1]).sum().backward()
    finalize_deferred_threshold_grads([layer], mode="half_wave")
    layer.thresholds.grad.mul_(2.0)
    torch.testing.assert_close(
        layer.thresholds.grad,
        torch.tensor([[-0.5]]),
        rtol=0,
        atol=0,
    )


def test_threshold_only_optimizer_receives_deferred_gradient():
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    layer.bandwidth = 0.1
    layer.q_points.requires_grad_(False)
    optimizer = torch.optim.SGD([layer.thresholds], lr=0.1)
    x = torch.tensor([[[[0.45, 0.55]]]])
    upstream = torch.tensor([[[[-2.0, -3.0]]]])
    out = CustomQuantFunction.apply(
        x,
        layer.q_points,
        layer.thresholds,
        layer.bandwidth,
        layer,
    )
    (out * upstream).sum().backward()
    assert layer.thresholds.grad is None
    assert layer.has_pending_threshold_side_sums()
    finalize_deferred_threshold_grads([layer], mode="half_wave")
    optimizer.step()
    torch.testing.assert_close(
        layer.thresholds,
        torch.tensor([[0.3]]),
        rtol=0,
        atol=1e-7,
    )


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_checkpoint_recompute_accumulates_side_sums_once(use_reentrant):
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=4,
        grouping_dim="token",
        quant_width=4,
    )
    x = torch.tensor(
        [[[0.0, 0.45, 0.55, 1.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    out = checkpoint(layer, x, use_reentrant=use_reentrant)
    out.sum().backward()
    assert layer._threshold_side_sum_updates == 1
    finalize_deferred_threshold_grads([layer], mode="half_wave")
    assert not layer.has_pending_threshold_side_sums()


def test_side_state_reset_freeze_and_state_dict_contract():
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    layer.accumulate_threshold_side_sums_(
        torch.tensor([[-1.0]]),
        torch.tensor([[0.0]]),
    )
    clear_threshold_side_sums_([layer])
    left, right = layer.threshold_side_sums(include_zeros=True)
    assert not layer.has_pending_threshold_side_sums()
    assert torch.count_nonzero(left).item() == 0
    assert torch.count_nonzero(right).item() == 0
    assert all("threshold_side_sums" not in key for key in layer.state_dict())

    layer.thresholds.requires_grad_(False)
    x = torch.tensor([[[[0.45, 0.55]]]])
    CustomQuantFunction.apply(
        x,
        layer.q_points,
        layer.thresholds,
        0.1,
        layer,
    ).sum().backward()
    assert not layer.has_pending_threshold_side_sums()
    assert layer.thresholds.grad is None


def test_uniform_affine_receives_deferred_threshold_contribution():
    layer = UniformAffineQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    layer.accumulate_threshold_side_sums_(
        torch.tensor([[-2.0]]),
        torch.tensor([[-3.0]]),
    )
    finalize_deferred_threshold_grads([layer], mode="half_wave")
    torch.testing.assert_close(layer.affine_low.grad, torch.tensor([1.0]), rtol=0, atol=0)
    torch.testing.assert_close(layer.affine_high.grad, torch.tensor([1.0]), rtol=0, atol=0)
