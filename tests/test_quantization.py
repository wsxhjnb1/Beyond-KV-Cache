import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from beyond.models.model_utils import (
    DISCOVER_CHAT_WRAP_PILE_DEFAULT,
    DISCOVER_CHAT_WRAP_PILE_PROMPT,
    DISCOVER_DATA_SEED,
    DISCOVER_DATASET_CHOICES,
    DISCOVER_EVAL_CHUNKS,
    DISCOVER_EVAL_SEED,
    DISCOVER_EXPERTS_IMPLEMENTATION_CHOICES,
    DISCOVER_VALIDATION_CHUNKS,
    SUPPORTED_DISCOVER_MODELS,
    _chat_generation_prefix,
    _encode_ids,
    _heldout_stage_specs_from_curriculum,
    _sample_to_segments,
    _source_targets_for_token_budget,
    _truncate_to_supervised_budget,
    canonical_model_id,
    detect_model_type,
    discover_default_chat_wrap_pile,
    discover_split_seeds,
    get_discover_sources,
    get_model_config,
    load_tokenizer,
    validate_discover_megatron_parallel_request,
)
from beyond.quantization.layers import (
    CustomQuantFunction,
    FullPrecisionBucketResidualAdapter,
    FullPrecisionChebyshevAdapter,
    MegatronQuantizedQKVLinear,
    NormalizedGroupQuantFunction,
    QuantizedLinear,
    QuantizedQKVLinear,
    UnifiedQuantLayer,
    UniformAffineQuantLayer,
    _tls,
    clear_threshold_side_sums_,
    combine_threshold_side_grads,
    finalize_deferred_threshold_grads,
    install_post_rope_k_quantization,
    quantize_post_rope_k_tensor,
)
from beyond.quantization.table_precision import TABLE_PRECISION_ABI
from beyond.quantization.training import (
    apply_quant_config,
    build_kv_quant_targets,
    extract_quant_config,
    find_kv_proj_layers,
    project_quant_points,
    resolve_experiment_control,
    save_quant_config,
    set_quant_params,
)


class _MegatronProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = _MegatronProjection()
        self.v_proj = _MegatronProjection()


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _Model(nn.Module):
    def __init__(self, num_hidden_layers=2):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=num_hidden_layers)
        self.layers = nn.ModuleList([_Layer() for _ in range(num_hidden_layers)])


class _ChatTemplateTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert messages[0] == {"role": "user", "content": DISCOVER_CHAT_WRAP_PILE_PROMPT}
        if len(messages) == 2:
            assert add_generation_prompt is False
            assert messages[1]["role"] == "assistant"
            return f"<user>{messages[0]['content']}</user><assistant><final>{messages[1]['content']}</final>"
        assert messages == [{"role": "user", "content": DISCOVER_CHAT_WRAP_PILE_PROMPT}]
        suffix = "<assistant>" if add_generation_prompt else ""
        return f"<user>{messages[0]['content']}</user>{suffix}"


class _NoChatTemplateTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("chat_template is not set")


class _QwenOfficialChatTemplateTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        rendered = ""
        for message in messages:
            rendered += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


def test_find_kv_proj_layers_ignores_megatron_wrapped_linear_children():
    model = _Model(num_hidden_layers=2)

    specs = find_kv_proj_layers(model)

    assert [spec["name"] for spec in specs] == [
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
        "layers.1.self_attn.k_proj",
        "layers.1.self_attn.v_proj",
    ]
    assert len(specs) == model.config.num_hidden_layers * 2
    assert not any(spec["name"].endswith(".linear") for spec in specs)
    assert [spec["grouping_dim"] for spec in specs] == [
        "token",
        "token",
        "token",
        "token",
    ]


def test_find_kv_proj_layers_returns_two_targets_for_leaf_fused_qkv_only():
    model = nn.Module()
    model.block = nn.Module()
    model.block.query_key_value = _MegatronProjection()

    specs = find_kv_proj_layers(model)

    assert specs == [
        {
            "name": "block.query_key_value",
            "proj_type": "query_key_value",
            "grouping_dim": "token",
            "kv_slice": "k",
        },
        {
            "name": "block.query_key_value",
            "proj_type": "query_key_value",
            "grouping_dim": "token",
            "kv_slice": "v",
        },
    ]


def test_find_kv_proj_layers_returns_two_targets_for_fused_qkv_proj():
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].self_attn = nn.Module()
    model.layers[0].self_attn.qkv_proj = _MegatronProjection()

    specs = find_kv_proj_layers(model)

    assert specs == [
        {
            "name": "layers.0.self_attn.qkv_proj",
            "proj_type": "qkv_proj",
            "grouping_dim": "token",
            "kv_slice": "k",
        },
        {
            "name": "layers.0.self_attn.qkv_proj",
            "proj_type": "qkv_proj",
            "grouping_dim": "token",
            "kv_slice": "v",
        },
    ]


def test_find_kv_proj_layers_skips_vision_tower_by_default():
    model = nn.Module()
    model.vision_tower = nn.Module()
    model.vision_tower.encoder = nn.ModuleList([_Layer()])
    model.language_model = _Model(num_hidden_layers=1)

    specs = find_kv_proj_layers(model)
    names = [spec["name"] for spec in specs]

    assert names == [
        "language_model.layers.0.self_attn.k_proj",
        "language_model.layers.0.self_attn.v_proj",
    ]

    specs_with_vision = find_kv_proj_layers(model, include_vision=True)
    assert any(spec["name"].startswith("vision_tower.") for spec in specs_with_vision)


def test_project_quant_points_sorts_in_place_without_bounding_values():
    q_points = torch.tensor([1.2, 0.4, -0.2, 0.4], dtype=torch.float32)

    project_quant_points(q_points, min_gap=1e-4)

    assert q_points.min().item() < 0.0
    assert q_points.max().item() > 1.0
    assert torch.all(q_points[1:] - q_points[:-1] >= 1e-4 - 1e-7)


def test_project_quant_points_sorts_each_table_row():
    q_points = torch.tensor(
        [[0.4, 0.0, 0.2], [1.1, -0.3, 0.5]],
        dtype=torch.float32,
    )

    project_quant_points(q_points, min_gap=1e-4)

    assert torch.all(q_points[:, 1:] - q_points[:, :-1] >= 1e-4 - 1e-7)
    assert q_points[0, 0].item() == pytest.approx(0.0)
    assert q_points[1, 0].item() < 0.0


def test_token_grouped_quant_layer_uses_group_tables_by_default():
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=6,
    )
    assert layer.table_axis == "group"
    with torch.no_grad():
        layer.q_points.copy_(
            torch.tensor(
                [
                    [0.0, 1.0],
                    [0.25, 0.75],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )
        layer.thresholds.fill_(0.5)
    x = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 10.0, 11.0]]])

    out = layer(x)

    assert torch.equal(
        out,
        torch.tensor([[[0.0, 1.0, 2.25, 2.75, 10.0, 11.0]]]),
    )


def test_token_grouped_quant_layer_can_use_one_table_per_group():
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        table_axis="group",
        quant_width=4,
    )
    with torch.no_grad():
        layer.q_points.copy_(
            torch.tensor(
                [
                    [0.0, 1.0],
                    [0.25, 0.75],
                ],
                dtype=torch.float32,
            )
        )
        layer.thresholds.fill_(0.5)
    x = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])

    out = layer(x)

    assert torch.equal(
        out,
        torch.tensor([[[0.0, 1.0, 2.25, 2.75]]]),
    )


@pytest.mark.parametrize("num_bits", [2, 3])
def test_token_grouped_quant_layer_supports_low_bit_group_tables(num_bits):
    layer = UnifiedQuantLayer(
        num_bits=num_bits,
        group_size=2,
        grouping_dim="token",
        table_axis="group",
        quant_width=4,
    )

    assert tuple(layer.q_points.shape) == (2, 1 << num_bits)
    assert tuple(layer.thresholds.shape) == (2, (1 << num_bits) - 1)


def test_uniform_affine_quantizer_has_exactly_two_parameters_per_table():
    layer = UniformAffineQuantLayer(
        num_bits=4,
        group_size=2,
        grouping_dim="token",
        table_axis="group",
        quant_width=6,
    )

    assert dict(layer.named_parameters()).keys() == {"affine_low", "affine_high"}
    assert sum(parameter.numel() for parameter in layer.parameters()) == 6
    q_points, thresholds = layer.materialize_quant_tables()
    master_q_points = (
        layer.affine_low.unsqueeze(-1)
        + (layer.affine_high - layer.affine_low).unsqueeze(-1) * layer.uniform_fractions
    )
    master_thresholds = (master_q_points[:, :-1] + master_q_points[:, 1:]) * 0.5
    assert torch.equal(q_points, master_q_points.to(torch.float16).to(torch.float32))
    assert torch.equal(thresholds, master_thresholds.to(torch.float16).to(torch.float32))


def test_uniform_affine_quantizer_projects_endpoints_and_never_learns_thresholds():
    layer = UniformAffineQuantLayer(
        num_bits=2,
        group_size=4,
        grouping_dim="token",
        quant_width=4,
    )
    with torch.no_grad():
        layer.affine_low.fill_(1.4)
        layer.affine_high.fill_(-0.2)

    layer.project_parameters_()
    q_points, thresholds = layer.materialize_quant_tables()

    assert 0.0 <= layer.affine_low.item() < layer.affine_high.item() <= 1.0
    master_q_points = (
        layer.affine_low.unsqueeze(-1)
        + (layer.affine_high - layer.affine_low).unsqueeze(-1) * layer.uniform_fractions
    )
    master_thresholds = (master_q_points[..., :-1] + master_q_points[..., 1:]) * 0.5
    assert torch.equal(q_points, master_q_points.to(torch.float16).to(torch.float32))
    assert torch.equal(thresholds, master_thresholds.to(torch.float16).to(torch.float32))


def test_uniform_affine_hard_forward_backpropagates_only_to_endpoints():
    layer = UniformAffineQuantLayer(
        num_bits=2,
        group_size=4,
        grouping_dim="token",
        quant_width=4,
    )
    x = torch.tensor([[[0.0, 0.2, 0.7, 1.0]]], dtype=torch.float32)

    layer(x).square().mean().backward()

    assert layer.affine_low.grad is not None
    assert layer.affine_high.grad is not None
    assert {name for name, _parameter in layer.named_parameters()} == {
        "affine_low",
        "affine_high",
    }


def test_uniform_affine_quantizer_refuses_nonuniform_resume_table():
    layer = UniformAffineQuantLayer(
        num_bits=2,
        group_size=4,
        grouping_dim="token",
        quant_width=4,
    )
    nonuniform = torch.tensor([[0.0, 0.1, 0.7, 1.0]])

    with pytest.raises(ValueError, match="refuses non-uniform"):
        set_quant_params(layer, nonuniform)


def test_reconstruction_capture_uses_hard_qdq_and_valid_token_mask():
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=4,
        grouping_dim="token",
        quant_width=4,
    )
    x = torch.tensor(
        [[[0.0, 0.2, 0.8, 1.0], [0.0, 0.4, 0.6, 1.0]]],
        dtype=torch.float32,
    )
    layer.enable_reconstruction_capture(token_mask=torch.tensor([[1, 0]]))

    out = layer(x)
    sse, elements = layer.consume_reconstruction_stats()
    layer.disable_reconstruction_capture()

    assert out.requires_grad is False
    assert sse.item() == pytest.approx(0.08, abs=1e-6)
    assert elements.item() == pytest.approx(4.0)
    (sse / elements).backward()
    finalize_deferred_threshold_grads([layer], mode="half_wave")
    assert layer.q_points.grad is not None
    assert layer.thresholds.grad is not None


def _run_custom_threshold_partition(partitions, mode="half_wave"):
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    layer.bandwidth = 0.1
    x = torch.tensor(
        [
            [[[0.45, 0.55]]],
            [[[0.45, 0.55]]],
        ],
        dtype=torch.float32,
    )
    upstream = torch.tensor(
        [
            [[[-2.0, -3.0]]],
            [[[3.0, 4.0]]],
        ],
        dtype=torch.float32,
    )
    for start, stop in partitions:
        out = CustomQuantFunction.apply(
            x[start:stop],
            layer.q_points,
            layer.thresholds,
            layer.bandwidth,
            layer,
        )
        (out * upstream[start:stop]).sum().backward()
    assert layer.thresholds.grad is None
    assert layer.has_pending_threshold_side_sums()
    finalize_deferred_threshold_grads([layer], mode=mode)
    return layer.q_points.grad.detach().clone(), layer.thresholds.grad.detach().clone()


@pytest.mark.parametrize(
    ("mode", "expected_threshold_grad"),
    [("half_wave", -1.0), ("raw", -2.0)],
)
def test_threshold_gradient_is_microbatch_partition_invariant(mode, expected_threshold_grad):
    full_q, full_t = _run_custom_threshold_partition([(0, 2)], mode=mode)
    split_q, split_t = _run_custom_threshold_partition([(0, 1), (1, 2)], mode=mode)

    torch.testing.assert_close(full_q, torch.tensor([[1.0, 1.0]]), rtol=0, atol=0)
    torch.testing.assert_close(split_q, full_q, rtol=0, atol=0)
    torch.testing.assert_close(
        full_t,
        torch.tensor([[expected_threshold_grad]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(split_t, full_t, rtol=0, atol=0)


def _run_normalized_threshold_partition(partitions):
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=4,
        grouping_dim="token",
        quant_width=4,
    )
    layer.bandwidth = 0.1
    x = torch.tensor(
        [
            [[0.0, 0.45, 0.55, 1.0]],
            [[0.0, 0.45, 0.55, 1.0]],
        ],
        dtype=torch.float32,
    )
    upstream = torch.tensor(
        [
            [[0.0, -2.0, -3.0, 0.0]],
            [[0.0, 3.0, 4.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    for start, stop in partitions:
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
    return layer.thresholds.grad.detach().clone()


def test_normalized_threshold_gradient_is_microbatch_partition_invariant():
    full = _run_normalized_threshold_partition([(0, 2)])
    split = _run_normalized_threshold_partition([(0, 1), (1, 2)])
    torch.testing.assert_close(full, torch.tensor([[-1.0]]), rtol=0, atol=0)
    torch.testing.assert_close(split, full, rtol=0, atol=0)


def test_threshold_sides_are_dp_averaged_before_half_wave():
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

    def fake_dp_average(packed):
        events.append("dp_average_raw_sides")
        remote_sides = torch.zeros_like(packed)
        remote_sides[0].fill_(3.0)
        remote_sides[1].fill_(4.0)
        packed.add_(remote_sides).div_(2.0)

    finalize_deferred_threshold_grads(
        [layer],
        mode="half_wave",
        data_reduce=fake_dp_average,
        include_all_enabled=True,
    )

    assert events == ["dp_average_raw_sides"]
    torch.testing.assert_close(
        layer.thresholds.grad,
        torch.tensor([[-0.5]]),
        rtol=0,
        atol=0,
    )


def test_threshold_side_sums_clear_between_optimizer_windows():
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    layer.accumulate_threshold_side_sums_(torch.tensor([[-1.0]]), torch.tensor([[0.0]]))
    clear_threshold_side_sums_([layer])
    assert not layer.has_pending_threshold_side_sums()
    left, right = layer.threshold_side_sums(include_zeros=True)
    assert torch.count_nonzero(left).item() == 0
    assert torch.count_nonzero(right).item() == 0


def test_frozen_thresholds_do_not_accumulate_deferred_side_sums():
    layer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=2,
    )
    layer.thresholds.requires_grad_(False)
    x = torch.tensor([[[[0.45, 0.55]]]], dtype=torch.float32)
    out = CustomQuantFunction.apply(
        x,
        layer.q_points,
        layer.thresholds,
        0.1,
        layer,
    )
    out.sum().backward()

    assert not layer.has_pending_threshold_side_sums()
    assert layer.thresholds.grad is None
    assert finalize_deferred_threshold_grads([layer], mode="half_wave") == 0


def test_uniform_affine_routes_deferred_threshold_gradient_to_endpoints():
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
    assert all("threshold_side_sums" not in key for key in layer.state_dict())


def test_threshold_side_combination_reads_mode_at_finalize_time(monkeypatch):
    left = torch.tensor([-1.0, 2.0])
    right = torch.tensor([-3.0, 4.0])
    monkeypatch.setenv("DISCOVER_THRESHOLD_GRAD_MODE", "raw")
    torch.testing.assert_close(
        combine_threshold_side_grads(left, right),
        torch.tensor([-4.0, 6.0]),
    )


def test_control_resolution_is_explicit_and_fail_closed():
    uniform = resolve_experiment_control("uniform_affine_nll")
    reconstruction = resolve_experiment_control("nonuniform_reconstruction")
    uncompressed = resolve_experiment_control("uncompressed_matched_nll")
    bucket_residual = resolve_experiment_control("uncompressed_bucket_residual_nll")

    assert uniform["quantizer_parameterization"] == "uniform_affine_endpoints"
    assert uniform["training_objective"] == "causal_lm_nll"
    assert reconstruction["training_objective"] == "kv_reconstruction_mse"
    assert uncompressed["quantizer_parameterization"] == "full_precision_chebyshev_residual"
    assert uncompressed["compressed_kv_cache"] is False
    assert bucket_residual["quantizer_parameterization"] == "full_precision_bucket_residual"
    assert bucket_residual["compressed_kv_cache"] is False
    assert bucket_residual["hard_bucket_assignment"] is True
    with pytest.raises(ValueError, match="fail-closed"):
        resolve_experiment_control("fisher_reconstruction")
    with pytest.raises(ValueError, match="fail-closed"):
        resolve_experiment_control("uncompressed_affine_nll")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_full_precision_matched_adapter_is_exact_identity_at_initialization(dtype):
    layer = FullPrecisionChebyshevAdapter(
        num_bits=4,
        group_size=4,
        grouping_dim="token",
        table_axis="group",
        quant_width=8,
    )
    x = torch.tensor(
        [[[0.0, -0.25, 0.75, 1.0, -3.0, -1.0, 2.0, 5.0]]],
        dtype=dtype,
    )

    output = layer(x)

    assert torch.equal(output, x)
    assert sum(parameter.numel() for parameter in layer.parameters()) == 2 * 31
    assert dict(layer.named_parameters()).keys() == {"coefficients"}


def test_full_precision_matched_adapter_is_continuous_bounded_and_trainable():
    layer = FullPrecisionChebyshevAdapter(
        num_bits=2,
        group_size=4,
        grouping_dim="token",
        table_axis="group",
        quant_width=4,
    )
    with torch.no_grad():
        layer.coefficients[:, 1].fill_(0.75)
    x = torch.tensor([[[0.0, 0.2, 0.7, 1.0]]], dtype=torch.float32)

    output = layer(x)
    output.square().mean().backward()

    assert not torch.equal(output, x)
    assert output.min().item() >= x.min().item()
    assert output.max().item() <= x.max().item()
    assert layer.coefficients.grad is not None
    assert torch.isfinite(layer.coefficients.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bucket_residual_adapter_is_exact_identity_and_parameter_matched(dtype):
    layer = FullPrecisionBucketResidualAdapter(
        num_bits=4,
        group_size=4,
        grouping_dim="token",
        table_axis="group",
        quant_width=8,
    )
    x = torch.tensor(
        [[[0.0, -0.25, 0.75, 1.0, -3.0, -1.0, 2.0, 5.0]]],
        dtype=dtype,
    )

    output = layer(x)

    assert torch.equal(output, x)
    assert sum(parameter.numel() for parameter in layer.parameters()) == 2 * 31
    assert dict(layer.named_parameters()).keys() == {"thresholds", "bucket_offsets"}
    assert torch.count_nonzero(layer.bucket_offsets).item() == 0


def test_bucket_residual_adapter_routes_offsets_without_double_input_gradient():
    layer = FullPrecisionBucketResidualAdapter(
        num_bits=2,
        group_size=4,
        grouping_dim="token",
        table_axis="group",
        quant_width=4,
    )
    layer.set_bucket_residual_state(
        torch.tensor([[0.1, -0.2, 0.3, -0.4]]),
        torch.tensor([[0.25, 0.50, 0.75]]),
    )
    layer.thresholds.requires_grad_(False)
    x = torch.tensor([[[0.0, 0.2, 0.6, 1.0]]], requires_grad=True)
    weights = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    output = layer(x)
    (output * weights).sum().backward()

    torch.testing.assert_close(
        output,
        torch.tensor([[[0.1, 0.3, 0.9, 0.6]]]),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(x.grad, weights, rtol=0, atol=0)
    torch.testing.assert_close(
        layer.bucket_offsets.grad,
        torch.tensor([[3.0, 0.0, 3.0, 4.0]]),
        rtol=0,
        atol=0,
    )


def test_bucket_residual_adapter_reuses_deferred_threshold_gradient():
    layer = FullPrecisionBucketResidualAdapter(
        num_bits=1,
        group_size=4,
        grouping_dim="token",
        table_axis="group",
        quant_width=4,
    )
    layer.set_bucket_residual_state(
        torch.tensor([[0.0, 2.0]]),
        torch.tensor([[0.5]]),
    )
    layer.bandwidth = 0.02
    x = torch.tensor([[[0.0, 0.49, 0.51, 1.0]]])

    layer(x).sum().backward()
    assert layer.thresholds.grad is None
    assert layer.has_pending_threshold_side_sums()
    assert finalize_deferred_threshold_grads([layer], mode="raw") == 1

    torch.testing.assert_close(
        layer.thresholds.grad,
        torch.tensor([[-4.0]]),
        rtol=0,
        atol=0,
    )


def test_build_kv_quant_targets_exposes_token_quant_width():
    model = _Model(num_hidden_layers=1)

    def factory(spec, _module):
        return nn.Identity()

    targets = build_kv_quant_targets(
        model,
        factory,
        layer_specs=find_kv_proj_layers(model),
        log_fn=None,
    )

    by_name = {target["name"]: target for target in targets}
    assert by_name["layers.0.self_attn.k_proj"]["quant_width"] == 4
    assert by_name["layers.0.self_attn.v_proj"]["quant_width"] == 4


def test_build_kv_quant_targets_infers_fused_qkv_proj_kv_widths():
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    attn = nn.Module()
    attn.num_heads = 4
    attn.num_key_value_heads = 2
    attn.head_dim = 2
    attn.qkv_proj = nn.Linear(4, 16, bias=False)
    model.layers[0].self_attn = attn

    def factory(spec, _module):
        return nn.Identity()

    targets = build_kv_quant_targets(
        model,
        factory,
        layer_specs=find_kv_proj_layers(model),
        log_fn=None,
    )

    assert [target["kv_slice"] for target in targets] == ["k", "v"]
    assert [target["quant_width"] for target in targets] == [4, 4]


def test_build_kv_quant_targets_infers_fused_qkv_proj_kv_widths_from_config():
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    attn = nn.Module()
    attn.config = SimpleNamespace(num_attention_heads=40, num_key_value_heads=10)
    attn.num_key_value_heads = 10
    attn.head_dim = 128
    attn.qkv_proj = nn.Linear(5120, 7680, bias=False)
    model.layers[0].self_attn = attn

    targets = build_kv_quant_targets(
        model,
        lambda _spec, _module: nn.Identity(),
        layer_specs=find_kv_proj_layers(model),
        log_fn=None,
    )

    assert [target["kv_slice"] for target in targets] == ["k", "v"]
    assert [target["quant_width"] for target in targets] == [1280, 1280]


def test_save_quant_config_writes_group_tables_by_default():
    model = nn.Module()
    model.layer = nn.Module()
    quantizer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=4,
    )
    model.layer.v_proj = QuantizedLinear(nn.Identity(), quantizer)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_quant_config(
            tmpdir,
            model,
            ["layer.v_proj"],
            1,
            QuantizedLinear,
            "step_1",
            log_fn=None,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)

    assert path.endswith(".pt")
    assert payload["layer.v_proj"]["table_axis"] == "group"
    assert payload["layer.v_proj"]["num_tables"] == 2
    assert payload["layer.v_proj"]["quant_points"].shape == (2, 2)
    assert payload["layer.v_proj"]["thresholds"].shape == (2, 1)
    assert payload["layer.v_proj"]["quant_points"].dtype == torch.float16
    assert payload["layer.v_proj"]["thresholds"].dtype == torch.float16
    assert payload["layer.v_proj"]["table_storage_dtype"] == "float16"
    assert payload["layer.v_proj"]["table_compute_dtype"] == "float32"
    assert payload["layer.v_proj"]["table_precision_abi"] == TABLE_PRECISION_ABI


def test_save_quant_config_writes_group_tables_when_enabled():
    model = nn.Module()
    model.layer = nn.Module()
    quantizer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        table_axis="group",
        quant_width=4,
    )
    model.layer.v_proj = QuantizedLinear(nn.Identity(), quantizer)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_quant_config(
            tmpdir,
            model,
            ["layer.v_proj"],
            1,
            QuantizedLinear,
            "step_1",
            log_fn=None,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["layer.v_proj"]["table_axis"] == "group"
    assert payload["layer.v_proj"]["num_tables"] == 2
    assert payload["layer.v_proj"]["quant_points"].shape == (2, 2)
    assert payload["layer.v_proj"]["thresholds"].shape == (2, 1)


def test_quant_config_embeds_metadata_and_apply_skips_it():
    model = nn.Module()
    model.layer = nn.Module()
    quantizer = UnifiedQuantLayer(
        num_bits=1,
        group_size=2,
        grouping_dim="token",
        quant_width=4,
    )
    model.layer.v_proj = QuantizedLinear(nn.Identity(), quantizer)
    metadata = {
        "artifact_type": "beyond_kv_quant_config",
        "base_model": "example/model",
        "model_revision": "abc123",
    }

    payload = extract_quant_config(
        model,
        ["layer.v_proj"],
        1,
        QuantizedLinear,
        log_fn=None,
        metadata=metadata,
    )

    assert payload["__metadata__"] == metadata
    assert apply_quant_config(model, payload, QuantizedLinear, log_fn=None) == 1


def test_save_uniform_affine_config_materializes_constraints_and_metadata():
    model = nn.Module()
    model.layer = nn.Module()
    quantizer = UniformAffineQuantLayer(
        num_bits=2,
        group_size=2,
        grouping_dim="token",
        quant_width=4,
    )
    with torch.no_grad():
        quantizer.affine_low.copy_(torch.tensor([0.1, 0.2]))
        quantizer.affine_high.copy_(torch.tensor([0.9, 0.8]))
    model.layer.v_proj = QuantizedLinear(nn.Identity(), quantizer)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_quant_config(
            tmpdir,
            model,
            ["layer.v_proj"],
            2,
            QuantizedLinear,
            "step_1",
            log_fn=None,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)["layer.v_proj"]

    assert payload["quantizer_parameterization"] == "uniform_affine_endpoints"
    assert payload["hard_forward"] is True
    assert payload["compressed_kv_cache"] is True
    expected_q_points = (
        quantizer.affine_low.unsqueeze(-1)
        + (quantizer.affine_high - quantizer.affine_low).unsqueeze(-1) * quantizer.uniform_fractions
    )
    expected_thresholds = (expected_q_points[:, :-1] + expected_q_points[:, 1:]) * 0.5
    assert torch.equal(payload["quant_points"], expected_q_points.to(torch.float16))
    assert torch.equal(payload["thresholds"], expected_thresholds.to(torch.float16))


def test_full_precision_matched_adapter_config_is_uncompressed_and_reloadable():
    model = nn.Module()
    model.layer = nn.Module()
    quantizer = FullPrecisionChebyshevAdapter(
        num_bits=2,
        group_size=2,
        grouping_dim="token",
        table_axis="group",
        quant_width=4,
    )
    with torch.no_grad():
        quantizer.coefficients.copy_(torch.arange(14, dtype=torch.float32).reshape(2, 7) / 100.0)
    model.layer.v_proj = QuantizedLinear(nn.Identity(), quantizer)
    payload = extract_quant_config(
        model,
        ["layer.v_proj"],
        2,
        QuantizedLinear,
        log_fn=None,
        metadata={"artifact_type": "beyond_uncompressed_kv_adapter_config"},
    )
    leaf = payload["layer.v_proj"]

    assert leaf["quantizer_parameterization"] == "full_precision_chebyshev_residual"
    assert leaf["hard_forward"] is False
    assert leaf["compressed_kv_cache"] is False
    assert leaf["cache_storage"] == "model_activation_dtype"
    assert "quant_points" not in leaf

    with torch.no_grad():
        quantizer.coefficients.zero_()
    assert apply_quant_config(model, payload, QuantizedLinear, log_fn=None) == 1
    assert torch.equal(
        quantizer.coefficients.cpu(),
        leaf["adapter_coefficients"],
    )


def test_bucket_residual_adapter_config_preserves_offset_order_and_thresholds():
    model = nn.Module()
    model.layer = nn.Module()
    quantizer = FullPrecisionBucketResidualAdapter(
        num_bits=2,
        group_size=2,
        grouping_dim="token",
        table_axis="group",
        quant_width=4,
    )
    offsets = torch.tensor(
        [[0.2, -0.4, 0.1, -0.3], [-0.5, 0.4, -0.2, 0.3]],
        dtype=torch.float32,
    )
    thresholds = torch.tensor(
        [[0.1, 0.5, 0.8], [0.2, 0.4, 0.9]],
        dtype=torch.float32,
    )
    quantizer.set_bucket_residual_state(offsets, thresholds)
    model.layer.v_proj = QuantizedLinear(nn.Identity(), quantizer)
    payload = extract_quant_config(
        model,
        ["layer.v_proj"],
        2,
        QuantizedLinear,
        log_fn=None,
        metadata={"artifact_type": "beyond_uncompressed_kv_adapter_config"},
    )
    leaf = payload["layer.v_proj"]

    assert leaf["quantizer_parameterization"] == "full_precision_bucket_residual"
    assert leaf["hard_forward"] is False
    assert leaf["hard_bucket_assignment"] is True
    assert leaf["compressed_kv_cache"] is False
    assert "quant_points" not in leaf
    assert torch.equal(leaf["bucket_offsets"], offsets)
    assert torch.equal(leaf["bucket_thresholds"], thresholds)

    with torch.no_grad():
        quantizer.bucket_offsets.zero_()
        quantizer.thresholds.copy_(torch.tensor([[0.25, 0.5, 0.75]]).repeat(2, 1))
    assert apply_quant_config(model, payload, QuantizedLinear, log_fn=None) == 1
    assert torch.equal(quantizer.bucket_offsets.cpu(), offsets)
    assert torch.equal(quantizer.thresholds.cpu(), thresholds)


@pytest.mark.parametrize("num_bits", [1, 2, 3, 4])
@pytest.mark.parametrize("group_size", [8, 24])
def test_triton_group_ste_quant_matches_python_reference(num_bits, group_size):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    try:
        from beyond.quantization.triton_ste import (
            triton_group_quant_backward_ste,
            triton_group_quant_forward,
        )
    except Exception:
        pytest.skip("requires Triton")

    torch.manual_seed(2)
    levels = 1 << num_bits
    x = torch.rand(2, 7, 4, group_size, device="cuda", dtype=torch.float32)
    q_points = torch.linspace(0, 1, levels, device="cuda").repeat(4, 1).contiguous()
    q_points = (q_points + torch.arange(4, device="cuda").float().unsqueeze(1) * 1e-4).contiguous()
    thresholds = ((q_points[:, :-1] + q_points[:, 1:]) / 2).contiguous()

    result = triton_group_quant_forward(x, q_points, thresholds)
    if result is None:
        pytest.skip("Triton grouped quant is unavailable for this shape")
    out, bins = result
    ref_out, ref_bins = CustomQuantFunction._forward_group_tables(
        x,
        q_points,
        thresholds,
    )

    assert torch.equal(bins.to(ref_bins.dtype).cpu(), ref_bins.cpu())
    torch.testing.assert_close(out.cpu(), ref_out.cpu(), rtol=0, atol=4e-3)

    grad = torch.randn_like(x)
    backward_result = triton_group_quant_backward_ste(
        grad,
        x,
        q_points,
        thresholds,
        bins,
        0.009,
    )
    assert backward_result is not None
    grad_x, grad_q, sum_left, sum_right = backward_result
    ref_grad_x, ref_grad_q, ref_sum_left, ref_sum_right = (
        CustomQuantFunction._backward_group_tables(
            grad,
            x,
            q_points,
            thresholds,
            ref_bins,
            0.009,
        )
    )

    assert torch.equal(grad_x.cpu(), ref_grad_x.cpu())
    torch.testing.assert_close(grad_q.cpu(), ref_grad_q.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sum_left.cpu(), ref_sum_left.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sum_right.cpu(), ref_sum_right.cpu(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("group_size", [24, 32])
def test_normalized_group_quant_fast_path_matches_reference_qtable_grads(group_size):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    torch.manual_seed(3)
    batch, seq, groups, bits = 2, 17, 7, 5
    width = groups * group_size
    x = torch.randn(
        batch,
        seq,
        width,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    grad = torch.randn_like(x)
    q_points = (
        torch.linspace(
            0,
            1,
            1 << bits,
            device="cuda",
            dtype=torch.float32,
        )
        .repeat(groups, 1)
        .contiguous()
    )
    q_points = (
        q_points + torch.arange(groups, device="cuda", dtype=torch.float32).unsqueeze(1) * 1e-4
    )
    thresholds = (q_points[:, :-1] + q_points[:, 1:]) * 0.5
    fast_layer = UnifiedQuantLayer(
        num_bits=bits,
        group_size=group_size,
        grouping_dim="token",
        quant_width=width,
    ).cuda()
    with torch.no_grad():
        fast_layer.q_points.copy_(q_points)
        fast_layer.thresholds.copy_(thresholds)

    out = NormalizedGroupQuantFunction.apply(
        x,
        fast_layer.q_points,
        fast_layer.thresholds,
        0.009,
        group_size,
        fast_layer,
    )
    (out * grad).float().sum().backward()
    finalize_deferred_threshold_grads([fast_layer], mode="half_wave")
    fast_grad_x = x.grad.detach().clone()
    fast_grad_q = fast_layer.q_points.grad.detach().clone()
    fast_grad_t = fast_layer.thresholds.grad.detach().clone()

    ref_layer = UnifiedQuantLayer(
        num_bits=bits,
        group_size=group_size,
        grouping_dim="token",
        quant_width=width,
    ).cuda()
    with torch.no_grad():
        ref_layer.q_points.copy_(q_points)
        ref_layer.thresholds.copy_(thresholds)
    x_ref = x.detach().clone().float().requires_grad_(True)
    x_g = x_ref.view(batch, seq, groups, group_size)
    mn = x_g.amin(dim=-1, keepdim=True)
    mx = x_g.amax(dim=-1, keepdim=True)
    scale = (mx - mn).clamp(min=1e-6)
    x_norm = ((x_g - mn) / scale).clamp(0.0, 1.0)
    xq_norm = CustomQuantFunction.apply(
        x_norm,
        ref_layer.q_points,
        ref_layer.thresholds,
        0.009,
        ref_layer,
    )
    ref_out = (xq_norm * scale + mn).view(batch, seq, width).to(torch.bfloat16)
    (ref_out * grad).float().sum().backward()
    finalize_deferred_threshold_grads([ref_layer], mode="half_wave")

    torch.testing.assert_close(out.cpu(), ref_out.cpu(), rtol=0, atol=4e-3)
    torch.testing.assert_close(
        fast_grad_q.cpu(), ref_layer.q_points.grad.cpu(), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        fast_grad_t.cpu(), ref_layer.thresholds.grad.cpu(), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(fast_grad_x.cpu(), grad.cpu(), rtol=0, atol=0)


class _ShiftQuantizer(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + self.shift


class _ScaleQuantizer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        return x * self.scale


@pytest.mark.parametrize("wrapper_cls", [QuantizedQKVLinear, MegatronQuantizedQKVLinear])
def test_fused_qkv_can_defer_k_quantization_until_rope(wrapper_cls):
    if hasattr(_tls, "active_k_quantizer"):
        delattr(_tls, "active_k_quantizer")
    k_quantizer = _ShiftQuantizer(10.0)
    v_quantizer = _ShiftQuantizer(20.0)
    wrapper = wrapper_cls(
        nn.Identity(),
        k_quantizer=k_quantizer,
        v_quantizer=v_quantizer,
        qkv_splits=(2, 2, 2),
    )
    wrapper.defer_k_quantize = True
    x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)

    out = wrapper(x)

    assert torch.equal(out[..., :2], x[..., :2])
    assert torch.equal(out[..., 2:4], x[..., 2:4])
    assert torch.equal(out[..., 4:6], x[..., 4:6] + 20.0)
    assert k_quantizer.calls == 0
    assert v_quantizer.calls == 1
    assert getattr(_tls, "active_k_quantizer") is k_quantizer
    delattr(_tls, "active_k_quantizer")


@pytest.mark.parametrize("unsqueeze_dim", [1, 2])
def test_post_rope_k_helper_quantizes_token_flattened_heads(unsqueeze_dim):
    quantizer = _ShiftQuantizer(3.0)
    if unsqueeze_dim == 1:
        k_rot = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    else:
        k_rot = torch.arange(2 * 4 * 3 * 5, dtype=torch.float32).reshape(2, 4, 3, 5)

    out = quantize_post_rope_k_tensor(k_rot, quantizer, unsqueeze_dim=unsqueeze_dim)

    assert torch.equal(out, k_rot + 3.0)
    assert quantizer.calls == 1


def test_install_post_rope_hook_defers_projection_and_quantizes_rope_output(monkeypatch):
    module = sys.modules[__name__]

    def apply_rotary_pos_emb(q, k, cos, sin, **_kwargs):
        del cos, sin
        return q + 1.0, k + 2.0

    monkeypatch.setattr(module, "apply_rotary_pos_emb", apply_rotary_pos_emb, raising=False)
    model = _Model(num_hidden_layers=1)
    quantizer = _ShiftQuantizer(5.0)
    wrapped = QuantizedLinear(nn.Identity(), quantizer)
    model.layers[0].self_attn.k_proj = wrapped

    count = install_post_rope_k_quantization(model, log_fn=None)
    assert count == 1
    assert wrapped.defer_quantize is True

    projected = wrapped(torch.zeros(1, 3, 10))
    assert torch.equal(projected, torch.zeros_like(projected))
    q = torch.zeros(1, 3, 2, 5)
    k = torch.zeros_like(q)
    q_rot, k_quantized = module.apply_rotary_pos_emb(
        q,
        k,
        None,
        None,
        unsqueeze_dim=2,
    )
    assert torch.equal(q_rot, torch.ones_like(q))
    assert torch.equal(k_quantized, torch.full_like(k, 7.0))
    assert not hasattr(_tls, "active_k_quantizer") or _tls.active_k_quantizer is None


@pytest.mark.parametrize("wrapper_cls", [QuantizedQKVLinear, MegatronQuantizedQKVLinear])
def test_fused_qkv_merge_preserves_quantizer_grads(wrapper_cls, monkeypatch):
    monkeypatch.setenv("DISCOVER_QKV_INPLACE", "1")
    linear = nn.Linear(6, 6, bias=False)
    k_quantizer = _ScaleQuantizer(2.0)
    v_quantizer = _ScaleQuantizer(3.0)
    wrapper = wrapper_cls(
        linear,
        k_quantizer=k_quantizer,
        v_quantizer=v_quantizer,
        qkv_splits=(2, 2, 2),
    )
    x = torch.randn(2, 3, 6)

    out = wrapper(x)
    out.float().sum().backward()

    assert linear.weight.grad is not None
    assert k_quantizer.scale.grad is not None
    assert v_quantizer.scale.grad is not None


def test_discover_megatron_parallel_request_accepts_tensor_parallel_grid():
    validate_discover_megatron_parallel_request(
        world_size=8,
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=1,
    )


def test_discover_megatron_parallel_request_allows_explicit_linear_only_tp_opt_in():
    validate_discover_megatron_parallel_request(
        world_size=8,
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=1,
        allow_linear_only_tensor_parallel=True,
    )


def test_discover_megatron_parallel_request_rejects_invalid_grid():
    with pytest.raises(ValueError, match="must be divisible"):
        validate_discover_megatron_parallel_request(
            world_size=6,
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
        )


def test_discover_dataset_recipes_include_supported_corpora():
    assert DISCOVER_DATASET_CHOICES == (
        "pile",
        "fineweb",
        "openwebmath",
        "fineweb2_cmn_hani",
        "fineweb2_multilingual_equal",
        "hotpotqa_distractor",
        "hotpot_2wiki_equal_input",
        "ultrachat_200k",
    )
    assert DISCOVER_CHAT_WRAP_PILE_DEFAULT is True

    sources = get_discover_sources("pile")
    assert len(sources) == 1
    assert sources[0]["path"] == "EleutherAI/the_pile_deduplicated"
    assert sources[0]["category"] == "ordinary_lm"
    assert sources[0]["chat_wrap"] is True
    assert sources[0]["chat_user_prompt"] == DISCOVER_CHAT_WRAP_PILE_PROMPT

    raw_sources = get_discover_sources("pile", chat_wrap_pile=False)
    assert raw_sources[0]["chat_wrap"] is False
    assert raw_sources[0]["chat_user_prompt"] == ""

    with pytest.raises(ValueError, match="Unsupported discover dataset"):
        get_discover_sources("unsupported")


def test_fineweb2_chinese_and_equal_multilingual_recipes_are_explicit():
    chinese_sources = get_discover_sources("fineweb2_cmn_hani", chat_wrap_pile=False)
    assert len(chinese_sources) == 1
    assert chinese_sources[0]["path"] == "HuggingFaceFW/fineweb-2"
    assert chinese_sources[0]["name"] == "cmn_Hani"
    assert chinese_sources[0]["fields"] == ("text",)

    multilingual_sources = get_discover_sources(
        "fineweb2_multilingual_equal",
        chat_wrap_pile=False,
    )
    assert [source["name"] for source in multilingual_sources] == [
        "sample-10BT",
        "cmn_Hani",
        "spa_Latn",
        "arb_Arab",
    ]
    assert [source["path"] for source in multilingual_sources] == [
        "HuggingFaceFW/fineweb",
        "HuggingFaceFW/fineweb-2",
        "HuggingFaceFW/fineweb-2",
        "HuggingFaceFW/fineweb-2",
    ]
    assert {source["target_tokens"] for source in multilingual_sources} == {25}
    assert {source["target_unit"] for source in multilingual_sources} == {"supervised_tokens"}
    assert {source["allow_chat_wrap"] for source in multilingual_sources} == {False}
    assert not any(
        source["chat_wrap"]
        for source in get_discover_sources("fineweb2_multilingual_equal", chat_wrap_pile=True)
    )
    scaled = _source_targets_for_token_budget(multilingual_sources, 4_000)
    assert [source["target_tokens"] for source in scaled] == [1_000] * 4


def test_hotpotqa_distractor_masks_context_and_question():
    source = get_discover_sources("hotpotqa_distractor")[0]
    segments = _sample_to_segments(
        {
            "context": {
                "title": ["Arthur's Magazine", "First for Women"],
                "sentences": [
                    ["Arthur's Magazine began in 1844."],
                    ["First for Women began in 1989."],
                ],
            },
            "question": "Which magazine was started first?",
            "answer": "Arthur's Magazine",
        },
        source,
        _NoChatTemplateTokenizer(),
    )

    assert segments[-2] == ("Arthur's Magazine", True)
    assert all(not has_loss for _text, has_loss in segments[:-2])
    assert "Arthur's Magazine began in 1844." in "".join(text for text, _ in segments)
    assert "Which magazine was started first?" in "".join(text for text, _ in segments)


def test_hotpot_2wiki_equal_input_recipe_is_pinned_and_exactly_balanced():
    sources = get_discover_sources(
        "hotpot_2wiki_equal_input",
        chat_wrap_pile=True,
        chat_user_prompt="This must not be applied.",
    )

    assert [source["path"] for source in sources] == [
        "hotpotqa/hotpot_qa",
        "framolfese/2WikiMultihopQA",
    ]
    assert [source["target_tokens"] for source in sources] == [50, 50]
    assert [source["revision"] for source in sources] == [
        "1908d6afbbead072334abe2965f91bd2709910ab",
        "fe713bfbd1afbca1a65246741a75890405d56a3a",
    ]
    assert [source["name"] for source in sources] == ["distractor", None]
    assert [source["split"] for source in sources] == ["train", "train"]
    assert {source["fields"] for source in sources} == {("context", "question", "answer")}
    assert {source["format"] for source in sources} == {"multihop_qa"}
    assert {source["target_unit"] for source in sources} == {"tokens"}
    assert {source["allow_chat_wrap"] for source in sources} == {False}
    assert not any(source["chat_wrap"] for source in sources)
    assert not any(source["chat_user_prompt"] for source in sources)

    scaled = _source_targets_for_token_budget(sources, 4_000)
    assert [source["target_tokens"] for source in scaled] == [2_000, 2_000]


@pytest.mark.parametrize(
    ("source_index", "sample", "expected_context", "expected_question", "expected_answer"),
    (
        (
            0,
            {
                "context": {
                    "title": ["Hotpot title"],
                    "sentences": [["Hotpot sentence one.", "Hotpot sentence two."]],
                },
                "question": "What is the Hotpot answer?",
                "answer": "Hotpot answer",
                "supporting_facts": {"title": ["Hotpot title"], "sent_id": [0]},
            },
            "[Hotpot title]\nHotpot sentence one. Hotpot sentence two.",
            "What is the Hotpot answer?",
            "Hotpot answer",
        ),
        (
            1,
            {
                "context": [
                    ["2Wiki title", ["2Wiki sentence one.", "2Wiki sentence two."]],
                    {"title": "Second title", "sentences": ["Second passage."]},
                ],
                "question": "What is the 2Wiki answer?",
                "answer": "2Wiki answer",
                "evidences": [["subject", "relation", "object"]],
            },
            "[2Wiki title]\n2Wiki sentence one. 2Wiki sentence two.",
            "What is the 2Wiki answer?",
            "2Wiki answer",
        ),
    ),
)
def test_hotpot_2wiki_formats_both_context_schemas_and_masks_inputs(
    source_index,
    sample,
    expected_context,
    expected_question,
    expected_answer,
):
    source = get_discover_sources(
        "hotpot_2wiki_equal_input",
        chat_wrap_pile=True,
    )[source_index]

    segments = _sample_to_segments(sample, source, _NoChatTemplateTokenizer())
    rendered = "".join(text for text, _has_loss in segments)

    assert expected_context in rendered
    assert expected_question in rendered
    assert [(text, has_loss) for text, has_loss in segments if has_loss] == [
        (expected_answer, True)
    ]
    assert all(
        not has_loss
        for text, has_loss in segments
        if expected_context in text or expected_question in text
    )
    assert "This must not be applied." not in rendered


def test_ultrachat_train_sft_targets_assistant_messages_only():
    source = get_discover_sources("ultrachat_200k")[0]
    assert source["path"] == "HuggingFaceH4/ultrachat_200k"
    assert source["split"] == "train_sft"
    segments = _sample_to_segments(
        {
            "messages": [
                {"role": "user", "content": "Give a short answer."},
                {"role": "assistant", "content": "A short answer."},
            ]
        },
        source,
        _NoChatTemplateTokenizer(),
    )

    targets = [text for text, has_loss in segments if has_loss]
    assert targets == ["A short answer."]
    assert all(not has_loss for text, has_loss in segments if "Give a short answer." in text)


def test_supervised_budget_truncation_counts_only_unmasked_labels():
    input_ids = [10, 11, 12, 13, 14, 15]
    labels = [-100, -100, 12, -100, 14, 15]

    truncated_ids, truncated_labels = _truncate_to_supervised_budget(input_ids, labels, 2)

    assert truncated_ids == [10, 11, 12, 13, 14]
    assert truncated_labels == [-100, -100, 12, -100, 14]


def test_pile_chat_wrap_adds_masked_prompt_before_target_text():
    source = get_discover_sources("pile")[0]
    segments = _sample_to_segments(
        {"text": "The document body."},
        source,
        _ChatTemplateTokenizer(),
    )

    assert segments == [
        (f"<user>{DISCOVER_CHAT_WRAP_PILE_PROMPT}</user><assistant><final>", False),
        ("The document body.", True),
    ]


def test_pile_chat_wrap_without_template_keeps_plain_lm_text():
    source = get_discover_sources("pile")[0]
    segments = _sample_to_segments(
        {"text": "The document body."},
        source,
        _NoChatTemplateTokenizer(),
    )

    assert segments == [("The document body.", True)]


def test_qwen3_chat_prefix_matches_official_generation_prompt_template():
    tokenizer = _QwenOfficialChatTemplateTokenizer()
    messages = [{"role": "user", "content": DISCOVER_CHAT_WRAP_PILE_PROMPT}]

    official_prefix = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    assert _chat_generation_prefix(tokenizer, DISCOVER_CHAT_WRAP_PILE_PROMPT) == official_prefix
    assert official_prefix == (
        f"<|im_start|>user\n{DISCOVER_CHAT_WRAP_PILE_PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    )


def test_ministral3_chat_prefix_uses_tokenized_official_template():
    class MistralCommonBackend:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return [1, 3, 100, 4]

    tokenizer = MistralCommonBackend()

    prefix = _chat_generation_prefix(tokenizer, DISCOVER_CHAT_WRAP_PILE_PROMPT)

    assert prefix == [1, 3, 100, 4]
    assert tokenizer.calls == [
        (
            [{"role": "user", "content": DISCOVER_CHAT_WRAP_PILE_PROMPT}],
            {"add_generation_prompt": True, "return_dict": False},
        )
    ]


def test_encode_ids_accepts_pre_tokenized_segments():
    class FailingTokenizer:
        def encode(self, *args, **kwargs):
            raise AssertionError("pre-tokenized segment should not be encoded")

    assert _encode_ids(FailingTokenizer(), [1, 3, 100, 4]) == [1, 3, 100, 4]


def test_discover_heldout_budgets_are_fixed_chunk_counts():
    curriculum = [{"name": "main_8k", "seqlen": 8192, "token_budget": 50_000_000}]

    validation_specs = _heldout_stage_specs_from_curriculum(curriculum, DISCOVER_VALIDATION_CHUNKS)
    eval_specs = _heldout_stage_specs_from_curriculum(curriculum, DISCOVER_EVAL_CHUNKS)

    assert DISCOVER_VALIDATION_CHUNKS == 150
    assert DISCOVER_EVAL_CHUNKS == 150
    assert validation_specs[0]["sample_budget"] == 150
    assert validation_specs[0]["token_budget"] == 150 * 8192
    assert eval_specs[0]["sample_budget"] == 150
    assert eval_specs[0]["token_budget"] == 150 * 8192


def test_discover_uses_qwen3_moe_instruct_model_id():
    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in SUPPORTED_DISCOVER_MODELS
    assert canonical_model_id("qwen3-30b-a3b") == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert canonical_model_id("qwen3-30b-a3b-instruct") == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert canonical_model_id("Qwen/Qwen3-30B-A3B") == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert detect_model_type("Qwen/Qwen3-30B-A3B-Instruct-2507") == "qwen"


def test_discover_supports_qwen3_8b_base_without_chat_wrap():
    model_id = "Qwen/Qwen3-8B-Base"

    assert model_id in SUPPORTED_DISCOVER_MODELS
    assert canonical_model_id("qwen3-8b-base") == model_id
    assert canonical_model_id(model_id) == model_id
    assert detect_model_type(model_id) == "qwen"
    assert discover_default_chat_wrap_pile(model_id) is False

    config = get_model_config("qwen", model_id, attn_implementation="eager")
    assert config["attn_implementation"] == "eager"
    assert "experts_implementation" not in config


def test_discover_data_and_eval_seeds_are_independent_of_optimizer_seed():
    split_seeds = discover_split_seeds(
        data_seed=DISCOVER_DATA_SEED,
        eval_seed=DISCOVER_EVAL_SEED,
        stage_index=0,
    )

    assert split_seeds == {
        "train": DISCOVER_DATA_SEED + 10_000,
        "validation": DISCOVER_EVAL_SEED,
        "eval": DISCOVER_EVAL_SEED + 10_000_000,
    }
    assert "optimizer" not in split_seeds


def test_discover_defaults_to_ministral3_14b_instruct_model_id():
    model_id = "mistralai/Ministral-3-14B-Instruct-2512-BF16"

    assert SUPPORTED_DISCOVER_MODELS[0] == model_id
    assert model_id in SUPPORTED_DISCOVER_MODELS
    assert canonical_model_id("ministral-3-14b") == model_id
    assert canonical_model_id("ministral3-14b-instruct") == model_id
    assert canonical_model_id("mistralai/Ministral-3-14B-Instruct-2512") == model_id
    assert detect_model_type(model_id) == "mistral3"


def test_ministral3_load_tokenizer_uses_official_mistral_common_backend(monkeypatch):
    import beyond.models.model_utils as model_utils

    calls = []

    class FakeMistralCommonBackend:
        pad_token = "<pad>"
        padding_side = "left"

        @classmethod
        def from_pretrained(cls, model_id):
            calls.append(model_id)
            return cls()

    monkeypatch.setattr(model_utils, "MistralCommonBackend", FakeMistralCommonBackend)

    tokenizer = load_tokenizer(
        "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "mistral3",
    )

    assert calls == ["mistralai/Ministral-3-14B-Instruct-2512-BF16"]
    assert tokenizer.name_or_path == "mistralai/Ministral-3-14B-Instruct-2512-BF16"
    assert tokenizer.padding_side == "right"


def test_discover_chat_wrap_default_enabled_for_supported_instruct_models():
    assert discover_default_chat_wrap_pile("mistralai/Ministral-3-14B-Instruct-2512-BF16") is True
    assert discover_default_chat_wrap_pile("Qwen/Qwen3-30B-A3B-Instruct-2507") is True
    assert discover_default_chat_wrap_pile("meta-llama/Llama-3.1-8B-Instruct") is True
    assert discover_default_chat_wrap_pile("Qwen/Qwen3-8B-Base") is False


def test_discover_uses_instruct_llama31_model_id():
    assert "meta-llama/Llama-3.1-8B-Instruct" in SUPPORTED_DISCOVER_MODELS
    assert "meta-llama/Llama-3.1-8B" not in SUPPORTED_DISCOVER_MODELS
    assert canonical_model_id("llama-3.1-8b") == "meta-llama/Llama-3.1-8B-Instruct"
    assert canonical_model_id("llama-3.1-8b-ins") == "meta-llama/Llama-3.1-8B-Instruct"
    assert canonical_model_id("meta-llama/Llama-3.1-8B") == "meta-llama/Llama-3.1-8B-Instruct"
    assert detect_model_type("meta-llama/Llama-3.1-8B-Instruct") == "llama"


def test_qwen3_auto_attention_uses_standard_candidate_order():
    config = get_model_config(
        "qwen", "Qwen/Qwen3-30B-A3B-Instruct-2507", attn_implementation="auto"
    )

    assert config["attn_implementation"] == config["_attn_implementation_candidates"][0]
    assert "sdpa" in config["_attn_implementation_candidates"]
    assert config["_attn_implementation_candidates"][-1] == "eager"
    assert config["experts_implementation"] == "eager"


def test_ministral3_auto_attention_uses_standard_candidate_order_without_moe_config():
    config = get_model_config(
        "mistral3",
        "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        attn_implementation="auto",
    )

    assert config["attn_implementation"] == config["_attn_implementation_candidates"][0]
    assert "sdpa" in config["_attn_implementation_candidates"]
    assert config["_attn_implementation_candidates"][-1] == "eager"
    assert "experts_implementation" not in config


def test_qwen3_experts_implementation_can_be_overridden():
    config = get_model_config(
        "qwen",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        experts_implementation="grouped_mm",
    )

    assert "grouped_mm" in DISCOVER_EXPERTS_IMPLEMENTATION_CHOICES
    assert config["experts_implementation"] == "grouped_mm"
