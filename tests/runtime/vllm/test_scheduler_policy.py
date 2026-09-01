from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from beyond.runtime.vllm import packed_backend as mod  # noqa: E402


def _request(*, pending: int = 1, speculative: bool = False, encoder: bool = False):
    return SimpleNamespace(
        num_tokens_with_spec=pending,
        num_output_placeholders=0,
        num_computed_tokens=0,
        spec_token_ids=[1] if speculative else [],
        has_encoder_inputs=encoder,
    )


def _scheduler(
    *,
    performance_mode: str = "throughput",
    prefix_caching: bool = False,
    running=None,
    waiting=None,
    max_num_running_reqs: int = 2,
):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(performance_mode=performance_mode),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        running=[_request()] if running is None else running,
        waiting=[object()] if waiting is None else waiting,
        skipped_waiting=[],
        max_num_running_reqs=max_num_running_reqs,
    )


def _engine_config(
    *,
    max_num_seqs: int = 128,
    max_num_batched_tokens: int = 16384,
    max_model_len: int = 4096,
    block_size: int = 128,
    performance_mode: str = "throughput",
    cudagraph_capture_sizes=None,
):
    if cudagraph_capture_sizes is None:
        cudagraph_capture_sizes = [1, 2, 4, 8, 256, 512]
    return SimpleNamespace(
        performance_mode=performance_mode,
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=list(cudagraph_capture_sizes),
            max_cudagraph_capture_size=max(cudagraph_capture_sizes),
        ),
    )


def test_runtime_audit_records_homogeneous_chunked_shapes(monkeypatch):
    monkeypatch.setenv("BEYOND_RUNTIME_AUDIT", "1")
    mod.reset_runtime_audit()

    mod._runtime_audit_chunked_plan(
        q_starts=[0, 64, 160],
        seq_lens=[16_384, 8_192],
    )
    mod._runtime_audit_chunked_plan(
        q_starts=[0, 64, 160],
        seq_lens=[16_384, 8_192],
    )

    assert mod.runtime_audit_snapshot()["chunked_plan_shapes"] == [
        {
            "q_lens": [64, 96],
            "context_lens": [16_320, 8_096],
            "seq_lens": [16_384, 8_192],
            "calls": 2,
        }
    ]
    mod.reset_runtime_audit()
    assert mod.runtime_audit_snapshot()["chunked_plan_shapes"] == []


def test_prefill_priority_defaults_to_uncached_throughput_mode(monkeypatch):
    monkeypatch.delenv("BEYOND_PREFILL_PRIORITY_SCHEDULER", raising=False)
    assert mod._beyond_can_prioritize_waiting_prefill(_scheduler())
    assert not mod._beyond_can_prioritize_waiting_prefill(_scheduler(performance_mode="balanced"))
    assert not mod._beyond_can_prioritize_waiting_prefill(_scheduler(prefix_caching=True))


def test_prefill_priority_environment_override(monkeypatch):
    monkeypatch.setenv("BEYOND_PREFILL_PRIORITY_SCHEDULER", "1")
    assert mod._beyond_can_prioritize_waiting_prefill(_scheduler(performance_mode="balanced"))
    monkeypatch.setenv("BEYOND_PREFILL_PRIORITY_SCHEDULER", "0")
    assert not mod._beyond_can_prioritize_waiting_prefill(_scheduler())


def test_scheduler_policy_preserves_explicit_max_num_seqs(monkeypatch):
    monkeypatch.delenv("BEYOND_MAX_NUM_SEQS_CAP", raising=False)
    monkeypatch.setattr(mod, "BEYOND_FULL_DECODE_MAX_BATCH", 0)
    config = _engine_config(max_num_seqs=128)

    mod._beyond_apply_scheduler_policy(config, explicit_max_num_seqs=True)

    assert config.scheduler_config.max_num_seqs == 128
    assert mod.BEYOND_FULL_DECODE_MAX_BATCH == 128


def test_scheduler_policy_still_caps_automatic_vllm_default(monkeypatch):
    monkeypatch.delenv("BEYOND_MAX_NUM_SEQS_CAP", raising=False)
    monkeypatch.setattr(mod, "BEYOND_FULL_DECODE_MAX_BATCH", 0)
    config = _engine_config(max_num_seqs=256)

    mod._beyond_apply_scheduler_policy(config, explicit_max_num_seqs=False)

    assert config.scheduler_config.max_num_seqs == 32
    assert mod.BEYOND_FULL_DECODE_MAX_BATCH == 32


def test_scheduler_policy_explicit_beyond_cap_overrides_explicit_vllm_value(
    monkeypatch,
):
    monkeypatch.setenv("BEYOND_MAX_NUM_SEQS_CAP", "64")
    monkeypatch.setattr(mod, "BEYOND_FULL_DECODE_MAX_BATCH", 0)
    config = _engine_config(max_num_seqs=128)

    mod._beyond_apply_scheduler_policy(config, explicit_max_num_seqs=True)

    assert config.scheduler_config.max_num_seqs == 64
    assert mod.BEYOND_FULL_DECODE_MAX_BATCH == 64


def test_high_concurrency_graph_policy_adds_sparse_anchors_through_b1024():
    config = _engine_config(max_num_seqs=1024)

    changed = mod._beyond_apply_high_concurrency_cudagraph_policy(
        config,
        explicit_max_num_seqs=True,
        explicit_cudagraph_roster=False,
    )

    assert changed
    assert config.compilation_config.cudagraph_capture_sizes == [
        1,
        2,
        4,
        8,
        256,
        512,
        640,
        768,
        896,
        1024,
    ]
    assert config.compilation_config.max_cudagraph_capture_size == 1024


@pytest.mark.parametrize(
    ("config", "explicit_roster"),
    [
        (_engine_config(max_num_seqs=512), False),
        (_engine_config(max_num_seqs=1024, performance_mode="balanced"), False),
        (_engine_config(max_num_seqs=1024), True),
    ],
)
def test_high_concurrency_graph_policy_preserves_non_selected_configs(
    config,
    explicit_roster,
):
    original_sizes = list(config.compilation_config.cudagraph_capture_sizes)
    original_max = config.compilation_config.max_cudagraph_capture_size

    changed = mod._beyond_apply_high_concurrency_cudagraph_policy(
        config,
        explicit_max_num_seqs=True,
        explicit_cudagraph_roster=explicit_roster,
    )

    assert not changed
    assert config.compilation_config.cudagraph_capture_sizes == original_sizes
    assert config.compilation_config.max_cudagraph_capture_size == original_max


def test_fast_graph_pool_estimate_scales_with_full_roster_and_model_layers(
    monkeypatch,
):
    monkeypatch.setattr(mod, "BEYOND_ENABLE_PIECEWISE_GRAPH", 1)
    monkeypatch.setattr(mod, "BEYOND_FULL_DECODE_MAX_BATCH", 1024)
    capture_sizes = (
        [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16)) + [640, 768, 896, 1024]
    )
    hf_config = SimpleNamespace(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=32,
    )
    runner = SimpleNamespace(
        dtype=mod.torch.bfloat16,
        model_config=SimpleNamespace(
            dtype=mod.torch.bfloat16,
            hf_config=SimpleNamespace(hidden_size=1),
            hf_text_config=hf_config,
        ),
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=1024,
            cudagraph_capture_sizes=capture_sizes,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=131072),
    )

    estimate = mod._beyond_fast_cudagraph_pool_estimate_bytes(runner)

    captured_rows = sum(capture_sizes)
    qkv_width = (32 + 2 * 8) * 128
    structural = 3 * captured_rows * 32 * qkv_width * 2
    overhead = max(512 << 20, len(capture_sizes) * (4 << 20))
    expected = structural + overhead
    quantum = 256 << 20
    expected = (expected + quantum - 1) // quantum * quantum
    assert estimate == expected
    assert estimate == 63 * (256 << 20)


def test_fast_graph_pool_estimate_excludes_piecewise_only_batches(monkeypatch):
    monkeypatch.setattr(mod, "BEYOND_ENABLE_PIECEWISE_GRAPH", 1)
    monkeypatch.setattr(mod, "BEYOND_FULL_DECODE_MAX_BATCH", 128)
    capture_sizes = [1, 2, 4] + list(range(8, 256, 8)) + [256]
    text_config = SimpleNamespace(
        head_dim=128,
        hidden_size=5120,
        intermediate_size=16384,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=40,
    )
    runner = SimpleNamespace(
        dtype=mod.torch.bfloat16,
        model_config=SimpleNamespace(
            dtype=mod.torch.bfloat16,
            hf_config=SimpleNamespace(text_config=text_config),
        ),
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=256,
            cudagraph_capture_sizes=capture_sizes,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=16384,
            max_num_seqs=128,
        ),
    )

    estimate = mod._beyond_fast_cudagraph_pool_estimate_bytes(runner)

    full_sizes = [size for size in capture_sizes if size <= 128]
    qkv_width = (32 + 2 * 8) * 128
    structural = 3 * sum(full_sizes) * 40 * qkv_width * 2
    overhead = max(512 << 20, len(full_sizes) * (4 << 20))
    quantum = 256 << 20
    expected = (structural + overhead + quantum - 1) // quantum * quantum
    assert estimate == expected
    assert estimate == 9 * (256 << 20)


def test_fast_graph_pool_estimate_uses_small_roster_driver_floor(monkeypatch):
    monkeypatch.setattr(mod, "BEYOND_ENABLE_PIECEWISE_GRAPH", 1)
    monkeypatch.setattr(mod, "BEYOND_FULL_DECODE_MAX_BATCH", 1)
    text_config = SimpleNamespace(
        head_dim=128,
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_hidden_layers=48,
    )
    runner = SimpleNamespace(
        dtype=mod.torch.bfloat16,
        model_config=SimpleNamespace(
            dtype=mod.torch.bfloat16,
            hf_config=SimpleNamespace(text_config=text_config),
        ),
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=1,
            cudagraph_capture_sizes=[1],
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=4096,
            max_num_seqs=1,
        ),
    )

    estimate = mod._beyond_fast_cudagraph_pool_estimate_bytes(runner)

    # Qwen's B1 structural graph payload is about 1.4 MiB.  Adding the
    # 128-MiB small-roster driver allowance and rounding to the same quantum
    # deliberately yields a 256-MiB reserve, over 5x its measured B300 pool.
    assert estimate == 2 * (128 << 20)


def test_fast_graph_pool_estimate_uses_validated_b8_small_roster_reserve(monkeypatch):
    monkeypatch.setattr(mod, "BEYOND_ENABLE_PIECEWISE_GRAPH", 1)
    monkeypatch.setattr(mod, "BEYOND_FULL_DECODE_MAX_BATCH", 8)
    text_config = SimpleNamespace(
        head_dim=128,
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_hidden_layers=48,
    )
    runner = SimpleNamespace(
        dtype=mod.torch.bfloat16,
        model_config=SimpleNamespace(
            dtype=mod.torch.bfloat16,
            hf_config=SimpleNamespace(text_config=text_config),
        ),
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=16,
            cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=4096,
            max_num_seqs=8,
        ),
    )

    estimate = mod._beyond_fast_cudagraph_pool_estimate_bytes(runner)

    assert estimate == 2 * (128 << 20)


def test_fast_graph_pool_estimate_long_context_moe_keeps_full_graphs(monkeypatch):
    monkeypatch.setattr(mod, "BEYOND_ENABLE_PIECEWISE_GRAPH", 1)
    monkeypatch.setattr(mod, "BEYOND_FULL_DECODE_MAX_BATCH", 8)
    monkeypatch.setattr(mod, "BEYOND_MOE_FULL_GRAPH_MAX_MODEL_LEN", 131072)
    text_config = SimpleNamespace(
        head_dim=128,
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_hidden_layers=48,
    )
    model_config = SimpleNamespace(
        dtype=mod.torch.bfloat16,
        hf_config=SimpleNamespace(text_config=text_config),
        is_moe=True,
        max_model_len=262144,
    )
    runner = SimpleNamespace(
        dtype=mod.torch.bfloat16,
        model_config=model_config,
        vllm_config=SimpleNamespace(model_config=model_config),
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=16,
            cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=4096,
            max_num_seqs=8,
        ),
    )

    estimate = mod._beyond_fast_cudagraph_pool_estimate_bytes(runner)

    assert estimate == 2 * (128 << 20)


def test_long_context_moe_bounds_only_full_graph_dummy_profile(monkeypatch):
    monkeypatch.setattr(mod, "BEYOND_MOE_FULL_GRAPH_MAX_MODEL_LEN", 131072)
    monkeypatch.setattr(mod, "BEYOND_MOE_FULL_GRAPH_PROFILE_SEQ_LEN", 128)
    model_config = SimpleNamespace(is_moe=True, max_model_len=262144)
    runner = SimpleNamespace(
        max_model_len=262144,
        max_num_tokens=4096,
        vllm_config=SimpleNamespace(model_config=model_config),
    )

    assert (
        mod._beyond_full_decode_profile_seq_lens(
            runner,
            SimpleNamespace(num_tokens=2),
        )
        == 128
    )
    assert (
        mod._beyond_full_decode_profile_seq_lens(
            runner,
            SimpleNamespace(num_tokens=8),
        )
        == 128
    )


def test_short_context_moe_retains_scheduler_sized_full_graph_profile(monkeypatch):
    monkeypatch.setattr(mod, "BEYOND_MOE_FULL_GRAPH_MAX_MODEL_LEN", 131072)
    monkeypatch.setattr(mod, "BEYOND_MOE_FULL_GRAPH_PROFILE_SEQ_LEN", 128)
    model_config = SimpleNamespace(is_moe=True, max_model_len=131072)
    runner = SimpleNamespace(
        max_model_len=131072,
        max_num_tokens=4096,
        vllm_config=SimpleNamespace(model_config=model_config),
    )

    assert (
        mod._beyond_full_decode_profile_seq_lens(
            runner,
            SimpleNamespace(num_tokens=2),
        )
        == 2048
    )


def test_runner_dimensions_use_nested_text_config():
    text_config = SimpleNamespace(
        hidden_size=5120,
        intermediate_size=16384,
        num_hidden_layers=40,
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config),
        )
    )

    assert mod._beyond_runner_hf_config(runner) is text_config
    assert mod._beyond_runner_hidden_size(runner, 32, 128) == 5120
    assert mod._beyond_runner_intermediate_size(runner, 32, 128) == 16384
    assert mod._beyond_runner_num_hidden_layers(runner) == 40


@pytest.mark.parametrize(
    "scheduler",
    [
        _scheduler(running=[], waiting=[object()]),
        _scheduler(waiting=[]),
        _scheduler(max_num_running_reqs=1),
        _scheduler(running=[_request(pending=2)]),
        _scheduler(running=[_request(speculative=True)]),
        _scheduler(running=[_request(encoder=True)]),
    ],
)
def test_prefill_priority_rejects_unsafe_scheduler_states(monkeypatch, scheduler):
    monkeypatch.delenv("BEYOND_PREFILL_PRIORITY_SCHEDULER", raising=False)
    assert not mod._beyond_can_prioritize_waiting_prefill(scheduler)
