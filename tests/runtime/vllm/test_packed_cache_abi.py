from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from beyond.runtime.vllm import packed_backend as backend  # noqa: E402
from quant.beyond_cute import CODE_LAYOUT_HND_TOKEN_WORD  # noqa: E402


@pytest.mark.parametrize("num_kv_heads", [4, 8])
def test_production_layout_is_block128_hnd(num_kv_heads: int):
    layout = backend._paged_layout(
        backend.BEYOND_PACKED_BLOCK_SIZE,
        num_kv_heads,
        128,
    )

    assert layout.block_size == 128
    assert layout.code_layout == CODE_LAYOUT_HND_TOKEN_WORD
    assert layout.abi_version == 2
    assert layout.v_stats_group_major
    assert layout.block_i32 == num_kv_heads * 5120


def test_backend_requires_supported_exact_physical_block_size():
    packed = backend.BEYOND_PACKED_BLOCK_SIZE

    assert backend.BeyondPackedBackend.get_supported_kernel_block_sizes() == [64, packed]
    assert backend.BeyondPackedBackend.get_preferred_block_size(16) == packed
    assert backend.BeyondPackedBackend.supports_block_size(None)
    assert backend.BeyondPackedBackend.supports_block_size(packed)
    assert backend.BeyondPackedBackend.supports_block_size(64)
    assert not backend.BeyondPackedBackend.supports_block_size(32)
    assert not backend.BeyondPackedBackend.supports_block_size(2 * packed)


@pytest.mark.parametrize("block_size", [64, 128])
def test_runner_prewarm_layout_uses_active_cache_block_size(block_size: int):
    runner = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
    )

    layout = backend._beyond_runner_packed_layout(runner, num_kv_heads=8)

    assert layout is not None
    assert layout.block_size == block_size
    assert layout.num_kv_heads == 8


@pytest.mark.parametrize("block_size", [None, 0, 32, 256])
def test_runner_prewarm_layout_rejects_inactive_block_size(block_size):
    runner = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
    )

    assert backend._beyond_runner_packed_layout(runner, num_kv_heads=8) is None


@pytest.mark.parametrize(
    ("seq_len", "capacity", "expected"),
    [
        (1, 4096, 128),
        (1024, 262144, 1024),
        (1025, 262144, 2048),
        (4096, 262144, 4096),
        (65537, 262144, 131072),
        (200000, 200064, 200064),
    ],
)
def test_sm103_decode_length_bucket(seq_len: int, capacity: int, expected: int):
    assert backend._sm103_decode_length_bucket(seq_len, capacity) == expected


def test_sm103_decode_length_bucket_rejects_overflow():
    with pytest.raises(ValueError, match="exceeds block-table capacity"):
        backend._sm103_decode_length_bucket(1025, 1024)


@pytest.mark.parametrize(
    ("seq_len", "expected"),
    [(1, 64), (64, 64), (65, 128), (1025, 2048)],
)
def test_sm103_decode_length_bucket_supports_block64(seq_len: int, expected: int):
    assert backend._sm103_decode_length_bucket(seq_len, 4096, 64) == expected


def test_sm103_graph_compile_bucket_covers_full_replay_capacity():
    assert (
        backend._sm103_decode_compile_bucket(
            128,
            262144,
            cuda_graph_enabled=True,
        )
        == 262144
    )
    assert (
        backend._sm103_decode_compile_bucket(
            1025,
            262144,
            cuda_graph_enabled=False,
        )
        == 2048
    )
    assert (
        backend._sm103_decode_compile_bucket(
            65,
            4096,
            cuda_graph_enabled=True,
            block_size=64,
        )
        == 4096
    )


@pytest.mark.parametrize(
    (
        "max_num_seqs",
        "token_cap",
        "model_len",
        "num_heads",
        "num_kv_heads",
        "expected",
    ),
    [
        (2, 256, 256, 32, 4, ((1, 1, 1), (17, 17, 1))),
        (2, 256, 256, 32, 8, ((1, 1, 1), (190, 95, 4))),
        (32, 8192, 256, 32, 4, ((1, 1, 1), (368, 23, 4))),
        (32, 8192, 256, 32, 8, ((1, 1, 1), (190, 38, 4))),
        (32, 128, 256, 32, 8, ((1, 1, 1), (33, 33, 1))),
        (32, 8192, 256, 32, 2, ()),
    ],
)
def test_packed_prefill_prewarm_shapes_cover_reachable_binaries(
    max_num_seqs,
    token_cap,
    model_len,
    num_heads,
    num_kv_heads,
    expected,
):
    assert (
        backend._beyond_packed_prefill_prewarm_shapes(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=token_cap,
            max_model_len=model_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("max_tokens", "num_kv_heads", "write_dq", "expected"),
    [
        (256, 4, False, ((1, 1),)),
        (512, 4, False, ((1, 1), (307, 4))),
        (256, 8, False, ((1, 1), (164, 4))),
        (256, 8, True, ((1, 1), (190, 4))),
    ],
)
def test_writer_prewarm_counts_cover_each_reachable_launch_policy(
    max_tokens,
    num_kv_heads,
    write_dq,
    expected,
):
    assert (
        backend._beyond_packed_writer_prewarm_token_counts(
            max_tokens=max_tokens,
            num_kv_heads=num_kv_heads,
            write_dq=write_dq,
        )
        == expected
    )


@pytest.mark.parametrize(
    (
        "model_dtype",
        "max_model_len",
        "max_num_seqs",
        "max_dense_tokens",
        "num_kv_heads",
        "expected",
    ),
    [
        (
            torch.bfloat16,
            1024,
            2,
            262144,
            8,
            ((1, 1, 128, False, 8, 4, True),),
        ),
        (
            torch.float16,
            1024,
            2,
            262144,
            8,
            ((1, 1, 128, False, 8, 4, True),),
        ),
        (
            torch.bfloat16,
            262144,
            8,
            262144,
            8,
            (
                (1, 1, 128, False, 8, 4, True),
                (8, 513, 1024, True, 4, 1, True),
                (8, 2049, 4096, True, 4, 1, False),
            ),
        ),
        (
            torch.bfloat16,
            262144,
            8,
            524288,
            4,
            (
                (1, 1, 128, False, 8, 4, True),
                (8, 1025, 2048, True, 4, 1, True),
                (8, 8193, 16384, True, 4, 1, False),
            ),
        ),
    ],
)
def test_readback_prewarm_cases_deduplicate_length_buckets(
    model_dtype,
    max_model_len,
    max_num_seqs,
    max_dense_tokens,
    num_kv_heads,
    expected,
):
    assert (
        backend._beyond_packed_readback_prewarm_cases(
            model_dtype=model_dtype,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_dense_tokens=max_dense_tokens,
            num_kv_heads=num_kv_heads,
        )
        == expected
    )


@pytest.mark.parametrize(
    (
        "model_dtype",
        "batch_size",
        "max_seq_bucket",
        "num_heads",
        "tokens_per_program",
        "expected_flat",
    ),
    [
        (torch.float16, 1, 16384, 4, 8, True),
        (torch.float16, 2, 16384, 4, 8, False),
        (torch.float16, 1, 16384, 8, 8, False),
        (torch.bfloat16, 1, 16384, 8, 4, True),
        (torch.bfloat16, 1, 32768, 8, 4, False),
        (torch.bfloat16, 1, 131072, 4, 4, False),
        (torch.bfloat16, 1, 262144, 4, 4, True),
    ],
)
def test_readback_grid_policy_uses_only_gated_2d_shapes(
    model_dtype,
    batch_size,
    max_seq_bucket,
    num_heads,
    tokens_per_program,
    expected_flat,
):
    assert (
        backend._select_packed_readback_use_flat_grid(
            model_dtype=model_dtype,
            batch_size=batch_size,
            max_seq_bucket=max_seq_bucket,
            num_heads=num_heads,
            tokens_per_program=tokens_per_program,
        )
        is expected_flat
    )


@pytest.mark.parametrize(
    (
        "max_model_len",
        "token_cap",
        "num_heads",
        "num_kv_heads",
        "num_sms",
        "expected",
    ),
    [
        (512, 256, 32, 8, 148, ()),
        (
            1024,
            256,
            32,
            8,
            148,
            ((2, 513, 2, 1, 5), (33, 513, 2, 2, 5)),
        ),
        (
            8192,
            256,
            32,
            4,
            148,
            (
                (2, 513, 2, 1, 5),
                (17, 513, 2, 2, 5),
                (2, 4097, 33, 1, 6),
            ),
        ),
        (8192, 16, 32, 4, 148, ((2, 513, 2, 1, 5), (2, 4097, 33, 1, 6))),
    ],
)
def test_chunked_prefill_fa4_prewarm_cases_cover_compile_signatures(
    max_model_len,
    token_cap,
    num_heads,
    num_kv_heads,
    num_sms,
    expected,
):
    assert (
        backend._beyond_chunked_prefill_fa4_prewarm_cases(
            max_model_len=max_model_len,
            max_num_batched_tokens=token_cap,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_sms=num_sms,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("computed", "prompt", "num_reqs", "expected"),
    [
        ([255, 256], [256, 256], 2, True),
        ([256, 257], [256, 256], 2, False),
        ([256, 256, 0], [256, 256, 1024], 2, False),
    ],
)
def test_prefix_tail_graph_boundary(computed, prompt, num_reqs, expected):
    input_batch = SimpleNamespace(
        num_computed_tokens_cpu=computed,
        num_prompt_tokens=prompt,
    )

    assert backend._beyond_has_pending_prompt_tokens(input_batch, num_reqs) is expected


def test_slot_mapping_prewarm_covers_b1_dense_decode_derivation(monkeypatch):
    class FakeTensor:
        def __init__(self, shape, strides):
            self.shape = shape
            self._strides = strides

        def stride(self, dim):
            return self._strides[dim]

    prompt_calls = []
    dense_calls = []
    synchronizations = []
    table = FakeTensor((4, 4096), (4096, 1))
    physical_block_table = SimpleNamespace(
        block_size=64,
        block_table=SimpleNamespace(gpu=table),
    )
    block_table = SimpleNamespace(
        block_tables=[physical_block_table],
        compute_slot_mapping=lambda *args: prompt_calls.append(args),
    )
    runner = SimpleNamespace(
        device="cuda",
        input_batch=SimpleNamespace(block_table=block_table),
    )

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                dense_calls.append((grid, args, kwargs))

            return launch

    def fake_tensor(values, **_kwargs):
        return FakeTensor((len(values),), (1,))

    def fake_vector(size, **_kwargs):
        return FakeTensor((int(size),), (1,))

    monkeypatch.setattr(backend, "_TRITON_OK", True)
    monkeypatch.setattr(
        backend,
        "_derive_dense_decode_slot_mapping_kernel",
        FakeKernel(),
    )
    monkeypatch.setattr(backend.torch, "tensor", fake_tensor)
    monkeypatch.setattr(backend.torch, "zeros", fake_vector)
    monkeypatch.setattr(backend.torch, "ones", fake_vector)
    monkeypatch.setattr(backend.torch, "empty", fake_vector)
    monkeypatch.setattr(backend.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        backend.torch.cuda,
        "synchronize",
        lambda device: synchronizations.append(device),
    )

    backend._beyond_prewarm_slot_mapping(runner)

    assert len(prompt_calls) == 2
    assert len(dense_calls) == 1
    grid, args, kwargs = dense_calls[0]
    assert grid == (1,)
    assert args[1] is table
    assert kwargs["BLOCK_TABLE_WIDTH"] == 4096
    assert kwargs["BLOCK_SIZE"] == 64
    assert kwargs["num_warps"] == 1
    assert len(synchronizations) == 1
    assert runner._beyond_slot_mapping_prewarmed


@pytest.mark.parametrize(
    ("is_moe", "expected"),
    [(False, False), (True, True)],
)
def test_moe_full_decode_graph_policy_uses_model_config(is_moe, expected):
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(is_moe=is_moe))

    assert backend._beyond_moe_requires_padded_full_decode(vllm_config) is expected


@pytest.mark.parametrize(
    ("is_moe", "max_model_len", "expected"),
    [
        (False, 262144, False),
        (True, 131072, False),
        (True, 131073, True),
        (True, 262144, True),
    ],
)
def test_native_long_context_moe_uses_bounded_full_graph_profile(
    monkeypatch,
    is_moe,
    max_model_len,
    expected,
):
    monkeypatch.setattr(backend, "BEYOND_MOE_FULL_GRAPH_MAX_MODEL_LEN", 131072)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_moe=is_moe,
            max_model_len=max_model_len,
        )
    )

    assert backend._beyond_long_context_moe_uses_bounded_full_graph_profile(vllm_config) is expected


def test_fp16_moe_automatically_uses_flashinfer_cutlass():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(is_moe=True, dtype=torch.float16),
        kernel_config=SimpleNamespace(moe_backend="auto"),
    )

    assert backend._beyond_apply_moe_dtype_policy(vllm_config)
    assert vllm_config.kernel_config.moe_backend == "flashinfer_cutlass"


@pytest.mark.parametrize(
    ("dtype", "is_moe", "selected"),
    [
        (torch.bfloat16, True, "auto"),
        (torch.float16, False, "auto"),
        (torch.float16, True, "triton"),
    ],
)
def test_moe_dtype_policy_preserves_supported_or_explicit_backend(
    dtype,
    is_moe,
    selected,
):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(is_moe=is_moe, dtype=dtype),
        kernel_config=SimpleNamespace(moe_backend=selected),
    )

    assert not backend._beyond_apply_moe_dtype_policy(vllm_config)
    assert vllm_config.kernel_config.moe_backend == selected


def test_moe_full_decode_graph_keys_drop_only_unsafe_batch_one():
    graph_key = namedtuple("GraphKey", ("num_tokens",))
    keys = {graph_key(1), graph_key(2), graph_key(4), graph_key(8)}

    selected = backend._beyond_select_full_decode_graph_keys(
        keys,
        full_decode_batch_cap=4,
        pad_single_token=True,
    )

    assert {key.num_tokens for key in selected} == {2, 4}


def test_dense_full_decode_graph_keys_retain_exact_batch_one():
    graph_key = namedtuple("GraphKey", ("num_tokens",))
    keys = {graph_key(1), graph_key(2), graph_key(4)}

    selected = backend._beyond_select_full_decode_graph_keys(
        keys,
        full_decode_batch_cap=2,
        pad_single_token=False,
    )

    assert {key.num_tokens for key in selected} == {1, 2}


def test_moe_full_decode_graph_keys_fail_closed_without_safe_bucket():
    graph_key = namedtuple("GraphKey", ("num_tokens",))
    selected = backend._beyond_select_full_decode_graph_keys(
        {graph_key(1)},
        full_decode_batch_cap=1,
        pad_single_token=True,
    )

    assert selected == set()


def test_direct_cute_graph_buffers_are_retained_once_per_capture(monkeypatch):
    impl = backend.BeyondPackedImpl.__new__(backend.BeyondPackedImpl)
    monkeypatch.setattr(backend, "_cuda_is_capturing", lambda: True)
    first = torch.empty(4)
    second = torch.empty(4)

    impl._retain_sm103_cudagraph_buffers(first, second)
    impl._retain_sm103_cudagraph_buffers(first, second)

    assert len(impl._sm103_cudagraph_keepalive) == 1
    retained = next(iter(impl._sm103_cudagraph_keepalive.values()))
    assert retained[0] is first
    assert retained[1] is second


def test_dense_decode_slot_mapping_cache_is_cleared_with_request_plan():
    metadata = SimpleNamespace(
        _beyond_dense_decode_slot_mapping_gpu_i32=torch.empty(4, dtype=torch.int32),
        _beyond_dense_decode_slot_mapping_gpu_i32_signature=(4, 128),
    )

    backend._clear_beyond_index_caches(metadata)

    assert not hasattr(metadata, "_beyond_dense_decode_slot_mapping_gpu_i32")
    assert not hasattr(metadata, "_beyond_dense_decode_slot_mapping_gpu_i32_signature")


def test_dense_decode_plan_retains_zero_length_graph_padding():
    metadata = SimpleNamespace()

    backend._attach_beyond_plan(
        metadata,
        num_reqs=4,
        num_actual_tokens=4,
        q_starts=[0, 1, 2, 3, 4],
        seq_lens=[1024, 0, 0, 0],
        max_query_len=1,
        max_seq_len=1024,
        is_prefilling_list=None,
    )

    assert metadata._beyond_single_token_decode_batch
    assert metadata._beyond_decode_req_list == [0, 1, 2, 3]
    assert metadata._beyond_q_indices_dense


def test_dense_prefix_cache_tail_is_decode_kernel_eligible():
    metadata = SimpleNamespace()

    backend._attach_beyond_plan(
        metadata,
        num_reqs=2,
        num_actual_tokens=2,
        q_starts=[0, 1, 2],
        seq_lens=[1024, 1024],
        max_query_len=1,
        max_seq_len=1024,
        is_prefilling_list=[True, True],
    )

    assert not metadata._beyond_single_token_decode_batch
    assert metadata._beyond_decode_req_list == [0, 1]
    assert backend._beyond_dense_decode_batch_eligible(
        num_reqs=2,
        num_actual_tokens=2,
        prefill_reqs=metadata._beyond_prefill_req_list,
        chunked_prefill_reqs=metadata._beyond_chunked_prefill_req_list,
        decode_reqs=metadata._beyond_decode_req_list,
        q_indices_dense=metadata._beyond_q_indices_dense,
    )


def test_dense_decode_plan_does_not_reclassify_first_token_prefill():
    metadata = SimpleNamespace()

    backend._attach_beyond_plan(
        metadata,
        num_reqs=2,
        num_actual_tokens=2,
        q_starts=[0, 1, 2],
        seq_lens=[1, 0],
        max_query_len=1,
        max_seq_len=1,
        is_prefilling_list=None,
    )

    assert not metadata._beyond_single_token_decode_batch
    assert metadata._beyond_prefill_req_list == [0]
