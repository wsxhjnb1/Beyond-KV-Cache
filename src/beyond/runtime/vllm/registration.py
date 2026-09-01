"""vLLM registry wiring for the Beyond packed attention backend."""

from __future__ import annotations


def register_backend() -> None:
    """Register the CUSTOM backend and install its vLLM compatibility hooks."""
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        from vllm.v1.attention.backends.registry import register_backend as register
    except ImportError:
        return  # Older vLLM versions do not expose the registry API.

    import torch

    from beyond.runtime.vllm import packed_backend as backend

    backend.refresh_env_tunables()
    qualname = (
        f"{backend.BeyondPackedBackend.__module__}.{backend.BeyondPackedBackend.__qualname__}"
    )
    register(AttentionBackendEnum.CUSTOM, qualname)
    backend._patch_vllm_beyond_scheduler_policy()
    backend._patch_vllm_copy_and_call_static_buffers()
    backend._patch_vllm_aot_replay_warmup()

    # The packed int32 layout is semantically one full-attention cache per
    # layer, so vLLM's stock FullAttentionManager remains the correct owner.
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        spec_manager_map,
    )

    spec_manager_map.setdefault(backend.BeyondPackedSpec, FullAttentionManager)

    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.backend import AttentionType

    if not getattr(Attention.get_kv_cache_spec, "_beyond_patched", False):
        original_get_kv_cache_spec = Attention.get_kv_cache_spec

        def get_kv_cache_spec_with_beyond(self, vllm_config):
            if self.attn_backend.get_name() == "CUSTOM":
                assert self.attn_type == AttentionType.DECODER, (
                    "BeyondPacked only supports DECODER attention"
                )
                return backend.BeyondPackedSpec(
                    block_size=vllm_config.cache_config.block_size,
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    head_size_v=self.head_size_v,
                    dtype=torch.int32,
                )
            return original_get_kv_cache_spec(self, vllm_config)

        get_kv_cache_spec_with_beyond._beyond_patched = True
        Attention.get_kv_cache_spec = get_kv_cache_spec_with_beyond

    backend._patch_vllm_decode_graph_buckets()
