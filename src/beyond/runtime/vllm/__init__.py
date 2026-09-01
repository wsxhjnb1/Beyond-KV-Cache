"""vLLM fake-QDQ and packed-cache integrations.

The package stays lazy so importing :mod:`beyond` never imports vLLM, Triton,
or CUDA-facing modules.
"""

__all__ = [
    "BeyondPackedBackend",
    "BeyondPackedImpl",
    "BeyondPackedSpec",
    "register_backend",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "register_backend":
        from .registration import register_backend

        return register_backend
    from . import packed_backend

    return getattr(packed_backend, name)
