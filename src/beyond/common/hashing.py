"""Small, dependency-free hashing primitives shared across domains."""

from __future__ import annotations

import hashlib
from os import PathLike
from pathlib import Path
from typing import BinaryIO

DEFAULT_BLOCK_SIZE = 1024 * 1024


def sha256_stream(handle: BinaryIO, *, block_size: int = DEFAULT_BLOCK_SIZE) -> str:
    """Return the SHA-256 digest of a binary stream from its current position."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(block_size), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str | PathLike[str], *, block_size: int = DEFAULT_BLOCK_SIZE) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    with Path(path).open("rb") as handle:
        return sha256_stream(handle, block_size=block_size)


__all__ = ["DEFAULT_BLOCK_SIZE", "sha256_file", "sha256_stream"]
