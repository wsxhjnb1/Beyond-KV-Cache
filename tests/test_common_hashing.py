from __future__ import annotations

from io import BytesIO

import pytest

from beyond.common.hashing import sha256_file, sha256_stream


def test_sha256_stream_and_file_match(tmp_path):
    payload = b"Beyond\x00quantization\n" * 17
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    assert sha256_stream(BytesIO(payload), block_size=7) == sha256_file(path, block_size=11)


def test_sha256_stream_rejects_nonpositive_block_size():
    with pytest.raises(ValueError, match="positive"):
        sha256_stream(BytesIO(b"payload"), block_size=0)
