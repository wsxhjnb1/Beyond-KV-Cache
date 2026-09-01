from __future__ import annotations

import importlib

import pytest

from beyond.runtime.vllm import config


@pytest.mark.parametrize("value", ["1", "true", "YES", "enabled"])
def test_env_bool_accepts_enabled_spellings(monkeypatch, value):
    monkeypatch.setenv("BEYOND_TEST_BOOL", value)
    assert config.env_bool("BEYOND_TEST_BOOL", False) is True


@pytest.mark.parametrize("value", ["0", "false", "No", "disabled", "none"])
def test_env_bool_accepts_disabled_spellings(monkeypatch, value):
    monkeypatch.setenv("BEYOND_TEST_BOOL", value)
    assert config.env_bool("BEYOND_TEST_BOOL", True) is False


def test_env_readers_use_defaults_for_missing_or_invalid_values(monkeypatch):
    monkeypatch.delenv("BEYOND_TEST_VALUE", raising=False)
    assert config.env_int("BEYOND_TEST_VALUE", 17) == 17
    assert config.env_bool("BEYOND_TEST_VALUE", True) is True
    monkeypatch.setenv("BEYOND_TEST_VALUE", "not-an-int")
    assert config.env_int("BEYOND_TEST_VALUE", 19) == 19


def test_env_bits_and_real_layout_fail_closed(monkeypatch):
    monkeypatch.setenv("BEYOND_TEST_BITS", "4")
    assert config.env_bits("BEYOND_TEST_BITS") == 4
    monkeypatch.setenv("BEYOND_TEST_BITS", "8")
    with pytest.raises(ValueError, match="supports bits"):
        config.env_bits("BEYOND_TEST_BITS")

    config.validate_real_packed_layout(4, 32, 32)
    with pytest.raises(NotImplementedError, match="only 4-bit"):
        config.validate_real_packed_layout(3, 32, 32)
    with pytest.raises(NotImplementedError, match="k_group_size"):
        config.validate_real_packed_layout(4, 64, 32)
    with pytest.raises(NotImplementedError, match="v_group_size"):
        config.validate_real_packed_layout(4, 32, 64)


def test_registration_module_import_does_not_require_vllm():
    module = importlib.import_module("beyond.runtime.vllm.registration")
    assert callable(module.register_backend)
