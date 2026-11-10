"""The documented allowlist formats must load instead of crashing at startup."""

import pytest

from src.config import Settings


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("123456789,987654321", [123456789, 987654321]),
        ("123456789", [123456789]),
        ("[123456789, 987654321]", [123456789, 987654321]),
        (" 1 , -1002 ", [1, -1002]),
        ("", []),
    ],
)
def test_allowlist_formats_load_from_env_file(tmp_path, monkeypatch, raw, expected):
    monkeypatch.delenv("ALLOWED_TELEGRAM_USER_IDS", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text(f"ALLOWED_TELEGRAM_USER_IDS={raw}\n", encoding="utf-8")

    assert Settings(_env_file=env_file).ALLOWED_TELEGRAM_USER_IDS == expected


def test_allowlist_comma_list_loads_from_environment(monkeypatch):
    monkeypatch.setenv("ALLOWED_TELEGRAM_USER_IDS", "11,22")
    assert Settings(_env_file=None).ALLOWED_TELEGRAM_USER_IDS == [11, 22]
