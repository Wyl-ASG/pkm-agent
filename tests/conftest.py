"""Isolate the test suite from real configuration, vaults, indexes, and remotes.

These variables are set before any `src` module is imported. Environment variables
take priority over `.env` and `.env.local`, so tests never touch real credentials,
the real vault, the real Qdrant storage, or the real Git remote.
"""

import os
import tempfile
from pathlib import Path

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="pkm-agent-tests-"))

os.environ.update(
    {
        "VAULT_PATH": str(_TEST_ROOT / "vault"),
        "QDRANT_STORAGE_PATH": str(_TEST_ROOT / "qdrant"),
        "LLM_PROVIDER": "mock",
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_SECRET_TOKEN": "test-secret",
        "ALLOWED_TELEGRAM_USER_IDS": "1",
        "GIT_REPO_URL": "",
        "GIT_BRANCH": "main",
        "SSH_KEY_PATH": "",
        "MFA_SECRET": "",
        "SCHEDULED_BRIEFING_ENABLED": "false",
        "ENABLE_FILE_WATCHER": "false",
    }
)
