"""Vault backup failures must reach the owner instead of failing silently."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import git
import pytest

from src.vault import git_engine as git_engine_module
from src.vault.git_engine import GitEngine


def _configure(repo: git.Repo) -> None:
    repo.git.config("user.email", "test@example.com")
    repo.git.config("user.name", "Test")


def _remote_with_clone(tmp_path: Path, name: str = "server") -> tuple[Path, GitEngine]:
    """Create a bare remote with one commit and a server clone driven by GitEngine."""
    remote = tmp_path / "remote.git"
    git.Repo.init(remote, bare=True, initial_branch="main")
    seed = git.Repo.clone_from(str(remote), tmp_path / "seed")
    _configure(seed)
    (tmp_path / "seed" / "2026-11-10.md").write_text("# Today\n## Log\n- start\n", encoding="utf-8")
    seed.git.add(A=True)
    seed.git.commit("-m", "init")
    seed.git.push("origin", "HEAD:main")

    clone = git.Repo.clone_from(str(remote), tmp_path / name)
    _configure(clone)
    return remote, GitEngine(vault_path=tmp_path / name, branch="main")


def _write(engine: GitEngine, name: str, text: str = "note\n") -> None:
    (engine.vault_path / name).write_text(text, encoding="utf-8")


def test_unreachable_remote_alerts_once_then_reports_recovery(tmp_path):
    remote, engine = _remote_with_clone(tmp_path)
    engine.repo.remotes.origin.set_url(str(tmp_path / "missing.git"))

    _write(engine, "a.md")
    assert engine.commit_and_push_sync("capture 1") is False
    alerts = engine.pop_alerts()
    assert len(alerts) == 1
    assert "Vault backup is failing" in alerts[0]
    assert "only on this server: 1" in alerts[0]

    _write(engine, "b.md")
    assert engine.commit_and_push_sync("capture 2") is False
    assert engine.pop_alerts() == [], "a failure that is already reported must not alert on every capture"
    status = engine.sync_status()
    assert status["failing"] is True and status["unpushed_commits"] == 2

    engine.repo.remotes.origin.set_url(str(remote))
    assert engine.commit_and_push_sync("scheduled retry") is True
    assert engine.pop_alerts() == ["Vault backup recovered: everything on this server is pushed again."]
    assert engine.sync_status() == {
        "remote_configured": True,
        "unpushed_commits": 0,
        "failing": False,
        "last_error": None,
    }
    assert {"a.md", "b.md"} <= set(git.Repo(remote).git.ls_tree("--name-only", "main").splitlines())


def test_conflict_with_another_device_names_the_file(tmp_path):
    remote, engine = _remote_with_clone(tmp_path)
    phone = git.Repo.clone_from(str(remote), tmp_path / "phone")
    _configure(phone)
    with open(tmp_path / "phone" / "2026-11-10.md", "a", encoding="utf-8") as f:
        f.write("- edit from phone\n")
    phone.git.commit("-am", "phone edit")
    phone.git.push("origin", "HEAD:main")

    with open(engine.vault_path / "2026-11-10.md", "a", encoding="utf-8") as f:
        f.write("- capture from telegram\n")
    assert engine.commit_and_push_sync("capture") is False

    alerts = engine.pop_alerts()
    assert len(alerts) == 1
    assert "conflicting edits to 2026-11-10.md" in alerts[0]
    assert "only on this server: 1" in alerts[0]
    assert not (engine.vault_path / ".git" / "rebase-merge").exists()


def test_ongoing_failure_is_repeated_after_a_day(tmp_path, monkeypatch):
    _, engine = _remote_with_clone(tmp_path)
    engine.repo.remotes.origin.set_url(str(tmp_path / "missing.git"))
    clock = [1000.0]
    monkeypatch.setattr(git_engine_module.time, "monotonic", lambda: clock[0])

    _write(engine, "a.md")
    engine.commit_and_push_sync("capture 1")
    assert len(engine.pop_alerts()) == 1

    clock[0] += git_engine_module.SYNC_ALERT_REPEAT_SECONDS - 1
    _write(engine, "b.md")
    engine.commit_and_push_sync("capture 2")
    assert engine.pop_alerts() == []

    clock[0] += 2
    _write(engine, "c.md")
    engine.commit_and_push_sync("capture 3")
    alerts = engine.pop_alerts()
    assert len(alerts) == 1 and "only on this server: 3" in alerts[0]


def test_status_does_not_create_a_repository(tmp_path):
    engine = GitEngine(vault_path=tmp_path / "vault", branch="main")
    (tmp_path / "vault").mkdir()
    assert engine.sync_status()["remote_configured"] is False
    assert not (tmp_path / "vault" / ".git").exists()


def test_error_text_hides_credentials_in_remote_urls():
    err = Exception("Cmd failed\n  stderr: 'fatal: unable to access https://user:secret-token@example.com/v.git/'")
    text = git_engine_module._short_error(err)
    assert "secret-token" not in text
    assert text.startswith("fatal: unable to access https://***@example.com")


@pytest.mark.asyncio
async def test_scheduled_backup_retries_and_messages_the_owner(tmp_path):
    import src.main as main

    remote, engine = _remote_with_clone(tmp_path)
    engine.repo.remotes.origin.set_url(str(tmp_path / "missing.git"))
    _write(engine, "a.md")
    engine.commit_sync("captured while the remote was down")

    sent = AsyncMock()
    with patch.object(main, "git_engine", engine), patch.object(main, "send_telegram_message", sent):
        await main.run_scheduled_vault_backup()
        assert sent.await_count == len(main.settings.ALLOWED_TELEGRAM_USER_IDS)
        assert "Vault backup is failing" in sent.await_args.args[1]

        sent.reset_mock()
        engine.repo.remotes.origin.set_url(str(remote))
        await main.run_scheduled_vault_backup()
        assert "recovered" in sent.await_args.args[1]

        sent.reset_mock()
        await main.run_scheduled_vault_backup()
        sent.assert_not_awaited()

    assert "a.md" in git.Repo(remote).git.ls_tree("--name-only", "main").splitlines()


@pytest.mark.asyncio
async def test_health_reports_vault_backup_state():
    import src.main as main

    response = await main.health_check()
    body = json.loads(response.body)
    assert set(body["components"]["vault_backup"]) == {
        "remote_configured",
        "unpushed_commits",
        "failing",
        "last_error",
    }
