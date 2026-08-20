"""Unit tests for Git transaction safety, working tree checks, and conflict handling."""

import pytest
from pathlib import Path
from src.vault.git_engine import GitEngine


def test_git_engine_initialization_and_clean_commit(tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    engine = GitEngine(vault_path=vault_dir)
    assert engine.repo is not None

    # Check status of new repo
    status = engine.check_working_tree()
    assert isinstance(status, dict)

    # Create a test note
    test_note = vault_dir / "Test.md"
    test_note.write_text("# Test Note\nHello world.", encoding="utf-8")

    # Commit
    res = engine.commit_sync("Initial test commit")
    assert res is True

    # Try committing again when not dirty (should return True without creating empty commit)
    res_clean = engine.commit_sync("No changes commit")
    assert res_clean is True


@pytest.mark.asyncio
async def test_git_engine_transaction(tmp_path):
    vault_dir = tmp_path / "vault_tx"
    vault_dir.mkdir()

    engine = GitEngine(vault_path=vault_dir)

    async with engine.transaction("Transaction update"):
        (vault_dir / "Daily.md").write_text("# Daily Note", encoding="utf-8")

    # Working tree should now be clean after transaction committed
    assert not engine.repo.is_dirty(untracked_files=True)


def test_git_engine_conflict_and_rebase_abort(tmp_path):
    """Test that git engine correctly detects clean tree and handles rebase abort without crashing."""
    vault_dir = tmp_path / "vault_conflict"
    vault_dir.mkdir()

    engine = GitEngine(vault_path=vault_dir)
    engine.commit_sync("Initial commit")

    # Should report no conflicts
    status = engine.check_working_tree()
    assert status["conflicts"] == []
    assert isinstance(status["untracked"], list)

    # Calling abort rebase when no rebase in progress should be a safe no-op
    engine._abort_rebase_if_in_progress(engine.repo)
    assert not engine.repo.is_dirty(untracked_files=True)


def test_git_engine_repair_broken_head(tmp_path):
    """Test that git engine recovers gracefully when HEAD points to a non-existent missing SHA commit."""
    vault_dir = tmp_path / "vault_broken_head"
    vault_dir.mkdir()

    engine = GitEngine(vault_path=vault_dir)
    (vault_dir / "Note1.md").write_text("Hello", encoding="utf-8")
    engine.commit_sync("Initial commit")

    # Manually corrupt the branch ref file to point to a non-existent commit SHA
    fake_sha = "3f4ff950f915e15c5a6b3938bcf743da174ea32f"
    branch_ref_file = vault_dir / ".git" / "refs" / "heads" / engine.branch
    branch_ref_file.write_text(fake_sha + "\n", encoding="utf-8")

    # Verify that repair handles the corrupted HEAD ref file
    repaired = engine._repair_broken_head()
    assert repaired is True

    # Now add a new file and commit - should succeed cleanly without crashing on missing SHA
    (vault_dir / "Note2.md").write_text("World", encoding="utf-8")
    res = engine.commit_sync("Commit after broken HEAD repair")
    assert res is True

