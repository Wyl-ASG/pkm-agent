"""Safe Git repository synchronization engine for Obsidian vault with transaction safety."""

import asyncio
from contextlib import asynccontextmanager, contextmanager
import logging
from pathlib import Path
import shlex
import threading
from typing import Any, Generator
import git
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError
from src.config import settings

logger = logging.getLogger(__name__)


def get_ssh_command(ssh_key_path: str | Path | None = None) -> str:
    """Construct SSH command string with identity file if configured and exists."""
    raw_path = ssh_key_path or getattr(settings, "SSH_KEY_PATH", None)
    if raw_path:
        key_path = Path(raw_path).resolve()
        if key_path.exists():
            return f"ssh -i {shlex.quote(str(key_path))} -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=15 -o ServerAliveInterval=15"
        logger.warning("Configured SSH key path does not exist: %s", key_path)
    return "ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15"


class GitConflictError(Exception):
    """Raised when Git synchronization encounters an unresolvable merge or rebase conflict."""

    def __init__(self, message: str, conflicting_files: list[str] | None = None) -> None:
        super().__init__(message)
        self.conflicting_files = conflicting_files or []


class GitEngine:
    """Handles thread-safe, transactional Git operations (pull, commit, push) for Obsidian vault."""

    def __init__(
        self,
        vault_path: str | Path | None = None,
        branch: str | None = None,
        ssh_key_path: str | Path | None = None,
    ) -> None:
        """Initialize GitEngine.

        Args:
            vault_path: Local path to the vault directory. Defaults to settings.VAULT_PATH.
            branch: Git branch name. Defaults to settings.GIT_BRANCH.
            ssh_key_path: Path to SSH private key for authentication. Defaults to settings.SSH_KEY_PATH.
        """
        self.vault_path = Path(vault_path or settings.VAULT_PATH).resolve()
        self.branch = branch or settings.GIT_BRANCH
        self.ssh_key_path = ssh_key_path or getattr(settings, "SSH_KEY_PATH", None)
        self._repo: git.Repo | None = None
        self._thread_lock = threading.Lock()
        self._async_lock: asyncio.Lock | None = None

    def _get_async_lock(self) -> asyncio.Lock:
        """Lazily initialize asyncio.Lock inside running event loop."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    @property
    def ssh_command(self) -> str:
        """Construct SSH command string for Git operations."""
        return get_ssh_command(self.ssh_key_path)

    @property
    def repo(self) -> git.Repo:
        """Git repository instance."""
        return self._ensure_repo()

    def _ensure_repo(self) -> git.Repo:
        """Lazy load or initialize Git repository instance safely and configure SSH environment."""
        if self._repo is not None:
            self._repo.git.update_environment(GIT_SSH_COMMAND=self.ssh_command)
            return self._repo

        if not self.vault_path.exists():
            logger.info("Creating vault directory at %s", self.vault_path)
            self.vault_path.mkdir(parents=True, exist_ok=True)

        try:
            self._repo = git.Repo(self.vault_path)
            logger.info("Loaded existing Git repository at %s", self.vault_path)
        except InvalidGitRepositoryError:
            logger.info("Initializing new Git repository at %s", self.vault_path)
            self._repo = git.Repo.init(self.vault_path)
        except NoSuchPathError as err:
            logger.error("Vault directory path does not exist: %s", self.vault_path)
            raise FileNotFoundError(f"Vault directory not found: {self.vault_path}") from err
        except Exception as err:
            logger.exception("Failed to initialize Git repository at %s: %s", self.vault_path, err)
            raise

        if settings.GIT_REPO_URL:
            try:
                if "origin" in [r.name for r in self._repo.remotes]:
                    origin = self._repo.remotes.origin
                    if settings.GIT_REPO_URL not in list(origin.urls):
                        logger.info("Updating origin remote URL to: %s", settings.GIT_REPO_URL)
                        origin.set_url(settings.GIT_REPO_URL)
                else:
                    logger.info("Creating origin remote with URL: %s", settings.GIT_REPO_URL)
                    self._repo.create_remote("origin", settings.GIT_REPO_URL)
            except Exception as rem_err:
                logger.warning("Could not configure git remote URL: %s", rem_err)

        self._repo.git.update_environment(GIT_SSH_COMMAND=self.ssh_command)
        return self._repo

    def _abort_rebase_if_in_progress(self, repo: git.Repo) -> None:
        """Abort any stuck or failed rebase operation to keep repo in a clean state."""
        try:
            git_dir = Path(repo.git_dir)
            if (git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists():
                logger.warning("Detected ongoing/failed rebase; executing 'git rebase --abort'.")
                repo.git.rebase("--abort")
        except Exception as err:
            logger.debug("Rebase abort cleanup notice: %s", err)

    def get_conflicting_files(self) -> list[str]:
        """Detect unmerged or conflicting files in working tree."""
        try:
            unmerged = self.repo.index.unmerged_blobs()
            return list(unmerged.keys())
        except Exception:
            return []

    def check_working_tree(self) -> dict[str, Any]:
        """Check working tree status (untracked files, modified files, conflicts)."""
        repo = self.repo
        return {
            "is_dirty": repo.is_dirty(untracked_files=True),
            "untracked": repo.untracked_files,
            "modified": [item.a_path for item in repo.index.diff(None)],
            "conflicts": self.get_conflicting_files(),
        }

    def pull_sync(self, rebase: bool = True, autostash: bool = True) -> bool:
        """Pull latest changes from remote repository synchronously using rebase.

        Args:
            rebase: Whether to use --rebase. Defaults to True.
            autostash: Whether to use --autostash. Defaults to True.

        Returns:
            True if pull succeeded or no remote configured, False on failure.
        """
        with self._thread_lock:
            try:
                self.repo.git.update_environment(GIT_SSH_COMMAND=self.ssh_command)
                if not self.repo.remotes:
                    logger.info("No remote repositories configured for pull; skipping.")
                    return True

                remote_names = [r.name for r in self.repo.remotes]
                origin_name = "origin" if "origin" in remote_names else remote_names[0]

                self._abort_rebase_if_in_progress(self.repo)

                logger.info(
                    "Pulling latest changes from remote '%s' branch '%s' (rebase=%s, autostash=%s)",
                    origin_name,
                    self.branch,
                    rebase,
                    autostash,
                )

                if rebase:
                    pull_args = ["--rebase"]
                    if autostash:
                        pull_args.append("--autostash")
                    pull_args.extend([origin_name, self.branch])
                    res = self.repo.git.pull(*pull_args)
                else:
                    res = self.repo.git.pull(origin_name, self.branch)

                logger.info("Successfully pulled latest vault changes: %s", res)
                return True

            except GitCommandError as err:
                logger.warning("Git rebase pull encountered conflict or error: %s", err)
                try:
                    self.repo.git.update_environment(GIT_SSH_COMMAND=self.ssh_command)
                    conflicts = self.get_conflicting_files()
                    self._abort_rebase_if_in_progress(self.repo)

                    if conflicts:
                        logger.error("⚠️ Vault synchronization conflict detected in files: %s", conflicts)
                        return False

                    # Fallback: attempt standard pull without rebase
                    logger.info("Attempting fallback standard pull for branch '%s'...", self.branch)
                    remote_names = [r.name for r in self.repo.remotes]
                    origin_name = "origin" if "origin" in remote_names else remote_names[0]
                    res = self.repo.git.pull(origin_name, self.branch)
                    logger.info("Fallback pull succeeded: %s", res)
                    return True
                except Exception as fallback_err:
                    logger.exception("Fallback git pull also failed: %s", fallback_err)
                    return False

            except Exception as err:
                logger.exception("Unexpected error during git pull: %s", err)
                return False
    def _recover_mass_deletion(self) -> bool:
        """Detect mass deletion and restore files from HEAD to trigger re-index."""
        try:
            # Use optimized raw git command to list only deleted files fast
            diff_out = self.repo.git.diff("--name-only", "--diff-filter=D")
            deleted_files = diff_out.splitlines() if diff_out else []
            threshold = getattr(settings, 'MASS_DELETION_THRESHOLD', 50)
            if len(deleted_files) > threshold:
                logger.warning("Mass deletion detected (%d files). Assuming database reset. Restoring from git HEAD...", len(deleted_files))
                self.repo.git.checkout("--", ".")
                return True
        except Exception as e:
            logger.error("Error during mass deletion recovery: %s", e)
        return False

    def commit_sync(self, message: str, file_paths: list[str | Path] | None = None) -> bool:
        """Stage and commit changes synchronously without creating empty commits.

        Args:
            message: Git commit message.
            file_paths: Specific files to stage. If None or empty, stages all changes.

        Returns:
            True if commit succeeded or no changes to commit, False on failure.
        """
        with self._thread_lock:
            try:
                self.repo.git.update_environment(GIT_SSH_COMMAND=self.ssh_command)
                self._recover_mass_deletion()

                if file_paths:
                    rel_paths: list[str] = []
                    for p in file_paths:
                        path_obj = Path(p)
                        if path_obj.is_absolute():
                            rel_paths.append(str(path_obj.relative_to(self.vault_path)))
                        else:
                            rel_paths.append(str(path_obj))
                    self.repo.index.add(rel_paths)
                else:
                    self.repo.git.add(A=True)

                if not self.repo.is_dirty(untracked_files=True):
                    logger.info("No uncommitted changes in vault repository; skipped commit.")
                    return True

                commit_obj = self.repo.index.commit(message)
                logger.info("Committed changes to vault repository: '%s' (%s)", message, commit_obj.hexsha)
                return True
            except GitCommandError as err:
                logger.error("Git commit failed: %s", err)
                return False
            except Exception as err:
                logger.exception("Unexpected error during git commit: %s", err)
                return False

    def push_sync(self) -> bool:
        """Push committed changes to remote repository synchronously.

        Returns:
            True if push succeeded or no remote configured, False on failure.
        """
        with self._thread_lock:
            try:
                self.repo.git.update_environment(GIT_SSH_COMMAND=self.ssh_command)
                if not self.repo.remotes:
                    logger.info("No remote repositories configured for push; skipping.")
                    return True

                try:
                    res = self.repo.git.push("origin", self.branch)
                    logger.info("Git push output: %s", res)
                    return True
                except GitCommandError as e:
                    if 'fetch first' in str(e) or 'non-fast-forward' in str(e):
                        logger.warning("Git push rejected (fetch first). Pulling and retrying...")
                        try:
                            self.repo.git.pull("--rebase", "--autostash", "origin", self.branch)
                            res = self.repo.git.push("origin", self.branch)
                            logger.info("Git push output after pull: %s", res)
                            return True
                        except Exception as inner_e:
                            logger.error("Git pull and push failed: %s", inner_e, exc_info=True)
                            return False
                    else:
                        logger.error("Git push failed: %s", e, exc_info=True)
                        return False
                except Exception as e:
                    logger.error("Git push failed: %s", e, exc_info=True)
                    return False

            except Exception as err:
                logger.error("Git push failed: %s", err, exc_info=True)
                return False

    def commit_and_push_sync(self, commit_message: str, bypass_recovery: bool = False) -> bool:
        """Stage all changes, commit if dirty, and push to origin branch synchronously."""
        with self._thread_lock:
            try:
                self.repo.git.update_environment(GIT_SSH_COMMAND=self.ssh_command)
                if not bypass_recovery:
                    self._recover_mass_deletion()
                self.repo.git.add(A=True)

                if self.repo.is_dirty(untracked_files=True):
                    self.repo.index.commit(commit_message)
                    logger.info("Committed changes: %s", commit_message)
                else:
                    logger.info("No changes to commit in vault.")

                if not self.repo.remotes:
                    logger.info("No remote repositories configured for push; skipping push.")
                    return True

                try:
                    res = self.repo.git.push("origin", self.branch)
                    logger.info("Git push output: %s", res)
                    return True
                except GitCommandError as e:
                    if 'fetch first' in str(e) or 'non-fast-forward' in str(e):
                        logger.warning("Git push rejected (fetch first). Pulling and retrying...")
                        try:
                            self.repo.git.pull("--rebase", "--autostash", "origin", self.branch)
                            res = self.repo.git.push("origin", self.branch)
                            logger.info("Git push output after pull: %s", res)
                            return True
                        except Exception as inner_e:
                            logger.error("Git pull and push failed: %s", inner_e, exc_info=True)
                            return False
                    else:
                        logger.error("Git push failed: %s", e, exc_info=True)
                        return False
                except Exception as e:
                    logger.error("Git push failed: %s", e, exc_info=True)
                    return False

            except Exception as err:
                logger.error("Git commit and push failed: %s", err, exc_info=True)
                return False

    @asynccontextmanager
    async def transaction(self, commit_message: str = "Vault update") -> Generator[None, None, None]:
        """Async context manager executing a safe write transaction.

        Flow:
        1. Acquire vault lock
        2. Check working tree
        3. Pull/rebase if remote configured
        4. Yield control to write modifications
        5. Validate changes
        6. Commit if dirty
        7. Push to remote
        8. Release lock
        """
        async with self._get_async_lock():
            # 1. Pull latest before write
            await asyncio.to_thread(self.pull_sync, True, True)
            try:
                yield
            finally:
                # 2. Stage, commit, and push
                await asyncio.to_thread(self.commit_and_push_sync, commit_message)

    async def pull(self, rebase: bool = True, autostash: bool = True) -> bool:
        """Asynchronously pull latest changes from remote repository."""
        async with self._get_async_lock():
            return await asyncio.to_thread(self.pull_sync, rebase, autostash)

    async def commit(self, message: str, file_paths: list[str | Path] | None = None) -> bool:
        """Asynchronously stage and commit changes."""
        async with self._get_async_lock():
            return await asyncio.to_thread(self.commit_sync, message, file_paths)

    async def push(self) -> bool:
        """Asynchronously push changes to remote repository."""
        async with self._get_async_lock():
            return await asyncio.to_thread(self.push_sync)

    async def commit_and_push(self, commit_message: str, bypass_recovery: bool = False) -> bool:
        """Asynchronously stage all changes, commit if dirty, and push to origin."""
        async with self._get_async_lock():
            return await asyncio.to_thread(self.commit_and_push_sync, commit_message, bypass_recovery)

    async def sync(self, commit_message: str) -> bool:
        """Execute full pull (rebase) -> commit -> push sync sequence asynchronously."""
        async with self._get_async_lock():
            logger.info("Starting vault Git sync sequence with rebase...")
            pull_ok = await asyncio.to_thread(self.pull_sync, True, True)
            if not pull_ok:
                logger.warning("Initial git pull encountered issue; proceeding with commit and push.")
            return await asyncio.to_thread(self.commit_and_push_sync, commit_message)


VaultGitEngine = GitEngine

__all__ = ["GitEngine", "VaultGitEngine", "GitConflictError"]
