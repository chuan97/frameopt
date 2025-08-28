"""utility functions for scripts."""

import subprocess
import time
from pathlib import Path

__all__ = [
    "timestamp_utc",
    "ensure_dir",
    "git_sha",
    "repo_root_from_here",
    "rel_name_from_config",
    "assert_git_clean",
]


def timestamp_utc() -> str:
    """Return current UTC timestamp formatted as YYYYMMDD_%H%M%S."""
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def ensure_dir(p: Path) -> None:
    """Create directory p and parents if they don't exist."""
    p.mkdir(parents=True, exist_ok=True)


def git_sha(repo_root: Path) -> str:
    """Return full git SHA for repo_root, or 'UNKNOWN' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def assert_git_clean(repo_root: Path) -> None:
    """
    Raise if there are uncommitted *tracked* changes (staged or unstaged).
    Untracked files are ignored to avoid false positives from scratch artifacts.
    """
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"], cwd=str(repo_root), text=True
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repo_root),
            text=True,
        )
        if status.strip():
            raise RuntimeError(
                "Repository has uncommitted tracked changes. "
                "For certifiable experiments, commit or stash changes and retry."
            )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Git check failed. Are you inside a git repo at {repo_root}? {e}"
        ) from e


def repo_root_from_here() -> Path:
    """Two levels up from this file (the repository root)."""
    return Path(__file__).resolve().parents[2]


def rel_name_from_config(path: Path, anchor: str) -> str:
    """
    Derive a stable name from a config file path relative to a known anchor directory.

    Example:
      path = /repo/configs/models/projection/quick.yaml, anchor="configs"
      -> "models/projection/quick"
    """
    try:
        idx = path.parts.index(anchor)
    except ValueError:
        # Fallback: use stem without extension
        return path.with_suffix("").name
    rel = Path(*path.parts[idx + 1 :]).with_suffix("")
    return str(rel).replace("\\", "/")
