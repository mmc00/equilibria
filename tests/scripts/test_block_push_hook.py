from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gtap" / "claude_hooks" / "block_push_without_gates.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("block_push_without_gates", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_push_target_is_the_git_c_path() -> None:
    """A session started in one worktree pushes another with ``git -C <path> push``:
    the stamp to check is that worktree's, not the hook's own."""
    module = _load_module()

    cmd = "git -C /work/family-phase0 push -u origin perf/x"
    assert module.target_dir(cmd, "/work/jax") == Path("/work/family-phase0")


def test_the_push_target_follows_a_leading_cd() -> None:
    module = _load_module()

    cmd = "cd /work/family-phase0 && gh pr create --base main --fill"
    assert module.target_dir(cmd, "/work/jax") == Path("/work/family-phase0")


def test_a_relative_git_c_path_is_taken_from_the_cwd() -> None:
    module = _load_module()

    assert module.target_dir("git -C ../other push", "/work/jax") == Path(
        "/work/jax/../other"
    )


def test_without_a_path_the_target_is_the_cwd() -> None:
    module = _load_module()

    assert module.target_dir("git push origin main", "/work/jax") == Path("/work/jax")


def test_the_target_is_the_cwd_when_the_command_cannot_be_parsed() -> None:
    module = _load_module()

    assert module.target_dir("git push 'unclosed", "/work/jax") == Path("/work/jax")
