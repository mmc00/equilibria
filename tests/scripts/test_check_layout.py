"""check_layout.py: closed root, lowercase English names, no loose scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "check_layout.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("check_layout", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_root_folder_is_rejected() -> None:
    assert _module().violations(["newdir/x.py"])


def test_new_root_file_is_rejected() -> None:
    assert _module().violations(["notes.txt"])


def test_known_root_entries_pass() -> None:
    paths = ["src/a.py", "README.md", "data/README.md", ".github/workflows/x.yml"]
    assert _module().violations(paths) == []


def test_spanish_name_is_rejected() -> None:
    assert _module().violations(["tests/test_sin_shock.py"])


def test_spanish_folder_is_rejected() -> None:
    assert _module().violations(["docs/resultados_viejos/x.md", "docs/notas/x.md"])


def test_accent_is_rejected() -> None:
    assert _module().violations(["docs/notes/análisis.md"])


def test_word_inside_a_token_is_not_spanish() -> None:
    assert _module().violations(["tests/test_decomp_design.py"]) == []


def test_uppercase_still_rejected() -> None:
    assert _module().violations(["tests/Foo.py"])


def test_loose_script_rejected() -> None:
    assert _module().violations(["scripts/foo.py"])


def test_violation_names_the_rule() -> None:
    (msg,) = _module().violations(["newdir/x.py"])
    assert "root" in msg and "data/README.md" in msg


def test_repo_passes_today() -> None:
    assert _module().main() == 0


def test_tracked_names_keep_their_accents(tmp_path: Path) -> None:
    """git quotes non-ASCII paths unless -z is used (review #4)."""
    import subprocess

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init", "-q")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "año.md").write_text("x")
    git("add", ".")
    mod = _module()
    paths = mod._tracked(tmp_path)
    assert paths == ["docs/año.md"]
    (msg,) = mod.violations(paths)
    assert "not English" in msg


def test_layout_hooks_run_on_windows_too() -> None:
    """`python3` is often missing on Windows: the hooks use pre-commit's python (review #5)."""
    import yaml

    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    ours = {
        h["id"]: h
        for repo in config["repos"]
        for h in repo["hooks"]
        if h["id"] in {"check-layout", "check-commit-msg"}
    }
    assert set(ours) == {"check-layout", "check-commit-msg"}
    for hook in ours.values():
        assert hook["language"] == "python"
        assert hook["entry"].split()[0] == "python"
