"""check_commit_msg.py: commit messages are English (maintainer's rule, 2026-10-28)."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "check_commit_msg.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("check_commit_msg", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_english_message_passes() -> None:
    assert _module().spanish_words("fix(gtap): pair rows by the name dictionary") == []


def test_spanish_message_fails() -> None:
    msg = "fix(gtap): el gate empareja las filas por el diccionario"
    assert _module().spanish_words(msg)


def test_accent_fails() -> None:
    assert _module().spanish_words("docs: añade la guía")


def test_trailer_and_backticks_are_ignored() -> None:
    msg = (
        "test: rename `test_sin_registro_de_bloques`\n\n"
        "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
    )
    assert _module().spanish_words(msg) == []


def test_git_comment_lines_are_ignored() -> None:
    msg = "feat: add check\n# Por favor ingrese el mensaje de los cambios\n"
    assert _module().spanish_words(msg) == []


def test_one_borrowed_word_passes() -> None:
    assert _module().spanish_words("feat: plot sin(x) residuals") == []


def test_msg_file_cli(tmp_path: Path) -> None:
    f = tmp_path / "COMMIT_EDITMSG"
    f.write_text("prueba de los mensajes\n")
    assert _module().main([str(f)]) == 1
    f.write_text("test the messages\n")
    assert _module().main([str(f)]) == 0


def test_range_reports_each_bad_commit(tmp_path: Path, capsys: Any) -> None:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("commit", "-q", "--allow-empty", "-m", "root")
    base = git("rev-parse", "HEAD").strip()
    git("commit", "-q", "--allow-empty", "-m", "feat: add the parser")
    git("commit", "-q", "--allow-empty", "-m", "fix: arregla el lector de los datos")
    bad = git("rev-parse", "--short", "HEAD").strip()
    assert _module().main(["--range", f"{base}..HEAD", "--repo", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert bad in out and "add the parser" not in out


# Lines from real English commits of this repo that the first version flagged
# (review #3): identifiers with `_`, SHA prefixes, a repeated word, an accent.
REAL_ENGLISH = [
    "feat(gtap): build_equations_all_periods — build once, no del/recreate\n\n"
    "replacing the 3x del_component/re-merge in build_equations_intra.",
    "gempack: regenerate fixtures with the standard closure (no dpsave swap)\n\n"
    "emitted by run_gempack_matrix.py as of de5f830 (dpsave exogenous, no swaps).\n"
    "swapped dpsave(r)=del_tbalry(r) for non-residual regions",
    "gtap(capfixdp): converges but does NOT reproduce GEMPACK\n\n"
    "that swap fixes the real del_tbalry (trade balance as %-of-world-income)",
    "gtap(newton_tr): mumps needs zero-free-diagonal column permutation\n\n"
    "solve (J P) y = -F, then un-permute p[colperm]=y. Also y is reused.",
    "docs: pin the numéraire to the world price index",
]


def test_real_english_commits_pass() -> None:
    for msg in REAL_ENGLISH:
        assert _module().spanish_words(msg) == [], msg


def test_spanish_with_common_short_words_fails() -> None:
    assert _module().spanish_words("chore(deps): actualizar ruff a 0.16.7 en el lock")


def test_message_is_read_as_utf8_whatever_the_locale(tmp_path: Path) -> None:
    """A cp1252/latin-1 locale must not hide accents (review #5, Windows)."""
    import os
    import sys

    import pytest

    env = {**os.environ, "LC_ALL": "en_US.ISO8859-1", "PYTHONUTF8": "0"}
    probe = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8=0",
            "-c",
            "import locale; print(locale.getpreferredencoding(False))",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if "8859" not in probe.stdout:
        pytest.skip("no latin-1 locale on this machine")
    f = tmp_path / "msg"
    f.write_bytes("docs: añade la guía\n".encode())
    r = subprocess.run(
        [sys.executable, "-X", "utf8=0", str(SCRIPT), str(f)],
        env=env,
        capture_output=True,
    )
    assert r.returncode == 1
