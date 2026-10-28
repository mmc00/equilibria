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
