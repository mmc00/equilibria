"""The Claude Code PostToolUse hook runs the skills checker on the skill a Write or
Edit just touched (specs/2026-11-26-skills-uber-method-design.md)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "dev"))
sys.path.insert(0, str(ROOT / "tests" / "scripts"))
SCRIPT = ROOT / "scripts" / "dev" / "claude_hooks" / "check_skill_on_edit.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_skill_on_edit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HOOK = _load()

from test_check_skills import make_skill  # noqa: E402  # ty: ignore[unresolved-import]


def _payload(path: Path | str) -> dict:
    return {"tool_name": "Edit", "tool_input": {"file_path": str(path)}}


def _pending(root: Path, *names: str) -> None:
    (root / "scripts" / "dev").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "dev" / "skills_pending.txt").write_text(
        "".join(f"{n}\n" for n in names)
    )


def test_skill_of_maps_agents_and_claude_paths(tmp_path):
    assert HOOK.skill_of(str(tmp_path / ".agents/skills/x/SKILL.md"), tmp_path) == "x"
    assert (
        HOOK.skill_of(str(tmp_path / ".claude/skills/x/evals/prompts.yaml"), tmp_path)
        == "x"
    )
    assert HOOK.skill_of(str(tmp_path / "src/a.py"), tmp_path) is None


def test_findings_on_a_non_pending_skill_exit_2_on_stderr(tmp_path):
    folder = make_skill(tmp_path, fm_name="other")
    _pending(tmp_path)
    code, text = HOOK.run(_payload(folder / "SKILL.md"), tmp_path)
    assert code == 2
    assert "demo: frontmatter:" in text


def test_findings_on_a_pending_skill_exit_0_as_context(tmp_path):
    folder = make_skill(tmp_path, fm_name="other")
    _pending(tmp_path, "demo")
    code, text = HOOK.run(_payload(folder / "SKILL.md"), tmp_path)
    assert code == 0
    out = json.loads(text)["hookSpecificOutput"]
    assert out["hookEventName"] == "PostToolUse"
    assert "demo: frontmatter:" in out["additionalContext"]


def test_clean_skill_exits_0_silent(tmp_path):
    folder = make_skill(tmp_path)
    _pending(tmp_path)
    assert HOOK.run(_payload(folder / "spec.md"), tmp_path) == (0, "")


def test_outside_path_non_skill_or_bad_json_exits_0_silent(
    tmp_path, monkeypatch, capsys
):
    make_skill(tmp_path)
    assert HOOK.run(_payload("/elsewhere/.agents/skills/demo/SKILL.md"), tmp_path) == (
        0,
        "",
    )
    assert HOOK.run(_payload(tmp_path / "src" / "a.py"), tmp_path) == (0, "")
    assert HOOK.run({"tool_input": {}}, tmp_path) == (0, "")
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("not json"))
    assert HOOK.main(root=tmp_path) == 0
    assert capsys.readouterr() == ("", "")


# MAR-257: the final review's deferred minors


def test_a_loose_file_under_the_skills_folder_exits_0_silent(tmp_path):
    make_skill(tmp_path)
    _pending(tmp_path)
    loose = tmp_path / ".agents" / "skills" / "notes.md"
    loose.write_text("notes\n")
    assert HOOK.run(_payload(loose), tmp_path) == (0, "")


@pytest.mark.parametrize("payload", [{"tool_input": "x"}, {"tool_input": ["x"]}])
def test_a_tool_input_that_is_not_a_map_exits_0_silent(tmp_path, payload):
    assert HOOK.run(payload, tmp_path) == (0, "")
