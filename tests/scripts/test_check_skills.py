"""scripts/dev/check_skills.py holds equilibria's skills to Uber's skills method
(specs/2026-11-26-skills-uber-method-design.md)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "scripts" / "dev"
sys.path.insert(0, str(DEV))


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, DEV / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CS = _load("check_skills")

GOOD_STEPS = """## Steps

1. Run the cascade on the case.

   ```bash
   uv run python scripts/gtap/cascade_orchestrator.py --dataset gtap7_3x3
   ```

   Check: the last line names a verdict.
2. Read the first dirty layer with `uv run python scripts/gtap/triage.py`.
   Check: the layer's headline names a family.
"""

GOOD_ERRORS = """## Errors

| Failure | Fix |
|---|---|
| `gams_nl_missing` | regenerate the .nl with `--local-gams` |
| the solve hangs | rerun with `--tool-timeout 600` |
"""

GOOD_SPEC = """# spec

## Goal

Find why a model differs from GAMS.

## Outputs

The first layer that explains the gap.

## Tools

`cascade_orchestrator.py`.

## Forbidden

Loosen a tolerance. Use `.venv/bin/python`.

## Agents

Claude Code (Opus).
"""

GOOD_PROMPTS = """- id: p1
  kind: golden
  prompt: "Is gtap7_3x3 clean?"
  expect: "yes"
- id: p2
  kind: trajectory
  prompt: "Why does 3x3 differ?"
  must_run: ["cascade_orchestrator.py"]
  expect: "names a layer"
- id: p3
  kind: trajectory
  prompt: "Why does 10x7 differ?"
  must_run: ["cascade_orchestrator.py --dataset gtap7_10x7"]
  must_not_run: ["git push"]
  expect: "names a layer"
- id: p4
  kind: trajectory
  prompt: "Check the shock only."
  must_run:
    - "--periods shock"
  expect: "shock verdict"
- id: p5
  kind: author
  prompt: "Explain the verdict to Marlon."
  expect: "plain words"
"""


def make_skill(root: Path, name: str = "demo", **over: Any) -> Path:
    """A skill that meets every rule; ``over`` bends one piece: ``description``,
    ``fm_name``, ``body``, ``spec``, ``prompts`` (None drops the file), ``link``,
    ``gitignore``, ``readme`` (False drops it), ``raw_skill`` (the whole SKILL.md),
    ``stamp`` (False drops evals/last_run.md, a dict overrides its keys)."""
    folder = root / ".agents" / "skills" / name
    (folder / "evals").mkdir(parents=True, exist_ok=True)
    desc = over.get("description", "Use when a model differs from GAMS.")
    body = over.get("body", f"# Demo\n\n{GOOD_STEPS}\n{GOOD_ERRORS}")
    fm_name = over.get("fm_name", name)
    raw = over.get("raw_skill")
    (folder / "SKILL.md").write_text(
        raw
        if raw is not None
        else f"---\nname: {fm_name}\ndescription: {desc}\n---\n{body}"
    )
    spec = over.get("spec", GOOD_SPEC)
    if spec is not None:
        (folder / "spec.md").write_text(spec)
    prompts = over.get("prompts", GOOD_PROMPTS)
    if prompts is not None:
        (folder / "evals" / "prompts.yaml").write_text(prompts)
    claude = root / ".claude" / "skills"
    claude.mkdir(parents=True, exist_ok=True)
    if over.get("link", True) and not (claude / name).exists():
        (claude / name).symlink_to(Path("../../.agents/skills") / name)
    gi = root / ".gitignore"
    lines = gi.read_text().splitlines() if gi.exists() else []
    if over.get("gitignore", True):
        lines.append(f"!/.claude/skills/{name}")
    gi.write_text("\n".join(lines) + "\n")
    readme = root / ".agents" / "skills" / "README.md"
    text = (
        readme.read_text() if readme.exists() else "| Skill | Use when |\n|---|---|\n"
    )
    if over.get("readme", True):
        text += f"| [{name}]({name}/SKILL.md) | demo |\n"
    readme.write_text(text)
    stamp_hook = getattr(CS, "write_test_stamp", None)
    stamp = over.get("stamp", {})
    if stamp is not False and stamp_hook is not None:
        stamp_hook(folder, **stamp)
    return folder


def rules(findings) -> list[str]:
    return sorted(f.rule for f in findings)


def test_a_good_skill_passes(tmp_path):
    make_skill(tmp_path)
    assert CS.check_skill(tmp_path, "demo") == []


def test_name_must_match_folder(tmp_path):
    make_skill(tmp_path, fm_name="other")
    assert rules(CS.check_skill(tmp_path, "demo")) == ["frontmatter"]


def test_description_over_100_words(tmp_path):
    make_skill(tmp_path, description="Use when " + "word " * 100)
    assert rules(CS.check_skill(tmp_path, "demo")) == ["frontmatter"]


def test_description_without_use_when(tmp_path):
    make_skill(tmp_path, description="Debug a model that differs from GAMS.")
    assert rules(CS.check_skill(tmp_path, "demo")) == ["frontmatter"]


def test_no_frontmatter_or_crlf_is_a_finding_not_a_crash(tmp_path):
    make_skill(tmp_path, raw_skill="# Demo\n\nno frontmatter here\n")
    assert "frontmatter" in rules(CS.check_skill(tmp_path, "demo"))
    crlf = (
        "---\r\nname: demo2\r\ndescription: Use when a model differs.\r\n---\r\n"
        f"# Demo\r\n\r\n{GOOD_STEPS}\n{GOOD_ERRORS}".replace("\n", "\r\n")
    )
    make_skill(tmp_path, "demo2", raw_skill=crlf)
    assert "frontmatter" not in rules(CS.check_skill(tmp_path, "demo2"))


def test_body_of_500_lines_fails_and_499_passes(tmp_path):
    base = f"# Demo\n\n{GOOD_STEPS}\n{GOOD_ERRORS}"
    pad = 499 - len(base.splitlines())
    make_skill(tmp_path, "ok", body=base + "x\n" * pad)
    assert CS.check_skill(tmp_path, "ok") == []
    make_skill(tmp_path, "long", body=base + "x\n" * (pad + 1))
    assert rules(CS.check_skill(tmp_path, "long")) == ["length"]


def test_missing_link_gitignore_or_readme_row(tmp_path):
    make_skill(tmp_path, link=False, gitignore=False, readme=False)
    assert rules(CS.check_skill(tmp_path, "demo")) == ["wiring", "wiring", "wiring"]


def _pending(root: Path, *names: str) -> None:
    (root / "scripts" / "dev").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "dev" / "skills_pending.txt").write_text(
        "# not migrated yet\n" + "".join(f"{n}\n" for n in names)
    )


def test_main_exit_codes(tmp_path, capsys):
    make_skill(tmp_path, fm_name="other")
    _pending(tmp_path)
    assert CS.main([], root=tmp_path) == 1
    assert "demo: frontmatter:" in capsys.readouterr().out

    _pending(tmp_path, "demo")
    assert CS.main([], root=tmp_path) == 0
    assert "warning: demo: frontmatter:" in capsys.readouterr().out

    make_skill(tmp_path, "clean")
    _pending(tmp_path, "demo", "clean")
    assert CS.main([], root=tmp_path) == 1
    assert "clean: pending: passes every rule; remove it from pending" in (
        capsys.readouterr().out
    )

    _pending(tmp_path, "demo", "elsewhere")
    assert CS.main([], root=tmp_path) == 0
    assert "warning: elsewhere: pending: not in the repo yet" in capsys.readouterr().out


def test_main_checks_only_named_skills(tmp_path, capsys):
    make_skill(tmp_path, "good")
    make_skill(tmp_path, "bad", fm_name="nope")
    _pending(tmp_path)
    assert CS.main(["good"], root=tmp_path) == 0
    assert capsys.readouterr().out == ""
