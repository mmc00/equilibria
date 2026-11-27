"""scripts/dev/skill_yaml.py reads the fixed YAML subset of a skill's
evals/prompts.yaml with the stdlib, and fails on anything outside it."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "scripts" / "dev"
_spec = importlib.util.spec_from_file_location("skill_yaml", DEV / "skill_yaml.py")
assert _spec is not None and _spec.loader is not None
SY = importlib.util.module_from_spec(_spec)
sys.modules["skill_yaml"] = SY
_spec.loader.exec_module(SY)


def test_reads_plain_single_and_double_quoted_scalars():
    text = "- id: p1\n  kind: 'golden'\n  prompt: \"Is \\\\. a dot?\"\n  expect: yes\n"
    assert SY.read_prompts(text) == [
        {"id": "p1", "kind": "golden", "prompt": "Is \\. a dot?", "expect": "yes"}
    ]


def test_reads_flow_and_block_lists():
    text = (
        '- id: p1\n  must_run: ["a b", \'c\', d]\n  must_not_run:\n    - "git push"\n'
        "    - gh\n"
    )
    assert SY.read_prompts(text) == [
        {"id": "p1", "must_run": ["a b", "c", "d"], "must_not_run": ["git push", "gh"]}
    ]


def test_strips_comments_but_not_hash_inside_quotes():
    text = '# top\n- id: p1  # the id\n  prompt: "issue #12"  # why\n'
    assert SY.read_prompts(text) == [{"id": "p1", "prompt": "issue #12"}]


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("- id: p1\n  prompt: |\n    many lines\n", 2),
        ("- id: &a p1\n", 1),
        ("- id: p1\n  nested:\n    key: value\n", 3),
    ],
)
def test_block_scalar_anchor_or_nested_map_raises_with_line(text, line):
    with pytest.raises(SY.SubsetError) as err:
        SY.read_prompts(text)
    assert err.value.line == line


def test_the_hash_ignores_dotfiles_and_git_ignored_files(tmp_path):
    """Final review: a Finder .DS_Store or an ignored file must not stale the stamp."""
    import subprocess

    folder = tmp_path / "skill"
    folder.mkdir()
    (folder / "SKILL.md").write_text("x\n")
    before = SY.skill_hash(folder)
    (folder / ".DS_Store").write_bytes(b"\0finder")
    assert SY.skill_hash(folder) == before
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("*.log\n")
    (folder / "run.log").write_text("noise\n")
    assert SY.skill_hash(folder) == before
    (folder / "spec.md").write_text("new\n")
    assert SY.skill_hash(folder) != before
