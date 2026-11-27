"""scripts/dev/run_skill_evals.py plays a skill's prompts through an agent in a
throwaway worktree, with and without the skill, and stamps evals/last_run.md
(specs/2026-11-26-skills-uber-method-design.md). The agent here is a fake: a
small script that prints canned stream-json."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

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


RE = _load("run_skill_evals")
SY = _load("skill_yaml")


def _stream(commands: list[str], answer: str, errors: int = 0) -> str:
    lines = []
    for i, cmd in enumerate(commands):
        lines.append(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "running"},
                        {
                            "type": "tool_use",
                            "id": f"t{i}",
                            "name": "Bash",
                            "input": {"command": cmd},
                        },
                    ]
                },
            }
        )
        lines.append(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"t{i}",
                            "is_error": i < errors,
                        },
                    ]
                },
            }
        )
    lines.append(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "r",
                        "name": "Read",
                        "input": {"file_path": "x"},
                    },
                ]
            },
        }
    )
    lines.append({"type": "result", "result": answer})
    return "\n".join(json.dumps(x) for x in lines) + "\n"


def _fake_agent(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "fake_agent.py"
    script.write_text(textwrap.dedent(body))
    return [sys.executable, str(script)]


@pytest.fixture
def repo(tmp_path):
    """A git repo with a committed skill ``demo`` and its .claude link."""
    root = tmp_path / "repo"
    folder = root / ".agents" / "skills" / "demo"
    (folder / "evals").mkdir(parents=True)
    (folder / "SKILL.md").write_text("committed\n")
    (root / ".claude" / "skills").mkdir(parents=True)
    (root / ".claude" / "skills" / "demo").symlink_to(Path("../../.agents/skills/demo"))

    def run(*a: str) -> None:
        subprocess.run(["git", *a], cwd=root, check=True, capture_output=True)

    run("init", "-q")
    run("-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    return root


def _worktrees(root: Path) -> int:
    out = subprocess.run(
        ["git", "worktree", "list"], cwd=root, capture_output=True, text=True
    )
    return len(out.stdout.strip().splitlines())


def test_commands_from_stream_reads_bash_tool_uses_only():
    stream = _stream(["uv run pytest -q", "git status"], "done").splitlines()
    assert RE.commands_from_stream(stream) == ["uv run pytest -q", "git status"]


def test_grade_needs_every_must_run_and_no_must_not_run():
    p = {
        "must_run": [r"cascade_orchestrator\.py", "--periods shock"],
        "must_not_run": ["git push"],
    }
    ok = ["uv run python scripts/gtap/cascade_orchestrator.py --periods shock"]
    assert RE.grade(p, ok)
    assert not RE.grade(p, ["uv run python scripts/gtap/cascade_orchestrator.py"])
    assert not RE.grade(p, ok + ["git push origin x"])


SPY = """
    import json, os, sys
    from pathlib import Path
    skill = Path(".agents/skills/demo/SKILL.md")
    Path(os.environ["SPY"]).write_text(json.dumps({
        "cwd": os.getcwd(),
        "skill": skill.read_text() if skill.exists() else None,
        "link": os.path.lexists(".claude/skills/demo"),
        "argv": sys.argv[1:],
    }))
    print(json.dumps({"type": "result", "result": "an answer"}))
"""


def test_run_prompt_uses_a_throwaway_worktree_with_the_working_copy_of_the_skill(
    repo, tmp_path, monkeypatch
):
    (repo / ".agents/skills/demo/SKILL.md").write_text("uncommitted edit\n")
    spy = tmp_path / "spy.json"
    monkeypatch.setenv("SPY", str(spy))
    before = _worktrees(repo)
    res = RE.run_prompt(
        {"prompt": "go"},
        "demo",
        with_skill=True,
        agent_cmd=_fake_agent(tmp_path, SPY),
        timeout_s=60,
        max_turns=5,
        repo=repo,
    )
    seen = json.loads(spy.read_text())
    assert Path(seen["cwd"]).resolve() != repo.resolve()
    assert seen["skill"] == "uncommitted edit\n"
    assert res["answer"] == "an answer"
    assert _worktrees(repo) == before


def test_without_skill_removes_the_folder_and_link(repo, tmp_path, monkeypatch):
    spy = tmp_path / "spy.json"
    monkeypatch.setenv("SPY", str(spy))
    RE.run_prompt(
        {"prompt": "go"},
        "demo",
        with_skill=False,
        agent_cmd=_fake_agent(tmp_path, SPY),
        timeout_s=60,
        max_turns=5,
        repo=repo,
    )
    seen = json.loads(spy.read_text())
    assert seen["skill"] is None and seen["link"] is False


def test_the_agent_is_contained(repo, tmp_path, monkeypatch):
    spy = tmp_path / "spy.json"
    monkeypatch.setenv("SPY", str(spy))
    RE.run_prompt(
        {"prompt": "go"},
        "demo",
        with_skill=True,
        agent_cmd=_fake_agent(tmp_path, SPY),
        timeout_s=60,
        max_turns=7,
        repo=repo,
    )
    argv = json.loads(spy.read_text())["argv"]
    denied = argv[argv.index("--disallowedTools") + 1 :]
    for tool in (
        "Bash(git push:*)",
        "Bash(gh:*)",
        "Bash(git commit:*)",
        "Bash(uv run python scripts/gtap/run_parity_gates.py:*)",
    ):
        assert tool in denied
    assert argv[argv.index("--max-turns") + 1] == "7"
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv


def test_worktree_removed_on_interrupt(repo, tmp_path):
    agent = _fake_agent(
        tmp_path,
        """
        import os, signal, time
        os.kill(os.getppid(), signal.SIGINT)
        time.sleep(5)
    """,
    )
    before = _worktrees(repo)
    with pytest.raises(KeyboardInterrupt):
        RE.run_prompt(
            {"prompt": "go"},
            "demo",
            with_skill=True,
            agent_cmd=agent,
            timeout_s=60,
            max_turns=5,
            repo=repo,
        )
    assert _worktrees(repo) == before


def test_write_last_run_stamps_hash_and_rate(repo):
    folder = repo / ".agents/skills/demo"
    results = [{"id": f"p{i}", "kind": "trajectory", "passed": i < 4} for i in range(5)]
    path = RE.write_last_run(
        folder,
        results,
        {"commit": "abc1234", "agent": "claude", "model": "m", "ablation_wins": "3/5"},
    )
    stamp = SY.read_last_run(path)
    assert stamp["skill_hash"] == SY.skill_hash(folder)
    assert stamp["pass_rate"] == "0.80"
    assert stamp["ablation_wins"] == "3/5"


def test_ablation_judge_order_is_random_and_labels_hidden():
    orders = []
    for seed in range(6):
        text, with_skill_is = RE.judge_prompt(
            {"prompt": "why?", "expect": "a layer"},
            "ANSWER-WITH",
            "ANSWER-WITHOUT",
            random.Random(seed),
        )
        assert "skill" not in text.lower()
        assert "ANSWER-WITH" in text and "ANSWER-WITHOUT" in text
        orders.append(with_skill_is)
    assert set(orders) == {"A", "B"}


SPY_TREE = """
    import json, os
    from pathlib import Path
    Path(os.environ["SPY"]).write_text(json.dumps({
        "evals": Path(".agents/skills/demo/evals").exists(),
        "link": os.path.lexists(".claude/skills/demo"),
        "skill": Path(".claude/skills/demo/SKILL.md").exists(),
    }))
    print(json.dumps({"type": "result", "result": "ok"}))
"""


def test_the_agent_does_not_see_its_grading(repo, tmp_path, monkeypatch):
    (repo / ".agents/skills/demo/evals/prompts.yaml").write_text("- id: p1\n")
    spy = tmp_path / "spy.json"
    monkeypatch.setenv("SPY", str(spy))
    RE.run_prompt(
        {"prompt": "go"},
        "demo",
        with_skill=True,
        agent_cmd=_fake_agent(tmp_path, SPY_TREE),
        timeout_s=60,
        max_turns=5,
        repo=repo,
    )
    assert json.loads(spy.read_text())["evals"] is False


def test_a_skill_not_in_head_is_linked_for_the_run(repo, tmp_path, monkeypatch):
    new = repo / ".agents/skills/fresh"
    new.mkdir(parents=True)
    (new / "SKILL.md").write_text("new skill\n")
    spy = tmp_path / "spy.json"
    monkeypatch.setenv("SPY", str(spy))
    agent = _fake_agent(tmp_path, SPY_TREE.replace("demo", "fresh"))
    RE.run_prompt(
        {"prompt": "go"},
        "fresh",
        with_skill=True,
        agent_cmd=agent,
        timeout_s=60,
        max_turns=5,
        repo=repo,
    )
    seen = json.loads(spy.read_text())
    assert seen["link"] is True and seen["skill"] is True


def test_the_run_has_an_explicit_permission_mode_and_allowlist(
    repo, tmp_path, monkeypatch
):
    spy = tmp_path / "spy.json"
    monkeypatch.setenv("SPY", str(spy))
    RE.run_prompt(
        {"prompt": "go"},
        "demo",
        with_skill=True,
        agent_cmd=_fake_agent(tmp_path, SPY),
        timeout_s=60,
        max_turns=5,
        repo=repo,
    )
    argv = json.loads(spy.read_text())["argv"]
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert "--allowedTools" in argv
    denied = argv[argv.index("--disallowedTools") + 1 :]
    for tool in (
        "Bash(git -C:*)",
        "Bash(git stash:*)",
        "Bash(gams:*)",
        "Bash(python3 scripts/gtap/run_parity_gates.py:*)",
    ):
        assert tool in denied, tool


def test_a_denied_command_does_not_count_for_must_run():
    stream = _stream(
        ["uv run python scripts/gtap/cascade_orchestrator.py"], "x", errors=1
    )
    assert RE.commands_from_stream(stream.splitlines()) == []


def test_the_judge_runs_with_no_tools_outside_the_repo(tmp_path, monkeypatch):
    spy = tmp_path / "judge.json"
    monkeypatch.setenv("SPY", str(spy))
    agent = _fake_agent(
        tmp_path,
        """
        import json, os, sys
        from pathlib import Path
        Path(os.environ["SPY"]).write_text(json.dumps({"cwd": os.getcwd(), "argv": sys.argv[1:]}))
        print("A")
    """,
    )
    assert RE._ask(agent, "pick", 60, None) == "A"
    seen = json.loads(spy.read_text())
    assert seen["argv"][seen["argv"].index("--tools") + 1] == ""
    assert not Path(seen["cwd"]).resolve().is_relative_to(ROOT.resolve())


def test_a_failed_worktree_add_leaves_nothing(repo, tmp_path, monkeypatch):
    made = []
    real = RE.tempfile.mkdtemp

    def spy_mkdtemp(**kw):
        made.append(real(**kw))
        return made[-1]

    monkeypatch.setattr(RE.tempfile, "mkdtemp", spy_mkdtemp)
    monkeypatch.setattr(
        RE, "_git", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        RE.run_prompt(
            {"prompt": "go"},
            "demo",
            with_skill=True,
            agent_cmd=["true"],
            timeout_s=60,
            max_turns=5,
            repo=repo,
        )
    assert made and not Path(made[0]).exists()


def test_sigterm_becomes_keyboard_interrupt():
    import signal

    with RE._interrupt_on_term():
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)


# MAR-257: the final review's deferred minors

MIX_PROMPTS = """- id: g1
  kind: golden
  prompt: "q"
  expect: "a"
- id: t1
  kind: trajectory
  prompt: "q"
  must_run: ["ok"]
  expect: "a"
- id: t2
  kind: trajectory
  prompt: "q"
  must_run: ["ok"]
  expect: "a"
- id: a1
  kind: author
  prompt: "q"
  expect: "a"
"""


def _main_with(repo, monkeypatch, prompts: str, passed: dict[str, bool]) -> int:
    """main over ``prompts`` with the agent and judge faked: ``passed`` per id."""
    (repo / ".agents/skills/demo/evals/prompts.yaml").write_text(prompts)

    def fake_run(prompt, skill, **kw):
        ok = passed.get(str(prompt.get("id")), True)
        return {
            "commands": ["ok"] if ok else [],
            "answer": "x",
            "passed": ok,
            "agent_error": "",
            "minutes": 0.0,
            "tool_calls": 0,
            "tool_errors": 0,
        }

    monkeypatch.setattr(RE, "run_prompt", fake_run)
    monkeypatch.setattr(
        RE, "golden_matches", lambda p, *a, **k: passed.get(str(p.get("id")), True)
    )
    return RE.main(["demo"], root=repo)


def test_main_exits_1_under_the_70_percent_pass_rate(repo, monkeypatch, capsys):
    assert _main_with(repo, monkeypatch, MIX_PROMPTS, {"t1": False, "t2": False}) == 1
    assert "under 70%" in capsys.readouterr().out
    assert _main_with(repo, monkeypatch, MIX_PROMPTS, {}) == 0


def test_a_failed_agent_run_fails_its_prompt(repo, tmp_path):
    agent = _fake_agent(
        tmp_path,
        """
        import json, sys
        print(json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t0", "name": "Bash",
             "input": {"command": "ok"}}]}}))
        print(json.dumps({"type": "result", "result": "half"}))
        sys.exit(3)
    """,
    )
    res = RE.run_prompt(
        {"prompt": "go", "must_run": ["ok"]},
        "demo",
        with_skill=True,
        agent_cmd=agent,
        timeout_s=60,
        max_turns=5,
        repo=repo,
    )
    assert res["commands"] == ["ok"]
    assert res["passed"] is False
    assert res["agent_error"] == "the agent exited 3"


def test_a_failed_agent_run_is_not_sent_to_the_golden_judge(repo, monkeypatch):
    (repo / ".agents/skills/demo/evals/prompts.yaml").write_text(
        '- id: g1\n  kind: golden\n  prompt: "q"\n  expect: "a"\n'
    )
    monkeypatch.setattr(
        RE,
        "run_prompt",
        lambda p, s, **kw: {
            "commands": [],
            "answer": "",
            "passed": False,
            "agent_error": "the agent exited 1",
            "minutes": 0.0,
            "tool_calls": 0,
            "tool_errors": 0,
        },
    )
    monkeypatch.setattr(RE, "golden_matches", lambda *a, **k: True)
    assert RE.main(["demo"], root=repo) == 1
    stamp = SY.read_last_run(repo / ".agents/skills/demo/evals/last_run.md")
    assert stamp["pass_rate"] == "0.00"


def test_a_bad_prompts_yaml_is_one_error_line_not_a_traceback(
    repo, monkeypatch, capsys
):
    (repo / ".agents/skills/demo/evals/prompts.yaml").write_text(
        "- id: p1\n  prompt: |\n    many\n"
    )
    assert RE.main(["demo"], root=repo) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: demo: evals/prompts.yaml line 2:")
    assert "Traceback" not in err


def test_the_run_reports_the_prompt_mix(repo, monkeypatch, capsys):
    _main_with(repo, monkeypatch, MIX_PROMPTS, {})
    out = capsys.readouterr().out
    assert (
        "prompts: 1 golden, 2 trajectory, 1 author "
        "(25% / 50% / 25%; the method's target is 5% / 80% / 15%)"
    ) in out
    stamp = SY.read_last_run(repo / ".agents/skills/demo/evals/last_run.md")
    assert stamp["mix"] == "1 golden, 2 trajectory, 1 author"
