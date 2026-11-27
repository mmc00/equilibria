"""run_skill_evals.py: play a skill's test prompts through an agent and stamp the run.

Uber's skills method, as specs/2026-11-26-skills-uber-method-design.md sets it:

    uv run python scripts/dev/run_skill_evals.py port-to-jax --ablation

How each prompt of ``evals/prompts.yaml`` runs:
1. It runs in a throwaway worktree of HEAD, with the working copy of the skill's
   folder copied in, so uncommitted edits are what get tested.
2. The agent is ``claude -p`` and its run is contained:
   - ``git push``, ``gh``, ``git commit`` and the GTAP gates are denied;
   - it has a turn cap and a timeout.
3. The prompt is graded by its kind:
   - ``trajectory``: by ``must_run`` and ``must_not_run`` over the commands the agent ran;
   - ``golden``: by a judge, against ``expect``;
   - ``author``: printed for the author, and it does not count toward the pass rate.
4. With ``--ablation``, each prompt also runs without the skill. A judge then
   picks the better of the two answers, unlabeled and in random order.

The run writes ``evals/last_run.md``, which check_skills.py requires to be
fresh. This costs real agent runs: a few minutes and some money per prompt.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from datetime import date
from pathlib import Path

DEV = Path(__file__).resolve().parent
sys.path.insert(0, str(DEV))

from skill_yaml import LAST_RUN, read_prompts, skill_hash  # noqa: E402

ROOT = DEV.parents[1]
SKILLS = Path(".agents") / "skills"
ALLOWED = ("Bash", "Read", "Grep", "Glob", "Edit", "Write")
DENIED = (
    "Bash(git push:*)",
    "Bash(gh:*)",
    "Bash(git commit:*)",
    "Bash(git -C:*)",
    "Bash(git stash:*)",
    "Bash(git branch:*)",
    "Bash(git worktree:*)",
    "Bash(gams:*)",
    "Bash(uv run python scripts/gtap/run_parity_gates.py:*)",
    "Bash(python3 scripts/gtap/run_parity_gates.py:*)",
)


def commands_from_stream(lines: Iterable[str]) -> list[str]:
    """The ``command`` of every Bash tool call in the agent's stream-json whose
    result was not an error: a denied or failed command does not count."""
    calls: list[tuple[str, str]] = []
    failed: set[str] = set()
    for line in lines:
        try:
            msg = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        for block in (msg.get("message") or {}).get("content") or []:
            if msg.get("type") == "assistant" and block.get("type") == "tool_use":
                cmd = (block.get("input") or {}).get("command")
                if block.get("name") == "Bash" and isinstance(cmd, str):
                    calls.append((str(block.get("id")), cmd))
            elif msg.get("type") == "user" and block.get("type") == "tool_result":
                if block.get("is_error"):
                    failed.add(str(block.get("tool_use_id")))
    return [cmd for tid, cmd in calls if tid not in failed]


def _stream_stats(lines: list[str]) -> tuple[str, int, int]:
    """(final answer, tool calls, tool errors) of the stream."""
    answer, calls, errors = "", 0, 0
    for line in lines:
        try:
            msg = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        content = (msg.get("message") or {}).get("content") or []
        if msg.get("type") == "assistant":
            calls += sum(1 for b in content if b.get("type") == "tool_use")
        elif msg.get("type") == "user":
            errors += sum(
                1
                for b in content
                if b.get("type") == "tool_result" and b.get("is_error")
            )
        elif msg.get("type") == "result":
            answer = str(msg.get("result") or "")
    return answer, calls, errors


def _as_list(value) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def grade(prompt: dict, commands: list[str]) -> bool:
    """Every ``must_run`` regex matches some command and no ``must_not_run`` does."""
    must = _as_list(prompt.get("must_run"))
    never = _as_list(prompt.get("must_not_run"))
    return all(any(re.search(p, c) for c in commands) for p in must) and not any(
        re.search(p, c) for p in never for c in commands
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def run_prompt(
    prompt: dict,
    skill: str,
    *,
    with_skill: bool,
    agent_cmd: list[str],
    timeout_s: int,
    max_turns: int,
    repo: Path = ROOT,
    model: str | None = None,
) -> dict:
    """One contained agent run of ``prompt`` in a throwaway worktree of HEAD."""
    tmp = Path(tempfile.mkdtemp(prefix=f"skill-eval-{skill}-"))
    work = tmp / "wt"
    try:
        _git(repo, "worktree", "add", "--detach", str(work), "HEAD")
        target = work / SKILLS / skill
        link = work / ".claude" / "skills" / skill
        if target.exists():
            shutil.rmtree(target)
        if with_skill:
            # the agent must not read its own grading (prompts.yaml, last_run.md)
            shutil.copytree(
                repo / SKILLS / skill, target, ignore=shutil.ignore_patterns("evals")
            )
            if not (link.is_symlink() or link.exists()):
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(Path("../..") / SKILLS / skill)
        elif link.is_symlink() or link.exists():
            link.unlink()
        argv = [
            *agent_cmd,
            "-p",
            str(prompt.get("prompt", "")),
            "--output-format",
            "stream-json",
            "--verbose",
            "--max-turns",
            str(max_turns),
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            *ALLOWED,
            "--disallowedTools",
            *DENIED,
        ]
        if model:
            argv += ["--model", model]
        start = time.monotonic()
        try:
            proc = subprocess.run(
                argv, cwd=work, capture_output=True, text=True, timeout=timeout_s
            )
            lines = proc.stdout.splitlines()
        except subprocess.TimeoutExpired as err:
            out = err.stdout or ""
            lines = (out.decode() if isinstance(out, bytes) else out).splitlines()
        minutes = (time.monotonic() - start) / 60
        commands = commands_from_stream(lines)
        answer, calls, errors = _stream_stats(lines)
        return {
            "commands": commands,
            "answer": answer,
            "passed": grade(prompt, commands),
            "minutes": minutes,
            "tool_calls": calls,
            "tool_errors": errors,
        }
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(work)],
            cwd=repo,
            capture_output=True,
        )
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(["git", "worktree", "prune"], cwd=repo, capture_output=True)


def judge_prompt(
    prompt: dict, with_answer: str, without_answer: str, rng: random.Random
) -> tuple[str, str]:
    """(the judge's prompt, which label, A or B, holds ``with_answer``). The two
    answers are unlabeled and their order is random."""
    first = rng.random() < 0.5
    a, b = (with_answer, without_answer) if first else (without_answer, with_answer)
    text = (
        "Two answers to the same request follow. Pick the better one for a "
        "developer of this repo: correct, specific, and following the repo's "
        "rules. Reply with exactly one letter, A or B.\n\n"
        f"Request:\n{prompt.get('prompt', '')}\n\n"
        f"What a good answer does:\n{prompt.get('expect', '')}\n\n"
        f"Answer A:\n{a}\n\nAnswer B:\n{b}\n"
    )
    return text, "A" if first else "B"


def _ask(agent_cmd: list[str], text: str, timeout_s: int, model: str | None) -> str:
    """One judge call: no tools, run from an empty temp folder, since ``text``
    holds the agent's own (untrusted) answers."""
    argv = [*agent_cmd, "-p", text, "--max-turns", "1", "--tools", ""]
    if model:
        argv += ["--model", model]
    with tempfile.TemporaryDirectory(prefix="skill-judge-") as cwd:
        try:
            out = subprocess.run(
                argv, cwd=cwd, capture_output=True, text=True, timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            return ""
    return out.stdout.strip()


@contextlib.contextmanager
def _interrupt_on_term():
    """SIGTERM and SIGHUP (a closed terminal) raise KeyboardInterrupt, so the
    worktree cleanup in ``run_prompt`` runs for them too."""

    def handler(signum, frame):
        raise KeyboardInterrupt

    old = {s: signal.signal(s, handler) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def golden_matches(
    prompt: dict, answer: str, agent_cmd: list[str], timeout_s: int, model: str | None
) -> bool:
    text = (
        "Does the answer below meet the expected result? Reply with exactly YES or NO.\n\n"
        f"Request:\n{prompt.get('prompt', '')}\n\nExpected:\n{prompt.get('expect', '')}\n\n"
        f"Answer:\n{answer}\n"
    )
    return _ask(agent_cmd, text, timeout_s, model).upper().startswith("YES")


def write_last_run(folder: Path, results: list[dict], meta: dict) -> Path:
    """Stamp ``evals/last_run.md`` with the pass rate over graded prompts and the
    folder's hash; ``meta`` gives commit, agent, model and ablation_wins."""
    graded = [r for r in results if r.get("kind") != "author"]
    rate = sum(1 for r in graded if r.get("passed")) / len(graded) if graded else 0.0
    fields = {
        "commit": meta.get("commit", ""),
        "date": meta.get("date", date.today().isoformat()),
        "agent": meta.get("agent", ""),
        "model": meta.get("model", ""),
        "pass_rate": f"{rate:.2f}",
        "ablation_wins": meta.get("ablation_wins", "not run"),
        "skill_hash": skill_hash(folder),
    }
    lines = [
        "# Last eval run",
        "",
        *(f"{k}: {v}" for k, v in fields.items()),
        "",
        "## Prompts",
        "",
    ]
    for r in results:
        mark = (
            "author"
            if r.get("kind") == "author"
            else ("pass" if r.get("passed") else "FAIL")
        )
        lines.append(f"- {r.get('id')}: {mark}")
    path = folder / LAST_RUN
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("skill")
    ap.add_argument("--agent", default="claude", help="the agent command (claude)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs", type=int, default=1, help="runs per prompt")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per run")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument(
        "--ablation", action="store_true", help="also run without the skill and judge"
    )
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    folder = ROOT / SKILLS / args.skill
    prompts = read_prompts(
        (folder / "evals" / "prompts.yaml").read_text(encoding="utf-8")
    )
    agent = [args.agent]
    rng = random.Random(args.seed)
    results, wins, pairs = [], 0, 0
    deltas = {"tool_calls": 0, "tool_errors": 0, "minutes": 0.0}
    with _interrupt_on_term():
        for p in prompts:
            for _ in range(args.runs):
                r = run_prompt(
                    p,
                    args.skill,
                    with_skill=True,
                    agent_cmd=agent,
                    timeout_s=args.timeout,
                    max_turns=args.max_turns,
                    model=args.model,
                )
                kind = p.get("kind")
                if kind == "golden":
                    r["passed"] = golden_matches(
                        p, r["answer"], agent, args.timeout, args.model
                    )
                if kind == "author":
                    print(
                        f"[author] {p.get('id')}: expect: {p.get('expect')}\n{r['answer']}\n"
                    )
                if args.ablation:
                    bare = run_prompt(
                        p,
                        args.skill,
                        with_skill=False,
                        agent_cmd=agent,
                        timeout_s=args.timeout,
                        max_turns=args.max_turns,
                        model=args.model,
                    )
                    text, label = judge_prompt(p, r["answer"], bare["answer"], rng)
                    pairs += 1
                    wins += (
                        _ask(agent, text, args.timeout, args.model).strip().upper()[:1]
                        == label
                    )
                    for k in deltas:
                        deltas[k] += bare[k] - r[k]
                results.append({"id": p.get("id"), "kind": kind, "passed": r["passed"]})
                print(
                    f"{p.get('id')}: {'author' if kind == 'author' else ('pass' if r['passed'] else 'FAIL')}"
                )
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    path = write_last_run(
        folder,
        results,
        {
            "commit": commit,
            "agent": args.agent,
            "model": args.model or "default",
            "ablation_wins": f"{wins}/{pairs}" if args.ablation else "not run",
        },
    )
    graded = [r for r in results if r["kind"] != "author"]
    rate = sum(r["passed"] for r in graded) / len(graded) if graded else 0.0
    print(
        f"pass rate {rate:.0%} over {len(graded)} graded run(s); stamp: {path.relative_to(ROOT)}"
    )
    if args.ablation:
        print(
            f"ablation: the skill's answer won {wins}/{pairs}; without the skill: "
            f"{deltas['tool_calls']:+d} tool calls, {deltas['tool_errors']:+d} tool errors, "
            f"{deltas['minutes']:+.1f} min"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
