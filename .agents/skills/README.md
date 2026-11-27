# Agent skills of this repo

Skills in the open Agent Skills format (`<name>/SKILL.md`), for any coding agent. Codex reads this folder; Claude Code reads `.claude/skills/<name>`, a link to the same folder.

| Skill | Use when |
|---|---|
| [port-to-jax](port-to-jax/SKILL.md) | moving a model to the JAX (family) backend, or a family run of a non-GTAP template fails |

To add one: create `.agents/skills/<name>/SKILL.md`, link it with `ln -s ../../.agents/skills/<name> .claude/skills/<name>`, add a `!/.claude/skills/<name>` line to `.gitignore`, and list it here.

Every skill follows Uber's skills method (`specs/2026-11-26-skills-uber-method-design.md`): a `spec.md`, a `SKILL.md` with numbered steps (a command and a `Check:` each) and an `## Errors` table, `evals/prompts.yaml`, and a fresh `evals/last_run.md` from `uv run python scripts/dev/run_skill_evals.py <name> --ablation`. `python scripts/dev/check_skills.py` checks it before each commit, in CI and from Claude Code; a skill in `scripts/dev/skills_pending.txt` only warns until it is migrated.
