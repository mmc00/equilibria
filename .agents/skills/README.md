# Agent skills of this repo

Skills in the open Agent Skills format (`<name>/SKILL.md`), for any coding agent. Codex reads this folder; Claude Code reads `.claude/skills/<name>`, a link to the same folder.

| Skill | Use when |
|---|---|
| [port-to-jax](port-to-jax/SKILL.md) | moving a model to the JAX (family) backend, or a family run of a non-GTAP template fails |

To add one: create `.agents/skills/<name>/SKILL.md`, link it with `ln -s ../../.agents/skills/<name> .claude/skills/<name>`, add a `!/.claude/skills/<name>` line to `.gitignore`, and list it here.
