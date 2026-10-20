# Specs and changes

`specs/` holds what each architecture area does **today** (the baseline) and
what we have decided to change (open changes). It is written for people who
develop equilibria. User documentation lives in `docs/site/`, `docs/guides/`
and `examples/`.

## Two levels

```
specs/<area>/
├── specs/specs.md          the present: one requirement per behavior
├── changes/<slug>/         changes in progress
├── changes/archive/<YYYY-MM-DD>-<slug>/   closed changes, never edited again
└── context.md              short: vocabulary, risky files, commands, known risks
```

- **Baseline** (`specs/<area>/specs/`): the current behavior. Each requirement
  is a `### Requirement: <name>` block; the name is its identity.
- **Change** (`specs/<area>/changes/<slug>/`): `proposal.md` (why, what, issues),
  an optional `design.md` (how, when it is not obvious) and `specs/specs.md`
  with deltas under `## ADDED Requirements`, `## MODIFIED Requirements` (the
  full requirement in its new form) and `## REMOVED Requirements`.

`specs/areas.toml` maps each area to the code paths it covers, lists
deliberately unowned (latent) code, and lists the user-doc pages of each area
under `[docs]`. Decisions that are expensive to reverse go in `docs/adr/` as
immutable ADRs (`NNNN-<slug>.md`).

## The cycle: propose, sync, archive

1. **Propose.** Create `specs/<area>/changes/<slug>/` with the proposal and the
   deltas. Open one GitHub issue per task and list them in the proposal.
2. **Implement and sync.** One task is one branch is one PR. Name the branch
   `<type>/<issue>-<slug>`, where `<slug>` is the change slug. In that PR,
   sync to the baseline whatever the task leaves complete.
3. **Archive.** The last PR of the change moves it to `changes/archive/`.

## When a change is needed

| Situation | Change? |
| --- | --- |
| Typo, formatting, bug with no behavior change | No (label `no-delta`) |
| Bug that shows the spec was wrong | Yes, `MODIFIED` |
| New functionality | Yes, `ADDED` |
| Refactor | Yes, with no deltas (if it produces deltas, it was not a refactor) |
| Decision that is expensive to reverse | Also an ADR |

## Commands

Run from the repository root:

```bash
python scripts/dev/spec_tools.py new-area <area>          # scaffold specs/<area>/
python scripts/dev/spec_tools.py new-change <area> <slug> # scaffold a change
python scripts/dev/spec_tools.py new-adr <slug>           # next numbered ADR
python scripts/dev/spec_tools.py archive <area> <slug>    # close a change
python scripts/dev/spec_tools.py check-pr --base origin/main --labels "<a,b>"
```

## How it is enforced

From cheapest to most expensive:

1. The PR template (`.github/pull_request_template.md`): issue, change,
   deltas implemented, spec checkboxes, parity gates, and what was not done.
2. The `spec-check` CI job runs `check-pr`. It **warns** when an area's code
   changes without a spec change, when a path is not mapped to any area, and
   when an area's baseline changes without touching its user-doc pages.
   It **fails** when a delta does not match the baseline (`MODIFIED` or
   `REMOVED` of a missing requirement, a conflicting `ADDED`) or when a
   requirement is edited by hand outside a change.
3. Review.

Labels that skip checks: `no-delta` (code change with no spec delta),
`bootstrap` (initial baseline or editorial fix to a baseline) and `no-docs`
(spec change that needs no user-doc update).

Set labels before pushing: adding a label later does not re-run the check
(re-push or re-run the job).

The existing parity gates (`scripts/gtap/run_parity_gates.py`) are separate
and unchanged.

## User documentation

Specs are for people who develop; the docs are for people who use. They travel
together without blocking: when a baseline changes, update the pages listed
for its area in `[docs]`, or label the PR `no-docs`. Specs and ADRs are not
published on the docs site.
