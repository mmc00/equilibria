from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "spec_tools.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("spec_tools", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _areas(m: Any) -> Any:
    return m.Areas(
        areas={
            "templates": ("src/equilibria/templates/base.py",),
            "templates/gtap": ("src/equilibria/templates/gtap/",),
            "solver": ("src/equilibria/solver/",),
        },
        unowned=("src/equilibria/templates/gtap_sparse/",),
    )


def test_area_for_prefers_longest_prefix() -> None:
    m = _load_module()
    a = _areas(m)
    assert m.area_for("src/equilibria/templates/gtap/driver.py", a) == "templates/gtap"
    assert m.area_for("src/equilibria/templates/base.py", a) == "templates"


def test_area_for_unowned_is_none() -> None:
    m = _load_module()
    assert m.area_for("src/equilibria/templates/gtap_sparse/x.py", _areas(m)) is None


def test_unmapped_flags_unknown_src_path() -> None:
    m = _load_module()
    paths = ["src/equilibria/nuevo.py", "src/equilibria/solver/x.py"]
    assert m.unmapped(paths, _areas(m)) == ["src/equilibria/nuevo.py"]
    assert m.unmapped(["src/equilibria/templates/gtap_sparse/x.py"], _areas(m)) == []


def test_real_areas_toml_maps_all_src() -> None:
    m = _load_module()
    areas = m.load_areas(ROOT / "specs" / "areas.toml")
    out = subprocess.run(
        ["git", "ls-files", "src/equilibria"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert m.unmapped(out, areas) == []


def test_areas_without_spec() -> None:
    m = _load_module()
    a = _areas(m)
    assert m.areas_without_spec(["src/equilibria/solver/x.py"], a) == ["solver"]
    changed = ["src/equilibria/solver/x.py", "specs/solver/specs/specs.md"]
    assert m.areas_without_spec(changed, a) == []
    # a nested area's spec does not satisfy its parent area
    changed = [
        "src/equilibria/templates/base.py",
        "specs/templates/gtap/specs/specs.md",
    ]
    assert m.areas_without_spec(changed, a) == ["templates"]


def test_parse_requirements_normalizes_heading() -> None:
    m = _load_module()
    reqs, warns = m.parse_requirements("###  requirement:  Foo \nSource: x\n\ntext\n")
    assert list(reqs) == ["Foo"]
    assert warns == []


def test_parse_requirements_warns_on_malformed() -> None:
    m = _load_module()
    reqs, warns = m.parse_requirements("### Requirment: Foo\nSource: x\n")
    assert reqs == {}
    assert len(warns) == 1
    assert "Requirment" in warns[0]


DELTA = """\
## ADDED Requirements

### Requirement: New
Source: a

new body

## MODIFIED Requirements

### Requirement: Old
Before: the old text
Source: a

updated body

## REMOVED Requirements

### Requirement: Gone
Reason: obsolete
"""


def test_parse_deltas_strips_before_lines() -> None:
    m = _load_module()
    d = m.parse_deltas(DELTA)
    assert list(d.added) == ["New"]
    assert list(d.modified) == ["Old"]
    assert d.removed == ["Gone"]
    assert "Before:" not in d.modified["Old"]
    assert "updated body" in d.modified["Old"]


def _deltas(m: Any, added=None, modified=None, removed=None) -> Any:
    return m.Deltas(
        added=added or {}, modified=modified or {}, removed=removed or [], warnings=[]
    )


def test_check_delta_modified_unknown_name() -> None:
    m = _load_module()
    errs = m.check_delta({}, _deltas(m, modified={"Ghost": "x"}))
    assert len(errs) == 1 and "Ghost" in errs[0]


def test_check_delta_added_duplicate_different_body() -> None:
    m = _load_module()
    errs = m.check_delta({"A": "old"}, _deltas(m, added={"A": "new"}))
    assert len(errs) == 1 and "duplicate" in errs[0]


def test_check_delta_added_already_synced_ok() -> None:
    m = _load_module()
    assert m.check_delta({"A": "same"}, _deltas(m, added={"A": "same"})) == []


def test_check_delta_rename_is_removed_plus_added() -> None:
    m = _load_module()
    d = _deltas(m, added={"B": "b"}, removed=["A"])
    assert m.check_delta({"A": "a"}, d) == []
    assert m.check_delta({"B": "b"}, d) == []


def test_hand_edits_detects_change_without_delta() -> None:
    m = _load_module()
    old, new = {"X": "1", "Y": "2"}, {"X": "9", "Y": "2"}
    assert m.hand_edits(old, new, []) == ["X"]
    assert m.hand_edits(old, new, [_deltas(m, modified={"X": "9"})]) == []


def test_subheadings_and_code_fences_stay_in_body() -> None:
    m = _load_module()
    md = (
        "### Requirement: A\nSource: x\n\ntext\n"
        "#### Scenario: s\nmore\n```\n# c\n### Requirement: fake\n```\n"
        "### Requirement: B\nSource: y\n"
    )
    reqs, warns = m.parse_requirements(md)
    assert warns == []
    assert list(reqs) == ["A", "B"]
    for part in ("text", "#### Scenario: s", "more", "# c", "### Requirement: fake"):
        assert part in reqs["A"]

    d = m.parse_deltas("## ADDED Requirements\n\n" + md)
    assert d.warnings == []
    assert "# c" in d.added["A"] and "#### Scenario: s" in d.added["A"]


# --- CLI (toy git repo) -----------------------------------------------------

REQ_V1 = "### Requirement: Solver converges\nSource: issue 1\n\nIt converges.\n"
REQ_V2 = "### Requirement: Solver converges\nSource: issue 1\n\nIt converges fast.\n"
TOY_AREAS = (
    '[areas]\nsolver = ["src/equilibria/solver/"]\n'
    '[docs]\nsolver = ["docs/site/guide/path_capi.md", "docs/guides/solver_*.md"]\n'
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def _write(repo: Path, rel: str, text: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _commit(repo: Path, msg: str) -> None:
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c", "user.name=t", "-c", "user.email=t@t",
        "commit", "-q", "-m", msg,
    )  # fmt: skip


@pytest.fixture
def toy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    shutil.copytree(ROOT / "specs" / "_templates", repo / "specs" / "_templates")
    _write(repo, "specs/areas.toml", TOY_AREAS)
    _write(repo, "src/equilibria/solver/a.py", "x = 1\n")
    _commit(repo, "base")
    monkeypatch.chdir(repo)
    return repo


def _run(m: Any, *argv: str) -> int:
    return m.main(list(argv))


def test_new_change_copies_template(toy: Path) -> None:
    m = _load_module()
    (toy / "specs" / "solver").mkdir()
    assert _run(m, "new-change", "solver", "faster") == 0
    base = toy / "specs" / "solver" / "changes" / "faster"
    assert (base / "proposal.md").is_file()
    assert (base / "design.md").is_file()
    assert (base / "specs" / "specs.md").is_file()


def test_new_change_refuses_existing(toy: Path) -> None:
    m = _load_module()
    (toy / "specs" / "solver").mkdir()
    assert _run(m, "new-change", "solver", "faster") == 0
    assert _run(m, "new-change", "solver", "faster") != 0
    assert _run(m, "new-change", "nope", "faster") != 0


def test_new_area_copies_and_refuses_existing(toy: Path) -> None:
    m = _load_module()
    assert _run(m, "new-area", "solver") == 0
    assert (toy / "specs" / "solver" / "specs" / "specs.md").is_file()
    assert (toy / "specs" / "solver" / "context.md").is_file()
    assert _run(m, "new-area", "solver") != 0


def test_new_adr_numbers_after_last(toy: Path) -> None:
    m = _load_module()
    _write(toy, "docs/adr/0008-x.md", "# x\n")
    assert _run(m, "new-adr", "my-decision") == 0
    assert (toy / "docs" / "adr" / "0009-my-decision.md").is_file()


def test_archive_moves_change(toy: Path) -> None:
    m = _load_module()
    (toy / "specs" / "solver").mkdir()
    _run(m, "new-change", "solver", "faster")
    assert _run(m, "archive", "solver", "faster") == 0
    assert not (toy / "specs/solver/changes/faster").exists()
    archived = list((toy / "specs/solver/changes/archive").glob("*-faster"))
    assert len(archived) == 1


def test_check_pr_warns_area_without_spec(
    toy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _load_module()
    _write(toy, "specs/solver/specs/specs.md", REQ_V1)
    _commit(toy, "spec")
    _git(toy, "branch", "base")
    _write(toy, "src/equilibria/solver/x.py", "y = 2\n")
    _commit(toy, "code")
    assert _run(m, "check-pr", "--base", "base") == 0
    out = capsys.readouterr().out
    assert "::warning::" in out and "has code changes but no spec change" in out


def test_check_pr_no_spec_warning_before_area_has_baseline(
    toy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _load_module()
    _git(toy, "branch", "base")
    _write(toy, "src/equilibria/solver/x.py", "y = 2\n")
    _commit(toy, "code")
    assert _run(m, "check-pr", "--base", "base") == 0
    assert "has code changes but no spec change" not in capsys.readouterr().out


def _hand_edit_repo(toy: Path) -> None:
    _write(toy, "specs/solver/specs/specs.md", REQ_V1)
    _commit(toy, "spec")
    _git(toy, "branch", "base")
    _write(toy, "specs/solver/specs/specs.md", REQ_V2)
    _commit(toy, "edit")


def test_check_pr_errors_on_hand_edit(
    toy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _load_module()
    _hand_edit_repo(toy)
    assert _run(m, "check-pr", "--base", "base") == 1
    assert "::error::" in capsys.readouterr().out


def test_check_pr_no_delta_label_skips(toy: Path) -> None:
    m = _load_module()
    _hand_edit_repo(toy)
    assert _run(m, "check-pr", "--base", "base", "--labels", "no-delta") == 0


def test_check_pr_covered_by_open_change(toy: Path) -> None:
    m = _load_module()
    _write(toy, "specs/solver/specs/specs.md", REQ_V1)
    _commit(toy, "spec")
    _git(toy, "branch", "base")
    _write(toy, "specs/solver/specs/specs.md", REQ_V2)
    delta = "## MODIFIED Requirements\n\n" + REQ_V2
    _write(toy, "specs/solver/changes/c1/specs/specs.md", delta)
    _commit(toy, "edit with delta")
    assert _run(m, "check-pr", "--base", "base") == 0


def test_check_pr_ignores_templates_and_archive(toy: Path) -> None:
    m = _load_module()
    _git(toy, "branch", "base")
    _write(toy, "specs/_templates/new-area/specs/specs.md", REQ_V1)
    _write(toy, "specs/solver/changes/archive/2026-01-01-c/specs/specs.md", REQ_V1)
    _commit(toy, "tpl")
    assert _run(m, "check-pr", "--base", "base") == 0


def test_check_pr_missing_base(toy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    m = _load_module()
    assert _run(m, "check-pr", "--base", "nope") == 2
    assert "base ref not found: nope" in capsys.readouterr().out


def test_load_areas_reads_docs_table(tmp_path: Path) -> None:
    m = _load_module()
    f = tmp_path / "areas.toml"
    f.write_text(TOY_AREAS, encoding="utf-8")
    a = m.load_areas(f)
    assert a.docs == {
        "solver": ("docs/site/guide/path_capi.md", "docs/guides/solver_*.md")
    }
    f.write_text('[areas]\nsolver = ["src/"]\n', encoding="utf-8")
    assert m.load_areas(f).docs == {}


def test_stale_docs_warns_when_baseline_changes_without_docs() -> None:
    m = _load_module()
    a = m.Areas(
        areas={"solver": ("src/equilibria/solver/",)},
        unowned=(),
        docs={"solver": ("docs/site/guide/path_capi.md", "docs/guides/solver_*.md")},
    )
    spec = "specs/solver/specs/specs.md"
    assert m.stale_docs([spec], a) == ["solver"]
    assert m.stale_docs([spec, "docs/site/guide/path_capi.md"], a) == []
    assert m.stale_docs([spec, "docs/guides/solver_notes.md"], a) == []
    # a change proposal is not the baseline
    assert m.stale_docs(["specs/solver/changes/c/specs/specs.md"], a) == []
    # area without a [docs] entry is not flagged
    assert m.stale_docs([spec], m.Areas(a.areas, ())) == []


def _baseline_change_repo(toy: Path) -> None:
    _write(toy, "specs/solver/specs/specs.md", REQ_V1)
    _commit(toy, "spec")
    _git(toy, "branch", "base")
    _write(toy, "specs/solver/specs/specs.md", REQ_V2)
    delta = "## MODIFIED Requirements\n\n" + REQ_V2
    _write(toy, "specs/solver/changes/c1/specs/specs.md", delta)
    _commit(toy, "edit with delta")


def test_check_pr_warns_stale_docs(
    toy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _load_module()
    _baseline_change_repo(toy)
    assert _run(m, "check-pr", "--base", "base") == 0
    out = capsys.readouterr().out
    assert "::warning::" in out and "user docs" in out and "solver" in out


def test_check_pr_no_docs_label_skips(
    toy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _load_module()
    _baseline_change_repo(toy)
    assert _run(m, "check-pr", "--base", "base", "--labels", "no-docs") == 0
    assert "user docs" not in capsys.readouterr().out


def test_real_docs_paths_exist() -> None:
    m = _load_module()
    areas = m.load_areas(ROOT / "specs" / "areas.toml")
    assert areas.docs, "specs/areas.toml has no [docs] table"
    for area, patterns in areas.docs.items():
        assert area in areas.areas, f"[docs] area {area!r} is not in [areas]"
        for pattern in patterns:
            assert list(ROOT.glob(pattern)), f"{area}: no file matches {pattern}"


# --- realistic baselines and the full propose -> sync -> archive flow ---------

R_NEWTON = "### Requirement: Newton\nSource: issue 1\n\nThe solver must use Newton.\n"
R_DAMPED = (
    "### Requirement: Newton\nSource: issue 1\n\nThe solver must use damped Newton.\n"
)
R_LOG = "### Requirement: Logging\nSource: issue 2\n\nIt must log each step.\n"
R_TOL = "### Requirement: Tolerance\nSource: issue 4\n\nIt must stop at 1e-8.\n"
R_MUMPS = "### Requirement: MUMPS\nSource: issue 3\n\nIt must factor with MUMPS.\n"
FULL_DELTA = (
    "## ADDED Requirements\n\n" + R_MUMPS + "\n"
    "## MODIFIED Requirements\n\n"
    "### Requirement: Newton\nBefore: The solver must use Newton.\n"
    "Source: issue 1\n\nThe solver must use damped Newton.\n\n"
    "## REMOVED Requirements\n\n"
    "### Requirement: Logging\nReason: replaced by tracing\n"
)
SPEC = "specs/solver/specs/specs.md"
CHANGE = "specs/solver/changes/mumps/specs/specs.md"


def _baseline(reqs: list[str], synced: str = "2026-10-01") -> str:
    """Baseline laid out like the pre-fix template: `Last synced:` at the end."""
    head = "# Solver specification\n\n## Purpose\n\nSolves.\n\n## Requirements\n\n"
    return head + "\n".join(reqs) + f"\nLast synced: {synced}\n"


def _branch(repo: Path, name: str, start: str = "main") -> None:
    _git(repo, "checkout", "-q", "-b", name, start)


def _merge(repo: Path, name: str) -> None:
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", name)


def _proposed(toy: Path, m: Any) -> None:
    """main has a baseline (Newton, Logging) and an open change `mumps`."""
    _write(toy, SPEC, _baseline([R_NEWTON, R_LOG, R_TOL]))
    _commit(toy, "bootstrap baseline")
    _branch(toy, "propose")
    _write(toy, CHANGE, FULL_DELTA)
    _write(toy, "specs/solver/changes/mumps/proposal.md", "# Why\n")
    _write(toy, "src/equilibria/solver/a.py", "x = 2\n")
    _commit(toy, "propose")
    assert _run(m, "check-pr", "--base", "main") == 0
    _merge(toy, "propose")


# Newton modified, Logging removed, Tolerance untouched, MUMPS appended at the end
SYNCED = [R_DAMPED, R_TOL, R_MUMPS]


def test_parse_requirements_drops_last_synced() -> None:
    m = _load_module()
    reqs, _ = m.parse_requirements(_baseline([R_NEWTON], "2026-10-20"))
    assert reqs["Newton"] == "Source: issue 1\n\nThe solver must use Newton."


def test_new_area_template_has_no_requirements_and_date_on_top() -> None:
    m = _load_module()
    text = (ROOT / "specs/_templates/new-area/specs/specs.md").read_text("utf-8")
    assert m.parse_requirements(text) == ({}, [])
    assert text.index("Last synced:") < text.index("## Requirements")


def test_sync_mismatches() -> None:
    m = _load_module()
    d = m.parse_deltas(FULL_DELTA)
    old = m.parse_requirements(_baseline([R_NEWTON, R_LOG, R_TOL]))[0]
    good = m.parse_requirements(_baseline(SYNCED))[0]
    assert m.hand_edits(old, good, [d]) == []
    assert m.sync_mismatches(old, good, [d]) == []
    bad_mumps = R_MUMPS.replace("MUMPS.", "anything.")
    bad = m.parse_requirements(_baseline([R_DAMPED, R_TOL, bad_mumps]))[0]
    assert m.sync_mismatches(old, bad, [d]) == ["MUMPS"]
    # a REMOVED requirement must actually be gone
    kept = m.parse_requirements(_baseline([R_DAMPED, R_LOG + "more\n", R_TOL]))[0]
    assert m.sync_mismatches(old, kept, [d]) == ["Logging"]


def test_check_pr_realistic_hand_edits(
    toy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _load_module()
    _write(toy, SPEC, _baseline([R_NEWTON, R_LOG, R_TOL]))
    _commit(toy, "bootstrap baseline")
    _branch(toy, "edit")
    _write(toy, SPEC, _baseline(SYNCED, "2026-10-20"))
    _commit(toy, "edit without change")
    assert _run(m, "check-pr", "--base", "main") == 1
    out = capsys.readouterr().out
    for name in ("Newton", "Logging", "MUMPS"):
        assert f"requirement {name!r} edited by hand" in out


def test_flow_sync_then_archive(toy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    m = _load_module()
    _proposed(toy, m)
    # PR: sync only (and bump Last synced)
    _branch(toy, "sync")
    _write(toy, SPEC, _baseline(SYNCED, "2026-10-20"))
    _commit(toy, "sync")
    assert _run(m, "check-pr", "--base", "main") == 0, capsys.readouterr().out
    _merge(toy, "sync")
    # PR: archive only, after the sync landed
    _branch(toy, "archive")
    assert _run(m, "archive", "solver", "mumps") == 0
    _commit(toy, "archive")
    assert _run(m, "check-pr", "--base", "main") == 0, capsys.readouterr().out


def test_flow_sync_and_archive_same_pr(
    toy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _load_module()
    _proposed(toy, m)
    _branch(toy, "last")
    _write(toy, SPEC, _baseline(SYNCED, "2026-10-20"))
    assert _run(m, "archive", "solver", "mumps") == 0
    _commit(toy, "sync and archive")
    assert _run(m, "check-pr", "--base", "main") == 0, capsys.readouterr().out


def test_flow_archive_still_checks_sync_text(
    toy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _load_module()
    _proposed(toy, m)
    _branch(toy, "last")
    _write(toy, SPEC, _baseline([R_DAMPED, R_TOL, R_MUMPS.replace("MUMPS.", "X.")]))
    assert _run(m, "archive", "solver", "mumps") == 0
    _commit(toy, "sync and archive")
    assert _run(m, "check-pr", "--base", "main") == 1
    assert "'MUMPS' does not match" in capsys.readouterr().out


def test_flow_sync_text_differs_from_delta(
    toy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _load_module()
    _proposed(toy, m)
    _branch(toy, "sync")
    _write(toy, SPEC, _baseline([R_DAMPED, R_TOL, R_MUMPS.replace("MUMPS.", "X.")]))
    _commit(toy, "sync with other text")
    assert _run(m, "check-pr", "--base", "main") == 1
    out = capsys.readouterr().out
    assert "'MUMPS' does not match" in out
    assert "'Newton'" not in out and "'Tolerance'" not in out
