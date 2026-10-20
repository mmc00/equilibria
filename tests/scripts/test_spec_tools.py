from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import Any

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
    ).stdout.split()
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
