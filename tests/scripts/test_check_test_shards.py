"""check_test_shards.py: every expected test runs exactly once across shards."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "check_test_shards.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("check_test_shards", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(nodeid: str, kind: str = "", message: str = "") -> str:
    """One <testcase> in the shape pytest writes with junit_family=xunit1."""
    file, _, rest = nodeid.partition("::")
    parts = rest.split("::")
    name = parts[-1]
    classname = ".".join([file.removesuffix(".py").replace("/", "."), *parts[:-1]])
    body = ""
    if kind == "failure":
        body = f'<failure message="{message}">boom</failure>'
    elif kind == "error":
        body = f'<error message="{message}">boom</error>'
    elif kind == "skipped":
        body = f'<skipped type="pytest.skip" message="{message}">x</skipped>'
    elif kind == "xfail":
        body = f'<skipped type="pytest.xfail" message="{message}"/>'
    return f'<testcase classname="{classname}" name="{name}" file="{file}">{body}</testcase>'


def _junit(path: Path, *cases: str) -> Path:
    path.write_text(f"<testsuites><testsuite>{''.join(cases)}</testsuite></testsuites>")
    return path


def test_complete_split_passes(tmp_path: Path) -> None:
    j1 = _junit(tmp_path / "1.xml", _case("tests/a.py::t1"), _case("tests/a.py::t2"))
    j2 = _junit(tmp_path / "2.xml", _case("tests/a.py::t3"))
    expected = ["tests/a.py::t1", "tests/a.py::t2", "tests/a.py::t3"]
    assert _module().check_shards(expected, [j1, j2]) == []


def test_missing_test_fails(tmp_path: Path) -> None:
    j1 = _junit(tmp_path / "1.xml", _case("tests/a.py::t1"), _case("tests/a.py::t2"))
    expected = ["tests/a.py::t1", "tests/a.py::t2", "tests/a.py::t3"]
    problems = _module().check_shards(expected, [j1])
    assert len(problems) == 1
    assert "a.py::t3" in problems[0] and "missing" in problems[0]


def test_duplicate_test_fails(tmp_path: Path) -> None:
    j1 = _junit(tmp_path / "1.xml", _case("tests/a.py::t1"))
    j2 = _junit(tmp_path / "2.xml", _case("tests/a.py::t1"))
    problems = _module().check_shards(["tests/a.py::t1"], [j1, j2])
    assert len(problems) == 1
    assert "a.py::t1" in problems[0] and "more than once" in problems[0]


def test_new_test_is_expected_once(tmp_path: Path) -> None:
    # A test absent from the durations file is still in the collected list.
    j1 = _junit(tmp_path / "1.xml", _case("tests/new.py::TestNew::test_x[a-1]"))
    assert _module().check_shards(["tests/new.py::TestNew::test_x[a-1]"], [j1]) == []


def test_missing_shard_report_fails() -> None:
    problems = _module().check_shards(["tests/a.py::t1"], [])
    assert problems and all("missing" in p for p in problems)


def test_failed_test_reported(tmp_path: Path) -> None:
    j1 = _junit(tmp_path / "1.xml", _case("tests/a.py::t1", "failure", "assert 1 == 2"))
    problems = _module().check_shards(["tests/a.py::t1"], [j1])
    assert len(problems) == 1
    assert "a.py::t1" in problems[0] and "failed" in problems[0]


def test_skip_reason_change_is_reported(tmp_path: Path) -> None:
    mod = _module()
    base = mod.load_junit(
        [_junit(tmp_path / "b.xml", _case("tests/a.py::t1", "skipped", "no X"))]
    )
    cur = mod.load_junit(
        [_junit(tmp_path / "c.xml", _case("tests/a.py::t1", "skipped", "no Y"))]
    )
    problems = mod.compare_runs(base, cur)
    assert len(problems) == 1
    assert "no X" in problems[0] and "no Y" in problems[0]


def test_outcome_change_is_reported(tmp_path: Path) -> None:
    mod = _module()
    base = mod.load_junit(
        [_junit(tmp_path / "b.xml", _case("tests/a.py::t1", "skipped", "no X"))]
    )
    cur = mod.load_junit([_junit(tmp_path / "c.xml", _case("tests/a.py::t1"))])
    problems = mod.compare_runs(base, cur)
    assert len(problems) == 1
    assert "skipped" in problems[0] and "passed" in problems[0]
    assert mod.load_junit(
        [_junit(tmp_path / "x.xml", _case("tests/a.py::t1", "xfail", "r"))]
    ) == {"tests/a.py::t1": ("xfailed", "r")}


def _module_case(file: str, kind: str, message: str = "") -> str:
    """A whole module skipped or failing at collection: pytest writes it with an
    empty classname and the dotted module as its name, in every shard."""
    name = file.removesuffix(".py").replace("/", ".")
    tag = "skipped" if kind == "skipped" else "error"
    return (
        f'<testcase classname="" name="{name}" file="{file}">'
        f'<{tag} message="{message}">x</{tag}></testcase>'
    )


def test_module_skip_in_every_shard_is_not_a_test(tmp_path: Path) -> None:
    m = _module()
    skip = _module_case("tests/b/test_opt.py", "skipped", "harpy3 not installed")
    j1 = _junit(tmp_path / "j1.xml", _case("tests/a/test_x.py::t1"), skip)
    j2 = _junit(tmp_path / "j2.xml", _case("tests/a/test_x.py::t2"), skip)
    expected = ["tests/a/test_x.py::t1", "tests/a/test_x.py::t2"]
    assert m.check_shards(expected, [j1, j2]) == []


def test_module_collection_error_is_reported(tmp_path: Path) -> None:
    m = _module()
    err = _module_case("tests/b/test_broken.py", "error", "ImportError")
    j1 = _junit(tmp_path / "j1.xml", _case("tests/a/test_x.py::t1"), err)
    problems = m.check_shards(["tests/a/test_x.py::t1"], [j1])
    assert len(problems) == 1
    assert "failed" in problems[0] and "test_broken" in problems[0]


def test_missing_test_of_unlisted_file_gets_refresh_hint(tmp_path: Path) -> None:
    j1 = _junit(tmp_path / "1.xml", _case("tests/a.py::t1"))
    problems = _module().check_shards(
        ["tests/a.py::t1", "tests/new/test_new.py::t1"], [j1], suite="python"
    )
    assert len(problems) == 1
    assert ".github/test_files_python.txt" in problems[0]
    assert "--refresh-files" in problems[0]


def test_missing_test_of_a_file_that_ran_has_no_hint(tmp_path: Path) -> None:
    j1 = _junit(tmp_path / "1.xml", _case("tests/a.py::t1"))
    problems = _module().check_shards(["tests/a.py::t1", "tests/a.py::t2"], [j1])
    assert len(problems) == 1 and "--refresh-files" not in problems[0]
