"""plan_test_groups.py: split the CI suites by file under a time budget."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "plan_test_groups.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("plan_test_groups", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_file_seconds_sums_per_file() -> None:
    got = _module().file_seconds(
        {"tests/a.py::t1": 1.5, "tests/a.py::C::t2[x]": 2.0, "tests/b.py::t": 4.0}
    )
    assert got == {"tests/a.py": 3.5, "tests/b.py": 4.0}


def test_big_file_gets_dedicated_split_groups() -> None:
    groups = _module().plan_groups(
        ["tests/big/test_big.py", "tests/small.py"],
        {"tests/big/test_big.py": 900.0, "tests/small.py": 10.0},
        budget=360,
        max_groups=10,
    )
    big = [g for g in groups if g["split"] is not None]
    assert [g["split"] for g in big] == [[1, 3], [2, 3], [3, 3]]
    assert all(g["paths"] == ["tests/big/test_big.py"] for g in big)
    assert big[0]["name"] == "test_big 1/3"
    packed = [g for g in groups if g["split"] is None]
    assert [g["paths"] for g in packed] == [["tests/small.py"]]


def test_small_files_packed_under_budget() -> None:
    seconds = {f"tests/f{i}.py": float(30 + 7 * i) for i in range(20)}
    groups = _module().plan_groups(list(seconds), seconds, budget=360, max_groups=10)
    for g in groups:
        assert sum(seconds[p] for p in g["paths"]) <= 360
    flat = [p for g in groups for p in g["paths"]]
    assert sorted(flat) == sorted(seconds)
    # minimal: total 20*30+7*190=1930 s needs at least 6 groups
    assert len(groups) == 6


def test_new_file_without_durations_is_placed() -> None:
    groups = _module().plan_groups(
        ["tests/new.py", "tests/old.py"],
        {"tests/old.py": 5.0},
        budget=360,
        max_groups=4,
    )
    flat = [p for g in groups for p in g["paths"]]
    assert sorted(flat) == ["tests/new.py", "tests/old.py"]


def test_too_many_groups_raises() -> None:
    seconds = {f"tests/f{i}.py": 300.0 for i in range(5)}
    with pytest.raises(ValueError, match="5"):
        _module().plan_groups(list(seconds), seconds, budget=360, max_groups=3)


def test_output_is_deterministic() -> None:
    seconds = {f"tests/f{i}.py": 50.0 for i in range(12)}
    files = list(seconds)
    a = _module().plan_groups(files, seconds, budget=360, max_groups=10)
    b = _module().plan_groups(list(reversed(files)), seconds, budget=360, max_groups=10)
    assert a == b


def test_matrix_json_shape(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    durations = tmp_path / "d.json"
    durations.write_text(
        json.dumps(
            {"tests/big.py::t": 800.0, "tests/a.py::t": 10.0, "tests/b.py::t": 20.0}
        )
    )
    listing = tmp_path / "files.txt"
    listing.write_text("tests/a.py\ntests/b.py\ntests/big.py\n")
    rc = _module().main(
        [
            "--durations",
            str(durations),
            "--files-list",
            str(listing),
            "--budget",
            "360",
            "--max-groups",
            "10",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert "\n" not in out
    include = json.loads(out)["include"]
    assert include and all(
        set(item) == {"group", "paths", "split"}
        and all(isinstance(v, str) for v in item.values())
        for item in include
    )
    assert {item["split"] for item in include} == {"", "1 3", "2 3", "3 3"}
    assert any(item["paths"] == "tests/a.py tests/b.py" for item in include)
