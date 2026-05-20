"""L5 (CI side): assert read_har(write_har(read_har(p))) matches the
harpy3 oracle dump committed under tests/babel/har/golden/.

The golden JSON files are produced by scripts/har/oracle_check.py in a
sandbox venv with harpy3 installed. CI never installs harpy3; it only
reads the committed JSON files and compares them to what our own reader
sees on the writer's output.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from equilibria.babel.har import read_har, write_har

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "src/equilibria/templates/reference/gtap/data"
GOLDEN_DIR = Path(__file__).parent / "golden"


def _golden_path(fixture: Path) -> Path:
    return GOLDEN_DIR / f"{fixture.parent.name}_{fixture.stem}.json"


FIXTURES = [
    DATA / "nus333/sets.har",
    DATA / "nus333/basedata.har",
    DATA / "nus333/baserate.har",
    DATA / "nus333/default.prm",
    DATA / "9x10/sets.har",
    DATA / "9x10/basedata.har",
]


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.name)
def test_writer_matches_oracle(fixture: Path, tmp_path: Path):
    if not fixture.exists():
        pytest.skip(f"fixture not present: {fixture}")
    gp = _golden_path(fixture)
    if not gp.exists():
        pytest.skip(f"golden not present: {gp}; run scripts/har/oracle_check.py refresh")

    golden = json.loads(gp.read_text())["headers"]

    out = tmp_path / f"og_{fixture.name}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        write_har(out, read_har(fixture))
    d = read_har(out)

    assert set(d.keys()) == set(golden.keys())
    for name, expected in golden.items():
        got = d[name]
        assert list(got.shape) == expected["shape"], f"shape mismatch for {name}"
        assert got.set_names == expected["set_names"], f"set_names mismatch for {name}"
        assert got.set_elements == expected["set_elements"], f"set_elements mismatch for {name}"
        if expected["dtype_kind"] in ("f",):
            tol = 1e-4
            assert abs(float(got.array.sum()) - expected["stats"]["sum"]) < tol * (
                1.0 + abs(expected["stats"]["sum"])
            ), f"sum mismatch for {name}"
        elif expected["dtype_kind"] == "i":
            assert int(got.array.sum()) == int(expected["stats"]["sum"]), (
                f"int sum mismatch for {name}"
            )
        else:
            arr = np.asarray(got.array).ravel()
            if arr.size == 0:
                # Empty set — golden may record empty string for "first".
                continue
            first = str(arr[0]).strip()
            assert first == str(expected["stats"]["first"]).strip(), (
                f"first-element mismatch for {name}"
            )
