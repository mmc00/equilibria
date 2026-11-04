"""Every Burfisher exercise with a GEMPACK solution has a notebook that replicates it.

Each notebook in ``examples/notebooks/gtap7/burfisher/`` builds the exercise from
scratch (data, closure, shock, solve) and checks every level against GEMPACK's
solution of the same ``.EXP``, with the criterion of the paper's Table 3a: every
cell within 0.5%. The notebooks are written by ``make_notebooks.py`` next to them;
the committed files must be exactly what it writes, so a change goes through it.

Not checked against GEMPACK here, each for its reason:

- ME4A, ME4B: neither GEMPACK ("Accuracy too low") nor equilibria (code 2) reach
  the shock, so there is nothing to compare.
- TBL53, GTAPV7: no GEMPACK Gragg solution yet (phase 3a-2 of one-data-folder
  adds them).
- ME9A: up to 0.77% from GEMPACK (``pe`` ROW LAND), while it matches GAMS. GEMPACK's
  own log rates many of its results to 2 figures; a rerun with more steps decides.
  Its notebook is kept, without the check.

NOT marked slow on purpose: the CI skips ``slow`` (see ``test_cge_examples_run.py``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NB_DIR = ROOT / "examples" / "notebooks" / "gtap7" / "burfisher"
GENERATOR = NB_DIR / "make_notebooks.py"

MATCHED = (
    "TBL45A", "TBL45B", "TBL45C", "TBL46A", "TBL46B", "TBL46C",
    "TBL54A", "TBL54B", "TBL55", "TBL62A", "TBL62B", "TBL62C",
    "TBL63A", "TBL63B", "TBL64", "TBL65A", "TBL65B", "TBL66",
    "TBL67", "TBL75A", "TBL75B", "TBL77", "TBL78", "TBL79",
    "TBL813", "TBL93", "TBL94", "ME3A", "ME3B", "ME3C",
    "ME5", "ME7A", "ME7B", "ME8", "ME9B", "ME9C", "ME9D",
)  # fmt: skip
UNCHECKED = ("ME9A",)


def _code(notebook: Path) -> str:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return "\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def _run(code: str, where: Path) -> subprocess.CompletedProcess:
    script = where / "notebook.py"
    script.write_text(code, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=NB_DIR,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def test_every_exercise_has_a_notebook() -> None:
    """And nothing else: an empty parametrize below would pass by vacuity."""
    found = {p.stem for p in NB_DIR.glob("*.ipynb")}
    assert found == {e.lower() for e in MATCHED + UNCHECKED}


def test_notebooks_are_what_the_generator_writes(tmp_path: Path) -> None:
    r = subprocess.run(
        [sys.executable, str(GENERATOR), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    written = sorted(p.name for p in tmp_path.glob("*.ipynb"))
    assert written == sorted(p.name for p in NB_DIR.glob("*.ipynb"))
    for name in written:
        assert (tmp_path / name).read_bytes() == (NB_DIR / name).read_bytes(), name


def test_notebooks_name_the_prm_files_as_they_are() -> None:
    """With the exact case: Linux (the CI) does not find 3x3CES.prm as 3x3ces.prm."""
    import re

    files = set(os.listdir(ROOT / "data" / "gtap7" / "nus333"))
    for nb in sorted(NB_DIR.glob("*.ipynb")):
        (prm,) = re.findall(r'default_path=har / "([^"]+)"', _code(nb))
        assert prm in files, (nb.name, prm)


@pytest.mark.needs_path
@pytest.mark.parametrize("exp", MATCHED)
def test_notebook_matches_gempack(exp: str, tmp_path: Path) -> None:
    r = _run(_code(NB_DIR / f"{exp.lower()}.ipynb"), tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "every cell within 0.5% of GEMPACK" in r.stdout


@pytest.mark.needs_path
def test_a_wrong_shock_fails(tmp_path: Path) -> None:
    """The check can fail: TBL55 with 9% instead of 10% more capital."""
    code = _code(NB_DIR / "tbl55.ipynb")
    shock = '"aft": {("USA", "CAPITAL"): 10.0}'
    assert shock in code
    r = _run(code.replace(shock, '"aft": {("USA", "CAPITAL"): 9.0}'), tmp_path)
    assert r.returncode != 0
    assert "not every cell within 0.5% of GEMPACK" in r.stderr
