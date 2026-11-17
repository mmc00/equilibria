"""The saved Pyomo photos: where they live and how the regeneration command writes them."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from equilibria.family import photos
from equilibria.sources import DATA_ROOT, load_source, sha256_file

ROOT = Path(__file__).resolve().parents[2]
REGEN = ROOT / "scripts" / "family" / "regen_photos.py"


def test_photo_dir_name():
    got = photos.photo_dir("gtap7_3x3", 1, "capFix", "gtap")
    assert got == DATA_ROOT / "gtap7" / "gtap7_3x3" / "equilibria_pyomo-photo_tol_today"
    got = photos.photo_dir("gtap7_3x3", 0, "capFix", "gtap")
    assert got.name == "equilibria_pyomo-photo_gtap_ifsub0_capFix"


def test_variants_cover_the_per_step_consumers():
    keep = {(ds, ifsub, savf): k for ds, ifsub, savf, _, k in photos.VARIANTS}
    for v in [
        ("gtap7_3x3", 1, "capFix"),
        ("gtap7_3x3", 0, "capFix"),
        ("gtap7_3x3", 1, "capFlex"),
        ("gtap7_10x7", 1, "capFix"),
    ]:
        assert keep[v] == "all"
    assert keep[("gtap7_15x10", 1, "capFix")] == "gate"


def test_regen_refuses_without_mumps():
    env = {**os.environ, "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}
    p = subprocess.run(
        [
            sys.executable,
            str(REGEN),
            "--dataset",
            "gtap7_3x3",
            "--root",
            "/nonexistent",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0
    assert "mumps" in p.stderr.lower()


@pytest.mark.slow
def test_write_photo_records_print_and_hashes(tmp_path):
    """Solves 3x3 with Pyomo (slow: PR CI never runs Pyomo)."""
    pytest.importorskip("mumps")
    spec = importlib.util.spec_from_file_location("regen_photos", REGEN)
    assert spec is not None and spec.loader is not None
    regen_photos = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regen_photos)

    out = regen_photos.write_photo(
        "gtap7_3x3", 1, "capFix", "gtap", "gate", root=tmp_path
    )
    assert out == tmp_path / "gtap7" / "gtap7_3x3" / "equilibria_pyomo-photo_tol_today"
    rec = load_source(out)
    assert (rec.kind, rec.tool, rec.dataset) == (
        "result",
        "equilibria-pyomo",
        "gtap7_3x3",
    )
    files = {p.name for p in out.iterdir()} - {"source.yaml"}
    assert set(rec.raw["outputs"]) == files
    assert {"final.pkl", "settled.pkl", "model_print.json"} <= files
    for name, sha in rec.raw["outputs"].items():
        assert sha256_file(out / name) == sha
    data = rec.raw["generated_from"]["data"]
    assert data["path"] == "gtap7/gtap7_3x3"
    assert set(data["sha256"]) == set(photos.DATA_FILES)
    mp = json.loads((out / "model_print.json").read_text())
    assert {"data", "closure", "pyomo_code", "shared_code", "env"} <= set(mp)
    assert mp["env"]["linsolve"] == "mumps"


def _regen_module():
    spec = importlib.util.spec_from_file_location("regen_photos", REGEN)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_regen_gates_run_the_pyomo_solve_cases_of_the_datasets(monkeypatch):
    """The gates are the cases that left PR CI (``pyomo_solve``), on the regenerated
    datasets, from every file that holds them."""
    regen = _regen_module()
    monkeypatch.setattr(regen.shutil, "which", lambda name: "/bin/" + name)
    cmd = regen.gate_command(["gtap7_3x3", "gtap7_10x7"])
    assert cmd[cmd.index("pytest") + 1 :][:3] == ["-q", "-m", "pyomo_solve"]
    assert cmd[cmd.index("-k") + 1] == "gtap7_3x3 or gtap7_10x7"
    files = {c for c in cmd if c.startswith("tests/")}
    assert files == {
        "tests/templates/gtap/test_gtap7_mcp_parity.py",
        "tests/templates/gtap/test_gtap7_nlp_parity.py",
        "tests/templates/gtap/test_gtap7_gempack_parity.py",
        "tests/templates/gtap/test_ifsub_equivalence.py",
    }


def test_regen_gates_refuse_to_run_without_gdxdump(monkeypatch):
    """Without gdxdump the GAMS gates would all skip and pass: refuse instead."""
    regen = _regen_module()
    monkeypatch.setattr(regen.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="gdxdump"):
        regen.gate_command(["gtap7_3x3"])
