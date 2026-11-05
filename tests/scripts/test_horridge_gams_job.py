"""Horridge's SIMPLE, GAMS side (phase 3a-2 Task 6, run on the Mac, decision D4).

``data/horridge_simple/tools/gams_job.py`` runs one .gms of the tpmh0103 archive
in a scratch folder and keeps its ``RESULTS.gdx`` as ``results.gdx``; the driver
records the five folders ``gams_<form>_<experiment>/`` through write_source.py.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from equilibria.sources import DATA_ROOT, load_source

MODEL = DATA_ROOT / "horridge_simple"
sys.path.insert(0, str(MODEL / "tools"))
import gams_job  # noqa: E402  # ty: ignore[unresolved-import]

OK_LST = """GAMS 53.5.0  5dc151ca May 5, 2026  DAX-DAC arm 64bit/macOS
**** SOLVER STATUS     1 Normal Completion
**** MODEL STATUS      1 Optimal
**** SOLVER STATUS     1 Normal Completion
**** MODEL STATUS      1 Optimal
"""


def test_the_five_jobs() -> None:
    """NLP and MCP for fixcap and tariff10; MPSGE has no tariff variant."""
    assert {(j.form, j.experiment): j.gms for j in gams_job.JOBS} == {
        ("nlp", "fixcap"): "simpleNLP.gms",
        ("mcp", "fixcap"): "simpleMCP.gms",
        ("mpsge", "fixcap"): "mpsgevh.gms",
        ("nlp", "tariff10"): "simpleNLP_tariff.gms",
        ("mcp", "tariff10"): "simpleMCP_tariff.gms",
    }
    assert {j.folder for j in gams_job.JOBS} == {
        "gams_nlp_fixcap", "gams_mcp_fixcap", "gams_mpsge_fixcap",
        "gams_nlp_tariff10", "gams_mcp_tariff10",
    }  # fmt: skip


def test_release_is_read_from_the_lst() -> None:
    assert gams_job.release(OK_LST) == "53.5.0"
    assert gams_job.release("no banner") is None


def test_the_last_solve_must_be_normal_and_optimal() -> None:
    """The simulation is the last solve; it must end normally and (locally) optimal."""
    assert gams_job.solve_problems(OK_LST) == []
    infeasible = OK_LST[::-1].replace("lamitpO 1", "elbisaefnI yllacoL 5", 1)[::-1]
    assert gams_job.solve_problems(infeasible) == [
        "MODEL STATUS      5 Locally Infeasible"
    ]
    stopped = OK_LST[::-1].replace("noitelpmoC lamroN 1", "tpurretnI ecruoseR 3", 1)[
        ::-1
    ]
    assert gams_job.solve_problems(stopped) == [
        "SOLVER STATUS     3 Resource Interrupt"
    ]
    assert gams_job.solve_problems("no solve here") == [
        "no SOLVER STATUS in the listing"
    ]


def test_earlier_solves_go_to_the_notes() -> None:
    """simpleNLP's first solve is Horridge's iterlim=0 calibration check: CONOPT
    calls it infeasible. It does not fail the job; the record's notes say so."""
    check = OK_LST.replace(
        "1 Normal Completion", "11 Internal Solver Failure", 1
    ).replace("1 Optimal", "5 Locally Infeasible", 1)
    assert gams_job.solve_problems(check) == []
    assert gams_job.earlier_solves(check) == [
        "solve 1 of 2: SOLVER STATUS 11 Internal Solver Failure, "
        "MODEL STATUS 5 Locally Infeasible"
    ]
    assert gams_job.earlier_solves(OK_LST) == [
        "solve 1 of 2: SOLVER STATUS 1 Normal Completion, MODEL STATUS 1 Optimal"
    ]


def test_command_records_through_write_source(tmp_path: Path) -> None:
    job = gams_job.JOBS[0]
    cmd = gams_job.job_command(job, tool_version="53.5.0", gams="gams", root=tmp_path)
    s = " ".join(cmd)
    assert "write_source.py" in s and "--kind result" in s
    assert "--tool gams --tool-version 53.5.0" in s
    assert f"--variant form={job.form} --variant experiment={job.experiment}" in s
    assert f"--model-file {tmp_path / 'horridge_simple'}" in s
    assert f"--folder {tmp_path / 'horridge_simple' / job.folder}" in s
    assert f"--notes-from {gams_job.NOTE_FILE}" in s
    assert f"--solver {job.solver}" in s


GAMS = shutil.which("gams")


@pytest.mark.skipif(GAMS is None, reason="gams is not on the PATH")
def test_run_keeps_only_the_results(tmp_path: Path) -> None:
    """A real GAMS run of simpleMCP: one results.gdx, no listing with local paths."""
    out = tmp_path / "out"
    version = gams_job.gams_release(GAMS)
    rc = gams_job.main(["run", "--form", "mcp", "--experiment", "fixcap",
                        "--out", str(out), "--gams", GAMS, "--tool-version", version])  # fmt: skip
    assert rc == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [gams_job.NOTE_FILE, "results.gdx"]
    )


# ── the recorded folders (absent until the Mac run is committed) ──


def _ch(gdx: Path, sym: str) -> dict[str, float]:
    gdxdump = shutil.which("gdxdump")
    if gdxdump is None:
        pytest.skip("gdxdump is not on the PATH")
    text = subprocess.run([str(gdxdump), str(gdx), f"symb={sym}", "format=csv"],
                          capture_output=True, text=True, check=True).stdout  # fmt: skip
    rows = [r.replace('"', "").split(",") for r in text.splitlines()[1:] if r.strip()]
    return {",".join(r[:-1]): float(r[-1]) for r in rows}


@pytest.mark.parametrize("exp", ["fixcap", "tariff10"])
def test_mcp_and_nlp_agree(exp: str) -> None:
    """The same model as MCP and NLP: the %-changes agree to ~8 digits."""
    mcp, nlp = MODEL / f"gams_mcp_{exp}", MODEL / f"gams_nlp_{exp}"
    if not (mcp / "source.yaml").exists() or not (nlp / "source.yaml").exists():
        pytest.skip("Horridge GAMS results not recorded yet")
    for sym in ("CH_Z", "CH_XFAC", "CH_P", "CH_PFAC"):
        a, b = _ch(mcp / "results.gdx", sym), _ch(nlp / "results.gdx", sym)
        assert a and b, sym
        # a GDX stores no zero cells: CH_P(mnf,imp) is 4e-8 in the MCP, 0 in the NLP
        cells = a.keys() | b.keys()
        assert max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in cells) < 1e-6, sym


def test_recorded_folders_name_their_variant() -> None:
    for job in gams_job.JOBS:
        folder = MODEL / job.folder
        if not (folder / "source.yaml").exists():
            pytest.skip("Horridge GAMS results not recorded yet")
        rec = load_source(folder)
        assert rec.tool == "gams" and rec.status == "verified"
        assert dict(rec.variant) == {"form": job.form, "experiment": job.experiment}
        assert set(rec.raw["outputs"]) == {"results.gdx"}
