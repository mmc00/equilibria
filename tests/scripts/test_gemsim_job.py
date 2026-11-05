"""gemsim_job.py (gragg) and target_shocks.py: Burfisher with Gragg on the levels TABs.

Step 5 of corridas_paper.ps1 in data/: each .exp of data/gtap7/nus333 runs under
GEMSIM on a TAB compiled from data/gtap7/gempack_rungtap375_levels. Johansen
1-step experiments become Gragg 2-4-6, and TBL53's ``target%`` shocks (which
only the RunGTAP GUI solves) become plain shocks (maintainer, 2026-11-03).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from equilibria.babel.har.symbols import HeaderArray
from equilibria.babel.har.writer import write_har
from equilibria.sources import DATA_ROOT, GEMPACK_MODEL_DIR

sys.path.insert(0, str(GEMPACK_MODEL_DIR / "tools"))
import gemsim_job  # noqa: E402  # ty: ignore[unresolved-import]
import target_shocks  # noqa: E402  # ty: ignore[unresolved-import]

ROOT = Path(__file__).resolve().parents[2]
NUS333 = DATA_ROOT / "gtap7" / "nus333"
COMM = ["AGR", "MFG", "SER"]


# --- target% (TBL53) ---------------------------------------------------------


def test_shk_reader_agrees_with_mk_exp_directos() -> None:
    sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
    from mk_exp_directos import leer_shk_etiquetado  # ty: ignore[unresolved-import]

    assert target_shocks.read_shk(NUS333 / "tfd.shk") == leer_shk_etiquetado(
        NUS333 / "tfd.shk"
    )


def test_tbl53_target_becomes_plain_shocks() -> None:
    text = (NUS333 / "tbl53.exp").read_text(encoding="latin-1")
    out, targets = target_shocks.translate(text, NUS333)
    shocks = [
        ln.strip() for ln in out.splitlines() if ln.lstrip().startswith("Shock tfd(")
    ]
    assert len(shocks) == 9 and len(targets) == 9
    # 100*[(1 + X/100)*(1 + HAT/100) - 1]: AGR,AGR,USA has a -3.95% rate, target 5%
    agr = next(s for s in shocks if s.startswith('Shock tfd("AGR", "AGR", "USA")'))
    assert float(agr.split("=")[1].rstrip(";")) == pytest.approx(9.321930, abs=1e-6)
    assert {t.target for t in targets} == {5.0, 10.0, 2.0}
    # every live line is a plain shock now; target% survives only in comments
    live = [ln for ln in out.splitlines() if not ln.lstrip().startswith("!")]
    assert not any("target%" in ln for ln in live)


def test_translate_leaves_other_exp_untouched() -> None:
    text = (NUS333 / "me9c.exp").read_text(encoding="latin-1")
    assert target_shocks.translate(text, NUS333) == (text, [])


def _updated(tmp_path: Path, power: float) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    shape = (3, 3, 2)
    vdfb = np.ones(shape, dtype=np.float32) * 100
    vdfp = vdfb * power
    sets = [COMM, COMM, ["USA", "ROW"]]
    names = ["COMM", "ACTS", "REG"]
    har = tmp_path / "updated.har"
    write_har(har, {
        "VDFB": HeaderArray("VDFB", "VDFB", "dom. purchases at basic prices", vdfb, names, sets),
        "VDFP": HeaderArray("VDFP", "VDFP", "dom. purchases at purchaser's prices", vdfp, names, sets),
    })  # fmt: skip
    return har


def test_targets_are_checked_in_the_updated_data(tmp_path: Path) -> None:
    targets = [target_shocks.Target("tfd", ("AGR", "AGR", "USA"), 5.0)]
    assert target_shocks.check(_updated(tmp_path / "hit", 1.05), targets) == []
    bad = target_shocks.check(_updated(tmp_path / "missed", 1.04), targets)
    assert len(bad) == 1 and "tfd" in bad[0] and "AGR" in bad[0]


# --- the gragg job -------------------------------------------------------------


def test_johansen_becomes_gragg_once() -> None:
    text = "Method = Johansen;\nSteps = 1;\nExogenous pop;\n"
    out, how = gemsim_job.gragg_method(text)
    assert how == "Johansen->Gragg 2-4-6"
    assert out.count("Method = Gragg;") == 1 and out.count("Steps = 2 4 6;") == 1
    same, how = gemsim_job.gragg_method("Method = Gragg;\nSteps = 6 8 10;\n")
    assert how == "as the .exp" and "Steps = 6 8 10;" in same
    with pytest.raises(gemsim_job.JobError, match="Method"):
        gemsim_job.gragg_method("Method = Gragg;\nMethod = Euler;\n")


def test_dir_exp_for_me8_and_tbl813() -> None:
    assert gemsim_job.exp_file(NUS333, "me8").name == "me8-dir.exp"
    assert gemsim_job.exp_file(NUS333, "tbl813").name == "tbl813-dir.exp"
    assert gemsim_job.exp_file(NUS333, "me9c").name == "me9c.exp"


def _fake_gemsim(console: str, files: tuple[str, ...]):
    seen: dict[str, object] = {}

    def run(cmd: list[str], cwd: Path, console_path: Path) -> int:
        seen["cwd"], seen["cmd"] = cwd, cmd
        out = console_path.parent
        for f in files:
            (out / f).write_bytes(b"x")
        console_path.write_text(console)
        return 0

    return run, seen


BANNER = " <GEMPACK Release 11.3.004   January 2015>\n"


def _gragg(tmp_path: Path, exp: str = "me9c") -> list[str]:
    out = tmp_path / "out"
    out.mkdir()
    return ["gragg", "--experiment", exp, "--package", str(NUS333),
            "--aux", str(tmp_path / "work" / "gtapuv7_levels" / "gtapuv7_levels"),  # as the driver
            "--gemsim", str(tmp_path / "gemsim.exe"),
            "--out", str(out), "--expect-release", "11.3.004"]  # fmt: skip


def test_runs_from_a_copy_of_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, seen = _fake_gemsim(
        BANNER + "completed without error\n", ("solution.sl4", "updated.har")
    )
    monkeypatch.setattr(gemsim_job, "run", run)
    assert gemsim_job.main(_gragg(tmp_path)) == 0
    cwd = Path(str(seen["cwd"]))
    assert not cwd.resolve().is_relative_to(DATA_ROOT)  # never runs inside data/
    # GEMSIM writes its current directory into the console and the .sl4: it is
    # the work folder (no user name), not %TEMP%
    assert cwd.resolve() == (tmp_path / "work" / "package" / "me9c").resolve()
    assert not cwd.exists()  # removed after the run
    cmf = (tmp_path / "out" / "run.cmf").read_text(encoding="latin-1")
    assert 'auxiliary files = "' in cmf and "gtapuv7_levels" in cmf
    assert 'solution file = "' in cmf and "Steps = 6 8 10;" in cmf


def test_gemsim_needs_normal_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gemsim writes the .sl4 even when it fails (ME4A/B: "Accuracy too low")."""
    run, _ = _fake_gemsim(
        BANNER + "Accuracy too low\n", ("solution.sl4", "updated.har")
    )
    monkeypatch.setattr(gemsim_job, "run", run)
    assert gemsim_job.main(_gragg(tmp_path)) == 1


def test_gragg_refuses_another_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = _fake_gemsim(" <GEMPACK Release 12.1.003   January 2020>\ncompleted without error\n",
                          ("solution.sl4", "updated.har"))  # fmt: skip
    monkeypatch.setattr(gemsim_job, "run", run)
    assert gemsim_job.main(_gragg(tmp_path)) == 1


def test_tbl53_checks_its_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TBL53 is recorded only if the updated data has the target tax rates."""

    def run(cmd: list[str], cwd: Path, console_path: Path) -> int:
        (console_path.parent / "solution.sl4").write_bytes(b"x")
        _updated(console_path.parent, 1.0)  # rates unchanged: the targets were missed
        console_path.write_text(BANNER + "completed without error\n")
        return 0

    monkeypatch.setattr(gemsim_job, "run", run)
    assert gemsim_job.main(_gragg(tmp_path, "tbl53")) == 1
    assert "target%" in (tmp_path / "out" / "run.cmf").read_text(encoding="latin-1")


# --- compiling the levels TAB -------------------------------------------------


def test_compile_refuses_an_unverified_model(tmp_path: Path) -> None:
    from tests.data._records import write_record

    root = tmp_path / "data"
    model = root / "gtap7" / "gempack_rungtap375_levels"
    model.mkdir(parents=True)
    (model / "gtapuv7_levels.tab").write_text("edited")
    (model / "gtapuv7_levels.sti").write_text("sti")
    write_record(model, {"kind": "model", "model": "gtap7", "source": "t",
                         "outputs": {"gtapuv7_levels.tab": "0" * 64,
                                     "gtapuv7_levels.sti": "0" * 64},
                         "status": "verified"})  # fmt: skip
    with pytest.raises(gemsim_job.JobError, match="does not verify"):
        gemsim_job.compile_levels_tab("gtapuv7_levels", levels_model=model,
                                      work=tmp_path / "work", tablo=tmp_path / "tablo.exe",
                                      root=root)  # fmt: skip


def test_compile_runs_tablo_in_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def run(cmd: list[str], cwd: Path, console_path: Path) -> int:
        calls.append(cwd)
        for ext in (".gss", ".gst"):
            (cwd / f"gtapuv7_levels{ext}").write_bytes(b"x")
        console_path.write_text(BANNER)
        return 0

    monkeypatch.setattr(gemsim_job, "run", run)
    work = tmp_path / "work"
    model = DATA_ROOT / "gtap7" / "gempack_rungtap375_levels"
    for _ in range(2):  # the second call finds it compiled from the same files
        gemsim_job.compile_levels_tab("gtapuv7_levels", levels_model=model, work=work,
                                      tablo=tmp_path / "tablo.exe", root=DATA_ROOT)  # fmt: skip
    assert calls == [work]
    assert (work / "gtapuv7_levels.tab").read_bytes() == (
        model / "gtapuv7_levels.tab"
    ).read_bytes()


def test_gemsim_scratch_files_are_not_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GEMSIM leaves <cmf>-GPXXX<n>.ud<k> work files and the solution's .slc next to
    the solution: not results (the .sl4 and the updated data are the record;
    maintainer, 2026-11-03)."""
    files = (
        "solution.sl4",
        "updated.har",
        "solution.slc",
        "solution.aal",
        "run-GPXXX7.ud8",
        "run-GPXXX2.ud3",
    )
    run, _ = _fake_gemsim(BANNER + "completed without error\n", files)
    monkeypatch.setattr(gemsim_job, "run", run)
    assert gemsim_job.main(_gragg(tmp_path)) == 0
    left = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert not any(
        "GPXXX" in name or name.endswith((".slc", ".aal")) for name in left
    ), left


# --- horridge and levels6 ------------------------------------------------------


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch):
    sys.path.insert(0, str(Path(__file__).parent))
    from _fake_gempack import FakeGempack

    f = FakeGempack()
    monkeypatch.setattr(gemsim_job, "run", f)
    monkeypatch.setattr(gemsim_job, "RETRY_PAUSE_S", 0)
    return f


def _out(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    return out


def test_horridge_runs_tablo_gemsim_sltoht(tmp_path: Path, fake) -> None:
    out = _out(tmp_path)
    argv = ["horridge", "--experiment", "tariff10", "--model-dir", str(DATA_ROOT / "horridge_simple"),
            "--tablo", "C:/GP/tablo.exe", "--gemsim", "C:/GP/gemsim.exe", "--sltoht", "C:/GP/sltoht.exe",
            "--release", "sltoht.exe=11.3.004", "--work", str(tmp_path / "work"),
            "--out", str(out), "--expect-release", "11.3.004"]  # fmt: skip
    assert gemsim_job.main(argv) == 0
    assert [p for p, _ in fake.calls] == ["tablo.exe", "gemsim.exe", "sltoht.exe"]
    assert {p.name for p in out.iterdir()} == {
        "run.cmf", "solution.sl4", "updated.har", "sl4dump.har", "tablo_console.txt",
        "gemsim_console.txt", "sltoht_console.txt",
    }  # fmt: skip


def _levels6(tmp_path: Path, rordelta: str) -> list[str]:
    closure = "capfix" if rordelta == "0" else "capflex"
    dump = (
        DATA_ROOT
        / "gtap7"
        / "gtap7_3x3"
        / f"gempack_tm10_s8-16-32_{closure}"
        / "sl4dump.har"
    )
    return ["levels6", "--dataset", "gtap7_3x3", "--rordelta", rordelta,
            "--aux", str(tmp_path / "work" / "gtapv7_levels" / "gtapv7_levels"),
            "--sl4dump", str(dump), "--gemsim", "C:/GP/gemsim.exe", "--sltoht", "C:/GP/sltoht.exe",
            "--work", str(tmp_path / "work"), "--out", str(_out(tmp_path)),
            "--expect-release", "11.3.004"]  # fmt: skip


def test_levels6_runs_the_patched_tab(tmp_path: Path, fake) -> None:
    assert gemsim_job.main(_levels6(tmp_path, "1")) == 0
    out = tmp_path / "out"
    assert {p.name for p in out.iterdir()} == {
        "run.cmf", "solution.sl4", "gemsim_console.txt", "sl4levels.har",
        "export_console.txt",
    }  # fmt: skip
    aux = str(tmp_path / "work" / "gtapv7_levels" / "gtapv7_levels")
    assert f"Auxiliary files = {aux} ;" in (out / "run.cmf").read_text()


def test_levels6_capfix_keeps_its_prm(tmp_path: Path, fake) -> None:
    """default_rdlt0.prm is _force_rdlt's .prm (it replaced the hand-made Windows one in 3b)."""
    sys.path.insert(0, str(GEMPACK_MODEL_DIR / "tools"))
    from run_gempack_matrix import _force_rdlt  # ty: ignore[unresolved-import]

    assert gemsim_job.main(_levels6(tmp_path, "0")) == 0
    prm = tmp_path / "out" / "default_rdlt0.prm"
    expected = tmp_path / "expected.prm"
    expected.write_bytes(
        (DATA_ROOT / "gtap7" / "gtap7_3x3" / "default.prm").read_bytes()
    )
    _force_rdlt(expected, 0)
    assert prm.read_bytes() == expected.read_bytes()


def test_levels6_keeps_the_xcheck_drops(tmp_path: Path, fake) -> None:
    """The patched TAB against the unpatched dump may drop variables (CNTendwr...):
    they are named in export_console.txt and the run still counts."""
    fake.mismatch = 2
    assert gemsim_job.main(_levels6(tmp_path, "1")) == 0
    assert "xcheck-mismatch=2" in (tmp_path / "out" / "export_console.txt").read_text()


def test_retry_waits_before_the_second_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake
) -> None:
    """An immediate retry failed like the first attempt (full 3a-2 run): wait first."""
    slept: list[float] = []
    monkeypatch.setattr(gemsim_job, "RETRY_PAUSE_S", 15)
    monkeypatch.setattr(gemsim_job.time, "sleep", slept.append)
    fake.fail_start.add("gemsim.exe")
    assert gemsim_job.main(_levels6(tmp_path, "1")) == 0
    assert slept == [15]
