"""``scripts/family/run_altertax.py``: the command, its outputs, its gate and the
replication of the user's base data by ``updated.har`` under ``default.prm``.

The command runs in a subprocess with the worktree's python (nus333 solves in ~0.2 min)."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from equilibria._local_refs import nus333_dir
from equilibria.babel.har.reader import read_har
from equilibria.family.altertax.tolerances import CHECK_TOL
from tests.family._prep import ROOT, _newton_mumps

pytest.importorskip("pyomo")
SCRIPT = ROOT / "scripts" / "family" / "run_altertax.py"
FIXTURES = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg"
OUTPUT_NAMES = [f"{n}.har" for n in ("updated", "summary", "decomp", "volume", "pct")]
COMPARED = {"levels", "updated", "summary", "volume", "decomp", "pct"}


def _run(args, cache, extra_env=None, timeout=480):
    env = {
        **os.environ,
        **_newton_mumps(),
        "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct",
        "EQUILIBRIA_FAMILY_CACHE": str(cache),
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
        **(extra_env or {}),
    }
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """One command run on nus333 with ``--compare`` against the burfisher fixtures."""
    out = tmp_path_factory.mktemp("out")
    p = _run(
        [nus333_dir(), "--out", out, "--compare", FIXTURES],
        tmp_path_factory.mktemp("cache"),
    )
    return out, p


def test_command_writes_five_outputs(solved):
    out, p = solved
    assert p.returncode == 0, p.stderr[-2000:]
    for name in OUTPUT_NAMES:
        assert (out / name).stat().st_size > 0, name
    run = json.loads((out / "run.json").read_text())
    assert run["codes"] == {"check": 1, "shock": 1}
    assert {"nit", "residual", "minutes"} <= set(run)


def test_compare_json_passes_against_the_fixtures(solved):
    out, p = solved
    assert p.returncode == 0, p.stderr[-2000:]
    cmp = json.loads((out / "compare.json").read_text())
    assert cmp["pass"] is True
    by = {o["output"]: o for o in cmp["outputs"]}
    assert set(by) == COMPARED
    for name, o in by.items():
        assert o["pass"] is True, name
        assert o["gate"], name
        assert isinstance(o["max"], float), name
        assert 0 < len(o["worst"]) <= 10, name
        assert "known_miss" in o, name


def test_compare_accepts_gempack_run_folder_names(tmp_path):
    """A GEMPACK run folder (its own file names, ``solution.sl4`` unzipped) is compared
    with no code change: the path to the future 3x3 ... 20x41 runs."""
    gem = tmp_path / "gem"
    gem.mkdir()
    rename = {
        "altax_updated.har": "updated.har",
        "altax_summary.har": "summary.har",
        "altax_decomp.har": "decomp.har",
        "altax_volume.har": "volume.har",
        "sl4levels_altax.har": "sl4levels.har",
    }
    for src, dst in rename.items():
        shutil.copy(FIXTURES / src, gem / dst)
    with (
        gzip.open(FIXTURES / "altax.sl4.gz", "rb") as a,
        (gem / "solution.sl4").open("wb") as b,
    ):
        shutil.copyfileobj(a, b)
    out = tmp_path / "out"
    p = _run([nus333_dir(), "--out", out, "--compare", gem], tmp_path / "cache")
    assert p.returncode == 0, p.stderr[-2000:]
    assert json.loads((out / "compare.json").read_text())["pass"] is True


def test_missing_gempack_file_is_a_clear_error(tmp_path):
    gem = tmp_path / "gem"
    gem.mkdir()
    shutil.copy(FIXTURES / "altax_summary.har", gem / "summary.har")
    p = _run([nus333_dir(), "--out", tmp_path / "out", "--compare", gem], tmp_path)
    assert p.returncode != 0
    assert "updated" in p.stderr and "solution.sl4" in p.stderr
    assert not (tmp_path / "out" / "updated.har").exists()  # failed before solving


def test_no_outputs_when_unsolved(tmp_path):
    """An iteration limit of 1 cannot solve the check period: run.json names the codes,
    no ``.har`` is written, the exit is non-zero and the message names the period."""
    out = tmp_path / "out"
    p = _run(
        [nus333_dir(), "--out", out],
        tmp_path / "cache",
        {"EQUILIBRIA_GTAP_SCIPY_MAXITER": "1"},
    )
    assert p.returncode != 0
    assert "check" in p.stderr, p.stderr[-2000:]
    run = json.loads((out / "run.json").read_text())
    assert any(c != 1 for c in run["codes"].values())
    assert not list(out.glob("*.har"))
    assert not (out / "compare.json").exists()


def test_updated_replicates_with_default_prm(solved, tmp_path, monkeypatch):
    """The user's base data replicates: ``updated.har`` loaded with nus333 ``default.prm``
    and solved with no shock gives codes 1 and the check period equals base within
    CHECK_TOL * max(1, |base|). CHECK_TOL (5e-6), not 1e-9: the 1.01 stand-ins for the
    unit elasticities apply wherever default.prm has them (esubd/esubm/sigmas = 1 as
    well as the production nests), and the 1.01 nests do not reproduce a 1.00 benchmark
    to the last digit."""
    from equilibria.family.altertax.closure import (
        ALTAX_RESIDUAL_REGION,
        altertax_closure,
    )
    from equilibria.family.altertax.compare import load_compare_script
    from equilibria.family.altertax.levels import family_levels
    from equilibria.family.altertax.run import UNIT_TRADE_ELASTICITIES
    from equilibria.family.driver import solve_gtap_family
    from equilibria.templates.gtap import GTAPParameters

    out, p = solved
    assert p.returncode == 0, p.stderr[-2000:]
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        monkeypatch.setenv(k, v)
    d = nus333_dir()
    params = GTAPParameters()
    params.load_from_har(
        basedata_path=out / "updated.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    # the stand-in of the route (see run.load_altertax_params) where default.prm has 1.0
    for name in UNIT_TRADE_ELASTICITIES:
        table = getattr(params.elasticities, name)
        for key, value in table.items():
            if value == 1.0:
                table[key] = 1.01
    params.shares.calibrate(
        params.benchmark, params.elasticities, params.sets, params.taxes
    )
    params._calibrate()
    res = solve_gtap_family(
        params,
        altertax_closure(),
        ALTAX_RESIDUAL_REGION,
        # the driver refuses an exactly-zero shock (it would apply its default +10 %
        # tariff), so the shock is a negligible 1e-6 % one: only the check period is read
        shock={"prdtx_rai": {("USA", "AGR", "AGR"): 1e-6}},
        cache_root=tmp_path / "c",
    )
    assert {k: int(res.results[k]["code"]) for k in ("check", "shock")} == {
        "check": 1,
        "shock": 1,
    }
    lv = family_levels(res.mp, load_compare_script().EQ_VARS)
    bad, n = [], 0
    for name, cells in lv.items():
        for key, chk in cells.items():
            if key[-1] != "check":
                continue
            base = cells.get((*key[:-1], "base"))
            if base is None or (np.isnan(base) and np.isnan(chk)):
                continue
            n += 1
            if abs(chk - base) > CHECK_TOL * max(1.0, abs(base)):
                bad.append((name, key, base, chk))
    assert n > 200  # 255 check cells of EQ_VARS on nus333
    assert not bad, bad[:10]
    assert read_har(str(out / "updated.har"))  # the file the check ran on


def test_coverage_row():
    import importlib.util

    path = ROOT / "scripts" / "gtap" / "coverage_matrix.py"
    spec = importlib.util.spec_from_file_location("coverage_matrix_altertax", path)
    assert spec is not None and spec.loader is not None
    cm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cm  # @dataclass looks its module up
    try:
        spec.loader.exec_module(cm)
    finally:
        sys.modules.pop(spec.name, None)
    assert "gempack" in cm.REFERENCES
    rows = [
        r
        for r in cm.rows_for("gtap7", "gempack", kind="altertax")
        if r.dataset == "nus333"
    ]
    assert len(rows) == 1
    assert rows[0] in cm.ROWS and rows[0].ci_status == "local"
    assert "run_altertax" in rows[0].ref
