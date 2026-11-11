"""The two ALTERTAX command-line entry points end with ``os._exit`` once their outputs are
written and flushed: the interpreter does not tear down the run's arrays (~6 GB on
20x41, 28-51 s). Seen from outside: the exit status and every printed line are there,
the files are complete, and an ``atexit`` hook of the process never runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.family._prep import ROOT, _newton_mumps

pytest.importorskip("pyomo")
SCRIPTS = ROOT / "scripts" / "family"

# runs the script as __main__ after registering an atexit hook that writes a sentinel
_LAUNCH = (
    "import atexit, runpy, sys\n"
    "sent = sys.argv[1]\n"
    "atexit.register(lambda: open(sent, 'w').write('atexit ran'))\n"
    "sys.argv = sys.argv[2:]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)


def _launch(script, args, tmp_path, cache=None):
    sent = tmp_path / "atexit.txt"
    env = {
        **os.environ,
        **_newton_mumps(),
        "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct",
        "EQUILIBRIA_FAMILY_CACHE": str(cache or tmp_path / "cache"),
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
    }
    p = subprocess.run(
        [sys.executable, "-c", _LAUNCH, str(sent), str(SCRIPTS / script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return p, sent


def test_run_altertax_exits_without_teardown(tmp_path):
    """A usage error (no basedata.har) returns 2 through the same exit."""
    p, sent = _launch(
        "run_altertax.py", [str(tmp_path), "--out", str(tmp_path)], tmp_path
    )
    assert p.returncode == 2, p.stderr[-2000:]
    assert "lacks basedata.har" in p.stderr
    assert not sent.exists()


def test_measure_exits_without_teardown_after_writing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    p, sent = _launch(
        "altertax_rungtap_measure.py", ["gtap7_3x3", "--out", str(out)], tmp_path
    )
    assert p.returncode == 0, p.stderr[-3000:]
    res = json.loads((out / "gtap7_3x3_ratio.json").read_text())  # complete JSON
    assert res["pass"] is True
    assert p.stdout.rstrip().endswith("overall PASS")  # the last line was flushed
    assert not sent.exists()
