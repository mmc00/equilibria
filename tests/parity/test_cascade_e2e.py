import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "bin" / "python")
REF = Path("/Users/marmol/proyectos2/equilibria_refs/"
           "gtap7_3x3_altertax_cd/out_altertax_ifsub0.gdx")
pytestmark = pytest.mark.skipif(not REF.exists(), reason="durable ref GDX absent")


def test_orchestrator_emits_parseable_json_with_provenance():
    proc = subprocess.run(
        [PY, "scripts/gtap/cascade_orchestrator.py",
         "--dataset", "gtap7_3x3", "--periods", "shock"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
    # stdout is exactly one JSON object
    report = json.loads(proc.stdout)
    assert report["dataset"] == "gtap7_3x3"
    assert report["ref"]["source"] == "durable"
    assert report["kkt_reader"] == "pure-python"
    assert "shock" in report["periods"]
    # provenance line is on stderr
    assert "kkt_reader:" in proc.stderr
