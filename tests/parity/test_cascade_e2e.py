import json
import subprocess
from pathlib import Path

import pytest

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "bin" / "python")
_FOLDER = find_reference(
    "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=0, fbep="header"
)
REF = None if _FOLDER is None else _FOLDER / "out.gdx"
pytestmark = pytest.mark.skipif(
    REF is None or not REF.exists(), reason="GAMS MCP altertax ref absent"
)


def test_orchestrator_emits_parseable_json_with_provenance():
    proc = subprocess.run(
        [
            PY,
            "scripts/gtap/cascade_orchestrator.py",
            "--dataset",
            "gtap7_3x3",
            "--periods",
            "shock",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    # stdout is exactly one JSON object
    report = json.loads(proc.stdout)
    assert report["dataset"] == "gtap7_3x3"
    assert report["ref"]["source"] == "gams_reference"
    assert report["kkt_reader"] == "pure-python"
    assert "shock" in report["periods"]
    # provenance line is on stderr
    assert "kkt_reader:" in proc.stderr
