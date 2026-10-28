"""write_source.py runs the generating command, times it and records it (ADR-0010)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from equilibria.sources import check_tree, load_source

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "write_source.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("write_source", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(root: Path) -> tuple[Path, Path]:
    model = root / "gtap7" / "gams" / "comp.gms"
    model.parent.mkdir(parents=True)
    model.write_text("model")
    data = root / "gtap7" / "gtap7_3x3"
    data.mkdir(parents=True)
    (data / "basedata.har").write_bytes(b"data")
    return model, data


def _argv(root: Path, folder: Path, command: list[str]) -> list[str]:
    model, data = _inputs(root)
    return [
        "--data-root",
        str(root),
        "--folder",
        str(folder),
        "--kind",
        "result",
        "--model",
        "gtap7",
        "--dataset",
        "gtap7_3x3",
        "--tool",
        "gams",
        "--variant",
        "ifsub=0",
        "--model-file",
        str(model),
        "--data-dir",
        str(data),
        "--",
        *command,
    ]


def test_machine_info_has_exactly_the_four_keys() -> None:
    info = _module().machine_info()
    assert set(info) == {"os", "cpu", "cores", "ram_gb"}
    assert info["cores"] >= 1 and info["ram_gb"] > 0


def test_runs_command_and_records_it(tmp_path: Path) -> None:
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_ifsub0"
    cmd = [sys.executable, "-c", "open('out.txt','w').write('x')"]
    assert _module().main(_argv(tmp_path, folder, cmd)) == 0
    raw = load_source(folder).raw
    assert set(raw["outputs"]) == {"out.txt"}
    assert raw["run"]["duration_s"] > 0
    assert raw["status"] == "verified"
    assert raw["variant"] == {"ifsub": 0}
    assert raw["generated_from"]["model"]["path"] == "gtap7/gams/comp.gms"


def test_failed_command_writes_nothing(tmp_path: Path) -> None:
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_ifsub0"
    cmd = [sys.executable, "-c", "raise SystemExit(3)"]
    assert _module().main(_argv(tmp_path, folder, cmd)) == 3
    assert not (folder / "source.yaml").exists()


def test_written_record_passes_check_tree(tmp_path: Path) -> None:
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_ifsub0"
    cmd = [sys.executable, "-c", "open('out.txt','w').write('x')"]
    _module().main(_argv(tmp_path, folder, cmd))
    msgs = check_tree(tmp_path)
    # Only the inputs built by the test lack their own records.
    assert all("no source.yaml" in m for m in msgs)
    assert not any("gams_ifsub0" in m for m in msgs)
