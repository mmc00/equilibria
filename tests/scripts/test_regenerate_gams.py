"""regenerate_gams.py runs the local GAMS jobs of phase 2a under write_source (ADR-0010)."""

from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.data._records import data_record, write_record

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gtap" / "regenerate_gams.py"


@pytest.fixture
def driver(monkeypatch: Any) -> Any:
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("regenerate_gams", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "regenerate_gams", module)  # @dataclass needs it
    spec.loader.exec_module(module)
    return module


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    write_record(root / "gtap7" / "gtap7_3x3", data_record(dataset="gtap7_3x3"))
    return root


def _gams(tmp_path: Path, version: str) -> Path:
    stub = tmp_path / "bin" / "gams"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(
        f"#!/bin/sh\necho '*** GAMS Base Module {version} 5dc151ca May 5, 2026'\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return stub


def test_jobs_are_the_table(driver: Any) -> None:
    js = driver.jobs()
    assert len(js) == 26
    assert sum(j.form == "nl" for j in js) == 10
    assert sum(j.form == "nlp" for j in js) == 14
    assert {(j.dataset, j.mode, j.ifsub) for j in js if j.form == "mcp"} == {
        ("gtap7_3x3", "gtap", 1),
        ("gtap7_3x3", "altertax", 1),
    }
    # no mcp for 3x4: above the demo cap
    assert {j.folder.name for j in js if j.dataset == "gtap7_3x4"} == {
        f"gams_{f}_altertax_ifsub{k}" for f in ("nl", "nlp") for k in (0, 1)
    }


def test_nl_jobs_keep_todays_phases(driver: Any) -> None:
    phases = {
        j.dataset: j.phases for j in driver.jobs() if j.form == "nl" and j.ifsub == 0
    }
    assert phases["gtap7_3x3"] == ("base", "check", "shock", "altertax")
    assert phases["gtap7_3x4"] == ("base", "shock", "altertax")


def test_job_command_records_what_ran(driver: Any) -> None:
    job = next(
        j
        for j in driver.jobs()
        if j.form == "nl" and j.dataset == "gtap7_3x3" and j.ifsub == 0
    )
    cmd = driver.job_command(job, tool_version="53.5.0")
    pairs = list(zip(cmd, cmd[1:], strict=False))
    for flag, value in [
        ("--variant", "form=nl"),
        ("--variant", "mode=altertax"),
        ("--variant", "ifsub=0"),
        ("--solver", "CONVERT"),
        ("--tool-version", "53.5.0"),
        ("--model-file", str(driver.DATA_ROOT / "gtap7" / "gams")),
        ("--ifsub", "0"),
        ("--out-dir", str(job.folder)),
    ]:
        assert (flag, value) in pairs, (flag, value)
    assert "--gams-local" in cmd and "--skip-python" in cmd


def test_resume_skips_recorded(driver: Any, tmp_path: Path, monkeypatch: Any) -> None:
    root = _root(tmp_path)
    ran: list[list[str]] = []
    monkeypatch.setattr(driver, "run_job", lambda cmd: ran.append(cmd) or 0)
    done = root / "gtap7" / "gtap7_3x3" / "gams_nl_altertax_ifsub0"
    write_record(
        done,
        {
            "kind": "model",
            "model": "gtap7",
            "source": "s",
            "outputs": {},
            "status": "verified",
        },
    )
    argv = ["--tool-version", "53.5.0", "--gams", str(_gams(tmp_path, "53.5.0")),
            "--data-root", str(root), "--datasets", "gtap7_3x3", "--only", "gams_nl_altertax_ifsub0",
            "--only", "gams_nl_altertax_ifsub1"]  # fmt: skip
    assert driver.main(argv) == 0
    assert len(ran) == 1 and "ifsub=1" in ran[0]


def test_failed_job_leaves_no_folder(
    driver: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    root = _root(tmp_path)
    ran: list[list[str]] = []

    def fail_first(cmd: list[str]) -> int:
        ran.append(cmd)
        folder = Path(cmd[cmd.index("--folder") + 1])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "half.nl").write_text("x")
        return 1 if len(ran) == 1 else 0

    monkeypatch.setattr(driver, "run_job", fail_first)
    argv = ["--tool-version", "53.5.0", "--gams", str(_gams(tmp_path, "53.5.0")),
            "--data-root", str(root), "--datasets", "gtap7_3x3", "--only", "gams_nl_altertax_ifsub0",
            "--only", "gams_nl_altertax_ifsub1"]  # fmt: skip
    assert driver.main(argv) != 0
    assert len(ran) == 2  # the next job still ran
    assert not (root / "gtap7" / "gtap7_3x3" / "gams_nl_altertax_ifsub0").exists()


def test_wrong_tool_version_stops(
    driver: Any, tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    root = _root(tmp_path)
    ran: list[list[str]] = []
    monkeypatch.setattr(driver, "run_job", lambda cmd: ran.append(cmd) or 0)
    argv = ["--tool-version", "53.5.0", "--gams", str(_gams(tmp_path, "53.6.0")),
            "--data-root", str(root), "--datasets", "gtap7_3x3"]  # fmt: skip
    assert driver.main(argv) != 0
    assert "53.6.0" in capsys.readouterr().err
    assert ran == []


def test_unknown_only_is_refused(driver: Any, tmp_path: Path, capsys: Any) -> None:
    argv = ["--tool-version", "53.5.0", "--gams", str(_gams(tmp_path, "53.5.0")),
            "--data-root", str(_root(tmp_path)), "--datasets", "gtap7_3x3",
            "--only", "gams_mcp_gtap_ifsub0"]  # fmt: skip
    assert driver.main(argv) == 2
    assert "gams_mcp_gtap_ifsub0" in capsys.readouterr().err
