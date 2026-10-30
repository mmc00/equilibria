"""A GAMS MCP reference solved on NEOS, written like a local run (phase 2b).

The NEOS XML-RPC proxy is faked: it serves the 2a 3x3 gtap MCP result back.
"""

from __future__ import annotations

import importlib
import io
import json
import xmlrpc.client
import zipfile
from pathlib import Path
from typing import Any

import pytest

from equilibria.sources import GAMS_TOOLS_DIR, dataset_dir

RESULT = dataset_dir("gtap7_3x3") / "gams_mcp_gtap_ifsub1"
PASSWORD = "s3cr3t-pw"
EMAIL = "someone@example.org"


def _lst(version: str = "53.5.0", status: str = "1") -> str:
    return (
        f"GAMS {version}  a8b3e4ba Sep 10, 2026          DAR arm 64bit/macOS\n"
        + "".join(f"**** MODEL STATUS      {status} Optimal\n" for _ in range(3))
    )


class FakeNeos:
    def __init__(
        self,
        statuses: tuple[str, ...] = ("Waiting", "Running", "Done"),
        lst: str | None = None,
        gdx: bool = True,
    ) -> None:
        self.statuses = list(statuses)
        self.lst = _lst() if lst is None else lst
        self.gdx = gdx
        self.sent = ""

    def ping(self) -> str:
        return "NeosServer is alive\n"

    def submitJob(self, xml: str) -> tuple[int, str]:  # noqa: N802 — NEOS API
        self.sent = xml
        return 4242, PASSWORD

    def getJobStatus(self, job: int, pw: str) -> str:  # noqa: N802
        assert (job, pw) == (4242, PASSWORD)
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def getFinalResults(self, job: int, pw: str) -> xmlrpc.client.Binary:  # noqa: N802
        return xmlrpc.client.Binary(f"NEOS log\n{self.lst}".encode())

    def getOutputFile(self, job: int, pw: str, name: str) -> xmlrpc.client.Binary:  # noqa: N802
        assert name == "solver-output.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            if self.gdx:
                z.write(RESULT / "out.gdx", "out.gdx")
            z.writestr("solve.lst", self.lst)
        return xmlrpc.client.Binary(buf.getvalue())


def _tools(monkeypatch: Any, name: str) -> Any:
    monkeypatch.syspath_prepend(str(GAMS_TOOLS_DIR))
    return importlib.import_module(name)


def _run(monkeypatch: Any, tmp_path: Path, fake: FakeNeos) -> Any:
    nj = _tools(monkeypatch, "neos_job")
    return nj, nj.run(
        RESULT / "model.gms", tmp_path / "out", email=EMAIL, proxy=fake, poll_s=0
    )


def test_neos_job_writes_outputs(monkeypatch: Any, tmp_path: Path) -> None:
    fake = FakeNeos()
    _, run = _run(monkeypatch, tmp_path, fake)
    out = tmp_path / "out"
    assert (out / "out.gdx").read_bytes() == (RESULT / "out.gdx").read_bytes()
    assert "MODEL STATUS" in (out / "model.lst").read_text()
    assert (out / "neos.log").exists()
    job = json.loads((out / "neos_job.json").read_text())
    assert set(job) == {
        "job", "submitted_utc", "finished_utc", "category", "solver",
        "input_method", "options", "gams_version", "sent_sha256",
    }  # fmt: skip
    assert (job["job"], job["category"], job["solver"], job["input_method"]) == (
        4242, "cp", "PATH", "GAMS",
    )  # fmt: skip
    assert run.job == 4242
    assert (
        "<category>cp</category>" in fake.sent and "<solver>PATH</solver>" in fake.sent
    )


@pytest.mark.parametrize("status", ["Failed", "Killed", "Bad job"])
def test_neos_failure_raises(monkeypatch: Any, tmp_path: Path, status: str) -> None:
    nj = _tools(monkeypatch, "neos_job")
    with pytest.raises(nj.NeosError, match=status):
        nj.run(
            RESULT / "model.gms", tmp_path / "out", email=EMAIL,
            proxy=FakeNeos(statuses=("Running", status)), poll_s=0,
        )  # fmt: skip


def test_neos_timeout_raises(monkeypatch: Any, tmp_path: Path) -> None:
    nj = _tools(monkeypatch, "neos_job")
    with pytest.raises(nj.NeosError, match="timed out"):
        nj.run(
            RESULT / "model.gms", tmp_path / "out", email=EMAIL,
            proxy=FakeNeos(statuses=("Running",)), poll_s=0, timeout_s=0,
        )  # fmt: skip


def test_neos_missing_gdx_raises(monkeypatch: Any, tmp_path: Path) -> None:
    nj = _tools(monkeypatch, "neos_job")
    with pytest.raises(nj.NeosError, match="gdx"):
        nj.run(RESULT / "model.gms", tmp_path / "out", email=EMAIL,
               proxy=FakeNeos(gdx=False), poll_s=0)  # fmt: skip


def test_neos_done_but_unsolved_is_refused(monkeypatch: Any, tmp_path: Path) -> None:
    nj = _tools(monkeypatch, "neos_job")
    with pytest.raises(nj.NeosError, match="5"):
        nj.run(RESULT / "model.gms", tmp_path / "out", email=EMAIL,
               proxy=FakeNeos(lst=_lst(status="5")), poll_s=0)  # fmt: skip


def test_neos_tool_version_from_lst(monkeypatch: Any, tmp_path: Path) -> None:
    _run(monkeypatch, tmp_path, FakeNeos(lst=_lst(version="52.5.0")))
    job = json.loads((tmp_path / "out" / "neos_job.json").read_text())
    assert job["gams_version"] == "52.5.0"


def _regen(monkeypatch: Any, tmp_path: Path, fake: FakeNeos) -> tuple[int, Path]:
    rgr = _tools(monkeypatch, "regen_gams_reference")
    nj = _tools(monkeypatch, "neos_job")

    def build(mode: str, dataset: str, ifsub: int, work: Path, fbep: str) -> Path:
        gms = work / "comp_gtap7_3x3_gtap_shock_ifsub1.gms"
        gms.write_text((RESULT / "model.gms").read_text())
        return gms

    monkeypatch.setattr(rgr, "_build", build)
    monkeypatch.setattr(nj, "default_proxy", lambda: fake)
    monkeypatch.setattr(nj, "POLL_S", 0)
    monkeypatch.setenv("NEOS_EMAIL", EMAIL)
    out = tmp_path / "result"
    argv = ["--dataset", "gtap7_3x3", "--mode", "gtap", "--form", "mcp", "--ifsub", "1",
            "--fbep", "header", "--where", "neos", "--out", str(out)]  # fmt: skip
    return rgr.main(argv), out


def test_neos_failure_leaves_no_folder(monkeypatch: Any, tmp_path: Path) -> None:
    rc, out = _regen(monkeypatch, tmp_path, FakeNeos(statuses=("Running", "Failed")))
    assert rc != 0
    assert not out.exists()


def test_neos_record_has_no_password(monkeypatch: Any, tmp_path: Path) -> None:
    rc, out = _regen(monkeypatch, tmp_path, FakeNeos())
    assert rc == 0
    assert {p.name for p in out.iterdir()} >= {"model.gms", "out.gdx", "neos_job.json"}
    for f in out.rglob("*"):
        blob = f.read_bytes()
        assert PASSWORD.encode() not in blob, f
        assert EMAIL.encode() not in blob, f
