"""write_source.py runs the generating command, times it and records it (ADR-0010)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

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


def test_parent_record_ignores_nested_records(tmp_path: Path) -> None:
    """A dataset folder recorded after its result subfolder exists (review #1)."""
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_ifsub0"
    cmd = [sys.executable, "-c", "open('out.txt','w').write('x')"]
    _module().main(_argv(tmp_path, folder, cmd))
    data = tmp_path / "gtap7" / "gtap7_3x3"
    argv = [
        "--data-root",
        str(tmp_path),
        "--folder",
        str(data),
        "--kind",
        "data",
        "--model",
        "gtap7",
        "--dataset",
        "gtap7_3x3",
        "--no-run",
        "--source",
        "test",
        "--license",
        "test",
    ]
    assert _module().main(argv) == 0
    assert set(load_source(data).raw["outputs"]) == {"basedata.har"}
    _module().main(_argv_again(tmp_path, folder))  # regenerate the child
    assert not any("gtap7_3x3:" in m for m in check_tree(tmp_path))


def _argv_again(root: Path, folder: Path) -> list[str]:
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
        str(root / "gtap7/gams/comp.gms"),
        "--data-dir",
        str(root / "gtap7/gtap7_3x3"),
        "--",
        sys.executable,
        "-c",
        "open('out.txt','w').write('y')",
    ]


def test_failed_rerun_removes_the_old_record(tmp_path: Path) -> None:
    """A failing regeneration must not leave a record that says verified (review #2)."""
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_ifsub0"
    ok = [sys.executable, "-c", "open('out.txt','w').write('x')"]
    _module().main(_argv(tmp_path, folder, ok))
    bad = [
        *_argv_again(tmp_path, folder)[:-3],
        sys.executable,
        "-c",
        "raise SystemExit(2)",
    ]
    assert _module().main(bad) == 2
    assert not (folder / "source.yaml").exists()


def test_inner_double_dash_is_kept(tmp_path: Path) -> None:
    """Only the separator before the command is dropped (review #6)."""
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_ifsub0"
    cmd = [
        sys.executable,
        "-c",
        "import sys; open('args.txt','w').write(repr(sys.argv[1:]))",
        "--",
        "x",
    ]
    _module().main(_argv(tmp_path, folder, cmd))
    assert (folder / "args.txt").read_text() == "['--', 'x']"


def test_bad_input_path_fails_before_running(tmp_path: Path) -> None:
    """A path outside the data root is an error before the (long) command runs (review #8)."""
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_ifsub0"
    outside = tmp_path.parent / "elsewhere.gms"
    outside.write_text("x")
    argv = _argv(
        tmp_path, folder, [sys.executable, "-c", "open('ran.txt','w').write('x')"]
    )
    argv[argv.index("--model-file") + 1] = str(outside)
    try:
        _module().main(argv)
    except SystemExit:
        pass
    assert not (folder / "ran.txt").exists()


class _FakeWinreg:
    HKEY_LOCAL_MACHINE = object()

    def __init__(self, name: str | None) -> None:
        self.name = name
        self.asked: list[tuple[str, str]] = []

    def OpenKey(self, root: object, path: str) -> str:  # noqa: N802 (winreg's API)
        if self.name is None:
            raise OSError("no such key")
        return path

    def QueryValueEx(self, key: str, value: str) -> tuple[str, int]:  # noqa: N802
        self.asked.append((key, value))
        return f"  {self.name}  ", 1


def test_windows_cpu_reads_the_registry_name() -> None:
    fake = _FakeWinreg("Intel(R) Core(TM) i7-12700H")
    assert _module()._windows_cpu(fake) == "Intel(R) Core(TM) i7-12700H"
    assert fake.asked == [
        (r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString")
    ]


def test_windows_cpu_falls_back_without_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the registry key: the architecture in the environment, not
    platform.processor(), which queries WMI (timed out in the 3a-2 run)."""
    monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "AMD64")
    assert _module()._windows_cpu(_FakeWinreg(None)) == "AMD64"


def test_model_derived_from_records_the_changes(tmp_path: Path) -> None:
    root = tmp_path / "data"
    base = root / "gtap7" / "base"
    base.mkdir(parents=True)
    (base / "a.gms").write_bytes(b"a\r\n")
    assert (
        _module().main(
            [
                "--data-root",
                str(root),
                "--folder",
                str(base),
                "--kind",
                "model",
                "--model",
                "gtap7",
                "--no-run",
                "--source",
                "upstream",
            ]
        )
        == 0
    )
    derived = root / "gtap7" / "derived"
    derived.mkdir()
    (derived / "a.gms").write_bytes(b"A\n")
    changes = tmp_path / "changes.yaml"
    changes.write_text(
        "- {file: a.gms, change: modified, what: upper case, why: test, commit: abc1234}\n"
    )
    assert (
        _module().main(
            [
                "--data-root",
                str(root),
                "--folder",
                str(derived),
                "--kind",
                "model",
                "--model",
                "gtap7",
                "--no-run",
                "--source",
                "ours",
                "--derived-from",
                "gtap7/base",
                "--changes",
                str(changes),
            ]
        )
        == 0
    )
    rec = load_source(derived)
    assert rec.raw["derived_from"] == "gtap7/base"
    assert rec.raw["changes"][0]["file"] == "a.gms"
    assert check_tree(root) == []


def test_unlisted_change_writes_no_record(tmp_path: Path) -> None:
    # the writer refuses a derived folder whose changes do not cover its edits
    root = tmp_path / "data"
    base = root / "gtap7" / "base"
    base.mkdir(parents=True)
    (base / "a.gms").write_bytes(b"a")
    _module().main(
        [
            "--data-root",
            str(root),
            "--folder",
            str(base),
            "--kind",
            "model",
            "--model",
            "gtap7",
            "--no-run",
            "--source",
            "upstream",
        ]
    )
    derived = root / "gtap7" / "derived"
    derived.mkdir()
    (derived / "a.gms").write_bytes(b"A")
    changes = tmp_path / "changes.yaml"
    changes.write_text("[]\n")
    assert (
        _module().main(
            [
                "--data-root",
                str(root),
                "--folder",
                str(derived),
                "--kind",
                "model",
                "--model",
                "gtap7",
                "--no-run",
                "--source",
                "ours",
                "--derived-from",
                "gtap7/base",
                "--changes",
                str(changes),
            ]
        )
        != 0
    )
    assert not (derived / "source.yaml").exists()


def test_model_folder_input_hashes_every_file(tmp_path: Path) -> None:
    # an include edited after the run makes the result stale, not only comp.gms
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_ifsub0"
    argv = _argv(
        tmp_path, folder, [sys.executable, "-c", "open('x.nl', 'w').write('nl')"]
    )
    model = Path(argv[argv.index("--model-file") + 1])
    (model.parent / "model.gms").write_text("equations")
    folder.mkdir(parents=True)
    argv[argv.index("--model-file") + 1] = str(model.parent)
    assert _module().main(argv) == 0
    hashes = load_source(folder).raw["generated_from"]["model"]["sha256"]
    assert set(hashes) == {"comp.gms", "model.gms"}
    (model.parent / "model.gms").write_text("edited")
    assert any("changed after" in m for m in check_tree(tmp_path))


_NEOS_JOB = {
    "job": 4242,
    "submitted_utc": "2026-10-30T18:00:00+00:00",
    "finished_utc": "2026-10-30T18:05:00+00:00",
    "category": "cp",
    "solver": "PATH",
    "input_method": "GAMS",
    "options": "convergence_tolerance 1e-9\n",
    "gams_version": "52.5.0",
    "sent_sha256": "0" * 64,
}


def _neos_cmd() -> list[str]:
    body = f"import json; json.dump({_NEOS_JOB!r}, open('neos_job.json', 'w'))"
    return [sys.executable, "-c", body]


def test_write_source_remote_neos(tmp_path: Path) -> None:
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_mcp_gtap_ifsub1"
    argv = _argv(tmp_path, folder, _neos_cmd())
    argv[argv.index("--") : argv.index("--")] = ["--remote", "neos"]
    assert _module().main(argv) == 0
    raw = load_source(folder).raw
    assert raw["run"]["machine"] == {"os": "NEOS server", "cpu": "", "cores": None,
                                     "ram_gb": None}  # fmt: skip
    assert raw["run"]["date"] == _NEOS_JOB["finished_utc"]
    assert raw["tool_version"] == "52.5.0"
    assert raw["neos"] == {k: v for k, v in _NEOS_JOB.items() if k != "gams_version"}
    assert "neos_job.json" in raw["outputs"]


def test_write_source_remote_neos_refuses_other_version(tmp_path: Path) -> None:
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_mcp_gtap_ifsub1"
    argv = _argv(tmp_path, folder, _neos_cmd())
    argv[argv.index("--") : argv.index("--")] = [
        "--remote",
        "neos",
        "--tool-version",
        "53.5.0",
    ]
    assert _module().main(argv) != 0
    assert not (folder / "source.yaml").exists()


def test_failed_command_leaves_no_new_folder(tmp_path: Path) -> None:
    # a NEOS job that fails mid-matrix must leave nothing behind (phase 2b)
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_mcp_gtap_ifsub1"
    cmd = [sys.executable, "-c", "raise SystemExit(3)"]
    assert _module().main(_argv(tmp_path, folder, cmd)) == 3
    assert not folder.exists()


def test_solver_options_are_recorded(tmp_path: Path) -> None:
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_mcp_gtap_ifsub1"
    body = "open('path.opt','w').write('* tight\\nconvergence_tolerance 1e-9\\ncrash_method pnewton\\n')"
    argv = _argv(tmp_path, folder, [sys.executable, "-c", body])
    argv[argv.index("--") : argv.index("--")] = [
        "--solver",
        "PATH",
        "--solver-options-file",
        "path.opt",
    ]
    assert _module().main(argv) == 0
    solver = load_source(folder).raw["solver"]
    assert solver == {
        "name": "PATH",
        "options": {"convergence_tolerance": "1e-9", "crash_method": "pnewton"},
    }


def test_no_options_file_records_no_options(tmp_path: Path) -> None:
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_mcp_gtap_ifsub1"
    argv = _argv(
        tmp_path, folder, [sys.executable, "-c", "open('out.txt','w').write('x')"]
    )
    argv[argv.index("--") : argv.index("--")] = [
        "--solver",
        "PATH",
        "--solver-options-file",
        "path.opt",
    ]
    assert _module().main(argv) == 0
    assert load_source(folder).raw["solver"] == {"name": "PATH"}


def _upstream(root: Path) -> Path:
    """A file of another result, the input a chained result cross-checks against."""
    dump = root / "gtap7" / "gtap7_3x3" / "gempack_tm10" / "sl4dump.har"
    dump.parent.mkdir(parents=True)
    dump.write_bytes(b"dump")
    return dump


def test_input_chains_a_result(tmp_path: Path) -> None:
    """--input NAME=PATH hashes another result's file; editing it makes this one stale."""
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "sltoht_tm10"
    cmd = [sys.executable, "-c", "open('out.txt','w').write('x')"]
    argv = _argv(tmp_path, folder, cmd)
    dump = _upstream(tmp_path)
    argv[argv.index("--") : argv.index("--")] = ["--input", f"sl4dump={dump}"]
    assert _module().main(argv) == 0
    chained = load_source(folder).raw["generated_from"]["sl4dump"]
    assert chained["path"] == "gtap7/gtap7_3x3/gempack_tm10/sl4dump.har"
    assert len(chained["sha256"]) == 64
    dump.write_bytes(b"edited")
    assert any(
        "sltoht_tm10: input sl4dump" in m and "changed after" in m
        for m in check_tree(tmp_path)
    )


def test_input_folder_hashes_every_file(tmp_path: Path) -> None:
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "sltoht_tm10"
    cmd = [sys.executable, "-c", "open('out.txt','w').write('x')"]
    argv = _argv(tmp_path, folder, cmd)
    dump = _upstream(tmp_path)
    argv[argv.index("--") : argv.index("--")] = ["--input", f"solution={dump.parent}"]
    assert _module().main(argv) == 0
    chained = load_source(folder).raw["generated_from"]["solution"]
    assert chained["path"] == "gtap7/gtap7_3x3/gempack_tm10"
    assert set(chained["sha256"]) == {"sl4dump.har"}


def test_input_outside_root_refused(tmp_path: Path) -> None:
    root = tmp_path / "data"
    folder = root / "gtap7" / "gtap7_3x3" / "sltoht_tm10"
    outside = tmp_path / "elsewhere.har"
    outside.write_bytes(b"x")
    argv = _argv(root, folder, [sys.executable, "-c", "open('ran.txt','w').write('x')"])
    argv[argv.index("--") : argv.index("--")] = ["--input", f"sl4dump={outside}"]
    with pytest.raises(SystemExit):
        _module().main(argv)
    assert not (folder / "ran.txt").exists()


def test_input_cannot_reuse_model_or_data(tmp_path: Path) -> None:
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "sltoht_tm10"
    argv = _argv(tmp_path, folder, [sys.executable, "-c", "pass"])
    dump = _upstream(tmp_path)
    argv[argv.index("--") : argv.index("--")] = ["--input", f"model={dump}"]
    with pytest.raises(SystemExit):
        _module().main(argv)


def test_external_is_recorded_not_checked(tmp_path: Path) -> None:
    """--external hashes a file kept out of the repository; check_tree never reads it."""
    root = tmp_path / "data"
    folder = root / "gtap7" / "gtap7_3x3" / "gemsim_levels6"
    exe = tmp_path / "runGTAP375" / "gtapv7.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"MZ")
    gss = tmp_path / "work" / "gtapv7_levels.gss"
    gss.parent.mkdir()
    gss.write_bytes(b"gss")
    argv = _argv(root, folder, [sys.executable, "-c", "open('out.txt','w').write('x')"])
    argv[argv.index("--") : argv.index("--")] = [
        "--external",
        f"gtapv7.exe={exe}@12.1.003",
        "--external",
        f"gtapv7_levels.gss={gss}",
    ]
    assert _module().main(argv) == 0
    external = load_source(folder).raw["external"]
    assert external["gtapv7.exe"]["path"] == str(exe)
    assert external["gtapv7.exe"]["release"] == "12.1.003"
    assert len(external["gtapv7.exe"]["sha256"]) == 64
    assert "release" not in external["gtapv7_levels.gss"]
    exe.unlink()
    gss.unlink()
    assert not any("gemsim_levels6" in m for m in check_tree(root))


def test_missing_external_fails_before_running(tmp_path: Path) -> None:
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gemsim_levels6"
    argv = _argv(
        tmp_path, folder, [sys.executable, "-c", "open('ran.txt','w').write('x')"]
    )
    argv[argv.index("--") : argv.index("--")] = [
        "--external",
        f"gtapv7.exe={tmp_path / 'nope.exe'}",
    ]
    with pytest.raises(SystemExit):
        _module().main(argv)
    assert not (folder / "ran.txt").exists()


def test_notes_from_file_is_appended_not_an_output(tmp_path: Path) -> None:
    """--notes-from FILE: what the command leaves there goes to notes, not to outputs."""
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gemsim_x"
    cmd = [sys.executable, "-c",
           "open('out.txt','w').write('x'); open('note.txt','w').write('retried once')"]  # fmt: skip
    argv = _argv(tmp_path, folder, cmd)
    argv[argv.index("--") : argv.index("--")] = ["--notes", "base", "--notes-from", "note.txt"]
    assert _module().main(argv) == 0
    raw = load_source(folder).raw
    assert raw["notes"] == "base; retried once"
    assert set(raw["outputs"]) == {"out.txt"}
    assert not (folder / "note.txt").exists()


def test_notes_from_absent_file_keeps_notes(tmp_path: Path) -> None:
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gemsim_x"
    argv = _argv(tmp_path, folder, [sys.executable, "-c", "open('out.txt','w').write('x')"])
    argv[argv.index("--") : argv.index("--")] = ["--notes-from", "note.txt"]
    assert _module().main(argv) == 0
    assert load_source(folder).raw["notes"] == ""


def _not_starting(m, monkeypatch: pytest.MonkeyPatch, times: int) -> list[int]:
    """subprocess.run fails to start the command ``times`` times (0xC0000142)."""
    import subprocess

    real, calls = m.subprocess.run, []

    def run(cmd, **kw):
        if not (isinstance(cmd, list) and cmd[:2] == [sys.executable, "-c"]):
            return real(cmd, **kw)  # not the recorded command (platform, git...)
        calls.append(1)
        if len(calls) <= times:
            return subprocess.CompletedProcess(cmd, 0xC0000142)
        return real(cmd, **kw)

    monkeypatch.setattr(m.subprocess, "run", run)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)  # the pause before a retry
    return calls


def test_command_that_does_not_start_is_retried_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3a-1 and 3a-2: now and then Windows does not start the command (the python
    of a job: STATUS_DLL_INIT_FAILED). One retry, and the record says so."""
    m = _module()
    calls = _not_starting(m, monkeypatch, 1)
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_ifsub0"
    cmd = [sys.executable, "-c", "open('out.txt','w').write('x')"]
    assert m.main(_argv(tmp_path, folder, cmd)) == 0
    assert len(calls) == 2
    assert "did not start" in load_source(folder).raw["notes"]


def test_a_second_failed_start_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = _module()
    calls = _not_starting(m, monkeypatch, 2)
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_ifsub0"
    assert m.main(_argv(tmp_path, folder, [sys.executable, "-c", "pass"])) != 0
    assert len(calls) == 2 and not (folder / "source.yaml").exists()


def test_windows_machine_info_does_not_touch_wmi(monkeypatch: pytest.MonkeyPatch) -> None:
    """platform's processor/release/version query WMI on Windows (Python 3.12), and
    WMI timed out mid-run (WinError 258, KeyError 'Architecture'): a solved job lost
    its record. machine_info builds the same strings without it."""
    from types import SimpleNamespace

    m = _module()

    def wmi(*a: object) -> str:
        raise OSError(258, "WMI timed out")

    monkeypatch.setattr(m.platform, "system", lambda: "Windows")
    for name in ("processor", "machine", "release", "version", "uname"):
        monkeypatch.setattr(m.platform, name, wmi)
    monkeypatch.setattr(m.sys, "getwindowsversion",
                        lambda: SimpleNamespace(major=10, minor=0, build=26200), raising=False)
    monkeypatch.setattr(m, "_windows_cpu", lambda: "13th Gen Intel(R) Core(TM) i7-1365U")
    monkeypatch.setattr(m, "_windows_ram", lambda: 31.4)
    info = m.machine_info()
    assert info["os"] == "Windows 11 (10.0.26200)"
    assert info["cpu"] == "13th Gen Intel(R) Core(TM) i7-1365U"
    assert info["ram_gb"] == 31.4


def test_retry_waits_before_the_second_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The immediate retry failed like the first attempt (full 3a-2 run): wait first."""
    m = _module()
    _not_starting(m, monkeypatch, 1)
    slept: list[float] = []
    monkeypatch.setattr(m.time, "sleep", slept.append)
    folder = tmp_path / "gtap7" / "gtap7_3x3" / "gams_ifsub0"
    cmd = [sys.executable, "-c", "open('out.txt','w').write('x')"]
    assert m.main(_argv(tmp_path, folder, cmd)) == 0
    assert slept == [m.RETRY_PAUSE_S] and m.RETRY_PAUSE_S >= 10
