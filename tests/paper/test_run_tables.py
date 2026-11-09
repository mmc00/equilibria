"""scripts/paper/run_tables.py: un comando por resultado del paper (§9).

El test no corre las tablas (horas, GAMS, PATH): verifica que cada paso apunte a
scripts y fixtures que existen en el repo, para que el paquete de replicacion no
cite un archivo borrado o renombrado."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "paper" / "run_tables.py"


def _mod():
    spec = importlib.util.spec_from_file_location("run_tables", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_hay_un_paso_por_tabla_del_paper() -> None:
    names = {s.name for s in _mod().steps(Path("out"))}
    assert {
        "table1",
        "table2",
        "table3",
        "table3a",
        "homogeneity",
        "model_size",
        "readers",
    } <= names


def test_cada_paso_apunta_a_archivos_que_existen() -> None:
    for s in _mod().steps(Path("out")):
        for f in s.files:
            assert (ROOT / f).exists(), (s.name, f)


def test_dry_run_no_corre_nada(capsys) -> None:
    assert _mod().main(["--dry-run", "--out", "out"]) == 0
    printed = capsys.readouterr().out
    assert (
        "compare_gempack_levels.py" in printed
        and "compare_burfisher_gempack.py" in printed
    )


NOTEBOOK_EXERCISES = ["TBL65B", "TBL94", "ME3C", "ME9A"]


def test_notebook_exercises_run_one_process_each() -> None:
    """Their closures are built in a notebook with @overwrite, which leaks into
    the next exercise solved in the same process: on 2026-11-08 TBL94 came out
    at 64% within 0.1% after TBL65B, against 100% alone. One command per
    exercise keeps each in its own process."""
    by_name = {s.name: s for s in _mod().steps(Path("out"))}
    for name, against in (
        ("table3a_notebooks", "gempack"),
        ("table3_notebooks", "gams"),
    ):
        cmds = by_name[name].cmds
        assert [c[c.index(against) + 1] for c in cmds] == NOTEBOOK_EXERCISES, name
        assert all("notebook_exercises.py" in c[1] for c in cmds), name
