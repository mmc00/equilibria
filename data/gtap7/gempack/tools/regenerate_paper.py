"""Regenerate the corridas_paper.ps1 runs and runs/ into data/ (ADR-0010, phase 3a-2).

Runs on the Windows machine with RunGTAP (gtapv7.exe, gtapuv7.exe) and GEMPACK
(tablo, gemsim, sltoht). Every job is one folder under data/, solved by a job
module of this folder (burfisher_job, levels_job, gemsim_job, swap_job) under
``write_source.py``, which times the run and writes the folder's ``source.yaml``.

The jobs come in groups (plan-phase-3a2.md); a group that reads another
group's result refuses a job whose prerequisite is not recorded yet:

    book          Burfisher's book solutions (gtapuv7.exe)            39
    book-sltoht   steps 1 and 6a: sltoht on them                      39
    levels        step 2: levels of matrix solutions (re-solved)       6
    horridge      step 3: Horridge's SIMPLE, GEMPACK side              3
    levels6       step 4: six more levels, patched condensed TAB      12
    gragg         step 5: Burfisher with Gragg, patched TABs          45
    gragg-sltoht  steps 6b and 7: sltoht on them                      45
    swap          runs/{nus333,9x10}_compare, swap closure            11
    swap-sltoht   levels of the two tm10 swap runs                     2
    validation    runs/gempack_20x41_validation, the recorded abort    1

As in 3a-1, a folder that already has a ``source.yaml`` is skipped, and a
failed job leaves no folder behind.

Usage (PowerShell, from the repo root):
    uv run python data/gtap7/gempack/tools/regenerate_paper.py `
        --gempack-release 12.1.003 --gp-release 11.3.004 --dry-run
    uv run python data/gtap7/gempack/tools/regenerate_paper.py `
        --gempack-release 12.1.003 --gp-release 11.3.004 --group book --only me9c
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PureWindowsPath

from equilibria.sources import DATA_ROOT, SOURCE_FILE, SourceError, load_source

ROOT = Path(__file__).resolve().parents[4]
TOOLS = Path(__file__).resolve().parent
WRITE_SOURCE = ROOT / "scripts" / "dev" / "write_source.py"
CMF_TEMPLATES = TOOLS / "cmf"

# Burfisher's experiments as committed today (tests/fixtures/gtap7_gempack/).
BOOK = (
    "gtapuv7", "gtapv7", "me3a", "me3b", "me4a", "me4b", "me5", "me7a", "me7b",
    "me8", "me9a", "me9b", "me9c", "me9d", "tbl45a", "tbl45b", "tbl45c", "tbl46a",
    "tbl46b", "tbl46c", "tbl54a", "tbl54b", "tbl55", "tbl62a", "tbl62b", "tbl62c",
    "tbl63a", "tbl63b", "tbl64", "tbl65a", "tbl66", "tbl67", "tbl75a", "tbl75b",
    "tbl77", "tbl78", "tbl79", "tbl813", "tbl93",
)  # fmt: skip
BOOK_LEVELS = ("tbl45a", "me9c", "gtapv7")  # step 1 also exports their levels
# step 5 as committed today (43), plus gtapv7 and tbl53 (maintainer, 2026-11-03):
# gtapv7.exp declares the closure of the CONDENSED model, so it runs on step 4's
# condensed gtapv7_levels; tbl53's target% shocks are translated to plain shocks
GRAGG = tuple(sorted(set(BOOK) | {"altax", "closures", "me3c", "tbl65b", "tbl94", "tbl53"}))
CONDENSED_GRAGG = ("gtapv7",)  # run on gtapv7_levels, not gtapuv7_levels
MATRIX = ("gtap7_3x3", "gtap7_3x4", "gtap7_5x5", "gtap7_10x7", "gtap7_15x10", "gtap7_20x41")
LEVELS = (*((ds, "capFlex") for ds in MATRIX[:5]), ("gtap7_20x41", "capFix"))
HORRIDGE = ("fixcap", "mobcap", "tariff10")
SWAP = (*(("nus333", k) for k in range(1, 11)), ("9x10", 10))
SWAP_LEVELS = (("nus333", 10), ("9x10", 10))
GROUPS = (
    "book", "book-sltoht", "levels", "horridge", "levels6", "gragg",
    "gragg-sltoht", "swap", "swap-sltoht", "validation",
)  # fmt: skip
STEPS = "8-16-32"
RDLT = {"capFlex": 1, "capFix": 0}
RELEASE_KIND = {"gempack": "gempack", "gemsim": "gp", "sltoht": "gp"}


@dataclass(frozen=True)
class Job:
    group: str
    name: str  # what --only matches: the folder name
    model: str
    dataset: str
    folder: Path
    tool: str
    script: str  # job module in tools/
    args: tuple[str, ...]
    data_dir: Path
    variant: Mapping[str, object] = field(default_factory=dict)
    inputs: Mapping[str, Path] = field(default_factory=dict)
    externals: Mapping[str, Path] = field(default_factory=dict)
    # release kind ("gempack" | "gp") of the externals that are executables
    external_release: Mapping[str, str] = field(default_factory=dict)
    solver: str | None = None
    needs: tuple[Path, ...] = ()  # folders whose record must exist first
    compiled: str | None = None  # the levels TAB tablo must compile first


@dataclass(frozen=True)
class Paths:
    root: Path
    work: Path
    gtapdir: PurePath
    gp: PurePath

    def exe(self, name: str) -> PurePath:
        return _join(self.gtapdir if name.startswith("gtap") else self.gp, name)


_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]|\\")


def _join(folder: PurePath, name: str) -> PurePath:
    """``folder / name``, kept a Windows path when ``folder`` is one.

    RunGTAP and GEMPACK live on Windows, but the job list is also built on
    Linux/macOS (tests, CI, --dry-run), where Path(r"C:\\runGTAP375") is a
    relative POSIX path and ``/`` would give C:\\runGTAP375/gtapuv7.exe.
    """
    text = str(folder)
    if _WINDOWS_PATH.search(text):
        return PureWindowsPath(text) / name
    return Path(text) / name


def _default_work() -> Path:
    """The compiled TABs go here (git-ignored), not to %TEMP%: the path is recorded in
    every external and command, and %TEMP% would name the Windows user."""
    env = os.environ.get("EQUILIBRIA_GEMPACK_WORK")
    return Path(env) / "paper" if env else ROOT / "runs" / "gempack_work"


def _tag(shock: float, closure: str) -> str:
    return f"tm{shock:g}_s{STEPS}_{closure.lower()}".replace(".", "p")


def jobs(
    groups: list[str],
    *,
    root: Path = DATA_ROOT,
    work: Path | None = None,
    gtapdir: PurePath = PureWindowsPath(r"C:\runGTAP375"),
    gp: PurePath = PureWindowsPath(r"C:\GP"),
) -> list[Job]:
    """Every job of ``groups``, in the order of GROUPS."""
    p = Paths(root, (work or _default_work()).resolve(), gtapdir, gp)
    build = {
        "book": _book, "book-sltoht": _book_sltoht, "levels": _levels,
        "horridge": _horridge, "levels6": _levels6, "gragg": _gragg,
        "gragg-sltoht": _gragg_sltoht, "swap": _swap, "swap-sltoht": _swap_sltoht,
        "validation": _validation,
    }  # fmt: skip
    return [j for g in GROUPS if g in groups for j in build[g](p)]


def _nus333(p: Paths) -> Path:
    return p.root / "gtap7" / "nus333"


def _levels_model(p: Paths) -> Path:
    return p.root / "gtap7" / "gempack_rungtap375_levels"


def _matrix_folder(p: Paths, ds: str, closure: str) -> Path:
    return p.root / "gtap7" / ds / f"gempack_{_tag(10, closure)}"


def _book(p: Paths) -> list[Job]:
    out = []
    for exp in BOOK:
        exe = "gtapv7.exe" if exp == "gtapv7" else "gtapuv7.exe"
        out.append(Job(
            group="book", name=exp, model="gtap7", dataset="nus333",
            folder=_nus333(p) / "gempack_burfisher" / exp, tool="gempack",
            script="burfisher_job.py",
            args=("--experiment", exp, "--package", str(_nus333(p)),
                  "--exe", str(p.exe(exe))),
            data_dir=_nus333(p), variant={"experiment": exp, "method": "book"},
            externals={exe: p.exe(exe)}, external_release={exe: "gempack"},
            solver=exe.removesuffix(".exe"),
        ))  # fmt: skip
    return out


def _sltoht_dump(p: Paths, group: str, exp: str, method: str, solution: Path,
                 folder: Path, levels: bool) -> Job:  # fmt: skip
    return Job(
        group=group, name=exp, model="gtap7", dataset="nus333", folder=folder,
        tool="sltoht", script="levels_job.py",
        args=("dump", "--solution", str(solution), "--sltoht", str(p.exe("sltoht.exe")),
              *(("--levels",) if levels else ())),
        data_dir=_nus333(p), variant={"experiment": exp, "method": method},
        inputs={"solution": solution},
        externals={"sltoht.exe": p.exe("sltoht.exe")},
        external_release={"sltoht.exe": "gp"}, needs=(solution,),
    )  # fmt: skip


def _book_sltoht(p: Paths) -> list[Job]:
    return [
        _sltoht_dump(p, "book-sltoht", exp, "book",
                     _nus333(p) / "gempack_burfisher" / exp,
                     _nus333(p) / "sltoht_burfisher" / exp, exp in BOOK_LEVELS)
        for exp in BOOK
    ]  # fmt: skip


def _gragg_sltoht(p: Paths) -> list[Job]:
    return [
        _sltoht_dump(p, "gragg-sltoht", exp, "gragg",
                     _nus333(p) / "gemsim_burfisher_gragg" / exp,
                     _nus333(p) / "sltoht_burfisher_gragg" / exp, True)
        for exp in GRAGG
    ]  # fmt: skip


def _resolve(p: Paths, group: str, ds: str, closure: str, sl4dump: Path,
             template: Path | None, needs: tuple[Path, ...]) -> Job:  # fmt: skip
    folder = p.root / "gtap7" / ds / f"sltoht_{_tag(10, closure)}"
    exes = {"gtapv7.exe": "gempack", "sltoht.exe": "gp"}
    how = (
        ("--template", str(template)) if template
        else ("--rordelta", str(RDLT[closure]))
    )  # fmt: skip
    return Job(
        group=group, name=folder.name, model="gtap7", dataset=ds, folder=folder,
        tool="sltoht", script="levels_job.py",
        args=("resolve", "--dataset", ds, "--shock", "10", "--steps", STEPS, *how,
              "--sl4dump", str(sl4dump), "--gtapv7", str(p.exe("gtapv7.exe")),
              "--sltoht", str(p.exe("sltoht.exe"))),
        data_dir=p.root / "gtap7" / ds,
        variant={"shock": 10, "steps": STEPS, "closure": closure},
        inputs={"sl4dump": sl4dump}, externals={e: p.exe(e) for e in exes},
        external_release=exes, solver="gtapv7", needs=needs,
    )  # fmt: skip


def _levels(p: Paths) -> list[Job]:
    out = []
    for ds, closure in LEVELS:
        matrix = _matrix_folder(p, ds, closure)
        out.append(_resolve(p, "levels", ds, closure, matrix / "sl4dump.har", None, (matrix,)))
    return out


def _horridge(p: Paths) -> list[Job]:
    model = p.root / "horridge_simple"
    exes = {"tablo.exe": "gp", "gemsim.exe": "gp", "sltoht.exe": "gp"}
    return [
        Job(
            group="horridge", name=f"gemsim_{exp}", model="horridge_simple",
            dataset="horridge_simple", folder=model / f"gemsim_{exp}", tool="gemsim",
            script="gemsim_job.py",
            args=("horridge", "--experiment", exp, "--model-dir", str(model),
                  *(a for e in exes for a in (f"--{e.removesuffix('.exe')}", str(p.exe(e))))),
            data_dir=model, variant={"experiment": exp},
            externals={e: p.exe(e) for e in exes}, external_release=exes,
            solver="gemsim", needs=(model,),
        )  # fmt: skip
        for exp in HORRIDGE
    ]


def _levels6(p: Paths) -> list[Job]:
    aux = _aux(p, "gtapv7_levels")
    exes = {"gemsim.exe": "gp", "sltoht.exe": "gp"}
    compiled = _compiled(aux)
    out = []
    for ds in MATRIX:
        for closure in ("capFlex", "capFix"):
            matrix = _matrix_folder(p, ds, closure)
            folder = p.root / "gtap7" / ds / f"gemsim_levels6_{_tag(10, closure)}"
            out.append(Job(
                group="levels6", name=folder.name, model="gtap7", dataset=ds,
                folder=folder, tool="gemsim", script="gemsim_job.py",
                args=("levels6", "--dataset", ds, "--rordelta", str(RDLT[closure]),
                      "--aux", str(aux), "--sl4dump", str(matrix / "sl4dump.har"),
                      "--gemsim", str(p.exe("gemsim.exe")),
                      "--sltoht", str(p.exe("sltoht.exe"))),
                data_dir=p.root / "gtap7" / ds,
                variant={"shock": 10, "steps": STEPS, "closure": closure,
                         "tab": "gtapv7_levels"},
                inputs={"sl4dump": matrix / "sl4dump.har", "levels_model": _levels_model(p)},
                externals={**{e: p.exe(e) for e in exes}, **compiled},
                external_release=exes, solver="gemsim",
                needs=(matrix, _levels_model(p)), compiled="gtapv7_levels",
            ))  # fmt: skip
    return out


def _aux(p: Paths, tab: str) -> Path:
    """Where tablo leaves the GEMSIM files of ``tab``, under --work."""
    return p.work / tab / tab


def _compiled(aux: Path) -> dict[str, Path]:
    return {f"{aux.name}{s}": aux.with_suffix(s) for s in (".gss", ".gst")}


def _gragg(p: Paths) -> list[Job]:
    out = []
    for exp in GRAGG:
        tab = "gtapv7_levels" if exp in CONDENSED_GRAGG else "gtapuv7_levels"
        aux = _aux(p, tab)
        out.append(Job(
            group="gragg", name=exp, model="gtap7", dataset="nus333",
            folder=_nus333(p) / "gemsim_burfisher_gragg" / exp, tool="gemsim",
            script="gemsim_job.py",
            args=("gragg", "--experiment", exp, "--package", str(_nus333(p)),
                  "--aux", str(aux), "--gemsim", str(p.exe("gemsim.exe"))),
            data_dir=_nus333(p), variant={"experiment": exp, "method": "gragg"},
            inputs={"levels_model": _levels_model(p)},
            externals={"gemsim.exe": p.exe("gemsim.exe"), **_compiled(aux)},
            external_release={"gemsim.exe": "gp"}, solver="gemsim",
            needs=(_levels_model(p),), compiled=tab,
        ))  # fmt: skip
    return out


def _swap_job(p: Paths, group: str, ds: str, shock: float, mode: str,
              variant_extra: Mapping[str, object]) -> Job:  # fmt: skip
    tag = _tag(shock, "dpsaveSwap")
    folder = p.root / "gtap7" / ds / f"gempack_{tag}"
    exes = {"gtapv7.exe": "gempack"} | ({"sltoht.exe": "gp"} if mode == "swap" else {})
    return Job(
        group=group, name=folder.name, model="gtap7", dataset=ds, folder=folder,
        tool="gempack", script="swap_job.py",
        args=(mode, "--dataset", ds, "--shock", f"{shock:g}",
              "--template", str(CMF_TEMPLATES / f"{ds}_tm10_dpsaveswap.cmf"),
              *(a for e in exes for a in (f"--{e.removesuffix('.exe')}", str(p.exe(e))))),
        data_dir=p.root / "gtap7" / ds,
        variant={"shock": shock, "steps": STEPS, "closure": "dpsaveSwap", **variant_extra},
        externals={e: p.exe(e) for e in exes}, external_release=exes, solver="gtapv7",
    )  # fmt: skip


def _swap(p: Paths) -> list[Job]:
    return [_swap_job(p, "swap", ds, k, "swap", {}) for ds, k in SWAP]


def _swap_sltoht(p: Paths) -> list[Job]:
    out = []
    for ds, k in SWAP_LEVELS:
        run = p.root / "gtap7" / ds / f"gempack_{_tag(k, 'dpsaveSwap')}"
        template = CMF_TEMPLATES / f"{ds}_tm10_dpsaveswap.cmf"
        out.append(_resolve(p, "swap-sltoht", ds, "dpsaveSwap", run / "sl4dump.har",
                            template, (run,)))  # fmt: skip
    return out


def _validation(p: Paths) -> list[Job]:
    return [_swap_job(p, "validation", "gtap7_20x41", 10, "abort", {"outcome": "abort"})]


def _variant_args(variant: Mapping[str, object]) -> list[str]:
    """``--variant k=v``; strings quoted so write_source keeps them strings."""
    out = []
    for k, v in variant.items():
        out += ["--variant", f'{k}="{v}"' if isinstance(v, str) else f"{k}={v:g}"]
    return out


def job_command(
    job: Job, *, gempack_release: str, gp_release: str, root: Path = DATA_ROOT
) -> list[str]:
    """write_source.py recording the job module for one job."""
    release = {"gempack": gempack_release, "gp": gp_release}
    tool_version = release[RELEASE_KIND[job.tool]]
    py = sys.executable
    externals = []
    for name, path in job.externals.items():
        kind = job.external_release.get(name)
        externals += ["--external", f"{name}={path}" + (f"@{release[kind]}" if kind else "")]
    return [
        py, str(WRITE_SOURCE),
        "--data-root", str(root),
        "--folder", str(job.folder),
        "--kind", "result", "--model", job.model, "--dataset", job.dataset,
        "--tool", job.tool, "--tool-version", tool_version,
        *_variant_args(job.variant),
        "--phases", "shock",
        "--model-file", str(root / "gtap7" / "gempack"),
        "--data-dir", str(job.data_dir),
        *(a for name, path in job.inputs.items() for a in ("--input", f"{name}={path}")),
        *externals,
        *(("--solver", job.solver) if job.solver else ()),
        "--",
        py, str(TOOLS / job.script), *job.args,
        "--out", str(job.folder), "--expect-release", tool_version,
    ]  # fmt: skip


def run_job(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False).returncode


def prepare_tab(tab: str, p: Paths) -> None:
    """Compile one levels TAB of data/ with tablo into --work, before its first job."""
    sys.path.insert(0, str(TOOLS))
    import gemsim_job  # noqa: PLC0415 (imported on use)

    gemsim_job.compile_levels_tab(
        tab, levels_model=_levels_model(p), work=_aux(p, tab).parent,
        tablo=p.exe("tablo.exe"), root=p.root,
    )  # fmt: skip


def _recorded(folder: Path) -> bool:
    try:
        return load_source(folder).status == "verified"
    except SourceError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gempack-release", required=True,
                    help="GEMPACK release gtapv7.exe prints (RunGTAP), e.g. 12.1.003")
    ap.add_argument("--gp-release", required=True,
                    help="GEMPACK release tablo/gemsim/sltoht print, e.g. 11.3.004")
    ap.add_argument("--group", action="append", choices=GROUPS, default=[])
    ap.add_argument("--only", action="append", default=[],
                    help="run only this folder name (e.g. me9c); repeatable")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--work", type=Path, default=None,
                    help="scratch for the compiled TABs (never under data/)")
    ap.add_argument("--gtapdir", type=Path, default=PureWindowsPath(r"C:\runGTAP375"))
    ap.add_argument("--gp", type=Path, default=PureWindowsPath(r"C:\GP"))
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args(argv)  # fmt: skip

    root = args.data_root.resolve()
    groups = args.group or list(GROUPS)
    p = Paths(root, (args.work or _default_work()).resolve(), args.gtapdir, args.gp)
    if p.work.is_relative_to(DATA_ROOT) or p.work.is_relative_to(root):
        print(f"--work {p.work} is under data/: compiled TABs never go there", file=sys.stderr)
        return 2
    every = jobs(groups, root=root, work=p.work, gtapdir=args.gtapdir, gp=args.gp)
    if args.datasets is not None:
        every = [j for j in every if j.dataset in args.datasets]
    unknown = sorted(set(args.only) - {j.name for j in every})
    if unknown:
        print(f"unknown --only {unknown}; valid: {sorted({j.name for j in every})}",
              file=sys.stderr)  # fmt: skip
        return 2
    selected = [j for j in every if not args.only or j.name in args.only]
    print(f"{len(selected)} job(s)")
    command = {"gempack_release": args.gempack_release, "gp_release": args.gp_release,
               "root": root}  # fmt: skip
    if args.dry_run:
        for j in selected:
            print(shlex.join(job_command(j, **command)))
        return 0

    exes = {path for j in selected for name, path in j.externals.items()
            if name.endswith(".exe")}  # fmt: skip
    missing = sorted(str(e) for e in exes if not Path(str(e)).exists())
    if missing:
        print(f"not found: {missing} (pass --gtapdir / --gp)", file=sys.stderr)
        return 2

    failed: list[str] = []
    prepared: dict[str, str] = {}  # levels TAB -> "" when compiled, else why not
    for n, j in enumerate(selected, 1):
        name = f"{j.group} {j.folder.relative_to(root).as_posix()}"
        head = f"[{n}/{len(selected)}] {name}"
        if _recorded(j.folder):
            print(f"{head}: already recorded, skipped")
            continue
        missing_needs = [d.relative_to(root).as_posix() for d in j.needs if not _recorded(d)]
        if missing_needs:
            failed.append(name)
            print(f"{head}: needs {', '.join(missing_needs)} recorded first; skipped")
            continue
        if j.compiled and j.compiled not in prepared:
            try:
                prepare_tab(j.compiled, p)
                prepared[j.compiled] = ""
            except Exception as err:  # noqa: BLE001 (reported; its jobs are skipped)
                prepared[j.compiled] = f"{type(err).__name__}: {err}"
        if j.compiled and prepared[j.compiled]:
            failed.append(name)
            print(f"{head}: {j.compiled} not compiled ({prepared[j.compiled]}); skipped")
            continue
        if j.folder.exists():  # left by a run that died: not an input of anything
            shutil.rmtree(j.folder)
        print(f"{head}: running", flush=True)
        rc = run_job(job_command(j, **command))
        if rc != 0:
            shutil.rmtree(j.folder, ignore_errors=True)
            failed.append(name)
            print(f"{head}: FAILED (exit {rc}), folder removed")
    print(f"\nrecorded {len(selected) - len(failed)}/{len(selected)}")
    for name in failed:
        print(f"  failed: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
