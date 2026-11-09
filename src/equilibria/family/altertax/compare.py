"""Levels gate of the ALTERTAX family solution against a GEMPACK run folder."""

from __future__ import annotations

import gzip
import importlib.util
import shutil
import tempfile
from pathlib import Path

from equilibria.compare import Report
from equilibria.family.altertax.levels import family_levels
from equilibria.family.altertax.run import AltertaxRun

_ROOT = Path(__file__).resolve().parents[4]

# the committed fixture names first, then GEMPACK's own run-folder names
LEVELS_NAMES = ("sl4levels_altax.har", "altax_sl4levels.har", "sl4levels.har")
# the solution itself, when no levels .har was exported (``from_har`` reads a ``.sl4``)
SL4_NAMES = ("solution.sl4", "altax.sl4", "solution.sl4.gz", "altax.sl4.gz")
UPDATED_NAMES = ("updated.har", "altax_updated.har", "altax.upd")


def load_compare_script():
    """``scripts/gtap/compare_gempack_levels.py`` as a module (``run``, ``EQ_VARS``)."""
    path = _ROOT / "scripts" / "gtap" / "compare_gempack_levels.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: compare_levels needs the repo checkout's scripts/ folder"
        )
    spec = importlib.util.spec_from_file_location("compare_gempack_levels", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find(folder: Path, names: tuple[str, ...]) -> Path:
    for n in names:
        if (folder / n).exists():
            return folder / n
    raise FileNotFoundError(
        f"{names[0]} not found in {folder} (also tried {names[1:]})"
    )


def compare_levels(
    run: AltertaxRun, gempack_dir: Path, basedata: Path | None = None
) -> Report:
    """Compare check (pre) and shock (post) levels with ``sl4levels`` + ``.upd`` found
    in ``gempack_dir``. ``basedata`` defaults to the nus333 ``basedata.har``."""
    from equilibria._local_refs import nus333_dir

    d = Path(gempack_dir)
    har = _find(d, LEVELS_NAMES + SL4_NAMES)
    upd = _find(d, UPDATED_NAMES)
    base = Path(basedata) if basedata is not None else nus333_dir() / "basedata.har"
    cg = load_compare_script()
    with tempfile.TemporaryDirectory() as tmp:
        if har.suffix == ".gz":  # an .sl4.gz is read unzipped
            plain = Path(tmp) / har.stem
            with gzip.open(har, "rb") as src, plain.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            har = plain
        return _levels_report(cg, run, har, base, upd)


def _levels_report(cg, run: AltertaxRun, har: Path, base: Path, upd: Path) -> Report:
    from equilibria.blocks.gtap._derived_params import xscale_data

    levels = family_levels(run.result.mp, cg.EQ_VARS)
    xscale = dict(xscale_data(run.params, run.params.sets))
    return cg.run(levels, xscale, har, base, upd)
