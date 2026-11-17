"""The saved Pyomo photos the family (JAX) route is checked against.

A photo is a Pyomo run of one dataset and closure, captured by
``scripts/family/regen_photos.py`` on a machine with MUMPS and committed under
``data/gtap7/<dataset>/equilibria_pyomo-photo_<closure tag>/`` with its
``source.yaml``: the family tests read it instead of solving Pyomo.  Its
``model_print.json`` (``fingerprint.ModelPrint``) says what it was made from, so a
change in the Pyomo-only code makes it stale (see ``freshness``).

This module is the read side; writing needs Pyomo and lives in the script.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from equilibria.sources import DATA_ROOT

PHOTO_FOLDER = "equilibria_pyomo-photo"
# the dataset files a photo's data print covers (the inputs _solve_shock loads)
DATA_FILES = ("basedata.har", "sets.har", "default.prm", "baserate.har")
LINSOLVE = "mumps"  # every photo is made with MUMPS (20x41 needs it)
REGEN = "scripts/family/regen_photos.py"

# (dataset, ifsub, savf, mode, keep): "all" keeps the per-step photos the prep tests
# read, "gate" only final.pkl, settled.pkl and the solve records.
VARIANTS: tuple[tuple[str, int, str, str, str], ...] = (
    ("gtap7_3x3", 1, "capFix", "gtap", "all"),
    ("gtap7_3x3", 0, "capFix", "gtap", "all"),
    ("gtap7_3x3", 1, "capFlex", "gtap", "all"),
    ("gtap7_10x7", 1, "capFix", "gtap", "all"),
    ("gtap7_15x10", 1, "capFix", "gtap", "gate"),
    ("gtap7_20x41", 1, "capFix", "gtap", "gate"),
)


def closure_tag(ifsub: int, savf: str, mode: str) -> str:
    """Folder suffix of a closure; today's default keeps the reference's old name."""
    if (ifsub, savf, mode) == (1, "capFix", "gtap"):
        return "tol_today"
    return f"{mode}_ifsub{ifsub}_{savf}"


def photo_dir(
    dataset: str,
    ifsub: int = 1,
    savf: str = "capFix",
    mode: str = "gtap",
    *,
    root: Path = DATA_ROOT,
) -> Path:
    family = dataset.split("_", 1)[0]
    return root / family / dataset / f"{PHOTO_FOLDER}_{closure_tag(ifsub, savf, mode)}"


def photo_closure(
    ifsub: int,
    savf: Literal["capFix", "capSFix", "capShrFix", "capFlex", "capFixDp"],
    unit_sigma: Literal["gams", "gempack"] = "gams",
):
    """The closure ``_solve_shock`` (the photos' Pyomo run) solves with."""
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=bool(ifsub),
        savf_flag=savf,
        numeraire="pnum",
        # GEMPACK is the reference: use the data's factor wedge (EVFP = EVFB +
        # FTRV + FBEP), the gtap7_gempack basis. The GAMS sign drops 15x10 to 93.9%.
        va_subsidy_basis="gempack",
        unit_sigma=unit_sigma,
    )


class PhotoError(RuntimeError):
    """A photo that is missing, damaged or made for another run."""


# the per-step capture names carry an upper-case kind (001_A_base.pkl, 002_A2_check.pkl);
# the repo keeps file names lowercase, so a photo stores them lowered
_STEP = re.compile(r"(\d{3})_([a-z]\d?)_(.+)")


def stored_name(name: str) -> str:
    """The name a capture file has inside the committed photo."""
    return name.lower()


def plain_name(stored: str) -> str:
    """The capture name back from ``stored_name`` (``.gz`` kept, if any)."""
    m = _STEP.fullmatch(stored)
    return f"{m[1]}_{m[2].upper()}_{m[3]}" if m else stored


def _regen_hint(dataset: str) -> str:
    return f"run {REGEN} --dataset {dataset} (with MUMPS)"


def _record(folder: Path, dataset: str) -> Mapping[str, Any]:
    from equilibria.sources import SourceError, load_source

    try:
        return load_source(folder).raw
    except SourceError as err:
        raise PhotoError(
            f"{folder}: no photo ({err}); {_regen_hint(dataset)}"
        ) from None


def _check_hashes(folder: Path, rec: Mapping[str, Any], dataset: str) -> None:
    from equilibria.sources import sha256_file

    for name, sha in rec["outputs"].items():
        f = folder / name
        if not f.is_file() or sha256_file(f) != sha:
            raise PhotoError(
                f"{folder}: {name} is missing or does not match its sha256; "
                f"{_regen_hint(dataset)}"
            )


def _cache_root() -> Path:
    import os

    base = (
        os.environ.get("EQUILIBRIA_FAMILY_CACHE")
        or Path.home() / ".cache/equilibria/family"
    )
    return Path(base) / "photos"


def unpacked(
    dataset: str,
    ifsub: int = 1,
    savf: str = "capFix",
    mode: str = "gtap",
    *,
    linsolve: str = LINSOLVE,
    root: Path = DATA_ROOT,
    cache_root: Path | None = None,
) -> Path:
    """The photo as plain files (the ``.gz`` outputs decompressed once into a cache
    keyed by the record's hash), after checking it is there, intact and made with
    ``linsolve``: a family run matches a photo at 1e-9 only with the same solver."""
    import gzip
    import hashlib
    import shutil
    import tempfile

    folder = photo_dir(dataset, ifsub, savf, mode, root=root)
    rec = _record(folder, dataset)
    made_with = rec.get("solver", {}).get("linsolve")
    if made_with != linsolve:
        raise PhotoError(
            f"{folder}: made with linsolve={made_with!r}, this run uses {linsolve!r}; "
            "a family run matches a photo only with the same linear solver"
        )
    _check_hashes(folder, rec, dataset)
    key = hashlib.sha256((folder / "source.yaml").read_bytes()).hexdigest()[:12]
    out = (cache_root or _cache_root()) / f"{dataset}_{folder.name}_{key}"
    if out.is_dir():
        return out
    # a temp folder of its own: pytest-xdist workers share the cache in CI
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out.parent, prefix=out.name + ".tmp-"))
    try:
        for name in rec["outputs"]:
            src, dst = folder / name, tmp / plain_name(name)
            if name.endswith(".gz"):
                dst.with_suffix("").write_bytes(gzip.decompress(src.read_bytes()))
            else:
                shutil.copyfile(src, dst)
        try:
            tmp.rename(out)
        except OSError:
            if not out.is_dir():  # not another worker finishing first
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def freshness(
    dataset: str,
    ifsub: int = 1,
    savf: str = "capFix",
    mode: str = "gtap",
    *,
    root: Path = DATA_ROOT,
) -> Literal["fresh", "shared_changed", "stale"]:
    """``stale`` when the data, the closure or the Pyomo-only code changed since the
    photo (regenerate it); ``shared_changed`` when only code both routes run changed
    (the family run must still match it at 1e-9)."""
    import json

    from equilibria.blocks.gtap.fingerprint import ModelPrint, model_print
    from equilibria.sources import dataset_dir

    folder = photo_dir(dataset, ifsub, savf, mode, root=root)
    rec = _record(folder, dataset)
    saved = ModelPrint.from_dict(json.loads((folder / "model_print.json").read_text()))
    now = model_print(
        data_files=[dataset_dir(dataset) / f for f in DATA_FILES],
        closure=photo_closure(ifsub, savf),  # type: ignore[arg-type]
        residual_region=rec["variant"]["residual_region"],
        linsolve=LINSOLVE,
    )
    if now is None or (now.data, now.closure, now.pyomo_code) != (
        saved.data,
        saved.closure,
        saved.pyomo_code,
    ):
        return "stale"
    return "fresh" if now.shared_code == saved.shared_code else "shared_changed"
