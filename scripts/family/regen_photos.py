"""Regenerate the saved Pyomo photos the family tests compare against.

    EQUILIBRIA_GTAP_TR_LINSOLVE=mumps <python with pymumps> \\
        scripts/family/regen_photos.py --dataset gtap7_3x3 [--dataset ...] | --all

Runs the Pyomo reference of every selected variant (``photos.VARIANTS``) into
``data/gtap7/<ds>/equilibria_pyomo-photo_<tag>/``, writes its ``model_print.json``
and ``source.yaml`` (ADR-0010), then the Pyomo parity tests of those datasets
against GAMS, so a regenerated photo is checked before it is committed.
MUMPS only: the family runs match a photo at 1e-9 only with the same linear
solver, and 20x41 needs MUMPS.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "family"), str(ROOT / "scripts" / "dev")]
from capture_prep import capture  # noqa: E402
from write_source import _git_commit, machine_info  # noqa: E402

from equilibria.blocks.gtap.fingerprint import model_print  # noqa: E402
from equilibria.family import photos  # noqa: E402
from equilibria.sources import DATA_ROOT, dataset_dir, sha256_file  # noqa: E402

GATE_TESTS = (
    "tests/templates/gtap/test_gtap7_mcp_parity.py",
    "tests/templates/gtap/test_gtap7_nlp_parity.py",
)


def _require_mumps() -> None:
    if os.environ.get("EQUILIBRIA_GTAP_TR_LINSOLVE") != photos.LINSOLVE:
        sys.exit("regen_photos: set EQUILIBRIA_GTAP_TR_LINSOLVE=mumps (photos are MUMPS)")
    try:
        import mumps  # noqa: F401
    except ImportError:
        sys.exit("regen_photos: pymumps (import mumps) is missing; use the eq-mumps env")


def _residual_region(dataset: str) -> str:
    from equilibria.templates.gtap import GTAPParameters

    d = dataset_dir(dataset)
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return list(p.sets.r)[-1]  # _solve_shock's default


def pack(folder: Path) -> None:
    """gzip every pickle in ``folder`` (GitHub refuses files over 100 MB: the 20x41
    final.pkl is ~294 MB, ~25 MB packed) and lower every name (the repo's layout
    rule); ``photos.unpacked`` restores both."""
    for f in sorted(folder.iterdir()):
        name = photos.stored_name(f.name)
        if f.suffix == ".pkl":
            data = gzip.compress(f.read_bytes(), 6, mtime=0)
            f.with_name(name + ".gz").write_bytes(data)
            f.unlink()
        elif name != f.name:
            f.rename(f.with_name(name))


def write_photo(
    dataset: str, ifsub: int, savf: str, mode: str, keep: str, *, root: Path = DATA_ROOT
) -> Path:
    """Capture one variant and record it; the old photo is replaced only on success."""
    out = photos.photo_dir(dataset, ifsub, savf, mode, root=root)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t0 = time.time()
    started = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    code = capture(dataset, tmp, ifsub=ifsub, savf=savf, mode=mode, keep=keep)
    duration = round(time.time() - t0, 3)
    if code != 1:
        raise RuntimeError(f"{dataset} ifsub={ifsub} {savf}: Pyomo shock code {code}")
    rr = _residual_region(dataset)
    data = dataset_dir(dataset)
    mp = model_print(
        data_files=[data / f for f in photos.DATA_FILES],
        closure=photos.photo_closure(ifsub, savf),
        residual_region=rr,
        linsolve=photos.LINSOLVE,
    )
    if mp is None:
        raise RuntimeError(f"{dataset}: the model print cannot cover data or closure")
    (tmp / "model_print.json").write_text(json.dumps(mp.to_dict(), indent=1) + "\n")
    pack(tmp)
    import yaml

    record = {
        "kind": "result",
        "model": "gtap7",
        "dataset": dataset,
        "tool": "equilibria-pyomo",
        "variant": {
            "mode": mode,
            "ifsub": ifsub,
            "savf": savf,
            "keep": keep,
            "residual_region": rr,
        },
        "phases": ["base", "check", "shock"],
        "generated_from": {
            "data": {
                "path": data.relative_to(DATA_ROOT).as_posix(),
                "sha256": {f: sha256_file(data / f) for f in photos.DATA_FILES},
            }
        },
        "solver": {"name": "newton_tr", "linsolve": photos.LINSOLVE},
        "command": " ".join(
            [photos.REGEN, "--dataset", dataset]
        ),
        "equilibria_commit": _git_commit(),
        "exit": {"code": 0},
        "outputs": {
            p.name: sha256_file(p) for p in sorted(tmp.iterdir()) if p.is_file()
        },
        "run": {"date": started, "duration_s": duration, "machine": machine_info()},
        "status": "verified",
        "notes": "",
    }
    (tmp / "source.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--root", type=Path, default=DATA_ROOT)
    ap.add_argument("--no-gates", action="store_true")
    a = ap.parse_args(argv)
    _require_mumps()
    chosen = [v for v in photos.VARIANTS if a.all or v[0] in a.dataset]
    if not chosen:
        sys.exit("regen_photos: no variant selected (--dataset DS or --all)")
    for ds, ifsub, savf, mode, keep in chosen:
        t = time.time()
        out = write_photo(ds, ifsub, savf, mode, keep, root=a.root)
        mb = sum(p.stat().st_size for p in out.iterdir()) / 2**20
        print(f"{ds} ifsub={ifsub} {savf}: {(time.time() - t) / 60:.2f} min, {mb:.1f} MB -> {out}")
    if a.no_gates:
        return 0
    datasets = sorted({v[0] for v in chosen})
    gate = [sys.executable, "-m", "pytest", "-q", *GATE_TESTS, "-k", " or ".join(datasets)]
    print("gates:", " ".join(gate), flush=True)
    return subprocess.run(gate, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
