"""Burfisher 3e / nus333: equilibria contra GEMPACK en NIVELES, ejercicio por ejercicio.

Cada ejercicio de ``run_burfisher.EXERCISES`` (el ``.prm``, los shocks y el cierre de
la Tabla 3) se resuelve con equilibria y se compara, pre y post, contra la solucion
GEMPACK del mismo ``.EXP`` con el mapeo GTAP de ``compare_gempack_levels.py``. The
book's runs are the records ``data/gtap7/nus333/gempack_burfisher/<exp>/``
(``solution.sl4``, ``updated.har``, ``run.cmf``), found with
``equilibria.sources.reference``.

El metodo de GEMPACK sale del command file (``Method = ...``; the last one wins). Seis ejercicios son Johansen de
un paso porque asi los resolvio el libro (TBL45A/B/C, TBL64, ME4A/B): una solucion en
niveles no tiene por que coincidir con esa extrapolacion lineal, y se reportan
aparte. Las variables sin nivel en el ``.sl4`` (``rore``, ``y``, ``pinv``, ``qinv``,
``pva``, ``pe``: GTAPV7 sin el parche de ``make_levels_tab.py``) quedan excluidas
con su motivo en el reporte.

Uso (desde la raiz, con PATH; el paquete del libro es ``data/gtap7/nus333``)::

    PYTHONPATH=src:. python scripts/gtap/compare_burfisher_gempack.py [--only TBL46A]

o, sin resolver, con niveles ya guardados (``--save-eq-dir`` o
``data/gtap7/nus333/equilibria_burfisher_levels/make_eq_burfisher_levels.py``)::

    uv run python scripts/gtap/compare_burfisher_gempack.py --eq-dir data/gtap7/nus333/equilibria_burfisher_levels
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from equilibria.sources import dataset_dir, find_reference, reference

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "gtap")]

import compare_gempack_levels as cg  # noqa: E402
import run_burfisher as rb  # noqa: E402

from equilibria.compare import Report  # noqa: E402

BASEDATA = dataset_dir("nus333") / "basedata.har"
# A GEMPACK source: "book" (the book's own runs, data/gtap7/nus333/gempack_burfisher/)
# or a folder of flat <exp>.sl4.gz/.upd/.log.gz files.
Source = str | Path
BOOK = "book"


def solution(exp: str, source: Source = BOOK) -> Path | None:
    """The folder of ``exp``'s GEMPACK run in ``source``, or None if it has none."""
    if isinstance(source, Path):
        return source if (source / f"{exp.lower()}.sl4.gz").exists() else None
    return find_reference("nus333", "gempack", experiment=exp.lower(), method=source)


def method(exp: str, source: Source = BOOK) -> str:
    """El metodo de la corrida GEMPACK: la ultima linea ``Method = X;`` que corrio."""
    if isinstance(source, Path):
        with gzip.open(source / f"{exp.lower()}.log.gz", "rb") as fh:
            raw = fh.read()
    else:
        rec = reference("nus333", "gempack", experiment=exp.lower(), method=source)
        raw = (rec / "run.cmf").read_bytes()
    # gemsim en Windows escribe el log en UTF-16 (con BOM)
    utf16 = raw[:2] in (b"\xff\xfe", b"\xfe\xff")
    text = raw.decode("utf-16" if utf16 else "utf-8", errors="replace")
    found = re.findall(r"(?im)^\s*method\s*=\s*(\w+)\s*;", text)
    if not found:
        raise ValueError(f"{exp}: el command file no dice el metodo")
    return found[-1].capitalize()


def run(
    exp: str, levels: dict, xscale: dict, source: Source = BOOK, *, linear: bool = False
) -> Report:
    """equilibria (``levels``, ``xscale``) contra la solucion GEMPACK de ``exp``.

    Los niveles salen de ``sl4levels_<exp>.har`` (sltoht) si existe; si no, del
    ``.sl4``, que tiene que traer ``PLEV``. The book's runs are read from their
    ``.sl4``. ``linear``: ``levels`` is a Johansen solution (see
    ``compare_gempack_levels.equilibria_side``)."""
    if not isinstance(source, Path):
        rec = reference("nus333", "gempack", experiment=exp.lower(), method=source)
        return cg.run(
            levels, xscale, rec / "solution.sl4", BASEDATA, rec / "updated.har",
            linear=linear,
        )  # fmt: skip
    fix = source
    upd = fix / f"{exp.lower()}.upd"
    har = fix / f"sl4levels_{exp.lower()}.har"
    if har.exists():
        return cg.run(levels, xscale, har, BASEDATA, upd, linear=linear)
    with tempfile.TemporaryDirectory() as tmp:
        sl4 = Path(tmp) / f"{exp.lower()}.sl4"
        with (
            gzip.open(fix / f"{exp.lower()}.sl4.gz", "rb") as src,
            sl4.open("wb") as dst,
        ):
            shutil.copyfileobj(src, dst)
        return cg.run(levels, xscale, sl4, BASEDATA, upd, linear=linear)


def solve(exp: str) -> tuple[dict, dict]:
    """(levels, xscale) de equilibria en el ejercicio, con el cierre de la Tabla 3."""
    from equilibria._local_refs import nus333_dir, path_capi_src
    from equilibria.blocks.gtap._derived_params import xscale_data

    sys.path.insert(0, str(path_capi_src()))
    m, p, codes = rb.solve_exercise(exp, nus333_dir())
    if set(codes.values()) != {1}:
        print(f"{exp:7s} equilibria no resolvio: {codes}", flush=True)
    return cg.model_levels(m), dict(xscale_data(p, p.sets))


def _line(exp: str, meth: str, r: Report) -> str:
    o = r.overall
    w = o["pct_within"]
    return (
        f"{exp:7s} {meth:8s} celdas={o['cells']:4d} <=0,1%={w[1e-3]:6.2f}% "
        f"<=0,5%={w[5e-3]:6.2f}% <=1%={w[1e-2]:6.2f}% max={o['max']:.3g} "
        f"en {o['max_variable']} {o['max_cell']}  sin pareja={len(r.unmatched)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--only", default="", help="EXP separados por coma")
    ap.add_argument("--eq-dir", type=Path, help="niveles de equilibria ya guardados")
    ap.add_argument("--save-eq-dir", type=Path, help="guardar los niveles resueltos")
    ap.add_argument("--json", type=Path, help="guardar los resultados en JSON")
    ap.add_argument(
        "--gempack-dir",
        type=lambda v: v if v == BOOK else Path(v),
        default=BOOK,
        help='"book" (the book\'s runs, the default) or a folder of GEMPACK runs',
    )
    a = ap.parse_args()

    todo = [e for e in a.only.split(",") if e] or list(rb.EXERCISES)
    results: dict[str, dict] = {}
    excluded: dict[str, str] = {}
    for exp in todo:
        if solution(exp, a.gempack_dir) is None:
            print(f"{exp:7s} sin solucion GEMPACK", flush=True)
            continue
        if a.eq_dir:
            eq = a.eq_dir / f"eq_burfisher_{exp.lower()}_levels.json.gz"
            if not eq.exists():
                print(f"{exp:7s} sin niveles de equilibria ({eq.name})", flush=True)
                continue
            levels, xscale = cg.load_eq_json(eq)
        else:
            levels, xscale = solve(exp)
            if a.save_eq_dir:
                path = a.save_eq_dir / f"eq_burfisher_{exp.lower()}_levels.json.gz"
                cg.save_eq_json(path, levels, xscale)
        meth = method(exp, a.gempack_dir)
        r = run(exp, levels, xscale, a.gempack_dir)
        print(_line(exp, meth, r), flush=True)
        excluded = r.excluded
        results[exp] = {"method": meth, "report": json.loads(r.to_json())}

    for meth in ("Gragg", "Johansen"):
        rs = [x["report"]["overall"] for x in results.values() if x["method"] == meth]
        if not rs:
            continue
        n = sum(o["cells"] for o in rs)
        within = {
            t: sum(o["cells"] * o["pct_within"][t] for o in rs) / n
            for t in rs[0]["pct_within"]
        }
        pct = "  ".join(f"<={100 * float(t):g}%={v:6.2f}%" for t, v in within.items())
        print(f"TOTAL {meth:8s} ejercicios={len(rs):2d} celdas={n:6d}  {pct}")
    for var, why in excluded.items():
        print(f"excluida: {var}: {why}")
    if a.json:
        a.json.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
