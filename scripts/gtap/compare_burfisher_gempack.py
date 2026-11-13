"""Burfisher 3e / nus333: equilibria contra GEMPACK en NIVELES, ejercicio por ejercicio.

Cada ejercicio de ``run_burfisher.EXERCISES`` (el ``.prm``, los shocks y el cierre de
la Tabla 3) se resuelve con equilibria y se compara, pre y post, contra la solucion
GEMPACK del mismo ``.EXP`` con el mapeo GTAP de ``compare_gempack_levels.py``. The
book's runs are the records ``data/gtap7/nus333/gempack_burfisher/<exp>/``
(``solution.sl4``, ``updated.har``, ``run.cmf``) and, with ``--gempack-dir gragg``,
their Gragg 2-4-6 reruns ``gemsim_burfisher_gragg/<exp>/``; both are found with
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
import json
import re
import sys
from pathlib import Path

from equilibria.sources import dataset_dir, find_reference, reference

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "gtap")]

import compare_gempack_levels as cg  # noqa: E402
import run_burfisher as rb  # noqa: E402

from equilibria.compare import Report  # noqa: E402

BASEDATA = dataset_dir("nus333") / "basedata.har"
# Where GEMPACK solved each exercise: "book" (the book's own runs,
# data/gtap7/nus333/gempack_burfisher/) or "gragg" (Gragg 2-4-6 on the levels TAB,
# gemsim_burfisher_gragg/ with their levels in sltoht_burfisher_gragg/). A Path is
# one run's folder (run.cmf, solution.sl4, updated.har, maybe sl4levels.har).
Source = str | Path
BOOK, GRAGG = "book", "gragg"
_TOOL = {BOOK: "gempack", GRAGG: "gemsim"}


def solution(exp: str, source: Source = BOOK) -> Path | None:
    """The folder of ``exp``'s GEMPACK run in ``source``, or None if it has none."""
    if isinstance(source, Path):
        return source if (source / "solution.sl4").exists() else None
    return find_reference(
        "nus333", _TOOL[source], experiment=exp.lower(), method=source
    )


def _run_dir(exp: str, source: Source) -> Path:
    if isinstance(source, Path):
        return source
    return reference("nus333", _TOOL[source], experiment=exp.lower(), method=source)


def _levels_har(exp: str, source: Source) -> Path | None:
    """sltoht's levels .har of the run, if it has one (the book's runs: none used)."""
    if source == GRAGG:
        rec = reference("nus333", "sltoht", experiment=exp.lower(), method=GRAGG)
        return rec / "sl4levels.har"
    if isinstance(source, Path) and (source / "sl4levels.har").exists():
        return source / "sl4levels.har"
    return None


def method(exp: str, source: Source = BOOK) -> str:
    """El metodo de la corrida GEMPACK: la ultima linea ``Method = X;`` que corrio."""
    raw = (_run_dir(exp, source) / "run.cmf").read_bytes()
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

    Los niveles salen del ``sl4levels.har`` (sltoht) de la corrida si existe; si no,
    del ``.sl4``, que tiene que traer ``PLEV``. ``linear``: ``levels`` is a Johansen
    solution (see ``compare_gempack_levels.equilibria_side``)."""
    d = _run_dir(exp, source)
    har = _levels_har(exp, source) or d / "solution.sl4"
    return cg.run(levels, xscale, har, BASEDATA, d / "updated.har", linear=linear)


def solve(exp: str) -> tuple[dict, dict]:
    """(levels, xscale) de equilibria en el ejercicio, con el cierre de la Tabla 3."""
    from equilibria._local_refs import nus333_dir, path_capi_src
    from equilibria.blocks.gtap._derived_params import xscale_data

    sys.path.insert(0, str(path_capi_src()))
    m, p, codes = rb.solve_exercise(exp, nus333_dir())
    if set(codes.values()) != {1}:
        print(f"{exp:7s} equilibria no resolvio: {codes}", flush=True)
    return cg.model_levels(m), dict(xscale_data(p, p.sets))


def exercises(only: str, source: Source) -> list[str]:
    """The exercises to compare: ``--only``'s, or every one of Table 3.

    A folder is one run, so it needs ``--only`` with exactly one exercise; else
    every exercise would be compared against that one run."""
    todo = [e for e in only.split(",") if e] or list(rb.EXERCISES)
    if isinstance(source, Path) and len(todo) != 1:
        raise ValueError(
            f"--gempack-dir {source} is one run: name its exercise with --only EXP"
        )
    return todo


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
        type=lambda v: v if v in _TOOL else Path(v),
        default=BOOK,
        help='"book" (the book\'s runs, the default), "gragg", or one run\'s folder '
        "(with --only EXP)",
    )
    a = ap.parse_args()

    try:
        todo = exercises(a.only, a.gempack_dir)
    except ValueError as err:
        ap.error(str(err))
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
