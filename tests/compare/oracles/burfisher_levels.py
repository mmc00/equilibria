"""Reproduce el oraculo del ejemplo 18 (tests/compare/_oracle.py) SIN equilibria.compare.

Dos ejercicios de Burfisher: TBL46A (arancel de USA a MFG de ROW, ``esubd0.8.prm``,
GEMPACK Gragg 2-4-6) y TBL45A (productividad de SER en USA, Johansen de 1 paso, como
el libro: no tiene por que coincidir con una solucion en niveles).
equilibria resuelto contra GEMPACK en NIVELES, pre y post, con las mismas formulas
del ejemplo 15 (``gtap7_map_3x3.py``). A = ``tests/compare/fixtures/
eq_burfisher_<exp>_levels.json.gz`` (``make_eq_burfisher_levels.py``); B = el ``.sl4``
de GEMPACK (``tests/fixtures/gtap7_gempack/burfisher/<exp>.sl4.gz``),
leido aca a mano: VARS/PLEV/VNCP dan donde empieza cada variable en LEVB (pre) y
LEVA (post); VCSP/VCNI/VCSN/STNM/SSZ/STEL, sus sets; orden Fortran. VDFB de
basedata.har (pre) y de ``<exp>.upd`` (post).

Estos ``.sl4`` son del GTAPV7 sin parche: ``rore``, ``y``, ``pinv``, ``qinv``, ``pva`` y
``pe`` no tienen nivel (PLEV = 0), asi que no se miden.

Uso (desde la raiz): ``uv run python tests/compare/oracles/burfisher_levels.py``
"""

from __future__ import annotations

import gzip
import itertools
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from gtap7_map_3x3 import GEMPACK_VARS, ROOT, _add_vdfb, load_eq, report, rows_of

from equilibria.babel.har.reader import read_har

FIX = ROOT / "tests" / "fixtures" / "gtap7_gempack" / "burfisher"
EQ = ROOT / "tests" / "compare" / "fixtures"
NUS333 = ROOT / "src" / "equilibria" / "templates" / "reference" / "gtap" / "data"


def sl4_levels(path: Path) -> dict[str, dict[tuple[str, tuple[str, ...]], float]]:
    """{var: {(periodo, indices): nivel}} de las GEMPACK_VARS con nivel."""
    d = read_har(str(path))
    s = lambda k: [str(x).strip() for x in np.asarray(d[k].array).ravel()]  # noqa: E731
    i = lambda k: np.asarray(d[k].array, dtype=int).ravel()  # noqa: E731
    f = lambda k: np.asarray(d[k].array, dtype=float).ravel()  # noqa: E731
    names, ncomp, plev = s("VARS"), i("VNCP"), i("PLEV")
    vcsp, vcni, vcsn = i("VCSP"), i("VCNI"), i("VCSN")
    stnm, ssz, stel = s("STNM"), i("SSZ"), s("STEL")
    levb, leva = f("LEVB"), f("LEVA")
    assert len(vcsp) == len(names), "VCSP no cubre VARS: otro modelo (ver #148)"
    out: dict[str, dict] = {}
    for j, name in enumerate(names):
        if name not in GEMPACK_VARS or plev[j] == 0:
            continue
        sets = []
        for sn in vcsn[vcsp[j] - 1 : vcsp[j] - 1 + vcni[j]]:
            start = int(ssz[: sn - 1].sum())
            sets.append(stel[start : start + ssz[sn - 1]])
        # Fortran: el primer indice varia mas rapido.
        keys = [k[::-1] for k in itertools.product(*sets[::-1])] or [()]
        assert len(keys) == ncomp[j], name
        a = plev[j] - 1
        cells = out.setdefault(name, {})
        for n, key in enumerate(keys):
            cells[("pre", tuple(key))] = float(levb[a + n])
            cells[("post", tuple(key))] = float(leva[a + n])
    return out


def run(exp: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sl4 = Path(tmp) / f"{exp}.sl4"
        with gzip.open(FIX / f"{exp}.sl4.gz", "rb") as src, sl4.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        B = sl4_levels(sl4)
    _add_vdfb(B, NUS333 / "nus333" / "basedata.har", FIX / f"{exp}.upd")
    print(f"== {exp}: variables de B con nivel: {len(B) - 1} (+ VDFB)")
    print(f"  sin nivel: {sorted(set(GEMPACK_VARS) - set(B))}")
    eq, xs = load_eq(EQ / f"eq_burfisher_{exp}_levels.json.gz")
    report(eq, rows_of(eq, xs, B))


def main() -> None:
    for exp in sys.argv[1:] or ("tbl46a", "tbl45a"):
        run(exp)


if __name__ == "__main__":
    main()
