"""Oraculo: el nivel PRE (LEVB) que GEMPACK debe reportar para las seis variables a
las que ``scripts/gtap/make_levels_tab.py`` les agrega ``orig_level``.

Sin equilibria.compare ni el parche del TAB: lee ``basedata.har`` y ``default.prm``
con ``babel.har.reader`` y aplica a mano las definiciones de nivel de GTAPv7.tab
(entre parentesis, la linea de gtapv7.tab de GTAP 11a donde se lee la definicion):

  y(r)     INCOME(r) = PRIVEXP + GOVEXP + SAVE                         (768)
           = sum_c (VDPP + VMPP) + sum_c (VDGP + VMGP) + SAVE
  qinv(r)  REGINV(r) = sum_c VIP = sum_c (VDIP + VMIP)                  (890)
  rore(r)  RORC * (KB / KE)^RORFLEX, con                                (2522, E_rore)
             RORC = [GROSSCAP - VDEP] / VKB,  GROSSCAP = sum_{ENDWC} VES (2464)
             KE   = VKB - VDEP + REGINV  (pinv = 1 en la base)
  pinv(r), pva(a,r), pe(e,r): indices de precio, 1 en la base.

Y lo contrasta con la base del GDX de GAMS del 3x3 (``out_gtap_shock_ifsub0.gdx``),
que define los mismos niveles por su cuenta (cal.gms:409-411, 671-673):
  regY = 1e-6 * INCOME,  xi = 1e-6 * REGINV,  rore = rore,  pi = pva = 1.

Uso (desde la raiz): ``uv run python tests/compare/oracles/levb_six_vars.py``
"""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from pathlib import Path

import numpy as np

from equilibria.sources import dataset_dir, has_dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _diff_core import GDXDUMP  # type: ignore  # noqa: E402

from equilibria.babel.har.reader import read_har  # noqa: E402

GDX_3X3 = ROOT / "tests/fixtures/gtap7/gtap7_3x3/out_gtap_shock_ifsub0.gdx"


def _arr(h, k):
    v = h[k]
    return np.asarray(getattr(v, "array", v), dtype=float)


def _labels(sets, k):
    v = sets[k]
    return [str(x).strip() for x in np.asarray(getattr(v, "array", v)).flatten()]


def levb(dataset: str) -> dict[str, dict[str, float]]:
    """Nivel PRE de y, qinv y rore por region, desde los datos (en millones de USD)."""
    d = dataset_dir(dataset)
    b = read_har(str(d / "basedata.har"))
    s = read_har(str(d / "sets.har"))
    p = read_har(str(d / "default.prm"))
    reg = _labels(s, "REG")
    endw = _labels(s, "ENDW")
    capital = [i for i, e in enumerate(endw) if e.lower() == "capital"]
    assert len(capital) == 1, endw  # ENDWC = {Capital} en los datasets del paper

    income = (
        (_arr(b, "VDPP") + _arr(b, "VMPP")).sum(axis=0)
        + (_arr(b, "VDGP") + _arr(b, "VMGP")).sum(axis=0)
        + _arr(b, "SAVE")
    )
    reginv = (_arr(b, "VDIP") + _arr(b, "VMIP")).sum(axis=0)
    vkb, vdep = _arr(b, "VKB"), _arr(b, "VDEP")
    grosscap = _arr(b, "EVOS")[capital[0]].sum(axis=0)  # VES = sum_a EVOS
    rorc = (grosscap - vdep) / vkb
    rore = rorc * (vkb / (vkb - vdep + reginv)) ** _arr(p, "RFLX")
    return {
        "y": dict(zip(reg, income.tolist(), strict=True)),
        "qinv": dict(zip(reg, reginv.tolist(), strict=True)),
        "rore": dict(zip(reg, rore.tolist(), strict=True)),
    }


def _gdx_base(symbol: str) -> dict[tuple[str, ...], float]:
    out = subprocess.run(
        [GDXDUMP, str(GDX_3X3), f"symb={symbol}", "format=csv"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # gdxdump format=csv de una variable: cabecera (dims..., "t", "Val"), nivel en Val.
    head, *rows = csv.reader(io.StringIO(out))
    assert head[-2:] == ["t", "Val"], head
    return {tuple(r[:-2]): float(r[-1]) for r in rows if r[-2] == "base"}


def main() -> None:
    for ds in ("gtap7_3x3", "gtap7_20x41"):
        if not has_dataset(ds):
            print(f"{ds}: sin datos")
            continue
        o = levb(ds)
        print(f"== {ds}")
        for v, by_r in o.items():
            items = list(by_r.items())
            print(
                f"  {v}: {len(items)} regiones; "
                + ", ".join(f"{r}={x:.10g}" for r, x in items[:3])
            )

    o = levb("gtap7_3x3")
    print("== 3x3 contra la base del GDX de GAMS")
    for v, sym, scale in (
        ("y", "regY", 1e-6),
        ("qinv", "xi", 1e-6),
        ("rore", "rore", 1.0),
    ):
        g = {k[0]: x for k, x in _gdx_base(sym).items()}
        worst = max(abs(o[v][r] * scale - g[r]) / abs(g[r]) for r in g)
        print(f"  {v} <-> {sym}: {len(g)} regiones, max d = {worst:.3e}")
    for sym in ("pi", "pva"):
        g = _gdx_base(sym)
        print(f"  {sym}: base min {min(g.values()):.12g} max {max(g.values()):.12g}")


if __name__ == "__main__":
    main()
