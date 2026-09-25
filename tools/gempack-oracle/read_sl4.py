#!/usr/bin/env python3
"""Extrae variables del .sl4 de GEMPACK por NOMBRE, sin adivinar posiciones.

El .sl4 guarda los resultados en tres arrays planos —CUMS (cambios %
acumulados), LEVB/LEVA (niveles pre/post)— y el mapa para indexarlos en
headers aparte:

    VARS  nombres de las 263 variables
    VNCP  cuantas componentes tiene cada una
    PCUM  donde empieza cada una dentro de CUMS   (base 1)
    PLEV  idem dentro de LEVB/LEVA, 0 = sin nivel
    VCSP/VCNI/VCSN  sobre que sets va cada variable
    STNM/SSZ/STEL   nombres, tamanos y elementos de los sets

Los indices van en orden Fortran (el primero varia mas rapido).

Uso:
    python read_sl4.py <archivo.sl4>                 # lista variables
    python read_sl4.py <archivo.sl4> --var qpa       # una variable, etiquetada
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


class SL4:
    def __init__(self, path: Path):
        from equilibria.babel.har import read_har
        self.d = read_har(path)
        self.vars = self._s("VARS")
        self.ncomp = self._i("VNCP")
        self.pcum = self._i("PCUM")
        self.plev = self._i("PLEV")
        self.cums = self._f("CUMS")
        self.levb = self._f("LEVB")
        self.leva = self._f("LEVA")
        self.setnm = self._s("STNM")
        self.setsz = self._i("SSZ")
        self.stel = self._s("STEL")
        self.vcsp = self._i("VCSP")
        self.vcni = self._i("VCNI")
        self.vcsn = self._i("VCSN")

    def _s(self, k): return [str(x).strip() for x in np.asarray(self.d[k].array).ravel()]
    def _i(self, k): return np.asarray(self.d[k].array, dtype=int).ravel()
    def _f(self, k): return np.asarray(self.d[k].array, dtype=float).ravel()

    def _idx(self, name: str) -> int:
        low = name.lower()
        for j, v in enumerate(self.vars):
            if v.lower() == low:
                return j
        raise KeyError(f"variable {name!r} no esta en el .sl4")

    def sets_of(self, name: str) -> list[tuple[str, list[str]]]:
        """Los sets sobre los que va la variable, con sus elementos."""
        j = self._idx(name)
        nums = self.vcsn[self.vcsp[j] - 1 : self.vcsp[j] - 1 + self.vcni[j]]
        out = []
        for sn in nums:
            size = int(self.setsz[sn - 1])
            start = int(self.setsz[: sn - 1].sum())
            out.append((self.setnm[sn - 1], self.stel[start : start + size]))
        return out

    def pct(self, name: str) -> np.ndarray:
        """Cambios % acumulados (lo que imprime el libro)."""
        j = self._idx(name)
        if self.pcum[j] == 0:
            raise KeyError(f"{name}: sin resultados en CUMS")
        return self.cums[self.pcum[j] - 1 : self.pcum[j] - 1 + self.ncomp[j]]

    def levels(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(pre, post) en NIVELES. KeyError si la variable no tiene nivel."""
        j = self._idx(name)
        if self.plev[j] == 0:
            raise KeyError(f"{name}: no es variable de niveles (plev=0)")
        a, n = self.plev[j] - 1, self.ncomp[j]
        return self.levb[a : a + n], self.leva[a : a + n]

    def labels(self, name: str) -> list[str]:
        """Etiquetas en orden Fortran: el primer indice varia mas rapido."""
        sets = self.sets_of(name)
        if not sets:
            return [""]
        out = [""]
        for _, elems in sets:
            out = [f"{e}|{o}" if o else e for o in out for e in elems] if False else [
                (f"{o}|{e}" if o else e) for e in elems for o in out
            ]
        # rehacer en orden Fortran explicito
        import itertools
        cols = [e for _, e in sets]
        out = ["|".join(t[::-1]) for t in itertools.product(*cols[::-1])]
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sl4", type=Path)
    ap.add_argument("--var")
    a = ap.parse_args()
    f = SL4(a.sl4)
    if not a.var:
        print(f"{len(f.vars)} variables en {a.sl4.name}")
        print(f"{'variable':<14}{'ncomp':>6}{'%':>4}{'niv':>5}  sets")
        for j, v in enumerate(f.vars):
            sets = ",".join(s for s, _ in f.sets_of(v))
            print(f"{v:<14}{f.ncomp[j]:>6}{'si' if f.pcum[j] else '-':>4}"
                  f"{'si' if f.plev[j] else '-':>5}  {sets}")
        return 0
    labs = f.labels(a.var)
    pct = f.pct(a.var)
    print(f"{a.var}  ({len(pct)} componentes)   sets: "
          f"{', '.join(s for s,_ in f.sets_of(a.var))}")
    try:
        pre, post = f.levels(a.var)
        print(f"{'clave':<22}{'pre':>14}{'post':>14}{'%':>10}")
        for k, lab in enumerate(labs[: len(pct)]):
            print(f"{lab:<22}{pre[k]:>14.6f}{post[k]:>14.6f}{pct[k]:>10.4f}")
    except KeyError:
        print(f"{'clave':<22}{'% cambio':>10}   (sin niveles)")
        for k, lab in enumerate(labs[: len(pct)]):
            print(f"{lab:<22}{pct[k]:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
