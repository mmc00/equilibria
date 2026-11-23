"""The frozen two-photo capture of the Pyomo period preparation (reader only).

The photos were written per solve call of the old Pyomo multiperiod driver, the
oracle for the array port; they are frozen now (MAR-218) and this module only
reads them:

* **A** — the whole multiperiod state at entry of the old Pyomo period recipe for
  period ``t`` (every Var cell of every period: value, fixed, lb, ub; every
  Constraint cell: active). **A2** is the same photo at the end of the preparation.
* **B** — the squared system right before PyomoNLP was built: active rows, free
  columns in paired order, every cell of the active period, and each active row's
  incidence.
* **P** / **S** — the active period's Var cells after the Newton writeback, and the
  solve's stats.
* ``final.pkl`` — every Var cell and row flag after the run (:func:`final_photo`
  takes the same photo of any solved model); ``settled.pkl`` — the dict
  ``calibrate_base`` returns.

Files are ``<n>_<tag>_<period>.pkl``, ``n`` counting the preparations of the run.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VarCells:
    keys: list[tuple]
    value: np.ndarray
    fixed: np.ndarray
    lb: np.ndarray
    ub: np.ndarray


@dataclass
class RowCells:
    keys: list[tuple]
    active: np.ndarray


@dataclass
class Photo:
    tag: str
    n: int
    period: str
    vars: dict[str, VarCells]
    rows: dict[str, RowCells]
    con_names: list[str] = field(default_factory=list)
    var_names: list[str] = field(default_factory=list)
    incidence: list[list[str]] = field(default_factory=list)
    n_deact: int = 0
    n_fixed: int = 0
    forced: list[tuple[str, str, bool]] = field(default_factory=list)


def _key(idx: Any) -> tuple:
    if idx is None:
        return ()
    return tuple(str(i) for i in idx) if isinstance(idx, tuple) else (str(idx),)


def _num(x: Any, none: float) -> float:
    return none if x is None else float(x)


def _vars(m: Any, period: str | None) -> dict[str, VarCells]:
    from pyomo.environ import Var

    out = {}
    for comp in m.component_objects(Var, descend_into=True):
        keys, val, fx, lb, ub = [], [], [], [], []
        for idx in comp:
            k = _key(idx)
            if period is not None and (not k or k[-1] != period):
                continue
            vd = comp[idx]
            keys.append(k)
            val.append(_num(vd.value, np.nan))
            fx.append(bool(vd.fixed))
            lb.append(_num(vd.lb, -np.inf))
            ub.append(_num(vd.ub, np.inf))
        if keys:
            out[comp.name] = VarCells(
                keys, np.array(val), np.array(fx, bool), np.array(lb), np.array(ub)
            )
    return out


def _rows(m: Any) -> dict[str, RowCells]:
    from pyomo.environ import Constraint

    out = {}
    for comp in m.component_objects(Constraint, descend_into=True):
        keys = [_key(idx) for idx in comp]
        act = np.array([comp[idx].active and comp.active for idx in comp], bool)
        out[comp.name] = RowCells(keys, act)
    return out


def load(path: Path) -> Photo:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def pairs(d: Path) -> list[tuple[Photo, Photo]]:
    """(A, B) per solve call, in call order (an A with no B — a skipped solve — is
    dropped)."""
    d = Path(d)
    out = []
    for b_path in sorted(d.glob("*_B_*.pkl")):
        b = load(b_path)
        a_path = d / f"{b.n:03d}_A_{b.period}.pkl"
        out.append((load(a_path), b))
    return out


def load_a2(d: Path, a: Photo) -> Photo | None:
    path = Path(d) / f"{a.n:03d}_A2_{a.period}.pkl"
    return load(path) if path.exists() else None


def final_photo(model: Any, n: int = 0) -> Photo:
    """Every Var cell of every period and every row flag of a solved Pyomo model."""
    return Photo("final", n, "all", _vars(model, None), _rows(model))
