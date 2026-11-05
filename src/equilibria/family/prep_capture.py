"""Two-photo capture of the Pyomo period preparation, the oracle for its array port.

Per solve call of the multiperiod driver:

* **A** — the whole multiperiod state at entry of ``PeriodPreparer.prepare(m, t)``
  (every Var cell of every period: value, fixed, lb, ub; every Constraint cell:
  active). **A2** is the same photo at the end of ``prepare``.
* **B** — the squared system right before PyomoNLP is built (``path_capi`` after
  its SOLVE_NLP deactivate/fix loop): active rows, free columns in paired order,
  every cell of the active period, and each active row's incidence.

Opt-in with ``EQUILIBRIA_FAMILY_PREP_CAPTURE=<dir>``; unset, every call returns at
once. Files are ``<n>_<tag>_<period>.pkl`` with ``n`` counting ``prepare`` calls in
this process, so the settle's check and the main check do not collide.
"""

from __future__ import annotations

import os
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ENV = "EQUILIBRIA_FAMILY_PREP_CAPTURE"
_state = {"n": 0}


def _dir() -> Path | None:
    d = os.environ.get(ENV)
    return Path(d) if d else None


def reset() -> None:
    _state["n"] = 0


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


def _write(photo: Photo) -> None:
    d = _dir()
    assert d is not None
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{photo.n:03d}_{photo.tag}_{photo.period}.pkl"
    with path.open("wb") as fh:
        pickle.dump(photo, fh, protocol=5)


def capture_state(m: Any, period: str, tag: str = "A") -> None:
    """Photo A (``tag="A"``, starts a new call) or A2 (end of the same ``prepare``)."""
    if _dir() is None:
        return
    if tag == "A":
        _state["n"] += 1
    _write(Photo(tag, _state["n"], period, _vars(m, None), _rows(m)))


def capture_squared(
    model: Any, constraints: list, free_variables: list, n_deact: int, n_fixed: int
) -> None:
    """Photo B: the squared system PyomoNLP is about to receive."""
    if _dir() is None:
        return
    from pyomo.core.expr.visitor import identify_variables

    periods = Counter(_key(c.index())[-1] for c in constraints if c.index() is not None)
    period = periods.most_common(1)[0][0]
    _write(
        Photo(
            "B",
            _state["n"],
            period,
            _vars(model, period),
            _rows(model),
            con_names=[c.name for c in constraints],
            var_names=[v.name for v in free_variables],
            incidence=[
                [v.name for v in identify_variables(c.body, include_fixed=True)]
                for c in constraints
            ],
            n_deact=int(n_deact),
            n_fixed=int(n_fixed),
        )
    )


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
