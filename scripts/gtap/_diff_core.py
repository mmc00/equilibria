"""Shared GAMS↔Python variable-by-variable diff helpers.

Used by diff_9x10_full.py and diff_nus333_full.py to walk every populated
Var symbol in a GAMS COMP-style GDX, locate the matching Pyomo Var, align
indices (peeling the GAMS time axis t∈{base,check,shock}), and report
match/diverge/missing counts plus worst-cell stats. Also writes a long-form
benchmark CSV consumed by the Sphinx docs site.

Includes the alias / derived-var / index-shape-tolerant lookup machinery
needed for full var-by-var parity (e.g. GAMS xa(r,i,aa) ↔ Python xaa,
GAMS pp(r,a,i) ↔ Python pp_rai, GAMS xi(r) = Python xiagg(r)).
"""
from __future__ import annotations
import csv
import subprocess
from pathlib import Path
from typing import Iterable

GDXDUMP = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump"
T_LABELS = {"base", "check", "shock"}
_DROPPABLE_HHD = {"hhd"}

CSV_FIELDS = [
    "dataset", "phase", "var", "py_var",
    "cells", "match", "diverge", "missing",
    "max_abs_err", "max_rel_err",
    "residual", "solve_seconds", "git_sha", "generated_at",
]

# GAMS → Python name aliases.
_NAME_ALIAS = {
    "pp": "pp_rai",
    "xa": "xaa",
}


class _DerivedVar:
    """Pseudo-Var: callable mapping (key)→value, behaves enough like a Pyomo Var
    for get_py_var_value to use it transparently."""

    def __init__(self, name, getter):
        self.name = name
        self._getter = getter

    def __contains__(self, key):
        try:
            return self._getter(key) is not None
        except Exception:
            return False

    def __getitem__(self, key):
        v = self._getter(key)
        if v is None:
            raise KeyError(key)
        return _DerivedScalar(v)

    def __iter__(self):
        return iter(())


class _DerivedScalar:
    def __init__(self, v):
        self._v = v

    @property
    def value(self):
        return self._v


def _value_or_zero(model, var_name, idx):
    v = getattr(model, var_name, None)
    if v is None:
        return None
    try:
        return float(v[idx].value if hasattr(v[idx], "value") else v[idx])
    except Exception:
        return None


def build_derived(model) -> dict:
    """Build derived 'GAMS-like' views for variables that don't have a direct Var."""
    out = {}
    from pyomo.core import value as _pyo_value

    # Phase 2.2+: xda/xma carry t as last index. Callers (compare_phase) always
    # append t before looking up derived vars, so these getters are 4-tuple only.
    def _xd(key):
        if len(key) == 4:
            return _value_or_zero(model, "xda", key)
        return None
    out["xd"] = _DerivedVar("xd(=xda)", _xd)

    def _xm(key):
        if len(key) == 4:
            return _value_or_zero(model, "xma", key)
        return None
    out["xm"] = _DerivedVar("xm(=xma)", _xm)

    def _xi(key):
        if len(key) != 1:
            return None
        return _value_or_zero(model, "xiagg", (key[0], "base"))
    out["xi"] = _DerivedVar("xi(=xiagg)", _xi)

    def _pg(r):
        try:
            sigmag = 1.01
            expo = 1.0 - sigmag
            terms = 0.0
            for i in model.i:
                gs = _pyo_value(model.g_share[r, i])
                if gs > 0.0:
                    pa = _pyo_value(model.pa[r, i, "gov", "base"])
                    terms += gs * pa ** expo
            if terms <= 0.0:
                return None
            return terms ** (1.0 / expo)
        except Exception:
            return None

    def _xg(key):
        if len(key) != 1:
            return None
        pg = _pg(key[0])
        if pg is None or pg <= 0.0:
            return None
        try:
            yg = _pyo_value(model.yg[key[0], "base"])
            return yg / pg
        except Exception:
            return None
    out["xg"] = _DerivedVar("xg(=yg/pg)", _xg)

    def _pg_outer(key):
        if len(key) != 1:
            return None
        return _pg(key[0])
    out["pg"] = _DerivedVar("pg(CES gov)", _pg_outer)

    return out


def list_populated_vars(gdx_path: Path) -> list[str]:
    out = subprocess.run(
        [GDXDUMP, str(gdx_path), "Symbols"],
        capture_output=True, text=True, check=True,
    ).stdout
    names = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[3] != "Var":
            continue
        try:
            n_records = int(parts[4])
        except ValueError:
            continue
        if n_records > 0:
            names.append(parts[1])
    return names


def gams_levels(gdx_path: Path, symbol: str) -> dict:
    res = subprocess.run(
        [GDXDUMP, str(gdx_path), "Format=csv", f"Symb={symbol}"],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return {}
    out: dict = {}
    reader = csv.reader(res.stdout.splitlines())
    next(reader, None)
    for row in reader:
        if len(row) < 2:
            continue
        *keys, val = row
        keys = tuple(k.strip('"') for k in keys)
        try:
            out[keys] = float(val)
        except ValueError:
            pass
    return out


def split_t(keys: tuple) -> tuple[tuple, str | None]:
    if keys and keys[-1] in T_LABELS:
        return keys[:-1], keys[-1]
    return keys, None


def _try_index(py_var, idx):
    from pyomo.core import value
    if isinstance(py_var, _DerivedVar):
        try:
            v = py_var._getter(idx)  # noqa: SLF001
            return float(v) if v is not None else None
        except Exception:
            return None
    try:
        if idx in py_var:
            return float(value(py_var[idx]))
    except Exception:
        pass
    try:
        for kk in py_var:
            if isinstance(kk, tuple) and tuple(str(x) for x in kk) == tuple(str(x) for x in idx):
                return float(value(py_var[kk]))
            if not isinstance(kk, tuple) and len(idx) == 1 and str(kk) == str(idx[0]):
                return float(value(py_var[kk]))
    except Exception:
        pass
    return None


def get_py_var_value(py_var, key: tuple) -> float | None:
    """Tolerates index-shape mismatches: drops a singleton 'hhd' dim, permutes last two dims."""
    from pyomo.core import value
    is_derived = isinstance(py_var, _DerivedVar)
    try:
        if not key:
            return float(value(py_var))
        if len(key) == 1 and not is_derived:
            k0 = key[0]
            if k0 in py_var:
                return float(value(py_var[k0]))
            for kk in py_var:
                if str(kk) == k0:
                    return float(value(py_var[kk]))
            return None
        v = _try_index(py_var, key)
        if v is not None:
            return v
        for i, k in enumerate(key):
            if k in _DROPPABLE_HHD:
                shrunk = key[:i] + key[i+1:]
                v = _try_index(py_var, shrunk)
                if v is not None:
                    return v
        if not is_derived and len(key) >= 2:
            permuted = key[:-2] + (key[-1], key[-2])
            v = _try_index(py_var, permuted)
            if v is not None:
                return v
        return None
    except Exception:
        return None


def find_py_var(model, name: str, derived: dict | None = None):
    """Try derived view, alias, literal name, lowercase, case-insensitive scan."""
    if derived is not None and name.lower() in derived:
        d = derived[name.lower()]
        return d, d.name
    aliased = _NAME_ALIAS.get(name.lower())
    if aliased is not None:
        v = getattr(model, aliased, None)
        if v is not None:
            return v, aliased
    v = getattr(model, name, None)
    if v is not None:
        return v, name
    v = getattr(model, name.lower(), None)
    if v is not None:
        return v, name.lower()
    target = name.lower()
    from pyomo.environ import Var
    for comp in model.component_objects(Var, active=True):
        if comp.name.lower() == target:
            return comp, comp.name
    return None, None


def compare_phase(model_py, gams_all: dict, t_label: str, tol_rel: float, tol_abs: float,
                  key_remap=None):
    n_total = n_match = n_diverge = n_missing = 0
    max_abs = max_rel = 0.0
    worst = None
    for full_key, g_val in gams_all.items():
        body, t = split_t(full_key)
        if t is not None and t != t_label:
            continue
        n_total += 1
        body_py = key_remap(body) if (body and key_remap) else body
        p_val = get_py_var_value(model_py, body_py) if body_py else None
        # Phase 2.1+ multi-period: production-block Vars now carry t as last
        # index. GAMS body was already peeled of t by split_t. Try appending
        # t_label back when the bare lookup misses. The Python model may use a
        # different t_set (e.g. only "base") even for shocked phases, so also
        # try common labels.
        if p_val is None and body_py:
            # Prefer the model's actual base-period label (model_py.t0) over a
            # hardcoded "base" so a future t_set rename doesn't silently break
            # parity diffs. Fall back to "base" when t0 isn't exposed.
            try:
                base_label = next(iter(model_py.t0))
            except Exception:
                base_label = "base"
            for t_try in (t_label, base_label):
                p_val = get_py_var_value(model_py, body_py + (t_try,))
                if p_val is not None:
                    break
        if p_val is None and not body:
            try:
                from pyomo.core import value
                p_val = float(value(model_py))
            except Exception:
                # model_py is an IndexedVar (t-indexed scalar) — try base label
                try:
                    base_label = next(iter(model_py.keys()))
                    p_val = float(value(model_py[base_label]))
                except Exception:
                    p_val = None
        # For t-indexed scalar Vars (body empty), try t_label or "base" as index
        if p_val is None and not body:
            from pyomo.core import value as _pv
            for _t in (t_label, "base"):
                try:
                    p_val = float(_pv(model_py[_t]))
                    break
                except Exception:
                    pass
        if p_val is None:
            n_missing += 1
            continue
        d = p_val - g_val
        rel = abs(d) / abs(g_val) if abs(g_val) > 1e-12 else (0.0 if abs(d) < tol_abs else float("inf"))
        if abs(d) <= tol_abs or rel <= tol_rel:
            n_match += 1
        else:
            n_diverge += 1
        if abs(d) > max_abs:
            max_abs = abs(d)
        if rel != float("inf") and rel > max_rel:
            max_rel = rel
        if worst is None or abs(d) > abs(worst[3]):
            worst = (full_key, p_val, g_val, d, rel)
    return {
        "n_total": n_total, "n_match": n_match, "n_diverge": n_diverge,
        "n_missing": n_missing, "max_abs": max_abs, "max_rel": max_rel,
        "worst": worst,
    }


def diff_phase_rows(
    *, dataset: str, phase: str, var_names: Iterable[str],
    gdx_path: Path, model_py, tol_rel: float, tol_abs: float,
    residual: float, git_sha: str, generated_at: str,
    derived: dict | None = None, key_remap=None,
    solve_seconds: float = 0.0,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    agg = {"vars_total": 0, "vars_match_all": 0, "vars_partial": 0, "vars_no_py": 0,
           "cells_total": 0, "cells_match": 0, "cells_diverge": 0, "cells_missing": 0}
    for name in var_names:
        agg["vars_total"] += 1
        gams_all = gams_levels(gdx_path, name)
        if not gams_all:
            continue
        py_var, py_name = find_py_var(model_py, name, derived=derived)
        if py_var is None:
            n_t = sum(1 for k in gams_all if split_t(k)[1] == phase)
            if n_t == 0:
                n_t = len(gams_all)
            agg["vars_no_py"] += 1
            agg["cells_total"] += n_t
            agg["cells_missing"] += n_t
            rows.append({
                "dataset": dataset, "phase": phase, "var": name, "py_var": "",
                "cells": n_t, "match": 0, "diverge": 0, "missing": n_t,
                "max_abs_err": "", "max_rel_err": "",
                "residual": f"{residual:.6e}",
                "solve_seconds": f"{solve_seconds:.3f}",
                "git_sha": git_sha, "generated_at": generated_at,
            })
            continue
        s = compare_phase(py_var, gams_all, phase, tol_rel=tol_rel, tol_abs=tol_abs,
                          key_remap=key_remap)
        agg["cells_total"] += s["n_total"]
        agg["cells_match"] += s["n_match"]
        agg["cells_diverge"] += s["n_diverge"]
        agg["cells_missing"] += s["n_missing"]
        if s["n_total"] == 0:
            continue
        if s["n_diverge"] == 0 and s["n_missing"] == 0:
            agg["vars_match_all"] += 1
        else:
            agg["vars_partial"] += 1
        rows.append({
            "dataset": dataset, "phase": phase, "var": name, "py_var": py_name or "",
            "cells": s["n_total"], "match": s["n_match"],
            "diverge": s["n_diverge"], "missing": s["n_missing"],
            "max_abs_err": f"{s['max_abs']:.6e}", "max_rel_err": f"{s['max_rel']:.6e}",
            "residual": f"{residual:.6e}",
            "solve_seconds": f"{solve_seconds:.3f}",
            "git_sha": git_sha, "generated_at": generated_at,
        })
    rows.append({
        "dataset": dataset, "phase": phase, "var": "__SUMMARY__", "py_var": "",
        "cells": agg["cells_total"], "match": agg["cells_match"],
        "diverge": agg["cells_diverge"], "missing": agg["cells_missing"],
        "max_abs_err": "", "max_rel_err": "",
        "residual": f"{residual:.6e}",
        "solve_seconds": f"{solve_seconds:.3f}",
        "git_sha": git_sha, "generated_at": generated_at,
    })
    return rows, agg


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def git_short_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"
