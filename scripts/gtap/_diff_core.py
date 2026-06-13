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
    "xd": "xda",             # GAMS xd[r,i,aa] (bilateral) → Python xda
    "xm": "xma",             # GAMS xm[r,i,aa] (bilateral) → Python xma
    "ytaxind": "ytax_ind",   # GAMS camelCase ytaxInd → Python ytax_ind
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

    def _xd(key):
        if len(key) != 3:
            return None
        return _value_or_zero(model, "xda", key)
    out["xd"] = _DerivedVar("xd(=xda)", _xd)

    def _xm(key):
        if len(key) != 3:
            return None
        return _value_or_zero(model, "xma", key)
    out["xm"] = _DerivedVar("xm(=xma)", _xm)

    def _xi(key):
        if len(key) != 1:
            return None
        return _value_or_zero(model, "xiagg", key[0])
    out["xi"] = _DerivedVar("xi(=xiagg)", _xi)

    def _pg(r):
        try:
            sigmag = 1.01
            expo = 1.0 - sigmag
            terms = 0.0
            for i in model.i:
                gs = _pyo_value(model.g_share[r, i])
                if gs > 0.0:
                    pa = _pyo_value(model.pa[r, i, "gov"])
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
            yg = _pyo_value(model.yg[key[0]])
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


def _strip_gtap_prefix(s: str) -> str:
    """Strip a single leading GAMS set-type prefix (a_, c_, f_) — lowercase input."""
    sl = s.lower()
    for pfx in ("a_", "c_", "f_"):
        if sl.startswith(pfx):
            return sl[len(pfx):]
    return sl


def _norm_key(key: tuple) -> tuple:
    """Normalise every element: strip GAMS prefixes and lowercase for comparison."""
    return tuple(_strip_gtap_prefix(str(k)) for k in key)


def get_py_var_value(py_var, key: tuple) -> float | None:
    """Tolerates index-shape mismatches: drops a singleton 'hhd' dim, permutes last two dims.

    Also strips GAMS set-type prefixes (a_, c_, f_) from key elements so that
    GAMS 'a_Food' matches Python 'Food'.
    """
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
            # Prefix-stripped fallback for singleton keys
            norm_k0 = _strip_gtap_prefix(str(k0))
            for kk in py_var:
                if _strip_gtap_prefix(str(kk)) == norm_k0:
                    return float(value(py_var[kk]))
            return None
        v = _try_index(py_var, key)
        if v is not None:
            return v
        # Prefix-stripping fallback: normalise GAMS key then direct lookup
        if not is_derived:
            norm = _norm_key(key)
            if norm != key:
                v = _try_index(py_var, norm)
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
        # Full fuzzy scan as last resort (handles mixed-case + prefix differences)
        if not is_derived:
            norm = _norm_key(key)
            for kk in py_var:
                py_key = kk if isinstance(kk, tuple) else (kk,)
                if len(py_key) == len(norm) and _norm_key(py_key) == norm:
                    return float(value(py_var[kk]))
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
        if p_val is None and not body:
            try:
                from pyomo.core import value
                p_val = float(value(model_py))
            except Exception:
                p_val = None
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


def list_populated_params(gdx_path: Path) -> list[str]:
    """Return GAMS symbol names that are Parameters (not Variables) with records > 0."""
    out = subprocess.run(
        [GDXDUMP, str(gdx_path), "Symbols"],
        capture_output=True, text=True, check=True,
    ).stdout
    names = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[3] != "Par":
            continue
        try:
            n_records = int(parts[4])
        except ValueError:
            continue
        if n_records > 0:
            names.append(parts[1])
    return names


def _find_py_param(model, name: str):
    """Locate a Pyomo Param by name (exact, lower, alias).  Returns (param, py_name) or (None, None)."""
    from pyomo.environ import Param
    # Exact
    v = getattr(model, name, None)
    if v is not None and hasattr(v, "__class__") and "Param" in type(v).__name__:
        return v, name
    # Lowercase
    v = getattr(model, name.lower(), None)
    if v is not None and "Param" in type(v).__name__:
        return v, name.lower()
    # Known aliases: GAMS alphaa → Python alphaa_hhd
    _PARAM_ALIAS = {
        "alphaa": "alphaa_hhd",
        "auh":    "auh",
        "betap":  "betap",
        "betag":  "betag",
        "betas":  "betas",
        "io":     None,   # Python has io_param in calibrated, not in model
        "af":     None,
        "and":    None,
        "ava":    None,
    }
    aliased = _PARAM_ALIAS.get(name.lower())
    if aliased is not None:
        v = getattr(model, aliased, None)
        if v is not None:
            return v, aliased
    if aliased is None and name.lower() in _PARAM_ALIAS:
        return None, None
    # Case-insensitive scan over all Params
    target = name.lower()
    for comp in model.component_objects(Param, active=True):
        if comp.name.lower() == target:
            return comp, comp.name
    return None, None


def _norm_key_elem(s: str) -> str:
    """Normalize a GAMS index element for fuzzy matching.

    Strips leading commodity prefixes like 'c_', 'a_', 'f_' that GAMS may add
    but Python drops, and lowercases for comparison.
    """
    s = s.lower()
    for pfx in ("c_", "a_", "f_"):
        if s.startswith(pfx):
            s = s[len(pfx):]
            break
    return s


def _keys_match(gams_key: tuple, py_key: tuple) -> bool:
    """True if both tuples have the same length and each element normalizes equal."""
    if len(gams_key) != len(py_key):
        return False
    return all(_norm_key_elem(str(g)) == _norm_key_elem(str(p))
               for g, p in zip(gams_key, py_key))


def _get_py_param_value(py_param, key: tuple):
    """Read a scalar value from a Pyomo Param, tolerating index mismatches like get_py_var_value.

    Handles commodity prefix differences (GAMS 'c_Food' vs Python 'Food') via
    fuzzy key normalisation.
    """
    from pyomo.core import value
    try:
        if not key:
            return float(value(py_param))
        if len(key) == 1:
            k0 = key[0]
            try:
                return float(value(py_param[k0]))
            except Exception:
                pass
            norm_k0 = _norm_key_elem(k0)
            for kk in py_param:
                if _norm_key_elem(str(kk)) == norm_k0:
                    return float(value(py_param[kk]))
            return None
        # Try direct
        try:
            return float(value(py_param[key]))
        except Exception:
            pass
        # Fuzzy scan: normalise both sides
        for kk in py_param:
            py_key = kk if isinstance(kk, tuple) else (kk,)
            if _keys_match(key, py_key):
                return float(value(py_param[kk]))
        # Drop 'hhd' dim then retry
        for i, k in enumerate(key):
            if k in _DROPPABLE_HHD:
                shrunk = key[:i] + key[i+1:]
                v = _get_py_param_value(py_param, shrunk)
                if v is not None:
                    return v
        # Permute last two dims
        if len(key) >= 2:
            permuted = key[:-2] + (key[-1], key[-2])
            try:
                return float(value(py_param[permuted]))
            except Exception:
                pass
        return None
    except Exception:
        return None


def compare_phase_param(py_param, gams_all: dict, t_label: str, tol_rel: float, tol_abs: float,
                        filter_aa: str | None = None):
    """Like compare_phase but reads from a Pyomo Param instead of a Var.

    filter_aa: if set, only compare GAMS cells where the 'aa' dimension equals this value.
    e.g. filter_aa='hhd' restricts alphaa(r,i,aa,t) to the household agent only.
    """
    n_total = n_match = n_diverge = n_missing = 0
    max_abs = max_rel = 0.0
    worst = None
    for full_key, g_val in gams_all.items():
        body, t = split_t(full_key)
        if t is not None and t != t_label:
            continue
        # Filter by agent dimension if requested
        if filter_aa is not None and filter_aa not in body:
            continue
        # Drop the aa dimension from body before looking up in Python param
        body_py = tuple(k for k in body if k != filter_aa) if filter_aa is not None else body
        n_total += 1
        p_val = _get_py_param_value(py_param, body_py)
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


# GAMS parameter symbols to compare against Python Params in altertax diff.
ALTERTAX_PARAM_NAMES = ["alphaa", "auh", "betap", "betag", "betas"]

# For parameters with an agent/household dimension in GAMS, filter to the
# relevant agent and drop that dimension before looking up in Python Param.
# e.g. GAMS alphaa(r,i,aa,t) → filter aa='hhd', Python alphaa_hhd(r,i)
_PARAM_FILTER_AA = {
    "alphaa": "hhd",
    "auh":    "hhd",
}


def diff_params_rows(
    *, dataset: str, phase: str, param_names: list[str],
    gdx_path: Path, model_py, tol_rel: float, tol_abs: float,
    residual: float, git_sha: str, generated_at: str,
    solve_seconds: float = 0.0,
) -> tuple[list[dict], dict]:
    """Compare GAMS parameters (stored as Vars in GDX) against Python Params.

    Returns same (rows, agg) format as diff_phase_rows so callers can merge the
    two tables.  Rows have py_var prefixed with 'param:' to distinguish them.
    """
    rows: list[dict] = []
    agg = {"vars_total": 0, "vars_match_all": 0, "vars_partial": 0, "vars_no_py": 0,
           "cells_total": 0, "cells_match": 0, "cells_diverge": 0, "cells_missing": 0}
    for name in param_names:
        agg["vars_total"] += 1
        gams_all = gams_levels(gdx_path, name)
        if not gams_all:
            # Try as parameter symbol (gdxdump treats it differently)
            res = subprocess.run(
                [GDXDUMP, str(gdx_path), "Format=csv", f"Symb={name}"],
                capture_output=True, text=True, check=False,
            )
            if res.returncode != 0 or not res.stdout.strip():
                continue
            import csv as _csv
            reader = _csv.reader(res.stdout.splitlines())
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                *keys, val = row
                keys = tuple(k.strip('"') for k in keys)
                try:
                    gams_all[keys] = float(val)
                except ValueError:
                    pass
        if not gams_all:
            continue
        py_param, py_name = _find_py_param(model_py, name)
        if py_param is None:
            n_t = sum(1 for k in gams_all if split_t(k)[1] == phase)
            if n_t == 0:
                n_t = len(gams_all)
            agg["vars_no_py"] += 1
            agg["cells_total"] += n_t
            agg["cells_missing"] += n_t
            rows.append({
                "dataset": dataset, "phase": phase, "var": f"[par]{name}", "py_var": "",
                "cells": n_t, "match": 0, "diverge": 0, "missing": n_t,
                "max_abs_err": "", "max_rel_err": "",
                "residual": f"{residual:.6e}",
                "solve_seconds": f"{solve_seconds:.3f}",
                "git_sha": git_sha, "generated_at": generated_at,
            })
            continue
        filter_aa = _PARAM_FILTER_AA.get(name.lower())
        s = compare_phase_param(py_param, gams_all, phase, tol_rel=tol_rel, tol_abs=tol_abs,
                                filter_aa=filter_aa)
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
            "dataset": dataset, "phase": phase, "var": f"[par]{name}", "py_var": f"param:{py_name}",
            "cells": s["n_total"], "match": s["n_match"],
            "diverge": s["n_diverge"], "missing": s["n_missing"],
            "max_abs_err": f"{s['max_abs']:.6e}", "max_rel_err": f"{s['max_rel']:.6e}",
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
