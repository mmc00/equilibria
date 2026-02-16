"""
Utilities to load PEP VAL_PAR parameters from .gdx or .xlsx sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values


def _norm_label(label: str) -> str:
    return str(label).strip().lower()


def _map_sector(label: str) -> str:
    return {
        "agr": "agr",
        "ind": "ind",
        "food": "ind",
        "othind": "ind",
        "ser": "ser",
        "adm": "adm",
    }.get(
        _norm_label(label), _norm_label(label)
    )


def _map_commodity(label: str) -> str:
    return {
        "agr": "agr",
        "food": "food",
        "othind": "othind",
        "ser": "ser",
        "adm": "adm",
    }.get(_norm_label(label), _norm_label(label))


def _map_agent(label: str) -> str:
    return {
        "hrp": "hrp",
        "hup": "hup",
        "hrr": "hrr",
        "hur": "hur",
        "firm": "firm",
        "gvt": "gvt",
        "row": "row",
    }.get(_norm_label(label), _norm_label(label))


def _empty_val_par() -> dict[str, Any]:
    return {
        "sigma_KD": {},
        "sigma_LD": {},
        "sigma_VA": {},
        "sigma_XT": {},
        "sigma_M": {},
        "sigma_XD": {},
        "sigma_X": {},
        "frisch": {},
        "sigma_Y": {},
        "sh0O": {},
        "tr0O": {},
        "ttdf0O": {},
        "ttdh0O": {},
    }


def _load_from_gdx(path: Path) -> dict[str, Any]:
    data = _empty_val_par()
    gdx = read_gdx(path)

    parj = read_parameter_values(gdx, "PARJ")
    for (j, par), val in parj.items():
        jj = _map_sector(j)
        pname = _norm_label(par)
        if pname == "sigma_kd":
            data["sigma_KD"][jj] = float(val)
        elif pname == "sigma_ld":
            data["sigma_LD"][jj] = float(val)
        elif pname == "sigma_va":
            data["sigma_VA"][jj] = float(val)
        elif pname == "sigma_xt":
            data["sigma_XT"][jj] = float(val)

    pari = read_parameter_values(gdx, "PARI")
    for (i, par), val in pari.items():
        ii = _map_commodity(i)
        pname = _norm_label(par)
        if pname == "sigma_m":
            data["sigma_M"][ii] = float(val)
        elif pname == "sigma_xd":
            data["sigma_XD"][ii] = float(val)

    parji = read_parameter_values(gdx, "PARJI")
    for (j, i), val in parji.items():
        data["sigma_X"][(_map_sector(j), _map_commodity(i))] = float(val)

    parag = read_parameter_values(gdx, "PARAG")
    for (row, col), val in parag.items():
        r = _norm_label(row)
        c = _map_agent(col)
        if r == "frisch" and c in {"hrp", "hup", "hrr", "hur"}:
            data["frisch"][c] = float(val)
        elif r in {"agr", "food", "othind", "ser", "adm"} and c in {"hrp", "hup", "hrr", "hur"}:
            data["sigma_Y"][(r, c)] = float(val)
        elif r == "sh0o" and c in {"hrp", "hup", "hrr", "hur"}:
            data["sh0O"][c] = float(val)
        elif r == "tr0o" and c in {"hrp", "hup", "hrr", "hur"}:
            data["tr0O"][c] = float(val)
        elif r == "ttdf0o" and c == "firm":
            data["ttdf0O"][c] = float(val)
        elif r == "ttdh0o" and c in {"hrp", "hup", "hrr", "hur"}:
            data["ttdh0O"][c] = float(val)

    return data


def _load_from_xlsx(path: Path) -> dict[str, Any]:
    data = _empty_val_par()
    df = pd.read_excel(path, sheet_name="PAR", header=None)

    def _find_section_row(tokens: tuple[str, ...]) -> int | None:
        for r in range(len(df)):
            c0 = df.iloc[r, 0] if df.shape[1] > 0 else None
            txt = _norm_label(c0) if pd.notna(c0) else ""
            if not txt:
                continue
            if any(tok in txt for tok in tokens):
                return r
        return None

    def _find_next_section_row(start_row: int) -> int:
        for r in range(start_row + 1, len(df)):
            c0 = df.iloc[r, 0] if df.shape[1] > 0 else None
            txt = _norm_label(c0) if pd.notna(c0) else ""
            if (
                "parameters indexed in" in txt
                or txt in {"parj", "pari", "parji", "parag"}
            ):
                return r
        return len(df)

    def _parse_table(section_row: int) -> tuple[list[str], list[pd.Series]]:
        header_row = section_row + 1
        end_row = _find_next_section_row(section_row)
        headers: list[str] = []
        for c in range(1, df.shape[1]):
            v = df.iloc[header_row, c]
            if pd.isna(v):
                continue
            headers.append(str(v).strip())
        rows: list[pd.Series] = []
        for r in range(header_row + 1, end_row):
            row_key = df.iloc[r, 0]
            if pd.isna(row_key):
                continue
            rows.append(df.iloc[r, :])
        return headers, rows

    # Prefer dynamic detection (works for both original and connect layouts)
    sec_j = _find_section_row(("parameters indexed in j", "parj"))
    sec_i = _find_section_row(("parameters indexed in i", "pari"))
    sec_ji = _find_section_row(("parameters indexed in j,i", "parji"))
    sec_ag = _find_section_row(("parameters indexed in ag", "parag"))

    if all(x is not None for x in (sec_j, sec_i, sec_ji, sec_ag)):
        j_cols, j_rows = _parse_table(sec_j)
        for row in j_rows:
            j = _map_sector(row.iloc[0])
            for ci, col_name in enumerate(j_cols, start=1):
                if ci >= len(row):
                    continue
                v = row.iloc[ci]
                if pd.isna(v):
                    continue
                pname = _norm_label(col_name)
                if pname == "sigma_kd":
                    data["sigma_KD"][j] = float(v)
                elif pname == "sigma_ld":
                    data["sigma_LD"][j] = float(v)
                elif pname == "sigma_va":
                    data["sigma_VA"][j] = float(v)
                elif pname == "sigma_xt":
                    data["sigma_XT"][j] = float(v)

        i_cols, i_rows = _parse_table(sec_i)
        for row in i_rows:
            i = _map_commodity(row.iloc[0])
            for ci, col_name in enumerate(i_cols, start=1):
                if ci >= len(row):
                    continue
                v = row.iloc[ci]
                if pd.isna(v):
                    continue
                pname = _norm_label(col_name)
                if pname == "sigma_m":
                    data["sigma_M"][i] = float(v)
                elif pname == "sigma_xd":
                    data["sigma_XD"][i] = float(v)

        ji_cols, ji_rows = _parse_table(sec_ji)
        i_headers = [_map_commodity(x) for x in ji_cols]
        for row in ji_rows:
            j = _map_sector(row.iloc[0])
            for ci, i in enumerate(i_headers, start=1):
                if ci >= len(row):
                    continue
                v = row.iloc[ci]
                if pd.isna(v):
                    continue
                data["sigma_X"][(j, i)] = float(v)

        ag_cols, ag_rows = _parse_table(sec_ag)
        h_headers = [_map_agent(x) for x in ag_cols]
        for row in ag_rows:
            pname = _norm_label(row.iloc[0])
            for ci, h in enumerate(h_headers, start=1):
                if ci >= len(row):
                    continue
                v = row.iloc[ci]
                if pd.isna(v):
                    continue
                if pname == "frisch" and h in {"hrp", "hup", "hrr", "hur"}:
                    data["frisch"][h] = float(v)
                elif pname in {"agr", "food", "othind", "ser", "adm"} and h in {"hrp", "hup", "hrr", "hur"}:
                    data["sigma_Y"][(pname, h)] = float(v)
                elif pname == "sh0o" and h in {"hrp", "hup", "hrr", "hur"}:
                    data["sh0O"][h] = float(v)
                elif pname == "tr0o" and h in {"hrp", "hup", "hrr", "hur"}:
                    data["tr0O"][h] = float(v)
                elif pname == "ttdf0o" and h == "firm":
                    data["ttdf0O"][h] = float(v)
                elif pname == "ttdh0o" and h in {"hrp", "hup", "hrr", "hur"}:
                    data["ttdh0O"][h] = float(v)

        return data

    # Fallback for known fixed-layout files
    # PARJ range A5:E10
    parj_rows = df.iloc[4:10, 0:5].copy()
    j_cols = [str(x).strip() for x in parj_rows.iloc[0, 1:].tolist() if pd.notna(x)]
    for _, row in parj_rows.iloc[1:].iterrows():
        if pd.isna(row.iloc[0]):
            continue
        j = _map_sector(row.iloc[0])
        for ci, col_name in enumerate(j_cols, start=1):
            v = row.iloc[ci]
            if pd.isna(v):
                continue
            pname = _norm_label(col_name)
            if pname == "sigma_kd":
                data["sigma_KD"][j] = float(v)
            elif pname == "sigma_ld":
                data["sigma_LD"][j] = float(v)
            elif pname == "sigma_va":
                data["sigma_VA"][j] = float(v)
            elif pname == "sigma_xt":
                data["sigma_XT"][j] = float(v)

    # PARI range A13:C18
    pari_rows = df.iloc[12:18, 0:3].copy()
    i_cols = [str(x).strip() for x in pari_rows.iloc[0, 1:].tolist() if pd.notna(x)]
    for _, row in pari_rows.iloc[1:].iterrows():
        if pd.isna(row.iloc[0]):
            continue
        i = _map_commodity(row.iloc[0])
        for ci, col_name in enumerate(i_cols, start=1):
            v = row.iloc[ci]
            if pd.isna(v):
                continue
            pname = _norm_label(col_name)
            if pname == "sigma_m":
                data["sigma_M"][i] = float(v)
            elif pname == "sigma_xd":
                data["sigma_XD"][i] = float(v)

    # PARJI range A21:F26
    parji_rows = df.iloc[20:26, 0:6].copy()
    i_headers = [_map_commodity(x) for x in parji_rows.iloc[0, 1:].tolist() if pd.notna(x)]
    for _, row in parji_rows.iloc[1:].iterrows():
        if pd.isna(row.iloc[0]):
            continue
        j = _map_sector(row.iloc[0])
        for ci, i in enumerate(i_headers, start=1):
            v = row.iloc[ci]
            if pd.isna(v):
                continue
            data["sigma_X"][(j, i)] = float(v)

    # PARAG range A29:F41
    parag_rows = df.iloc[28:41, 0:6].copy()
    h_headers = [_map_agent(x) for x in parag_rows.iloc[0, 1:].tolist() if pd.notna(x)]
    for _, row in parag_rows.iloc[1:].iterrows():
        if pd.isna(row.iloc[0]):
            continue
        pname = _norm_label(row.iloc[0])
        for ci, h in enumerate(h_headers, start=1):
            v = row.iloc[ci]
            if pd.isna(v):
                continue
            if pname == "frisch" and h in {"hrp", "hup", "hrr", "hur"}:
                data["frisch"][h] = float(v)
            elif pname in {"agr", "food", "othind", "ser", "adm"} and h in {"hrp", "hup", "hrr", "hur"}:
                data["sigma_Y"][(pname, h)] = float(v)
            elif pname == "sh0o" and h in {"hrp", "hup", "hrr", "hur"}:
                data["sh0O"][h] = float(v)
            elif pname == "tr0o" and h in {"hrp", "hup", "hrr", "hur"}:
                data["tr0O"][h] = float(v)
            elif pname == "ttdf0o" and h == "firm":
                data["ttdf0O"][h] = float(v)
            elif pname == "ttdh0o" and h in {"hrp", "hup", "hrr", "hur"}:
                data["ttdh0O"][h] = float(v)

    return data


def load_val_par(path: Path | str | None) -> dict[str, Any]:
    """Load VAL_PAR parameters from GDX or XLSX; return empty dict on failure."""
    if path is None:
        return _empty_val_par()
    p = Path(path)
    if not p.exists():
        return _empty_val_par()
    try:
        if p.suffix.lower() == ".gdx":
            return _load_from_gdx(p)
        if p.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            return _load_from_xlsx(p)
    except Exception:
        return _empty_val_par()
    return _empty_val_par()
