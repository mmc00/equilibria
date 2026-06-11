"""Minimal AMPL .nl file parser for GTAP model comparison.

Reads the .nl binary-text format (text mode only — Pyomo writes text .nl).
Also reads optional .col (variable names) and .row (constraint names) sidecars.

Returns a NLModel dataclass with structural and linear-coefficient data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NLModel:
    n_vars: int = 0
    n_cons: int = 0
    n_nonzeros: int = 0
    var_names: list[str] = field(default_factory=list)   # index → name
    con_names: list[str] = field(default_factory=list)   # index → name
    bounds: list[tuple[float, float]] = field(default_factory=list)  # (lb, ub) per var
    J: dict[int, dict[int, float]] = field(default_factory=dict)     # con_idx → {var_idx → coeff}
    has_nonlinear_body: set[int] = field(default_factory=set)        # con indices with non-trivial C-block

    def var_name(self, idx: int) -> str:
        return self.var_names[idx] if idx < len(self.var_names) else f"x[{idx}]"

    def con_name(self, idx: int) -> str:
        return self.con_names[idx] if idx < len(self.con_names) else f"c[{idx}]"


def parse_nl(nl_path: str | Path) -> NLModel:
    """Parse an AMPL .nl file (text mode). Reads sidecar .col/.row if present."""
    nl_path = Path(nl_path)
    m = NLModel()

    # Load sidecar names
    col_path = nl_path.with_suffix(".col")
    row_path = nl_path.with_suffix(".row")
    if col_path.exists():
        m.var_names = col_path.read_text().splitlines()
    if row_path.exists():
        m.con_names = row_path.read_text().splitlines()

    lines = nl_path.read_text().splitlines()

    _INF = 1e30
    header_parsed = False
    cur_section: str | None = None
    cur_con_idx: int | None = None
    cur_con_entries: dict[int, float] = {}
    cur_c_idx: int | None = None        # current C-block con index
    cur_c_first_line: bool = False       # True if we haven't seen any data line yet

    # .nl format: line 0 is format marker ("g3 1 1 0" etc.)
    # line 1 is "n_vars n_cons n_nonzeros n_ranges n_eqn ..."
    # Subsequent lines are section markers (single letter + args) or data rows.
    # Section markers: b (bounds), J (jacobian row), G (obj gradient), x (init),
    #                  r (constraint rhs/range), O (obj), C (nonlinear con body), etc.

    SECTION_LETTERS = set("bJGCxXOrOkd")

    for lineno, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue

        # Header parsing: line 0 = format marker, line 1 = problem dimensions
        if lineno == 0:
            continue  # format marker — skip

        if not header_parsed:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    m.n_vars = int(parts[0])
                    m.n_cons = int(parts[1])
                    if len(parts) >= 3:
                        m.n_nonzeros = int(parts[2])
                    header_parsed = True
                except ValueError:
                    pass
            continue

        # Check for section marker: a single letter followed by optional digits/spaces
        first_char = line[0]
        if first_char in SECTION_LETTERS:
            rest = line[1:].strip()
            # Distinguish "J42 5" (section marker) from data lines starting with letters
            # Section markers have the letter immediately followed by digits or end of line
            is_section = (len(line) == 1 or rest == "" or
                          (len(rest) > 0 and (rest[0].isdigit() or rest[0] == '-')))

            if is_section and first_char in "bJGCxXOr":
                # Flush previous J block before switching sections
                if cur_section == "J" and cur_con_idx is not None and cur_con_entries:
                    m.J[cur_con_idx] = dict(cur_con_entries)
                    cur_con_entries = {}
                    cur_con_idx = None

                cur_section = first_char

                if first_char == "J" and rest:
                    parts = rest.split()
                    if len(parts) >= 2:
                        cur_con_idx = int(parts[0])
                        cur_con_entries = {}
                elif first_char == "C" and rest:
                    # C<idx> marks start of nonlinear body for constraint idx.
                    # Bodies that are just "n0" (constant zero) are trivial → skip.
                    try:
                        cur_c_idx = int(rest.split()[0])
                        cur_c_first_line = True
                    except (ValueError, IndexError):
                        cur_c_idx = None
                        cur_c_first_line = False
                elif first_char == "b":
                    m.bounds = []
                continue

        # Data rows
        if cur_section == "b":
            parts = line.split()
            if parts:
                try:
                    btype = int(parts[0])
                    if btype == 0 and len(parts) >= 3:
                        m.bounds.append((float(parts[1]), float(parts[2])))
                    elif btype == 1 and len(parts) >= 2:
                        m.bounds.append((float(parts[1]), _INF))
                    elif btype == 2 and len(parts) >= 2:
                        m.bounds.append((-_INF, float(parts[1])))
                    elif btype == 3:
                        m.bounds.append((-_INF, _INF))
                    elif btype == 4 and len(parts) >= 2:
                        val = float(parts[1])
                        m.bounds.append((val, val))
                except (ValueError, IndexError):
                    pass

        elif cur_section == "C" and cur_c_idx is not None:
            # First data line of a C-block: if it's exactly "n0", the body is trivially zero.
            if cur_c_first_line:
                cur_c_first_line = False
                if line.strip() != "n0":
                    m.has_nonlinear_body.add(cur_c_idx)

        elif cur_section == "J" and cur_con_idx is not None:
            # Check if this is a new J sub-section header: "J<con_idx> <nnz>"
            if line[0] == "J" and len(line) > 1 and line[1].isdigit():
                # Flush current
                if cur_con_entries:
                    m.J[cur_con_idx] = dict(cur_con_entries)
                    cur_con_entries = {}
                parts = line[1:].split()
                if len(parts) >= 2:
                    cur_con_idx = int(parts[0])
            else:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        var_idx = int(parts[0])
                        coeff = float(parts[1])
                        cur_con_entries[var_idx] = coeff
                    except ValueError:
                        pass

    # Flush final J block
    if cur_section == "J" and cur_con_idx is not None and cur_con_entries:
        m.J[cur_con_idx] = dict(cur_con_entries)

    return m
