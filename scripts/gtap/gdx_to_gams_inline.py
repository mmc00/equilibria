"""Convert a GDX file to inline GAMS set/parameter declarations.

Mirrors what gtap_9x10_complete_inline.gms does for the 9x10 dataset:
emit each Set / Parameter as literal GAMS so the resulting .gms is
self-contained and NEOS can run it without a separate GDX upload.
"""
from __future__ import annotations
import argparse
from pathlib import Path

GAMS_SYS = "/Library/Frameworks/GAMS.framework/Versions/48/Resources"


def _domain_str(sym) -> str:
    """Return '(parent1,parent2,...)' if sym has a typed domain, else ''."""
    dom = getattr(sym, "domain_names", None)
    if not dom:
        return ""
    if all(d == "*" for d in dom):
        return ""
    parts = [d if d != "*" else "*" for d in dom]
    return "(" + ",".join(parts) + ")"


def emit_set(sym) -> str:
    """Emit a set as 'Set name(parent) / a, b, c /;' (or empty 'Set name(parent) /  /;')."""
    name = sym.name
    desc = (sym.description or "").replace("'", "")
    dom_str = _domain_str(sym)
    if sym.records is None or len(sym.records) == 0:
        return f"Set {name}{dom_str} '{desc}' /  /;\n"
    cols = list(sym.records.columns)
    key_cols = [c for c in cols if c != "element_text"]
    elements = []
    for _, row in sym.records.iterrows():
        elements.append(".".join(str(row[c]) for c in key_cols))
    body = ",\n   ".join(elements)
    return f"Set {name}{dom_str} '{desc}' /\n   {body}\n/;\n"


def emit_set_assignment(sym) -> str:
    """Emit set membership as element-by-element 'name(elem) = yes;' assignment.

    Use this when the set is already declared earlier in the file (so we
    can't redeclare it). Each element is added with an unquoted singleton.
    """
    name = sym.name
    if sym.records is None or len(sym.records) == 0:
        return f"* {name}: empty\n"
    cols = list(sym.records.columns)
    key_cols = [c for c in cols if c != "element_text"]
    lines = [f"* {name} membership"]
    for _, row in sym.records.iterrows():
        keys = ",".join(f"'{row[c]}'" for c in key_cols)
        lines.append(f"{name}({keys}) = yes ;")
    return "\n".join(lines) + "\n"


_GETDATA_RENAME = {
    # GDX symbol → name declared in getData.gms (the $load alias targets).
    "POP": "pop0",
    "rorFlex": "rorFlex0",
}
_GETDATA_SKIP = {
    # GDX symbols not consumed by upstream standard_gtap_7/getData.gms.
    # Skipping avoids "Unknown symbol" errors at compile time.
    "VOSB",
    # 9x10 GDX (gtap10 dataset) extras that getData.gms doesn't declare:
    "DVER", "DPSM", "ISEP", "CSEP", "MFRV", "OSEP", "TFRV", "VMRT", "XTRV",
    "RORFLEX",  # the elasticity rorFlex is loaded as `rorFlex0=rorFlex` so this
                # uppercase variant from a different file is redundant.
    # gtap7_15x10 (and other margin-structured datasets) GDX extras that
    # getData.gms does NOT declare — VMTS is a bilateral margin param, and the
    # rT* are pre-computed tax RATES that getData recomputes from the value
    # params (VDFB/VDFP/…), so injecting them is both redundant AND breaks the
    # compile ("$140 Unknown symbol"; NEOS jobs 19760911/12).  Present only in
    # datasets with an explicit margin/tax-rate block (absent in 3x3/5x5/10x7).
    "VMTS",
    "rTFD", "rTFE", "rTFM", "rTGD", "rTGM", "rTID", "rTIM", "rTIN",
    "rTMS", "rTO", "rTPD", "rTPM", "rTXS",
}


def emit_parameter(sym) -> str:
    """Emit element-by-element assignments — assumes parameter is already
    declared by getData.gms. Use uppercase names to match the original symbols.
    """
    if sym.name in _GETDATA_SKIP:
        return f"* skipped {sym.name} (not declared in getData)\n"
    name = _GETDATA_RENAME.get(sym.name, sym.name)
    domain = sym.domain_names
    n_dim = len(domain) if isinstance(domain, list) else 0

    if sym.records is None or len(sym.records) == 0:
        if n_dim == 0:
            return f"{name} = 0 ;\n"
        return f"* {name} (empty in GDX — relying on $onImplicitAssign suppression)\n"
    df = sym.records

    if n_dim == 0:
        v = float(df["value"].iloc[0])
        return f"{name} = {v:.10g} ;\n"

    val_col = "value" if "value" in df.columns else df.columns[-1]
    key_cols = [c for c in df.columns if c != val_col]
    lines = [f"* {name} data ({len(df)} cells)"]
    for _, row in df.iterrows():
        v = float(row[val_col])
        if v == 0.0:
            continue
        keys = ",".join(f"'{row[c]}'" for c in key_cols)
        lines.append(f"{name}({keys}) = {v:.10g} ;")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Path for set declarations block (out_sets.gms is created next to it).")
    ap.add_argument("--out-params", type=Path, default=None,
                    help="If given, writes parameter assignments here separately.")
    args = ap.parse_args()

    import gams.transfer as gt
    m = gt.Container(str(args.gdx), system_directory=GAMS_SYS)

    # Sets block: full Set declarations (must exist at compile time).
    set_lines = ["* === Inlined SETS from " + args.gdx.name + " ===\n"]
    for name, sym in m.data.items():
        if sym.__class__.__name__ in ("Set", "Alias"):
            set_lines.append(emit_set(sym))
    args.out.write_text("\n".join(set_lines))
    print(f"Wrote {args.out} ({args.out.stat().st_size:,} bytes)")

    # Params block: element-by-element assignments to already-declared params.
    out_params = args.out_params or args.out.with_name("inline_params.gms")
    par_lines = [
        "* === Inlined PARAMETER data from " + args.gdx.name + " ===\n",
        "$onImplicitAssign\n",
    ]
    for name, sym in m.data.items():
        if sym.__class__.__name__ == "Parameter":
            par_lines.append(emit_parameter(sym))
    out_params.write_text("\n".join(par_lines))
    print(f"Wrote {out_params} ({out_params.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
