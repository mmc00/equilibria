"""Build a PROCESSED v7_consolidated.gdx from a RAW GEMPACK GtapView GDX.

Some GTAPAgg datasets (e.g. gtap7_15x10) ship a v7_consolidated.gdx that is the
RAW GEMPACK GtapView dump: uppercase symbol names, raw GEMPACK rate parameters
(rTFD/rTFE/.../PTAX/ADRV/...), GtapView metadata sets (XXCR/DREG/MCOM/...), NO
calibration elasticities, and NO a_/c_ activity/commodity label prefixes.

The altertax bundle builder (build_gtap7_altertax_neos_bundle.py →
gdx_to_gams_inline.py) instead expects the PROCESSED schema that the good
gtap7_5x5/v7_consolidated.gdx has:

  8 SETS (lowercase): acts(a_), comm(c_), reg, endw, marg(c_), endwm, endwf, endws
  45 PARAMS (lowercase): 31 market params (from the RAW GDX, uppercase→lowercase,
     goods columns prefixed c_/a_ per GTAP semantics) + 14 calibration
     elasticities (esubva/esubd/esubm/esubc/esubq/esubs/esubt/esubg/esubi/
     etrae/etraq/incpar/subpar/rorFlex) read from default.prm.

The RAW-only GEMPACK rate params and GtapView metadata sets are DROPPED — the
processed model never declares them, so leaving them in causes "Unknown symbol".

Writes via the babel low-level write_gdx so the regular-domain metadata is kept
without enforcing domain membership — exactly like the 5x5 reference, which
itself carries benign domain "violations" (evfb's endowment labels sit under the
`acts` domain, etc.).

Usage:
  PYTHONPATH=<gams>/GMSPython/lib/python3.12/site-packages \
    uv run --no-project python scripts/gtap/build_v7_processed_from_raw.py \
      --raw datasets/gtap7_15x10/v7_consolidated.gdx \
      --prm datasets/gtap7_15x10/default.prm \
      --out /tmp/v7_15x10_processed.gdx

Then swap it into the dataset (back up the RAW first), build+solve the bundle,
and restore the RAW.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gams.transfer as gt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from equilibria.babel.har_to_gdx import read_har  # noqa: E402
from equilibria.babel.gdx.symbols import Parameter, Set  # noqa: E402
from equilibria.babel.gdx.writer import write_gdx  # noqa: E402

GAMS_SYS = "/Library/Frameworks/GAMS.framework/Versions/53/Resources"

# Per-parameter (lower-name, domain-list, per-column-prefix-list) measured from
# the good gtap7_5x5/v7_consolidated.gdx.  prefix: "" none, "c_" commodity,
# "a_" activity.  Domains mirror the 5x5 (all goods cols → acts; vtwr's 2nd
# goods col → comm).
MARKET = {
    "VDFB": ("vdfb", ["acts", "acts", "reg"], ["c_", "a_", ""]),
    "VDFP": ("vdfp", ["acts", "acts", "reg"], ["c_", "a_", ""]),
    "VMFB": ("vmfb", ["acts", "acts", "reg"], ["c_", "a_", ""]),
    "VMFP": ("vmfp", ["acts", "acts", "reg"], ["c_", "a_", ""]),
    "MAKB": ("makb", ["acts", "acts", "reg"], ["c_", "a_", ""]),
    "MAKS": ("maks", ["acts", "acts", "reg"], ["c_", "a_", ""]),
    "EVFB": ("evfb", ["acts", "acts", "reg"], ["", "a_", ""]),
    "EVFP": ("evfp", ["acts", "acts", "reg"], ["", "a_", ""]),
    "EVOS": ("evos", ["acts", "acts", "reg"], ["", "a_", ""]),
    "VDPB": ("vdpb", ["acts", "reg"], ["c_", ""]), "VDPP": ("vdpp", ["acts", "reg"], ["c_", ""]),
    "VMPB": ("vmpb", ["acts", "reg"], ["c_", ""]), "VMPP": ("vmpp", ["acts", "reg"], ["c_", ""]),
    "VDGB": ("vdgb", ["acts", "reg"], ["c_", ""]), "VDGP": ("vdgp", ["acts", "reg"], ["c_", ""]),
    "VMGB": ("vmgb", ["acts", "reg"], ["c_", ""]), "VMGP": ("vmgp", ["acts", "reg"], ["c_", ""]),
    "VDIB": ("vdib", ["acts", "reg"], ["c_", ""]), "VDIP": ("vdip", ["acts", "reg"], ["c_", ""]),
    "VMIB": ("vmib", ["acts", "reg"], ["c_", ""]), "VMIP": ("vmip", ["acts", "reg"], ["c_", ""]),
    "VST":  ("vst",  ["acts", "reg"], ["c_", ""]),
    "VXSB": ("vxsb", ["acts", "reg", "reg"], ["c_", "", ""]),
    "VFOB": ("vfob", ["acts", "reg", "reg"], ["c_", "", ""]),
    "VCIF": ("vcif", ["acts", "reg", "reg"], ["c_", "", ""]),
    "VMSB": ("vmsb", ["acts", "reg", "reg"], ["c_", "", ""]),
    "VTWR": ("vtwr", ["acts", "comm", "reg", "reg"], ["c_", "c_", "", ""]),
    "SAVE": ("save", ["acts"], [""]), "VDEP": ("vdep", ["acts"], [""]),
    "VKB":  ("vkb",  ["acts"], [""]), "POP":  ("pop",  ["acts"], [""]),
}

# default.prm elasticity header → (lower-name, domain-list, row-prefix, mode).
# mode: "2d" (sector|endw × reg), "1d_reg" (per region), "1d_marg" (per margin).
# drop_zero: GAMS stores parameters sparse; the 5x5 drops all-zero rows.
PRM = {
    "ESBV": ("esubva", ["acts", "reg"], "a_", "2d", False),
    "ETRQ": ("etraq",  ["acts", "reg"], "a_", "2d", False),
    "ESBD": ("esubd",  ["acts", "reg"], "c_", "2d", False),
    "ESBM": ("esubm",  ["acts", "reg"], "c_", "2d", False),
    "INCP": ("incpar", ["acts", "reg"], "c_", "2d", False),
    "SUBP": ("subpar", ["acts", "reg"], "c_", "2d", False),
    "ESBC": ("esubc",  ["acts", "reg"], "a_", "2d", True),
    "ESBT": ("esubt",  ["acts", "reg"], "a_", "2d", True),
    "ESBQ": ("esubq",  ["acts", "reg"], "c_", "2d", True),
    "ETRE": ("etrae",  ["acts", "reg"], "",   "2d", True),
    "ESBG": ("esubg",   ["acts"], "",   "1d_reg",  False),
    "RFLX": ("rorFlex", ["acts"], "",   "1d_reg",  False),
    "ESBI": ("esubi",   ["acts"], "",   "1d_reg",  True),
    "ESBS": ("esubs",   ["acts"], "c_", "1d_marg", False),
}


def build(raw_path: Path, prm_path: Path, out_path: Path) -> None:
    raw = gt.Container(str(raw_path), system_directory=GAMS_SYS)

    def setelems(name):
        return list(dict.fromkeys(raw[name].records.iloc[:, 0]))

    acts_raw = setelems("ACTS")
    comm_raw = setelems("COMM")
    reg_raw = setelems("REG")
    endw_raw = setelems("ENDW")
    marg_raw = setelems("MARG")
    endm_raw = setelems("ENDM")
    endf_raw = setelems("ENDF")
    ends_raw = setelems("ENDS")

    symbols = []

    def mkset(name, dom, elems):
        return Set(name=name, dimensions=1, domain=[dom], elements=[[e] for e in elems])

    symbols += [
        mkset("acts", "*", ["a_" + e for e in acts_raw]),
        mkset("comm", "*", ["c_" + e for e in comm_raw]),
        mkset("reg",  "*", reg_raw),
        mkset("endw", "*", endw_raw),
        mkset("marg", "comm", ["c_" + e for e in marg_raw]),
        mkset("endwm", "endw", endm_raw),
        mkset("endwf", "endw", endf_raw),
        mkset("endws", "endw", ends_raw),
    ]

    # ── 31 market params ──
    for upper, (lower, dom, col_pfx) in MARKET.items():
        if upper not in raw:
            raise SystemExit(f"RAW GDX missing market param {upper}")
        df = raw[upper].records
        idx_cols = list(df.columns[:-1])
        if len(col_pfx) != len(idx_cols):
            raise SystemExit(f"{upper}: prefix length {len(col_pfx)} != ncol {len(idx_cols)}")
        keymat = df.iloc[:, :-1].astype(str).values
        vals = df.iloc[:, -1].values
        recs = []
        for r in range(len(df)):
            keys = [col_pfx[j] + keymat[r, j] for j in range(len(idx_cols))]
            recs.append((keys, float(vals[r])))
        symbols.append(Parameter(name=lower, dimensions=len(dom), domain=dom, records=recs))

    # ── 14 elasticities ──
    prm = read_har(str(prm_path))
    for header, (lower, dom, row_pfx, mode, drop_zero) in PRM.items():
        if header not in prm:
            raise SystemExit(f"default.prm missing elasticity header {header}")
        ha = prm[header]
        a = ha.array
        recs = []
        if mode == "2d":
            rows, regs = ha.set_elements[0], ha.set_elements[1]
            for i, rl in enumerate(rows):
                for jr, rg in enumerate(regs):
                    v = float(a[i, jr])
                    if drop_zero and v == 0.0:
                        continue
                    lab = (row_pfx + rl) if row_pfx else rl
                    recs.append(([lab, rg], v))
        elif mode == "1d_reg":
            for j, rg in enumerate(ha.set_elements[0]):
                v = float(a[j])
                if drop_zero and v == 0.0:
                    continue
                recs.append(([rg], v))
        elif mode == "1d_marg":
            for j, lb in enumerate(ha.set_elements[0]):
                recs.append((["c_" + lb], float(a[j])))
        symbols.append(Parameter(name=lower, dimensions=len(dom), domain=dom, records=recs))

    write_gdx(str(out_path), symbols)
    nsets = sum(1 for s in symbols if isinstance(s, Set))
    nparams = sum(1 for s in symbols if isinstance(s, Parameter))
    print(f"WROTE {out_path}  (sets={nsets} params={nparams})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path, help="RAW GEMPACK v7 GDX")
    ap.add_argument("--prm", required=True, type=Path, help="default.prm with elasticities")
    ap.add_argument("--out", required=True, type=Path, help="processed GDX output path")
    args = ap.parse_args()
    build(args.raw, args.prm, args.out)


if __name__ == "__main__":
    main()
