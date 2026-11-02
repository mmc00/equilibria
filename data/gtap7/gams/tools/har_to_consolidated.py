"""Write a dataset's v7_consolidated.gdx from its HARs (basedata, sets, default.prm).

The GAMS bundle builders inline v7_consolidated.gdx: the GTAPAgg "processed"
schema, 8 sets and 45 parameters with lower-case names and a_/c_ prefixed
activity/commodity labels. Datasets that come without one (9x10, from the GTAP
Standard 7 GAMS distribution) get it from their HARs here; on the datasets that
ship one, the output equals it record by record
(tests/scripts/test_har_to_consolidated.py).

Labels get their prefix only when the HAR does not carry it already: gtap7_3x3's
HARs say `Food`, 9x10's say `c_Crops`.

Usage:
    uv run python data/gtap7/gams/tools/har_to_consolidated.py --dataset 9x10 \
        --out data/gtap7/9x10/v7_consolidated.gdx
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from equilibria.babel.gdx.symbols import Parameter, Set
from equilibria.babel.gdx.writer import write_gdx
from equilibria.babel.har_to_gdx import read_har
from equilibria.sources import dataset_dir

# set name -> (sets.har header, domain, label prefix)
SETS = {
    "acts": ("ACTS", "*", "a_"),
    "comm": ("COMM", "*", "c_"),
    "reg": ("REG", "*", ""),
    "endw": ("ENDW", "*", ""),
    "marg": ("MARG", "comm", "c_"),
    "endwm": ("ENDM", "endw", ""),
    "endwf": ("ENDF", "endw", ""),
    "endws": ("ENDS", "endw", ""),
}

# basedata.har header -> (name, domain, per-column label prefix). The domains
# mirror the GTAPAgg files (every goods column under acts, vtwr's second under comm).
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
    "VDPB": ("vdpb", ["acts", "reg"], ["c_", ""]),
    "VDPP": ("vdpp", ["acts", "reg"], ["c_", ""]),
    "VMPB": ("vmpb", ["acts", "reg"], ["c_", ""]),
    "VMPP": ("vmpp", ["acts", "reg"], ["c_", ""]),
    "VDGB": ("vdgb", ["acts", "reg"], ["c_", ""]),
    "VDGP": ("vdgp", ["acts", "reg"], ["c_", ""]),
    "VMGB": ("vmgb", ["acts", "reg"], ["c_", ""]),
    "VMGP": ("vmgp", ["acts", "reg"], ["c_", ""]),
    "VDIB": ("vdib", ["acts", "reg"], ["c_", ""]),
    "VDIP": ("vdip", ["acts", "reg"], ["c_", ""]),
    "VMIB": ("vmib", ["acts", "reg"], ["c_", ""]),
    "VMIP": ("vmip", ["acts", "reg"], ["c_", ""]),
    "VST": ("vst", ["acts", "reg"], ["c_", ""]),
    "VXSB": ("vxsb", ["acts", "reg", "reg"], ["c_", "", ""]),
    "VFOB": ("vfob", ["acts", "reg", "reg"], ["c_", "", ""]),
    "VCIF": ("vcif", ["acts", "reg", "reg"], ["c_", "", ""]),
    "VMSB": ("vmsb", ["acts", "reg", "reg"], ["c_", "", ""]),
    "VTWR": ("vtwr", ["acts", "comm", "reg", "reg"], ["c_", "c_", "", ""]),
    "SAVE": ("save", ["acts"], [""]),
    "VDEP": ("vdep", ["acts"], [""]),
    "VKB": ("vkb", ["acts"], [""]),
    "POP": ("pop", ["acts"], [""]),
}

# default.prm header -> (name, domain, per-column label prefix)
PRM = {
    "ESBV": ("esubva", ["acts", "reg"], ["a_", ""]),
    "ETRQ": ("etraq", ["acts", "reg"], ["a_", ""]),
    "ESBD": ("esubd", ["acts", "reg"], ["c_", ""]),
    "ESBM": ("esubm", ["acts", "reg"], ["c_", ""]),
    "INCP": ("incpar", ["acts", "reg"], ["c_", ""]),
    "SUBP": ("subpar", ["acts", "reg"], ["c_", ""]),
    "ESBC": ("esubc", ["acts", "reg"], ["a_", ""]),
    "ESBT": ("esubt", ["acts", "reg"], ["a_", ""]),
    "ESBQ": ("esubq", ["acts", "reg"], ["c_", ""]),
    "ETRE": ("etrae", ["acts", "reg"], ["", ""]),
    "ESBG": ("esubg", ["acts"], [""]),
    "RFLX": ("rorFlex", ["acts"], [""]),
    "ESBI": ("esubi", ["acts"], [""]),
    "ESBS": ("esubs", ["acts"], ["c_"]),
}


def _label(prefix: str, label: str) -> str:
    return label if not prefix or label.startswith(prefix) else prefix + label


def _parameter(har: dict, header: str, name: str, domain: list[str], prefixes: list[str]):
    h = har[header]
    labels = [[_label(p, e) for e in els] for p, els in zip(prefixes, h.set_elements)]
    if len(labels) != len(domain):
        raise SystemExit(f"{header}: {len(labels)} dimensions, expected {len(domain)}")
    records = []
    for idx in itertools.product(*(range(len(els)) for els in labels)):
        value = float(h.array[idx])
        if value != 0.0:  # GAMS stores parameters sparse
            records.append(([labels[k][i] for k, i in enumerate(idx)], value))
    return Parameter(name=name, dimensions=len(domain), domain=domain, records=records)


def build(folder: Path) -> list:
    """The 53 symbols of v7_consolidated.gdx for the dataset in ``folder``."""
    sets = read_har(str(folder / "sets.har"))
    base = read_har(str(folder / "basedata.har"))
    prm = read_har(str(folder / "default.prm"))
    symbols: list = [
        Set(
            name=name,
            dimensions=1,
            domain=[dom],
            elements=[[_label(prefix, str(e).strip())] for e in sets[header].array],
        )
        for name, (header, dom, prefix) in SETS.items()
    ]
    symbols += [_parameter(base, h, *spec) for h, spec in MARKET.items()]
    symbols += [_parameter(prm, h, *spec) for h, spec in PRM.items()]
    return symbols


def write(folder: Path, out: Path) -> None:
    write_gdx(str(out), build(folder))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    write(dataset_dir(a.dataset), a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
