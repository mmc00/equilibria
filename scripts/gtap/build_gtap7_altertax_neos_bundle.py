"""Build a self-contained NEOS altertax bundle from a v7_consolidated.gdx.

Works for any GTAPAgg dataset that has a v7_consolidated.gdx (gtap7_3x3,
gtap7_3x4, gtap7_5x5, …).  Mirrors build_altertax_neos_bundle.py but skips
the 9x10-specific GDX merge step — the consolidated GDX is already a single
merged file equivalent to in.gdx.

Usage:
    uv run python scripts/gtap/build_gtap7_altertax_neos_bundle.py --dataset gtap7_3x3
    uv run python scripts/gtap/build_gtap7_altertax_neos_bundle.py --dataset gtap7_3x4
    uv run python scripts/gtap/build_gtap7_altertax_neos_bundle.py --dataset gtap7_5x5

Output:
    output/<dataset>_altertax_neos_bundle/comp_<dataset>_altertax_neos.gms
    output/<dataset>_altertax_neos_bundle/in.gdx  (symlink/copy of v7_consolidated.gdx)

After running, submit the .gms to NEOS (solver=PATH, model=MCP) and save the
returned out.gdx to output/<dataset>_altertax_neos_bundle/out.gdx.
"""
from __future__ import annotations
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path("/Users/marmol/proyectos2/cge_babel/standard_gtap_7")
GAMS_SYS = "/Library/Frameworks/GAMS.framework/Versions/48/Resources"
DATASETS_DIR = ROOT / "datasets"


def _read(name: str) -> str:
    return (UPSTREAM / name).read_text()


def _resolve_mvar(arg_macro: str, arg_year: str) -> str:
    body = _read("mvar.gms")
    body = re.sub(r"^\$setargs\s+.*$", "", body, count=1, flags=re.MULTILINE)
    body = body.replace("%theMacro%", arg_macro).replace("%theYear%", arg_year)
    return body


def _resolve_solve(arg_solver: str) -> str:
    body = _read("solve.gms")
    body = body.replace("%1", arg_solver)
    return body


def _force_ifsub(text: str, ifsub: int) -> str:
    """Force the compile-time ifSUB switch. comp_altertax.gms uses
    `$setGlobal ifSUB 1`; replace it so we can generate both the ifsub0 and
    ifsub1 reference GDXs (the parity fixtures need both)."""
    new, n = re.subn(
        r"^\$setGlobal\s+ifSUB\s+\d+\s*$",
        f"$setGlobal ifSUB       {ifsub}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise SystemExit(f"Could not find `$setGlobal ifSUB` to set to {ifsub}.")
    return new


def _inline_includes(text: str) -> str:
    def _sub_batinclude(m: re.Match) -> str:
        target = m.group("file")
        args = (m.group("args") or "").split()
        if target == "solve.gms":
            return _resolve_solve(args[0] if args else "gtap")
        if target == "mvar.gms":
            return _resolve_mvar(*(args + ["", ""])[:2])
        body = _read(target)
        for i, a in enumerate(args, start=1):
            body = body.replace(f"%{i}", a)
        return body

    def _sub_include(m: re.Match) -> str:
        return _read(m.group("file"))

    pat_batinclude = re.compile(
        r"^\s*\$\$?batinclude\s+\"(?P<file>\S+\.(?:gms|inc))\"(?P<args>[^\n]*)$",
        re.MULTILINE,
    )
    pat_include = re.compile(
        r"^\s*\$\$?include\s+\"(?P<file>\S+\.(?:gms|inc))\"\s*$",
        re.MULTILINE,
    )

    for _ in range(8):
        new = pat_batinclude.sub(_sub_batinclude, text)
        new = pat_include.sub(_sub_include, new)
        if new == text:
            break
        text = new
    return text


def _strip_gdx_loads(text: str) -> str:
    pat = re.compile(
        r'^(\s*)(\$(?:gdxin|load[a-zA-Z]*)\b[^\n]*)$',
        re.MULTILINE,
    )
    return pat.sub(r'\1* [stripped for NEOS inline] \2', text)


def _strip_getdata_set_block(text: str) -> str:
    pat = re.compile(
        r'^sets\s*\n'
        r'\s*acts\s+"Activities"\s*\n'
        r'(?:\s*\w+[^\n]*\n)+?'
        r'\s*endws\(endw\)\s+"Sluggish factors"\s*\n'
        r';\s*$',
        re.MULTILINE,
    )
    def _commentify(m: re.Match) -> str:
        return "\n".join("* [getData set decl stripped] " + ln
                          for ln in m.group(0).splitlines())
    new_text, n = pat.subn(_commentify, text)
    if n == 0:
        raise SystemExit("Could not find getData.gms `sets acts comm ...;` block.")
    return new_text


def _insert_pwmg_fix_before_solve(text: str) -> str:
    fix = (
        "\n* === NEOS bundle fix: enforce pwmg convention (0 where tmarg=0) ===\n"
        "pwmg.l(r,i,rp,tsim)$(tmarg.l(r,i,rp,tsim) eq 0) = 0 ;\n"
    )
    pat = re.compile(
        r"(pwmg\.fx\(r,i,rp,tsim\)\$\(not tmgFlag\(r,i,rp\)\)\s*=\s*pwmg\.l\(r,i,rp,tsim\)\s*;)",
        re.IGNORECASE,
    )
    if not pat.search(text):
        raise SystemExit("Could not locate `pwmg.fx ... = pwmg.l` line.")
    return pat.sub(fix + r"\1", text, count=1)


def _insert_pdp_pmp_recalc_before_unload(text: str) -> str:
    fix = (
        "\n* === NEOS bundle fix: recompute pdp/pmp for ALL cells (alpha=0 too) ===\n"
        "pdp.l(r,i,aa,tsim) = pd.l(r,i,tsim) * (1 + dintx.l(r,i,aa,tsim)) ;\n"
        "pmp.l(r,i,aa,tsim) = pmt.l(r,i,tsim) * (1 + mintx.l(r,i,aa,tsim)) ;\n"
    )
    pat = re.compile(r'(execute_unload\s+"out\.gdx"\s*;)', re.IGNORECASE)
    if not pat.search(text):
        print("WARNING: could not locate execute_unload anchor.")
        return text
    return pat.sub(fix + r"\1", text, count=1)


def _patch_shock_to_tariff(text: str, tariff_increase: float = 0.10) -> str:
    old_block = (
        "   if(sameas(tsim,'shock'),\n"
        "      pnum.fx(tsim) = 1.5 ;\n"
        "   ) ;"
    )
    pct = tariff_increase * 100
    new_block = (
        f"   if(sameas(tsim,'shock'),\n"
        f"* === NEOS bundle: altertax tariff shock +{pct:.0f}% on all import tariffs ===\n"
        f"      imptx.fx(r,i,rp,tsim)$xwFlag(r,i,rp) =\n"
        f"         imptx.l(r,i,rp,tsim) * {1.0 + tariff_increase} ;\n"
        f"   ) ;"
    )
    if old_block not in text:
        raise SystemExit(
            "Could not locate numeraire shock block to replace.\n"
            "Expected:\n" + old_block
        )
    n = text.count(old_block)
    if n != 1:
        raise SystemExit(f"Expected 1 occurrence of shock block, found {n}.")
    return text.replace(old_block, new_block, 1)


def _get_dataset_sets(gdx_path: Path) -> dict:
    """Read reg/comm/endw sets from consolidated GDX.

    Prefers gams.transfer (real GAMS install) when available; falls back to
    equilibria's pure-Python GDX reader (no GAMS install required) otherwise.
    """
    try:
        import gams.transfer as gt
    except ImportError:
        from equilibria.babel.gdx.reader import read_gdx, read_set_elements, get_symbol
        d = read_gdx(str(gdx_path))

        def _read(name: str) -> list[str]:
            # v7_consolidated.gdx uses uppercase canonical GAMS set names
            # (REG/COMM/ENDW); solved out.gdx fixtures use lowercase.
            actual = name if get_symbol(d, name) is not None else name.upper()
            return [t[0] for t in read_set_elements(d, actual)]

        return {"reg": _read("reg"), "comm": _read("comm"), "endw": _read("endw")}
    c = gt.Container(str(gdx_path), system_directory=GAMS_SYS)
    return {
        "reg":  list(c["reg"].records["uni"].values),
        "comm": list(c["comm"].records["uni"].values),
        "endw": list(c["endw"].records["uni"].values),
    }


def _filter_metadata_sets(text: str) -> str:
    """Remove sets not needed by comp_altertax.gms from inline_data.

    Keeps only the 8 canonical GTAP sets: REG, COMM, MARG, ACTS, ENDW,
    ENDS (ENDWS), ENDM (ENDWM), ENDF (ENDWF).

    Removes:
    - GEMPACK metadata (XXCR, XXCD, XXCP, DREL) — illegal chars
    - GTAPAgg mapping sets (MCOM, MREG, MACT, MEND, MMAR) — repeated elements
    - GTAPAgg disaggregated sets (DCOM, DREG, DACT, DEND, DMAR) — not in model
    - Extra endowment subsets (ENDC, ENDL, ENDT, TARS) — not in model
    """
    # Sets to keep (uppercase names as they appear in inline_data)
    keep = {"REG", "COMM", "MARG", "ACTS", "ENDW",
            "ENDS", "ENDM", "ENDF",       # compact names (some GDX versions)
            "ENDWS", "ENDWM", "ENDWF"}    # full names (other GDX versions)
    pat = re.compile(
        r'^(Set\s+(\w+)\s+[^\n]*\n(?:.*\n)*?/\s*;\s*\n)',
        re.MULTILINE,
    )
    removed = []
    def _filter(m: re.Match) -> str:
        name = m.group(2).upper()
        if name in keep:
            return m.group(1)
        removed.append(name)
        return ""
    new = pat.sub(_filter, text)
    if removed:
        print(f"  filtered sets from inline_data: {removed}")
    return new


def _normalize_set_names(text: str) -> str:
    """Rename GDX set names to the canonical lowercase names expected by comp_altertax.gms.

    v7_consolidated.gdx uses uppercase/short names; getData.gms expects lowercase.
    Renames both the Set declaration header and the short-name variants of
    sluggish/mobile/specific endowment subsets.
    """
    renames = [
        ("REG",   "reg"),
        ("COMM",  "comm"),
        ("ACTS",  "acts"),
        ("ENDW",  "endw"),
        ("MARG",  "marg"),
        ("ENDS",  "endws"),   # short form
        ("ENDWS", "endws"),   # full form (already canonical)
        ("ENDM",  "endwm"),
        ("ENDWM", "endwm"),
        ("ENDF",  "endwf"),
        ("ENDWF", "endwf"),
    ]
    for src, dst in renames:
        if src == dst:
            continue
        # Rename only in Set declaration headers (first token after "Set ")
        text = re.sub(
            r'^(Set\s+)' + re.escape(src) + r'(\b)',
            lambda m, d=dst: m.group(1) + d + m.group(2),
            text, flags=re.MULTILINE,
        )
    return text


def _rename_pop_to_pop0(text: str) -> str:
    """Rename 'pop' → 'pop0' in inline_params.

    GAMS getData.gms uses $load pop0=pop (rename on load). The inliner
    generates assignments as 'pop(...)=...' but comp_altertax.gms declares
    'pop' as a Variable(r,t) — conflict. The parameter is called pop0.
    """
    # Match '* pop data' header and 'pop(...)=' assignments
    text = re.sub(r'^(\*\s*)pop(\s+(?:data|\(empty\)))', r'\1pop0\2',
                  text, flags=re.MULTILINE)
    text = re.sub(r'^pop\(', 'pop0(', text, flags=re.MULTILINE)
    return text


def _patch_dataset_sets(text: str, sets: dict) -> str:
    """Replace hardcoded 9x10 rres/rmuv/imuv set elements with dataset values.

    getData.gms contains hardcoded 9x10-specific element lists for:
      rres(r)  - residual region (last region)
      rmuv(r)  - RMUV price index regions (all regions)
      imuv(i)  - IMUV commodity basket (manufactured goods)
    """
    regions = sets["reg"]
    comms = sets["comm"]

    rres_elem = regions[-1]   # last region = residual (GAMS convention)
    rmuv_elems = ", ".join(regions)
    # imuv: non-agricultural, non-services commodities; use all if unclear
    imuv_elems = ", ".join(comms)

    def _replace_set_block(text: str, set_name: str, new_elements: str) -> str:
        pat = re.compile(
            r'(set\s+' + re.escape(set_name) + r'\s*\([^\)]+\)\s*"[^"]*"\s*/\s*\n)'
            r'((?:[^\n]*\n)*?)'
            r'(/\s*;)',
            re.MULTILINE | re.IGNORECASE,
        )
        def _sub(m: re.Match) -> str:
            return m.group(1) + f"   {new_elements}\n" + m.group(3)
        new, n = pat.subn(_sub, text)
        if n:
            print(f"  patched set {set_name} → {new_elements}")
        else:
            print(f"  WARNING: could not find set {set_name} to patch")
        return new

    text = _replace_set_block(text, "rres", rres_elem)
    text = _replace_set_block(text, "rmuv", rmuv_elems)
    text = _replace_set_block(text, "imuv", imuv_elems)
    return text


def _elements_of(text: str, set_name: str) -> set:
    """Extract the element names of an inline `Set <name> / ... /;` block."""
    m = re.search(
        rf'(?im)^\s*[Ss]et\s+{re.escape(set_name)}\b[^\n]*/\s*\n(.*?)\n\s*/\s*;',
        text, re.S,
    )
    if not m:
        return set()
    els = set()
    for ln in m.group(1).splitlines():
        ln = ln.split("*", 1)[0].strip().rstrip(",").strip()
        if not ln or ln.startswith("set."):
            continue
        tok = re.split(r"[\s,.]", ln)[0].strip().strip("'\"")
        if tok:
            els.add(tok)
    return els


def _fix_acts_comm_collision(text: str) -> str:
    """Drop the `set.comm` line from the `set is` composite when acts == comm.

    Ported from build_gtap7_pure_local_bundle.py (commit 9da8917). `set is` (SAM
    accounts) unions acts ∪ comm ∪ endw ∪ stdlab ∪ reg via `set.<name>` markers. In
    datasets whose activity and commodity elements share the SAME names (e.g.
    gtap7_3x4 / gtap7_15x10: both Food/Mnfcs/… with NO a_/c_ prefix, unlike
    3x3/5x5/10x7 which are prefixed), listing set.acts THEN set.comm redeclares those
    elements → GAMS "$172 Element is redefined" (verified: $onMulti does NOT help —
    it permits set re-declaration, not duplicate elements in one list). When
    acts == comm the union equals acts alone, so dropping `set.comm` from `set is` is
    exact and safe. Only fires on the collision (prefixed datasets untouched).
    """
    acts = _elements_of(text, "acts")
    comm = _elements_of(text, "comm")
    if not acts or acts != comm:
        return text  # prefixed (no collision) or sets not found — nothing to fix
    m = re.search(
        r'(set is "SAM accounts for aggregated SAM" /.*?/\s*;)', text, re.S
    )
    if not m:
        return text
    block = m.group(1)
    # DELETE the whole `set.comm` line + its header comment (a commented-out marker
    # left in place breaks GAMS: `*` inside a set element list is mishandled).
    new_block = re.sub(
        r'\n[ \t]*\*[ \t]*User-defined commodities[ \t]*\n(?:[ \t]*\n)*[ \t]*set\.comm[ \t]*',
        '',
        block, count=1,
    )
    if new_block == block:
        new_block = re.sub(r'\n[ \t]*set\.comm[ \t]*(?=\n)', '', block, count=1)
    if new_block == block:
        return text
    return text.replace(block, new_block, 1)


def run_inliner(in_gdx: Path, out_data: Path, out_params: Path) -> None:
    cmd = [
        "uv", "run", "--with", "gamsapi", "--with", "pandas",
        "python", str(ROOT / "scripts/gtap/gdx_to_gams_inline.py"),
        "--gdx", str(in_gdx),
        "--out", str(out_data),
        "--out-params", str(out_params),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _fbep_ftrv_har_as_assignments(har_path: Path, header: str, inlined: str) -> str:
    """Emit GAMS assignment lines for a factor header (FBEP/FTRV) read from a
    basedata.har (index order ENDW/factor, ACTS/activity, REG/region; raw values).

    Ported from build_gtap7_pure_local_bundle.py (commit d3dfb73): some datasets'
    v7_consolidated.gdx LOSE the FBEP (factor subsidy) symbol during aggregation
    — gtap7_3x3's v7 has no FBEP at all, so getData.gms's hardcoded `fbep=0` wins
    and the reference GDX is subsidy-BLIND (fctts=0, ytax(fs)=0). Python loads the
    real FBEP from basedata.har (fctts=-fbep/evfb), so a subsidy-blind reference
    mismatches Python for every subsidized (ag) factor. Inject the real header.

    The ACTS element gets whatever prefix the dataset's own inlined sets use for
    activities (gtap7_3x3 uses `a_Food`) — detected from the inlined EVFB block.
    """
    import sys as _sys
    _src = str(ROOT / "src")
    if _src not in _sys.path:
        _sys.path.insert(0, _src)
    from equilibria.babel.har.reader import read_har  # type: ignore
    import numpy as _np

    try:
        har = read_har(har_path, select_headers=[header])
    except Exception:
        return ""
    h = har.get(header)
    if h is None or getattr(h, "rank", 0) != 3:
        return ""

    act_prefix = ""
    m = re.search(r"(?:^|\n)evfb\('[^']+','([acfr])_", inlined)
    if m:
        act_prefix = m.group(1) + "_"

    endw_labels, acts_labels, reg_labels = h.set_elements
    lines = [f"* {header} data (injected from basedata.har)"]
    nz = _np.argwhere(h.array != 0)
    for idx in nz:
        f_lbl = endw_labels[idx[0]]
        a_lbl = acts_labels[idx[1]]
        r_lbl = reg_labels[idx[2]]
        val = float(h.array[tuple(idx)])
        lines.append(f"{header}('{f_lbl}','{act_prefix}{a_lbl}','{r_lbl}') = {val:.10g} ;")
    return "\n".join(lines) if len(lines) > 1 else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Dataset name, e.g. gtap7_3x3, gtap7_3x4, gtap7_5x5")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: output/<dataset>_altertax_neos_bundle)")
    ap.add_argument("--tariff", type=float, default=0.10,
                    help="Uniform tariff shock fraction (default: 0.10 = +10%%)")
    ap.add_argument("--ifsub", type=int, choices=(0, 1), default=1,
                    help="ifSUB compile switch (default 1); the parity fixtures need both")
    args = ap.parse_args()

    dataset = args.dataset
    src_gdx = DATASETS_DIR / dataset / "v7_consolidated.gdx"
    if not src_gdx.exists():
        raise SystemExit(f"v7_consolidated.gdx not found: {src_gdx}")

    out_dir = args.out_dir or ROOT / "output" / f"{dataset}_altertax_neos_bundle"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== [1/4] Copying {dataset} consolidated GDX as in.gdx ===")
    in_gdx = out_dir / "in.gdx"
    shutil.copy2(src_gdx, in_gdx)
    print(f"  copied → {in_gdx.name} ({in_gdx.stat().st_size:,} bytes)")

    # Read dataset-specific sets (reg/comm/endw) for patching.
    ds_sets = _get_dataset_sets(in_gdx)
    print(f"  dataset sets: reg={ds_sets['reg']}, comm={ds_sets['comm'][:4]}")

    print("\n=== [2/4] Inlining GDX → GAMS Set/Param declarations ===")
    data_block_path = out_dir / "inline_data.gms"
    params_path = out_dir / "inline_params.gms"
    run_inliner(in_gdx, data_block_path, params_path)
    print(f"  inline_data:   {data_block_path.stat().st_size:,} bytes")
    print(f"  inline_params: {params_path.stat().st_size:,} bytes")

    # Post-process inline files.
    data_block_text = _filter_metadata_sets(data_block_path.read_text())
    data_block_text = _normalize_set_names(data_block_text)
    data_block_path.write_text(data_block_text)
    params_text = _rename_pop_to_pop0(params_path.read_text())
    params_path.write_text(params_text)

    print("\n=== [3/4] Inlining $include directives ===")
    main_script = _read("comp_altertax.gms")
    main_script = _force_ifsub(main_script, args.ifsub)
    inlined = _inline_includes(main_script)
    print(f"  inlined size: {len(inlined):,} chars")

    print("\n=== [4/4] Stripping GDX loads, injecting data, applying fixes ===")
    inlined = _strip_gdx_loads(inlined)
    inlined = _strip_getdata_set_block(inlined)
    inlined = _patch_dataset_sets(inlined, ds_sets)

    # Inject inline SETS block right at the stripped set-decl anchor.
    data_block = data_block_path.read_text()
    marker = "* [getData set decl stripped]"
    idx = inlined.find(marker)
    if idx == -1:
        raise SystemExit("Could not find getData set-decl marker.")
    inlined = inlined[:idx] + "\n" + data_block + "\n" + inlined[idx:]

    # Drop set.comm from `set is` when acts == comm (bare/unprefixed datasets like
    # 3x4/15x10) — else GAMS "$172 Element is redefined". No-op for prefixed sets.
    # MUST run AFTER the data_block (which carries the `set is` composite) is inlined.
    inlined = _fix_acts_comm_collision(inlined)

    # Split inline_params into Dat-side and Prm-side.
    params_block = params_path.read_text()
    dat_param_names = {
        "VDFB", "VDFP", "VMFB", "VMFP", "VDPB", "VDPP", "VMPB", "VMPP",
        "VDGB", "VDGP", "VMGB", "VMGP", "VDIB", "VDIP", "VMIB", "VMIP",
        "EVFB", "EVFP", "EVOS", "VXSB", "VFOB", "VCIF", "VMSB",
        "VST", "VTWR", "SAVE", "VDEP", "VKB", "POP", "MAKS", "MAKB", "pop0",
    }
    # CASE-INSENSITIVE match (ported from build_gtap7_pure_local_bundle.py, commit
    # d3dfb73): the inliner emits comment anchors in LOWERCASE (`* ftrv data`,
    # `* evfb data`) but dat_param_names lists UPPERCASE. A case-sensitive
    # `current_target in dat_param_names` silently mis-routed FTRV/EVFB/... into
    # prm_lines instead of dat_lines → the raw factor-tax data was DROPPED → the
    # generated reference had ftrv=0, fcttx=0, ytax(ft)=0 for EVERY altertax
    # dataset this builder made. That's why the altertax parity gate saw
    # ytax[USA,ft]=0 in GAMS vs Python's real 3.099.
    dat_param_names_upper = {n.upper() for n in dat_param_names}
    header = params_block.split("\n")[0]
    dat_lines, prm_lines = [header], [header]
    current_target = None
    for line in params_block.splitlines()[1:]:
        m = re.match(r"^\*\s*([A-Za-z][A-Za-z0-9_]*)\s+(?:data|\(empty\))", line)
        if m:
            current_target = m.group(1)
        elif line.startswith("* skipped "):
            current_target = None
            continue
        elif line.startswith("$on") or line.startswith("$off"):
            dat_lines.append(line)
            prm_lines.append(line)
            continue
        target_dat = current_target.upper() in dat_param_names_upper if current_target else False
        (dat_lines if target_dat else prm_lines).append(line)
    dat_block = "\n".join(dat_lines)
    prm_block = "\n".join(prm_lines)

    ftrv_anchor = "ftrv(fp,a0,r) = evfp(fp,a0,r) - evfb(fp,a0,r) ;"
    f_idx = inlined.find(ftrv_anchor)
    if f_idx < 0:
        raise SystemExit("Could not find ftrv anchor for Dat params block.")
    inlined = inlined[:f_idx] + "\n" + dat_block + "\n\n" + inlined[f_idx:]

    # FBEP/FTRV injection (ported from build_gtap7_pure_local_bundle.py, 2026-07-16):
    # vanilla getData.gms hardcodes `fbep=0` and `ftrv=evfp-evfb` — it NEVER reads an
    # FBEP header — so a dataset whose v7 dropped FBEP yields a subsidy-BLIND reference
    # (fctts=0, ytax(fs)=0). gtap7_3x3's v7 has no FBEP at all. Python loads the real
    # FBEP/FTRV from basedata.har (fctts=-fbep/evfb), so the reference mismatched Python
    # for every subsidized (ag) factor: ytax[r,fs] py≈0.09 vs GAMS 0. Inject the real
    # header AFTER both anchors (GAMS is case-insensitive → `FBEP(...)` overrides `fbep=0`;
    # `FTRV(...)` overrides `ftrv=evfp-evfb` where the header has a nonzero value).
    anchor_end = inlined.find(ftrv_anchor) + len(ftrv_anchor)
    inject_lines = []
    for _sym in ("FBEP", "FTRV"):
        _blk = _fbep_ftrv_har_as_assignments(
            DATASETS_DIR / dataset / "basedata.har", _sym, inlined)
        if _blk.strip():
            inject_lines.append(_blk)
            _ncells = _blk.count(" = ")
            print(f"  injected {_sym} ({_ncells} cells) from basedata.har for {dataset}")
    if inject_lines:
        inject_txt = "\n\n" + "\n".join(inject_lines) + "\n"
        inlined = inlined[:anchor_end] + inject_txt + inlined[anchor_end:]

    rorflex_anchor = "* [stripped for NEOS inline] $loadDC etrae=etrae rorFlex0=rorFlex\n"
    a_idx = inlined.find(rorflex_anchor)
    if a_idx < 0:
        raise SystemExit("Could not locate rorFlex stripped-load anchor.")
    close_marker = "* [stripped for NEOS inline] $gdxin\n"
    close_idx = inlined.find(close_marker, a_idx)
    if close_idx < 0:
        raise SystemExit("Could not locate Prm.gdx close $gdxin marker.")
    insert_at = close_idx + len(close_marker)
    inlined = inlined[:insert_at] + "\n" + prm_block + "\n" + inlined[insert_at:]

    inlined = inlined.replace('"%outDir%/%simName%.csv"', '"COMP.csv"')
    inlined = inlined.replace('"%outDir%/%simName%DBG.csv"', '"COMPDBG.csv"')
    inlined = inlined.replace('"%outDir%/%simName%.gdx"', '"out.gdx"')

    inlined = _insert_pwmg_fix_before_solve(inlined)
    inlined = _insert_pdp_pmp_recalc_before_unload(inlined)
    inlined = _patch_shock_to_tariff(inlined, tariff_increase=args.tariff)

    out_gms = out_dir / f"comp_{dataset}_altertax_neos_ifsub{args.ifsub}.gms"
    out_gms.write_text(inlined)
    print(f"\n  wrote {out_gms.name} ({len(inlined):,} chars)")

    remaining = re.findall(r"\$\$?(?:bat)?include\s+\"(\S+)\"", inlined)
    if remaining:
        print(f"  WARNING: unresolved includes: {set(remaining)}")
    else:
        print("  All $include directives resolved.")

    active_macros = set()
    for line in inlined.splitlines():
        if line.lstrip().startswith("*"):
            continue
        for macro in ("%inDir%", "%BaseName%", "%outDir%", "%simName%"):
            if macro in line:
                active_macros.add(macro)
    if active_macros:
        print(f"  NOTE: leftover macros in active code: {active_macros}")

    print(f"\nNext: submit {out_gms.name} to NEOS (solver=PATH, model=MCP)")
    print(f"      save returned out.gdx → {out_dir}/out.gdx")


if __name__ == "__main__":
    main()
