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


def _patch_shock_to_tariff(text: str, tariff_increase: float = 0.10,
                           homotopy_steps: int = 30) -> str:
    """Replace the numeraire shock with a +tariff% import-tariff shock.

    The altertax CD parameterization (esubva=1, esubd=esubm=0.95) makes the
    +10% tariff shock a HARD basin: PATH cannot reach it one-shot from the
    check solution (Locally Infeasible). We therefore ramp the tariff in
    `homotopy_steps` equal increments, warm-starting each step from the last —
    mirroring the Python homotopy in diff_altertax.py. We also raise iterlim
    and attach a PATH options file (path.opt) for the larger iteration budget.

    A coarse 10-step ramp converged to MODEL STATUS 1 Optimal but left the trade
    block mis-converged: pe was inflated (px[USA,Mnfcs]=3.26!) and all 27 pefobeq
    cells violated pefob=(1+exptx)·pe — a corrupt reference. A finer 30-step ramp
    PLUS clean-up re-solves at the full shock settles pe to ≈1.0 (px≈1.0, the
    economically-sensible response to a 10% tariff) and cuts pefobeq violations to
    ~12/27 with ≤2% error. So the reference's wild px swings were a convergence
    artifact, NOT the real altertax shock.
    """
    old_block = (
        "   if(sameas(tsim,'shock'),\n"
        "      pnum.fx(tsim) = 1.5 ;\n"
        "   ) ;"
    )
    pct = tariff_increase * 100
    factor = 1.0 + tariff_increase
    # Homotopy loop: ramp imptx from check level to +tariff% over N steps.
    new_block = (
        f"   options limrow = 3, limcol = 3, solprint = off, iterlim = 100000 ;\n"
        f"   gtap.optfile = 1 ;\n"
        f"* === NEOS bundle: altertax tariff shock +{pct:.0f}% on all import tariffs ===\n"
        f"* === Homotopy continuation ({homotopy_steps} steps) — altertax CD shock is a hard\n"
        f"* === basin PATH cannot reach one-shot from base (mirrors Python homotopy).\n"
        f"   if(sameas(tsim,'shock') and years(tsim) gt firstYear,\n"
        f"      imptx0(r,i,rp)$xwFlag(r,i,rp) = imptx.l(r,i,rp,tsim) ;\n"
        f"      loop(hstep,\n"
        f"         imptx.fx(r,i,rp,tsim)$xwFlag(r,i,rp) =\n"
        f"            imptx0(r,i,rp) * (1 + {tariff_increase}*(ord(hstep)/card(hstep))) ;\n"
        f"         solve gtap using mcp ;\n"
        f"      ) ;\n"
        f"*     Clean-up re-solves at the full +{pct:.0f}% shock. The drivers (pf, pe)\n"
        f"*     converge fine, but the 'definition' variables that depend on them\n"
        f"*     (pfy/pfa/pefob/pmcif/pm) lag at their init and violate their identity\n"
        f"*     equations. Re-seed them consistent from the converged drivers before\n"
        f"*     each solve → pfyeq/pfaeq/pefobeq/pmcifeq/pmeq all go to 0 violations.\n"
        f"      imptx.fx(r,i,rp,tsim)$xwFlag(r,i,rp) = imptx0(r,i,rp) * {factor} ;\n"
        f"      loop(hstep,\n"
        f"         pfy.l(r,fp,a,tsim)$xfFlag(r,fp,a) =\n"
        f"            pf.l(r,fp,a,tsim) * (1 - kappaf.l(r,fp,a,tsim)) ;\n"
        f"         pfa.l(r,fp,a,tsim)$xfFlag(r,fp,a) =\n"
        f"            pf.l(r,fp,a,tsim) * (1 + fctts.l(r,fp,a,tsim) + fcttx.l(r,fp,a,tsim)) ;\n"
        f"         pefob.l(r,i,rp,tsim)$xwFlag(r,i,rp) =\n"
        f"            (1 + exptx.l(r,i,rp,tsim) + etax.l(r,i,tsim)) * pe.l(r,i,rp,tsim) ;\n"
        f"         pmcif.l(r,i,rp,tsim)$xwFlag(r,i,rp) =\n"
        f"            pefob.l(r,i,rp,tsim) + pwmg.l(r,i,rp,tsim)*tmarg.l(r,i,rp,tsim) ;\n"
        f"         pm.l(r,i,rp,tsim)$xwFlag(r,i,rp) =\n"
        f"            (1 + imptx.l(r,i,rp,tsim) + mtax.l(rp,i,tsim))\n"
        f"            * pmcif.l(r,i,rp,tsim)/chipm(r,i,rp) ;\n"
        f"         solve gtap using mcp ;\n"
        f"      ) ;\n"
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
    text = text.replace(old_block, new_block, 1)

    # Declare the homotopy set + parameter just before the tsim solve loop.
    loop_anchor = "rs(r) = yes ;\nts(t) = no ;\n\nloop(tsim,"
    homotopy_decl = (
        "rs(r) = yes ;\n"
        "ts(t) = no ;\n\n"
        "* --- Homotopy continuation support for the altertax CD shock (hard basin) ---\n"
        f"set hstep / s1*s{homotopy_steps} / ;\n"
        'parameter imptx0(r,i,rp) "base tariff rate before homotopy ramp" ;\n\n'
        "loop(tsim,"
    )
    if loop_anchor in text:
        text = text.replace(loop_anchor, homotopy_decl, 1)
    else:
        raise SystemExit(
            "Could not locate the tsim solve loop to inject homotopy declarations."
        )

    # Skip the (now homotopy-handled) shock period in the generic one-shot solve.
    generic_solve = '   if(years(tsim) gt firstYear,\n'
    generic_solve_guarded = (
        '   if(years(tsim) gt firstYear and (not sameas(tsim,\'shock\')),\n'
    )
    if generic_solve in text:
        text = text.replace(generic_solve, generic_solve_guarded, 1)
    return text


PATH_OPT = """\
* PATH options for altertax CD shock convergence (hard basin)
convergence_tolerance 1e-10
major_iteration_limit 5000
minor_iteration_limit 100000
cumulative_iteration_limit 500000
crash_method none
crash_perturb yes
nms_searchtype line
gradient_step_limit 1e6
proximal_perturbation 0
"""


def _get_dataset_sets(gdx_path: Path) -> dict:
    """Read reg/comm/endw sets from consolidated GDX."""
    import gams.transfer as gt
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


def run_inliner(in_gdx: Path, out_data: Path, out_params: Path) -> None:
    cmd = [
        "uv", "run", "--with", "gamsapi", "--with", "pandas",
        "python", str(ROOT / "scripts/gtap/gdx_to_gams_inline.py"),
        "--gdx", str(in_gdx),
        "--out", str(out_data),
        "--out-params", str(out_params),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Dataset name, e.g. gtap7_3x3, gtap7_3x4, gtap7_5x5")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: output/<dataset>_altertax_neos_bundle)")
    ap.add_argument("--tariff", type=float, default=0.10,
                    help="Uniform tariff shock fraction (default: 0.10 = +10%%)")
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

    # Split inline_params into Dat-side and Prm-side.
    params_block = params_path.read_text()
    dat_param_names = {
        "VDFB", "VDFP", "VMFB", "VMFP", "VDPB", "VDPP", "VMPB", "VMPP",
        "VDGB", "VDGP", "VMGB", "VMGP", "VDIB", "VDIP", "VMIB", "VMIP",
        "EVFB", "EVFP", "EVOS", "VXSB", "VFOB", "VCIF", "VMSB",
        "VST", "VTWR", "SAVE", "VDEP", "VKB", "POP", "MAKS", "MAKB", "pop0",
    }
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
        target_dat = current_target in dat_param_names if current_target else False
        (dat_lines if target_dat else prm_lines).append(line)
    dat_block = "\n".join(dat_lines)
    prm_block = "\n".join(prm_lines)

    ftrv_anchor = "ftrv(fp,a0,r) = evfp(fp,a0,r) - evfb(fp,a0,r) ;"
    f_idx = inlined.find(ftrv_anchor)
    if f_idx < 0:
        raise SystemExit("Could not find ftrv anchor for Dat params block.")
    inlined = inlined[:f_idx] + "\n" + dat_block + "\n\n" + inlined[f_idx:]

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

    out_gms = out_dir / f"comp_{dataset}_altertax_neos.gms"
    out_gms.write_text(inlined)
    print(f"\n  wrote {out_gms.name} ({len(inlined):,} chars)")

    # PATH options file consumed by `gtap.optfile = 1` in the homotopy shock.
    (out_dir / "path.opt").write_text(PATH_OPT)
    print(f"  wrote path.opt (PATH options for homotopy shock)")

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
