"""Build a self-contained, LOCALLY-runnable PURE-gtap tariff-shock bundle.

This is the PURE STD_GTAP model (NOT altertax). It assembles a single .gms
from a dataset's v7_consolidated.gdx that:

  1. Loads the dataset sets/params inline (no external GDX at execution).
  2. Solves the baseline (loop tsim: base -> check), `Solve gtap using mcp`.
  3. Applies a +10% tm_pct import-tariff shock on the 'shock' period:
        imptx.fx = (1 + imptx.l) * 1.10 - 1   (gated by xwFlag, so the
        diagonal r==r / no-flow routes are skipped automatically).
     This matches the Python gate's _apply_imptx_shock(factor=0.10):
        imptx_new = (1 + imptx_old) * (1 + 0.10) - 1.
  4. Solves the shock period and execute_unloads to out.gdx.

It is produced for BOTH ifSUB=0 and ifSUB=1 (two .gms / two out.gdx) and is
meant to be run LOCALLY with GAMS v53 (community license) — gtap7_3x3 fits the
~2500 row/col limit.

Differences from build_gtap7_altertax_neos_bundle.py:
  * starts from comp.gms (PURE) instead of comp_altertax.gms,
  * NO altertax elasticity overrides,
  * shock = tm_pct import tariff (not the numeraire pnum.fx jump),
  * NO homotopy ramp / path.opt (the PURE +10% tariff is a soft basin),
  * runs locally (no NEOS submission),
  * pdp/pmp recalc and pwmg=0 are already in postsim.gms / iterloop.gms.

Usage:
    uv run python scripts/gtap/build_gtap7_pure_local_bundle.py \
        --dataset gtap7_3x3 --ifsub 0
    uv run python scripts/gtap/build_gtap7_pure_local_bundle.py \
        --dataset gtap7_3x3 --ifsub 1
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

# Reuse the proven inlining machinery from the altertax NEOS builder.
import build_gtap7_altertax_neos_bundle as alt

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = alt.UPSTREAM
DATASETS_DIR = ROOT / "datasets"


def _patch_shock_to_tariff(text: str, tariff_increase: float = 0.10) -> str:
    """Replace the numeraire shock block with a +tariff% tm_pct import shock.

    comp.gms shock block:
        if(sameas(tsim,'shock'),
           pnum.fx(tsim) = 1.5 ;
        ) ;

    Replace with the tm_pct power-of-tariff shock, gated by xwFlag so the
    diagonal (r==r) and no-flow routes are skipped (xwFlag=0 there).
    """
    old_block = (
        "   if(sameas(tsim,'shock'),\n"
        "      pnum.fx(tsim) = 1.5 ;\n"
        "   ) ;"
    )
    factor = 1.0 + tariff_increase
    pct = tariff_increase * 100
    new_block = (
        f"* === PURE gtap tm_pct shock: +{pct:.0f}% on the POWER of all bilateral\n"
        f"* === import tariffs.  imptx_new = (1+imptx)*{factor} - 1, gated by xwFlag\n"
        f"* === (so the diagonal r==r / no-flow routes, where xwFlag=0, are skipped).\n"
        f"* === Matches the Python gate's _apply_imptx_shock(factor={tariff_increase}).\n"
        f"   if(sameas(tsim,'shock'),\n"
        f"      imptx.fx(r,i,rp,tsim)$xwFlag(r,i,rp) =\n"
        f"         (1 + imptx.l(r,i,rp,tsim)) * {factor} - 1 ;\n"
        f"   ) ;"
    )
    if old_block not in text:
        raise SystemExit(
            "Could not locate numeraire shock block to replace.\nExpected:\n"
            + old_block
        )
    n = text.count(old_block)
    if n != 1:
        raise SystemExit(f"Expected 1 occurrence of shock block, found {n}.")
    return text.replace(old_block, new_block, 1)


def _inject_path_opt_for_shock(text: str) -> str:
    """Attach a PATH `path.opt` to the shock `solve gtap using mcp` (ifSUB=1).

    The ifSUB=1 shock on the large datasets (10x7, 15x10) is NOT the "soft basin"
    the ifSUB=0 / small-dataset shock is: with the default PATH settings PATH
    DIVERGES from the check warm-start — the residual climbs (regYeq 3.2 → factYeq
    13.8 → xfeq 43.7), stalls on `xfeq(USA,SkLab,Manuf)` at 34.15 for three majors,
    then blows to inf on `eveq(USA,hhd)` and quits with `** EXIT - iteration limit`
    at residual ~3-4 (verified in NEOS jobs 19761836/7 solve.log).  The saved out.gdx
    is that non-converged point, so it violates its OWN regYeq (regY ≠ factY+yTaxInd
    by 1.2-1.5) — that is what made Python's parity read ~66% against it.

    The faithful fix is solver STABILISATION, not a model change: a proximal
    perturbation to damp the diverging Newton steps, plus higher major/minor limits
    and a better crash so PATH can climb out of the stall.  This is standard PATH
    tuning for a hard MCP restart; it changes no equation and hardcodes no value.

    Gated to ifSUB=1 (the ifSUB=0 bundle is byte-identical and already converges to
    a 100%-parity out.gdx — do not perturb it).
    """
    opt_lines = "\n".join([
        "convergence_tolerance 1e-9",
        "major_iteration_limit 2000",
        "minor_iteration_limit 100000",
        "cumulative_iteration_limit 1000000",
        "proximal_perturbation 1e-2",
        "crash_method pnewton",
        "crash_perturb yes",
        "nms_initial_reference_factor 2",
        "gradient_step_limit 20",
        "restart_limit 5",
        "lemke_start automatic",
        "time_limit 3600",
    ])
    optfile_block = (
        "\n* === PATH stabilisation for the ifSUB=1 shock restart (see\n"
        "* === _inject_path_opt_for_shock): default PATH diverges from the check\n"
        "* === warm-start on the large datasets; a proximal perturbation + higher\n"
        "* === iteration limits let it converge.  Faithful (solver tuning, no eq change).\n"
        "$onecho > path.opt\n"
        f"{opt_lines}\n"
        "$offecho\n"
        "   gtap.optfile = 1 ;\n"
    )
    anchor = "   options limrow = 3, limcol = 3, solprint = off, iterlim = 1000 ;"
    if text.count(anchor) != 1:
        raise SystemExit(
            f"Expected exactly 1 shock-solve options anchor, found {text.count(anchor)}."
        )
    return text.replace(anchor, anchor + "\n" + optfile_block, 1)


def _force_ifsub(text: str, ifsub: int) -> str:
    """Force the ifSUB compile-time switch to a fixed value.

    comp.gms uses `$setGlobal ifSUB 1`.  Replace with the requested value.
    """
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

    `set is` (SAM accounts) unions acts ∪ comm ∪ endw ∪ stdlab ∪ reg by listing each
    `set.<name>` marker.  In datasets whose activity and commodity elements share the
    SAME names (e.g. gtap7_15x10: both are Rice/Grains/… with NO a_/c_ prefix, unlike
    3x3/5x5/10x7 which are prefixed), listing set.acts THEN set.comm redeclares those
    elements and GAMS raises "$172 Element is redefined" (verified locally: $onMulti
    does NOT help — it permits set RE-declaration, not duplicate elements in one list).
    When acts == comm the union equals acts alone, so dropping the `set.comm` line from
    the `set is` block is exact and safe.  Only fires on the collision (prefixed
    datasets are untouched).  set.comm stays in the OTHER composites (aa, i) which are
    subsets of the already-complete `is`.
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
    # DELETE the `set.comm` line entirely (and its `*  User-defined commodities`
    # header comment).  A commented-out marker LEFT IN PLACE breaks GAMS ("$182
    # Closing / missing", "$409 Unrecognizable item") — a `*` line inside a set
    # element list is mishandled — so remove the whole line, leaving the block a
    # clean acts ∪ endw ∪ stdlab ∪ reg (== the intended union, since comm == acts).
    new_block = re.sub(
        r'\n[ \t]*\*[ \t]*User-defined commodities[ \t]*\n(?:[ \t]*\n)*[ \t]*set\.comm[ \t]*',
        '',
        block, count=1,
    )
    if new_block == block:
        # Fallback: at least remove the bare set.comm line.
        new_block = re.sub(r'\n[ \t]*set\.comm[ \t]*(?=\n)', '', block, count=1)
    if new_block == block:
        return text
    return text.replace(block, new_block, 1)


def _fbep_ftrv_har_as_assignments(har_path: Path, header: str, inlined: str) -> str:
    """Emit GAMS assignment lines for a factor header (FBEP/FTRV) read from a
    basedata.har, matching the inliner's own `* <NAME> data (N cells)` block
    format (index order ENDW/factor, ACTS/activity, REG/region; raw values).

    The ACTS element gets whatever prefix the dataset's own inlined sets use for
    activities (gtap7_3x3 uses `a_Food`; gtap7_15x10 uses bare `Rice`) — detected
    from the already-built `inlined` .gms so the emitted keys resolve against the
    dataset's real sets (else GAMS Error 170 domain violation at compile).
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

    # Detect the activity-set prefix from the inlined .gms's own EVFB block
    # (same (ENDW, ACTS, REG) domain): e.g. `evfb('Land','a_Food','USA')` → a_.
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
                    help="Dataset name, e.g. gtap7_3x3")
    ap.add_argument("--ifsub", type=int, choices=(0, 1), required=True,
                    help="ifSUB mode: 0 (full) or 1 (reduced).")
    ap.add_argument("--tariff", type=float, default=0.10,
                    help="Uniform tariff power shock fraction (default 0.10).")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    dataset = args.dataset
    ifsub = args.ifsub
    src_gdx = DATASETS_DIR / dataset / "v7_consolidated.gdx"
    if not src_gdx.exists():
        raise SystemExit(f"v7_consolidated.gdx not found: {src_gdx}")

    out_dir = args.out_dir or ROOT / "output" / f"{dataset}_pure_local_bundle"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== [1/4] Copying {dataset} consolidated GDX as in.gdx ===")
    in_gdx = out_dir / "in.gdx"
    shutil.copy2(src_gdx, in_gdx)
    print(f"  copied -> {in_gdx.name} ({in_gdx.stat().st_size:,} bytes)")

    ds_sets = alt._get_dataset_sets(in_gdx)
    print(f"  dataset sets: reg={ds_sets['reg']}, comm={ds_sets['comm']}")

    print("\n=== [2/4] Inlining GDX -> GAMS Set/Param declarations ===")
    data_block_path = out_dir / "inline_data.gms"
    params_path = out_dir / "inline_params.gms"
    alt.run_inliner(in_gdx, data_block_path, params_path)

    data_block_text = alt._filter_metadata_sets(data_block_path.read_text())
    data_block_text = alt._normalize_set_names(data_block_text)
    data_block_path.write_text(data_block_text)
    params_text = alt._rename_pop_to_pop0(params_path.read_text())
    params_path.write_text(params_text)

    print("\n=== [3/4] Inlining $include directives (PURE comp.gms) ===")
    main_script = alt._read("comp.gms")
    main_script = _force_ifsub(main_script, ifsub)
    inlined = alt._inline_includes(main_script)
    print(f"  inlined size: {len(inlined):,} chars")

    print("\n=== [4/4] Stripping GDX loads, injecting data, applying shock ===")
    inlined = alt._strip_gdx_loads(inlined)
    inlined = alt._strip_getdata_set_block(inlined)
    inlined = alt._patch_dataset_sets(inlined, ds_sets)

    # Inject the inline SETS block at the stripped set-decl anchor.
    data_block = data_block_path.read_text()
    marker = "* [getData set decl stripped]"
    idx = inlined.find(marker)
    if idx == -1:
        raise SystemExit("Could not find getData set-decl marker.")
    inlined = inlined[:idx] + "\n" + data_block + "\n" + inlined[idx:]

    # Fix the acts==comm element collision in the `set is` composite (unprefixed
    # datasets like gtap7_15x10 → "$172 Element is redefined").  See
    # _fix_acts_comm_collision.  No-op for prefixed datasets (3x3/5x5/10x7).
    inlined = _fix_acts_comm_collision(inlined)

    # Split inline_params into Dat-side and Prm-side (same split as altertax).
    params_block = params_path.read_text()
    dat_param_names = {
        "VDFB", "VDFP", "VMFB", "VMFP", "VDPB", "VDPP", "VMPB", "VMPP",
        "VDGB", "VDGP", "VMGB", "VMGP", "VDIB", "VDIP", "VMIB", "VMIP",
        "EVFB", "EVFP", "EVOS", "VXSB", "VFOB", "VCIF", "VMSB",
        "VST", "VTWR", "SAVE", "VDEP", "VKB", "POP", "MAKS", "MAKB", "pop0",
    }
    # BUG FIX (found 2026-07-14): the inliner's `* <name> data (N cells)`
    # marker comments carry the GAMS parameter's LOWERCASE spelling (e.g.
    # `* evfb data (36 cells)`), but `dat_param_names` above lists the
    # UPPERCASE GDX header names (`EVFB`, `EVFP`, ...). The membership check
    # `current_target in dat_param_names` was therefore case-sensitive and
    # ALWAYS false for every one of these params — evfb/evfp (and every
    # other dat_param_names entry) silently fell into prm_lines instead of
    # dat_lines, so `dat_block` never actually contained their real values.
    # This is what let `ftrv = evfp - evfb` (inserted right before dat_block)
    # evaluate on GAMS's implicit-zero defaults regardless of where dat_block
    # was positioned — fixing the insertion ORDER alone (see the ftrv_anchor
    # block below) was not sufficient without ALSO fixing this membership
    # check, since the wrongly-placed data would just end up in prm_block
    # instead, contributing nothing at either position. Normalize both sides
    # to uppercase for the comparison.
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

    # BUG FIX (found 2026-07-14): the original getData.gms computes
    # `ftrv = evfp - evfb` (and `fbep = 0`) RIGHT AFTER its own
    # `execute_load "...Dat.gdx", ..., evfb, evfp, evos, ...` — i.e. the
    # data injection happens BEFORE the ftrv calculation in the real GAMS
    # source. This inliner previously inserted `dat_block` (which carries
    # the real evfb/evfp/etc. values) AFTER the ftrv anchor line instead of
    # BEFORE it — so in every generated bundle (local AND NEOS, both bundle
    # builders share this exact anchor/insertion pattern), `ftrv` was
    # computed while evfb/evfp were still at GAMS's implicit default (0),
    # making ftrv (and therefore fcttx, in periods/models that source fcttx
    # from ftrv) silently and universally zero — confirmed via a `display`
    # instrumentation of a real local GAMS run showing EVFB/EVFP/ftrv all
    # "( ALL 0.000 )" right after the (mis-ordered) assignment. This directly
    # explains gtap7_3x3's own gams_levels(ref,'fcttx') reading all-zero:
    # not GAMS's genuine behavior, but an artifact of this inliner's bundle
    # (including the reference GDX fixtures generated by it/its NEOS twin).
    ftrv_anchor = "ftrv(fp,a0,r) = evfp(fp,a0,r) - evfb(fp,a0,r) ;"
    f_idx = inlined.find(ftrv_anchor)
    if f_idx < 0:
        raise SystemExit("Could not find ftrv anchor for Dat params block.")
    inlined = inlined[:f_idx] + dat_block + "\n\n" + inlined[f_idx:]

    rorflex_anchor = "* [stripped for NEOS inline] $loadDC etrae=etrae rorFlex0=rorFlex\n"
    a_idx = inlined.find(rorflex_anchor)
    if a_idx < 0:
        raise SystemExit("Could not locate rorFlex stripped-load anchor.")
    close_marker = "* [stripped for NEOS inline] $gdxin\n"
    close_idx = inlined.find(close_marker, a_idx)
    if close_idx < 0:
        raise SystemExit("Could not locate Prm.gdx close $gdxin marker.")

    # Some v7_consolidated.gdx datasets (e.g. gtap7_15x10, gtap7_3x4) don't
    # embed the CDE/CES elasticity parameters (incpar/subpar/esubi/...) —
    # unlike gtap7_3x3/5x5/10x7/20x41, which already carry them under GAMS
    # lowercase names. Without them cal.gms's CDE income-allocation module
    # (eh0/bh0 = incpar/subpar) divides by zero. Detect the gap and inject
    # from the dataset's own default.prm (GEMPACK HAR) via the same
    # _prm_har_as_assignments emitter nl_compare.py already uses for this
    # exact case.
    #
    # DETECTION BUG (found 2026-07-13 via a gtap7_5x5 CONVERT run failing to
    # compile): the inliner (gdx_to_gams_inline.py) NEVER emits a literal
    # "parameter incpar(...) ;" declaration line for ANY dataset — it only
    # emits a `* incpar data (N cells)` comment followed by bare assignment
    # lines (`incpar('c_Agri','USA') = ... ;`). The old check ("parameter
    # incpar(" not in prm_block) is therefore ALWAYS true, firing the
    # injection unconditionally — including for gtap7_5x5, which genuinely
    # already has incpar/subpar embedded (confirmed via gdxdump + this
    # inliner's own output). The redundant injection happened to be
    # harmless there only because THIS session's prefix fix to
    # _prm_har_as_assignments (1cdd60d) coincidentally re-derived the SAME
    # c_/a_-prefixed keys 5x5 already uses — but gtap7_15x10 uses NO
    # prefix at all on its comm/acts set elements (`Rice`, not `c_Rice` —
    # confirmed via its own inlined `Set comm .../ Rice, Grains, ... /`
    # block), so the SAME injection there emits keys ("c_Rice") that don't
    # exist in that dataset's own sets, hard-failing GAMS Error 170 (domain
    # violation) once CONVERT's stricter compile is used (PATH's own solve
    # path tolerates it, which is why this went unnoticed for weeks).
    #
    # Correct detection: check for the comment marker the inliner ACTUALLY
    # emits (`* incpar data`), not a declaration line that never exists.
    #
    # PREFIX BUG (found 2026-07-13 right after the detection-bug fix above):
    # whether comm/acts elements carry a c_/a_ prefix in THIS dataset's own
    # inlined sets is not universal either — gtap7_5x5 does, gtap7_15x10
    # does not (bare `Rice`/`Grains` for both comm and acts). Injecting the
    # wrong convention hard-fails GAMS Error 170 at compile. Detect from the
    # already-built `inlined` text itself rather than assuming.
    if "* incpar data" not in prm_block and "* INCPAR data" not in prm_block:
        prm_har_path = DATASETS_DIR / dataset / "default.prm"
        if prm_har_path.exists():
            import nl_compare as _nlc
            use_prefix = _nlc._detect_prm_prefix_convention(inlined)
            elast_block = _nlc._prm_har_as_assignments(prm_har_path, use_prefix=use_prefix)
            # comp.gms (via getData.gms, already inlined) declares these
            # params with real domains — a fresh `parameter name(*,*) ;`
            # redeclaration is GAMS error 184. Keep only the assignment
            # lines (`name(...) = val ;`), drop the declaration lines.
            assign_lines = [
                ln for ln in elast_block.splitlines()
                if not re.match(r"^\s*parameter\s+\w+\(", ln)
            ]
            elast_block = "\n".join(assign_lines)
            if elast_block.strip():
                prm_block += "\n" + elast_block
                print(f"  injected default.prm elasticities (incpar/subpar/esub*) for {dataset}")

    # FBEP/FTRV injection (found 2026-07-16): some datasets' v7_consolidated.gdx
    # LOSE the FBEP (factor subsidy) / FTRV (factor tax) symbols during
    # aggregation — gtap7_3x3's v7 has no FBEP symbol at all, so the inliner
    # never emits a `* FBEP data` block and getData.gms's hardcoded `fbep=0`
    # (and its `ftrv = evfp - evfb` fallback) win, producing a subsidy-BLIND
    # reference GDX (fctts=0, pfa/pf wedge = net rtf). But the dataset's OWN
    # basedata.har DOES carry the real FBEP/FTRV (3x3: 12/36 nonzero cells,
    # FBEP total -0.1674 — IDENTICAL to 15x10's 177-cell total, i.e. a faithful
    # aggregation of the same underlying subsidies). Python loads FBEP/FTRV from
    # that basedata.har and computes fctts=-fbep/evfb, fcttx=ftrv/evfb — so a
    # subsidy-blind reference mismatches Python for every subsidized (ag) factor.
    # Fix: when the inlined .gms lacks the FBEP/FTRV data blocks (v7 dropped
    # them), inject them from basedata.har with the dataset's own set-prefix
    # convention, so the regenerated reference GDX carries the real subsidies.
    for _sym, _n_idx in (("FBEP", 3), ("FTRV", 3)):
        if f"* {_sym} data" in inlined or f"\n{_sym}(" in inlined:
            continue  # already present (from the v7 via the inliner)
        _blk = _fbep_ftrv_har_as_assignments(
            DATASETS_DIR / dataset / "basedata.har", _sym, inlined)
        if _blk.strip():
            prm_block += "\n" + _blk
            _ncells = _blk.count(" = ")
            print(f"  injected {_sym} ({_ncells} cells) from basedata.har for {dataset}")

    insert_at = close_idx + len(close_marker)
    inlined = inlined[:insert_at] + "\n" + prm_block + "\n" + inlined[insert_at:]

    # Redirect output files to the bundle dir (no %outDir%/%simName% macros).
    inlined = inlined.replace('"%outDir%/%simName%.csv"', '"COMP.csv"')
    inlined = inlined.replace('"%outDir%/%simName%DBG.csv"', '"COMPDBG.csv"')
    inlined = inlined.replace('"%outDir%/%simName%.gdx"', '"out.gdx"')

    # Apply the tm_pct tariff shock (replaces the numeraire pnum.fx jump).
    inlined = _patch_shock_to_tariff(inlined, tariff_increase=args.tariff)

    # ifSUB=1 only: attach a PATH path.opt to stabilise the diverging shock restart
    # (default PATH hits the iteration limit at residual ~3-4, saving a non-converged
    # out.gdx that violates its own regYeq).  ifSUB=0 stays byte-identical.
    if ifsub == 1:
        inlined = _inject_path_opt_for_shock(inlined)
        print("  injected PATH path.opt for the ifSUB=1 shock restart")

    out_gms = out_dir / f"comp_{dataset}_gtap_shock_ifsub{ifsub}.gms"
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

    print(f"\nNext: run LOCALLY with GAMS v53:")
    print(f"  gams {out_gms.name} curDir={out_dir}  (-> out.gdx)")


if __name__ == "__main__":
    main()
