#!/usr/bin/env python3
"""Run the core PEP scenario pack and optionally enforce GAMS parity by scenario."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from equilibria.baseline import GAMSNLPReferenceManifest  # noqa: E402
from equilibria.simulations import PepSimulator  # noqa: E402
from equilibria.simulations import export_tax, government_spending, import_price, import_shock  # noqa: E402

DEFAULT_SAM_FILE = REPO_ROOT / "src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx"
DEFAULT_VAL_PAR_FILE = REPO_ROOT / "src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the public PEP core scenario pack "
            "(base + export_tax + import_price_agr + import_shock + government_spending)."
        )
    )
    parser.add_argument("--sam-file", type=Path, default=DEFAULT_SAM_FILE)
    parser.add_argument("--val-par-file", type=Path, default=DEFAULT_VAL_PAR_FILE)
    parser.add_argument("--gdxdump-bin", type=str, default="gdxdump")
    parser.add_argument("--method", choices=["auto", "ipopt", "path"], default="ipopt")
    parser.add_argument("--init-mode", choices=["excel", "gams"], default="excel")
    parser.add_argument("--solve-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--compare-abs-tol", type=float, default=1e-6)
    parser.add_argument("--compare-rel-tol", type=float, default=1e-6)
    parser.add_argument("--no-dynamic-sets", action="store_true")
    parser.add_argument("--import-price-commodity", type=str, default="agr")
    parser.add_argument("--import-price-multiplier", type=float, default=1.25)
    parser.add_argument("--import-shock-multiplier", type=float, default=1.25)
    parser.add_argument("--export-tax-multiplier", type=float, default=0.75)
    parser.add_argument("--government-spending-multiplier", type=float, default=1.2)
    parser.add_argument("--sam-qa-mode", choices=["off", "warn", "hard_fail"], default="off")
    parser.add_argument("--cri-fix-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping scenario name -> {results_gdx, slice}. "
            "Supported keys: base, export_tax, import_price_agr, import_shock, government_spending."
        ),
    )
    parser.add_argument(
        "--require-reference-manifest",
        action="store_true",
        help="Fail if the reference manifest is missing or any required scenario entry is absent.",
    )
    parser.add_argument("--save-report", type=Path, default=None)
    return parser


def _load_reference_manifest(path: Path | None, *, required: bool) -> dict[str, dict[str, str]]:
    if path is None:
        if required:
            raise FileNotFoundError("reference manifest is required")
        return {}
    if not path.exists():
        raise FileNotFoundError(f"reference manifest not found: {path}")

    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("reference manifest must be a JSON object")

    if raw.get("schema_version") == "pep_gams_nlp_reference/v1":
        manifest = GAMSNLPReferenceManifest.model_validate(raw)
        if manifest.scenario_references is None:
            raise ValueError("official GAMS NLP reference manifest is missing scenario_references")
        out: dict[str, dict[str, str]] = {}
        for name, reference in manifest.scenario_references.items():
            out[str(name).strip().lower()] = {
                "results_gdx": reference.results_gdx.path,
                "slice": reference.slice,
            }
        return out

    out: dict[str, dict[str, str]] = {}
    for name, payload in raw.items():
        if not isinstance(payload, dict):
            raise ValueError(f"reference manifest entry '{name}' must be an object")
        results_gdx = payload.get("results_gdx")
        if not results_gdx:
            raise ValueError(f"reference manifest entry '{name}' is missing 'results_gdx'")
        out[str(name).strip().lower()] = {
            "results_gdx": str(results_gdx),
            "slice": str(payload.get("slice", "sim1")).strip().lower(),
        }
    return out


def _run_one(
    simulator: PepSimulator,
    *,
    scenario_name: str,
    scenario: Any | None,
    reference_manifest: dict[str, dict[str, str]],
    compare_abs_tol: float,
    compare_rel_tol: float,
    require_reference_manifest: bool,
) -> dict[str, Any]:
    ref = reference_manifest.get(scenario_name)
    if require_reference_manifest and ref is None:
        raise KeyError(f"missing reference manifest entry for '{scenario_name}'")

    if scenario is None:
        raw = simulator.run_scenarios(
            scenarios=[],
            include_base=True,
            warm_start=False,
            reference_results_gdx=(ref["results_gdx"] if ref else None),
            base_reference_slice=(ref["slice"] if ref else "base"),
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
        )
        entry = raw["base"]
    else:
        raw = simulator.run_scenarios(
            scenarios=[scenario],
            include_base=False,
            warm_start=False,
            reference_results_gdx=(ref["results_gdx"] if ref else None),
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
        )
        entry = raw["scenarios"][0]

    if ref is not None and entry["reference_slice"] != ref["slice"]:
        entry["reference_slice"] = ref["slice"]
        if isinstance(entry.get("comparison"), dict):
            entry["comparison"]["gams_slice"] = ref["slice"]

    return entry


def _print_entry(name: str, entry: dict[str, Any]) -> None:
    solve = entry["solve"]
    print("-" * 84)
    print(f"Scenario: {name}")
    print(
        "  solve: converged={converged} iterations={iterations} residual={residual:.3e}".format(
            converged=solve["converged"],
            iterations=solve["iterations"],
            residual=float(solve["final_residual"]),
        )
    )
    comparison = entry.get("comparison")
    if isinstance(comparison, dict):
        print(
            "  compare[{slice_}]: passed={passed} compared={compared} mismatches={mismatches} missing={missing}".format(
                slice_=comparison["gams_slice"],
                passed=comparison["passed"],
                compared=comparison["compared"],
                mismatches=comparison["mismatches"],
                missing=comparison["missing"],
            )
        )
        print(
            "  max_abs_diff={abs_:.6e} max_rel_diff={rel_:.6e}".format(
                abs_=float(comparison["max_abs_diff"]),
                rel_=float(comparison["max_rel_diff"]),
            )
        )


def main() -> int:
    args = _build_parser().parse_args()
    try:
        reference_manifest = _load_reference_manifest(
            args.reference_manifest,
            required=args.require_reference_manifest,
        )
    except Exception as exc:
        print(f"Invalid reference manifest: {exc}")
        return 2

    scenario_pack = [
        ("base", None),
        ("export_tax", export_tax(multiplier=args.export_tax_multiplier)),
        (
            f"import_price_{args.import_price_commodity.strip().lower()}",
            import_price(
                commodity=args.import_price_commodity,
                multiplier=args.import_price_multiplier,
            ),
        ),
        ("import_shock", import_shock(multiplier=args.import_shock_multiplier)),
        (
            "government_spending",
            government_spending(multiplier=args.government_spending_multiplier),
        ),
    ]

    simulator = PepSimulator(
        sam_file=args.sam_file,
        val_par_file=args.val_par_file,
        gdxdump_bin=args.gdxdump_bin,
        init_mode=args.init_mode,
        method=args.method,
        solve_tolerance=args.solve_tolerance,
        max_iterations=args.max_iterations,
        dynamic_sets=(not args.no_dynamic_sets),
        sam_qa_mode=args.sam_qa_mode,
        cri_fix_mode=args.cri_fix_mode,
    ).fit()

    entries: dict[str, Any] = {}
    failed_solve = False
    failed_compare = False
    for name, scenario in scenario_pack:
        try:
            entry = _run_one(
                simulator,
                scenario_name=name,
                scenario=scenario,
                reference_manifest=reference_manifest,
                compare_abs_tol=args.compare_abs_tol,
                compare_rel_tol=args.compare_rel_tol,
                require_reference_manifest=args.require_reference_manifest,
            )
        except Exception as exc:
            print(f"Scenario '{name}' failed before solve: {exc}")
            return 2
        entries[name] = entry
        if not bool(entry["solve"]["converged"]):
            failed_solve = True
        comparison = entry.get("comparison")
        if args.require_reference_manifest and not isinstance(comparison, dict):
            failed_compare = True
        if isinstance(comparison, dict) and not bool(comparison["passed"]):
            failed_compare = True

    report = {
        "config": {
            "sam_file": str(args.sam_file),
            "val_par_file": str(args.val_par_file),
            "gdxdump_bin": str(args.gdxdump_bin),
            "dynamic_sets": (not args.no_dynamic_sets),
            "init_mode": args.init_mode,
            "method": args.method,
            "solve_tolerance": args.solve_tolerance,
            "max_iterations": args.max_iterations,
            "compare_abs_tol": args.compare_abs_tol,
            "compare_rel_tol": args.compare_rel_tol,
            "reference_manifest": str(args.reference_manifest) if args.reference_manifest else None,
            "require_reference_manifest": bool(args.require_reference_manifest),
            "sam_qa_mode": args.sam_qa_mode,
            "cri_fix_mode": args.cri_fix_mode,
        },
        "scenarios": entries,
    }

    print("=" * 84)
    print("PEP CORE SCENARIOS GATE")
    print("=" * 84)
    for name, entry in entries.items():
        _print_entry(name, entry)
    print("-" * 84)

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, indent=2))
        print(f"Saved report: {args.save_report}")

    return 0 if (not failed_solve and not failed_compare) else 2


if __name__ == "__main__":
    raise SystemExit(main())
