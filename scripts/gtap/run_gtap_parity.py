#!/usr/bin/env python3
"""GTAP Parity Test Runner

This script runs GTAP parity tests comparing Python results against
CGEBox GAMS baseline.

Usage:
    # Run parity check with existing GAMS results
    python run_gtap_parity.py --gdx-file data/asa7x5.gdx \\
        --gams-results results/gams_baseline.gdx
    
    # Run with specific tolerance
    python run_gtap_parity.py --gdx-file data/asa7x5.gdx \\
        --gams-results results/gams_baseline.gdx \\
        --tolerance 1e-8
    
    # Run and save report
    python run_gtap_parity.py --gdx-file data/asa7x5.gdx \\
        --gams-results results/gams_baseline.gdx \\
        --output report.txt
    
    # Run specific closure
    python run_gtap_parity.py --gdx-file data/asa7x5.gdx \\
        --gams-results results/gams_baseline.gdx \\
        --closure trade_policy

Exit Codes:
    0 - Parity check passed
    1 - Parity check failed
    2 - Error (missing files, solve failure, etc.)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

# Add equilibria to path if running standalone
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from equilibria.templates.gtap.gtap_parity_pipeline import (
    GTAPParityRunner,
    run_gtap_parity_test,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--gdx-file',
    type=click.Path(exists=True, path_type=Path, dir_okay=False),
    required=True,
    help='Path to GTAP data GDX file'
)
@click.option(
    '--gams-results',
    type=click.Path(exists=True, path_type=Path, dir_okay=False),
    required=True,
    help='Path to GAMS results GDX file'
)
@click.option(
    '--closure',
    default='gtap_standard',
    type=click.Choice(['gtap_standard', 'trade_policy', 'cgebox_full', 'single_region']),
    help='Closure type to use'
)
@click.option(
    '--tolerance',
    type=float,
    default=1e-6,
    help='Tolerance for parity comparison'
)
@click.option(
    '--solver',
    type=click.Choice(['ipopt', 'path', 'conopt']),
    default='ipopt',
    help='Solver to use for Python model'
)
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    help='Output file for detailed report'
)
@click.option(
    '--json-output',
    type=click.Path(path_type=Path),
    help='Output file for JSON results'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
def main(
    gdx_file: Path,
    gams_results: Path,
    closure: str,
    tolerance: float,
    solver: str,
    output: Optional[Path],
    json_output: Optional[Path],
    verbose: bool,
):
    """Run GTAP parity test comparing Python vs GAMS."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    click.echo("=" * 70)
    click.echo("GTAP Parity Test")
    click.echo("=" * 70)
    click.echo(f"Data file:      {gdx_file}")
    click.echo(f"GAMS results:   {gams_results}")
    click.echo(f"Closure:        {closure}")
    click.echo(f"Tolerance:      {tolerance}")
    click.echo(f"Solver:         {solver}")
    click.echo("=" * 70)
    click.echo()
    
    try:
        # Initialize runner
        with click.progressbar(length=1, label='Initializing') as bar:
            runner = GTAPParityRunner(
                gdx_file=gdx_file,
                gams_results_gdx=gams_results,
                closure=closure,
                solver=solver,
                tolerance=tolerance,
            )
            bar.update(1)
        
        # Run Python model
        click.echo("\nRunning Python GTAP model...")
        py_result = runner.run_python()
        
        if not py_result.success:
            click.echo(click.style(f"✗ Python solve failed: {py_result.message}", fg="red"))
            sys.exit(2)
        
        click.echo(click.style(f"✓ Python converged", fg="green"))
        click.echo(f"  Iterations:   {py_result.iterations}")
        click.echo(f"  Solve time:   {py_result.solve_time:.2f}s")
        click.echo(f"  Walras check: {py_result.walras_value:.2e}")
        
        # Run parity check
        click.echo("\nRunning parity comparison...")
        comparison = runner.run_parity_check()
        
        # Generate report
        report = runner.generate_report(comparison)
        click.echo("\n" + report)
        
        # Save outputs
        if output:
            output.write_text(report)
            click.echo(f"\nReport saved to: {output}")
        
        if json_output:
            json_data = {
                "config": {
                    "gdx_file": str(gdx_file),
                    "gams_results": str(gams_results),
                    "closure": closure,
                    "tolerance": tolerance,
                    "solver": solver,
                },
                "python_result": {
                    "success": py_result.success,
                    "status": py_result.status.value,
                    "iterations": py_result.iterations,
                    "solve_time": py_result.solve_time,
                    "walras_value": py_result.walras_value,
                },
                "comparison": comparison.to_dict(),
            }
            json_output.write_text(json.dumps(json_data, indent=2, default=str))
            click.echo(f"JSON results saved to: {json_output}")
        
        # Exit with appropriate code
        if comparison.passed:
            click.echo(click.style("\n✓ PARITY CHECK PASSED", fg="green", bold=True))
            sys.exit(0)
        else:
            click.echo(click.style("\n✗ PARITY CHECK FAILED", fg="red", bold=True))
            click.echo(f"\n{comparison.n_mismatches} mismatches found")
            click.echo(f"Max absolute difference: {comparison.max_abs_diff:.2e}")
            click.echo(f"Max relative difference: {comparison.max_rel_diff:.2e}")
            sys.exit(1)
            
    except FileNotFoundError as e:
        click.echo(click.style(f"✗ File not found: {e}", fg="red"), err=True)
        sys.exit(2)
    except ImportError as e:
        click.echo(click.style(f"✗ Import error: {e}", fg="red"), err=True)
        click.echo("Make sure all dependencies are installed: pip install -e .", err=True)
        sys.exit(2)
    except Exception as e:
        logger.exception("Unexpected error")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(2)


if __name__ == '__main__':
    main()
