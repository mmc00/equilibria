#!/usr/bin/env python3
"""GTAP CLI - Command-line interface for GTAP CGE model

Usage:
    python run_gtap.py info --gdx-file data/asa7x5.gdx
    python run_gtap.py calibrate --gdx-file data/asa7x5.gdx
    python run_gtap.py solve --gdx-file data/asa7x5.gdx --solver ipopt
    python run_gtap.py shock --gdx-file data/asa7x5.gdx --shock-file shock.yaml

Commands:
    info        Display GTAP data information
    calibrate   Calibrate model from GDX
    solve       Solve the baseline model
    shock       Apply shock and solve

Example:
    # Run baseline
    python run_gtap.py solve --gdx-file data/asa7x5.gdx
    
    # Apply 10% import tariff shock
    python run_gtap.py shock --gdx-file data/asa7x5.gdx \\
        --variable rtms --index '(USA,agr,EUR)' --value 0.10
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

# Add equilibria to path if running standalone
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from equilibria.templates.gtap import (
    GTAPModelEquations,
    GTAPParameters,
    GTAPSets,
    GTAPSolver,
    build_gtap_contract,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """GTAP CGE Model CLI - CGEBox Implementation"""
    ctx.ensure_object(dict)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    ctx.obj['logger'] = logging.getLogger(__name__)


@cli.command()
@click.option(
    '--gdx-file',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to GTAP GDX file'
)
@click.pass_context
def info(ctx, gdx_file):
    """Display GTAP data information"""
    logger = ctx.obj['logger']
    logger.info(f"Loading GTAP data from {gdx_file}")
    
    try:
        # Load sets
        sets = GTAPSets()
        sets.load_from_gdx(gdx_file)
        
        click.echo(f"\n{'='*60}")
        click.echo(f"GTAP Data Information")
        click.echo(f"{'='*60}")
        click.echo(f"File: {gdx_file}")
        click.echo(f"Aggregation: {sets.aggregation_name}")
        click.echo(f"")
        click.echo(f"Sets:")
        click.echo(f"  Regions:      {sets.n_regions:3d} - {', '.join(sets.r)}")
        click.echo(f"  Commodities:  {sets.n_commodities:3d} - {', '.join(sets.i)}")
        click.echo(f"  Activities:   {sets.n_activities:3d} - {', '.join(sets.a)}")
        click.echo(f"  Factors:      {sets.n_factors:3d} - {', '.join(sets.f)}")
        click.echo(f"")
        click.echo(f"Factor Mobility:")
        click.echo(f"  Mobile:   {sets.n_mobile_factors} - {', '.join(sets.mf)}")
        click.echo(f"  Specific: {sets.n_specific_factors} - {', '.join(sets.sf)}")
        click.echo(f"")
        
        # Validate
        is_valid, errors = sets.validate()
        if is_valid:
            click.echo(click.style("✓ Sets are valid", fg="green"))
        else:
            click.echo(click.style("✗ Set validation errors:", fg="red"))
            for error in errors:
                click.echo(f"  - {error}")
        
        # Load parameters summary
        params = GTAPParameters()
        params.load_from_gdx(gdx_file)
        
        click.echo(f"\nParameters:")
        click.echo(f"  Elasticities:    {len(params.elasticities.esubva) + len(params.elasticities.esubm)} loaded")
        click.echo(f"  Benchmark:       {len(params.benchmark.vom) + len(params.benchmark.vfm)} flows")
        click.echo(f"  Tax rates:       {len(params.taxes.rto) + len(params.taxes.rtms)} rates")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--gdx-file',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to GTAP GDX file'
)
@click.option(
    '--closure',
    default='gtap_standard',
    help='Closure type (gtap_standard, trade_policy, cgebox_full)'
)
@click.option(
    '--solver',
    type=click.Choice(['ipopt', 'path', 'conopt']),
    default='ipopt',
    help='Solver to use'
)
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    help='Output file for results (JSON)'
)
@click.pass_context
def solve(ctx, gdx_file, closure, solver, output):
    """Solve the GTAP baseline model"""
    logger = ctx.obj['logger']
    logger.info(f"Solving GTAP model from {gdx_file}")
    
    try:
        # Load data
        with click.progressbar(length=3, label='Loading data') as bar:
            params = GTAPParameters()
            params.load_from_gdx(gdx_file)
            bar.update(1)
            
            contract = build_gtap_contract(closure)
            bar.update(1)
            
            equations = GTAPModelEquations(params.sets, params, contract.closure)
            model = equations.build_model()
            bar.update(1)
        
        # Solve
        click.echo(f"\nSolving with {solver}...")
        gtap_solver = GTAPSolver(model, contract.closure, solver_name=solver)
        
        with click.progressbar(length=1, label='Solving') as bar:
            result = gtap_solver.solve()
            bar.update(1)
        
        # Display results
        click.echo(f"\n{'='*60}")
        click.echo(f"Solution Results")
        click.echo(f"{'='*60}")
        
        if result.success:
            click.echo(click.style(f"✓ Converged successfully", fg="green"))
        else:
            click.echo(click.style(f"✗ Did not converge", fg="red"))
        
        click.echo(f"Status:       {result.status.value}")
        click.echo(f"Iterations:   {result.iterations}")
        click.echo(f"Solve time:   {result.solve_time:.2f}s")
        click.echo(f"Walras check: {result.walras_value:.2e}")
        
        if result.objective_value is not None:
            click.echo(f"Objective:    {result.objective_value:.6f}")
        
        # Save results
        if output:
            results_data = {
                "status": result.status.value,
                "success": result.success,
                "iterations": result.iterations,
                "solve_time": result.solve_time,
                "walras_value": result.walras_value,
                "message": result.message,
            }
            with open(output, 'w') as f:
                json.dump(results_data, f, indent=2)
            click.echo(f"\nResults saved to: {output}")
        
        # Exit with appropriate code
        sys.exit(0 if result.success else 1)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--gdx-file',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to GTAP GDX file'
)
@click.option(
    '--variable',
    required=True,
    help='Variable to shock (e.g., rtms, rtxs)'
)
@click.option(
    '--index',
    required=True,
    help='Index tuple (e.g., "(USA,agr,EUR)")'
)
@click.option(
    '--value',
    type=float,
    required=True,
    help='New value for the shock'
)
@click.option(
    '--solver',
    type=click.Choice(['ipopt', 'path', 'conopt']),
    default='ipopt',
    help='Solver to use'
)
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    help='Output file for results (JSON)'
)
@click.pass_context
def shock(ctx, gdx_file, variable, index, value, solver, output):
    """Apply a shock and solve the model"""
    logger = ctx.obj['logger']
    logger.info(f"Applying shock: {variable}{index} = {value}")
    
    try:
        # Parse index
        try:
            idx = eval(index)
            if not isinstance(idx, tuple):
                idx = (idx,)
        except:
            idx = (index,)
        
        # Load and solve baseline
        params = GTAPParameters()
        params.load_from_gdx(gdx_file)
        
        contract = build_gtap_contract("trade_policy")  # Allow tax changes
        equations = GTAPModelEquations(params.sets, params, contract.closure)
        model = equations.build_model()
        
        gtap_solver = GTAPSolver(model, contract.closure, solver_name=solver)
        
        # Apply shock
        shock_def = {"variable": variable, "index": idx, "value": value}
        gtap_solver.apply_shock(shock_def)
        
        # Solve
        click.echo(f"Solving with shock...")
        result = gtap_solver.solve()
        
        # Display results
        click.echo(f"\n{'='*60}")
        click.echo(f"Shock Results")
        click.echo(f"{'='*60}")
        click.echo(f"Shock: {variable}{index} = {value}")
        
        if result.success:
            click.echo(click.style(f"✓ Converged successfully", fg="green"))
        else:
            click.echo(click.style(f"✗ Did not converge", fg="red"))
        
        click.echo(f"Status:       {result.status.value}")
        click.echo(f"Walras check: {result.walras_value:.2e}")
        
        if output:
            with open(output, 'w') as f:
                json.dump({
                    "shock": {"variable": variable, "index": idx, "value": value},
                    "status": result.status.value,
                    "success": result.success,
                    "walras_value": result.walras_value,
                }, f, indent=2)
            click.echo(f"\nResults saved to: {output}")
        
        sys.exit(0 if result.success else 1)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
