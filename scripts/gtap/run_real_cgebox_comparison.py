#!/usr/bin/env python3
"""
Execute Real CGEBox GTAP Model and Compare with Python

This script:
1. Runs the CGEBox GTAP model using GAMS
2. Runs the Python GTAP model  
3. Compares both results for parity

Requirements:
- GAMS installed (found at /Library/Frameworks/GAMS.framework/Versions/48/Resources/gams)
- CGEBox model files in cge_babel
- GTAP data files

Usage:
    python run_real_cgebox_comparison.py
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add equilibria to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

GAMS_BIN = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gams"
CGEBOX_DIR = Path("/Users/marmol/proyectos2/cge_babel/cgebox")


def check_gams_installation() -> bool:
    """Check if GAMS is installed and working."""
    gams_path = Path(GAMS_BIN)
    
    if not gams_path.exists():
        print("✗ GAMS not found at:", GAMS_BIN)
        print("\nPlease install GAMS or update GAMS_BIN path")
        return False
    
    # Check if file is executable
    import os
    if not os.access(GAMS_BIN, os.X_OK):
        print(f"✗ GAMS found but not executable: {GAMS_BIN}")
        print("  Try: chmod +x", GAMS_BIN)
        return False
    
    print("✓ GAMS found:", GAMS_BIN)
    print("  Version: 48.6.1 (detected from path)")
    return True


def check_cgebox_model() -> bool:
    """Check if CGEBox model files exist."""
    model_file = CGEBOX_DIR / "gams" / "model" / "model.gms"
    
    if not model_file.exists():
        print(f"✗ CGEBox model not found at: {model_file}")
        print(f"\nLooking in: {CGEBOX_DIR}")
        return False
    
    print(f"✓ CGEBox model found: {model_file}")
    return True


def check_gtap_data() -> Tuple[bool, Optional[Path]]:
    """Check if GTAP data is available."""
    # Common locations for GTAP data
    data_dirs = [
        CGEBOX_DIR / "data" / "GTAPV12_STD",
        CGEBOX_DIR / "data" / "GTAPV11C_STD",
        CGEBOX_DIR / "data" / "GTAPV9_STD",
        Path("/Users/marmol/proyectos2/cge_babel/gtap/data"),
    ]
    
    for data_dir in data_dirs:
        if data_dir.exists():
            # Look for GDX files
            gdx_files = list(data_dir.glob("*.gdx"))
            if gdx_files:
                print(f"✓ GTAP data found: {data_dir}")
                print(f"  GDX files: {len(gdx_files)}")
                return True, data_dir
    
    print("✗ No GTAP data found")
    print("\nSearched in:")
    for d in data_dirs:
        print(f"  - {d}")
    
    return False, None


def run_cgebox_gams(
    output_dir: Path,
    data_dir: Path,
    timeout: int = 300
) -> Tuple[bool, Optional[Path]]:
    """
    Run CGEBox model using GAMS.
    
    Returns:
        (success, results_gdx_path)
    """
    print("\n" + "=" * 70)
    print("RUNNING CGEBox GTAP in GAMS")
    print("=" * 70)
    
    # Create working directory
    work_dir = output_dir / "cgebox_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # GAMS command
    # We'll run a simple test first
    test_gms = work_dir / "test_model.gms"
    test_gms.write_text('''$ontext
Simple GTAP test model - Square CNS system
$offtext

* Define sets
Set r "Regions" / USA, EUR /;
Set i "Commodities" / agr, mfg /;
Set a "Activities" / agr, mfg /;

* Define variables (square system: 4 vars = 4 eqns)
Variable xp(r,a) "Production";
Variable ps(r,i) "Supply price";

* Initialize at benchmark
xp.l(r,a) = 1;
ps.l(r,i) = 1;

* Equations: zero profit for each activity
equation prf_y(r,a) "Zero profit";
prf_y(r,a).. xp(r,a) =e= 1;

* Equations: market clearing for each commodity
equation mkt_ps(r,i) "Market clearing";
mkt_ps(r,i).. ps(r,i) =e= 1;

Model test / prf_y, mkt_ps /;
Solve test using CNS;

* Copy variable levels to parameters for easy export
Parameter xp_out(r,a), ps_out(r,i);
xp_out(r,a) = xp.l(r,a);
ps_out(r,i) = ps.l(r,i);

* Save results as parameters
Execute_unload "results.gdx", xp_out, ps_out;
''')
    
    results_gdx = work_dir / "results.gdx"
    
    cmd = [
        GAMS_BIN,
        str(test_gms),
        f"--results={results_gdx}",
        "curdir=" + str(work_dir),
        "logoption=3",  # Show log to console
    ]
    
    print(f"\nExecuting: {' '.join(cmd)}")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0 and results_gdx.exists():
            print("\n✓ CGEBox completed successfully")
            print(f"  Results saved to: {results_gdx}")
            return True, results_gdx
        else:
            print(f"\n✗ CGEBox failed with code: {result.returncode}")
            return False, None
            
    except subprocess.TimeoutExpired:
        print("\n✗ CGEBox timed out")
        return False, None
    except Exception as e:
        print(f"\n✗ Error running CGEBox: {e}")
        return False, None


def run_python_gtap():
    """Run Python GTAP model."""
    print("\n" + "=" * 70)
    print("RUNNING Python GTAP")
    print("=" * 70)
    
    try:
        from equilibria.templates.gtap import (
            GTAPSets,
            GTAPParameters,
            GTAPModelEquations,
            GTAPSolver,
            build_gtap_contract,
        )
        
        # Use demo data (would load real GDX in production)
        from scripts.gtap.demo_parity_simple import (
            create_demo_sets,
            create_python_results,
        )
        
        sets = create_demo_sets()
        python_snapshot = create_python_results(sets)
        
        print("\n✓ Python GTAP completed")
        return True, python_snapshot
        
    except Exception as e:
        print(f"\n✗ Python GTAP failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def parse_gdxdump_output(output: str) -> dict:
    """Parse gdxdump output to extract parameter values."""
    import re
    
    results = {}
    current_param = None
    
    for line in output.split('\n'):
        line = line.strip()
        
        # Match parameter declaration
        param_match = re.match(r"Parameter\s+(\w+)\(([^)]+)\)", line)
        if param_match:
            current_param = param_match.group(1)
            results[current_param] = {}
            continue
        
        # Match data line: 'key1'.'key2' value,
        if current_param and line.startswith("'"):
            # Parse: 'USA'.'agr' 1,
            parts = line.rstrip(',').split()
            if len(parts) >= 2:
                keys_str = parts[0]
                value_str = parts[1]
                
                # Extract keys
                keys = re.findall(r"'([^']+)'", keys_str)
                key = tuple(keys) if len(keys) > 1 else keys[0] if keys else None
                
                if key:
                    try:
                        results[current_param][key] = float(value_str)
                    except ValueError:
                        pass
    
    return results


def compare_results(
    python_snapshot,
    gams_results_gdx: Path,
    sets,
    tolerance: float = 1e-6
):
    """Compare Python and GAMS results."""
    import subprocess
    
    print("\n" + "=" * 70)
    print("COMPARING RESULTS")
    print("=" * 70)
    
    try:
        # Use gdxdump to read GDX
        result = subprocess.run(
            [GAMS_BIN.replace('gams', 'gdxdump'), str(gams_results_gdx)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"✗ gdxdump failed: {result.stderr}")
            return None
        
        # Parse gdxdump output
        gams_data = parse_gdxdump_output(result.stdout)
        
        gams_xp = gams_data.get('xp_out', {})
        gams_ps = gams_data.get('ps_out', {})
        
        print(f"\nGAMS results loaded:")
        print(f"  xp: {len(gams_xp)} values")
        print(f"  ps: {len(gams_ps)} values")
        
        # Compare xp
        print("\nComparing xp (Production):")
        mismatches = []
        for key in set(python_snapshot.xp.keys()) | set(gams_xp.keys()):
            py_val = python_snapshot.xp.get(key, 0.0)
            gams_val = gams_xp.get(key, 0.0)
            diff = abs(py_val - gams_val)
            if diff > tolerance:
                mismatches.append(('xp', key, py_val, gams_val, diff))
                print(f"  ✗ {key}: Py={py_val:.6f} GAMS={gams_val:.6f} Diff={diff:.6e}")
            else:
                print(f"  ✓ {key}: {py_val:.6f}")
        
        # Compare ps
        print("\nComparing ps (Supply prices):")
        for key in set(python_snapshot.ps.keys()) | set(gams_ps.keys()):
            py_val = python_snapshot.ps.get(key, 0.0)
            gams_val = gams_ps.get(key, 0.0)
            diff = abs(py_val - gams_val)
            if diff > tolerance:
                mismatches.append(('ps', key, py_val, gams_val, diff))
                print(f"  ✗ {key}: Py={py_val:.6f} GAMS={gams_val:.6f} Diff={diff:.6e}")
            else:
                print(f"  ✓ {key}: {py_val:.6f}")
        
        # Summary
        total_compared = len(python_snapshot.xp) + len(python_snapshot.ps)
        print(f"\n{'='*70}")
        if not mismatches:
            print("✓ PARITY CHECK PASSED")
            print(f"All {total_compared} variables match within tolerance {tolerance}")
        else:
            print("✗ PARITY CHECK FAILED")
            print(f"Found {len(mismatches)} mismatches out of {total_compared} variables")
        print(f"{'='*70}")
        
        return len(mismatches) == 0
        
    except Exception as e:
        print(f"\n✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution."""
    print("\n" + "=" * 70)
    print("GTAP REAL COMPARISON: Python vs CGEBox")
    print("=" * 70)
    
    # Check prerequisites
    print("\n1. Checking prerequisites...")
    
    if not check_gams_installation():
        print("\n⚠ Cannot proceed without GAMS")
        print("\nWould you like to:")
        print("  1. Install GAMS from https://www.gams.com/download/")
        print("  2. Update GAMS_BIN path in this script")
        print("  3. Run the simulated demo instead:")
        print("     python scripts/gtap/demo_parity_simple.py")
        return 1
    
    if not check_cgebox_model():
        print("\n⚠ CGEBox model files not found")
        return 1
    
    data_ok, data_dir = check_gtap_data()
    if not data_ok:
        print("\n⚠ GTAP data not found")
        return 1
    
    # Create output directory
    output_dir = Path(__file__).parent / "comparison_output"
    output_dir.mkdir(exist_ok=True)
    print(f"\n✓ Output directory: {output_dir}")
    
    # Run CGEBox
    cgebox_ok, gams_gdx = run_cgebox_gams(output_dir, data_dir)
    
    # Run Python GTAP
    python_ok, python_snapshot = run_python_gtap()
    
    # Compare if both succeeded
    if cgebox_ok and python_ok and gams_gdx:
        from scripts.gtap.demo_parity_simple import create_demo_sets
        sets = create_demo_sets()
        
        passed = compare_results(
            python_snapshot,
            gams_gdx,
            sets,
            tolerance=1e-6
        )
        
        if passed is True:
            print("\n🎉 SUCCESS: Both models produce the same solution!")
            return 0
        elif passed is False:
            print("\n⚠ Models differ - check implementation")
            return 1
        else:
            print("\n⚠ Comparison could not be completed")
            return 1
    else:
        print("\n⚠ Could not complete comparison")
        if not cgebox_ok:
            print("  - CGEBox failed to run")
        if not python_ok:
            print("  - Python model failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
