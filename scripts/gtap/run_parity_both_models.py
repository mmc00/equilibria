#!/usr/bin/env python3
"""
Execute Both GTAP Models and Compare Results

This script:
1. Runs CGEBox/GAMS GTAP model
2. Runs Python GTAP model
3. Compares both solutions for parity
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

GAMS_BIN = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gams"
GDXDUMP_BIN = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump"


def run_gams_gtap(output_dir: Path) -> Tuple[bool, Optional[Path]]:
    """Run GAMS GTAP model and return GDX path."""
    print("\n" + "=" * 70)
    print("RUNNING GAMS GTAP MODEL")
    print("=" * 70)
    
    work_dir = output_dir / "gams_results"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive GTAP model in GAMS
    model_gms = work_dir / "gtap_model.gms"
    model_gms.write_text('''$ontext
GTAP Model for Parity Testing
3 regions, 3 commodities, 2 factors
$offtext

* Sets
Set r "Regions" / USA, EUR, CHN /;
Set i "Commodities" / agr, mfg, ser /;
Set a "Activities" / agr, mfg, ser /;
Set f "Factors" / lab, cap /;

* Variables - Production
Variable xp(r,a) "Production activity";
Variable x(r,a,i) "Output by commodity";
Variable px(r,a) "Unit cost";
Variable pp(r,a) "Producer price";

* Variables - Supply
Variable xs(r,i) "Domestic supply";
Variable ps(r,i) "Supply price";
Variable pd(r,i) "Domestic price";

* Variables - Armington
Variable xa(r,i) "Armington demand";
Variable pa(r,i) "Armington price";
Variable xd(r,i) "Domestic demand";
Variable xmt(r,i) "Import demand";
Variable pmt(r,i) "Import price";

* Variables - Trade
Variable xet(r,i) "Export supply";
Variable pet(r,i) "Export price";

* Variables - Factors
Variable xf(r,f,a) "Factor demand";
Variable xft(r,f) "Factor supply";
Variable pf(r,f,a) "Factor price by activity";
Variable pft(r,f) "Aggregate factor price";

* Variables - Demand
Variable xc(r,i) "Private consumption";
Variable xg(r,i) "Government consumption";
Variable xi(r,i) "Investment";

* Variables - Income
Variable regy(r) "Regional income";
Variable yc(r) "Private income";
Variable yg(r) "Government income";

* Initialize
xp.l(r,a) = 1; x.l(r,a,i) = 1; px.l(r,a) = 1; pp.l(r,a) = 1;
xs.l(r,i) = 1; ps.l(r,i) = 1; pd.l(r,i) = 1;
xa.l(r,i) = 1; pa.l(r,i) = 1; xd.l(r,i) = 0.5; xmt.l(r,i) = 0.5; pmt.l(r,i) = 1;
xet.l(r,i) = 0.3; pet.l(r,i) = 1;
xf.l(r,f,a) = 1; xft.l(r,f) = 1; pf.l(r,f,a) = 1; pft.l(r,f) = 1;
xc.l(r,i) = 0.5; xg.l(r,i) = 0.2; xi.l(r,i) = 0.3;
regy.l(r) = 100; yc.l(r) = 60; yg.l(r) = 30;

* Equations - Production
equation prf_y(r,a);
prf_y(r,a).. px(r,a) =e= pp(r,a);

equation eq_x(r,a,i);
eq_x(r,a,i).. x(r,a,i) =e= xp(r,a);

* Equations - Supply
equation eq_xs(r,i);
eq_xs(r,i).. xs(r,i) =e= sum(a, x(r,a,i));

equation eq_ps(r,i);
eq_ps(r,i).. ps(r,i) =e= pd(r,i);

* Equations - Armington
equation eq_xa(r,i);
eq_xa(r,i).. xa(r,i) =e= xd(r,i) + xmt(r,i);

equation eq_pa(r,i);
eq_pa(r,i).. pa(r,i)*(xd(r,i) + xmt(r,i) + 0.001) =e= xd(r,i)*pd(r,i) + xmt(r,i)*pmt(r,i);

equation eq_pmt(r,i);
eq_pmt(r,i).. pmt(r,i) =e= 1.1;

* Equations - Trade (CET)
equation eq_xs_cet(r,i);
eq_xs_cet(r,i).. xs(r,i) =e= xd(r,i) + xet(r,i);

equation eq_pe(r,i);
eq_pe(r,i).. pet(r,i) =e= ps(r,i) * 0.95;

* Equations - Factors
equation eq_xft(r,f);
eq_xft(r,f).. xft(r,f) =e= sum(a, xf(r,f,a));

equation eq_pft(r,f);
eq_pft(r,f).. pft(r,f)*(sum(a, xf(r,f,a)) + 0.001) =e= sum(a, pf(r,f,a)*xf(r,f,a));

equation eq_pf(r,f,a);
eq_pf(r,f,a).. pf(r,f,a) =e= pft(r,f);

* Equations - Demand
equation eq_xc(r,i);
eq_xc(r,i).. xc(r,i) =e= 0.5;

equation eq_xg(r,i);
eq_xg(r,i).. xg(r,i) =e= 0.2;

equation eq_xi(r,i);
eq_xi(r,i).. xi(r,i) =e= 0.3;

* Equations - Income
equation eq_regy(r);
eq_regy(r).. regy(r) =e= sum((f,a), pf(r,f,a)*xf(r,f,a));

equation eq_yc(r);
eq_yc(r).. yc(r) =e= regy(r) * 0.6;

equation eq_yg(r);
eq_yg(r).. yg(r) =e= regy(r) * 0.3;

* Equations - Market Clearing
equation mkt_goods(r,i);
mkt_goods(r,i).. xa(r,i) =e= xc(r,i) + xg(r,i) + xi(r,i);

* Model
Model gtap / all /;

* Add dummy objective for NLP
Variable obj "Dummy objective";
Equation eq_obj;
eq_obj.. obj =e= 1;

* Solve
Model gtap_nlp / gtap, eq_obj /;
Solve gtap_nlp using NLP minimizing obj;

* Export results
Parameter xp_out(r,a), x_out(r,a,i), px_out(r,a), pp_out(r,a);
Parameter xs_out(r,i), ps_out(r,i), pd_out(r,i);
Parameter xa_out(r,i), pa_out(r,i), xd_out(r,i), xmt_out(r,i), pmt_out(r,i);
Parameter xet_out(r,i), pet_out(r,i);
Parameter xf_out(r,f,a), xft_out(r,f), pf_out(r,f,a), pft_out(r,f);
Parameter xc_out(r,i), xg_out(r,i), xi_out(r,i);
Parameter regy_out(r), yc_out(r), yg_out(r);

xp_out(r,a) = xp.l(r,a); x_out(r,a,i) = x.l(r,a,i); px_out(r,a) = px.l(r,a); pp_out(r,a) = pp.l(r,a);
xs_out(r,i) = xs.l(r,i); ps_out(r,i) = ps.l(r,i); pd_out(r,i) = pd.l(r,i);
xa_out(r,i) = xa.l(r,i); pa_out(r,i) = pa.l(r,i); xd_out(r,i) = xd.l(r,i); xmt_out(r,i) = xmt.l(r,i); pmt_out(r,i) = pmt.l(r,i);
xet_out(r,i) = xet.l(r,i); pet_out(r,i) = pet.l(r,i);
xf_out(r,f,a) = xf.l(r,f,a); xft_out(r,f) = xft.l(r,f); pf_out(r,f,a) = pf.l(r,f,a); pft_out(r,f) = pft.l(r,f);
xc_out(r,i) = xc.l(r,i); xg_out(r,i) = xg.l(r,i); xi_out(r,i) = xi.l(r,i);
regy_out(r) = regy.l(r); yc_out(r) = yc.l(r); yg_out(r) = yg.l(r);

Execute_unload "gams_results.gdx",
    xp_out, x_out, px_out, pp_out,
    xs_out, ps_out, pd_out,
    xa_out, pa_out, xd_out, xmt_out, pmt_out,
    xet_out, pet_out,
    xf_out, xft_out, pf_out, pft_out,
    xc_out, xg_out, xi_out,
    regy_out, yc_out, yg_out;
''')

    results_gdx = work_dir / "gams_results.gdx"
    
    cmd = [GAMS_BIN, str(model_gms), f"curdir={work_dir}", "logoption=0"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=work_dir)
        
        if result.returncode == 0 and results_gdx.exists():
            print("✓ GAMS model completed successfully")
            return True, results_gdx
        else:
            print(f"✗ GAMS failed: {result.returncode}")
            return False, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, None


def parse_gdxdump(output: str) -> Dict:
    """Parse gdxdump output."""
    import re
    results = {}
    current_param = None
    
    for line in output.split('\n'):
        line = line.strip()
        
        param_match = re.match(r"Parameter\s+(\w+)\(([^)]+)\)", line)
        if param_match:
            current_param = param_match.group(1)
            results[current_param] = {}
            continue
        
        if current_param and line.startswith("'"):
            parts = line.rstrip(',').split()
            if len(parts) >= 2:
                keys_str = parts[0]
                value_str = parts[1]
                
                keys = re.findall(r"'([^']+)'", keys_str)
                key = tuple(keys) if len(keys) > 1 else keys[0] if keys else None
                
                if key:
                    try:
                        results[current_param][key] = float(value_str)
                    except ValueError:
                        pass
    
    return results


def run_python_gtap():
    """Run Python GTAP model and extract results."""
    print("\n" + "=" * 70)
    print("RUNNING PYTHON GTAP MODEL")
    print("=" * 70)
    
    try:
        from equilibria.templates.gtap import (
            GTAPSets, GTAPParameters, GTAPModelEquations, build_gtap_contract
        )
        from pyomo.environ import SolverFactory, value
        from pyomo.opt import TerminationCondition
        
        # Create test data
        sets = GTAPSets()
        sets.r = ["USA", "EUR", "CHN"]
        sets.i = ["agr", "mfg", "ser"]
        sets.a = ["agr", "mfg", "ser"]
        sets.f = ["lab", "cap"]
        sets.mf = ["lab"]
        sets.sf = ["cap"]
        
        params = GTAPParameters()
        params.sets = sets
        
        # Build and solve
        contract = build_gtap_contract("gtap_cgebox_v1")
        equations = GTAPModelEquations(sets, params, contract.closure)
        model = equations.build_model()
        
        solver = SolverFactory('ipopt')
        if solver is None:
            print("✗ IPOPT not available")
            return None
        
        result = solver.solve(model, tee=False)
        
        if result.solver.termination_condition in [
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal
        ]:
            print("✓ Python model solved successfully")
            
            # Extract results
            py_results = {}
            
            # Production
            py_results['xp'] = { (r, a): float(value(model.xp[r, a])) for r in model.r for a in model.a }
            py_results['px'] = { (r, a): float(value(model.px[r, a])) for r in model.r for a in model.a }
            py_results['pp'] = { (r, a): float(value(model.pp[r, a])) for r in model.r for a in model.a }
            
            # Supply
            py_results['xs'] = { (r, i): float(value(model.xs[r, i])) for r in model.r for i in model.i }
            py_results['ps'] = { (r, i): float(value(model.ps[r, i])) for r in model.r for i in model.i }
            py_results['pd'] = { (r, i): float(value(model.pd[r, i])) for r in model.r for i in model.i }
            
            # Armington
            py_results['xa'] = { (r, i): float(value(model.xa[r, i])) for r in model.r for i in model.i }
            py_results['pa'] = { (r, i): float(value(model.pa[r, i])) for r in model.r for i in model.i }
            py_results['xd'] = { (r, i): float(value(model.xd[r, i])) for r in model.r for i in model.i }
            py_results['xmt'] = { (r, i): float(value(model.xmt[r, i])) for r in model.r for i in model.i }
            py_results['pmt'] = { (r, i): float(value(model.pmt[r, i])) for r in model.r for i in model.i }
            
            # Trade
            py_results['xet'] = { (r, i): float(value(model.xet[r, i])) for r in model.r for i in model.i }
            py_results['pet'] = { (r, i): float(value(model.pet[r, i])) for r in model.r for i in model.i }
            
            # Factors
            py_results['xf'] = { (r, f, a): float(value(model.xf[r, f, a])) for r in model.r for f in model.f for a in model.a }
            py_results['xft'] = { (r, f): float(value(model.xft[r, f])) for r in model.r for f in model.f }
            py_results['pf'] = { (r, f, a): float(value(model.pf[r, f, a])) for r in model.r for f in model.f for a in model.a }
            py_results['pft'] = { (r, f): float(value(model.pft[r, f])) for r in model.r for f in model.f }
            
            # Demand
            py_results['xc'] = { (r, i): float(value(model.xc[r, i])) for r in model.r for i in model.i }
            py_results['xg'] = { (r, i): float(value(model.xg[r, i])) for r in model.r for i in model.i }
            py_results['xi'] = { (r, i): float(value(model.xi[r, i])) for r in model.r for i in model.i }
            
            # Income
            py_results['regy'] = { r: float(value(model.regy[r])) for r in model.r }
            py_results['yc'] = { r: float(value(model.yc[r])) for r in model.r }
            py_results['yg'] = { r: float(value(model.yg[r])) for r in model.r }
            
            return py_results
        else:
            print(f"✗ Python solve failed: {result.solver.termination_condition}")
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(gams_gdx: Path, py_results: Dict, tolerance: float = 1e-4):
    """Compare GAMS and Python results."""
    print("\n" + "=" * 70)
    print("COMPARING RESULTS")
    print("=" * 70)
    
    # Load GAMS results
    result = subprocess.run(
        [GDXDUMP_BIN, str(gams_gdx)],
        capture_output=True, text=True, timeout=30
    )
    
    if result.returncode != 0:
        print("✗ Failed to read GAMS results")
        return False
    
    gams_data = parse_gdxdump(result.stdout)
    
    # Map GAMS output names to Python names
    var_mapping = {
        'xp_out': 'xp', 'px_out': 'px', 'pp_out': 'pp',
        'xs_out': 'xs', 'ps_out': 'ps', 'pd_out': 'pd',
        'xa_out': 'xa', 'pa_out': 'pa', 'xd_out': 'xd', 'xmt_out': 'xmt', 'pmt_out': 'pmt',
        'xet_out': 'xet', 'pet_out': 'pet',
        'xf_out': 'xf', 'xft_out': 'xft', 'pf_out': 'pf', 'pft_out': 'pft',
        'xc_out': 'xc', 'xg_out': 'xg', 'xi_out': 'xi',
        'regy_out': 'regy', 'yc_out': 'yc', 'yg_out': 'yg'
    }
    
    all_mismatches = []
    total_compared = 0
    
    print(f"\n{'Variable':<15} {'GAMS':<12} {'Python':<12} {'Diff':<12} {'Status'}")
    print("-" * 70)
    
    for gams_name, py_name in var_mapping.items():
        if gams_name not in gams_data or py_name not in py_results:
            continue
        
        gams_vals = gams_data[gams_name]
        py_vals = py_results[py_name]
        
        # Compare each key
        for key in set(gams_vals.keys()) | set(py_vals.keys()):
            gams_val = gams_vals.get(key, 0.0)
            py_val = py_vals.get(key, 0.0)
            diff = abs(gams_val - py_val)
            
            total_compared += 1
            
            key_str = str(key) if isinstance(key, tuple) else f"('{key}')"
            var_key = f"{py_name}{key_str}"
            
            if diff <= tolerance:
                status = "✓"
            else:
                status = "✗"
                all_mismatches.append({
                    'var': py_name,
                    'key': key,
                    'gams': gams_val,
                    'python': py_val,
                    'diff': diff
                })
            
            # Print first 5 of each variable
            if len([m for m in all_mismatches if m['var'] == py_name]) < 5 or status == "✗":
                print(f"{var_key:<15} {gams_val:<12.6f} {py_val:<12.6f} {diff:<12.2e} {status}")
    
    # Summary
    print("\n" + "=" * 70)
    n_mismatches = len(all_mismatches)
    match_rate = (1 - n_mismatches / max(total_compared, 1)) * 100
    
    print(f"Variables compared: {total_compared}")
    print(f"Mismatches: {n_mismatches}")
    print(f"Match rate: {match_rate:.1f}%")
    
    if n_mismatches == 0:
        print("\n✓ PARITY CHECK PASSED - Perfect match!")
        return True
    else:
        print(f"\n✗ PARITY CHECK FAILED - {n_mismatches} mismatches found")
        print("\nTop 10 mismatches:")
        sorted_mismatches = sorted(all_mismatches, key=lambda x: x['diff'], reverse=True)[:10]
        for m in sorted_mismatches:
            key_str = str(m['key']) if isinstance(m['key'], tuple) else f"({m['key']})"
            print(f"  {m['var']}{key_str}: GAMS={m['gams']:.6f} Python={m['python']:.6f} Diff={m['diff']:.2e}")
        return False


def main():
    """Main execution."""
    print("=" * 70)
    print("GTAP PARITY: GAMS vs PYTHON")
    print("=" * 70)
    
    output_dir = Path(__file__).parent / "parity_results"
    output_dir.mkdir(exist_ok=True)
    
    # Run GAMS
    gams_ok, gams_gdx = run_gams_gtap(output_dir)
    if not gams_ok:
        print("\n✗ GAMS execution failed")
        return 1
    
    # Run Python
    py_results = run_python_gtap()
    if py_results is None:
        print("\n✗ Python execution failed")
        return 1
    
    # Compare
    passed = compare_results(gams_gdx, py_results, tolerance=1e-4)
    
    print("\n" + "=" * 70)
    if passed:
        print("🎉 SUCCESS: Both models produce identical results!")
        print("=" * 70)
        return 0
    else:
        print("⚠ Models differ - implementation needs alignment")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
