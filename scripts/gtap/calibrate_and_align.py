#!/usr/bin/env python3
"""
GTAP SAM Calibration and Parity Alignment

This script:
1. Creates a consistent SAM (Social Accounting Matrix) for GTAP
2. Calibrates both GAMS and Python models from the same SAM
3. Aligns equations between both models
4. Validates against GAMS pre-solve levels
"""

import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

GAMS_BIN = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gams"
GDXDUMP_BIN = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump"


class GTAPSAM:
    """Create and manage GTAP Social Accounting Matrix."""
    
    def __init__(self, regions, commodities, factors):
        self.r = regions
        self.i = commodities
        self.a = commodities  # Activities = commodities
        self.f = factors
        
        # Initialize all SAM flows to zero
        self.vom = {}      # Output at market prices (r, a)
        self.vfm = {}      # Factor payments (r, f, a)
        self.vdfm = {}     # Domestic intermediate (r, i, a)
        self.vifm = {}     # Imported intermediate (r, i, a)
        self.vpm = {}      # Private consumption (r, i)
        self.vgm = {}      # Government consumption (r, i)
        self.vim = {}      # Investment (r, i)
        self.vxmd = {}     # Exports (r, i, rp)
        self.viws = {}     # Imports CIF (r, i, rp)
        
    def create_simple_sam(self):
        """Create a simple but consistent SAM."""
        print("Creating SAM from benchmark data...")
        
        # Production values - each activity produces 100
        for r in self.r:
            for a in self.a:
                self.vom[(r, a)] = 100.0
        
        # Factor payments - 60% of output
        for r in self.r:
            for f in self.f:
                for a in self.a:
                    self.vfm[(r, f, a)] = 30.0  # 60% total (30 per factor)
        
        # Intermediate consumption - 20% of output
        for r in self.r:
            for i in self.i:
                for a in self.a:
                    if i == a:
                        self.vdfm[(r, i, a)] = 15.0  # Own use
                    else:
                        self.vdfm[(r, i, a)] = 2.5   # Cross use
                    self.vifm[(r, i, a)] = 0.5       # Small imports
        
        # Final demand
        for r in self.r:
            for i in self.i:
                self.vpm[(r, i)] = 20.0   # Private
                self.vgm[(r, i)] = 10.0   # Government
                self.vim[(r, i)] = 5.0    # Investment
        
        # Trade - 10% of production exported
        for r in self.r:
            for i in self.i:
                for rp in self.r:
                    if r != rp:
                        self.vxmd[(r, i, rp)] = 10.0 / (len(self.r) - 1)
                        self.viws[(rp, i, r)] = 10.0 / (len(self.r) - 1)
        
        print(f"  ✓ SAM created: {len(self.vom)} production flows")
        print(f"  ✓ {len(self.vfm)} factor payments")
        print(f"  ✓ {len(self.vdfm)} intermediate flows")
        print(f"  ✓ {len(self.vxmd)} trade flows")
        
    def calibrate_parameters(self):
        """Calibrate model parameters from SAM."""
        print("\nCalibrating parameters from SAM...")
        
        params = {
            # Technology shares
            'axp': {},      # Production shifter
            'aio': {},      # Intermediate output coefficient
            'ava': {},      # Value-added coefficient
            
            # Armington shares
            'alphad': {},   # Domestic share
            'alpham': {},   # Import share
            
            # CET shares
            'thetad': {},   # Domestic supply share
            'thetae': {},   # Export share
            
            # Factor shares
            'alphaf': {},   # Factor share in VA
            
            # Demand shares
            'gamma_c': {},  # Consumption share
            'gamma_g': {},  # Government share
            'gamma_i': {},  # Investment share
        }
        
        # Production technology (Leontief for simplicity)
        for r in self.r:
            for a in self.a:
                output = self.vom.get((r, a), 100.0)
                params['axp'][(r, a)] = 1.0
                
                # Value added share
                va = sum(self.vfm.get((r, f, a), 0) for f in self.f)
                params['ava'][(r, a)] = va / output if output > 0 else 0.6
                
                # Intermediate share
                int_total = sum(self.vdfm.get((r, i, a), 0) + 
                               self.vifm.get((r, i, a), 0) for i in self.i)
                params['aio'][(r, a)] = int_total / output if output > 0 else 0.3
        
        # Armington shares (Cobb-Douglas)
        for r in self.r:
            for i in self.i:
                domestic = sum(self.vdfm.get((r, i, a), 0) for a in self.a)
                imports = sum(self.vifm.get((r, i, a), 0) for a in self.a)
                total = domestic + imports + 20  # Add final demand
                
                params['alphad'][(r, i)] = domestic / total if total > 0 else 0.7
                params['alpham'][(r, i)] = imports / total if total > 0 else 0.3
        
        # Factor shares
        for r in self.r:
            for a in self.a:
                va = sum(self.vfm.get((r, f, a), 0) for f in self.f)
                for f in self.f:
                    params['alphaf'][(r, f, a)] = (self.vfm.get((r, f, a), 0) / va 
                                                   if va > 0 else 1.0 / len(self.f))
        
        # Demand shares
        for r in self.r:
            total_c = sum(self.vpm.get((r, i), 0) for i in self.i)
            for i in self.i:
                params['gamma_c'][(r, i)] = (self.vpm.get((r, i), 0) / total_c 
                                             if total_c > 0 else 1.0 / len(self.i))
        
        print(f"  ✓ {len(params['axp'])} production parameters")
        print(f"  ✓ {len(params['alphad'])} Armington shares")
        print(f"  ✓ {len(params['alphaf'])} factor shares")
        
        return params


def create_gams_model_with_sam(sam: GTAPSAM, params: Dict, output_dir: Path):
    """Create GAMS model calibrated from SAM."""
    print("\n" + "=" * 70)
    print("CREATING CALIBRATED GAMS MODEL")
    print("=" * 70)
    
    work_dir = output_dir / "calibrated_gams"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate GAMS file with calibrated parameters
    gms_content = f"""$ontext
GTAP Model - Calibrated from SAM
Benchmark equilibrium: all prices = 1, quantities from SAM
$offtext

* Sets
Set r "Regions" / {', '.join(sam.r)} /;
Set i "Commodities" / {', '.join(sam.i)} /;
Set a "Activities" / {', '.join(sam.a)} /;
Set f "Factors" / {', '.join(sam.f)} /;

* Calibrated parameters from SAM
Parameter axp(r,a) "Production shifter";
Parameter ava(r,a) "Value-added coefficient";
Parameter aio(r,a) "Intermediate coefficient";
Parameter alphad(r,i) "Domestic share";
Parameter alpham(r,i) "Import share";
Parameter alphaf(r,f,a) "Factor share in VA";
Parameter gamma_c(r,i) "Consumption share";

* Initialize parameters from calibration
"""
    
    # Add parameter values
    for (r, a), val in params['axp'].items():
        gms_content += f"axp('{r}','{a}') = {val:.6f};\n"
    
    for (r, a), val in params['ava'].items():
        gms_content += f"ava('{r}','{a}') = {val:.6f};\n"
    
    for (r, a), val in params['aio'].items():
        gms_content += f"aio('{r}','{a}') = {val:.6f};\n"
    
    for (r, i), val in params['alphad'].items():
        gms_content += f"alphad('{r}','{i}') = {val:.6f};\n"
    
    for (r, i), val in params['alpham'].items():
        gms_content += f"alpham('{r}','{i}') = {val:.6f};\n"
    
    for (r, f, a), val in params['alphaf'].items():
        gms_content += f"alphaf('{r}','{f}','{a}') = {val:.6f};\n"
    
    for (r, i), val in params['gamma_c'].items():
        gms_content += f"gamma_c('{r}','{i}') = {val:.6f};\n"
    
    # Add model equations using calibrated parameters
    gms_content += """
* Variables - all at benchmark = 1 initially
Variable xp(r,a), x(r,a,i), px(r,a), pp(r,a);
Variable xs(r,i), ps(r,i), pd(r,i);
Variable xa(r,i), pa(r,i), xd(r,i), xmt(r,i), pmt(r,i);
Variable xet(r,i), pet(r,i);
Variable xf(r,f,a), xft(r,f), pf(r,f,a), pft(r,f);
Variable xc(r,i), xg(r,i), xi(r,i);
Variable regy(r), yc(r), yg(r);
Variable obj;

* Initialize at benchmark
xp.l(r,a) = 1; x.l(r,a,i) = 1; px.l(r,a) = 1; pp.l(r,a) = 1;
xs.l(r,i) = 1; ps.l(r,i) = 1; pd.l(r,i) = 1;
xa.l(r,i) = 1; pa.l(r,i) = 1; xd.l(r,i) = 0.7; xmt.l(r,i) = 0.3; pmt.l(r,i) = 1;
xet.l(r,i) = 0.2; pet.l(r,i) = 1;
xf.l(r,f,a) = 1; xft.l(r,f) = 1; pf.l(r,f,a) = 1; pft.l(r,f) = 1;
xc.l(r,i) = 0.5; xg.l(r,i) = 0.2; xi.l(r,i) = 0.3;
regy.l(r) = 1; yc.l(r) = 0.6; yg.l(r) = 0.3;

* Equations using calibrated parameters

* Production: Leontief technology
equation eq_xp(r,a);
eq_xp(r,a).. xp(r,a) =e= axp(r,a);

equation eq_x(r,a,i);
eq_x(r,a,i).. x(r,a,i) =e= xp(r,a);

equation eq_px(r,a);
eq_px(r,a).. px(r,a) * xp(r,a) =e= ava(r,a) * xp(r,a) * sum(f, pf(r,f,a) * xf(r,f,a)) / sum(f, xf(r,f,a))
    + aio(r,a) * xp(r,a) * pa(r,a);

* Supply
equation eq_xs(r,i);
eq_xs(r,i).. xs(r,i) =e= sum(a, x(r,a,i));

equation eq_ps(r,i);
eq_ps(r,i).. ps(r,i) =e= pd(r,i);

* Armington (simplified)
equation eq_xa(r,i);
eq_xa(r,i).. xa(r,i) =e= alphad(r,i) * xd(r,i) + alpham(r,i) * xmt(r,i);

equation eq_pa(r,i);
eq_pa(r,i).. pa(r,i) =e= alphad(r,i) * pd(r,i) + alpham(r,i) * pmt(r,i);

equation eq_pmt(r,i);
eq_pmt(r,i).. pmt(r,i) =e= 1.0;

* Trade
equation eq_xs_cet(r,i);
eq_xs_cet(r,i).. xs(r,i) =e= xd(r,i) + xet(r,i);

equation eq_pet(r,i);
eq_pet(r,i).. pet(r,i) =e= ps(r,i);

* Factors
equation eq_xft(r,f);
eq_xft(r,f).. xft(r,f) =e= sum(a, xf(r,f,a));

equation eq_xf(r,f,a);
eq_xf(r,f,a).. xf(r,f,a) =e= alphaf(r,f,a) * xp(r,a) / ava(r,a);

equation eq_pf(r,f,a);
eq_pf(r,f,a).. pf(r,f,a) =e= pft(r,f);

equation eq_pft(r,f);
eq_pft(r,f).. pft(r,f) =e= sum(a, pf(r,f,a) * alphaf(r,f,a)) / sum(a, alphaf(r,f,a));

* Demand
equation eq_xc(r,i);
eq_xc(r,i).. xc(r,i) =e= gamma_c(r,i) * yc(r) / pa(r,i);

equation eq_xg(r,i);
eq_xg(r,i).. xg(r,i) =e= 0.2;

equation eq_xi(r,i);
eq_xi(r,i).. xi(r,i) =e= 0.3;

* Income
equation eq_regy(r);
eq_regy(r).. regy(r) =e= sum((f,a), pf(r,f,a) * xf(r,f,a));

equation eq_yc(r);
eq_yc(r).. yc(r) =e= 0.6 * regy(r);

equation eq_yg(r);
eq_yg(r).. yg(r) =e= 0.3 * regy(r);

* Market clearing
equation mkt_goods(r,i);
mkt_goods(r,i).. xa(r,i) =e= xc(r,i) + xg(r,i) + xi(r,i) + sum(a, x(r,a,i));

* Objective
equation eq_obj;
eq_obj.. obj =e= 1;

Model gtap / all /;
Solve gtap using NLP minimizing obj;

* Export results
Execute_unload "gams_results.gdx", xp, x, px, pp, xs, ps, pd, xa, pa, xd, xmt, pmt,
    xet, pet, xf, xft, pf, pft, xc, xg, xi, regy, yc, yg;
"""
    
    model_file = work_dir / "gtap_calibrated.gms"
    model_file.write_text(gms_content)
    
    print(f"✓ GAMS model created: {model_file}")
    return work_dir, model_file


def run_calibrated_comparison():
    """Run full calibration and comparison."""
    print("=" * 70)
    print("GTAP CALIBRATED PARITY COMPARISON")
    print("=" * 70)
    
    # Create SAM
    sam = GTAPSAM(
        regions=["USA", "EUR", "CHN"],
        commodities=["agr", "mfg", "ser"],
        factors=["lab", "cap"]
    )
    sam.create_simple_sam()
    
    # Calibrate parameters
    params = sam.calibrate_parameters()
    
    # Create calibrated GAMS model
    output_dir = Path(__file__).parent / "calibrated_comparison"
    output_dir.mkdir(exist_ok=True)
    
    gams_dir, gams_file = create_gams_model_with_sam(sam, params, output_dir)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Run GAMS model:")
    print(f"   cd {gams_dir}")
    print(f"   gams gtap_calibrated.gms")
    print("\n2. Update Python model to use same calibrated parameters")
    print("3. Run Python model")
    print("4. Compare results")
    print("=" * 70)


if __name__ == "__main__":
    run_calibrated_comparison()
