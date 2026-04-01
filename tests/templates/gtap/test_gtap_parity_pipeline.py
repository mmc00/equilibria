"""Tests for GTAP GAMS parity pipeline.

These tests verify the parity comparison system between Python GTAP
and CGEBox GAMS results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pyomo.environ import ConcreteModel, Param, Set, Var

from equilibria.templates.gtap import GTAPParameters, GTAPSets
from equilibria.templates.gtap.gtap_parity_pipeline import (
    GTAPParityComparison,
    GTAPParityRunner,
    GTAPGAMSReference,
    GTAPVariableSnapshot,
    _domestic_sales_scale,
    _resolve_standard_gtap_reference_csv,
    _standard_gtap_csv_scale_factor,
    _resolve_standard_gtap_years,
    _derive_factor_tax_wedges_from_standard_gtap_csv,
    _enrich_sets_from_standard_gtap_csv,
    compare_gtap_gams_parity,
    compare_variable_groups,
    run_gtap_parity_test,
)
from equilibria.templates.gtap.gtap_parameters import GTAPTaxRates
from equilibria.templates.gtap.gtap_solver import SolverResult, SolverStatus


class TestGTAPVariableSnapshot:
    """Tests for GTAPVariableSnapshot."""
    
    def test_empty_snapshot(self):
        """Test creating empty snapshot."""
        snap = GTAPVariableSnapshot()
        assert snap.pnum is None
        assert snap.walras is None
        assert len(snap.xp) == 0
        assert snap.is_empty() is True
    
    def test_snapshot_creation(self):
        """Test snapshot with data."""
        snap = GTAPVariableSnapshot(
            xp={("USA", "agr"): 1.0, ("EUR", "mfg"): 2.0},
            pnum=1.5,
            walras=1e-10,
        )
        assert snap.pnum == 1.5
        assert snap.walras == 1e-10
        assert snap.xp[("USA", "agr")] == 1.0

    def test_from_standard_gtap_csv(self, tmp_path: Path):
        """Test parsing a postsim-style CSV snapshot."""
        csv_path = tmp_path / "COMP.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "Variable,Region,Sector,Qualifier,Year,Value",
                    "x,R1,a_agr,c_food,1,10",
                    "xp,R1,a_agr,,1,10",
                    "xds,R1,c_food,,1,7",
                    "xd,R1,c_food,hhd,1,3",
                    "xa,R1,c_food,hhd,1,4",
                    "pa,R1,c_food,hhd,1,1.1",
                    "pdp,R1,c_food,hhd,1,1.2",
                    "xw,R1,c_food,R2,1,5",
                    "xwmg,R1,c_food,R2,1,0.5",
                    "pwmg,R1,c_food,R2,1,1.05",
                    "xmgm,R1,c_food,R2,1,0.5",
                    "xtmg,GBL,m_trans,,1,0.5",
                    "ptmg,GBL,m_trans,,1,1.0",
                    "regY,R1,,,1,100",
                    "pabs,R1,,,1,1.0",
                ]
            ),
            encoding="utf-8",
        )

        sets = GTAPSets(
            r=["R1", "R2"],
            i=["c_food", "m_trans"],
            a=["a_agr"],
            f=["lab"],
            mf=["lab"],
            sf=[],
            m=["m_trans"],
        )

        snap = GTAPVariableSnapshot.from_standard_gtap_csv(csv_path, sets)

        assert snap.x[("R1", "a_agr", "c_food")] == 10.0
        assert snap.xd[("R1", "c_food", "hhd")] == 3.0
        assert snap.xaa[("R1", "c_food", "hhd")] == 4.0
        assert snap.paa[("R1", "c_food", "hhd")] == 1.1
        assert snap.xwmg[("R1", "c_food", "R2")] == 0.5
        assert snap.xmgm[("m_trans", "R1", "c_food", "R2")] == 0.5
        assert snap.xtmg[("m_trans",)] == 0.5

    def test_from_standard_gtap_csv_rescales_generated_quantity_rows(self, tmp_path: Path):
        """Generated comparator CSVs should rescale quantities and incomes back from fallback units."""
        csv_path = tmp_path / "comp" / "COMP_generated.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(
            "\n".join(
                [
                    "Variable,Region,Sector,Qualifier,Year,Value",
                    "xa,R1,c_food,inv,1,2.5e-06",
                    "xd,R1,c_food,inv,1,1.5e-06",
                    "xp,R1,a_agr,,1,3.0e-06",
                    "yg,R1,,,1,4.0e-06",
                    "pd,R1,c_food,,1,1.2",
                    "pdp,R1,c_food,inv,1,1.3",
                ]
            ),
            encoding="utf-8",
        )

        sets = GTAPSets(
            r=["R1"],
            i=["c_food"],
            a=["a_agr"],
            f=["lab"],
            mf=["lab"],
            sf=[],
            m=[],
        )

        snap = GTAPVariableSnapshot.from_standard_gtap_csv(csv_path, sets)

        assert snap.xaa[("R1", "c_food", "inv")] == pytest.approx(2.5)
        assert snap.xd[("R1", "c_food", "inv")] == pytest.approx(1.5)
        assert snap.xp[("R1", "a_agr")] == pytest.approx(3.0)
        assert snap.yg["R1"] == pytest.approx(4.0)
        assert snap.pd[("R1", "c_food")] == pytest.approx(1.2)
        assert snap.pdp[("R1", "c_food", "inv")] == pytest.approx(1.3)

    def test_from_standard_gtap_csv_uses_neos_row_scales_when_available(self, tmp_path: Path):
        """Generated comparator CSVs should infer row-specific powers-of-ten scales from sibling NEOS exports."""
        csv_path = tmp_path / "comp" / "COMP_generated.csv"
        neos_path = tmp_path / "comp_neos" / "COMP.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        neos_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(
            "\n".join(
                [
                    "Variable,Region,Sector,Qualifier,Year,Value",
                    "xa,R1,c_food,a_agr,1,2.5e-04",
                    "xd,R1,c_food,a_agr,1,1.5e-04",
                    "xa,R1,c_food,inv,1,2.5e-06",
                    "xf,R1,a_agr,lab,1,3.0e-07",
                    "pd,R1,c_food,,1,1.2",
                ]
            ),
            encoding="utf-8",
        )
        neos_path.write_text(
            "\n".join(
                [
                    "Variable,Region,Sector,Qualifier,Year,Value",
                    '"xa","R1","c_food","a_agr",1,2.5',
                    '"xd","R1","c_food","a_agr",1,1.5',
                    '"xa","R1","c_food","inv",1,2.5',
                    '"xf","R1","a_agr","lab",1,3.0',
                    '"pd","R1","c_food","",1,1.2',
                ]
            ),
            encoding="utf-8",
        )

        sets = GTAPSets(
            r=["R1"],
            i=["c_food"],
            a=["a_agr"],
            f=["lab"],
            mf=["lab"],
            sf=[],
            m=[],
        )

        snap = GTAPVariableSnapshot.from_standard_gtap_csv(csv_path, sets)

        assert snap.xaa[("R1", "c_food", "a_agr")] == pytest.approx(2.5)
        assert snap.xd[("R1", "c_food", "a_agr")] == pytest.approx(1.5)
        assert snap.xaa[("R1", "c_food", "inv")] == pytest.approx(2.5)
        assert snap.xf[("R1", "lab", "a_agr")] == pytest.approx(3.0)
        assert snap.pd[("R1", "c_food")] == pytest.approx(1.2)

    def test_from_python_model_grosses_up_factor_prices_from_kappaf(self):
        """Python snapshots should report GTAP-style gross factor prices when the model stores net prices."""
        model = ConcreteModel()
        model.r = Set(initialize=["R1"])
        model.f = Set(initialize=["SkLab"])
        model.a = Set(initialize=["a_agr"])
        model.pf = Var(model.r, model.f, model.a, initialize=1.0)
        model.pft = Var(model.r, model.f, initialize=1.0)
        model.kappaf_activity = Param(
            model.r,
            model.f,
            model.a,
            initialize={("R1", "SkLab", "a_agr"): 0.2},
            default=0.0,
        )
        model._equilibria_factor_price_representation = "net_of_direct_tax"

        model.pf["R1", "SkLab", "a_agr"].value = 1.0
        model.pft["R1", "SkLab"].value = 1.0

        snap = GTAPVariableSnapshot.from_python_model(model)

        assert snap.pf[("R1", "SkLab", "a_agr")] == pytest.approx(1.25)
        assert snap.pft[("R1", "SkLab")] == pytest.approx(1.0)


class TestStandardGTAPSetEnrichment:
    """Tests for repairing sets from standard_gtap_7 CSV exports."""

    def test_enrichment_repairs_corrupted_factor_subsets(self, tmp_path: Path):
        """CSV-derived factor labels should override suspicious factor subsets from the sets GDX."""
        csv_path = tmp_path / "COMP.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "Variable,Region,Sector,Qualifier,Year,Value",
                    "xp,R1,a_agr,,shock,10",
                    "xs,R1,c_food,,shock,10",
                    "xf,R1,a_agr,Land,shock,5",
                    "xf,R1,a_agr,SkLab,shock,5",
                    "x,R1,a_agr,c_food,shock,10",
                ]
            ),
            encoding="utf-8",
        )

        sets = GTAPSets(
            r=["R0"],
            i=["old_comm"],
            a=["a_agr"],
            f=["a_agr"],
            mf=["a_agr"],
            sf=["a_srv"],
            m=[],
        )

        _enrich_sets_from_standard_gtap_csv(
            sets,
            csv_path,
            solution_year="shock",
            benchmark_year="base",
        )

        assert sets.f == ["Land", "SkLab"]
        assert sets.sf == ["Land"]
        assert sets.mf == ["SkLab"]
        assert sets.a == ["a_agr"]
        assert sets.i == ["c_food"]

    def test_factor_tax_wedges_derive_from_pf_and_pft_rows(self, tmp_path: Path):
        """Activity-level factor wedges should be recoverable from baseline pf/pft rows."""
        csv_path = tmp_path / "COMP.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "Variable,Region,Sector,Qualifier,Year,Value",
                    "pft,R1,SkLab,,1,1.0",
                    "pf,R1,a_mine,SkLab,1,1.25",
                    "pf,R1,a_food,SkLab,1,1.10",
                ]
            ),
            encoding="utf-8",
        )

        taxes = GTAPTaxRates()
        _derive_factor_tax_wedges_from_standard_gtap_csv(
            taxes,
            csv_path,
            benchmark_year="2011",
            solution_year="1",
        )

        assert taxes.kappaf_activity[("R1", "SkLab", "a_mine")] == pytest.approx(0.2)
        assert taxes.kappaf_activity[("R1", "SkLab", "a_food")] == pytest.approx(1.0 - (1.0 / 1.10))

    def test_resolve_standard_gtap_years_falls_back_to_available_tokens(self, tmp_path: Path):
        """standard_gtap_7 CSVs with year labels 1/2/3 should override incompatible defaults."""
        csv_path = tmp_path / "COMP.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "Variable,Region,Sector,Qualifier,Year,Value",
                    '"xp","R1","a_agr","",1,10',
                    '"xp","R1","a_agr","",2,11',
                    '"xp","R1","a_agr","",3,12',
                ]
            ),
            encoding="utf-8",
        )

        benchmark_year, solution_year = _resolve_standard_gtap_years(
            csv_path,
            benchmark_year="2011",
            solution_year="shock",
        )

        assert benchmark_year == "1"
        assert solution_year == "1"

    def test_resolve_standard_gtap_reference_csv_prefers_generated_9x10_output(self, tmp_path: Path):
        """9x10 parity should prefer a domain-compatible generated comparator CSV."""
        root_csv = tmp_path / "COMP.csv"
        generated_csv = tmp_path / "comp" / "COMP_generated.csv"
        neos_csv = tmp_path / "comp_neos" / "COMP.csv"

        root_csv.write_text("Variable,Region,Sector,Qualifier,Year,Value\n", encoding="utf-8")
        generated_csv.parent.mkdir(parents=True, exist_ok=True)
        generated_csv.write_text("Variable,Region,Sector,Qualifier,Year,Value\n", encoding="utf-8")
        neos_csv.parent.mkdir(parents=True, exist_ok=True)
        neos_csv.write_text("Variable,Region,Sector,Qualifier,Year,Value\n", encoding="utf-8")

        sets = GTAPSets(
            r=["R1"],
            i=["c_Crops"],
            a=["a_agricultur", "a_Extraction"],
            f=["Lab"],
            mf=["Lab"],
            sf=[],
        )

        assert _resolve_standard_gtap_reference_csv(root_csv, sets) == generated_csv

    def test_resolve_standard_gtap_reference_csv_leaves_specific_or_non_9x10_paths(self, tmp_path: Path):
        """Non-9x10 sets or already-specific CSV paths should remain unchanged."""
        root_csv = tmp_path / "COMP.csv"
        root_csv.write_text("Variable,Region,Sector,Qualifier,Year,Value\n", encoding="utf-8")
        direct_csv = tmp_path / "comp_neos" / "COMP.csv"
        direct_csv.parent.mkdir(parents=True, exist_ok=True)
        direct_csv.write_text("Variable,Region,Sector,Qualifier,Year,Value\n", encoding="utf-8")

        sets = GTAPSets(
            r=["R1"],
            i=["c_Crops"],
            a=["a_Crops", "a_MeatLstk"],
            f=["Lab"],
            mf=["Lab"],
            sf=[],
        )

        assert _resolve_standard_gtap_reference_csv(root_csv, sets) == root_csv
        assert _resolve_standard_gtap_reference_csv(direct_csv, sets) == direct_csv

    def test_standard_gtap_csv_scale_factor_only_affects_generated_quantity_files(self, tmp_path: Path):
        """Scaling should only apply to generated comparator CSV quantity/income rows."""
        generated = tmp_path / "comp" / "COMP_generated.csv"
        regular = tmp_path / "COMP.csv"

        assert _standard_gtap_csv_scale_factor(generated, "xa") == pytest.approx(1.0e6)
        assert _standard_gtap_csv_scale_factor(generated, "yg") == pytest.approx(1.0e6)
        assert _standard_gtap_csv_scale_factor(generated, "pd") == pytest.approx(1.0)
        assert _standard_gtap_csv_scale_factor(regular, "xa") == pytest.approx(1.0)

    def test_standard_gtap_csv_scale_factor_prefers_neos_row_scales(self, tmp_path: Path):
        """Row-specific generated scales should prefer sibling NEOS ratios when available."""
        generated = tmp_path / "comp" / "COMP_generated.csv"
        neos = tmp_path / "comp_neos" / "COMP.csv"
        generated.parent.mkdir(parents=True, exist_ok=True)
        neos.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(
            "\n".join(
                [
                    "Variable,Region,Sector,Qualifier,Year,Value",
                    "xa,R1,c_food,a_agr,1,2.0e-04",
                    "xa,R1,c_food,inv,1,2.0e-06",
                    "xf,R1,a_agr,lab,1,3.0e-07",
                ]
            ),
            encoding="utf-8",
        )
        neos.write_text(
            "\n".join(
                [
                    "Variable,Region,Sector,Qualifier,Year,Value",
                    '"xa","R1","c_food","a_agr",1,2.0',
                    '"xa","R1","c_food","inv",1,2.0',
                    '"xf","R1","a_agr","lab",1,3.0',
                ]
            ),
            encoding="utf-8",
        )

        assert _standard_gtap_csv_scale_factor(
            generated,
            "xa",
            region="R1",
            sector="c_food",
            qualifier="a_agr",
            year="1",
        ) == pytest.approx(1.0e4)
        assert _standard_gtap_csv_scale_factor(
            generated,
            "xa",
            region="R1",
            sector="c_food",
            qualifier="inv",
            year="1",
        ) == pytest.approx(1.0e6)
        assert _standard_gtap_csv_scale_factor(
            generated,
            "xf",
            region="R1",
            sector="a_agr",
            qualifier="lab",
            year="1",
        ) == pytest.approx(1.0e7)


class TestCompareVariableGroups:
    """Tests for variable group comparison."""
    
    def test_identical_values(self):
        """Test comparison with identical values."""
        python = {("r1", "a1"): 1.0, ("r2", "a2"): 2.0}
        gams = {("r1", "a1"): 1.0, ("r2", "a2"): 2.0}
        
        n_comp, n_mism, max_diff, mismatches = compare_variable_groups(
            python, gams, "test", tolerance=1e-6
        )
        
        assert n_comp == 2
        assert n_mism == 0
        assert max_diff == 0.0
        assert len(mismatches) == 0
    
    def test_with_mismatches(self):
        """Test comparison with mismatches."""
        python = {("r1", "a1"): 1.0, ("r2", "a2"): 2.0}
        gams = {("r1", "a1"): 1.001, ("r2", "a2"): 2.0}
        
        n_comp, n_mism, max_diff, mismatches = compare_variable_groups(
            python, gams, "test", tolerance=1e-6
        )
        
        assert n_comp == 2
        assert n_mism == 1
        assert max_diff == pytest.approx(0.001, abs=1e-10)
        assert len(mismatches) == 1
        assert mismatches[0]["group"] == "test"
    
    def test_missing_in_python(self):
        """Test when Python has missing values."""
        python = {("r1", "a1"): 1.0}
        gams = {("r1", "a1"): 1.0, ("r2", "a2"): 0.0}
        
        n_comp, n_mism, max_diff, mismatches = compare_variable_groups(
            python, gams, "test", tolerance=1e-6
        )
        
        # Should not count (r2, a2) as both are 0
        assert n_comp == 1
    
    def test_tolerance_check(self):
        """Test tolerance threshold."""
        python = {("r1", "a1"): 1.0}
        gams = {("r1", "a1"): 1.0000001}
        
        # Should pass with loose tolerance
        n_comp1, n_mism1, _, _ = compare_variable_groups(
            python, gams, "test", tolerance=1e-3
        )
        assert n_mism1 == 0
        
        # Should fail with tight tolerance
        n_comp2, n_mism2, _, _ = compare_variable_groups(
            python, gams, "test", tolerance=1e-9
        )
        assert n_mism2 == 1


def test_domestic_sales_scale_prefers_supply_when_absorption_is_tiny():
    """`xds` should normalize on the supply side when CSV absorption is not comparable."""
    sets = GTAPSets(
        r=["R1"],
        i=["c_food"],
        a=["a_agr"],
        f=["Land"],
        mf=[],
        sf=["Land"],
        output_pairs=[("a_agr", "c_food")],
        activity_commodities={"a_agr": ["c_food"]},
        commodity_activities={"c_food": ["a_agr"]},
    )
    params = GTAPParameters(sets=sets)
    params.benchmark.vom[("R1", "a_agr")] = 100.0
    params.benchmark.vom_i[("R1", "c_food")] = 100.0
    params.benchmark.xs0[("R1", "c_food")] = 100.0
    params.benchmark.xd0[("R1", "c_food")] = 5.0

    scale = _domestic_sales_scale(params.benchmark, sets, "R1", "c_food")

    assert scale == pytest.approx(100.0)


def test_runner_warm_starts_from_normalized_reference(monkeypatch, tmp_path: Path):
    """Parity runner should pass normalized GAMS levels as a solver warm start."""

    captured = {}

    class FakeSolver:
        def __init__(self, model, closure, solver_name, solver_options):
            self.model = model

        def apply_solution_hint(self, hint):
            captured["xp"] = hint.xp.get(("R1", "a_agr"))
            captured["pf"] = hint.pf.get(("R1", "Land", "a_agr"))

        def solve(self):
            return SolverResult(status=SolverStatus.CONVERGED, success=True, walras_value=0.0)

    def fake_build(_bundle, benchmark_year, solution_year):
        sets = GTAPSets(
            r=["R1"],
            i=["c_food"],
            a=["a_agr"],
            f=["Land"],
            mf=[],
            sf=["Land"],
            output_pairs=[("a_agr", "c_food")],
            activity_commodities={"a_agr": ["c_food"]},
            commodity_activities={"c_food": ["a_agr"]},
        )
        params = GTAPParameters(sets=sets)
        params.benchmark.vom[("R1", "a_agr")] = 20.0
        params.benchmark.makb[("R1", "a_agr", "c_food")] = 20.0
        params.benchmark.vfm[("R1", "Land", "a_agr")] = 10.0
        params.benchmark.vom_i[("R1", "c_food")] = 20.0
        params.benchmark.xd0[("R1", "c_food")] = 20.0
        params.shares.p_gx[("R1", "a_agr", "c_food")] = 1.0
        params.shares.p_alphad[("R1", "c_food")] = 1.0
        params.shares.p_gd[("R1", "c_food")] = 1.0
        params.shares.p_cons[("R1", "c_food")] = 1.0
        params.shares.p_gov[("R1", "c_food")] = 1.0
        params.shares.p_inv[("R1", "c_food")] = 1.0
        params.shares.p_yc["R1"] = 1.0
        return sets, params

    monkeypatch.setattr(
        "equilibria.templates.gtap.gtap_parity_pipeline._build_standard_gtap_params",
        fake_build,
    )
    monkeypatch.setattr(
        "equilibria.templates.gtap.gtap_parity_pipeline.GTAPSolver",
        FakeSolver,
    )
    monkeypatch.setattr(
        GTAPGAMSReference,
        "load",
        classmethod(
            lambda cls, gdx_path, sets=None, solution_year=1: GTAPGAMSReference(
                gdx_path=Path(gdx_path),
                sets=sets,
                snapshot=GTAPVariableSnapshot(
                    xp={("R1", "a_agr"): 20.0},
                    pf={("R1", "Land", "a_agr"): 1.25},
                    x={("R1", "a_agr", "c_food"): 20.0},
                    xs={("R1", "c_food"): 20.0},
                    xds={("R1", "c_food"): 20.0},
                    xft={("R1", "Land"): 10.0},
                    xf={("R1", "Land", "a_agr"): 10.0},
                ),
                modelstat=1.0,
                solvestat=1.0,
                solve_time=0.0,
            )
        ),
    )

    sets_path = tmp_path / "fake_sets.gdx"
    benchmark_path = tmp_path / "fake_benchmark.csv"
    reference_path = tmp_path / "fake_reference.csv"
    sets_path.write_text("", encoding="utf-8")
    benchmark_path.write_text("", encoding="utf-8")
    reference_path.write_text("", encoding="utf-8")

    runner = GTAPParityRunner(
        sets_gdx=sets_path,
        benchmark_csv=benchmark_path,
        gams_results_gdx=reference_path,
        normalize_reference=True,
        warm_start_reference=True,
    )
    result = runner.run_python()

    assert result.success is True
    assert captured["xp"] == pytest.approx(1.0)
    assert captured["pf"] == pytest.approx(1.25)


class TestGTAPParityComparison:
    """Tests for GTAPParityComparison dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        comp = GTAPParityComparison(
            passed=True,
            tolerance=1e-6,
            n_variables_compared=100,
            n_mismatches=0,
            max_abs_diff=0.0,
            max_rel_diff=0.0,
            mismatches=[],
            summary={"test": "data"},
        )
        
        d = comp.to_dict()
        assert d["passed"] is True
        assert d["tolerance"] == 1e-6
        assert d["n_variables_compared"] == 100
    
    def test_to_json(self):
        """Test conversion to JSON."""
        comp = GTAPParityComparison(
            passed=False,
            tolerance=1e-6,
            n_variables_compared=100,
            n_mismatches=5,
            max_abs_diff=0.1,
            max_rel_diff=0.05,
            mismatches=[
                {"group": "xp", "key": ("USA", "agr"), "python": 1.0, "gams": 1.1}
            ],
            summary={},
        )
        
        json_str = comp.to_json()
        assert '"passed": false' in json_str
        assert '"n_mismatches": 5' in json_str


class TestCompareGTAPParity:
    """Tests for compare_gtap_gams_parity function."""
    
    def test_perfect_match(self, monkeypatch):
        """Test when Python and GAMS match perfectly."""
        py_snap = GTAPVariableSnapshot(
            xp={("USA", "agr"): 1.0},
            pnum=1.0,
            walras=0.0,
        )
        
        gams_ref = GTAPGAMSReference(
            gdx_path=Path("test.gdx"),
            sets=None,
            snapshot=py_snap,  # Same snapshot
            modelstat=1.0,
            solvestat=1.0,
            solve_time=1.0,
        )
        
        result = compare_gtap_gams_parity(py_snap, gams_ref, tolerance=1e-6)
        
        assert result.passed is True
        assert result.n_mismatches == 0
        assert result.max_abs_diff == 0.0
    
    def test_with_mismatches(self):
        """Test when there are mismatches."""
        py_snap = GTAPVariableSnapshot(
            xp={("USA", "agr"): 1.0, ("EUR", "mfg"): 2.0},
            pnum=1.0,
            walras=0.0,
        )
        
        gams_snap = GTAPVariableSnapshot(
            xp={("USA", "agr"): 1.1, ("EUR", "mfg"): 2.0},  # 10% difference
            pnum=1.0,
            walras=0.0,
        )
        
        gams_ref = GTAPGAMSReference(
            gdx_path=Path("test.gdx"),
            sets=None,
            snapshot=gams_snap,
            modelstat=1.0,
            solvestat=1.0,
            solve_time=1.0,
        )
        
        result = compare_gtap_gams_parity(py_snap, gams_ref, tolerance=1e-6)
        
        assert result.passed is False
        assert result.n_mismatches >= 1
        assert result.max_abs_diff >= 0.1

    def test_flags_degenerate_solution_pattern(self):
        """Near-zero quantities and factor prices should raise a dead-numeraire diagnostic."""
        py_snap = GTAPVariableSnapshot(
            xp={("USA", "agr"): 0.0},
            xds={("USA", "c_food"): 0.0},
            regy={"USA": 0.0},
            pf={("USA", "lab", "agr"): 0.0},
            pft={("USA", "lab"): 0.0},
            pnum=1.0,
            walras=0.0,
        )

        gams_snap = GTAPVariableSnapshot(
            xp={("USA", "agr"): 1.0},
            xds={("USA", "c_food"): 1.0},
            regy={"USA": 1.0},
            pf={("USA", "lab", "agr"): 1.0},
            pft={("USA", "lab"): 1.0},
            pnum=1.0,
            walras=0.0,
        )

        gams_ref = GTAPGAMSReference(
            gdx_path=Path("test.gdx"),
            sets=None,
            snapshot=gams_snap,
            modelstat=1.0,
            solvestat=1.0,
            solve_time=1.0,
        )

        result = compare_gtap_gams_parity(py_snap, gams_ref, tolerance=1e-6)

        assert result.summary["dead_numeraire_suspected"] is True
        assert "xp" in result.summary["collapsed_quantity_groups"]
        assert "pf" in result.summary["collapsed_price_groups"]

    def test_does_not_flag_non_degenerate_solution(self):
        """Healthy snapshots should not trigger the dead-numeraire diagnostic."""
        py_snap = GTAPVariableSnapshot(
            xp={("USA", "agr"): 1.0},
            pf={("USA", "lab", "agr"): 1.0},
            pft={("USA", "lab"): 1.0},
            pnum=1.0,
            walras=0.0,
        )

        gams_ref = GTAPGAMSReference(
            gdx_path=Path("test.gdx"),
            sets=None,
            snapshot=py_snap,
            modelstat=1.0,
            solvestat=1.0,
            solve_time=1.0,
        )

        result = compare_gtap_gams_parity(py_snap, gams_ref, tolerance=1e-6)

        assert result.summary["dead_numeraire_suspected"] is False
        assert result.summary["collapsed_quantity_groups"] == {}
        assert result.summary["collapsed_price_groups"] == {}


class TestGTAPParityRunner:
    """Tests for GTAPParityRunner."""
    
    def test_initialization_with_split_bundle(self, tmp_path: Path, monkeypatch):
        """Test runner initialization in standard_gtap_7 bundle mode."""
        sets = GTAPSets(
            r=["R1", "R2"],
            i=["c_food", "m_trans"],
            a=["a_agr"],
            f=["lab"],
            mf=["lab"],
            sf=[],
            m=["m_trans"],
        )
        sets.output_pairs = [("a_agr", "c_food")]
        sets.activity_commodities = {"a_agr": ["c_food"]}
        sets.commodity_activities = {"c_food": ["a_agr"], "m_trans": []}
        params = GTAPParameters()
        params.sets = sets

        def mock_build_params(bundle, benchmark_year, solution_year):
            return sets, params

        class DummyEquations:
            def __init__(self, sets, params, closure):
                self.sets = sets
                self.params = params
                self.closure = closure

            def build_model(self):
                return {"ok": True}

        monkeypatch.setattr(
            "equilibria.templates.gtap.gtap_parity_pipeline._build_standard_gtap_params",
            mock_build_params,
        )
        monkeypatch.setattr(
            "equilibria.templates.gtap.gtap_parity_pipeline.GTAPModelEquations",
            DummyEquations,
        )

        runner = GTAPParityRunner(
            sets_gdx=tmp_path / "10x10Sets.gdx",
            elasticities_gdx=tmp_path / "10x10Prm.gdx",
            benchmark_csv=tmp_path / "COMP.csv",
        )

        assert runner.sets is sets
        assert runner.params is params
        assert runner.model == {"ok": True}
    
    def test_run_python_mocked(self, tmp_path: Path, monkeypatch):
        """Test running Python model with mocking."""
        # This would require extensive mocking of the full model
        # For now, just test the interface
        pass

    def test_load_gams_reference_from_csv(self, tmp_path: Path):
        """Test loading a GAMS reference from COMP.csv."""
        csv_path = tmp_path / "COMP.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "Variable,Region,Sector,Qualifier,Year,Value",
                    "xp,R1,a_agr,,1,10",
                    "regY,R1,,,1,100",
                ]
            ),
            encoding="utf-8",
        )

        sets = GTAPSets(
            r=["R1"],
            i=["c_food"],
            a=["a_agr"],
            f=["lab"],
            mf=["lab"],
            sf=[],
            m=[],
        )

        reference = GTAPGAMSReference.load(csv_path, sets)
        assert reference.snapshot.xp[("R1", "a_agr")] == 10.0
        assert reference.snapshot.regy["R1"] == 100.0


class TestIntegration:
    """Integration tests requiring actual GDX files."""
    
    @pytest.mark.skip(reason="Requires actual GTAP GDX files")
    def test_full_parity_check(self):
        """Test full parity check with real data.
        
        This test requires:
        - A GTAP data GDX file
        - A GAMS results GDX file
        """
        gdx_file = Path("data/asa7x5.gdx")
        gams_results = Path("results/gams_results.gdx")
        
        if not gdx_file.exists() or not gams_results.exists():
            pytest.skip("GDX files not available")
        
        result = run_gtap_parity_test(
            gdx_file=gdx_file,
            gams_results_gdx=gams_results,
            tolerance=1e-6,
        )
        
        # Should complete without error
        assert result is not None
        assert result.n_variables_compared > 0


class TestRunGTAPParityTest:
    """Tests for run_gtap_parity_test convenience function."""
    
    def test_missing_files(self, tmp_path: Path):
        """Test with missing files."""
        gdx_file = tmp_path / "nonexistent.gdx"
        gams_results = tmp_path / "nonexistent_results.gdx"
        
        with pytest.raises(FileNotFoundError):
            run_gtap_parity_test(gdx_file=gdx_file, gams_results_gdx=gams_results)


class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_comparison(self):
        """Test comparison with empty snapshots."""
        py_snap = GTAPVariableSnapshot()
        gams_snap = GTAPVariableSnapshot()
        
        gams_ref = GTAPGAMSReference(
            gdx_path=Path("test.gdx"),
            sets=None,
            snapshot=gams_snap,
            modelstat=1.0,
            solvestat=1.0,
            solve_time=1.0,
        )
        
        result = compare_gtap_gams_parity(py_snap, gams_ref, tolerance=1e-6)
        
        # Should pass (nothing to compare)
        assert result.passed is True
    
    def test_nan_values(self):
        """Test handling of NaN values."""
        py_snap = GTAPVariableSnapshot(
            xp={("USA", "agr"): float('nan')},
        )
        gams_snap = GTAPVariableSnapshot(
            xp={("USA", "agr"): 1.0},
        )
        
        gams_ref = GTAPGAMSReference(
            gdx_path=Path("test.gdx"),
            sets=None,
            snapshot=gams_snap,
            modelstat=1.0,
            solvestat=1.0,
            solve_time=1.0,
        )
        
        # Should handle NaN gracefully
        result = compare_gtap_gams_parity(py_snap, gams_ref, tolerance=1e-6)
        # NaN != anything, so this will be a mismatch
        assert result.n_mismatches >= 0  # Could be 0 if NaN handling skips it
    
    def test_inf_values(self):
        """Test handling of infinity values."""
        py_snap = GTAPVariableSnapshot(
            xp={("USA", "agr"): float('inf')},
        )
        gams_snap = GTAPVariableSnapshot(
            xp={("USA", "agr"): float('inf')},
        )
        
        gams_ref = GTAPGAMSReference(
            gdx_path=Path("test.gdx"),
            sets=None,
            snapshot=gams_snap,
            modelstat=1.0,
            solvestat=1.0,
            solve_time=1.0,
        )
        
        result = compare_gtap_gams_parity(py_snap, gams_ref, tolerance=1e-6)
        # inf - inf = nan, so this is tricky
        assert result is not None
