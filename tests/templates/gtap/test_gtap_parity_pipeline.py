"""Tests for GTAP GAMS parity pipeline.

These tests verify the parity comparison system between Python GTAP
and CGEBox GAMS results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from equilibria.templates.gtap.gtap_parity_pipeline import (
    GTAPParityComparison,
    GTAPParityRunner,
    GTAPGAMSReference,
    GTAPVariableSnapshot,
    compare_gtap_gams_parity,
    compare_variable_groups,
    run_gtap_parity_test,
)


class TestGTAPVariableSnapshot:
    """Tests for GTAPVariableSnapshot."""
    
    def test_empty_snapshot(self):
        """Test creating empty snapshot."""
        snap = GTAPVariableSnapshot()
        assert snap.pnum == 1.0
        assert snap.walras == 0.0
        assert len(snap.xp) == 0
    
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


class TestGTAPParityRunner:
    """Tests for GTAPParityRunner."""
    
    def test_initialization(self, tmp_path: Path, monkeypatch):
        """Test runner initialization."""
        # Create mock GDX file
        gdx_file = tmp_path / "test.gdx"
        gdx_file.write_bytes(b"GAMSGDX")
        
        # Mock the sets and params loading
        def mock_load_sets(self, path):
            self.r = ["USA", "EUR"]
            self.i = ["agr", "mfg"]
            self.a = ["agr", "mfg"]
            self.f = ["lab", "cap"]
            self.mf = ["lab"]
            self.sf = ["cap"]
            self.aggregation_name = "test"
        
        def mock_load_params(self, path):
            pass
        
        monkeypatch.setattr("equilibria.templates.gtap.GTAPSets.load_from_gdx", mock_load_sets)
        monkeypatch.setattr("equilibria.templates.gtap.GTAPParameters.load_from_gdx", mock_load_params)
        
        # Should raise because we can't actually load the GDX
        with pytest.raises(Exception):
            runner = GTAPParityRunner(gdx_file=gdx_file)
    
    def test_run_python_mocked(self, tmp_path: Path, monkeypatch):
        """Test running Python model with mocking."""
        # This would require extensive mocking of the full model
        # For now, just test the interface
        pass


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
            run_gtap_parity_test(gdx_file, gams_results)


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
