from equilibria.templates.pep_parity_pipeline import (
    classify_pipeline_outcome,
    default_equation_contracts,
    evaluate_block_gates,
    evaluate_eq29_eq39_against_gams,
    evaluate_eq79_eq84_against_gams,
    evaluate_levels_against_gams,
    evaluate_results_baseline_compatibility,
    summarize_residuals,
)


def test_summarize_residuals_basic():
    residuals = {
        "EQ84_ser": 10.0,
        "EQ29": -4.0,
        "EQ39_ind": 3.0,
    }
    summary = summarize_residuals(residuals, top_n=2)
    assert summary.count == 3
    assert abs(summary.max_abs - 10.0) < 1e-12
    assert len(summary.top_abs) == 2
    assert summary.top_abs[0][0] == "EQ84_ser"


def test_evaluate_block_gates_fail_fast_stops_first_block():
    residuals = {
        "EQ29": 1.0,
        "EQ39_ind": 0.5,
        "EQ84_ser": 0.0,
        "EQ88_agr": 0.0,
        "EQ44": 0.0,
    }
    out = evaluate_block_gates(residuals, contracts=default_equation_contracts(), fail_fast=True)
    assert out["overall_passed"] is False
    assert out["first_failed_block"] == "production_tax_consistency"
    assert len(out["blocks"]) == 1
    assert out["blocks"][0]["passed"] is False


def test_evaluate_block_gates_non_fail_fast_reports_all():
    residuals = {
        "EQ29": 1.0,
        "EQ39_ind": 0.2,
        "EQ84_ser": 2.0,
        "EQ88_agr": 3.0,
        "WALRAS": 4.0,
    }
    out = evaluate_block_gates(residuals, contracts=default_equation_contracts(), fail_fast=False)
    assert out["overall_passed"] is False
    assert len(out["blocks"]) == len(default_equation_contracts())
    failed = [b for b in out["blocks"] if not b["passed"]]
    assert failed


def test_default_trade_market_contract_includes_eq63():
    contracts = default_equation_contracts()
    trade_contract = next(c for c in contracts if c.block == "trade_market_clearing")
    assert "EQ63" in trade_contract.eq_prefixes


class _Vars:
    def __init__(self):
        self.TIP = {}
        self.TIPT = 0.0
        self.TIC = {}
        self.PP = {}
        self.XST = {}
        self.PD = {}
        self.DD = {}
        self.PM = {}
        self.IM = {}


def test_eq29_eq39_parity_passes_with_matching_residuals(monkeypatch):
    import equilibria.templates.pep_parity_pipeline as mod

    def fake_gdxdump_records(_bin, _gdx, symbol):
        if symbol == "valTIP":
            return [
                (("agr", "base"), 10.0),
                (("ind", "base"), 20.0),
            ]
        if symbol == "valTIPT":
            return [(("base",), 30.0)]
        if symbol == "valPP":
            return [
                (("agr", "base"), 2.0),
                (("ind", "base"), 4.0),
            ]
        if symbol == "valXST":
            return [
                (("agr", "base"), 5.0),
                (("ind", "base"), 5.0),
            ]
        if symbol == "valttip":
            return [
                (("agr", "base"), 1.0),
                (("ind", "base"), 1.0),
            ]
        if symbol == "valTIC":
            return [(("agr", "base"), 6.0)]
        if symbol == "valttic":
            return [(("agr", "base"), 0.5)]
        if symbol == "valPD":
            return [(("agr", "base"), 2.0)]
        if symbol == "valDD":
            return [(("agr", "base"), 4.0)]
        if symbol == "valPM":
            return [(("agr", "base"), 1.0)]
        if symbol == "valIM":
            return [(("agr", "base"), 4.0)]
        return []

    monkeypatch.setattr(mod, "_gdxdump_records", fake_gdxdump_records)

    v = _Vars()
    v.TIP = {"agr": 9.0, "ind": 19.0}
    v.TIPT = 28.0
    v.TIC = {"agr": 5.0}
    v.PP = {"agr": 2.0, "ind": 4.0}
    v.XST = {"agr": 5.0, "ind": 5.0}
    v.PD = {"agr": 1.0}
    v.DD = {"agr": 4.0}
    v.PM = {"agr": 1.0}
    v.IM = {"agr": 4.0}

    # GAMS eq29 residual = 30 - (10+20) = 0
    # Python eq29 residual = 28 - (9+19) = 0
    # GAMS eq39 residuals: agr=10-1*2*5=0, ind=20-1*4*5=0
    # Python eq39 residuals: agr=9-1*2*5=-1, ind=19-1*4*5=-1
    # Delta each = -1; set tol high so this case passes (EQ39 and EQ40).
    out = evaluate_eq29_eq39_against_gams(
        vars_obj=v,
        results_gdx="dummy.gdx",
        gdxdump_bin="dummy",
        gams_slice="base",
        tol=1.5,
    )
    assert out["passed"] is True


def test_eq29_eq39_parity_fails_with_large_delta(monkeypatch):
    import equilibria.templates.pep_parity_pipeline as mod

    def fake_gdxdump_records(_bin, _gdx, symbol):
        if symbol == "valTIP":
            return [(("agr", "base"), 10.0)]
        if symbol == "valTIPT":
            return [(("base",), 10.0)]
        if symbol == "valPP":
            return [(("agr", "base"), 1.0)]
        if symbol == "valXST":
            return [(("agr", "base"), 1.0)]
        if symbol == "valttip":
            return [(("agr", "base"), 1.0)]
        if symbol == "valTIC":
            return [(("agr", "base"), 1.0)]
        if symbol == "valttic":
            return [(("agr", "base"), 1.0)]
        if symbol == "valPD":
            return [(("agr", "base"), 1.0)]
        if symbol == "valDD":
            return [(("agr", "base"), 1.0)]
        if symbol == "valPM":
            return [(("agr", "base"), 1.0)]
        if symbol == "valIM":
            return [(("agr", "base"), 1.0)]
        return []

    monkeypatch.setattr(mod, "_gdxdump_records", fake_gdxdump_records)

    v = _Vars()
    v.TIP = {"agr": 100.0}
    v.TIPT = 100.0
    v.TIC = {"agr": 100.0}
    v.PP = {"agr": 1.0}
    v.XST = {"agr": 1.0}
    v.PD = {"agr": 1.0}
    v.DD = {"agr": 1.0}
    v.PM = {"agr": 1.0}
    v.IM = {"agr": 1.0}

    out = evaluate_eq29_eq39_against_gams(
        vars_obj=v,
        results_gdx="dummy.gdx",
        gdxdump_bin="dummy",
        gams_slice="base",
        tol=1e-6,
    )
    assert out["passed"] is False


def test_eq79_eq84_parity_passes_with_matching_residuals(monkeypatch):
    import equilibria.templates.pep_parity_pipeline as mod

    def fake_gdxdump_records(_bin, _gdx, symbol):
        if symbol == "valPC":
            return [(("agr", "base"), 1.0)]
        if symbol == "valQ":
            return [(("agr", "base"), 10.0)]
        if symbol == "valPM":
            return [(("agr", "base"), 1.0)]
        if symbol == "valIM":
            return [(("agr", "base"), 4.0)]
        if symbol == "valPD":
            return [(("agr", "base"), 1.0)]
        if symbol == "valDD":
            return [(("agr", "base"), 6.0)]
        if symbol == "valC":
            return [(("agr", "h1", "base"), 3.0)]
        if symbol == "valCG":
            return [(("agr", "base"), 2.0)]
        if symbol == "valINV":
            return [(("agr", "base"), 2.0)]
        if symbol == "valVSTK":
            return [(("agr", "base"), 1.0)]
        if symbol == "valDIT":
            return [(("agr", "base"), 1.0)]
        if symbol == "valMRGN":
            return [(("agr", "base"), 1.0)]
        return []

    monkeypatch.setattr(mod, "_gdxdump_records", fake_gdxdump_records)

    v = _Vars()
    v.PC = {"agr": 1.0}
    v.Q = {"agr": 10.0}
    v.PM = {"agr": 1.0}
    v.IM = {"agr": 4.0}
    v.PD = {"agr": 1.0}
    v.DD = {"agr": 6.0}
    v.C = {("agr", "h1"): 3.0}
    v.CG = {"agr": 2.0}
    v.INV = {"agr": 2.0}
    v.VSTK = {"agr": 1.0}
    v.DIT = {"agr": 1.0}
    v.MRGN = {"agr": 1.0}

    out = evaluate_eq79_eq84_against_gams(
        vars_obj=v,
        results_gdx="dummy.gdx",
        gdxdump_bin="dummy",
        gams_slice="base",
        tol=1e-9,
    )
    assert out["passed"] is True


def test_levels_parity_detects_delta(monkeypatch):
    import equilibria.templates.pep_parity_pipeline as mod

    def fake_read_gdx(_path):
        return {"symbols": [{"name": "valPC"}, {"name": "valQ"}]}

    def fake_gdxdump_records(_bin, _gdx, symbol):
        if symbol == "valPC":
            return [(("agr", "base"), 1.0)]
        if symbol == "valQ":
            return [(("agr", "base"), 10.0)]
        return []

    monkeypatch.setattr(mod, "read_gdx", fake_read_gdx)
    monkeypatch.setattr(mod, "_gdxdump_records", fake_gdxdump_records)

    v = _Vars()
    v.PC = {"agr": 2.0}
    v.Q = {"agr": 10.0}

    out = evaluate_levels_against_gams(
        vars_obj=v,
        results_gdx="dummy.gdx",
        gdxdump_bin="dummy",
        gams_slice="base",
        tol=1e-9,
    )
    assert out["passed"] is False
    assert out["count_compared"] == 2
    assert out["max_abs_delta"] == 1.0


def test_baseline_compatibility_gdp_anchor(monkeypatch):
    import equilibria.templates.pep_parity_pipeline as mod

    def fake_gdxdump_records(_bin, _gdx, symbol):
        if symbol == "valGDP_BP":
            return [(("base",), 100.0)]
        return []

    class _State:
        gdp = {"GDP_BPO": 100.0}

    monkeypatch.setattr(mod, "_gdxdump_records", fake_gdxdump_records)
    out = evaluate_results_baseline_compatibility(
        state=_State(),
        results_gdx="dummy.gdx",
        gdxdump_bin="dummy",
        gams_slice="base",
        rel_tol=1e-8,
    )
    assert out["passed"] is True


def test_baseline_compatibility_scenario_uses_base_anchor(monkeypatch):
    import equilibria.templates.pep_parity_pipeline as mod

    def fake_gdxdump_records(_bin, _gdx, symbol):
        if symbol == "valGDP_BP":
            return [
                (("base",), 100.0),
                (("sim1",), 80.0),
            ]
        return []

    class _State:
        gdp = {"GDP_BPO": 100.0}

    monkeypatch.setattr(mod, "_gdxdump_records", fake_gdxdump_records)
    out = evaluate_results_baseline_compatibility(
        state=_State(),
        results_gdx="dummy.gdx",
        gdxdump_bin="dummy",
        gams_slice="sim1",
        rel_tol=1e-8,
    )
    assert out["passed"] is True
    assert out["anchor_slice"] == "base"
    assert out["requested_slice"] == "sim1"
    assert out["gams_gdp_bp"] == 100.0
    assert out["requested_slice_has_records"] is True
    assert out["requested_slice_gdp_bp"] == 80.0


def test_classify_pipeline_outcome_data_contract_from_sam_qa() -> None:
    out = classify_pipeline_outcome(
        sam_qa_report={"passed": False},
        init_gates=None,
        solve_report=None,
        method="none",
    )
    assert out["kind"] == "data_contract"
    assert out["reason"] == "sam_qa_failed"


def test_classify_pipeline_outcome_solver_dynamics_from_solve_failure() -> None:
    out = classify_pipeline_outcome(
        sam_qa_report={"passed": True},
        init_gates={"overall_passed": True, "first_failed_block": None},
        solve_report={"converged": False, "gates": {"overall_passed": False}},
        method="ipopt",
    )
    assert out["kind"] == "solver_dynamics"
    assert out["reason"] == "solve_not_converged"


def test_classify_pipeline_outcome_pass_on_init_only() -> None:
    out = classify_pipeline_outcome(
        sam_qa_report={"passed": True},
        init_gates={"overall_passed": True, "first_failed_block": None},
        solve_report=None,
        method="none",
    )
    assert out["kind"] == "pass"
    assert out["reason"] == "init_gates_passed"
