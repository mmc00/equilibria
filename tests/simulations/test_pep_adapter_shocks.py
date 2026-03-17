from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from equilibria.simulations.adapters.pep import PepAdapter
from equilibria.simulations.types import Scenario, Shock
from equilibria.templates.pep_calibration_unified import PEPModelState


def test_pep_adapter_applies_scalar_go_shocks() -> None:
    state = PEPModelState(consumption={"GO": 100.0})
    adapter = PepAdapter(sam_file="dummy.gdx", val_par_file=None)

    adapter.apply_shock(state, Shock(var="G", op="scale", values=1.2))
    assert state.consumption["GO"] == pytest.approx(120.0)

    adapter.apply_shock(state, Shock(var="G", op="add", values=5.0))
    assert state.consumption["GO"] == pytest.approx(125.0)

    adapter.apply_shock(state, Shock(var="G", op="set", values=80.0))
    assert state.consumption["GO"] == pytest.approx(80.0)


def test_pep_adapter_applies_indexed_pwm_wildcard_and_specific() -> None:
    state = PEPModelState(
        sets={"I": ["agr", "ser"]},
        trade={"PWMO": {"agr": 1.0, "ser": 2.0}},
    )
    adapter = PepAdapter(sam_file="dummy.gdx", val_par_file=None)

    adapter.apply_shock(
        state,
        Shock(var="PWM", op="scale", values={"*": 1.25, "agr": 2.0}),
    )

    assert state.trade["PWMO"]["agr"] == pytest.approx(2.5)
    assert state.trade["PWMO"]["ser"] == pytest.approx(2.5)


def test_pep_adapter_rejects_unknown_index_for_indexed_shock() -> None:
    state = PEPModelState(
        sets={"I": ["agr"]},
        trade={"PWMO": {"agr": 1.0}},
    )
    adapter = PepAdapter(sam_file="dummy.gdx", val_par_file=None)

    with pytest.raises(ValueError, match="unknown indices"):
        adapter.apply_shock(
            state,
            Shock(var="PWM", op="set", values={"bad": 1.0}),
        )


def test_pep_adapter_syncs_export_tax_aggregates_after_ttix_shock() -> None:
    state = PEPModelState(
        sets={"I": ["agr", "ser"]},
        trade={
            "ttixO": {"agr": 0.1, "ser": 0.2},
            "EXDO": {"agr": 10.0, "ser": 5.0},
            "PEO": {"agr": 2.0, "ser": 3.0},
            "tmrg_X": {
                ("agr", "agr"): 0.1,
                ("ser", "agr"): 0.2,
                ("agr", "ser"): 0.0,
                ("ser", "ser"): 0.1,
            },
            "PCO": {"agr": 1.0, "ser": 2.0},
        },
        income={
            "TICTO": 10.0,
            "TIMTO": 20.0,
            "TIXTO": 0.0,
            "TPRCTSO": 0.0,
            "YGKO": 100.0,
            "TDHTO": 5.0,
            "TDFTO": 7.0,
            "TPRODNO": 8.0,
            "YGTRO": 9.0,
            "YGO": 0.0,
        },
    )
    adapter = PepAdapter(sam_file="dummy.gdx", val_par_file=None)

    adapter.apply_shock(state, Shock(var="ttix", op="scale", values={"*": 2.0}))

    assert state.trade["ttixO"]["agr"] == pytest.approx(0.2)
    assert state.trade["ttixO"]["ser"] == pytest.approx(0.4)
    assert state.trade["TIXO"]["agr"] == pytest.approx(5.0)
    assert state.trade["TIXO"]["ser"] == pytest.approx(6.4)
    assert state.income["TIXTO"] == pytest.approx(11.4)
    assert state.income["TPRCTSO"] == pytest.approx(41.4)
    assert state.income["YGO"] == pytest.approx(170.4)


def test_pep_adapter_available_shocks_include_domain_members_when_sets_known() -> None:
    adapter = PepAdapter(sam_file="dummy.gdx", val_par_file=None)
    adapter._sets = {"I": ["agr", "ser"]}  # internal cache filled by fit_base_state
    catalog = adapter.available_shocks()
    by_var = {item.var: item for item in catalog}
    assert by_var["G"].members is None
    assert by_var["PWM"].members == ("agr", "ser")
    assert by_var["PWX"].members == ("agr", "ser")
    assert by_var["ttix"].members == ("agr", "ser")


def test_pep_adapter_fit_base_state_uses_excel_dynamic_sam_for_excel_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeExcelDynamicSAM:
        def __init__(self, *, sam_file: Path, val_par_file: Path | None, accounts: dict[str, str] | None):
            captured["sam_file"] = sam_file
            captured["val_par_file"] = val_par_file
            captured["accounts"] = accounts

        def calibrate(self) -> PEPModelState:
            return PEPModelState(sets={"I": ["agr"], "J": ["agr"]})

    monkeypatch.setattr(
        "equilibria.simulations.adapters.pep.PEPModelCalibratorExcelDynamicSAM",
        _FakeExcelDynamicSAM,
    )
    monkeypatch.setattr(
        "equilibria.simulations.adapters.pep.should_apply_cri_pep_fix",
        lambda *_args, **_kwargs: False,
    )

    adapter = PepAdapter(
        sam_file="sam-cri.xlsx",
        val_par_file="val.xlsx",
        dynamic_sets=True,
        accounts={"gvt": "gvt", "row": "row"},
    )

    state = adapter.fit_base_state()

    assert captured["sam_file"] == Path("sam-cri.xlsx")
    assert captured["val_par_file"] == Path("val.xlsx")
    assert captured["accounts"] == {"gvt": "gvt", "row": "row"}
    assert state.sets["I"] == ["agr"]
    assert adapter._sets["I"] == ["agr"]


def test_pep_adapter_fit_base_state_uses_dynamic_sam_for_gdx_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeDynamicSAM:
        def __init__(self, *, sam_file: Path, val_par_file: Path | None, accounts: dict[str, str] | None):
            captured["sam_file"] = sam_file
            captured["val_par_file"] = val_par_file
            captured["accounts"] = accounts

        def calibrate(self) -> PEPModelState:
            return PEPModelState(sets={"I": ["agr", "ser"]})

    monkeypatch.setattr(
        "equilibria.simulations.adapters.pep.PEPModelCalibratorDynamicSAM",
        _FakeDynamicSAM,
    )

    adapter = PepAdapter(
        sam_file="sam-cri.gdx",
        val_par_file="val.gdx",
        dynamic_sets=True,
        accounts={"gvt": "gvt"},
    )

    state = adapter.fit_base_state()

    assert captured["sam_file"] == Path("sam-cri.gdx")
    assert captured["val_par_file"] == Path("val.gdx")
    assert captured["accounts"] == {"gvt": "gvt"}
    assert state.sets["I"] == ["agr", "ser"]
    assert adapter._sets["I"] == ["agr", "ser"]


def test_pep_adapter_prepares_runtime_cri_excel_with_transform_and_qa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeExcelDynamicSAM:
        def __init__(self, *, sam_file: Path, val_par_file: Path | None, accounts: dict[str, str] | None):
            captured["cal_sam_file"] = sam_file
            captured["cal_val_par_file"] = val_par_file
            captured["cal_accounts"] = accounts

        def calibrate(self) -> PEPModelState:
            return PEPModelState(sets={"I": ["agr"]})

    def _fake_transform(**kwargs: object) -> dict[str, object]:
        captured["transform"] = kwargs
        return {"after": {"balance": {"max_row_col_abs_diff": 0.0}}}

    class _FakeQAReport:
        passed = True

        def save_json(self, path: Path | str) -> None:
            captured["qa_save"] = Path(path)

    def _fake_run_sam_qa_from_file(**kwargs: object) -> _FakeQAReport:
        captured["qa"] = kwargs
        return _FakeQAReport()

    monkeypatch.setattr(
        "equilibria.simulations.adapters.pep.PEPModelCalibratorExcelDynamicSAM",
        _FakeExcelDynamicSAM,
    )
    monkeypatch.setattr(
        "equilibria.simulations.adapters.pep.should_apply_cri_pep_fix",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "equilibria.simulations.adapters.pep.transform_sam_to_pep_compatible",
        _fake_transform,
    )
    monkeypatch.setattr(
        "equilibria.simulations.adapters.pep.run_sam_qa_from_file",
        _fake_run_sam_qa_from_file,
    )
    monkeypatch.setattr(
        "equilibria.simulations.adapters.pep.format_report_summary",
        lambda report: f"summary(passed={report.passed})",
    )

    adapter = PepAdapter(
        sam_file="SAM-CRI-gams.xlsx",
        val_par_file="VAL_PAR-CRI-gams.xlsx",
        dynamic_sets=True,
        sam_qa_mode="warn",
        sam_qa_report="output/qa.json",
    )

    state = adapter.fit_base_state()

    assert state.sets["I"] == ["agr"]
    assert adapter._runtime_sam_file == Path("output/SAM-CRI-gams-pep-compatible.xlsx")
    assert captured["transform"] == {
        "input_sam": Path("SAM-CRI-gams.xlsx"),
        "output_sam": Path("output/SAM-CRI-gams-pep-compatible.xlsx"),
        "report_json": Path("output/SAM-CRI-gams-pep-compatible-report.json"),
        "target_mode": "geomean",
        "margin_commodity": "ser",
    }
    assert captured["qa"] == {
        "sam_file": Path("output/SAM-CRI-gams-pep-compatible.xlsx"),
        "dynamic_sam": True,
        "accounts": adapter.accounts,
        "balance_rel_tol": 1e-06,
        "gdp_rel_tol": 0.08,
        "max_samples": 8,
        "strict_structural": False,
    }
    assert captured["cal_sam_file"] == Path("output/SAM-CRI-gams-pep-compatible.xlsx")


def test_pep_adapter_solve_state_passes_contract_and_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeSolver:
        def __init__(self, **kwargs: object) -> None:
            captured["kwargs"] = kwargs
            self.params = {"ok": True}

        def solve(self, *, method: str) -> SimpleNamespace:
            captured["method"] = method
            return SimpleNamespace(
                converged=True,
                iterations=1,
                final_residual=0.0,
                message="ok",
                variables=SimpleNamespace(GDP_BP=1.0, GDP_MP=1.0, CTH={}, IT=0.0, EXD={}, IM={}),
            )

        def validate_solution(self, solution: object) -> dict[str, object]:
            captured["validated_solution"] = solution
            return {"passed": True}

    monkeypatch.setattr("equilibria.simulations.adapters.pep.PEPModelSolver", _FakeSolver)

    adapter = PepAdapter(
        sam_file="sam.gdx",
        val_par_file="val.xlsx",
        contract="pep_nlp_v1",
        config="default_ipopt",
    )
    adapter._runtime_sam_file = Path("sam.gdx")

    state = PEPModelState()
    solver, solution, validation = adapter.solve_state(
        state,
        initial_vars=None,
        reference_results_gdx=None,
        reference_slice="sim1",
        scenario=Scenario(
            name="government_spending",
            shocks=[Shock(var="G", op="scale", values=1.2)],
            closure={"fixed": ["G", "CAB", "PWM", "CMIN", "VSTK", "TR_SELF"]},
        ),
    )

    assert isinstance(solver, _FakeSolver)
    assert validation == {"passed": True}
    assert solution.converged is True
    kwargs = captured["kwargs"]
    assert kwargs["contract"].name == "pep_nlp_v1"
    assert kwargs["config"].name == "default_ipopt"
    assert kwargs["sam_file"] == Path("sam.gdx")
    assert captured["method"] == "ipopt"
    assert kwargs["contract"].closure.fixed == ("G", "CAB", "PWM", "CMIN", "VSTK", "TR_SELF")


def test_pep_adapter_hard_fail_raises_on_failed_sam_qa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeQAReport(SimpleNamespace):
        def save_json(self, path: Path | str) -> None:
            _ = path

    monkeypatch.setattr(
        "equilibria.simulations.adapters.pep.should_apply_cri_pep_fix",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "equilibria.simulations.adapters.pep.run_sam_qa_from_file",
        lambda **_kwargs: _FakeQAReport(passed=False),
    )
    monkeypatch.setattr(
        "equilibria.simulations.adapters.pep.format_report_summary",
        lambda _report: "SAM QA FAIL",
    )

    adapter = PepAdapter(
        sam_file="SAM-CRI-gams.xlsx",
        val_par_file="VAL_PAR-CRI-gams.xlsx",
        dynamic_sets=True,
        sam_qa_mode="hard_fail",
    )

    with pytest.raises(RuntimeError, match="SAM QA failed: SAM QA FAIL"):
        adapter.fit_base_state()
