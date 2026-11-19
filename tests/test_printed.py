"""``tests/_printed`` keeps the results and catches a changed number."""

from __future__ import annotations

from tests._printed import GOLDEN, mismatches, results

TBL55 = GOLDEN["notebooks"]["TBL55"]


def test_results_drop_the_chatter_and_the_codes() -> None:
    kept = results(TBL55)
    assert kept[0] == "454 cells (pre and post) against GEMPACK Gragg"
    assert not [line for line in kept if "codes:" in line or "Jacobian" in line]


def test_the_golden_matches_itself() -> None:
    assert mismatches(TBL55, TBL55) == []


def test_a_changed_number_is_caught() -> None:
    got = [line.replace("28713.6", "28713.7") for line in TBL55]
    assert mismatches(got, TBL55) == [
        "line 4: 28713.7 != 28713.6 in "
        "'qst          post  USA,SER                        28713.6       "
        "28707.5   0.021%'"
    ]


def test_a_changed_text_or_a_missing_line_is_caught() -> None:
    assert mismatches([*TBL55[:-1], "not every cell"], TBL55)
    assert mismatches(TBL55[:-1], TBL55)


def test_another_solver_code_line_is_not_a_result() -> None:
    old = ["codes: {'base': 1, 'check': 1, 'shock': 1}", "a 1.5"]
    core = ["codes: {'check': 'optimal', 'shock': 'optimal'}", "a 1.5"]
    assert mismatches(core, old) == []


def test_pyomo_warnings_are_not_results() -> None:
    """The old driver's Johansen path set pwmg to 0 below its bound; Pyomo said so."""
    old = [
        "WARNING (W1002): Setting Var 'pwmg[USA,AGR,USA,shock]' to a numeric value",
        "`0.0` outside the bounds (0.001, None).",
        "    See also https://pyomo.readthedocs.io/en/stable/errors.html#w1002",
        "a 1.5",
    ]
    assert mismatches(["a 1.5"], old) == []


def test_a_printed_optimal_is_the_old_code_1() -> None:
    """``print(results["shock"]["code"])``: PATH's 1 on the old driver, "optimal" now."""
    assert mismatches(["optimal", "a 1.5"], ["1", "a 1.5"]) == []
    assert mismatches(["code 2"], ["1"])


QUICKSTART = GOLDEN["docs"]["gtap_quickstart.md"]


def test_a_failed_solve_panel_is_a_mismatch() -> None:
    """The panel's title says whether the solve succeeded: it is a result."""
    head = next(line for line in QUICKSTART if "SOLVE COMPLETE" in line)
    failed = head.replace("  SOLVE COMPLETE  ", "  SOLVE FAILED (code optimal)  ")
    assert mismatches(
        [failed if line == head else line for line in QUICKSTART], QUICKSTART
    )


def test_the_result_dict_keeps_its_code() -> None:
    """``print(results["shock"])``: the code is a result, the residual's value is not."""
    old = ["{'code': 1, 'residual': 8.589158046959055e-14}"]
    assert mismatches(["{'code': 'optimal', 'residual': 5.1e-13}"], old) == []
    assert mismatches(["{'code': 'code 2', 'residual': 5.1e-13}"], old)
