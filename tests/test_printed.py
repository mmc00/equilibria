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
    got = [
        line.replace("within 0.1%: 100.00%", "within 0.1%: 99.78%") for line in TBL55
    ]
    assert mismatches(got, TBL55) == [
        "line 1: 99.78 != 100.00 in "
        "'within 0.1%: 100.00%   within 0.5%: 100.00%   within 1%: 100.00%'"
    ]


def test_the_largest_differences_table_is_not_compared() -> None:
    """Its rows are the near-zero cells (qxs ~1e-5) that move in the 4th digit
    between Linux and the Mac, sorted by diffs that tie at 0.000%: TBL77 lists
    other cells on the CI. The headline (cells, % within) is the result."""
    rows = [i for i, line in enumerate(TBL55) if line.startswith("qst ")]
    got = list(TBL55)
    got[rows[0]] = (
        "qxs          post  USA,SER,USA                1.49991e-05   1.49991e-05   0.000%"
    )
    assert mismatches(got, TBL55) == []


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


def test_the_largest_difference_line_moves_by_at_most_its_band() -> None:
    """TBL64's headline: 0.029% on Linux against 0.026% on the Mac passes, a move
    past ``LARGEST_DIFF_ABS`` does not, and every other line keeps ``rel``."""
    old = ["largest difference: 0.026%"]
    assert mismatches(["largest difference: 0.029%"], old) == []
    assert mismatches(["largest difference: 0.032%"], old)
    assert mismatches(["within 0.5%: 99.29%"], ["within 0.5%: 99.28%"])
