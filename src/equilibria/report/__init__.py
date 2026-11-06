"""End-of-solve report.

``summary(m, results)`` prints a short panel after a solve (termination code,
Walras, residuals, worst equations) and returns the same numbers.
"""

from equilibria.report.summary import SolveSummary, WorstEquation, summary

__all__ = ["SolveSummary", "WorstEquation", "summary"]
