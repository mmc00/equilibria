"""End-of-solve report.

``summary(m, results)`` prints a short panel after a solve (termination code,
Walras, residuals, worst equations), writes the same numbers to the
``equilibria`` logger (INFO; a WARNING for what needs attention) and returns
them.
"""

from equilibria.report.summary import SolveSummary, WorstEquation, summary

__all__ = ["SolveSummary", "WorstEquation", "summary"]
