"""Contract definitions for SAM QA gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class SAMContractSpec:
    """Defines one SAM QA contract gate."""

    code: str
    title: str
    category: str
    description: str
    severity: Severity
    abs_tol: float
    rel_tol: float


def default_sam_contracts(
    *,
    balance_rel_tol: float = 1e-6,
    gdp_rel_tol: float = 0.08,
) -> dict[str, SAMContractSpec]:
    """Default structural QA contracts used by Phase 1."""
    return {
        "EXP001": SAMContractSpec(
            code="EXP001",
            title="Export Value Balance",
            category="exports_domestic_supply",
            description=(
                "For each commodity i: "
                "SAM('X',i,'AG','ROW') = sum_j SAM('J',j,'X',i) "
                "+ sum_ij SAM('I',ij,'X',i) + SAM('AG','GVT','X',i)"
            ),
            severity="error",
            abs_tol=1e-6,
            rel_tol=balance_rel_tol,
        ),
        "DSP001": SAMContractSpec(
            code="DSP001",
            title="Commodity Account Balance",
            category="exports_domestic_supply",
            description=(
                "For each commodity i account, SAM row and column totals must match."
            ),
            severity="error",
            abs_tol=1e-6,
            rel_tol=balance_rel_tol,
        ),
        "MRG001": SAMContractSpec(
            code="MRG001",
            title="Domestic Margin Denominator",
            category="margins",
            description=(
                "If domestic margin demand exists for commodity ij, "
                "DDO(ij)+IMO(ij) must be strictly positive."
            ),
            severity="error",
            abs_tol=1e-9,
            rel_tol=0.0,
        ),
        "MRG002": SAMContractSpec(
            code="MRG002",
            title="Export Margin Denominator",
            category="margins",
            description=(
                "If export margins exist for commodity ij, "
                "sum_j EXO(j,ij) must be strictly positive."
            ),
            severity="error",
            abs_tol=1e-9,
            rel_tol=0.0,
        ),
        "TAX001": SAMContractSpec(
            code="TAX001",
            title="Import Tax Base",
            category="tax_base",
            description=(
                "If TIMO(i) is non-zero, import base IMO(i) must be strictly positive."
            ),
            severity="error",
            abs_tol=1e-9,
            rel_tol=0.0,
        ),
        "TAX002": SAMContractSpec(
            code="TAX002",
            title="Export Tax Base",
            category="tax_base",
            description=(
                "If TIXO(i) is non-zero, EXDO(i)-TIXO(i) must be strictly positive."
            ),
            severity="error",
            abs_tol=1e-9,
            rel_tol=0.0,
        ),
        "TAX003": SAMContractSpec(
            code="TAX003",
            title="Commodity Tax Base",
            category="tax_base",
            description=(
                "If TICO(i) is non-zero, the commodity tax denominator from EQ40-style "
                "components must be strictly positive."
            ),
            severity="error",
            abs_tol=1e-9,
            rel_tol=0.0,
        ),
        "TAX004": SAMContractSpec(
            code="TAX004",
            title="Production Tax Base",
            category="tax_base",
            description=(
                "If TIPO(j) is non-zero, production value-added + intermediate base must "
                "be strictly positive."
            ),
            severity="error",
            abs_tol=1e-9,
            rel_tol=0.0,
        ),
        "MAC001": SAMContractSpec(
            code="MAC001",
            title="Savings-Investment Balance",
            category="macro_closure",
            description=(
                "sum_ag SAM('OTH','INV','AG',ag) must match "
                "sum_i [SAM('I',i,'OTH','INV') + SAM('I',i,'OTH','VSTK')]."
            ),
            severity="error",
            abs_tol=1e-6,
            rel_tol=balance_rel_tol,
        ),
        "MAC002": SAMContractSpec(
            code="MAC002",
            title="GDP Proxy Closure",
            category="macro_closure",
            description=(
                "Income-side GDP proxy and expenditure-side GDP proxy should be close "
                "(relative gap threshold configurable)."
            ),
            severity="error",
            abs_tol=1e-6,
            rel_tol=gdp_rel_tol,
        ),
    }
