# GAMS Parity Matrix

This matrix defines baseline parity targets between GAMS symbols and Equilibria solver initialization variables.

## Core Price Variables

| GAMS Symbol | Equilibria Variable | Source in Calibrated State |
|---|---|---|
| `PD(i)` | `vars.PD[i]` | `state.trade["PDO"][i]` |
| `PM(i)` | `vars.PM[i]` | `state.trade["PMO"][i]` |
| `PC(i)` | `vars.PC[i]` | `state.trade["PCO"][i]` |
| `PL(i)` | `vars.PL[i]` | `state.trade["PLO"][i]` |
| `PWM(i)` | `vars.PWM[i]` | `state.trade["PWMO"][i]` |
| `PP(j)` | `vars.PP[j]` | `state.production["PPO"][j]` |
| `PVA(j)` | `vars.PVA[j]` | `state.production["PVAO"][j]` |
| `PCI(j)` | `vars.PCI[j]` | `state.production["PCIO"][j]` |

## Tax and Government Aggregates

| GAMS Symbol | Equilibria Variable | Source in Calibrated State |
|---|---|---|
| `TIWT` | `vars.TIWT` | `state.income["TIWTO"]` |
| `TIKT` | `vars.TIKT` | `state.income["TIKTO"]` |
| `TPRODN` | `vars.TPRODN` | `state.income["TPRODNO"]` |
| `TICT` | `vars.TICT` | `state.income["TICTO"]` |
| `TIMT` | `vars.TIMT` | `state.income["TIMTO"]` |
| `TIXT` | `vars.TIXT` | `state.income["TIXTO"]` |
| `YG` | `vars.YG` | `state.income["YGO"]` |

## Macro and ROW Aggregates

| GAMS Symbol | Equilibria Variable | Source in Calibrated State |
|---|---|---|
| `GDP_IB` | `vars.GDP_IB` | `state.gdp["GDP_IBO"]` |
| `SROW` | `vars.SROW` | `-state.income["CABO"]` |
| `CAB` | `vars.CAB` | `state.income["CABO"]` |

## Transfer Orientation Notes

- Household transfer income equation in GAMS uses `SUM[ag, TR(h,ag)]` in `EQ13`.
- Household consumption budget uses `SUM[agng, TR(agng,h)]` in `EQ15`.
- Government transfer income uses `SUM[agng, TR('gvt',agng)]` in `EQ34`.
- ROW equations use:
  - `SUM[agd, TR('row',agd)]` in `EQ44`
  - `SUM[agd, TR(agd,'row')]` in `EQ45`

These orientation conventions must be preserved exactly in strict parity mode.
