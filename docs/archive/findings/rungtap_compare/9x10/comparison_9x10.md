# 9x10 — equilibria vs RunGTAP (capFix closure, 10% tm shock)

**Setup:** identical NUS333 methodology extended to 10 regions × 10 commodities ×
9 activities × 5 factors. Both engines use capFix closure with NAmerica as residual.

RunGTAP CMF: `swap dpsave(r) = del_tbalry(r)` for all 9 non-residual regions.
Equilibria: `GTAPClosureConfig(savf_flag='capFix', if_sub=False)`, `residual_region='NAmerica'`.

## Utility per capita — equilibria vs RunGTAP

| Region | equilibria u %Δ | RunGTAP u %Δ | gap (pp) |
|---|---:|---:|---:|
| Oceania | -0.5683% | -0.6245% | +0.0562 |
| EastAsia | -0.6706% | -0.6605% | -0.0101 |
| SEAsia | -1.4058% | -1.0772% | -0.3286 |
| SouthAsia | -0.5561% | -0.5868% | +0.0307 |
| NAmerica | -0.0052% | -0.0281% | +0.0229 |
| LatinAmer | -0.5159% | -0.5445% | +0.0286 |
| EU_28 | -0.6758% | -0.3289% | -0.3469 |
| MENA | -1.2282% | -1.3453% | +0.1171 |
| SSA | -0.9070% | -1.0161% | +0.1091 |
| RestofWorld | -1.4115% | -1.4984% | +0.0869 |

**Avg |gap|:** 0.1137 pp
**Max |gap|:** 0.3469 pp

## Regional income (regy nominal) — equilibria vs RunGTAP

| Region | equilibria regy %Δ | RunGTAP y %Δ | gap (pp) |
|---|---:|---:|---:|
| Oceania | +2.8428% | +2.7640% | +0.0789 |
| EastAsia | +1.0532% | +1.1183% | -0.0651 |
| SEAsia | +0.0950% | +0.4269% | -0.3318 |
| SouthAsia | +1.9958% | +1.8667% | +0.1291 |
| NAmerica | +3.6163% | +3.4799% | +0.1364 |
| LatinAmer | +3.0247% | +2.9242% | +0.1005 |
| EU_28 | +1.5164% | +1.7348% | -0.2183 |
| MENA | +3.5387% | +3.3916% | +0.1471 |
| SSA | +4.2182% | +3.9916% | +0.2266 |
| RestofWorld | +1.8088% | +1.8020% | +0.0068 |

## Bottom line

- **Per-region u %Δ:** 8 of 10 regions within 0.15 pp of RunGTAP.
- **2 outlier regions** (SEAsia, EU_28) where the dpsave shift under the swap is extreme (|dpsave| > 14% in RunGTAP) — those exhibit larger linear-approx gaps.
- **NAmerica (residual):** essentially exact (gap < 0.025 pp).
- The full per-region EV via `welfare_shadow.integrate()` matches RunGTAP to ~1.7% at the world total — see `verify_shadow_9x10.py` for the breakdown.