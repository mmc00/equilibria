# Phase 2b — FBEP header vs zero, per result

Each GAMS result exists in two variants (#187): `fbep=header` (basedata.har's FBEP/FTRV injected after getData.gms's `fbep = 0`) and `fbep=zero` (the official getData.gms). What differs between the two folders of each result.

## Local results (Task 4)

The .nl (CONVERT) of every phase differs in every dataset: FBEP enters the factor-price equations.

| Result | .nl files that differ |
|---|---|
| gtap7_10x7 `gams_nl_altertax_ifsub0` | gams_altertax.nl, gams_base.nl, gams_check.nl, gams_shock.nl |
| gtap7_10x7 `gams_nl_altertax_ifsub1` | gams_altertax.nl, gams_base.nl, gams_check.nl, gams_shock.nl |
| gtap7_15x10 `gams_nl_altertax_ifsub0` | gams_altertax.nl, gams_base.nl, gams_shock.nl |
| gtap7_15x10 `gams_nl_altertax_ifsub1` | gams_altertax.nl, gams_base.nl, gams_shock.nl |
| gtap7_3x3 `gams_nl_altertax_ifsub0` | gams_altertax.nl, gams_base.nl, gams_check.nl, gams_shock.nl |
| gtap7_3x3 `gams_nl_altertax_ifsub1` | gams_altertax.nl, gams_base.nl, gams_check.nl, gams_shock.nl |
| gtap7_3x4 `gams_nl_altertax_ifsub0` | gams_altertax.nl, gams_base.nl, gams_shock.nl |
| gtap7_3x4 `gams_nl_altertax_ifsub1` | gams_altertax.nl, gams_base.nl, gams_shock.nl |
| gtap7_5x5 `gams_nl_altertax_ifsub0` | gams_altertax.nl, gams_base.nl, gams_check.nl, gams_shock.nl |
| gtap7_5x5 `gams_nl_altertax_ifsub1` | gams_altertax.nl, gams_base.nl, gams_check.nl, gams_shock.nl |

NLP and MCP solves: `gdxdiff RelEps 1e-6` of `out.gdx`. The subsidy moves the whole equilibrium, so most symbols differ; the full list is below each count.

| Result | Symbols that differ |
|---|---|
| gtap7_10x7 `gams_nlp_altertax_ifsub0` | 140 symbols |
| gtap7_10x7 `gams_nlp_altertax_ifsub1` | 130 symbols |
| gtap7_10x7 `gams_nlp_gtap_ifsub0` | 139 symbols |
| gtap7_10x7 `gams_nlp_gtap_ifsub1` | 131 symbols |
| gtap7_3x3 `gams_mcp_altertax_ifsub1` | 156 symbols |
| gtap7_3x3 `gams_mcp_gtap_ifsub1` | 167 symbols |
| gtap7_3x3 `gams_nlp_altertax_ifsub0` | 137 symbols |
| gtap7_3x3 `gams_nlp_altertax_ifsub1` | 127 symbols |
| gtap7_3x3 `gams_nlp_gtap_ifsub0` | 136 symbols |
| gtap7_3x3 `gams_nlp_gtap_ifsub1` | 138 symbols |
| gtap7_3x4 `gams_nlp_altertax_ifsub0` | 136 symbols |
| gtap7_3x4 `gams_nlp_altertax_ifsub1` | 127 symbols |
| gtap7_5x5 `gams_nlp_altertax_ifsub0` | zero not recorded |
| gtap7_5x5 `gams_nlp_altertax_ifsub1` | 127 symbols |
| gtap7_5x5 `gams_nlp_gtap_ifsub0` | 149 symbols |
| gtap7_5x5 `gams_nlp_gtap_ifsub1` | 134 symbols |

gtap7_5x5 `gams_nlp_altertax_ifsub0` has no zero variant: IPOPT stops in the check period (model status 7, evaluation errors in `pfeq`, `x**c` with x < 0); the header variant solves. See the 2b ledger.

### Symbol lists

<details><summary>gtap7_10x7 `gams_nlp_altertax_ifsub0`</summary>

af, and, arent, ava, betag, betap, betas, chiSave, chif, cv, ev, factY, fctts, fcttx, ftrv, gdpmp, gx, kapEnd, nd, ndeq, p, pa, pabs, paeq, pcons, pd, pdeq, pdp, pdpeq, pe, peeq, pefob, pefobeq, peq, pet, peteq, pf, pfa, pfact, pfaeq, pft, pfy, pg, pgdpmp, pgeq, phi, phiP, phieq, pi, pieq, pigbl, pm, pmcif, pmcifeq, pmeq, pmp, pmpeq, pmt, pmteq, pmuv, pnd, pndeq, pp, ppeq, ps, psave, pseq, ptmg, ptmgeq, pva, pvaeq, pwmg, pwmgeq, px, pxeq, regY, rgdpmp, rorc, rore, rorg, rsav, rsaveq, sam, savf, u, ug, uh, us, va, vaeq, x, xa, xaceq, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xi, xigbl, xm, xmeq, xmgm, xmgmeq, xmt, xmteq, xp, xpFlag, xpeq, xs, xscale, xseq, xtmg, xtmgeq, xw, xweq, xwmg, xwmgeq, yc, yceq, yg, ygeq, yi, ytax, ytaxInd, ytaxTot, ytaxshr, zcons, zconseq

</details>

<details><summary>gtap7_10x7 `gams_nlp_altertax_ifsub1`</summary>

af, and, arent, ava, betag, betap, betas, chiSave, chif, cv, ev, factY, fctts, fcttx, ftrv, gdpmp, gx, kapEnd, nd, ndeq, p, pa, pabs, paeq, pcons, pd, pdeq, pdp, pe, peeq, pefob, peq, pet, peteq, pf, pfa, pfact, pft, pfy, pg, pgdpmp, pgeq, phi, phiP, phieq, pi, pieq, pigbl, pm, pmcif, pmp, pmt, pmteq, pmuv, pnd, pndeq, pp, ps, psave, pseq, ptmg, ptmgeq, pva, pvaeq, pwmg, px, pxeq, regY, rgdpmp, rorc, rore, rorg, rsav, rsaveq, sam, savf, u, ug, uh, us, va, vaeq, x, xa, xaceq, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xi, xigbl, xm, xmeq, xmgm, xmt, xmteq, xp, xpFlag, xpeq, xs, xscale, xseq, xtmg, xtmgeq, xw, xweq, xwmg, yc, yceq, yg, ygeq, yi, ytax, ytaxInd, ytaxTot, ytaxshr, zcons, zconseq

</details>

<details><summary>gtap7_10x7 `gams_nlp_gtap_ifsub0`</summary>

af, and, arent, ava, betag, betap, betas, chiSave, chif, cv, ev, factY, fctts, fcttx, ftrv, gdpmp, gx, kapEnd, nd, ndeq, p, pa, pabs, paeq, pcons, pd, pdeq, pdp, pdpeq, pe, peeq, pefob, pefobeq, peq, pet, peteq, pf, pfa, pfact, pfaeq, pft, pfy, pg, pgdpmp, pgeq, phi, phiP, phieq, pi, pieq, pigbl, pm, pmcif, pmcifeq, pmeq, pmp, pmpeq, pmt, pmteq, pmuv, pnd, pndeq, pp, ppeq, ps, psave, pseq, ptmg, ptmgeq, pva, pvaeq, pwmg, pwmgeq, px, pxeq, regY, rgdpmp, rorc, rore, rorg, rsav, rsaveq, sam, savf, u, ug, uh, us, va, vaeq, x, xa, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xi, xigbl, xm, xmeq, xmgm, xmgmeq, xmt, xmteq, xp, xpFlag, xpeq, xs, xscale, xseq, xtmg, xtmgeq, xw, xweq, xwmg, xwmgeq, yc, yceq, yg, ygeq, yi, ytax, ytaxInd, ytaxTot, ytaxshr, ytaxshreq, zcons

</details>

<details><summary>gtap7_10x7 `gams_nlp_gtap_ifsub1`</summary>

af, and, arent, ava, betag, betap, betas, chiSave, chif, cv, ev, factY, fctts, fcttx, ftrv, gdpmp, gx, kapEnd, nd, ndeq, p, pa, pabs, paeq, pcons, pd, pdeq, pdp, pe, peeq, pefob, peq, pet, peteq, pf, pfa, pfact, pfeq, pft, pfy, pg, pgdpmp, pgeq, phi, phiP, phieq, pi, pieq, pigbl, pm, pmcif, pmp, pmt, pmteq, pmuv, pnd, pndeq, pp, ps, psave, pseq, ptmg, ptmgeq, pva, pvaeq, pwmg, px, pxeq, regY, rgdpmp, rorc, rore, rorg, rsav, rsaveq, sam, savf, u, ug, uh, us, va, vaeq, x, xa, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xi, xigbl, xm, xmeq, xmgm, xmt, xmteq, xp, xpFlag, xpeq, xs, xscale, xseq, xtmg, xtmgeq, xw, xweq, xwmg, yc, yceq, yg, ygeq, yi, ytax, ytaxInd, ytaxTot, ytaxshr, ytaxshreq, zcons, zconseq

</details>

<details><summary>gtap7_3x3 `gams_mcp_altertax_ifsub1`</summary>

af, and, arent, arenteq, ava, betag, betap, betas, capAccteq, chif, chifeq, cv, cveq, ev, eveq, factY, factYeq, fctts, fcttx, ftrv, gdpmp, gdpmpeq, gx, kapEnd, kapEndeq, nd, ndeq, p, pa, pabs, pabseq, paeq, pcons, pconseq, pd, pdeq, pdp, pe, peeq, pefob, peq, pet, peteq, pf, pfa, pfact, pfacteq, pfeq, pft, pfteq, pfy, pg, pgdpmp, pgdpmpeq, pgeq, phi, phiP, phiPeq, phieq, pi, pieq, pigbl, pigbleq, pm, pmcif, pmp, pmt, pmteq, pmuv, pmuveq, pnumeq, pp, ps, psave, psaveeq, pseq, regY, regYeq, rgdpmp, rgdpmpeq, rorc, rorceq, rore, roreeq, rorg, rorgeq, rsav, rsaveq, sam, savf, savfeq, u, ueq, ug, ugeq, uh, uheq, us, useq, va, vaeq, walraseq, x, xa, xaceq, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xgeq, xi, xieq, xigbl, xigbleq, xm, xmeq, xmgm, xmt, xmteq, xp, xpeq, xs, xseq, xtmg, xtmgeq, xw, xweq, xwmg, yc, yceq, yg, ygeq, yi, yieq, ytax, ytaxInd, ytaxIndeq, ytaxTot, ytaxToteq, ytaxeq, ytaxshr, ytaxshreq, zcons, zconseq

</details>

<details><summary>gtap7_3x3 `gams_mcp_gtap_ifsub1`</summary>

af, and, arent, arenteq, ava, betag, betap, betas, capAccteq, chiSave, chiSaveeq, chif, chifeq, cv, cveq, ev, eveq, factY, factYeq, fctts, fcttx, ftrv, gdpmp, gdpmpeq, gx, kapEnd, kapEndeq, nd, ndeq, p, pa, pabs, pabseq, paeq, pcons, pconseq, pd, pdeq, pdp, pe, peeq, pefob, peq, pet, peteq, pf, pfa, pfact, pfacteq, pfeq, pft, pfteq, pfy, pg, pgdpmp, pgdpmpeq, pgeq, phi, phiP, phiPeq, phieq, pi, pieq, pigbl, pigbleq, pm, pmcif, pmp, pmt, pmteq, pmuv, pmuveq, pnd, pndeq, pnumeq, pp, ps, psave, psaveeq, pseq, ptmg, ptmgeq, pva, pvaeq, pwmg, px, pxeq, regY, regYeq, rgdpmp, rgdpmpeq, rorc, rorceq, rore, roreeq, rorg, rorgeq, rsav, rsaveq, sam, savf, savfeq, u, ueq, ug, ugeq, uh, uheq, us, useq, va, vaeq, walraseq, x, xa, xaceq, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xgeq, xi, xieq, xigbl, xigbleq, xm, xmeq, xmgm, xmt, xmteq, xp, xpeq, xs, xseq, xtmg, xtmgeq, xw, xweq, xwmg, yc, yceq, yg, ygeq, yi, yieq, ytax, ytaxInd, ytaxIndeq, ytaxTot, ytaxToteq, ytaxeq, ytaxshr, ytaxshreq, zcons, zconseq

</details>

<details><summary>gtap7_3x3 `gams_nlp_altertax_ifsub0`</summary>

af, and, arent, ava, betag, betap, betas, chiSave, chif, cv, ev, factY, fctts, fcttx, ftrv, gdpmp, gx, kapEnd, nd, ndeq, p, pa, pabs, paeq, pcons, pd, pdeq, pdp, pdpeq, pe, peeq, pefob, pefobeq, peq, pet, peteq, pf, pfa, pfact, pfaeq, pft, pfy, pg, pgdpmp, pgeq, phi, phiP, phieq, pi, pieq, pigbl, pm, pmcif, pmcifeq, pmeq, pmp, pmpeq, pmt, pmteq, pmuv, pnd, pndeq, pp, ppeq, ps, psave, pseq, ptmg, ptmgeq, pva, pvaeq, pwmg, pwmgeq, px, pxeq, regY, rgdpmp, rorc, rore, rorg, rsav, rsaveq, sam, savf, u, ug, uh, us, va, vaeq, x, xa, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xi, xigbl, xm, xmeq, xmgm, xmgmeq, xmt, xmteq, xp, xpeq, xs, xseq, xtmg, xtmgeq, xw, xweq, xwmg, xwmgeq, yc, yceq, yg, ygeq, yi, ytax, ytaxInd, ytaxTot, ytaxshr, zcons, zconseq

</details>

<details><summary>gtap7_3x3 `gams_nlp_altertax_ifsub1`</summary>

af, and, arent, ava, betag, betap, betas, chiSave, chif, cv, ev, factY, fctts, fcttx, ftrv, gdpmp, gx, kapEnd, nd, ndeq, p, pa, pabs, paeq, pcons, pd, pdeq, pdp, pe, peeq, pefob, peq, pet, peteq, pf, pfa, pfact, pft, pfy, pg, pgdpmp, pgeq, phi, phiP, phieq, pi, pieq, pigbl, pm, pmcif, pmp, pmt, pmteq, pmuv, pnd, pndeq, pp, ps, psave, pseq, ptmg, ptmgeq, pva, pvaeq, pwmg, px, pxeq, regY, rgdpmp, rorc, rore, rorg, rsav, rsaveq, sam, savf, u, ug, uh, us, va, vaeq, x, xa, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xi, xigbl, xm, xmeq, xmgm, xmt, xmteq, xp, xpeq, xs, xseq, xtmg, xtmgeq, xw, xweq, xwmg, yc, yceq, yg, ygeq, yi, ytax, ytaxInd, ytaxTot, ytaxshr, zcons, zconseq

</details>

<details><summary>gtap7_3x3 `gams_nlp_gtap_ifsub0`</summary>

af, and, arent, ava, betag, betap, betas, chiSave, chif, cv, ev, factY, fctts, fcttx, ftrv, gdpmp, gx, kapEnd, nd, ndeq, p, pa, pabs, paeq, pcons, pd, pdeq, pdp, pdpeq, pe, peeq, pefob, pefobeq, peq, pet, peteq, pf, pfa, pfact, pfaeq, pft, pfy, pg, pgdpmp, pgeq, phi, phiP, phieq, pi, pieq, pigbl, pm, pmcif, pmcifeq, pmeq, pmp, pmpeq, pmt, pmteq, pmuv, pnd, pndeq, pp, ppeq, ps, psave, pseq, ptmg, ptmgeq, pva, pvaeq, pwmg, pwmgeq, px, pxeq, regY, rgdpmp, rorc, rore, rorg, rsav, rsaveq, sam, savf, u, ug, uh, us, va, vaeq, x, xa, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xi, xigbl, xm, xmeq, xmgm, xmgmeq, xmt, xmteq, xp, xpeq, xs, xseq, xtmg, xtmgeq, xw, xweq, xwmg, xwmgeq, yc, yceq, yg, ygeq, yi, ytax, ytaxInd, ytaxTot, ytaxshr, zcons

</details>

<details><summary>gtap7_3x3 `gams_nlp_gtap_ifsub1`</summary>

af, and, arent, arenteq, ava, betag, betap, betas, chiSave, chif, chifeq, cv, cveq, ev, eveq, factY, fctts, fcttx, ftrv, gdpmp, gx, kapEnd, kapEndeq, nd, ndeq, p, pa, pabs, pabseq, paeq, pcons, pd, pdeq, pdp, pe, peeq, pefob, peq, pet, peteq, pf, pfa, pfact, pft, pfy, pg, pgdpmp, pgdpmpeq, pgeq, phi, phiP, phieq, pi, pieq, pigbl, pm, pmcif, pmp, pmt, pmteq, pmuv, pnd, pndeq, pp, ps, psave, pseq, ptmg, ptmgeq, pva, pvaeq, pwmg, px, pxeq, regY, rgdpmp, rorc, rorceq, rore, roreeq, rorg, rorgeq, rsav, rsaveq, sam, savf, u, ug, uh, us, va, vaeq, x, xa, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xi, xigbl, xm, xmeq, xmgm, xmt, xmteq, xp, xpeq, xs, xseq, xtmg, xtmgeq, xw, xweq, xwmg, yc, yceq, yg, ygeq, yi, ytax, ytaxInd, ytaxTot, ytaxeq, ytaxshr, ytaxshreq, zcons

</details>

<details><summary>gtap7_3x4 `gams_nlp_altertax_ifsub0`</summary>

af, and, arent, ava, betag, betap, betas, chif, cv, ev, factY, fctts, fcttx, ftrv, gdpmp, gx, kapEnd, nd, ndeq, p, pa, pabs, paeq, pcons, pd, pdeq, pdp, pdpeq, pe, peeq, pefob, pefobeq, peq, pet, peteq, pf, pfa, pfact, pfaeq, pft, pfy, pg, pgdpmp, pgeq, phi, phiP, phieq, pi, pieq, pigbl, pm, pmcif, pmcifeq, pmeq, pmp, pmpeq, pmt, pmteq, pmuv, pnd, pndeq, pp, ppeq, ps, psave, pseq, ptmg, ptmgeq, pva, pvaeq, pwmg, pwmgeq, px, pxeq, regY, rgdpmp, rorc, rore, rorg, rsav, rsaveq, sam, savf, u, ug, uh, us, va, vaeq, x, xa, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xi, xigbl, xm, xmeq, xmgm, xmgmeq, xmt, xmteq, xp, xpeq, xs, xseq, xtmg, xtmgeq, xw, xweq, xwmg, xwmgeq, yc, yceq, yg, ygeq, yi, ytax, ytaxInd, ytaxTot, ytaxshr, zcons, zconseq

</details>

<details><summary>gtap7_3x4 `gams_nlp_altertax_ifsub1`</summary>

af, and, arent, ava, betag, betap, betas, chiSave, chif, cv, ev, factY, fctts, fcttx, ftrv, gdpmp, gx, kapEnd, nd, ndeq, p, pa, pabs, paeq, pcons, pd, pdeq, pdp, pe, peeq, pefob, peq, pet, peteq, pf, pfa, pfact, pft, pfy, pg, pgdpmp, pgeq, phi, phiP, phieq, pi, pieq, pigbl, pm, pmcif, pmp, pmt, pmteq, pmuv, pnd, pndeq, pp, ps, psave, pseq, ptmg, ptmgeq, pva, pvaeq, pwmg, px, pxeq, regY, rgdpmp, rorc, rore, rorg, rsav, rsaveq, sam, savf, u, ug, uh, us, va, vaeq, x, xa, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xi, xigbl, xm, xmeq, xmgm, xmt, xmteq, xp, xpeq, xs, xseq, xtmg, xtmgeq, xw, xweq, xwmg, yc, yceq, yg, ygeq, yi, ytax, ytaxInd, ytaxTot, ytaxshr, zcons, zconseq

</details>

<details><summary>gtap7_5x5 `gams_nlp_altertax_ifsub1`</summary>

af, and, arent, ava, betag, betap, betas, chiSave, chif, cv, ev, factY, fctts, fcttx, ftrv, gdpmp, gx, kapEnd, nd, ndeq, p, pa, pabs, paeq, pcons, pd, pdeq, pdp, pe, peeq, pefob, peq, pet, peteq, pf, pfa, pfact, pft, pfy, pg, pgdpmp, pgeq, phi, phiP, phieq, pi, pieq, pigbl, pm, pmcif, pmp, pmt, pmteq, pmuv, pnd, pndeq, pp, ps, psave, pseq, ptmg, ptmgeq, pva, pvaeq, pwmg, px, pxeq, regY, rgdpmp, rorc, rore, rorg, rsav, rsaveq, sam, savf, u, ug, uh, us, va, vaeq, x, xa, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xi, xigbl, xm, xmeq, xmgm, xmt, xmteq, xp, xpeq, xs, xseq, xtmg, xtmgeq, xw, xweq, xwmg, yc, yceq, yg, ygeq, yi, ytax, ytaxInd, ytaxTot, ytaxshr, zcons, zconseq

</details>

<details><summary>gtap7_5x5 `gams_nlp_gtap_ifsub0`</summary>

af, and, arent, arenteq, ava, betag, betap, betas, chiSave, chif, chifeq, cv, cveq, ev, eveq, factY, fctts, fcttx, ftrv, gdpmp, gx, kapEnd, kapEndeq, nd, ndeq, p, pa, pabs, pabseq, paeq, pcons, pd, pdeq, pdp, pdpeq, pe, peeq, pefob, pefobeq, peq, pet, peteq, pf, pfa, pfact, pfacteq, pfaeq, pft, pfy, pfyeq, pg, pgdpmp, pgeq, phi, phiP, phieq, pi, pieq, pigbl, pm, pmcif, pmcifeq, pmeq, pmp, pmpeq, pmt, pmteq, pmuv, pnd, pndeq, pp, ppeq, ps, psave, pseq, ptmg, ptmgeq, pva, pvaeq, pwmg, pwmgeq, px, pxeq, regY, rgdpmp, rorc, rorceq, rore, roreeq, rorg, rorgeq, rsav, rsaveq, sam, savf, u, ug, uh, us, va, vaeq, x, xa, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xi, xigbl, xm, xmeq, xmgm, xmgmeq, xmt, xmteq, xp, xpeq, xs, xseq, xtmg, xtmgeq, xw, xweq, xwmg, xwmgeq, yc, yceq, yg, ygeq, yi, ytax, ytaxInd, ytaxTot, ytaxeq, ytaxshr, ytaxshreq, zcons

</details>

<details><summary>gtap7_5x5 `gams_nlp_gtap_ifsub1`</summary>

af, and, arent, arenteq, ava, betag, betap, betas, chiSave, chif, cv, ev, factY, fctts, fcttx, ftrv, gdpmp, gx, kapEnd, kapEndeq, nd, ndeq, p, pa, pabs, pabseq, paeq, pcons, pd, pdeq, pdp, pe, peeq, pefob, peq, pet, peteq, pf, pfa, pfact, pfacteq, pft, pfy, pg, pgdpmp, pgdpmpeq, pgeq, phi, phiP, phieq, pi, pieq, pigbl, pm, pmcif, pmp, pmt, pmteq, pmuv, pnd, pndeq, pp, ps, psave, pseq, ptmg, ptmgeq, pva, pvaeq, pwmg, px, pxeq, regY, rgdpmp, rorc, rorceq, rore, rorg, rsav, rsaveq, sam, savf, u, ug, uh, us, va, vaeq, x, xa, xageq, xaieq, xapeq, xatmgeq, xcshr, xcshreq, xd, xdeq, xds, xdseq, xeq, xet, xeteq, xf, xfeq, xg, xi, xigbl, xm, xmeq, xmgm, xmt, xmteq, xp, xpeq, xs, xseq, xtmg, xtmgeq, xw, xweq, xwmg, yc, yceq, yg, ygeq, yi, ytax, ytaxInd, ytaxTot, ytaxeq, ytaxshr, ytaxshreq, zcons

</details>

