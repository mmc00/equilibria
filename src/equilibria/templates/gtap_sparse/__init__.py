"""GTAP sparse-trade template.

OUR levels block model with padding trade routes (base flow ~0, e.g. non-existent
bilateral routes and unused transport-margin cells) fixed to their ~0 benchmark
value and their route constraints deactivated. This is the "condensation" GEMPACK
does implicitly: routes with zero base flow never enter the active system, so the
Jacobian loses the extreme-scale (1e11..1e14) entries those degenerate cells
contribute — the source of IPOPT's overflow (inf_pr 2.29e9) on the largest datasets.

Sibling of templates/gtap_loglevels: same blocks (equilibria.blocks.gtap), own
composer + multiperiod driver. Does NOT touch the production model.
"""
