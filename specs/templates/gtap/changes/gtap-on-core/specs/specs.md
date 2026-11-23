## MODIFIED Requirements

### Requirement: A model without the ShockBlock is rejected
Source: change tariff-to-shock-block, ADR-0001

The `GTAP` template must raise `ValueError` before any build or solve when asked
for a model without the ShockBlock (`builder="monolith"`, the monolith
`GTAPMultiPeriodModel`): there is nowhere to write the shock, the default tariff
included (`tests/templates/gtap/test_gtap_recipe.py::test_bad_builder`).
