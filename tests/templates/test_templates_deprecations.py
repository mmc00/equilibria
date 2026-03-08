from __future__ import annotations

import warnings


def test_templates_pep_runner_export_warns_deprecated() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        from equilibria import templates

        _ = templates.PEPScenarioParityRunner

    assert any(isinstance(item.message, DeprecationWarning) for item in caught)


def test_pep_scenario_parity_runner_init_warns_deprecated() -> None:
    from equilibria.templates.pep_scenario_parity import PEPScenarioParityRunner

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _ = PEPScenarioParityRunner()

    assert any("deprecated" in str(item.message).lower() for item in caught)
