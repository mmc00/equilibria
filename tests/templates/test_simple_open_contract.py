from __future__ import annotations

from equilibria.contracts import (
    ModelBoundsConfig,
    ModelClosureConfig,
    ModelContract,
    ModelEquationConfig,
    ModelReferenceConfig,
    ModelRuntimeConfig,
)
from equilibria.templates import (
    SimpleOpenBoundsConfig,
    SimpleOpenClosureConfig,
    SimpleOpenContract,
    SimpleOpenEconomy,
    SimpleOpenEquationConfig,
    SimpleOpenReferenceConfig,
    SimpleOpenRuntimeConfig,
    build_simple_open_closure_config,
    build_simple_open_contract,
    build_simple_open_runtime_config,
)


def test_simple_open_contract_builders_default() -> None:
    contract = build_simple_open_contract("simple_open_v1")
    runtime_config = build_simple_open_runtime_config("default_template")

    assert isinstance(contract, SimpleOpenContract)
    assert contract.name == "simple_open_v1"
    assert contract.closure.name == "simple_open_default"
    assert contract.closure.numeraire == "PFX"
    assert contract.equations.include == ("EQ_VA", "EQ_INT", "EQ_CET")
    assert contract.bounds.free == ("ER", "CAB")

    assert isinstance(runtime_config, SimpleOpenRuntimeConfig)
    assert runtime_config.problem_type == "template"
    assert runtime_config.solver == "none"
    assert runtime_config.jacobian_mode == "analytic"
    assert runtime_config.reference.enabled is False


def test_simple_open_contract_accepts_mapping_override() -> None:
    contract = build_simple_open_contract(
        {
            "closure": {
                "name": "flexible_external_balance",
                "fixed": ["PFX"],
                "endogenous": ["ER", "CAB", "FSAV"],
            },
            "equations": {
                "activation_masks": "all_active",
            },
        }
    )

    assert contract.closure.name == "flexible_external_balance"
    assert contract.closure.fixed == ("PFX",)
    assert contract.closure.endogenous == ("ER", "CAB", "FSAV")
    assert contract.equations.activation_masks == "all_active"


def test_simple_open_types_inherit_generic_base() -> None:
    assert issubclass(SimpleOpenClosureConfig, ModelClosureConfig)
    assert issubclass(SimpleOpenEquationConfig, ModelEquationConfig)
    assert issubclass(SimpleOpenBoundsConfig, ModelBoundsConfig)
    assert issubclass(SimpleOpenContract, ModelContract)
    assert issubclass(SimpleOpenReferenceConfig, ModelReferenceConfig)
    assert issubclass(SimpleOpenRuntimeConfig, ModelRuntimeConfig)


def test_simple_open_template_resolves_contract_and_runtime() -> None:
    template = SimpleOpenEconomy(
        contract="simple_open_v1",
        runtime_config={"name": "default_template"},
    )
    info = template.get_info()

    assert isinstance(template.contract, SimpleOpenContract)
    assert isinstance(template.runtime_config, SimpleOpenRuntimeConfig)
    assert info["contract"]["name"] == "simple_open_v1"
    assert info["runtime_config"]["name"] == "default_template"


def test_simple_open_closure_builder_named_variant() -> None:
    closure = build_simple_open_closure_config("flexible_external_balance")

    assert closure.name == "flexible_external_balance"
    assert closure.fixed == ("PFX",)
    assert closure.endogenous == ("ER", "CAB", "FSAV")
