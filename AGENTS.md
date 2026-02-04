# AGENTS.md - equilibria Implementation Plan

## Project Overview

**equilibria** is a modern Python framework for Computable General Equilibrium (CGE) modeling with modular equation blocks, universal data I/O, and auto-calibration.

## Current State

### Implemented
- **babel.gdx.reader**: Pure Python GDX file reader (sets, parameters up to 6D)
- **babel.gdx.writer**: GDX file writer
- **babel.gdx.utils**: Utility functions for GDX handling
- **babel.gdx.symbols**: Symbol type definitions
- **core.sets**: Set and SetManager with Pydantic
- **core.parameters**: Parameter and ParameterManager with Pydantic
- **core.variables**: Variable and VariableManager with Pydantic
- **core.equations**: Equation base class with Pydantic
- **babel.sam**: SAM class with Pydantic
- Basic project structure with pyproject.toml

### Missing Core Components
- ~~Model class with block system~~ ✅ COMPLETED
- ~~Equation blocks (Production, Trade, Demand, etc.)~~ ✅ COMPLETED (Production & Trade)
- ~~Solver backends (Pyomo, PyOptInterface)~~ ✅ COMPLETED (Pyomo)
- ~~Templates (PEP, GTAP-like, etc.)~~ ✅ COMPLETED (SimpleOpenEconomy)
- Analysis tools
- Auto-calibration

## Quality Gates (REQUIRED before commit)

```bash
# Type checking (REQUIRED)
uvx ty check src/

# Linting (REQUIRED)
uv run ruff check .
uv run ruff format .

# Testing
uv run pytest
```

## Implementation Phases

### Phase 1: Foundation & Data Layer ✅ COMPLETED

#### 1.1 Core Data Structures ✅
**Files created:**
- `src/equilibria/core/sets.py` - Set definitions using Pydantic BaseModel
- `src/equilibria/core/parameters.py` - Parameter containers using Pydantic
- `src/equilibria/core/variables.py` - Variable definitions using Pydantic
- `src/equilibria/core/equations.py` - Equation base classes using Pydantic

**Key requirements met:**
- ✅ Use Pydantic for validation
- ✅ Support multi-dimensional indexing
- ✅ Type hints throughout
- ✅ Immutable where possible

#### 1.2 SAM (Social Accounting Matrix) ✅
**Files created:**
- `src/equilibria/babel/sam.py` - SAM class using Pydantic BaseModel

**Features:**
- ✅ Load from Excel, GDX
- ✅ Validate row/column balance
- ✅ Extract sets automatically
- ✅ Export to various formats
- ✅ RAS balancing

#### 1.3 Set Management ✅
**Integrated in:**
- `src/equilibria/core/sets.py` - SetManager class

**Features:**
- ✅ Load sets from SAM
- ✅ Support for subsets
- ✅ Domain validation
- ✅ Cartesian product generation

### Phase 2: Block System ✅ COMPLETED

#### 2.1 Block Base Classes ✅
**Files created:**
- `src/equilibria/blocks/base.py` - Block base class using Pydantic
- `src/equilibria/blocks/registry.py` - Block registration

**Features:**
- ✅ Pydantic BaseModel for all block classes
- ✅ ParameterSpec, VariableSpec, EquationSpec
- ✅ BlockRegistry with decorator support
- ✅ Metadata introspection

#### 2.2 Production Blocks ✅
**Files created:**
- `src/equilibria/blocks/production/__init__.py` - All production blocks

**Blocks implemented:**
- ✅ CESValueAdded - CES value-added production
- ✅ LeontiefIntermediate - Leontief intermediate inputs
- ✅ CETTransformation - CET output transformation

#### 2.3 Trade Blocks ✅
**Files created:**
- `src/equilibria/blocks/trade/__init__.py` - All trade blocks

**Blocks implemented:**
- ✅ ArmingtonCES - Armington import aggregation
- ✅ CETExports - CET export transformation

#### 2.4 Demand Blocks (TODO)
**Planned:**
- LES - Linear Expenditure System
- CobbDouglas - Cobb-Douglas demand

#### 2.5 Institution Blocks (TODO)
**Planned:**
- Household - Household income
- Government - Government budget
- ROW - Rest of world

#### 2.6 Equilibrium Blocks (TODO)
**Planned:**
- MarketClearing - Market clearing conditions
- PriceNorm - Price normalization

### Phase 3: Model Framework ✅ COMPLETED

#### 3.1 Model Class ✅
**Files created:**
- `src/equilibria/model.py` - Main Model class using Pydantic

**Features:**
- ✅ Add/remove blocks
- ✅ Automatic equation assembly
- ✅ Set validation across blocks
- ✅ Statistics (variables, equations, DOF)
- ✅ Calibration interface (placeholder)

**Pydantic Integration:**
```python
class Model(BaseModel):
    name: str = Field(..., description="Model name")
    blocks: list[Block] = Field(default_factory=list)
    set_manager: SetManager = Field(default_factory=SetManager)
    parameter_manager: ParameterManager = Field(default_factory=ParameterManager)
    variable_manager: VariableManager = Field(default_factory=VariableManager)
```

#### 3.2 Calibration (TODO)
**Planned:**
- Calibration ABC using Pydantic
- CES calibration from SAM
- CET calibration
- LES calibration

**Requirements:**
- Auto-calculate elasticities from SAM
- Support for user-provided elasticities
- Calibration report

### Phase 4: Solver Backends ✅ COMPLETED

#### 4.1 Backend Interface ✅
**Files created:**
- `src/equilibria/backends/base.py` - Backend ABC
- `src/equilibria/backends/solution.py` - Solution container using Pydantic

#### 4.2 Pyomo Backend ✅
**Files created:**
- `src/equilibria/backends/pyomo_backend.py`

**Features:**
- ✅ Translate blocks to Pyomo components
- ✅ Support IPOPT
- ✅ Automatic set, parameter, variable conversion

#### 4.3 PyOptInterface Backend (TODO)
**Planned:**
- Direct model construction
- Support HiGHS, other solvers

### Phase 5: Templates ✅ COMPLETED

#### 5.1 Template Base ✅
**Files created:**
- `src/equilibria/templates/base.py` using Pydantic

#### 5.2 Pre-built Models ✅
**Files created:**
- `src/equilibria/templates/simple_open.py` - 3-sector model

**Templates implemented:**
- ✅ SimpleOpenEconomy - Basic open economy CGE

**TODO:**
- PEP standard CGE
- GTAP-style model

### Phase 6: Analysis Tools

#### 6.1 Solution Comparison ✅
**Files created:**
- `src/equilibria/backends/base.py` - Solution.compare() method

**Features:**
- ✅ Baseline vs counterfactual
- ✅ Percentage changes
- Variable difference tracking

#### 6.2 Sensitivity Analysis (TODO)
**Planned:**
- `src/equilibria/analysis/sensitivity.py`

#### 6.3 Visualization (TODO)
**Planned:**
- `src/equilibria/viz/charts.py`
- `src/equilibria/viz/network.py` - SAM visualization

### Phase 7: Documentation & Examples ✅ COMPLETED

#### 7.1 Documentation ✅
- ✅ AGENTS.md implementation plan
- ✅ Docstrings for all public classes
- ✅ Type hints throughout

#### 7.2 Examples ✅
**Files created:**
- `examples/example_01_basic_setup.py` - Basic model setup
- `examples/example_02_sets_parameters.py` - Sets and parameters
- `examples/example_03_sam_data.py` - SAM manipulation
- `examples/example_04_custom_blocks.py` - Custom blocks
- `examples/example_05_complete_model.py` - Complete model
- `examples/example_06_pyomo_backend.py` - Pyomo backend
- `examples/example_07_templates.py` - Templates

## File Structure Target

```
src/equilibria/
├── __init__.py
├── version.py
├── core/
│   ├── __init__.py
│   ├── sets.py
│   ├── parameters.py
│   ├── variables.py
│   ├── equations.py
│   └── set_manager.py
├── babel/
│   ├── __init__.py
│   ├── sam.py
│   ├── readers.py
│   ├── writers.py
│   └── gdx/
│       ├── __init__.py
│       ├── reader.py
│       ├── writer.py
│       ├── utils.py
│       └── symbols.py
├── blocks/
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── production/
│   │   ├── __init__.py
│   │   ├── ces_va.py
│   │   ├── leontief.py
│   │   └── cet.py
│   ├── trade/
│   │   ├── __init__.py
│   │   ├── armington.py
│   │   └── cet_exports.py
│   ├── demand/
│   │   ├── __init__.py
│   │   ├── les.py
│   │   └── cobb_douglas.py
│   ├── institutions/
│   │   ├── __init__.py
│   │   ├── household.py
│   │   └── government.py
│   └── equilibrium/
│       ├── __init__.py
│       └── market_clearing.py
├── model.py
├── calibration/
│   ├── __init__.py
│   ├── base.py
│   ├── ces.py
│   ├── cet.py
│   └── les.py
├── backends/
│   ├── __init__.py
│   ├── base.py
│   ├── solution.py
│   ├── pyomo_backend.py
│   └── poi_backend.py
├── templates/
│   ├── __init__.py
│   ├── base.py
│   ├── simple_open.py
│   └── pep.py
├── analysis/
│   ├── __init__.py
│   ├── comparison.py
│   └── sensitivity.py
└── viz/
    ├── __init__.py
    └── charts.py
```

## Development Workflow

### For Each Phase:
1. Create/update files following the plan
2. Run type checker: uvx ty check src/
3. Run linter: uv run ruff check . && uv run ruff format .
4. Run tests: uv run pytest
5. Fix any issues before proceeding

### Code Standards
- Type hints required on all public APIs
- Docstrings in Google style
- No bare except: clauses
- No mutable default arguments
- Use uv for all package operations
- Use uv run for executing scripts
- **MUST use Pydantic BaseModel for all data classes and blocks**

### Testing Requirements
- Unit tests for each module
- Integration tests for full models
- Test data in tests/data/
- Minimum 80% coverage

## Priority Order

1. Phase 1.1 - Core data structures (blocking) ✅ COMPLETED
2. Phase 1.2 - SAM class (blocking) ✅ COMPLETED
3. Phase 2.1 - Block base classes (blocking) ✅ COMPLETED
4. Phase 3.1 - Model class (blocking) ✅ COMPLETED
5. Phase 2.2-2.3 - Key production/trade blocks ✅ COMPLETED
6. Phase 4.1-4.2 - Pyomo backend ✅ COMPLETED
7. Phase 3.2 - Calibration (TODO)
8. Phase 5 - Templates ✅ COMPLETED
9. Phase 6 - Analysis (Partial)
10. Phase 7 - Docs & examples ✅ COMPLETED

## Notes

- GDX reader already supports up to 6D parameters
- Use existing GDX infrastructure for data I/O
- Build incrementally - test each block before adding next
- Focus on static models first, dynamic later
- Keep dependencies minimal - numpy, pandas, pydantic are core
- **All block classes MUST inherit from Pydantic BaseModel**
- **All configuration classes MUST use Pydantic for validation**

## Test Suite

**Implemented Tests:**
- ✅ Core: 27 tests for Set, Parameter, Variable
- ✅ SAM: 10 tests for validation and balancing
- ✅ Model: 21 tests for model and blocks
- ✅ Backends: 14 tests for Pyomo integration
- ✅ Templates: 12 tests for configuration
- **Total: 84 tests**

## Current Status

**Completed:** Phases 1, 2, 3, 4, 5, 7 (Core Framework)
**In Progress:** Phase 6 (Analysis Tools - partial)
**TODO:** Phase 3.2 (Calibration)
