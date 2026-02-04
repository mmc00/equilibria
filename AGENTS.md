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
- Model class with block system
- Equation blocks (Production, Trade, Demand, etc.)
- Solver backends (Pyomo, PyOptInterface)
- Templates (PEP, GTAP-like, etc.)
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

### Phase 2: Block System

#### 2.1 Block Base Classes
**Files to create:**
- `src/equilibria/blocks/base.py` - Block base class using Pydantic
- `src/equilibria/blocks/registry.py` - Block registration

**Requirements:**
- **MUST use Pydantic BaseModel** for all block classes
- Declare required sets using Pydantic fields
- Declare parameters with validation
- Declare variables with bounds
- Declare equations
- Metadata introspection via Pydantic model schema
- Support for block configuration through Pydantic models

**Pydantic Integration:**
```python
from pydantic import BaseModel, Field

class Block(BaseModel):
    name: str = Field(..., description="Block name")
    required_sets: list[str] = Field(default_factory=list)
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    variables: dict[str, VariableSpec] = Field(default_factory=dict)
    equations: list[EquationSpec] = Field(default_factory=list)
```

#### 2.2 Production Blocks
**Files to create:**
- `src/equilibria/blocks/production/ces_va.py` - CES value-added
- `src/equilibria/blocks/production/leontief.py` - Leontief intermediate
- `src/equilibria/blocks/production/cet.py` - CET transformation
- `src/equilibria/blocks/production/nested_ces.py` - Nested CES

**Equations needed:**
- CES aggregation
- First-order conditions (labor, capital)
- Zero-profit conditions

#### 2.3 Trade Blocks
**Files to create:**
- `src/equilibria/blocks/trade/armington.py` - Armington CES
- `src/equilibria/blocks/trade/cet_exports.py` - CET exports
- `src/equilibria/blocks/trade/small_open.py` - Small open economy

#### 2.4 Demand Blocks
**Files to create:**
- `src/equilibria/blocks/demand/les.py` - Linear Expenditure System
- `src/equilibria/blocks/demand/cobb_douglas.py` - Cobb-Douglas

#### 2.5 Institution Blocks
**Files to create:**
- `src/equilibria/blocks/institutions/household.py` - Household income
- `src/equilibria/blocks/institutions/government.py` - Government budget
- `src/equilibria/blocks/institutions/row.py` - Rest of world

#### 2.6 Equilibrium Blocks
**Files to create:**
- `src/equilibria/blocks/equilibrium/market_clearing.py` - Market clearing
- `src/equilibria/blocks/equilibrium/price_norm.py` - Price normalization

### Phase 3: Model Framework

#### 3.1 Model Class
**Files to create:**
- `src/equilibria/model.py` - Main Model class using Pydantic

**Features:**
- Add/remove blocks
- Automatic equation assembly
- Set validation across blocks
- Statistics (variables, equations, DOF)
- Calibration interface

**Pydantic Integration:**
```python
class Model(BaseModel):
    name: str = Field(..., description="Model name")
    blocks: list[Block] = Field(default_factory=list)
    set_manager: SetManager = Field(default_factory=SetManager)
    parameter_manager: ParameterManager = Field(default_factory=ParameterManager)
    variable_manager: VariableManager = Field(default_factory=VariableManager)
```

#### 3.2 Calibration
**Files to create:**
- `src/equilibria/calibration/base.py` - Calibration ABC using Pydantic
- `src/equilibria/calibration/ces.py` - CES calibration from SAM
- `src/equilibria/calibration/cet.py` - CET calibration
- `src/equilibria/calibration/les.py` - LES calibration

**Requirements:**
- Auto-calculate elasticities from SAM
- Support for user-provided elasticities
- Calibration report

### Phase 4: Solver Backends

#### 4.1 Backend Interface
**Files to create:**
- `src/equilibria/backends/base.py` - Backend ABC
- `src/equilibria/backends/solution.py` - Solution container using Pydantic

#### 4.2 Pyomo Backend
**Files to create:**
- `src/equilibria/backends/pyomo_backend.py`

**Requirements:**
- Translate blocks to Pyomo components
- Support IPOPT, CONOPT
- Handle complementarity conditions

#### 4.3 PyOptInterface Backend
**Files to create:**
- `src/equilibria/backends/poi_backend.py`

**Requirements:**
- Direct model construction
- Support HiGHS, other solvers

### Phase 5: Templates

#### 5.1 Template Base
**Files to create:**
- `src/equilibria/templates/base.py` using Pydantic

#### 5.2 Pre-built Models
**Files to create:**
- `src/equilibria/templates/simple_open.py` - 3-sector model
- `src/equilibria/templates/pep.py` - PEP standard CGE
- `src/equilibria/templates/gtap_like.py` - GTAP-style

### Phase 6: Analysis Tools

#### 6.1 Solution Comparison
**Files to create:**
- `src/equilibria/analysis/comparison.py`

**Features:**
- Baseline vs counterfactual
- Percentage changes
- Export to Excel

#### 6.2 Sensitivity Analysis
**Files to create:**
- `src/equilibria/analysis/sensitivity.py`

#### 6.3 Visualization
**Files to create:**
- `src/equilibria/viz/charts.py`
- `src/equilibria/viz/network.py` - SAM visualization

### Phase 7: Documentation & Examples

#### 7.1 Documentation
- API docs for all public classes
- Tutorial notebooks
- Model template guides

#### 7.2 Examples
- `examples/basic_cge.py` - Simple model
- `examples/tax_policy.py` - Tax analysis
- `examples/trade_war.py` - Multi-region

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
3. Phase 2.1 - Block base classes (blocking)
4. Phase 3.1 - Model class (blocking)
5. Phase 2.2-2.3 - Key production/trade blocks
6. Phase 4.1-4.2 - Pyomo backend
7. Phase 3.2 - Calibration
8. Phase 5 - Templates
9. Phase 6 - Analysis
10. Phase 7 - Docs & examples

## Notes

- GDX reader already supports up to 6D parameters
- Use existing GDX infrastructure for data I/O
- Build incrementally - test each block before adding next
- Focus on static models first, dynamic later
- Keep dependencies minimal - numpy, pandas, pydantic are core
- **All block classes MUST inherit from Pydantic BaseModel**
- **All configuration classes MUST use Pydantic for validation**
