"""Tests for Pyomo backend."""

import numpy as np
import pytest

from equilibria import Model
from equilibria.backends import PyomoBackend, Solution
from equilibria.blocks import CESValueAdded
from equilibria.core import Set


class TestSolution:
    """Tests for Solution class."""

    def test_solution_creation(self):
        """Test solution creation."""
        sol = Solution(
            model_name="Test",
            status="optimal",
            objective_value=0.0,
            variables={"VA": np.array([100.0, 150.0])},
            solve_time=1.5,
            iterations=42,
        )
        assert sol.model_name == "Test"
        assert sol.status == "optimal"
        assert sol.solve_time == 1.5

    def test_solution_get_variable(self):
        """Test getting variable values."""
        sol = Solution(
            model_name="Test",
            variables={"VA": np.array([100.0, 150.0])},
        )
        va = sol.get_variable("VA")
        assert np.array_equal(va, np.array([100.0, 150.0]))

    def test_solution_get_missing_variable(self):
        """Test getting missing variable."""
        sol = Solution(model_name="Test")
        assert sol.get_variable("MISSING") is None

    def test_solution_compare_equal(self):
        """Test comparing equal solutions."""
        sol1 = Solution(
            model_name="Test",
            variables={"VA": np.array([100.0, 150.0])},
        )
        sol2 = Solution(
            model_name="Test",
            variables={"VA": np.array([100.0, 150.0])},
        )

        comparison = sol1.compare(sol2)
        assert comparison["is_equal"] is True

    def test_solution_compare_different(self):
        """Test comparing different solutions."""
        sol1 = Solution(
            model_name="Test",
            variables={"VA": np.array([100.0, 150.0])},
        )
        sol2 = Solution(
            model_name="Test",
            variables={"VA": np.array([105.0, 150.0])},
        )

        comparison = sol1.compare(sol2, tolerance=1e-6)
        assert comparison["is_equal"] is False
        assert "VA" in comparison["differences"]

    def test_solution_to_dict(self):
        """Test solution serialization."""
        sol = Solution(
            model_name="Test",
            status="optimal",
            variables={"VA": np.array([100.0])},
        )
        d = sol.to_dict()

        assert d["model_name"] == "Test"
        assert d["status"] == "optimal"
        assert d["variables"]["VA"] == [100.0]


class TestPyomoBackend:
    """Tests for PyomoBackend class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model."""
        model = Model(name="TestModel")
        model.add_set(Set(name="J", elements=("sec1", "sec2")))
        model.add_set(Set(name="I", elements=("labor", "capital")))
        model.add_set(Set(name="F", elements=("firm",)))
        model.add_block(CESValueAdded(sigma=0.8))
        return model

    def test_backend_creation(self):
        """Test backend creation."""
        backend = PyomoBackend(solver="ipopt")
        assert backend.solver == "ipopt"

    def test_backend_build(self, simple_model):
        """Test building Pyomo model."""
        backend = PyomoBackend()
        backend.build(simple_model)

        assert backend.pyomo_model is not None
        assert hasattr(backend.pyomo_model, "J")
        assert hasattr(backend.pyomo_model, "I")

    def test_backend_build_creates_sets(self, simple_model):
        """Test that sets are created."""
        backend = PyomoBackend()
        backend.build(simple_model)

        # Check sets exist
        assert hasattr(backend.pyomo_model, "J")
        assert hasattr(backend.pyomo_model, "I")

        # Check set contents
        j_set = list(backend.pyomo_model.J)
        assert "sec1" in j_set
        assert "sec2" in j_set

    def test_backend_build_creates_parameters(self, simple_model):
        """Test that parameters are created."""
        backend = PyomoBackend()
        backend.build(simple_model)

        # Check parameters exist
        assert hasattr(backend.pyomo_model, "sigma_VA")
        assert hasattr(backend.pyomo_model, "beta_VA")

    def test_backend_build_creates_variables(self, simple_model):
        """Test that variables are created."""
        backend = PyomoBackend()
        backend.build(simple_model)

        # Check variables exist
        assert hasattr(backend.pyomo_model, "VA")
        assert hasattr(backend.pyomo_model, "FD")

    def test_backend_build_without_model_raises(self):
        """Test that solve without build raises error."""
        backend = PyomoBackend()
        with pytest.raises(RuntimeError):
            backend.solve()

    def test_list_available_solvers(self):
        """Test listing available solvers."""
        backend = PyomoBackend()
        solvers = backend.list_available_solvers()

        # Should return a list
        assert isinstance(solvers, list)
        # IPOPT should be available if installed

    def test_backend_repr(self):
        """Test string representation."""
        backend = PyomoBackend(solver="ipopt")
        repr_str = repr(backend)
        assert "PyomoBackend" in repr_str
        assert "ipopt" in repr_str
