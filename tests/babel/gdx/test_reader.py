"""
equilibria/tests/babel/gdx/test_reader.py - Tests for GDX reader module.

Unit tests for equilibria.babel.gdx.reader module.

Tests cover:
- read_gdx(): Main entry point for reading GDX files
- read_header(): Header parsing with version, endianness, platform
- read_symbol_table(): Symbol table parsing
- Error handling: FileNotFoundError, ValueError for invalid files
- Integration tests with real GDX files
"""

# Standard library imports
from __future__ import annotations

import struct
from pathlib import Path

# Third-party imports
import pytest

# Local imports
from equilibria.babel.gdx.reader import (
    get_equations,
    get_parameters,
    get_sets,
    get_symbol,
    get_variables,
    read_domains_from_bytes,
    read_gdx,
    read_header_from_bytes,
    read_symbol_table_from_bytes,
    read_uel_from_bytes,
)

# Path to test fixtures
FIXTURES_DIR: Path = Path(__file__).parent.parent.parent / "fixtures"


class TestReadGdx:
    """Tests for read_gdx() function."""

    def test_file_not_found_raises_error(self) -> None:
        """Should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="GDX file not found"):
            read_gdx("nonexistent_file.gdx")

    def test_invalid_extension_raises_error(self, tmp_path: Path) -> None:
        """Should raise ValueError for non-.gdx extension."""
        invalid_file: Path = tmp_path / "data.txt"
        invalid_file.write_bytes(b"dummy content")

        with pytest.raises(ValueError, match="Expected .gdx file"):
            read_gdx(invalid_file)

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        """Should accept Path objects as input."""
        gdx_file: Path = tmp_path / "test.gdx"
        content: bytes = b"\x00" * 19 + b"GAMSGDX" + b"\x07\x00\x00\x00"
        content += b"\x00" * 100
        gdx_file.write_bytes(data=content)

        result: dict = read_gdx(gdx_file)

        assert result["filepath"] == str(object=gdx_file)
        assert "header" in result
        assert "symbols" in result

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """Should accept string paths as input."""
        gdx_file: Path = tmp_path / "test.gdx"
        content: bytes = b"\x00" * 19 + b"GAMSGDX" + b"\x07\x00\x00\x00"
        content += b"\x00" * 100
        gdx_file.write_bytes(data=content)

        result: dict = read_gdx(str(gdx_file))

        assert result["filepath"] == str(object=gdx_file)

    def test_returns_correct_structure(self, tmp_path: Path) -> None:
        """Should return dict with filepath, header, symbols, elements, domains."""
        gdx_file: Path = tmp_path / "test.gdx"
        content: bytes = b"\x00" * 19 + b"GAMSGDX" + b"\x07\x00\x00\x00"
        content += b"\x00" * 100
        gdx_file.write_bytes(data=content)

        result: dict = read_gdx(gdx_file)

        assert set(result.keys()) == {
            "filepath",
            "header",
            "symbols",
            "elements",
            "domains",
        }
        assert isinstance(result["header"], dict)
        assert isinstance(result["symbols"], list)
        assert isinstance(result["elements"], list)
        assert isinstance(result["domains"], list)


class TestReadHeader:
    """Tests for read_header_from_bytes() function."""

    def _create_gdx_header(
        self,
        version: int = 7,
        producer: str = "GDX Library C++ V7 Windows",
    ) -> bytes:
        """Create a GDX header with proper format."""
        header: bytes = b"\x00" * 19
        header += b"GAMSGDX"
        header += struct.pack("<I", version)
        header += b"\x00" * 2
        header += producer.encode(encoding="utf-8")
        header += b"\x00" * (200 - len(header))
        return header

    def test_parses_version(self) -> None:
        """Should correctly parse version number."""
        header: bytes = self._create_gdx_header(version=7)
        result: dict = read_header_from_bytes(header)
        assert result["version"] == 7

    def test_parses_version_6(self) -> None:
        """Should correctly parse version 6."""
        header: bytes = self._create_gdx_header(version=6)
        result: dict = read_header_from_bytes(header)
        assert result["version"] == 6

    def test_detects_windows_platform(self) -> None:
        """Should detect Windows platform from producer string."""
        header: bytes = self._create_gdx_header(producer="GDX Library Windows 64bit")
        result: dict = read_header_from_bytes(header)
        assert result["platform"] == "Windows"

    def test_detects_linux_platform(self) -> None:
        """Should detect Linux platform from producer string."""
        header: bytes = self._create_gdx_header(producer="GDX Library Linux x86_64")
        result: dict = read_header_from_bytes(header)
        assert result["platform"] == "Linux"

    def test_detects_macos_platform(self) -> None:
        """Should detect macOS platform from producer string."""
        header: bytes = self._create_gdx_header(producer="GDX Library arm64 macOS")
        result: dict = read_header_from_bytes(header)
        assert result["platform"] == "macOS"

    def test_unknown_platform_returns_unknown(self) -> None:
        """Should return 'unknown' for unrecognized platform."""
        header: bytes = self._create_gdx_header(producer="GDX Library Unknown System")
        result: dict = read_header_from_bytes(header)
        assert result["platform"] == "unknown"

    def test_returns_correct_structure(self) -> None:
        """Should return dict with version, endianness, platform, producer."""
        header: bytes = self._create_gdx_header()
        result: dict = read_header_from_bytes(header)
        assert "version" in result
        assert "endianness" in result
        assert "platform" in result
        assert "producer" in result

    def test_no_magic_returns_zero_version(self) -> None:
        """Should return version 0 if GAMSGDX magic not found."""
        header: bytes = b"\x00" * 100
        result: dict = read_header_from_bytes(header)
        assert result["version"] == 0


class TestReadSymbolTable:
    """Tests for read_symbol_table_from_bytes() function."""

    def _create_symbol_entry(
        self,
        name: str,
        type_flag: int,
        dimension: int,
        records: int,
        description: str = "",
    ) -> bytes:
        """Create a single symbol entry in GDX format."""
        name_bytes: bytes = name.encode("utf-8")
        desc_bytes: bytes = description.encode("utf-8")

        entry: bytes = bytes([len(name_bytes)])
        entry += name_bytes
        entry += bytes([type_flag])

        # 25 bytes metadata
        metadata: bytes = b"\x00" * 7
        metadata += bytes([dimension])
        metadata += b"\x00" * 8
        metadata += struct.pack("<I", records)
        metadata += b"\x00" * 5  # 7+1+8+4+5 = 25 bytes
        entry += metadata

        entry += bytes([len(desc_bytes)])
        entry += desc_bytes
        entry += b"\x00" * 6

        return entry

    def _create_symbol_table(self, symbols: list) -> bytes:
        """Create a complete symbol table with _SYMB_ marker."""
        data: bytes = b"_SYMB_"
        data += struct.pack("<I", len(symbols))

        for name, type_flag, dim, records, desc in symbols:
            data += self._create_symbol_entry(name, type_flag, dim, records, desc)

        return data

    def test_empty_symbol_table(self) -> None:
        """Should return empty list for zero symbols."""
        data: bytes = b"_SYMB_" + struct.pack("<I", 0)
        result: list = read_symbol_table_from_bytes(data)
        assert result == []

    def test_no_symb_marker_returns_empty(self) -> None:
        """Should return empty list if _SYMB_ marker not found."""
        data: bytes = b"\x00" * 100
        result: list = read_symbol_table_from_bytes(data)
        assert result == []

    def test_single_set_symbol(self) -> None:
        """Should parse a single set symbol correctly."""
        data: bytes = self._create_symbol_table([("i", 0x01, 1, 3, "industries")])
        result: list = read_symbol_table_from_bytes(data)

        assert len(result) == 1
        assert result[0]["name"] == "i"
        assert result[0]["type"] == 0
        assert result[0]["type_name"] == "set"
        assert result[0]["dimension"] == 1
        assert result[0]["records"] == 3
        assert result[0]["description"] == "industries"

    def test_single_parameter_symbol(self) -> None:
        """Should parse a parameter symbol correctly."""
        data: bytes = self._create_symbol_table(
            [("price", 0x3F, 1, 10, "market prices")]
        )
        result: list = read_symbol_table_from_bytes(data)

        assert len(result) == 1
        assert result[0]["name"] == "price"
        assert result[0]["type"] == 1
        assert result[0]["type_name"] == "parameter"

    def test_multiple_symbols(self) -> None:
        """Should parse multiple symbols correctly."""
        data: bytes = self._create_symbol_table(
            [
                ("i", 0x01, 1, 3, "industries"),
                ("j", 0x01, 1, 5, "commodities"),
                ("sam", 0x66, 2, 15, "SAM matrix"),
            ]
        )
        result: list = read_symbol_table_from_bytes(data)

        assert len(result) == 3
        assert result[0]["name"] == "i"
        assert result[1]["name"] == "j"
        assert result[2]["name"] == "sam"
        assert result[2]["dimension"] == 2

    def test_symbol_without_description(self) -> None:
        """Should handle symbols with empty description."""
        data: bytes = self._create_symbol_table([("x", 0x01, 1, 2, "")])
        result: list = read_symbol_table_from_bytes(data)

        assert len(result) == 1
        assert result[0]["description"] == ""

    def test_alias_type(self) -> None:
        """Should correctly identify alias type."""
        data: bytes = self._create_symbol_table([("j", 0x20, 1, 3, "alias")])
        result: list = read_symbol_table_from_bytes(data)

        assert len(result) == 1
        assert result[0]["type"] == 4
        assert result[0]["type_name"] == "alias"


@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / "fixtures" / "simple_test.gdx").exists(),
    reason="Test fixture simple_test.gdx not found",
)
class TestIntegrationWithRealGdx:
    """Integration tests using real GDX files generated by GAMS."""

    def test_read_real_gdx_file(self) -> None:
        """Should read a real GDX file generated by GAMS."""
        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "simple_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        # Header checks
        assert result["header"]["version"] == 7
        assert result["header"]["endianness"] == "little"
        assert result["header"]["platform"] == "macOS"

        # Symbols checks
        assert len(result["symbols"]) == 4

        symbols_by_name: dict = {s["name"]: s for s in result["symbols"]}

        # Check set 'i'
        assert "i" in symbols_by_name
        assert symbols_by_name["i"]["type_name"] == "set"
        assert symbols_by_name["i"]["dimension"] == 1
        assert symbols_by_name["i"]["records"] == 3

        # Check parameter 'sam'
        assert "sam" in symbols_by_name
        assert symbols_by_name["sam"]["type_name"] == "parameter"
        assert symbols_by_name["sam"]["dimension"] == 2
        assert symbols_by_name["sam"]["records"] == 9

    def test_read_uel_elements(self) -> None:
        """Should read unique element list from real GDX file."""
        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "simple_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        # Check elements
        assert "elements" in result
        elements: list = result["elements"]
        assert len(elements) == 6
        assert "agr" in elements
        assert "mfg" in elements
        assert "srv" in elements
        assert "food" in elements
        assert "goods" in elements
        assert "services" in elements

    def test_read_domains(self) -> None:
        """Should read domain definitions from real GDX file."""
        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "simple_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        # Check domains
        assert "domains" in result
        domains: list = result["domains"]
        assert "i" in domains
        assert "j" in domains

    def test_get_symbol_helper(self) -> None:
        """Should get symbol by name."""
        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "simple_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        sym = get_symbol(result, "price")
        assert sym is not None
        assert sym["name"] == "price"
        assert sym["type_name"] == "parameter"

        # Non-existent symbol
        assert get_symbol(result, "nonexistent") is None

    def test_get_sets_helper(self) -> None:
        """Should get all sets."""
        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "simple_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        sets: list = get_sets(result)
        assert len(sets) == 1
        assert sets[0]["name"] == "i"

    def test_get_parameters_helper(self) -> None:
        """Should get all parameters."""
        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "simple_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        params: list = get_parameters(result)
        assert len(params) == 2
        param_names: list = [p["name"] for p in params]
        assert "price" in param_names
        assert "sam" in param_names


class TestReadUel:
    """Tests for read_uel_from_bytes() function."""

    def _create_uel(self, elements: list[str]) -> bytes:
        """Create UEL structure."""
        data: bytes = b"_UEL_"
        data += struct.pack("<I", len(elements))
        for elem in elements:
            elem_bytes: bytes = elem.encode("utf-8")
            data += bytes([len(elem_bytes)])
            data += elem_bytes
        return data

    def test_empty_uel(self) -> None:
        """Should return empty list for zero elements."""
        data: bytes = b"_UEL_" + struct.pack("<I", 0)
        result: list = read_uel_from_bytes(data)
        assert result == []

    def test_no_uel_marker(self) -> None:
        """Should return empty list if _UEL_ not found."""
        data: bytes = b"\x00" * 100
        result: list = read_uel_from_bytes(data)
        assert result == []

    def test_single_element(self) -> None:
        """Should read single element."""
        data: bytes = self._create_uel(["agr"])
        result: list = read_uel_from_bytes(data)

        assert len(result) == 1
        assert result[0] == "agr"

    def test_multiple_elements(self) -> None:
        """Should read multiple elements."""
        data: bytes = self._create_uel(["agr", "mfg", "srv"])
        result: list = read_uel_from_bytes(data)

        assert len(result) == 3
        assert result == ["agr", "mfg", "srv"]


class TestReadDomains:
    """Tests for read_domains_from_bytes() function."""

    def _create_domains(self, domains: list[str]) -> bytes:
        """Create DOMS structure."""
        data: bytes = b"_DOMS_"
        data += struct.pack("<I", len(domains))
        for dom in domains:
            dom_bytes: bytes = dom.encode("utf-8")
            data += bytes([len(dom_bytes)])
            data += dom_bytes
        return data

    def test_empty_domains(self) -> None:
        """Should return empty list for zero domains."""
        data: bytes = b"_DOMS_" + struct.pack("<I", 0)
        result: list = read_domains_from_bytes(data)
        assert result == []

    def test_no_doms_marker(self) -> None:
        """Should return empty list if _DOMS_ not found."""
        data: bytes = b"\x00" * 100
        result: list = read_domains_from_bytes(data)
        assert result == []

    def test_single_domain(self) -> None:
        """Should read single domain."""
        data: bytes = self._create_domains(["i"])
        result: list = read_domains_from_bytes(data)

        assert len(result) == 1
        assert result[0] == "i"

    def test_multiple_domains(self) -> None:
        """Should read multiple domains."""
        data: bytes = self._create_domains(["i", "j", "k"])
        result: list = read_domains_from_bytes(data)

        assert len(result) == 3
        assert result == ["i", "j", "k"]


@pytest.mark.skipif(
    not (
        Path(__file__).parent.parent.parent
        / "fixtures"
        / "variables_equations_test.gdx"
    ).exists(),
    reason="Test fixture variables_equations_test.gdx not found",
)
class TestVariablesEquationsFixture:
    """Tests using variables_equations_test.gdx fixture."""

    def test_read_variables(self) -> None:
        """Should correctly identify variable symbols."""
        gdx_path: Path = (
            Path(__file__).parent.parent.parent
            / "fixtures"
            / "variables_equations_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        variables: list = get_variables(result)
        var_names: list = [v["name"] for v in variables]

        assert "X" in var_names
        assert "Y" in var_names
        assert "obj" in var_names

    def test_read_equations(self) -> None:
        """Should correctly identify equation symbols."""
        gdx_path: Path = (
            Path(__file__).parent.parent.parent
            / "fixtures"
            / "variables_equations_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        equations: list = get_equations(result)
        eq_names: list = [e["name"] for e in equations]

        assert "eq_output" in eq_names
        assert "eq_total" in eq_names

    def test_variable_has_type_flag(self) -> None:
        """Symbols should include raw type_flag for debugging."""
        gdx_path: Path = (
            Path(__file__).parent.parent.parent
            / "fixtures"
            / "variables_equations_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        sym = get_symbol(result, "X")
        assert sym is not None
        assert "type_flag" in sym
        assert isinstance(sym["type_flag"], int)


@pytest.mark.skipif(
    not (
        Path(__file__).parent.parent.parent / "fixtures" / "multidim_test.gdx"
    ).exists(),
    reason="Test fixture multidim_test.gdx not found",
)
class TestMultidimFixture:
    """Tests using multidim_test.gdx fixture."""

    def test_read_3d_parameter(self) -> None:
        """Should read 3-dimensional parameter."""
        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "multidim_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        sym = get_symbol(result, "data3d")
        assert sym is not None
        assert sym["type_name"] == "parameter"
        assert sym["dimension"] == 3
        assert sym["records"] == 60  # 4 regions * 5 periods * 3 sectors

    def test_read_multiple_sets(self) -> None:
        """Should read multiple sets with different type flags."""
        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "multidim_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        sets: list = get_sets(result)
        set_names: list = [s["name"] for s in sets]

        assert "r" in set_names
        assert "t" in set_names
        assert "s" in set_names
        assert len(sets) == 3

    def test_elements_from_multiple_sets(self) -> None:
        """Should contain elements from all sets in UEL."""
        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "multidim_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        elements: list = result["elements"]

        # Regions
        assert "north" in elements
        assert "south" in elements

        # Time periods
        assert "t1" in elements
        assert "t5" in elements

        # Sectors
        assert "s1" in elements
        assert "s3" in elements


@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / "fixtures" / "sparse_test.gdx").exists(),
    reason="Test fixture sparse_test.gdx not found",
)
class TestSparseFixture:
    """Tests using sparse_test.gdx fixture."""

    def test_sparse_parameter_records(self) -> None:
        """Should correctly count sparse parameter records."""
        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "sparse_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        sym = get_symbol(result, "sparse_param")
        assert sym is not None
        # Only 2 values defined: sparse_param("a")=1 and sparse_param("e")=5
        assert sym["records"] == 2

    def test_sparse_set_elements(self) -> None:
        """Should read all set elements even if parameter is sparse."""
        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "sparse_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        elements: list = result["elements"]
        # Set has 4 elements: a, c, e, g
        assert "a" in elements
        assert "c" in elements
        assert "e" in elements
        assert "g" in elements


@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / "fixtures" / "variables_equations_test.gdx").exists(),
    reason="Test fixture variables_equations_test.gdx not found",
)
class TestReadParameterValues:
    """Tests for read_parameter_values() function."""

    def test_read_2d_parameter(self) -> None:
        """Should read values from 2D parameter."""
        from equilibria.babel.gdx.reader import read_parameter_values

        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "variables_equations_test.gdx"
        )
        result: dict = read_gdx(gdx_path)
        values: dict = read_parameter_values(result, "sam")

        # sam is 3x3 = 9 values
        assert len(values) == 9

        # Check specific values from the GAMS file
        assert values[("agr", "food")] == 100.0
        assert values[("agr", "goods")] == 50.0
        assert values[("mfg", "goods")] == 200.0

    def test_read_1d_parameter_partial(self) -> None:
        """Should read explicitly stored values from 1D parameter."""
        from equilibria.babel.gdx.reader import read_parameter_values

        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "variables_equations_test.gdx"
        )
        result: dict = read_gdx(gdx_path)
        values: dict = read_parameter_values(result, "price")

        # May not read all values due to GDX compression
        # But should read at least some
        assert len(values) >= 1

    def test_parameter_not_found_raises(self) -> None:
        """Should raise ValueError for non-existent symbol."""
        from equilibria.babel.gdx.reader import read_parameter_values

        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "variables_equations_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        with pytest.raises(ValueError, match="not found"):
            read_parameter_values(result, "nonexistent")

    def test_non_parameter_raises(self) -> None:
        """Should raise ValueError when reading non-parameter symbol."""
        from equilibria.babel.gdx.reader import read_parameter_values

        gdx_path: Path = (
            Path(__file__).parent.parent.parent / "fixtures" / "variables_equations_test.gdx"
        )
        result: dict = read_gdx(gdx_path)

        with pytest.raises(ValueError, match="not a parameter"):
            read_parameter_values(result, "i")  # i is a set, not parameter
