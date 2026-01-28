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
    read_gdx,
    read_header_from_bytes,
    read_symbol_table_from_bytes,
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
        """Should return dict with filepath, header, and symbols keys."""
        gdx_file: Path = tmp_path / "test.gdx"
        content: bytes = b"\x00" * 19 + b"GAMSGDX" + b"\x07\x00\x00\x00"
        content += b"\x00" * 100
        gdx_file.write_bytes(data=content)

        result: dict = read_gdx(gdx_file)

        assert set(result.keys()) == {"filepath", "header", "symbols"}
        assert isinstance(result["header"], dict)
        assert isinstance(result["symbols"], list)


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
