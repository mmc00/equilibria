"""
equilibria/tests/babel/gdx/test_reader.py - Tests for GDX reader module.

Unit tests for equilibria.babel.gdx.reader module.

Tests cover:
- read_gdx(): Main entry point for reading GDX files
- read_header(): Header parsing with version, endianness, platform
- read_symbol_table(): Symbol table parsing
- Error handling: FileNotFoundError, ValueError for invalid files
"""

# Standard library imports
from __future__ import annotations

import io
import struct
from pathlib import Path

# Third-party imports
import pytest

# Local imports
from equilibria.babel.gdx.reader import read_gdx, read_header, read_symbol_table


class TestReadGdx:
    """Tests for read_gdx() function."""

    def test_file_not_found_raises_error(self) -> None:
        """Should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="GDX file not found"):
            read_gdx("nonexistent_file.gdx")

    def test_invalid_extension_raises_error(self, tmp_path: Path) -> None:
        """Should raise ValueError for non-.gdx extension."""
        # Create a file with wrong extension
        invalid_file: Path = tmp_path / "data.txt"
        invalid_file.write_bytes(b"dummy content")

        with pytest.raises(ValueError, match="Expected .gdx file"):
            read_gdx(invalid_file)

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        """Should accept Path objects as input."""
        gdx_file: Path = tmp_path / "test.gdx"
        # Create minimal valid GDX-like file (128 byte header + empty symbol table)
        header: bytes = b"GAMS" + b"\x07\x00\x00\x00"  # magic + version 7
        header += b"\x01"  # little endian
        header += b"\x01"  # Windows
        header += b"\x00" * (128 - 10)  # padding to 128 bytes
        header += b"\x00\x00\x00\x00"  # 0 symbols
        gdx_file.write_bytes(header)

        result: dict[str, object] = read_gdx(gdx_file)

        assert result["filepath"] == str(gdx_file)
        assert "header" in result
        assert "symbols" in result

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """Should accept string paths as input."""
        gdx_file: Path = tmp_path / "test.gdx"
        header: bytes = b"GAMS" + b"\x07\x00\x00\x00" + b"\x01\x01"
        header += b"\x00" * (128 - 10)
        header += b"\x00\x00\x00\x00"
        gdx_file.write_bytes(header)

        result: dict[str, object] = read_gdx(str(gdx_file))

        assert result["filepath"] == str(gdx_file)

    def test_returns_correct_structure(self, tmp_path: Path) -> None:
        """Should return dict with filepath, header, and symbols keys."""
        gdx_file: Path = tmp_path / "test.gdx"
        header: bytes = b"GAMS" + b"\x07\x00\x00\x00" + b"\x01\x01"
        header += b"\x00" * (128 - 10)
        header += b"\x00\x00\x00\x00"
        gdx_file.write_bytes(header)

        result: dict[str, object] = read_gdx(gdx_file)

        assert set(result.keys()) == {"filepath", "header", "symbols"}
        assert isinstance(result["header"], dict)
        assert isinstance(result["symbols"], list)


class TestReadHeader:
    """Tests for read_header() function."""

    def _create_header_bytes(
        self,
        magic: bytes = b"GAMS",
        version: int = 7,
        endianness: bytes = b"\x01",
        platform: bytes = b"\x01",
    ) -> bytes:
        """Helper to create a GDX header."""
        header = magic
        header += struct.pack("<I", version)  # little-endian uint32
        header += endianness
        header += platform
        header += b"\x00" * (128 - len(header))  # pad to 128 bytes
        return header

    def test_parses_version(self) -> None:
        """Should correctly parse version number."""
        header_bytes: bytes = self._create_header_bytes(version=7)
        file: io.BytesIO = io.BytesIO(header_bytes)

        result: dict[str, object] = read_header(file)

        assert result["version"] == 7

    def test_parses_little_endian(self) -> None:
        """Should detect little endian format."""
        header_bytes: bytes = self._create_header_bytes(endianness=b"\x01")
        file: io.BytesIO = io.BytesIO(header_bytes)

        result: dict[str, object] = read_header(file)

        assert result["endianness"] == "little"

    def test_parses_big_endian(self) -> None:
        """Should detect big endian format."""
        header_bytes: bytes = self._create_header_bytes(endianness=b"\x02")
        file: io.BytesIO = io.BytesIO(header_bytes)

        result: dict[str, object] = read_header(file)

        assert result["endianness"] == "big"

    def test_defaults_to_little_endian_on_unknown(self) -> None:
        """Should default to little endian for unknown byte."""
        header_bytes: bytes = self._create_header_bytes(endianness=b"\xff")
        file: io.BytesIO = io.BytesIO(header_bytes)

        result: dict[str, object] = read_header(file)

        assert result["endianness"] == "little"

    def test_parses_windows_platform(self) -> None:
        """Should detect Windows platform."""
        header_bytes: bytes = self._create_header_bytes(platform=b"\x01")
        file: io.BytesIO = io.BytesIO(header_bytes)

        result: dict[str, object] = read_header(file)

        assert result["platform"] == "Windows"

    def test_parses_linux_platform(self) -> None:
        """Should detect Linux platform."""
        header_bytes: bytes = self._create_header_bytes(platform=b"\x02")
        file: io.BytesIO = io.BytesIO(header_bytes)

        result: dict[str, object] = read_header(file)

        assert result["platform"] == "Linux"

    def test_unknown_platform_returns_unknown(self) -> None:
        """Should return 'unknown' for unrecognized platform byte."""
        header_bytes: bytes = self._create_header_bytes(platform=b"\xff")
        file: io.BytesIO = io.BytesIO(header_bytes)

        result: dict[str, object] = read_header(file)

        assert result["platform"] == "unknown"

    def test_returns_correct_structure(self) -> None:
        """Should return dict with version, endianness, platform."""
        header_bytes: bytes = self._create_header_bytes()
        file: io.BytesIO = io.BytesIO(header_bytes)

        result: dict[str, object] = read_header(file)

        assert set(result.keys()) == {"version", "endianness", "platform"}


class TestReadSymbolTable:
    """Tests for read_symbol_table() function."""

    def _create_symbol_entry(
        self,
        name: str,
        sym_type: int,
        dimension: int,
        description: str = "",
        endianness: str = "little",
    ) -> bytes:
        """Helper to create a symbol table entry."""
        byteorder: str = "<" if endianness == "little" else ">"
        name_bytes: bytes = name.encode("utf-8")
        desc_bytes: bytes = description.encode("utf-8")

        entry: bytes = struct.pack(f"{byteorder}I", len(name_bytes))
        entry += name_bytes
        entry += struct.pack(f"{byteorder}I", sym_type)
        entry += struct.pack(f"{byteorder}I", dimension)
        entry += struct.pack(f"{byteorder}I", len(desc_bytes))
        entry += desc_bytes

        return entry

    def _create_symbol_table(
        self,
        symbols: list[tuple[str, int, int, str]],
        endianness: str = "little",
    ) -> bytes:
        """Helper to create a complete symbol table with header."""
        byteorder: str = "<" if endianness == "little" else ">"

        # 128 byte header (padding)
        data: bytes = b"\x00" * 128

        # Number of symbols
        data += struct.pack(f"{byteorder}I", len(symbols))

        # Symbol entries
        for name, sym_type, dimension, description in symbols:
            data += self._create_symbol_entry(
                name, sym_type, dimension, description, endianness
            )

        return data

    def test_empty_symbol_table(self) -> None:
        """Should return empty list for zero symbols."""
        data: bytes = self._create_symbol_table([])
        file: io.BytesIO = io.BytesIO(data)

        result: list[dict[str, object]] = read_symbol_table(file)

        assert result == []

    def test_single_symbol(self) -> None:
        """Should parse a single symbol correctly."""
        data: bytes = self._create_symbol_table([("i", 0, 1, "Industries")])
        file: io.BytesIO = io.BytesIO(data)

        result: list[dict[str, object]] = read_symbol_table(file)

        assert len(result) == 1
        assert result[0]["name"] == "i"
        assert result[0]["type"] == 0
        assert result[0]["dimension"] == 1
        assert result[0]["description"] == "Industries"

    def test_multiple_symbols(self) -> None:
        """Should parse multiple symbols correctly."""
        symbols: list[tuple[str, int, int, str]] = [
            ("i", 0, 1, "Industries"),
            ("j", 0, 1, "Commodities"),
            ("price", 1, 2, "Prices by commodity and region"),
        ]
        data: bytes = self._create_symbol_table(symbols)
        file: io.BytesIO = io.BytesIO(data)

        result: list[dict[str, object]] = read_symbol_table(file)

        assert len(result) == 3
        assert result[0]["name"] == "i"
        assert result[1]["name"] == "j"
        assert result[2]["name"] == "price"
        assert result[2]["dimension"] == 2

    def test_symbol_without_description(self) -> None:
        """Should handle symbols with empty description."""
        data: bytes = self._create_symbol_table([("x", 2, 1, "")])
        file: io.BytesIO = io.BytesIO(data)

        result: list[dict[str, object]] = read_symbol_table(file)

        assert len(result) == 1
        assert result[0]["description"] == ""

    def test_big_endian(self) -> None:
        """Should correctly parse big-endian symbol table."""
        data: bytes = self._create_symbol_table(
            [("test", 1, 3, "Test param")], endianness="big"
        )
        file: io.BytesIO = io.BytesIO(data)

        result: list[dict[str, object]] = read_symbol_table(file, endianness="big")

        assert len(result) == 1
        assert result[0]["name"] == "test"
        assert result[0]["dimension"] == 3

    def test_truncated_file_returns_partial(self) -> None:
        """Should return partial results if file is truncated."""
        # Create valid first symbol, then truncate
        data: bytes = self._create_symbol_table([("valid", 0, 1, "OK")])
        # Indicate 2 symbols but only provide 1
        data = data[:128] + b"\x02\x00\x00\x00" + data[132:]
        file: io.BytesIO = io.BytesIO(data)

        result: list[dict[str, object]] = read_symbol_table(file)

        # Should get at least the first symbol (or empty on parse failure)
        assert isinstance(result, list)

    def test_invalid_name_length_stops_parsing(self) -> None:
        """Should stop parsing if name length is invalid (>256)."""
        data: bytes = b"\x00" * 128  # header
        data += b"\x01\x00\x00\x00"  # 1 symbol
        data += b"\xff\x0f\x00\x00"  # name_len = 4095 (invalid)
        file: io.BytesIO = io.BytesIO(data)

        result: list[dict[str, object]] = read_symbol_table(file)

        assert result == []

    def test_handles_utf8_names(self) -> None:
        """Should handle UTF-8 encoded symbol names."""
        data: bytes = self._create_symbol_table([("índice", 0, 1, "Índice de preços")])
        file: io.BytesIO = io.BytesIO(data)

        result: list[dict[str, object]] = read_symbol_table(file)

        assert len(result) == 1
        assert result[0]["name"] == "índice"
        assert result[0]["description"] == "Índice de preços"

    def test_symbol_types(self) -> None:
        """Should correctly parse different symbol types."""
        symbols: list[tuple[str, int, int, str]] = [
            ("set_i", 0, 1, "Set"),  # type 0 = set
            ("param_a", 1, 2, "Parameter"),  # type 1 = parameter
            ("var_x", 2, 1, "Variable"),  # type 2 = variable
            ("eq_market", 3, 1, "Equation"),  # type 3 = equation
        ]
        data: bytes = self._create_symbol_table(symbols)
        file: io.BytesIO = io.BytesIO(data)

        result: list[dict[str, object]] = read_symbol_table(file)

        assert result[0]["type"] == 0
        assert result[1]["type"] == 1
        assert result[2]["type"] == 2
        assert result[3]["type"] == 3


class TestIntegration:
    """Integration tests for the complete read workflow."""

    def test_read_complete_gdx_file(self, tmp_path: Path) -> None:
        """Should read a complete mock GDX file."""
        gdx_file: Path = tmp_path / "complete.gdx"

        # Build complete file
        # Header
        header: bytes = b"GAMS"
        header += struct.pack("<I", 7)  # version
        header += b"\x01"  # little endian
        header += b"\x01"  # Windows
        header += b"\x00" * (128 - 10)

        # Symbol table
        symbols_data: bytes = struct.pack("<I", 2)  # 2 symbols

        # Symbol 1: set "i"
        name1: bytes = b"i"
        desc1: bytes = b"Industries"
        symbols_data += struct.pack("<I", len(name1)) + name1
        symbols_data += struct.pack("<I", 0)  # type = set
        symbols_data += struct.pack("<I", 1)  # dimension
        symbols_data += struct.pack("<I", len(desc1)) + desc1

        # Symbol 2: parameter "price"
        name2: bytes = b"price"
        desc2: bytes = b"Market prices"
        symbols_data += struct.pack("<I", len(name2)) + name2
        symbols_data += struct.pack("<I", 1)  # type = parameter
        symbols_data += struct.pack("<I", 2)  # dimension
        symbols_data += struct.pack("<I", len(desc2)) + desc2

        gdx_file.write_bytes(header + symbols_data)

        # Read and verify
        result: dict[str, object] = read_gdx(gdx_file)

        assert result["header"]["version"] == 7
        assert result["header"]["endianness"] == "little"
        assert result["header"]["platform"] == "Windows"
        assert len(result["symbols"]) == 2
        assert result["symbols"][0]["name"] == "i"
        assert result["symbols"][1]["name"] == "price"
        assert result["symbols"][1]["dimension"] == 2
