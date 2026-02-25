"""Data models for SAM workflow execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from equilibria.sam_tools.aggregation import aggregate_dataframe, load_mapping
from equilibria.sam_tools.balancing import RASBalanceResult, RASBalancer
from equilibria.sam_tools.enums import SAMFormat


class Sam(BaseModel):
    """Contenedor base para una SAM, asegura matriz cuadrada con cuentas consistentes."""

    dataframe: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _normalize_accounts(keys: Iterable[Any]) -> list[tuple[str, ...]]:
        normalized: list[tuple[str, ...]] = []
        for entry in keys:
            if isinstance(entry, tuple):
                parts = tuple(str(part).strip() for part in entry)
            else:
                parts = (str(entry).strip(),)
            normalized.append(parts)
        return normalized

    @staticmethod
    def _ensure_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        if df.ndim != 2:
            raise ValueError("El SAM debe ser una matriz bidimensional")
        rows, cols = df.shape
        if rows != cols:
            raise ValueError("El SAM debe ser cuadrado: filas y columnas iguales")
        df.index = pd.MultiIndex.from_tuples(Sam._normalize_accounts(df.index))
        df.columns = pd.MultiIndex.from_tuples(Sam._normalize_accounts(df.columns))
        if set(df.index.tolist()) != set(df.columns.tolist()):
            raise ValueError("Las mismas cuentas deben aparecer en filas y columnas")
        return df.astype(float)

    @model_validator(mode="after")
    def _validate(self) -> Sam:
        normalized = self._ensure_dataframe(self.dataframe)
        object.__setattr__(self, "dataframe", normalized)
        return self

    @staticmethod
    def _build_square_dataframe(
        matrix: np.ndarray,
        row_keys: Sequence[tuple[str, str]],
        col_keys: Sequence[tuple[str, str]],
    ) -> pd.DataFrame:
        normalized_rows = [tuple(str(part).strip() for part in key) for key in row_keys]
        normalized_cols = [tuple(str(part).strip() for part in key) for key in col_keys]
        combined: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for key in normalized_rows + normalized_cols:
            if key not in seen:
                combined.append(key)
                seen.add(key)
        size = len(combined)
        square = np.zeros((size, size), dtype=float)
        index_map = {key: idx for idx, key in enumerate(combined)}
        for i, row_key in enumerate(normalized_rows):
            for j, col_key in enumerate(normalized_cols):
                square[index_map[row_key], index_map[col_key]] = matrix[i, j]
        multi_index = pd.MultiIndex.from_tuples(combined)
        return pd.DataFrame(square, index=multi_index, columns=multi_index)

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        row_keys: Sequence[tuple[str, str]],
        col_keys: Sequence[tuple[str, str]],
    ) -> Sam:
        if matrix.shape != (len(row_keys), len(col_keys)):
            raise ValueError("La matriz no coincide con los índices provistos")
        df = cls._build_square_dataframe(matrix, row_keys, col_keys)
        return cls(dataframe=df)

    @property
    def row_keys(self) -> list[tuple[str, ...]]:
        return [tuple(key) for key in self.dataframe.index.tolist()]

    @property
    def col_keys(self) -> list[tuple[str, ...]]:
        return [tuple(key) for key in self.dataframe.columns.tolist()]

    @property
    def matrix(self) -> np.ndarray:
        matrix = self.dataframe.to_numpy(dtype=float, copy=False)
        matrix.setflags(write=True)
        return matrix

    def update_matrix(self, matrix: np.ndarray) -> None:
        new_df = pd.DataFrame(
            matrix,
            index=self.dataframe.index,
            columns=self.dataframe.columns,
        )
        self.replace_dataframe(new_df)

    def to_dataframe(self) -> pd.DataFrame:
        return self.dataframe.copy()

    def replace_dataframe(self, frame: pd.DataFrame) -> None:
        normalized = self._ensure_dataframe(frame)
        object.__setattr__(self, "dataframe", normalized)

    def aggregate(self, mapping_path: Path) -> Sam:
        mapping, ordered = load_mapping(mapping_path)
        df = self.to_dataframe()
        aggregated = aggregate_dataframe(df, mapping, ordered)
        category = self.row_keys[0][0] if self.row_keys else "RAW"
        multi_index = pd.MultiIndex.from_tuples([(category, label) for label in aggregated.index])
        new_df = pd.DataFrame(aggregated.to_numpy(dtype=float), index=multi_index, columns=multi_index)
        self.replace_dataframe(new_df)
        return self

    def balance_ras(
        self,
        *,
        ras_type: str = "arithmetic",
        tolerance: float = 1e-9,
        max_iterations: int = 200,
    ) -> RASBalanceResult:
        result = RASBalancer().balance_dataframe(
            self.to_dataframe(),
            ras_type=ras_type,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        self.replace_dataframe(result.matrix)
        return result


class SamTransform(BaseModel):
    """Metadata holder that embedds a square ``Sam`` and tracks source info."""

    sam: Sam
    source_path: Path
    source_format: str
    raw_df: pd.DataFrame | None = None
    data_start_row: int | None = None
    data_start_col: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def row_keys(self) -> list[tuple[str, ...]]:
        return [tuple(key) for key in self.sam.row_keys]

    @property
    def col_keys(self) -> list[tuple[str, ...]]:
        return [tuple(key) for key in self.sam.col_keys]

    @property
    def matrix(self) -> np.ndarray:
        return self.sam.matrix

    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        self.sam.update_matrix(value)

    def replace_sam(self, new_sam: Sam) -> None:
        object.__setattr__(self, "sam", new_sam)




class SAMWorkflowConfig(BaseModel):
    """Resolved workflow config from YAML."""

    name: str
    country: str | None
    input_path: Path
    input_format: SAMFormat
    output_path: Path
    output_format: SAMFormat
    input_options: dict[str, Any] = Field(default_factory=dict)
    transforms: list[dict[str, Any]] = Field(default_factory=list)
    report_path: Path | None
    output_symbol: str

    model_config = ConfigDict(arbitrary_types_allowed=True)
