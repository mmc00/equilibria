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

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        row_keys: Sequence[tuple[str, str]],
        col_keys: Sequence[tuple[str, str]],
        ) -> Sam:
        if matrix.shape != (len(row_keys), len(col_keys)):
            raise ValueError("La matriz no coincide con los índices provistos")
        df = pd.DataFrame(
            matrix,
            index=pd.MultiIndex.from_tuples(row_keys),
            columns=pd.MultiIndex.from_tuples(col_keys),
        )
        return cls(dataframe=df)

    @property
    def row_keys(self) -> list[tuple[str, ...]]:
        return [tuple(key) for key in self.dataframe.index.tolist()]

    @property
    def col_keys(self) -> list[tuple[str, ...]]:
        return [tuple(key) for key in self.dataframe.columns.tolist()]

    @property
    def matrix(self) -> np.ndarray:
        return self.dataframe.to_numpy(dtype=float)

    def update_matrix(self, matrix: np.ndarray) -> None:
        if matrix.shape != self.matrix.shape:
            raise ValueError("La nueva matriz debe conservar la forma original")
        new_df = pd.DataFrame(matrix, index=self.dataframe.index, columns=self.dataframe.columns)
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
    """In-memory SAM representation used during transformations."""

    sam: Sam
    row_keys: list[tuple[str, str]]
    col_keys: list[tuple[str, str]]
    source_path: Path
    source_format: str
    raw_df: pd.DataFrame | None = None
    data_start_row: int | None = None
    data_start_col: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_matrix_shape(self) -> SamTransform:
        rows, cols = self.sam.matrix.shape
        if rows != len(self.row_keys):
            raise ValueError(
                f"matrix rows ({rows}) do not match row_keys ({len(self.row_keys)})"
            )
        if cols != len(self.col_keys):
            raise ValueError(
                f"matrix cols ({cols}) do not match col_keys ({len(self.col_keys)})"
            )
        if tuple(self.row_keys) != tuple(self.sam.row_keys):
            raise ValueError("Los row_keys deben coincidir con la SAM base")
        if tuple(self.col_keys) != tuple(self.sam.col_keys):
            raise ValueError("Los col_keys deben coincidir con la SAM base")
        return self

    @property
    def matrix(self) -> np.ndarray:
        return self.sam.matrix

    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        self.sam.update_matrix(value)

    def refresh_keys(self) -> None:
        self.row_keys = [tuple(key) for key in self.sam.row_keys]
        self.col_keys = [tuple(key) for key in self.sam.col_keys]


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
