"""Utilities for analysing target columns before Hot Deck fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import PATHS, TARGETS


@dataclass(frozen=True)
class TargetGroups:
    """Names of targets grouped by the number of observed values."""

    constant: list[str]
    binary: list[str]
    multiclass: list[str]


class TargetAnalyzer:
    """Inspect target types, class balance, and unusable targets."""

    def __init__(self) -> None:
        self.groups_: Optional[TargetGroups] = None
        self.summary_: Optional[pd.DataFrame] = None
        self.is_fitted_: bool = False

    def fit(self, targets: pd.DataFrame) -> "TargetAnalyzer":
        """Analyse target columns and store their metadata."""

        self._validate_targets(targets)

        records: list[dict[str, Any]] = []
        constant: list[str] = []
        binary: list[str] = []
        multiclass: list[str] = []

        for column in targets.columns:
            series = targets[column]
            value_counts = series.value_counts(dropna=False)
            unique_count = int(series.nunique(dropna=False))

            if unique_count <= 1:
                target_type = "constant"
                constant.append(column)
            elif unique_count == 2:
                target_type = "binary"
                binary.append(column)
            else:
                target_type = "multiclass"
                multiclass.append(column)

            majority_count = int(value_counts.iloc[0])
            minority_count = int(value_counts.iloc[-1])
            majority_share = majority_count / len(series)

            records.append(
                {
                    "target": column,
                    "target_type": target_type,
                    "unique_values": unique_count,
                    "missing_values": int(series.isna().sum()),
                    "majority_value": value_counts.index[0],
                    "majority_count": majority_count,
                    "majority_share": majority_share,
                    "minority_count": minority_count,
                    "is_highly_imbalanced": bool(
                        unique_count > 1 and majority_share >= 0.90
                    ),
                }
            )

        self.groups_ = TargetGroups(
            constant=constant,
            binary=binary,
            multiclass=multiclass,
        )
        self.summary_ = pd.DataFrame(records)
        self.is_fitted_ = True
        return self

    def get_summary(self) -> pd.DataFrame:
        """Return one row of analysis for every target column."""

        self._check_fitted()
        return self.summary_.copy()

    def get_groups(self) -> TargetGroups:
        """Return constant, binary, and multiclass target names."""

        self._check_fitted()
        return self.groups_

    def get_modeling_targets(self) -> list[str]:
        """Return targets that contain more than one observed value."""

        groups = self.get_groups()
        return groups.binary + groups.multiclass

    def remove_constant_targets(
        self, targets: pd.DataFrame
    ) -> pd.DataFrame:
        """Remove targets that cannot be evaluated or learned."""

        self._check_fitted()
        self._validate_targets(targets)

        expected = set(self.summary_["target"])
        actual = set(targets.columns)
        if expected != actual:
            raise ValueError(
                "Target columns do not match the fitted target analysis."
            )

        return targets[self.get_modeling_targets()].copy()

    def overall_report(self) -> dict[str, Any]:
        """Return a compact project-level target report."""

        groups = self.get_groups()
        summary = self.get_summary()

        return {
            "total_targets": len(summary),
            "constant_targets": len(groups.constant),
            "binary_targets": len(groups.binary),
            "multiclass_targets": len(groups.multiclass),
            "highly_imbalanced_targets": int(
                summary["is_highly_imbalanced"].sum()
            ),
            "constant_target_names": groups.constant,
        }

    @staticmethod
    def _validate_targets(targets: pd.DataFrame) -> None:
        if not isinstance(targets, pd.DataFrame):
            raise TypeError("targets must be a pandas DataFrame.")
        if targets.empty:
            raise ValueError("targets is empty.")
        if targets.columns.duplicated().any():
            duplicates = targets.columns[
                targets.columns.duplicated()
            ].tolist()
            raise ValueError(
                f"targets contains duplicate columns: {duplicates}"
            )
        non_numeric = targets.select_dtypes(
            exclude=np.number
        ).columns.tolist()
        if non_numeric:
            raise TypeError(
                f"targets contains non-numeric columns: {non_numeric}"
            )

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before requesting results.")


if __name__ == "__main__":
    from src.preprocessing import load_datasets

    data = load_datasets(
        PATHS.train_predictors,
        PATHS.train_targets,
        PATHS.test_predictors,
    )

    analyzer = TargetAnalyzer().fit(data.train_targets)

    print("Overall target report:")
    print(analyzer.overall_report())

    modeling_targets = (
        analyzer.remove_constant_targets(data.train_targets)
        if TARGETS.remove_constant_targets
        else data.train_targets.copy()
    )
    print(f"Modeling target count: {modeling_targets.shape[1]}")

    print("\nFirst 10 target summaries:")
    print(analyzer.get_summary().head(10).to_string(index=False))
