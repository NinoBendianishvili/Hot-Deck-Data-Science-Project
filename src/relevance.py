"""Raw mutual-information weighting for decoded bitmap groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from src.config import DIAGNOSTICS, PATHS, RELEVANCE


@dataclass(frozen=True)
class RelevanceReport:
    """Summary of learned per-target group relevance."""

    target_count: int
    group_count: int
    top_groups: dict[str, list[str]]
    selected_group_counts: dict[str, int]


class TargetRelevanceWeighter:
    """Learn one raw mutual-information weight per target and bitmap group."""

    def __init__(
        self,
        weight_floor: float = RELEVANCE.weight_floor,
    ) -> None:
        if weight_floor < 0:
            raise ValueError("weight_floor must be non-negative.")

        self.weight_floor = float(weight_floor)
        self.group_columns_: Optional[list[str]] = None
        self.target_columns_: Optional[list[str]] = None
        self.weights_: Optional[pd.DataFrame] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        donor_predictors: pd.DataFrame,
        donor_targets: pd.DataFrame,
    ) -> "TargetRelevanceWeighter":
        """Fit target-by-group raw mutual information using donor rows only."""

        self._validate_categorical_frame(
            donor_predictors,
            "donor_predictors",
        )
        self._validate_targets(donor_targets)

        if len(donor_predictors) != len(donor_targets):
            raise ValueError(
                "donor_predictors and donor_targets must have the same "
                "number of rows."
            )
        if not donor_predictors.index.equals(donor_targets.index):
            raise ValueError(
                "donor_predictors and donor_targets must have identical "
                "aligned indices."
            )

        self.group_columns_ = donor_predictors.columns.tolist()
        self.target_columns_ = donor_targets.columns.tolist()
        predictor_arrays = {
            group: donor_predictors[group].to_numpy()
            for group in self.group_columns_
        }
        normalized_rows: dict[str, np.ndarray] = {}

        for target in self.target_columns_:
            target_values = donor_targets[target].to_numpy()
            scores = np.array(
                [
                    mutual_info_score(
                        predictor_arrays[group],
                        target_values,
                    )
                    for group in self.group_columns_
                ],
                dtype=np.float64,
            )
            normalized_rows[target] = self._normalize(scores)

        self.weights_ = pd.DataFrame(
            normalized_rows,
            index=self.group_columns_,
        ).T
        self.is_fitted_ = True
        return self

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize positive raw mutual-information scores to sum to one."""

        values = np.asarray(scores, dtype=np.float64).copy()
        values[~np.isfinite(values)] = 0.0
        values = np.maximum(values, 0.0)
        positive = values > 0

        if not positive.any():
            return np.full(
                len(values),
                1.0 / len(values),
                dtype=np.float64,
            )

        normalized = np.zeros(len(values), dtype=np.float64)
        selected = values[positive]
        if self.weight_floor > 0:
            selected = np.maximum(selected, self.weight_floor)
        normalized[positive] = selected / selected.sum()
        return normalized

    def get_weights(self, target: str) -> np.ndarray:
        """Return normalized bitmap-group weights for one target."""

        self._check_fitted()
        if target not in self.weights_.index:
            raise KeyError(f"Unknown target: {target}")
        return self.weights_.loc[target].to_numpy(dtype=np.float64)

    def get_weight_table(self) -> pd.DataFrame:
        """Return normalized target-by-group weights."""

        self._check_fitted()
        return self.weights_.copy()

    def get_report(
        self,
        top_n: int = DIAGNOSTICS.top_relevance_groups,
    ) -> RelevanceReport:
        """Return the most relevant bitmap groups for every target."""

        self._check_fitted()
        top_groups = {
            target: self.weights_.loc[target]
            .loc[lambda row: row > 0]
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
            for target in self.target_columns_
        }
        selected_group_counts = {
            target: int((self.weights_.loc[target] > 0).sum())
            for target in self.target_columns_
        }
        return RelevanceReport(
            target_count=len(self.target_columns_ or []),
            group_count=len(self.group_columns_ or []),
            top_groups=top_groups,
            selected_group_counts=selected_group_counts,
        )

    def get_parameters(self) -> dict[str, object]:
        """Return the fitted raw-MI relevance configuration."""

        self._check_fitted()
        return {
            "method": "raw_mutual_information",
            "weight_floor": self.weight_floor,
            "target_count": len(self.target_columns_ or []),
            "group_count": len(self.group_columns_ or []),
        }

    @staticmethod
    def _validate_categorical_frame(
        frame: pd.DataFrame,
        frame_name: str,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{frame_name} must be a pandas DataFrame.")
        if frame.empty:
            raise ValueError(f"{frame_name} is empty.")
        if frame.columns.duplicated().any():
            raise ValueError(
                f"{frame_name} contains duplicate column names."
            )
        if frame.isna().any().any():
            raise ValueError(f"{frame_name} contains missing values.")
        non_numeric = frame.select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric:
            raise TypeError(
                f"{frame_name} contains non-numeric groups: {non_numeric}"
            )

    @staticmethod
    def _validate_targets(targets: pd.DataFrame) -> None:
        if not isinstance(targets, pd.DataFrame):
            raise TypeError("donor_targets must be a pandas DataFrame.")
        if targets.empty:
            raise ValueError("donor_targets is empty.")
        if targets.columns.duplicated().any():
            raise ValueError(
                "donor_targets contains duplicate column names."
            )
        if targets.isna().any().any():
            raise ValueError("donor_targets contains missing values.")

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before requesting weights.")


if __name__ == "__main__":
    from src.feature_engineering import BitmapFeatureTransformer
    from src.preprocessing import DataPreprocessor, load_datasets
    from src.target_analysis import TargetAnalyzer

    data = load_datasets(
        PATHS.train_predictors,
        PATHS.train_targets,
        PATHS.test_predictors,
    )
    clean_train = DataPreprocessor().fit_transform(data.train_predictors)
    decoded_train = BitmapFeatureTransformer().fit_transform(clean_train)
    analyzer = TargetAnalyzer().fit(data.train_targets)
    modeling_targets = analyzer.remove_constant_targets(data.train_targets)
    weighter = TargetRelevanceWeighter(
        weight_floor=RELEVANCE.weight_floor,
    ).fit(decoded_train, modeling_targets)
    report = weighter.get_report(
        top_n=DIAGNOSTICS.top_relevance_groups,
    )

    print("Relevance parameters:")
    print(weighter.get_parameters())
    print(
        f"Weighted {report.target_count} targets over "
        f"{report.group_count} decoded bitmap groups."
    )
    for target in list(report.top_groups)[:3]:
        print(
            f"  {target}: {report.top_groups[target]} "
            f"({report.selected_group_counts[target]} groups retained)"
        )
