"""Data loading and conservative preprocessing utilities.

The predictor columns are pre-encoded one-hot bitmap groups.  Preprocessing
therefore preserves the full column structure: it does not remove constants,
impute each bit independently, or apply per-column frequency filtering.
Missing/non-finite cells are replaced with zero, which downstream bitmap
handling interprets as an unknown category when a whole group is all zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from src.config import PATHS, PREPROCESSING


@dataclass(frozen=True)
class DatasetBundle:
    """Container for the three datasets used by the project."""

    train_predictors: pd.DataFrame
    train_targets: pd.DataFrame
    test_predictors: pd.DataFrame


class DataPreprocessor:
    """Validate binary bitmap matrices without damaging group structure."""

    def __init__(
        self,
        imputation_strategy: str = PREPROCESSING.imputation_strategy,
    ) -> None:
        if imputation_strategy not in {"unknown_category", "zero"}:
            raise ValueError(
                "Bitmap predictors support only the 'unknown_category' "
                "(or equivalent 'zero') strategy."
            )
        self.imputation_strategy = imputation_strategy
        self.input_columns_: Optional[list[str]] = None
        self.output_columns_: Optional[list[str]] = None
        self.missing_values_seen_: int = 0
        self.is_fitted_: bool = False

    def fit(self, train_predictors: pd.DataFrame) -> "DataPreprocessor":
        """Record the training schema and validate binary values."""

        self._validate_predictor_frame(train_predictors, "train_predictors")
        cleaned = self._replace_infinite_values(train_predictors)
        self._validate_binary_or_missing(cleaned, "train_predictors")

        self.input_columns_ = cleaned.columns.tolist()
        self.output_columns_ = self.input_columns_.copy()
        self.missing_values_seen_ = int(cleaned.isna().sum().sum())
        self.is_fitted_ = True
        return self

    def transform(self, predictors: pd.DataFrame) -> pd.DataFrame:
        """Preserve all bits and map missing/non-finite cells to zero."""

        if not self.is_fitted_:
            raise RuntimeError("Call fit() before transform().")

        self._validate_predictor_frame(predictors, "predictors")
        self._validate_columns(predictors)

        cleaned = self._replace_infinite_values(predictors)
        self._validate_binary_or_missing(cleaned, "predictors")

        result = cleaned.fillna(0).astype(np.uint8)
        return result[self.output_columns_].copy()

    def fit_transform(
        self,
        train_predictors: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fit on and transform training predictors."""

        return self.fit(train_predictors).transform(train_predictors)

    def get_report(self) -> dict[str, Any]:
        """Return fitted preprocessing decisions."""

        if not self.is_fitted_:
            raise RuntimeError("Call fit() before requesting a report.")

        return {
            "input_feature_count": len(self.input_columns_ or []),
            "output_feature_count": len(self.output_columns_ or []),
            "removed_constant_columns": [],
            "imputation_strategy": self.imputation_strategy,
            "training_missing_values_mapped_to_unknown": (
                self.missing_values_seen_
            ),
            "preserves_bitmap_columns": True,
        }

    @staticmethod
    def _replace_infinite_values(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.replace([np.inf, -np.inf], np.nan).copy()

    @staticmethod
    def _validate_binary_or_missing(
        frame: pd.DataFrame,
        frame_name: str,
    ) -> None:
        values = frame.to_numpy()
        non_missing = values[~pd.isna(values)]
        if not np.isin(non_missing, [0, 1]).all():
            raise ValueError(
                f"{frame_name} must contain only binary values 0 and 1 "
                "(plus optional missing values)."
            )

    @staticmethod
    def _validate_predictor_frame(
        frame: pd.DataFrame,
        frame_name: str,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{frame_name} must be a pandas DataFrame.")
        if frame.empty:
            raise ValueError(f"{frame_name} is empty.")
        if frame.columns.duplicated().any():
            duplicates = frame.columns[frame.columns.duplicated()].tolist()
            raise ValueError(
                f"{frame_name} contains duplicate columns: {duplicates}"
            )
        non_numeric = frame.select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric:
            raise TypeError(
                f"{frame_name} contains non-numeric columns: {non_numeric}"
            )

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        expected = self.input_columns_ or []
        actual = frame.columns.tolist()
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise ValueError(
                "Predictor columns do not match the fitted training data in "
                f"name and order. Missing: {missing}; extra: {extra}"
            )


def load_datasets(
    train_predictors_path: Union[str, Path],
    train_targets_path: Union[str, Path],
    test_predictors_path: Union[str, Path],
) -> DatasetBundle:
    """Load and validate the three project CSV files."""

    paths = {
        "train_predictors": Path(train_predictors_path),
        "train_targets": Path(train_targets_path),
        "test_predictors": Path(test_predictors_path),
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"{name} must be a CSV file: {path}")

    bundle = DatasetBundle(
        train_predictors=pd.read_csv(paths["train_predictors"]),
        train_targets=pd.read_csv(paths["train_targets"]),
        test_predictors=pd.read_csv(paths["test_predictors"]),
    )
    _validate_dataset_bundle(bundle)
    return bundle


def _validate_dataset_bundle(bundle: DatasetBundle) -> None:
    if len(bundle.train_predictors) != len(bundle.train_targets):
        raise ValueError(
            "Training predictors and targets must have the same row count. "
            f"Received {len(bundle.train_predictors)} predictor rows and "
            f"{len(bundle.train_targets)} target rows."
        )

    if bundle.train_predictors.columns.tolist() != (
        bundle.test_predictors.columns.tolist()
    ):
        raise ValueError(
            "Training and test predictor columns must match in name and order."
        )

    DataPreprocessor._validate_predictor_frame(
        bundle.train_predictors,
        "train_predictors",
    )
    DataPreprocessor._validate_predictor_frame(
        bundle.test_predictors,
        "test_predictors",
    )
    DataPreprocessor._validate_predictor_frame(
        bundle.train_targets,
        "train_targets",
    )


def summarize_datasets(bundle: DatasetBundle) -> pd.DataFrame:
    """Create a compact quality report for all three datasets."""

    records: list[dict[str, Any]] = []
    for name, frame in {
        "train_predictors": bundle.train_predictors,
        "train_targets": bundle.train_targets,
        "test_predictors": bundle.test_predictors,
    }.items():
        records.append(
            {
                "dataset": name,
                "rows": frame.shape[0],
                "columns": frame.shape[1],
                "missing_values": int(frame.isna().sum().sum()),
                "infinite_values": int(
                    np.isinf(frame.to_numpy(dtype=float)).sum()
                ),
                "duplicate_rows": int(frame.duplicated().sum()),
                "constant_columns": int(
                    (frame.nunique(dropna=False) <= 1).sum()
                ),
            }
        )
    return pd.DataFrame(records)


if __name__ == "__main__":
    data = load_datasets(
        PATHS.train_predictors,
        PATHS.train_targets,
        PATHS.test_predictors,
    )

    print("Raw-data quality report:")
    print(summarize_datasets(data).to_string(index=False))

    preprocessor = DataPreprocessor()
    clean_train = preprocessor.fit_transform(data.train_predictors)
    clean_test = preprocessor.transform(data.test_predictors)

    print("\nPreprocessing report:")
    print(preprocessor.get_report())
    print(f"Clean train shape: {clean_train.shape}")
    print(f"Clean test shape: {clean_test.shape}")
