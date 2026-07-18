"""Bitmap-group discovery and categorical feature transformation.

The supplied predictor matrix is not a collection of independent binary
variables.  It is a sequence of disjoint, exhaustive one-hot bitmap blocks.
This module reconstructs those blocks, removes structural identifiers and
redundant groups, and decodes each retained bitmap to one categorical value.

A decoded group contributes one mismatch to categorical Hamming distance,
regardless of whether the original bitmap used two, five, or eleven columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import FEATURES, PATHS


@dataclass(frozen=True)
class BitmapGroup:
    """Metadata for one consecutive one-hot encoded variable."""

    name: str
    columns: tuple[str, ...]
    start_column: str
    end_column: str
    size: int


@dataclass(frozen=True)
class FeatureSelectionReport:
    """Summary of bitmap reconstruction and structural removals."""

    input_feature_count: int
    detected_group_count: int
    output_group_count: int
    removed_identity_groups: list[str]
    removed_identity_columns: list[str]
    removed_constant_groups: list[str]
    removed_constant_columns: list[str]
    removed_duplicate_groups: list[str]
    removed_derived_groups: list[str]
    retained_groups: list[str]
    unknown_category_value: int


class BitmapFeatureTransformer:
    """Detect consecutive one-hot blocks and decode them to categories.

    Parameters
    ----------
    drop_identity_groups:
        Remove identity-style blocks.  A block is classified as an identity
        block when it has one column per training row and every column is active
        in exactly one row.  In this project that is ``pred_0``–``pred_1014``.
    drop_constant_groups:
        Remove groups whose decoded value never changes.  This includes a
        one-column all-one bitmap.
    drop_derived_groups:
        Remove exact coarser groups that are deterministic functions of a more
        detailed group.  Keeping both would count the same hierarchy twice.
    unknown_category_value:
        Code used when a transformed row has no active bit in a known group.
        The final test set contains such all-zero blocks; they are treated as
        an explicit unknown/unseen category rather than imputed bit by bit.
    """

    def __init__(
        self,
        *,
        drop_identity_groups: bool = FEATURES.drop_identity_groups,
        drop_constant_groups: bool = FEATURES.drop_constant_groups,
        drop_derived_groups: bool = FEATURES.drop_derived_groups,
        drop_duplicate_groups: bool = FEATURES.drop_duplicate_groups,
        unknown_category_value: int = FEATURES.unknown_category_value,
    ) -> None:
        if not isinstance(unknown_category_value, int):
            raise TypeError("unknown_category_value must be an integer.")

        self.drop_identity_groups = drop_identity_groups
        self.drop_constant_groups = drop_constant_groups
        self.drop_derived_groups = drop_derived_groups
        self.drop_duplicate_groups = drop_duplicate_groups
        self.unknown_category_value = unknown_category_value

        self.input_columns_: Optional[list[str]] = None
        self.detected_groups_: Optional[list[BitmapGroup]] = None
        self.output_groups_: Optional[list[BitmapGroup]] = None
        self.removed_identity_groups_: Optional[list[BitmapGroup]] = None
        self.removed_constant_groups_: Optional[list[BitmapGroup]] = None
        self.removed_duplicate_groups_: Optional[list[BitmapGroup]] = None
        self.removed_derived_groups_: Optional[list[BitmapGroup]] = None
        self.derived_from_: Optional[dict[str, str]] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        train_predictors: pd.DataFrame,
    ) -> "BitmapFeatureTransformer":
        """Infer the bitmap schema from the complete training predictors."""

        self._validate_binary_frame(
            train_predictors,
            frame_name="train_predictors",
        )
        if train_predictors.isna().any().any():
            raise ValueError(
                "train_predictors contains missing values. Run the data "
                "preprocessor before bitmap detection."
            )

        self.input_columns_ = train_predictors.columns.tolist()
        self.detected_groups_ = self._detect_consecutive_one_hot_groups(
            train_predictors
        )

        decoded_all = self._decode_groups(
            train_predictors,
            self.detected_groups_,
            allow_unknown=False,
        )

        self.removed_identity_groups_ = []
        self.removed_constant_groups_ = []
        candidate_groups: list[BitmapGroup] = []

        for group in self.detected_groups_:
            decoded = decoded_all[group.name]

            is_identity = (
                group.size == len(train_predictors)
                and train_predictors[list(group.columns)]
                .sum(axis=0)
                .eq(1)
                .all()
            )
            is_constant = decoded.nunique(dropna=False) <= 1

            if self.drop_identity_groups and is_identity:
                self.removed_identity_groups_.append(group)
            elif self.drop_constant_groups and is_constant:
                self.removed_constant_groups_.append(group)
            else:
                candidate_groups.append(group)

        self.removed_duplicate_groups_ = []
        if self.drop_duplicate_groups:
            candidate_names = [group.name for group in candidate_groups]
            duplicate_mask = decoded_all[candidate_names].T.duplicated(
                keep="first"
            )
            duplicate_names = set(duplicate_mask[duplicate_mask].index)
            self.removed_duplicate_groups_ = [
                group for group in candidate_groups
                if group.name in duplicate_names
            ]
            candidate_groups = [
                group for group in candidate_groups
                if group.name not in duplicate_names
            ]

        self.removed_derived_groups_ = []
        self.derived_from_ = {}
        if self.drop_derived_groups:
            self.removed_derived_groups_, self.derived_from_ = (
                self._find_redundant_derived_groups(
                    decoded_all,
                    candidate_groups,
                )
            )

        removed_derived_names = {
            group.name for group in self.removed_derived_groups_
        }
        self.output_groups_ = [
            group
            for group in candidate_groups
            if group.name not in removed_derived_names
        ]

        if not self.output_groups_:
            raise ValueError(
                "No usable bitmap groups remain after structural filtering."
            )

        self.is_fitted_ = True
        return self

    def transform(self, predictors: pd.DataFrame) -> pd.DataFrame:
        """Decode fitted bitmap groups for donor, recipient, or test rows."""

        self._check_fitted()
        self._validate_binary_frame(
            predictors,
            frame_name="predictors",
        )
        self._validate_columns(predictors)

        return self._decode_groups(
            predictors,
            self.output_groups_ or [],
            allow_unknown=True,
        )

    def fit_transform(
        self,
        train_predictors: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fit on and decode the training predictor matrix."""

        return self.fit(train_predictors).transform(train_predictors)

    def get_group_table(self) -> pd.DataFrame:
        """Return one readable record per detected group."""

        self._check_fitted()

        identity_names = {
            group.name for group in self.removed_identity_groups_ or []
        }
        constant_names = {
            group.name for group in self.removed_constant_groups_ or []
        }
        duplicate_names = {
            group.name for group in self.removed_duplicate_groups_ or []
        }
        derived_names = {
            group.name for group in self.removed_derived_groups_ or []
        }
        retained_names = {group.name for group in self.output_groups_ or []}

        records: list[dict[str, Any]] = []
        for group in self.detected_groups_ or []:
            if group.name in identity_names:
                status = "removed_identity"
            elif group.name in constant_names:
                status = "removed_constant"
            elif group.name in duplicate_names:
                status = "removed_duplicate_group"
            elif group.name in derived_names:
                status = "removed_derived"
            elif group.name in retained_names:
                status = "retained"
            else:
                status = "removed"

            records.append(
                {
                    "group": group.name,
                    "start_column": group.start_column,
                    "end_column": group.end_column,
                    "bitmap_size": group.size,
                    "status": status,
                    "derived_from": (self.derived_from_ or {}).get(group.name),
                    "columns": ", ".join(group.columns),
                }
            )

        return pd.DataFrame(records)

    def get_report(self) -> FeatureSelectionReport:
        """Return the structural feature-selection report."""

        self._check_fitted()

        identity = self.removed_identity_groups_ or []
        constant = self.removed_constant_groups_ or []
        duplicate = self.removed_duplicate_groups_ or []
        derived = self.removed_derived_groups_ or []
        retained = self.output_groups_ or []

        return FeatureSelectionReport(
            input_feature_count=len(self.input_columns_ or []),
            detected_group_count=len(self.detected_groups_ or []),
            output_group_count=len(retained),
            removed_identity_groups=[group.name for group in identity],
            removed_identity_columns=[
                column for group in identity for column in group.columns
            ],
            removed_constant_groups=[group.name for group in constant],
            removed_constant_columns=[
                column for group in constant for column in group.columns
            ],
            removed_duplicate_groups=[group.name for group in duplicate],
            removed_derived_groups=[group.name for group in derived],
            retained_groups=[group.name for group in retained],
            unknown_category_value=self.unknown_category_value,
        )

    @staticmethod
    def _detect_consecutive_one_hot_groups(
        predictors: pd.DataFrame,
    ) -> list[BitmapGroup]:
        """Greedily recover the shortest consecutive exact one-hot blocks."""

        columns = predictors.columns.tolist()
        values = predictors.to_numpy(dtype=np.uint8)
        groups: list[BitmapGroup] = []
        start = 0

        while start < len(columns):
            running_sum = np.zeros(len(predictors), dtype=np.uint16)
            end_found: Optional[int] = None

            for end in range(start, len(columns)):
                running_sum += values[:, end]

                if np.any(running_sum > 1):
                    break
                if np.all(running_sum == 1):
                    end_found = end + 1
                    break

            if end_found is None:
                sample = columns[start : min(start + 8, len(columns))]
                raise ValueError(
                    "Could not reconstruct an exact consecutive one-hot "
                    f"group beginning at {columns[start]}. Nearby columns: "
                    f"{sample}."
                )

            group_columns = tuple(columns[start:end_found])
            group_number = len(groups)
            groups.append(
                BitmapGroup(
                    name=f"bitmap_{group_number:02d}",
                    columns=group_columns,
                    start_column=group_columns[0],
                    end_column=group_columns[-1],
                    size=len(group_columns),
                )
            )
            start = end_found

        return groups

    def _decode_groups(
        self,
        predictors: pd.DataFrame,
        groups: list[BitmapGroup],
        *,
        allow_unknown: bool,
    ) -> pd.DataFrame:
        records: dict[str, np.ndarray] = {}

        for group in groups:
            group_values = predictors[list(group.columns)].to_numpy(
                dtype=np.uint8
            )
            row_sums = group_values.sum(axis=1)

            if np.any(row_sums > 1):
                bad_rows = predictors.index[row_sums > 1].tolist()[:10]
                raise ValueError(
                    f"Bitmap group {group.name} ({group.start_column}–"
                    f"{group.end_column}) has multiple active categories in "
                    f"rows {bad_rows}."
                )

            if not allow_unknown and np.any(row_sums == 0):
                bad_rows = predictors.index[row_sums == 0].tolist()[:10]
                raise ValueError(
                    f"Training bitmap group {group.name} contains all-zero "
                    f"rows {bad_rows}; exact one-hot training groups are "
                    "required for schema discovery."
                )

            decoded = np.argmax(group_values, axis=1).astype(np.int16)
            decoded[row_sums == 0] = self.unknown_category_value
            records[group.name] = decoded

        return pd.DataFrame(records, index=predictors.index)

    @staticmethod
    def _find_redundant_derived_groups(
        decoded: pd.DataFrame,
        groups: list[BitmapGroup],
    ) -> tuple[list[BitmapGroup], dict[str, str]]:
        """Find exact coarser variables deterministically derived from detail.

        If every category of a higher-cardinality group maps to exactly one
        category of a lower-cardinality group, the lower-cardinality group is
        an exact parent/aggregate.  It is removed to avoid double-counting.
        """

        by_name = {group.name: group for group in groups}
        removed_names: set[str] = set()
        derived_from: dict[str, str] = {}

        for detailed in groups:
            detailed_values = decoded[detailed.name]
            detailed_cardinality = int(detailed_values.nunique())

            for coarse in groups:
                if detailed.name == coarse.name:
                    continue
                coarse_values = decoded[coarse.name]
                coarse_cardinality = int(coarse_values.nunique())

                if detailed_cardinality <= coarse_cardinality:
                    continue

                maximum_mapping_count = (
                    pd.DataFrame(
                        {
                            "detailed": detailed_values,
                            "coarse": coarse_values,
                        }
                    )
                    .groupby("detailed", dropna=False)["coarse"]
                    .nunique(dropna=False)
                    .max()
                )

                if int(maximum_mapping_count) == 1:
                    removed_names.add(coarse.name)
                    derived_from[coarse.name] = detailed.name

        removed = [
            by_name[name]
            for name in [group.name for group in groups]
            if name in removed_names
        ]
        return removed, derived_from

    @staticmethod
    def _validate_binary_frame(
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
                f"{frame_name} contains duplicate names: {duplicates}"
            )

        non_numeric = frame.select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric:
            raise TypeError(
                f"{frame_name} contains non-numeric columns: {non_numeric}"
            )

        values = frame.to_numpy()
        non_missing = values[~pd.isna(values)]
        if not np.isin(non_missing, [0, 1]).all():
            raise ValueError(
                f"{frame_name} must contain only binary values 0 and 1."
            )

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        expected = self.input_columns_ or []
        actual = frame.columns.tolist()
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise ValueError(
                "Predictor columns do not match the fitted bitmap schema in "
                f"name and order. Missing: {missing}; extra: {extra}"
            )

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before transform().")


if __name__ == "__main__":
    from src.preprocessing import DataPreprocessor, load_datasets

    data = load_datasets(
        PATHS.train_predictors,
        PATHS.train_targets,
        PATHS.test_predictors,
    )

    preprocessor = DataPreprocessor()
    clean_train = preprocessor.fit_transform(data.train_predictors)
    clean_test = preprocessor.transform(data.test_predictors)

    transformer = BitmapFeatureTransformer(
        drop_identity_groups=FEATURES.drop_identity_groups,
        drop_constant_groups=FEATURES.drop_constant_groups,
        drop_derived_groups=FEATURES.drop_derived_groups,
        drop_duplicate_groups=FEATURES.drop_duplicate_groups,
        unknown_category_value=FEATURES.unknown_category_value,
    )
    decoded_train = transformer.fit_transform(clean_train)
    decoded_test = transformer.transform(clean_test)

    report = transformer.get_report()
    print("Bitmap feature report:")
    print(
        {
            "input_feature_count": report.input_feature_count,
            "detected_group_count": report.detected_group_count,
            "output_group_count": report.output_group_count,
            "removed_identity_column_count": len(
                report.removed_identity_columns
            ),
            "removed_constant_groups": report.removed_constant_groups,
            "removed_duplicate_groups": report.removed_duplicate_groups,
            "removed_derived_groups": report.removed_derived_groups,
        }
    )
    print("\nDetected groups:")
    print(
        transformer.get_group_table()[
            [
                "group",
                "start_column",
                "end_column",
                "bitmap_size",
                "status",
                "derived_from",
            ]
        ].to_string(index=False)
    )
    print(f"\nDecoded train shape: {decoded_train.shape}")
    print(f"Decoded test shape: {decoded_test.shape}")
    print(
        "Test unknown-category cells:",
        int((decoded_test == transformer.unknown_category_value).sum().sum()),
    )
