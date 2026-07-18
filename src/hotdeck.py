"""Raw-MI-weighted Hot Deck prediction for decoded bitmap groups."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import MODEL, PATHS, RELEVANCE, TARGETS


def stable_smallest(values: list[Any]) -> Any:
    """Return the smallest numeric target value deterministically."""

    return min(values)


class WeightedHotDeckImputer:
    """Predict targets from raw-MI-weighted categorical donor distances."""

    def __init__(
        self,
        n_neighbors: int = MODEL.n_neighbors,
        voting: str = MODEL.voting,
        chunk_size: int = MODEL.chunk_size,
    ) -> None:
        if not isinstance(n_neighbors, int):
            raise TypeError("n_neighbors must be an integer.")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        if voting != "unweighted":
            raise ValueError("voting must be 'unweighted'.")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")

        self.n_neighbors = n_neighbors
        self.voting = voting
        self.chunk_size = chunk_size
        self.donor_predictors_: Optional[pd.DataFrame] = None
        self.donor_targets_: Optional[pd.DataFrame] = None
        self.group_weights_: Optional[pd.DataFrame] = None
        self.predictor_columns_: list[str] = []
        self.target_columns_: list[str] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        donor_predictors: pd.DataFrame,
        donor_targets: pd.DataFrame,
        feature_weights: pd.DataFrame,
    ) -> "WeightedHotDeckImputer":
        """Store aligned donors and normalized target-by-group weights."""

        self._validate_categorical_predictors(
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
        if self.n_neighbors > len(donor_predictors):
            raise ValueError(
                "n_neighbors cannot exceed the number of donor rows."
            )
        self._validate_group_weights(
            feature_weights,
            donor_predictors,
            donor_targets,
        )

        ordered_weights = feature_weights.loc[
            donor_targets.columns,
            donor_predictors.columns,
        ].astype(np.float64)
        row_sums = ordered_weights.sum(axis=1)
        zero_rows = row_sums <= 0
        if zero_rows.any():
            ordered_weights.loc[zero_rows, :] = 1.0
            row_sums = ordered_weights.sum(axis=1)

        self.donor_predictors_ = donor_predictors.copy()
        self.donor_targets_ = donor_targets.copy()
        self.predictor_columns_ = donor_predictors.columns.tolist()
        self.target_columns_ = donor_targets.columns.tolist()
        self.group_weights_ = ordered_weights.div(row_sums, axis=0)
        self.is_fitted_ = True
        return self

    def predict(self, recipient_predictors: pd.DataFrame) -> pd.DataFrame:
        """Predict using the model's configured global neighbor count."""

        return self.predict_for_neighbor_counts(
            recipient_predictors,
            (self.n_neighbors,),
        )[self.n_neighbors]

    def predict_for_neighbor_counts(
        self,
        recipient_predictors: pd.DataFrame,
        neighbor_counts: Sequence[int],
    ) -> dict[int, pd.DataFrame]:
        """Predict several global k values from one weighted-distance pass."""

        self._check_fitted()
        self._validate_categorical_predictors(
            recipient_predictors,
            "recipient_predictors",
        )
        self._validate_predictor_columns(recipient_predictors)
        counts = self._validate_neighbor_counts(neighbor_counts)

        donor_values = self.donor_predictors_.to_numpy(dtype=np.int32)
        recipient_values = recipient_predictors.to_numpy(dtype=np.int32)
        weight_matrix = self.group_weights_.to_numpy(dtype=np.float32)
        recipient_count = len(recipient_predictors)
        target_count = len(self.target_columns_)
        outputs = {
            k: np.empty((recipient_count, target_count), dtype=object)
            for k in counts
        }

        for chunk_start in range(0, recipient_count, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, recipient_count)
            chunk = recipient_values[chunk_start:chunk_end]
            mismatches = (
                chunk[:, None, :] != donor_values[None, :, :]
            ).astype(np.float32)
            distances = np.einsum(
                "cdg,tg->cdt",
                mismatches,
                weight_matrix,
                optimize=True,
                dtype=np.float32,
            )

            for target_position, target in enumerate(self.target_columns_):
                target_distances = distances[:, :, target_position]
                nearest = np.argsort(
                    target_distances,
                    axis=1,
                    kind="stable",
                )[:, : counts[-1]]
                target_values = self.donor_targets_[target]

                for local_row, donor_positions in enumerate(nearest):
                    output_row = chunk_start + local_row
                    ordered_values = target_values.iloc[donor_positions]
                    for k in counts:
                        outputs[k][output_row, target_position] = self._vote(
                            ordered_values.iloc[:k]
                        )

        return {
            k: pd.DataFrame(
                values,
                columns=self.target_columns_,
                index=recipient_predictors.index,
            ).infer_objects(copy=False)
            for k, values in outputs.items()
        }

    def get_parameters(self) -> dict[str, Any]:
        """Return fitted model settings and dimensions."""

        self._check_fitted()
        return {
            "distance_metric": (
                "target_specific_weighted_categorical_hamming_by_bitmap_group"
            ),
            "aggregation": self.voting,
            "n_neighbors": self.n_neighbors,
            "donor_rows": len(self.donor_predictors_),
            "predictor_group_count": len(self.predictor_columns_),
            "target_count": len(self.target_columns_),
        }

    def _validate_neighbor_counts(
        self,
        neighbor_counts: Sequence[int],
    ) -> list[int]:
        counts = sorted(set(int(value) for value in neighbor_counts))
        if not counts or counts[0] < 1:
            raise ValueError("neighbor_counts must contain positive integers.")
        if counts[-1] > len(self.donor_predictors_):
            raise ValueError(
                "A requested neighbor count exceeds the number of donors."
            )
        return counts

    @staticmethod
    def _vote(values: pd.Series) -> Any:
        value_list = values.tolist()
        counts: dict[Any, int] = {}
        for value in value_list:
            counts[value] = counts.get(value, 0) + 1
        highest_count = max(counts.values())
        tied_values = {
            value for value, count in counts.items() if count == highest_count
        }
        for value in value_list:
            if value in tied_values:
                return value
        return stable_smallest(list(tied_values))

    @staticmethod
    def _validate_categorical_predictors(
        frame: pd.DataFrame,
        frame_name: str,
    ) -> None:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{frame_name} must be a non-empty DataFrame.")
        if frame.columns.duplicated().any():
            raise ValueError(f"{frame_name} contains duplicate columns.")
        if frame.isna().any().any():
            raise ValueError(f"{frame_name} contains missing values.")
        if frame.select_dtypes(exclude=np.number).shape[1]:
            raise TypeError(f"{frame_name} must contain numeric groups.")

    @staticmethod
    def _validate_targets(targets: pd.DataFrame) -> None:
        if not isinstance(targets, pd.DataFrame) or targets.empty:
            raise ValueError("donor_targets must be a non-empty DataFrame.")
        if targets.columns.duplicated().any():
            raise ValueError("donor_targets contains duplicate columns.")
        if targets.isna().any().any():
            raise ValueError("donor_targets contains missing values.")

    @staticmethod
    def _validate_group_weights(
        weights: pd.DataFrame,
        predictors: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> None:
        if not isinstance(weights, pd.DataFrame):
            raise TypeError("feature_weights must be a DataFrame.")
        missing_targets = sorted(set(targets.columns) - set(weights.index))
        missing_groups = sorted(set(predictors.columns) - set(weights.columns))
        if missing_targets:
            raise ValueError(
                "feature_weights is missing rows for targets: "
                f"{missing_targets}"
            )
        if missing_groups:
            raise ValueError(
                "feature_weights is missing columns for predictor groups: "
                f"{missing_groups}"
            )
        selected = weights.loc[targets.columns, predictors.columns]
        values = selected.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("feature_weights must be finite and non-negative.")

    def _validate_predictor_columns(
        self,
        recipient_predictors: pd.DataFrame,
    ) -> None:
        if recipient_predictors.columns.tolist() != self.predictor_columns_:
            raise ValueError(
                "Recipient predictor groups must match donor groups in order."
            )

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict().")


if __name__ == "__main__":
    from src.feature_engineering import BitmapFeatureTransformer
    from src.preprocessing import DataPreprocessor, load_datasets
    from src.relevance import TargetRelevanceWeighter
    from src.target_analysis import TargetAnalyzer

    data = load_datasets(
        PATHS.train_predictors,
        PATHS.train_targets,
        PATHS.test_predictors,
    )
    clean_train = DataPreprocessor().fit_transform(data.train_predictors)
    decoded_train = BitmapFeatureTransformer().fit_transform(clean_train)
    analyzer = TargetAnalyzer().fit(data.train_targets)
    modeling_targets = (
        analyzer.remove_constant_targets(data.train_targets)
        if TARGETS.remove_constant_targets
        else data.train_targets.copy()
    )
    weighter = TargetRelevanceWeighter(
        weight_floor=RELEVANCE.weight_floor,
    ).fit(decoded_train, modeling_targets)
    model = WeightedHotDeckImputer().fit(
        decoded_train,
        modeling_targets,
        weighter.get_weight_table(),
    )
    print(model.get_parameters())
