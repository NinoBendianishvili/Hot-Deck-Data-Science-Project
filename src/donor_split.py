"""Reproducible donor-recipient folds for Hot Deck evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import KFold

from src.config import EVALUATION, FEATURES, PATHS, TARGETS


@dataclass(frozen=True)
class DonorSplit:
    """One fold containing donor training rows and held-out recipients."""

    fold: int
    donor_predictors: pd.DataFrame
    donor_targets: pd.DataFrame
    recipient_predictors: pd.DataFrame
    recipient_targets: pd.DataFrame


class DonorSplitter:
    """Generate shuffled reproducible donor-recipient folds."""

    def __init__(
        self,
        n_splits: int = EVALUATION.n_splits,
        random_state: int = EVALUATION.random_state,
    ) -> None:
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer of at least 2.")
        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer.")
        self.n_splits = n_splits
        self.random_state = random_state

    def split(
        self,
        predictors: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> Iterator[DonorSplit]:
        """Yield every row once as a held-out recipient."""

        self._validate_inputs(predictors, targets)
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        for fold, (donor_positions, recipient_positions) in enumerate(
            splitter.split(predictors),
            start=1,
        ):
            yield DonorSplit(
                fold=fold,
                donor_predictors=predictors.iloc[donor_positions],
                donor_targets=targets.iloc[donor_positions],
                recipient_predictors=predictors.iloc[recipient_positions],
                recipient_targets=targets.iloc[recipient_positions],
            )

    def _validate_inputs(
        self,
        predictors: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> None:
        if not isinstance(predictors, pd.DataFrame):
            raise TypeError("predictors must be a DataFrame.")
        if not isinstance(targets, pd.DataFrame):
            raise TypeError("targets must be a DataFrame.")
        if predictors.empty or targets.empty:
            raise ValueError("predictors and targets must not be empty.")
        if len(predictors) != len(targets):
            raise ValueError("predictors and targets must have equal rows.")
        if not predictors.index.equals(targets.index):
            raise ValueError("predictors and targets must have aligned indices.")
        if self.n_splits > len(predictors):
            raise ValueError("n_splits cannot exceed the number of rows.")


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
    decoded_train = BitmapFeatureTransformer(
        drop_identity_groups=FEATURES.drop_identity_groups,
        drop_constant_groups=FEATURES.drop_constant_groups,
        drop_derived_groups=FEATURES.drop_derived_groups,
        drop_duplicate_groups=FEATURES.drop_duplicate_groups,
        unknown_category_value=FEATURES.unknown_category_value,
    ).fit_transform(clean_train)
    analyzer = TargetAnalyzer().fit(data.train_targets)
    modeling_targets = (
        analyzer.remove_constant_targets(data.train_targets)
        if TARGETS.remove_constant_targets
        else data.train_targets.copy()
    )
    folds = DonorSplitter().split(decoded_train, modeling_targets)
    for split in folds:
        print(
            {
                "fold": split.fold,
                "donor_rows": len(split.donor_predictors),
                "recipient_rows": len(split.recipient_predictors),
            }
        )
