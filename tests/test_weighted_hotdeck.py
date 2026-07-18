"""Tests for the per-target weighted Hot Deck fusion component."""

import pandas as pd
import pytest

from src.hotdeck import WeightedHotDeckImputer


def make_donor_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors = pd.DataFrame(
        {
            "pred_0": [0, 0, 1, 1],
            "pred_1": [0, 1, 0, 1],
            "pred_2": [0, 1, 1, 1],
        },
        index=[10, 11, 12, 13],
    )

    targets = pd.DataFrame(
        {
            "tar_0": [0, 0, 1, 1],
            "tar_1": [2, 2, 3, 3],
        },
        index=[10, 11, 12, 13],
    )

    return predictors, targets


def uniform_weights(predictors: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        1.0,
        index=targets.columns,
        columns=predictors.columns,
    )


def test_uniform_weights_select_exact_matching_donor() -> None:
    donor_predictors, donor_targets = make_donor_data()
    recipients = pd.DataFrame(
        {"pred_0": [1], "pred_1": [0], "pred_2": [1]},
        index=[100],
    )

    weights = uniform_weights(donor_predictors, donor_targets)

    weighted = WeightedHotDeckImputer(n_neighbors=1).fit(
        donor_predictors, donor_targets, weights
    ).predict(recipients)

    assert weighted.loc[100, "tar_0"] == 1
    assert weighted.loc[100, "tar_1"] == 3


def test_zero_weight_feature_is_ignored_in_matching() -> None:
    donor_predictors, donor_targets = make_donor_data()
    recipients = pd.DataFrame(
        {"pred_0": [0], "pred_1": [0], "pred_2": [0]},
        index=[100],
    )

    weights = uniform_weights(donor_predictors, donor_targets)
    weights.loc[:, "pred_0"] = 0.0

    model = WeightedHotDeckImputer(n_neighbors=1).fit(
        donor_predictors, donor_targets, weights
    )
    predictions = model.predict(recipients)

    assert predictions.loc[100, "tar_0"] == 0


def test_different_targets_can_select_different_neighbours() -> None:
    donor_predictors, donor_targets = make_donor_data()
    recipients = pd.DataFrame(
        {"pred_0": [1], "pred_1": [1], "pred_2": [0]},
        index=[100],
    )

    weights = pd.DataFrame(
        {
            "pred_0": [10.0, 0.1],
            "pred_1": [0.1, 10.0],
            "pred_2": [0.1, 0.1],
        },
        index=["tar_0", "tar_1"],
    )

    model = WeightedHotDeckImputer(n_neighbors=1).fit(
        donor_predictors, donor_targets, weights
    )
    predictions = model.predict(recipients)

    assert predictions.loc[100, "tar_0"] in (1,)
    assert predictions.loc[100, "tar_1"] in (2, 3)


def test_prediction_shape_and_index_are_preserved() -> None:
    donor_predictors, donor_targets = make_donor_data()
    recipients = pd.DataFrame(
        {"pred_0": [0, 1], "pred_1": [1, 1], "pred_2": [1, 0]},
        index=[200, 201],
    )
    weights = uniform_weights(donor_predictors, donor_targets)

    predictions = WeightedHotDeckImputer(n_neighbors=1).fit(
        donor_predictors, donor_targets, weights
    ).predict(recipients)

    assert predictions.shape == (2, 2)
    assert predictions.index.tolist() == [200, 201]
    assert predictions.columns.tolist() == ["tar_0", "tar_1"]


def test_rejects_missing_target_row_in_weights() -> None:
    donor_predictors, donor_targets = make_donor_data()
    weights = uniform_weights(donor_predictors, donor_targets).drop(
        index="tar_1"
    )

    with pytest.raises(ValueError, match="missing rows"):
        WeightedHotDeckImputer(n_neighbors=1).fit(
            donor_predictors, donor_targets, weights
        )


def test_rejects_missing_feature_column_in_weights() -> None:
    donor_predictors, donor_targets = make_donor_data()
    weights = uniform_weights(donor_predictors, donor_targets).drop(
        columns="pred_2"
    )

    with pytest.raises(ValueError, match="missing columns"):
        WeightedHotDeckImputer(n_neighbors=1).fit(
            donor_predictors, donor_targets, weights
        )


def test_rejects_negative_weights() -> None:
    donor_predictors, donor_targets = make_donor_data()
    weights = uniform_weights(donor_predictors, donor_targets)
    weights.iloc[0, 0] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        WeightedHotDeckImputer(n_neighbors=1).fit(
            donor_predictors, donor_targets, weights
        )


def test_predictions_are_reproducible() -> None:
    donor_predictors, donor_targets = make_donor_data()
    recipients = donor_predictors.iloc[[0, 1]].copy()
    weights = uniform_weights(donor_predictors, donor_targets)

    model = WeightedHotDeckImputer(n_neighbors=2).fit(
        donor_predictors, donor_targets, weights
    )

    first = model.predict(recipients)
    second = model.predict(recipients)

    pd.testing.assert_frame_equal(first, second)


def test_multiple_neighbor_counts_reuse_one_fitted_model() -> None:
    donor_predictors, donor_targets = make_donor_data()
    recipients = donor_predictors.iloc[[0, 1]].copy()
    weights = uniform_weights(donor_predictors, donor_targets)

    predictions = WeightedHotDeckImputer(n_neighbors=3).fit(
        donor_predictors,
        donor_targets,
        weights,
    ).predict_for_neighbor_counts(recipients, (1, 3))
    direct = WeightedHotDeckImputer(n_neighbors=1).fit(
        donor_predictors,
        donor_targets,
        weights,
    ).predict(recipients)

    assert set(predictions) == {1, 3}
    pd.testing.assert_frame_equal(predictions[1], direct)
