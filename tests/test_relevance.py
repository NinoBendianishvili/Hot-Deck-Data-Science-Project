"""Tests for per-target feature relevance weighting."""

import numpy as np
import pandas as pd
import pytest

from src.relevance import TargetRelevanceWeighter


def make_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    informative = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 5)
    noise = rng.integers(0, 2, size=len(informative))

    predictors = pd.DataFrame(
        {
            "informative": informative,
            "noise": noise,
        }
    )
    targets = pd.DataFrame({"tar_0": informative})

    return predictors, targets


def test_informative_feature_gets_more_weight_than_noise() -> None:
    predictors, targets = make_data()

    weighter = TargetRelevanceWeighter().fit(predictors, targets)
    weights = weighter.get_weights("tar_0")

    informative_weight = weights[predictors.columns.get_loc("informative")]
    noise_weight = weights[predictors.columns.get_loc("noise")]

    assert informative_weight > noise_weight


def test_weights_are_non_negative_and_finite() -> None:
    predictors, targets = make_data()

    weighter = TargetRelevanceWeighter().fit(predictors, targets)
    table = weighter.get_weight_table()

    assert (table.to_numpy() >= 0).all()
    assert np.isfinite(table.to_numpy()).all()


def test_constant_target_falls_back_to_uniform_weights() -> None:
    predictors, targets = make_data()
    targets["tar_constant"] = 0

    weighter = TargetRelevanceWeighter().fit(predictors, targets)
    weights = weighter.get_weights("tar_constant")

    assert np.allclose(weights, weights[0])


def test_get_weights_rejects_unknown_target() -> None:
    predictors, targets = make_data()
    weighter = TargetRelevanceWeighter().fit(predictors, targets)

    with pytest.raises(KeyError):
        weighter.get_weights("does_not_exist")


def test_report_lists_top_features_per_target() -> None:
    predictors, targets = make_data()
    weighter = TargetRelevanceWeighter().fit(predictors, targets)

    report = weighter.get_report(top_n=1)

    assert report.target_count == 1
    assert report.group_count == 2
    assert report.top_groups["tar_0"] == ["informative"]


def test_rejects_unfitted_access() -> None:
    with pytest.raises(RuntimeError):
        TargetRelevanceWeighter().get_weights("tar_0")


def test_rejects_misaligned_indices() -> None:
    predictors, targets = make_data()
    targets.index = targets.index + 1000

    with pytest.raises(ValueError, match="aligned indices"):
        TargetRelevanceWeighter().fit(predictors, targets)
