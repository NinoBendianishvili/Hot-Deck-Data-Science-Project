"""Tests for evaluation and baseline utilities."""

import pandas as pd
import pytest

from src.evaluation import (
    MajorityClassBaseline,
    MultiTargetEvaluator,
    evaluate_and_select_global_k_five_fold,
    select_global_neighbor_count,
)


def test_perfect_predictions_receive_perfect_scores() -> None:
    actual = pd.DataFrame(
        {
            "tar_0": [0, 1, 0],
            "tar_1": [2, 2, 3],
        },
        index=[10, 11, 12],
    )

    result = MultiTargetEvaluator().evaluate(
        actual,
        actual.copy(),
    )

    assert result.overall_metrics[
        "cell_accuracy"
    ] == pytest.approx(1.0)

    assert result.overall_metrics[
        "hamming_loss"
    ] == pytest.approx(0.0)

    assert result.overall_metrics[
        "exact_match_accuracy"
    ] == pytest.approx(1.0)

    assert result.overall_metrics[
        "mean_target_f1_macro"
    ] == pytest.approx(1.0)


def test_cell_accuracy_and_exact_match_are_distinct() -> None:
    actual = pd.DataFrame(
        {
            "tar_0": [0, 1],
            "tar_1": [0, 1],
        }
    )
    predicted = pd.DataFrame(
        {
            "tar_0": [0, 1],
            "tar_1": [1, 1],
        }
    )

    result = MultiTargetEvaluator().evaluate(
        actual,
        predicted,
    )

    assert result.overall_metrics[
        "cell_accuracy"
    ] == pytest.approx(0.75)

    assert result.overall_metrics[
        "hamming_loss"
    ] == pytest.approx(0.25)

    assert result.overall_metrics[
        "exact_match_accuracy"
    ] == pytest.approx(0.50)


def test_majority_baseline_repeats_donor_modes() -> None:
    donor_targets = pd.DataFrame(
        {
            "tar_0": [0, 0, 1],
            "tar_1": [2, 3, 3],
        }
    )

    predictions = MajorityClassBaseline().fit(
        donor_targets
    ).predict(
        pd.Index([100, 101])
    )

    assert predictions.loc[100, "tar_0"] == 0
    assert predictions.loc[100, "tar_1"] == 3
    assert predictions.shape == (2, 2)


def test_evaluator_rejects_misaligned_indices() -> None:
    actual = pd.DataFrame(
        {"tar_0": [0, 1]},
        index=[0, 1],
    )
    predicted = pd.DataFrame(
        {"tar_0": [0, 1]},
        index=[10, 11],
    )

    with pytest.raises(
        ValueError,
        match="aligned indices",
    ):
        MultiTargetEvaluator().evaluate(
            actual,
            predicted,
        )


def test_five_fold_evaluation_holds_out_every_row_once() -> None:
    predictors = pd.DataFrame(
        {
            "pred_0": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            "pred_1": [0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
        },
        index=range(10),
    )
    targets = pd.DataFrame(
        {
            "tar_0": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        },
        index=range(10),
    )

    result = evaluate_and_select_global_k_five_fold(
        predictors,
        targets,
        neighbor_candidates=(1, 3, 7),
        n_splits=5,
    )

    assert set(result.model_comparison["model"]) == {
        "target_relevance_hotdeck",
        "majority_baseline",
    }
    assert result.fold_metrics.shape[0] == 10
    assert result.fold_metrics.groupby("model")["test_rows"].sum().eq(10).all()
    assert result.neighbor_selection.shape[0] == 3
    assert result.neighbor_selection["selected"].sum() == 1
    assert result.selected_k in (1, 3, 7)


def test_global_k_selection_uses_accuracy_within_f1_tolerance() -> None:
    metrics = pd.DataFrame(
        {
            "n_neighbors": [1, 3, 5],
            "mean_target_f1_macro": [0.6000, 0.5994, 0.5980],
            "cell_accuracy": [0.70, 0.80, 0.90],
        }
    )

    selected_k, selection = select_global_neighbor_count(
        metrics,
        f1_tolerance=0.001,
    )

    assert selected_k == 3
    assert selection.loc[
        selection["n_neighbors"] == 5,
        "within_f1_tolerance",
    ].item() is False


def test_global_k_selection_uses_smallest_k_for_complete_tie() -> None:
    metrics = pd.DataFrame(
        {
            "n_neighbors": [3, 5],
            "mean_target_f1_macro": [0.6000, 0.5995],
            "cell_accuracy": [0.80, 0.80],
        }
    )

    selected_k, _ = select_global_neighbor_count(metrics)

    assert selected_k == 3

