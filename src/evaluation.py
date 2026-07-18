"""Five-fold evaluation utilities for bitmap-aware Hot Deck prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import (
    DIAGNOSTICS,
    EVALUATION,
    FEATURES,
    MODEL,
    PATHS,
    RELEVANCE,
    TARGETS,
)
from src.donor_split import DonorSplitter
from src.hotdeck import WeightedHotDeckImputer, stable_smallest
from src.relevance import TargetRelevanceWeighter


@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation output for a prediction matrix."""

    overall_metrics: dict[str, float]
    per_target_metrics: pd.DataFrame


@dataclass(frozen=True)
class FiveFoldEvaluationResult:
    """Outputs from global-neighbor selection and final evaluation."""

    selected_k: int
    neighbor_selection: pd.DataFrame
    model_comparison: pd.DataFrame
    fold_metrics: pd.DataFrame
    model_results: dict[str, EvaluationResult]


class MultiTargetEvaluator:
    """Evaluate categorical predictions across many target columns."""

    def evaluate(
        self,
        actual: pd.DataFrame,
        predicted: pd.DataFrame,
    ) -> EvaluationResult:
        """Calculate overall and per-target classification metrics."""

        self._validate_frames(actual, predicted)
        actual = actual[predicted.columns]
        per_target_records: list[dict[str, Any]] = []

        for column in actual.columns:
            y_true = actual[column]
            y_pred = predicted[column]
            per_target_records.append(
                {
                    "target": column,
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision_macro": precision_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),
                    "recall_macro": recall_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),
                    "f1_macro": f1_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),
                    "unique_actual_values": int(y_true.nunique()),
                }
            )

        per_target = pd.DataFrame(per_target_records)
        actual_values = actual.to_numpy()
        predicted_values = predicted.to_numpy()
        overall_metrics = {
            "cell_accuracy": float(np.mean(actual_values == predicted_values)),
            "hamming_loss": float(np.mean(actual_values != predicted_values)),
            "exact_match_accuracy": float(
                np.mean(np.all(actual_values == predicted_values, axis=1))
            ),
            "mean_target_accuracy": float(per_target["accuracy"].mean()),
            "mean_target_precision_macro": float(
                per_target["precision_macro"].mean()
            ),
            "mean_target_recall_macro": float(
                per_target["recall_macro"].mean()
            ),
            "mean_target_f1_macro": float(per_target["f1_macro"].mean()),
        }
        return EvaluationResult(overall_metrics, per_target)

    @staticmethod
    def _validate_frames(
        actual: pd.DataFrame,
        predicted: pd.DataFrame,
    ) -> None:
        if not isinstance(actual, pd.DataFrame):
            raise TypeError("actual must be a pandas DataFrame.")
        if not isinstance(predicted, pd.DataFrame):
            raise TypeError("predicted must be a pandas DataFrame.")
        if actual.empty or predicted.empty:
            raise ValueError("actual and predicted must not be empty.")
        if len(actual) != len(predicted):
            raise ValueError(
                "actual and predicted must have the same number of rows."
            )
        if not actual.index.equals(predicted.index):
            raise ValueError(
                "actual and predicted must have identical aligned indices."
            )
        if actual.columns.tolist() != predicted.columns.tolist():
            raise ValueError(
                "actual and predicted must have identical target columns "
                "in the same order."
            )
        if actual.isna().any().any() or predicted.isna().any().any():
            raise ValueError(
                "actual and predicted must not contain missing values."
            )


class MajorityClassBaseline:
    """Predict every target using its most common donor value."""

    def __init__(self) -> None:
        self.majority_values_: Optional[pd.Series] = None
        self.is_fitted_: bool = False

    def fit(self, donor_targets: pd.DataFrame) -> "MajorityClassBaseline":
        if not isinstance(donor_targets, pd.DataFrame):
            raise TypeError("donor_targets must be a pandas DataFrame.")
        if donor_targets.empty:
            raise ValueError("donor_targets is empty.")
        if donor_targets.isna().any().any():
            raise ValueError("donor_targets contains missing values.")

        values: dict[str, Any] = {}
        for column in donor_targets.columns:
            modes = donor_targets[column].mode(dropna=False)
            values[column] = stable_smallest(modes.tolist())

        self.majority_values_ = pd.Series(values)
        self.is_fitted_ = True
        return self

    def predict(self, recipient_index: pd.Index) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict().")
        repeated = np.tile(
            self.majority_values_.to_numpy(),
            (len(recipient_index), 1),
        )
        return pd.DataFrame(
            repeated,
            columns=self.majority_values_.index,
            index=recipient_index,
        )


def _coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        try:
            result[column] = pd.to_numeric(result[column])
        except (TypeError, ValueError):
            pass
    return result


def select_global_neighbor_count(
    candidate_metrics: pd.DataFrame,
    f1_tolerance: float = EVALUATION.f1_tolerance,
) -> tuple[int, pd.DataFrame]:
    """Select one global k using F1 tolerance, accuracy and smallest k."""

    required = {
        "n_neighbors",
        EVALUATION.primary_metric,
        EVALUATION.secondary_metric,
    }
    missing = sorted(required - set(candidate_metrics.columns))
    if missing:
        raise ValueError(f"candidate_metrics is missing columns: {missing}")
    if candidate_metrics.empty:
        raise ValueError("candidate_metrics must not be empty.")
    if f1_tolerance < 0:
        raise ValueError("f1_tolerance must be non-negative.")
    if EVALUATION.final_tie_breaker != "smallest_k":
        raise ValueError("final_tie_breaker must be 'smallest_k'.")

    selection = candidate_metrics.copy()
    best_f1 = float(selection[EVALUATION.primary_metric].max())
    selection["f1_difference_from_best"] = (
        best_f1 - selection[EVALUATION.primary_metric]
    )
    selection["within_f1_tolerance"] = (
        selection["f1_difference_from_best"]
        <= f1_tolerance + np.finfo(float).eps
    )
    eligible = selection[selection["within_f1_tolerance"]].sort_values(
        [EVALUATION.secondary_metric, "n_neighbors"],
        ascending=[False, True],
        kind="stable",
    )
    selected_k = int(eligible.iloc[0]["n_neighbors"])
    selection["selected"] = selection["n_neighbors"].eq(selected_k)
    return selected_k, selection.sort_values("n_neighbors").reset_index(drop=True)


def evaluate_and_select_global_k_five_fold(
    predictors: pd.DataFrame,
    targets: pd.DataFrame,
    neighbor_candidates: tuple[int, ...] = MODEL.neighbor_candidates,
    n_splits: int = EVALUATION.n_splits,
    random_state: int = EVALUATION.random_state,
) -> FiveFoldEvaluationResult:
    """Select one global k while holding out every row exactly once."""

    if len(predictors) != len(targets):
        raise ValueError("predictors and targets must have the same row count.")
    if not predictors.index.equals(targets.index):
        raise ValueError("predictors and targets must have aligned indices.")
    if n_splits < 2 or n_splits > len(predictors):
        raise ValueError("n_splits must be between 2 and the number of rows.")
    candidates = tuple(sorted(set(int(k) for k in neighbor_candidates)))
    if not candidates or candidates[0] < 1:
        raise ValueError("neighbor_candidates must contain positive integers.")
    smallest_training_fold = len(predictors) - int(
        np.ceil(len(predictors) / n_splits)
    )
    if candidates[-1] > smallest_training_fold:
        raise ValueError(
            "The largest neighbor candidate exceeds a training fold size."
        )

    candidate_oof = {
        k: pd.DataFrame(index=targets.index, columns=targets.columns)
        for k in candidates
    }
    majority_oof = pd.DataFrame(index=targets.index, columns=targets.columns)
    evaluator = MultiTargetEvaluator()
    all_fold_records: list[dict[str, Any]] = []
    splitter = DonorSplitter(
        n_splits=n_splits,
        random_state=random_state,
    )

    for split in splitter.split(predictors, targets):
        fold = split.fold
        donor_predictors = split.donor_predictors
        donor_targets = split.donor_targets
        recipient_predictors = split.recipient_predictors
        recipient_targets = split.recipient_targets

        weighter = TargetRelevanceWeighter(
            weight_floor=RELEVANCE.weight_floor,
        ).fit(donor_predictors, donor_targets)
        candidate_predictions = WeightedHotDeckImputer(
            n_neighbors=candidates[-1],
            voting=MODEL.voting,
            chunk_size=MODEL.chunk_size,
        ).fit(
            donor_predictors,
            donor_targets,
            weighter.get_weight_table(),
        ).predict_for_neighbor_counts(recipient_predictors, candidates)
        majority_predictions = MajorityClassBaseline().fit(
            donor_targets
        ).predict(recipient_predictors.index)

        for k, predictions in candidate_predictions.items():
            candidate_oof[k].loc[recipient_targets.index] = predictions
            fold_result = evaluator.evaluate(recipient_targets, predictions)
            all_fold_records.append(
                {
                    "fold": fold,
                    "model": "target_relevance_hotdeck",
                    "distance_weighting": (
                        "target_group_raw_mutual_information"
                    ),
                    "voting": MODEL.voting,
                    "n_neighbors": k,
                    "training_rows": len(donor_predictors),
                    "test_rows": len(recipient_predictors),
                    **fold_result.overall_metrics,
                }
            )
        majority_oof.loc[recipient_targets.index] = majority_predictions
        majority_result = evaluator.evaluate(
            recipient_targets,
            majority_predictions,
        )
        all_fold_records.append(
            {
                "fold": fold,
                "model": "majority_baseline",
                "distance_weighting": "none",
                "voting": "majority",
                "n_neighbors": np.nan,
                "training_rows": len(donor_predictors),
                "test_rows": len(recipient_predictors),
                **majority_result.overall_metrics,
            }
        )

    all_fold_metrics = pd.DataFrame(all_fold_records)
    candidate_results: dict[int, EvaluationResult] = {}
    candidate_records: list[dict[str, Any]] = []
    for k, predictions in candidate_oof.items():
        result = evaluator.evaluate(targets, _coerce_numeric(predictions))
        candidate_results[k] = result
        folds = all_fold_metrics[
            (all_fold_metrics["model"] == "target_relevance_hotdeck")
            & (all_fold_metrics["n_neighbors"] == k)
        ]
        candidate_records.append(
            {
                "n_neighbors": k,
                **result.overall_metrics,
                "fold_cell_accuracy_mean": folds["cell_accuracy"].mean(),
                "fold_cell_accuracy_std": folds["cell_accuracy"].std(ddof=1),
                "fold_mean_target_f1_macro_mean": folds[
                    "mean_target_f1_macro"
                ].mean(),
                "fold_mean_target_f1_macro_std": folds[
                    "mean_target_f1_macro"
                ].std(ddof=1),
            }
        )

    selected_k, neighbor_selection = select_global_neighbor_count(
        pd.DataFrame(candidate_records)
    )
    selected_result = candidate_results[selected_k]
    majority_result = evaluator.evaluate(
        targets,
        _coerce_numeric(majority_oof),
    )
    selected_folds = all_fold_metrics[
        (all_fold_metrics["model"] == "target_relevance_hotdeck")
        & (all_fold_metrics["n_neighbors"] == selected_k)
    ]
    majority_folds = all_fold_metrics[
        all_fold_metrics["model"] == "majority_baseline"
    ]
    fold_metrics = pd.concat(
        [selected_folds, majority_folds],
        ignore_index=True,
    ).sort_values(["fold", "model"]).reset_index(drop=True)

    def comparison_record(
        model: str,
        result: EvaluationResult,
        folds: pd.DataFrame,
        distance_weighting: str,
        voting: str,
        n_neighbors: float,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "evaluation": f"{n_splits}_fold_out_of_fold",
            "distance_weighting": distance_weighting,
            "voting": voting,
            "n_neighbors": n_neighbors,
            **result.overall_metrics,
            "fold_cell_accuracy_mean": folds["cell_accuracy"].mean(),
            "fold_cell_accuracy_std": folds["cell_accuracy"].std(ddof=1),
            "fold_mean_target_f1_macro_mean": folds[
                "mean_target_f1_macro"
            ].mean(),
            "fold_mean_target_f1_macro_std": folds[
                "mean_target_f1_macro"
            ].std(ddof=1),
        }

    model_comparison = pd.DataFrame(
        [
            comparison_record(
                "target_relevance_hotdeck",
                selected_result,
                selected_folds,
                "target_group_raw_mutual_information",
                MODEL.voting,
                float(selected_k),
            ),
            comparison_record(
                "majority_baseline",
                majority_result,
                majority_folds,
                "none",
                "majority",
                np.nan,
            ),
        ]
    )
    return FiveFoldEvaluationResult(
        selected_k=selected_k,
        neighbor_selection=neighbor_selection,
        model_comparison=model_comparison,
        fold_metrics=fold_metrics,
        model_results={
            "target_relevance_hotdeck": selected_result,
            "majority_baseline": majority_result,
        },
    )


def summarize_by_target_class_count(
    per_target_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize performance by the number of observed target classes."""

    return (
        per_target_metrics.groupby("unique_actual_values")
        .agg(
            target_count=("target", "count"),
            mean_accuracy=("accuracy", "mean"),
            mean_precision_macro=("precision_macro", "mean"),
            mean_recall_macro=("recall_macro", "mean"),
            mean_f1_macro=("f1_macro", "mean"),
        )
        .reset_index()
    )


if __name__ == "__main__":
    from src.feature_engineering import BitmapFeatureTransformer
    from src.preprocessing import DataPreprocessor, load_datasets
    from src.target_analysis import TargetAnalyzer

    PATHS.outputs_dir.mkdir(parents=True, exist_ok=True)
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
    transformer.get_group_table().to_csv(PATHS.bitmap_groups, index=False)
    report = transformer.get_report()
    print("Bitmap structure:")
    print(
        {
            "raw_columns": report.input_feature_count,
            "detected_groups": report.detected_group_count,
            "removed_identity_columns": len(report.removed_identity_columns),
            "removed_constant_groups": report.removed_constant_groups,
            "removed_derived_groups": report.removed_derived_groups,
            "decoded_predictor_groups": report.output_group_count,
            "test_unknown_category_cells": int(
                (decoded_test == transformer.unknown_category_value).sum().sum()
            ),
        }
    )
    analyzer = TargetAnalyzer().fit(data.train_targets)
    modeling_targets = (
        analyzer.remove_constant_targets(data.train_targets)
        if TARGETS.remove_constant_targets
        else data.train_targets.copy()
    )
    evaluation = evaluate_and_select_global_k_five_fold(
        decoded_train,
        modeling_targets,
    )
    evaluation.model_comparison.to_csv(PATHS.model_comparison, index=False)
    evaluation.fold_metrics.to_csv(PATHS.fold_metrics, index=False)
    evaluation.neighbor_selection.to_csv(
        PATHS.neighbor_selection,
        index=False,
    )
    official = evaluation.model_results["target_relevance_hotdeck"]
    official.per_target_metrics.to_csv(PATHS.best_target_metrics, index=False)
    class_summary = summarize_by_target_class_count(
        official.per_target_metrics
    )
    class_summary.to_csv(PATHS.best_class_summary, index=False)

    print("\nSelected model configuration:")
    print(
        {
            "folds": EVALUATION.n_splits,
            "mutual_information": "raw",
            "neighbor_candidates": MODEL.neighbor_candidates,
            "f1_tolerance": EVALUATION.f1_tolerance,
            "selected_n_neighbors": evaluation.selected_k,
            "voting": MODEL.voting,
        }
    )
    print("\nNeighbor selection:")
    print(evaluation.neighbor_selection.to_string(index=False))
    print("\nOut-of-fold model comparison:")
    print(evaluation.model_comparison.to_string(index=False))
    print("\nFold metrics:")
    print(evaluation.fold_metrics.to_string(index=False))
    print("\nPerformance by target class count:")
    print(class_summary.to_string(index=False))
    print("\nThirty lowest-F1 targets:")
    print(
        official.per_target_metrics.sort_values("f1_macro")
        .head(DIAGNOSTICS.lowest_f1_targets)
        .to_string(index=False)
    )
