"""End-to-end bitmap-group-aware Hot Deck prediction pipeline."""

from __future__ import annotations

import json

import pandas as pd

from src.config import EVALUATION, FEATURES, MODEL, PATHS, RELEVANCE, TARGETS
from src.evaluation import (
    evaluate_and_select_global_k_five_fold,
    summarize_by_target_class_count,
)
from src.feature_engineering import BitmapFeatureTransformer
from src.hotdeck import WeightedHotDeckImputer
from src.preprocessing import DataPreprocessor, load_datasets
from src.relevance import TargetRelevanceWeighter
from src.target_analysis import TargetAnalyzer


def run_pipeline() -> pd.DataFrame:
    """Fit on all labeled rows and predict all final test targets."""

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
    selected_k = evaluation.selected_k

    weighter = TargetRelevanceWeighter(
        weight_floor=RELEVANCE.weight_floor,
    ).fit(
        decoded_train,
        modeling_targets,
    )
    model = WeightedHotDeckImputer(
        n_neighbors=selected_k,
        voting=MODEL.voting,
        chunk_size=MODEL.chunk_size,
    ).fit(
        decoded_train,
        modeling_targets,
        weighter.get_weight_table(),
    )
    modeling_predictions = model.predict(decoded_test)

    constant_targets = (
        set(analyzer.get_groups().constant)
        if TARGETS.remove_constant_targets
        else set()
    )
    prediction_columns = {
        column: (
            pd.Series(
                data.train_targets[column].iloc[0],
                index=decoded_test.index,
            )
            if column in constant_targets
            else modeling_predictions[column]
        )
        for column in data.train_targets.columns
    }
    final_predictions = pd.DataFrame(
        prediction_columns,
        index=decoded_test.index,
    )

    PATHS.outputs_dir.mkdir(parents=True, exist_ok=True)
    final_predictions.to_csv(PATHS.predictions, index=False)
    evaluation.model_comparison.to_csv(PATHS.model_comparison, index=False)
    evaluation.fold_metrics.to_csv(PATHS.fold_metrics, index=False)
    evaluation.neighbor_selection.to_csv(
        PATHS.neighbor_selection,
        index=False,
    )
    official_evaluation = evaluation.model_results[
        "target_relevance_hotdeck"
    ]
    official_evaluation.per_target_metrics.to_csv(
        PATHS.best_target_metrics,
        index=False,
    )
    summarize_by_target_class_count(
        official_evaluation.per_target_metrics
    ).to_csv(
        PATHS.best_class_summary,
        index=False,
    )

    transformer.get_group_table().to_csv(
        PATHS.bitmap_groups,
        index=False,
    )
    weighter.get_weight_table().to_csv(
        PATHS.target_group_weights,
    )

    metadata = {
        "n_neighbors": selected_k,
        "neighbor_candidates": list(MODEL.neighbor_candidates),
        "neighbor_selection_folds": EVALUATION.n_splits,
        "neighbor_selection_primary_metric": EVALUATION.primary_metric,
        "neighbor_selection_secondary_metric": EVALUATION.secondary_metric,
        "neighbor_selection_f1_tolerance": EVALUATION.f1_tolerance,
        "neighbor_selection_final_tie_breaker": (
            EVALUATION.final_tie_breaker
        ),
        "distance": "target-specific weighted categorical Hamming",
        "group_weighting": "raw mutual information",
        "voting": MODEL.voting,
        "raw_predictor_columns": data.train_predictors.shape[1],
        "decoded_predictor_groups": decoded_train.shape[1],
        "training_rows": len(decoded_train),
        "test_rows": len(decoded_test),
        "modeling_targets": modeling_targets.shape[1],
        "constant_targets_restored": len(constant_targets),
        "test_unknown_category_cells": int(
            (decoded_test == transformer.unknown_category_value).sum().sum()
        ),
        "output_file": str(
            PATHS.predictions.relative_to(PATHS.project_root)
        ),
    }
    PATHS.pipeline_metadata.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return final_predictions


if __name__ == "__main__":
    predictions = run_pipeline()
    print(f"Created predictions with shape {predictions.shape}.")
    print(
        "Saved to "
        f"{PATHS.predictions.relative_to(PATHS.project_root)}"
    )