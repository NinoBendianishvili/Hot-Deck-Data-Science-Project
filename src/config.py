"""Central configuration for data paths, modeling, evaluation, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PathConfig:
    """Project input and output locations."""

    project_root: Path = PROJECT_ROOT
    datasets_dir: Path = PROJECT_ROOT / "datasets"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    train_predictors: Path = datasets_dir / "train_predictors.csv"
    train_targets: Path = datasets_dir / "train_targets.csv"
    test_predictors: Path = datasets_dir / "test_predictors.csv"
    predictions: Path = outputs_dir / "test_target_predictions.csv"
    pipeline_metadata: Path = outputs_dir / "pipeline_metadata.json"
    bitmap_groups: Path = outputs_dir / "bitmap_group_structure.csv"
    target_group_weights: Path = outputs_dir / "target_group_weights.csv"
    model_comparison: Path = outputs_dir / "model_comparison.csv"
    fold_metrics: Path = outputs_dir / "five_fold_metrics.csv"
    neighbor_selection: Path = outputs_dir / "neighbor_selection.csv"
    best_target_metrics: Path = outputs_dir / "per_target_metrics.csv"
    best_class_summary: Path = outputs_dir / "performance_by_class_count.csv"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Settings for bitmap-preserving preprocessing."""

    imputation_strategy: str = "unknown_category"


@dataclass(frozen=True)
class FeatureConfig:
    """Settings for bitmap discovery and structural filtering."""

    drop_identity_groups: bool = True
    drop_constant_groups: bool = True
    drop_derived_groups: bool = True
    drop_duplicate_groups: bool = True
    unknown_category_value: int = -1


@dataclass(frozen=True)
class TargetConfig:
    """Settings for target selection."""

    remove_constant_targets: bool = True


@dataclass(frozen=True)
class RelevanceConfig:
    """Settings for raw mutual-information group weighting."""

    weight_floor: float = 1e-6


@dataclass(frozen=True)
class ModelConfig:
    """Settings for target-weighted Hot Deck prediction."""

    n_neighbors: int = 11
    neighbor_candidates: tuple[int, ...] = (5, 7, 9, 11, 13, 15)
    voting: str = "unweighted"
    chunk_size: int = 16


@dataclass(frozen=True)
class EvaluationConfig:
    """Settings for reproducible five-fold evaluation and diagnostics."""

    random_state: int = 42
    n_splits: int = 5
    primary_metric: str = "mean_target_f1_macro"
    secondary_metric: str = "cell_accuracy"
    f1_tolerance: float = 0.001
    final_tie_breaker: str = "smallest_k"


@dataclass(frozen=True)
class DiagnosticConfig:
    """Settings for standalone stage reports."""

    top_relevance_groups: int = 5
    lowest_f1_targets: int = 30


PATHS = PathConfig()
PREPROCESSING = PreprocessingConfig()
FEATURES = FeatureConfig()
TARGETS = TargetConfig()
RELEVANCE = RelevanceConfig()
MODEL = ModelConfig()
EVALUATION = EvaluationConfig()
DIAGNOSTICS = DiagnosticConfig()
