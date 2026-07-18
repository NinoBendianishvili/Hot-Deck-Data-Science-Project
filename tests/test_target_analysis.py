"""Tests for target analysis."""

import pandas as pd

from src.target_analysis import TargetAnalyzer


def test_target_analyzer_groups_targets_correctly() -> None:
    targets = pd.DataFrame(
        {
            "constant": [1, 1, 1, 1],
            "binary": [0, 1, 0, 1],
            "multiclass": [0, 1, 2, 1],
        }
    )

    analyzer = TargetAnalyzer().fit(targets)
    groups = analyzer.get_groups()

    assert groups.constant == ["constant"]
    assert groups.binary == ["binary"]
    assert groups.multiclass == ["multiclass"]


def test_remove_constant_targets() -> None:
    targets = pd.DataFrame(
        {
            "constant": [5, 5, 5],
            "usable": [0, 1, 0],
        }
    )

    analyzer = TargetAnalyzer().fit(targets)
    result = analyzer.remove_constant_targets(targets)

    assert result.columns.tolist() == ["usable"]
