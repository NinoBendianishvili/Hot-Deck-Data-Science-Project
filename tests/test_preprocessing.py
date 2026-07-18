"""Unit tests for bitmap-preserving preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import DataPreprocessor


def test_preprocessor_preserves_columns_and_maps_missing_to_zero() -> None:
    train = pd.DataFrame(
        {
            "pred_0": [1.0, 0.0, np.nan, 0.0],
            "pred_1": [0.0, 1.0, 0.0, 1.0],
            "constant": [1.0, 1.0, 1.0, 1.0],
        }
    )

    preprocessor = DataPreprocessor()
    transformed = preprocessor.fit_transform(train)

    assert transformed.columns.tolist() == train.columns.tolist()
    assert transformed.isna().sum().sum() == 0
    assert transformed.loc[2, "pred_0"] == 0
    assert preprocessor.get_report()["removed_constant_columns"] == []
    assert preprocessor.get_report()["preserves_bitmap_columns"] is True


def test_transform_rejects_mismatched_columns() -> None:
    train = pd.DataFrame(
        {"pred_0": [0, 1], "pred_1": [1, 0]}
    )
    invalid_test = pd.DataFrame(
        {"pred_0": [0], "different_column": [1]}
    )

    preprocessor = DataPreprocessor().fit(train)

    with pytest.raises(ValueError, match="columns do not match"):
        preprocessor.transform(invalid_test)
