"""Tests for bitmap-group discovery and decoding."""

import pandas as pd
import pytest

from src.feature_engineering import BitmapFeatureTransformer


def make_bitmap_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_0": [1, 0, 0, 0],
            "id_1": [0, 1, 0, 0],
            "id_2": [0, 0, 1, 0],
            "id_3": [0, 0, 0, 1],
            "parent_0": [1, 1, 0, 1],
            "parent_1": [0, 0, 1, 0],
            "detail_0": [1, 0, 0, 1],
            "detail_1": [0, 1, 0, 0],
            "detail_2": [0, 0, 1, 0],
            "constant": [1, 1, 1, 1],
        }
    )


def test_transformer_removes_identity_constant_and_derived_parent() -> None:
    predictors = make_bitmap_data()
    transformer = BitmapFeatureTransformer()
    decoded = transformer.fit_transform(predictors)
    report = transformer.get_report()

    assert report.detected_group_count == 4
    assert len(report.removed_identity_columns) == 4
    assert report.removed_constant_groups == ["bitmap_03"]
    assert report.removed_derived_groups == ["bitmap_01"]
    assert decoded.columns.tolist() == ["bitmap_02"]
    assert decoded["bitmap_02"].tolist() == [0, 1, 2, 0]


def test_transform_encodes_all_zero_test_group_as_unknown() -> None:
    train = make_bitmap_data()
    transformer = BitmapFeatureTransformer().fit(train)
    test = train.iloc[[0]].copy()
    test.loc[:, ["detail_0", "detail_1", "detail_2"]] = 0

    decoded = transformer.transform(test)

    assert decoded.iloc[0, 0] == -1


def test_raw_duplicate_columns_inside_different_groups_are_preserved() -> None:
    predictors = make_bitmap_data()
    transformer = BitmapFeatureTransformer(
        drop_derived_groups=False,
    )
    decoded = transformer.fit_transform(predictors)

    assert decoded.columns.tolist() == ["bitmap_01", "bitmap_02"]


def test_transform_rejects_multiple_active_categories() -> None:
    predictors = make_bitmap_data()
    transformer = BitmapFeatureTransformer().fit(predictors)
    invalid = predictors.iloc[[0]].copy()
    invalid.loc[:, "detail_1"] = 1

    with pytest.raises(ValueError, match="multiple active categories"):
        transformer.transform(invalid)


def test_fit_rejects_non_binary_values() -> None:
    predictors = make_bitmap_data()
    predictors.loc[0, "detail_0"] = 2

    with pytest.raises(ValueError, match="only binary values"):
        BitmapFeatureTransformer().fit(predictors)


def test_exact_duplicate_decoded_group_is_removed_as_a_whole() -> None:
    predictors = pd.DataFrame(
        {
            "a_0": [1, 0, 1, 0],
            "a_1": [0, 1, 0, 1],
            "b_0": [1, 0, 1, 0],
            "b_1": [0, 1, 0, 1],
        }
    )
    transformer = BitmapFeatureTransformer(
        drop_identity_groups=False,
        drop_constant_groups=False,
        drop_derived_groups=False,
        drop_duplicate_groups=True,
    )
    decoded = transformer.fit_transform(predictors)

    assert decoded.columns.tolist() == ["bitmap_00"]
    assert transformer.get_report().removed_duplicate_groups == [
        "bitmap_01"
    ]
