"""Tests for the feature pipeline, focused on the no-leakage guarantee."""
from __future__ import annotations

from src.features.pipeline import (
    FeatureBuilder,
    feature_columns,
    placement_feature_columns,
)


def test_observed_signals_masked_until_their_checkpoint(dataset):
    features = dataset.features_test
    for column, first_index in [
        ("obs_assignment_delay_min", 1),
        ("obs_actual_prep_min", 2),
        ("obs_to_restaurant_min", 2),
    ]:
        before = features[features["checkpoint_index"] < first_index][column]
        after = features[features["checkpoint_index"] >= first_index][column]
        assert before.isna().all()
        assert after.notna().all()


def test_placement_rows_expose_no_future_signal(dataset):
    cp0 = dataset.features_test[dataset.features_test["checkpoint_index"] == 0]
    observed = [c for c in dataset.features_test.columns if c.startswith("obs_")]
    assert cp0[observed].isna().all().all()


def test_placement_features_have_no_missing_values(dataset):
    cp0 = dataset.features_test[dataset.features_test["checkpoint_index"] == 0]
    assert cp0[placement_feature_columns(dataset.features_test)].notna().all().all()


def test_target_is_present_and_non_negative(dataset):
    target = dataset.features_test["remaining_min"]
    assert target.notna().all()
    assert (target >= -1e-6).all()


def test_placement_columns_are_subset_of_all_features(dataset):
    all_cols = set(feature_columns(dataset.features_test))
    placement = set(placement_feature_columns(dataset.features_test))
    assert placement.issubset(all_cols)
    assert not any(c.startswith("obs_") for c in placement)


def test_builder_requires_fit(simulated):
    orders, events = simulated
    builder = FeatureBuilder()
    try:
        builder.transform(orders, events)
    except RuntimeError:
        return
    raise AssertionError("transform before fit should raise RuntimeError")
