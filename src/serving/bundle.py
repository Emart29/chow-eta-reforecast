"""Serving artifact bundle.

The API needs more than a trained model: it needs the exact fitted feature
pipeline used in training, the re-forecaster, the risk classifier, its decision
threshold, and the column orders each model expects. This module trains those
together on one split and persists them as a single bundle, so serving and
training never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
from xgboost import XGBClassifier, XGBRegressor

from src import config
from src.features.pipeline import FeatureBuilder, feature_columns
from src.models import dynamic
from src.models.dataset import build_dataset
from src.risk import detector

BUNDLE_PATH = config.MODELS_DIR / "serving_bundle.joblib"


@dataclass
class ServingBundle:
    """Everything the serving layer needs to turn a raw order into predictions."""

    builder: FeatureBuilder
    eta_model: XGBRegressor
    eta_feature_columns: list[str]
    risk_model: XGBClassifier
    risk_feature_columns: list[str]
    risk_threshold: float


def build_and_save() -> ServingBundle:
    """Train the serving models on the standard split and persist the bundle."""
    config.ensure_dirs()
    dataset = build_dataset()

    eta_model, _ = dynamic.train(dataset)
    risk_model, risk_metrics, _ = detector.train(dataset)

    eta_cols = feature_columns(dataset.features_test)
    bundle = ServingBundle(
        builder=dataset.builder,
        eta_model=eta_model,
        eta_feature_columns=eta_cols,
        risk_model=risk_model,
        risk_feature_columns=[*eta_cols, "reforecast_growth"],
        risk_threshold=float(risk_metrics["threshold"]),
    )
    joblib.dump(bundle, BUNDLE_PATH)
    return bundle


def load_bundle() -> ServingBundle:
    """Load the persisted serving bundle, or explain how to create it."""
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Serving bundle not found at {BUNDLE_PATH}. "
            "Build it with `python -m src.serving.bundle`."
        )
    return joblib.load(BUNDLE_PATH)


def main() -> None:
    # Import the module under its real name so pickled classes are tagged
    # ``src.serving.bundle`` rather than ``__main__`` when run with ``-m``.
    from src.serving import bundle as bundle_module  # noqa: PLW0406 (intentional, see above)

    bundle = bundle_module.build_and_save()
    print("Serving bundle trained and saved.")
    print(f"  eta features:   {len(bundle.eta_feature_columns)}")
    print(f"  risk features:  {len(bundle.risk_feature_columns)}")
    print(f"  risk threshold: {bundle.risk_threshold:.3f}")
    print(f"  saved -> {BUNDLE_PATH}")


if __name__ == "__main__":
    main()
