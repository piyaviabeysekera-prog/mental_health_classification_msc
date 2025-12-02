"""Explainability phase: RF + SHAP + sensitivity analysis.

This module provides a reproducible explainability and sensitivity phase for
the project. It trains a RandomForest on the full enriched dataset (including
composite indices), computes global SHAP importances, and runs a small
sensitivity experiment that removes the composite features to measure their
impact on Macro F1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from joblib import dump

from .config import (
    DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    MODELS_DIR,
    RANDOM_SEED,
)
from .utils import ensure_dir, set_global_seeds


def load_feature_dataframe() -> pd.DataFrame:
    path = DATA_DIR / "features" / "merged_with_composites.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Expected enriched feature file at {path}")
    return pd.read_parquet(path)


def split_features_labels(
    df: pd.DataFrame, label_col: str = "label", drop_cols: List[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    if drop_cols is None:
        drop_cols = []
    cols_to_drop = set(drop_cols + [label_col])
    feature_cols = [c for c in df.columns if c not in cols_to_drop]
    X = df[feature_cols].copy()
    y = df[label_col].astype(int).copy()
    return X, y


def build_rf_model(random_state: int = RANDOM_SEED) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )


def run_global_shap(max_samples: int = 500, shap_class: int = 2) -> pd.DataFrame:
    set_global_seeds()
    ensure_dir(FIGURES_DIR)
    ensure_dir(TABLES_DIR)
    ensure_dir(MODELS_DIR / "explainability")

    df = load_feature_dataframe()
    X, y = split_features_labels(df, label_col="label", drop_cols=["subject"])

    rf = build_rf_model()
    rf.fit(X, y)

    dump(rf, MODELS_DIR / "explainability" / "rf_explainability_model.joblib")

    if len(X) > max_samples:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=RANDOM_SEED)
        sample_idx = next(sss.split(X, y))[0]
        X_sample = X.iloc[sample_idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)

    # Normalize shap_values into a per-feature mean |SHAP| (1-D)
    if isinstance(shap_values, list):
        # list of arrays, each shape (n_samples, n_features)
        if shap_class >= len(shap_values):
            shap_arr = np.stack([np.array(sv) for sv in shap_values])  # (n_classes, n_samples, n_features)
            shap_target = np.mean(shap_arr, axis=0)
            mean_abs_shap = np.mean(np.abs(shap_arr), axis=(0, 1))
        else:
            shap_target = np.array(shap_values[shap_class])  # (n_samples, n_features)
            mean_abs_shap = np.mean(np.abs(shap_target), axis=0)
    else:
        sv = np.array(shap_values)
        if sv.ndim == 3:
            # shape (n_samples, n_features, n_outputs)
            if shap_class >= sv.shape[2]:
                shap_target = np.mean(sv, axis=2)
            else:
                shap_target = sv[:, :, shap_class]
            mean_abs_shap = np.mean(np.abs(shap_target), axis=0)
        else:
            shap_target = sv
            mean_abs_shap = np.mean(np.abs(shap_target), axis=0)

    mean_abs_shap = np.asarray(mean_abs_shap).ravel()
    shap_importance = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs_shap}).sort_values(
        "mean_abs_shap", ascending=False
    )

    top10 = shap_importance.head(10)
    top10.to_csv(TABLES_DIR / "shap_top_features.csv", index=False)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_target, X_sample, show=False, plot_type="bar", max_display=20)
    plt.title("Global SHAP Feature Importance (RF explainability model)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_global.png", dpi=300, bbox_inches="tight")
    plt.close()

    return top10


def run_sensitivity() -> pd.DataFrame:
    set_global_seeds()
    ensure_dir(TABLES_DIR)
    ensure_dir(FIGURES_DIR)

    df = load_feature_dataframe()
    base_drop = ["subject"]
    X_full, y = split_features_labels(df, "label", drop_cols=base_drop)

    variants: Dict[str, List[str]] = {}
    variants["full_with_composites"] = list(X_full.columns)
    for comp in ["SRI", "RS", "PL"]:
        variants[f"no_{comp}"] = [c for c in X_full.columns if c != comp]
    variants["no_SRI_RS_PL"] = [c for c in X_full.columns if c not in ["SRI", "RS", "PL"]]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    rows = []
    for name, cols in variants.items():
        X_variant = X_full[cols]
        rf = build_rf_model()
        f1_scores = cross_val_score(rf, X_variant, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        rows.append({
            "variant": name,
            "n_features": len(cols),
            "f1_macro_mean": float(np.mean(f1_scores)),
            "f1_macro_std": float(np.std(f1_scores)),
        })

    sens_df = pd.DataFrame(rows)
    sens_df.to_csv(TABLES_DIR / "sensitivity.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(sens_df["variant"], sens_df["f1_macro_mean"], marker="o")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Macro F1")
    plt.title("Sensitivity to Composite Removal (RF, 5-fold CV)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sensitivity_spider.png", dpi=300, bbox_inches="tight")
    plt.close()

    return sens_df


def phase_E_explainability_and_sensitivity() -> None:
    top_shap = run_global_shap()
    sens_df = run_sensitivity()

    print("Phase E complete.")
    print("Top SHAP features (head):")
    print(top_shap)
    print("\nSensitivity summary:")
    print(sens_df)

