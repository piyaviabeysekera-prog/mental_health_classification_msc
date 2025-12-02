"""
Phase F — Fairness, Reject Option & Packaging.

This module does three things:

1. Train a cross-validated probabilistic RF model on the enriched
   feature set and compute a *fairness / uncertainty proxy*:
   - per-subject variance of predicted high-risk probability.
2. Build a simple *reject-option* band around ambiguous probabilities
   (e.g. 0.45–0.55) and show how error rate vs. coverage changes.
3. Package all key tables & figures into final “thesis” folders and
   write a manifest.json describing what was produced.

It operates purely on:
- data_stage/features/merged_with_composites.parquet
- existing tables in reports/tables
- existing figures in reports/figures
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import shutil
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score

from .config import DATA_DIR, TABLES_DIR, FIGURES_DIR, RANDOM_SEED
from .utils import ensure_dir, set_global_seeds


# ---------------------------------------------------------------------
# Helpers: loading data and building a simple probabilistic RF
# ---------------------------------------------------------------------


def _load_enriched_df() -> pd.DataFrame:
    """
    Load the enriched feature dataframe produced in Phase B.

    Expected path:
        data_stage/features/merged_with_composites.parquet
    """
    path = DATA_DIR / "features" / "merged_with_composites.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Expected enriched feature file at {path}")

    df = pd.read_parquet(path)
    df["label"] = df["label"].astype(int)
    df["subject"] = df["subject"].astype(str)
    return df


def _build_rf() -> RandomForestClassifier:
    """
    Define a reasonably strong probabilistic model for fairness analysis.

    We reuse a RandomForest with similar capacity to earlier phases,
    but here we care mainly about *probabilities* rather than squeezing
    out every last F1 point.
    """
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
    )


# ---------------------------------------------------------------------
# 1. Fairness / uncertainty proxy
# ---------------------------------------------------------------------


def compute_fairness_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train a cross-validated probabilistic RF and compute, for each sample:
        - predicted probability of the high-risk class (label 2)
        - predicted label (argmax of probabilities)

    Then aggregate *per subject* to obtain:
        - n_windows
        - subject_accuracy
        - mean_prob_class2
        - var_prob_class2 (uncertainty proxy)
        - std_prob_class2
        - mean_true_label (average observed label)

    The result is written to:
        reports/tables/fairness_summary.csv
    """
    set_global_seeds()
    ensure_dir(TABLES_DIR)

    df = _load_enriched_df()

    # Use all features except subject + label
    feature_cols = [c for c in df.columns if c not in ["subject", "label"]]
    X = df[feature_cols].values
    y = df["label"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rf = _build_rf()

    # cross_val_predict with method="predict_proba" gives us an out-of-fold
    # probability estimate for every sample, using only train folds.
    proba = cross_val_predict(
        rf,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
        verbose=0,
    )

    # Labels are {0,1,2}, so column index 2 is "high-risk" probability.
    prob_class2 = proba[:, 2]
    pred_labels = np.argmax(proba, axis=1)

    df = df.copy()
    df["prob_class2"] = prob_class2
    df["pred_label"] = pred_labels

    # Aggregate by subject as a simple fairness / stability proxy.
    grouped = (
        df.groupby("subject")
        .agg(
            n_windows=("label", "size"),
            subject_accuracy=("pred_label", lambda s: float(np.mean(s == df.loc[s.index, "label"]))),
            mean_prob_class2=("prob_class2", "mean"),
            var_prob_class2=("prob_class2", "var"),
            std_prob_class2=("prob_class2", "std"),
            mean_true_label=("label", "mean"),
        )
        .reset_index()
    )

    out_path = TABLES_DIR / "fairness_summary.csv"
    grouped.to_csv(out_path, index=False)
    print(f"[Phase F] Fairness summary written to: {out_path}")

    # Return both the grouped summary and the per-window dataframe with probs
    return grouped, df


# ---------------------------------------------------------------------
# 2. Reject-option band analysis
# ---------------------------------------------------------------------


def compute_reject_stats(
    df_with_probs: pd.DataFrame,
    bands: List[Tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """
    Compute simple reject-option statistics for a set of probability bands.

    For each band [low, high]:

        - 'coverage_full'            : fraction of all windows used (always 1.0)
        - 'error_rate_full'          : 1 - accuracy with no rejection
        - 'coverage_outside_band'    : fraction of windows that remain after
                                       rejecting ambiguous ones
        - 'error_rate_outside_band'  : 1 - accuracy on the *non-rejected* set

    The idea: as we widen the reject band, we expect lower error on the
    accepted windows but lower coverage (fewer automated decisions).

    Results are written to:
        reports/tables/reject_stats.csv
    """
    ensure_dir(TABLES_DIR)

    if bands is None:
        # Default: narrow band around 0.5, plus two slightly wider bands.
        bands = [(0.45, 0.55), (0.40, 0.60), (0.35, 0.65)]

    y_true = df_with_probs["label"].values
    y_pred = df_with_probs["pred_label"].values
    prob = df_with_probs["prob_class2"].values

    base_acc = accuracy_score(y_true, y_pred)
    base_err = 1.0 - base_acc

    rows: List[Dict[str, Any]] = []

    for low, high in bands:
        reject_mask = (prob >= low) & (prob <= high)
        keep_mask = ~reject_mask

        if keep_mask.sum() == 0:
            # Degenerate band: everything rejected.
            cov_keep = 0.0
            err_keep = np.nan
        else:
            acc_keep = accuracy_score(y_true[keep_mask], y_pred[keep_mask])
            err_keep = 1.0 - acc_keep
            cov_keep = float(keep_mask.mean())

        rows.append(
            {
                "band_low": low,
                "band_high": high,
                "coverage_full": 1.0,
                "error_rate_full": float(base_err),
                "coverage_outside_band": cov_keep,
                "error_rate_outside_band": float(err_keep),
            }
        )

    reject_df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "reject_stats.csv"
    reject_df.to_csv(out_path, index=False)
    print(f"[Phase F] Reject-option statistics written to: {out_path}")

    return reject_df


# ---------------------------------------------------------------------
# 3. Packaging and manifest generation
# ---------------------------------------------------------------------


def _try_get_git_hash(root: Path) -> str:
    """
    Best-effort attempt to get the current git commit hash.
    If the project is not a git repo or git is unavailable, return 'unknown'.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def package_for_thesis() -> Dict[str, List[str]]:
    """
    Copy key tables and figures into final “thesis” folders and write
    a manifest.json describing what has been packaged.

    Final locations:
        reports/tables/thesis_final/
        reports/figures/thesis_final/

    Returns a dict of {"tables": [...], "figures": [...]} for logging.
    """
    final_tables_dir = TABLES_DIR / "thesis_final"
    final_figures_dir = FIGURES_DIR / "thesis_final"
    ensure_dir(final_tables_dir)
    ensure_dir(final_figures_dir)

    # Key CSV tables that the thesis will reference directly.
    table_files = [
        "loso_baselines.csv",
        "ensembles_per_fold.csv",
        "calibration.csv",
        "tiers_costs.csv",
        "feature_family_ablation.csv",
        "sensitivity.csv",
        "shap_top_features.csv",
        "fairness_summary.csv",
        "reject_stats.csv",
    ]

    copied_tables: List[str] = []
    for name in table_files:
        src = TABLES_DIR / name
        if src.exists():
            dst = final_tables_dir / name
            shutil.copy2(src, dst)
            copied_tables.append(str(dst.relative_to(final_tables_dir)))
        else:
            print(f"[Phase F] Warning: expected table not found: {src}")

    # Key figures that are useful for the write-up.
    figure_files = [
        "eda_distributions.png",
        "corr_heatmap.png",
        "composites_by_label.png",
        "feature_family_bar.png",
        "calibration_plots.png",
        "roc_pr_curves.png",      # if produced in Phase D
        "shap_global.png",
        "sensitivity_spider.png",
    ]

    copied_figures: List[str] = []
    for name in figure_files:
        src = FIGURES_DIR / name
        if src.exists():
            dst = final_figures_dir / name
            shutil.copy2(src, dst)
            copied_figures.append(str(dst.relative_to(final_figures_dir)))
        else:
            print(f"[Phase F] Note: figure not found (skipped): {src}")

    # Build manifest.json at reports/tables level.
    manifest = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "git_hash": _try_get_git_hash(root=Path(__file__).resolve().parents[1]),
        "final_tables_dir": str(final_tables_dir),
        "final_figures_dir": str(final_figures_dir),
        "tables": copied_tables,
        "figures": copied_figures,
    }

    manifest_path = TABLES_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Phase F] Manifest written to: {manifest_path}")
    return {"tables": copied_tables, "figures": copied_figures}


# ---------------------------------------------------------------------
# Orchestrator: Phase F entrypoint
# ---------------------------------------------------------------------


def phase_F_fairness_and_packaging() -> None:
    """
    High-level Phase F driver.

    1. Compute fairness_summary.csv
    2. Compute reject_stats.csv
    3. Package key artefacts and write manifest.json
    """
    set_global_seeds()

    fairness_df, per_window_df = compute_fairness_summary()
    reject_df = compute_reject_stats(per_window_df)

    packaged = package_for_thesis()

    print("\n[Phase F] Fairness summary (head):")
    print(fairness_df.head())

    print("\n[Phase F] Reject-option statistics:")
    print(reject_df)

    print("\n[Phase F] Packaged artefacts:")
    print(packaged)
