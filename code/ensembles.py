from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import joblib

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
import matplotlib.pyplot as plt

from xgboost import XGBClassifier  # pip install xgboost

from .config import DATA_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR
from .utils import (
    ensure_dir,
    set_global_seeds,
    log_run_metadata,
    append_runlog_md_row,
)
from .baselines import _load_enriched_dataset, _select_feature_columns


# ---------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------
def _compute_macro_pr_auc(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> float:
    pr_aucs: List[float] = []
    for idx, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        precision, recall, _ = precision_recall_curve(y_bin, y_proba[:, idx])
        if len(precision) > 1 and len(recall) > 1:
            pr_aucs.append(auc(recall, precision))
    if not pr_aucs:
        return float("nan")
    return float(np.mean(pr_aucs))


def _evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")

    try:
        metrics["auroc_macro"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro"
        )
    except Exception:
        metrics["auroc_macro"] = float("nan")

    metrics["pr_auc_macro"] = _compute_macro_pr_auc(y_true, y_proba, classes)
    return metrics


def _compute_multiclass_brier(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> float:
    Y_true = label_binarize(y_true, classes=classes)
    if Y_true.shape[1] == 1:  # binary edge case
        Y_true = np.hstack([1 - Y_true, Y_true])
    return float(np.mean(np.sum((Y_true - y_proba) ** 2, axis=1)))


# ---------------------------------------------------------------------
# Feature family ablations
# ---------------------------------------------------------------------
def _define_feature_families(df: pd.DataFrame) -> Dict[str, List[str]]:
    all_features = [c for c in df.columns if c not in {"subject", "label"}]

    core_features = [
        c
        for c in all_features
        if not any(
            kw in c
            for kw in [
                "SRI",
                "RS",
                "PL",
                "smoker",
                "coffee",
                "sport_today",
                "feel_ill",
            ]
        )
    ]

    eda_features = [c for c in all_features if c.startswith("EDA")]
    hrv_features = [c for c in all_features if c.startswith("BVP")]
    composite_cols = [c for c in all_features if c in {"SRI", "RS", "PL"}]
    core_plus_composites = list(sorted(set(core_features + composite_cols)))

    families: Dict[str, List[str]] = {
        "core_stats": core_features,
        "eda_only": eda_features,
        "hrv_only": hrv_features,
        "core_plus_composites": core_plus_composites,
    }
    return families


def _run_loso_for_family(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    subjects = sorted(df["subject"].unique())
    results: List[Dict] = []

    for fold_id, test_subject in enumerate(subjects):
        train_df = df[df["subject"] != test_subject]
        test_df = df[df["subject"] == test_subject]

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        classes = model.classes_

        metrics = _evaluate_predictions(y_test, y_pred, y_proba, classes)
        metrics.update({"fold_id": fold_id, "test_subject": test_subject})
        results.append(metrics)

    df_res = pd.DataFrame(results)
    return df_res


def _run_feature_family_ablations(df: pd.DataFrame) -> pd.DataFrame:
    families = _define_feature_families(df)
    rows: List[Dict] = []

    for family_name, feature_cols in families.items():
        if not feature_cols:
            continue
        family_df = _run_loso_for_family(df, feature_cols)
        family_df["family"] = family_name
        rows.append(family_df)

    if not rows:
        return pd.DataFrame()

    ablations = pd.concat(rows, ignore_index=True)

    mean_rows = (
        ablations.groupby("family")[ ["f1_macro", "auroc_macro", "pr_auc_macro"] ]
        .mean()
        .reset_index()
    )
    mean_rows["fold_id"] = "mean"
    mean_rows["test_subject"] = "ALL"

    ablations = pd.concat([ablations, mean_rows], ignore_index=True)
    return ablations


def _plot_feature_family_bars(ablations: pd.DataFrame, out_path: Path) -> None:
    summary = (
        ablations[ablations["fold_id"] == "mean"].sort_values("f1_macro", ascending=False).reset_index(drop=True)
    )
    families = summary["family"].tolist()
    f1_scores = summary["f1_macro"].tolist()

    plt.figure(figsize=(8, 5))
    plt.bar(families, f1_scores)
    plt.ylabel("Macro F1")
    plt.xlabel("Feature family")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Ensembles + calibration + tiers
# ---------------------------------------------------------------------
def _build_xgb_model(n_classes: int) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    return model


def _run_ensembles_and_calibration(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subjects = sorted(df["subject"].unique())
    ensemble_rows: List[Dict] = []
    calibration_rows: List[Dict] = []
    tier_records: List[Dict] = []

    ensure_dir(MODELS_DIR / "ensembles")

    all_pre_probs_class2: List[float] = []
    all_post_probs_class2: List[float] = []
    all_true_class2: List[int] = []

    for fold_id, test_subject in enumerate(subjects):
        train_df = df[df["subject"] != test_subject]
        test_df = df[df["subject"] == test_subject]

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        n_classes = len(np.unique(y_train))
        xgb = _build_xgb_model(n_classes=n_classes)
        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

        voting = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb)], voting="soft")

        voting.fit(X_train_scaled, y_train)
        y_pred_base = voting.predict(X_test_scaled)
        y_proba_base = voting.predict_proba(X_test_scaled)
        classes = voting.classes_

        base_metrics = _evaluate_predictions(y_test, y_pred_base, y_proba_base, classes)
        base_metrics.update({
            "fold_id": fold_id,
            "test_subject": test_subject,
            "model": "voting_ensemble",
            "stage": "pre_calibration",
        })
        ensemble_rows.append(base_metrics)

        calib = CalibratedClassifierCV(estimator=voting, method="isotonic", cv=3)
        calib.fit(X_train_scaled, y_train)

        y_pred_cal = calib.predict(X_test_scaled)
        y_proba_cal = calib.predict_proba(X_test_scaled)

        cal_metrics = _evaluate_predictions(y_test, y_pred_cal, y_proba_cal, classes)
        cal_metrics.update({
            "fold_id": fold_id,
            "test_subject": test_subject,
            "model": "voting_ensemble",
            "stage": "post_calibration",
        })
        ensemble_rows.append(cal_metrics)

        brier_pre = _compute_multiclass_brier(y_test, y_proba_base, classes)
        brier_post = _compute_multiclass_brier(y_test, y_proba_cal, classes)

        calibration_rows.append({
            "fold_id": fold_id,
            "test_subject": test_subject,
            "model": "voting_ensemble",
            "brier_pre": brier_pre,
            "brier_post": brier_post,
        })

        if 2 in classes:
            idx_high = list(classes).index(2)
            prob_high_pre = y_proba_base[:, idx_high]
            prob_high_post = y_proba_cal[:, idx_high]
            all_pre_probs_class2.extend(prob_high_pre.tolist())
            all_post_probs_class2.extend(prob_high_post.tolist())
            all_true_class2.extend((y_test == 2).astype(int).tolist())

            for p, y_true in zip(prob_high_post, y_test):
                if p < 0.33:
                    tier = "low"
                elif p < 0.66:
                    tier = "medium"
                else:
                    tier = "high"

                tier_records.append({
                    "fold_id": fold_id,
                    "test_subject": test_subject,
                    "true_label": int(y_true),
                    "p_high_calibrated": float(p),
                    "tier": tier,
                })

        model_path = MODELS_DIR / "ensembles" / f"ensemble_fold_{fold_id}.pkl"
        joblib.dump(calib, model_path)

    calibration_df = pd.DataFrame(calibration_rows)
    tiers_df = pd.DataFrame(tier_records)

    if not tiers_df.empty:
        cost_map = {0: 1.0, 1: 2.0, 2: 5.0}
        tier_stats = []
        for tier_name, group in tiers_df.groupby("tier"):
            n = len(group)
            counts = group["true_label"].value_counts().to_dict()
            n0 = counts.get(0, 0)
            n1 = counts.get(1, 0)
            n2 = counts.get(2, 0)
            mean_prob = float(group["p_high_calibrated"].mean())
            expected_cost = (n0 * cost_map[0] + n1 * cost_map[1] + n2 * cost_map[2]) / max(n, 1)

            tier_stats.append({
                "tier": tier_name,
                "n_samples": n,
                "n_label0": n0,
                "n_label1": n1,
                "n_label2": n2,
                "mean_p_high": mean_prob,
                "expected_cost": expected_cost,
            })
        tiers_summary = pd.DataFrame(tier_stats)
    else:
        tiers_summary = pd.DataFrame()

    ensembles_df = pd.DataFrame(ensemble_rows)
    if not ensembles_df.empty:
        mean_rows = (
            ensembles_df.groupby(["model", "stage"])[["f1_macro", "auroc_macro", "pr_auc_macro"]]
            .mean()
            .reset_index()
        )
        mean_rows["fold_id"] = "mean"
        mean_rows["test_subject"] = "ALL"
        ensembles_df = pd.concat([ensembles_df, mean_rows], ignore_index=True)

    if all_true_class2 and all_pre_probs_class2 and all_post_probs_class2:
        from sklearn.calibration import calibration_curve

        y_true_bin = np.array(all_true_class2)
        probs_pre = np.array(all_pre_probs_class2)
        probs_post = np.array(all_post_probs_class2)

        frac_pos_pre, mean_pred_pre = calibration_curve(y_true_bin, probs_pre, n_bins=10)
        frac_pos_post, mean_pred_post = calibration_curve(y_true_bin, probs_post, n_bins=10)

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(mean_pred_pre, frac_pos_pre, marker="o", label="Pre-calibration")
        plt.plot(mean_pred_post, frac_pos_post, marker="s", label="Post-calibration")
        plt.xlabel("Predicted probability (class 2)")
        plt.ylabel("Observed frequency (class 2)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "calibration_plots.png", dpi=300)
        plt.close()

    return ensembles_df, calibration_df, tiers_summary


def run_phase_D() -> None:
    set_global_seeds()
    ensure_dir(TABLES_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(MODELS_DIR)

    df = _load_enriched_dataset()
    df["subject"] = df["subject"].astype(str)
    df["label"] = df["label"].astype(int)
    all_feature_cols = _select_feature_columns(df)

    ablations = _run_feature_family_ablations(df)
    ablation_path = TABLES_DIR / "feature_family_ablation.csv"
    ablations.to_csv(ablation_path, index=False)

    _plot_feature_family_bars(ablations, FIGURES_DIR / "feature_family_bar.png")

    families = _define_feature_families(df)
    best_family_cols = families.get("core_plus_composites", all_feature_cols)

    ensembles_df, calibration_df, tiers_summary = _run_ensembles_and_calibration(df, best_family_cols)

    ensembles_path = TABLES_DIR / "ensembles_per_fold.csv"
    calibration_path = TABLES_DIR / "calibration.csv"
    tiers_path = TABLES_DIR / "tiers_costs.csv"

    ensembles_df.to_csv(ensembles_path, index=False)
    calibration_df.to_csv(calibration_path, index=False)
    tiers_summary.to_csv(tiers_path, index=False)

    details = {
        "feature_family_ablation": str(ablation_path),
        "ensembles_per_fold": str(ensembles_path),
        "calibration_csv": str(calibration_path),
        "tiers_costs_csv": str(tiers_path),
    }

    json_path = log_run_metadata(
        phase_name="phase_D_ensembles_calibration_tiers",
        status="success",
        details=details,
    )

    append_runlog_md_row(
        timestamp_utc=json_path.name.split("_")[-1].replace(".json", ""),
        phase_name="phase_D_ensembles_calibration_tiers",
        status="success",
        notes=(
            "Feature ablations, voting ensemble (RF + XGBoost), calibration "
            "and risk tiers completed."
        ),
    )

    print(f"Phase D complete. Logged: {json_path}")
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import joblib

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
import matplotlib.pyplot as plt

from xgboost import XGBClassifier  # pip install xgboost

from .config import DATA_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR
from .utils import (
    ensure_dir,
    set_global_seeds,
    log_run_metadata,
    append_runlog_md_row,
)
from .baselines import _load_enriched_dataset, _select_feature_columns


# ---------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------
def _compute_macro_pr_auc(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> float:
    pr_aucs: List[float] = []
    for idx, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        precision, recall, _ = precision_recall_curve(y_bin, y_proba[:, idx])
        if len(precision) > 1 and len(recall) > 1:
            pr_aucs.append(auc(recall, precision))
    if not pr_aucs:
        return float("nan")
    return float(np.mean(pr_aucs))


def _evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")

    try:
        metrics["auroc_macro"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro"
        )
    except Exception:
        metrics["auroc_macro"] = float("nan")

    metrics["pr_auc_macro"] = _compute_macro_pr_auc(y_true, y_proba, classes)
    return metrics


def _compute_multiclass_brier(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> float:
    Y_true = label_binarize(y_true, classes=classes)
    if Y_true.shape[1] == 1:  # binary edge case
        Y_true = np.hstack([1 - Y_true, Y_true])
    return float(np.mean(np.sum((Y_true - y_proba) ** 2, axis=1)))


# ---------------------------------------------------------------------
# Feature family ablations
# ---------------------------------------------------------------------
def _define_feature_families(df: pd.DataFrame) -> Dict[str, List[str]]:
    all_features = [c for c in df.columns if c not in {"subject", "label"}]

    core_features = [
        c
        for c in all_features
        if not any(
            kw in c
            for kw in [
                "SRI",
                "RS",
                "PL",
                "smoker",
                "coffee",
                "sport_today",
                "feel_ill",
            ]
        )
    ]

    eda_features = [c for c in all_features if c.startswith("EDA")]
    hrv_features = [c for c in all_features if c.startswith("BVP")]
    composite_cols = [c for c in all_features if c in {"SRI", "RS", "PL"}]
    core_plus_composites = list(sorted(set(core_features + composite_cols)))

    families: Dict[str, List[str]] = {
        "core_stats": core_features,
        "eda_only": eda_features,
        "hrv_only": hrv_features,
        "core_plus_composites": core_plus_composites,
    }
    return families


def _run_loso_for_family(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    subjects = sorted(df["subject"].unique())
    results: List[Dict] = []

    for fold_id, test_subject in enumerate(subjects):
        train_df = df[df["subject"] != test_subject]
        test_df = df[df["subject"] == test_subject]

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        classes = model.classes_

        metrics = _evaluate_predictions(y_test, y_pred, y_proba, classes)
        metrics.update({"fold_id": fold_id, "test_subject": test_subject})
        results.append(metrics)

    df_res = pd.DataFrame(results)
    return df_res


def _run_feature_family_ablations(df: pd.DataFrame) -> pd.DataFrame:
    families = _define_feature_families(df)
    rows: List[Dict] = []

    for family_name, feature_cols in families.items():
        if not feature_cols:
            continue
        family_df = _run_loso_for_family(df, feature_cols)
        family_df["family"] = family_name
        rows.append(family_df)

    if not rows:
        return pd.DataFrame()

    ablations = pd.concat(rows, ignore_index=True)

    mean_rows = (
        ablations.groupby("family")[ ["f1_macro", "auroc_macro", "pr_auc_macro"] ]
        .mean()
        .reset_index()
    )
    mean_rows["fold_id"] = "mean"
    mean_rows["test_subject"] = "ALL"

    ablations = pd.concat([ablations, mean_rows], ignore_index=True)
    return ablations


def _plot_feature_family_bars(ablations: pd.DataFrame, out_path: Path) -> None:
    summary = (
        ablations[ablations["fold_id"] == "mean"].sort_values("f1_macro", ascending=False).reset_index(drop=True)
    )
    families = summary["family"].tolist()
    f1_scores = summary["f1_macro"].tolist()

    plt.figure(figsize=(8, 5))
    plt.bar(families, f1_scores)
    plt.ylabel("Macro F1")
    plt.xlabel("Feature family")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Ensembles + calibration + tiers
# ---------------------------------------------------------------------
def _build_xgb_model(n_classes: int) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    return model


def _run_ensembles_and_calibration(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subjects = sorted(df["subject"].unique())
    ensemble_rows: List[Dict] = []
    calibration_rows: List[Dict] = []
    tier_records: List[Dict] = []

    ensure_dir(MODELS_DIR / "ensembles")

    all_pre_probs_class2: List[float] = []
    all_post_probs_class2: List[float] = []
    all_true_class2: List[int] = []

    for fold_id, test_subject in enumerate(subjects):
        train_df = df[df["subject"] != test_subject]
        test_df = df[df["subject"] == test_subject]

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        n_classes = len(np.unique(y_train))
        xgb = _build_xgb_model(n_classes=n_classes)
        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

        voting = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb)], voting="soft")

        voting.fit(X_train_scaled, y_train)
        y_pred_base = voting.predict(X_test_scaled)
        y_proba_base = voting.predict_proba(X_test_scaled)
        classes = voting.classes_

        base_metrics = _evaluate_predictions(y_test, y_pred_base, y_proba_base, classes)
        base_metrics.update({
            "fold_id": fold_id,
            "test_subject": test_subject,
            "model": "voting_ensemble",
            "stage": "pre_calibration",
        })
        ensemble_rows.append(base_metrics)

        calib = CalibratedClassifierCV(estimator=voting, method="isotonic", cv=3)
        calib.fit(X_train_scaled, y_train)

        y_pred_cal = calib.predict(X_test_scaled)
        y_proba_cal = calib.predict_proba(X_test_scaled)

        cal_metrics = _evaluate_predictions(y_test, y_pred_cal, y_proba_cal, classes)
        cal_metrics.update({
            "fold_id": fold_id,
            "test_subject": test_subject,
            "model": "voting_ensemble",
            "stage": "post_calibration",
        })
        ensemble_rows.append(cal_metrics)

        brier_pre = _compute_multiclass_brier(y_test, y_proba_base, classes)
        brier_post = _compute_multiclass_brier(y_test, y_proba_cal, classes)

        calibration_rows.append({
            "fold_id": fold_id,
            "test_subject": test_subject,
            "model": "voting_ensemble",
            "brier_pre": brier_pre,
            "brier_post": brier_post,
        })

        if 2 in classes:
            idx_high = list(classes).index(2)
            prob_high_pre = y_proba_base[:, idx_high]
            prob_high_post = y_proba_cal[:, idx_high]
            all_pre_probs_class2.extend(prob_high_pre.tolist())
            all_post_probs_class2.extend(prob_high_post.tolist())
            all_true_class2.extend((y_test == 2).astype(int).tolist())

            for p, y_true in zip(prob_high_post, y_test):
                if p < 0.33:
                    tier = "low"
                elif p < 0.66:
                    tier = "medium"
                else:
                    tier = "high"

                tier_records.append({
                    "fold_id": fold_id,
                    "test_subject": test_subject,
                    "true_label": int(y_true),
                    "p_high_calibrated": float(p),
                    "tier": tier,
                })

        model_path = MODELS_DIR / "ensembles" / f"ensemble_fold_{fold_id}.pkl"
        joblib.dump(calib, model_path)

    calibration_df = pd.DataFrame(calibration_rows)
    tiers_df = pd.DataFrame(tier_records)

    if not tiers_df.empty:
        cost_map = {0: 1.0, 1: 2.0, 2: 5.0}
        tier_stats = []
        for tier_name, group in tiers_df.groupby("tier"):
            n = len(group)
            counts = group["true_label"].value_counts().to_dict()
            n0 = counts.get(0, 0)
            n1 = counts.get(1, 0)
            n2 = counts.get(2, 0)
            mean_prob = float(group["p_high_calibrated"].mean())
            expected_cost = (n0 * cost_map[0] + n1 * cost_map[1] + n2 * cost_map[2]) / max(n, 1)

            tier_stats.append({
                "tier": tier_name,
                "n_samples": n,
                "n_label0": n0,
                "n_label1": n1,
                "n_label2": n2,
                "mean_p_high": mean_prob,
                "expected_cost": expected_cost,
            })
        tiers_summary = pd.DataFrame(tier_stats)
    else:
        tiers_summary = pd.DataFrame()

    ensembles_df = pd.DataFrame(ensemble_rows)
    if not ensembles_df.empty:
        mean_rows = (
            ensembles_df.groupby(["model", "stage"])[["f1_macro", "auroc_macro", "pr_auc_macro"]]
            .mean()
            .reset_index()
        )
        mean_rows["fold_id"] = "mean"
        mean_rows["test_subject"] = "ALL"
        ensembles_df = pd.concat([ensembles_df, mean_rows], ignore_index=True)

    if all_true_class2 and all_pre_probs_class2 and all_post_probs_class2:
        from sklearn.calibration import calibration_curve

        y_true_bin = np.array(all_true_class2)
        probs_pre = np.array(all_pre_probs_class2)
        probs_post = np.array(all_post_probs_class2)

        frac_pos_pre, mean_pred_pre = calibration_curve(y_true_bin, probs_pre, n_bins=10)
        frac_pos_post, mean_pred_post = calibration_curve(y_true_bin, probs_post, n_bins=10)

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(mean_pred_pre, frac_pos_pre, marker="o", label="Pre-calibration")
        plt.plot(mean_pred_post, frac_pos_post, marker="s", label="Post-calibration")
        plt.xlabel("Predicted probability (class 2)")
        plt.ylabel("Observed frequency (class 2)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "calibration_plots.png", dpi=300)
        plt.close()

    return ensembles_df, calibration_df, tiers_summary


def run_phase_D() -> None:
    set_global_seeds()
    ensure_dir(TABLES_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(MODELS_DIR)

    df = _load_enriched_dataset()
    df["subject"] = df["subject"].astype(str)
    df["label"] = df["label"].astype(int)
    all_feature_cols = _select_feature_columns(df)

    ablations = _run_feature_family_ablations(df)
    ablation_path = TABLES_DIR / "feature_family_ablation.csv"
    ablations.to_csv(ablation_path, index=False)

    _plot_feature_family_bars(ablations, FIGURES_DIR / "feature_family_bar.png")

    families = _define_feature_families(df)
    best_family_cols = families.get("core_plus_composites", all_feature_cols)

    ensembles_df, calibration_df, tiers_summary = _run_ensembles_and_calibration(df, best_family_cols)

    ensembles_path = TABLES_DIR / "ensembles_per_fold.csv"
    calibration_path = TABLES_DIR / "calibration.csv"
    tiers_path = TABLES_DIR / "tiers_costs.csv"

    ensembles_df.to_csv(ensembles_path, index=False)
    calibration_df.to_csv(calibration_path, index=False)
    tiers_summary.to_csv(tiers_path, index=False)

    details = {
        "feature_family_ablation": str(ablation_path),
        "ensembles_per_fold": str(ensembles_path),
        "calibration_csv": str(calibration_path),
        "tiers_costs_csv": str(tiers_path),
    }

    json_path = log_run_metadata(
        phase_name="phase_D_ensembles_calibration_tiers",
        status="success",
        details=details,
    )

    append_runlog_md_row(
        timestamp_utc=json_path.name.split("_")[-1].replace(".json", ""),
        phase_name="phase_D_ensembles_calibration_tiers",
        status="success",
        notes=(
            "Feature ablations, voting ensemble (RF + XGBoost), calibration "
            "and risk tiers completed."
        ),
    )

    print(f"Phase D complete. Logged: {json_path}")
# ---------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import joblib

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve,
)
import matplotlib.pyplot as plt

from xgboost import XGBClassifier  # pip install xgboost

from .config import DATA_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR
from .utils import (
    ensure_dir,
    set_global_seeds,
    log_run_metadata,
    append_runlog_md_row,
)
from .baselines import _load_enriched_dataset, _select_feature_columns


# ---------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------
def _compute_macro_pr_auc(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> float:
    """
    To compute macro-averaged PR-AUC across classes, so that precision–
    recall performance is summarised in a way that is robust to class
    imbalance for the purposes of ensemble comparison in this project.
    """
    pr_aucs: List[float] = []
    for idx, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        precision, recall, _ = precision_recall_curve(y_bin, y_proba[:, idx])
        if len(precision) > 1 and len(recall) > 1:
            pr_aucs.append(auc(recall, precision))
    if not pr_aucs:
        return float("nan")
    return float(np.mean(pr_aucs))

def _evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, float]:
    """
    To derive F1, AUROC and PR-AUC metrics from predictions and
    probabilities, so that model performance is summarised using
    complementary views of classification quality in this project.
    """
    metrics: Dict[str, float] = {}
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")

    try:
        metrics["auroc_macro"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro"
        )
    except ValueError:
        metrics["auroc_macro"] = float("nan")

    metrics["pr_auc_macro"] = _compute_macro_pr_auc(y_true, y_proba, classes)
    return metrics

def _compute_multiclass_brier(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> float:
    """
    To compute a multiclass Brier score using one-hot encoded labels,
    so that calibration quality can be compared before and after
    isotonic correction for the purposes of risk tiering in this project.
    """
    Y_true = label_binarize(y_true, classes=classes)
    if Y_true.shape[1] == 1:  # binary edge case
        Y_true = np.hstack([1 - Y_true, Y_true])
    return float(np.mean(np.sum((Y_true - y_proba) ** 2, axis=1)))

# ---------------------------------------------------------------------
# Feature family ablations
# ---------------------------------------------------------------------
def _define_feature_families(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    To partition the full feature set into interpretable families,
    so that ablation experiments can isolate the incremental value
    of EDA-only features, HRV surrogates, and composite indices in
    this project.
    """
    all_features = [c for c in df.columns if c not in {"subject", "label"}]

    # Core stats: broad but excluding obviously derived composites
    core_features = [
        c
        for c in all_features
        if not any(
            kw in c
            for kw in [
                "SRI",
                "RS",
                "PL",
                "smoker",
                "coffee",
                "sport_today",
                "feel_ill",
            ]
        )
    ]

    # EDA-only decomposition
    eda_features = [c for c in all_features if c.startswith("EDA")]

    # HRV surrogates and BVP-based features
    hrv_features = [c for c in all_features if c.startswith("BVP")]

    # Core + composites (SRI, RS, PL) as primary candidate for best family
    composite_cols = [c for c in all_features if c in {"SRI", "RS", "PL"}]
    core_plus_composites = list(sorted(set(core_features + composite_cols)))

    families: Dict[str, List[str]] = {
        "core_stats": core_features,
        "eda_only": eda_features,
        "hrv_only": hrv_features,
        "core_plus_composites": core_plus_composites,
    }
    return families

def _run_loso_for_family(
    df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """
    To run a Random Forest baseline under LOSO for a given feature
    family, so that relative performance contributions of each family
    can be quantified for the purposes of ablation analysis.
    """
    subjects = sorted(df["subject"].unique())
    results: List[Dict] = []

    for fold_id, test_subject in enumerate(subjects):
        train_df = df[df["subject"] != test_subject]
        test_df = df[df["subject"] == test_subject]

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        classes = model.classes_

        metrics = _evaluate_predictions(y_test, y_pred, y_proba, classes)
        metrics.update(
            {
                "fold_id": fold_id,
                "test_subject": test_subject,
            }
        )
        results.append(metrics)

    df_res = pd.DataFrame(results)
    return df_res

def _run_feature_family_ablations(df: pd.DataFrame) -> pd.DataFrame:
    """
    To execute ablation experiments across predefined feature families,
    so that the incremental value of composite indices (SRI, RS, PL)
    can be empirically demonstrated for this project.
    """
    families = _define_feature_families(df)
    rows: List[Dict] = []

    for family_name, feature_cols in families.items():
        if not feature_cols:
            continue
        family_df = _run_loso_for_family(df, feature_cols)
        family_df["family"] = family_name
        rows.append(family_df)

    if not rows:
        return pd.DataFrame()

    ablations = pd.concat(rows, ignore_index=True)

    # Add mean metrics per family for quick summary
    mean_rows = (
        ablations.groupby("family")[["f1_macro", "auroc_macro", "pr_auc_macro"]]
        .mean()
        .reset_index()
    )
    mean_rows["fold_id"] = "mean"
    mean_rows["test_subject"] = "ALL"

    ablations = pd.concat([ablations, mean_rows], ignore_index=True)
    return ablations

def _plot_feature_family_bars(ablations: pd.DataFrame, out_path: Path) -> None:
    """
    To visualise mean F1 performance across feature families, so that
    the effect of adding composite indices can be inspected at a glance
    for the purposes of motivating ensemble design in this project.
    """
    summary = (
        ablations[ablations["fold_id"] == "mean"]
        .sort_values("f1_macro", ascending=False)
        .reset_index(drop=True)
    )
    families = summary["family"].tolist()
    f1_scores = summary["f1_macro"].tolist()

    plt.figure(figsize=(8, 5))
    plt.bar(families, f1_scores)
    plt.ylabel("Macro F1")
    plt.xlabel("Feature family")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# ---------------------------------------------------------------------
# Ensembles + calibration + tiers
# ---------------------------------------------------------------------
def _build_xgb_model(n_classes: int) -> XGBClassifier:
    """
    To construct an XGBoost classifier with sensible defaults for
    multiclass physiological data, so that the ensemble uses a strong
    gradient-boosted tree component in this project.
    """
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    return model

def _run_ensembles_and_calibration(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    To train XGBoost + Random Forest ensembles under LOSO, calibrate
    their probabilities, and derive tiered risk estimates, so that the
    main predictive and calibration results of the project are produced.
    """
    subjects = sorted(df["subject"].unique())
    ensemble_rows: List[Dict] = []
    calibration_rows: List[Dict] = []
    tier_records: List[Dict] = []

    ensure_dir(MODELS_DIR / "ensembles")

    all_pre_probs_class2: List[float] = []
    all_post_probs_class2: List[float] = []
    all_true_class2: List[int] = []

    for fold_id, test_subject in enumerate(subjects):
        train_df = df[df["subject"] != test_subject]
        test_df = df[df["subject"] == test_subject]

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Base models
        n_classes = len(np.unique(y_train))
        xgb = _build_xgb_model(n_classes=n_classes)
        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

        # Soft voting ensemble combining Random Forest + XGBoost
        voting = VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb)],
            voting="soft",
        )

        # Fit ensemble on scaled training data
        voting.fit(X_train_scaled, y_train)
        y_pred_base = voting.predict(X_test_scaled)
        y_proba_base = voting.predict_proba(X_test_scaled)
        classes = voting.classes_

        # Metrics before calibration
        base_metrics = _evaluate_predictions(
            y_test, y_pred_base, y_proba_base, classes
        )
        base_metrics.update(
            {
                "fold_id": fold_id,
                "test_subject": test_subject,
                "model": "voting_ensemble",
                "stage": "pre_calibration",
            }
        )
        ensemble_rows.append(base_metrics)

        # Calibration using isotonic regression on training data
        calib = CalibratedClassifierCV(
            estimator=voting,
            method="isotonic",
            cv=3,
        )
        calib.fit(X_train_scaled, y_train)

        y_pred_cal = calib.predict(X_test_scaled)
        y_proba_cal = calib.predict_proba(X_test_scaled)

        # Metrics after calibration
        cal_metrics = _evaluate_predictions(
            y_test, y_pred_cal, y_proba_cal, classes
        )
        cal_metrics.update(
            {
                "fold_id": fold_id,
                "test_subject": test_subject,
                "model": "voting_ensemble",
                "stage": "post_calibration",
            }
        )
        ensemble_rows.append(cal_metrics)

        # Brier scores for calibration quality
        brier_pre = _compute_multiclass_brier(y_test, y_proba_base, classes)
        brier_post = _compute_multiclass_brier(y_test, y_proba_cal, classes)

        calibration_rows.append(
            {
                "fold_id": fold_id,
                "test_subject": test_subject,
                "model": "voting_ensemble",
                "brier_pre": brier_pre,
                "brier_post": brier_post,
            }
        )

        # Collect probabilities for high-risk class (assumed label = 2)
        if 2 in classes:
            idx_high = list(classes).index(2)
            prob_high_pre = y_proba_base[:, idx_high]
            prob_high_post = y_proba_cal[:, idx_high]
            all_pre_probs_class2.extend(prob_high_pre.tolist())
            all_post_probs_class2.extend(prob_high_post.tolist())
            all_true_class2.extend((y_test == 2).astype(int).tolist())

            # Tier assignment based on calibrated probabilities
            for p, y_true in zip(prob_high_post, y_test):
                if p < 0.33:
                    tier = "low"
                elif p < 0.66:
                    tier = "medium"
                else:
                    tier = "high"

                tier_records.append(
                    {
                        "fold_id": fold_id,
                        "test_subject": test_subject,
                        "true_label": int(y_true),
                        "p_high_calibrated": float(p),
                        "tier": tier,
                    }
                )

        # Save calibrated ensemble model per fold
        model_path = MODELS_DIR / "ensembles" / f"ensemble_fold_{fold_id}.pkl"
        joblib.dump(calib, model_path)

    # Build calibration summary and tier statistics
    calibration_df = pd.DataFrame(calibration_rows)
    tiers_df = pd.DataFrame(tier_records)

    # Aggregate tier-level expected “costs” using a simple cost scheme
    if not tiers_df.empty:
        cost_map = {0: 1.0, 1: 2.0, 2: 5.0}
        tier_stats = []
        for tier_name, group in tiers_df.groupby("tier"):
            n = len(group)
            counts = group["true_label"].value_counts().to_dict()
            n0 = counts.get(0, 0)
            n1 = counts.get(1, 0)
            n2 = counts.get(2, 0)
            mean_prob = float(group["p_high_calibrated"].mean())
            expected_cost = (
                n0 * cost_map[0] + n1 * cost_map[1] + n2 * cost_map[2]
            ) / max(n, 1)

            tier_stats.append(
                {
                    "tier": tier_name,
                    "n_samples": n,
                    "n_label0": n0,
                    "n_label1": n1,
                    "n_label2": n2,
                    "mean_p_high": mean_prob,
                    "expected_cost": expected_cost,
                }
            )
        tiers_summary = pd.DataFrame(tier_stats)
    else:
        tiers_summary = pd.DataFrame()

    # Build ensemble metrics DataFrame and add mean rows
    ensembles_df = pd.DataFrame(ensemble_rows)
    if not ensembles_df.empty:
        mean_rows = (
            ensembles_df.groupby(["model", "stage"])[
                ["f1_macro", "auroc_macro", "pr_auc_macro"]
            ]
            .mean()
            .reset_index()
        )
        mean_rows["fold_id"] = "mean"
        mean_rows["test_subject"] = "ALL"
        ensembles_df = pd.concat([ensembles_df, mean_rows], ignore_index=True)

    # Reliability curves for high-risk class probabilities
    if all_true_class2 and all_pre_probs_class2 and all_post_probs_class2:
        from sklearn.calibration import calibration_curve

        y_true_bin = np.array(all_true_class2)
        probs_pre = np.array(all_pre_probs_class2)
        probs_post = np.array(all_post_probs_class2)

        frac_pos_pre, mean_pred_pre = calibration_curve(
            y_true_bin, probs_pre, n_bins=10
        )
        frac_pos_post, mean_pred_post = calibration_curve(
            y_true_bin, probs_post, n_bins=10
        )

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(mean_pred_pre, frac_pos_pre, marker="o", label="Pre-calibration")
        plt.plot(mean_pred_post, frac_pos_post, marker="s", label="Post-calibration")
        plt.xlabel("Predicted probability (class 2)")
        plt.ylabel("Observed frequency (class 2)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "calibration_plots.png", dpi=300)
        plt.close()

    return ensembles_df, calibration_df, tiers_summary

# ---------------------------------------------------------------------
# Public entry point for Phase D
# ---------------------------------------------------------------------
def run_phase_D() -> None:
    """
    To perform feature-family ablations, train ensemble models, apply
    probability calibration and derive risk tiers, so that the main
    predictive and calibration results required for this thesis are
    generated in a single, reproducible phase.
    """
    set_global_seeds()
    ensure_dir(TABLES_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(MODELS_DIR)

    # 1. Load enriched dataset and feature set
    df = _load_enriched_dataset()
    df["subject"] = df["subject"].astype(str)
    df["label"] = df["label"].astype(int)
    all_feature_cols = _select_feature_columns(df)

    # 2. Feature family ablations with Random Forest baseline
    ablations = _run_feature_family_ablations(df)
    ablation_path = TABLES_DIR / "feature_family_ablation.csv"
    ablations.to_csv(ablation_path, index=False)

    _plot_feature_family_bars(
        ablations, FIGURES_DIR / "feature_family_bar.png"
    )

    # For ensembles, use the core_plus_composites family
    families = _define_feature_families(df)
    best_family_cols = families.get("core_plus_composites", all_feature_cols)

    # 3. Ensembles + calibration + tiers on best feature family
    ensembles_df, calibration_df, tiers_summary = _run_ensembles_and_calibration(
        df, best_family_cols
    )

    ensembles_path = TABLES_DIR / "ensembles_per_fold.csv"
    calibration_path = TABLES_DIR / "calibration.csv"
    tiers_path = TABLES_DIR / "tiers_costs.csv"

    ensembles_df.to_csv(ensembles_path, index=False)
    calibration_df.to_csv(calibration_path, index=False)
    tiers_summary.to_csv(tiers_path, index=False)

    # 4. Log run metadata
    details = {
        "feature_family_ablation": str(ablation_path),
        "ensembles_per_fold": str(ensembles_path),
        "calibration_csv": str(calibration_path),
        "tiers_costs_csv": str(tiers_path),
    }

    json_path = log_run_metadata(
        phase_name="phase_D_ensembles_calibration_tiers",
        status="success",
        details=details,
    )

    append_runlog_md_row(
        timestamp_utc=json_path.name.split("_")[-1].replace(".json", ""),
        phase_name="phase_D_ensembles_calibration_tiers",
        status="success",
        notes=(
            "Feature ablations, voting ensemble (RF + XGBoost), calibration "
            "and risk tiers completed."
        ),
    )

    print(f"Phase D complete. Logged: {json_path}")
# ============================================
# file: code/ensembles.py
# ============================================
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import joblib

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve,
)
import matplotlib.pyplot as plt

from xgboost import XGBClassifier  # pip install xgboost

from .config import DATA_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR
from .utils import (
    ensure_dir,
    set_global_seeds,
    log_run_metadata,
    append_runlog_md_row,
)
from .baselines import _load_enriched_dataset, _select_feature_columns


# ---------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------
def _compute_macro_pr_auc(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> float:
    """
    To compute macro-averaged PR-AUC across classes, so that precision–
    recall performance is summarised in a way that is robust to class
    imbalance for the purposes of ensemble comparison in this project.
    """
    pr_aucs: List[float] = []
    for idx, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        precision, recall, _ = precision_recall_curve(y_bin, y_proba[:, idx])
        if len(precision) > 1 and len(recall) > 1:
            pr_aucs.append(auc(recall, precision))
    if not pr_aucs:
        return float("nan")
    return float(np.mean(pr_aucs))


def _evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, float]:
    """
    To derive F1, AUROC and PR-AUC metrics from predictions and
    probabilities, so that model performance is summarised using
    complementary views of classification quality in this project.
    """
    metrics: Dict[str, float] = {}
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")

    try:
        metrics["auroc_macro"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro"
        )
    except ValueError:
        metrics["auroc_macro"] = float("nan")

    metrics["pr_auc_macro"] = _compute_macro_pr_auc(y_true, y_proba, classes)
    return metrics


def _compute_multiclass_brier(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> float:
    """
    To compute a multiclass Brier score using one-hot encoded labels,
    so that calibration quality can be compared before and after
    isotonic correction for the purposes of risk tiering in this project.
    """
    Y_true = label_binarize(y_true, classes=classes)
    if Y_true.shape[1] == 1:  # binary edge case
        Y_true = np.hstack([1 - Y_true, Y_true])
    return float(np.mean(np.sum((Y_true - y_proba) ** 2, axis=1)))


# ---------------------------------------------------------------------
# Feature family ablations
# ---------------------------------------------------------------------
def _define_feature_families(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    To partition the full feature set into interpretable families,
    so that ablation experiments can isolate the incremental value
    of EDA-only features, HRV surrogates, and composite indices in
    this project.
    """
    all_features = [c for c in df.columns if c not in {"subject", "label"}]

    # Core stats: broad but excluding obviously derived composites
    core_features = [
        c
        for c in all_features
        if not any(
            kw in c
            for kw in [
                "SRI",
                "RS",
                "PL",
                "smoker",
                "coffee",
                "sport_today",
                "feel_ill",
            ]
        )
    ]

    # EDA-only decomposition
    eda_features = [c for c in all_features if c.startswith("EDA")]

    # HRV surrogates and BVP-based features
    hrv_features = [c for c in all_features if c.startswith("BVP")]

    # Core + composites (SRI, RS, PL) as primary candidate for best family
    composite_cols = [c for c in all_features if c in {"SRI", "RS", "PL"}]
    core_plus_composites = list(sorted(set(core_features + composite_cols)))

    families: Dict[str, List[str]] = {
        "core_stats": core_features,
        "eda_only": eda_features,
        "hrv_only": hrv_features,
        "core_plus_composites": core_plus_composites,
    }
    return families


def _run_loso_for_family(
    df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """
    To run a Random Forest baseline under LOSO for a given feature
    family, so that relative performance contributions of each family
    can be quantified for the purposes of ablation analysis.
    """
    subjects = sorted(df["subject"].unique())
    results: List[Dict] = []

    for fold_id, test_subject in enumerate(subjects):
        train_df = df[df["subject"] != test_subject]
        test_df = df[df["subject"] == test_subject]

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        classes = model.classes_

        metrics = _evaluate_predictions(y_test, y_pred, y_proba, classes)
        metrics.update(
            {
                "fold_id": fold_id,
                "test_subject": test_subject,
            }
        )
        results.append(metrics)

    df_res = pd.DataFrame(results)
    return df_res


def _run_feature_family_ablations(df: pd.DataFrame) -> pd.DataFrame:
    """
    To execute ablation experiments across predefined feature families,
    so that the incremental value of composite indices (SRI, RS, PL)
    can be empirically demonstrated for this project.
    """
    families = _define_feature_families(df)
    rows: List[Dict] = []

    for family_name, feature_cols in families.items():
        if not feature_cols:
            continue
        family_df = _run_loso_for_family(df, feature_cols)
        family_df["family"] = family_name
        rows.append(family_df)

    if not rows:
        return pd.DataFrame()

    ablations = pd.concat(rows, ignore_index=True)

    # Add mean metrics per family for quick summary
    mean_rows = (
        ablations.groupby("family")[["f1_macro", "auroc_macro", "pr_auc_macro"]]
        .mean()
        .reset_index()
    )
    mean_rows["fold_id"] = "mean"
    mean_rows["test_subject"] = "ALL"

    ablations = pd.concat([ablations, mean_rows], ignore_index=True)
    return ablations


def _plot_feature_family_bars(ablations: pd.DataFrame, out_path: Path) -> None:
    """
    To visualise mean F1 performance across feature families, so that
    the effect of adding composite indices can be inspected at a glance
    for the purposes of motivating ensemble design in this project.
    """
    summary = (
        ablations[ablations["fold_id"] == "mean"]
        .sort_values("f1_macro", ascending=False)
        .reset_index(drop=True)
    )
    families = summary["family"].tolist()
    f1_scores = summary["f1_macro"].tolist()

    plt.figure(figsize=(8, 5))
    plt.bar(families, f1_scores)
    plt.ylabel("Macro F1")
    plt.xlabel("Feature family")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Ensembles + calibration + tiers
# ---------------------------------------------------------------------
def _build_xgb_model(n_classes: int) -> XGBClassifier:
    """
    To construct an XGBoost classifier with sensible defaults for
    multiclass physiological data, so that the ensemble uses a strong
    gradient-boosted tree component in this project.
    """
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    return model


def _run_ensembles_and_calibration(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    To train XGBoost + Random Forest ensembles under LOSO, calibrate
    their probabilities, and derive tiered risk estimates, so that the
    main predictive and calibration results of the project are produced.
    """
    subjects = sorted(df["subject"].unique())
    ensemble_rows: List[Dict] = []
    calibration_rows: List[Dict] = []
    tier_records: List[Dict] = []

    ensure_dir(MODELS_DIR / "ensembles")

    all_pre_probs_class2: List[float] = []
    all_post_probs_class2: List[float] = []
    all_true_class2: List[int] = []

    for fold_id, test_subject in enumerate(subjects):
        train_df = df[df["subject"] != test_subject]
        test_df = df[df["subject"] == test_subject]

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Base models
        n_classes = len(np.unique(y_train))
        xgb = _build_xgb_model(n_classes=n_classes)
        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

        # Soft voting ensemble combining Random Forest + XGBoost
        voting = VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb)],
            voting="soft",
        )

        # Fit ensemble on scaled training data
        voting.fit(X_train_scaled, y_train)
        y_pred_base = voting.predict(X_test_scaled)
        y_proba_base = voting.predict_proba(X_test_scaled)
        classes = voting.classes_

        # Metrics before calibration
        base_metrics = _evaluate_predictions(
            y_test, y_pred_base, y_proba_base, classes
        )
        base_metrics.update(
            {
                "fold_id": fold_id,
                "test_subject": test_subject,
                "model": "voting_ensemble",
                "stage": "pre_calibration",
            }
        )
        ensemble_rows.append(base_metrics)

        # Calibration using isotonic regression on training data
        calib = CalibratedClassifierCV(
            estimator=voting,
            method="isotonic",
            cv=3,
        )
        calib.fit(X_train_scaled, y_train)

        y_pred_cal = calib.predict(X_test_scaled)
        y_proba_cal = calib.predict_proba(X_test_scaled)

        # Metrics after calibration
        cal_metrics = _evaluate_predictions(
            y_test, y_pred_cal, y_proba_cal, classes
        )
        cal_metrics.update(
            {
                "fold_id": fold_id,
                "test_subject": test_subject,
                "model": "voting_ensemble",
                "stage": "post_calibration",
            }
        )
        ensemble_rows.append(cal_metrics)

        # Brier scores for calibration quality
        brier_pre = _compute_multiclass_brier(y_test, y_proba_base, classes)
        brier_post = _compute_multiclass_brier(y_test, y_proba_cal, classes)

        calibration_rows.append(
            {
                "fold_id": fold_id,
                "test_subject": test_subject,
                "model": "voting_ensemble",
                "brier_pre": brier_pre,
                "brier_post": brier_post,
            }
        )

        # Collect probabilities for high-risk class (assumed label = 2)
        if 2 in classes:
            idx_high = list(classes).index(2)
            prob_high_pre = y_proba_base[:, idx_high]
            prob_high_post = y_proba_cal[:, idx_high]
            all_pre_probs_class2.extend(prob_high_pre.tolist())
            all_post_probs_class2.extend(prob_high_post.tolist())
            all_true_class2.extend((y_test == 2).astype(int).tolist())

            # Tier assignment based on calibrated probabilities
            for p, y_true in zip(prob_high_post, y_test):
                if p < 0.33:
                    tier = "low"
                elif p < 0.66:
                    tier = "medium"
                else:
                    tier = "high"

                tier_records.append(
                    {
                        "fold_id": fold_id,
                        "test_subject": test_subject,
                        "true_label": int(y_true),
                        "p_high_calibrated": float(p),
                        "tier": tier,
                    }
                )

        # Save calibrated ensemble model per fold
        model_path = MODELS_DIR / "ensembles" / f"ensemble_fold_{fold_id}.pkl"
        joblib.dump(calib, model_path)

    # Build calibration summary and tier statistics
    calibration_df = pd.DataFrame(calibration_rows)
    tiers_df = pd.DataFrame(tier_records)

    # Aggregate tier-level expected “costs” using a simple cost scheme
    if not tiers_df.empty:
        cost_map = {0: 1.0, 1: 2.0, 2: 5.0}
        tier_stats = []
        for tier_name, group in tiers_df.groupby("tier"):
            n = len(group)
            counts = group["true_label"].value_counts().to_dict()
            n0 = counts.get(0, 0)
            n1 = counts.get(1, 0)
            n2 = counts.get(2, 0)
            mean_prob = float(group["p_high_calibrated"].mean())
            expected_cost = (
                n0 * cost_map[0] + n1 * cost_map[1] + n2 * cost_map[2]
            ) / max(n, 1)

            tier_stats.append(
                {
                    "tier": tier_name,
                    "n_samples": n,
                    "n_label0": n0,
                    "n_label1": n1,
                    "n_label2": n2,
                    "mean_p_high": mean_prob,
                    "expected_cost": expected_cost,
                }
            )
        tiers_summary = pd.DataFrame(tier_stats)
    else:
        tiers_summary = pd.DataFrame()

    # Build ensemble metrics DataFrame and add mean rows
    ensembles_df = pd.DataFrame(ensemble_rows)
    if not ensembles_df.empty:
        mean_rows = (
            ensembles_df.groupby(["model", "stage"])[
                ["f1_macro", "auroc_macro", "pr_auc_macro"]
            ]
            .mean()
            .reset_index()
        )
        mean_rows["fold_id"] = "mean"
        mean_rows["test_subject"] = "ALL"
        ensembles_df = pd.concat([ensembles_df, mean_rows], ignore_index=True)

    # Reliability curves for high-risk class probabilities
    if all_true_class2 and all_pre_probs_class2 and all_post_probs_class2:
        from sklearn.calibration import calibration_curve

        y_true_bin = np.array(all_true_class2)
        probs_pre = np.array(all_pre_probs_class2)
        probs_post = np.array(all_post_probs_class2)

        frac_pos_pre, mean_pred_pre = calibration_curve(
            y_true_bin, probs_pre, n_bins=10
        )
        frac_pos_post, mean_pred_post = calibration_curve(
            y_true_bin, probs_post, n_bins=10
        )

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(mean_pred_pre, frac_pos_pre, marker="o", label="Pre-calibration")
        plt.plot(mean_pred_post, frac_pos_post, marker="s", label="Post-calibration")
        plt.xlabel("Predicted probability (class 2)")
        plt.ylabel("Observed frequency (class 2)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "calibration_plots.png", dpi=300)
        plt.close()

    return ensembles_df, calibration_df, tiers_summary


# ---------------------------------------------------------------------
# Public entry point for Phase D
# ---------------------------------------------------------------------
def run_phase_D() -> None:
    """
    To perform feature-family ablations, train ensemble models, apply
    probability calibration and derive risk tiers, so that the main
    predictive and calibration results required for this thesis are
    generated in a single, reproducible phase.
    """
    set_global_seeds()
    ensure_dir(TABLES_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(MODELS_DIR)

    # 1. Load enriched dataset and feature set
    df = _load_enriched_dataset()
    df["subject"] = df["subject"].astype(str)
    df["label"] = df["label"].astype(int)
    all_feature_cols = _select_feature_columns(df)

    # 2. Feature family ablations with Random Forest baseline
    ablations = _run_feature_family_ablations(df)
    ablation_path = TABLES_DIR / "feature_family_ablation.csv"
    ablations.to_csv(ablation_path, index=False)

    _plot_feature_family_bars(
        ablations, FIGURES_DIR / "feature_family_bar.png"
    )

    # For ensembles, use the core_plus_composites family
    families = _define_feature_families(df)
    best_family_cols = families.get("core_plus_composites", all_feature_cols)

    # 3. Ensembles + calibration + tiers on best feature family
    ensembles_df, calibration_df, tiers_summary = _run_ensembles_and_calibration(
        df, best_family_cols
    )

    ensembles_path = TABLES_DIR / "ensembles_per_fold.csv"
    calibration_path = TABLES_DIR / "calibration.csv"
    tiers_path = TABLES_DIR / "tiers_costs.csv"

    ensembles_df.to_csv(ensembles_path, index=False)
    calibration_df.to_csv(calibration_path, index=False)
    tiers_summary.to_csv(tiers_path, index=False)

    # 4. Log run metadata
    details = {
        "feature_family_ablation": str(ablation_path),
        "ensembles_per_fold": str(ensembles_path),
        "calibration_csv": str(calibration_path),
        "tiers_costs_csv": str(tiers_path),
    }

    json_path = log_run_metadata(
        phase_name="phase_D_ensembles_calibration_tiers",
        status="success",
        details=details,
    )

    append_runlog_md_row(
        timestamp_utc=json_path.name.split("_")[-1].replace(".json", ""),
        phase_name="phase_D_ensembles_calibration_tiers",
        status="success",
        notes=(
            "Feature ablations, voting ensemble (RF + XGBoost), calibration "
            "and risk tiers completed."
        ),
    )

    print(f"Phase D complete. Logged: {json_path}")
