# ============================================
# file: code/baselines.py
# ============================================
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

from .config import DATA_DIR, MODELS_DIR, TABLES_DIR
from .utils import (
    ensure_dir,
    set_global_seeds,
    log_run_metadata,
    append_runlog_md_row,
)


def _load_enriched_dataset() -> pd.DataFrame:
    features_path = DATA_DIR / "features" / "merged_with_composites.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Expected enriched parquet at {features_path}, but it was not found.")
    df = pd.read_parquet(features_path)

    if "subject" not in df.columns or "label" not in df.columns:
        raise ValueError("Enriched dataset must contain 'subject' and 'label' columns.")

    df["subject"] = df["subject"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"subject", "label"}
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


def _compute_macro_pr_auc(y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> float:
    pr_aucs: List[float] = []
    for idx, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        precision, recall, _ = precision_recall_curve(y_bin, y_proba[:, idx])
        if len(precision) > 1 and len(recall) > 1:
            pr_aucs.append(auc(recall, precision))
    if not pr_aucs:
        return float("nan")
    return float(np.mean(pr_aucs))


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    try:
        metrics["auroc_macro"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        metrics["auroc_macro"] = float("nan")
    metrics["pr_auc_macro"] = _compute_macro_pr_auc(y_true, y_proba, classes)
    return metrics


def _build_loso_folds(df: pd.DataFrame) -> pd.DataFrame:
    subjects = sorted(df["subject"].unique())
    fold_rows: List[Tuple[int, str]] = []
    for fold_id, subj in enumerate(subjects):
        fold_rows.append((fold_id, subj))
    folds_df = pd.DataFrame(fold_rows, columns=["fold_id", "subject"])
    return folds_df


def _train_scalers_and_models(df: pd.DataFrame, feature_cols: List[str], folds_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(MODELS_DIR)
    scalers_dir = MODELS_DIR / "scalers"
    baselines_dir = MODELS_DIR / "baselines"
    ensure_dir(scalers_dir)
    ensure_dir(baselines_dir)
    ensure_dir(TABLES_DIR)

    shuffle_rows: List[Dict] = []
    baseline_rows: List[Dict] = []

    for _, row in folds_df.iterrows():
        fold_id = int(row["fold_id"])
        test_subject = str(row["subject"])

        train_df = df[df["subject"] != test_subject]
        test_df = df[df["subject"] == test_subject]

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        scaler_path = scalers_dir / f"fold_{fold_id}.pkl"
        joblib.dump(scaler, scaler_path)

        # shuffled-label control
        y_train_shuffled = np.random.permutation(y_train)
        logreg_shuffle = LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial")
        logreg_shuffle.fit(X_train_scaled, y_train_shuffled)
        y_pred_sc = logreg_shuffle.predict(X_test_scaled)
        y_proba_sc = logreg_shuffle.predict_proba(X_test_scaled)
        classes_sc = logreg_shuffle.classes_
        metrics_sc = _evaluate_predictions(y_test, y_pred_sc, y_proba_sc, classes_sc)
        metrics_sc.update({"fold_id": fold_id, "test_subject": test_subject, "model": "logreg_shuffled"})
        shuffle_rows.append(metrics_sc)

        # Logistic Regression (real labels)
        logreg = LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial")
        logreg.fit(X_train_scaled, y_train)
        y_pred_lr = logreg.predict(X_test_scaled)
        y_proba_lr = logreg.predict_proba(X_test_scaled)
        classes_lr = logreg.classes_
        metrics_lr = _evaluate_predictions(y_test, y_pred_lr, y_proba_lr, classes_lr)
        metrics_lr.update({"fold_id": fold_id, "test_subject": test_subject, "model": "logreg"})
        baseline_rows.append(metrics_lr)
        logreg_path = baselines_dir / f"logreg_fold_{fold_id}.pkl"
        joblib.dump(logreg, logreg_path)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        y_pred_rf = rf.predict(X_test_scaled)
        y_proba_rf = rf.predict_proba(X_test_scaled)
        classes_rf = rf.classes_
        metrics_rf = _evaluate_predictions(y_test, y_pred_rf, y_proba_rf, classes_rf)
        metrics_rf.update({"fold_id": fold_id, "test_subject": test_subject, "model": "random_forest"})
        baseline_rows.append(metrics_rf)
        rf_path = baselines_dir / f"random_forest_fold_{fold_id}.pkl"
        joblib.dump(rf, rf_path)

    shuffle_df = pd.DataFrame(shuffle_rows)
    baseline_df = pd.DataFrame(baseline_rows)

    if not shuffle_df.empty:
        mean_sc = (shuffle_df.groupby("model")[ ["f1_macro", "auroc_macro", "pr_auc_macro"] ].mean().reset_index())
        mean_sc["fold_id"] = "mean"
        mean_sc["test_subject"] = "ALL"
        shuffle_df = pd.concat([shuffle_df, mean_sc], ignore_index=True)

    if not baseline_df.empty:
        mean_bl = (baseline_df.groupby("model")[ ["f1_macro", "auroc_macro", "pr_auc_macro"] ].mean().reset_index())
        mean_bl["fold_id"] = "mean"
        mean_bl["test_subject"] = "ALL"
        baseline_df = pd.concat([baseline_df, mean_bl], ignore_index=True)

    return shuffle_df, baseline_df


def run_phase_C() -> None:
    set_global_seeds()
    ensure_dir(TABLES_DIR)
    ensure_dir(MODELS_DIR)

    df = _load_enriched_dataset()
    feature_cols = _select_feature_columns(df)

    folds_df = _build_loso_folds(df)
    folds_path = TABLES_DIR / "loso_folds.csv"
    folds_df.to_csv(folds_path, index=False)

    shuffle_df, baseline_df = _train_scalers_and_models(df, feature_cols, folds_df)

    shuffle_path = TABLES_DIR / "shuffle_control.csv"
    baseline_path = TABLES_DIR / "loso_baselines.csv"

    shuffle_df.to_csv(shuffle_path, index=False)
    baseline_df.to_csv(baseline_path, index=False)

    details = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_subjects": int(df["subject"].nunique()),
        "shuffle_control_path": str(shuffle_path),
        "baselines_path": str(baseline_path),
    }

    json_path = log_run_metadata(phase_name="phase_C_loso_scaling_baselines", status="success", details=details)

    append_runlog_md_row(timestamp_utc=json_path.name.split("_")[-1].replace(".json", ""), phase_name="phase_C_loso_scaling_baselines", status="success", notes="LOSO folds, scalers, shuffled-label control and baselines completed.")

    print(f"Phase C complete. Logged: {json_path}")
# ============================================
# file: code/baselines.py
# ============================================
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

from .config import DATA_DIR, MODELS_DIR, TABLES_DIR
from .utils import (
    ensure_dir,
    set_global_seeds,
    log_run_metadata,
    append_runlog_md_row,
)


def _load_enriched_dataset() -> pd.DataFrame:
    """
    To load the enriched dataset with composite indices from Phase B,
    so that LOSO splitting and baseline modelling operate on the same
    feature space used throughout the project for the purposes of
    consistent risk stratification experiments.
    """
    features_path = DATA_DIR / "features" / "merged_with_composites.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Expected enriched parquet at {features_path}, but it was not found."
        )
    df = pd.read_parquet(features_path)

    # Enforce basic schema expectations
    if "subject" not in df.columns or "label" not in df.columns:
        raise ValueError("Enriched dataset must contain 'subject' and 'label' columns.")

    df["subject"] = df["subject"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    To derive the list of feature columns used for modelling, so that
    identifiers and labels are excluded while all raw and composite
    predictors are retained for the purposes of baseline comparison
    in this project.
    """
    exclude = {"subject", "label"}
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


def _compute_macro_pr_auc(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> float:
    """
    To compute macro-averaged PR-AUC across classes, so that precision–
    recall performance is summarised in a way that is robust to class
    imbalance for the purposes of baseline model evaluation.
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

    # AUROC (macro, one-vs-rest)
    try:
        metrics["auroc_macro"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro"
        )
    except ValueError:
        metrics["auroc_macro"] = float("nan")

    # PR-AUC (macro)
    metrics["pr_auc_macro"] = _compute_macro_pr_auc(y_true, y_proba, classes)

    return metrics


def _build_loso_folds(df: pd.DataFrame) -> pd.DataFrame:
    """
    To assign each subject to its own leave-one-subject-out (LOSO) fold,
    so that each model evaluation reflects generalisation to unseen
    individuals for the purposes of insurance-relevant risk modelling
    in this project.
    """
    subjects = sorted(df["subject"].unique())
    fold_rows: List[Tuple[int, str]] = []

    for fold_id, subj in enumerate(subjects):
        fold_rows.append((fold_id, subj))

    folds_df = pd.DataFrame(fold_rows, columns=["fold_id", "subject"])
    return folds_df


def _train_scalers_and_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    folds_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    To iterate over LOSO folds, fit per-fold scalers and baseline
    models, and collect evaluation metrics, so that both shuffled-label
    controls and real-label baselines are available for diagnosing
    leakage and establishing reference performance in this project.
    """
    ensure_dir(MODELS_DIR)
    scalers_dir = MODELS_DIR / "scalers"
    baselines_dir = MODELS_DIR / "baselines"
    ensure_dir(scalers_dir)
    ensure_dir(baselines_dir)
    ensure_dir(TABLES_DIR)

    shuffle_rows: List[Dict] = []
    baseline_rows: List[Dict] = []

    for _, row in folds_df.iterrows():
        fold_id = int(row["fold_id"])
        test_subject = str(row["subject"])

        train_df = df[df["subject"] != test_subject]
        test_df = df[df["subject"] == test_subject]

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        # -----------------------------
        # Per-fold imputer + scaler (train only)
        # -----------------------------
        imputer = SimpleImputer(strategy="mean")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        imputer_path = scalers_dir / f"imputer_fold_{fold_id}.pkl"
        scaler_path = scalers_dir / f"scaler_fold_{fold_id}.pkl"
        joblib.dump(imputer, imputer_path)
        joblib.dump(scaler, scaler_path)

        # -----------------------------
        # Shuffled-label control
        # -----------------------------
        y_train_shuffled = np.random.permutation(y_train)

        logreg_shuffle = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            multi_class="multinomial",
        )
        logreg_shuffle.fit(X_train_scaled, y_train_shuffled)

        y_pred_sc = logreg_shuffle.predict(X_test_scaled)
        y_proba_sc = logreg_shuffle.predict_proba(X_test_scaled)
        classes_sc = logreg_shuffle.classes_

        metrics_sc = _evaluate_predictions(y_test, y_pred_sc, y_proba_sc, classes_sc)
        metrics_sc.update(
            {
                "fold_id": fold_id,
                "test_subject": test_subject,
                "model": "logreg_shuffled",
            }
        )
        shuffle_rows.append(metrics_sc)

        # -----------------------------
        # Real-label baselines
        # -----------------------------
        # Logistic Regression
        logreg = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            multi_class="multinomial",
        )
        logreg.fit(X_train_scaled, y_train)

        y_pred_lr = logreg.predict(X_test_scaled)
        y_proba_lr = logreg.predict_proba(X_test_scaled)
        classes_lr = logreg.classes_

        metrics_lr = _evaluate_predictions(y_test, y_pred_lr, y_proba_lr, classes_lr)
        metrics_lr.update(
            {
                "fold_id": fold_id,
                "test_subject": test_subject,
                "model": "logreg",
            }
        )
        baseline_rows.append(metrics_lr)

        logreg_path = baselines_dir / f"logreg_fold_{fold_id}.pkl"
        joblib.dump(logreg, logreg_path)

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train_scaled, y_train)

        y_pred_rf = rf.predict(X_test_scaled)
        y_proba_rf = rf.predict_proba(X_test_scaled)
        classes_rf = rf.classes_

        metrics_rf = _evaluate_predictions(y_test, y_pred_rf, y_proba_rf, classes_rf)
        metrics_rf.update(
            {
                "fold_id": fold_id,
                "test_subject": test_subject,
                "model": "random_forest",
            }
        )
        baseline_rows.append(metrics_rf)

        rf_path = baselines_dir / f"random_forest_fold_{fold_id}.pkl"
        joblib.dump(rf, rf_path)

    shuffle_df = pd.DataFrame(shuffle_rows)
    baseline_df = pd.DataFrame(baseline_rows)

    # Add mean rows per model for quick inspection
    if not shuffle_df.empty:
        mean_sc = (
            shuffle_df.groupby("model")[["f1_macro", "auroc_macro", "pr_auc_macro"]]
            .mean()
            .reset_index()
        )
        mean_sc["fold_id"] = "mean"
        mean_sc["test_subject"] = "ALL"
        shuffle_df = pd.concat([shuffle_df, mean_sc], ignore_index=True)

    if not baseline_df.empty:
        mean_bl = (
            baseline_df.groupby("model")[["f1_macro", "auroc_macro", "pr_auc_macro"]]
            .mean()
            .reset_index()
        )
        mean_bl["fold_id"] = "mean"
        mean_bl["test_subject"] = "ALL"
        baseline_df = pd.concat([baseline_df, mean_bl], ignore_index=True)

    return shuffle_df, baseline_df


def run_phase_C() -> None:
    """
    To construct LOSO folds, train per-fold scalers, run shuffled-label
    controls and fit baseline models, so that the pipeline produces
    leakage checks and benchmark performance estimates for the purposes
    of validating the modelling design in this project.
    """
    set_global_seeds()
    ensure_dir(TABLES_DIR)
    ensure_dir(MODELS_DIR)

    # 1) Load enriched data
    df = _load_enriched_dataset()
    feature_cols = _select_feature_columns(df)

    # 2) Build LOSO folds
    folds_df = _build_loso_folds(df)
    folds_path = TABLES_DIR / "loso_folds.csv"
    folds_df.to_csv(folds_path, index=False)

    # 3) Train scalers, shuffled control and baselines
    shuffle_df, baseline_df = _train_scalers_and_models(df, feature_cols, folds_df)

    shuffle_path = TABLES_DIR / "shuffle_control.csv"
    baseline_path = TABLES_DIR / "loso_baselines.csv"

    shuffle_df.to_csv(shuffle_path, index=False)
    baseline_df.to_csv(baseline_path, index=False)

    # 4) Log run metadata
    details = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_subjects": int(df["subject"].nunique()),
        "shuffle_control_path": str(shuffle_path),
        "baselines_path": str(baseline_path),
    }

    json_path = log_run_metadata(
        phase_name="phase_C_loso_scaling_baselines",
        status="success",
        details=details,
    )

    append_runlog_md_row(
        timestamp_utc=json_path.name.split("_")[-1].replace(".json", ""),
        phase_name="phase_C_loso_scaling_baselines",
        status="success",
        notes="LOSO folds, scalers, shuffled-label control and baselines completed.",
    )

    print(f"Phase C complete. Logged: {json_path}")
