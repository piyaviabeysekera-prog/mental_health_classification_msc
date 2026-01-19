# ============================================
# file: code/phase_G.py
# Phase G: Heterogeneous Multi-Model Ensemble & Comparative Performance Audit
# ============================================
"""
Phase G: Comprehensive heterogeneous ensemble evaluation with generalization gap analysis.

This phase compares six individual models (LogReg, RF, XGBoost, ExtraTrees, LightGBM, CatBoost)
using strict Leave-One-Subject-Out (LOSO) cross-validation, creates a soft-voting ensemble,
and calculates generalization gaps to audit overfitting.

Non-Destructive: This phase creates new outputs without modifying any previous phases.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
)

from .config import DATA_DIR, MODELS_DIR, TABLES_DIR, RANDOM_SEED
from .utils import (
    ensure_dir,
    set_global_seeds,
    log_run_metadata,
    append_runlog_md_row,
)

# Optional imports for advanced models (graceful fallback if not available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. Phase G will skip XGBoost models.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not installed. Phase G will skip LightGBM models.")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not installed. Phase G will skip CatBoost models.")


def _load_enriched_dataset() -> pd.DataFrame:
    """Load the Phase B enriched dataset with composites."""
    features_path = DATA_DIR / "features" / "merged_with_composites.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Expected enriched parquet at {features_path}. "
            "Ensure Phase B (composites) has been run first."
        )
    df = pd.read_parquet(features_path)

    if "subject" not in df.columns or "label" not in df.columns:
        raise ValueError("Enriched dataset must contain 'subject' and 'label' columns.")

    df["subject"] = df["subject"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Exclude subject and label columns; return all feature columns."""
    exclude = {"subject", "label"}
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


def _compute_macro_pr_auc(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> float:
    """Compute macro-averaged Precision-Recall AUC."""
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
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> Dict[str, float]:
    """Calculate all evaluation metrics for a single fold."""
    metrics: Dict[str, float] = {}
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    
    try:
        metrics["auroc_macro"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["auroc_macro"] = float("nan")
    
    metrics["pr_auc_macro"] = _compute_macro_pr_auc(y_true, y_proba, classes)
    return metrics


def _build_loso_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Build Leave-One-Subject-Out fold structure."""
    subjects = sorted(df["subject"].unique())
    fold_rows: List[Tuple[int, str]] = []
    for fold_id, subj in enumerate(subjects):
        fold_rows.append((fold_id, subj))
    folds_df = pd.DataFrame(fold_rows, columns=["fold_id", "subject"])
    return folds_df


def _train_and_evaluate_models(
    df: pd.DataFrame, feature_cols: List[str], folds_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train 6 individual models + voting ensemble using LOSO CV.
    
    Returns:
        individual_results_df: Per-fold metrics for 6 individual models
        ensemble_results_df: Per-fold metrics for voting ensemble
    """
    ensure_dir(MODELS_DIR)
    ensure_dir(TABLES_DIR)
    
    phase_g_models_dir = MODELS_DIR / "phase_G"
    ensure_dir(phase_g_models_dir)
    
    individual_rows: List[Dict] = []
    ensemble_rows: List[Dict] = []
    
    n_folds = len(folds_df)
    
    for fold_idx, (_, fold_row) in enumerate(folds_df.iterrows()):
        fold_id = int(fold_row["fold_id"])
        test_subject = str(fold_row["subject"])
        
        print(f"  Processing fold {fold_idx + 1}/{n_folds} (test_subject={test_subject})...")
        
        # Split data
        train_df = df[df["subject"] != test_subject]
        test_df = df[df["subject"] == test_subject]
        
        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Handle missing values
        imputer = SimpleImputer(strategy="mean")
        X_train_scaled = imputer.fit_transform(X_train_scaled)
        X_test_scaled = imputer.transform(X_test_scaled)
        
        # ====== INDIVIDUAL MODELS ======
        trained_models = {}
        
        # 1. Logistic Regression
        print(f"    Training LogisticRegression...")
        logreg = LogisticRegression(
            max_iter=1000, class_weight="balanced", multi_class="multinomial", random_state=RANDOM_SEED
        )
        logreg.fit(X_train_scaled, y_train)
        trained_models["logreg"] = logreg
        
        y_pred_lr_train = logreg.predict(X_train_scaled)
        y_proba_lr_train = logreg.predict_proba(X_train_scaled)
        y_pred_lr_test = logreg.predict(X_test_scaled)
        y_proba_lr_test = logreg.predict_proba(X_test_scaled)
        classes_lr = logreg.classes_
        
        train_metrics_lr = _evaluate_predictions(y_train, y_pred_lr_train, y_proba_lr_train, classes_lr)
        test_metrics_lr = _evaluate_predictions(y_test, y_pred_lr_test, y_proba_lr_test, classes_lr)
        
        train_metrics_lr.update({"fold_id": fold_id, "test_subject": test_subject, "model": "LogisticRegression", "stage": "train"})
        test_metrics_lr.update({"fold_id": fold_id, "test_subject": test_subject, "model": "LogisticRegression", "stage": "test"})
        individual_rows.append(train_metrics_lr)
        individual_rows.append(test_metrics_lr)
        
        # 2. Random Forest
        print(f"    Training RandomForest...")
        rf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)
        trained_models["random_forest"] = rf
        
        y_pred_rf_train = rf.predict(X_train_scaled)
        y_proba_rf_train = rf.predict_proba(X_train_scaled)
        y_pred_rf_test = rf.predict(X_test_scaled)
        y_proba_rf_test = rf.predict_proba(X_test_scaled)
        classes_rf = rf.classes_
        
        train_metrics_rf = _evaluate_predictions(y_train, y_pred_rf_train, y_proba_rf_train, classes_rf)
        test_metrics_rf = _evaluate_predictions(y_test, y_pred_rf_test, y_proba_rf_test, classes_rf)
        
        train_metrics_rf.update({"fold_id": fold_id, "test_subject": test_subject, "model": "RandomForest", "stage": "train"})
        test_metrics_rf.update({"fold_id": fold_id, "test_subject": test_subject, "model": "RandomForest", "stage": "test"})
        individual_rows.append(train_metrics_rf)
        individual_rows.append(test_metrics_rf)
        
        # 3. XGBoost
        if XGBOOST_AVAILABLE:
            print(f"    Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1, 
                objective="multi:softprob", num_class=3, random_state=RANDOM_SEED, n_jobs=-1
            )
            xgb_model.fit(X_train_scaled, y_train, verbose=0)
            trained_models["xgboost"] = xgb_model
            
            y_pred_xgb_train = xgb_model.predict(X_train_scaled)
            y_proba_xgb_train = xgb_model.predict_proba(X_train_scaled)
            y_pred_xgb_test = xgb_model.predict(X_test_scaled)
            y_proba_xgb_test = xgb_model.predict_proba(X_test_scaled)
            classes_xgb = xgb_model.classes_
            
            train_metrics_xgb = _evaluate_predictions(y_train, y_pred_xgb_train, y_proba_xgb_train, classes_xgb)
            test_metrics_xgb = _evaluate_predictions(y_test, y_pred_xgb_test, y_proba_xgb_test, classes_xgb)
            
            train_metrics_xgb.update({"fold_id": fold_id, "test_subject": test_subject, "model": "XGBoost", "stage": "train"})
            test_metrics_xgb.update({"fold_id": fold_id, "test_subject": test_subject, "model": "XGBoost", "stage": "test"})
            individual_rows.append(train_metrics_xgb)
            individual_rows.append(test_metrics_xgb)
        
        # 4. Extra Trees Classifier
        print(f"    Training ExtraTrees...")
        et = ExtraTreesClassifier(
            n_estimators=200, class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=-1
        )
        et.fit(X_train_scaled, y_train)
        trained_models["extra_trees"] = et
        
        y_pred_et_train = et.predict(X_train_scaled)
        y_proba_et_train = et.predict_proba(X_train_scaled)
        y_pred_et_test = et.predict(X_test_scaled)
        y_proba_et_test = et.predict_proba(X_test_scaled)
        classes_et = et.classes_
        
        train_metrics_et = _evaluate_predictions(y_train, y_pred_et_train, y_proba_et_train, classes_et)
        test_metrics_et = _evaluate_predictions(y_test, y_pred_et_test, y_proba_et_test, classes_et)
        
        train_metrics_et.update({"fold_id": fold_id, "test_subject": test_subject, "model": "ExtraTrees", "stage": "train"})
        test_metrics_et.update({"fold_id": fold_id, "test_subject": test_subject, "model": "ExtraTrees", "stage": "test"})
        individual_rows.append(train_metrics_et)
        individual_rows.append(test_metrics_et)
        
        # 5. LightGBM
        if LIGHTGBM_AVAILABLE:
            print(f"    Training LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1, num_leaves=31,
                objective="multiclass", num_class=3, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
            )
            lgb_model.fit(X_train_scaled, y_train)
            trained_models["lightgbm"] = lgb_model
            
            y_pred_lgb_train = lgb_model.predict(X_train_scaled)
            y_proba_lgb_train = lgb_model.predict_proba(X_train_scaled)
            y_pred_lgb_test = lgb_model.predict(X_test_scaled)
            y_proba_lgb_test = lgb_model.predict_proba(X_test_scaled)
            classes_lgb = lgb_model.classes_
            
            train_metrics_lgb = _evaluate_predictions(y_train, y_pred_lgb_train, y_proba_lgb_train, classes_lgb)
            test_metrics_lgb = _evaluate_predictions(y_test, y_pred_lgb_test, y_proba_lgb_test, classes_lgb)
            
            train_metrics_lgb.update({"fold_id": fold_id, "test_subject": test_subject, "model": "LightGBM", "stage": "train"})
            test_metrics_lgb.update({"fold_id": fold_id, "test_subject": test_subject, "model": "LightGBM", "stage": "test"})
            individual_rows.append(train_metrics_lgb)
            individual_rows.append(test_metrics_lgb)
        
        # 6. CatBoost
        if CATBOOST_AVAILABLE:
            print(f"    Training CatBoost...")
            cb_model = CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.1, random_state=RANDOM_SEED, verbose=0
            )
            cb_model.fit(X_train_scaled, y_train)
            trained_models["catboost"] = cb_model
            
            y_pred_cb_train = cb_model.predict(X_train_scaled)
            y_proba_cb_train = cb_model.predict_proba(X_train_scaled)
            y_pred_cb_test = cb_model.predict(X_test_scaled)
            y_proba_cb_test = cb_model.predict_proba(X_test_scaled)
            classes_cb = cb_model.classes_
            
            train_metrics_cb = _evaluate_predictions(y_train, y_pred_cb_train, y_proba_cb_train, classes_cb)
            test_metrics_cb = _evaluate_predictions(y_test, y_pred_cb_test, y_proba_cb_test, classes_cb)
            
            train_metrics_cb.update({"fold_id": fold_id, "test_subject": test_subject, "model": "CatBoost", "stage": "train"})
            test_metrics_cb.update({"fold_id": fold_id, "test_subject": test_subject, "model": "CatBoost", "stage": "test"})
            individual_rows.append(train_metrics_cb)
            individual_rows.append(test_metrics_cb)
        
        # ====== VOTING ENSEMBLE ======
        print(f"    Creating VotingClassifier (soft voting)...")
        ensemble_estimators = [
            ("logreg", logreg),
            ("rf", rf),
            ("et", et),
        ]
        
        if XGBOOST_AVAILABLE:
            ensemble_estimators.append(("xgb", trained_models["xgboost"]))
        if LIGHTGBM_AVAILABLE:
            ensemble_estimators.append(("lgb", trained_models["lightgbm"]))
        if CATBOOST_AVAILABLE:
            ensemble_estimators.append(("cb", trained_models["catboost"]))
        
        voting_clf = VotingClassifier(estimators=ensemble_estimators, voting="soft")
        voting_clf.fit(X_train_scaled, y_train)
        
        y_pred_ens_train = voting_clf.predict(X_train_scaled)
        y_proba_ens_train = voting_clf.predict_proba(X_train_scaled)
        y_pred_ens_test = voting_clf.predict(X_test_scaled)
        y_proba_ens_test = voting_clf.predict_proba(X_test_scaled)
        classes_ens = voting_clf.classes_
        
        train_metrics_ens = _evaluate_predictions(y_train, y_pred_ens_train, y_proba_ens_train, classes_ens)
        test_metrics_ens = _evaluate_predictions(y_test, y_pred_ens_test, y_proba_ens_test, classes_ens)
        
        train_metrics_ens.update({"fold_id": fold_id, "test_subject": test_subject, "model": "VotingEnsemble", "stage": "train"})
        test_metrics_ens.update({"fold_id": fold_id, "test_subject": test_subject, "model": "VotingEnsemble", "stage": "test"})
        ensemble_rows.append(train_metrics_ens)
        ensemble_rows.append(test_metrics_ens)
        
        # Save models for this fold
        logreg_path = phase_g_models_dir / f"logreg_fold_{fold_id}.pkl"
        rf_path = phase_g_models_dir / f"random_forest_fold_{fold_id}.pkl"
        et_path = phase_g_models_dir / f"extra_trees_fold_{fold_id}.pkl"
        ensemble_path = phase_g_models_dir / f"voting_ensemble_fold_{fold_id}.pkl"
        
        joblib.dump(logreg, logreg_path)
        joblib.dump(rf, rf_path)
        joblib.dump(et, et_path)
        joblib.dump(voting_clf, ensemble_path)
    
    # Compile results into DataFrames
    individual_df = pd.DataFrame(individual_rows)
    ensemble_df = pd.DataFrame(ensemble_rows)
    
    # Calculate generalization gaps (Training F1 - Testing F1) for individual models
    individual_df["generalization_gap"] = 0.0
    for model in individual_df["model"].unique():
        for fold_id in individual_df["fold_id"].unique():
            train_mask = (individual_df["model"] == model) & (individual_df["fold_id"] == fold_id) & (individual_df["stage"] == "train")
            test_mask = (individual_df["model"] == model) & (individual_df["fold_id"] == fold_id) & (individual_df["stage"] == "test")
            
            if train_mask.any() and test_mask.any():
                train_f1 = individual_df.loc[train_mask, "f1_macro"].values[0]
                test_f1 = individual_df.loc[test_mask, "f1_macro"].values[0]
                gap = train_f1 - test_f1
                individual_df.loc[train_mask, "generalization_gap"] = gap
                individual_df.loc[test_mask, "generalization_gap"] = gap
    
    # Calculate generalization gaps for ensemble
    ensemble_df["generalization_gap"] = 0.0
    for fold_id in ensemble_df["fold_id"].unique():
        train_mask = (ensemble_df["fold_id"] == fold_id) & (ensemble_df["stage"] == "train")
        test_mask = (ensemble_df["fold_id"] == fold_id) & (ensemble_df["stage"] == "test")
        
        if train_mask.any() and test_mask.any():
            train_f1 = ensemble_df.loc[train_mask, "f1_macro"].values[0]
            test_f1 = ensemble_df.loc[test_mask, "f1_macro"].values[0]
            gap = train_f1 - test_f1
            ensemble_df.loc[train_mask, "generalization_gap"] = gap
            ensemble_df.loc[test_mask, "generalization_gap"] = gap
    
    return individual_df, ensemble_df


def _aggregate_results(
    individual_df: pd.DataFrame, ensemble_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create summary statistics per model.
    
    Returns:
        individual_summary_df: Summary metrics for individual models
        ensemble_summary_df: Summary metrics for ensemble
    """
    # Individual models summary
    individual_test = individual_df[individual_df["stage"] == "test"].copy()
    individual_summary = individual_test.groupby("model").agg({
        "f1_macro": ["mean", "std", "count"],
        "accuracy": ["mean", "std"],
        "auroc_macro": ["mean", "std"],
        "pr_auc_macro": ["mean", "std"],
        "generalization_gap": ["mean", "std"],
    }).reset_index()
    
    individual_summary.columns = [
        "model",
        "f1_macro_mean", "f1_macro_std", "n_folds",
        "accuracy_mean", "accuracy_std",
        "auroc_macro_mean", "auroc_macro_std",
        "pr_auc_macro_mean", "pr_auc_macro_std",
        "generalization_gap_mean", "generalization_gap_std",
    ]
    
    # Ensemble summary
    ensemble_test = ensemble_df[ensemble_df["stage"] == "test"].copy()
    ensemble_summary = ensemble_test.groupby("model").agg({
        "f1_macro": ["mean", "std", "count"],
        "accuracy": ["mean", "std"],
        "auroc_macro": ["mean", "std"],
        "pr_auc_macro": ["mean", "std"],
        "generalization_gap": ["mean", "std"],
    }).reset_index()
    
    ensemble_summary.columns = [
        "model",
        "f1_macro_mean", "f1_macro_std", "n_folds",
        "accuracy_mean", "accuracy_std",
        "auroc_macro_mean", "auroc_macro_std",
        "pr_auc_macro_mean", "pr_auc_macro_std",
        "generalization_gap_mean", "generalization_gap_std",
    ]
    
    return individual_summary, ensemble_summary


def run_phase_G() -> None:
    """Main entry point for Phase G."""
    print("\n" + "=" * 80)
    print("PHASE G: Heterogeneous Multi-Model Ensemble & Comparative Performance Audit")
    print("=" * 80 + "\n")
    
    set_global_seeds()
    ensure_dir(TABLES_DIR)
    ensure_dir(MODELS_DIR)
    
    print("Loading enriched dataset...")
    df = _load_enriched_dataset()
    feature_cols = _select_feature_columns(df)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Number of subjects: {df['subject'].nunique()}")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}\n")
    
    print("Building LOSO folds...")
    folds_df = _build_loso_folds(df)
    print(f"Number of folds: {len(folds_df)}\n")
    
    print("Training and evaluating models (this may take several minutes)...")
    individual_df, ensemble_df = _train_and_evaluate_models(df, feature_cols, folds_df)
    
    print("\nAggregating results...")
    individual_summary, ensemble_summary = _aggregate_results(individual_df, ensemble_df)
    
    # Save outputs
    individual_path = TABLES_DIR / "phase_G_individual_performance.csv"
    ensemble_path = TABLES_DIR / "phase_G_ensemble_performance.csv"
    individual_folds_path = TABLES_DIR / "phase_G_individual_fold_metrics.csv"
    ensemble_folds_path = TABLES_DIR / "phase_G_ensemble_fold_metrics.csv"
    
    individual_summary.to_csv(individual_path, index=False)
    ensemble_summary.to_csv(ensemble_path, index=False)
    individual_df.to_csv(individual_folds_path, index=False)
    ensemble_df.to_csv(ensemble_folds_path, index=False)
    
    print(f"\n✓ Individual model summary saved to: {individual_path}")
    print(f"✓ Ensemble model summary saved to: {ensemble_path}")
    print(f"✓ Individual fold metrics saved to: {individual_folds_path}")
    print(f"✓ Ensemble fold metrics saved to: {ensemble_folds_path}")
    
    # Display summary tables
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODELS SUMMARY (Test Set Performance)")
    print("=" * 80)
    print(individual_summary.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("VOTING ENSEMBLE SUMMARY (Test Set Performance)")
    print("=" * 80)
    print(ensemble_summary.to_string(index=False))
    
    # Log metadata
    details = {
        "n_rows": int(df.shape[0]),
        "n_features": int(len(feature_cols)),
        "n_subjects": int(df["subject"].nunique()),
        "n_folds": int(len(folds_df)),
        "models_trained": list(individual_df["model"].unique()),
        "individual_summary_path": str(individual_path),
        "ensemble_summary_path": str(ensemble_path),
        "individual_fold_metrics_path": str(individual_folds_path),
        "ensemble_fold_metrics_path": str(ensemble_folds_path),
        "xgboost_available": XGBOOST_AVAILABLE,
        "lightgbm_available": LIGHTGBM_AVAILABLE,
        "catboost_available": CATBOOST_AVAILABLE,
    }
    
    json_path = log_run_metadata(
        phase_name="phase_G_heterogeneous_ensemble",
        status="success",
        details=details,
    )
    
    append_runlog_md_row(
        timestamp_utc=json_path.name.split("_")[-1].replace(".json", ""),
        phase_name="phase_G_heterogeneous_ensemble",
        status="success",
        notes="Heterogeneous ensemble with 6 models, LOSO CV, generalization gap analysis complete.",
    )
    
    print(f"\n✓ Phase G complete. Logged metadata to: {json_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_phase_G()
