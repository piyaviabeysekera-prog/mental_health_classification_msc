"""
Extract existing LOSO metrics from the project without training any models.

This script:
1. Recursively scans reports/ and models/ for CSV files containing metric columns
2. Identifies metric files by detecting f1_macro, auroc_macro, pr_auc_macro, fold_id, etc.
3. Concatenates all fold rows into a single dataframe
4. Outputs two summary files:
   - loso_all_models_fold_metrics.csv (all fold rows for all models)
   - loso_all_models_summary.csv (mean/std/count per model)
5. If XGBoost/ensemble metrics are missing, attempts to build them from prediction files
6. Adds clear logging of which files were used and which models were found

Usage:
    python tools/extract_existing_loso_metrics.py
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define root paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = PROJECT_ROOT / "models"


def find_metric_files() -> List[Path]:
    """Recursively find CSV files in reports/ and models/ directories."""
    metric_files = []
    
    search_dirs = [REPORTS_DIR, MODELS_DIR]
    for search_dir in search_dirs:
        if search_dir.exists():
            metric_files.extend(search_dir.rglob("*.csv"))
    
    logger.info(f"Found {len(metric_files)} CSV files in search directories")
    return metric_files


def is_metric_file(df: pd.DataFrame) -> bool:
    """
    Check if a dataframe contains LOSO metric columns.
    Metric files should have columns like: f1_macro, auroc_macro, pr_auc_macro, fold_id, test_subject, model
    """
    metric_cols = {"f1_macro", "auroc_macro", "pr_auc_macro"}
    fold_cols = {"fold_id", "test_subject"}
    model_col = {"model"}
    
    has_metrics = bool(metric_cols & set(df.columns))
    has_fold_info = bool(fold_cols & set(df.columns))
    has_model = bool(model_col & set(df.columns))
    
    return has_metrics and has_fold_info and has_model


def extract_metrics_from_files(metric_files: List[Path]) -> pd.DataFrame:
    """
    Read all metric files and extract rows with metric columns.
    Return concatenated dataframe of all metrics.
    """
    all_metrics = []
    used_files = []
    
    for file_path in metric_files:
        try:
            df = pd.read_csv(file_path)
            
            if is_metric_file(df):
                used_files.append(file_path)
                all_metrics.append(df)
                logger.info(f"✓ Extracted metrics from: {file_path.relative_to(PROJECT_ROOT)}")
        
        except Exception as e:
            logger.debug(f"Skipped {file_path.name}: {e}")
    
    if not all_metrics:
        logger.warning("No metric files found with standard metric columns")
        return pd.DataFrame()
    
    combined = pd.concat(all_metrics, ignore_index=True)
    logger.info(f"✓ Extracted metrics from {len(used_files)} files")
    logger.info(f"✓ Total metric rows: {len(combined)}")
    
    return combined


def find_prediction_files() -> List[Path]:
    """Find prediction CSV files that might contain y_true, y_pred, proba_* columns."""
    pred_files = []
    
    for search_dir in [REPORTS_DIR, MODELS_DIR]:
        if search_dir.exists():
            # Look for files with 'pred' or 'prob' in the name
            pred_files.extend(search_dir.rglob("*pred*.csv"))
            pred_files.extend(search_dir.rglob("*prob*.csv"))
    
    return list(set(pred_files))  # Remove duplicates


def is_prediction_file(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Check if dataframe contains prediction columns: y_true, y_pred (or y_actual, y_predicted)
    and fold info. Return tuple of (is_pred_file, model_name).
    """
    has_true = any(col in df.columns for col in ['y_true', 'y_actual', 'label', 'target'])
    has_pred = any(col in df.columns for col in ['y_pred', 'y_predicted', 'prediction'])
    has_proba = any(col.startswith('proba_') or col.startswith('prob_') for col in df.columns)
    has_fold = any(col in df.columns for col in ['fold_id', 'fold', 'test_subject', 'subject'])
    
    is_pred = (has_true and (has_pred or has_proba)) and has_fold
    
    # Try to infer model name from file name
    model_name = None
    if is_pred:
        filename = df.attrs.get('filename', '')
        if 'xgboost' in filename.lower() or 'xgb' in filename.lower():
            model_name = 'xgboost'
        elif 'ensemble' in filename.lower():
            model_name = 'ensemble'
    
    return is_pred, model_name


def compute_metrics_from_predictions(pred_df: pd.DataFrame, model_name: str) -> Optional[pd.DataFrame]:
    """
    Compute F1, AUROC, and PR-AUC from prediction dataframe.
    Assumes columns: y_true (or equivalent), y_pred (or proba_*), fold_id, test_subject
    """
    try:
        # Standardize column names
        y_true_col = next((col for col in pred_df.columns if col in ['y_true', 'y_actual', 'label', 'target']), None)
        y_pred_col = next((col for col in pred_df.columns if col in ['y_pred', 'y_predicted', 'prediction']), None)
        proba_cols = [col for col in pred_df.columns if col.startswith('proba_') or col.startswith('prob_')]
        fold_col = next((col for col in pred_df.columns if col in ['fold_id', 'fold']), None)
        subject_col = next((col for col in pred_df.columns if col in ['test_subject', 'subject']), None)
        
        if not y_true_col:
            return None
        
        y_true = pred_df[y_true_col].astype(int)
        
        # Compute F1 (macro)
        if y_pred_col and y_pred_col in pred_df.columns:
            y_pred = pred_df[y_pred_col].astype(int)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            f1_macro = np.nan
        
        # Compute AUROC (macro, one-vs-rest)
        auroc_macro = np.nan
        if len(proba_cols) >= 3:  # At least 3 classes
            y_proba = pred_df[[col for col in proba_cols if col in pred_df.columns]].values
            try:
                auroc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            except Exception:
                auroc_macro = np.nan
        
        # Compute PR-AUC (macro, one-vs-rest, for high-risk class 2)
        pr_auc_macro = np.nan
        if len(proba_cols) >= 3:
            y_proba = pred_df[[col for col in proba_cols if col in pred_df.columns]].values
            try:
                # For each class, compute PR-AUC (one-vs-rest)
                pr_aucs = []
                for cls in range(y_proba.shape[1]):
                    y_true_bin = (y_true == cls).astype(int)
                    if len(np.unique(y_true_bin)) > 1:  # Only if both classes present
                        precision, recall, _ = precision_recall_curve(y_true_bin, y_proba[:, cls])
                        pr_auc = auc(recall, precision)
                        pr_aucs.append(pr_auc)
                pr_auc_macro = np.mean(pr_aucs) if pr_aucs else np.nan
            except Exception:
                pr_auc_macro = np.nan
        
        # Create result row
        fold_id = pred_df[fold_col].iloc[0] if fold_col else np.nan
        test_subject = pred_df[subject_col].iloc[0] if subject_col else 'unknown'
        
        return pd.DataFrame({
            'f1_macro': [f1_macro],
            'auroc_macro': [auroc_macro],
            'pr_auc_macro': [pr_auc_macro],
            'fold_id': [fold_id],
            'test_subject': [test_subject],
            'model': [model_name]
        })
    
    except Exception as e:
        logger.debug(f"Could not compute metrics from predictions: {e}")
        return None


def extract_metrics_from_predictions(pred_files: List[Path]) -> pd.DataFrame:
    """
    Attempt to extract metrics from prediction files if standard metric CSVs don't exist.
    """
    pred_metrics = []
    found_models = set()
    
    for file_path in pred_files:
        try:
            df = pd.read_csv(file_path)
            df.attrs['filename'] = file_path.name
            
            is_pred, model_name = is_prediction_file(df)
            
            if is_pred and model_name:
                logger.info(f"Found prediction file: {file_path.relative_to(PROJECT_ROOT)} (model: {model_name})")
                
                # Try to compute metrics by fold
                fold_col = next((col for col in df.columns if col in ['fold_id', 'fold']), None)
                if fold_col:
                    for fold_id in df[fold_col].unique():
                        fold_df = df[df[fold_col] == fold_id]
                        metrics = compute_metrics_from_predictions(fold_df, model_name)
                        if metrics is not None:
                            pred_metrics.append(metrics)
                            found_models.add(model_name)
        
        except Exception as e:
            logger.debug(f"Skipped {file_path.name}: {e}")
    
    if pred_metrics:
        combined = pd.concat(pred_metrics, ignore_index=True)
        logger.info(f"✓ Extracted metrics from {len(found_models)} models via prediction files")
        return combined
    
    return pd.DataFrame()


def create_summary(all_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics per model: mean, std, count of F1, AUROC, PR-AUC.
    """
    if all_metrics.empty:
        return pd.DataFrame()
    
    # Ensure numeric columns
    numeric_cols = ['f1_macro', 'auroc_macro', 'pr_auc_macro']
    for col in numeric_cols:
        if col in all_metrics.columns:
            all_metrics[col] = pd.to_numeric(all_metrics[col], errors='coerce')
    
    # Group by model
    grouped = all_metrics.groupby('model')[numeric_cols].agg(['mean', 'std', 'count'])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    
    return grouped


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("LOSO METRICS EXTRACTION (No Model Training)")
    logger.info("=" * 80)
    
    # Step 1: Find and extract from metric files
    logger.info("\n[STEP 1] Scanning for existing metric files...")
    metric_files = find_metric_files()
    all_metrics = extract_metrics_from_files(metric_files)
    
    # Step 2: If no metrics found, try prediction files
    if all_metrics.empty:
        logger.info("\n[STEP 2] No metric files found. Attempting to build from prediction files...")
        pred_files = find_prediction_files()
        logger.info(f"Found {len(pred_files)} prediction files. Extracting...")
        pred_metrics = extract_metrics_from_predictions(pred_files)
        
        if not pred_metrics.empty:
            all_metrics = pred_metrics
        else:
            logger.error("No metric files or prediction files found!")
            return
    else:
        logger.info("\n[STEP 2] Skipping prediction files (metric files already found)")
    
    # Step 3: Create summary
    logger.info("\n[STEP 3] Creating summary statistics...")
    summary = create_summary(all_metrics)
    
    # Step 4: Output files
    logger.info("\n[STEP 4] Writing output files...")
    
    fold_metrics_path = TABLES_DIR / "loso_all_models_fold_metrics.csv"
    summary_path = TABLES_DIR / "loso_all_models_summary.csv"
    
    all_metrics.to_csv(fold_metrics_path, index=False)
    logger.info(f"✓ Wrote fold metrics: {fold_metrics_path.relative_to(PROJECT_ROOT)}")
    logger.info(f"  Rows: {len(all_metrics)}, Columns: {list(all_metrics.columns)}")
    
    summary.to_csv(summary_path, index=False)
    logger.info(f"✓ Wrote summary: {summary_path.relative_to(PROJECT_ROOT)}")
    logger.info(f"  Models found: {list(summary['model'].values)}")
    
    # Step 5: Display summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS (by model)")
    logger.info("=" * 80)
    print(summary.to_string(index=False))
    
    # Step 6: Model count by fold
    logger.info("\n" + "=" * 80)
    logger.info("MODELS BY FOLD")
    logger.info("=" * 80)
    fold_summary = all_metrics.groupby(['fold_id', 'model']).size().unstack(fill_value=0)
    print(fold_summary)
    
    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
