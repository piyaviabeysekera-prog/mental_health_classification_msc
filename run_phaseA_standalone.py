import sys
sys.path.insert(0, '.')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from code.config import MERGED_CSV_PATH, TABLES_DIR, FIGURES_DIR, PHYSIO_RANGES
from code.utils import ensure_dir, log_run_metadata, append_runlog_md_row, set_global_seeds

from datetime import datetime


def run_phase_A():
    set_global_seeds()
    ensure_dir(TABLES_DIR)
    ensure_dir(FIGURES_DIR)

    df = pd.read_csv(MERGED_CSV_PATH)

    required_cols = [
        "EDA_phasic_mean", "EDA_phasic_std", "EDA_tonic_std", "EDA_mean",
        "BVP_mean", "BVP_std", "BVP_peak_freq",
        "TEMP_mean", "TEMP_std", "TEMP_min", "TEMP_max", "TEMP_slope",
        "net_acc_mean", "subject", "label"
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    numeric_cols = [c for c in required_cols if c not in ["subject", "label"]]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["subject"] = df["subject"].astype(str)
    df["label"] = df["label"].astype(int)

    schema_summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "min": df.min(numeric_only=True),
        "max": df.max(numeric_only=True)
    })
    schema_summary.to_csv(TABLES_DIR / "schema_summary.csv", index=False)

    missingness = df.isna().mean().reset_index()
    missingness.columns = ["column", "missing_fraction"]
    missingness.to_csv(TABLES_DIR / "missingness.csv", index=False)

    class_balance = df.groupby("subject")["label"].value_counts().unstack(fill_value=0)
    class_balance.to_csv(TABLES_DIR / "class_balance_by_subject.csv")

    violations = []
    eda_neg = df[df["EDA_mean"] < PHYSIO_RANGES["EDA_min"]].shape[0]
    if eda_neg > 0:
        violations.append(("EDA_mean", eda_neg))

    temp_low = df[df["TEMP_mean"] < PHYSIO_RANGES["TEMP_min"]].shape[0]
    temp_high = df[df["TEMP_mean"] > PHYSIO_RANGES["TEMP_max"]].shape[0]
    if temp_low > 0: violations.append(("TEMP_mean_low", temp_low))
    if temp_high > 0: violations.append(("TEMP_mean_high", temp_high))

    bvp_low = df[df["BVP_peak_freq"] < PHYSIO_RANGES["BVP_freq_min"]].shape[0]
    bvp_high = df[df["BVP_peak_freq"] > PHYSIO_RANGES["BVP_freq_max"]].shape[0]
    if bvp_low > 0: violations.append(("BVP_peak_freq_low", bvp_low))
    if bvp_high > 0: violations.append(("BVP_peak_freq_high", bvp_high))

    violations_df = pd.DataFrame(violations, columns=["feature", "n_violations"])
    violations_df.to_csv(TABLES_DIR / "range_violations.csv", index=False)

    important_features = [
        "EDA_phasic_mean", "EDA_phasic_std", "EDA_mean",
        "BVP_mean", "BVP_peak_freq",
        "TEMP_mean", "TEMP_slope",
        "net_acc_mean"
    ]

    plt.figure(figsize=(16, 12))
    for i, col in enumerate(important_features, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(col)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_distributions.png", dpi=300)
    plt.close()

    plt.figure(figsize=(14, 10))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap for Physiological Features")
    plt.savefig(FIGURES_DIR / "corr_heatmap.png", dpi=300)
    plt.close()

    json_path = log_run_metadata(
        phase_name="phase_A_ingestion_and_eda",
        status="success",
        details={
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "n_range_violations": len(violations_df)
        }
    )

    append_runlog_md_row(
        timestamp_utc=json_path.name.split("_")[-1].replace(".json", ""),
        phase_name="phase_A_ingestion_and_eda",
        status="success",
        notes="Ingestion, schema check, missingness, EDA, range validation completed."
    )

    print(f"Phase A complete. Logged: {json_path}")
    return df

if __name__ == '__main__':
    df = run_phase_A()
    print('rows:', len(df))
