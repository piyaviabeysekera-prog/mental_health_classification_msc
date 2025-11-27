# ============================================
# file: code/main_pipeline.py
# ============================================
import argparse
import logging
from datetime import datetime

from .config import ROOT_DIR
from . import utils


def phase_0_scaffolding() -> None:
    """
    To perform one-time project scaffolding actions such as directory
    creation and initial run logging, so that later modelling phases
    can rely on a consistent filesystem layout for the purposes of
    reproducible experimentation in this project.
    """
    utils.init_basic_logging()
    utils.set_global_seeds()

    logging.info("Running Phase 0: scaffolding")
    logging.info(f"Project root resolved as: {ROOT_DIR}")

    # Create standard directory structure
    utils.initialise_scaffolding_directories()
    logging.info("Ensured standard directories exist under the project root.")

    # Ensure RUNLOG.md exists
    utils.ensure_runlog_md()
    logging.info("Ensured RUNLOG.md exists in the reports directory.")

    # Create a dummy run JSON and append to RUNLOG.md
    now_utc = datetime.utcnow().isoformat() + "Z"
    dummy_details = {
        "note": "Initial scaffolding run to verify directory and logging setup.",
    }

    json_path = utils.log_run_metadata(
        phase_name="phase_0_scaffolding",
        status="success",
        details=dummy_details,
    )
    utils.append_runlog_md_row(
        timestamp_utc=now_utc,
        phase_name="phase_0_scaffolding",
        status="success",
        notes="Initial scaffolding and dummy run recorded.",
    )

    logging.info(f"Dummy run metadata written to: {json_path}")
    logging.info("Phase 0 scaffolding completed successfully.\n")


def run_pipeline(selected_phases: list[str]) -> None:
    """
    To route requested phase names to their corresponding functions, so
    that subsets of the pipeline can be executed flexibly for the
    purposes of iterative development and debugging in this project.
    """
    # Build mapping at runtime so later-defined phases (e.g. phase_A)
    # are included even if they're defined after this function in the file.
    phase_map = {"phase_0": phase_0_scaffolding}
    if "phase_A_ingestion_and_eda" in globals():
        phase_map["phase_A"] = globals()["phase_A_ingestion_and_eda"]

    for phase_name in selected_phases:
        if phase_name not in phase_map:
            logging.error(f"Unknown phase name requested: {phase_name}")
            continue
        func = phase_map[phase_name]
        func()


def parse_args() -> argparse.Namespace:
    """
    To parse command-line arguments for selecting pipeline phases, so
    that the same script can drive different parts of the workflow for
    the purposes of interactive experiment control in this project.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Mental health risk stratification pipeline - main driver script."
        )
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["phase_0"],
        help=(
            "List of phase identifiers to run in order. "
            "Example: --phases phase_0 phase_A"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    utils.init_basic_logging()
    logging.info(f"Requested phases: {args.phases}")
    run_pipeline(args.phases)


# ============================================
# PHASE A — INGESTION + SCHEMA + BASIC EDA
# ============================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from .config import (
    MERGED_CSV_PATH,
    TABLES_DIR,
    FIGURES_DIR,
    PHYSIO_RANGES
)
from .utils import (
    ensure_dir,
    log_run_metadata,
    append_runlog_md_row,
    set_global_seeds
)


def phase_A_ingestion_and_eda():
    """
    To load the merged dataset, validate its schema and physiologic plausibility,
    and produce initial exploratory insights so that downstream composite creation
    and modelling phases operate on a verified, clean foundation for the purposes
    of robust and reproducible risk stratification in this project.
    """

    set_global_seeds()
    ensure_dir(TABLES_DIR)
    ensure_dir(FIGURES_DIR)

    # ----------------------------------------
    # 1. INGESTION
    # ----------------------------------------
    df = pd.read_csv(MERGED_CSV_PATH)

    # Required columns
    required_cols = [
        "EDA_phasic_mean", "EDA_phasic_std", "EDA_tonic_std", "EDA_mean",
        "BVP_mean", "BVP_std", "BVP_peak_freq",
        "TEMP_mean", "TEMP_std", "TEMP_min", "TEMP_max", "TEMP_slope",
        "net_acc_mean", "subject", "label"
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Enforce dtype expectations
    numeric_cols = [c for c in required_cols if c not in ["subject", "label"]]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["subject"] = df["subject"].astype(str)
    df["label"] = df["label"].astype(int)

    # ----------------------------------------
    # 2. SCHEMA SUMMARY
    # ----------------------------------------
    schema_summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "min": df.min(numeric_only=True),
        "max": df.max(numeric_only=True)
    })
    schema_summary.to_csv(TABLES_DIR / "schema_summary.csv", index=False)

    # ----------------------------------------
    # 3. MISSINGNESS CHECK
    # ----------------------------------------
    missingness = df.isna().mean().reset_index()
    missingness.columns = ["column", "missing_fraction"]
    missingness.to_csv(TABLES_DIR / "missingness.csv", index=False)

    # ----------------------------------------
    # 4. CLASS BALANCE BY SUBJECT
    # ----------------------------------------
    class_balance = df.groupby("subject")["label"].value_counts().unstack(fill_value=0)
    class_balance.to_csv(TABLES_DIR / "class_balance_by_subject.csv")

    # ----------------------------------------
    # 5. RANGE VIOLATIONS (PHYSIOLOGICAL SANITY)
    # ----------------------------------------
    violations = []

    # EDA non-negativity
    eda_neg = df[df["EDA_mean"] < PHYSIO_RANGES["EDA_min"]].shape[0]
    if eda_neg > 0:
        violations.append(("EDA_mean", eda_neg))

    # TEMP out of plausible human bounds
    temp_low = df[df["TEMP_mean"] < PHYSIO_RANGES["TEMP_min"]].shape[0]
    temp_high = df[df["TEMP_mean"] > PHYSIO_RANGES["TEMP_max"]].shape[0]
    if temp_low > 0: violations.append(("TEMP_mean_low", temp_low))
    if temp_high > 0: violations.append(("TEMP_mean_high", temp_high))

    # BVP_peak_freq out of HRV plausible surrogate band
    bvp_low = df[df["BVP_peak_freq"] < PHYSIO_RANGES["BVP_freq_min"]].shape[0]
    bvp_high = df[df["BVP_peak_freq"] > PHYSIO_RANGES["BVP_freq_max"]].shape[0]
    if bvp_low > 0: violations.append(("BVP_peak_freq_low", bvp_low))
    if bvp_high > 0: violations.append(("BVP_peak_freq_high", bvp_high))

    violations_df = pd.DataFrame(violations, columns=["feature", "n_violations"])
    violations_df.to_csv(TABLES_DIR / "range_violations.csv", index=False)

    # ----------------------------------------
    # 6. BASIC EDA — DISTRIBUTIONS
    # ----------------------------------------
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

    # ----------------------------------------
    # 7. CORRELATION HEATMAP
    # ----------------------------------------
    plt.figure(figsize=(14, 10))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap for Physiological Features")
    plt.savefig(FIGURES_DIR / "corr_heatmap.png", dpi=300)
    plt.close()

    # ----------------------------------------
    # 8. LOG RUN
    # ----------------------------------------
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

phase_map = {
    "phase_0": phase_0_scaffolding,
    "phase_A": phase_A_ingestion_and_eda,
}
# ============================================
# file: code/main_pipeline.py
# ============================================
import argparse
import logging
from datetime import datetime

from .config import ROOT_DIR
from . import utils


def phase_0_scaffolding() -> None:
    """
    To perform one-time project scaffolding actions such as directory
    creation and initial run logging, so that later modelling phases
    can rely on a consistent filesystem layout for the purposes of
    reproducible experimentation in this project.
    """
    utils.init_basic_logging()
    utils.set_global_seeds()

    logging.info("Running Phase 0: scaffolding")
    logging.info(f"Project root resolved as: {ROOT_DIR}")

    # Create standard directory structure
    utils.initialise_scaffolding_directories()
    logging.info("Ensured standard directories exist under the project root.")

    # Ensure RUNLOG.md exists
    utils.ensure_runlog_md()
    logging.info("Ensured RUNLOG.md exists in the reports directory.")

    # Create a dummy run JSON and append to RUNLOG.md
    now_utc = datetime.utcnow().isoformat() + "Z"
    dummy_details = {
        "note": "Initial scaffolding run to verify directory and logging setup.",
    }

    json_path = utils.log_run_metadata(
        phase_name="phase_0_scaffolding",
        status="success",
        details=dummy_details,
    )
    utils.append_runlog_md_row(
        timestamp_utc=now_utc,
        phase_name="phase_0_scaffolding",
        status="success",
        notes="Initial scaffolding and dummy run recorded.",
    )

    logging.info(f"Dummy run metadata written to: {json_path}")
    logging.info("Phase 0 scaffolding completed successfully.\n")


def run_pipeline(selected_phases: list[str]) -> None:
    """
    To route requested phase names to their corresponding functions, so
    that subsets of the pipeline can be executed flexibly for the
    purposes of iterative development and debugging in this project.
    """
    phase_map = {
        "phase_0": phase_0_scaffolding,
        # Placeholders for later phases, to be implemented incrementally:
        # "phase_A": phase_A_ingestion_and_eda,
        # "phase_B": phase_B_composites,
        # "phase_C": phase_C_loso_baselines,
        # "phase_D": phase_D_ensembles_calibration,
        # "phase_E": phase_E_explainability_sensitivity,
        # "phase_F": phase_F_fairness_packaging,
    }

    for phase_name in selected_phases:
        if phase_name not in phase_map:
            logging.error(f"Unknown phase name requested: {phase_name}")
            continue
        func = phase_map[phase_name]
        func()


def parse_args() -> argparse.Namespace:
    """
    To parse command-line arguments for selecting pipeline phases, so
    that the same script can drive different parts of the workflow for
    the purposes of interactive experiment control in this project.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Mental health risk stratification pipeline - main driver script."
        )
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["phase_0"],
        help=(
            "List of phase identifiers to run in order. "
            "Example: --phases phase_0 phase_A"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    utils.init_basic_logging()
    logging.info(f"Requested phases: {args.phases}")
    run_pipeline(args.phases)


# ============================================
# PHASE A — INGESTION + SCHEMA + BASIC EDA
# ============================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from .config import (
    MERGED_CSV_PATH,
    TABLES_DIR,
    FIGURES_DIR,
    PHYSIO_RANGES
)
from .utils import (
    ensure_dir,
    log_run_metadata,
    append_runlog_md_row,
    set_global_seeds
)


def phase_A_ingestion_and_eda():
    """
    To load the merged dataset, validate its schema and physiologic plausibility,
    and produce initial exploratory insights so that downstream composite creation
    and modelling phases operate on a verified, clean foundation for the purposes
    of robust and reproducible risk stratification in this project.
    """

    set_global_seeds()
    ensure_dir(TABLES_DIR)
    ensure_dir(FIGURES_DIR)

    # ----------------------------------------
    # 1. INGESTION
    # ----------------------------------------
    df = pd.read_csv(MERGED_CSV_PATH)

    # Required columns
    required_cols = [
        "EDA_phasic_mean", "EDA_phasic_std", "EDA_tonic_std", "EDA_mean",
        "BVP_mean", "BVP_std", "BVP_peak_freq",
        "TEMP_mean", "TEMP_std", "TEMP_min", "TEMP_max", "TEMP_slope",
        "net_acc_mean", "subject", "label"
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Enforce dtype expectations
    numeric_cols = [c for c in required_cols if c not in ["subject", "label"]]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["subject"] = df["subject"].astype(str)
    df["label"] = df["label"].astype(int)

    # ----------------------------------------
    # 2. SCHEMA SUMMARY
    # ----------------------------------------
    schema_summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "min": df.min(numeric_only=True),
        "max": df.max(numeric_only=True)
    })
    schema_summary.to_csv(TABLES_DIR / "schema_summary.csv", index=False)

    # ----------------------------------------
    # 3. MISSINGNESS CHECK
    # ----------------------------------------
    missingness = df.isna().mean().reset_index()
    missingness.columns = ["column", "missing_fraction"]
    missingness.to_csv(TABLES_DIR / "missingness.csv", index=False)

    # ----------------------------------------
    # 4. CLASS BALANCE BY SUBJECT
    # ----------------------------------------
    class_balance = df.groupby("subject")["label"].value_counts().unstack(fill_value=0)
    class_balance.to_csv(TABLES_DIR / "class_balance_by_subject.csv")

    # ----------------------------------------
    # 5. RANGE VIOLATIONS (PHYSIOLOGICAL SANITY)
    # ----------------------------------------
    violations = []

    # EDA non-negativity
    eda_neg = df[df["EDA_mean"] < PHYSIO_RANGES["EDA_min"]].shape[0]
    if eda_neg > 0:
        violations.append(("EDA_mean", eda_neg))

    # TEMP out of plausible human bounds
    temp_low = df[df["TEMP_mean"] < PHYSIO_RANGES["TEMP_min"]].shape[0]
    temp_high = df[df["TEMP_mean"] > PHYSIO_RANGES["TEMP_max"]].shape[0]
    if temp_low > 0: violations.append(("TEMP_mean_low", temp_low))
    if temp_high > 0: violations.append(("TEMP_mean_high", temp_high))

    # BVP_peak_freq out of HRV plausible surrogate band
    bvp_low = df[df["BVP_peak_freq"] < PHYSIO_RANGES["BVP_freq_min"]].shape[0]
    bvp_high = df[df["BVP_peak_freq"] > PHYSIO_RANGES["BVP_freq_max"]].shape[0]
    if bvp_low > 0: violations.append(("BVP_peak_freq_low", bvp_low))
    if bvp_high > 0: violations.append(("BVP_peak_freq_high", bvp_high))

    violations_df = pd.DataFrame(violations, columns=["feature", "n_violations"])
    violations_df.to_csv(TABLES_DIR / "range_violations.csv", index=False)

    # ----------------------------------------
    # 6. BASIC EDA — DISTRIBUTIONS
    # ----------------------------------------
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

    # ----------------------------------------
    # 7. CORRELATION HEATMAP
    # ----------------------------------------
    plt.figure(figsize=(14, 10))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap for Physiological Features")
    plt.savefig(FIGURES_DIR / "corr_heatmap.png", dpi=300)
    plt.close()

    # ----------------------------------------
    # 8. LOG RUN
    # ----------------------------------------
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

phase_map = {
    "phase_0": phase_0_scaffolding,
    "phase_A": phase_A_ingestion_and_eda,
}

