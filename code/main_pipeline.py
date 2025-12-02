"""Main pipeline entrypoint for the project.

Defines phases and exposes `run_pipeline` to execute them.
"""
from __future__ import annotations

from .composites import run_phase_B
from .baselines import run_phase_C
from .ensembles import run_phase_D
from .explainability import (
    phase_E_explainability_and_sensitivity,
)

import argparse
import logging
from datetime import datetime
from typing import Callable, Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from .config import (
    ROOT_DIR,
    MERGED_CSV_PATH,
    TABLES_DIR,
    FIGURES_DIR,
    PHYSIO_RANGES,
)
from . import utils


def phase_0_scaffolding() -> None:
    utils.init_basic_logging()
    utils.set_global_seeds()
    logging.info("Running Phase 0: scaffolding")
    utils.initialise_scaffolding_directories()
    utils.ensure_runlog_md()

    now_utc = datetime.utcnow().isoformat() + "Z"
    details = {"note": "Initial scaffolding run."}
    json_path = utils.log_run_metadata("phase_0_scaffolding", "success", details)
    utils.append_runlog_md_row(now_utc, "phase_0_scaffolding", "success", "Scaffolding run recorded.")
    logging.info(f"Dummy run metadata written to: {json_path}")


def phase_A_ingestion_and_eda() -> pd.DataFrame:
    utils.set_global_seeds()
    utils.ensure_dir(TABLES_DIR)
    utils.ensure_dir(FIGURES_DIR)

    df = pd.read_csv(MERGED_CSV_PATH)
    df["subject"] = df["subject"].astype(str)
    df["label"] = df["label"].astype(int)

    # Simple outputs
    df.to_csv(TABLES_DIR / "ingested_preview.csv", index=False)
    logging.info("Phase A: ingestion complete")
    return df


def phase_B_composites() -> None:
    df = run_phase_B()
    logging.info(f"Phase B completed. Enriched dataframe shape: {df.shape}")


def phase_C_loso_baselines() -> None:
    run_phase_C()


def phase_D_ensembles() -> None:
    run_phase_D()

def phase_E() -> None:
    """
    To delegate to the dedicated explainability and sensitivity phase so that
    Phase E can be invoked from the generic pipeline runner for the purposes
    of SHAP-based interpretation and composite robustness analysis in this
    project.
    """
    phase_E_explainability_and_sensitivity()


def run_pipeline(selected_phases: List[str]) -> None:
    phase_map: Dict[str, Callable] = {
        "phase_0": phase_0_scaffolding,
        "phase_A": phase_A_ingestion_and_eda,
        "phase_B": phase_B_composites,
        "phase_C": phase_C_loso_baselines,
        "phase_D": phase_D_ensembles,
        "phase_E": phase_E_explainability_and_sensitivity,  # or phase_E if you used the wrapper
    }
   


    for phase_name in selected_phases:
        if phase_name not in phase_map:
            logging.error(f"Unknown phase: {phase_name}")
            continue
        phase_map[phase_name]()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Pipeline driver."))
    parser.add_argument("--phases", nargs="+", default=["phase_0"], help=("List of phases to run."),)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    utils.init_basic_logging()
    logging.info(f"Requested phases: {args.phases}")
    run_pipeline(args.phases)
"""
Main pipeline entrypoint for the project. Defines scaffolding and
ingestion/EDA phases and exposes a simple `run_pipeline` function
that maps phase names to functions.
"""
from __future__ import annotations
from .composites import run_phase_B
from .baselines import run_phase_C
from .ensembles import run_phase_D

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    ROOT_DIR,
    MERGED_CSV_PATH,
    TABLES_DIR,
    FIGURES_DIR,
    PHYSIO_RANGES,
)
from . import utils


def phase_0_scaffolding() -> None:
    utils.init_basic_logging()
    utils.set_global_seeds()

    logging.info("Running Phase 0: scaffolding")
    logging.info(f"Project root resolved as: {ROOT_DIR}")

    utils.initialise_scaffolding_directories()
    logging.info("Ensured standard directories exist under the project root.")

    utils.ensure_runlog_md()
    logging.info("Ensured RUNLOG.md exists in the reports directory.")

    now_utc = datetime.utcnow().isoformat() + "Z"
    details = {"note": "Initial scaffolding run to verify directory and logging setup."}

    json_path = utils.log_run_metadata("phase_0_scaffolding", "success", details)
    utils.append_runlog_md_row(now_utc, "phase_0_scaffolding", "success", "Initial scaffolding and dummy run recorded.")
    logging.info(f"Dummy run metadata written to: {json_path}")


def phase_A_ingestion_and_eda() -> pd.DataFrame:
    """Load merged CSV, validate schema and ranges, produce simple EDA tables/figures."""
    utils.set_global_seeds()
    utils.ensure_dir(TABLES_DIR)
    utils.ensure_dir(FIGURES_DIR)

    df = pd.read_csv(MERGED_CSV_PATH)

    required_cols = [
        "EDA_phasic_mean", "EDA_phasic_std", "EDA_tonic_std", "EDA_mean",
        "BVP_mean", "BVP_std", "BVP_peak_freq",
        "TEMP_mean", "TEMP_std", "TEMP_min", "TEMP_max", "TEMP_slope",
        "net_acc_mean", "subject", "label",
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
        "max": df.max(numeric_only=True),
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
    if temp_low > 0:
        violations.append(("TEMP_mean_low", temp_low))
    if temp_high > 0:
        violations.append(("TEMP_mean_high", temp_high))

    bvp_low = df[df["BVP_peak_freq"] < PHYSIO_RANGES["BVP_freq_min"]].shape[0]
    bvp_high = df[df["BVP_peak_freq"] > PHYSIO_RANGES["BVP_freq_max"]].shape[0]
    if bvp_low > 0:
        violations.append(("BVP_peak_freq_low", bvp_low))
    if bvp_high > 0:
        violations.append(("BVP_peak_freq_high", bvp_high))

    violations_df = pd.DataFrame(violations, columns=["feature", "n_violations"])
    violations_df.to_csv(TABLES_DIR / "range_violations.csv", index=False)

    important_features = [
        "EDA_phasic_mean", "EDA_phasic_std", "EDA_mean",
        "BVP_mean", "BVP_peak_freq",
        "TEMP_mean", "TEMP_slope",
        "net_acc_mean",
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

    json_path = utils.log_run_metadata(
        phase_name="phase_A_ingestion_and_eda",
        status="success",
        details={
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "n_range_violations": len(violations_df),
        },
    )

    utils.append_runlog_md_row(
        timestamp_utc=json_path.name.split("_")[-1].replace(".json", ""),
        phase_name="phase_A_ingestion_and_eda",
        status="success",
        notes="Ingestion, schema check, missingness, EDA, range validation completed.",
    )

    print(f"Phase A complete. Logged: {json_path}")
    return df


def phase_B_composites() -> None:
    """
    To execute composite index construction and related artefact
    generation, so that the modelling pipeline can use SRI, RS and PL
    alongside raw features for the purposes of ensemble-based risk
    stratification in this project.
    """
    df = run_phase_B()
    print(f"Phase B completed. Enriched dataframe shape: {df.shape}")


def phase_C_loso_baselines() -> None:
    """
    To invoke the LOSO folding, scaling, shuffle control and baseline
    training routine, so that a single phase identifier triggers the
    full baseline modelling setup for the purposes of evaluating model
    validity in this project.
    """
    run_phase_C()


def phase_D_ensembles() -> None:
    """
    To invoke feature-family ablations, ensemble training, calibration
    and tier derivation, so that a single phase identifier generates
    the main predictive and calibration results required for this
    project.
    """
    run_phase_D()


def run_pipeline(selected_phases: List[str]) -> None:
    phase_map: Dict[str, Callable] = {
        "phase_0": phase_0_scaffolding,
        "phase_A": phase_A_ingestion_and_eda,
        "phase_B": phase_B_composites,
        "phase_C": phase_C_loso_baselines,
        "phase_D": phase_D_ensembles,
        "phase_E": phase_E_explainability_and_sensitivity,
    }

    for phase_name in selected_phases:
        if phase_name not in phase_map or phase_map[phase_name] is None:
            logging.error(f"Unknown phase name requested: {phase_name}")
            continue
        func = phase_map[phase_name]
        func()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Mental health risk stratification pipeline - main driver script."))
    parser.add_argument("--phases", nargs="+", default=["phase_0"], help=("List of phase identifiers to run in order. Example: --phases phase_0 phase_A"),)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    utils.init_basic_logging()
    logging.info(f"Requested phases: {args.phases}")
    run_pipeline(args.phases)
"""
Main pipeline entrypoint for the project. Defines scaffolding and
ingestion/EDA phases and exposes a simple `run_pipeline` function
that maps phase names to functions.
"""
from __future__ import annotations
from .composites import run_phase_B
from .baselines import run_phase_C
from .ensembles import run_phase_D

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    ROOT_DIR,
    MERGED_CSV_PATH,
    TABLES_DIR,
    FIGURES_DIR,
    PHYSIO_RANGES,
)
from . import utils


def phase_0_scaffolding() -> None:
    utils.init_basic_logging()
    utils.set_global_seeds()

    logging.info("Running Phase 0: scaffolding")
    logging.info(f"Project root resolved as: {ROOT_DIR}")

    utils.initialise_scaffolding_directories()
    logging.info("Ensured standard directories exist under the project root.")

    utils.ensure_runlog_md()
    logging.info("Ensured RUNLOG.md exists in the reports directory.")

    now_utc = datetime.utcnow().isoformat() + "Z"
    details = {"note": "Initial scaffolding run to verify directory and logging setup."}

    json_path = utils.log_run_metadata("phase_0_scaffolding", "success", details)
    utils.append_runlog_md_row(now_utc, "phase_0_scaffolding", "success", "Initial scaffolding and dummy run recorded.")
    logging.info(f"Dummy run metadata written to: {json_path}")


def phase_A_ingestion_and_eda() -> pd.DataFrame:
    """Load merged CSV, validate schema and ranges, produce simple EDA tables/figures."""
    utils.set_global_seeds()
    utils.ensure_dir(TABLES_DIR)
    utils.ensure_dir(FIGURES_DIR)

    df = pd.read_csv(MERGED_CSV_PATH)

    required_cols = [
        "EDA_phasic_mean", "EDA_phasic_std", "EDA_tonic_std", "EDA_mean",
        "BVP_mean", "BVP_std", "BVP_peak_freq",
        "TEMP_mean", "TEMP_std", "TEMP_min", "TEMP_max", "TEMP_slope",
        "net_acc_mean", "subject", "label",
    ]
"""
Main pipeline entrypoint for the project. Defines scaffolding and
ingestion/EDA phases and exposes a simple `run_pipeline` function
that maps phase names to functions.
"""
from __future__ import annotations
from .composites import run_phase_B
from .baselines import run_phase_C
from .ensembles import run_phase_D

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    ROOT_DIR,
    MERGED_CSV_PATH,
    TABLES_DIR,
    FIGURES_DIR,
    PHYSIO_RANGES,
)
from . import utils


def phase_0_scaffolding() -> None:
    utils.init_basic_logging()
    utils.set_global_seeds()

    logging.info("Running Phase 0: scaffolding")
    logging.info(f"Project root resolved as: {ROOT_DIR}")

    utils.initialise_scaffolding_directories()
    logging.info("Ensured standard directories exist under the project root.")

    utils.ensure_runlog_md()
    logging.info("Ensured RUNLOG.md exists in the reports directory.")

    now_utc = datetime.utcnow().isoformat() + "Z"
    details = {"note": "Initial scaffolding run to verify directory and logging setup."}

    json_path = utils.log_run_metadata("phase_0_scaffolding", "success", details)
    utils.append_runlog_md_row(now_utc, "phase_0_scaffolding", "success", "Initial scaffolding and dummy run recorded.")
    logging.info(f"Dummy run metadata written to: {json_path}")


def phase_A_ingestion_and_eda() -> pd.DataFrame:
    """Load merged CSV, validate schema and ranges, produce simple EDA tables/figures."""
    utils.set_global_seeds()
    utils.ensure_dir(TABLES_DIR)
    utils.ensure_dir(FIGURES_DIR)

    df = pd.read_csv(MERGED_CSV_PATH)

    required_cols = [
        "EDA_phasic_mean", "EDA_phasic_std", "EDA_tonic_std", "EDA_mean",
        "BVP_mean", "BVP_std", "BVP_peak_freq",
        "TEMP_mean", "TEMP_std", "TEMP_min", "TEMP_max", "TEMP_slope",
        "net_acc_mean", "subject", "label",
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
        "max": df.max(numeric_only=True),
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
    if temp_low > 0:
        violations.append(("TEMP_mean_low", temp_low))
    if temp_high > 0:
        violations.append(("TEMP_mean_high", temp_high))

    bvp_low = df[df["BVP_peak_freq"] < PHYSIO_RANGES["BVP_freq_min"]].shape[0]
    bvp_high = df[df["BVP_peak_freq"] > PHYSIO_RANGES["BVP_freq_max"]].shape[0]
    if bvp_low > 0:
        violations.append(("BVP_peak_freq_low", bvp_low))
    if bvp_high > 0:
        violations.append(("BVP_peak_freq_high", bvp_high))

    violations_df = pd.DataFrame(violations, columns=["feature", "n_violations"])
    violations_df.to_csv(TABLES_DIR / "range_violations.csv", index=False)

    important_features = [
        "EDA_phasic_mean", "EDA_phasic_std", "EDA_mean",
        "BVP_mean", "BVP_peak_freq",
        "TEMP_mean", "TEMP_slope",
        "net_acc_mean",
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

    json_path = utils.log_run_metadata(
        phase_name="phase_A_ingestion_and_eda",
        status="success",
        details={
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "n_range_violations": len(violations_df),
        },
    )

    utils.append_runlog_md_row(
        timestamp_utc=json_path.name.split("_")[-1].replace(".json", ""),
        phase_name="phase_A_ingestion_and_eda",
        status="success",
        notes="Ingestion, schema check, missingness, EDA, range validation completed.",
    )

    print(f"Phase A complete. Logged: {json_path}")
    return df


def phase_B_composites() -> None:
    """
    To execute composite index construction and related artefact
    generation, so that the modelling pipeline can use SRI, RS and PL
    alongside raw features for the purposes of ensemble-based risk
    stratification in this project.
    """
    df = run_phase_B()
    print(f"Phase B completed. Enriched dataframe shape: {df.shape}")


def phase_C_loso_baselines() -> None:
    """
    To invoke the LOSO folding, scaling, shuffle control and baseline
    training routine, so that a single phase identifier triggers the
    full baseline modelling setup for the purposes of evaluating model
    validity in this project.
    """
    run_phase_C()


def phase_D_ensembles() -> None:
    """
    To invoke feature-family ablations, ensemble training, calibration
    and tier derivation, so that a single phase identifier generates
    the main predictive and calibration results required for this
    project.
    """
    run_phase_D()


def run_pipeline(selected_phases: List[str]) -> None:
    phase_map: Dict[str, Callable] = {
        "phase_0": phase_0_scaffolding,
        "phase_A": phase_A_ingestion_and_eda,
        "phase_B": phase_B_composites,
        "phase_C": phase_C_loso_baselines,
        "phase_D": phase_D_ensembles,
        "phase_E": phase_E_explainability_and_sensitivity,
    }

    for phase_name in selected_phases:
        if phase_name not in phase_map or phase_map[phase_name] is None:
            logging.error(f"Unknown phase name requested: {phase_name}")
            continue
        func = phase_map[phase_name]
        func()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Mental health risk stratification pipeline - main driver script."))
    parser.add_argument("--phases", nargs="+", default=["phase_0"], help=("List of phase identifiers to run in order. Example: --phases phase_0 phase_A"),)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    utils.init_basic_logging()
    logging.info(f"Requested phases: {args.phases}")
    run_pipeline(args.phases)
        df = run_phase_B()

        print(f"Phase B completed. Enriched dataframe shape: {df.shape}")





