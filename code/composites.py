# ============================================
# file: code/composites.py
# ============================================
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    MERGED_CSV_PATH,
    TABLES_DIR,
    FIGURES_DIR,
)
from .utils import ensure_dir, set_global_seeds

import matplotlib.pyplot as plt
import seaborn as sns


def load_base_dataframe() -> pd.DataFrame:
    """
    To load the base merged dataset and enforce the minimal schema
    required for composite index construction, so that the composite
    engineering logic operates on a consistent and correctly typed
    dataframe for the purposes of reproducible feature generation in
    this project.
    """
    df = pd.read_csv(MERGED_CSV_PATH)

    required_cols = [
        "EDA_phasic_mean", "EDA_phasic_std", "EDA_tonic_std", "EDA_mean",
        "BVP_mean",
        "TEMP_mean", "TEMP_slope",
        "net_acc_mean",
        "subject", "label"
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for composites: {missing_cols}")

    numeric_cols = [c for c in required_cols if c not in ["subject", "label"]]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["subject"] = df["subject"].astype(str)
    df["label"] = df["label"].astype(int)

    # Optional: drop leftover index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def compute_sri(df: pd.DataFrame) -> pd.DataFrame:
    """
    To construct the Stress Reactivity Index (SRI) as the product of
    z-scored phasic amplitude and a spike-rate proxy, so that each
    window receives a single scalar capturing the joint intensity and
    variability of phasic EDA for the purposes of representing stress
    reactivity in this project.
    """
    # phasic amplitude proxy: EDA_phasic_mean
    # spike-rate proxy: EDA_phasic_std (higher variability ~ more spikes)
    sri_cols = ["EDA_phasic_mean", "EDA_phasic_std"]

    scaler = StandardScaler()
    z_vals = scaler.fit_transform(df[sri_cols])

    df["z_phasic_amp"] = z_vals[:, 0]
    df["z_spike_proxy"] = z_vals[:, 1]

    df["SRI"] = df["z_phasic_amp"] * df["z_spike_proxy"]
    return df


def compute_rs(df: pd.DataFrame) -> pd.DataFrame:
    """
    To construct the Recovery Speed (RS) index as a monotonic proxy
    of 1/tau, where tau reflects slower recovery when tonic EDA is
    volatile and temperature trends away from baseline, so that higher
    RS represents faster and more stable recovery for the purposes of
    resilience-oriented risk stratification in this project.
    """
    # Use EDA_tonic_std (baseline instability) and TEMP_slope (trend)
    # High EDA_tonic_std  -> slower recovery
    # High positive TEMP_slope (warming back up) -> faster recovery
    cols = []
    if "EDA_tonic_std" in df.columns:
        cols.append("EDA_tonic_std")
    cols.append("TEMP_slope")

    scaler = StandardScaler()
    z_rs = scaler.fit_transform(df[cols].fillna(method="ffill").fillna(method="bfill"))

    if "EDA_tonic_std" in cols:
        df["z_EDA_tonic_std"] = z_rs[:, cols.index("EDA_tonic_std")]
        df["z_TEMP_slope"] = z_rs[:, cols.index("TEMP_slope")]
        # tau_proxy increases with instability and decreases with favourable slope
        tau_proxy = df["z_EDA_tonic_std"] - df["z_TEMP_slope"]
    else:
        df["z_TEMP_slope"] = z_rs[:, 0]
        tau_proxy = -df["z_TEMP_slope"]

    # RS is defined such that higher RS = faster, more stable recovery (lower tau_proxy)
    df["RS"] = -tau_proxy
    return df


def compute_pl(df: pd.DataFrame) -> pd.DataFrame:
    """
    To construct the Physiological Lability (PL) index by aggregating
    the variance of first differences across key features within each
    subject, so that each window receives a scalar capturing short-term
    volatility in autonomic and behavioural signals for the purposes of
    identifying unstable regulation patterns in this project.
    """
    pl_features = ["EDA_mean", "BVP_mean", "TEMP_mean", "net_acc_mean"]

    # Sort within subject by original row order to approximate temporal order
    df = df.sort_values(by=["subject"]).copy()

    diff_cols: List[str] = []
    for feat in pl_features:
        if feat in df.columns:
            diff_col = f"{feat}_diff"
            df[diff_col] = df.groupby("subject")[feat].diff()
            diff_cols.append(diff_col)

    if diff_cols:
        df["PL"] = df[diff_cols].pow(2).mean(axis=1)
    else:
        df["PL"] = np.nan

    return df


def build_feature_dictionary(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    To create a feature dictionary table describing raw and composite
    variables, so that the modelling pipeline remains interpretable and
    transparently documented for the purposes of thesis reporting in
    this project.
    """
    rows: List[Tuple[str, str, str]] = []

    for col in df.columns:
        if col in ["SRI", "RS", "PL"]:
            feature_type = "composite"
            if col == "SRI":
                desc = (
                    "Stress Reactivity Index: product of z-scored EDA_phasic_mean "
                    "and EDA_phasic_std, representing intensity x variability of "
                    "phasic EDA per window."
                )
            elif col == "RS":
                desc = (
                    "Recovery Speed: inverse recovery proxy derived from z-scored "
                    "EDA_tonic_std and TEMP_slope, where higher values indicate "
                    "faster and more stable return towards baseline."
                )
            else:  # PL
                desc = (
                    "Physiological Lability: mean squared first difference across "
                    "EDA_mean, BVP_mean, TEMP_mean and net_acc_mean within each "
                    "subject, capturing short-term volatility in physiology."
                )
        elif col in ["subject", "label"]:
            feature_type = "identifier" if col == "subject" else "target"
            desc = "Subject identifier." if col == "subject" else "Window-level class label."
        else:
            feature_type = "raw_feature"
            desc = "Raw or precomputed feature from merged.csv."

        rows.append((col, feature_type, desc))

    dict_df = pd.DataFrame(rows, columns=["feature_name", "feature_type", "description"])
    ensure_dir(output_path.parent)
    dict_df.to_csv(output_path, index=False)
    return dict_df


def plot_composites_by_label(df: pd.DataFrame, output_path: Path) -> None:
    """
    To visualise the distributions of SRI, RS and PL across class labels,
    so that their discriminative behaviour can be inspected for the
    purposes of validating composite usefulness in this project.
    """
    ensure_dir(output_path.parent)

    plt.figure(figsize=(15, 10))

    composites = ["SRI", "RS", "PL"]
    for i, comp in enumerate(composites, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(data=df, x="label", y=comp)
        plt.title(comp)
        plt.xlabel("Label")
        plt.ylabel(comp)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_phase_B() -> pd.DataFrame:
    """
    To orchestrate composite calculation, feature dictionary creation
    and composite sanity visualisation, so that a single callable
    function can produce the enriched dataset and documentation for the
    purposes of downstream modelling in this project.
    """
    set_global_seeds()
    ensure_dir(TABLES_DIR)
    ensure_dir(FIGURES_DIR)

    df = load_base_dataframe()
    df = compute_sri(df)
    df = compute_rs(df)
    df = compute_pl(df)

    # Save enriched dataset as parquet
    features_dir = Path(MERGED_CSV_PATH).parents[1] / "features"
    ensure_dir(features_dir)
    enriched_path = features_dir / "merged_with_composites.parquet"
    df.to_parquet(enriched_path, index=False)

    # Feature dictionary
    dict_path = TABLES_DIR / "feature_dictionary.csv"
    _ = build_feature_dictionary(df, dict_path)

    # Composite sanity plots
    composites_plot_path = FIGURES_DIR / "composites_by_label.png"
    plot_composites_by_label(df, composites_plot_path)

    return df
# ============================================
# file: code/composites.py
# ============================================
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    MERGED_CSV_PATH,
    TABLES_DIR,
    FIGURES_DIR,
)
from .utils import ensure_dir, set_global_seeds

import matplotlib.pyplot as plt
import seaborn as sns


def load_base_dataframe() -> pd.DataFrame:
    """
    To load the base merged dataset and enforce the minimal schema
    required for composite index construction, so that the composite
    engineering logic operates on a consistent and correctly typed
    dataframe for the purposes of reproducible feature generation in
    this project.
    """
    df = pd.read_csv(MERGED_CSV_PATH)

    required_cols = [
        "EDA_phasic_mean", "EDA_phasic_std", "EDA_tonic_std", "EDA_mean",
        "BVP_mean",
        "TEMP_mean", "TEMP_slope",
        "net_acc_mean",
        "subject", "label"
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for composites: {missing_cols}")

    numeric_cols = [c for c in required_cols if c not in ["subject", "label"]]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["subject"] = df["subject"].astype(str)
    df["label"] = df["label"].astype(int)

    # Optional: drop leftover index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def compute_sri(df: pd.DataFrame) -> pd.DataFrame:
    """
    To construct the Stress Reactivity Index (SRI) as the product of
    z-scored phasic amplitude and a spike-rate proxy, so that each
    window receives a single scalar capturing the joint intensity and
    variability of phasic EDA for the purposes of representing stress
    reactivity in this project.
    """
    # phasic amplitude proxy: EDA_phasic_mean
    # spike-rate proxy: EDA_phasic_std (higher variability ~ more spikes)
    sri_cols = ["EDA_phasic_mean", "EDA_phasic_std"]

    scaler = StandardScaler()
    z_vals = scaler.fit_transform(df[sri_cols])

    df["z_phasic_amp"] = z_vals[:, 0]
    df["z_spike_proxy"] = z_vals[:, 1]

    df["SRI"] = df["z_phasic_amp"] * df["z_spike_proxy"]
    return df


def compute_rs(df: pd.DataFrame) -> pd.DataFrame:
    """
    To construct the Recovery Speed (RS) index as a monotonic proxy
    of 1/tau, where tau reflects slower recovery when tonic EDA is
    volatile and temperature trends away from baseline, so that higher
    RS represents faster and more stable recovery for the purposes of
    resilience-oriented risk stratification in this project.
    """
    # Use EDA_tonic_std (baseline instability) and TEMP_slope (trend)
    # High EDA_tonic_std  -> slower recovery
    # High positive TEMP_slope (warming back up) -> faster recovery
    cols = []
    if "EDA_tonic_std" in df.columns:
        cols.append("EDA_tonic_std")
    cols.append("TEMP_slope")

    scaler = StandardScaler()
    z_rs = scaler.fit_transform(df[cols].fillna(method="ffill").fillna(method="bfill"))

    if "EDA_tonic_std" in cols:
        df["z_EDA_tonic_std"] = z_rs[:, cols.index("EDA_tonic_std")]
        df["z_TEMP_slope"] = z_rs[:, cols.index("TEMP_slope")]
        # tau_proxy increases with instability and decreases with favourable slope
        tau_proxy = df["z_EDA_tonic_std"] - df["z_TEMP_slope"]
    else:
        df["z_TEMP_slope"] = z_rs[:, 0]
        tau_proxy = -df["z_TEMP_slope"]

    # RS is defined such that higher RS = faster, more stable recovery (lower tau_proxy)
    df["RS"] = -tau_proxy
    return df


def compute_pl(df: pd.DataFrame) -> pd.DataFrame:
    """
    To construct the Physiological Lability (PL) index by aggregating
    the variance of first differences across key features within each
    subject, so that each window receives a scalar capturing short-term
    volatility in autonomic and behavioural signals for the purposes of
    identifying unstable regulation patterns in this project.
    """
    pl_features = ["EDA_mean", "BVP_mean", "TEMP_mean", "net_acc_mean"]

    # Sort within subject by original row order to approximate temporal order
    df = df.sort_values(by=["subject"]).copy()

    diff_cols: List[str] = []
    for feat in pl_features:
        if feat in df.columns:
            diff_col = f"{feat}_diff"
            df[diff_col] = df.groupby("subject")[feat].diff()
            diff_cols.append(diff_col)

    if diff_cols:
        df["PL"] = df[diff_cols].pow(2).mean(axis=1)
    else:
        df["PL"] = np.nan

    return df


def build_feature_dictionary(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    To create a feature dictionary table describing raw and composite
    variables, so that the modelling pipeline remains interpretable and
    transparently documented for the purposes of thesis reporting in
    this project.
    """
    rows: List[Tuple[str, str, str]] = []

    for col in df.columns:
        if col in ["SRI", "RS", "PL"]:
            feature_type = "composite"
            if col == "SRI":
                desc = (
                    "Stress Reactivity Index: product of z-scored EDA_phasic_mean "
                    "and EDA_phasic_std, representing intensity x variability of "
                    "phasic EDA per window."
                )
            elif col == "RS":
                desc = (
                    "Recovery Speed: inverse recovery proxy derived from z-scored "
                    "EDA_tonic_std and TEMP_slope, where higher values indicate "
                    "faster and more stable return towards baseline."
                )
            else:  # PL
                desc = (
                    "Physiological Lability: mean squared first difference across "
                    "EDA_mean, BVP_mean, TEMP_mean and net_acc_mean within each "
                    "subject, capturing short-term volatility in physiology."
                )
        elif col in ["subject", "label"]:
            feature_type = "identifier" if col == "subject" else "target"
            desc = "Subject identifier." if col == "subject" else "Window-level class label."
        else:
            feature_type = "raw_feature"
            desc = "Raw or precomputed feature from merged.csv."

        rows.append((col, feature_type, desc))

    dict_df = pd.DataFrame(rows, columns=["feature_name", "feature_type", "description"])
    ensure_dir(output_path.parent)
    dict_df.to_csv(output_path, index=False)
    return dict_df


def plot_composites_by_label(df: pd.DataFrame, output_path: Path) -> None:
    """
    To visualise the distributions of SRI, RS and PL across class labels,
    so that their discriminative behaviour can be inspected for the
    purposes of validating composite usefulness in this project.
    """
    ensure_dir(output_path.parent)

    plt.figure(figsize=(15, 10))

    composites = ["SRI", "RS", "PL"]
    for i, comp in enumerate(composites, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(data=df, x="label", y=comp)
        plt.title(comp)
        plt.xlabel("Label")
        plt.ylabel(comp)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_phase_B() -> pd.DataFrame:
    """
    To orchestrate composite calculation, feature dictionary creation
    and composite sanity visualisation, so that a single callable
    function can produce the enriched dataset and documentation for the
    purposes of downstream modelling in this project.
    """
    set_global_seeds()
    ensure_dir(TABLES_DIR)
    ensure_dir(FIGURES_DIR)

    df = load_base_dataframe()
    df = compute_sri(df)
    df = compute_rs(df)
    df = compute_pl(df)

    # Save enriched dataset as parquet
    features_dir = Path(MERGED_CSV_PATH).parents[1] / "features"
    ensure_dir(features_dir)
    enriched_path = features_dir / "merged_with_composites.parquet"
    df.to_parquet(enriched_path, index=False)

    # Feature dictionary
    dict_path = TABLES_DIR / "feature_dictionary.csv"
    _ = build_feature_dictionary(df, dict_path)

    # Composite sanity plots
    composites_plot_path = FIGURES_DIR / "composites_by_label.png"
    plot_composites_by_label(df, composites_plot_path)

    return df
