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