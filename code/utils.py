# ============================================
# file: code/utils.py
# ============================================
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Relative import from the same package
from .config import (
    ROOT_DIR,
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    RUNS_DIR,
    RUNLOG_MD_PATH,
    RANDOM_SEED,
)


def ensure_dir(path: Path) -> Path:
    """
    To guarantee that a directory path exists on disk, so that later
    save operations do not fail unexpectedly for the purposes of
    making the pipeline robust to missing folders in this project.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seeds(seed: int = RANDOM_SEED) -> None:
    """
    To enforce a consistent random seed across Python, NumPy and other
    libraries, so that experimental results can be reproduced for the
    purposes of stable and auditable model comparisons in this project.
    """
    import random

    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def get_git_hash() -> Optional[str]:
    """
    To capture the current git commit hash, so that generated artefacts
    can be tied back to a specific code version for the purposes of
    traceable and versioned experiment logging in this project.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(ROOT_DIR),
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def init_basic_logging() -> None:
    """
    To configure a simple logging setup for console output, so that
    phase-level progress messages are consistently formatted for the
    purposes of quick inspection and debugging in this project.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def log_run_metadata(
    phase_name: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    To record structured metadata about a pipeline phase execution, so
    that each run can be audited, filtered and referenced later for the
    purposes of documenting the modelling lifecycle in this project.
    """
    ensure_dir(RUNS_DIR)

    timestamp = datetime.utcnow().isoformat() + "Z"
    git_hash = get_git_hash()

    payload: Dict[str, Any] = {
        "timestamp_utc": timestamp,
        "phase": phase_name,
        "status": status,
        "git_hash": git_hash,
    }
    if details:
        payload["details"] = details

    # File name based on time and phase
    safe_phase = phase_name.replace(" ", "_")
    json_path = RUNS_DIR / f"run_{safe_phase}_{timestamp.replace(':', '-')}.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return json_path


def ensure_runlog_md() -> None:
    """
    To create an initial RUNLOG.md file if it does not exist, so that
    high-level run summaries can be written in a human-readable format
    for the purposes of tracking important experiments in this project.
    """
    ensure_dir(REPORTS_DIR)

    if not RUNLOG_MD_PATH.exists():
        header = (
            "# Run Log\n\n"
            "| Timestamp (UTC) | Phase | Status | Notes |\n"
            "|-----------------|-------|--------|-------|\n"
        )
        RUNLOG_MD_PATH.write_text(header, encoding="utf-8")


def append_runlog_md_row(
    timestamp_utc: str,
    phase_name: str,
    status: str,
    notes: str = "",
) -> None:
    """
    To append a single markdown table row to RUNLOG.md, so that core
    information about each executed phase is visible at a glance for
    the purposes of quick manual review in this project.
    """
    ensure_runlog_md()
    row = f"| {timestamp_utc} | {phase_name} | {status} | {notes} |\n"
    with RUNLOG_MD_PATH.open("a", encoding="utf-8") as f:
        f.write(row)


def initialise_scaffolding_directories() -> None:
    """
    To enforce the standard project directory structure under the repo,
    so that all subsequent phases save artefacts into predictable paths
    for the purposes of keeping the modelling pipeline organised in
    this project.
    """
    ensure_dir(DATA_DIR / "emoma_csv")
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(TABLES_DIR)
    ensure_dir(RUNS_DIR)
# ============================================
# file: code/utils.py
# ============================================
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Relative import from the same package
from .config import (
    ROOT_DIR,
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    RUNS_DIR,
    RUNLOG_MD_PATH,
    RANDOM_SEED,
)


def ensure_dir(path: Path) -> Path:
    """
    To guarantee that a directory path exists on disk, so that later
    save operations do not fail unexpectedly for the purposes of
    making the pipeline robust to missing folders in this project.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seeds(seed: int = RANDOM_SEED) -> None:
    """
    To enforce a consistent random seed across Python, NumPy and other
    libraries, so that experimental results can be reproduced for the
    purposes of stable and auditable model comparisons in this project.
    """
    import random

    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def get_git_hash() -> Optional[str]:
    """
    To capture the current git commit hash, so that generated artefacts
    can be tied back to a specific code version for the purposes of
    traceable and versioned experiment logging in this project.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(ROOT_DIR),
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def init_basic_logging() -> None:
    """
    To configure a simple logging setup for console output, so that
    phase-level progress messages are consistently formatted for the
    purposes of quick inspection and debugging in this project.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def log_run_metadata(
    phase_name: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    To record structured metadata about a pipeline phase execution, so
    that each run can be audited, filtered and referenced later for the
    purposes of documenting the modelling lifecycle in this project.
    """
    ensure_dir(RUNS_DIR)

    timestamp = datetime.utcnow().isoformat() + "Z"
    git_hash = get_git_hash()

    payload: Dict[str, Any] = {
        "timestamp_utc": timestamp,
        "phase": phase_name,
        "status": status,
        "git_hash": git_hash,
    }
    if details:
        payload["details"] = details

    # File name based on time and phase
    safe_phase = phase_name.replace(" ", "_")
    json_path = RUNS_DIR / f"run_{safe_phase}_{timestamp.replace(':', '-')}.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return json_path


def ensure_runlog_md() -> None:
    """
    To create an initial RUNLOG.md file if it does not exist, so that
    high-level run summaries can be written in a human-readable format
    for the purposes of tracking important experiments in this project.
    """
    ensure_dir(REPORTS_DIR)

    if not RUNLOG_MD_PATH.exists():
        header = (
            "# Run Log\n\n"
            "| Timestamp (UTC) | Phase | Status | Notes |\n"
            "|-----------------|-------|--------|-------|\n"
        )
        RUNLOG_MD_PATH.write_text(header, encoding="utf-8")


def append_runlog_md_row(
    timestamp_utc: str,
    phase_name: str,
    status: str,
    notes: str = "",
) -> None:
    """
    To append a single markdown table row to RUNLOG.md, so that core
    information about each executed phase is visible at a glance for
    the purposes of quick manual review in this project.
    """
    ensure_runlog_md()
    row = f"| {timestamp_utc} | {phase_name} | {status} | {notes} |\n"
    with RUNLOG_MD_PATH.open("a", encoding="utf-8") as f:
        f.write(row)


def initialise_scaffolding_directories() -> None:
    """
    To enforce the standard project directory structure under the repo,
    so that all subsequent phases save artefacts into predictable paths
    for the purposes of keeping the modelling pipeline organised in
    this project.
    """
    ensure_dir(DATA_DIR / "emoma_csv")
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(TABLES_DIR)
    ensure_dir(RUNS_DIR)