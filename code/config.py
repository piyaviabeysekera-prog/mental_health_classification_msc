# file: code/config.py
# ============================================
from pathlib import Path

# To centralise all important paths and constants, so that later phases
# reference a single source of truth for the purposes of maintaining a
# reproducible and organised modelling pipeline in this project.

# Project root directory (one level above this file)
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# Core directories
DATA_DIR: Path = ROOT_DIR / "data_stage"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"
RUNS_DIR: Path = REPORTS_DIR / "runs"

# Default random seed for deterministic behaviour
RANDOM_SEED: int = 42

# Physiological sanity ranges used later in QC and EDA stages
PHYSIO_RANGES = {
    "EDA_min": 0.0,           # microsiemens, EDA should not be negative
    "TEMP_min": 30.0,         # degrees Celsius, plausible lower bound
    "TEMP_max": 40.0,         # degrees Celsius, plausible upper bound
    "BVP_freq_min": 0.8,      # Hz, lower heart rate surrogate bound (~48 bpm)
    "BVP_freq_max": 2.5,      # Hz, upper heart rate surrogate bound (~150 bpm)
}

# Default location of the main merged CSV (can be adjusted if needed)
MERGED_CSV_PATH: Path = DATA_DIR / "emoma_csv" / "merged.csv"

# Name of the run log markdown file
RUNLOG_MD_PATH: Path = REPORTS_DIR / "RUNLOG.md"

# file: code/config.py
# ============================================
from pathlib import Path

# To centralise all important paths and constants, so that later phases
# reference a single source of truth for the purposes of maintaining a
# reproducible and organised modelling pipeline in this project.

# Project root directory (one level above this file)
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# Core directories
DATA_DIR: Path = ROOT_DIR / "data_stage"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"
RUNS_DIR: Path = REPORTS_DIR / "runs"

# Default random seed for deterministic behaviour
RANDOM_SEED: int = 42

# Physiological sanity ranges used later in QC and EDA stages
PHYSIO_RANGES = {
    "EDA_min": 0.0,           # microsiemens, EDA should not be negative
    "TEMP_min": 30.0,         # degrees Celsius, plausible lower bound
    "TEMP_max": 40.0,         # degrees Celsius, plausible upper bound
    "BVP_freq_min": 0.8,      # Hz, lower heart rate surrogate bound (~48 bpm)
    "BVP_freq_max": 2.5,      # Hz, upper heart rate surrogate bound (~150 bpm)
}

# Default location of the main merged CSV (can be adjusted if needed)
MERGED_CSV_PATH: Path = DATA_DIR / "emoma_csv" / "merged.csv"

# Name of the run log markdown file
RUNLOG_MD_PATH: Path = REPORTS_DIR / "RUNLOG.md"