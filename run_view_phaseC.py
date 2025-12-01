import pandas as pd
from pathlib import Path

TABLES = Path("reports") / "tables"
folds_csv = TABLES / "loso_folds.csv"
shuffle_csv = TABLES / "shuffle_control.csv"
baselines_csv = TABLES / "loso_baselines.csv"

print(f"Reading: {folds_csv}")
folds = pd.read_csv(folds_csv)
print("\nLOSO folds (first 10 rows):")
print(folds.head(10).to_string(index=False))

print("\nReading shuffle control:", shuffle_csv)
shuffle = pd.read_csv(shuffle_csv)
mean_rows = shuffle[shuffle["fold_id"] == "mean"]
example_rows = shuffle[shuffle["fold_id"] != "mean"].head(3)
print("\nShuffle control - mean rows:")
print(mean_rows.to_string(index=False))
print("\nShuffle control - example folds:")
print(example_rows.to_string(index=False))

print("\nReading baselines:", baselines_csv)
baselines = pd.read_csv(baselines_csv)
mean_rows_bl = baselines[baselines["fold_id"] == "mean"]
example_rows_bl = baselines[baselines["fold_id"] != "mean"].head(3)
print("\nBaselines - mean rows:")
print(mean_rows_bl.to_string(index=False))
print("\nBaselines - example folds:")
print(example_rows_bl.to_string(index=False))
