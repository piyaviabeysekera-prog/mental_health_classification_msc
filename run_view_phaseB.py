import pandas as pd
from pathlib import Path

PARQUET = Path("data_stage") / "features" / "merged_with_composites.parquet"
DICT_CSV = Path("reports") / "tables" / "feature_dictionary.csv"
PLOT = Path("reports") / "figures" / "composites_by_label.png"

print(f"Reading parquet: {PARQUET}")
df = pd.read_parquet(PARQUET)

print("\nFirst 5 rows (first 10 columns):")
print(df.iloc[:5, :10].to_string(index=False))

print("\nFeature dictionary head (20 rows):")
fd = pd.read_csv(DICT_CSV)
print(fd.head(20).to_string(index=False))

print("\nComposites plot located at:", PLOT.resolve())
