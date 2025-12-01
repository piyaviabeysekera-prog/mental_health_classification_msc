import pandas as pd
from pathlib import Path

ROOT = Path('.')
MERGED = ROOT / 'data_stage' / 'emoma_csv' / 'merged.csv'
TABLES = ROOT / 'reports' / 'tables'

# Read data and tables
print('Reading merged CSV from:', MERGED)
df = pd.read_csv(MERGED)

schema_summary = pd.read_csv(TABLES / 'schema_summary.csv')
class_balance = pd.read_csv(TABLES / 'class_balance_by_subject.csv')
violations_df = pd.read_csv(TABLES / 'range_violations.csv')

# 1. df.head()
print('\n--- df.head() ---')
print(df.head())
print("\nDescription: First 5 rows of the merged dataset; shows feature columns, subject and label. Useful to inspect raw values, types, and obvious anomalies (NaNs, extreme values).\n")

# 2. schema_summary.head(10) twice (per request)
print('--- schema_summary.head(10) ---')
print(schema_summary.head(10))
print('\nDescription: Schema summary lists each column, its dtype, number of unique values, and observed min/max (numeric). Helps identify mis-typed columns and ranges.\n')

print('--- schema_summary.head(10) (repeat) ---')
print(schema_summary.head(10))

# 3. class_balance.head(10) twice
print('\n--- class_balance.head(10) ---')
print(class_balance.head(10))
print('\nDescription: Class balance by subject. Each row lists counts per label for a subject — good to spot subjects with imbalanced labels or missing classes.\n')

print('--- class_balance.head(10) (repeat) ---')
print(class_balance.head(10))

# 4. violations_df
print('\n--- violations_df (range_violations.csv) ---')
print(violations_df)
print('\nDescription: Counts of rows violating physiologic sanity ranges. Use to decide filtering, unit corrections, or further QC.\n')

print('Done')
