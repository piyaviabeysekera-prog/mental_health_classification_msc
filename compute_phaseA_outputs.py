import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path('.')
MERGED = ROOT / 'data_stage' / 'emoma_csv' / 'merged.csv'
TABLES = ROOT / 'reports' / 'tables'
FIGURES = ROOT / 'reports' / 'figures'
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

PHYSIO_RANGES = {
    "EDA_min": 0.0,
    "TEMP_min": 30.0,
    "TEMP_max": 40.0,
    "BVP_freq_min": 0.8,
    "BVP_freq_max": 2.5,
}

print('Reading', MERGED)
df = pd.read_csv(MERGED)
print('rows,cols', df.shape)

required_cols = [
    "EDA_phasic_mean", "EDA_phasic_std", "EDA_tonic_std", "EDA_mean",
    "BVP_mean", "BVP_std", "BVP_peak_freq",
    "TEMP_mean", "TEMP_std", "TEMP_min", "TEMP_max", "TEMP_slope",
    "net_acc_mean", "subject", "label"
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print('Missing required columns:', missing_cols)

numeric_cols = [c for c in required_cols if c not in ['subject','label']]
# coerce numeric
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df['subject'] = df['subject'].astype(str)
try:
    df['label'] = df['label'].astype(int)
except Exception:
    pass

schema_summary = pd.DataFrame({
    'column': df.columns,
    'dtype': df.dtypes.astype(str),
    'n_unique': df.nunique(),
    'min': df.min(numeric_only=True),
    'max': df.max(numeric_only=True)
})
schema_summary.to_csv(TABLES / 'schema_summary.csv', index=False)

missingness = df.isna().mean().reset_index()
missingness.columns = ['column','missing_fraction']
missingness.to_csv(TABLES / 'missingness.csv', index=False)

class_balance = df.groupby('subject')['label'].value_counts().unstack(fill_value=0)
class_balance.to_csv(TABLES / 'class_balance_by_subject.csv')

violations = []
eda_neg = df[df['EDA_mean'] < PHYSIO_RANGES['EDA_min']].shape[0]
if eda_neg>0: violations.append(('EDA_mean', eda_neg))

temp_low = df[df['TEMP_mean'] < PHYSIO_RANGES['TEMP_min']].shape[0]
temp_high = df[df['TEMP_mean'] > PHYSIO_RANGES['TEMP_max']].shape[0]
if temp_low>0: violations.append(('TEMP_mean_low', temp_low))
if temp_high>0: violations.append(('TEMP_mean_high', temp_high))

bvp_low = df[df['BVP_peak_freq'] < PHYSIO_RANGES['BVP_freq_min']].shape[0]
bvp_high = df[df['BVP_peak_freq'] > PHYSIO_RANGES['BVP_freq_max']].shape[0]
if bvp_low>0: violations.append(('BVP_peak_freq_low', bvp_low))
if bvp_high>0: violations.append(('BVP_peak_freq_high', bvp_high))

violations_df = pd.DataFrame(violations, columns=['feature','n_violations'])
violations_df.to_csv(TABLES / 'range_violations.csv', index=False)

important_features = [
    'EDA_phasic_mean','EDA_phasic_std','EDA_mean',
    'BVP_mean','BVP_peak_freq',
    'TEMP_mean','TEMP_slope','net_acc_mean'
]

plt.figure(figsize=(16,12))
for i,col in enumerate(important_features,1):
    plt.subplot(3,3,i)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(col)
plt.tight_layout()
plt.savefig(FIGURES / 'eda_distributions.png', dpi=300)
plt.close()

plt.figure(figsize=(14,10))
sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', center=0)
plt.title('Correlation Heatmap for Physiological Features')
plt.savefig(FIGURES / 'corr_heatmap.png', dpi=300)
plt.close()

print('Wrote tables to', TABLES)
print('Wrote figures to', FIGURES)
