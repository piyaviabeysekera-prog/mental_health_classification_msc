import pandas as pd
import numpy as np

# Read the summary
summary = pd.read_csv('reports/tables/loso_all_models_summary.csv')

# Read the fold metrics
fold_metrics = pd.read_csv('reports/tables/loso_all_models_fold_metrics.csv')

# Extract voting ensemble metrics (pre-calibration)
ensemble_pre = fold_metrics[(fold_metrics['model'] == 'voting_ensemble') & (fold_metrics['stage'] == 'pre_calibration')]
ensemble_post = fold_metrics[(fold_metrics['model'] == 'voting_ensemble') & (fold_metrics['stage'] == 'post_calibration')]

print('VOTING ENSEMBLE PERFORMANCE (Pre-Calibration)')
print('=' * 70)
print(f'F1-Macro:   {ensemble_pre["f1_macro"].mean():.4f} ± {ensemble_pre["f1_macro"].std():.4f}')
print(f'AUROC:      {ensemble_pre["auroc_macro"].mean():.4f} ± {ensemble_pre["auroc_macro"].std():.4f}')
print(f'PR-AUC:     {ensemble_pre["pr_auc_macro"].mean():.4f} ± {ensemble_pre["pr_auc_macro"].std():.4f}')
print(f'Folds:      {ensemble_pre["fold_id"].nunique()}')
print(f'Total evals: {len(ensemble_pre)}')

print('\n\nVOTING ENSEMBLE PERFORMANCE (Post-Calibration)')
print('=' * 70)
print(f'F1-Macro:   {ensemble_post["f1_macro"].mean():.4f} ± {ensemble_post["f1_macro"].std():.4f}')
print(f'AUROC:      {ensemble_post["auroc_macro"].mean():.4f} ± {ensemble_post["auroc_macro"].std():.4f}')
print(f'PR-AUC:     {ensemble_post["pr_auc_macro"].mean():.4f} ± {ensemble_post["pr_auc_macro"].std():.4f}')
print(f'Folds:      {ensemble_post["fold_id"].nunique()}')
print(f'Total evals: {len(ensemble_post)}')

print('\n\nCALIBRATION IMPACT')
print('=' * 70)
f1_delta = ensemble_post['f1_macro'].mean() - ensemble_pre['f1_macro'].mean()
auroc_delta = ensemble_post['auroc_macro'].mean() - ensemble_pre['auroc_macro'].mean()
pr_auc_delta = ensemble_post['pr_auc_macro'].mean() - ensemble_pre['pr_auc_macro'].mean()

print(f'F1 Change:      {f1_delta:+.4f} ({(f1_delta/ensemble_pre["f1_macro"].mean()*100):+.2f}%)')
print(f'AUROC Change:   {auroc_delta:+.4f} ({(auroc_delta/ensemble_pre["auroc_macro"].mean()*100):+.2f}%)')
print(f'PR-AUC Change:  {pr_auc_delta:+.4f} ({(pr_auc_delta/ensemble_pre["pr_auc_macro"].mean()*100):+.2f}%)')

print('\n\nALL MODELS SUMMARY')
print('=' * 70)
print(summary.to_string(index=False))
