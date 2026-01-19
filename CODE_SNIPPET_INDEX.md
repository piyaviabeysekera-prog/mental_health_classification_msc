 i# Code Snippet Index for Thesis Write-Up

**Purpose**: Quick lookup table mapping each phase to relevant source files (with line ranges) and output files.  
**How to use**: For each phase section in your thesis, refer to the code lines listed here, copy them into your write-up, and cite the files.

---

## **Phase 0: Scaffolding**

### Relevant Input Files
- **`code/main_pipeline.py`** → Lines 28–41 (function `phase_0_scaffolding()`)
  - Sets up logging, seeds, directories, and run metadata
- **`code/utils.py`** → Lines (key helper functions)
  - `init_basic_logging()` — initializes logging
  - `set_global_seeds()` — sets random seeds for reproducibility
  - `initialise_scaffolding_directories()` — creates `reports/`, `models/`, `data_stage/`
  - `ensure_runlog_md()` — ensures `RUNLOG.md` exists

### Code Snippet Locations
| File | Lines | What It Does |
|------|-------|-------------|
| `code/main_pipeline.py` | 28–41 | Phase 0 entry point & orchestration |
| `code/utils.py` | (search for function) | Helper functions for scaffolding |
| `code/config.py` | (top section) | Directory path definitions |

### Output Files Generated
- `reports/RUNLOG.md` — run log with timestamp & phase metadata
- `reports/run_metadata/phase_0_scaffolding_*.json` — metadata JSON per run

---

## **Phase A: Ingestion & Exploratory Data Analysis (EDA)**

### Relevant Input Files
- **`code/main_pipeline.py`** → Lines 44–54 (function `phase_A_ingestion_and_eda()`)
  - Loads CSV, coerces types, saves preview
  - Main orchestration wrapper (calls actual EDA logic from phase A implementation)

**Note**: Phase A in `main_pipeline.py` is simplified. The full EDA logic was originally in earlier versions.  
For complete details, see the actual implementations in Phase A runner scripts (if preserved) or refer to:
- **Data loading**: `code/config.py` (path definition) + pandas read_csv
- **Type coercion & schema validation**: Phase A function in main_pipeline
- **Visualization generation**: matplotlib & seaborn code (likely in Phase A modules or runners)

### Code Snippet Locations
| File | Lines | What It Does |
|------|-------|-------------|
| `code/main_pipeline.py` | 44–54 | Phase A entry point |
| `code/config.py` | (search MERGED_CSV_PATH) | Input data path definition |
| `code/utils.py` | (search ensure_dir) | Directory creation helper |

### Output Files Generated
- `reports/tables/ingested_preview.csv` — first few rows of data
- `reports/tables/schema_summary.csv` — data types & ranges (if created)
- `reports/tables/missingness.csv` — % missing per feature (if created)
- `reports/tables/class_balance_by_subject.csv` — per-subject label distribution (if created)
- `reports/figures/eda_distributions.png` — histograms (9 features)
- `reports/figures/corr_heatmap.png` — correlation matrix heatmap

---

## **Phase B: Composite Feature Construction**

### Relevant Input Files
- **`code/main_pipeline.py`** → Lines 57–59 (function `phase_B_composites()`)
  - Calls `run_phase_B()` which orchestrates composite creation
- **`code/composites.py`** → Full file (primary implementation)
  - `run_phase_B()` — main entry point
  - `_create_sri()`, `_create_rs()`, `_create_pl()` — composite index creation
  - Feature family definitions and merging logic

### Code Snippet Locations
| File | Lines | What It Does |
|------|-------|-------------|
| `code/main_pipeline.py` | 57–59 | Phase B entry point |
| `code/composites.py` | (search for `run_phase_B`) | Composite orchestration |
| `code/composites.py` | (search for `_create_sri`, `_create_rs`, `_create_pl`) | Individual composite calculations |
| `code/composites.py` | (end of file) | Merging composites with base features |

### Output Files Generated
- `data_stage/features/merged_with_composites.parquet` — enriched dataset (base features + SRI, RS, PL)
- `reports/tables/feature_dictionary.csv` — feature definitions & composite recipes (if created)
- `reports/figures/composites_by_label.png` — composite distributions by class (if created)

---

## **Phase C: LOSO Baselines (Leave-One-Subject-Out Evaluation)**

### Relevant Input Files
- **`code/main_pipeline.py`** → Lines 62–63 (function `phase_C_loso_baselines()`)
  - Calls `run_phase_C()` from baselines module
- **`code/baselines.py`** → Full file (primary implementation)
  - `run_phase_C()` — main orchestration (lines ~486–523)
  - `_build_loso_folds()` — creates LOSO fold assignments (lines ~312–328)
  - `_train_scalers_and_models()` — per-fold training loop (lines ~334–482)
  - `_evaluate_predictions()` — computes metrics (lines ~63–75)
  - `_compute_macro_pr_auc()` — precision-recall AUC helper (lines ~49–62)

### Code Snippet Locations
| File | Lines | What It Does |
|------|-------|-------------|
| `code/main_pipeline.py` | 62–63 | Phase C entry point |
| `code/baselines.py` | ~486–523 | `run_phase_C()` — full orchestration |
| `code/baselines.py` | ~312–328 | `_build_loso_folds()` — creates fold assignments |
| `code/baselines.py` | ~334–482 | `_train_scalers_and_models()` — training loop (contains scaling, imputation, model training, shuffled controls) |
| `code/baselines.py` | ~63–75 | `_evaluate_predictions()` — metric computation (F1, AUROC, PR-AUC) |

### Output Files Generated
- `reports/tables/loso_folds.csv` — fold assignments (subject per fold)
- `reports/tables/loso_baselines.csv` — per-fold baseline metrics (fold_id, test_subject, model, F1, AUROC, PR-AUC)
- `reports/tables/shuffle_control.csv` — shuffled-label control metrics (validates real learning)
- `models/scalers/imputer_fold_*.pkl` — saved imputers (per fold)
- `models/scalers/scaler_fold_*.pkl` — saved StandardScaler (per fold)
- `models/baselines/logreg_fold_*.pkl` — saved LogisticRegression models
- `models/baselines/rf_fold_*.pkl` — saved RandomForest models

---

## **Phase D: Ensemble Models, Calibration & Tiering**

### Relevant Input Files
- **`code/main_pipeline.py`** → Lines 66–67 (function `phase_D_ensembles()`)
  - Calls `run_phase_D()` from ensembles module
- **`code/ensembles.py`** → Full file (primary implementation)
  - `run_phase_D()` — main orchestration (search for function definition)
  - `_run_ensembles_and_calibration()` — per-fold ensemble + calibration loop (lines ~217–400+)
  - `_run_feature_family_ablations()` — feature family sensitivity tests (lines ~160–185)
  - `_run_loso_for_family()` — train model on single feature family (lines ~116–145)
  - `_define_feature_families()` — feature groupings (lines ~100–115)
  - `_evaluate_predictions()` — metric computation (lines shared with baselines.py)

### Code Snippet Locations
| File | Lines | What It Does |
|------|-------|-------------|
| `code/main_pipeline.py` | 66–67 | Phase D entry point |
| `code/ensembles.py` | (search `def run_phase_D`) | Full Phase D orchestration |
| `code/ensembles.py` | ~217–400+ | `_run_ensembles_and_calibration()` — ensemble training, calibration, tiering |
| `code/ensembles.py` | ~160–185 | `_run_feature_family_ablations()` — ablation loop |
| `code/ensembles.py` | ~116–145 | `_run_loso_for_family()` — per-family LOSO training |
| `code/ensembles.py` | ~100–115 | `_define_feature_families()` — feature family definitions |

### Key Code Sections to Extract
- **VotingClassifier creation** (search for `VotingClassifier` in ensembles.py)
  ```python
  ensemble = VotingClassifier(
      estimators=[('rf', rf), ('xgb', xgb)],
      voting='soft'
  )
  ensemble.fit(X_train_scaled, y_train)
  ```

- **Calibration logic** (search for `CalibratedClassifierCV`)
  ```python
  calibrator = CalibratedClassifierCV(ensemble, method='sigmoid', cv=5)
  calibrator.fit(X_train_scaled, y_train)
  ```

- **Tier derivation** (search for tier/cost threshold logic)

### Output Files Generated
- `reports/tables/ensembles_per_fold.csv` — per-fold ensemble metrics (F1, AUROC, PR-AUC, calibration scores)
- `reports/tables/feature_family_ablation.csv` — performance per family (family, fold_id, F1, AUROC, PR-AUC)
- `reports/tables/calibration.csv` — calibration statistics (pre/post calibration)
- `reports/tables/tiers_costs.csv` — risk tier thresholds & associated costs (if created)
- `reports/figures/calibration_plots.png` — reliability diagrams (predicted vs. actual probability)
- `models/ensembles/ensemble_fold_*.pkl` — saved VotingClassifier (per fold)
- `models/ensembles/calibrator_fold_*.pkl` — saved CalibratedClassifierCV (per fold)

---

## **Phase E: Explainability (SHAP) & Sensitivity Analysis**

### Relevant Input Files
- **`code/main_pipeline.py`** → Lines 70–71 (function `phase_E()`)
  - Calls `phase_E_explainability_and_sensitivity()` from explainability module
- **`code/explainability.py`** → Full file (primary implementation)
  - `phase_E_explainability_and_sensitivity()` — main orchestration (search for function)
  - `run_global_shap()` — SHAP importance computation (lines search for function)
  - `run_sensitivity()` — feature family removal experiments (lines search for function)
  - `_load_feature_dataframe()` — data loading helper
  - `_build_rf_model()` — RandomForest trainer
  - `split_features_labels()` — data prep helper

### Code Snippet Locations
| File | Lines | What It Does |
|------|-------|-------------|
| `code/main_pipeline.py` | 70–71 | Phase E entry point |
| `code/explainability.py` | (search `phase_E_explainability_and_sensitivity`) | Phase E orchestration |
| `code/explainability.py` | (search `run_global_shap`) | SHAP computation |
| `code/explainability.py` | (search `run_sensitivity`) | Sensitivity experiments (feature removal) |
| `code/explainability.py` | (search `_build_rf_model`) | RandomForest training |

### Key Code Sections to Extract
- **SHAP computation** (search for TreeExplainer)
  ```python
  import shap
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(X_sample)
  mean_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=0)
  ```

- **Sensitivity loop** (search for feature family removal in run_sensitivity)

### Output Files Generated
- `reports/tables/shap_top_features.csv` — ranked features by SHAP importance (feature_name, mean_abs_shap)
- `reports/figures/shap_global.png` — bar plot of top SHAP features
- `reports/tables/sensitivity.csv` — F1-macro per family removed (family, f1_with_family_removed, f1_drop_pct)
- `reports/figures/sensitivity_spider.png` — radar plot of sensitivity (% performance drop per family)
- `models/explainability/rf_explainability_model.joblib` — saved RandomForest model (for reproducibility)

---

## **Phase F: Fairness / Uncertainty Proxy + Reject-Option + Packaging**

### Relevant Input Files
- **`code/main_pipeline.py`** → Lines 74–75 (function `phase_F()`)
  - Calls `phase_F_fairness_and_packaging()` from fairness_packaging module
- **`code/fairness_packaging.py`** → Full file (primary implementation)
  - `phase_F_fairness_and_packaging()` — main orchestration (search for function)
  - `compute_fairness_summary()` — per-subject uncertainty aggregation (lines search for function)
  - `compute_reject_stats()` — reject-option band analysis (lines search for function)
  - `package_for_thesis()` — file packaging & manifest generation (lines search for function)
  - `_load_enriched_df()` — data loading helper
  - `_build_rf()` — RandomForest builder

### Code Snippet Locations
| File | Lines | What It Does |
|------|-------|-------------|
| `code/main_pipeline.py` | 74–75 | Phase F entry point |
| `code/fairness_packaging.py` | (search `phase_F_fairness_and_packaging`) | Phase F orchestration |
| `code/fairness_packaging.py` | (search `compute_fairness_summary`) | Fairness/uncertainty computation |
| `code/fairness_packaging.py` | (search `compute_reject_stats`) | Reject-option analysis |
| `code/fairness_packaging.py` | (search `package_for_thesis`) | Packaging & manifest |

### Key Code Sections to Extract
- **Cross-validated probabilities** (search for cross_val_predict)
  ```python
  from sklearn.model_selection import cross_val_predict
  y_proba_oof = cross_val_predict(rf, X, y, method='predict_proba', cv=5)
  ```

- **Per-subject uncertainty aggregation** (search for groupby and variance in compute_fairness_summary)

- **Reject-option bands** (search for band logic in compute_reject_stats)
  ```python
  in_band = (y_proba[:, 2] >= band_low) & (y_proba[:, 2] <= band_high)
  coverage = (~in_band).sum() / len(y)
  error_outside = (y_pred[~in_band] != y_true[~in_band]).mean()
  ```

- **Manifest generation** (search for json.dump and git hash)

### Output Files Generated
- `reports/tables/fairness_summary.csv` — per-subject stats (subject, n_windows, accuracy, variance, std_prob_class2, mean_true_label)
- `reports/tables/reject_stats.csv` — reject-option trade-offs (band_low, band_high, coverage, error_rate)
- `reports/tables/manifest.json` — inventory (generated_at_utc, git_hash, tables list, figures list)
- `reports/tables/thesis_final/` — curated tables (copies of key CSVs)
- `reports/figures/thesis_final/` — curated figures (copies of key PNGs)

---

## **Quick Reference: File Locations**

### Core Implementation Files (Phase Logic)
- **Phase 0**: `code/main_pipeline.py`, `code/utils.py`, `code/config.py`
- **Phase A**: `code/main_pipeline.py`, `code/config.py`
- **Phase B**: `code/composites.py`, `code/config.py`
- **Phase C**: `code/baselines.py`
- **Phase D**: `code/ensembles.py`, `code/baselines.py` (shared helpers)
- **Phase E**: `code/explainability.py`
- **Phase F**: `code/fairness_packaging.py`

### Orchestration File
- **`code/main_pipeline.py`**: Central hub (all phases callable via `run_pipeline(selected_phases)`)

### Configuration & Utilities
- **`code/config.py`**: Path definitions, hyperparameters, feature ranges
- **`code/utils.py`**: Logging, seeding, directory creation, metadata logging

---

## **For Your Thesis Write-Up**

### Workflow
1. Identify which phase section you're writing (e.g., "Methods: LOSO Baselines")
2. Refer to this index and locate the relevant files + line ranges (e.g., Phase C → `code/baselines.py` lines 312–328)
3. Open the actual file in your editor and copy the snippet
4. Paste into your thesis with a caption like:
   ```
   Code 3.2.1: LOSO fold construction (code/baselines.py, lines 312–328)
   ```
5. Cite the git hash for reproducibility

### Template for Thesis Section
```markdown
## 3.2 Phase C: LOSO Baselines

### 3.2.1 Fold Construction
[Reference from index: code/baselines.py, lines 312–328]
[Paste code snippet here]

### 3.2.2 Scaling & Training
[Reference from index: code/baselines.py, lines 334–482]
[Paste code snippet here]

### 3.2.3 Outputs
[List from index: loso_folds.csv, loso_baselines.csv, shuffle_control.csv]
```

---

**Last Updated**: December 2025  
**Git Hash**: (Include in thesis for reproducibility)

