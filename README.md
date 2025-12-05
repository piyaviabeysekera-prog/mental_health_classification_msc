Mental Health Risk Stratification — MSc Project

This repository contains the code, data processing, modelling, explainability and fairness/packaging steps used for an MSc project that proposes an ensemble machine-learning approach for insurance risk-stratification of mental health risk using wearable data.

This README provides a phase-wise breakdown (Phase 0, A, B, C, D, E, F) mapping the code to the conceptual steps you can use in a viva or the thesis write-up. For each phase I list the purpose, the main files involved, the key operations (1,2,3 style), inputs and outputs, and short notes you can use when explaining the work.

---

**Quick Start**

- Prepare a Python environment with the required packages (example):

```powershell
python -m pip install -r requirements.txt
```

- To run one or more phases via the main driver:

```powershell
python code/main_pipeline.py --phases phase_A
python code/main_pipeline.py --phases phase_A phase_B phase_C
```

If you prefer to run the existing per-phase runner scripts present in the repo root (e.g. `run_phaseA.py`, `run_phaseD_runner.py`), those also invoke the code in `code/` and are preserved.

**Dependencies (high-level)**: Python 3.8+; pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, shap, xgboost. Use the provided environment/requirements if available.

---

**Project layout (important files)**

- `code/` : main python package containing the pipeline modules.
- `data_stage/` : prepared features (e.g. `merged_with_composites.parquet`).
- `models/` : saved model artefacts (per-fold ensembles, explainability RFs).
- `reports/tables/` and `reports/figures/` : CSV/PNG outputs for the thesis.

---

**Phase 0 — Scaffolding**

- Purpose: Prepare the project directories, logging and basic run metadata.
- Main files: `code/main_pipeline.py`, `code/utils.py`, `code/config.py`.
- Steps:
	1. Initialise logging and deterministic seeds.
	2. Create standard directories under `reports/`, `models/`, `data_stage/`.
 3. Record a small run metadata JSON and append a row to `RUNLOG.md`.
- Inputs: none (scaffolding routine only).
- Outputs: run metadata JSON, `RUNLOG.md` entry.
- Viva notes: emphasise reproducibility and that every pipeline run writes metadata to aid result provenance.

**Phase A — Ingestion & Exploratory Data Analysis (EDA)**

- Purpose: Load the merged features CSV, validate schema & ranges, and produce primary EDA tables/figures used to understand the sensor-derived features and class balance.
- Main files: `code/main_pipeline.py` (phase entry), `code/config.py` (paths & ranges), `code/utils.py` (helpers).
- Steps:
	1. Read `MERGED_CSV_PATH` into a DataFrame and coerce types for `subject` and `label`.
	2. Produce schema summary, missingness table, class-balance by subject and range-violation checks.
	3. Save EDA figures (histograms, correlation heatmap) to `reports/figures/` and tables to `reports/tables/`.
- Inputs: `merged.csv` (or `data_stage/features/merged_with_composites.parquet` depending on workflow).
- Outputs: `reports/tables/schema_summary.csv`, `missingness.csv`, `class_balance_by_subject.csv`, `reports/figures/eda_distributions.png`, `reports/figures/corr_heatmap.png`.
- Viva notes: explain how features were sanity-checked (physiological ranges in `config.py`) and why per-subject balance matters for LOSO.

**Phase B — Composite Feature Construction**

- Purpose: Compute composite indices (e.g. SRI, RS, PL) from base features and produce an enriched feature table used downstream by models.
- Main files: `code/composites.py`, `code/utils.py`, `code/config.py`.
- Steps:
	1. Load base features and create composite indices (domain-motivated transformations aggregating multiple sensors into a single index).
	2. Merge composite indices with the base feature table to produce an enriched dataset.
	3. Persist the enriched dataset (parquet/csv) for modelling phases and create a feature dictionary if needed.
- Inputs: base merged features file.
- Outputs: `data_stage/features/merged_with_composites.parquet` (or equivalent), feature dictionary tables and composite plots.
- Viva notes: highlight the motivation for composites (reduce dimensionality, capture domain constructs) and describe how they were validated.

**Phase C — LOSO Baselines (Leave-One-Subject-Out evaluation)**

- Purpose: Establish baseline predictive performance with careful cross-validation to avoid leakage — LOSO ensures per-subject independence during evaluation.
- Main files: `code/baselines.py`, `code/utils.py`, `code/config.py`.
- Steps:
	1. Build LOSO splits where each subject is held out in turn as the test set.
 2. Train baseline models (e.g. RandomForest, XGBoost) within each fold using appropriate scaling and pipeline steps.
	3. Collect per-fold performance metrics, calibration statistics and save results to `reports/tables/`.
- Inputs: enriched feature table from Phase B.
- Outputs: `reports/tables/loso_baselines.csv`, calibration tables/plots.
- Viva notes: discuss why LOSO is appropriate for wearable/subject-level generalisation and the precautions taken (no leakage in preprocessing, seed control).

**Phase D — Ensemble Models, Calibration & Tiering**

- Purpose: Train and combine multiple models into an ensemble, calibrate output probabilities and derive risk tiers suitable for insurance decisioning.
- Main files: `code/ensembles.py`, `code/utils.py`, `code/config.py`, `run_phaseD_runner.py` (runner preserved)
- Steps:
	1. Train per-fold ensemble models using selected feature families (feature-family ablations are supported to assess contribution).
	2. Calibrate ensemble probabilities and compute tier boundaries or cost-driven thresholds.
	3. Save per-fold models into `models/` and write ensemble results and calibration plots to `reports/`.
- Inputs: enriched dataset, baseline/fold metadata.
- Outputs: `models/` (saved ensembles), `reports/tables/ensembles_per_fold.csv`, `reports/figures/calibration_plots.png`, tier/cost tables.
- Viva notes: explain ensemble composition, why calibration matters for risk stratification, and how tier thresholds were chosen/validated.

**Phase E — Explainability (SHAP) & Sensitivity Analysis**

- Purpose: Produce model-agnostic, shardable explainability outputs to support interpretability claims (global SHAP for feature importance) and sensitivity experiments to test robustness when removing feature families.
- Main files: `code/explainability.py`, `joblib` model persistence in `models/`.
- Steps:
	1. Train a RandomForest on the enriched dataset (or load a persisted explainability model).
	2. Compute SHAP values (sampled stratified subset) and save global mean-absolute SHAP importance ranking.
	3. Run sensitivity experiments (re-train/evauate with feature families removed) and summarise impact on performance.
- Inputs: enriched dataset, saved RF model if available.
- Outputs: `reports/tables/shap_top_features.csv`, `reports/figures/shap_global.png`, `reports/tables/sensitivity.csv`, `reports/figures/sensitivity_spider.png`.
- Viva notes: be ready to explain SHAP basics (local vs global importance) and why mean absolute SHAP is used for ranking. Also discuss sensitivity experiments and what they imply about feature family utility.

**Phase F — Fairness / Uncertainty Proxy + Reject-Option + Packaging**

- Purpose: Estimate an uncertainty/fairness proxy using cross-validated probabilistic RandomForest (out-of-fold probabilities), evaluate a reject-option band (abstain when probability near 0.5) and package the curated thesis-ready artefacts plus a `manifest.json` describing the final files.
- Main files: `code/fairness_packaging.py`, plus the final manifest in `reports/tables/manifest.json`.
- Steps:
	1. Run `cross_val_predict(..., method='predict_proba')` to obtain out-of-fold class probabilities; aggregate per-subject statistics (variance/mean of probabilities) as an uncertainty proxy.
	2. For several symmetric bands around 0.5 (e.g., 0.45-0.55), compute coverage and error rates with and without abstention (reject-option analysis) and write `reject_stats.csv`.
	3. Collect the final curated tables/figures list and copy them to `reports/tables/thesis_final/` and `reports/figures/thesis_final/`, writing `manifest.json` with git hash and timestamp.
- Inputs: enriched dataset and optionally trained models.
- Outputs: `reports/tables/fairness_summary.csv`, `reports/tables/reject_stats.csv`, `reports/tables/manifest.json`, plus copies under `thesis_final/`.
- Viva notes: discuss the rationale for the uncertainty proxy (per-subject variance), how reject-option can reduce error at the cost of coverage, and how this connects to insurance operation (handling uncertain cases more carefully).

---

**Notes for viva preparation (short talking points)**

- Data & preprocessing: explain how wearable signals were transformed, why composite indices were created, and how range checks/missingness were handled.
- Evaluation protocol: emphasise LOSO cross-validation and explain why it prevents subject leakage; discuss metrics used (F1-macro, accuracy, calibration curves).
- Model choice & calibration: describe ensemble rationale and why calibrated probabilities matter for risk stratification and downstream decisions (pricing, tiering, interventions).
- Explainability: summarise SHAP output interpretation and sensitivity experiments; be ready to show top features and how removing a family changes performance.
- Fairness & uncertainty: describe the probabilistic proxy, reject-option trade-off and how packaging collects thesis-ready artefacts with provenance metadata.

---

If you want, I can:

- Expand the README with sample command lines for each preserved `run_phase*.py` file present in the repository root.
- Create a concise `requirements.txt` or Dockerfile for reproducible runs.
- Produce a one-page summary slide that maps each phase to the key figures/tables you should show in your viva.

Tell me which of the above you want next and I'll add it.
