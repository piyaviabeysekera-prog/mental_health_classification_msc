# Phase E & F Re-run Analysis: Conflicts, Safety, and Comparison Strategy

## ⚠️ Will It Cause Conflicts?

### **YES, File Conflicts Will Occur** (But Can Be Managed)

#### **Phase E File Overwrite Risk:**
```
Current outputs (from previous run):
├─ reports/tables/shap_top_features.csv         ← WILL BE OVERWRITTEN
├─ reports/tables/sensitivity.csv               ← WILL BE OVERWRITTEN
├─ reports/figures/shap_global.png              ← WILL BE OVERWRITTEN
└─ models/explainability/rf_explainability_model.joblib  ← WILL BE OVERWRITTEN

If you re-run without backing up:
⚠️ Previous results permanently lost
⚠️ No way to compare old vs new
```

#### **Phase F File Overwrite Risk:**
```
Current outputs (from previous run):
├─ reports/tables/fairness_summary.csv          ← WILL BE OVERWRITTEN
├─ reports/tables/reject_stats.csv              ← WILL BE OVERWRITTEN
├─ reports/tables/manifest.json                 ← WILL BE UPDATED
└─ reports/tables/thesis_final/ (all files)     ← WILL BE OVERWRITTEN
   ├─ shap_top_features.csv
   ├─ fairness_summary.csv
   └─ ... (all thesis finals)

If you re-run without backing up:
⚠️ All thesis_final packaging lost
⚠️ Git hash / timestamp changed
⚠️ Previous version unrecoverable
```

---

## ✅ Is It Safe to Run?

### **YES, Technically Safe — But Requires Preparation**

#### **Safety Level: 8/10**
- ✅ No database corruption risk
- ✅ No model poisoning (Phase G models independent)
- ✅ No data leakage introduced
- ✅ Can always be re-run if something breaks
- ⚠️ **BUT: Must back up existing results first**

#### **Safe Pre-Run Checklist:**
```
Before running Phase E:
□ Backup reports/tables/shap_top_features.csv → shap_top_features_BACKUP_Jan20.csv
□ Backup reports/tables/sensitivity.csv → sensitivity_BACKUP_Jan20.csv
□ Backup reports/figures/shap_global.png → shap_global_BACKUP_Jan20.png

Before running Phase F:
□ Backup reports/tables/fairness_summary.csv → fairness_summary_BACKUP_Jan20.csv
□ Backup reports/tables/reject_stats.csv → reject_stats_BACKUP_Jan20.csv
□ Backup reports/tables/thesis_final/ → thesis_final_BACKUP_Jan20/
□ Backup reports/tables/manifest.json → manifest_BACKUP_Jan20.json

Result: Old and new results side-by-side ✅
```

---

## 🔍 Critical Issue: Current Phase E/F Don't Use Phase G Models

### **THE CATCH:**

Phase E and F are currently **hardcoded to train their own RandomForest**:

```python
# Phase E: explainability.py (lines 55-62)
def build_rf_model(random_state: int = RANDOM_SEED) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )  # ← This is NOT the Phase G ensemble!

# Phase F: fairness_packaging.py (lines 52-70)
def _build_rf() -> RandomForestClassifier:
    """Define a reasonably strong probabilistic model for fairness analysis."""
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
    )  # ← This is NOT the Phase G ensemble either!
```

### **What This Means:**

| Scenario | What Happens | Result |
|----------|--------------|--------|
| Run Phase E as-is | Trains a NEW RandomForest (not Phase G) | Same SHAP results as before (RF is RF) |
| Run Phase F as-is | Trains a NEW RandomForest (not Phase G) | Same fairness results as before |
| What you ACTUALLY want | Run SHAP/fairness on Phase G 6-model ensemble | Requires code modification ⚠️ |

---

## 🛠️ To Use Phase G Ensemble in E & F, You'd Need To:

### **Option 1: Modify Phase E to Accept an Ensemble Model**
```python
# Current (hardcoded RF):
def run_global_shap(max_samples: int = 500) -> pd.DataFrame:
    rf = build_rf_model()  # ← Always creates RF
    rf.fit(X, y)
    explainer = shap.TreeExplainer(rf)
    
# Modified (accept any model):
def run_global_shap(model=None, max_samples: int = 500) -> pd.DataFrame:
    if model is None:
        model = build_rf_model()  # fallback
    else:
        model = model  # use provided ensemble
    
    model.fit(X, y)  # or load pre-trained
    explainer = shap.TreeExplainer(model)  # or KernelExplainer
    ...
```

**Effort**: 20-30 minutes | **Risk**: Low | **Benefit**: Ensemble-specific SHAP

---

### **Option 2: Create Wrapper Scripts**
```python
# NEW FILE: run_phase_E_on_ensemble_G.py
import joblib
from code.explainability import phase_E_explainability_and_sensitivity

# Load pre-trained Phase G ensemble from fold 0
ensemble = joblib.load('models/phase_G/voting_ensemble_fold_0.pkl')

# Apply SHAP analysis
# (would require custom code, not in existing Phase E)

# Save to: reports/tables/phase_G_explainability/
```

**Effort**: 1-2 hours | **Risk**: Medium | **Benefit**: Ensemble-specific, customizable

---

### **Option 3: Do Nothing & Accept Current Setup**
```python
# Phase E/F use generic RandomForest
# This is actually reasonable because:
# - Feature importance rankings likely stable across models
# - SHAP values show "what features matter" in general
# - Fair comparison baseline (not specific to one model)
```

**Effort**: 0 | **Risk**: 0 | **Benefit**: Fast, defensible

---

## 📊 Comparison Strategy (If You Decide to Run)

### **How to Compare Old vs New Results Safely**

#### **Step 1: Backup Everything**
```powershell
# Backup Phase E results
Copy-Item reports/tables/shap_top_features.csv `
           reports/tables/shap_top_features_BEFORE_PHASE_G.csv
Copy-Item reports/tables/sensitivity.csv `
           reports/tables/sensitivity_BEFORE_PHASE_G.csv

# Backup Phase F results
Copy-Item reports/tables/fairness_summary.csv `
           reports/tables/fairness_summary_BEFORE_PHASE_G.csv
Copy-Item reports/tables/reject_stats.csv `
           reports/tables/reject_stats_BEFORE_PHASE_G.csv

# Backup entire thesis_final
Copy-Item reports/tables/thesis_final reports/tables/thesis_final_BEFORE_PHASE_G -Recurse
```

#### **Step 2: Run Phase E & F**
```python
from code.main_pipeline import run_pipeline
run_pipeline(["phase_E", "phase_F"])

# This will:
# ✓ Train new RandomForest (independent of Phase G)
# ✓ Recompute SHAP (likely similar rankings)
# ✓ Recompute fairness (likely similar distributions)
# ✓ Regenerate thesis_final packaging
```

#### **Step 3: Compare Results**
```python
import pandas as pd

# Compare SHAP rankings
old_shap = pd.read_csv('reports/tables/shap_top_features_BEFORE_PHASE_G.csv')
new_shap = pd.read_csv('reports/tables/shap_top_features.csv')

print("Old top 5 features:")
print(old_shap.head())
print("\nNew top 5 features:")
print(new_shap.head())
print("\nRanking changed?", not old_shap['feature'].equals(new_shap['feature']))

# Compare sensitivity
old_sens = pd.read_csv('reports/tables/sensitivity_BEFORE_PHASE_G.csv')
new_sens = pd.read_csv('reports/tables/sensitivity.csv')

print("\nOld sensitivity results:")
print(old_sens)
print("\nNew sensitivity results:")
print(new_sens)

# Compare fairness
old_fair = pd.read_csv('reports/tables/fairness_summary_BEFORE_PHASE_G.csv')
new_fair = pd.read_csv('reports/tables/fairness_summary.csv')

print("\nOld fairness variance (mean):", old_fair['var_prob_class2'].mean())
print("New fairness variance (mean):", new_fair['var_prob_class2'].mean())
```

---

## 🎯 Should You Actually Do This?

### **Probability of Useful Insights: 40% (Modest)**

#### **What You'll Likely Find:**
```
SHAP feature rankings:
├─ OLD: EDA_tonic_max, EDA_tonic_mean, EDA_phasic_min, ...
├─ NEW: EDA_tonic_max, EDA_tonic_mean, EDA_phasic_min, ...  (same!)
└─ Reason: RF is deterministic, features haven't changed

Sensitivity impact:
├─ OLD: Removing SRI reduces F1 by ~5%
├─ NEW: Removing SRI reduces F1 by ~5%  (same!)
└─ Reason: Same feature set, cross-validation on same data

Fairness uncertainty:
├─ OLD: Per-subject variance: σ² = 0.04-0.12
├─ NEW: Per-subject variance: σ² = 0.04-0.12  (similar!)
└─ Reason: Same subjects, similar model capacity
```

#### **Why Results Will Be ~Same:**
1. **Deterministic randomness**: Fixed random seed (RANDOM_SEED = 42)
2. **Same data**: Phase B enriched features unchanged
3. **Same model type**: Both use 400-tree RandomForest
4. **Same validation**: Both use 5-fold cross-validation

---

## 💡 Better Alternative: Phase G-Specific Analysis

Instead of re-running Phase E/F as-is, consider this **higher-value approach**:

```python
# NEW: Extract feature importance from Phase G models directly
import joblib
from pathlib import Path

models_dir = Path("models/phase_G")

# Load fold 0 of each model type
xgb = joblib.load(models_dir / "xgboost_fold_0.pkl")
lgb = joblib.load(models_dir / "lightgbm_fold_0.pkl")
cat = joblib.load(models_dir / "catboost_fold_0.pkl")

# Extract feature importances
xgb_importance = xgb.feature_importances_
lgb_importance = lgb.booster_.feature_importance()
cat_importance = cat.feature_importances_

# Compare across 6 models
# Question: Do all models agree on important features?
# Insight: If all 6 models prioritize EDA metrics, that's strong evidence
```

**Value**: HIGH (ensemble-specific insights)
**Effort**: 1-2 hours
**Benefit**: Direct connection to Phase G performance

---

## ✅ Final Recommendation

### **For Your Immediate Needs (Thesis Submission):**

| Approach | Time | Value | Risk | Recommendation |
|----------|------|-------|------|-----------------|
| Re-run Phase E/F as-is | 20 min | **Low** (same results) | Low | ❌ Skip |
| Modify Phase E/F for ensemble | 2-3 hrs | **Moderate** (ensemble-specific) | Medium | ⚠️ Optional |
| Extract Phase G feature importance | 1-2 hrs | **High** (direct insights) | Low | ✅ **DO THIS** |
| Use existing Phase E/F results | 0 min | **High** (already valid) | None | ✅ **DO THIS** |

### **Recommended Strategy:**
1. ✅ **Keep existing Phase E/F results** (valid and useful)
2. ✅ **Extract Phase G feature importance** (new insights)
3. ❌ **Skip re-running Phase E/F** (no new information)
4. ❌ **Don't modify Phase E/F** (too much work for marginal gain)

---

## 📋 Summary Decision Matrix

```
QUESTION: Should I re-run Phase E & F on Phase G ensemble?

CONFLICT?           YES  ✓ (files will overwrite)
SAFE?               YES  ✓ (if backed up first)
EASY?               YES  ✓ (just run pipeline)

BUT:

Will results differ?     NO  ✗ (same RandomForest seed)
Will I learn new things? NO  ✗ (features already analyzed)
Worth the time?          NO  ✗ (20 min for zero new insights)

VERDICT:             ❌ SKIP - Better alternatives exist
ALTERNATIVE:         ✅ Extract Phase G model feature importances instead
```

---

## If You Still Want to Compare (Adventurous Route)

### **Here's the Exact Procedure:**

```powershell
# Step 1: Create backup folder
New-Item -Path "backups/phase_EF_before_phaseG" -ItemType Directory -Force

# Step 2: Backup Phase E files
Copy-Item "reports/tables/shap_top_features.csv" `
          "backups/phase_EF_before_phaseG/" -Force
Copy-Item "reports/tables/sensitivity.csv" `
          "backups/phase_EF_before_phaseG/" -Force
Copy-Item "reports/figures/shap_global.png" `
          "backups/phase_EF_before_phaseG/" -Force

# Step 3: Backup Phase F files
Copy-Item "reports/tables/fairness_summary.csv" `
          "backups/phase_EF_before_phaseG/" -Force
Copy-Item "reports/tables/reject_stats.csv" `
          "backups/phase_EF_before_phaseG/" -Force
Copy-Item "reports/tables/thesis_final" `
          "backups/phase_EF_before_phaseG/thesis_final_backup" -Recurse -Force

# Step 4: Run Phase E & F
python -c "from code.main_pipeline import run_pipeline; run_pipeline(['phase_E', 'phase_F'])"

# Step 5: Compare (results will be same, so comparison confirms determinism)
```

---

*Conclusion: Safe to run, but unnecessary. Focus on Phase G insights instead.*
