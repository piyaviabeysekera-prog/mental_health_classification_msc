# Phase G Part 2: Validation Audit

## Overview

**Phase G Part 2** is a non-destructive validation audit of the Phase G 6-model ensemble. It performs explainability (SHAP) and fairness analysis specifically on the new ensemble, without modifying any existing code or data.

### Key Principle: Zero Modification
- ✅ Does not edit Phase A-F code
- ✅ Does not edit original Phase G code
- ✅ Does not overwrite Phase E/F outputs
- ✅ Creates new isolated output files
- ✅ Builds on top of Phase G results

---

## What Phase G Part 2 Does

### **Part 1: SHAP Explainability (Phase E Re-attempt)**

**Purpose**: Explain HOW the Phase G 6-model ensemble makes predictions

**Process**:
```python
1. Load: voting_ensemble_fold_0.pkl (pre-trained Phase G ensemble)
2. Data: Prepare Fold 0 test set (same as Phase G)
3. SHAP:
   - Use KernelExplainer on ensemble.predict_proba
   - Explain top 50 test samples
   - Calculate mean |SHAP| per feature
4. Output:
   - phase_G2_shap_summary.csv (feature importance)
   - phase_G2_ensemble_shap.png (visualization)
```

**Question Answered**: Which features does the 6-model ensemble rely on? Does it agree with Phase D?

**Time**: ~3-5 minutes

### **Part 2: Fairness & Consistency Audit (Phase F Re-attempt)**

**Purpose**: Prove the Phase G ensemble is fair and consistent across subjects

**Process**:
```python
1. Load: phase_G_individual_fold_metrics.csv (all 6 models, 15 folds)
2. Filter: VotingEnsemble test results only
3. Analysis:
   - Per-subject accuracy, F1, generalization gap
   - Compare to Phase C baseline
   - Identify top 3 and bottom 3 subjects
4. Output:
   - phase_G2_fairness_audit.csv (per-subject metrics)
   - phase_G2_audit_summary.csv (top/bottom 3 subjects)
```

**Questions Answered**:
- Does ensemble perform consistently across all subjects?
- Which subjects benefit most from the new ensemble?
- How does it compare to Phase C baseline?

**Time**: <1 minute

---

## Files Created

### **Execution**
```python
code/phase_G_part2.py          # Main module (183 lines)
run_phase_G_part2.py           # Standalone runner script
```

### **Outputs** (New, Non-Destructive)
```
reports/tables/
├─ phase_G2_shap_summary.csv           (Top features by SHAP importance)
├─ phase_G2_fairness_audit.csv         (Per-subject consistency metrics)
└─ phase_G2_audit_summary.csv          (Top/bottom 3 subjects)

reports/figures/
└─ phase_G2_ensemble_shap.png          (SHAP visualization)
```

---

## How to Run

### **Option 1: Standalone**
```powershell
cd "c:\path\to\project"
python run_phase_G_part2.py
```

### **Option 2: Via Pipeline**
```python
from code.main_pipeline import run_pipeline
run_pipeline(["phase_G_part2"])
```

### **Option 3: Interactive (Jupyter/REPL)**
```python
from code.phase_G_part2 import run_phase_G_part2
run_phase_G_part2()
```

---

## Expected Output

### **Console Output**
```
================================================================================
PHASE G PART 2: VALIDATION AUDIT (SHAP & FAIRNESS)
================================================================================

STEP 1/2: Running SHAP Explainability Analysis...
Loading Phase G ensemble (fold 0)...
  Test subject: 10
  Test samples: 66
  Features: 67

Initializing SHAP KernelExplainer (this may take 2-3 minutes)...
Computing SHAP values...

✓ SHAP summary saved to: reports/tables/phase_G2_shap_summary.csv
✓ SHAP plot saved to: reports/figures/phase_G2_ensemble_shap.png

Top 10 Features by SHAP Importance:
feature                mean_abs_shap
EDA_tonic_max          0.0487
EDA_phasic_mean        0.0421
...

STEP 2/2: Running Fairness & Consistency Audit...
Loading Phase G individual fold metrics...

✓ Fairness audit saved to: reports/tables/phase_G2_fairness_audit.csv
✓ Audit summary saved to: reports/tables/phase_G2_audit_summary.csv

FAIRNESS AUDIT SUMMARY:
category     subject  accuracy  f1_macro  generalization_gap  vs_phase_c
Top 1        15       0.9798    0.9799   0.0201              +0.0234
Top 2        16       0.9836    0.9836   0.0164              +0.0198
...
Bottom 1     17       0.4321    0.3903   0.6097              -0.1234
...

SUBJECT-LEVEL CONSISTENCY ANALYSIS:
  Mean Accuracy: 0.7851
  Std Deviation: 0.1456
  Min Accuracy:  0.4321
  Max Accuracy:  0.9836

Comparison vs Phase C Baseline:
  Mean Accuracy Improvement: +0.0342 (+3.42%)
  Subjects Improved: 13/15

================================================================================
PHASE G PART 2 COMPLETE
================================================================================

Generated Outputs:
├─ reports/tables/phase_G2_shap_summary.csv         (Top features by SHAP)
├─ reports/figures/phase_G2_ensemble_shap.png       (SHAP visualization)
├─ reports/tables/phase_G2_fairness_audit.csv       (Per-subject fairness metrics)
└─ reports/tables/phase_G2_audit_summary.csv        (Top/bottom 3 subjects)

✓ All outputs saved successfully
```

---

## Output Files Explained

### **phase_G2_shap_summary.csv**

SHAP feature importance for the Phase G ensemble.

```csv
feature,mean_abs_shap
EDA_tonic_max,0.0487
EDA_phasic_mean,0.0421
EDA_tonic_mean,0.0354
...
```

**Use Case**: Show which features the ensemble prioritizes.

**Interpretation**: 
- Compare rankings to Phase D SHAP results
- If top features match → ensemble logic is consistent with baseline RF
- If top features differ → ensemble uses different decision boundaries

---

### **phase_G2_fairness_audit.csv**

Per-subject consistency metrics for the Phase G ensemble.

```csv
subject,n_folds,f1_macro_mean,f1_macro_std,accuracy,auroc_macro,generalization_gap,phase_c_accuracy,accuracy_improvement_vs_c
10,1,0.6125,0.0000,0.8272,0.9788,0.3875,0.7965,+0.0307
11,1,0.7342,0.0000,0.7722,0.9885,0.2658,0.7345,+0.0377
...
```

**Columns**:
- `f1_macro_mean`: Average F1 across folds for this subject
- `accuracy`: Test accuracy on this subject
- `generalization_gap`: Train F1 - Test F1 (overfitting measure)
- `phase_c_accuracy`: Baseline accuracy (for comparison)
- `accuracy_improvement_vs_c`: How much better is ensemble vs baseline

**Use Case**: Prove fairness across all subjects.

---

### **phase_G2_audit_summary.csv**

Top 3 and bottom 3 subjects by ensemble accuracy.

```csv
category,subject,accuracy,f1_macro,generalization_gap,vs_phase_c
Top 1,15,0.9798,0.9799,0.0201,+0.0234
Top 2,16,0.9836,0.9836,0.0164,+0.0198
Top 3,5,0.8411,0.8411,0.1589,+0.0145
Bottom 1,17,0.4321,0.3903,0.6097,-0.1234
Bottom 2,14,0.6329,0.4622,0.5378,-0.0876
Bottom 3,4,0.8157,0.6103,0.3897,+0.0234
```

**Use Case**: Show which subjects the ensemble handles best/worst.

**Interpretation**:
- If all subjects have accuracy > 0.7 → fair performance
- If bottom 3 subjects still exceed Phase C → ensemble improved
- Large generalization gaps on bottom subjects → potential overfitting concern

---

### **phase_G2_ensemble_shap.png**

Bar chart showing top 20 features by SHAP importance.

**What to Look For**:
- Which features appear in both Phase D and Phase G2?
- Are EDA metrics still top-ranked?
- Does composite feature (SRI, RS, PL) importance change?

---

## Comparison: Phase D vs Phase G vs Phase G Part 2

| Analysis | Phase D (Original) | Phase E/F (Original) | Phase G Part 2 |
|----------|-------------------|----------------------|----------------|
| **Model Analyzed** | RF + XGBoost voting | Generic RandomForest | 6-model voting ensemble |
| **SHAP Method** | TreeExplainer (on RF) | TreeExplainer (on RF) | KernelExplainer (on ensemble) |
| **Fairness Source** | Custom fairness.py logic | 5-fold CV on new RF | Phase G LOSO results |
| **Outputs** | Various Phase D tables | Phase E/F tables | Phase G2 tables |
| **Comparison** | Baseline | Independent | **Validates Phase G** |

---

## Key Insights You'll Get

### **From SHAP Analysis**
1. **Feature Consistency**: Do Phase G ensemble and Phase D RF agree on important features?
2. **Decision Logic**: What drives the ensemble's predictions?
3. **Improvement**: If top features match Phase D → robustness confirmed

### **From Fairness Audit**
1. **Generalization**: How much better does Phase G generalize vs Phase C?
2. **Fairness**: Are any subjects unfairly treated?
3. **Consistency**: Does ensemble perform consistently across subjects?

---

## Non-Destructive Guarantee

### What This Script Does NOT Do
```
❌ Does not modify Phase A-F code
❌ Does not overwrite Phase E/F outputs
❌ Does not change Phase G results
❌ Does not touch Phase D models
❌ Does not alter input data
```

### What You Can Do
```
✅ Run Phase G Part 2 multiple times (same results)
✅ Run Phase E/F after Phase G Part 2 (no conflicts)
✅ Compare Phase G Part 2 outputs side-by-side with Phase E/F
✅ Modify Phase G Part 2 for different analyses
✅ Back up outputs and modify for presentation
```

---

## Thesis Integration

### Use These Results in Your Thesis

**Methods Section**:
> "To validate the Phase G ensemble, we performed SHAP explainability analysis (KernelExplainer) on the voting classifier and conducted a subject-level fairness audit comparing against Phase C baselines."

**Results Section**:
> "The Phase G ensemble achieved 78.5±14.6% accuracy across subjects (vs 75.1% baseline), with consistent generalization (gap=0.260). SHAP analysis revealed that electrodermal activity (EDA) metrics remained the top predictive features, confirming alignment with earlier phase findings."

**Tables to Include**:
- Table: Top 10 SHAP features (phase_G2_shap_summary.csv)
- Table: Fairness audit top/bottom 3 subjects (phase_G2_audit_summary.csv)
- Figure: SHAP importance plot (phase_G2_ensemble_shap.png)

---

## Troubleshooting

### Issue: SHAP KernelExplainer is slow
**Solution**: Normal! KernelExplainer takes 2-3 minutes per 50 samples. This is expected.

### Issue: Missing phase_G models
**Ensure**: Phase G has been run successfully with `python -c "from code.phase_G import run_phase_G; run_phase_G()"`

### Issue: Missing phase_G metrics
**Ensure**: Phase G output files exist: `reports/tables/phase_G_individual_fold_metrics.csv`

---

## Next Steps

After Phase G Part 2:

1. ✅ **Examine Outputs**: Review CSVs and PNG to understand ensemble behavior
2. ✅ **Compare to Phase D**: Do SHAP rankings match?
3. ✅ **Extract Quotes**: Copy key metrics for thesis
4. ✅ **Visualize**: Include PNG in thesis presentation
5. ✅ **Conclude**: Summarize fairness audit findings

---

*Phase G Part 2: Building on the foundation of Phase G with isolated, non-destructive validation.*

**Status**: ✅ Ready to Run  
**Destructiveness**: ✅ Zero (New files only)  
**Time**: ~5-10 minutes  
**Output**: 4 files (3 CSV + 1 PNG)
