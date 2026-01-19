# Integration Guide: Adding Phase G to main_pipeline.py

## Overview

Phase G is **completely standalone** and does NOT require modification to existing phases. However, if you want Phase G to run as part of your full pipeline orchestration, this guide shows how to integrate it.

**Important**: You can run Phase G independently without any integration. This is optional.

## Option 1: Run Phase G Independently (Recommended)

Simply execute in Python:

```python
from code.phase_G import run_phase_G

run_phase_G()
```

This is the **safest approach** and requires no changes to existing code.

## Option 2: Integrate into main_pipeline.py (Advanced)

If you want Phase G to run as part of your orchestrated pipeline, follow these steps:

### Step 1: Check current main_pipeline.py structure

View the end of main_pipeline.py:

```bash
tail -50 code/main_pipeline.py
```

Look for the `main()` function that orchestrates phases.

### Step 2: Add Phase G import at top of main_pipeline.py

Add this import section near other phase imports:

```python
# Around line 20-40, near other imports:
from .phase_G import run_phase_G  # Phase G: Heterogeneous ensemble
```

### Step 3: Add command-line argument for Phase G

In the argument parser section, add:

```python
# Around line 100-150, near other phase arguments:
parser.add_argument(
    "--run-phase-G",
    action="store_true",
    help="Run Phase G: Heterogeneous Multi-Model Ensemble & Comparative Performance Audit"
)

parser.add_argument(
    "--run-all",
    action="store_true",
    help="Run all phases (A-G)"
)
```

### Step 4: Add Phase G execution block to main()

In the `main()` function, add this block at the end:

```python
# Around line 300-350, at the end of other phase blocks:

if args.run_phase_G or args.run_all:
    print("\n" + "="*80)
    print("PHASE G: Heterogeneous Multi-Model Ensemble & Comparative Performance Audit")
    print("="*80)
    try:
        run_phase_G()
        print("✓ Phase G completed successfully")
    except Exception as e:
        print(f"✗ Phase G failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
```

### Step 5: Update help text

Find the `if __name__ == "__main__":` section and update docstring:

```python
if __name__ == "__main__":
    """
    Orchestrate all phases of ML pipeline.
    
    Phases:
    - Phase A: Schema validation & EDA
    - Phase B: Composite feature engineering (SRI, RS, PL)
    - Phase C: LOSO baselines (LogReg, RF + negative control)
    - Phase D: Ensemble & calibration
    - Phase E: Explainability (SHAP)
    - Phase F: Fairness evaluation
    - Phase G: Heterogeneous ensemble & generalization audit  <-- NEW
    
    Example: python main_pipeline.py --run-all
    Example: python main_pipeline.py --run-phase-A --run-phase-G
    """
```

## Option 3: Create a Simplified Wrapper Script (Recommended)

Create a new file: `code/run_full_pipeline_with_G.py`

```python
"""
Orchestration script to run phases 0-G in sequence.
This is a simplified alternative to modifying main_pipeline.py.
"""

from pathlib import Path
import sys

from .main_pipeline import run_phase_A
from .composites import run_phase_B
from .baselines import run_phase_C
from .ensembles import run_phase_D
from .explainability import run_phase_E
from .fairness_packaging import run_phase_F
from .phase_G import run_phase_G

def run_full_pipeline_with_phase_G():
    """Execute all phases A-G in sequence."""
    
    phases = [
        ("A", "Schema Validation & EDA", run_phase_A),
        ("B", "Composite Feature Engineering", run_phase_B),
        ("C", "LOSO Baselines", run_phase_C),
        ("D", "Ensemble & Calibration", run_phase_D),
        ("E", "Explainability (SHAP)", run_phase_E),
        ("F", "Fairness Evaluation", run_phase_F),
        ("G", "Heterogeneous Ensemble Audit", run_phase_G),
    ]
    
    failed_phases = []
    
    for phase_letter, phase_name, phase_func in phases:
        try:
            print(f"\n{'='*80}")
            print(f"PHASE {phase_letter}: {phase_name}")
            print('='*80)
            phase_func()
            print(f"✓ Phase {phase_letter} completed successfully")
        except Exception as e:
            print(f"✗ Phase {phase_letter} failed: {e}")
            import traceback
            traceback.print_exc()
            failed_phases.append(phase_letter)
    
    print(f"\n{'='*80}")
    print("PIPELINE SUMMARY")
    print('='*80)
    
    if not failed_phases:
        print("✓ All phases A-G completed successfully")
        return 0
    else:
        print(f"✗ Failed phases: {', '.join(failed_phases)}")
        return 1

if __name__ == "__main__":
    exit_code = run_full_pipeline_with_phase_G()
    sys.exit(exit_code)
```

Then run via:

```bash
python code/run_full_pipeline_with_G.py
```

## Phase G Dependencies: What It Needs to Run

Phase G **requires** that Phase B (composites) has already been run:

```
Phase B creates: data_stage/features/merged_with_composites.parquet
                 ↓
            Phase G loads this file
```

**Phases G doesn't require**: Phase C, D, E, or F (it trains its own models)

**Dependency chain**:

```
Phase B (Composites)
      ↓
Phase G (Heterogeneous Ensemble)  ← INDEPENDENT THREAD
      ↓
[Your thesis results]
```

Phase G runs **in parallel** with Phase C, D, E, F conceptually, though you likely run them sequentially.

## Expected Runtime

If integrating into full pipeline:

```
Phase A: ~2-3 min
Phase B: ~5 min
Phase C: ~8 min
Phase D: ~10 min
Phase E: ~5 min
Phase F: ~3 min
Phase G: ~10-15 min (NEW)
─────────────────────
TOTAL:  ~43-56 min (vs ~33-40 min without Phase G)
```

## Minimal Integration (Recommended)

If you just want Phase G in your pipeline without modifying main_pipeline.py, create a simple runner:

```python
# File: run_phase_G.py (in project root)
from code.phase_G import run_phase_G

if __name__ == "__main__":
    run_phase_G()
```

Execute via:

```bash
python run_phase_G.py
```

## Verification: Phase G Doesn't Modify Previous Phases

After running Phase G, verify nothing was changed:

```bash
# Check that Phase B output is unchanged
ls -lah data_stage/features/merged_with_composites.parquet

# Check that Phase C output is unchanged
ls -lah reports/tables/loso_baselines.csv
ls -lah reports/tables/shuffle_control.csv

# Check that Phase D output is unchanged
ls -lah reports/tables/ensembles_per_fold.csv
```

All timestamps should be older than Phase G execution time, confirming no modifications.

Phase G outputs are isolated:

```
✓ reports/tables/phase_G_individual_performance.csv      (NEW)
✓ reports/tables/phase_G_ensemble_performance.csv        (NEW)
✓ reports/tables/phase_G_individual_fold_metrics.csv     (NEW)
✓ reports/tables/phase_G_ensemble_fold_metrics.csv       (NEW)
✓ models/phase_G/                                        (NEW)
```

## Recommendations

### For Rapid Thesis Writing (Most Common)

1. Run Phase G independently:
   ```python
   from code.phase_G import run_phase_G
   run_phase_G()
   ```

2. Extract metrics:
   ```python
   import pandas as pd
   summary = pd.read_csv('reports/tables/phase_G_individual_performance.csv')
   ensemble = pd.read_csv('reports/tables/phase_G_ensemble_performance.csv')
   ```

3. Write thesis section citing Phase G results

### For Production Pipeline

Use Option 3 (wrapper script) to maintain clean orchestration without modifying main_pipeline.py.

### For Maximum Integration

Modify main_pipeline.py as shown in Option 2, but **test thoroughly first** to ensure no breakage.

---

**Recommended**: Start with Option 1 (independent execution), then migrate to Option 3 (wrapper) if needed. Avoid Option 2 unless you have confidence modifying main_pipeline.py.
