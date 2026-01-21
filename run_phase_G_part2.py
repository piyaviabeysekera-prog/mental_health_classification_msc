#!/usr/bin/env python
"""
Runner script for Phase G Part 2: Validation Audit on Phase G Ensemble.

Usage:
    python run_phase_G_part2.py

This script:
- Does NOT modify any existing code or data
- Loads the pre-trained Phase G ensemble
- Performs SHAP analysis on the ensemble
- Runs fairness audit vs Phase C baseline
- Saves results to new isolated output files
"""

from code.phase_G_part2 import run_phase_G_part2

if __name__ == "__main__":
    run_phase_G_part2()
