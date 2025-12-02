"""
Convenience script to run Phase F (fairness + reject option + packaging)
without manually passing CLI arguments.
"""

from code.main_pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(["phase_F"])