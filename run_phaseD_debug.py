import traceback

try:
    from code.ensembles import run_phase_D
    print("Imported run_phase_D; invoking...")
    run_phase_D()
    print("run_phase_D completed")
except Exception:
    print("Exception during run_phase_D:")
    traceback.print_exc()
