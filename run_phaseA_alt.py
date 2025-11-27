import importlib.util
from pathlib import Path

mp_path = Path('code') / 'main_pipeline.py'
spec = importlib.util.spec_from_file_location('mp_alt', str(mp_path))
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)

if hasattr(mp, 'phase_A_ingestion_and_eda'):
    df = mp.phase_A_ingestion_and_eda()
    print('Phase A completed, rows:', len(df))
else:
    print('phase_A_ingestion_and_eda not found in module')
