# Helper to run phase A ingestion + EDA reproducibly
import importlib.util
import sys
from pathlib import Path

repo_root = Path(__file__).parent.resolve()
pkg_dir = repo_root / 'code'

# Load package module 'code' from code/__init__.py
pkg_name = 'code'
spec = importlib.util.spec_from_file_location(pkg_name, str(pkg_dir / '__init__.py'))
pkg = importlib.util.module_from_spec(spec)
pkg.__path__ = [str(pkg_dir)]
sys.modules[pkg_name] = pkg

# Load submodules explicitly by file
for mod_name in ('config', 'utils', 'main_pipeline'):
    full_name = f"{pkg_name}.{mod_name}"
    spec = importlib.util.spec_from_file_location(full_name, str(pkg_dir / f"{mod_name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)

# Run the Phase A function
from code.main_pipeline import phase_A_ingestion_and_eda

if __name__ == '__main__':
    df = phase_A_ingestion_and_eda()
    print('Phase A completed, rows:', len(df))
