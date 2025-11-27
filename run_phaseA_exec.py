import importlib.util
import sys
from pathlib import Path

repo_root = Path(__file__).parent.resolve()
pkg_dir = repo_root / 'code'

pkg_name = 'code'
# create package skeleton module and register
spec_pkg = importlib.util.spec_from_file_location(pkg_name, str(pkg_dir / '__init__.py'))
pkg = importlib.util.module_from_spec(spec_pkg)
pkg.__path__ = [str(pkg_dir)]
sys.modules[pkg_name] = pkg

# load config and utils as submodules
for mod in ('config', 'utils'):
    full = f"{pkg_name}.{mod}"
    spec = importlib.util.spec_from_file_location(full, str(pkg_dir / f"{mod}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)

# load main_pipeline as a submodule
full = f"{pkg_name}.main_pipeline"
spec = importlib.util.spec_from_file_location(full, str(pkg_dir / 'main_pipeline.py'))
module = importlib.util.module_from_spec(spec)
sys.modules[full] = module
spec.loader.exec_module(module)

mp = sys.modules[full]
if hasattr(mp, 'phase_A_ingestion_and_eda'):
    df = mp.phase_A_ingestion_and_eda()
    print('Phase A completed, rows:', len(df))
else:
    print('phase_A_ingestion_and_eda not found')

print('module keys count:', len(mp.__dict__))
print(sorted([k for k in mp.__dict__.keys() if not k.startswith('__')]))

if hasattr(mp, 'phase_A_ingestion_and_eda'):
    df = mp.phase_A_ingestion_and_eda()
    print('Phase A completed, rows:', len(df))
else:
    print('phase_A_ingestion_and_eda not found')
