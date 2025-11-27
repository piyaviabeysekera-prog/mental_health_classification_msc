# Helper to run phase 0 scaffolding reproducibly
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

# Load config and utils as submodules under the package name
for mod_name in ('config', 'utils'):
    full_name = f"{pkg_name}.{mod_name}"
    spec = importlib.util.spec_from_file_location(full_name, str(pkg_dir / f"{mod_name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)

# Now call the utilities to perform scaffolding
from code import utils

utils.init_basic_logging()
utils.set_global_seeds()
utils.initialise_scaffolding_directories()
utils.ensure_runlog_md()

from datetime import datetime
now = datetime.utcnow().isoformat() + 'Z'
json_path = utils.log_run_metadata('phase_0_scaffolding', 'success', {'note': 'Initial scaffolding run (helper script).'})
utils.append_runlog_md_row(now, 'phase_0_scaffolding', 'success', 'Initial scaffolding and dummy run recorded.')

print('Phase 0 scaffolding completed. Wrote:', json_path)
