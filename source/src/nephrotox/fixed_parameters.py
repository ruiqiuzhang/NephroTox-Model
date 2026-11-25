"""Explicit opt-in execution of original, preselected parameter values."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
FIXED_PARAMETER_FILE=ROOT/'configs/original_fixed_parameters.json'
MODES=('search','fixed_original')


def fixed_parameter_records():
    data=json.loads(FIXED_PARAMETER_FILE.read_text(encoding='utf-8'))
    if data['parameter_mode']!='fixed_original' or len(data['models'])!=43:
        raise ValueError('Original fixed-parameter registry is incomplete')
    return data['models']


def configure_parameter_mode(results_dir, mode):
    if mode not in MODES:
        raise ValueError(f'Unsupported parameter mode: {mode}')
    path=Path(results_dir)/'parameter_mode.json'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf-8'))['parameter_mode']
        if previous!=mode:
            raise ValueError('Cannot mix fixed and searched configurations in the same results directory')
        return
    if (Path(results_dir)/'modeling/protocol.json').exists() and mode!='search':
        raise ValueError('An existing search run cannot be converted to a fixed-parameter run')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'schema_version':1,'parameter_mode':mode},indent=2),encoding='utf-8')


def read_parameter_mode(results_dir):
    path=Path(results_dir)/'parameter_mode.json'
    mode=json.loads(path.read_text(encoding='utf-8'))['parameter_mode'] if path.exists() else 'search'
    if mode not in MODES:
        raise ValueError('Invalid saved parameter mode')
    return mode
