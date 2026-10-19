"""Load one explicitly selected representation with source/split integrity checks."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import ROOT, load_data
from .features import load_descriptor_names
from .preprocessing import prepare_fold


def load_fold(representation='rdkit166', fold=None, results_dir=None):
    """fold=1..10 supplies train/validation; None supplies development/test.

    Only the selected numerical representation enters X. Names, identifiers,
    outcome, structure groups and fold metadata are never included as predictors.
    """
    folder = Path(results_dir) if results_dir else ROOT / 'results'
    run = json.loads((folder / 'run_manifest.json').read_text(encoding='utf-8'))
    if run.get('status') != 'complete' or not run.get('independent_validation_passed'):
        raise ValueError('Results are incomplete or have not passed independent validation')
    rows, info = load_data()
    if run['source']['rows_fingerprint'] != info['rows_fingerprint']:
        raise ValueError('Results were generated from different DATA records')
    manifest = json.loads((folder / 'features/feature_manifest.json').read_text(encoding='utf-8'))
    if representation not in manifest['representations']:
        raise ValueError(f'Unknown representation: {representation}')
    file = folder / f'features/{representation}.npz'
    if hashlib.sha256(file.read_bytes()).hexdigest() != manifest['representations'][representation]['file_sha256']:
        raise ValueError('Feature NPZ failed its integrity check')
    with np.load(file, allow_pickle=False) as saved:
        matrix = saved['X']
        if saved['compound_id'].tolist() != [r['compound_id'] for r in rows]:
            raise ValueError('Feature IDs/order differ from DATA')
        if saved['standardized_smiles'].tolist() != [r['standardized_smiles'] for r in rows]:
            raise ValueError('Feature molecular structures differ from DATA')
        if not np.array_equal(saved['row_index'], np.arange(len(rows))):
            raise ValueError('Feature row indices differ from DATA')
        if representation == 'rdkit166' and saved['descriptor_names'].tolist() != load_descriptor_names(ROOT / 'configs/rdkit_descriptors_166.txt'):
            raise ValueError('Descriptor names/order differ from the fixed configuration')
    if not np.isfinite(matrix).all():
        raise ValueError('This release requires all raw feature values to be finite')
    assignments = json.loads((folder / 'split_payload.json').read_text(encoding='utf-8'))['assignments']
    return prepare_fold(matrix, rows, assignments, representation, fold)
