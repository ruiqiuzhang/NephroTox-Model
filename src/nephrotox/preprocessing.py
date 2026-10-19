"""Training-only preprocessing with explicit source and group-isolation guards."""
from numbers import Integral

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .split import GROUP_KEY_FIELDS, leakage_checks, require

FEATURE_WIDTHS = {'morgan2048':2048, 'maccs166':166, 'atompair1024':1024, 'torsion1024':1024, 'rdkit166':166}


def get_fold_indices(rows, assignments, fold=None):
    """Return global zero-based indices: (training, validation-or-test)."""
    require(fold is None or (isinstance(fold,Integral) and not isinstance(fold,bool) and 1<=fold<=10),
            'fold must be None or an integer in 1..10')
    ids = [r['compound_id'] for r in rows]
    require(len(ids)>0 and len(set(ids))==len(ids), 'DATA IDs must be nonempty and unique')
    require(ids==[r['compound_id'] for r in assignments], 'DATA/assignment IDs or order differ')
    for original, assigned in zip(rows, assignments):
        require(type(original['label']) is int and original['label'] in (0,1) and
                type(assigned['label']) is int and assigned['label']==original['label'],
                f'DATA/assignment label mismatch: {original["compound_id"]}')
    leakage_checks(assignments)
    dev = np.array([r['dataset_partition']=='development' for r in assignments],dtype=bool)
    folds = np.array([r['cv_validation_fold'] or 0 for r in assignments],dtype=int)
    for number in range(1,11):
        labels = {r['label'] for r in assignments if r['dataset_partition']=='development' and r['cv_validation_fold']==number}
        require(labels=={0,1},f'Validation fold {number} empty or lacks one class')
    if fold is None:
        training, evaluation = dev, ~dev
    else:
        training, evaluation = dev & (folds!=fold), dev & (folds==fold)
    require(training.any() and evaluation.any() and not (training&evaluation).any(), 'Empty/overlapping training and evaluation subsets')
    require(fold is None or not ((training|evaluation)&~dev).any(), 'Test rows entered CV')
    return np.flatnonzero(training), np.flatnonzero(evaluation)


def prepare_fold(X, rows, assignments, representation, fold=None):
    """Fit imputer/scaler only on training; never fit on validation/test values.

    Binary fingerprints remain unchanged. RDKit166 permits NaN for train-only
    median imputation, rejects infinity and preserves all 166 columns.
    """
    require(representation in FEATURE_WIDTHS, f'Unknown representation: {representation}')
    matrix = np.asarray(X)
    require(np.issubdtype(matrix.dtype,np.number) and matrix.ndim==2 and
            matrix.shape==(len(rows),FEATURE_WIDTHS[representation]), 'Feature matrix shape/type differs from DATA/representation')
    require(not np.isinf(matrix).any(), 'Feature matrix contains infinity')
    if representation!='rdkit166':
        require(np.isin(matrix,[0,1]).all(), 'Fingerprint features must be binary bits')
    training, evaluation = get_fold_indices(rows,assignments,fold)
    a,b = matrix[training],matrix[evaluation]
    preprocessing = {}
    if representation=='rdkit166':
        imputer = SimpleImputer(strategy='median',keep_empty_features=True)
        scaler = StandardScaler()
        a = scaler.fit_transform(imputer.fit_transform(a))
        b = scaler.transform(imputer.transform(b))
        preprocessing = {'imputer':imputer,'scaler':scaler}
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite features after training-only preprocessing')
    ids = np.array([r['compound_id'] for r in rows],dtype=str)
    labels = np.array([r['label'] for r in rows],dtype=int)
    other = 'test' if fold is None else 'validation'
    return {'X_train':a,'X_'+other:b, 'training_ids':ids[training],'other_ids':ids[evaluation],
            'training_indices':training,'other_indices':evaluation,
            'y_train':labels[training],'y_'+other:labels[evaluation], 'preprocessing':preprocessing}
