"""Structure-isolated holdout/CV computed only from the caller's DATA rows.

No file I/O, historical diagnostics or molecular-structure mutation occurs here.
All chemistry normalization is restricted to disposable grouping graph copies.
"""
from __future__ import annotations

import collections
import hashlib
import json
from numbers import Integral

import numpy as np
import rdkit
import scipy
import sklearn
from rdkit import Chem
from rdkit.Chem import rdMolHash
from rdkit.Chem.MolStandardize import rdMolStandardize as ms
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedGroupKFold

SEED = 20260810
GROUP_KEY_FIELDS = (
    'parent_group_key', 'protomer_group_key', 'scaffold_group_key',
    'raw_scaffold_group_key', 'tautomer_scaffold_group_key',
    'element_parent_group_key', 'element_scaffold_group_key',
)


def require(condition, message):
    """Data-integrity checks remain active under python -O."""
    if not condition:
        raise ValueError(str(message))


def _key(prefix, signature):
    return prefix + hashlib.sha256(signature.encode('utf8')).hexdigest()[:20]


def validate_rows(rows):
    require(len(rows) >= 20, 'At least 20 DATA rows are required')
    require(len({r['compound_id'] for r in rows}) == len(rows), 'Duplicate DATA compound IDs')
    for row in rows:
        for field in ('compound_id', 'standardized_smiles', 'representation_type'):
            require(isinstance(row[field], str) and bool(row[field].strip()), f'Empty DATA field: {field}')
        require(type(row['label']) is int and row['label'] in (0, 1), 'DATA labels must be integer 0 or 1')
    require({r['label'] for r in rows} == {0, 1}, 'DATA must contain both labels')


def _typed_scaffold(molecule, coordination):
    parts = []
    for fragment in Chem.GetMolFrags(molecule, asMols=True):
        if coordination and not any(a.GetAtomicNum() == 6 for a in fragment.GetAtoms()):
            continue
        scaffold = MurckoScaffold.GetScaffoldForMol(fragment)
        if scaffold.GetNumAtoms():
            parts.append(Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False))
    return '.'.join(sorted(parts))


def build_grouping_keys(rows):
    """Seven grouping keys; source isomeric SMILES are never rewritten."""
    normalizer, reionizer, uncharger = ms.Normalizer(), ms.Reionizer(), ms.Uncharger(force=True)
    enumerator = ms.TautomerEnumerator()
    enumerator.SetMaxTautomers(1000)
    enumerator.SetMaxTransforms(1000)
    scaffold_cache = {}
    metadata = []
    for row in rows:
        molecule = Chem.MolFromSmiles(row['standardized_smiles'])
        require(molecule is not None, f'Invalid DATA SMILES: {row["compound_id"]}')
        diagnostic = Chem.Mol(molecule)
        Chem.RemoveStereochemistry(diagnostic)
        for atom in diagnostic.GetAtoms():
            atom.SetIsotope(0)
        diagnostic = Chem.RemoveHs(diagnostic)
        coordination = row['representation_type'] == 'complete_coordination_entity'
        raw_scaffold = _typed_scaffold(diagnostic, coordination)
        # Never remove fragments or disconnect a metal / ligand. A complete
        # coordination parent keeps all its heavy atoms, counterions and ligands.
        if not coordination:
            diagnostic = uncharger.uncharge(reionizer.reionize(normalizer.normalize(diagnostic)))
        parent_smiles = Chem.MolToSmiles(diagnostic, canonical=True, isomericSmiles=False)
        protomer = rdMolHash.MolHash(diagnostic, rdMolHash.HashFunction.HetAtomProtomerv2)
        if coordination:
            parent, status = parent_smiles, 'not_enumerated_complete_coordination_entity'
        else:
            enumeration = enumerator.Enumerate(diagnostic)
            parent = Chem.MolToSmiles(enumerator.PickCanonical(enumeration), canonical=True, isomericSmiles=False)
            status = str(enumeration.status)
        scaffold = _typed_scaffold(diagnostic, coordination)
        if scaffold not in scaffold_cache:
            if scaffold:
                enumeration = enumerator.Enumerate(Chem.MolFromSmiles(scaffold))
                canonical = Chem.MolToSmiles(enumerator.PickCanonical(enumeration), canonical=True, isomericSmiles=False)
                scaffold_cache[scaffold] = canonical, str(enumeration.status)
            else:
                scaffold_cache[scaffold] = '', 'acyclic_parent_fallback'
        canonical_scaffold, scaffold_status = scaffold_cache[scaffold]
        element_parent = rdMolHash.MolHash(diagnostic, rdMolHash.HashFunction.ElementGraph)
        element_scaffold = (rdMolHash.MolHash(Chem.MolFromSmiles(scaffold), rdMolHash.HashFunction.ElementGraph)
                            if scaffold else element_parent)
        metadata.append({
            'compound_id': row['compound_id'], 'label': row['label'],
            'parent_group_key': _key('P_', parent), 'parent_group_signature': parent,
            'protomer_group_key': _key('H_', protomer), 'protomer_group_signature': protomer,
            'grouping_parent_smiles': parent_smiles, 'tautomer_enumeration_status': status,
            'scaffold_group_key': _key('S_', scaffold) if scaffold else _key('A_', parent),
            'grouping_scaffold_smiles': scaffold,
            'raw_scaffold_group_key': _key('R_', raw_scaffold) if raw_scaffold else _key('A_', parent),
            'grouping_raw_scaffold_smiles': raw_scaffold,
            'tautomer_scaffold_group_key': _key('T_', canonical_scaffold) if canonical_scaffold else _key('A_', parent),
            'grouping_tautomer_scaffold_smiles': canonical_scaffold,
            'scaffold_tautomer_enumeration_status': scaffold_status,
            'element_parent_group_key': _key('EP_', element_parent),
            'element_parent_signature': element_parent,
            'element_scaffold_group_key': _key('ES_' if scaffold else 'EA_', element_scaffold),
            'element_scaffold_signature': element_scaffold,
            'scaffold_grouping_rule': 'typed_Murcko' if scaffold else 'acyclic_complete_parent_fallback',
            'complete_coordination_entity': coordination,
        })
    return metadata


def connected_groups(rows, metadata):
    """Connected components of ANY shared key, including transitive links."""
    require(len(rows) == len(metadata), 'DATA and grouping metadata lengths differ')
    require([r['compound_id'] for r in rows] == [r['compound_id'] for r in metadata], 'Grouping IDs/order differ from DATA')
    require(len({r['compound_id'] for r in rows}) == len(rows), 'Duplicate DATA compound IDs')
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    lookups = {field: {} for field in GROUP_KEY_FIELDS}
    for index, item in enumerate(metadata):
        for field in GROUP_KEY_FIELDS:
            require(isinstance(item.get(field), str) and bool(item[field]), f'Missing grouping key: {field}')
            if item[field] in lookups[field]:
                a, b = find(index), find(lookups[field][item[field]])
                parents[max(a, b)] = min(a, b)
            else:
                lookups[field][item[field]] = index
    members = collections.defaultdict(list)
    for index in range(len(rows)):
        members[find(index)].append(index)
    ordered = sorted(members.values(), key=lambda indexes: min(rows[i]['compound_id'] for i in indexes))
    groups = []
    for number, indexes in enumerate(ordered, 1):
        group_id = f'G{number:04d}'
        for index in indexes:
            metadata[index]['structure_group_id'] = group_id
            metadata[index]['structure_group_size'] = len(indexes)
        positives = sum(rows[i]['label'] for i in indexes)
        groups.append({'group_id': group_id, 'n': len(indexes), 'positive': positives,
                       'negative': len(indexes) - positives,
                       'members': [rows[i]['compound_id'] for i in indexes]})
    return groups


def grouped_holdout(rows, metadata, groups, seed):
    target_n = round(len(rows) * .2)
    target_positive = round(sum(r['label'] for r in rows) * target_n / len(rows))
    y = np.array([r['label'] for r in rows], dtype=int)
    g = np.array([r['structure_group_id'] for r in metadata])
    require(len(groups) >= 5 and min(np.bincount(y)) >= 5, 'Insufficient groups/classes for five-way holdout initialization')
    initializer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, held = next(initializer.split(np.zeros((len(rows), 1)), y, g))
    initial = set(g[held])
    sizes = np.array([group['n'] for group in groups], dtype=float)
    positives = np.array([group['positive'] for group in groups], dtype=float)
    desired = np.array([group['group_id'] in initial for group in groups])
    cost = sizes * np.where(desired, -1., 1.) + np.random.default_rng(seed).uniform(0, 1e-4, len(groups))
    constraint = LinearConstraint(csr_matrix(np.vstack([sizes, positives])), [target_n, target_positive], [target_n, target_positive])
    result = milp(cost, integrality=np.ones(len(groups)), bounds=Bounds(0, 1), constraints=constraint,
                  options={'time_limit': 60, 'mip_rel_gap': 0})
    require(result.success and result.x is not None, f'Whole-group holdout quota infeasible or solver failed: {result.message}')
    selected = {groups[i]['group_id'] for i, value in enumerate(result.x) if value > .5}
    test = np.array([i for i, group in enumerate(g) if group in selected], dtype=int)
    development = np.array([i for i, group in enumerate(g) if group not in selected], dtype=int)
    require(len(test) == target_n and int(y[test].sum()) == target_positive, 'Holdout row/class quotas not met')
    return development, test, {'initial_method': 'first fold of StratifiedGroupKFold(n_splits=5)',
                              'initial_test_rows': len(held), 'initial_test_positive': int(y[held].sum()),
                              'target_test_rows': target_n, 'target_test_positive': target_positive,
                              'solver': 'scipy.optimize.milp / HiGHS', 'solver_message': result.message,
                              'groups_reassigned_after_initial': sorted(initial.symmetric_difference(selected)),
                              'objective': 'Minimize row-weighted changes to fixed initial groups; tiny seeded tie-break; no model or features used'}


def leakage_checks(assignments):
    checks = {}
    for row in assignments:
        require(row['dataset_partition'] in {'development', 'test'}, 'Unknown partition')
        fold = row['cv_validation_fold']
        require((row['dataset_partition'] == 'test' and fold is None) or
                (row['dataset_partition'] == 'development' and type(fold) is int and 1 <= fold <= 10),
                f'Invalid CV fold: {row["compound_id"]}')
    for field in ('structure_group_id',) + GROUP_KEY_FIELDS:
        partitions, folds = collections.defaultdict(set), collections.defaultdict(set)
        for row in assignments:
            require(isinstance(row.get(field), str) and bool(row[field]), f'Missing grouping key: {field}')
            partitions[row[field]].add(row['dataset_partition'])
            if row['dataset_partition'] == 'development':
                folds[row[field]].add(row['cv_validation_fold'])
        checks[field + '_cross_partition_overlap'] = sum(len(v) > 1 for v in partitions.values())
        checks[field + '_cross_cv_fold_overlap'] = sum(len(v) > 1 for v in folds.values())
    require(all(value == 0 for value in checks.values()), f'Structure leakage detected: {checks}')
    return checks


def split(rows, seed=SEED):
    """Return assignments/groups/manifest from DATA rows, preserving input rows."""
    validate_rows(rows)
    require(isinstance(seed, Integral) and not isinstance(seed, bool) and 0 <= seed < 2**32, 'seed must be an integer in 0..2**32-1')
    seed = int(seed)
    metadata = build_grouping_keys(rows)
    groups = connected_groups(rows, metadata)
    development, test, holdout = grouped_holdout(rows, metadata, groups, seed)
    require(len({metadata[i]['structure_group_id'] for i in development}) >= 10, 'Fewer than ten development groups')
    y = np.array([rows[i]['label'] for i in development], dtype=int)
    require(min(np.bincount(y, minlength=2)) >= 10, 'Fewer than ten development examples of one class')
    g = np.array([metadata[i]['structure_group_id'] for i in development])
    folds = np.zeros(len(development), dtype=int)
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    for fold, (_, held) in enumerate(cv.split(np.zeros((len(development), 1)), y, g), 1):
        folds[held] = fold
    for offset, index in enumerate(development):
        metadata[index].update(dataset_partition='development', cv_validation_fold=int(folds[offset]))
    for index in test:
        metadata[index].update(dataset_partition='test', cv_validation_fold=None)
    checks = leakage_checks(metadata)
    fold_counts = []
    for fold in range(1, 11):
        held = [r for r in metadata if r['dataset_partition'] == 'development' and r['cv_validation_fold'] == fold]
        positive = sum(r['label'] for r in held)
        require(0 < positive < len(held), f'Validation fold {fold} empty or lacks one class')
        fold_counts.append({'fold': fold, 'validation_n': len(held), 'validation_positive': positive,
                            'validation_negative': len(held)-positive, 'training_n': len(development)-len(held),
                            'validation_groups': len({r['structure_group_id'] for r in held})})
    counts = {}
    for partition in ('development', 'test'):
        subset = [r for r in metadata if r['dataset_partition'] == partition]
        positive = sum(r['label'] for r in subset)
        counts[partition] = {'n': len(subset), 'positive': positive, 'negative': len(subset)-positive,
                             'groups': len({r['structure_group_id'] for r in subset})}
    manifest = {'n': len(rows), 'random_seed': seed, 'partition_counts': counts,
                'group_count': len(groups), 'parent_group_count': len({r['parent_group_key'] for r in metadata}),
                'nonempty_scaffold_count': len({r['grouping_scaffold_smiles'] for r in metadata if r['grouping_scaffold_smiles']}),
                'acyclic_rows': sum(not r['grouping_scaffold_smiles'] for r in metadata),
                'cross_validation': fold_counts, 'holdout_method': holdout, 'leakage_checks': checks,
                'tautomer_enumeration_status_counts': dict(collections.Counter(r['tautomer_enumeration_status'] for r in metadata)),
                'tautomer_enumeration_limit_records': [r['compound_id'] for r in metadata if r['tautomer_enumeration_status'] not in {'Completed','not_enumerated_complete_coordination_entity'}],
                'scaffold_tautomer_enumeration_status_counts': dict(collections.Counter(r['scaffold_tautomer_enumeration_status'] for r in metadata)),
                'scaffold_tautomer_limit_records': [r['compound_id'] for r in metadata if r['scaffold_tautomer_enumeration_status'] not in {'Completed','acyclic_parent_fallback'}],
                'largest_groups': sorted(groups, key=lambda item: (-item['n'],item['group_id']))[:10],
                'software': {'rdkit': rdkit.__version__, 'scikit_learn': sklearn.__version__, 'scipy': scipy.__version__, 'numpy': np.__version__},
                'source_identity_sha256': hashlib.sha256(json.dumps([(r['compound_id'],r.get('PubChem_CID'),r['standardized_smiles'],r['label']) for r in rows],ensure_ascii=False,separators=(',',':')).encode('utf8')).hexdigest(),
                'data_policy': 'Original DATA SMILES, specified stereo, names, CIDs and labels unchanged; graph normalization only in grouping copies.',
                'grouping_policy': 'Union of seven keys: canonical tautomer parent, v2 protomer parent, original/normalized/tautomer typed Murcko scaffolds, ElementGraph parent/scaffold. ElementGraph conservatively also groups some saturation/redox analogues. Full-parent/scaffold enumeration limits are 1000 tautomers and 1000 transforms; enumeration-independent keys remain enforced.',
                'acyclic_policy': 'Empty Murcko scaffolds use parent keys; unrelated acyclic graphs are not put in one empty group. This is not a guarantee against every high-similarity analogue.',
                'fold_interpretation': 'Fold k: development fold k validates, other nine development folds train. Test has no CV fold and never enters cross-validation.',
                'split_selection_policy': 'Seed fixed before training. Only molecular groups, sample counts and labels determine membership; no features/models/scores used.'}
    return {'assignments': metadata, 'groups': groups, 'manifest': manifest}
