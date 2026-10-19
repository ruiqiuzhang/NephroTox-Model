"""Independent validation of the current DATA source and newly generated results.

No historical datasets, baseline workbooks, chemical caches or legacy inventory
are opened. Molecular features are recomputed directly with RDKit primitives;
the producer's feature, splitting and preprocessing functions are not imported.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolHash
from rdkit.Chem.MolStandardize import rdMolStandardize as ms
from rdkit.Chem.Scaffolds import MurckoScaffold


GROUP_KEYS = ('parent_group_key', 'protomer_group_key', 'scaffold_group_key',
              'raw_scaffold_group_key', 'tautomer_scaffold_group_key',
              'element_parent_group_key', 'element_scaffold_group_key')
FP_WIDTHS = {'morgan2048': 2048, 'maccs166': 166, 'atompair1024': 1024, 'torsion1024': 1024}


class ValidationError(ValueError):
    """A failed validation. Unhandled failures make the pipeline exit nonzero."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def records_fingerprint(rows):
    text = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _short(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]


def validate_source(rows, source_info=None):
    require(bool(rows), 'No DATA records supplied')
    ids = [r['compound_id'] for r in rows]
    require(len(ids) == len(set(ids)) and all(isinstance(v, str) and v for v in ids), 'DATA IDs missing or duplicated')
    require(set(r['label'] for r in rows) == {0, 1} and all(type(r['label']) is int for r in rows), 'DATA labels must contain integer binary 0 and 1')
    require(all(type(r['PubChem_CID']) is int and r['PubChem_CID'] > 0 and r['standardized_smiles'] for r in rows), 'DATA CID/SMILES missing')
    if source_info and source_info.get('rows_fingerprint'):
        require(records_fingerprint(rows) == source_info['rows_fingerprint'], 'DATA rows fingerprint mismatch (including labels and row order)')
    if source_info and 'n_rows' in source_info:
        require(len(rows) == source_info['n_rows'], 'DATA source row count mismatch')
    if source_info and 'label_counts' in source_info:
        expected = {int(k): v for k, v in source_info['label_counts'].items()}
        require(dict(Counter(r['label'] for r in rows)) == expected, 'DATA source label counts mismatch')


def validate_group_isolation(assignments):
    """Check actual assignments, not the producer's claimed overlap counters."""
    for key in ('structure_group_id', *GROUP_KEYS):
        partitions, folds = defaultdict(set), defaultdict(set)
        for row in assignments:
            require(bool(row.get(key)), f'Missing group key {key}')
            partitions[row[key]].add(row['dataset_partition'])
            if row['dataset_partition'] == 'development':
                folds[row[key]].add(row['cv_validation_fold'])
        require(all(len(v) == 1 for v in partitions.values()), f'Group overlap across development/test: {key}')
        require(all(len(v) == 1 for v in folds.values()), f'Group overlap across CV folds: {key}')


def validate_split(rows, split):
    assignments = split['assignments']
    require([r['compound_id'] for r in assignments] == [r['compound_id'] for r in rows], 'Split ID order mismatch')
    require([r['label'] for r in assignments] == [r['label'] for r in rows], 'Split labels differ from DATA')
    identity = [(r['compound_id'], r['PubChem_CID'], r['standardized_smiles'], r['label']) for r in rows]
    identity_hash = hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    require(split['manifest']['source_identity_sha256'] == identity_hash, 'Split source identity/label fingerprint mismatch')
    development = [r for r in assignments if r['dataset_partition'] == 'development']
    test = [r for r in assignments if r['dataset_partition'] == 'test']
    target_test = round(len(rows) * .20)
    target_positive = round(sum(r['label'] for r in rows) * target_test / len(rows))
    require(len(development) + len(test) == len(rows) and len(test) == target_test,
            '80/20 partition coverage/count mismatch')
    require(sum(r['label'] for r in test) == target_positive and sum(1-r['label'] for r in test) == target_test-target_positive,
            'Test class quotas differ from the declared stratification rule')
    require({r['cv_validation_fold'] for r in development} == set(range(1, 11)) and all(r['cv_validation_fold'] is None for r in test), 'Development folds/test blank fold mismatch')
    validate_group_isolation(assignments)
    for partition in ('development', 'test'):
        selected = [r for r in assignments if r['dataset_partition'] == partition]
        expected = {'n': len(selected), 'positive': sum(r['label'] for r in selected),
                    'negative': sum(1-r['label'] for r in selected), 'groups': len({r['structure_group_id'] for r in selected})}
        require(split['manifest']['partition_counts'][partition] == expected, f'Partition summary differs from assignments: {partition}')
    for fold in range(1, 11):
        valid = [r for r in development if r['cv_validation_fold'] == fold]
        train = [r for r in development if r['cv_validation_fold'] != fold]
        require({r['label'] for r in valid} == {r['label'] for r in train} == {0, 1}, f'CV{fold:02d} does not contain both classes')
        train_ids, valid_ids = {r['compound_id'] for r in train}, {r['compound_id'] for r in valid}
        require(not train_ids & valid_ids and train_ids | valid_ids == {r['compound_id'] for r in development}, f'CV{fold:02d} training/validation not complementary')
        expected = {'fold': fold, 'validation_n': len(valid), 'validation_positive': sum(r['label'] for r in valid),
                    'validation_negative': sum(1-r['label'] for r in valid), 'training_n': len(train),
                    'validation_groups': len({r['structure_group_id'] for r in valid})}
        require(split['manifest']['cross_validation'][fold-1] == expected, f'CV{fold:02d} summary differs from assignments')
    # Independent DFS reconstructs seven-key connected components.
    key_members = [defaultdict(list) for _ in GROUP_KEYS]
    adjacency = defaultdict(set)
    for i, row in enumerate(assignments):
        for key, mapping in zip(GROUP_KEYS, key_members):
            mapping[row[key]].append(i)
    for mapping in key_members:
        for members in mapping.values():
            for i in members[1:]:
                adjacency[members[0]].add(i)
                adjacency[i].add(members[0])
    visited, components = set(), set()
    for i in range(len(rows)):
        if i in visited:
            continue
        stack, members = [i], set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            members.add(assignments[node]['compound_id'])
            stack.extend(adjacency[node] - visited)
        components.add(frozenset(members))
    recorded = defaultdict(set)
    for r in assignments:
        recorded[r['structure_group_id']].add(r['compound_id'])
    require(components == {frozenset(v) for v in recorded.values()}, 'Seven-key connected-component members mismatch')
    require(all(r['structure_group_size'] == len(recorded[r['structure_group_id']]) for r in assignments), 'Structure group sizes mismatch')
    require(split['manifest']['group_count'] == len(recorded), 'Group count summary mismatch')


def _scaffold(molecule, coordination):
    values = []
    for fragment in Chem.GetMolFrags(molecule, asMols=True):
        if coordination and not any(a.GetAtomicNum() == 6 for a in fragment.GetAtoms()):
            continue
        value = MurckoScaffold.GetScaffoldForMol(fragment)
        if value.GetNumAtoms():
            values.append(Chem.MolToSmiles(value, isomericSmiles=False))
    return '.'.join(sorted(values))


def _validate_group_graphs(rows, assignments):
    normalizer, reionizer, uncharger = ms.Normalizer(), ms.Reionizer(), ms.Uncharger(force=True)
    copies = []
    for row, actual in zip(rows, assignments):
        molecule = Chem.MolFromSmiles(row['standardized_smiles'])
        require(molecule is not None, f'Invalid DATA SMILES: {row["compound_id"]}')
        Chem.RemoveStereochemistry(molecule)
        for atom in molecule.GetAtoms():
            atom.SetIsotope(0)
        molecule = Chem.RemoveHs(molecule)
        coordination = row.get('representation_type') == 'complete_coordination_entity'
        raw_scaf = _scaffold(molecule, coordination)
        if not coordination:
            molecule = uncharger.uncharge(reionizer.reionize(normalizer.normalize(molecule)))
        parent = Chem.MolToSmiles(molecule, isomericSmiles=False)
        scaf = _scaffold(molecule, coordination)
        protomer = rdMolHash.MolHash(molecule, rdMolHash.HashFunction.HetAtomProtomerv2)
        ep = rdMolHash.MolHash(Chem.MolFromSmiles(parent), rdMolHash.HashFunction.ElementGraph)
        es = rdMolHash.MolHash(Chem.MolFromSmiles(scaf), rdMolHash.HashFunction.ElementGraph) if scaf else ep
        expected = {'grouping_parent_smiles': parent, 'grouping_raw_scaffold_smiles': raw_scaf, 'grouping_scaffold_smiles': scaf,
                    'protomer_group_signature': protomer, 'protomer_group_key': 'H_' + _short(protomer),
                    'scaffold_group_key': 'S_' + _short(scaf) if scaf else 'A_' + _short(actual['parent_group_signature']),
                    'raw_scaffold_group_key': 'R_' + _short(raw_scaf) if raw_scaf else 'A_' + _short(actual['parent_group_signature']),
                    'element_parent_signature': ep, 'element_parent_group_key': 'EP_' + _short(ep),
                    'element_scaffold_signature': es, 'element_scaffold_group_key': ('ES_' if scaf else 'EA_') + _short(es),
                    'parent_group_key': 'P_' + _short(actual['parent_group_signature']),
                    'tautomer_scaffold_group_key': 'T_' + _short(actual['grouping_tautomer_scaffold_smiles']) if actual['grouping_tautomer_scaffold_smiles'] else 'A_' + _short(actual['parent_group_signature'])}
        require(all(actual.get(k) == v for k, v in expected.items()), f'Independently recomputed group graph/key mismatch: {row["compound_id"]}')
        copies.append((molecule, scaf, coordination))
    # Fixed positions sampled from this DATA population, with no historical IDs.
    indexes = sorted(set(np.linspace(0, len(rows) - 1, min(12, len(rows)), dtype=int).tolist()))
    enum = ms.TautomerEnumerator()
    enum.SetMaxTautomers(1000)
    enum.SetMaxTransforms(1000)
    with rdBase.BlockLogs():
        for i in indexes:
            molecule, scaf, coordination = copies[i]
            expected_parent = Chem.MolToSmiles(molecule, isomericSmiles=False) if coordination else Chem.MolToSmiles(enum.PickCanonical(enum.Enumerate(molecule)), isomericSmiles=False)
            expected_scaf = Chem.MolToSmiles(enum.PickCanonical(enum.Enumerate(Chem.MolFromSmiles(scaf))), isomericSmiles=False) if scaf else ''
            require(assignments[i]['parent_group_signature'] == expected_parent and assignments[i]['grouping_tautomer_scaffold_smiles'] == expected_scaf, f'Canonical tautomer sample mismatch: {rows[i]["compound_id"]}')
    return {'all_rows_fast_keys_recomputed': 5, 'canonical_tautomer_sample_rows': len(indexes)}


def validate_feature_payload(rows, matrices, payload, descriptor_names):
    """Check actual new JSON encodings against aligned raw matrices."""
    require(payload['manifest']['input_records_sha256'] == records_fingerprint(rows), 'Feature source fingerprint mismatch (including labels)')
    ids = [r['compound_id'] for r in rows]
    require([r['compound_id'] for r in payload['fingerprints']] == ids and [r['compound_id'] for r in payload['rdkit166']] == ids, 'Feature JSON ID/order mismatch')
    for i, row in enumerate(payload['fingerprints']):
        require(row['row_index'] == i, 'Fingerprint row_index mismatch')
        for name, width in FP_WIDTHS.items():
            bits = row[name]
            require(isinstance(bits, str) and len(bits) == width and not set(bits) - {'0', '1'}, f'Invalid fingerprint bitstring: {name}')
            decoded = np.frombuffer(bits.encode('ascii'), dtype=np.uint8) - 48
            require(np.array_equal(decoded, matrices[name][i]), f'Fingerprint bit mismatch: {row["compound_id"]}/{name}')
    for i, row in enumerate(payload['rdkit166']):
        values = np.asarray([row[name] for name in descriptor_names], dtype=float)
        require(row['row_index'] == i and np.isfinite(values).all() and np.allclose(values, matrices['rdkit166'][i], rtol=1e-12, atol=1e-12), f'Descriptor mismatch: {row["compound_id"]}')


def _validate_features(rows, directory, descriptor_file, source_info):
    manifest = _read(directory / 'feature_manifest.json')
    require(manifest['input_records_sha256'] == records_fingerprint(rows), 'Feature source records fingerprint mismatch')
    require(manifest['source'] == (source_info or {}), 'Feature source metadata mismatch')
    names = [s.strip() for s in descriptor_file.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    require(len(names) == len(set(names)) == 166, 'Expected 166 unique frozen descriptor names')
    require(manifest['descriptor_file_sha256'] == _sha(descriptor_file), 'Descriptor configuration fingerprint mismatch')
    ids, smiles = [r['compound_id'] for r in rows], [r['standardized_smiles'] for r in rows]
    molecules = [Chem.MolFromSmiles(s) for s in smiles]
    require(all(m is not None for m in molecules), 'Invalid source molecule')
    snapshots = [Chem.MolToSmiles(Chem.Mol(m), isomericSmiles=True) for m in molecules]
    generators = {'morgan2048': rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True),
                  'atompair1024': rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024, includeChirality=True),
                  'torsion1024': rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024, includeChirality=True)}
    matrices = {}
    for name, width in [*FP_WIDTHS.items(), ('rdkit166', 166)]:
        file = directory / f'{name}.npz'
        require(_sha(file) == manifest['representations'][name]['file_sha256'], f'NPZ file fingerprint mismatch: {name}')
        with np.load(file, allow_pickle=False) as archive:
            X = archive['X']
            require(X.shape == (len(rows), width) and np.isfinite(X).all(), f'Feature dimensions/nonfinite values: {name}')
            require(archive['compound_id'].tolist() == ids and archive['standardized_smiles'].tolist() == smiles and np.array_equal(archive['row_index'], np.arange(len(rows))), f'NPZ identity/order mismatch: {name}')
            require(str(archive['descriptor_file_sha256'].item()) == _sha(descriptor_file) and json.loads(str(archive['source_info_json'].item())) == (source_info or {}), f'NPZ source/config metadata mismatch: {name}')
            if name == 'rdkit166':
                require(archive['descriptor_names'].tolist() == names, 'NPZ descriptor order mismatch')
            matrices[name] = X
        definition = manifest['representations'][name]
        require(definition['shape'] == list(X.shape) and definition['dtype'] == str(X.dtype) and definition['missing_values'] == 0, f'Feature manifest dimensions/type/missingness mismatch: {name}')
        dtype = np.dtype('<f8') if np.issubdtype(X.dtype, np.floating) else np.dtype('u1')
        numeric = hashlib.sha256()
        numeric.update(json.dumps({'shape': list(X.shape), 'dtype': dtype.str}, sort_keys=True).encode())
        numeric.update(np.asarray(X, dtype=dtype).tobytes(order='C'))
        require(numeric.hexdigest() == definition['numeric_sha256'], f'Feature numeric fingerprint mismatch: {name}')
        if name != 'rdkit166':
            require(definition['include_chirality'] == (name != 'maccs166') and (name != 'morgan2048' or definition['radius'] == 2), f'Fingerprint definition metadata mismatch: {name}')
            expected = np.zeros((len(rows), width), dtype=np.uint8)
            for i, molecule in enumerate(molecules):
                if name == 'maccs166':
                    full = np.zeros(167, dtype=np.uint8)
                    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(molecule), full)
                    expected[i] = full[1:]
                else:
                    DataStructs.ConvertToNumpyArray(generators[name].GetFingerprint(molecule), expected[i])
            require(np.array_equal(expected, X), f'Independently recomputed fingerprint differs: {name}')
    functions = dict(Descriptors.descList)
    expected = np.full((len(rows), 166), np.nan)
    with rdBase.BlockLogs():
        for i, smiles in enumerate(smiles):
            molecule = Chem.MolFromSmiles(smiles)
            for j, name in enumerate(names):
                expected[i, j] = float(functions[name](molecule))
    require(np.isfinite(expected).all() and np.allclose(expected, matrices['rdkit166'], rtol=1e-12, atol=1e-12), 'Independently recomputed 166-descriptor matrix differs')
    validate_feature_payload(rows, matrices, _read(directory / 'features_payload.json'), names)
    require([Chem.MolToSmiles(Chem.Mol(m), isomericSmiles=True) for m in molecules] == snapshots, 'Source graph/stereo changed while validating')
    return matrices, names


def _validate_preprocessing(rows, assignments, X, directory, names):
    ids = np.asarray([r['compound_id'] for r in rows])
    stages, parameter_rows = [], []
    for fold in [None, *range(1, 11)]:
        stage = 'final_development' if fold is None else f'CV{fold:02d}'
        train = np.asarray([i for i, r in enumerate(assignments) if r['dataset_partition'] == 'development' and (fold is None or r['cv_validation_fold'] != fold)])
        other = np.asarray([i for i, r in enumerate(assignments) if (r['dataset_partition'] == 'test' if fold is None else r['dataset_partition'] == 'development' and r['cv_validation_fold'] == fold)])
        values = X[train]
        medians = np.nanmedian(values, axis=0)
        imputed = np.where(np.isnan(values), medians, values)
        means, scales = np.mean(imputed, axis=0), np.std(imputed, axis=0, ddof=0)
        variance = scales ** 2
        epsilon = np.finfo(np.float64).eps
        constant_bound = len(train) * epsilon * variance + (len(train) * means * epsilon) ** 2
        scales[variance <= constant_bound] = 1.
        with np.load(directory / f'{stage}.npz', allow_pickle=False) as archive:
            require(archive['training_ids'].tolist() == ids[train].tolist() and archive['other_ids'].tolist() == ids[other].tolist(), f'Preprocessing training/held-out identity mismatch: {stage}')
            require(np.array_equal(archive['training_indices'], train) and np.array_equal(archive['other_indices'], other), f'Preprocessing fit indices leak across split: {stage}')
            for key, expected in [('medians', medians), ('means', means), ('scales', scales)]:
                require(np.isfinite(archive[key]).all() and np.allclose(archive[key], expected, rtol=1e-10, atol=1e-12), f'Train-only preprocessing statistic mismatch: {stage}/{key}')
        stages.append({'stage': stage, 'training': len(train), 'held_out': len(other)})
        for j, name in enumerate(names):
            parameter_rows.append({'stage': stage, 'descriptor': name, 'n_fit': len(train),
                                   'training_median': medians[j], 'training_mean': means[j], 'training_scale': scales[j],
                                   'missing_training_values': int(np.isnan(values[:, j]).sum())})
    saved_json = _read(directory / 'parameters.json')
    with (directory / 'parameters.csv').open(encoding='utf-8-sig', newline='') as handle:
        saved_csv = list(csv.DictReader(handle))
    for kind, saved in [('JSON', saved_json), ('CSV', saved_csv)]:
        require(len(saved) == len(parameter_rows), f'Preprocessing {kind} parameter count mismatch')
        for actual, expected in zip(saved, parameter_rows):
            for key, wanted in expected.items():
                if isinstance(wanted, str):
                    ok = actual.get(key) == wanted
                else:
                    ok = np.isclose(float(actual[key]), wanted, rtol=1e-10, atol=1e-12)
                require(ok, f'Preprocessing {kind} parameter mismatch: {expected["stage"]}/{expected["descriptor"]}/{key}')
    return stages


def _validate_csvs(directory, rows, split, matrices, names):
    assignments = split['assignments']
    path = directory / 'split_assignments.csv'
    with path.open(encoding='utf-8-sig', newline='') as handle:
        saved = list(csv.DictReader(handle))
    require(len(saved) == len(assignments), 'Split CSV row count mismatch')
    for actual, expected in zip(saved, assignments):
        require(actual['compound_id'] == expected['compound_id'], 'Split CSV ID order mismatch')
        for key, value in actual.items():
            require(key in expected and value == ('' if expected[key] is None else str(expected[key])), f'Split CSV field mismatch: {expected["compound_id"]}/{key}')
    with (directory / 'cross_validation_folds.csv').open(encoding='utf-8-sig', newline='') as handle:
        cv_rows = list(csv.DictReader(handle))
    require(len(cv_rows) == 10 and all(all(str(wanted[key]) == value for key, value in actual.items()) for actual, wanted in zip(cv_rows, split['manifest']['cross_validation'])), 'Cross-validation CSV summaries differ')
    schema = _read(directory / 'modeling_table_schema.json')
    with (directory / 'modeling_data.csv').open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        exported = list(reader)
    validate_modeling_rows(rows, assignments, exported, headers, schema, matrices, names)


def validate_modeling_rows(rows, assignments, exported, headers, schema, matrices, names):
    """Independently compare every delivered CSV field with source/arrays/folds."""
    fp_columns = {'Morgan2048_bits': 'morgan2048', 'MACCS166_bits': 'maccs166', 'AtomPair1024_bits': 'atompair1024', 'TopologicalTorsion1024_bits': 'torsion1024'}
    expected_headers = list(rows[0]) + ['dataset_partition', 'cv_validation_fold', 'structure_group_id', 'structure_group_size'] + [f'CV{k:02d}_role' for k in range(1,11)] + list(fp_columns) + ['RDKit_' + name for name in names]
    require(headers == expected_headers == schema['headers'] and len(set(headers)) == len(headers) and all(headers), 'Modeling CSV/source schema column mismatch')
    require(schema['columns'] == len(headers), 'Modeling CSV schema column count mismatch')
    require(len(exported) == len(rows) == schema['rows'], 'Modeling CSV row count mismatch')
    for i, (actual, source, assignment) in enumerate(zip(exported, rows, assignments)):
        require(all(actual[key] == str(value) for key, value in source.items()), f'Modeling CSV changed DATA fields: {source["compound_id"]}')
        for key in ['dataset_partition', 'cv_validation_fold', 'structure_group_id', 'structure_group_size']:
            require(actual[key] == ('' if assignment[key] is None else str(assignment[key])), f'Modeling CSV split/fold mismatch: {source["compound_id"]}/{key}')
        for fold in range(1, 11):
            expected = 'test' if assignment['dataset_partition'] == 'test' else 'validation' if assignment['cv_validation_fold'] == fold else 'train'
            require(actual[f'CV{fold:02d}_role'] == expected, f'Modeling CSV CV role mismatch: {source["compound_id"]}/{fold}')
        for column, representation in fp_columns.items():
            bits = actual[column]
            require(len(bits) == FP_WIDTHS[representation] and not set(bits) - {'0', '1'} and np.array_equal(np.frombuffer(bits.encode('ascii'), dtype=np.uint8)-48, matrices[representation][i]), f'Modeling CSV fingerprint mismatch: {source["compound_id"]}/{column}')
        require(np.allclose(np.asarray([float(actual['RDKit_'+name]) for name in names]), matrices['rdkit166'][i], rtol=1e-12, atol=1e-12), f'Modeling CSV descriptor mismatch: {source["compound_id"]}')


def validate(rows, artifacts_dir, descriptor_file, report_path, source_info=None):
    """Validate DATA-derived artifacts; write a report and raise on any failure."""
    directory, descriptor_file, report_path = Path(artifacts_dir), Path(descriptor_file), Path(report_path)
    protected = {directory / name for name in ['split_payload.json', 'split_manifest.json', 'source_snapshot.json', 'run_manifest.json',
                                               'modeling_table_schema.json', 'features/features_payload.json', 'features/feature_manifest.json',
                                               'features/stereo_retention.json', 'preprocessing/parameters.json', 'preprocessing/preprocessing_manifest.json']}
    protected.add(descriptor_file)
    protected.update(directory.rglob('*.npz'))
    require(report_path.suffix.lower() == '.json' and report_path.resolve() not in {p.resolve() for p in protected}, 'Validation report cannot overwrite source/artifact files')
    if report_path.exists():
        previous = _read(report_path)
        require(isinstance(previous, dict) and 'checks' in previous and 'issues' in previous,
                'Validation report cannot overwrite an existing data/manifest file')
    before = records_fingerprint(rows)
    report = {'passed': False, 'scope': 'Current DATA rows and newly generated artifacts only', 'rows': len(rows),
              'source': source_info or {}, 'input_records_sha256': before, 'rdkit_version': rdBase.rdkitVersion,
              'checks': {}, 'issues': []}
    try:
        validate_source(rows, source_info)
        report['checks']['DATA_identity_label_fingerprint'] = True
        split = _read(directory / 'split_payload.json')
        validate_split(rows, split)
        report['checks']['all_holdout_CV_and_seven_key_components'] = True
        report['group_graph_checks'] = _validate_group_graphs(rows, split['assignments'])
        report['checks']['independent_group_graphs'] = True
        matrices, names = _validate_features(rows, directory / 'features', descriptor_file, source_info)
        report['checks']['all_FP_bits_and166_descriptor_values_independently_recomputed'] = True
        report['preprocessing'] = _validate_preprocessing(rows, split['assignments'], matrices['rdkit166'], directory / 'preprocessing', names)
        report['checks']['all11_training_only_preprocessing_stages'] = True
        _validate_csvs(directory, rows, split, matrices, names)
        report['checks']['all_modeling_split_fold_and_parameter_CSV_JSON_values_match'] = True
        require(before == records_fingerprint(rows), 'DATA rows changed during validation')
        report['checks']['DATA_rows_unchanged'] = True
        report['passed'] = True
        report['features'] = {'descriptor_cells_recomputed': len(rows) * len(names), 'descriptor_nonfinite_values': 0, 'FP_widths': FP_WIDTHS}
    except Exception as error:
        report['issues'].append({'type': type(error).__name__, 'message': str(error)})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if not report['passed']:
        raise ValidationError('Independent validation failed: ' + report['issues'][0]['message'])
    return report
