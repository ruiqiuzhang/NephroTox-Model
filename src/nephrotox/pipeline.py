"""Complete offline numerical reproduction, using only the supplied DATA file."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import numpy as np
from rdkit import rdBase

from .data import load_data, rows_fingerprint
from .features import compute, load_descriptor_names
from .preprocessing import prepare_fold
from .split import split

ROOT = Path(__file__).resolve().parents[2]
REPRESENTATIONS = ('morgan2048', 'maccs166', 'atompair1024', 'torsion1024', 'rdkit166')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, headers, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def export_modeling_table(rows, assignments, output, descriptors):
    payload = json.loads((output / 'features/features_payload.json').read_text(encoding='utf-8'))
    ids = [r['compound_id'] for r in rows]
    for collection in (assignments, payload['fingerprints'], payload['rdkit166']):
        if [r['compound_id'] for r in collection] != ids:
            raise ValueError('Source, features and split ID order differ during export')
    group_headers = ['dataset_partition', 'cv_validation_fold', 'structure_group_id', 'structure_group_size']
    role_headers = [f'CV{k:02d}_role' for k in range(1, 11)]
    fp_columns = {'Morgan2048_bits': 'morgan2048', 'MACCS166_bits': 'maccs166',
                  'AtomPair1024_bits': 'atompair1024', 'TopologicalTorsion1024_bits': 'torsion1024'}
    source_columns = list(rows[0])
    headers = source_columns + group_headers + role_headers + list(fp_columns) + ['RDKit_' + d for d in descriptors]
    table = []
    for i, original in enumerate(rows):
        row = dict(original)
        assignment = assignments[i]
        if assignment['label'] != original['label']:
            raise ValueError('Source label changed during export')
        row.update({key: assignment[key] for key in group_headers})
        for k, header in enumerate(role_headers, 1):
            row[header] = ('test' if assignment['dataset_partition'] == 'test' else
                           'validation' if assignment['cv_validation_fold'] == k else 'train')
        row.update({header: payload['fingerprints'][i][name] for header, name in fp_columns.items()})
        row.update({'RDKit_' + d: payload['rdkit166'][i][d] for d in descriptors})
        if any(row[key] != original[key] for key in source_columns):
            raise ValueError('A source field was changed during export')
        table.append(row)
    write_csv(output / 'modeling_data.csv', headers, table)
    write_json(output / 'modeling_table_schema.json', {
        'rows': len(rows), 'columns': len(headers), 'headers': headers,
        'source_columns': source_columns, 'label_column': 'label',
        'fingerprint_columns': fp_columns,
        'descriptor_columns': ['RDKit_' + d for d in descriptors],
        'fingerprint_encoding': 'Fixed-width 0/1 text, left character = matrix bit 0; MACCS starts at original bit 1',
        'csv_reading': 'Read fingerprint columns as strings to retain leading zeroes. NPZ matrices avoid spreadsheet auto-conversion.',
        'excluded_predictors': 'Identity, source, label, split, fold, role and group metadata must not enter numeric predictors.',
    })


def save_preprocessing(rows, assignments, output, descriptors):
    folder = output / 'preprocessing'
    folder.mkdir(parents=True, exist_ok=True)
    parameters, checks = [], []
    for representation in REPRESENTATIONS:
        with np.load(output / f'features/{representation}.npz', allow_pickle=False) as saved:
            matrix = saved['X']
            if saved['compound_id'].tolist() != [r['compound_id'] for r in rows]:
                raise ValueError('NPZ/DATA identity mismatch')
        for fold in (None, *range(1, 11)):
            stage = 'final_development' if fold is None else f'CV{fold:02d}'
            result = prepare_fold(matrix, rows, assignments, representation, fold)
            if representation == 'rdkit166':
                imputer, scaler = result['preprocessing']['imputer'], result['preprocessing']['scaler']
                np.savez_compressed(folder / (stage + '.npz'),
                    medians=imputer.statistics_, means=scaler.mean_, scales=scaler.scale_,
                    training_ids=result['training_ids'], other_ids=result['other_ids'],
                    training_indices=result['training_indices'], other_indices=result['other_indices'])
                for j, descriptor in enumerate(descriptors):
                    parameters.append({'stage': stage, 'descriptor': descriptor,
                        'n_fit': len(result['training_ids']), 'training_median': float(imputer.statistics_[j]),
                        'training_mean': float(scaler.mean_[j]), 'training_scale': float(scaler.scale_[j]),
                        'missing_training_values': int(np.isnan(matrix[result['training_indices'], j]).sum())})
            checks.append({'representation': representation, 'stage': stage,
                'training_rows': len(result['training_ids']), 'evaluation_rows': len(result['other_ids']),
                'training_only_preprocessing': representation == 'rdkit166', 'all_values_finite': True})
    write_csv(folder / 'parameters.csv', list(parameters[0]), parameters)
    write_json(folder / 'parameters.json', parameters)
    write_json(folder / 'preprocessing_manifest.json', {'checked_uses': len(checks), 'checks': checks,
        'raw_features_unchanged': True, 'test_never_in_cross_validation': True,
        'models_trained': False, 'cv_performance_computed': False})


def run(input_path=None, output=None, seed=20260810, validate_only=False):
    output = Path(output) if output else ROOT / 'results'
    output = output.resolve()
    protected = [ROOT, ROOT / 'data', ROOT / 'src', ROOT / 'configs', ROOT / 'tests', ROOT / 'docs']
    if output in protected or any(base != ROOT and base in output.parents for base in protected):
        raise ValueError('Output must not overwrite source code or input data directories')
    rows, source_info = load_data(input_path)
    descriptor_file = ROOT / 'configs/rdkit_descriptors_166.txt'
    if validate_only:
        from .validate import validate
        return validate(rows, output, descriptor_file, output / 'independent_validation.json', source_info=source_info)
    output.mkdir(parents=True, exist_ok=True)
    run_info = {'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': source_info, 'random_seed': seed, 'reused_features_or_group_diagnostics': False,
        'python_version': platform.python_version(), 'platform': platform.system(),
        'dependencies': {name: version(name) for name in ['numpy','scipy','scikit-learn','rdkit','openpyxl']},
        'models_trained': False, 'cv_performance_computed': False, 'figure_generated': False}
    write_json(output / 'run_manifest.json', run_info)
    try:
        write_json(output / 'source_snapshot.json', {'source_info': source_info, 'headers': list(rows[0]), 'rows': rows})
        print(f'[1/6] DATA verified: {len(rows)} rows, {len(rows[0])} columns', flush=True)
        descriptors = load_descriptor_names(descriptor_file)
        feature_manifest = compute(rows, output / 'features', descriptor_file, source_info=source_info)
        print('[2/6] Five representations recomputed from original isomeric SMILES', flush=True)
        with rdBase.BlockLogs():
            split_result = split(rows, seed=seed)
        split_result['manifest']['source'] = source_info
        write_json(output / 'split_payload.json', split_result)
        write_json(output / 'split_manifest.json', split_result['manifest'])
        assignments = split_result['assignments']
        write_csv(output / 'split_assignments.csv', list(assignments[0]), assignments)
        folds = split_result['manifest']['cross_validation']
        write_csv(output / 'cross_validation_folds.csv', list(folds[0]), folds)
        print('[3/6] Entire structure groups assigned to holdout and ten development folds', flush=True)
        save_preprocessing(rows, assignments, output, descriptors)
        print('[4/6] All five representations checked across eleven training/evaluation uses', flush=True)
        export_modeling_table(rows, assignments, output, descriptors)
        print('[5/6] Complete modeling table and split/feature manifests exported', flush=True)
        from .validate import validate
        report = validate(rows, output, descriptor_file, output / 'independent_validation.json', source_info=source_info)
        if not report.get('passed', report.get('summary', {}).get('passed', False)):
            raise ValueError('Independent validation did not report success')
        if rows_fingerprint(rows) != source_info['rows_fingerprint']:
            raise ValueError('The loaded DATA records were mutated')
        load_data(input_path)  # Recheck the sole input workbook at the end.
        run_info.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
            partition_counts=split_result['manifest']['partition_counts'],
            group_count=split_result['manifest']['group_count'],
            source_values_unchanged=True, independent_validation_passed=True,
            descriptor_nonfinite_values=feature_manifest['nonfinite_descriptor_values'])
        write_json(output / 'run_manifest.json', run_info)
        print('[6/6] Independent source, feature, grouping and training-only validation passed', flush=True)
        print(json.dumps({'output': str(output), 'partitions': run_info['partition_counts'],
                          'groups': run_info['group_count'], 'status': 'complete'}, ensure_ascii=False), flush=True)
        return run_info
    except Exception as error:
        run_info.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json(output / 'run_manifest.json', run_info)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Exact DATA workbook; defaults to the bundled byte-identical source')
    parser.add_argument('--output', type=Path, help='Results directory; defaults to this repository/results')
    parser.add_argument('--seed', type=int, default=20260810)
    parser.add_argument('--validate-only', action='store_true', help='Independently check an existing output against DATA')
    args = parser.parse_args()
    run(args.input, args.output, args.seed, args.validate_only)
