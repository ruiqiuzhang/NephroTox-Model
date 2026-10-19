"""Negative controls mutate real newly computed artifacts, never formal DATA."""
from __future__ import annotations
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from nephrotox.features import compute, _numeric_hash
from nephrotox.pipeline import export_modeling_table
from nephrotox.validate import (GROUP_KEYS, ValidationError, _validate_features,
                               records_fingerprint, validate_feature_payload,
                               validate_group_isolation, validate_source,
                               validate_modeling_rows, validate)

ROOT = Path(__file__).resolve().parents[1]


class ValidationNegativeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='nephrotox-validation-tests-')
        cls.directory = Path(cls.temporary.name) / 'features'
        cls.rows = [
            {'compound_id': 'TEST01', 'name': 'ethanol', 'PubChem_CID': 702, 'standardized_smiles': 'CCO', 'label': 0},
            {'compound_id': 'TEST02', 'name': 'ethylamine', 'PubChem_CID': 6341, 'standardized_smiles': 'CCN', 'label': 1},
            {'compound_id': 'TEST03', 'name': 'lactic acid', 'PubChem_CID': 612, 'standardized_smiles': 'C[C@H](O)C(=O)O', 'label': 0},
            {'compound_id': 'TEST04', 'name': 'acetic acid', 'PubChem_CID': 176, 'standardized_smiles': 'CC(=O)O', 'label': 1},
        ]
        cls.source_info = {'rows_fingerprint': records_fingerprint(cls.rows), 'source_policy': 'synthetic unit-test controls, not modeling DATA'}
        cls.descriptor_file = ROOT / 'configs/rdkit_descriptors_166.txt'
        compute(cls.rows, cls.directory, cls.descriptor_file, cls.source_info)
        cls.payload = json.loads((cls.directory / 'features_payload.json').read_text(encoding='utf-8'))
        cls.names = cls.payload['manifest']['representations']['rdkit166']['descriptor_order']
        cls.matrices = {}
        for name in ['morgan2048', 'maccs166', 'atompair1024', 'torsion1024', 'rdkit166']:
            with np.load(cls.directory / (name + '.npz'), allow_pickle=False) as archive:
                cls.matrices[name] = archive['X']
        cls.model_assignments = cls.assignments()
        output = Path(cls.temporary.name)
        export_modeling_table(cls.rows, cls.model_assignments, output, cls.names)
        cls.model_schema = json.loads((output / 'modeling_table_schema.json').read_text(encoding='utf-8'))
        with (output / 'modeling_data.csv').open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            cls.model_headers = reader.fieldnames
            cls.model_rows = list(reader)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_unmodified_new_artifacts_pass(self):
        validate_source(self.rows, self.source_info)
        validate_feature_payload(self.rows, self.matrices, self.payload, self.names)
        _validate_features(self.rows, self.directory, self.descriptor_file, self.source_info)

    def test_one_valid_binary_bit_flip_fails(self):
        changed = copy.deepcopy(self.payload)
        bits = changed['fingerprints'][0]['morgan2048']
        changed['fingerprints'][0]['morgan2048'] = ('1' if bits[0] == '0' else '0') + bits[1:]
        with self.assertRaisesRegex(ValidationError, 'Fingerprint bit mismatch'):
            validate_feature_payload(self.rows, self.matrices, changed, self.names)

    def test_one_descriptor_value_change_fails(self):
        changed = copy.deepcopy(self.payload)
        changed['rdkit166'][0][self.names[0]] += 1.0
        with self.assertRaisesRegex(ValidationError, 'Descriptor mismatch'):
            validate_feature_payload(self.rows, self.matrices, changed, self.names)

    def test_swapped_labels_fail_even_with_unchanged_class_counts(self):
        changed = copy.deepcopy(self.rows)
        changed[0]['label'], changed[1]['label'] = changed[1]['label'], changed[0]['label']
        with self.assertRaisesRegex(ValidationError, 'fingerprint mismatch'):
            validate_source(changed, self.source_info)
        with self.assertRaisesRegex(ValidationError, 'source fingerprint mismatch'):
            validate_feature_payload(changed, self.matrices, self.payload, self.names)

    def test_boolean_labels_cannot_bypass_binary_label_guard(self):
        changed = copy.deepcopy(self.rows)
        changed[0]['label'] = False
        with self.assertRaisesRegex(ValidationError, 'integer binary'):
            validate_source(changed)

    @staticmethod
    def assignments():
        result = []
        for i in range(4):
            group = 'A' if i < 2 else 'B'
            row = {'compound_id': f'TEST{i+1:02d}', 'label': i % 2, 'structure_group_id': group,
                   'structure_group_size': 2,
                   'dataset_partition': 'development' if i < 2 else 'test',
                   'cv_validation_fold': 1 if i < 2 else None}
            row.update({key: key + group for key in GROUP_KEYS})
            result.append(row)
        return result

    def test_move_same_parent_group_member_to_test_fails(self):
        changed = self.assignments()
        # Swap two rows so aggregate development/test counts remain unchanged.
        changed[1]['dataset_partition'], changed[1]['cv_validation_fold'] = 'test', None
        changed[2]['dataset_partition'], changed[2]['cv_validation_fold'] = 'development', 1
        with self.assertRaisesRegex(ValidationError, 'Group overlap across development/test'):
            validate_group_isolation(changed)

    def test_change_validation_fold_inside_group_fails(self):
        changed = self.assignments()
        changed[1]['cv_validation_fold'] = 2
        with self.assertRaisesRegex(ValidationError, 'Group overlap across CV folds'):
            validate_group_isolation(changed)

    def test_actual_producer_modeling_csv_passes(self):
        validate_modeling_rows(self.rows, self.model_assignments, self.model_rows, self.model_headers,
                               self.model_schema, self.matrices, self.names)

    def test_modeling_csv_source_field_change_fails(self):
        changed = copy.deepcopy(self.model_rows)
        changed[0]['name'] = 'changed source identity'
        with self.assertRaisesRegex(ValidationError, 'changed DATA fields'):
            validate_modeling_rows(self.rows, self.model_assignments, changed, self.model_headers,
                                   self.model_schema, self.matrices, self.names)

    def test_modeling_csv_role_change_fails(self):
        changed = copy.deepcopy(self.model_rows)
        changed[0]['CV01_role'] = 'train'
        with self.assertRaisesRegex(ValidationError, 'CSV CV role mismatch'):
            validate_modeling_rows(self.rows, self.model_assignments, changed, self.model_headers,
                                   self.model_schema, self.matrices, self.names)

    def test_modeling_csv_descriptor_change_fails(self):
        changed = copy.deepcopy(self.model_rows)
        column = 'RDKit_' + self.names[0]
        changed[0][column] = str(float(changed[0][column]) + 1.0)
        with self.assertRaisesRegex(ValidationError, 'CSV descriptor mismatch'):
            validate_modeling_rows(self.rows, self.model_assignments, changed, self.model_headers,
                                   self.model_schema, self.matrices, self.names)

    def test_report_cannot_overwrite_source_snapshot(self):
        output = Path(self.temporary.name)
        snapshot = output / 'source_snapshot.json'
        snapshot.write_text(json.dumps({'rows': self.rows}), encoding='utf-8')
        before = snapshot.read_bytes()
        with self.assertRaisesRegex(ValidationError, 'cannot overwrite'):
            validate(self.rows, output, self.descriptor_file, snapshot, self.source_info)
        self.assertEqual(before, snapshot.read_bytes())

    def _tampered_npz(self, directory, name):
        shutil.copytree(self.directory, directory)
        path = directory / (name + '.npz')
        with np.load(path, allow_pickle=False) as archive:
            contents = {key: archive[key] for key in archive.files}
        contents['X'] = contents['X'].copy()
        if name == 'rdkit166':
            contents['X'][0, 0] += 1.0
        else:
            contents['X'][0, 0] ^= 1
        np.savez_compressed(path, **contents)
        # Update the transport hash too: fresh molecular recomputation must
        # detect this change rather than merely rejecting a stale file hash.
        manifest_path = directory / 'feature_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['representations'][name]['file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['representations'][name]['numeric_sha256'] = _numeric_hash(contents['X'])
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')

    def test_npz_bit_tamper_with_updated_transport_hash_still_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'features'
            self._tampered_npz(directory, 'morgan2048')
            with self.assertRaisesRegex(ValidationError, 'Independently recomputed fingerprint differs'):
                _validate_features(self.rows, directory, self.descriptor_file, self.source_info)

    def test_npz_descriptor_tamper_with_updated_hash_still_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'features'
            self._tampered_npz(directory, 'rdkit166')
            with self.assertRaisesRegex(ValidationError, 'Independently recomputed 166-descriptor matrix differs'):
                _validate_features(self.rows, directory, self.descriptor_file, self.source_info)

    def test_invalid_validation_exits_nonzero_under_python_optimized(self):
        changed = self.assignments()
        changed[1]['cv_validation_fold'] = 2
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(ROOT / 'src')
        result = subprocess.run([sys.executable, '-X', 'utf8', '-O', '-c',
                                 'import json,sys; from nephrotox.validate import validate_group_isolation; validate_group_isolation(json.loads(sys.argv[1]))',
                                 json.dumps(changed)], capture_output=True, text=True, encoding='utf-8', env=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Group overlap across CV folds', result.stderr)


if __name__ == '__main__':
    unittest.main()
