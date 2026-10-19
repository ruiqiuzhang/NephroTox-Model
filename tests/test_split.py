"""Synthetic critical regression tests; never read scientific or historical data."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from nephrotox.split import GROUP_KEY_FIELDS, build_grouping_keys, connected_groups, leakage_checks, split
from nephrotox.preprocessing import get_fold_indices, prepare_fold


def synthetic_rows(n=60):
    return [{'compound_id':f'TEST{i:04d}', 'name':f'Synthetic fixture {i}',
             'PubChem_CID':i+1, 'standardized_smiles':'C'*(i+1),
             'label':i%2, 'representation_type':'single_parent'} for i in range(n)]


class StructureSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = synthetic_rows()
        cls.result = split(cls.rows)

    def test_seven_keys_participate_in_transitive_components(self):
        rows = synthetic_rows(3)
        for position, field in enumerate(GROUP_KEY_FIELDS):
            with self.subTest(key=field):
                metadata = [dict(compound_id=r['compound_id'],label=r['label'],
                                 **{key:f'{key}:{i}' for key in GROUP_KEY_FIELDS}) for i,r in enumerate(rows)]
                metadata[1][field] = metadata[0][field]
                bridge = GROUP_KEY_FIELDS[(position+1)%len(GROUP_KEY_FIELDS)]
                metadata[2][bridge] = metadata[1][bridge]
                groups = connected_groups(rows, metadata)
                self.assertEqual(len(groups),1)
                self.assertEqual(groups[0]['members'],[r['compound_id'] for r in rows])

    def test_chemistry_keys_preserve_source_stereo_and_group_equivalents(self):
        smiles = ['N[C@H](C)C(=O)O','N[C@@H](C)C(=O)O','CN','C[NH3+]',
                  '[13CH3]O','CO','O=c1cccc[nH]1','Oc1ccccn1','Cc1ccccc1','Oc1ccccc1','CCO','CCN']
        rows = synthetic_rows(len(smiles))
        for row,value in zip(rows,smiles):
            row['standardized_smiles']=value
        original = copy.deepcopy(rows)
        metadata = build_grouping_keys(rows)
        connected_groups(rows, metadata)
        self.assertEqual(rows,original)
        for a,b in [(0,1),(2,3),(4,5),(6,7),(8,9)]:
            self.assertEqual(metadata[a]['structure_group_id'],metadata[b]['structure_group_id'])
        self.assertNotEqual(metadata[10]['structure_group_id'],metadata[11]['structure_group_id'])
        self.assertTrue(all(metadata[i]['scaffold_group_key'] for i in [10,11]))

    def test_exact_holdout_and_ten_grouped_folds(self):
        manifest = self.result['manifest']
        self.assertEqual(manifest['partition_counts']['test']['n'],12)
        self.assertEqual(manifest['partition_counts']['test']['positive'],6)
        self.assertEqual(manifest['partition_counts']['development']['n'],48)
        self.assertEqual(len(manifest['cross_validation']),10)
        self.assertTrue(all(v==0 for v in manifest['leakage_checks'].values()))
        self.assertEqual(len(manifest['leakage_checks']),16)
        for fold in manifest['cross_validation']:
            self.assertGreater(fold['validation_positive'],0)
            self.assertGreater(fold['validation_negative'],0)

    def test_deterministic_and_input_unmodified(self):
        original = copy.deepcopy(self.rows)
        repeat = split(self.rows)
        self.assertEqual(self.rows,original)
        self.assertEqual(self.result['assignments'],repeat['assignments'])
        self.assertEqual(self.result['groups'],repeat['groups'])

    def test_missing_grouping_key_fails(self):
        metadata = copy.deepcopy(self.result['assignments'])
        del metadata[0]['element_parent_group_key']
        with self.assertRaisesRegex(ValueError,'Missing grouping key'):
            connected_groups(self.rows,metadata)

    def test_group_leak_and_invalid_fold_fail(self):
        metadata = copy.deepcopy(self.result['assignments'])
        a = next(r for r in metadata if r['dataset_partition']=='development')
        b = next(r for r in metadata if r['dataset_partition']=='test')
        a['element_parent_group_key']=b['element_parent_group_key']
        with self.assertRaisesRegex(ValueError,'Structure leakage'):
            leakage_checks(metadata)
        for fold in [0,11,1.0,True]:
            with self.subTest(fold=fold),self.assertRaises(ValueError):
                get_fold_indices(self.rows,self.result['assignments'],fold)

    def test_changed_label_or_test_fold_fails(self):
        for field,value in [('label',2),('cv_validation_fold',1)]:
            assignments = copy.deepcopy(self.result['assignments'])
            row = next(r for r in assignments if r['dataset_partition']=='test')
            row[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                get_fold_indices(self.rows,assignments,1)

    def test_optimized_python_still_fails_on_bad_input(self):
        code = "from nephrotox.split import split; rows=[dict(compound_id='duplicate',standardized_smiles='C',label=i%2,representation_type='single_parent')for i in range(20)]; split(rows)"
        environment = dict(os.environ,PYTHONPATH=str(ROOT/'src'))
        process = subprocess.run([sys.executable,'-X','utf8','-O','-c',code],env=environment,
                                 capture_output=True,text=True,encoding='utf8')
        self.assertNotEqual(process.returncode,0)
        self.assertIn('Duplicate DATA compound IDs',process.stderr)

    def test_evaluation_values_cannot_change_fitted_preprocessing(self):
        assignments=self.result['assignments']
        for fold in [None,1]:
            with self.subTest(fold=fold):
                training,evaluation=get_fold_indices(self.rows,assignments,fold)
                matrix=np.arange(len(self.rows)*166,dtype=float).reshape(len(self.rows),166)
                matrix[training[0],0]=np.nan
                before=prepare_fold(matrix,self.rows,assignments,'rdkit166',fold)
                changed=matrix.copy();changed[evaluation]+=1_000_000
                after=prepare_fold(changed,self.rows,assignments,'rdkit166',fold)
                for name,attribute in [('imputer','statistics_'),('scaler','mean_'),('scaler','scale_')]:
                    np.testing.assert_array_equal(getattr(before['preprocessing'][name],attribute),
                                                  getattr(after['preprocessing'][name],attribute))
                np.testing.assert_array_equal(before['X_train'],after['X_train'])
                key='X_test' if fold is None else 'X_validation'
                self.assertFalse(np.array_equal(before[key],after[key]))
                self.assertFalse(set(before['training_ids'])&set(before['other_ids']))

    def test_fingerprints_stay_binary_and_invalid_shape_fails(self):
        matrix=np.zeros((len(self.rows),166),dtype=np.uint8)
        matrix[::2,0]=1
        result=prepare_fold(matrix,self.rows,self.result['assignments'],'maccs166',1)
        np.testing.assert_array_equal(result['X_train'],matrix[result['training_indices']])
        self.assertEqual(result['preprocessing'],{})
        with self.assertRaises(ValueError):
            prepare_fold(matrix[:,:165],self.rows,self.result['assignments'],'maccs166',1)
        matrix[0,0]=2
        with self.assertRaises(ValueError):
            prepare_fold(matrix,self.rows,self.result['assignments'],'maccs166',1)


if __name__=='__main__':
    unittest.main(verbosity=2)
