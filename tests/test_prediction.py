"""New-query features must equal the training inputs for the same molecules."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from nephrotox.modeling_common import training_context,raw_matrix,prepare_count_features

spec=importlib.util.spec_from_file_location("prediction_entry",ROOT/"scripts/predict.py")
prediction=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prediction)


class PredictionFeatureTests(unittest.TestCase):
    def test_all_query_representations_equal_frozen_training_features(self):
        context=training_context()
        prepare_count_features(context)
        index=context["development"][::83]
        smiles=context["smiles"][index].tolist()
        for rep in ["morgan2048","maccs166","atompair1024","torsion1024","rdkit166",
                    "morgan_count2048","atompair_count1024","torsion_count1024"]:
            with self.subTest(representation=rep):
                self.assertTrue(np.array_equal(prediction.inference_features(smiles,rep),raw_matrix(rep,context)[index]))

    def test_query_validation_preserves_stereo_and_rejects_mixture(self):
        smiles=["C[C@H](O)C(=O)O","C[C@@H](O)C(=O)O","CC(O)C(=O)O"]
        self.assertEqual(prediction.inference_features(smiles,"graph").tolist(),smiles)
        for bad in [["CCO.[Na+]"],["*CC"]]:
            with self.assertRaises(ValueError):
                prediction.inference_features(bad,"graph")


if __name__=="__main__":
    unittest.main()
