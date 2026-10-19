"""Guard the data boundary and resumable selection, including corrupt caches."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nephrotox.modeling_common import training_context, raw_matrix
from nephrotox.modeling_pipeline import _search


class ModelingBoundaryTests(unittest.TestCase):
    def test_graph_selects_the_highest_cv_score_instead_of_fixing_reference(self):
        context = {"y":np.tile([0,1],36), "groups":np.repeat([f"g{i}" for i in range(36)],2),
                   "ids":np.array([f"a{i}" for i in range(72)])}
        job = {"model_id":"graph__GINE", "representation":"graph", "estimator_key":"GINE"}
        configurations = [{"signal":0},{"signal":1},{"signal":2}]
        class Candidate:
            def __init__(self,params): self.params=params
            def fit(self,X,y,groups,sample_ids,log_dir): self.fit_metadata_={"fit_seconds":0.0}; return self
            def predict_proba(self,X):
                p=np.where(X==1,.8,.2) if self.params["signal"]==1 else np.where(X==1,.2,.8)
                if self.params["signal"]==2: p=np.full(len(X),.5)
                return np.column_stack([1-p,p])
        with tempfile.TemporaryDirectory() as temp, patch("nephrotox.modeling_pipeline._candidate_list",return_value=configurations), patch("nephrotox.modeling_pipeline._estimator",side_effect=lambda job,params,evaluation,threads:Candidate(params)) as fit_factory:
            out = Path(temp)
            params = _search(job,context["y"].copy(),context,np.arange(72),out,"protocol",1)
            selected = json.loads((out/"selection.json").read_text(encoding="utf-8"))
            self.assertEqual(params,configurations[1])
            self.assertEqual(selected["selection_method"],"seeded_random_search")
            self.assertEqual(selected["selected_candidate_id_in_code"],1)
            self.assertEqual(len(selected["trials"]),3)
            self.assertTrue((out/"inner_splits.json").exists())
            self.assertEqual(len(list(out.rglob("trial.json"))),9)
            self.assertEqual(fit_factory.call_count,9)
            resumed = _search(job,context["y"].copy(),context,np.arange(72),out,"protocol",1)
            self.assertEqual(resumed,configurations[1])
            self.assertEqual(fit_factory.call_count,9)

    def test_current_data_training_view_masks_holdout(self):
        ctx = training_context()
        self.assertEqual(len(ctx["development"]), 1338)
        self.assertEqual(len(ctx["test"]), 335)
        self.assertTrue((ctx["y"][ctx["test"]] == -1).all())
        self.assertFalse(set(ctx["groups"][ctx["development"]]) & set(ctx["groups"][ctx["test"]]))
        X = raw_matrix("maccs166", ctx)
        self.assertEqual(X.shape, (1673,166))
        self.assertTrue(np.isin(X, [0,1]).all())

    def test_saved_tuning_score_cannot_silently_change_selection(self):
        rng = np.random.default_rng(8)
        X = rng.integers(0, 2, (72, 8))
        y = np.tile([0,1], 36)
        ctx = {"y":y, "groups":np.repeat([f"g{i}" for i in range(36)],2),
               "ids":np.array([f"i{i}" for i in range(72)])}
        job = {"model_id":"test__rf", "representation":"maccs166", "estimator_key":"rf"}
        candidate = {"n_estimators":3, "max_depth":3, "min_samples_leaf":1, "max_features":"sqrt"}
        with tempfile.TemporaryDirectory() as temp, patch("nephrotox.modeling_pipeline._candidate_list", return_value=[candidate]):
            out = Path(temp)
            _search(job, X, ctx, np.arange(72), out, "test-protocol", 1)
            trial = out / "candidate_01/inner_1/trial.json"
            data = json.loads(trial.read_text(encoding="utf-8"))
            data["roc_auc"] = 1-data["roc_auc"]
            trial.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"AUC/ID"):
                _search(job, X, ctx, np.arange(72), out, "test-protocol", 1)
            ctx["y"][0] = -1
            with self.assertRaisesRegex(ValueError,"Holdout labels"):
                _search(job, X, ctx, np.arange(72), out, "test-protocol", 1)


if __name__ == "__main__":
    unittest.main()
