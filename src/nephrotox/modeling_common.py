"""DATA-only inputs and immutable, group-isolated modeling protocol.

The training view masks every holdout label. All numerical preprocessing is
inside each estimator fit; raw matrices are deliberately returned here.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from datetime import datetime, timezone
import platform

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from .data import ROOT, load_data
from .features import load_descriptor_names
from .preprocessing import get_fold_indices
from .search_space import load_search_design

BINARY_REPRESENTATIONS = ["morgan2048", "maccs166", "atompair1024", "torsion1024", "rdkit166"]
COUNT_REPRESENTATIONS = ["morgan_count2048", "atompair_count1024", "torsion_count1024"]
ALGORITHMS = ["rf", "svm", "mlp", "xgb", "dnn"]
PARAMETER_DESIGN_PATH = ROOT / "configs/original_parameter_design.json"
PARAMETER_DESIGN = json.loads(PARAMETER_DESIGN_PATH.read_text(encoding="utf-8"))
SEARCH_DESIGN_PATH = ROOT / "configs/model_search_spaces.json"
SEARCH_DESIGN = load_search_design()
SEEDS = list(PARAMETER_DESIGN["seeds"])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")
    temp.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def training_context(results_dir=None):
    folder = Path(results_dir) if results_dir else ROOT / "results"
    rows, info = load_data()
    run = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete" or not run.get("independent_validation_passed"):
        raise ValueError("DATA/features/splits must pass independent validation before training")
    if run["source"]["rows_fingerprint"] != info["rows_fingerprint"]:
        raise ValueError("Run/source fingerprint mismatch")
    assignments = json.loads((folder / "split_payload.json").read_text(encoding="utf-8"))["assignments"]
    dev, test = get_fold_indices(rows, assignments)
    ids = np.array([r["compound_id"] for r in rows], dtype=str)
    # Do not provide names, CID, labels, row order or group keys as predictors.
    y = np.full(len(rows), -1, dtype=int)
    y[dev] = [rows[i]["label"] for i in dev]
    return {"ids": ids, "smiles": np.array([r["standardized_smiles"] for r in rows]),
            "y": y, "groups": np.array([a["structure_group_id"] for a in assignments]),
            "fold": np.array([a["cv_validation_fold"] or 0 for a in assignments]),
            "development": dev, "test": test, "source": info, "results_dir": folder,
            "assignment_sha256": sha256(folder / "split_payload.json")}


def raw_matrix(representation, context):
    folder = context["results_dir"]
    if representation in COUNT_REPRESENTATIONS:
        path = folder / f"features/{representation}.npz"
        manifest_path = folder / "features/count_feature_manifest.json"
        if not path.exists() or not manifest_path.exists():
            raise ValueError("Count features absent; run --prepare first")
    else:
        path = folder / f"features/{representation}.npz"
        manifest_path = folder / "features/feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["representations"][representation]
    if "count" in representation and manifest["source_rows_fingerprint"] != context["source"]["rows_fingerprint"]:
        raise ValueError("Count features derive from different DATA")
    if "count" not in representation and manifest["source"]["rows_fingerprint"] != context["source"]["rows_fingerprint"]:
        raise ValueError("Features derive from different DATA")
    if sha256(path) != expected["file_sha256"]:
        raise ValueError(f"Feature integrity failure: {representation}")
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data["compound_id"], context["ids"]) or not np.array_equal(data["standardized_smiles"], context["smiles"]):
            raise ValueError("Feature identities/structures differ from current DATA")
        if not np.array_equal(data["row_index"], np.arange(len(context["ids"]))):
            raise ValueError("Feature row order differs from current DATA")
        if representation == "rdkit166" and data["descriptor_names"].tolist() != load_descriptor_names(ROOT / "configs/rdkit_descriptors_166.txt"):
            raise ValueError("Descriptor order mismatch")
        matrix = np.array(data["X"], copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite raw matrix")
    if list(matrix.shape) != expected["shape"] or matrix.shape[0] != len(context["ids"]):
        raise ValueError("Feature dimensions differ from manifest")
    return matrix


def prepare_count_features(context):
    folder = context["results_dir"] / "features"
    manifest_path = folder / "count_feature_manifest.json"
    if manifest_path.exists():
        for rep in COUNT_REPRESENTATIONS:
            raw_matrix(rep, context)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    generators = {
        "morgan_count2048": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True),
        "atompair_count1024": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024, includeChirality=True),
        "torsion_count1024": rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024, includeChirality=True),
    }
    molecules = [Chem.MolFromSmiles(s) for s in context["smiles"]]
    manifest = {"source_rows_fingerprint": context["source"]["rows_fingerprint"],
                "rdkit_version": rdBase.rdkitVersion, "representations": {},
                "definition": "RDKit GetCountFingerprint; hashed counts, same dimensions and chirality settings as corresponding binary generators; not count-simulated binary bits"}
    for rep, generator in generators.items():
        width = 2048 if "2048" in rep else 1024
        matrix = np.zeros((len(molecules), width), dtype=np.int32)
        for i, mol in enumerate(molecules):
            DataStructs.ConvertToNumpyArray(generator.GetCountFingerprint(mol), matrix[i])
        path = folder / f"{rep}.npz"
        np.savez_compressed(path, X=matrix, compound_id=context["ids"], standardized_smiles=context["smiles"],
                            row_index=np.arange(len(molecules)))
        manifest["representations"][rep] = {"file_sha256": sha256(path), "shape": list(matrix.shape),
            "dtype": str(matrix.dtype), "numeric_sha256": hashlib.sha256(matrix.astype("<i4").tobytes()).hexdigest(),
            "includeChirality": True, "generator_method": "GetCountFingerprint"}
    json_write(manifest_path, manifest)
    return manifest


def registry():
    jobs = []
    for rep in BINARY_REPRESENTATIONS + COUNT_REPRESENTATIONS:
        for algorithm in ALGORITHMS:
            jobs.append({"model_id": f"{rep}__{algorithm}", "representation": rep,
                         "algorithm": algorithm.upper(), "estimator_key": algorithm,
                         "selection_method": "finite_prespecified_candidates", "configuration_count": 3,
                         "role": "sensitivity_count" if "count" in rep else "main"})
    for graph in ["D-MPNN", "GINE", "AttentiveFP"]:
        jobs.append({"model_id": f"graph__{graph.replace('-', '')}", "representation": "graph",
                     "algorithm": graph.replace("-", ""), "estimator_key": graph, "role": "main",
                     "selection_method": "seeded_random_search",
                     "configuration_count": SEARCH_DESIGN["graph_candidates_per_architecture"]})
    return jobs


def build_protocol(context, graph_candidates):
    from .modeling_classical import CANDIDATES
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "scipy", "scikit-learn", "rdkit", "torch", "xgboost", "joblib"]}
    fitting_files = ["modeling_common.py", "modeling_pipeline.py", "modeling_classical.py", "modeling_neural.py", "modeling_graph.py", "search_space.py"]
    return {"schema_version": 1, "source_excel_sha256": context["source"]["sha256"],
        "source_rows_fingerprint": context["source"]["rows_fingerprint"],
        "split_payload_sha256": context["assignment_sha256"], "registry": registry(),
        "outer_validation": "fixed ten development structure-group folds; test labels masked in trainer",
        "inner_validation": {"splitter": "StratifiedGroupKFold", "folds": 3, "shuffle": True, "seed": SEEDS[0]},
        "search": "conventional: compare three original S2A candidates; graph: compare original reference plus 19 seeded random configurations from declared additional ranges; every candidate is fitted and scored in three grouped inner folds; highest mean ROC-AUC wins, ties by candidate order",
        "candidate_parameters": CANDIDATES, "graph_candidate_parameters": graph_candidates,
        "parameter_design_file": "configs/original_parameter_design.json",
        "parameter_design_sha256": sha256(PARAMETER_DESIGN_PATH),
        "parameter_source_references": PARAMETER_DESIGN["source_references"],
        "search_design_file": "configs/model_search_spaces.json",
        "search_design_sha256": sha256(SEARCH_DESIGN_PATH),
        "search_space_definition": SEARCH_DESIGN,
        "graph_selection_mode": SEARCH_DESIGN["graph_method"],
        "graph_reference_parameters": PARAMETER_DESIGN["graph_reference_configurations"],
        "graph_reference_configuration_ids": PARAMETER_DESIGN["graph_reference_configuration_ids"],
        "original_graph_table_selection_mode": PARAMETER_DESIGN["original_graph_table_selection_mode"],
        "implementation_training_defaults": PARAMETER_DESIGN["implementation_training_defaults"],
        "algorithm_fixed_parameters": {
            "RF": {"criterion":"gini", "bootstrap":True, "class_weight":"balanced", "min_samples_split":2,
                   "min_weight_fraction_leaf":0.0,"max_leaf_nodes":None,"min_impurity_decrease":0.0,"max_samples":None,
                   "ccp_alpha":0.0,"oob_score":False,"warm_start":False,"random_state":SEEDS[0],"n_jobs":"--threads"},
            "SVM": {"kernel":"rbf","class_weight":"balanced","probability":False,"shrinking":True,"tol":0.001,
                    "cache_size_MB":512,"max_iter":-1,"random_state":SEEDS[0],
                    "calibration":{"method":"sigmoid","cv":"explicit 3-fold StratifiedGroupKFold","ensemble":False,"n_jobs":1}},
            "MLP": {"activation":"relu","solver":"adam","batch_size":128,"max_iter":200,"early_stopping":False,
                    "tol":0.0,"n_iter_no_change":201,"shuffle":True,"learning_rate":"constant",
                    "beta_1":0.9,"beta_2":0.999,"epsilon":1e-8,"random_state":SEEDS[0]},
            "XGB": {"booster":"gbtree","objective":"binary:logistic","eval_metric":"logloss","tree_method":"hist",
                    "device":"cpu","scale_pos_weight":"negative/positive count in this fit subset","random_state":SEEDS[0],
                    "early_stopping":False,"missing":"NaN sentinel; current DATA raw features must be finite","n_jobs":"--threads",
                    "reg_lambda":1.0,"min_child_weight":1},
            "DNN": {"max_epochs":80,"patience":10,"min_delta":0.0001,"optimizer":"AdamW","betas":[0.9,0.999],"eps":1e-8,
                    "learning_rate_schedule":"ReduceLROnPlateau(mode=max, patience=4, factor=0.5) on training-internal grouped validation ROC-AUC; replay actual epoch LR prefix during all-fit-subset refit",
                    "activation":"ReLU","output":"single logit",
                    "loss":"BCEWithLogitsLoss; pos_weight = fit-subset negatives/positives",
                    "epoch_monitor":"ROC-AUC on first training-internal 5-fold StratifiedGroupKFold validation fold",
                    "refit":"fresh same-seed initialization; complete fit subset; fixed selected epoch"},
            "graph_models": {"optimizer":"AdamW","betas":[0.9,0.999],"eps":1e-8,"learning_rate_schedule":"constant",
                    "loss":"BCEWithLogitsLoss; training-subset positive weight", "epoch_monitor":"unweighted BCE on grouped internal holdout",
                    "refit":"fresh same-seed initialization; complete fit subset; fixed selected epoch",
                    "architecture_source":"modeling_graph.py; full per-fit model_definition saved", "readout_steps":"used by AttentiveFP only"}},
        "preprocessing": {"binary_fingerprints":"unchanged",
            "rdkit166":"fit-subset median imputer; StandardScaler additionally for SVM/MLP/DNN; no scaler for RF/XGB",
            "count_fingerprints":"raw counts for RF/XGB; log1p then fit-subset StandardScaler for SVM/MLP/DNN",
            "scope":"every inner fit, calibration subfit, epoch selector and refit has independently fitted preprocessing"},
        "search_seeds": [SEEDS[0]], "neural_evaluation_seeds": SEEDS,
        "classical_evaluation_seed": SEEDS[0], "probability_aggregation": "neural only: arithmetic mean across three seeds; classical: single fitted model",
        "final_training": "repeat the declared candidate search for every conventional and graph model on all development rows; refit the highest-mean-inner-AUC configuration on all development rows; no outer-fold ensembling; no inherited selected parameters or epochs",
        "neural_early_stopping": "epoch chosen using a further training-internal grouped split, then refit all fit-subset rows; see per-fit logs for monitor and limits",
        "svm_probability": "3-fold grouped out-of-fold sigmoid calibration, then base estimator refit on all fit rows",
        "primary_model": "highest mean of ten outer-fold ROC-AUC among the 25 conventional main models; ties by model_id; holdout not used",
        "classification_threshold": 0.5, "ci": {"method": "class-stratified bootstrap", "resamples": 2000, "seed": 20260810},
        "ci_sensitivity": "2000 structure-group bootstrap resamples for development-selected primary and three graph models; conditional on saved fitted predictions",
        "paired_comparison": "primary vs other main models plus all three graph pairs; two-sided paired bootstrap AUC difference; Holm family correction",
        "applicability_domain": "Morgan2048 chiral binary Tanimoto; dev leave-self-out lower fifth percentile for max and mean5 neighbor similarity; OOF per-fold training scope",
        "label_cliffs": "opposite binary labels; Morgan2048 chiral Tanimoto >=0.85; exploratory, no potency claims",
        "structural_alerts": "predefined 166 MACCS keys; development-label Fisher tests; BH across all 166; association only",
        "attribution": {"method": "TreeSHAP for predeclared MACCS RF and descriptor RF; graph bond removal with atom features fixed", "clipping_percentile": 95},
        "versions": versions, "python": platform.python_version(), "compute": "CPU; bounded worker threads; per-fit elapsed wall times logged",
        "fitting_source_sha256": {name: sha256(Path(__file__).parent / name) for name in fitting_files},
        "test_reporting": "only after every registered model is complete and the global lock verifies file hashes",
        "endpoint_limit": "existing binary endpoint carried unchanged from DATA; chemical identity validation does not establish label provenance or absence of renal toxicity"}


def freeze_protocol(context, graph_candidates):
    out = context["results_dir"] / "modeling"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    protocol = build_protocol(context, graph_candidates)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != protocol:
            raise ValueError("Modeling protocol differs from frozen protocol; create a separate run, never mix results")
    else:
        json_write(path, protocol)
        json_write(out / "protocol_freeze.json", {"frozen_at_utc": utc_now(), "protocol_sha256": sha256(path),
                                                   "no_test_metrics_computed": True})
    return protocol, sha256(path)
