# NephroTox manuscript training and prediction

This release contains the training code, original fitted models and per-compound predictions for the 1,673-compound nephrotoxicity benchmark. The primary model is the MACCS random forest (MACCS–RF). All 43 manuscript models are ready for prediction.

| Task | Files |
| --- | --- |
| Predict with the published models | [Prediction script](source/scripts/predict.py), [example inputs](validation/inference_queries.csv), [43 model IDs and weight hashes](reference/modeling/model_lock.json) |
| Train the models | [Training script](source/scripts/run_complete_modeling.py), [modeling CSV](source/data/NephroTox_Modeling_Data.csv), [fixed parameters](source/configs/original_fixed_parameters.json) |
| Inspect the reported results | [Development out-of-fold predictions](reference/modeling/reports/oof_predictions.csv), [held-out predictions](reference/modeling/reports/test_predictions.csv), [model performance](reference/modeling/reports/model_summary.csv) |
| Check the methods and supplementary tables | [Modeling protocol](docs/MODELING_PROTOCOL.md), [partition and fold assignments](reference/split_assignments.csv), [workbook guide](WORKBOOK_MAP.md) |

Run the commands below from the repository root. The archived environment used CPython 3.14.5 on Windows with CPU PyTorch, RDKit and scikit-learn. Install the [pinned dependencies](source/requirements-modeling.txt) in a virtual environment.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --only-binary=:all: -r source/requirements-modeling.txt
```

On a POSIX shell, use `python3.14 -m venv .venv` and `source .venv/bin/activate` before running the same pip command. The [environment record](reference/modeling/benchmarks/runtime_environment.json) identifies the original platform and package versions.

## Predict with the manuscript weights

The default model is the original MACCS–RF identified in the [development-set selection record](reference/modeling/reports/primary_selection.json). Supply a CSV with a unique `compound_id` and a pre-standardized, single-component `standardized_smiles` on each row.

```sh
python source/scripts/predict.py --input validation/inference_queries.csv --output predictions.csv
```

To use another original model, pass its directory under `reference/modeling/`. For example, the following command runs the three-seed AttentiveFP ensemble.

```sh
python source/scripts/predict.py --model-dir reference/modeling/graph__AttentiveFP --input validation/inference_queries.csv --output source/results/attentivefp_predictions.csv
```

The output CSV contains each compound ID, its submitted SMILES, the model ID, the probability of label 1, the predicted label and the decision threshold. A probability of 0.5 or higher is classified as label 1.

## Train the manuscript models

The training command reads the supplied CSV, prepares features and trains 25 conventional models for the main comparison, 15 count-fingerprint models and three graph models. The original stratified random split assigned 1,338 compounds to development and 335 to the held-out test set. Each model used one fixed configuration for ten development folds and the final development-set fit.

```sh
python source/scripts/run_complete_modeling.py --results source/results/manuscript --split-mode legacy_random --parameter-mode fixed_original --workers 2 --threads 2
```

Start with a fresh results directory. To resume an interrupted run, repeat the same command with the same directory. Completed model files are written to `source/results/manuscript/modeling/<model_id>/` and numeric reports to `source/results/manuscript/modeling/reports/`. The prediction script also accepts a fitted directory from this workflow via `--model-dir`.

The original fits read a workbook. The [public CSV](source/data/NephroTox_Modeling_Data.csv) preserves its 1,673 ordered records and 12 typed fields. The [input provenance](source/data/input_provenance.json) documents this correspondence. Five of the seven fitting modules are byte-identical to the originals. The two modules adapted to record the CSV input also have [original source snapshots](reference/fitting_source/). The [fitting-source record](source/data/fitting_source_provenance.json) lists their hashes.

## Verify the supplied files and results

The following commands use Python's standard library. They check the package and model locks and recalculate the reported ROC-AUC, average precision, Brier score and confusion counts from the archived predictions.

```sh
python verify_release.py
python verify_model_metrics.py
```

To check the separate supplementary workbook, pass `--workbook "../Supplementary Material/NephroTox_Supplementary_Data.xlsx"` to each command, using its actual location. The archived [inference check](validation/checkpoint_inference.json) covers all 43 final models and all 335 held-out compounds. The following command regenerates the [paired fingerprint comparisons](validation/paired_binary_count.csv) from the saved predictions.

```sh
python source/scripts/export_binary_count_tables.py --results reference/modeling --output source/results/fingerprint_comparisons
```

## Manuscript and deployment versions

The [frozen protocol](reference/modeling/protocol.json) identifies the stratified random partitions and fixed configurations used for this manuscript. An earlier public protocol described structure-group isolation and nested candidate selection for a separate experiment.

On 24 November 2026, the [website introduction](https://nephro.toxic-ai.cn/introduction) described the 1,673-compound study. The [online service](https://nephro.toxic-ai.cn/api/v1/health) reported version 1.0.0. Its [deployment metadata](https://nephro.toxic-ai.cn/api/v1/models) described a 1,853-compound cohort with 1,482 development and 371 test compounds. The [model lock](reference/modeling/model_lock.json) identifies the 43 original manuscript weights included here.
