# Manuscript modeling protocol

The manuscript compares 25 conventional models, 15 count-fingerprint sensitivity models and three molecular graph models on the same 1,673 compounds (842 positive and 831 reference-negative). The [archived training protocol](../reference/modeling/protocol.json) has SHA-256 `dcab35fef8def12907c144146cec6a021c827ebc0970b16666a67d6fa0f29393`. The [model lock](../reference/modeling/model_lock.json) identifies each final estimator, its metadata and its predictions.

## Data and partitioning

The [input CSV](../source/data/NephroTox_Modeling_Data.csv) supplies compound IDs, standardized isomeric SMILES, binary labels and the other original compound fields. The reader validates the column order and restores integer and floating-point fields before checking the record fingerprint. The [provenance record](../source/data/input_provenance.json) links the CSV to the historical workbook through the same 1,673 ordered records. The archived protocol identifies the original training input. New protocols identify the CSV used for each refit.

The label-stratified random holdout assigned 1,338 compounds to development and 335 to testing. The split and the ten shuffled, stratified development folds used seed 20260810. Seventy-one structure groups occurred in both partitions. This is an internal random-holdout evaluation. See the [compound assignments](../reference/split_assignments.csv) and [outer validation folds](../reference/cross_validation_folds.csv).

## Features and preprocessing

Morgan, atom-pair and topological-torsion fingerprints include chirality. MACCS uses keys 1 through 166. The graph representation contains 221 ordered atom features and 33 ordered bond features. The [feature code](../source/src/nephrotox/features.py), [descriptor order](../source/configs/rdkit_descriptors_166.txt) and [graph feature schema](../source/configs/graph_feature_schema.json) define these inputs.

Binary fingerprints went directly into the models. The descriptor pipeline uses fitting-subset medians for missing values. All descriptor values for these 1,673 compounds were finite, as recorded in the [feature manifest](../reference/features/feature_manifest.json). SVM, MLP and DNN models also used fitting-subset standardization for the descriptor inputs. RF and XGBoost used raw count fingerprints. SVM, MLP and DNN used a log(1 + count) transform followed by fitting-subset standardization. The outer validation and test rows were held out when estimating preprocessing parameters.

## Model fitting and selection

Each combination used one [fixed configuration](../source/configs/original_fixed_parameters.json) for its ten outer fits and its full-development fit. These values had been chosen in an earlier dataset with overlapping compounds. The 180 [candidate designs](../source/configs/model_candidate_parameters.csv) in supplementary Table S1A describe available settings. Their generation methods, proposed inner folds and scoring rules are design metadata. Tables S1B and S1C identify the 43 configurations and 473 fits actually used. No hyperparameter search was executed for the manuscript run.

Within each fitting subset, SVM used the native LIBSVM five-fold sigmoid calibration. MLP used a random 15% internal holdout for early stopping. For DNN and graph models a group-aware holdout within the fitting subset selected the epoch by ROC-AUC. The selected epoch count then determined the length of a fresh full-subset fit for the same seed. Epoch limits were 80 for the DNN and 200 for graph models. Graph training used batch size 32, patience 20, minimum ROC-AUC improvement 0.0001 and gradient clipping at 5.

The three neural seeds were 20260810, 20260811 and 20260812. The ensemble probability was their arithmetic mean. For the outer RF, MLP and XGBoost fits, the seed was 20260810 + 100 times the archived candidate ID + the zero-based fold index. For the outer SVM fits it was 20260810 + 1000 + the zero-based fold index. Final conventional fits used 20260810. The per-fit metadata record the settings actually used.

The [primary-model ranking](../reference/modeling/reports/primary_selection.json) used the unweighted mean of ten validation-fold ROC-AUC values across the 25 main conventional models. This selected MACCS–RF. The held-out predictions used the final full-development fits. For classification the probability threshold was 0.5.

## Reproduction and results

The [prediction command](../README.md#predict-with-the-manuscript-weights) uses the original manuscript weights. The [training command](../README.md#train-the-manuscript-models) runs all 43 models from the supplied CSV. The [workbook guide](../WORKBOOK_MAP.md) maps the inputs, parameters, predictions and metrics to their supplementary worksheets. The [fitting-source record](../source/data/fitting_source_provenance.json) identifies the original implementation by SHA-256 and links it to the current CSV-based code.
