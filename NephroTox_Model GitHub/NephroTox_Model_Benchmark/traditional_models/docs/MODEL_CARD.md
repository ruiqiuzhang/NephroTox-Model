# Model card

## Model family

The benchmark contains 25 final classifiers: five molecular representations crossed with RF, SVM, MLP, XGBoost, and DNN. All combinations use the same locked split and folds.

## Selection and evaluation

Hyperparameters and DNN architecture are selected only by development 10-fold mean ROC-AUC. Final models are refitted on all 1,482 development compounds and evaluated once on 371 stratified-random independent-test compounds. Test performance does not alter model selection or ranking.

## Outputs

Each experiment saves the selected parameters, OOF probabilities, final model/preprocessing artifact, test probabilities, threshold predictions, confusion counts, required metrics, calibration table, runtime, file size, and SHA-256 hash.

## Appropriate use

Models may support research prioritization and early screening within an applicability domain similar to the training chemistry. Predictions should be treated as computational evidence, not as proof of renal safety or toxicity.

## Known risks

- A random split can place close analogs in development and test; Tanimoto and applicability-domain results must accompany aggregate performance.
- Labels may contain evidence heterogeneity and incomplete negatives.
- Probability calibration can drift on new sources or chemical domains.
- A model with high aggregate AUC can still miss important toxic compounds.
- External and prospective validation remain necessary.

## Reproducibility

Use the stored configuration, source hash, split manifest, feature hashes, and model hashes. Do not compare a modified model against the locked test and then reuse that test result as if it were an untouched final evaluation.
