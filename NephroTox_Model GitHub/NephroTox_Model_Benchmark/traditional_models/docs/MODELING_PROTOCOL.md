# Modeling protocol

## Objective

Benchmark five molecular representations with RF, SVM, MLP, XGBoost, and DNN for binary nephrotoxicity prediction while preventing chemical-series leakage and test-set-guided model development.

## Audited input

The only modeling source is `data/NephroTox_Dataset.xlsx`, sheet `Modeling_Final`. It contains 1,853 unique compounds: 931 positive and 922 reference-negative. The preparation stage verifies the file SHA-256, schema, row count, missingness, binary labels, QC status, SMILES validity, and uniqueness of compound ID, standardized SMILES, standardized InChIKey, and connectivity key.

## Molecular representations

1. Morgan fingerprint: radius 2, 2,048 bits, chirality enabled.
2. MACCS keys: 166 defined RDKit bits after removing the unused zeroth bit.
3. Hashed atom-pair fingerprint: 1,024 bits, chirality enabled.
4. Hashed topological-torsion fingerprint: 1,024 bits, chirality enabled.
5. RDKit descriptors: 166 version-controlled names in `configs/rdkit_descriptors_166.txt`.

The fourth representation is a true topological-torsion fingerprint. Descriptor names are frozen so an RDKit upgrade cannot silently change the feature definition.

## Independent test split

The primary revised holdout is a deterministic stratified-random split with seed `20260810`. The 8:2 ratio is used once: 80% is the development set and 20% is the locked independent test. No additional validation set is removed from the data; validation occurs only through development-set cross-validation.

- Development: 1,482 compounds (745 positive, 737 negative).
- Independent test: 371 compounds (186 positive, 185 negative).

The random protocol does not constrain scaffold overlap. Bemis-Murcko group overlap and training-test Morgan Tanimoto similarities are therefore reported transparently as chemical-space and applicability-domain analyses.

## Development-only 10-fold selection

The development set is divided with `StratifiedKFold(n_splits=10, shuffle=True, random_state=20260810)`. Validation folds contain 148-149 compounds and retain the class ratio. For each iteration, nine folds train the candidate and the remaining fold supplies development validation.

Each of the 25 representation-model combinations evaluates three predeclared candidates from `configs/experiment.json`. Selection uses mean development 10-fold ROC-AUC. Ties are resolved by lower fold-to-fold ROC-AUC SD and then lower candidate ID. The following are saved for every candidate and fold:

- Parameters, sample counts, ROC-AUC, and runtime.
- Per-compound candidate OOF score.
- DNN best epoch and complete epoch-level training history.
- Selected-candidate per-compound OOF probability and metrics.

No test feature or label is used for preprocessing, early stopping, hyperparameter selection, architecture selection, threshold selection, or model ranking.

## Preprocessing and algorithms

Fingerprint inputs remain binary. For descriptors, median imputation is fitted within each training fold. Descriptor scaling is fitted within each training fold for SVM, MLP, and DNN.

- RF uses balanced class weights.
- RBF-SVM is selected by outer-fold decision scores; probability fitting is performed only for the selected candidate and final refit.
- MLP uses ReLU, Adam, fold-local preprocessing, and development-only early stopping.
- XGBoost uses fold-local `scale_pos_weight` and histogram tree construction.
- DNN uses two fully connected hidden layers, BatchNorm, ReLU, dropout, AdamW, weighted binary cross-entropy, learning-rate reduction, and early stopping. Positive-class weight is computed from the current training fold.

The final DNN epoch count is the median of the selected architecture's ten best epochs. Three predeclared seeds (`20260810`, `20260811`, and `20260812`) are fitted on the full development set and their test probabilities are averaged. Seeds are never ranked by test performance.

## Candidate lock and one-time test evaluation

The lock stage requires complete development-only outputs for all 25 combinations. It records SHA-256 hashes for the configuration, source implementation, descriptor list, protocol files, feature matrices, selected parameters, candidate results, and OOF predictions. Only after this manifest is created and verified can the evaluation stage open the separately stored test labels.

Each selected model is refitted on the full development set and evaluated once on the independent test. An interrupted evaluation may resume only under the identical lock; completed combinations are not retrained or overwritten. A completed evaluation manifest prevents a second test run.

## Outcomes and uncertainty

The classification threshold is fixed at 0.5. Required test outcomes are ROC-AUC, accuracy, sensitivity, specificity, precision, F1, MCC, and balanced accuracy. PR-AUC, Brier score, and 10-bin expected calibration error are retained as secondary outcomes.

Confidence intervals use 2,000 class-stratified percentile-bootstrap replicates. All 25 models receive 95% CIs. The top two combinations are defined by development CV before opening the test set and compared by paired bootstrap for ROC-AUC and MCC.

## Figure contract

Figure 4 retains the original manuscript's five model rows, five-representation ROC curves, and five Blues confusion matrices per row. Figure 5 retains six independent-test metric line panels and the red-white-blue model-metric heatmap; panel A reports independent-test AUC with 2,000-replicate stratified-bootstrap 95% confidence intervals, while development 10-fold CV mean ± SD is reported in Supplementary Table S1. Morgan, MACCS, AtomPairs, TopologicalTorsion, and Descriptor keep the original red square, blue circle, yellow upward triangle, green downward triangle, and purple diamond encoding. Supplementary Figure S1 reports training-test maximum Morgan Tanimoto similarity and the normalized distance from the decision threshold, defined as `2 × |p − 0.5|`. MACCS-RF is labelled as the CV-selected primary model, whereas Descriptor-RF is labelled as the test-best descriptive model and is not used to revise model selection.

Public split tables are exported as `data/splits/dataset_split.csv` (all 1,853 compounds) and `data/splits/cv_folds.csv` (the 1,482 development compounds with one-based and zero-based fold identifiers).

Figures are generated only from saved prediction and summary CSVs. Primary exports are editable SVG and PDF; 600 dpi TIFF and PNG are supplied for submission and preview. Figure 5 includes MCC in the heatmap and ROC-AUC 95% CI whiskers while retaining the original visual system.

## Reproducibility controls

- Hash-locked source workbook and protocol files.
- Persistent row-level split and fold assignments.
- Deterministic stratified-random holdout and stratified 10-fold CV.
- Fold-local imputation, scaling, class weighting, and early stopping.
- Separate development, test-feature, and sealed test-label files.
- Development-only candidate outputs for all 25 combinations.
- Lock-manifest verification before test access.
- Saved final models, preprocessing objects, predictions, metrics, CIs, figures, and SHA-256 artifact manifests.
- Automated metric reconstruction and editable-figure validation.

## Applicability domain

For each independent-test molecule, the maximum Morgan (radius 2, 2,048-bit) Tanimoto similarity to any development molecule and its nearest neighbor are saved. The applicability-domain threshold is defined without test outcomes as the 5th percentile of leave-one-out nearest-neighbor similarities within the development set. Test compounds at or above this threshold are classed as inside the domain. Performance is reported for inside/outside-domain subsets and fixed maximum-similarity intervals for all 25 models.
