# Formal package cleanup audit

## Scope

This delivery archive is restricted to the completed traditional-model run `random10_locked_v1`:

- five molecular representations: Morgan, MACCS, AtomPairs, TopologicalTorsion, and 166 RDKit descriptors;
- five algorithms: RF, SVM, MLP, XGBoost, and feed-forward DNN;
- stratified-random 80:20 split with seed `20260810`;
- 1,482 development compounds and 371 independent-test compounds;
- development-only ten-fold model selection;
- 25 locked experiments, one-time independent-test evaluation, 2,000 bootstrap replicates, figures, supplementary tables, and final model artifacts.

## Removed from the delivery package

- legacy five-fold configuration and its obsolete data manifest;
- alternative scaffold10 configuration and locked dataset;
- legacy `random_holdout`, `scaffold_holdout`, and `scaffold10_locked` split files;
- duplicate compatibility entry scripts superseded by the numbered random10 workflow;
- unused legacy reporting/validation modules containing five-fold terminology;
- internal requirement/design notes that mixed the superseded scaffold protocol with the formal random10 protocol.

The exclusions affect no file referenced by the `random10_locked_v1` lock, no formal feature matrix, no tuning/OOF result, no final model, no test prediction, and no publication output.

## Retained canonical workflow

```text
00_prepare_data.py
01_generate_features.py
02_tune_models.py
03_lock_candidates.py
04_evaluate_locked_test.py
05_summarize_results.py
06_validate_outputs.py
07_export_supplementary_tables.py
run_all.py
```

The package manifest is regenerated after cleanup. Before release, the complete 25-experiment output validator passed, the model lock remained unchanged, and the test suite passed 9/9 checks. ZIP integrity, the regenerated manifest, protocol hashes, and the absence of excluded paths are verified again from the rebuilt archive itself.
