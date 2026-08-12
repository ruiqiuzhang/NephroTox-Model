# Reproducibility checklist

- [ ] Workbook hash matches `configs/experiment.json`.
- [ ] Python and package versions are recorded.
- [ ] `scripts/00_prepare_data.py` reports 1,482 development and 371 test compounds.
- [ ] The stratified split reports 1,482 development and 371 independent-test compounds with the expected class counts.
- [ ] Five feature matrices have the expected row order and dimensions.
- [ ] Every representation-model combination has three candidates and ten folds.
- [ ] Every selected OOF file contains 1,482 unique rows.
- [ ] The lock manifest contains 25 experiments and passes hash verification.
- [ ] The test evaluation manifest confirms one post-lock evaluation.
- [ ] Every test prediction file contains the same 371 compound IDs in the same order.
- [ ] All metrics can be recomputed from probabilities and threshold 0.5.
- [ ] All final model artifacts match their SHA-256 values.
- [ ] The CI table contains 25 combinations.
- [ ] Figure 4 and Figure 5 exist as SVG, PDF, TIFF, and PNG.
- [ ] Training-test Tanimoto, applicability-domain source data, and the AD figure are present.
- [ ] SVG text remains editable.
- [ ] `results/random10_locked/output_validation.json` reports `PASS`.
