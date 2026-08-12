# Data provenance

`NephroTox_Dataset.xlsx` is the author-supplied, quality-controlled modeling
workbook. Only the `Modeling_Final` sheet is used by the pipeline.

- Rows: 1,853 compounds
- Labels: 931 nephrotoxic (1), 922 non-nephrotoxic (0)
- SHA-256: `435b0551b1475d2a3dd60de3d451140526521a5ea1b918d578b92172972ef220`
- Required QC state: every row must have `QC_status = PASS`

The data-preparation script checks the hash, schema, row count, missingness,
binary labels, duplicate identifiers, and SMILES validity before it creates any
split. If the workbook changes, the pipeline stops until the new file is
reviewed and its hash is intentionally updated in `configs/experiment.json`.

The revised protocol writes three logically separated files:

- `data/locked/random10/development.csv`: 1,482 development compounds with labels and 10-fold assignments.
- `data/locked/random10/test_features.csv`: 371 test compounds without labels.
- `data/locked/random10/sealed/test_labels.csv`: the independently hashed label file opened only after candidate locking.

The only included split assignment is the formal `data/splits/random10_locked.csv`, accompanied by the publication-facing `dataset_split.csv` and `cv_folds.csv`. Legacy and alternative split files are intentionally excluded from this package.

Before making a public GitHub repository, the author should confirm that the
underlying source databases permit redistribution of the compiled records.
