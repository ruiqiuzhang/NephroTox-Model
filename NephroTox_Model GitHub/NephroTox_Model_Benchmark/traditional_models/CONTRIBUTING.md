# Contributing

Changes that affect data, features, splits, candidates, model code, thresholds, or final-refit rules create a new experimental protocol. Do not overwrite the locked `random10_locked_v1` results.

1. Create a new run ID and output directory.
2. Document the scientific reason for the change.
3. Run development-only tests and tuning without reading the existing locked test outcomes for selection.
4. Add or update automated tests.
5. Run `python -m pytest` before submitting a change.
6. Do not commit secrets, personal access tokens, or unlicensed source data.

Large `.joblib`, `.pt`, `.xlsx`, and TIFF files require Git LFS. Include updated artifact hashes whenever a binary changes.
