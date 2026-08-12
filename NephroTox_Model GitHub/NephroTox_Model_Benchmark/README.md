# NephroTox model benchmark

GitHub-oriented release of the NephroTox benchmark, organized into two clearly separated model families:

- **Traditional models:** 25 combinations of five molecular representations and five algorithms, with reproducible Python source code, locked data splits, tests, selected hyperparameters, independent-test metrics, and publication tables.
- **Molecular graph models:** D-MPNN, GINE, and AttentiveFP, plus the three-model deployment consensus, with locked configurations, development-confirmation metrics, independent-test metrics, predictions, post-analysis tables, and figure source data.

The supplied molecular-graph archive contains **results and locked configurations only**. It does not contain the training or inference source code, so this release does not claim end-to-end reproducibility for those models. See [molecular_graph_models/README.md](molecular_graph_models/README.md) for the exact boundary.

## Repository layout

```text
.
├── traditional_models/          # Source code and curated results for 25 traditional/DNN combinations
├── molecular_graph_models/      # Curated artifacts for D-MPNN, GINE, AttentiveFP, and consensus
├── tables/                      # Unified CSV and Excel tables for review and reuse
├── scripts/validate_release.py  # Lightweight repository-integrity check
└── RELEASE_MANIFEST_SHA256.csv  # File sizes and SHA-256 checksums
```

## Model inventory

| Family | Representations / architectures | Algorithms | Independent-test coverage | Source code |
|---|---|---|---|---|
| Traditional and feed-forward DNN | Morgan, MACCS, AtomPairs, TopologicalTorsion, RDKit descriptors | RF, SVM, MLP, XGBoost, DNN | 25 combinations | Included |
| Molecular graph | D-MPNN, GINE, AttentiveFP | Three graph architectures plus probability consensus | 3 models + consensus | Not present in supplied archive |

The primary machine-readable comparison is [tables/model_performance_independent_test.csv](tables/model_performance_independent_test.csv). The formatted workbook [tables/NephroTox_Model_Tables.xlsx](tables/NephroTox_Model_Tables.xlsx) contains the same normalized tables plus scope and cleanup notes.

## Traditional-model setup

Python 3.11 or newer is required.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -e traditional_models
python -m pytest traditional_models/tests
```

Run the numbered workflow from `traditional_models/`. Detailed protocol and leakage controls are documented in [traditional_models/docs/MODELING_PROTOCOL.md](traditional_models/docs/MODELING_PROTOCOL.md).

## Result interpretation

- Model selection and independent-test evaluation are separate stages. Do not rank models using independent-test results as if those results were development criteria.
- The traditional protocol identifies MACCS-RF as the development-CV-selected primary model. Descriptor-RF is retained as a test-best descriptive comparison, not as a retrospectively selected primary model.
- Molecular-graph development-confirmation values and independent-test values are stored separately. The consensus is a deployment ensemble and is not an additional trained architecture.
- Metric values in the unified tables are copied from the supplied locked outputs; no values were re-estimated during repository cleanup.

## What was removed

The release excludes redundant nested ZIP files, superseded graph-result snapshots, model binaries (`.pt`, `.joblib`), fold/member-level predictions, tuning OOF matrices, training logs, temporary inspection files, and large TIFF exports. Compact PNG/SVG/PDF figures, source-data CSV files, summary metrics, configurations, and audit tables are retained.

The original source archives remain unchanged outside this release directory.

## Release validation

Run the dependency-free integrity check from the repository root:

```bash
python scripts/validate_release.py
```

It checks required files, JSON/CSV readability, the normalized 29-row independent-test table, banned large-artifact types, and GitHub's 100 MB per-file limit.

## Citation and licensing

Citation metadata are provided in [CITATION.cff](CITATION.cff). No software or dataset license was selected in the supplied material. Before publishing this repository, the author must choose an explicit license and confirm that every dataset component may be redistributed; see [LICENSE_NOTICE.md](LICENSE_NOTICE.md).

