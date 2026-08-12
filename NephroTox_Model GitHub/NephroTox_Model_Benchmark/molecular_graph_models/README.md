# Molecular graph model artifacts

This directory contains the final supplied artifacts for three molecular graph architectures:

- D-MPNN
- GINE
- AttentiveFP

It also contains the locked three-model probability consensus used for deployment-oriented evaluation.

## Scope boundary

The source archive contains result tables, selected configurations, model locks, ensemble predictions, post-analysis outputs, and publication figures. It contains no Python source, notebook, environment specification sufficient to recreate training, or executable inference entry point. Accordingly:

- development and independent-test results are auditable from the retained tables;
- trained graph weights and repeated/fold-level intermediates were intentionally excluded from the GitHub-oriented release;
- training and inference cannot be reproduced from this directory alone;
- graph source code should be added in a later release only if it can be matched to the retained configuration and lock identifiers.

## Retained structure

```text
results/
├── selection/                    # Locked configuration and selection manifests
├── confirmation/                 # Compact development-confirmation summaries
├── reporting/manuscript_aligned/ # Development metrics and bootstrap CIs
├── final/                        # Ensemble test probabilities and model locks
├── test_evaluation/              # Independent-test metrics, CIs, and labeled predictions
├── consensus/                    # Development/test consensus predictions and lock
├── analysis/                     # Applicability, explainability, label cliffs, structural alerts
└── figures/                      # GitHub-friendly figures and source data; TIFF removed
```

The normalized cross-family comparison is in `../tables/model_performance_independent_test.csv`. Original metric names are preserved in the source tables under `results/`.

