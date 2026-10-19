# NephroTox current DATA model card

## Workflow scope

This code-only release specifies the modeling process and its scientific interpretation. `configs/modeling_protocol.json` defines the complete comparison. Model weights, numerical reports and validation logs are generated under `results/`; figures use their script-configured output directories. Generated artifacts are not required repository contents. The primary model and its performance are read from generated reports rather than fixed in this card.

The requested benchmark comprises 25 conventional main combinations, three main graph models and 15 count-fingerprint sensitivity combinations. The three graph architectures are explicitly implemented D-MPNN, GINE and AttentiveFP variants with documented features, operators and training rules. They are not a claim to reproduce missing historical weights or the defaults of a third-party package.

## Intended use

Research use: compare established structure-based classifiers under one molecular grouping/split protocol, prioritize compounds for additional investigation and examine uncertainty, chemical coverage and model sensitivity. Predictions are conditional on the inherited dataset endpoint and the current chemical representation.

The workflow does not establish clinical renal safety, causal nephrotoxicity, dose-specific risk, injury severity, onset, renal accumulation, mechanistic pathways or an approved decision rule for patients. It does not replace pharmacovigilance adjudication, laboratory assays, animal studies or clinical assessment. No clinical deployment, prospective evaluation or external-cohort validation is claimed here.

## Data identity and endpoint

The authoritative workbook is `data/NephroTox_Modeling_Data.xlsx`, sheet `Modeling_Final`, with SHA256 `2832a2b0679bf724b6acc3197e3ad8e4d2c15e0572b2bc9bb76b217f4a0ba752`. It contains 1,673 compounds: 842 positive and 831 reference-negative. Labels are not regenerated during model fitting. The same isomeric SMILES and specified/unspecified stereo information enter all current feature and graph pipelines; representation-specific limitations remain.

PubChem CID, names, formula and standardized identity fields aid chemical traceability. They are not evidence accession numbers for the toxicity endpoint. A complete row-level ledger of original renal outcomes, dates/versions, primary evidence, reference-negative selection and conflict adjudication is not present in the current DATA. This limitation remains unresolved; chemical identity checks cannot substitute for it. Reference-negative does not mean proven absence of renal toxicity under all dose/exposure conditions.

The binary endpoint also aggregates heterogeneous possible kidney outcomes. Subtype, timing, severity, dose, duration, pharmacokinetics, comedication and patient susceptibility are not modeled. The data should not be represented as a homogeneous experimentally measured potency endpoint.

## Evaluation population

| Population | N | Positive | Reference-negative |
| --- | ---: | ---: | ---: |
| Development | 1,338 | 673 | 665 |
| Heldout test | 335 | 169 | 166 |

The 80:20 allocation and development ten-fold assignments isolate connected molecular parent/scaffold groups. The 959 groups comprise 754 development and 205 test groups. Grouping is deliberately conservative across multiple signatures and their transitive connections. This constrains identified structural leakage, but does not make the test a new source, a later time period or a prospective cohort. Similar chemical environments can remain across distinct groups.

All 43 models share these populations. Conventional algorithms retain the original three supplementary candidates. Each graph architecture performs a newly specified finite random search over 20 candidates: the original Recommended baseline plus 19 unique sampled configurations, with seed 20260810 and ranges declared in `configs/model_search_spaces.json`. These graph ranges were added for this release; `prespecified_no_hpo` describes only the historical table, not the present workflow. All candidates use three inner group folds inside each outer-training subset; selected configurations are evaluated on the untouched outer fold. Full-development tuning is repeated before final fitting, and neural training epochs are selected only inside the fitting population. Old selected winners, performance and epoch counts are not reused as current-data evidence. The primary model is chosen only from development-fold results among the 25 conventional main combinations; test-best performance does not determine it.

## Inputs, stereochemistry and missingness

The five main inputs are chiral Morgan2048, MACCS166, chiral AtomPair1024, chiral topological-torsion1024 and the explicitly listed RDKit166 descriptors. Count sensitivity uses true `GetCountFingerprint` matrices for the three applicable families. Molecular graphs encode the declared 2D atom/bond schema, including specified stereo; current schema dimensions and exact names are exported with the run. Unknown stereocentres are not completed. MACCS and the selected descriptors do not independently distinguish enantiomers; chiral hashes can also collide.

No 3D conformer, explicit exposure, pH-dependent microstate distribution or clinical covariate is an input. Grouping copies do not replace feature-input structures. Invalid matrices and structures fail rather than being silently removed. Descriptor imputation/scaling and count preprocessing are fitted only within the corresponding training subset, including internal calibration/epoch-selection fits. DNN selects epochs and its plateau learning-rate schedule internally, then replays that recorded schedule prefix during fresh fitting on the complete fitting subset. Graph epoch selection is similarly training-internal, with constant configured learning rates. Names and identifiers are never predictors.

## Model output and threshold

Each model outputs a positive-class probability-like score. The primary binary threshold is prespecified at 0.5. SVM scores receive explicit grouped out-of-fold sigmoid calibration within each fit; other model outputs are not automatically calibrated probabilities of clinical harm. DNN and graph scores average seed-specific sigmoid probabilities for seeds 20260810, 20260811 and 20260812 in both OOF and final fitting. Conventional classical models use one fit with seed 20260810.

Calibration curves, Brier score and log loss evaluate saved outputs; a threshold-distance measure is not labelled confidence. Calibration evaluated on the internal test does not justify post-test recalibration or threshold adjustment. Any later operational threshold or recalibration policy requires a separately validated development procedure.

## Results and artifacts to consult

Execution produces the following evidence tables; they need not pre-exist in the repository:

- `results/modeling/model_lock.json` for the complete model registry, source/protocol/artifact hashes and test-reporting authorization.
- `results/modeling/reports/primary_selection.json` for the development-only primary selection.
- `results/modeling/reports/model_summary.csv` and `perfold_metrics.csv` for metrics and intervals.
- `results/modeling/reports/oof_predictions.csv` and `test_predictions.csv` for ID-level predictions.
- `results/modeling/reports/paired_auc.csv` and `group_bootstrap_ci.csv` for corrected comparisons and structural-group sensitivity intervals.
- `results/modeling/reliability/` for AD thresholds, neighbors, coverage, error cases, calibration bins, discordant-label pairs and development-only structural-key associations.
- `results/modeling/benchmarks/` for repeated feature/inference timings, model-loading time, hardware/software/thread information and raw measurement tables.
- Per-model candidate, selection, epoch, effective-parameter and serialization logs for reconstruction of each fit.

All numerical reports must derive from the same validated global lock. AUC or ranking alone does not establish clinical safety or mechanistic validity.

## Uncertainty and failure modes

Reported bootstrap intervals condition on the saved fitted predictions. They do not include repeated retraining, new partition selection or the uncertainty of inherited labels. Class-stratified observation resampling is accompanied by a whole-structure-group sensitivity analysis for prespecified models. Outer-fold scores are correlated; multiple seeds are not independent biological replicates.

AD measures similarity to the current training chemical coverage, not a guarantee of correctness. Maximum and mean-five-neighbor Tanimoto thresholds are estimated from training leave-self-out values. Every OOF compound uses its own outer-training references and thresholds. Sparse or single-class subgroups can have unstable or undefined metrics and are labelled accordingly. In-domain predictions can fail, and out-of-domain predictions may be unreliable even at extreme scores.

High-similarity opposite labels may reflect genuine biological differences, unmeasured context, annotation uncertainty or representation limits. They are called binary discordant-label pairs; they are not confirmed quantitative activity cliffs. Pair statistics sharing compounds are dependent. Exact row-level label evidence is needed to resolve suspected misannotation.

## Interpretation boundaries

TreeSHAP on MACCS-RF and descriptor-RF is descriptive of those predeclared models. It is not automatically an explanation of a different chosen primary. Correlated keys/descriptors and hashed feature collisions limit chemical interpretation. MACCS numbering explicitly maps original keys 1–166 to model columns 0–165, with RDKit definitions retained.

Graph bond occlusion measures prediction change after deleting both directed message edges while keeping atom features fixed. It neither simulates a reaction nor proves that a bond causes kidney injury. Raw values and the common 95th-percentile display rule are distinguished. Development-only MACCS enrichment is exploratory association, not a kidney-specific toxicophore; Fisher/OR summaries do not account for within-scaffold dependence.

The former target-prediction/PPI/GO/KEGG analysis is removed. The retained four-cluster positive-only Figure 2 is descriptive chemical organization; its clusters do not establish shared mechanisms and cannot support cluster-specific specificity or ROC-AUC without negative samples.

## Computational efficiency

The post-lock benchmark rebuilds binary/descriptor features three times, count features once, and cache-cleared molecular graph features three times from all 1,673 current SMILES. Exact numerical correspondence and graph-schema/repeat checks protect timing-input identity. Loaded-model prediction uses one warm-up, three repeats of the fixed 335-record batch and ten single-record repeats for each of the first ten development compounds. Numerical features/cached graphs are already available for prediction; loading and feature construction are measured separately. Summaries retain variability and raw calls, hardware/OS/memory, library versions, bounded threads and artifact hashes. These measurements characterize the declared local CPU process, not webserver throughput or concurrency, and do not presume that the shortest fingerprint is faster or more accurate.

## Scientific limitations and extension requirements

- Execute the complete registry and derive comparisons only after the global model lock verifies the run.
- Retain every candidate trial, final configuration, OOF prediction, epoch history, seed result, model artifact and environment manifest as local reproducibility evidence.
- Supply a lawful source/label provenance ledger where records exist; disclose irrecoverable source evidence instead of inventing identifiers.
- Establish temporal, genuinely source-independent or prospective external validation before making broad generalization claims.
- Use harmonized, non-overlapping endpoints/data for comparisons with external tools; none is established by this release.
- Compare dose/exposure and renal subtypes only when appropriate annotated data become available.
- Validate any proposed biological mechanism experimentally; no such experiment is included.
- Benchmark production infrastructure separately before claiming response-time or concurrent-user capacity.
- Synchronize manuscript methods, results, figure numbering, references and reviewer replies with generated outputs.

## Reproducibility and presentation

The protocol fixes the source/split hashes, candidate configurations, grouped validation, seeds, threshold and inference aggregation. Actual package versions and fitting-source hashes are captured by the run. Current execution uses CPU and bounded numerical threads. Installation and execution are documented in `MODELING_PROTOCOL.md`; the complete candidate JSON and per-fit effective defaults remain machine-readable.

Arial is required for all current/future figures, including title, panel letter, axis, tick, legend, table and molecular atom text. Silent font substitution is disallowed. Original panel layouts and colors are retained where compatible with the revised scientific content. Visual consistency does not establish predictive or biological validity.
