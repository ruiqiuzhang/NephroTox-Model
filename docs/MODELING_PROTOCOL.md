# Current DATA modeling protocol

## Scope and code configuration

This document specifies the complete 43-model workflow, its inputs, evaluation rules and reproducible execution. The GitHub package contains process code, configuration and the current input workbook. Training artifacts and numerical reports are generated locally under `results/`; figures use the output directory configured by each script. Generated artifacts are not required to be present in the code-only repository.

The published machine-readable configuration is `configs/modeling_protocol.json`. Execution creates a run-specific protocol and digest under `results/modeling/`, including fitting-source hashes. A fitting-code or protocol change requires a separate frozen run. Historical performance tables, legacy feature matrices and previously selected best-model parameters are not numerical inputs.

This release retains the original comparison structure: 25 conventional representation–algorithm combinations and three molecular graph architectures. Fifteen count-fingerprint combinations are a separately labelled sensitivity analysis. The former target-prediction/PPI/GO/KEGG analysis is excluded. Figure 2 remains a descriptive four-cluster structural analysis; cluster membership is not used as an endpoint or as a model predictor.

## Authoritative input and endpoint

The sole scientific input is `data/NephroTox_Modeling_Data.xlsx`, worksheet `Modeling_Final`, which is a byte-identical copy of the supplied current DATA workbook.

| Integrity field | Frozen value |
| --- | --- |
| Workbook SHA256 | `2832a2b0679bf724b6acc3197e3ad8e4d2c15e0572b2bc9bb76b217f4a0ba752` |
| Ordered-row fingerprint | `51909c995bb610e02cb51e7ddc902b8224722a9ec31bbd9c5b95a9a3cdfc605d` |
| Split-payload SHA256 | `7a2d350020899c419ea15fe4fc4747ed82d4e1f4573043ce7bf286e532bfc6f3` |
| Records | 1,673 |
| Positive / reference-negative | 842 / 831 |

The existing DATA `label` values are carried unchanged. Positive means the inherited nephrotoxicity-associated binary label; reference-negative is an operational class and does not establish absence of renal toxicity. The current workbook does not supply a complete record-level chain of original renal terms, source versions, evidence accessions and label-adjudication decisions. PubChem CID documents chemical identity, not toxicity-label evidence. This run does not recreate missing source evidence, establish causal drug–kidney injury relationships, or repeat pharmacovigilance signal extraction. No unsupported PRR/ROR selection rule is inferred from the current table.

Only the active DATA fields are used for identity checks. Five entirely empty unnamed spreadsheet columns are not predictors. Names, CID, labels, row position, fold number and group keys are excluded from all feature matrices. Input SMILES and specified stereochemistry remain authoritative; unknown stereocentres are not completed.

## Fixed development/test split and structural isolation

| Population | Compounds | Positive | Reference-negative | Structure groups |
| --- | ---: | ---: | ---: | ---: |
| Development | 1,338 | 673 | 665 | 754 |
| Heldout test | 335 | 169 | 166 | 205 |
| Total | 1,673 | 842 | 831 | 959 |

The development fraction is 1,338/1,673 and the test fraction is 335/1,673, implementing the requested approximately 80:20 allocation. The split seed is 20260810. All models use the same test set and the same ten outer development validation folds from `results/split_assignments.csv` and `results/cross_validation_folds.csv`. The test is an **internal structure-group heldout test**, not a source-independent, temporal or prospective external cohort.

Structural groups are connected components linked by any shared grouping key, including transitive links: parent, protomer, typed Murcko scaffold, raw scaffold, tautomer scaffold, element parent and element scaffold. Acyclic molecules use a complete-parent fallback rather than one universal empty-scaffold group. Grouping transformations act on disposable molecular copies; the feature-input SMILES are not replaced by grouping structures. See `SPLIT_PROTOCOL.md` and `src/nephrotox/split.py` for the exact operations and allocation method.

Holdout and outer-fold construction require zero overlap for the declared group keys. Inner search, SVM calibration and neural epoch selection also require group-disjoint fitting and assessment subsets. These constraints prevent leakage under the implemented parent/scaffold definitions. They do not guarantee that all cross-partition molecules are chemically dissimilar, eliminate fingerprint collisions or prove external generalization. The holdout construction uses group composition and labels to meet the fixed allocation; it uses no model predictions or performance scores.

## Molecular representations

| Representation | Definition |
| --- | --- |
| Morgan2048 | RDKit Morgan generator, radius 2, 2,048 bits, `includeChirality=True`, `GetFingerprint` |
| MACCS166 | `GenMACCSKeys`, discard unused original bit 0; retain original keys 1–166 in order |
| AtomPair1024 | RDKit AtomPair generator, 1,024 bits, `includeChirality=True`, `GetFingerprint` |
| Torsion1024 | RDKit topological-torsion generator, 1,024 bits, `includeChirality=True`, `GetFingerprint` |
| RDKit166 | Exact 166-name list in `configs/rdkit_descriptors_166.txt`; never truncate a version-dependent registry |
| Count sensitivity | `GetCountFingerprint` from the corresponding chiral Morgan/AtomPair/Torsion generator, with the same hashed width |
| Molecular graphs | Direct 2D molecular connectivity and specified stereochemical features from the same DATA SMILES |

The binary features use the named RDKit generator defaults in the pinned version. In particular, binary AtomPair/Torsion count simulation and true hashed integer count fingerprints are distinct representations; they must not be described as identical encodings with only a different array dtype. The count matrices are not constructed by decoding or counting set bits in binary vectors. Hashed representations can collide.

MACCS matrix column 0 means original key 1; column 165 means key 166. Definitions and count thresholds are supplied from RDKit `MACCSkeys.smartsPatts`, including algorithmic notes for keys 1, 125 and 166. MACCS166 and the selected RDKit descriptors do not independently distinguish enantiomers. Preserving stereo in DATA therefore does not imply that every representation is stereo-sensitive. Chiral fingerprints and graph features retain supported specified information without inventing unknown configurations.

The current graph schema is `nephrotox-graph-v2-canonical-stereo-original-indices`: 221 atom features and 33 bond features. It includes canonical-traversal chiral tags/permutations, CIP codes and bond stereo/direction, mapped back to original atom/bond indices. Exact ordered feature names are returned by `graph_feature_schema()` and exported with the run. There is no 3D conformer ensemble, pH-dependent microstate enumeration, image rasterization, ImageNet initialization or downloaded pretrained model. See `GRAPH_MODELS.md` for architecture operators and checkpoint semantics.

## Original supplementary parameter design

`configs/original_parameter_design.json` records the source supplementary parameter tables and their machine-readable mapping. Conventional RF, SVM, MLP, XGBoost and DNN each retain the original three candidate configurations. The Recommended graph-model table supplies one original configuration per architecture and explicitly labelled the historical design `prespecified_no_hpo`. In this release those configurations are baseline candidates, and graph search spaces are newly specified in `configs/model_search_spaces.json`; the old label is not the current training procedure. Old selected winners, numerical performance and final epoch counts are not inputs or conclusions for the current DATA. Group-isolated calibration, epoch selection and outer assessment are performed on current fitting populations.

## Nested candidate selection and honest OOF assessment

1. Reserve one of the fixed ten development folds as outer validation. It is not available for preprocessing fitting, candidate ranking, early stopping or probability-calibration fitting.
2. For each conventional/count combination and each graph architecture, on the remaining outer-training rows create three `StratifiedGroupKFold` inner folds, with shuffle enabled and seed 20260810.
3. Evaluate every frozen candidate in each inner fold. Conventional models compare the original three listed configurations. Each graph model compares 20 configurations: the original baseline plus 19 unique configurations sampled from the declared graph search space using seed 20260810. Candidate generation is deterministic and frozen before performance evaluation. A custom Python loop uses scikit-learn group splits and the implemented estimator APIs; graph parameter selection is a finite random search, not Bayesian optimization, Optuna, an exhaustive grid or an unrestricted manual search.
4. Rank candidates by the unweighted mean of the three inner ROC-AUC values, using saved validation probabilities. An exact tie selects the first candidate in the frozen list.
5. Refit the chosen candidate using only that outer-training population, including its own further grouped calibration/epoch procedure. Predict the untouched outer fold.
6. Repeat for all ten outer folds. Every development compound receives exactly one OOF probability from a model that did not fit or select parameters using that compound's outer fold.
7. Repeat the same three-fold candidate search on all 1,338 development records. Freeze the selected configuration and refit on all development records. Final prediction uses this refit, not an ensemble of outer-fold models.

Each graph candidate is actually fitted and evaluated in the three inner folds. The selected graph configuration is then fitted on the outer-training population and predicts the untouched outer fold, using the same assessment populations as conventional models. Its training epoch is selected afresh inside each fitting subset; an inherited final epoch is never copied into the current fit. Full-development graph selection repeats the same candidate search before fitting its final model. The newly supplied search spaces and sampled trials must not be represented as evidence already present in the old supplementary tables.

Three candidates per conventional algorithm and 20 per graph architecture are listed completely in `configs/modeling_protocol.json`. The 43-model registry and its 180 parameter entries are also supplied as `configs/model_registry.csv` and `configs/model_candidate_parameters.csv`: 40 conventional/count combinations × three candidates = 120 entries, plus three graph architectures × 20 = 60 entries. The scope is a finite declared comparison, not proof that each algorithm reached its global optimum. Search uses one initialization seed, 20260810. The same three neural evaluation seeds—20260810, 20260811 and 20260812—are used for each chosen outer fit and the final full-development fit. DNN/graph probabilities are arithmetic means of member sigmoid probabilities. RF/SVM/MLP/XGB use one fitted model at seed 20260810. Repeated classical probabilities written into a three-row storage array are duplicate storage, not three independent fits.

The conventional primary is selected from the 25 main conventional combinations by the mean of ten outer-fold ROC-AUC values, with model ID as tie-breaker. Count sensitivity models are excluded from retrospective primary selection. A representation comparator is also chosen using development performance. No test-best model is promoted to primary. Neither side-by-side representations nor within-architecture seed averaging constitute cross-representation feature fusion or stacking.

## Training-only preprocessing and model details

Every estimator receives a raw feature matrix for its particular fitting subset. Preprocessing is encapsulated inside `fit`, including the repeated subfits used for SVM calibration and neural epoch selection. No global scaler/imputer is fitted before CV.

| Input / model | Processing |
| --- | --- |
| Binary fingerprints, all models | Preserve 0/1 values; no fitted scaling |
| RDKit166, RF/XGB | Median imputation fitted on that fit subset only |
| RDKit166, SVM/MLP/DNN | Training-only median imputation, followed by training-only standard scaling |
| Count, RF/XGB | Original nonnegative integer counts |
| Count, SVM/MLP/DNN | `log1p` transformation followed by training-only standard scaling |
| Graphs | Deterministic per-molecule construction; no population-fitted featurizer |

The raw-matrix loader requires finite validated input. Imputation is an explicitly scoped estimator operation, not permission to silently include invalid feature rows. Unsupported matrices, invalid structures and missing structural groups fail explicitly.

RF uses balanced class weights, fixed random seed and the candidate tree/depth/leaf/feature settings. SVM uses an RBF `SVC` with `probability=False` and balanced class weights. Its probabilities are obtained by three-fold **grouped out-of-fold sigmoid calibration**, with `CalibratedClassifierCV(ensemble=False)`; the final base estimator is refitted on the full fitting subset. The library's ungrouped internal SVC probability routine is not used.

MLP is scikit-learn `MLPClassifier`, ReLU/Adam, batch size 128, with internal random validation and early stopping disabled. It runs the declared fixed 200-iteration budget; loss curves, iteration count and convergence warnings are retained. A maximum-iteration warning is not hidden or described as convergence. DNN is a separate PyTorch architecture with two candidate-defined hidden layers, BatchNorm, ReLU, dropout and AdamW; it is not merely a relabelled scikit-learn MLP. XGBoost uses CPU histogram trees and a fixed candidate-defined tree count, with binary-logistic loss and training-only negative/positive class ratio; it does not use an outer-validation early-stop set.

The complete effective estimator defaults, not only searched parameters, are recorded in per-fit metadata. XGBoost's legitimate NaN missing-value sentinel is documented as the string `NaN (XGBoost missing-value sentinel)` in strict JSON metadata; the actual estimator retains its NaN sentinel. Nonfinite prediction values, metrics or unexpected parameter values are not silently converted.

## Neural epoch selection and refitting

| Component | DNN | D-MPNN / GINE / AttentiveFP |
| --- | --- | --- |
| Grouped epoch holdout | First fold of a further five-fold group split of the current fit subset | Approximately 20% group-disjoint holdout, using declared deterministic grouped candidates |
| Monitor | Internal validation ROC-AUC, maximize | Internal validation unweighted BCE, minimize |
| Epoch cap | 80 | 60 |
| Patience / minimum improvement | 10 / 0.0001 | 8 / 0.0001 |
| Training objective | Positive-weighted BCE | Positive-weighted BCE |
| Optimizer | AdamW | AdamW |
| Learning rate | Candidate initial value; `ReduceLROnPlateau(mode="max", patience=4, factor=0.5)` on training-internal validation ROC-AUC | Fixed configured value |
| Gradient clipping | No additional clipping | Norm 5.0 |
| Final fit-subset model | Fresh initialization, all fit-subset rows, selected fixed epoch count | Same |

Positive class weight is calculated from the rows actually used by each training stage. Epoch-selection preprocessing is fitted only on its internal training side. The full fit-subset refit uses a fresh preprocessing fit on that full subset. The early-stop checkpoint is discarded before refitting, with the same seed used for fresh initialization. DNN retains the original plateau scheduler: the training-internal selector records the actual learning rate used at every epoch; the full fit-subset model then replays the recorded prefix through the selected epoch without using an outer-validation or test monitor. Graph models use a constant learning rate. The graph epoch cap, patience, minimum-improvement and AdamW/constant-rate training rules are explicitly supplied implementation settings, distinguished from the original Recommended architecture/hyperparameter fields. Selected epochs, all selection/refit curves, input IDs, groups and per-seed metadata are retained. Reaching an epoch cap does not prove convergence.

## Global test lock and reporting

`training_context()` masks every test label as -1. Test structures may be passed for inference after each final configuration lock, but test labels do not enter training or tuning. Formal test evaluation is allowed only after **every registered model** has complete metadata, a saved model and aligned prediction arrays with verified hashes. `lock_all()` then writes `results/modeling/model_lock.json`. The report loader revalidates source, protocol, model, metadata, prediction, ID and split correspondence before extracting test labels. A missing model or modified artifact prevents reporting.

The classification threshold is fixed at 0.5. Metric calculations use the same saved probabilities and predictions for ROC-AUC, ACC, sensitivity, specificity, precision, F1, MCC, balanced accuracy, AP, Brier score, log loss and TN/FP/FN/TP. AP is stepwise average precision, not trapezoidal PR-AUC. AUC is undefined for single-class subsets; undefined ROC-AUC/BA in AD subgroups are left empty rather than claimed as informative values.

The primary 95% intervals use 2,000 class-stratified bootstrap draws, seed 20260810. A prespecified sensitivity analysis resamples complete structural groups for the development-selected primary and the three graph models. These intervals are conditional on saved fitted predictions and do not include model retraining or alternative split uncertainty. Paired AUC comparisons share resampled observations, use centered two-sided bootstrap differences with a plus-one correction, and apply Holm correction across the primary-versus-main and three graph-pair comparison family. Ten overlapping outer training fits and multiple seed fits are not independent biological replicates.

Read numerical conclusions from the following generated tables rather than copying values into this protocol:

| Conclusion | Authoritative generated fields |
| --- | --- |
| Primary identity and rule | `reports/primary_selection.json`: `primary_model_id`, `selection_metric`, `test_used_for_selection` |
| Main performance | `reports/model_summary.csv`: `model_id`, `partition`, metric names and `_lower` / `_upper` fields |
| Fold variability | `reports/perfold_metrics.csv`: `model_id`, `fold`, metrics |
| Prediction evidence | `reports/oof_predictions.csv`, `reports/test_predictions.csv`: ID, label, probability, fold/group |
| Statistical comparisons | `reports/paired_auc.csv`: difference, interval, `p_bootstrap`, `p_holm`, family size |
| Group uncertainty | `reports/group_bootstrap_ci.csv` and saved bootstrap draws |
| Measured computational cost | `reports/runtime_summary.csv`: measured fitting/inference time and model bytes; see timing-scope notes |
| Repeated efficiency benchmark | `benchmarks/feature_benchmark.csv`, `runtime_benchmark.csv`, raw measurement tables and `runtime_environment.json` |

For all models, per-fit search evidence is stored under `<model_id>/outer_XX/candidate_XX/inner_X/`, with `trial.json`, prediction hashes and selection records. Graph records include the actual inner-fold scores of all 20 candidates. Full-development selection, final-fit logs, model weights, configuration locks and final prediction arrays remain separately traceable. The supplementary search plot separates the 40 conventional/count combinations with three candidates from the three graph architectures with 20 candidates; every plotted search score is verified against saved inner validation probabilities.

`scripts/benchmark_models.py` performs the separate post-lock efficiency experiment without retraining, threshold selection or test-label scoring. Fresh SMILES parsing, numerical feature computation and allocation are timed for all 1,673 records: three repeats for binary fingerprints/descriptors, one for each count family, and three cache-cleared graph-construction repeats. Recomputed numerical matrices must match their frozen counterparts exactly; graph schema and repeat hashes are checked. Counts and binary matrices are timed separately rather than assuming equal computational cost.

For each locked model, joblib loading is timed separately. After an explicit warm-up, the fixed 335-record test batch is predicted three times; the first ten development compounds in DATA order are each predicted ten times individually. Prediction timing includes fitted preprocessing and declared probability aggregation, with precomputed numerical inputs/cached graphs; feature generation, workbook reading, disk writing and model loading are excluded from these prediction intervals and reported separately. The script records raw measurements, means, medians, variability, single-call 95th percentile, model bytes, CPU/memory/OS, package versions, thread settings and model/source/protocol hashes. The operating-system file cache is not flushed for model-loading measurement. Wall times depend on concurrent workload; this is a local CPU efficiency experiment, not a deployed-server latency, throughput or concurrent-user benchmark. No speed advantage for MACCS or another representation is presumed.

## Reliability, interpretation and figure limits

AD uses the chiral Morgan2048 binary Tanimoto matrix. Test thresholds are the fifth percentiles of development leave-self-out maximum and mean-five-neighbor similarities. For OOF analysis, both reference compounds and threshold estimation are restricted to each corresponding outer-training fold. The two AD definitions are reported separately; probability distance from 0.5 is labelled threshold distance, not calibrated confidence. Displayed domain CIs resample whole structural groups with the replicate count stated in `reliability_manifest.json`.

Binary discordant-label pairs use Tanimoto >=0.85 from current compounds. Development endpoints use their OOF probabilities; test endpoints use final-model probabilities. Pair endpoint IDs and probability origins are explicit. Shared endpoints are deduplicated for compound-level metrics. These binary label differences are not quantitative potency-based activity cliffs.

All 166 MACCS keys form the predefined association family. Two-sided Fisher tests and BH adjustment use development labels only. Full definitions, support counts and all P/q values are exported. Corrected OR intervals are exploratory compound-level summaries and do not model within-scaffold dependence. No associated key is declared a kidney-specific toxicophore or causal mechanism.

TreeSHAP is restricted to the predeclared MACCS-RF and descriptor-RF explanation models. A different development-selected primary does not retroactively turn these into explanations of that primary. MACCS dependencies and descriptor correlations limit attribution interpretation. Graph bond occlusion removes both directed edges while retaining original atoms and atom features; delta is original probability minus perturbed-graph probability. The perturbed graph is not claimed to be a chemically valid reaction product. Raw deltas are retained; the 95th-percentile display rule must match figure annotations and captions.

All main and supplementary figures use the shared Arial policy, including molecular atom glyphs. The original panel layouts and model order are retained when their scientific meaning remains valid. Numerical results, model rankings, case selection and uncertainty annotations are recalculated. The original target/network/enrichment figure is removed, so former Figures 4–8 become current Figures 3–7.

## Chinese execution guide

在仓库根目录安装 `requirements-modeling.txt` 后，先运行数据/特征/划分复现及独立校验。正式建模必须使用同一 `results` 目录中的固定 DATA 快照、指纹和分组；不得接入旧表或旧预测。

完整自动流程入口如下，包含数据准备、全部模型、全局锁、报告、可靠性、解释、计时和绘图：

```powershell
python scripts/run_complete_modeling.py --results results --workers 2 --threads 2
```

若按阶段执行，可使用：

```powershell
python reproduce.py --output results
python reproduce.py --output results --validate-only
python scripts/run_modeling.py --prepare --results results
python scripts/run_modeling.py --all --threads 2 --results results
python scripts/run_modeling.py --lock-and-report --results results
python scripts/benchmark_models.py --results results/modeling --threads 2
python scripts/draw_model_figures.py --results results/modeling --output results/modeling/figures --supplementary
python scripts/draw_reliability_figures.py --results results/modeling
```

最后一条绘图命令要求先运行可靠性分析 `report_features(...)`，生成同一锁下的 `results/modeling/reliability`；完整自动流程会执行此步骤。若尚未生成，该绘图命令应报错，不应代入历史值。分步调用时，通过 Python API 传入当前 DATA 行、固定 `assignments`、经哈希核验的 `morgan2048` 和 `maccs166` 矩阵。

训练可以按 `--model-id` 独立执行或从缓存恢复；只有协议、训练/验证ID及保存预测哈希一致的缓存才可复用。拟合源码变化后应建立新的冻结运行，不能混用旧缓存。不要将 smoke test 的缩小迭代次数或少量记录当作正式实验参数。

正文中的样本数、AUC、主模型和阈值应读取本次运行自动导出的表格。模型评估不能填补逐条毒性证据来源、剂量/暴露、外部队列、时间验证或湿实验的缺口。
