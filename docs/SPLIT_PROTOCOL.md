# DATA-only molecular groups and evaluation protocol

The sole scientific input is the current `DATA` workbook selected by the runner.
`nephrotox.split.split(rows, seed=20260810)` receives the parsed DATA rows and
performs no file reads, cache lookups, historical recovery or row mutation.
The real modeling SMILES retains all supplied stereo and isotope annotations.

## Seven indivisible grouping constraints

Disposable grouping graphs have stereo and isotope annotations removed, followed
by explicit-H removal. Ordinary parent copies pass through RDKit Normalizer,
Reionizer and force Uncharger. No largest-fragment operation occurs. The seven
keys are:

1. Canonical tautomer SMILES of the normalized achiral parent.
2. `HetAtomProtomerv2` hash of the normalized achiral parent.
3. Typed Bemis–Murcko scaffold before parent normalization.
4. Typed Bemis–Murcko scaffold after parent normalization.
5. Canonical tautomer SMILES of the normalized typed scaffold.
6. Element-labelled heavy-atom connectivity (`ElementGraph`) of the full parent.
7. Element-labelled heavy-atom connectivity (`ElementGraph`) of the scaffold.

Union-find constructs connected components of **any** shared key, including
transitive links through different keys. A component is never split between
development/test or between development validation folds. Stereo variants share
a group while their actual stereochemical structures remain distinct.

Typed scaffold keys retain atom identities and bond orders. The additional
ElementGraph constraints preserve elements/connectivity while ignoring bond
orders, aromaticity, H count and charge. They are deliberately conservative and
can also keep saturation or redox analogues together. Empty acyclic Murcko
scaffolds fall back to parent keys, so unrelated acyclic molecules do not share
one empty-string group. This does not guarantee isolation of every broadly
similar acyclic family or every highly similar molecular pair.

The generic implementation retains every atom/fragment of a complete
coordination parent and combines all nonempty organic ligand scaffolds. It never
substitutes the largest ligand or a bare metal ion for the complete input.

Full-parent and scaffold tautomer enumeration each has explicit limits of
1,000 tautomers and 1,000 transformations. Every enumeration status and every
limit-hit compound ID is reported. A truncated canonical key is not described
as exhaustive; enumeration-independent ElementGraph constraints still apply.

## Holdout and development cross-validation

The first fold of seeded five-fold StratifiedGroupKFold initializes the 20%
test set. Binary integer optimization moves whole groups as little as possible
to obtain `round(N*0.2)` rows with the nearest proportional positive count. For
the current 1,673-row DATA with 842 positive and 831 negative records, the target
is 335 test rows (169 positive, 166 negative) and 1,338 development rows.
Only group membership, row counts and labels influence this step. The seed is
fixed in advance; no descriptor values, models or predictive scores are used.
Infeasible quotas or failed optimization raise an error rather than split a
group or silently emit a different protocol.

Development receives ten seeded StratifiedGroupKFold validation-fold numbers.
On round `k`, development fold `k` validates and the other nine development
folds train. Test rows have a null fold and never enter CV. Fold sizes may differ
because structural groups are indivisible. Every validation fold must contain
both classes. Sixteen overlap checks cover all seven keys plus connected group
IDs across the holdout and CV folds; any overlap raises a nonzero failure.

## Train-only feature processing

`get_fold_indices(rows, assignments, fold)` returns global zero-based training
and evaluation indices. `fold=1..10` means development training/validation;
`fold=None` means full development training and independent test evaluation.
It validates source IDs/order, labels, fold semantics and all grouping fields.

`prepare_fold(X, rows, assignments, representation, fold)` accepts the five
published feature representations. Four fingerprints stay binary and unchanged.
RDKit166 uses a median SimpleImputer and StandardScaler fitted solely on the
training indices. Validation/test values are transformed without refitting.
NaN values can be imputed from training; infinity is rejected. Feature selection
or a predictive model, if added later, must likewise be fitted on training only.

All integrity guards use explicit exceptions and remain active with `python -O`.
Synthetic unittest cases exercise all seven transitive constraints, stereo /
protomer / isotope grouping, source immutability, holdout quotas, CV isolation,
error exit under optimization, and an evaluation-only perturbation proving that
validation/test values cannot change the fitted preprocessing statistics.
