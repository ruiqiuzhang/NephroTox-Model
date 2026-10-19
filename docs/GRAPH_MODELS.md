# Molecular graph models

`src/nephrotox/modeling_graph.py` implements D-MPNN, GINE and AttentiveFP with
RDKit and ordinary PyTorch operations. No pretrained model, PyTorch Geometric,
DGL or compiled graph extension is required. All fitting and prediction use
CPU. `n_jobs=2` limits each fit to two PyTorch threads.

## Hyperparameter search and reference candidate

All three models use `selection_mode="seeded_random_search"`. The ranges,
sampling distributions, budget and random seed are declared in
`configs/model_search_spaces.json`; `src/nephrotox/search_space.py` generates
the candidate list deterministically. Each architecture compares **20 unique
configurations**: one original reference and 19 seeded random samples. The
sampling seed is 20260810. This is a finite randomized search, not an exhaustive
search or a claim of global optimality.

The parameters below reproduce the **Locked parameters** column of the supplied
supplementary Table S3, `S3 Graph Configs`, solely as the reference configuration
at candidate index 0. `GRAPH_REFERENCE_CONFIGS` stores its original parameter
names. It is not fixed as the winner. No historical scores or selected epochs
are used to choose a configuration for the current DATA.

The executable search ranges are:

| Architecture | Categorical search dimensions | Architectural constants |
| --- | --- | --- |
| D-MPNN | Message width {128,256,384}; message depth {3,4,5,6}; message dropout {0,0.1,0.2}; FFN width {128,256,300,512}; FFN hidden layers {1,2,3}; FFN dropout {0.1,0.25,0.4}; batch size {32,64} | Sum aggregation |
| GINE | Hidden width {128,256,384}; graph layers {3,4,5}; dropout {0.1,0.2,0.3}; batch size {32,64} | Two-layer update MLP, batch normalization, gated-attention pooling, learned epsilon |
| AttentiveFP | Hidden width {128,256,384}; graph layers {2,3,4}; readout timesteps {1,2,3}; dropout {0.1,0.2,0.3}; batch size {32,64} | Single-head attention |

For every architecture, learning rate is sampled log-uniformly over
[0.0001,0.001] and weight decay over [0.000001,0.001]. Categorical values are
sampled by `sklearn.model_selection.ParameterSampler`; continuous dimensions
use `scipy.stats.loguniform`. Candidate generation uses no molecular labels,
fold scores or test information. The same reproducibly generated candidate
list is reused at all selection stages.

| Parameter | D-MPNN | GINE | AttentiveFP |
| --- | --- | --- | --- |
| Message/hidden width | `message_hidden_dim=256` | `hidden_dim=256` | `hidden_channels=256` |
| Message/graph depth | `message_depth=5` | `num_layers=4` | `num_layers=3` |
| Message dropout | `message_dropout=0.0` | `dropout=0.2` | `dropout=0.2` |
| Molecular pooling | `aggregation="sum"` | `pooling="gated_attention"` | Atom-to-molecule recurrent attention |
| FFN | `ffn_hidden_dim=300`, `ffn_num_layers=3`, `ffn_dropout=0.25` | Two-affine-map binary prediction head | Linear binary prediction head |
| GINE update | — | `mlp_num_layers=2`, `normalization="batch_norm"`, `train_epsilon=True` | — |
| Molecular readout steps | — | — | `num_timesteps=2` |
| Batch size | 32 | 32 | 32 |
| Learning rate | 0.0002875575337885439 | 0.0003 | 0.0003 |
| Weight decay | 0.00009791096446029439 | 0.0001 | 0.0001 |

Supplementary Table S3 does not specify an epoch cap, patience, gradient
clipping, optimizer type or holdout rule. The executable implementation makes
these additional choices explicit: AdamW; constant learning rate; maximum 60
epoch-selection epochs; patience 8; minimum BCE improvement 0.0001; gradient
norm clipping 5.0; approximately 20% grouped holdout inside each fitting subset.
These are fixed implementation settings, separate from the searched network
and optimizer parameters. The
table's previously reported final epoch counts are not fixed inputs to training
on the current DATA. Each fitting subset selects its own epoch as described below.

## Interface and data isolation

```python
from nephrotox.modeling_graph import GRAPH_CANDIDATES, make_graph_estimator

estimator = make_graph_estimator(
    "D-MPNN", selected_parameters,
    seed=20260810, seeds=[20260810, 20260811, 20260812], n_jobs=2,
)
estimator.fit(training_smiles, training_labels, training_structure_groups,
              sample_ids=training_compound_ids)
probabilities = estimator.predict_proba(evaluation_smiles)
```

`fit` receives only the applicable training rows and has no argument for outer
validation or held-out test data. The runner uses the same fixed grouped outer
10 folds for every architecture and feature representation. For each outer
training subset, every candidate is evaluated using the same three inner
structure-group folds. Mean inner ROC-AUC selects the configuration; ties go
to the first candidate in the reproducibly generated list. Each inner fit
uses seed 20260810. The selected configuration is then fitted to the entire
outer-training subset, and only then predicts the outer-validation subset.

Final parameter selection repeats the same 20-candidate, three-fold grouped
search using only the full development set, then fits all development rows.
Development OOF predictions and this final development refit use the same
three seeds above. The test labels are not available during either selection
or fitting. Candidate generation alone does not constitute an executed search;
per-candidate scores and selected parameters are written only when training runs.
Member sigmoid probabilities are averaged arithmetically. Binary predictions
use probability >=0.5. The graph probabilities are not automatically calibrated.

## Graph features and stereochemistry

`graph_feature_schema()` returns feature names in exact numerical column order,
dimensions and schema version. All architectures use **221 atom features** and
**33 bond features**. Every categorical field has an explicit unknown channel.
These deterministic features require no fitting or population-level statistics.

| Object | Feature definition |
| --- | --- |
| Atom | Atomic number 1–118; total degree 0–10 including H; formal charge −5–5; total H count 0–8; hybridization; radical-electron count 0–4; canonical-traversal RDKit chiral tag 0–8; canonical non-tetrahedral chiral permutation 0–30; CIP R/S/r/s/unspecified; aromatic and ring flags; mass/100; explicit isotope/300 |
| Bond | RDKit bond-type category; stereo 0–7; direction 0–6; conjugation and ring flags |
| Connectivity | Two directed edges per original RDKit bond; reverse-edge indices; original bond indices; molecule-specific batch membership |

RDKit assigns labels only to stereo explicitly represented by the input SMILES.
Unknown stereocentres are not completed. Traversal-relative @/@@, non-tetrahedral
permutations and slash directions are read from canonical isomeric SMILES and
mapped back using `_smilesAtomOutputOrder`. Original input atom and bond indices
are retained. Equivalent randomized isomeric SMILES therefore represent the
same stereo features while keeping the original occlusion numbering. There is
no stereochemistry removal, AddHs, conformer generation or non-isomeric
conversion. These are molecular connectivity/stereo models, not 3D models.

Invalid, empty, disconnected or dummy-atom input molecules raise errors. A
single-atom molecule is supported. No input record is silently dropped.

## Architecture definitions

The implementation choices below specify operators not given by the parameter
table; widths, depths and other searched settings follow each actual candidate.
The numerical dimensions quoted below describe the original reference candidate.
Package-default numerical identity is not claimed.

**D-MPNN.** A directed edge v→w is initialized from source-atom and bond
features by a bias-free linear map and ReLU. For the remaining depth−1 steps,
incoming u→v edge states are summed excluding the immediate reverse w→v,
projected by a shared bias-free map, added to the initial preactivation and
passed through ReLU/message dropout. Incoming edge states at each atom are
summed and concatenated with original atom features, then projected with
ReLU/message dropout. Atom states are **summed** per molecule. Following the
[Chemprop v2 FFN convention](https://chemprop.readthedocs.io/en/latest/_modules/chemprop/nn/ffn.html),
`ffn_num_layers=3` specifies three hidden layers: 256→300→300→300→1, with ReLU
and 0.25 dropout between affine maps. Message dropout and FFN dropout are
separate parameters. This follows the
directed nonbacktracking construction of
[Yang et al.](https://doi.org/10.1021/acs.jcim.9b00237) and
[Chemprop](https://chemprop.readthedocs.io/en/latest/autoapi/chemprop/nn/message_passing/base/index.html).

**GINE.** A linear atom projection and ReLU produce 256-channel states. Each
of four layers forms ReLU(h_j+W_edge e_ji), sums incoming messages and adds
(1+epsilon)h_i, with a learned epsilon initially zero. A two-affine-map MLP of
width 256 with an intervening ReLU, batch normalization, ReLU and dropout
produces a residual update. No artificial self-loop edges are added because
the epsilon term provides the self state. A learned gate
Linear(256,256)→tanh→Linear(256,1) is softmax-normalized separately in each
molecule; its weighted sum pools the node states. A 256→256→1 ReLU/dropout head
produces the logit. For the exceptional training batch containing only one atom,
batch normalization uses its running statistics instead of invalid one-item
batch statistics. Evaluation always uses running statistics. This uses the
[GINE operator](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html)
from [Hu et al.](https://arxiv.org/abs/1905.12265).

**AttentiveFP.** The first neighborhood-attention layer conditions attention
keys on source-atom and bond features; later layers use projected atom states.
Single-head additive attention is normalized over incoming neighbors. No
artificial self-loop edges are added. ELU/dropout messages update atom states
with GRU cells and ReLU through three layers. The initial molecular state is
the sum of atom states followed by ReLU. Two atom-to-molecule attention/GRU/ReLU
steps refine it before dropout and a linear binary head. The attention uses one
head, which is an implementation definition rather than an omitted tuned
parameter. This follows the recurrent molecular readout of
[Xiong et al.](https://doi.org/10.1021/acs.jmedchem.9b00959) and the
[PyG architecture description](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/attentive_fp.html).

## Epoch selection and refitting

For each seed, a group-disjoint two-class holdout is selected strictly inside
the rows supplied to that `fit`. Stratified group splits and deterministic
group-shuffle candidates are considered; the split closest to the declared
size/class proportion is used. If a defensible two-class grouped split cannot
be formed, fitting fails explicitly; there is no ungrouped fallback.

Training uses weighted binary cross entropy, with positive weight
N_negative/N_positive calculated only on the applicable training portion.
Unweighted holdout BCE selects an epoch. The selection model is discarded and
a newly initialized model at the same seed is fitted to all supplied fitting
rows for exactly the selected number of epochs. Its positive weight is
recalculated on those fitting rows. The optimizer and constant learning rate
are the same in epoch selection and refit. Outer validation and test labels
cannot select the epoch.

`fit_metadata_` records all training/holdout IDs and groups, group-overlap checks,
loss curves, selected epochs, full refit curves, parameters, member seeds,
architecture definitions and hashes. Passing `log_dir` optionally writes these
records and per-seed state dictionaries during execution. The GitHub source
package contains no generated curves, metrics or model weights.

## Checkpoints and bond occlusion

`joblib.dump(estimator, path)` stores the estimator. `save_checkpoint(path)` and
`load_graph_estimator(path)` provide CPU state dictionaries with restricted
`weights_only=True` loading, architecture/schema checks and strict state loading.
Only trusted files should be loaded with joblib.

`predict_with_removed_bonds(smiles, [bond_index])` removes both directed edges
for the specified original RDKit bond indices, retains all atom features and
reruns the network. Delta probability is original minus masked P(positive).
This perturbation does not edit, re-sanitize or re-valence the chemical molecule
and is not a simulated reaction or evidence of a toxic mechanism. An empty mask
reproduces ordinary prediction.

## Focused tests

`tests/test_modeling_graph.py` covers original-reference values and mapping,
20-candidate reproducibility/uniqueness and actual architecture construction,
independent D-MPNN FFN/message dropout, GINE normalization/gated pooling,
AttentiveFP depth/readout steps, graph semantics, stereo/traversal invariance,
original-index occlusion, single atoms, batch isolation, grouped epoch selection,
fixed-epoch refitting, seed aggregation and serialization. Small synthetic
fixtures exercise the training interface without reading historical datasets
or creating scientific result files in the repository.
