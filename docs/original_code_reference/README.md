# Original source reference

The `traditional_models` directory preserves the original Python source,
candidate configuration and dependency declaration without alteration. It is
provided to document the code used as a methodological reference. Its earlier
sample counts, paths and random-split assumptions are not the current analysis.

The executable current implementation is `src/nephrotox/` and the entry point is
`scripts/run_complete_modeling.py`. It accepts the current 1,673-record DATA
workbook only. No historical data matrices, model weights, predictions or
performance tables have been copied into this reference directory. The previous
graph-model release did not contain a trainable implementation; the current
three graph architectures have new, documented source and tests.
