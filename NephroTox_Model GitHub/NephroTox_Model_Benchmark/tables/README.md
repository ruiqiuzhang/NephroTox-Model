# Normalized model tables

This directory provides GitHub-friendly comparison tables derived only from the supplied locked outputs.

## Files

- `model_performance_independent_test.csv`: all 25 traditional combinations, three molecular graph models, and the graph consensus in one normalized schema.
- `model_inventory.csv`: one row per model system with family, representation/architecture, selection role, source-code availability, and retained artifact scope.
- `NephroTox_Model_Tables.xlsx`: formatted workbook with scope notes, filterable source tables, and a cleanup log.

## Metric conventions

- `roc_auc`, `pr_auc`, `accuracy`, `precision`, `sensitivity`, `specificity`, `f1`, `mcc`, and `balanced_accuracy` are unitless proportions.
- CI columns contain bootstrap 95% confidence bounds when supplied.
- Blank CI cells mean that the source artifact did not provide a directly comparable interval in that table.
- `selection_role` must be read before comparing rows. Independent-test performance was not used to revise the locked primary-model designation.

