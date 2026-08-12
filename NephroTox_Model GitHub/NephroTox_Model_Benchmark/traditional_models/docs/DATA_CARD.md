# Data card

## Dataset

`NephroTox_Dataset.xlsx` contains 1,853 quality-controlled small molecules used for binary nephrotoxicity modeling. The `Modeling_Final` sheet is the only sheet read by the pipeline.

| Property | Value |
|---|---:|
| Total compounds | 1,853 |
| Nephrotoxic | 931 |
| Reference-negative | 922 |
| Invalid standardized SMILES | 0 |
| Duplicate standardized structures | 0 |
| Acyclic compounds | 128 |

The required workbook SHA-256 is `435b0551b1475d2a3dd60de3d451140526521a5ea1b918d578b92172972ef220`.

## Intended use

The dataset supports retrospective method development and comparative evaluation of in-silico nephrotoxicity classifiers. It is not a substitute for experimental toxicology, clinical judgment, regulatory evaluation, or patient-level risk prediction.

## Labels and limitations

The negative class is described as **reference-negative**, not universally proven non-nephrotoxic. Source evidence can vary in completeness and endpoint definition. The current modeling table does not provide sufficient harmonized metadata for source-specific, phenotype-specific, or temporal validation. Results therefore estimate performance for this compiled binary endpoint and should not be generalized without external validation.

## Redistribution

Before public release, the author must confirm that records compiled from the underlying source databases may be redistributed in this form. If redistribution is not permitted, publish the code, schema, hashes, and reconstruction instructions without the workbook.

