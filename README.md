# Hot Deck Implementation Project

## Overview

This project implements a modular data science pipeline for predicting 191 categorical target variables for 660 unlabeled records. The 1,015 labeled records form the donor pool. For each target, the model identifies similar donors and transfers the most common observed target value among the nearest donors.

The main modeling decision is based on the structure discovered in the predictor data. Although the input contains 1,097 binary columns, those columns are not independent measurements. They form consecutive one-hot encoded bitmap groups representing categorical variables. Treating every bit independently would give large categorical variables more influence simply because they contain more columns. The pipeline therefore discovers the bitmap boundaries, decodes each retained group into one categorical predictor, and measures donor similarity at group level.

Donor similarity is target-specific. Raw mutual information estimates how relevant each decoded predictor group is to each target. These relevance values weight a categorical Hamming distance, allowing the same recipient to have different nearest donors for different target columns. After donor selection, voting is unweighted: every selected donor contributes one vote.

The implementation covers the requested preprocessing, feature engineering, reproducible donor splitting, Hot Deck fusion, evaluation, pipeline integration, independent module execution, configuration, testing, and documentation requirements.

The final target-weighted Hot Deck model selected k=11 through five-fold cross-validation. It achieved approximately 77% cell accuracy and 60% mean target macro F1, compared with approximately 73% accuracy and 39% macro F1 for the majority baseline.


## Assignment requirements and implementation

| Requirement | Implementation |
|---|---|
| Data preprocessing | `src/preprocessing.py` loads the datasets, validates their alignment and converts missing or non-finite bitmap cells into a representation that can later be decoded as unknown |
| Feature engineering or dimensionality reduction | `src/feature_engineering.py` discovers one-hot bitmap groups, removes structurally unsuitable groups and decodes retained groups into categorical predictors |
| Donor split modeling | `src/donor_split.py` creates reproducible shuffled donor/recipient folds and guarantees that every labeled row is held out once |
| Hot Deck fusion | `src/hotdeck.py` performs target-specific weighted donor matching and deterministic unweighted voting |
| Target relevance | `src/relevance.py` learns raw mutual-information weights for every target and predictor group |
| Evaluation | `src/evaluation.py` creates out-of-fold predictions, selects a global neighbor count and reports classification metrics against a fold-specific majority baseline |
| Pipeline integration | `src/pipeline.py` runs evaluation, uses the selected neighbor count, refits on all labeled rows and generates final test predictions |
| Central configuration | `src/config.py` contains all data paths, output paths and changeable modeling or evaluation parameters |
| Independent execution | Every main module can be imported without side effects and executed separately with `python -m` |
| Testing | `tests/` contains unit tests for all critical transformations, split behavior, distance logic, evaluation and configuration |

## End-to-end workflow

1. Load the training predictors, training targets and test predictors.
2. Validate row counts, indices, column consistency and predictor values.
3. Replace missing or non-finite bitmap cells with zero so an unavailable group can be represented explicitly after decoding.
4. Discover consecutive exact one-hot bitmap blocks from the training predictors.
5. Remove identity, constant, duplicate and deterministically derived bitmap groups.
6. Decode every retained bitmap group into one categorical predictor.
7. Analyze targets and temporarily remove constant targets from model fitting.
8. Create five reproducible donor/recipient folds.
9. Within every fold, fit raw mutual-information group weights using donor rows only.
10. Produce out-of-fold predictions for each configured global neighbor candidate.
11. Select one global neighbor count using macro F1, the configured tolerance, cell accuracy and a deterministic tie-breaker.
12. Compare the selected model with a majority-class baseline fitted independently in each fold.
13. Refit group relevance and the selected Hot Deck model using all labeled rows.
14. Predict the 189 modeled targets for the 660 final recipient rows.
15. Restore constant targets and the original 191-column order.
16. Save predictions, learned weights, evaluation tables, bitmap structure and pipeline metadata.

## Why bitmap grouping helped

The training predictor matrix contains 26 consecutive exact one-hot blocks with the following discovered sizes:

```text
1015, 2, 10, 2, 2, 5, 1, 6, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 11, 2, 7, 4
```

These boundaries are inferred from the training data; they are not encoded as fixed column ranges in the source code.

Decoding valid one-hot bitmap groups does not change donor rankings under unweighted Hamming distance, because two different categories always differ in exactly two bitmap cells. The groups were decoded to enable meaningful group-level weighting. Raw mutual information was then used to assign target-specific relevance weights to the decoded groups, allowing the distance calculation to emphasize predictors that are more informative for each target. This weighted approach significantly improved both macro F1 and cell accuracy compared with the unweighted model.

### Structural filtering

The configured transformer removes:

- `pred_0`–`pred_1014`, a 1,015-column row-identity bitmap. It uniquely identifies training rows, while final test rows are all zero in this block, so it cannot generalize.
- `pred_1036`, a one-column constant bitmap with no discriminative information.
- `pred_1029`–`pred_1030`, an exact coarse parent of the more detailed five-category group `pred_1031`–`pred_1035`. Keeping both would count the same hierarchy twice.
- Any exact duplicate decoded group, removing the duplicate as a complete categorical variable rather than deleting individual dummy columns.

After filtering, 23 decoded categorical predictor groups remain.

## Preprocessing decisions

The assignment requests missing-value handling, outlier treatment and scaling where appropriate. These operations are adapted to the actual data representation:

- An all-zero retained group is decoded to `FEATURES.unknown_category_value`, which defaults to `-1`.
- Conventional numeric outlier clipping is not applied: valid predictor cells can only be zero or one, so any other finite value is a schema error rather than a statistical outlier.
- Standardization and normalization are not applied to decoded categories because their integer codes are labels, not ordered numeric magnitudes.

This preserves the semantic structure of the predictors and avoids creating invalid bitmap combinations.

## Target analysis

`TargetAnalyzer` classifies target columns as constant, binary or multiclass. Constant targets contain no learnable variation, so they can be excluded from relevance fitting, distance calculation and cross-validation. Their observed constant values are restored after prediction so the final output retains all 191 target columns in their original order.

The analysis module also produces summaries used for diagnostics and for interpreting performance by target class count.

## Donor splitting

`DonorSplitter` wraps a shuffled five-fold split with `random_state=42` by default. For each fold it returns:

- donor predictors;
- donor targets;
- recipient predictors;
- recipient targets;
- the fold number.

With 1,015 labeled rows and five folds, each fold contains 812 donor rows and 203 recipient rows. Every row is used as a recipient exactly once and as a donor in the remaining four folds. Matching predictor and target indices are required, and repeated runs with the same configuration produce the same partitions.

This simulates the final task: recipient predictors are available, recipient targets are hidden from the model, and observed donor targets are transferred only after matching.

## Raw mutual-information group weighting

For target $t$ and decoded predictor group $g$, the relevance module calculates:

$$
s_{t,g}=I(X_g;Y_t)
$$

where $I(X_g;Y_t)$ is raw mutual information between the categorical predictor group and the categorical target. Positive scores are floored when necessary and normalized within each target:

$$
w_{t,g}=\frac{s_{t,g}}{\sum_j s_{t,j}}
$$

If every group has zero relevance for a target, the implementation uses equal group weights as a deterministic fallback. The result is a 189 × 23 weight matrix for the nonconstant modeled targets.

During cross-validation, weights are fitted only from the donor portion of each fold. During final prediction, they are refitted from all labeled rows. Recipient targets and final test data are never used to learn relevance.

## Target-specific Hot Deck distance

For recipient $r$, donor $d$ and target $t$, donor distance is:

$$
D_t(r,d)=\sum_g w_{t,g}\,\mathbf{1}(x_{r,g}\neq x_{d,g})
$$

The indicator is zero when the decoded group values match and one when they differ. Each target has its own relevance vector, so donor rankings can differ by target even for the same recipient.

The model uses stable ordering when distances tie. This makes donor selection reproducible and allows the ordered donor list calculated for the largest candidate value to be reused for every smaller candidate value.

## Unweighted voting

The word *weighted* describes the distance used to choose donors, not the final vote. Once the nearest $k$ donors for a target are selected, every donor receives one equal vote. The predicted target value is the mode of their observed values.

If multiple classes receive the same number of votes, the value supported by the closest selected donor wins. A final stable ordering rule keeps the result deterministic.

## Five-fold evaluation and global neighbor selection

Evaluation creates one out-of-fold prediction for every labeled row and every configured neighbor candidate. The current candidate range is:

```python
(5, 7, 9, 11, 13, 15)
```

Odd values reduce the frequency of voting ties. A single global $k$ is selected rather than tuning a separate value for every target, keeping the final model simpler and more stable when the dataset changes.

The selection rule is:

1. Find the candidate with the highest combined out-of-fold mean target macro F1.
2. Retain candidates no more than `0.001` below that value.
3. Among eligible candidates, select the one with the highest combined out-of-fold cell accuracy.
4. If both metrics remain tied, select the smaller neighbor count.

The expanded search selected `k=11`. Because candidates on both sides of 11 were evaluated, the selected value is not an artificial consequence of stopping the search at its previous upper boundary.

### Majority baseline

The official comparison contains only:

- the selected bitmap-group-aware, raw-MI-weighted Hot Deck model; and
- a majority-class baseline.

The baseline learns the most frequent value of every target using only the donor rows in each fold. It is recalculated per fold to preserve the same leakage-free evaluation conditions as the Hot Deck model. Its computational cost is small compared with donor-distance calculation.

### Reported metrics

- `cell_accuracy`: fraction of all modeled target cells predicted correctly;
- `hamming_loss`: fraction of modeled target cells predicted incorrectly;
- `exact_match_accuracy`: fraction of rows for which all 189 modeled targets are correct;
- `mean_target_accuracy`: mean of independently calculated per-target accuracies;
- `mean_target_precision_macro`: mean per-target macro precision;
- `mean_target_recall_macro`: mean per-target macro recall;
- `mean_target_f1_macro`: mean per-target macro F1.

Macro F1 is the primary selection metric because it gives minority classes meaningful influence. Cell accuracy is the secondary metric because it summarizes the overall proportion of correct output cells. Exact-match accuracy is expected to be very low or zero because a row must be correct on all 189 modeled targets simultaneously.

### Cross-validation result

| Model | Selected k | Cell accuracy | Mean target macro F1 |
|---|---:|---:|---:|
| Majority-class baseline | — | 0.734543 | 0.394610 |
| **Bitmap-group-aware weighted-distance Hot Deck** | **11** | **0.771548** | **0.602990** |

These are combined out-of-fold estimates over the 1,015 labeled rows, not scores on the hidden final test targets. The fold tables and per-target reports should be used alongside the headline averages when assessing stability and difficult target columns.

The selected k=11 model achieved approximately 77% cell accuracy and 60% macro F1, compared with 73% and 39% for the majority baseline. The larger improvement in macro F1 shows that the model predicts minority classes better instead of only favoring common values. Binary targets performed best, with about 80% accuracy and 65% macro F1. Four- and five-class targets were more difficult, reaching only about 51% accuracy and 35% and 30% macro F1, respectively. This is likely because the available donor records are divided across more classes, leaving fewer examples for rare values. With eleven-neighbor majority voting, frequent classes can also outvote a correct but rare nearby donor. Three-class targets achieved high accuracy but low macro F1, which suggests strong class imbalance. Future improvements could therefore include better distance weighting, distance-weighted voting, or selecting a suitable neighbor count separately for each target.

## Evaluation versus final prediction

The two commands have different responsibilities:

```bash
python -m src.evaluation
```

This command performs five-fold evaluation, selects the global neighbor count and updates the evaluation CSV files. It does not create final test predictions, learned full-data weights or pipeline metadata.

```bash
python -m src.pipeline
```

This command performs the integrated workflow. It runs the same configured five-fold neighbor selection, reads `evaluation.selected_k`, refits the model on all labeled rows and predicts the final test rows. It therefore uses `k=11` when evaluation selects 11; it does not use the standalone default merely because that value exists in the configuration.

`MODEL.n_neighbors` is the default used when `WeightedHotDeckImputer` or `python -m src.hotdeck` is run without an explicit value. `MODEL.neighbor_candidates` controls the cross-validated search used by evaluation and the complete pipeline.

## Central configuration

All changeable paths and parameters are defined in `src/config.py`.

### Important modeling settings

| Setting | Current value | Meaning |
|---|---|---|
| `FEATURES.drop_identity_groups` | `True` | Removes row-identity bitmaps |
| `FEATURES.drop_constant_groups` | `True` | Removes groups with no variation |
| `FEATURES.drop_derived_groups` | `True` | Removes redundant exact parent groups |
| `FEATURES.drop_duplicate_groups` | `True` | Removes duplicate decoded categorical groups |
| `FEATURES.unknown_category_value` | `-1` | Code used for all-zero or unavailable groups |
| `TARGETS.remove_constant_targets` | `True` | Excludes constant targets while modeling and restores them afterward |
| `RELEVANCE.weight_floor` | `1e-6` | Minimum positive relevance used before normalization |
| `MODEL.n_neighbors` | `7` | Default for standalone Hot Deck construction; the integrated pipeline uses the cross-validated selected value |
| `MODEL.neighbor_candidates` | `(5, 7, 9, 11, 13, 15)` | Global values evaluated by cross-validation |
| `MODEL.voting` | `unweighted` | Gives each selected donor one vote |
| `MODEL.chunk_size` | `16` | Number of recipient rows processed together during distance calculation |

### Evaluation settings

| Setting | Current value | Meaning |
|---|---:|---|
| `EVALUATION.random_state` | `42` | Reproducible fold shuffling |
| `EVALUATION.n_splits` | `5` | Number of donor/recipient folds |
| `EVALUATION.primary_metric` | `mean_target_f1_macro` | Main neighbor-selection metric |
| `EVALUATION.secondary_metric` | `cell_accuracy` | Tie-breaking metric within the F1 tolerance |
| `EVALUATION.f1_tolerance` | `0.001` | Maximum acceptable difference from the best macro F1 |
| `EVALUATION.final_tie_breaker` | `smallest_k` | Final deterministic selection rule |

`PATHS` defines all input and output locations relative to the project directory, so commands do not depend on machine-specific absolute paths.

## Installation

Python 3.9 or newer is recommended.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run all commands below from the project root while the virtual environment is active.

## Running the project

### Run the complete final pipeline

```bash
python -m src.pipeline
```

This selects the configured global neighbor count, fits the official model on all labeled rows and writes a 660 × 191 prediction matrix.

### Run evaluation only

```bash
python -m src.evaluation
```

This updates the cross-validation and neighbor-selection reports without generating final test predictions.

### Run each module independently

```bash
python -m src.preprocessing
python -m src.feature_engineering
python -m src.target_analysis
python -m src.donor_split
python -m src.relevance
python -m src.hotdeck
python -m src.evaluation
python -m src.pipeline
```

The `if __name__ == "__main__"` entry point in each module provides a focused diagnostic demonstration. Importing a module does not run its diagnostic code.

## Output files

| File | Contents |
|---|---|
| `outputs/bitmap_group_structure.csv` |  Discovered bitmap boundaries, decoded names and retention status |
| `outputs/neighbor_selection.csv` | Metrics, F1 differences, tolerance eligibility and selection status for every candidate k |
| `outputs/model_comparison.csv` | Headline out-of-fold comparison of the selected official model and majority baseline |
| `outputs/five_fold_metrics.csv` | Fold-level metrics for the selected model and majority baseline |
| `outputs/per_target_metrics.csv` | Out-of-fold accuracy, precision, recall and F1 for each modeled target |
| `outputs/performance_by_class_count.csv` | Per-target metrics aggregated by the number of observed classes in each target |
| `outputs/target_group_weights.csv` | Full-data raw-MI weight matrix for 189 targets × 23 decoded groups |
| `outputs/pipeline_metadata.json` | Selected k, configuration, data dimensions and structural counts for the final run |
| `outputs/test_target_predictions.csv` | Final 660 × 191 target prediction matrix |


## Testing

Run the complete test suite with:

```bash
python -m pytest -q
```

The suite contains 33 tests covering:

- missing-value handling and column consistency;
- bitmap discovery, decoding and structural filtering;
- invalid non-binary or multi-active bitmap detection;
- constant, binary and multiclass target analysis;
- reproducible donor splitting and index validation;
- raw mutual-information relevance and fallback behavior;
- target-specific weighted categorical distance;
- deterministic unweighted voting;
- reuse of ordered donors across multiple neighbor candidates;
- global neighbor selection and tie-breaking;
- out-of-fold coverage, metric correctness and index alignment.

## Computational considerations

Cross-validation is the most expensive stage because distance is target-specific and must compare each recipient with every donor. The implementation limits unnecessary work in two ways:

- recipient rows are processed in configurable chunks, limiting intermediate memory use;
- distances and ordered donors are calculated once per fold up to the largest candidate k, then reused to generate predictions for all smaller candidates.

The majority baseline, target summaries and CSV generation are inexpensive relative to distance calculation. Runtime depends on hardware, candidate count and dataset size. Increasing the dataset or the largest candidate increases work, but adding smaller candidate values does not require a separate complete distance calculation.

## Four-day implementation timeline

The project was organized as a focused four-day first-week delivery. Each day ended with an independently runnable, testable increment.

| Day | Focus | Main deliverables |
|---|---|---|
| **Day 1 — Standard implementation and data audit** | Understand the files and establish a correct baseline workflow | Dataset loading and validation; missing-value policy; target profiling; standard donor matching prototype; majority-class baseline; initial metric implementation; package and test structure |
| **Day 2 — Bitmap-aware modeling** | Use the structure discovered in the 1,097 predictors | Consecutive one-hot group detection; identity, constant, duplicate and derived-group filtering; categorical decoding; unknown-category handling; group-level categorical Hamming distance; raw mutual-information weights per target |
| **Day 3 — Experiments and model selection** | Test whether the more structured approach generalizes and tune its main global parameter | Reproducible five-fold donor splitting; leakage-free fold-local relevance fitting; global k comparison over 5–15; macro-F1-first selection with an accuracy tolerance rule; per-target and class-count diagnostics; comparison with the majority baseline |
| **Day 4 — Integration and cleanup** | Turn the experiment into a reproducible submission | End-to-end pipeline; refit on all labeled rows; constant-target restoration; output reports and metadata; removal of unused experimental branches; centralized configuration; PEP8 cleanup; unit-test completion; comprehensive README and execution checks |

### Milestones after the first delivery

Future development could explore more sophisticated methods for weighting predictor groups in the donor-distance calculation, beyond raw mutual information, to better represent nonlinear relationships and interactions between predictors and targets. The voting stage could also be extended with distance-weighted voting, giving closer donors more influence than donors near the edge of the selected neighborhood. Finally, the current single global neighbor count could be made target-specific, allowing each target to select its own optimal value of k based on cross-validation while applying suitable regularization or selection constraints to avoid overfitting.


## Error handling and validation

The package raises clear errors for conditions that would otherwise produce misleading output, including:

- missing input files;
- empty datasets;
- mismatched training predictor and target row counts;
- mismatched train/test predictor columns;
- duplicate column names;
- non-binary finite predictor values;
- invalid one-hot groups with multiple active categories;
- misaligned predictor and target indices;
- impossible fold counts or neighbor counts;
- missing, negative or non-finite relevance weights;
- prediction columns or indices that do not align with the expected targets.

## Removed unused and experimental code

To keep the submitted pipeline focused, unused compatibility functions and experimental branches were removed, including:

- raw-cell donor estimation paths that ignored bitmap grouping;
- adjusted and permutation mutual-information branches;
- automatic multiclass voting experiments;
- inverse-distance vote weighting;
- unused binary feature-selection helpers and legacy report aliases;
- redundant relevance-table accessors and validation aliases;
- duplicated data paths, output paths and repeated model constants.

The main classes, diagnostic entry points and independently executable pipeline stages remain because they directly support the assignment’s modularity requirement.

## Limitations

- The structural interpretation of consecutive one-hot blocks is inferred from the supplied data. A production system should also accept an explicit schema when available.
- Cross-validation estimates performance on the labeled sample; final test targets are unavailable, so final test accuracy cannot be reported.
- Rare classes and targets with four or five classes remain more difficult than common binary targets.
- Raw mutual information measures marginal relevance and does not explicitly model interactions between predictor groups.
- A single global neighbor count favors simplicity and stability but may not be optimal for every individual target.

These limitations are visible in the diagnostic CSV files and provide clear directions for future work without complicating the official submitted model.
