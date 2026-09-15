# AgentFault failure taxonomy classifier

This package consumes complete-trajectory feature rows. It reuses the existing
77 `feature_engineering.trajectory_features.FEATURE_COLUMNS` and the existing
preprocessing builder. It does not read step features, alter feature extraction,
change the workflow, or implement failure detection/localization.

## Commands

Run from the repository root:

```powershell
python -m pip install -r requirements-taxonomy.txt
python scripts/train_taxonomy_classifier.py
python scripts/evaluate_taxonomy_classifier.py
python scripts/predict_taxonomy.py --trajectory-id AF-0029_fault_KNOW_CITATION_MISMATCH_step_7
python scripts/predict_taxonomy.py --features path/to/row.csv
python -m unittest tests.test_taxonomy_classifier tests.test_feature_engineering -v
```

Training accepts `--features`, `--manifest`, `--model-dir`, `--results-dir`, and
`--random-state` (42 by default). Use `--embedded-split` only when the feature CSV
already contains the split assignments. The evaluator supports equivalent input
and output arguments and checks dataset/manifest SHA256 against model metadata.
Evaluation reproduces fixed-model results; it never chooses or retrains a model.
The prediction CLI accepts one or more CSV rows, or filters one trajectory ID.
Labels may be present in that CSV but are dropped before prediction.

## Data and taxonomy contracts

`utils.py` explicitly maps the existing `injector.faults.FaultType` enum members
to six parent categories. Coverage must match the entire enum. Prefix guesses are
not used. Clean/unlabelled records are excluded and counted; a valid subtype needs
an injected-fault flag or an observed failure outcome to be eligible. Unknown
nonempty taxonomy labels raise an error, preventing silent spelling mistakes.

The existing manifest is joined one-to-one by trajectory ID, with task identity
checks. Duplicate IDs, missing assignments, crossed task groups, invalid splits,
and conflicts with embedded splits are rejected. `base_task_id`, when available,
is additionally checked for grouping. There is no automatic re-split.

Only the declared predictors enter any sklearn Pipeline. Labels, task and
trajectory IDs, source, injection provenance, and split are excluded. The feature
dictionary is also checked when available next to the input CSV. Missing
predictor columns or infinite values cause an explicit error. Raw CSVs are never
modified. Each candidate fits its own preprocessing on training rows alone.

## Training and selection

- Logistic Regression: C = 0.1, 1, 10; balanced class weights; standardization.
- Random Forest: (150 trees, max depth 5, minimum split 2) and (250 trees,
  unrestricted depth, minimum split 4), both with balanced class weights.
- XGBoost, when installed: (100 trees, depth 2, learning rate .05) and (150 trees,
  depth 3, learning rate .1); subsample/column sample .9; balanced sample weights.
- Seven total candidates; seed 42; CPU execution with one estimator worker.
- Selection uses validation Macro-F1 only. Exact ties keep the earlier candidate
  (Logistic Regression before Random Forest before XGBoost). The selected model
  remains fitted on training only; validation is not included in a final refit.
- Test predictions are made only after selection and subtype training are done.
  Every candidate's validation report is saved in `validation_trials.json`;
  `model_comparison.csv` contains the best validation configuration per family.
- Absent training categories are listed explicitly. Model probabilities only
  contain categories actually observed in training. At least two training
  categories and nonempty training/validation splits are required.

Subtype models use a fixed 150-tree balanced Random Forest, with separately fitted
training-only preprocessing. A category needs at least six training rows, at
least two observed subtypes, and at least two examples in every observed subtype.
Otherwise the subtype model is unavailable, with a saved reason. This is a
conservative prototype feasibility rule, not a statistical sufficiency claim.

## Inference and artifacts

```python
from ml.taxonomy.inference import predict_taxonomy, TaxonomyPredictor

prediction = predict_taxonomy(feature_row)
# For repeated calls, load once:
predictor = TaxonomyPredictor('models/taxonomy')
prediction = predictor.predict(feature_row)
```

`best_model.joblib` is a bundle containing the selected category Pipeline, label
encoder, category-specific subtype Pipelines, feature contract, and metadata.
The preprocessing and label encoder are also saved separately for inspection.
`label_mapping.json` and `metadata.json` describe class order, training IDs,
parameters, input hashes, software versions, absent classes and subtype coverage.
The active subtype models in the bundle/metadata are authoritative. Load trusted
local joblib files only.

Inference routes through the predicted category. A missing subtype model returns
null subtype/confidence. Confidence is the model's raw maximum predict_proba
value, not a calibrated probability of correctness. No manual normalization,
calibration, SHAP, or diagnosis orchestration is included.

## Metrics and interpretation

Category metrics cover the union of trained categories and true evaluation
categories. All six are represented in the current train/validation/test splits.
Category reports include Accuracy, Macro Precision/Recall/F1, Weighted F1 and
per-class support. The saved confusion matrix has true labels on rows.

Subtype reports separate **oracle routing** (given the true category) from
**predicted routing** (end-to-end). Unavailable predictions count as incorrect.
Subtype macro scores explicitly include all declared subtypes (20 globally or
the declared subtypes within each category); zero-support labels contribute zero.
This denominator is recorded in each report. Weighted scores use actual support.
Hierarchical accuracy requires both category and subtype to be correct.

The current dataset has only 100 synthetic faulty examples. No clean records are
excluded from this particular run because none exist in its feature CSV. All six
categories and 20 subtypes are present in training, but SECURITY subtype training
is skipped because SEC_PROMPT_INJECTION has only one training example.

The audit detects potential single-feature category proxies in repeated training
values and checks repeated predictor vectors across training/test after model
selection. Actual results show 13/15 test vectors also occur in training despite
different task IDs. The selected model heavily relies on payload lengths. These
are observable features, not explicit labels, but likely encode synthetic
injection/template patterns. Scores are not evidence of generalization to new
tasks or real failures. The existing grouped split is preserved as requested.

## Outputs

`models/taxonomy/`: model bundle, preprocessing, label encoder, mapping, metadata,
and one standalone pipeline per available subtype category.

`results/taxonomy/`: validation comparison/trials, category report, confusion
matrix PNG/CSV, test predictions, subtype report, feature importances, proxy
audit, and `review3_summary.json`.

The focused feature/taxonomy tests pass. Full discovery also reaches two existing
tests importing the missing `injector.generator.TrajectoryGenerator`; those
pre-existing errors are outside this classifier implementation.
