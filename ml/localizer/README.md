# AgentFault Root-Cause Step Localizer

Binary step classification followed by ranking within a complete faulty
trajectory. This package is independent of the taxonomy classifier and LangGraph.
It does not detect whether a trajectory failed, calibrate probabilities, or
implement early prediction, neural models, SHAP, or diagnosis orchestration.

## Run

From the existing repository root:

```powershell
python -m pip install -r requirements-localizer.txt
python scripts/train_root_cause_localizer.py
python scripts/evaluate_root_cause_localizer.py
python scripts/predict_root_cause.py --trajectory-id AF-0004_fault_TOOL_WRONG_TOOL_step_5
python -m unittest tests.test_root_cause_localizer tests.test_feature_engineering tests.test_taxonomy_classifier -v
```

Training/evaluation accept `--features`, `--manifest`, `--trajectory-metadata`,
`--model-dir`, and `--results-dir`. Training also accepts `--random-state` (42).
`--embedded-split` uses existing step-row splits; `--embedded-fault-metadata` uses
existing step-row fault flags instead of reading the separate metadata table.
Neither option generates assignments or labels. Prediction accepts `--features`
and `--model-dir` plus the required trajectory ID.

## Verified existing schema

- Primary input: `data/features/step_features.csv`, one row per recorded step.
- IDs and ranking metadata: `trajectory_id`, `task_id`, `step_index`.
- Target: existing binary `is_root_cause`.
- Predictors: exactly the 28 entries in the existing step `FEATURE_COLUMNS`.
  These include `step_type`, `agent_id`, `tool_name`, status, payload sizes,
  observed errors, cumulative counts, and semantic/prefix similarities.
- Split: `data/features/split_manifest.csv`, joined many-to-one by trajectory ID.
- Fault eligibility/origin metadata: `fault_injected` and `origin_step` are absent
  from the step CSV. Only these fields and task/trajectory IDs are read from
  `trajectory_features.csv`; no trajectory-level predictors enter the model.
- `normalized_step_position` is retrospective metadata and excluded because it
  uses final trajectory length. Raw `step_index` is also excluded from scoring.

The loader checks the existing feature dictionary when available. All steps of
a trajectory must share task identity and split. All related tasks/base tasks
must stay in one split; conflicts cause an error before filtering or training.

## Labels and invalid data

Only injected-fault trajectories with exactly one positive label are trained.
Clean trajectories remain excluded; positive markers on clean trajectories are
invalid. Missing/zero-positive faults, multiple positives, missing or duplicate
step IDs, invalid predictors, and inconsistent origins are reported and skipped.
No labels are repaired or invented. Indices may have gaps and are not renumbered.

When origin metadata is known, its consistency is checked using the existing
injector rule: if the original step was removed, the next surviving step (or the
last surviving step) carries the recorded root label. This checks an existing
label; it never derives a new positive. Origin metadata never reaches scoring.

Clean/faulty counts describe fault flags; invalid and missing-label diagnostic
counts can overlap for malformed records. A zero-positive faulty trajectory is
reported as missing-root, not a clean example. Schema/split failures are fatal
because grouped evaluation cannot safely proceed without those contracts.

## Model selection and preprocessing

Seven modest candidates are compared:

- Logistic Regression: C .1, 1, 10; balanced class weights; standardized numerics.
- Random Forest: 150 trees/depth 6/min leaf 2, or 250 trees/unlimited depth/min
  leaf 1; balanced class weights.
- XGBoost: 100 trees/depth 2/learning rate .05, or 150 trees/depth 3/rate .1;
  subsample and column sampling .9. `scale_pos_weight` is training negatives
  divided by training positives. Missing XGBoost is reported and skipped.

Each candidate gets its own existing `build_preprocessor('step')`: median
imputation, one-hot categorical encoding with unknown-category support, and
optional scaling. It fits on training steps only. All-missing training numeric
columns use the existing sklearn `keep_empty_features` fallback at the encoded
stage; raw CSVs remain unchanged.

Selection maximizes validation MRR, then validation exact-step accuracy. Remaining
ties keep the earlier candidate. Validation/test are never included in fitting;
the selected pipeline is not refitted. The test set is scored only after selection
and artifact saving. Standalone evaluation reproduces the frozen model and checks
input SHA256 hashes; it does not tune any setting.

## Ranking and metrics

Probabilities are raw binary `predict_proba` outputs for class 1. They are not
calibrated and do not sum to one across trajectory steps. The highest-probability
step is the predicted root. Ties use smaller step index; labels are never consulted.

- Exact-step accuracy: true root has rank 1.
- Top-3 accuracy: true root has rank at most 3 (all available steps if fewer).
- MRR: mean of 1 / true-root rank.
- Mean absolute step distance: mean absolute difference of the actual recorded
  predicted/root step indices, preserving gaps.
- Top-5 accuracy: only trajectories with at least five steps; eligible count is
  saved, and the metric is null if none qualify.
- Secondary binary metrics: precision/recall/F1 at fixed threshold .5, PR-AUC
  measured as average precision, and ROC-AUC. The threshold is not tuned.

Step scoring consumes only local and prefix-safe predictors. The semantic model
is pretrained and encodes each text independently; previous-step similarity uses
only earlier meaningful steps. Ranking considers the complete supplied set of
steps. This complete-trajectory ranker is not an early-failure predictor.

## Reusable inference

```python
from ml.localizer.inference import predict_root_cause, RootCauseLocalizer

result = predict_root_cause(step_feature_rows)
# Reuse a loaded model for repeated calls:
localizer = RootCauseLocalizer('models/localizer')
result = localizer.predict(step_feature_rows)
```

Provide one nonempty trajectory with its 28 predictors and IDs/step indices. Labels
are not required. Output contains predicted root step, its probability, and all
steps ranked by probability (therefore at least the top three when available).
Ground-truth labels are never returned by normal inference. A supplied clean
trajectory would still be ranked: eligibility/detection belongs to the caller.

The saved bundle contains the fitted pipeline, predictor contract, and metadata.
A separate fitted preprocessor is also saved. Load only trusted joblib files.
The package does not import or load the taxonomy classifier; future predicted
taxonomy context would require an explicit new feature contract.

## Actual prototype limits

The current data contains 100 valid faulty trajectories, 100 positive and 715
negative steps. There are no clean examples or invalid roots in this input.
The existing split gives 572/119/124 train/validation/test steps in 70/15/15
trajectories. Training XGBoost positive weight is 502/70 = 7.1714285714.

The saved selected model is XGBoost. On 15 held-out trajectories it reaches .60
exact-step accuracy, .9333 Top-3, .7333 MRR, and .8 mean absolute step distance.
However, 120 of 124 test step vectors (including all 15 positive ones) also occur
in training. Observable categorical values `wrong_agent` and `wrong_tool` appear
among important features. These originate from runtime fields changed by the
injector, not provenance/label columns, but reveal synthetic injection patterns.
Error status is likewise an observable field; direct injected metadata is absent
from predictors. These results do not establish generalization to new tasks.

## Files

`data.py`: loading, eligibility and grouped-split validation.
`utils.py`: leakage contract, training-only weights and artifact helpers.
`ranking.py`: deterministic ranking and localization metrics.
`train.py`: seven-candidate search and model persistence.
`evaluate.py`: frozen-model binary/ranking reports, importance, overlap audit and
an example chosen by sorted test trajectory ID rather than by correctness.
`inference.py`: independent scoring/ranking API.

Artifacts are saved in `models/localizer/` and `results/localizer/`. The summary,
per-step predictions, trajectory rankings, feature importances and leakage audit
are intended for Review 3. Existing feature engineering and taxonomy files are
unchanged. Full test discovery still reaches two unrelated legacy tests importing
the missing `injector.generator.TrajectoryGenerator`.
