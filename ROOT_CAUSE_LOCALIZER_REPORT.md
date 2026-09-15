# AgentFault Root-Cause Localizer: Review 3 report

Implemented in the existing repository at `C:/Users/PRAKUL/OneDrive/ドキュメント/ChatGPT/agentfault`. Existing schema, feature engineering, taxonomy code, source datasets, and grouped split were not changed.

## Commands

```powershell
python -m pip install -r requirements-localizer.txt
python scripts/train_root_cause_localizer.py
python scripts/evaluate_root_cause_localizer.py
python scripts/predict_root_cause.py --trajectory-id AF-0004_fault_TOOL_WRONG_TOOL_step_5
python -m unittest tests.test_root_cause_localizer tests.test_feature_engineering tests.test_taxonomy_classifier -v
```

## Input and label validation

Primary training table: `data/features/step_features.csv`. Predictors are exactly the existing 28 step FEATURE_COLUMNS. Fault flags and origins are read from the trajectory table only as eligibility/label-validation metadata; no trajectory-level features are used.

Valid faulty trajectories: **100**. Positive root steps: **100**. Negative steps: **715**.
Clean trajectories: 0; missing root labels: 0; invalid trajectories: 0; multiple-positive trajectories: 0.

| Split | Trajectories | Steps | Positive | Negative |
|---|---:|---:|---:|---:|
| train | 70 | 572 | 70 | 502 |
| validation | 15 | 119 | 15 | 104 |
| test | 15 | 124 | 15 | 109 |

Training imbalance weighting: Logistic Regression and Random Forest use balanced class weights. XGBoost scale_pos_weight = 502 / 70 = 7.1714285714, computed exclusively from training steps.

## Validation comparison

| Model | Exact-step accuracy | Top-3 accuracy | MRR |
|---|---:|---:|---:|
| Logistic Regression | 0.466667 | 0.733333 | 0.628571 |
| Random Forest | 0.600000 | 0.666667 | 0.707778 |
| XGBoost | 0.600000 | 0.866667 | 0.724444 |

Selected **XGBoost** by highest validation MRR, with exact-step accuracy as the secondary criterion. Seven modest candidates were fitted. No settings were changed based on held-out results.
Best parameters: `{"n_estimators": 150, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9, "scale_pos_weight": 7.171428571428572}`.

## Final test ranking metrics

Test faulty trajectories: **15**.

| Metric | Value |
|---|---:|
| exact_step_accuracy | 0.600000 |
| top3_accuracy | 0.933333 |
| mrr | 0.733333 |
| mean_absolute_step_distance | 0.800000 |
| top5_accuracy | 0.933333 |

Top-5 eligible trajectories (at least five steps): 15. Ties favor the smaller recorded step index. Absolute distance uses recorded indices, including any gaps.

## Secondary binary metrics

| Metric | Value |
|---|---:|
| precision | 0.450000 |
| recall | 0.600000 |
| f1 | 0.514286 |
| pr_auc | 0.675337 |
| roc_auc | 0.876453 |

Binary threshold is fixed at .5. PR-AUC is average precision. Probabilities are raw model outputs, uncalibrated, and not normalized across steps.

## Top 15 feature importances

| Feature | Importance |
|---|---:|
| numeric__previous_step_same_type | 0.128381 |
| categorical__status_ERROR | 0.098316 |
| numeric__previous_step_same_agent | 0.073899 |
| numeric__has_error | 0.071910 |
| numeric__output_length_chars | 0.056970 |
| numeric__output_length_words | 0.049709 |
| categorical__agent_id_wrong_agent | 0.049312 |
| numeric__input_length_chars | 0.047967 |
| categorical__agent_id_researcher | 0.047479 |
| numeric__input_length_words | 0.046374 |
| categorical__tool_name_calculator | 0.041637 |
| numeric__previous_step_similarity | 0.041178 |
| numeric__cumulative_handoffs | 0.038548 |
| numeric__is_llm_step | 0.026248 |
| categorical__tool_name_wrong_tool | 0.024980 |

## Real held-out localization example

Trajectory: `AF-0004_fault_TOOL_WRONG_TOOL_step_5`. Actual root-cause step: **5**. Predicted step: **5**. Result: **CORRECT**.

This is the first held-out trajectory sorted by ID, rather than an example chosen for correctness.

| Rank | Step | Probability | Step type | Agent | Tool |
|---|---:|---:|---|---|---|
| 1 | 5 | 0.974188 | TOOL_CALL | researcher | wrong_tool |
| 2 | 2 | 0.453140 | HANDOFF | planner | UNKNOWN |
| 3 | 7 | 0.359917 | LLM_CALL | writer | UNKNOWN |
| 4 | 3 | 0.317008 | RETRIEVAL | researcher | UNKNOWN |
| 5 | 6 | 0.126741 | HANDOFF | researcher | UNKNOWN |
| 6 | 4 | 0.107335 | LLM_CALL | researcher | UNKNOWN |
| 7 | 1 | 0.020882 | LLM_CALL | planner | UNKNOWN |
| 8 | 8 | 0.003704 | LLM_CALL | writer | UNKNOWN |

Ground truth appears only in evaluation outputs. The reusable inference API and normal prediction CLI do not return or use root labels.

## Leakage and generalization audit

Explicit target/provenance columns are excluded by a hard feature contract. This includes is_root_cause, origin_step, fault types/categories, injection parameters, modified/original values, outcome, split, IDs, and normalized_step_position. Step indices are only ranking/output metadata.

Preprocessing fits only on training steps. All steps of a trajectory retain the original split, and task/base-task grouping is validated. Prefix-safe feature extraction and independent row scoring are checked; no future-context features were added. No taxonomy context enters this model.

- Small synthetic dataset; repeated templates, runtime sentinel values, payload sizes and missingness may encode injection patterns. Raw probabilities are uncalibrated and are not normalized across steps.
- Step predictors use only observed local/prefix features. Final ranking uses the complete supplied trajectory; this does not implement early failure prediction.
- 120/124 test step feature vectors occur in training, including 15/15 positive steps. Distinct task IDs do not eliminate synthetic-template reuse.
- Top observable categorical features include synthetic injection sentinels: categorical__agent_id_wrong_agent, categorical__tool_name_wrong_tool. These are runtime field values, not label columns, but may reveal injection patterns.

The source audit traced wrong_tool to injector/operators.py assigning StepRecord.tool_name, and wrong_agent to its assignment of StepRecord.agent_id. Error-related predictors use observed StepRecord.status/output.error through feature_engineering/common.py. These are actual runtime field values, not injected provenance dictionaries. Nevertheless, they expose synthetic injection patterns and limit external validity.

The existing manifest was deliberately preserved. Repeated step vectors under different task IDs mean the evaluation is not independent of template reuse. These results should not be presented as real-world generalization or calibrated causal certainty.

## Files created

### Source, tests, dependencies, documentation

- `ml/localizer/__init__.py`
- `ml/localizer/data.py`
- `ml/localizer/evaluate.py`
- `ml/localizer/inference.py`
- `ml/localizer/ranking.py`
- `ml/localizer/README.md`
- `ml/localizer/train.py`
- `ml/localizer/utils.py`
- `requirements-localizer.txt`
- `ROOT_CAUSE_LOCALIZER_REPORT.md`
- `scripts/evaluate_root_cause_localizer.py`
- `scripts/predict_root_cause.py`
- `scripts/train_root_cause_localizer.py`
- `tests/test_root_cause_localizer.py`

### Saved model paths

- `models/localizer/best_localizer.joblib`
- `models/localizer/model_metadata.json`
- `models/localizer/preprocessing.joblib`

### Evaluation outputs

- `results/localizer/binary_classification_report.json`
- `results/localizer/example_localization.json`
- `results/localizer/feature_importance.csv`
- `results/localizer/leakage_audit.json`
- `results/localizer/localization_metrics.json`
- `results/localizer/model_comparison.csv`
- `results/localizer/review3_summary.json`
- `results/localizer/test_step_predictions.csv`
- `results/localizer/test_trajectory_rankings.csv`
- `results/localizer/validation_trials.json`

Modified pre-existing files: **none**. The model bundle contains the selected fitted pipeline, feature contract and metadata; preprocessing is also saved separately.

## Tests and execution

**9 localizer tests, 14 feature tests, and 6 taxonomy tests pass (29 total).** Full discovery reports two pre-existing import errors: tests/test_injector.py and tests/test_pipeline.py import the missing injector.generator.TrajectoryGenerator. Those unrelated files were not changed.

Tests cover leakage exclusions, step/task/base-task split integrity, training-only imbalance weights and imputation, persistence, finite probabilities, descending/tie ranking, exact/Top-3/MRR/distance formulas, missing/multiple roots, malformed indices, clean-label rejection, and independence of earlier scores from later rows. Test fixtures are temporary and never enter the training dataset.

The training CLI, standalone evaluation CLI, and prediction CLI were all executed on the real repository dataset. Standalone evaluation reproduces a frozen model and does not retune against the test set.
