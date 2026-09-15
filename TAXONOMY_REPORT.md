# AgentFault Taxonomy Classifier: Review 3 report

Implemented in the existing repository using the feature-engineering column contract and preprocessing. Existing feature, trajectory, injector, dataset, and split files were not modified. No detector, localizer, calibration, SHAP, or diagnosis agent was implemented.

## Commands

```powershell
python -m pip install -r requirements-taxonomy.txt
python scripts/train_taxonomy_classifier.py
python scripts/evaluate_taxonomy_classifier.py
python scripts/predict_taxonomy.py --trajectory-id AF-0029_fault_KNOW_CITATION_MISMATCH_step_7
python scripts/predict_taxonomy.py --features path/to/row.csv
python -m unittest tests.test_taxonomy_classifier tests.test_feature_engineering -v
```

## Dataset and class distribution

Usable failure-labelled samples: **100**. Excluded clean/unlabelled rows: **0**. Existing split: **70 train / 15 validation / 15 test**. All six categories are represented; all 20 subtypes occur in training.

| Category | Train | Validation | Test |
|---|---:|---:|---:|
| PLANNING | 13 | 1 | 1 |
| TOOL | 14 | 3 | 3 |
| KNOWLEDGE | 9 | 4 | 2 |
| MULTI_AGENT | 14 | 3 | 3 |
| CONTROL | 9 | 1 | 5 |
| SECURITY | 11 | 3 | 1 |

## Validation selection

| Model | Validation Macro-F1 |
|---|---:|
| Logistic Regression | 0.753968 |
| Random Forest | 0.777778 |
| XGBoost | 0.777778 |

Selected **Random Forest**, using validation Macro-F1 only. Random Forest and XGBoost tied; the fixed candidate-order tie rule retains Random Forest. No hyperparameters were changed after observing test results. Standalone evaluation reproduces the frozen model.

Best parameters: `{"n_estimators": 250, "max_depth": null, "min_samples_split": 4, "class_weight": "balanced"}`.

## Final held-out category metrics

Accuracy: **0.866667**; Macro-F1: **0.758333**; Weighted-F1: **0.843333**.
Macro Precision: **0.766667**; Macro Recall: **0.777778**.

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| CONTROL | 1.0000 | 1.0000 | 1.0000 | 5 |
| KNOWLEDGE | 1.0000 | 1.0000 | 1.0000 | 2 |
| MULTI_AGENT | 0.6000 | 1.0000 | 0.7500 | 3 |
| PLANNING | 1.0000 | 1.0000 | 1.0000 | 1 |
| SECURITY | 0.0000 | 0.0000 | 0.0000 | 1 |
| TOOL | 1.0000 | 0.6667 | 0.8000 | 3 |

Confusion matrix: `results/taxonomy/category_confusion_matrix.png` (also saved as CSV).

## Subtype and hierarchical results

Trained models: CONTROL, KNOWLEDGE, MULTI_AGENT, PLANNING, TOOL.
SECURITY subtype model is unavailable: training has only one SEC_PROMPT_INJECTION sample. The rule requires at least six rows, two observed subtypes, and two samples in each observed subtype. Category-level SECURITY prediction remains supported.

Hierarchical accuracy (category AND subtype correct): **0.666667**.
Routed subtype accuracy: **0.666667**; Macro-F1: **0.375000**; Weighted-F1: **0.700000**.
Routed subtype prediction coverage: **1.000000**. Unavailable predictions count as incorrect.

Subtype Macro-F1 includes all 20 declared subtypes; zero-support subtypes contribute zero. Within-category oracle metrics use all declared subtypes in that category. This explains lower macro values even when a tiny supported subset is classified correctly.

### Oracle-category subtype metrics

These assume the true category is supplied and are not end-to-end performance.

| Category | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| PLANNING | 1.0000 | 0.3333 | 1.0000 |
| TOOL | 1.0000 | 0.7500 | 1.0000 |
| KNOWLEDGE | 1.0000 | 0.6667 | 1.0000 |
| MULTI_AGENT | 0.6667 | 0.2500 | 0.6667 |
| CONTROL | 0.6000 | 0.5000 | 0.7000 |

### Routed per-subtype report

| Subtype | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| CTRL_EXCESSIVE_EXPLORATION | 0.0000 | 0.0000 | 0.0000 | 0 |
| CTRL_LOOP | 1.0000 | 0.3333 | 0.5000 | 3 |
| CTRL_PREMATURE_TERMINATION | 1.0000 | 1.0000 | 1.0000 | 2 |
| KNOW_CITATION_MISMATCH | 1.0000 | 1.0000 | 1.0000 | 1 |
| KNOW_CONTEXT_TRUNCATION | 1.0000 | 1.0000 | 1.0000 | 1 |
| KNOW_RETRIEVAL_FAILURE | 0.0000 | 0.0000 | 0.0000 | 0 |
| MA_INCORRECT_HANDOFF | 0.0000 | 0.0000 | 0.0000 | 0 |
| MA_INFORMATION_LOSS | 1.0000 | 1.0000 | 1.0000 | 2 |
| MA_MISSING_RESPONSIBILITY | 0.0000 | 0.0000 | 0.0000 | 0 |
| MA_ROLE_OVERLAP | 0.0000 | 0.0000 | 0.0000 | 1 |
| PLAN_INCORRECT_SUCCESS_CRITERIA | 1.0000 | 1.0000 | 1.0000 | 1 |
| PLAN_INVALID_DEPENDENCY | 0.0000 | 0.0000 | 0.0000 | 0 |
| PLAN_MISSING_STEP | 0.0000 | 0.0000 | 0.0000 | 0 |
| SEC_CROSS_USER_DATA_LEAKAGE | 0.0000 | 0.0000 | 0.0000 | 0 |
| SEC_PROMPT_INJECTION | 0.0000 | 0.0000 | 0.0000 | 1 |
| SEC_UNAUTHORIZED_ACTION | 0.0000 | 0.0000 | 0.0000 | 0 |
| TOOL_FAILED_RECOVERY | 1.0000 | 1.0000 | 1.0000 | 1 |
| TOOL_UNNECESSARY_CALL | 0.0000 | 0.0000 | 0.0000 | 0 |
| TOOL_WRONG_ARGUMENT | 1.0000 | 1.0000 | 1.0000 | 1 |
| TOOL_WRONG_TOOL | 0.0000 | 0.0000 | 0.0000 | 1 |

## Top feature importances

| Feature | Importance |
|---|---:|
| numeric__average_step_output_length | 0.143680 |
| numeric__total_output_chars | 0.136062 |
| numeric__average_step_input_length | 0.094968 |
| numeric__total_input_chars | 0.082478 |
| numeric__avg_consecutive_step_similarity | 0.035352 |
| numeric__semantic_drift_slope | 0.031990 |
| numeric__total_input_words | 0.031313 |
| numeric__total_output_words | 0.027747 |
| numeric__total_steps | 0.025868 |
| numeric__avg_task_step_similarity | 0.023100 |
| numeric__handoffs_per_step | 0.020014 |
| numeric__first_error_step_position | 0.018204 |
| numeric__max_step_output_length | 0.017711 |
| numeric__consecutive_identical_action_max | 0.016519 |
| numeric__avg_semantic_drift | 0.015766 |

## Leakage controls and limitations

Only the 77 declared predictive columns enter model pipelines. IDs, taxonomy labels, outcome, origin, source, split, and injection provenance are excluded. Tests verify label mutation does not change model inputs/probabilities and preprocessing is fitted on train only. Group assignments are preserved and checked.

**This does not eliminate synthetic data overlap.** The audit found:

- Small synthetic dataset: lengths, missingness, repeated actions and semantic scores can reveal injection patterns; metrics are prototype-only. Probabilities are uncalibrated.
- Payload-size features account for 45.7% of selected-model importance. These observable sizes can encode synthetic injection signatures; inspect source feature definitions before generalizing.
- 13/15 test rows have exactly the same predictor vector as a training row despite distinct task IDs. Synthetic template reuse limits generalization claims.

Top size features were traced to provenance-filtered observed payload lengths in feature_engineering/common.py and step_features.py; no filename, origin, operator, or fault label enters these values. Nevertheless, changed payload sizes expose injection patterns. The existing grouped split was preserved as requested. A genuinely independent task/template dataset is needed before claiming real-world performance.

## Real held-out prediction

```json
{
  "trajectory_id": "AF-0029_fault_KNOW_CITATION_MISMATCH_step_7",
  "predicted_category": "KNOWLEDGE",
  "category_confidence": 0.9554033821359191,
  "category_probabilities": {
    "CONTROL": 0.001,
    "KNOWLEDGE": 0.9554033821359191,
    "MULTI_AGENT": 0.002220183486238532,
    "PLANNING": 0.004360316494217894,
    "SECURITY": 0.021756943571697874,
    "TOOL": 0.015259174311926605
  },
  "predicted_subtype": "KNOW_CITATION_MISMATCH",
  "subtype_confidence": 0.9666666666666667
}
```
Actual category/subtype: KNOWLEDGE / KNOW_CITATION_MISMATCH. Both correct; labels were used only for evaluation.

## Files created

### Source and tests

- `ml/__init__.py`
- `ml/taxonomy/__init__.py`
- `ml/taxonomy/data.py`
- `ml/taxonomy/evaluate.py`
- `ml/taxonomy/inference.py`
- `ml/taxonomy/README.md`
- `ml/taxonomy/train.py`
- `ml/taxonomy/utils.py`
- `requirements-taxonomy.txt`
- `scripts/evaluate_taxonomy_classifier.py`
- `scripts/predict_taxonomy.py`
- `scripts/train_taxonomy_classifier.py`
- `TAXONOMY_REPORT.md`
- `tests/test_taxonomy_classifier.py`

### Saved model paths

- `models/taxonomy/best_model.joblib`
- `models/taxonomy/label_encoder.joblib`
- `models/taxonomy/label_mapping.json`
- `models/taxonomy/metadata.json`
- `models/taxonomy/preprocessing.joblib`
- `models/taxonomy/subtype_control.joblib`
- `models/taxonomy/subtype_knowledge.joblib`
- `models/taxonomy/subtype_multi_agent.joblib`
- `models/taxonomy/subtype_planning.joblib`
- `models/taxonomy/subtype_tool.joblib`

### Evaluation artifacts

- `results/taxonomy/category_classification_report.json`
- `results/taxonomy/category_confusion_matrix.csv`
- `results/taxonomy/category_confusion_matrix.png`
- `results/taxonomy/evaluation.log`
- `results/taxonomy/feature_importance.csv`
- `results/taxonomy/model_comparison.csv`
- `results/taxonomy/proxy_audit.json`
- `results/taxonomy/review3_summary.json`
- `results/taxonomy/subtype_classification_report.json`
- `results/taxonomy/test_predictions.csv`
- `results/taxonomy/validation_trials.json`

Modified pre-existing files: **none**. Model/result files above are newly generated artifacts. The model bundle includes the fitted category preprocessor/model, label encoder, five subtype pipelines, and metadata; separate preprocessing and encoder files are provided too.

## Test results

**6 taxonomy tests + 14 feature tests pass (20 total)**. Full discovery runs 22 entries and reports two pre-existing import errors: tests/test_injector.py and tests/test_pipeline.py import TrajectoryGenerator, which the existing injector.generator does not define. Those unrelated files were not changed.

Training, standalone evaluation, and CLI prediction were all run with the actual repository dataset. The confusion matrix was visually inspected. No tests assert a particular accuracy value.
