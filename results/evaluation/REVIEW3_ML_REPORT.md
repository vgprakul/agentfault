# Review 3: AgentFault ML Evaluation

## Dataset and split

100 failure-labelled trajectories; 815 valid faulty step rows. Existing grouped assignments preserved. Split integrity passed.
| split | trajectories | steps | positive_steps | negative_steps |
|---|---|---|---|---|
| train | 70 | 572 | 70 | 502 |
| validation | 15 | 119 | 15 | 104 |
| test | 15 | 124 | 15 | 109 |

## Taxonomy classifier

Best model: **Random Forest**. Test samples: 15. Accuracy 0.8667, Macro-F1 0.7583, Weighted-F1 0.8433.
| category | precision | recall | f1-score | support |
|---|---|---|---|---|
| PLANNING | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| TOOL | 1.0000 | 0.6667 | 0.8000 | 3.0000 |
| KNOWLEDGE | 1.0000 | 1.0000 | 1.0000 | 2.0000 |
| MULTI_AGENT | 0.6000 | 1.0000 | 0.7500 | 3.0000 |
| CONTROL | 1.0000 | 1.0000 | 1.0000 | 5.0000 |
| SECURITY | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

Subtype accuracy 0.6666666666666666; conditional on correct category 0.7692307692307693; hierarchical accuracy 0.6666666666666666.
Unavailable subtype models: {"SECURITY": "Requires >=6 training rows, >=2 subtype classes and >=2 rows in every observed subtype"}

Main category errors:
| trajectory_id | actual_category | predicted_category | confidence |
|---|---|---|---|
| AF-0018_fault_SEC_PROMPT_INJECTION_step_4 | SECURITY | MULTI_AGENT | 0.5660 |
| AF-0004_fault_TOOL_WRONG_TOOL_step_5 | TOOL | MULTI_AGENT | 0.5660 |

## Root-cause localizer

Best model: **XGBoost**. Test trajectories: 15. Exact-step 0.6000, Top-3 0.9333, Top-5 0.9333333333333333, MRR 0.7333, mean distance 0.8000, median distance 0.0000.
Secondary step metrics: PR-AUC 0.6753, recall 0.6000, F1 0.5143.

Main localization errors:
| trajectory_id | actual_root_cause_step | predicted_root_cause_step | actual_root_cause_rank | absolute_step_distance |
|---|---|---|---|---|
| AF-0014_fault_MA_ROLE_OVERLAP_step_2 | 2 | 5 | 2 | 3 |
| AF-0018_fault_SEC_PROMPT_INJECTION_step_4 | 4 | 5 | 6 | 1 |
| AF-0036_fault_CTRL_PREMATURE_TERMINATION_step_7 | 7 | 5 | 3 | 2 |
| AF-0055_fault_CTRL_LOOP_step_7 | 7 | 5 | 3 | 2 |
| AF-0076_fault_CTRL_PREMATURE_TERMINATION_step_7 | 7 | 5 | 3 | 2 |
| AF-0095_fault_CTRL_LOOP_step_7 | 7 | 5 | 3 | 2 |

### Performance by failure category
| category | trajectories | exact_step_accuracy | top3_accuracy | mrr | mean_absolute_step_distance |
|---|---|---|---|---|---|
| CONTROL | 5 | 0.2000 | 1.0000 | 0.4667 | 1.6000 |
| KNOWLEDGE | 2 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| MULTI_AGENT | 3 | 0.6667 | 1.0000 | 0.8333 | 1.0000 |
| PLANNING | 1 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| SECURITY | 1 | 0.0000 | 0.0000 | 0.1667 | 1.0000 |
| TOOL | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

### Performance by trajectory length
short < 8, medium = 8, long > 8 steps (training median; tied quantiles)
| length_group | trajectories | exact_step_accuracy | top3_accuracy | mrr | mean_absolute_step_distance | minimum_steps | maximum_steps |
|---|---|---|---|---|---|---|---|
| short | 2 | 0.0000 | 1.0000 | 0.3333 | 2.0000 | 7 | 7 |
| medium | 10 | 0.8000 | 0.9000 | 0.8667 | 0.4000 | 8 | 8 |
| long | 3 | 0.3333 | 1.0000 | 0.5556 | 1.3333 | 10 | 10 |

## Baselines
| module | model | accuracy | macro_f1 | exact_step_accuracy | top3_accuracy | top5_accuracy | mrr | mean_absolute_step_distance |
|---|---|---|---|---|---|---|---|---|
| taxonomy | Trained Random Forest | 0.8667 | 0.7583 |  |  |  |  |  |
| taxonomy | Majority: TOOL | 0.2000 | 0.0556 |  |  |  |  |  |
| localizer | Trained XGBoost |  |  | 0.6000 | 0.9333 | 0.9333 | 0.7333 | 0.8000 |
| localizer | Random step |  |  | 0.0000 | 0.2000 | 0.4000 | 0.2367 | 2.7333 |
| localizer | Earliest step |  |  | 0.0667 | 0.1333 | 0.4667 | 0.2494 | 4.4000 |

Majority label uses training counts only (taxonomy order resolves ties). Random-step is a uniform random ranking per trajectory with seed 42; earliest-step orders by recorded index. Baseline labels are used only for scoring.

## Candidate model comparison
| module | model | validation_macro_f1 | validation_mrr | validation_exact_step_accuracy | selected |
|---|---|---|---|---|---|
| taxonomy | Logistic Regression | 0.7540 |  |  | False |
| taxonomy | Random Forest | 0.7778 |  |  | True |
| taxonomy | XGBoost | 0.7778 |  |  | False |
| localizer | Logistic Regression |  | 0.6286 | 0.4667 | False |
| localizer | Random Forest |  | 0.7078 | 0.6000 | False |
| localizer | XGBoost |  | 0.7244 | 0.6000 | True |

Selection is from saved validation results only. No retraining or test-based model selection occurred.

## Limitations

Only 15 test trajectories per model; category and length subgroups are very small. No claims of causality or production readiness. Probabilities are uncalibrated. Subtype macro metrics use only actual subtypes present in this split, unlike the earlier all-20-subtype report. Top-5 uses only trajectories with at least five steps.

- taxonomy: 13/15 test predictor vectors also occur in training despite task grouping.
- taxonomy: Four payload-size features contribute 45.7% of importance and may expose injection/template patterns.
- localizer: 120/124 test predictor vectors also occur in training despite task grouping.
- localizer: Observable synthetic sentinel values among top features: categorical__agent_id_wrong_agent, categorical__tool_name_wrong_tool

Probability metric skips: {}
Machine-readable metrics, confidence buckets, error tables, feature audits and plots are in this directory.

## Implementation and files

New source files: ml/evaluation/__init__.py, taxonomy_metrics.py, localization_metrics.py, plots.py, model_comparison.py, integrity.py, review3_report.py, README.md; scripts/evaluate_ml.py; tests/test_ml_evaluation.py.
No pre-existing source files, trained models, feature datasets, or module-specific prediction files were modified.

Generated evaluation artifacts:
- baseline_comparison.csv
- evaluation_metadata.json
- leakage_audit.json
- localization_errors.csv
- localizer_by_category.csv
- localizer_by_length.csv
- localizer_metrics.json
- localizer_model_comparison.png
- localizer_rank_distribution.png
- localizer_step_distance_distribution.png
- model_comparison.csv
- review3_ml_summary.json
- split_integrity.json
- taxonomy_confidence_buckets.csv
- taxonomy_confusion_matrix.png
- taxonomy_confusion_matrix_normalized.png
- taxonomy_errors.csv
- taxonomy_metrics.json
- taxonomy_model_comparison.png
- taxonomy_per_class_f1.png
- REVIEW3_ML_REPORT.md