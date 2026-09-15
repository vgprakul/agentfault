# Unified AgentFault ML evaluation

Run from the existing repository root:

```powershell
python scripts/evaluate_ml.py
python -m unittest tests.test_ml_evaluation -v
```

Uses the already installed taxonomy/localizer dependencies (pandas, numpy,
scikit-learn, joblib, matplotlib, and XGBoost for the saved localizer). No new
dependencies are introduced. No models are trained or changed.

## Inputs and reuse

- Feature tables and manifest: `data/features/`.
- Saved model bundles: `models/taxonomy/best_model.joblib` and
  `models/localizer/best_localizer.joblib`.
- Prediction tables: `results/taxonomy/test_predictions.csv` and
  `results/localizer/test_step_predictions.csv`.
- Saved trajectory rankings: `results/localizer/test_trajectory_rankings.csv`.
- Validation comparisons: each existing result directory's `model_comparison.csv`.

Options: `--feature-dir`, `--model-dir`, `--results-dir`, `--output`, and
`--random-seed` (42 by default). Missing models produce an explicit training
command; there is no implicit retraining option.

The package reuses taxonomy classification metrics, localizer ranking/binary
metrics, model loaders, and feature-importance extraction. Existing prediction
tables are checked for exact test coverage, ground-truth agreement, and agreement
with fresh inference from the frozen models. Current dataset hashes must match
saved training metadata. Stale/malformed predictions fail clearly. If prediction
files are absent, frozen-model predictions are saved only in the evaluation
directory. Existing result directories are not overwritten.

## Metric conventions

- Category macro/per-class metrics use classes present in the evaluated ground
  truth. Confusion axes additionally include a predicted-only class if necessary
  to retain actual misclassification counts; unused classes are never added.
- Subtype macro/per-class metrics likewise use only observed actual subtypes.
  This differs from the older classifier report's denominator of all 20 subtypes.
  Null subtype predictions count as incorrect. Conditional subtype accuracy uses
  only samples with a correct category and known actual subtype. Full hierarchical
  accuracy requires both labels correct.
- Log loss uses the saved model's probability support in explicit class order.
  A true class outside that support makes log loss unavailable. One-vs-rest
  ROC/PR-AUC is skipped per class when positives or negatives are absent; macro
  aggregates average only valid classes. PR-AUC uses average precision. Missing
  probabilities are reported, never synthesized or renormalized.
- Confidence buckets are left inclusive/right exclusive, except the final bucket
  includes 1.0. Empty buckets have null accuracy. This is descriptive analysis,
  not calibration or ECE.
- Localizer metrics reuse existing ranks: decreasing probability, then increasing
  step index for ties. Only valid single-root trajectories qualify. Malformed
  root-label groups are reported/skipped by the loader/metric adapter.
- Top-5 uses trajectories with at least five steps, matching the current localizer
  convention. Eligible count is saved. Distances use recorded indices, preserving
  gaps. Binary threshold remains .5; accuracy is not promoted as localization's
  main metric.

## Baselines and grouping

The taxonomy majority class comes exclusively from training labels; ties follow
the existing taxonomy category order. The seeded random baseline creates a
uniform random ranking of steps independently of labels. Its scores are ordering
scores, not meaningful probabilities. Earliest-step uses ascending recorded step
index. Root labels are read only when scoring baselines. A single seeded random
run on 15 trajectories is noisy; do not treat it as an exact theoretical average.

Length bins come from training trajectory-length terciles. Tied terciles fall back
to below/equal/above training median. Current bins are short <8, medium =8, long >8
steps. Empty groups are omitted; no groups are fabricated. Actual taxonomy labels
join localizer results only for evaluation breakdowns and never enter inference.

## Integrity, audits, reproducibility

`split_integrity.json` is written before evaluation proceeds. It checks trajectory,
task, and available base-task assignments, embedded split consistency, identity
matches, and saved model training IDs. Serious violations stop evaluation.

Feature-name terms (fault/origin/root_cause/injected/label/target/subtype/category/
outcome) trigger inspection warnings rather than an automatic leakage verdict.
The audit also saves top importances and checks duplicate train/test predictor
vectors. Actual current overlap is 13/15 taxonomy vectors and 120/124 localizer
step vectors. Payload sizes and synthetic runtime sentinel values remain strong
proxy risks even though explicit target columns are excluded.

Evaluation metadata includes UTC timestamp, seed, input/model paths and SHA256s,
counts, evaluated classes, and Git commit when available. This repository has no
commit, so null Git metadata is valid. Model comparison reads existing validation
results only; missing comparison files are recorded without retraining.

## Outputs

All new outputs are in `results/evaluation/`: category/subtype/probability metrics,
localizer/binary metrics, confidence buckets, unified comparison and baselines,
error tables, category/length breakdowns, split integrity, leakage audit,
reproducibility metadata, Review 3 JSON/Markdown report, and seven PNG plots.

The current hold-out consists of 15 trajectories per model. Small subgroups,
absent subtypes, uncalibrated scores and repeated synthetic templates limit the
interpretation. The framework does not claim causality or production readiness.
