# Complete-trajectory failure detection

The target comes exclusively from `outcome_status`. SUCCESS/PASS map to 0;
FAIL/FAILURE/ERROR/PARTIAL_SUCCESS map to 1. Unknown outcomes are reported and
excluded. Fault injection is not evidence of an unsuccessful outcome.

Run from the repository root:

```powershell
python scripts/train_failure_detector.py
python scripts/evaluate_failure_detector.py
python scripts/predict_failure.py --trajectory-id <id>
```

Training compares modest Logistic Regression, Random Forest and optional XGBoost
grids. Selection uses validation FAILURE F1 at threshold 0.5; ties retain the
earlier candidate. Preprocessing and weights use training rows only. The saved
joblib contains both preprocessing and classifier. Threshold analysis uses
validation only and never changes the default threshold automatically.

## Current data limitation

The primary feature table contains 100 FAIL outcomes and no SUCCESS outcomes.
Default training therefore stops with a data audit and blocked summary instead
of fitting a misleading one-class model. Existing real recordings contain three
SUCCESS outcomes, all in one training task. They do not supply successful
validation/test examples. Neither files nor split assignments are modified.

Explicit additional feature/manifest pairs can be supplied with
`--additional-features` and `--additional-manifest` (repeatable). A limited run
with a single-class validation/test split additionally requires
`--allow-single-class-evaluation`. Such results cannot establish binary
discrimination: ROC-AUC/PR-AUC and false-positive rate are unavailable on an
all-failure test set. Source differences between recordings and synthetic data
may act as proxies. Obtain representative successful and failed tasks in every
existing split before drawing performance conclusions.

The authorized limited prototype was trained with:

```powershell
python scripts/train_failure_detector.py --additional-features data/features/real/trajectory_features.csv --additional-manifest data/features/real/split_manifest.csv --allow-single-class-evaluation
```

Inference accepts only the existing declared feature columns; labels and IDs
are excluded. Model probabilities are uncalibrated. Joblib files must be trusted.
