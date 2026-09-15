"""Modest validation-selected binary models; never infer outcomes from faults."""
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from feature_engineering.preprocessing import build_preprocessor
from feature_engineering.trajectory_features import FEATURE_COLUMNS
from .data import DEFAULT_FEATURES, DEFAULT_MANIFEST, load_dataset, feature_frame
from .utils import save_json, validate_features, training_weights, OUTCOME_MAPPING
from .inference import probabilities
from .evaluate import metrics, evaluate


def candidates(seed, ratio):
    for c in [.1, 1., 10.]:
        yield 'Logistic Regression', {'C': c}, LogisticRegression(C=c, class_weight='balanced', max_iter=3000, random_state=seed), True
    for trees, depth, minimum in [(150, 5, 2), (250, None, 4)]:
        params=dict(n_estimators=trees, max_depth=depth, min_samples_split=minimum, class_weight='balanced')
        yield 'Random Forest', params, RandomForestClassifier(**params, random_state=seed, n_jobs=1), False
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print('XGBoost unavailable; optional candidate skipped.')
    else:
        for trees, depth, rate in [(100, 2, .05), (150, 3, .1)]:
            params=dict(n_estimators=trees, max_depth=depth, learning_rate=rate, subsample=.9, colsample_bytree=.9)
            yield 'XGBoost', params, XGBClassifier(**params, scale_pos_weight=ratio, eval_metric='logloss', random_state=seed, n_jobs=1), False


def train(features=DEFAULT_FEATURES, manifest=DEFAULT_MANIFEST, model_dir='models/failure_detector',
          results_dir='results/failure_detector', seed=42, additional_features=None,
          additional_manifests=None, allow_single_class_evaluation=False):
    validate_features()
    rows, info=load_dataset(features, manifest, additional_features, additional_manifests)
    output=Path(results_dir); output.mkdir(parents=True, exist_ok=True)
    save_json(output/'data_audit.json', info)
    save_json(output/'split_integrity.json', info['split_integrity'])
    print('=== AgentFault Failure Detector ===')
    print('Loaded trajectory rows:', info['loaded_rows'])
    print('SUCCESS samples:', info['success_samples'], 'FAILURE samples:', info['failure_samples'])
    for name, counts in info['split_counts'].items(): print(name, counts)
    training=rows.loc[rows['split']=='train']; validation=rows.loc[rows['split']=='validation']; test=rows.loc[rows['split']=='test']
    reason=None
    if training['failure_label'].nunique()!=2:
        reason='Training requires both actual SUCCESS and FAILURE outcomes. The current training split contains only one class; fault_injected cannot substitute for outcome labels.'
    elif validation.empty or test.empty:
        reason='Existing validation and test splits must contain labelled rows.'
    elif not allow_single_class_evaluation and (validation['failure_label'].nunique()!=2 or test['failure_label'].nunique()!=2):
        reason='Validation/test lack both outcomes. Add representative data, or explicitly use --allow-single-class-evaluation for a limited prototype with unavailable discrimination metrics.'
    if reason:
        save_json(output/'review3_summary.json', {'status':'blocked', 'reason':reason, 'best_model':None,
            'test_samples':len(test), 'accuracy':None, 'precision':None, 'recall':None, 'f1':None,
            'roc_auc':None, 'pr_auc':None, 'default_threshold':.5, 'data':info})
        raise ValueError(reason)
    weights=training_weights(training)
    warnings=['Small dataset; probabilities are uncalibrated. Inspect observable text lengths and error counts for synthetic/template proxies.']
    if validation['failure_label'].nunique()!=2 or test['failure_label'].nunique()!=2:
        warnings.append('Validation/test contain a single outcome. F1 cannot establish SUCCESS/FAILURE discrimination; ROC-AUC and PR-AUC are unavailable.')
    if additional_features:
        warnings.append('Multiple existing datasets combined without changing splits. Source/instrumentation differences may predict outcomes; inspect source confounding.')
    if training.loc[training['failure_label']==0,'task_id'].nunique()<2:
        warnings.append('SUCCESS training examples cover fewer than two tasks; generalization cannot be established.')
    print('Selection metric: validation FAILURE F1 at threshold 0.50; ties retain the earlier candidate.')
    fitted=[]; best=None; best_score=-1
    for name, parameters, estimator, scale in candidates(seed, weights['scale_pos_weight']):
        pipeline=Pipeline([('preprocessing', build_preprocessor('trajectory', standardize=scale)), ('classifier', estimator)])
        pipeline.fit(feature_frame(training), training['failure_label'])
        result=metrics(validation['failure_label'], probabilities(pipeline, validation))
        fitted.append({'model':name, 'parameters':str(parameters), 'validation_f1':result['f1'], 'validation_pr_auc':result['pr_auc'], 'validation_recall':result['recall']})
        print(name, parameters, 'Validation F1:', result['f1'], 'PR-AUC:', result['pr_auc'])
        if result['f1']>best_score:
            best_score=result['f1']; best=(name, parameters, pipeline, result)
    comparison=pd.DataFrame(fitted)
    comparison.to_csv(output/'candidate_results.csv', index=False)
    comparison.sort_values('validation_f1', ascending=False, kind='stable').drop_duplicates('model').to_csv(output/'model_comparison.csv', index=False)
    name, parameters, pipeline, validation_metrics=best
    metadata={'best_model':name, 'model_type':name, 'parameters':parameters, 'training_timestamp':datetime.now(timezone.utc).isoformat(),
        'feature_count':len(FEATURE_COLUMNS), 'feature_names':FEATURE_COLUMNS, 'class_mapping':{'SUCCESS':0,'FAILURE':1},
        'outcome_mapping':OUTCOME_MAPPING, 'validation_metrics':validation_metrics, 'selection_metric':'validation_failure_f1',
        'default_threshold':.5, 'random_seed':seed, 'inputs':info['inputs'], 'data':info, 'weights':weights,
        'training_trajectory_ids':training['trajectory_id'].tolist(), 'warnings':warnings}
    destination=Path(model_dir); destination.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline':pipeline, 'feature_columns':FEATURE_COLUMNS, 'metadata':metadata}, destination/'best_failure_detector.joblib')
    save_json(destination/'model_metadata.json', metadata)
    print('Best model:', name)
    return evaluate(destination, output)
