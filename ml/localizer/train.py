"""Train binary step models; select by validation MRR then exact-step accuracy."""
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from feature_engineering.preprocessing import build_preprocessor
from feature_engineering.step_features import FEATURE_COLUMNS
from .data import DEFAULT_FEATURES, DEFAULT_MANIFEST, DEFAULT_METADATA, feature_frame, load_dataset
from .evaluate import binary_metrics, evaluate
from .inference import root_probabilities
from .ranking import localization_metrics
from .utils import file_hash, save_json, training_weights, validate_features


def candidates(scale_pos_weight, seed=42):
    result = []
    for c in [.1,1.,10.]:
        result.append(('Logistic Regression',{'C':c},LogisticRegression(C=c,class_weight='balanced',max_iter=3000,random_state=seed),True))
    for trees,depth,leaf in [(150,6,2),(250,None,1)]:
        parameters = dict(n_estimators=trees,max_depth=depth,min_samples_leaf=leaf,class_weight='balanced')
        result.append(('Random Forest',parameters,RandomForestClassifier(**parameters,random_state=seed,n_jobs=1),False))
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print('XGBoost unavailable; skipping optional model.')
    else:
        for trees,depth,rate in [(100,2,.05),(150,3,.1)]:
            parameters = dict(n_estimators=trees,max_depth=depth,learning_rate=rate,
                              subsample=.9,colsample_bytree=.9,scale_pos_weight=scale_pos_weight)
            result.append(('XGBoost',parameters,XGBClassifier(**parameters,objective='binary:logistic',eval_metric='logloss',tree_method='hist',random_state=seed,n_jobs=1),False))
    return result


def selection_key(metrics):
    return metrics['mrr'],metrics['exact_step_accuracy']


def train(features=DEFAULT_FEATURES, manifest=DEFAULT_MANIFEST, trajectory_metadata=DEFAULT_METADATA,
          model_dir='models/localizer', results_dir='results/localizer', seed=42):
    validate_features()
    rows,info = load_dataset(features,manifest,trajectory_metadata)
    print('AgentFault Root-Cause Localizer')
    for key in ['loaded_step_rows','faulty_trajectories','usable_faulty_trajectories','clean_trajectories',
                'missing_root_cause_labels','invalid_trajectories','multiple_positive_trajectories','positive_steps','negative_steps']:
        print(f'{key}: {info[key]}')
    for issue in info['issues']:
        print('SKIPPED:',issue)
    print('Existing split counts:',info['split_counts'])
    training = rows.loc[rows['split']=='train'].copy()
    validation = rows.loc[rows['split']=='validation'].copy()
    if training.empty or validation.empty:
        raise ValueError('Usable train and validation trajectories are required; no automatic resplit')
    weights = training_weights(training)
    print('Training-only class weights:',weights)
    x_train = feature_frame(training)
    trials, family_best, best = [], {}, None
    for name,parameters,classifier,scale in candidates(weights['scale_pos_weight'],seed):
        print(f'Training {name}: {parameters}',flush=True)
        pipeline = Pipeline([('preprocessing',build_preprocessor('step',standardize=scale)),('classifier',classifier)])
        pipeline.fit(x_train,training['is_root_cause'].astype(int))
        probabilities = root_probabilities(pipeline,validation)
        ranking,_,_ = localization_metrics(validation,probabilities)
        entry = {'model':name,'parameters':parameters,'ranking':ranking,
                 'binary':binary_metrics(validation['is_root_cause'],probabilities)}
        trials.append(entry)
        print(f'Validation MRR: {ranking["mrr"]:.6f}; Exact-Step: {ranking["exact_step_accuracy"]:.6f}; Top-3: {ranking["top3_accuracy"]:.6f}',flush=True)
        if name not in family_best or selection_key(ranking)>selection_key(family_best[name]['ranking']):
            family_best[name] = entry
        if best is None or selection_key(ranking)>selection_key(best[0]['ranking']):
            best = entry,pipeline
    output,models = Path(results_dir),Path(model_dir)
    output.mkdir(parents=True,exist_ok=True)
    models.mkdir(parents=True,exist_ok=True)
    comparison = [{'model':name,**{'validation_'+key:entry['ranking'][key] for key in ['exact_step_accuracy','top3_accuracy','mrr','mean_absolute_step_distance']},
                   'parameters':str(entry['parameters'])} for name,entry in family_best.items()]
    pd.DataFrame(comparison).to_csv(output/'model_comparison.csv',index=False)
    save_json(output/'validation_trials.json',trials)
    warnings = ['Small synthetic dataset; repeated templates, runtime sentinel values, payload sizes and missingness may encode injection patterns. Raw probabilities are uncalibrated and are not normalized across steps.',
                'Step predictors use only observed local/prefix features. Final ranking uses the complete supplied trajectory; this does not implement early failure prediction.']
    versions = {}
    for package in ['scikit-learn','pandas','numpy','joblib','xgboost']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {'created_utc':datetime.now(timezone.utc).isoformat(),'best_model':best[0]['model'],
        'best_parameters':best[0]['parameters'],'random_state':seed,'data_counts':info,
        'training_weights':weights,'validation_comparison':comparison,
        'selection_rule':'validation MRR, then exact-step accuracy; ties retain earlier candidate; no test-based tuning or refit',
        'training_trajectory_ids':training['trajectory_id'].unique().tolist(),
        'training_step_keys':training[['trajectory_id','step_index']].to_dict(orient='records'),
        'validation_trajectory_ids':validation['trajectory_id'].unique().tolist(),
        'feature_columns':list(FEATURE_COLUMNS),'all_missing_training_features':x_train.columns[x_train.isna().all()].tolist(),
        'input_sha256':{'step_features':file_hash(features),'manifest':file_hash(manifest),'trajectory_metadata':file_hash(trajectory_metadata)},
        'versions':versions,'warnings':warnings,'probabilities_calibrated':False}
    bundle = {'pipeline':best[1],'metadata':metadata,'feature_columns':list(FEATURE_COLUMNS)}
    joblib.dump(bundle,models/'best_localizer.joblib')
    joblib.dump(best[1].named_steps['preprocessing'],models/'preprocessing.joblib')
    save_json(models/'model_metadata.json',metadata)
    print(f'Best model: {best[0]["model"]}\nSaved: {models/"best_localizer.joblib"}')
    # Test scoring occurs only after model selection is frozen and artifacts saved.
    return evaluate(model_dir,results_dir,features,manifest,trajectory_metadata)
