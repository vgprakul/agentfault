"""Small validation-selected category search and train-only subtype models."""
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from feature_engineering.preprocessing import build_preprocessor
from feature_engineering.trajectory_features import FEATURE_COLUMNS
from .data import DEFAULT_FEATURES, DEFAULT_MANIFEST, feature_frame, load_dataset
from .evaluate import evaluate, metrics
from .utils import CATEGORIES, CATEGORY_SUBTYPES, SUBTYPE_TO_CATEGORY, file_hash, save_json, validate_features


def candidates(seed=42):
    result = []
    for c in [.1,1.,10.]:
        result.append(('Logistic Regression',{'C':c},LogisticRegression(C=c,class_weight='balanced',max_iter=3000,random_state=seed),True))
    for trees,depth,minimum in [(150,5,2),(250,None,4)]:
        parameters = dict(n_estimators=trees,max_depth=depth,min_samples_split=minimum,class_weight='balanced')
        result.append(('Random Forest',parameters,RandomForestClassifier(**parameters,random_state=seed,n_jobs=1),False))
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print('XGBoost unavailable: skipping optional dependency.')
    else:
        for trees,depth,rate in [(100,2,.05),(150,3,.1)]:
            parameters = dict(n_estimators=trees,max_depth=depth,learning_rate=rate,subsample=.9,colsample_bytree=.9)
            result.append(('XGBoost',parameters,XGBClassifier(**parameters,eval_metric='mlogloss',random_state=seed,n_jobs=1,tree_method='hist'),False))
    return result


def proxy_audit(frame):
    """Flag training-only observable signatures; do not tune on test behaviour."""
    selected = feature_frame(frame)
    warnings = []
    suspicious = []
    labels = frame['fault_category'].reset_index(drop=True)
    for column in FEATURE_COLUMNS:
        values = selected[column].reset_index(drop=True).fillna('MISSING')
        if not 1 < values.nunique() <= max(2,len(frame)//2):
            continue
        counts = pd.crosstab(values,labels)
        repeated_groups = counts.loc[counts.sum(axis=1)>=2]
        covered = int(repeated_groups.to_numpy().sum())
        purity = float(repeated_groups.max(axis=1).sum()/covered) if covered else 0
        if covered >= .5*len(frame) and purity >= .9:
            suspicious.append({'feature':column,'repeated_value_category_purity':purity,'covered_training_rows':covered})
    if suspicious:
        warnings.append('Potential synthetic label proxies among observable features: '+', '.join(x['feature'] for x in suspicious)+'. These are not explicit labels, but may encode injection/template patterns.')
    signatures = pd.util.hash_pandas_object(selected,index=False).to_numpy()
    signature_labels = pd.DataFrame({'signature':signatures,'category':labels})
    conflicting = int(signature_labels.groupby('signature')['category'].nunique().gt(1).sum())
    if conflicting:
        warnings.append(f'{conflicting} identical training feature vectors map to multiple categories; current features cannot uniquely distinguish these examples.')
    warnings.append('Small synthetic dataset: lengths, missingness, repeated actions and semantic scores can reveal injection patterns; metrics are prototype-only. Probabilities are uncalibrated.')
    return {'explicit_leakage_columns':[], 'suspicious_observable_features':suspicious,
            'conflicting_training_feature_vectors':conflicting,
            'all_missing_training_features':selected.columns[selected.isna().all()].tolist(), 'warnings':warnings}


def train(features=DEFAULT_FEATURES, manifest=DEFAULT_MANIFEST, model_dir='models/taxonomy', results_dir='results/taxonomy', seed=42):
    validate_features()
    frame, info = load_dataset(features,manifest)
    train_rows = frame.loc[frame['split']=='train'].copy()
    validation = frame.loc[frame['split']=='validation'].copy()
    if len(train_rows)==0 or len(validation)==0:
        raise ValueError('Existing split must contain labelled training and validation rows; no automatic resplit')
    encoder = LabelEncoder().fit(train_rows['fault_category'])
    if len(encoder.classes_) < 2:
        raise ValueError('At least two observed training categories are required')
    print('AgentFault Taxonomy Classifier')
    print(f'Loaded trajectory feature rows: {info["loaded_rows"]}\nFailure-labelled trajectories: {len(frame)}\nExcluded clean/unlabelled: {info["excluded_unlabelled_or_clean"]}')
    print('Split counts:',info['split_counts'])
    print('Class distribution by split:',info['class_distribution'])
    classes = encoder.classes_.tolist()
    absent = sorted(set(CATEGORIES)-set(classes))
    print('Categories present in training:',classes,'\nCategories absent:',absent)
    audit = proxy_audit(train_rows)
    for warning in audit['warnings']:
        print('WARNING:',warning)
    x_train,x_validation = feature_frame(train_rows),feature_frame(validation)
    y_train = encoder.transform(train_rows['fault_category'])
    validation_labels = sorted(set(classes)|set(validation['fault_category']))
    trials, model_best, best = [], {}, None
    for name, parameters, classifier, scale in candidates(seed):
        print(f'Training {name}: {parameters}',flush=True)
        pipeline = Pipeline([('preprocessing',build_preprocessor('trajectory',standardize=scale)),('classifier',classifier)])
        fit_parameters = {'classifier__sample_weight':compute_sample_weight('balanced',y_train)} if name=='XGBoost' else {}
        pipeline.fit(x_train,y_train,**fit_parameters)
        predicted = encoder.inverse_transform(pipeline.predict(x_validation).astype(int))
        scores = metrics(validation['fault_category'],predicted,validation_labels)
        entry = {'model':name,'parameters':parameters,**scores}
        trials.append(entry)
        print(f'Validation Macro-F1: {scores["macro_f1"]:.6f}',flush=True)
        if name not in model_best or scores['macro_f1'] > model_best[name]['macro_f1']:
            model_best[name] = entry
        # Strict greater-than gives deterministic ties: earlier/simpler candidate wins.
        if best is None or scores['macro_f1'] > best[0]['macro_f1']:
            best = (entry,pipeline)
    model_path, output = Path(model_dir),Path(results_dir)
    model_path.mkdir(parents=True,exist_ok=True)
    output.mkdir(parents=True,exist_ok=True)
    comparison = [{k:v for k,v in entry.items() if k not in {'per_class','metric_labels','parameters'}} | {'parameters':str(entry['parameters'])} for entry in model_best.values()]
    pd.DataFrame(comparison).rename(columns={k:'validation_'+k for k in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1']}).to_csv(output/'model_comparison.csv',index=False)
    save_json(output/'validation_trials.json',trials)
    subtype_models, unavailable, subtype_info = {}, {}, {}
    for category in CATEGORIES:
        subset = train_rows.loc[train_rows['fault_category']==category]
        counts = subset['fault_subtype'].value_counts()
        subtype_info[category] = {'training_counts':counts.to_dict(),
            'absent_training_subtypes':sorted({s.value for s in CATEGORY_SUBTYPES[category]}-set(counts.index))}
        if len(subset)<6 or len(counts)<2 or counts.min()<2:
            unavailable[category] = 'Requires >=6 training rows, >=2 subtype classes and >=2 rows in every observed subtype'
            continue
        model = Pipeline([('preprocessing',build_preprocessor('trajectory')),
                          ('classifier',RandomForestClassifier(n_estimators=150,class_weight='balanced',random_state=seed,n_jobs=1))])
        model.fit(feature_frame(subset),subset['fault_subtype'])
        subtype_models[category] = model
        joblib.dump(model,model_path/f'subtype_{category.lower()}.joblib')
    versions = {}
    for package in ['scikit-learn','pandas','numpy','joblib','xgboost']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {**info,'created_utc':datetime.now(timezone.utc).isoformat(),'random_state':seed,
        'best_model':best[0]['model'],'best_parameters':best[0]['parameters'],
        'validation_macro_f1':{name:entry['macro_f1'] for name,entry in model_best.items()},
        'classes_trained':classes,'categories_absent':absent,'feature_columns':list(FEATURE_COLUMNS),
        'training_trajectory_ids':train_rows['trajectory_id'].tolist(),
        'validation_trajectory_ids':validation['trajectory_id'].tolist(),
        'feature_file_sha256':file_hash(features),'manifest_sha256':file_hash(manifest) if manifest else None,
        'subtype_training':subtype_info,'subtype_unavailable':unavailable,'proxy_audit':audit,
        'selection':'validation Macro-F1; ties retain earlier candidate; no refit on validation/test',
        'versions':versions,'probabilities_calibrated':False,
        'xgboost_available':any(x['model']=='XGBoost' for x in trials)}
    bundle = {'pipeline':best[1],'label_encoder':encoder,'subtype_models':subtype_models,
              'feature_columns':list(FEATURE_COLUMNS),'metadata':metadata}
    joblib.dump(bundle,model_path/'best_model.joblib')
    joblib.dump(best[1].named_steps['preprocessing'],model_path/'preprocessing.joblib')
    joblib.dump(encoder,model_path/'label_encoder.joblib')
    save_json(model_path/'label_mapping.json',{'category_classes':classes,'subtype_to_category':SUBTYPE_TO_CATEGORY})
    save_json(model_path/'metadata.json',metadata)
    save_json(output/'proxy_audit.json',audit)
    print(f'Best model: {best[0]["model"]}\nSaved model: {model_path/"best_model.joblib"}')
    # Only now, after all model selection and subtype fitting, access test predictions.
    return evaluate(model_dir,results_dir,features,manifest)
