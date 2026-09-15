"""Frozen-model binary and trajectory-ranking evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report,
                             precision_recall_fscore_support, roc_auc_score)

from .data import DEFAULT_FEATURES, DEFAULT_MANIFEST, DEFAULT_METADATA, feature_frame, load_dataset
from .inference import RootCauseLocalizer
from .ranking import localization_metrics
from .utils import file_hash, save_json


def binary_metrics(labels, probabilities):
    labels = np.asarray(labels,dtype=int)
    predicted = np.asarray(probabilities)>=.5
    precision,recall,f1,_ = precision_recall_fscore_support(labels,predicted,average='binary',zero_division=0)
    both_classes = len(np.unique(labels))==2
    return {'precision':float(precision),'recall':float(recall),'f1':float(f1),
        'pr_auc':float(average_precision_score(labels,probabilities)) if (labels==1).any() else None,
        'roc_auc':float(roc_auc_score(labels,probabilities)) if both_classes else None,
        'threshold':.5,'pr_auc_definition':'Average precision (non-interpolated precision-recall integral)',
        'classification_report':classification_report(labels,predicted,labels=[0,1],output_dict=True,zero_division=0)}


def feature_importance(pipeline):
    classifier = pipeline.named_steps['classifier']
    importance = classifier.feature_importances_ if hasattr(classifier,'feature_importances_') else np.abs(classifier.coef_).mean(axis=0)
    names = pipeline.named_steps['preprocessing'].get_feature_names_out()
    return pd.DataFrame({'feature':names,'importance':importance}).sort_values('importance',ascending=False).reset_index(drop=True)


def evaluate(model_dir='models/localizer', results_dir='results/localizer', features=DEFAULT_FEATURES,
             manifest=DEFAULT_MANIFEST, trajectory_metadata=DEFAULT_METADATA):
    localizer = RootCauseLocalizer(model_dir)
    metadata = localizer.bundle['metadata']
    for name,path in [('step_features',features),('manifest',manifest),('trajectory_metadata',trajectory_metadata)]:
        if file_hash(path)!=metadata['input_sha256'][name]:
            raise ValueError(f'{name} differs from the recorded training inputs')
    rows,info = load_dataset(features,manifest,trajectory_metadata)
    test = rows.loc[rows['split']=='test'].copy()
    if test.empty:
        raise ValueError('No usable faulty test trajectories in existing split')
    if set(test['trajectory_id']) & set(metadata['training_trajectory_ids']):
        raise ValueError('Training/test trajectory overlap')
    probabilities = localizer.score_steps(test)
    metrics,steps,rankings = localization_metrics(test,probabilities)
    binary = binary_metrics(test['is_root_cause'],probabilities)
    output = Path(results_dir)
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'binary_classification_report.json',binary)
    save_json(output/'localization_metrics.json',metrics)
    steps.to_csv(output/'test_step_predictions.csv',index=False)
    rankings.to_csv(output/'test_trajectory_rankings.csv',index=False)
    importance = feature_importance(localizer.bundle['pipeline'])
    importance.to_csv(output/'feature_importance.csv',index=False)
    train = rows.loc[rows['split']=='train']
    training_hashes = set(pd.util.hash_pandas_object(feature_frame(train),index=False))
    repeated = pd.util.hash_pandas_object(feature_frame(test),index=False).isin(training_hashes)
    positive_overlap = int(repeated.to_numpy()[test['is_root_cause'].to_numpy()==1].sum())
    warnings = list(metadata['warnings'])
    if repeated.any():
        warnings.append(f'{int(repeated.sum())}/{len(test)} test step feature vectors occur in training, including {positive_overlap}/{int(test["is_root_cause"].sum())} positive steps. Distinct task IDs do not eliminate synthetic-template reuse.')
    suspicious_names = [name for name in importance.head(20)['feature'] if any(token in name.lower() for token in ['wrong_tool','wrong_agent','invalid_','failed_recovery'])]
    if suspicious_names:
        warnings.append('Top observable categorical features include synthetic injection sentinels: '+', '.join(suspicious_names)+'. These are runtime field values, not label columns, but may reveal injection patterns.')
    if metrics['exact_step_accuracy']>=.95:
        warnings.append('Near-perfect Rank-1 accuracy: inspect duplicate vectors and synthetic sentinels before making generalization claims. No labels, origin indices, or injection metadata were passed to the model.')
    summary = {'best_model':metadata['best_model'],'test_faulty_trajectories':metrics['faulty_trajectories'],
        **{k:v for k,v in metrics.items() if k!='faulty_trajectories'},
        'data_counts':info,'validation_comparison':metadata['validation_comparison'],
        'binary_metrics':{k:v for k,v in binary.items() if k!='classification_report'},
        'top_important_features':importance.head(15).to_dict(orient='records'),
        'test_steps_matching_train':int(repeated.sum()),'test_positive_steps_matching_train':positive_overlap,
        'warnings':warnings,'probabilities_calibrated':False}
    save_json(output/'review3_summary.json',summary)
    save_json(output/'leakage_audit.json',{'explicit_leakage_columns':[],
        'excluded_metadata':['trajectory_id','task_id','step_index','normalized_step_position','origin_step','fault_injected'],
        'predictor_count':len(localizer.bundle['feature_columns']),
        'test_steps_matching_train':int(repeated.sum()),'suspicious_top_feature_names':suspicious_names,'warnings':warnings})
    # Deterministic real example: first test trajectory sorted by ID, not selected for success.
    example = rankings.iloc[0].to_dict()
    selected = test.loc[test['trajectory_id']==example['trajectory_id']]
    interpretation = steps.loc[steps['trajectory_id']==example['trajectory_id']].merge(
        selected[['trajectory_id','step_index','step_type','agent_id','tool_name']],on=['trajectory_id','step_index'],validate='one_to_one')
    example['ranked_steps'] = interpretation.sort_values('predicted_rank').to_dict(orient='records')
    save_json(output/'example_localization.json',example)
    print('\n=== Review 3 Root-Cause Localization Summary ===')
    for key in ['best_model','test_faulty_trajectories','exact_step_accuracy','top3_accuracy','mrr','mean_absolute_step_distance','top5_accuracy']:
        print(f'{key}: {summary[key]}')
    print('Top important features:\n'+importance.head(15).to_string(index=False))
    for warning in warnings:
        print('WARNING:',warning)
    print(f'\nTrajectory: {example["trajectory_id"]}\nActual root-cause step: {example["actual_root_cause_step"]}')
    for row in example['ranked_steps']:
        print(f'Step {row["step_index"]}: {row["root_cause_probability"]:.6f} | {row["step_type"]} | {row["agent_id"]} | {row["tool_name"]}')
    print('Exact prediction:', 'CORRECT' if example['exact_match'] else 'INCORRECT')
    return summary
