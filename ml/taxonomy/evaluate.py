"""Evaluate a frozen selected model; never choose hyperparameters here."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from .data import DEFAULT_FEATURES, DEFAULT_MANIFEST, feature_frame, load_dataset
from .inference import TaxonomyPredictor
from .utils import CATEGORY_SUBTYPES, SUBTYPE_TO_CATEGORY, file_hash, save_json


def metrics(actual, predicted, labels):
    if len(actual) == 0:
        return None
    precision, recall, f1, _ = precision_recall_fscore_support(actual,predicted,labels=labels,average='macro',zero_division=0)
    weighted = precision_recall_fscore_support(actual,predicted,labels=labels,average='weighted',zero_division=0)[2]
    return {'accuracy':float(accuracy_score(actual,predicted)), 'macro_precision':float(precision),
            'macro_recall':float(recall),'macro_f1':float(f1),'weighted_f1':float(weighted),
            'metric_labels':list(labels),
            'per_class':classification_report(actual,predicted,labels=labels,output_dict=True,zero_division=0)}


def feature_importance(bundle):
    pipeline = bundle['pipeline']
    classifier = pipeline.named_steps['classifier']
    values = classifier.feature_importances_ if hasattr(classifier,'feature_importances_') else np.abs(classifier.coef_).mean(axis=0)
    names = pipeline.named_steps['preprocessing'].get_feature_names_out()
    return pd.DataFrame({'feature':names,'importance':values}).sort_values('importance',ascending=False).reset_index(drop=True)


def evaluate(model_dir='models/taxonomy', results_dir='results/taxonomy', features=DEFAULT_FEATURES, manifest=DEFAULT_MANIFEST):
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictor = TaxonomyPredictor(model_dir)
    metadata = predictor.bundle['metadata']
    if file_hash(features) != metadata['feature_file_sha256']:
        raise ValueError('Evaluation features differ from the recorded training dataset')
    if manifest is not None and file_hash(manifest) != metadata['manifest_sha256']:
        raise ValueError('Evaluation manifest differs from the recorded training manifest')
    frame, info = load_dataset(features,manifest)
    test = frame.loc[frame['split']=='test'].copy()
    if not len(test):
        raise ValueError('No labelled held-out test rows')
    if set(test['trajectory_id']) & set(metadata['training_trajectory_ids']):
        raise ValueError('Training IDs overlap test IDs')
    predictions = predictor.predict_many(test)
    predicted_categories = [p['predicted_category'] for p in predictions]
    category_labels = sorted(set(metadata['classes_trained']) | set(test['fault_category']))
    category_metrics = metrics(test['fault_category'],predicted_categories,category_labels)
    save_json(output/'category_classification_report.json',category_metrics)
    matrix = confusion_matrix(test['fault_category'],predicted_categories,labels=category_labels)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(9,7))
    ConfusionMatrixDisplay(matrix,display_labels=category_labels).plot(ax=ax,cmap='Blues',colorbar=False,xticks_rotation=35)
    ax.set_title('AgentFault category classifier: held-out test set')
    fig.tight_layout()
    fig.savefig(output/'category_confusion_matrix.png',dpi=160)
    plt.close(fig)
    pd.DataFrame(matrix,index=category_labels,columns=category_labels).to_csv(output/'category_confusion_matrix.csv')
    rows = []
    for (_, actual), prediction in zip(test.iterrows(),predictions):
        rows.append({'trajectory_id':actual['trajectory_id'],'task_id':actual['task_id'],
            'actual_category':actual['fault_category'],'predicted_category':prediction['predicted_category'],
            'category_confidence':prediction['category_confidence'],
            'category_probabilities':json.dumps(prediction['category_probabilities']),
            'actual_subtype':actual['fault_subtype'],'predicted_subtype':prediction['predicted_subtype'],
            'subtype_confidence':prediction['subtype_confidence'],
            'category_correct':actual['fault_category']==prediction['predicted_category'],
            'subtype_correct':actual['fault_subtype']==prediction['predicted_subtype']})
    pd.DataFrame(rows).to_csv(output/'test_predictions.csv',index=False)
    oracle = {}
    for category, model in predictor.bundle['subtype_models'].items():
        subset = test.loc[test['fault_category']==category]
        labels = [s.value for s in CATEGORY_SUBTYPES[category]]
        oracle[category] = metrics(subset['fault_subtype'],model.predict(feature_frame(subset)),labels) if len(subset) else None
    routed_subtypes = [p['predicted_subtype'] or 'UNAVAILABLE' for p in predictions]
    subtype_metrics = {'oracle_category_metrics':oracle,
        'routed_metrics':metrics(test['fault_subtype'],routed_subtypes,sorted(SUBTYPE_TO_CATEGORY)),
        'prediction_coverage':sum(p['predicted_subtype'] is not None for p in predictions)/len(test),
        'hierarchical_accuracy':sum(r['category_correct'] and r['subtype_correct'] for r in rows)/len(rows),
        'unavailable_models':metadata['subtype_unavailable'],
        'note':'Oracle metrics assume the true category; routed metrics use the predicted category. Unavailable subtype predictions count as incorrect. Macro subtype metrics cover the 20 declared subtypes.'}
    save_json(output/'subtype_classification_report.json',subtype_metrics)
    importance = feature_importance(predictor.bundle)
    importance.to_csv(output/'feature_importance.csv',index=False)
    # Audit repeated synthetic feature vectors after freezing model selection.
    train_features = feature_frame(frame.loc[frame['split']=='train'])
    training_signatures = set(pd.util.hash_pandas_object(train_features,index=False).tolist())
    test_signatures = pd.util.hash_pandas_object(feature_frame(test),index=False)
    repeated = int(test_signatures.isin(training_signatures).sum())
    warnings = list(metadata['proxy_audit']['warnings'])
    size_features = {'numeric__average_step_output_length','numeric__total_output_chars',
                     'numeric__average_step_input_length','numeric__total_input_chars'}
    size_importance = float(importance.loc[importance['feature'].isin(size_features),'importance'].sum())
    if size_importance >= .3:
        warnings.append(f'Payload-size features account for {size_importance:.1%} of selected-model importance. These observable sizes can encode synthetic injection signatures; inspect source feature definitions before generalizing.')
    if repeated:
        warnings.append(f'{repeated}/{len(test)} test rows have exactly the same predictor vector as a training row despite distinct task IDs. Synthetic template reuse limits generalization claims.')
    if category_metrics['accuracy'] >= .95:
        warnings.append('Near-perfect test accuracy: review synthetic injection signatures and duplicate feature vectors; this is not evidence of real-world generalization.')
    summary = {**info,'classes_trained':metadata['classes_trained'],'categories_absent':metadata['categories_absent'],
        'best_category_model':metadata['best_model'],'selection_metric':'validation macro_f1',
        'validation_macro_f1':metadata['validation_macro_f1'],
        'test_accuracy':category_metrics['accuracy'],'test_macro_f1':category_metrics['macro_f1'],
        'test_weighted_f1':category_metrics['weighted_f1'],
        'subtype_models_trained':sorted(predictor.bundle['subtype_models']),
        'subtype_unavailable':metadata['subtype_unavailable'],
        'hierarchical_accuracy':subtype_metrics['hierarchical_accuracy'],
        'routed_subtype_macro_f1':subtype_metrics['routed_metrics']['macro_f1'],
        'routed_subtype_weighted_f1':subtype_metrics['routed_metrics']['weighted_f1'],
        'top_important_features':importance.head(15).to_dict(orient='records'),
        'test_vectors_also_in_train':repeated,'payload_size_importance':size_importance,'warnings':warnings,
        'probabilities_calibrated':False}
    save_json(output/'review3_summary.json',summary)
    save_json(output/'proxy_audit.json',{**metadata['proxy_audit'],
        'post_selection_test_vectors_also_in_train':repeated,
        'payload_size_importance':size_importance,'warnings':warnings})
    for warning in warnings:
        print('WARNING:',warning)
    print('\n=== Review 3 Taxonomy Classifier Summary ===')
    for name in ['usable_rows','split_counts','classes_trained','best_category_model','test_accuracy','test_macro_f1','test_weighted_f1','subtype_models_trained','hierarchical_accuracy']:
        print(f'{name}: {summary[name]}')
    print('Top important features:\n'+importance.head(15).to_string(index=False))
    return summary
