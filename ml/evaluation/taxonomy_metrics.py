"""Extend existing taxonomy metrics with observed-class and probability reporting."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score, average_precision_score
from ml.taxonomy.evaluate import metrics as existing_metrics
from ml.taxonomy.utils import CATEGORIES, SUBTYPE_TO_CATEGORY


def observed_order(values):
    observed = set(values)
    return [c for c in CATEGORIES if c in observed] + sorted(observed-set(CATEGORIES))


def classification_metrics(actual, predicted):
    labels = observed_order(actual)
    result = existing_metrics(actual,predicted,labels)
    if result is None:
        return None
    weighted = result['per_class']['weighted avg']
    result.update(weighted_precision=float(weighted['precision']),weighted_recall=float(weighted['recall']))
    result['per_class'] = {label:result['per_class'][label] for label in labels}
    result['classes_evaluated'] = labels
    return result


def probability_metrics(frame):
    skipped = {}
    if 'category_probabilities' not in frame or frame['category_probabilities'].isna().all():
        return {'log_loss':None,'ovr_roc_auc':None,'ovr_pr_auc':None,'skipped':{'all':'No saved class probabilities'}},None
    try:
        maps = [json.loads(v) if isinstance(v,str) else v for v in frame['category_probabilities']]
        classes = sorted(maps[0])
        if not classes or any(not isinstance(m,dict) or set(m)!=set(classes) for m in maps):
            raise ValueError('Inconsistent probability classes')
        if set(classes)-set(CATEGORIES):
            raise ValueError('Unknown probability class')
        probabilities = np.array([[m[c] for c in classes] for m in maps],dtype=float)
    except (TypeError,KeyError,ValueError) as exc:
        raise ValueError(f'Malformed category probabilities: {exc}') from exc
    if not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any() or not np.allclose(probabilities.sum(axis=1),1,atol=1e-5):
        raise ValueError('Probabilities must be finite in [0,1] and sum to one')
    for i,predicted in enumerate(frame['predicted_category']):
        if predicted not in classes or not np.isclose(probabilities[i,classes.index(predicted)],probabilities[i].max(),atol=1e-5):
            raise ValueError('Predicted category is inconsistent with probability maximum')
    actual = frame['actual_category'].to_numpy()
    loss = None
    if set(actual)-set(classes):
        skipped['log_loss'] = 'True categories are absent from model probability support'
    elif len(classes)<2:
        skipped['log_loss'] = 'At least two model probability classes required'
    else:
        loss = float(log_loss(actual,probabilities,labels=classes))
    roc,pr = {},{}
    for index,category in enumerate(classes):
        target = (actual==category).astype(int)
        if target.min()==target.max():
            skipped[category] = 'One-vs-rest ROC/PR-AUC unavailable: no positive or no negative examples in evaluated split'
        else:
            roc[category] = float(roc_auc_score(target,probabilities[:,index]))
            pr[category] = float(average_precision_score(target,probabilities[:,index]))
    return {'log_loss':loss,'ovr_roc_auc':float(np.mean(list(roc.values()))) if roc else None,
        'ovr_pr_auc':float(np.mean(list(pr.values()))) if pr else None,
        'per_class_roc_auc':roc,'per_class_pr_auc':pr,'probability_classes':classes,
        'aggregation':'Macro mean over valid one-vs-rest classes only; PR-AUC is average precision',
        'skipped':skipped},probabilities.max(axis=1)


def confidence_analysis(frame, model_confidence=None):
    if 'category_confidence' not in frame and model_confidence is None:
        return {'unavailable_reason':'No confidence values'},pd.DataFrame(columns=['bucket','count','accuracy'])
    confidence = pd.to_numeric(frame['category_confidence'],errors='raise').to_numpy() if 'category_confidence' in frame else model_confidence
    if not np.isfinite(confidence).all() or ((confidence<0)|(confidence>1)).any():
        raise ValueError('Malformed confidence values')
    if model_confidence is not None and not np.allclose(confidence,model_confidence,atol=1e-5):
        raise ValueError('Saved confidence disagrees with probabilities')
    correct = frame['actual_category'].eq(frame['predicted_category']).to_numpy()
    buckets = []
    for low,high in zip([0,.2,.4,.6,.8],[.2,.4,.6,.8,1]):
        mask = (confidence>=low)&((confidence<high) if high<1 else (confidence<=high))
        buckets.append({'bucket':f'{low:.1f}-{high:.1f}','count':int(mask.sum()),
                        'accuracy':float(correct[mask].mean()) if mask.any() else None})
    return {'average_confidence_correct':float(confidence[correct].mean()) if correct.any() else None,
        'average_confidence_incorrect':float(confidence[~correct].mean()) if (~correct).any() else None,
        'buckets':buckets,'bucket_rule':'left inclusive, right exclusive; last bucket includes 1.0'},pd.DataFrame(buckets)


def evaluate_taxonomy(frame, unavailable_subtypes=None):
    required = {'trajectory_id','task_id','actual_category','predicted_category'}
    if not required<=set(frame) or frame.empty:
        raise ValueError('Nonempty taxonomy predictions with IDs and labels required')
    if frame['trajectory_id'].duplicated().any() or frame[list(required)].isna().any().any():
        raise ValueError('Missing labels/IDs or duplicate taxonomy trajectories')
    if (set(frame['actual_category'])|set(frame['predicted_category']))-set(CATEGORIES):
        raise ValueError('Unknown taxonomy category')
    result = classification_metrics(frame['actual_category'],frame['predicted_category'])
    # Include observed predicted-only classes on matrix axes so errors never disappear.
    matrix_labels = observed_order(set(frame['actual_category'])|set(frame['predicted_category']))
    result['confusion_matrix_labels'] = matrix_labels
    result['confusion_matrix'] = confusion_matrix(frame['actual_category'],frame['predicted_category'],labels=matrix_labels).tolist()
    result['samples'] = len(frame)
    result['probability_metrics'],confidence = probability_metrics(frame)
    result['confidence_analysis'],buckets = confidence_analysis(frame,confidence)
    result['subtype'] = {'unavailable_models':unavailable_subtypes or {}}
    if {'actual_subtype','predicted_subtype'}<=set(frame):
        valid = frame['actual_subtype'].notna()
        sub = frame.loc[valid].copy()
        if len(sub):
            if set(sub['actual_subtype'])-set(SUBTYPE_TO_CATEGORY) or set(sub['predicted_subtype'].dropna())-set(SUBTYPE_TO_CATEGORY):
                raise ValueError('Unknown subtype label')
            if not sub['actual_subtype'].map(SUBTYPE_TO_CATEGORY).eq(sub['actual_category']).all():
                raise ValueError('Actual subtype/category mismatch')
            available = sub['predicted_subtype'].notna()
            if not sub.loc[available,'predicted_subtype'].map(SUBTYPE_TO_CATEGORY).eq(sub.loc[available,'predicted_category']).all():
                raise ValueError('Predicted subtype/category mismatch')
            correct_category = sub['actual_category'].eq(sub['predicted_category'])
            correct_subtype = sub['actual_subtype'].eq(sub['predicted_subtype'])
            result['subtype'].update(classification_metrics(sub['actual_subtype'],sub['predicted_subtype'].fillna('UNAVAILABLE')))
            result['subtype'].update(evaluated_samples=len(sub),category_only_accuracy=result['accuracy'],
                conditional_on_correct_category_accuracy=float(correct_subtype[correct_category].mean()) if correct_category.any() else None,
                conditional_sample_count=int(correct_category.sum()),hierarchical_accuracy=float((correct_category&correct_subtype).mean()),
                prediction_coverage=float(available.mean()),
                denominator_note='Subtype macro metrics use only subtypes present in ground truth in this split; unavailable predictions count as incorrect')
        else:
            result['subtype']['unavailable_reason'] = 'No actual subtype labels'
    else:
        result['subtype']['unavailable_reason'] = 'Subtype columns unavailable'
    return result,buckets
