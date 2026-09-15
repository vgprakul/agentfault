"""Reuse existing ranking/binary metrics and add grouped analyses and baselines."""
import numpy as np
import pandas as pd
from ml.localizer.ranking import localization_metrics as existing_metrics
from ml.localizer.evaluate import binary_metrics
from ml.localizer.data import valid_step_indices


def evaluate_localization(predictions):
    required = {'trajectory_id','task_id','step_index','actual_is_root_cause','root_cause_probability'}
    if not required<=set(predictions) or predictions.empty:
        raise ValueError('Nonempty step predictions with IDs, targets and probabilities required')
    if predictions[['trajectory_id','task_id']].isna().any().any():
        raise ValueError('Missing trajectory/task IDs')
    accepted,skipped = [],[]
    for trajectory_id,group in predictions.groupby('trajectory_id'):
        if not valid_step_indices(group) or not group['actual_is_root_cause'].isin([0,1]).all() or group['actual_is_root_cause'].sum()!=1:
            skipped.append({'trajectory_id':trajectory_id,'reason':'Invalid step IDs or missing/multiple root positives'})
        else:
            accepted.append(group)
    if not accepted:
        raise ValueError('No valid single-root trajectories')
    valid = pd.concat(accepted,ignore_index=True)
    rows = valid.rename(columns={'actual_is_root_cause':'is_root_cause'})
    metrics,steps,rankings = existing_metrics(rows,rows['root_cause_probability'].to_numpy())
    metrics['median_absolute_step_distance'] = float(rankings['absolute_step_distance'].median())
    metrics['binary'] = binary_metrics(rows['is_root_cause'],rows['root_cause_probability'])
    metrics['skipped_trajectories'] = skipped
    return metrics,steps,rankings


def localization_baseline(rows, strategy, seed=42):
    """Predict using indices/randomness only; root labels are read only by metrics."""
    ordered = rows.sort_values(['trajectory_id','step_index']).copy()
    rng = np.random.default_rng(seed)
    values = []
    for _,group in ordered.groupby('trajectory_id',sort=True):
        n = len(group)
        if strategy=='random':
            # Uniform random permutation, giving each step equal chance of every rank.
            values.extend((rng.permutation(n)+1)/(n+1))
        elif strategy=='earliest':
            values.extend(np.arange(n,0,-1)/(n+1))
        else:
            raise ValueError('Unknown localization baseline')
    metrics,_,_ = existing_metrics(ordered,values)
    metrics['baseline_seed'] = seed if strategy=='random' else None
    return metrics


def summarize_group(group):
    return {'trajectories':len(group),'exact_step_accuracy':float(group['exact_match'].mean()),
            'top3_accuracy':float(group['in_top3'].mean()),'mrr':float(group['reciprocal_rank'].mean()),
            'mean_absolute_step_distance':float(group['absolute_step_distance'].mean())}


def breakdowns(rankings, category_metadata, training_steps):
    labels = category_metadata[['trajectory_id','actual_category']]
    if labels['trajectory_id'].duplicated().any():
        raise ValueError('Duplicate category evaluation metadata')
    joined = rankings.merge(labels,on='trajectory_id',how='left',validate='one_to_one')
    category_rows = [{'category':category,**summarize_group(group)} for category,group in joined.dropna(subset=['actual_category']).groupby('actual_category')]
    # Quantile boundaries derive from training lengths, never outcomes/test labels.
    lengths = training_steps.groupby('trajectory_id').size()
    if lengths.empty:
        raise ValueError('Training trajectory lengths required for length bins')
    low,high = np.quantile(lengths,[1/3,2/3])
    if low==high:
        # Repeated template lengths otherwise produce an empty middle quantile.
        center = float(lengths.median())
        bins = np.where(rankings['total_steps']<center,'short',np.where(rankings['total_steps']==center,'medium','long'))
        definition = f'short < {center:g}, medium = {center:g}, long > {center:g} steps (training median; tied quantiles)'
    else:
        bins = np.where(rankings['total_steps']<=low,'short',np.where(rankings['total_steps']<=high,'medium','long'))
        definition = f'short <= {low:g}, medium <= {high:g}, long > {high:g} steps (training terciles)'
    grouped = rankings.assign(length_group=bins)
    length_rows = [{'length_group':name,**summarize_group(group),'minimum_steps':int(group['total_steps'].min()),'maximum_steps':int(group['total_steps'].max())}
                   for name in ['short','medium','long'] for group in [grouped.loc[grouped['length_group']==name]] if len(group)]
    return pd.DataFrame(category_rows),pd.DataFrame(length_rows),{'definition':definition,'missing_category_trajectories':int(joined['actual_category'].isna().sum())}
