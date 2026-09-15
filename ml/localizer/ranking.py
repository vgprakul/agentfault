"""Deterministic within-trajectory ranks and localization metrics."""
import numpy as np
import pandas as pd
from .data import valid_step_indices


def rank_steps(rows, probabilities):
    probabilities = np.asarray(probabilities,dtype=float)
    if probabilities.shape!=(len(rows),) or not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any():
        raise ValueError('One finite probability in [0,1] is required per step')
    if rows.empty or rows[['trajectory_id','task_id']].isna().any().any():
        raise ValueError('Nonempty steps with trajectory/task IDs required')
    for _, group in rows.groupby('trajectory_id'):
        if not valid_step_indices(group) or group['task_id'].nunique()!=1:
            raise ValueError('Invalid/duplicate step IDs or inconsistent task identity')
    ranked = rows[['trajectory_id','task_id','step_index']].copy()
    ranked['step_index'] = pd.to_numeric(ranked['step_index']).astype(int)
    ranked['root_cause_probability'] = probabilities
    # No label-based tie-breaking. Equal probabilities favor smaller observed step ID.
    ranked = ranked.sort_values(['trajectory_id','root_cause_probability','step_index'],ascending=[True,False,True],kind='stable')
    ranked['predicted_rank'] = ranked.groupby('trajectory_id').cumcount()+1
    return ranked


def localization_metrics(rows, probabilities):
    ranked = rank_steps(rows,probabilities)
    labels = rows[['trajectory_id','step_index','is_root_cause']].copy()
    labels['step_index'] = pd.to_numeric(labels['step_index']).astype(int)
    ranked = ranked.merge(labels,on=['trajectory_id','step_index'],validate='one_to_one')
    trajectories = []
    for trajectory_id,group in ranked.groupby('trajectory_id',sort=True):
        if not group['is_root_cause'].isin([0,1]).all() or group['is_root_cause'].sum()!=1:
            raise ValueError('Ranking evaluation requires exactly one true root per trajectory')
        actual = group.loc[group['is_root_cause']==1].iloc[0]
        predicted = group.loc[group['predicted_rank']==1].iloc[0]
        true_rank = int(actual['predicted_rank'])
        distance = abs(int(predicted['step_index'])-int(actual['step_index']))
        trajectories.append({'trajectory_id':trajectory_id,'task_id':actual['task_id'],
            'actual_root_cause_step':int(actual['step_index']),
            'predicted_root_cause_step':int(predicted['step_index']),
            'predicted_probability':float(predicted['root_cause_probability']),
            'actual_root_cause_rank':true_rank,'exact_match':true_rank==1,'in_top3':true_rank<=3,
            'reciprocal_rank':1/true_rank,'absolute_step_distance':distance,
            'in_top5':true_rank<=5 if len(group)>=5 else None,'total_steps':len(group)})
    table = pd.DataFrame(trajectories)
    top5 = table.loc[table['total_steps']>=5,'in_top5']
    metrics = {'faulty_trajectories':len(table),'exact_step_accuracy':float(table['exact_match'].mean()),
        'top3_accuracy':float(table['in_top3'].mean()),'mrr':float(table['reciprocal_rank'].mean()),
        'mean_absolute_step_distance':float(table['absolute_step_distance'].mean()),
        'top5_accuracy':float(top5.mean()) if len(top5) else None,'top5_eligible_trajectories':len(top5),
        'tie_rule':'probability descending, then step_index ascending'}
    steps = ranked.rename(columns={'is_root_cause':'actual_is_root_cause'}).sort_values(['trajectory_id','step_index'])
    return metrics,steps,table
