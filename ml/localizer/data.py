"""Validate step rows, retain known single-root faults, preserve grouped splits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from feature_engineering.step_features import FEATURE_COLUMNS
from .utils import validate_features

DEFAULT_FEATURES = 'data/features/step_features.csv'
DEFAULT_MANIFEST = 'data/features/split_manifest.csv'
DEFAULT_METADATA = 'data/features/trajectory_features.csv'
CATEGORICAL = {'step_type','agent_id','tool_name','status'}


def feature_frame(rows):
    validate_features()
    missing = set(FEATURE_COLUMNS)-set(rows.columns)
    if missing:
        raise ValueError(f'Missing step predictors: {sorted(missing)}')
    result = rows.loc[:,FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        if column in CATEGORICAL:
            result[column] = result[column].fillna('UNKNOWN').astype(str)
        else:
            result[column] = pd.to_numeric(result[column].replace('',np.nan),errors='raise')
            if np.isinf(result[column].to_numpy(dtype=float)).any():
                raise ValueError(f'Infinite predictor: {column}')
    return result


def check_grouping(frame):
    for column in ['trajectory_id','task_id'] + (['base_task_id'] if 'base_task_id' in frame else []):
        if frame[column].isna().any() or frame.groupby(column)['split'].nunique().gt(1).any():
            raise ValueError(f'Missing {column} or group crosses train/validation/test')
    if frame.groupby('trajectory_id')['task_id'].nunique().gt(1).any():
        raise ValueError('A trajectory has multiple task IDs')
    if 'base_task_id' in frame and frame.groupby('trajectory_id')['base_task_id'].nunique().gt(1).any():
        raise ValueError('A trajectory has multiple base task IDs')
    if not frame['split'].isin(['train','validation','test']).all():
        raise ValueError('Unsupported or missing split assignment')


def valid_step_indices(group):
    indices = pd.to_numeric(group['step_index'],errors='coerce')
    return (indices.notna().all() and np.isfinite(indices).all() and
            indices.ge(0).all() and indices.eq(indices.astype(float).round()).all() and not indices.duplicated().any())


def load_dataset(features=DEFAULT_FEATURES, manifest=DEFAULT_MANIFEST, trajectory_metadata=DEFAULT_METADATA):
    rows = pd.read_csv(features,dtype={'trajectory_id':str,'task_id':str,'base_task_id':str})
    required = {'trajectory_id','task_id','step_index','is_root_cause'} | set(FEATURE_COLUMNS)
    if not required <= set(rows):
        raise ValueError(f'Missing required columns: {sorted(required-set(rows))}')
    info = {'loaded_step_rows':len(rows),'loaded_trajectories':int(rows['trajectory_id'].nunique()),
            'faulty_trajectories':0,'usable_faulty_trajectories':0,'clean_trajectories':0,
            'missing_root_cause_labels':0,'invalid_trajectories':0,'multiple_positive_trajectories':0,
            'unknown_fault_status_trajectories':0,'issues':[]}
    dictionary = Path(features).with_name('feature_dictionary.json')
    if dictionary.exists():
        entries = json.loads(dictionary.read_text(encoding='utf-8'))
        for column in FEATURE_COLUMNS:
            if entries.get('step.'+column,{}).get('role') != 'feature':
                raise ValueError(f'Dictionary does not authorize predictor {column}')
    missing_ids = rows['trajectory_id'].isna()
    if missing_ids.any():
        info['issues'].append(f'Skipped {int(missing_ids.sum())} rows with missing trajectory ID')
        rows = rows.loc[~missing_ids].copy()
    if manifest:
        assignments = pd.read_csv(manifest,dtype={'trajectory_id':str,'task_id':str})
        if not {'trajectory_id','task_id','split'} <= set(assignments):
            raise ValueError('Manifest requires trajectory_id, task_id, split')
        if assignments['trajectory_id'].duplicated().any():
            raise ValueError('Duplicate trajectory assignments in manifest')
        assignments['split'] = assignments['split'].str.lower()
        check_grouping(assignments)
        rows = rows.merge(assignments,on='trajectory_id',how='left',validate='many_to_one',suffixes=('','_manifest'))
        if not rows['task_id'].eq(rows['task_id_manifest']).all():
            raise ValueError('Missing manifest assignment or task identity mismatch')
        if 'split_manifest' in rows:
            if not rows['split'].str.lower().eq(rows['split_manifest']).all():
                raise ValueError('Embedded step split disagrees with manifest')
            rows['split'] = rows.pop('split_manifest')
        rows = rows.drop(columns='task_id_manifest')
    elif 'split' not in rows:
        raise ValueError('Existing manifest or embedded splits required; no automatic resplit')
    rows['split'] = rows['split'].str.lower()
    check_grouping(rows)
    if trajectory_metadata:
        # Read ONLY eligibility/label metadata, never trajectory predictors.
        metadata = pd.read_csv(trajectory_metadata,usecols=lambda c:c in {'trajectory_id','task_id','base_task_id','fault_injected','origin_step'},
                               dtype={'trajectory_id':str,'task_id':str,'base_task_id':str})
        if not {'trajectory_id','task_id','fault_injected'} <= set(metadata):
            raise ValueError('Trajectory metadata requires IDs and fault_injected')
        if metadata['trajectory_id'].isna().any() or metadata['trajectory_id'].duplicated().any():
            raise ValueError('Duplicate/missing IDs in trajectory metadata')
        rows = rows.merge(metadata,on='trajectory_id',how='left',validate='many_to_one',suffixes=('','_metadata'))
        if not rows['task_id'].eq(rows['task_id_metadata']).all():
            raise ValueError('Missing trajectory metadata or task mismatch')
        rows = rows.drop(columns='task_id_metadata')
        for column in ['fault_injected','origin_step','base_task_id']:
            if column+'_metadata' in rows:
                same = rows[column].eq(rows[column+'_metadata']) | (rows[column].isna() & rows[column+'_metadata'].isna())
                if not same.all():
                    raise ValueError(f'Step/trajectory metadata disagree on {column}')
                rows = rows.drop(columns=column+'_metadata')
        check_grouping(rows)
    if 'fault_injected' not in rows:
        raise ValueError('Fault status metadata is needed to distinguish clean from missing-root trajectories')
    accepted = []
    for trajectory_id, group in rows.groupby('trajectory_id',sort=False):
        reason = None
        flags = group['fault_injected'].astype(str).str.lower()
        if flags.nunique()!=1 or not flags.isin(['0','1','false','true','0.0','1.0']).all():
            info['unknown_fault_status_trajectories'] += 1
            info['issues'].append(f'{trajectory_id}: unknown/inconsistent fault status; skipped')
            continue
        faulty = flags.iloc[0] in {'1','true','1.0'}
        info['faulty_trajectories' if faulty else 'clean_trajectories'] += 1
        target = pd.to_numeric(group['is_root_cause'],errors='coerce')
        if not valid_step_indices(group):
            reason = 'missing, duplicate, negative or noninteger step IDs'
        elif not target.isin([0,1]).all():
            info['missing_root_cause_labels'] += 1
            reason = 'missing or nonbinary root labels'
        elif target.sum()>1:
            info['multiple_positive_trajectories'] += 1
            reason = 'multiple positive steps'
        elif not faulty and target.sum()!=0:
            reason = 'clean trajectory has a positive root label'
        elif faulty and target.sum()==0:
            info['missing_root_cause_labels'] += 1
            info['issues'].append(f'{trajectory_id}: no positive root label; skipped without repair')
            continue
        if reason:
            info['invalid_trajectories'] += 1
            info['issues'].append(f'{trajectory_id}: {reason}; skipped without repair')
            continue
        if not faulty:
            continue
        group = group.copy()
        group['step_index'] = pd.to_numeric(group['step_index']).astype(int)
        group['is_root_cause'] = target.astype(int)
        if 'origin_step' in group and group['origin_step'].notna().any():
            origins = pd.to_numeric(group['origin_step'],errors='coerce')
            if origins.isna().any() or not np.isfinite(origins).all() or origins.nunique()!=1 or not origins.eq(origins.round()).all() or origins.lt(0).any():
                reason = 'invalid or inconsistent origin metadata'
            else:
                # Validate the existing injector's removed-origin rule, never derive labels.
                indices = sorted(group['step_index'])
                expected = next((i for i in indices if i>=origins.iloc[0]),indices[-1])
                if group.loc[group['is_root_cause']==1,'step_index'].iloc[0]!=expected:
                    reason = 'positive label disagrees with existing injector origin semantics'
        try:
            feature_frame(group)
        except (ValueError,TypeError) as exc:
            reason = f'invalid predictors: {exc}'
        if reason:
            info['invalid_trajectories'] += 1
            info['issues'].append(f'{trajectory_id}: {reason}; skipped without repair')
            continue
        accepted.append(group)
    result = pd.concat(accepted,ignore_index=True) if accepted else rows.iloc[:0].copy()
    info['usable_faulty_trajectories'] = int(result['trajectory_id'].nunique())
    info['usable_step_rows'] = len(result)
    info['positive_steps'] = int(result['is_root_cause'].sum())
    info['negative_steps'] = int((result['is_root_cause']==0).sum())
    info['split_counts'] = {split:{'trajectories':int(group['trajectory_id'].nunique()),'steps':len(group),
        'positive_steps':int(group['is_root_cause'].sum()),'negative_steps':int((group['is_root_cause']==0).sum())}
        for split in ['train','validation','test'] for group in [result.loc[result['split']==split]]}
    return result,info
