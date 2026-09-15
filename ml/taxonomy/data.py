"""Strict feature/manifest joins; no resplitting and no implicit predictor selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.trajectory_features import FEATURE_COLUMNS
from .utils import CATEGORIES, SUBTYPE_TO_CATEGORY, validate_features

DEFAULT_FEATURES = 'data/features/trajectory_features.csv'
DEFAULT_MANIFEST = 'data/features/split_manifest.csv'


def feature_frame(frame):
    validate_features()
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required feature columns: {sorted(missing)}')
    selected = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        if column == 'dominant_agent':
            selected[column] = selected[column].fillna('UNKNOWN').astype(str)
        else:
            selected[column] = pd.to_numeric(selected[column].replace('', np.nan), errors='raise')
            if np.isinf(selected[column].to_numpy(dtype=float)).any():
                raise ValueError(f'Infinite value in feature {column}')
    return selected


def _unique_ids(frame, context):
    if frame['trajectory_id'].isna().any() or frame['trajectory_id'].duplicated().any():
        raise ValueError(f'{context}: missing or duplicate trajectory IDs')


def load_dataset(features=DEFAULT_FEATURES, manifest=DEFAULT_MANIFEST, dictionary=None):
    """Return failure-labelled rows plus counts; inspect every split before filtering."""
    frame = pd.read_csv(features, dtype={'trajectory_id':str, 'task_id':str, 'base_task_id':str})
    required = {'trajectory_id','task_id','fault_type','fault_injected','outcome_status'}
    if not required <= set(frame.columns):
        raise ValueError(f'Missing schema columns: {sorted(required-set(frame.columns))}')
    _unique_ids(frame,'features')
    feature_frame(frame)
    dictionary_path = Path(dictionary) if dictionary else Path(features).with_name('feature_dictionary.json')
    if dictionary_path.exists():
        entries = json.loads(dictionary_path.read_text(encoding='utf-8'))
        for column in FEATURE_COLUMNS:
            if entries.get('trajectory.'+column,{}).get('role') != 'feature':
                raise ValueError(f'Feature dictionary does not authorize predictor: {column}')
    if manifest is not None:
        assignments = pd.read_csv(manifest, dtype={'trajectory_id':str,'task_id':str})
        if not {'trajectory_id','task_id','split'} <= set(assignments):
            raise ValueError('Manifest needs trajectory_id, task_id, split')
        _unique_ids(assignments,'manifest')
        assignments['split'] = assignments['split'].str.lower()
        if assignments['task_id'].isna().any() or assignments.groupby('task_id')['split'].nunique().gt(1).any():
            raise ValueError('Manifest has missing tasks or a task in multiple splits')
        joined = frame.merge(assignments, on='trajectory_id', how='left', validate='one_to_one', suffixes=('','_manifest'))
        if joined['task_id_manifest'].isna().any() or not joined['task_id'].eq(joined['task_id_manifest']).all():
            raise ValueError('Feature/manifest task identity mismatch or missing assignment')
        if 'split_manifest' in joined:
            if not joined['split'].str.lower().eq(joined['split_manifest']).all():
                raise ValueError('Embedded split disagrees with manifest')
            joined['split'] = joined.pop('split_manifest')
        frame = joined.drop(columns='task_id_manifest')
    elif 'split' not in frame:
        raise ValueError('A manifest or embedded split column is required')
    frame['split'] = frame['split'].str.lower()
    if not frame['split'].isin(['train','validation','test']).all():
        raise ValueError('Only train/validation/test assignments are supported')
    group = 'base_task_id' if 'base_task_id' in frame else 'task_id'
    if frame[group].isna().any() or frame.groupby(group)['split'].nunique().gt(1).any():
        raise ValueError('Task group crosses splits or is missing')
    subtype = frame['fault_type'].fillna('').astype(str)
    unknown = set(subtype) - set(SUBTYPE_TO_CATEGORY) - {'','NONE','UNKNOWN'}
    if unknown:
        raise ValueError(f'Unknown taxonomy labels: {sorted(unknown)}')
    flags = frame['fault_injected'].astype(str).str.lower()
    if not flags.isin(['1','0','true','false']).all():
        raise ValueError('fault_injected must be boolean or 0/1')
    injected = flags.isin(['1','true'])
    genuine_failure = frame['outcome_status'].fillna('').str.upper().isin(['FAIL','FAILED','FAILURE','ERROR'])
    keep = subtype.isin(SUBTYPE_TO_CATEGORY) & (injected | genuine_failure)
    excluded = int((~keep).sum())
    result = frame.loc[keep].copy()
    result['fault_subtype'] = subtype[keep]
    result['fault_category'] = subtype[keep].map(SUBTYPE_TO_CATEGORY)
    info = {'loaded_rows':len(frame),'usable_rows':len(result),'excluded_unlabelled_or_clean':excluded,
            'class_distribution':{split:{c:int(((result['split']==split)&(result['fault_category']==c)).sum()) for c in CATEGORIES}
                                  for split in ['train','validation','test']},
            'split_counts':result['split'].value_counts().to_dict()}
    return result, info
