"""Load outcome targets and preserve existing groups, including explicit additions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from feature_engineering.trajectory_features import FEATURE_COLUMNS
from ml.evaluation.integrity import split_integrity
from .utils import outcome_to_label, validate_features, file_hash

DEFAULT_FEATURES='data/features/trajectory_features.csv'
DEFAULT_MANIFEST='data/features/split_manifest.csv'


def feature_frame(rows):
    validate_features()
    missing=set(FEATURE_COLUMNS)-set(rows)
    if missing:
        raise ValueError(f'Missing predictors: {sorted(missing)}')
    result=rows.loc[:,FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        if column=='dominant_agent':
            result[column]=result[column].fillna('UNKNOWN').astype(str)
        else:
            result[column]=pd.to_numeric(result[column].replace('',np.nan),errors='raise')
            if np.isinf(result[column].to_numpy(dtype=float)).any():
                raise ValueError(f'Infinite predictor: {column}')
    return result


def load_dataset(features=DEFAULT_FEATURES,manifest=DEFAULT_MANIFEST,additional_features=None,additional_manifests=None):
    additional_features=additional_features or []
    additional_manifests=additional_manifests or []
    if len(additional_features)!=len(additional_manifests):
        raise ValueError('Each additional feature file needs its existing manifest')
    frames,assignments,inputs=[],[],[]
    for feature_path,manifest_path in [(features,manifest)]+list(zip(additional_features,additional_manifests)):
        frame=pd.read_csv(feature_path,dtype={'trajectory_id':str,'task_id':str,'base_task_id':str})
        if not {'trajectory_id','task_id','outcome_status'}<=set(frame):
            raise ValueError('Trajectory CSV needs trajectory_id, task_id and outcome_status')
        if frame['trajectory_id'].isna().any() or frame['trajectory_id'].duplicated().any():
            raise ValueError('Missing/duplicate trajectory IDs')
        feature_frame(frame)
        dictionary=Path(feature_path).with_name('feature_dictionary.json')
        if dictionary.exists():
            entries=json.loads(dictionary.read_text(encoding='utf-8'))
            for column in FEATURE_COLUMNS:
                if entries.get('trajectory.'+column,{}).get('role')!='feature':
                    raise ValueError(f'Dictionary does not authorize predictor {column}')
        if manifest_path:
            assignment=pd.read_csv(manifest_path,dtype={'trajectory_id':str,'task_id':str,'base_task_id':str})
        elif 'split' in frame:
            assignment=frame[['trajectory_id','task_id','split']].copy()
        else:
            raise ValueError('An existing manifest or embedded split is required')
        report=split_integrity(assignment,{'trajectories':frame})
        if not report['valid']:
            raise ValueError(f'Split integrity failed: {report}')
        frames.append(frame); assignments.append(assignment)
        inputs.append({'features':str(Path(feature_path).resolve()),'manifest':str(Path(manifest_path).resolve()) if manifest_path else None,
                       'features_sha256':file_hash(feature_path),'manifest_sha256':file_hash(manifest_path)})
    rows=pd.concat(frames,ignore_index=True)
    assignment=pd.concat(assignments,ignore_index=True)
    if rows['trajectory_id'].duplicated().any():
        raise ValueError('Duplicate trajectories across input files')
    if 'base_task_id' in rows:
        rows['base_task_id']=rows['base_task_id'].fillna(rows['task_id'])
    if 'base_task_id' in assignment:
        assignment['base_task_id']=assignment['base_task_id'].fillna(assignment['task_id'])
    report=split_integrity(assignment,{'combined_trajectories':rows})
    if not report['valid']:
        raise ValueError(f'Combined split integrity failed: {report}')
    rows['split']=rows['trajectory_id'].map(assignment.set_index('trajectory_id')['split'].str.lower())
    rows['failure_label']=rows['outcome_status'].map(outcome_to_label)
    excluded=rows.loc[rows['failure_label'].isna(),['trajectory_id','outcome_status']].fillna('MISSING').to_dict(orient='records')
    usable=rows.loc[rows['failure_label'].notna()].copy()
    usable['failure_label']=usable['failure_label'].astype(int)
    counts={split:{'samples':len(group),'SUCCESS':int((group['failure_label']==0).sum()),'FAILURE':int((group['failure_label']==1).sum())}
            for split in ['train','validation','test'] for group in [usable.loc[usable['split']==split]]}
    info={'loaded_rows':len(rows),'usable_samples':len(usable),'excluded_outcomes':excluded,'split_counts':counts,
          'success_samples':int((usable['failure_label']==0).sum()),'failure_samples':int((usable['failure_label']==1).sum()),
          'split_integrity':report,'inputs':inputs}
    return usable,info
