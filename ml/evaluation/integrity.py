"""Split audits return a report before callers fail fast."""
import pandas as pd


def split_integrity(manifest, datasets):
    issues,overlaps = [],[]
    required={'trajectory_id','task_id','split'}
    if not required<=set(manifest):
        return {'valid':False,'number_of_overlaps':0,'overlap_examples':[], 'issues':['Manifest columns missing']}
    manifest=manifest.copy()
    manifest['split']=manifest['split'].str.lower()
    if manifest[list(required)].isna().any().any() or not manifest['split'].isin(['train','validation','test']).all():
        issues.append('Missing IDs or unsupported split values in manifest')
    for key in ['trajectory_id','task_id']+(['base_task_id'] if 'base_task_id' in manifest else []):
        bad=manifest.groupby(key)['split'].nunique()
        overlaps.extend({'source':'manifest','key':key,'value':str(v)} for v in bad[bad>1].index)
    if manifest['trajectory_id'].duplicated().any():
        issues.append('Duplicate trajectory assignments in manifest')
    else:
        indexed=manifest.set_index('trajectory_id')
        for name,frame in datasets.items():
            if not {'trajectory_id','task_id'}<=set(frame):
                issues.append(f'{name}: missing ID columns'); continue
            assigned=frame['trajectory_id'].map(indexed['split'])
            tasks=frame['trajectory_id'].map(indexed['task_id'])
            if frame[['trajectory_id','task_id']].isna().any().any() or assigned.isna().any() or not frame['task_id'].eq(tasks).all():
                issues.append(f'{name}: missing assignment or task identity mismatch')
            if frame.groupby('trajectory_id')['task_id'].nunique().gt(1).any():
                issues.append(f'{name}: a trajectory has multiple task IDs')
            if 'split' in frame:
                existing=frame['split'].str.lower()
                mismatch=existing.ne(assigned)
                overlaps.extend({'source':name,'key':'trajectory_id','value':str(v),'reason':'Embedded split differs from manifest'}
                                for v in frame.loc[mismatch,'trajectory_id'].unique())
            grouped=frame.assign(_evaluation_split=assigned)
            for key in ['trajectory_id','task_id']+(['base_task_id'] if 'base_task_id' in frame else []):
                if frame[key].isna().any():
                    issues.append(f'{name}: missing {key}')
                counts=grouped.groupby(key)['_evaluation_split'].nunique()
                overlaps.extend({'source':name,'key':key,'value':str(v)} for v in counts[counts>1].index)
    return {'valid':not issues and not overlaps,'number_of_overlaps':len(overlaps),
            'overlap_examples':overlaps[:20],'issues':issues,'manifest_trajectories':len(manifest),
            'split_trajectory_counts':manifest['split'].value_counts().to_dict(),
            'split_task_counts':manifest.groupby('split')['task_id'].nunique().to_dict()}
