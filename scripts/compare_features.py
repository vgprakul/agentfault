"""Print actual clean/tool/control feature rows and a root-cause sample."""
import csv
import json
from pathlib import Path


def read(path):
    with Path(path).open(encoding='utf-8') as file:
        return list(csv.DictReader(file))


def comparison():
    faulty=read('data/features/trajectory_features.csv')
    clean=read('data/features/real/trajectory_features.csv')
    selected=[next(r for r in clean if r['fault_injected']=='0'),
              next(r for r in faulty if r['fault_type'].startswith('TOOL_')),
              next(r for r in faulty if r['fault_type']=='CTRL_LOOP')]
    names=['total_steps','total_tool_calls','tool_error_count','repeated_action_count',
           'loop_indicator','avg_semantic_drift','max_semantic_drift']
    lines=['| Feature | Clean | Tool fault | Control fault |','|---|---:|---:|---:|']
    for name in names:
        values=[f'{float(r[name]):.4f}' if r[name] and '.' in r[name] else r[name] or 'unavailable' for r in selected]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    root=next(r for r in read('data/features/step_features.csv') if r['trajectory_id']==selected[1]['trajectory_id'] and r['is_root_cause']=='1')
    return '\n'.join(lines),selected,root


if __name__=='__main__':
    table,selected,root=comparison()
    print(table)
    print('\nTrajectories:')
    for row in selected:
        print(row['trajectory_id'],row['fault_type'])
    print('\nActual step CSV root-cause row:')
    print(json.dumps(root,indent=2))
