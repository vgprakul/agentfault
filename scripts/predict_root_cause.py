"""Rank steps of a selected trajectory without loading ground-truth metadata."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from ml.localizer.inference import RootCauseLocalizer


def main():
    parser=argparse.ArgumentParser(description='AgentFault Root-Cause Localization')
    parser.add_argument('--trajectory-id',required=True)
    parser.add_argument('--features',default='data/features/step_features.csv')
    parser.add_argument('--model-dir',default='models/localizer')
    args=parser.parse_args()
    rows=pd.read_csv(args.features,dtype={'trajectory_id':str,'task_id':str})
    rows=rows.loc[rows['trajectory_id']==args.trajectory_id]
    if rows.empty:
        parser.error('No steps match that trajectory ID')
    result=RootCauseLocalizer(args.model_dir).predict(rows)
    print('=== AgentFault Root-Cause Localization ===')
    print(f'Trajectory: {args.trajectory_id}\nPredicted root-cause step: {result["predicted_root_cause_step"]}\nConfidence: {result["root_cause_probability"]:.6f}')
    print('Ranked suspicious steps:')
    for rank,step in enumerate(result['ranked_steps'],1):
        print(f'{rank}. Step {step["step_index"]} - {step["probability"]:.6f}')


if __name__=='__main__':
    main()
