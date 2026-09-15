"""Predict from complete-trajectory features; labels are not inference inputs."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from ml.failure_detector.inference import FailureDetector

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', default='data/features/trajectory_features.csv')
    parser.add_argument('--trajectory-id')
    parser.add_argument('--model-dir', default='models/failure_detector')
    parser.add_argument('--threshold', type=float, default=.5)
    args=parser.parse_args()
    try:
        rows=pd.read_csv(args.features)
        if args.trajectory_id: rows=rows.loc[rows['trajectory_id']==args.trajectory_id]
        if rows.empty: raise ValueError('No matching feature rows')
        detector=FailureDetector(args.model_dir)
        for (_, row), result in zip(rows.iterrows(), detector.predict_many(rows, args.threshold)):
            print('=== AgentFault Failure Detection ===')
            print('Trajectory:', row.get('trajectory_id', 'UNKNOWN'))
            print('Failure probability:', result['failure_probability'])
            print('Predicted outcome:', result['predicted_outcome'])
            print('Threshold:', result['threshold'])
    except (ValueError, FileNotFoundError) as error: parser.exit(2, f'{error}\n')
