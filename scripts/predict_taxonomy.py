"""Predict from an ID in the feature CSV, or from an independent feature-row CSV."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from ml.taxonomy.inference import TaxonomyPredictor


def main():
    parser=argparse.ArgumentParser(description='AgentFault Taxonomy Prediction')
    parser.add_argument('--trajectory-id')
    parser.add_argument('--features',default='data/features/trajectory_features.csv')
    parser.add_argument('--model-dir',default='models/taxonomy')
    args=parser.parse_args()
    rows=pd.read_csv(args.features,dtype={'trajectory_id':str})
    if args.trajectory_id:
        rows=rows.loc[rows['trajectory_id']==args.trajectory_id]
        if len(rows)!=1:
            parser.error('Trajectory ID must match exactly one feature row')
    if not len(rows):
        parser.error('Feature CSV contains no rows')
    predictor=TaxonomyPredictor(args.model_dir)
    print('=== AgentFault Taxonomy Prediction ===')
    for result in predictor.predict_many(rows):
        print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
