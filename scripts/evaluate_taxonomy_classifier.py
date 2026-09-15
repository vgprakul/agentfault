"""Reproduce evaluation of the saved, frozen selected model."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ml.taxonomy.evaluate import evaluate


def main():
    parser=argparse.ArgumentParser(description='Evaluate saved AgentFault taxonomy classifier')
    parser.add_argument('--features',default='data/features/trajectory_features.csv')
    parser.add_argument('--manifest',default='data/features/split_manifest.csv')
    parser.add_argument('--embedded-split',action='store_true')
    parser.add_argument('--model-dir',default='models/taxonomy')
    parser.add_argument('--results-dir',default='results/taxonomy')
    args=parser.parse_args()
    evaluate(args.model_dir,args.results_dir,args.features,None if args.embedded_split else args.manifest)


if __name__=='__main__':
    main()
