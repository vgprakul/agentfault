"""Reproduce held-out evaluation of the saved, frozen localizer."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ml.localizer.evaluate import evaluate


def main():
    parser=argparse.ArgumentParser(description='Evaluate AgentFault Root-Cause Localizer')
    parser.add_argument('--features',default='data/features/step_features.csv')
    parser.add_argument('--manifest',default='data/features/split_manifest.csv')
    parser.add_argument('--trajectory-metadata',default='data/features/trajectory_features.csv')
    parser.add_argument('--embedded-split',action='store_true')
    parser.add_argument('--embedded-fault-metadata',action='store_true')
    parser.add_argument('--model-dir',default='models/localizer')
    parser.add_argument('--results-dir',default='results/localizer')
    args=parser.parse_args()
    evaluate(args.model_dir,args.results_dir,args.features,None if args.embedded_split else args.manifest,
             None if args.embedded_fault_metadata else args.trajectory_metadata)


if __name__=='__main__':
    main()
