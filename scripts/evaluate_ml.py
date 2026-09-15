"""Evaluate existing AgentFault ML models; never silently retrain."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ml.evaluation.review3_report import evaluate_all


def main():
    parser=argparse.ArgumentParser(description='Unified AgentFault ML Evaluation')
    parser.add_argument('--feature-dir',default='data/features')
    parser.add_argument('--model-dir',default='models')
    parser.add_argument('--results-dir',default='results')
    parser.add_argument('--output',default='results/evaluation')
    parser.add_argument('--random-seed',type=int,default=42)
    args=parser.parse_args()
    evaluate_all(args.feature_dir,args.model_dir,args.results_dir,args.output,args.random_seed)


if __name__=='__main__':
    main()
