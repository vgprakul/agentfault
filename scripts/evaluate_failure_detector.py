"""Evaluate a saved detector without retraining."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.failure_detector.evaluate import evaluate

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', default='models/failure_detector')
    parser.add_argument('--results-dir', default='results/failure_detector')
    parser.add_argument('--threshold', type=float, default=.5)
    args=parser.parse_args()
    try: evaluate(args.model_dir, args.results_dir, args.threshold)
    except (ValueError, FileNotFoundError) as error: parser.exit(2, f'{error}\n')
