"""Train using existing outcome labels and grouped splits."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.failure_detector.train import train

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', default='data/features/trajectory_features.csv')
    parser.add_argument('--manifest', default='data/features/split_manifest.csv')
    parser.add_argument('--additional-features', action='append')
    parser.add_argument('--additional-manifest', action='append')
    parser.add_argument('--allow-single-class-evaluation', action='store_true')
    parser.add_argument('--model-dir', default='models/failure_detector')
    parser.add_argument('--results-dir', default='results/failure_detector')
    parser.add_argument('--seed', type=int, default=42)
    args=parser.parse_args()
    try:
        train(args.features, args.manifest, args.model_dir, args.results_dir, args.seed,
              args.additional_features, args.additional_manifest, args.allow_single_class_evaluation)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'Training stopped: {error}\n')

if __name__=='__main__': main()
