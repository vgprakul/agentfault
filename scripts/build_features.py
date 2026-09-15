"""Run directly from the repository: python scripts/build_features.py --input PATH."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engineering.pipeline import build


def main():
    parser = argparse.ArgumentParser(description='AgentFault Feature Engineering')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='data/features')
    parser.add_argument('--no-semantic', action='store_true', help='Explicitly leave embedding features unavailable')
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()
    print('AgentFault Feature Engineering', flush=True)
    trajectories, steps, manifest, issues = build(args.input,args.output,not args.no_semantic,args.random_state)
    for issue in issues:
        print(f'Skipped/reported: {issue}')
    clean = sum(not row['fault_injected'] for row in trajectories)
    print(f'Loaded trajectories: {len(trajectories)}\nClean trajectories: {clean}\nFaulty trajectories: {len(trajectories)-clean}')
    print(f'Trajectory samples generated: {len(trajectories)}\nStep samples generated: {len(steps)}')
    for split in ['train','validation','test']:
        print(f'{split.title()} tasks: {len({r["task_id"] for r in manifest if r["split"] == split})}')
    print('Saved:')
    for name in ['trajectory_features.csv','step_features.csv','split_manifest.csv','feature_dictionary.json','build_report.json']:
        print(Path(args.output)/name)


if __name__ == '__main__':
    main()
