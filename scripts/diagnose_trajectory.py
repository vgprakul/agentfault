"""Run deterministic diagnosis; optional truth display is evaluation-only."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnosis.orchestration_agent import diagnose_trajectory, load_inputs
from diagnosis.schemas import DiagnosisConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory-id',required=True)
    parser.add_argument('--trajectory-input',action='append')
    parser.add_argument('--feature-dir',action='append')
    parser.add_argument('--output-dir',default='results/diagnosis')
    parser.add_argument('--top-k',type=int,default=3)
    parser.add_argument('--failure-threshold',type=float,default=.5)
    parser.add_argument('--category-threshold',type=float,default=.5)
    parser.add_argument('--localization-threshold',type=float,default=.5)
    parser.add_argument('--hypothesis-threshold',type=float,default=.5)
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--show-ground-truth',action='store_true')
    args=parser.parse_args()
    try:
        config=DiagnosisConfig(top_k=args.top_k,failure_threshold=args.failure_threshold,category_threshold=args.category_threshold,
            localization_threshold=args.localization_threshold,hypothesis_threshold=args.hypothesis_threshold)
        result=diagnose_trajectory(args.trajectory_id,args.trajectory_input,args.feature_dir,args.output_dir,config)
        truth=None
        if args.show_ground_truth:
            record,_,_,_=load_inputs(args.trajectory_id,args.trajectory_input,args.feature_dir)
            truth={'actual_fault_type':record.fault_type,'actual_origin_step':record.origin_step,
                   'predicted_fault_type':result['predicted_subtype'],'predicted_origin_step':result['predicted_root_cause_step']}
        if args.json:
            # Ground truth is a separate debug object and is not persisted in diagnosis artifacts.
            print(json.dumps({**result,**({'evaluation_only':truth} if truth else {})},indent=2,allow_nan=False))
        else:
            print('=== AgentFault Diagnosis ===')
            for key in ['trajectory_id','diagnosis_status','failure_detected','failure_probability','predicted_category','predicted_subtype','predicted_root_cause_step','root_cause_confidence','diagnosis_confidence']:
                print(f'{key}: {result[key]}')
            print('Ranked hypotheses:')
            for h in result['ranked_hypotheses']: print(f"  Step {h['step_index']} | score {h['score']:.6f} | {h['hypothesis']}")
            print('Selected hypothesis:',(result['selected_hypothesis'] or {}).get('hypothesis'))
            print('Propagation:', ' -> '.join(str(s['step_index']) for s in result['propagation_chain']) or 'Unavailable')
            print(result['natural_language_explanation'])
            if truth: print('Evaluation only:',json.dumps(truth))
            print('Saved to:',args.output_dir)
    except (ValueError,FileNotFoundError,KeyError) as error: parser.exit(2,f'Diagnosis stopped: {error}\n')


if __name__=='__main__': main()
