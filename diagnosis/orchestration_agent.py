"""Compose existing inference APIs and deterministic evidence hypotheses."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import pandas as pd
from feature_engineering.loader import load
from feature_engineering.trajectory_features import FEATURE_COLUMNS as TRAJECTORY_FEATURES
from ml.localizer.data import feature_frame as localizer_features
from ml.failure_detector.inference import FailureDetector
from ml.taxonomy.inference import TaxonomyPredictor
from ml.localizer.inference import RootCauseLocalizer
from ml.taxonomy.utils import save_json
from .schemas import DiagnosisConfig, ReplayRequest, interpret_replay
from .utils import probability, observable_steps, weighted
from .hypothesis_generation import generate
from .hypothesis_ranking import rank_hypotheses
from .explanation import propagation_chain, explain


def assemble_diagnosis(trajectory,failure_detection,taxonomy=None,localization=None,config=None,replay_result=None):
    """Consume supplied predictions. Ground-truth record fields are never consulted."""
    config=config or DiagnosisConfig()
    steps=observable_steps(trajectory)
    failure=probability(failure_detection.get('failure_probability'))
    result={'trajectory_id':trajectory.trajectory_id,'failure_detected':None if failure is None else failure>=config.failure_threshold,
        'failure_probability':failure,'predicted_category':None,'predicted_subtype':None,
        'predicted_root_cause_step':None,'root_cause_confidence':None,'ranked_suspicious_steps':[],
        'taxonomy_result':None,'ranked_hypotheses':[],'selected_hypothesis':None,'selected_evidence_step':None,
        'propagation_chain':[],'diagnosis_confidence':None,'counterfactual_status':'NOT_RUN','replay_request':None,
        'replay_interpretation':None,'recommended_next_action':None,'reasons':[],
        'configuration':asdict(config),'confidence_components':None,
        'limitations':['ML probabilities and diagnosis scores are uncalibrated.',
                      'Current detector validation/test contain failures only; synthetic source proxies and duplicate features limit reliability.',
                      'A hypothesis and chronological links do not establish a causal root cause.']}
    if failure is not None and failure<config.failure_threshold:
        result['diagnosis_status']='NO_FAILURE_DETECTED'
        result['natural_language_explanation']=explain(result)
        return result
    taxonomy=taxonomy or {}; localization=localization or {}
    category_confidence=probability(taxonomy.get('category_confidence'))
    subtype_confidence=probability(taxonomy.get('subtype_confidence')) if taxonomy.get('predicted_subtype') else None
    root_confidence=probability(localization.get('root_cause_probability'))
    # Only approved prediction fields are copied, even for externally supplied dictionaries.
    result['taxonomy_result']={key:taxonomy.get(key) for key in ['predicted_category','predicted_subtype','category_confidence','subtype_confidence','category_probabilities']}
    result.update(predicted_category=taxonomy.get('predicted_category'),predicted_subtype=taxonomy.get('predicted_subtype'),
                  predicted_root_cause_step=localization.get('predicted_root_cause_step'),root_cause_confidence=root_confidence)
    hypotheses,ranked_steps=generate(steps,taxonomy,localization,config.top_k)
    if ranked_steps and ((localization.get('predicted_root_cause_step') is not None and localization['predicted_root_cause_step']!=ranked_steps[0]['step_index']) or (root_confidence is not None and root_confidence!=ranked_steps[0]['probability'])):
        raise ValueError('Localization top prediction must match the highest ranked probability')
    result['ranked_suspicious_steps']=ranked_steps[:max(3,config.top_k)]
    ranked,selected=rank_hypotheses(hypotheses,config)
    result['ranked_hypotheses']=ranked
    if taxonomy.get('predicted_category') is None: result['reasons'].append('Taxonomy category unavailable')
    for name,value,threshold in [('failure',failure,config.failure_threshold),('taxonomy',category_confidence,config.category_threshold),('localization',root_confidence,config.localization_threshold)]:
        if value is None or value<threshold: result['reasons'].append(f'{name} confidence is missing or below {threshold:.2f}')
    if not selected: result['reasons'].append('No hypothesis exceeded the configured confidence threshold')
    if result['reasons']:
        result['diagnosis_status']='UNDETERMINED'
    else:
        competitors=[h['score'] for h in ranked if h is not selected]
        # No runner-up supplies no evidence of separation; close-score earlier selection can have margin 0.
        margin=max(0,selected['score']-max(competitors)) if competitors else 0.0
        components={'root':selected['signals']['localization'],'category':category_confidence,'subtype':subtype_confidence,'margin':margin}
        result.update(diagnosis_status='HYPOTHESIS_SELECTED', selected_hypothesis=selected,
            selected_evidence_step=selected['evidence'], diagnosis_confidence=weighted(components,{'root':.4,'category':.3,'subtype':.2,'margin':.1}),
            confidence_components=components,propagation_chain=propagation_chain(steps,selected['step_index']))
        request=ReplayRequest(trajectory.trajectory_id,selected['step_index'],selected['hypothesis_type'],selected['proposed_change'])
        result['replay_request']=asdict(request)
        if replay_result is not None:
            interpretation=interpret_replay(request,replay_result)
            result['counterfactual_status']='SUPPLIED_RESULT'
            result['replay_interpretation']=interpretation
            if interpretation['status']=='REJECTED':
                result['diagnosis_status']='UNDETERMINED'
                result['reasons'].append('Supplied replay did not improve the score by the required amount; selected hypothesis rejected')
                result['selected_hypothesis']=None
                result['selected_evidence_step']=None
                result['diagnosis_confidence']=None
                result['propagation_chain']=[]
    result['natural_language_explanation']=explain(result)
    return result


class DiagnosisAgent:
    def __init__(self,config=None,failure_detector=None,taxonomy_classifier=None,localizer=None,
                 failure_model_dir='models/failure_detector',taxonomy_model_dir='models/taxonomy',localizer_model_dir='models/localizer'):
        self.config=config or DiagnosisConfig()
        self.failure=failure_detector; self.taxonomy=taxonomy_classifier; self.localizer=localizer
        self.model_dirs=(failure_model_dir,taxonomy_model_dir,localizer_model_dir)

    def diagnose(self,trajectory,trajectory_features,step_features=None):
        if self.failure is None: self.failure=FailureDetector(self.model_dirs[0])
        # Do not pass metadata/labels to trajectory predictors.
        features={name:trajectory_features[name] for name in TRAJECTORY_FEATURES}
        failure=self.failure.predict(features,threshold=self.config.failure_threshold)
        p=probability(failure.get('failure_probability'))
        if p is None or p<self.config.failure_threshold:
            return assemble_diagnosis(trajectory,failure,config=self.config)
        if self.taxonomy is None: self.taxonomy=TaxonomyPredictor(self.model_dirs[1])
        if self.localizer is None: self.localizer=RootCauseLocalizer(self.model_dirs[2])
        if step_features is None or step_features.empty: raise ValueError('Step features required for a predicted failure')
        if step_features['trajectory_id'].nunique()!=1 or str(step_features.iloc[0]['trajectory_id'])!=trajectory.trajectory_id:
            raise ValueError('Step features must match trajectory ID')
        if step_features['step_index'].tolist()!=[s.step_index for s in trajectory.steps]:
            raise ValueError('Feature step indices do not match captured trajectory')
        step_input=localizer_features(step_features).copy()
        step_input['trajectory_id']=trajectory.trajectory_id
        step_input['task_id']=trajectory.task_id
        step_input['step_index']=step_features['step_index'].to_numpy()
        taxonomy=self.taxonomy.predict(features)
        localization=self.localizer.predict(step_input)
        return assemble_diagnosis(trajectory,failure,taxonomy,localization,self.config)


def load_inputs(trajectory_id,trajectory_paths=None,feature_dirs=None):
    records=[]; diagnostics=[]
    for path in trajectory_paths or ['dataset/agentfault-100.jsonl','data/trajectories']:
        if not Path(path).exists(): continue
        loaded,_,issues=load(path); records.extend(r for r in loaded if r.trajectory_id==trajectory_id); diagnostics.extend(issues)
    if len(records)!=1: raise ValueError(f'Expected one captured trajectory for {trajectory_id}; found {len(records)}. Loader issues: {diagnostics[:3]}')
    matches=[]
    for directory in feature_dirs or ['data/features','data/features/real']:
        file=Path(directory)/'trajectory_features.csv'
        if not file.exists(): continue
        frame=pd.read_csv(file); match=frame.loc[frame['trajectory_id']==trajectory_id]
        if not match.empty:
            step_path=Path(directory)/'step_features.csv'
            steps=pd.read_csv(step_path) if step_path.exists() else pd.DataFrame()
            if not steps.empty: steps=steps.loc[steps['trajectory_id']==trajectory_id].sort_values('step_index')
            matches.append((match,steps,str(file.resolve())))
    if len(matches)!=1 or len(matches[0][0])!=1:
        raise ValueError('Expected one existing trajectory feature row. Run scripts/build_features.py for the captured input first.')
    frame,steps,path=matches[0]
    if str(frame.iloc[0]['task_id'])!=records[0].task_id: raise ValueError('Feature task ID differs from trajectory')
    return records[0],frame.iloc[0],steps,path


def diagnose_trajectory(trajectory_id,trajectory_paths=None,feature_dirs=None,output_dir='results/diagnosis',config=None):
    record,row,steps,path=load_inputs(trajectory_id,trajectory_paths,feature_dirs)
    result=DiagnosisAgent(config).diagnose(record,row,steps)
    result['feature_file']=path
    result['feature_provenance_note']='Existing feature cache; ID/task/step consistency checked. Payload freshness is not independently verified.'
    if output_dir is not None:
        # IDs are identifiers only, never parsed for fault labels; sanitize paths to prevent traversal.
        safe=''.join(c if c.isalnum() or c in '-_.' else '_' for c in trajectory_id).strip('.')[:160]
        if safe!=trajectory_id: safe+='_'+hashlib.sha256(trajectory_id.encode()).hexdigest()[:10]
        destination=Path(output_dir)/(safe+'_diagnosis.json')
        save_json(destination,result)
    return result
