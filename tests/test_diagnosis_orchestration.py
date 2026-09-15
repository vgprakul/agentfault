import copy
import unittest
from unittest.mock import Mock
import pandas as pd
from injector.schema import AgentConfig,AgentFaultRecord,Outcome,StepRecord
from feature_engineering.trajectory_features import FEATURE_COLUMNS
from feature_engineering.step_features import FEATURE_COLUMNS as STEP_FEATURES
from diagnosis.orchestration_agent import DiagnosisAgent,assemble_diagnosis
from diagnosis.schemas import DiagnosisConfig,ReplayRequest,ReplayResult,interpret_replay
from diagnosis.hypothesis_ranking import rank_hypotheses


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.record=AgentFaultRecord('opaque-id','task',AgentConfig('test','test',0,['agent']),Outcome('FAIL',0),True,
            'TOOL_WRONG_ARGUMENT',2,{'modified_value':'SECRET'},4,1,'INJECTED','TEST',[
                StepRecord(1,'LLM_CALL','agent',{'query':'Calculate 2+2'},{'plan':'calculate'}),
                StepRecord(2,'TOOL_CALL','agent',{'expression':'2-2'},{'error':'invalid operation'},tool_name='arbitrary',status='ERROR'),
                StepRecord(3,'TOOL_RESPONSE','agent',{'error':'invalid operation'},{'text':'no result'},tool_name='arbitrary'),
                StepRecord(4,'LLM_CALL','agent',{'text':'no result'},{'answer':'unavailable'})])
        self.failure={'failure_probability':.9,'predicted_outcome':'FAILURE'}
        self.taxonomy={'predicted_category':'TOOL','category_confidence':.9,'predicted_subtype':'TOOL_WRONG_ARGUMENT','subtype_confidence':.8}
        self.localization={'predicted_root_cause_step':2,'root_cause_probability':.9,
            'ranked_steps':[{'step_index':2,'probability':.9},{'step_index':3,'probability':.6},{'step_index':1,'probability':.3},{'step_index':4,'probability':.1}]}

    def result(self,**kwargs):
        return assemble_diagnosis(self.record,self.failure,self.taxonomy,self.localization,**kwargs)

    def test_success_early_exit(self):
        failure=Mock(); failure.predict.return_value={'failure_probability':.1,'predicted_outcome':'SUCCESS'}
        taxonomy=Mock(); localizer=Mock()
        agent=DiagnosisAgent(failure_detector=failure,taxonomy_classifier=taxonomy,localizer=localizer)
        result=agent.diagnose(self.record,{c:0 for c in FEATURE_COLUMNS})
        self.assertEqual(result['diagnosis_status'],'NO_FAILURE_DETECTED')
        taxonomy.predict.assert_not_called();localizer.predict.assert_not_called()

    def test_failure_calls_models_without_labels(self):
        failure=Mock(); failure.predict.return_value=self.failure
        taxonomy=Mock(); taxonomy.predict.return_value=self.taxonomy
        localizer=Mock(); localizer.predict.return_value=self.localization
        rows=pd.DataFrame([{**{c:0 for c in STEP_FEATURES},'trajectory_id':'opaque-id','step_index':i,'is_root_cause':i==2} for i in range(1,5)])
        agent=DiagnosisAgent(failure_detector=failure,taxonomy_classifier=taxonomy,localizer=localizer)
        agent.diagnose(self.record,{**{c:0 for c in FEATURE_COLUMNS},'origin_step':2},rows)
        taxonomy.predict.assert_called_once();localizer.predict.assert_called_once()
        self.assertNotIn('origin_step',taxonomy.predict.call_args.args[0])
        self.assertNotIn('is_root_cause',localizer.predict.call_args.args[0])
        # The existing ranking API needs task IDs for metadata, not as predictors.
        from ml.localizer.ranking import rank_steps
        ranked=rank_steps(localizer.predict.call_args.args[0],[.1,.9,.5,.2])
        self.assertEqual(ranked.iloc[0]['step_index'],2)

    def test_ranking_and_topk(self):
        result=self.result(config=DiagnosisConfig(top_k=2))
        scores=[h['score'] for h in result['ranked_hypotheses']]
        self.assertEqual(scores,sorted(scores,reverse=True));self.assertEqual(len(scores),2)
        self.assertEqual(result['selected_hypothesis']['step_index'],2)

    def test_redundancy_pruning(self):
        hypotheses=self.result()['ranked_hypotheses']
        ranked,_=rank_hypotheses(hypotheses+[copy.deepcopy(hypotheses[0])],DiagnosisConfig())
        self.assertEqual(len(ranked),3)

    def test_earlier_close_score_tiebreak(self):
        hypotheses=self.result()['ranked_hypotheses'][:2]
        for h in hypotheses: h['signals']={k:.8 for k in h['signals']}
        hypotheses[0]['step_index']=3;hypotheses[1]['step_index']=2
        _,selected=rank_hypotheses(hypotheses,DiagnosisConfig())
        self.assertEqual(selected['step_index'],2)

    def test_low_confidence(self):
        self.taxonomy['category_confidence']=.1
        result=self.result()
        self.assertEqual(result['diagnosis_status'],'UNDETERMINED')
        self.assertIsNone(result['selected_hypothesis'])

    def test_propagation_order_and_confidence(self):
        result=self.result()
        self.assertEqual([s['step_index'] for s in result['propagation_chain']],[2,3,4])
        self.assertTrue(all(not s['causal_effect_verified'] for s in result['propagation_chain']))
        self.assertTrue(0<=result['diagnosis_confidence']<=1)
        components=result['confidence_components']
        self.assertAlmostEqual(result['diagnosis_confidence'],.4*.9+.3*.9+.2*.8+.1*components['margin'])

    def test_no_fabricated_replay(self):
        result=self.result()
        self.assertEqual(result['counterfactual_status'],'NOT_RUN');self.assertIsNone(result['replay_interpretation'])

    def test_replay_interpretation(self):
        request=ReplayRequest('t',1,'TOOL','verify')
        self.assertEqual(interpret_replay(request,ReplayResult('t',1,'TOOL',0,.8))['status'],'SUPPORTED')
        self.assertEqual(interpret_replay(request,ReplayResult('t',1,'TOOL',.8,.5))['status'],'REJECTED')
        with self.assertRaises(ValueError): interpret_replay(request,ReplayResult('other',1,'TOOL',0,1))
        rejected=self.result(replay_result=ReplayResult('opaque-id',2,'TOOL_WRONG_ARGUMENT',1,0))
        self.assertEqual(rejected['diagnosis_status'],'UNDETERMINED')
        self.assertIsNone(rejected['selected_hypothesis'])

    def test_ground_truth_invariance(self):
        expected=self.result()
        self.record.origin_step=4;self.record.fault_type='CTRL_LOOP';self.record.fault_injected=False
        self.record.outcome=Outcome('SUCCESS',1);self.record.injection_params={'operator':'SECRET'}
        for step in self.record.steps:
            step.is_root_cause=True;step.metadata={'origin_step':999,'operator':'SECRET'}
            step.input['fault_type']='SECRET';step.input['modified_value']='SECRET'
        self.assertEqual(self.result(),expected)

    def test_missing_subtype_renormalizes(self):
        self.taxonomy['predicted_subtype']=None;self.taxonomy['subtype_confidence']=None
        result=self.result();c=result['confidence_components']
        self.assertAlmostEqual(result['diagnosis_confidence'],(.4*c['root']+.3*c['category']+.1*c['margin'])/.8)

    def test_invalid_predictions(self):
        self.localization['ranked_steps'][0]['probability']=float('nan')
        with self.assertRaises(ValueError): self.result()


if __name__=='__main__': unittest.main()
