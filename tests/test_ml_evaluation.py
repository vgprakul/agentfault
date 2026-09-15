"""Deterministic metric/baseline fixtures, independent of real-model accuracy."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from ml.evaluation.taxonomy_metrics import evaluate_taxonomy, classification_metrics
from ml.evaluation.localization_metrics import evaluate_localization, localization_baseline
from ml.evaluation.model_comparison import majority_baseline, comparisons
from ml.evaluation.integrity import split_integrity
from ml.evaluation.review3_report import ordered_cache, evaluate_all


def taxonomy_fixture():
    actual=['TOOL','TOOL','CONTROL','CONTROL']
    predicted=['TOOL','CONTROL','CONTROL','CONTROL']
    probability=[.8,.3,.2,.1]
    return pd.DataFrame({'trajectory_id':['a','b','c','d'],'task_id':['a','b','c','d'],
        'actual_category':actual,'predicted_category':predicted,
        'category_probabilities':[json.dumps({'TOOL':p,'CONTROL':1-p}) for p in probability],
        'category_confidence':[max(p,1-p) for p in probability]})


def localization_fixture():
    rows=[]
    for i,pattern in enumerate([[.9,.8,.7,.6,.5],[.8,.9,.7,.6,.5],[.5,.9,.8,.7,.1]]):
        for step,p in enumerate(pattern,1):
            rows.append(dict(trajectory_id=str(i),task_id=str(i),step_index=step,
                             actual_is_root_cause=int(step==1),root_cause_probability=p))
    return pd.DataFrame(rows)


class EvaluationTests(unittest.TestCase):
    def test_taxonomy_metrics_and_order(self):
        result,buckets=evaluate_taxonomy(taxonomy_fixture())
        self.assertAlmostEqual(result['accuracy'],.75)
        self.assertAlmostEqual(result['macro_f1'],((2/3)+.8)/2)
        self.assertEqual(result['classes_evaluated'],['TOOL','CONTROL'])
        self.assertEqual(result['confusion_matrix_labels'],['TOOL','CONTROL'])
        self.assertEqual(result['confusion_matrix'],[[1,1],[0,2]])
        self.assertEqual(buckets['count'].sum(),4)
        self.assertIsNotNone(result['probability_metrics']['log_loss'])
        self.assertTrue(0<=result['probability_metrics']['ovr_roc_auc']<=1)

    def test_missing_probability_class_safe(self):
        frame=taxonomy_fixture().iloc[:2].copy()
        result,_=evaluate_taxonomy(frame)
        self.assertEqual(result['classes_evaluated'],['TOOL'])
        self.assertIsNone(result['probability_metrics']['ovr_roc_auc'])
        self.assertIn('CONTROL',result['probability_metrics']['skipped'])
        # Predicted-only CONTROL stays on the matrix axes so the error is counted.
        self.assertEqual(np.array(result['confusion_matrix']).sum(),2)

    def test_subtype_conditional_and_unavailable(self):
        frame=taxonomy_fixture()
        frame['actual_subtype']=['TOOL_WRONG_TOOL','TOOL_WRONG_ARGUMENT','CTRL_LOOP','CTRL_LOOP']
        frame['predicted_subtype']=['TOOL_WRONG_TOOL',None,'CTRL_LOOP',None]
        result,_=evaluate_taxonomy(frame,{'CONTROL':'fixture unavailable'})
        self.assertAlmostEqual(result['subtype']['accuracy'],.5)
        self.assertAlmostEqual(result['subtype']['hierarchical_accuracy'],.5)
        self.assertAlmostEqual(result['subtype']['conditional_on_correct_category_accuracy'],2/3)
        self.assertEqual(len(result['subtype']['classes_evaluated']),3)

    def test_localization_metrics(self):
        result,steps,rankings=evaluate_localization(localization_fixture())
        self.assertAlmostEqual(result['exact_step_accuracy'],1/3)
        self.assertAlmostEqual(result['top3_accuracy'],2/3)
        self.assertAlmostEqual(result['mrr'],(1+.5+.25)/3)
        self.assertAlmostEqual(result['mean_absolute_step_distance'],2/3)
        self.assertEqual(result['median_absolute_step_distance'],1)
        self.assertEqual(result['top5_accuracy'],1)

    def test_majority_baseline_training_only(self):
        result=majority_baseline(['TOOL','TOOL','CONTROL'],['CONTROL','CONTROL','TOOL'])
        self.assertEqual(result['majority_class'],'TOOL')
        self.assertAlmostEqual(result['accuracy'],1/3)
        self.assertAlmostEqual(result['macro_f1'],.25)

    def test_random_and_earliest_baselines(self):
        rows=localization_fixture().rename(columns={'actual_is_root_cause':'is_root_cause'})
        first=localization_baseline(rows,'random',seed=42)
        self.assertEqual(first,localization_baseline(rows.sample(frac=1,random_state=1),'random',seed=42))
        rng=np.random.default_rng(42)
        ranks=[]
        for _ in range(3):
            order=rng.permutation(5)
            ranks.append(int((order>order[0]).sum())+1)
        self.assertAlmostEqual(first['mrr'],np.mean([1/r for r in ranks]))
        earliest=localization_baseline(rows,'earliest')
        self.assertEqual(earliest['exact_step_accuracy'],1)
        self.assertEqual(earliest['mrr'],1)

    def test_split_overlap_detection(self):
        manifest=pd.DataFrame({'trajectory_id':['a','b'],'task_id':['t1','t2'],'split':['train','test']})
        frame=manifest.drop(columns='split')
        self.assertTrue(split_integrity(manifest,{'steps':frame})['valid'])
        frame=frame.assign(base_task_id='shared')
        result=split_integrity(manifest,{'steps':frame})
        self.assertFalse(result['valid'])
        self.assertGreater(result['number_of_overlaps'],0)
        duplicated=pd.concat([manifest,manifest.iloc[[0]].assign(split='test')])
        self.assertFalse(split_integrity(duplicated,{})['valid'])

    def test_malformed_predictions(self):
        frame=taxonomy_fixture()
        frame.loc[0,'category_probabilities']='bad JSON'
        with self.assertRaises(ValueError): evaluate_taxonomy(frame)
        with self.assertRaises(ValueError): evaluate_taxonomy(pd.concat([taxonomy_fixture(),taxonomy_fixture()]))
        local=localization_fixture(); local.loc[0,'root_cause_probability']=np.nan
        with self.assertRaises(ValueError): evaluate_localization(local)
        local=localization_fixture(); local.loc[local['trajectory_id']=='0','actual_is_root_cause']=0
        result,_,_=evaluate_localization(local)
        self.assertEqual(len(result['skipped_trajectories']),1)
        self.assertEqual(result['faulty_trajectories'],2)

    def test_prediction_cache_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache.csv'
            taxonomy_fixture().to_csv(path,index=False)
            with self.assertRaises(ValueError): ordered_cache(path,taxonomy_fixture().iloc[:2],['trajectory_id'])

    def test_missing_model_does_not_train(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError,'train_taxonomy_classifier.py'):
                evaluate_all(model_dir=directory,output_dir=Path(directory)/'results')

    def test_missing_comparisons_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            table,missing=comparisons(Path(directory)/'taxonomy.csv',Path(directory)/'localizer.csv',{'taxonomy':'Random Forest','localizer':'XGBoost'})
            self.assertTrue(table.empty)
            self.assertEqual(len(missing),2)


if __name__=='__main__':
    unittest.main()
