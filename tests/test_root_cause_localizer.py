"""Root-label, grouped-split, model persistence and ranking behavior tests."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from feature_engineering.step_features import FEATURE_COLUMNS
from ml.localizer.data import feature_frame, load_dataset
from ml.localizer.inference import RootCauseLocalizer, predict_root_cause
from ml.localizer.ranking import localization_metrics, rank_steps
from ml.localizer.train import train
from ml.localizer.utils import training_weights, validate_features


def fixture(directory):
    """Synthetic unit-test fixture only; never added to the repository dataset."""
    template = pd.read_csv('data/features/step_features.csv').iloc[0].to_dict()
    steps,metadata,manifest = [],[],[]
    for i in range(10):
        trajectory_id,task_id = f'fixture-{i}',f'task-{i}'
        split = 'train' if i<6 else 'validation' if i<8 else 'test'
        metadata.append(dict(trajectory_id=trajectory_id,task_id=task_id,fault_injected=1,origin_step=2))
        manifest.append(dict(trajectory_id=trajectory_id,task_id=task_id,split=split))
        for index in range(1,5):
            row=dict(template)
            row.update(trajectory_id=trajectory_id,task_id=task_id,step_index=index,is_root_cause=int(index==2),
                       has_error=int(index==2),input_length_chars=10 if i<6 else 1000)
            steps.append(row)
    features=Path(directory)/'steps.csv'; meta=Path(directory)/'metadata.csv'; splits=Path(directory)/'splits.csv'
    pd.DataFrame(steps).to_csv(features,index=False)
    pd.DataFrame(metadata).to_csv(meta,index=False)
    pd.DataFrame(manifest).to_csv(splits,index=False)
    return features,splits,meta


class LocalizerTests(unittest.TestCase):
    def test_leakage_contract(self):
        validate_features()
        for column in ['is_root_cause','origin_step','fault_type','fault_category','injected_fault','modified_value',
                       'original_value','injection_params','outcome_status','split','step_index','normalized_step_position','trajectory_id']:
            with self.assertRaises(ValueError): validate_features(list(FEATURE_COLUMNS)+[column])
        rows,_=load_dataset()
        before=feature_frame(rows)
        rows['is_root_cause']=1-rows['is_root_cause']; rows['origin_step']=999; rows['fault_type']='SECRET'
        pd.testing.assert_frame_equal(before,feature_frame(rows))

    def test_current_dataset_and_grouping(self):
        rows,info=load_dataset()
        self.assertEqual((info['usable_faulty_trajectories'],info['positive_steps'],info['negative_steps']),(100,100,715))
        self.assertEqual(info['issues'],[])
        self.assertEqual(info['split_counts']['train']['steps'],572)
        for column in ['trajectory_id','task_id']:
            self.assertEqual(rows.groupby(column)['split'].nunique().max(),1)

    def test_split_violation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            features,manifest,meta=fixture(directory)
            rows=pd.read_csv(features)
            rows['split']='train'; rows.loc[0,'split']='test'; rows.to_csv(features,index=False)
            with self.assertRaises(ValueError): load_dataset(features,manifest,meta)
            with self.assertRaises(ValueError): load_dataset(features,None,meta)

    def test_base_task_grouping(self):
        with tempfile.TemporaryDirectory() as directory:
            features,manifest,meta=fixture(directory)
            rows=pd.read_csv(meta); rows['base_task_id']='shared-base'; rows.to_csv(meta,index=False)
            with self.assertRaises(ValueError): load_dataset(features,manifest,meta)

    def test_training_only_weights(self):
        rows,_=load_dataset()
        result=training_weights(rows.loc[rows['split']=='train'])
        self.assertEqual(result['positive_training_steps'],70)
        self.assertEqual(result['negative_training_steps'],502)
        self.assertAlmostEqual(result['scale_pos_weight'],502/70)
        with self.assertRaises(ValueError): training_weights(rows)

    def test_ranking_metrics(self):
        rows=[]; probabilities=[]
        patterns=[[.9,.8,.7,.6,.5],[.8,.9,.7,.6,.5],[.5,.9,.8,.7,.1]]
        for i,pattern in enumerate(patterns):
            for j,p in enumerate(pattern,1):
                rows.append(dict(trajectory_id=str(i),task_id=str(i),step_index=j,is_root_cause=int(j==1)))
                probabilities.append(p)
        metrics,steps,trajectories=localization_metrics(pd.DataFrame(rows),probabilities)
        self.assertAlmostEqual(metrics['exact_step_accuracy'],1/3)
        self.assertAlmostEqual(metrics['top3_accuracy'],2/3)
        self.assertAlmostEqual(metrics['mrr'],(1+.5+.25)/3)
        self.assertAlmostEqual(metrics['mean_absolute_step_distance'],2/3)
        self.assertEqual(metrics['top5_accuracy'],1)
        self.assertEqual(trajectories['actual_root_cause_rank'].tolist(),[1,2,4])

    def test_ties_and_invalid_probabilities(self):
        rows=pd.DataFrame([dict(trajectory_id='a',task_id='a',step_index=i) for i in [3,1,2]])
        self.assertEqual(rank_steps(rows,[.5,.5,.5])['step_index'].tolist(),[1,2,3])
        for values in [[.1,np.nan,.2],[.1,1.1,.2],[.1,.2]]:
            with self.assertRaises(ValueError): rank_steps(rows,values)

    def test_malformed_and_clean_never_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            features,manifest,meta=fixture(directory)
            rows=pd.read_csv(features); metadata=pd.read_csv(meta)
            rows.loc[rows['trajectory_id']=='fixture-0','is_root_cause']=0
            rows.loc[(rows['trajectory_id']=='fixture-1')&(rows['step_index']==3),'is_root_cause']=1
            rows.loc[(rows['trajectory_id']=='fixture-2')&(rows['step_index']==3),'step_index']=2
            rows.loc[(rows['trajectory_id']=='fixture-3')&(rows['step_index']==3),'step_index']=np.nan
            metadata.loc[metadata['trajectory_id'].isin(['fixture-4','fixture-5']),'fault_injected']=0
            rows.loc[rows['trajectory_id']=='fixture-4','is_root_cause']=0
            rows.to_csv(features,index=False); metadata.to_csv(meta,index=False)
            before=features.read_bytes()
            valid,info=load_dataset(features,manifest,meta)
            self.assertEqual(info['clean_trajectories'],2)
            self.assertEqual(info['missing_root_cause_labels'],1)
            self.assertEqual(info['multiple_positive_trajectories'],1)
            self.assertEqual(info['invalid_trajectories'],4)
            self.assertEqual(info['usable_faulty_trajectories'],4)
            self.assertEqual(valid['is_root_cause'].sum(),4)
            self.assertEqual(features.read_bytes(),before)

    def test_small_model_reload_and_label_free_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            features,manifest,meta=fixture(directory)
            models=Path(directory)/'model'
            candidate=[('Logistic Regression',{'C':1},LogisticRegression(class_weight='balanced',max_iter=1000),True)]
            with patch('ml.localizer.train.candidates',return_value=candidate),patch('ml.localizer.train.evaluate',return_value={}) as evaluator:
                train(features,manifest,meta,models,Path(directory)/'results')
                evaluator.assert_called_once()
            localizer=RootCauseLocalizer(models)
            metadata=localizer.bundle['metadata']
            self.assertEqual(len(metadata['training_step_keys']),24)
            self.assertEqual(set(metadata['training_trajectory_ids']),{f'fixture-{i}' for i in range(6)})
            columns=localizer.bundle['pipeline'].named_steps['preprocessing'].transformers_[0][2]
            statistics=localizer.bundle['pipeline'].named_steps['preprocessing'].named_transformers_['numeric'].named_steps['impute'].statistics_
            self.assertEqual(statistics[columns.index('input_length_chars')],10)
            rows=pd.read_csv(features); rows=rows.loc[rows['trajectory_id']=='fixture-9'].drop(columns='is_root_cause')
            result=predict_root_cause(rows,models)
            probabilities=[s['probability'] for s in result['ranked_steps']]
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue(all(0<=p<=1 for p in probabilities))
            self.assertEqual(probabilities,sorted(probabilities,reverse=True))
            self.assertEqual(result['root_cause_probability'],max(probabilities))
            self.assertEqual(result['predicted_root_cause_step'],result['ranked_steps'][0]['step_index'])
            self.assertNotIn('is_root_cause',str(result))
            rows['is_root_cause']=1; rows['origin_step']=999; rows['fault_category']='SECRET'
            self.assertEqual(result,localizer.predict(rows))
            # A model scores each row independently; later supplied rows cannot alter earlier scores.
            np.testing.assert_allclose(localizer.score_steps(rows.iloc[:2]),localizer.score_steps(rows)[:2])


if __name__=='__main__':
    unittest.main()
