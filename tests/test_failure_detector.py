import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from feature_engineering.trajectory_features import FEATURE_COLUMNS
from ml.failure_detector.data import load_dataset
from ml.failure_detector.utils import outcome_to_label, validate_features, training_weights
from ml.failure_detector.evaluate import metrics, majority_baseline, threshold_analysis
from ml.failure_detector.train import train
from ml.failure_detector.inference import FailureDetector


class FailureDetectorTests(unittest.TestCase):
    def fixture(self, directory, single=False):
        rows=[]
        for i in range(12):
            row={c:float(i % 2) for c in FEATURE_COLUMNS}
            row.update(trajectory_id=f't{i}', task_id=f'group{i}', dominant_agent='agent',
                       outcome_status='FAIL' if single or i%2 else 'SUCCESS', fault_injected=False)
            rows.append(row)
        frame=pd.DataFrame(rows)
        frame['split']=['train']*8+['validation']*2+['test']*2
        features=Path(directory)/'features.csv'; manifest=Path(directory)/'manifest.csv'
        frame.to_csv(features,index=False)
        frame[['trajectory_id','task_id','split']].to_csv(manifest,index=False)
        return features,manifest

    def test_mapping(self):
        for label in ['SUCCESS','success','PASS']: self.assertEqual(outcome_to_label(label),0)
        for label in ['FAIL','FAILURE','ERROR','PARTIAL_SUCCESS']: self.assertEqual(outcome_to_label(label),1)
        self.assertIsNone(outcome_to_label('unknown'))

    def test_leakage(self):
        validate_features()
        for column in ['outcome_status','origin_step','failure_label','fault_type','trajectory_id']:
            with self.assertRaises(ValueError): validate_features(FEATURE_COLUMNS+[column])

    def test_split_and_outcome(self):
        with tempfile.TemporaryDirectory() as temp:
            features,manifest=self.fixture(temp)
            rows,info=load_dataset(features,manifest)
            self.assertEqual(rows['failure_label'].sum(),6)  # All fault flags are false.
            self.assertTrue(info['split_integrity']['valid'])
            assignment=pd.read_csv(manifest)
            assignment.loc[10,'task_id']=assignment.loc[0,'task_id']
            assignment.to_csv(manifest,index=False)
            with self.assertRaises(ValueError): load_dataset(features,manifest)

    def test_weights_training_only(self):
        rows=pd.DataFrame({'split':['train']*4,'failure_label':[0,1,1,1]})
        self.assertAlmostEqual(training_weights(rows)['scale_pos_weight'],1/3)
        rows.loc[0,'split']='test'
        with self.assertRaises(ValueError): training_weights(rows)

    def test_metrics_and_thresholds(self):
        result=metrics([0,0,1,1],[.1,.6,.4,.9])
        for key in ['precision','recall','f1','false_positive_rate','false_negative_rate']:
            self.assertEqual(result[key],.5)
        table,_=threshold_analysis([0,0,1,1],[.1,.6,.4,.9])
        self.assertEqual(table.loc[table.threshold==.3,'recall'].iloc[0],1)
        self.assertEqual(majority_baseline([0,1,1],[0,1,1])['accuracy'],2/3)
        self.assertIsNone(metrics([1,1],[.8,.9])['roc_auc'])
        for labels,scores in [([.5],[.2]),([1],[np.nan]),([1],[1.1])]:
            with self.assertRaises(ValueError): metrics(labels,scores)

    def test_single_class_training_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            features,manifest=self.fixture(temp,True)
            with self.assertRaisesRegex(ValueError,'both actual'):
                train(features,manifest,Path(temp)/'model',Path(temp)/'results')
            self.assertFalse((Path(temp)/'model'/'best_failure_detector.joblib').exists())

    def test_train_reload_and_training_only_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp:
            features,manifest=self.fixture(temp)
            frame=pd.read_csv(features)
            frame.loc[8:,'total_steps']=9999
            frame.to_csv(features,index=False)
            candidate=[('Logistic Regression',{'C':1},LogisticRegression(class_weight='balanced'),True)]
            with patch('ml.failure_detector.train.candidates',return_value=candidate):
                train(features,manifest,Path(temp)/'model',Path(temp)/'results')
            detector=FailureDetector(Path(temp)/'model')
            preprocess=detector.bundle['pipeline'].named_steps['preprocessing']
            numerical=preprocess.transformers_[0][2]
            median=preprocess.named_transformers_['numeric'].named_steps['impute'].statistics_[numerical.index('total_steps')]
            self.assertEqual(median,.5)
            prediction=detector.predict(frame.iloc[0])
            self.assertTrue(np.isfinite(prediction['failure_probability']))
            self.assertGreaterEqual(prediction['failure_probability'],0)
            self.assertLessEqual(prediction['failure_probability'],1)
            self.assertTrue(detector.predict(frame.iloc[0],0)['predicted_failure'])
            self.assertFalse(detector.predict(frame.iloc[0],1)['predicted_failure'])
            self.assertNotIn('actual_outcome',prediction)


if __name__=='__main__': unittest.main()
