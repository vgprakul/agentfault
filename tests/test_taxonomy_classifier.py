"""Contracts and behavior tests; no dependence on exact model accuracy."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from injector.faults import FaultType
from feature_engineering.trajectory_features import FEATURE_COLUMNS
from ml.taxonomy.data import feature_frame, load_dataset
from ml.taxonomy.inference import TaxonomyPredictor, predict_taxonomy
from ml.taxonomy.train import train
from ml.taxonomy.utils import CATEGORIES, SUBTYPE_TO_CATEGORY, validate_features


class TaxonomyTests(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(set(SUBTYPE_TO_CATEGORY),{s.value for s in FaultType})
        self.assertEqual(SUBTYPE_TO_CATEGORY[FaultType.TOOL_WRONG_ARGUMENT.value],'TOOL')
        self.assertEqual(SUBTYPE_TO_CATEGORY[FaultType.CTRL_LOOP.value],'CONTROL')
        self.assertEqual(SUBTYPE_TO_CATEGORY[FaultType.PLAN_MISSING_STEP.value],'PLANNING')

    def test_leakage(self):
        validate_features()
        for label in ['fault_type','trajectory_id','origin_step','split','injection_operator','source_filename']:
            with self.assertRaises(ValueError):
                validate_features(list(FEATURE_COLUMNS)+[label])
        rows,_=load_dataset()
        a=feature_frame(rows)
        rows['fault_type']='SECRET'
        rows['trajectory_id']='SECRET'
        rows['origin_step']=999
        pd.testing.assert_frame_equal(a,feature_frame(rows))

    def test_existing_split(self):
        rows,info=load_dataset()
        self.assertEqual(info['split_counts'],{'train':70,'validation':15,'test':15})
        manifest=pd.read_csv('data/features/split_manifest.csv').set_index('trajectory_id')['split'].to_dict()
        self.assertTrue(all(manifest[r.trajectory_id]==r.split for r in rows.itertuples()))
        self.assertEqual(rows.groupby('task_id')['split'].nunique().max(),1)
        self.assertFalse(set(rows.loc[rows['split']=='train','trajectory_id']) & set(rows.loc[rows['split']=='test','trajectory_id']))

    def test_missing_feature_rejected(self):
        frame=pd.read_csv('data/features/trajectory_features.csv')
        with self.assertRaises(ValueError):
            feature_frame(frame.drop(columns=FEATURE_COLUMNS[0]))

    def test_manifest_integrity_and_clean_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            features=Path(directory)/'features.csv'; manifest=Path(directory)/'manifest.csv'
            frame=pd.read_csv('data/features/trajectory_features.csv').head(3).copy()
            frame['fault_injected']=0; frame['fault_type']='NONE'; frame['outcome_status']='SUCCESS'
            frame.to_csv(features,index=False)
            assignments=frame[['trajectory_id','task_id']].copy(); assignments['split']='train'; assignments.to_csv(manifest,index=False)
            rows,info=load_dataset(features,manifest)
            self.assertEqual(len(rows),0)
            self.assertEqual(info['excluded_unlabelled_or_clean'],3)
            assignments.loc[assignments.index[1],'task_id']=assignments.iloc[0]['task_id']
            assignments.loc[assignments.index[1],'split']='test'; assignments.to_csv(manifest,index=False)
            with self.assertRaises(ValueError): load_dataset(features,manifest)

    def test_small_training_reload_probabilities_missing_subtype(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            template=pd.read_csv('data/features/trajectory_features.csv').iloc[0].to_dict()
            rows=[]
            for i in range(16):
                row=dict(template)
                row.update(trajectory_id=f'sample-{i}',task_id=f'task-{i}',fault_injected=1,
                    fault_type='TOOL_WRONG_TOOL' if i%2 else 'PLAN_MISSING_STEP',outcome_status='FAIL',
                    total_steps=(5 if i%2 else 8) if i<8 else 1000+i)
                rows.append(row)
            frame=pd.DataFrame(rows)
            features=directory/'features.csv'; frame.to_csv(features,index=False)
            assignments=frame[['trajectory_id','task_id']].copy()
            assignments['split']=['train']*8+['validation']*4+['test']*4
            manifest=directory/'manifest.csv'; assignments.to_csv(manifest,index=False)
            model_dir=directory/'model'
            candidate=[('Logistic Regression',{'C':1},LogisticRegression(max_iter=1000,class_weight='balanced'),True)]
            # This test isolates training/artifact behavior; actual full evaluation is run by the CLI.
            with patch('ml.taxonomy.train.candidates',return_value=candidate),patch('ml.taxonomy.train.evaluate',return_value={}) as evaluate:
                train(features,manifest,model_dir,directory/'results')
                evaluate.assert_called_once()
            predictor=TaxonomyPredictor(model_dir)
            metadata=predictor.bundle['metadata']
            preprocessing=predictor.bundle['pipeline'].named_steps['preprocessing']
            self.assertEqual(preprocessing.named_transformers_['numeric'].named_steps['impute'].statistics_[0],6.5)
            self.assertEqual(set(metadata['training_trajectory_ids']),set(frame.iloc[:8]['trajectory_id']))
            self.assertEqual(set(metadata['classes_trained']),{'TOOL','PLANNING'})
            self.assertEqual(len(metadata['categories_absent']),4)
            prediction=predict_taxonomy(frame.iloc[-1],model_dir)
            self.assertIn(prediction['predicted_category'],CATEGORIES)
            self.assertIsNone(prediction['predicted_subtype'])
            self.assertIsNone(prediction['subtype_confidence'])
            probabilities=list(prediction['category_probabilities'].values())
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertAlmostEqual(sum(probabilities),1,places=5)
            self.assertEqual(set(prediction['category_probabilities']),{'TOOL','PLANNING'})
            # Mutating every label/identifier cannot change probabilities.
            mutated=frame.iloc[-1].copy(); mutated['fault_type']='SECRET'; mutated['origin_step']=99
            self.assertEqual(prediction['category_probabilities'],predictor.predict(mutated)['category_probabilities'])


if __name__=='__main__':
    unittest.main()
