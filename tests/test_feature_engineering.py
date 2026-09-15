"""Feature tests are independent of the archive's broken generator module."""
import csv
import json
import tempfile
import unittest
import importlib.util
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from injector.schema import AgentConfig, AgentFaultRecord, Outcome, StepRecord
from feature_engineering import loader, step_features, trajectory_features, semantic_features, splitting
from feature_engineering.pipeline import build, validate_columns, dictionary
from feature_engineering.common import canonical


def record():
    return AgentFaultRecord('test','task',AgentConfig('test','test',0,[]),Outcome('SUCCESS',1),
        False,None,None,None,3,1,'NATURAL','TRAIN',[
            StepRecord(1,'LLM_CALL','arbitrary',{'task':'Find the answer'},'Thinking',status='success'),
            StepRecord(2,'TOOL_CALL','arbitrary',{'b':2,'a':1},{'error':'failure'},tool_name='any_tool',status='error'),
            StepRecord(3,'TOOL_CALL','arbitrary',{'a':1,'b':2},{'result':3},tool_name='any_tool',status='success')])


def extract(r):
    embeddings = semantic_features.Embeddings('unused',enabled=False)
    sem, aggregate = semantic_features.extract(r,loader.task_text(r),embeddings)
    steps = step_features.extract(r,sem)
    return trajectory_features.extract(r,steps,aggregate),steps


class FeatureTests(unittest.TestCase):
    def test_counts_duplicates_and_missing(self):
        row,_ = extract(record())
        self.assertEqual(row['total_steps'],3)
        self.assertEqual(row['duplicate_tool_call_count'],1)
        self.assertEqual(row['tool_success_rate'],.5)
        self.assertEqual(row['total_errors'],1)
        self.assertIsNone(row['average_tool_latency_ms'])
        self.assertEqual(canonical({'a':1,'b':2}),canonical({'b':2,'a':1}))

    def test_empty(self):
        r=record(); r.steps=[]; r.num_steps=0
        row,steps=extract(r)
        self.assertEqual(row['tool_calls_per_step'],0)
        self.assertEqual(steps,[])
        self.assertIsNone(row['avg_semantic_drift'])

    def test_separate_result_and_loop(self):
        r=record()
        r.steps[2].output=None
        r.steps.append(StepRecord(4,'TOOL_RESULT','arbitrary',None,{'result':3},tool_name='any_tool',status='success'))
        r.num_steps=4
        row,_=extract(r)
        self.assertEqual(row['tool_result_missing_count'],0)
        r.steps.append(deepcopy(r.steps[2])); r.steps[-1].step_index=5; r.num_steps=5
        row,_=extract(r)
        self.assertEqual(row['loop_indicator'],1)

    def test_unknown_status(self):
        r=record()
        for s in r.steps:
            s.status=None; s.output=None
        row,_=extract(r)
        self.assertIsNone(row['tool_success_rate'])
        self.assertIsNone(row['failed_step_count'])

    def test_future_invariance(self):
        r=record(); _,before=extract(r)
        r.steps.append(StepRecord(4,'RETRY','future','future','future',status='error'))
        r.num_steps+=1
        _,after=extract(r)
        for a,b in zip(before,after):
            self.assertEqual({k:a[k] for k in step_features.FEATURE_COLUMNS},
                             {k:b[k] for k in step_features.FEATURE_COLUMNS})

    def test_label_and_provenance_invariance(self):
        r=record(); before,steps_before=extract(r)
        r.fault_injected=True; r.fault_type='TOOL_WRONG_ARGUMENT'; r.origin_step=2
        r.outcome.status='FAIL'; r.injection_params={'injected_value':'secret'}
        r.steps[1].is_root_cause=True
        r.steps[1].metadata={'fault_type':'SECRET','original_value':'SECRET'}
        r.steps[1].input['modified_value']='SECRET'
        after,steps_after=extract(r)
        self.assertEqual([before[k] for k in trajectory_features.FEATURE_COLUMNS],
                         [after[k] for k in trajectory_features.FEATURE_COLUMNS])
        self.assertEqual([[s[k] for k in step_features.FEATURE_COLUMNS] for s in steps_before],
                         [[s[k] for k in step_features.FEATURE_COLUMNS] for s in steps_after])
        validate_columns()

    def test_roots_and_ordering(self):
        r=record(); r.fault_injected=True; r.origin_step=2
        loader.validate(r)
        self.assertTrue(r.steps[1].is_root_cause)
        r.steps=[r.steps[0],r.steps[2]]; r.num_steps=2
        loader.validate(r)
        self.assertTrue(r.steps[-1].is_root_cause)
        r.fault_injected=False
        with self.assertRaises(ValueError): loader.validate(r)
        r=record(); r.steps.reverse()
        with self.assertRaises(ValueError): loader.validate(r)

    def test_group_split(self):
        records=[]
        for i in range(20):
            for j in range(3):
                r=record(); r.task_id=str(i); r.trajectory_id=f'{i}-{j}'; records.append(r)
        rows=splitting.split(records)
        splitting.validate(rows)
        self.assertEqual(rows,splitting.split(records))
        self.assertEqual(len({r['task_id'] for r in rows if r['split']=='train'}),14)
        rows[0]['split']='invalid'
        with self.assertRaises(ValueError): splitting.validate(rows)

    def test_semantics(self):
        r=record()
        with tempfile.TemporaryDirectory() as directory:
            e=semantic_features.Embeddings(directory)
            for value,vector in [('Find the answer',[1,0]),('Thinking',[1,0])]:
                e.cache[e.key(value)]=vector
            rows,aggregate=semantic_features.extract(r,'Find the answer',e)
            self.assertEqual(rows[0]['semantic_drift'],0)
            self.assertEqual(aggregate['avg_task_step_similarity'],1)
        self.assertEqual(semantic_features.cosine([1,0],[-1,0]),-1)
        self.assertIsNone(semantic_features.cosine([0,0],[1,0]))
        self.assertEqual(semantic_features.text({'fault_type':'SECRET','id':'id_1'}),'')

    def test_load_skip_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.jsonl'
            line=json.dumps(asdict(record()))
            path.write_text(line+'\nBAD JSON\n'+line)
            trajectories,steps,manifest,issues=build(path,Path(directory)/'out',semantic=False)
            self.assertEqual((len(trajectories),len(steps)),(1,3))
            self.assertEqual(len(issues),2)
            for name,count in [('trajectory_features',1),('step_features',3),('split_manifest',1)]:
                with (Path(directory)/'out'/f'{name}.csv').open() as file:
                    self.assertEqual(len(list(csv.DictReader(file))),count)

    def test_current_data(self):
        records,_,issues=loader.load('dataset/agentfault-100.jsonl')
        self.assertEqual(issues,[])
        self.assertEqual(len(records),100)
        self.assertEqual(sum(len(r.steps) for r in records),815)
        real,_,issues=loader.load('data/trajectories')
        self.assertEqual(issues,[])
        self.assertEqual(len(real),3)
        for r in real:
            self.assertFalse(any(s.is_root_cause for s in r.steps))

    def test_dictionary_coverage(self):
        entries=dictionary()
        for level,module in [('step',step_features),('trajectory',trajectory_features)]:
            for column in module.FEATURE_COLUMNS:
                self.assertIn(f'{level}.{column}',entries)

    def test_full_dataset_csv_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            rows,steps,manifest,issues=build('dataset/agentfault-100.jsonl',directory,semantic=False)
            self.assertEqual((len(rows),len(steps),len(manifest)),(100,815,100))
            self.assertEqual(issues,[])
            self.assertEqual(sum(s['is_root_cause'] for s in steps),100)
            with (Path(directory)/'step_features.csv').open() as file:
                self.assertEqual(len(list(csv.DictReader(file))),815)

    @unittest.skipUnless(importlib.util.find_spec('pandas') and importlib.util.find_spec('sklearn'), 'Optional preprocessing dependencies unavailable')
    def test_train_only_preprocessing(self):
        import pandas as pd
        from feature_engineering.preprocessing import build_preprocessor
        rows=[]
        for value in [1.,float('nan'),3.]:
            row={c:value for c in trajectory_features.FEATURE_COLUMNS}
            row['dominant_agent']='train_agent'
            row['fault_type']='SECRET_LABEL'
            rows.append(row)
        train=pd.DataFrame(rows)
        transform=build_preprocessor(standardize=True)
        transform.fit(train)
        validation=train.copy()
        validation['total_steps']=999
        validation['dominant_agent']='unseen_agent'
        transform.transform(validation)
        self.assertEqual(transform.named_transformers_['numeric'].named_steps['impute'].statistics_[0],2)
        self.assertFalse(any('fault_type' in x for x in transform.get_feature_names_out()))


if __name__=='__main__':
    unittest.main()
