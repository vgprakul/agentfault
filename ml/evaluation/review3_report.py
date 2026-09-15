"""One non-training evaluation entry point with checked prediction reuse."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd

from ml.taxonomy.data import load_dataset as load_taxonomy, feature_frame as taxonomy_features
from ml.localizer.data import load_dataset as load_localizer, feature_frame as step_features
from ml.taxonomy.inference import TaxonomyPredictor
from ml.localizer.inference import RootCauseLocalizer
from ml.taxonomy.evaluate import feature_importance as taxonomy_importance
from ml.localizer.evaluate import feature_importance as localizer_importance
from ml.localizer.utils import save_json, file_hash
from .taxonomy_metrics import evaluate_taxonomy
from .localization_metrics import evaluate_localization, localization_baseline, breakdowns
from .model_comparison import comparisons, majority_baseline
from .integrity import split_integrity
from .plots import taxonomy_plots, localizer_plots, comparison_plots


def ordered_cache(path, expected, keys):
    cache=pd.read_csv(path,dtype={'trajectory_id':str,'task_id':str})
    if not set(keys)<=set(cache) or cache.duplicated(keys).any():
        raise ValueError(f'Malformed or duplicate prediction keys: {path}')
    wanted=pd.MultiIndex.from_frame(expected[keys])
    found=pd.MultiIndex.from_frame(cache[keys])
    if set(wanted)!=set(found):
        raise ValueError(f'Prediction coverage does not match eligible test rows: {path}')
    cache.index=found
    return cache.loc[wanted].reset_index(drop=True)


def taxonomy_predictions(predictor,test,path,output):
    fresh=pd.DataFrame(predictor.predict_many(test))
    if path.exists():
        cached=ordered_cache(path,test,['trajectory_id'])
        for column,expected in [('task_id',test['task_id']),('actual_category',test['fault_category']),('actual_subtype',test['fault_subtype'])]:
            if column not in cached or cached[column].fillna('').tolist()!=expected.fillna('').tolist():
                raise ValueError(f'Saved taxonomy {column} disagrees with ground truth')
        for column in ['predicted_category','predicted_subtype']:
            if cached[column].fillna('').tolist()!=fresh[column].fillna('').tolist():
                raise ValueError('Saved taxonomy predictions differ from current frozen model')
        for column in ['category_confidence','subtype_confidence']:
            if not np.allclose(pd.to_numeric(cached[column]),pd.to_numeric(fresh[column]),atol=1e-6,equal_nan=True):
                raise ValueError('Saved taxonomy confidence differs from frozen model')
        for old,new in zip(cached['category_probabilities'],fresh['category_probabilities']):
            old=json.loads(old)
            if set(old)!=set(new) or not all(np.isclose(old[k],new[k],atol=1e-6) for k in new):
                raise ValueError('Saved taxonomy probabilities differ from frozen model')
        return cached,str(path)
    fresh['task_id']=test['task_id'].to_numpy()
    fresh['actual_category']=test['fault_category'].to_numpy()
    fresh['actual_subtype']=test['fault_subtype'].to_numpy()
    fresh['category_probabilities']=fresh['category_probabilities'].map(json.dumps)
    destination=output/'taxonomy_predictions.csv'
    fresh.to_csv(destination,index=False)
    return fresh,str(destination)


def localizer_predictions(localizer,test,path,output):
    probabilities=localizer.score_steps(test)
    if path.exists():
        cached=ordered_cache(path,test,['trajectory_id','step_index'])
        if cached['task_id'].tolist()!=test['task_id'].tolist() or cached['actual_is_root_cause'].tolist()!=test['is_root_cause'].tolist():
            raise ValueError('Saved localizer labels/task IDs disagree with step data')
        if not np.allclose(cached['root_cause_probability'],probabilities,atol=1e-6):
            raise ValueError('Saved localizer probabilities differ from frozen model')
        return cached,str(path)
    fresh=test[['trajectory_id','task_id','step_index','is_root_cause']].rename(columns={'is_root_cause':'actual_is_root_cause'}).copy()
    fresh['root_cause_probability']=probabilities
    destination=output/'localizer_predictions.csv'
    fresh.to_csv(destination,index=False)
    return fresh,str(destination)


def audit_models(taxonomy,localizer,tax_rows,local_rows,tax_metrics,local_metrics):
    terms=['fault','origin','root_cause','injected','label','target','subtype','category','outcome']
    audits={}
    for name,model,rows,features,importance in [
        ('taxonomy',taxonomy.bundle,tax_rows,taxonomy_features,taxonomy_importance(taxonomy.bundle)),
        ('localizer',localizer.bundle,local_rows,step_features,localizer_importance(localizer.bundle['pipeline']))]:
        names=list(model['feature_columns'])+model['pipeline'].named_steps['preprocessing'].get_feature_names_out().tolist()
        flagged=sorted({n for n in names if any(term in n.lower() for term in terms)})
        training=features(rows.loc[rows['split']=='train'])
        test=features(rows.loc[rows['split']=='test'])
        repeated=int(pd.util.hash_pandas_object(test,index=False).isin(set(pd.util.hash_pandas_object(training,index=False))).sum())
        top=importance.head(15).to_dict(orient='records')
        warnings=[]
        if flagged:
            warnings.append('Feature-name terms require inspection, not an automatic leakage verdict: '+', '.join(flagged))
        if repeated:
            warnings.append(f'{repeated}/{len(test)} test predictor vectors also occur in training despite task grouping.')
        sentinels=[r['feature'] for r in top if any(token in r['feature'] for token in ['wrong_tool','wrong_agent','invalid_'])]
        if sentinels:
            warnings.append('Observable synthetic sentinel values among top features: '+', '.join(sentinels))
        if name=='taxonomy':
            size_sum=float(importance.loc[importance['feature'].isin(['numeric__average_step_output_length','numeric__total_output_chars','numeric__average_step_input_length','numeric__total_input_chars']),'importance'].sum())
            warnings.append(f'Four payload-size features contribute {size_sum:.1%} of importance and may expose injection/template patterns.')
        score=tax_metrics['accuracy'] if name=='taxonomy' else local_metrics['exact_step_accuracy']
        if score>=.95:
            warnings.append('WARNING: performance is unusually high; inspect for target/proxy leakage.')
        audits[name]={'suspicious_feature_names':flagged,'top_feature_importances':top,
                      'matching_test_vectors':repeated,'test_vectors':len(test),'warnings':warnings}
    return audits


def markdown_table(frame,columns=None):
    frame=frame[columns] if columns else frame
    lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
    for row in frame.itertuples(index=False,name=None):
        values=[('' if pd.isna(v) else f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)) for v in row]
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def write_report(output,summary,tax,loc,baselines,category,length,tax_errors,local_errors,audit,counts,comparison):
    per_class=pd.DataFrame(tax['per_class']).T.reset_index(names='category')
    text=['# Review 3: AgentFault ML Evaluation','',
        '## Dataset and split','',
        f"{counts['taxonomy']['usable_rows']} failure-labelled trajectories; {counts['localizer']['usable_step_rows']} valid faulty step rows. Existing grouped assignments preserved. Split integrity passed.",
        markdown_table(pd.DataFrame(counts['localizer']['split_counts']).T.reset_index(names='split')),'',
        '## Taxonomy classifier','',
        f"Best model: **{summary['taxonomy_classifier']['best_model']}**. Test samples: {tax['samples']}. Accuracy {tax['accuracy']:.4f}, Macro-F1 {tax['macro_f1']:.4f}, Weighted-F1 {tax['weighted_f1']:.4f}.",
        markdown_table(per_class), '',
        f"Subtype accuracy {tax.get('subtype',{}).get('accuracy')}; conditional on correct category {tax.get('subtype',{}).get('conditional_on_correct_category_accuracy')}; hierarchical accuracy {tax.get('subtype',{}).get('hierarchical_accuracy')}.",
        'Unavailable subtype models: '+json.dumps(tax['subtype'].get('unavailable_models',{})), '',
        'Main category errors:',markdown_table(tax_errors.head(5),['trajectory_id','actual_category','predicted_category','confidence']),'',
        '## Root-cause localizer','',
        f"Best model: **{summary['root_cause_localizer']['best_model']}**. Test trajectories: {loc['faulty_trajectories']}. Exact-step {loc['exact_step_accuracy']:.4f}, Top-3 {loc['top3_accuracy']:.4f}, Top-5 {loc['top5_accuracy']}, MRR {loc['mrr']:.4f}, mean distance {loc['mean_absolute_step_distance']:.4f}, median distance {loc['median_absolute_step_distance']:.4f}.",
        f"Secondary step metrics: PR-AUC {loc['binary']['pr_auc']:.4f}, recall {loc['binary']['recall']:.4f}, F1 {loc['binary']['f1']:.4f}.",'',
        'Main localization errors:',markdown_table(local_errors.head(6),['trajectory_id','actual_root_cause_step','predicted_root_cause_step','actual_root_cause_rank','absolute_step_distance']),'',
        '### Performance by failure category',markdown_table(category),'',
        '### Performance by trajectory length',loc['length_bins']['definition'],markdown_table(length),'',
        '## Baselines',markdown_table(baselines), '',
        f'Majority label uses training counts only (taxonomy order resolves ties). Random-step is a uniform random ranking per trajectory with seed {summary["baselines"]["random_step"]["baseline_seed"]}; earliest-step orders by recorded index. Baseline labels are used only for scoring.','',
        '## Candidate model comparison',markdown_table(comparison.reindex(columns=['module','model','validation_macro_f1','validation_mrr','validation_exact_step_accuracy','selected'])),'',
        'Selection is from saved validation results only. No retraining or test-based model selection occurred.','',
        '## Limitations','',
        'Only 15 test trajectories per model; category and length subgroups are very small. No claims of causality or production readiness. Probabilities are uncalibrated. Subtype macro metrics use only actual subtypes present in this split, unlike the earlier all-20-subtype report. Top-5 uses only trajectories with at least five steps.','']
    for module,report in audit.items():
        text.extend(f'- {module}: {warning}' for warning in report['warnings'])
    text.extend(['','Probability metric skips: '+json.dumps(tax['probability_metrics']['skipped']),
                 'Machine-readable metrics, confidence buckets, error tables, feature audits and plots are in this directory.'])
    text.extend(['','## Implementation and files','',
                 'New source files: ml/evaluation/__init__.py, taxonomy_metrics.py, localization_metrics.py, plots.py, model_comparison.py, integrity.py, review3_report.py, README.md; scripts/evaluate_ml.py; tests/test_ml_evaluation.py.',
                 'No pre-existing source files, trained models, feature datasets, or module-specific prediction files were modified.','',
                 'Generated evaluation artifacts:'])
    text.extend('- '+p.name for p in sorted(output.iterdir()) if p.is_file() and p.name!='REVIEW3_ML_REPORT.md')
    text.append('- REVIEW3_ML_REPORT.md')
    (output/'REVIEW3_ML_REPORT.md').write_text('\n'.join(text),encoding='utf-8')


def evaluate_all(feature_dir='data/features',model_dir='models',results_dir='results',output_dir='results/evaluation',seed=42):
    features,models,results,output=map(Path,[feature_dir,model_dir,results_dir,output_dir])
    output.mkdir(parents=True,exist_ok=True)
    print('=== AgentFault ML Evaluation ===',flush=True)
    for name,file,command in [('Taxonomy','taxonomy/best_model.joblib','train_taxonomy_classifier.py'),('Localizer','localizer/best_localizer.joblib','train_root_cause_localizer.py')]:
        if not (models/file).exists():
            raise FileNotFoundError(f'{name} model not found. Run: python scripts/{command}')
    manifest=features/'split_manifest.csv'; trajectory=features/'trajectory_features.csv'; steps=features/'step_features.csv'
    print('Checking split integrity...',flush=True)
    raw={name:pd.read_csv(path,dtype={'trajectory_id':str,'task_id':str,'base_task_id':str}) for name,path in [('trajectory',trajectory),('steps',steps)]}
    assignments=pd.read_csv(manifest,dtype={'trajectory_id':str,'task_id':str})
    integrity=split_integrity(assignments,raw)
    save_json(output/'split_integrity.json',integrity)
    if not integrity['valid']:
        raise ValueError('Split integrity FAILED; inspect split_integrity.json')
    taxonomy=TaxonomyPredictor(models/'taxonomy'); localizer=RootCauseLocalizer(models/'localizer')
    tax_meta=taxonomy.bundle['metadata']; local_meta=localizer.bundle['metadata']
    split_map=assignments.set_index('trajectory_id')['split'].str.lower()
    bad=[tid for meta in [tax_meta,local_meta] for tid in meta['training_trajectory_ids'] if split_map.get(tid)!='train']
    if bad:
        integrity.update(valid=False,number_of_overlaps=len(set(bad)),overlap_examples=sorted(set(bad))[:20])
        save_json(output/'split_integrity.json',integrity)
        raise ValueError('Saved model training IDs are outside the training split')
    if tax_meta['feature_file_sha256']!=file_hash(trajectory) or tax_meta['manifest_sha256']!=file_hash(manifest):
        raise ValueError('Taxonomy inputs differ from trained model metadata')
    for key,path in [('step_features',steps),('manifest',manifest),('trajectory_metadata',trajectory)]:
        if local_meta['input_sha256'][key]!=file_hash(path):
            raise ValueError(f'Localizer input changed: {key}')
    print('PASSED',flush=True)
    tax_rows,tax_info=load_taxonomy(trajectory,manifest)
    local_rows,local_info=load_localizer(steps,manifest,trajectory)
    tax_test=tax_rows.loc[tax_rows['split']=='test'].reset_index(drop=True)
    local_test=local_rows.loc[local_rows['split']=='test'].reset_index(drop=True)
    if tax_test.empty or local_test.empty:
        raise ValueError('Both modules require nonempty eligible test data')
    tax_pred,tax_source=taxonomy_predictions(taxonomy,tax_test,results/'taxonomy/test_predictions.csv',output)
    local_pred,local_source=localizer_predictions(localizer,local_test,results/'localizer/test_step_predictions.csv',output)
    tax,buckets=evaluate_taxonomy(tax_pred,tax_meta.get('subtype_unavailable'))
    loc,recomputed_steps,rankings=evaluate_localization(local_pred)
    if 'predicted_rank' in local_pred:
        checked=local_pred.merge(recomputed_steps[['trajectory_id','step_index','predicted_rank']],on=['trajectory_id','step_index'],suffixes=('','_checked'))
        if not checked['predicted_rank'].eq(checked['predicted_rank_checked']).all():
            raise ValueError('Saved localizer step ranks disagree with probabilities')
    saved_rankings=results/'localizer/test_trajectory_rankings.csv'
    if saved_rankings.exists():
        saved=ordered_cache(saved_rankings,rankings,['trajectory_id'])
        for column in ['actual_root_cause_step','predicted_root_cause_step','actual_root_cause_rank','absolute_step_distance']:
            if not np.array_equal(saved[column].to_numpy(),rankings[column].to_numpy()):
                raise ValueError('Saved trajectory rankings disagree with recomputed ranks')
    majority=majority_baseline(tax_rows.loc[tax_rows['split']=='train','fault_category'],tax_pred['actual_category'])
    random=localization_baseline(local_test,'random',seed)
    earliest=localization_baseline(local_test,'earliest',seed)
    baseline_rows=[{'module':'taxonomy','model':'Trained '+tax_meta['best_model'],'accuracy':tax['accuracy'],'macro_f1':tax['macro_f1']},
        {'module':'taxonomy','model':'Majority: '+majority['majority_class'],'accuracy':majority['accuracy'],'macro_f1':majority['macro_f1']}]
    for name,value in [('Trained '+local_meta['best_model'],loc),('Random step',random),('Earliest step',earliest)]:
        baseline_rows.append({'module':'localizer','model':name,**{k:value[k] for k in ['exact_step_accuracy','top3_accuracy','top5_accuracy','mrr','mean_absolute_step_distance']}})
    baselines=pd.DataFrame(baseline_rows)
    comparison,missing=comparisons(results/'taxonomy/model_comparison.csv',results/'localizer/model_comparison.csv',{'taxonomy':tax_meta['best_model'],'localizer':local_meta['best_model']})
    for entry in comparison.to_dict(orient='records'):
        if entry['module']=='taxonomy':
            expected=tax_meta['validation_macro_f1'].get(entry['model'])
            if expected is None or not np.isclose(entry['validation_macro_f1'],expected):
                raise ValueError('Taxonomy comparison differs from saved model selection metadata')
        else:
            expected=next((v for v in local_meta['validation_comparison'] if v['model']==entry['model']),None)
            if expected is None or any(not np.isclose(entry[k],expected[k]) for k in ['validation_mrr','validation_exact_step_accuracy']):
                raise ValueError('Localizer comparison differs from saved model selection metadata')
    evaluation_labels=tax_rows[['trajectory_id','fault_category']].rename(columns={'fault_category':'actual_category'})
    by_category,by_length,length_bins=breakdowns(rankings,evaluation_labels,local_rows.loc[local_rows['split']=='train'])
    loc['length_bins']=length_bins
    tax_errors=tax_pred.loc[tax_pred['actual_category']!=tax_pred['predicted_category']].rename(columns={'category_confidence':'confidence'})
    tax_errors=tax_errors[['trajectory_id','task_id','actual_category','predicted_category','confidence','actual_subtype','predicted_subtype']]
    local_errors=rankings.loc[~rankings['exact_match'],['trajectory_id','task_id','actual_root_cause_step','predicted_root_cause_step','actual_root_cause_rank','predicted_probability','absolute_step_distance']]
    audit=audit_models(taxonomy,localizer,tax_rows,local_rows,tax,loc)
    summary={'taxonomy_classifier':{'best_model':tax_meta['best_model'],'test_samples':len(tax_test),**{k:tax[k] for k in ['accuracy','macro_f1','weighted_f1']}},
        'root_cause_localizer':{'best_model':local_meta['best_model'],'test_trajectories':loc['faulty_trajectories'],**{k:loc[k] for k in ['exact_step_accuracy','top3_accuracy','top5_accuracy','mrr','mean_absolute_step_distance','median_absolute_step_distance']}},
        'baselines':{'taxonomy_majority':majority,'random_step':random,'earliest_step':earliest},'split_integrity':integrity}
    for file,value in [('taxonomy_metrics.json',tax),('localizer_metrics.json',loc),('review3_ml_summary.json',summary),('leakage_audit.json',audit)]:
        save_json(output/file,value)
    for file,table in [('model_comparison.csv',comparison),('baseline_comparison.csv',baselines),('taxonomy_errors.csv',tax_errors),
                       ('localization_errors.csv',local_errors),('localizer_by_category.csv',by_category),('localizer_by_length.csv',by_length),('taxonomy_confidence_buckets.csv',buckets)]:
        table.to_csv(output/file,index=False)
    taxonomy_plots(tax,output); localizer_plots(rankings,output); comparison_plots(comparison,output)
    try:
        git=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,timeout=5)
        commit=git.stdout.strip() if git.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):
        commit=None
    paths={'taxonomy_model':models/'taxonomy/best_model.joblib','localizer_model':models/'localizer/best_localizer.joblib',
           'trajectory_features':trajectory,'step_features':steps,'manifest':manifest,'taxonomy_predictions':Path(tax_source),'localizer_predictions':Path(local_source)}
    save_json(output/'evaluation_metadata.json',{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'random_seed':seed,'git_commit':commit,
        'paths':{k:str(p.resolve()) for k,p in paths.items()},'sha256':{k:file_hash(p) for k,p in paths.items()},
        'dataset_counts':{'taxonomy':tax_info,'localizer':local_info},'classes_evaluated':tax['classes_evaluated'],
        'missing_comparison_files':missing,'prediction_policy':'Existing predictions verified against frozen models and current labels; no model fitting'})
    write_report(output,summary,tax,loc,baselines,by_category,by_length,tax_errors,local_errors,audit,{'taxonomy':tax_info,'localizer':local_info},comparison)
    for title,key in [('Taxonomy Classifier','taxonomy_classifier'),('Root-Cause Localizer','root_cause_localizer')]:
        print('\n## '+title)
        for metric,value in summary[key].items(): print(f'{metric}: {value}')
    print(f'\n## Baselines\nTaxonomy majority Macro-F1: {majority["macro_f1"]:.6f}\nRandom-step localization MRR: {random["mrr"]:.6f}\nEarliest-step localization MRR: {earliest["mrr"]:.6f}')
    for module,report in audit.items():
        for warning in report['warnings']: print(f'WARNING ({module}): {warning}')
    print(f'\nResults saved to: {output}')
    return summary
