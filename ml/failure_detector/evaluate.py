"""Failure-positive metrics, fixed-threshold analysis and saved-model evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,precision_recall_fscore_support,confusion_matrix,
    classification_report,roc_auc_score,average_precision_score,roc_curve,precision_recall_curve)
from .data import load_dataset,feature_frame
from .inference import FailureDetector,probabilities
from .utils import save_json,validate_threshold


def metrics(labels,scores,threshold=.5):
    threshold=validate_threshold(threshold)
    y=np.asarray(labels); p=np.asarray(scores,dtype=float)
    if y.shape!=p.shape or not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Nonempty binary labels and finite probabilities in [0,1] required')
    pred=(p>=threshold).astype(int)
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    both=len(np.unique(y))==2
    return {'accuracy':float(accuracy_score(y,pred)),'precision':float(precision),'recall':float(recall),'f1':float(f1),
        'roc_auc':float(roc_auc_score(y,p)) if both else None,
        'pr_auc':float(average_precision_score(y,p)) if both else None,
        'tn':tn,'fp':fp,'fn':fn,'tp':tp,'false_positive_rate':fp/(fp+tn) if fp+tn else None,
        'false_negative_rate':fn/(fn+tp) if fn+tp else None,'threshold':threshold,
        'unavailable_metrics':{} if both else {'roc_auc':'Evaluated split has only one outcome class',
            'pr_auc':'Single-class average precision is non-discriminative; reported unavailable',
            **({'false_positive_rate':'No SUCCESS examples'} if not tn+fp else {'false_negative_rate':'No FAILURE examples'})},
        'per_class':classification_report(y,pred,labels=[0,1],target_names=['SUCCESS','FAILURE'],output_dict=True,zero_division=0)}


def majority_baseline(training_labels,test_labels):
    counts=pd.Series(training_labels).value_counts()
    if counts.empty: raise ValueError('Training labels required')
    majority=int(sorted(counts[counts==counts.max()].index)[0])
    result=metrics(test_labels,np.full(len(test_labels),majority,dtype=float))
    result['majority_class']=majority
    return result


def threshold_analysis(labels,scores):
    rows=[]
    for threshold in [.3,.4,.5,.6,.7]:
        result=metrics(labels,scores,threshold)
        rows.append({k:result[k] for k in ['threshold','precision','recall','f1','false_positive_rate','false_negative_rate']})
    frame=pd.DataFrame(rows)
    # Ties favor proximity to the unchanged default threshold, then lower threshold.
    ordered=frame.assign(distance=(frame['threshold']-.5).abs()).sort_values(['f1','distance','threshold'],ascending=[False,True,True])
    recall=frame.assign(distance=(frame['threshold']-.5).abs()).sort_values(['recall','precision','distance','threshold'],ascending=[False,False,True,True])
    return frame,{'best_f1_threshold':float(ordered.iloc[0]['threshold']),
        'high_recall_threshold':float(recall.iloc[0]['threshold']),
        'selection_note':'Validation-only discussion; default threshold remains 0.50 unless explicitly supplied'}


def feature_importance(pipeline):
    classifier=pipeline.named_steps['classifier']
    values=classifier.feature_importances_ if hasattr(classifier,'feature_importances_') else np.abs(classifier.coef_).mean(axis=0)
    return pd.DataFrame({'feature':pipeline.named_steps['preprocessing'].get_feature_names_out(),'importance':values}).sort_values('importance',ascending=False)


def plots(labels,scores,result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,ax=plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay(np.array([[result['tn'],result['fp']],[result['fn'],result['tp']]]),display_labels=['SUCCESS','FAILURE']).plot(ax=ax,colorbar=False,cmap='Blues')
    ax.set_title('Failure detector: held-out counts')
    fig.tight_layout();fig.savefig(output/'confusion_matrix.png',dpi=150);plt.close(fig)
    for name in ['roc_curve','precision_recall_curve']:
        fig,ax=plt.subplots(figsize=(6,4))
        if len(set(labels))<2:
            ax.text(.5,.5,'Unavailable: held-out data contains\nonly one outcome class.',ha='center',va='center',transform=ax.transAxes)
            ax.set_axis_off()
        elif name=='roc_curve':
            x,y,_=roc_curve(labels,scores);ax.plot(x,y);ax.plot([0,1],[0,1],'--',color='gray')
            ax.set(xlabel='False positive rate',ylabel='Failure recall')
        else:
            precision,recall,_=precision_recall_curve(labels,scores);ax.plot(recall,precision)
            ax.set(xlabel='Failure recall',ylabel='Failure precision')
        ax.set_title(name.replace('_',' ').title())
        fig.tight_layout();fig.savefig(output/f'{name}.png',dpi=150);plt.close(fig)


def evaluate(model_dir='models/failure_detector',results_dir='results/failure_detector',threshold=.5):
    detector=FailureDetector(model_dir)
    metadata=detector.bundle['metadata']
    inputs=metadata['data']['inputs']
    rows,info=load_dataset(inputs[0]['features'],inputs[0]['manifest'],
        [v['features'] for v in inputs[1:]],[v['manifest'] for v in inputs[1:]])
    if info['inputs']!=inputs: raise ValueError('Evaluation inputs changed since model fitting')
    training=rows.loc[rows['split']=='train']; validation=rows.loc[rows['split']=='validation']; test=rows.loc[rows['split']=='test']
    if test.empty: raise ValueError('No test rows')
    if set(test['trajectory_id'])&set(metadata['training_trajectory_ids']): raise ValueError('Training/test overlap')
    p=probabilities(detector.bundle['pipeline'],test)
    result=metrics(test['failure_label'],p,threshold)
    baseline=majority_baseline(training['failure_label'],test['failure_label'])
    analysis,discussion=threshold_analysis(validation['failure_label'],probabilities(detector.bundle['pipeline'],validation))
    analysis.insert(0,'split','validation')
    output=Path(results_dir);output.mkdir(parents=True,exist_ok=True)
    analysis.to_csv(output/'threshold_analysis.csv',index=False)
    predictions=test[['trajectory_id','task_id','outcome_status','failure_label']].rename(columns={'outcome_status':'actual_outcome','failure_label':'actual_failure_label'}).copy()
    predictions['predicted_failure_label']=(p>=threshold).astype(int);predictions['failure_probability']=p
    predictions['prediction_correct']=predictions['actual_failure_label'].eq(predictions['predicted_failure_label'])
    predictions.to_csv(output/'test_predictions.csv',index=False)
    save_json(output/'classification_report.json',result)
    pd.DataFrame([{'model':name,**{k:value[k] for k in ['accuracy','precision','recall','f1']}}
                  for name,value in [(metadata['best_model'],result),('Training majority baseline',baseline)]]).to_csv(output/'baseline_comparison.csv',index=False)
    importance=feature_importance(detector.bundle['pipeline']);importance.to_csv(output/'feature_importance.csv',index=False)
    warnings=list(metadata['warnings'])
    suspicious=[c for c in detector.bundle['feature_columns'] if any(t in c.lower() for t in ['fault','outcome','injected','label','target','root','origin','failure_type'])]
    if suspicious: warnings.append('Inspect suspicious feature names: '+', '.join(suspicious))
    if result['accuracy']>=.95:
        warnings.append('WARNING: near-perfect scores require proxy-leakage inspection; single-class test scores can be matched by an always-FAILURE baseline.')
    training_features=feature_frame(training)
    repeats=int(pd.util.hash_pandas_object(feature_frame(test),index=False).isin(set(pd.util.hash_pandas_object(training_features,index=False))).sum())
    if repeats: warnings.append(f'{repeats}/{len(test)} test feature vectors also occur in training.')
    # Inspect actual training distributions, without using held-out labels to select predictors.
    proxy_rows=[]
    for column in feature_frame(training):
        if column=='dominant_agent': continue
        for label, group in training.groupby('failure_label'):
            values=group[column]
            proxy_rows.append({'feature':column,'training_outcome':'FAILURE' if label else 'SUCCESS',
                'count':len(group),'missing_count':int(values.isna().sum()),
                'minimum':values.min(),'median':values.median() if values.notna().any() else None,'maximum':values.max()})
    pd.DataFrame(proxy_rows).to_csv(output/'training_feature_audit.csv',index=False)
    source_counts=[]
    if 'source' in training:
        source_counts=training.groupby(['source','failure_label'],dropna=False).size().reset_index(name='samples').to_dict(orient='records')
        if training.groupby('source')['failure_label'].nunique().eq(1).all() and training['source'].nunique()>1:
            warnings.append('Training source perfectly separates outcomes. Source is excluded, but instrumentation features can encode it indirectly; this is a concrete proxy-leakage risk.')
    save_json(output/'leakage_audit.json',{'suspicious_feature_names':suspicious,'test_vectors_matching_train':repeats,
        'training_source_outcome_counts':source_counts,
        'top_features':importance.head(15).to_dict(orient='records'),'warnings':warnings})
    summary={'status':'limited_single_class_evaluation' if test['failure_label'].nunique()<2 else 'evaluated',
        'best_model':metadata['best_model'],'test_samples':len(test),
        **{k:result[k] for k in ['accuracy','precision','recall','f1','roc_auc','pr_auc','tn','fp','fn','tp']},
        'default_threshold':.5,'evaluated_threshold':threshold,'data':info,'validation_metrics':metadata['validation_metrics'],
        'threshold_discussion':discussion,'majority_baseline':baseline,'warnings':warnings}
    save_json(output/'review3_summary.json',summary)
    plots(test['failure_label'].to_numpy(),p,result,output)
    # Test metrics are provenance only: neither model nor preprocessing is refitted.
    save_json(Path(model_dir)/'model_metadata.json',{**metadata,'test_metrics':result,'threshold_discussion':discussion})
    print('\nFinal Test Results')
    for key in ['accuracy','precision','recall','f1','roc_auc','pr_auc']: print(f'{key}: {result[key]}')
    print('Threshold discussion:',discussion)
    print('Top features:\n'+importance.head(15).to_string(index=False))
    for warning in warnings: print('WARNING:',warning)
    return summary
