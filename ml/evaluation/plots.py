"""Small static matplotlib plots; no model fitting or calibration."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def finish(fig,path):
    fig.tight_layout()
    fig.savefig(path,dpi=150)
    plt.close(fig)


def summary_plot(summary, output):
    """Plot existing held-out metrics without recomputing or changing them."""
    panels = [
        ('taxonomy_classifier', 'Taxonomy classifier',
         [('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1'), ('weighted_f1', 'Weighted-F1')]),
        ('root_cause_localizer', 'Root-cause localizer',
         [('exact_step_accuracy', 'Exact step'), ('top3_accuracy', 'Top-3'), ('mrr', 'MRR')]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (key, title, metrics), color in zip(axes, panels, ['#3478ac', '#348b73']):
        values = summary[key]
        available = [(label, values.get(name)) for name, label in metrics
                     if values.get(name) is not None]
        bars = ax.bar([label for label, _ in available],
                      [value for _, value in available], color=color, width=.6)
        ax.bar_label(bars, fmt='%.3f', padding=4)
        ax.set(ylim=(0, 1.12), ylabel='Held-out test score (0–1)',
               title=f"{title}\n{values['best_model']}")
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('AgentFault ML Evaluation', fontsize=16)
    fig.text(.5, .01, 'Different tasks and metrics; bar heights are not a direct comparison between modules.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    path = output / 'ml_evaluation_summary.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def taxonomy_plots(metrics,output):
    matrix=np.array(metrics['confusion_matrix'])
    labels=metrics['confusion_matrix_labels']
    for normalized in [False,True]:
        values=matrix.astype(float)
        if normalized:
            values=np.divide(values,values.sum(axis=1,keepdims=True),out=np.zeros_like(values),where=values.sum(axis=1,keepdims=True)!=0)
        fig,ax=plt.subplots(figsize=(9,7))
        ax.imshow(values,cmap='Blues',vmin=0,vmax=1 if normalized else None)
        ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,
               xlabel='Predicted category',ylabel='Actual category',title='Taxonomy confusion matrix'+(' (row normalized)' if normalized else ' (counts)'))
        plt.setp(ax.get_xticklabels(),rotation=35,ha='right')
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j,i,f'{values[i,j]:.2f}' if normalized else str(matrix[i,j]),ha='center',va='center',
                        color='white' if values[i,j]>values.max()/2 else '#123456')
        finish(fig,output/('taxonomy_confusion_matrix_normalized.png' if normalized else 'taxonomy_confusion_matrix.png'))
    labels=metrics['classes_evaluated']
    fig,ax=plt.subplots(figsize=(9,4.5))
    bars=ax.bar(labels,[metrics['per_class'][c]['f1-score'] for c in labels],color='#3478ac')
    ax.bar_label(bars,fmt='%.2f')
    ax.set(ylim=(0,1.12),ylabel='F1',title='Taxonomy per-class test F1')
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right')
    finish(fig,output/'taxonomy_per_class_f1.png')


def localizer_plots(rankings,output):
    for column,name,title,xlabel in [('actual_root_cause_rank','localizer_rank_distribution.png','Rank of the true root-cause step','True root rank'),
                                     ('absolute_step_distance','localizer_step_distance_distribution.png','Root-cause absolute step distance','Absolute step-index distance')]:
        counts=rankings[column].value_counts().sort_index()
        fig,ax=plt.subplots(figsize=(7,4))
        ax.bar(counts.index,counts.values,color='#348b73')
        ax.set(xticks=range(int(counts.index.min()),int(counts.index.max())+1),xlabel=xlabel,ylabel='Trajectories',title=title)
        ax.yaxis.get_major_locator().set_params(integer=True)
        finish(fig,output/name)


def comparison_plots(comparison,output):
    for module in ['taxonomy','localizer']:
        if comparison.empty: continue
        data=comparison.loc[comparison['module']==module]
        if data.empty: continue
        fig,ax=plt.subplots(figsize=(8,4.5))
        x=np.arange(len(data))
        if module=='taxonomy':
            bars=ax.bar(x,data['validation_macro_f1'],color='#3478ac')
            ax.bar_label(bars,fmt='%.3f')
            ax.set(ylabel='Validation Macro-F1')
        else:
            for offset,column,label,color in [(-.18,'validation_mrr','MRR','#348b73'),(.18,'validation_exact_step_accuracy','Exact step','#d89b35')]:
                bars=ax.bar(x+offset,data[column],width=.36,label=label,color=color)
                ax.bar_label(bars,fmt='%.3f',fontsize=9)
            ax.legend(loc='upper left')
            ax.set(ylabel='Validation score')
        ax.set(xticks=x,xticklabels=data['model'],ylim=(0,1.15),title=f'{module.title()} candidate comparison')
        finish(fig,output/f'{module}_model_comparison.png')
