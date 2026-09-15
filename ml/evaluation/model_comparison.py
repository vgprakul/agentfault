"""Read existing validation comparisons; construct a train-derived majority baseline."""
from pathlib import Path
import pandas as pd
from .taxonomy_metrics import classification_metrics
from ml.taxonomy.utils import CATEGORIES


def majority_baseline(training_labels, actual):
    counts = pd.Series(training_labels).value_counts()
    if counts.empty:
        raise ValueError('Majority baseline requires labelled training rows')
    tied = set(counts[counts==counts.max()].index)
    majority = next(c for c in CATEGORIES if c in tied)
    result = classification_metrics(actual,[majority]*len(actual))
    result['majority_class'] = majority
    result['tie_rule'] = 'existing CATEGORIES order; training counts only'
    return result


def comparisons(taxonomy_path, localizer_path, selected):
    frames,missing = [],[]
    for module,path,primary,secondary in [('taxonomy',taxonomy_path,'validation_macro_f1',None),
                                           ('localizer',localizer_path,'validation_mrr','validation_exact_step_accuracy')]:
        path=Path(path)
        if not path.exists():
            missing.append(str(path)); continue
        frame=pd.read_csv(path)
        if not {'model',primary}<=set(frame) or frame['model'].duplicated().any():
            raise ValueError(f'Malformed model comparison: {path}')
        for column in [primary]+([secondary] if secondary else []):
            values=pd.to_numeric(frame[column],errors='raise')
            if not values.between(0,1).all():
                raise ValueError(f'Invalid validation metric {column}')
        frame['module']=module
        frame['selection_metric']=primary
        frame['selected']=frame['model'].eq(selected[module])
        frames.append(frame)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(),missing
