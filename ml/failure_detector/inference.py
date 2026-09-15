"""Outcome prediction independent of labels, taxonomy and LangGraph."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import feature_frame
from .utils import validate_features, validate_threshold


def probabilities(pipeline,rows):
    positive=np.flatnonzero(pipeline.classes_==1)
    if len(positive)!=1 or len(pipeline.classes_)!=2:
        raise ValueError('A two-class SUCCESS/FAILURE model is required')
    values=pipeline.predict_proba(feature_frame(rows))[:,positive[0]]
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError('Invalid model probabilities')
    return values


class FailureDetector:
    def __init__(self,model_dir='models/failure_detector'):
        path=Path(model_dir)/'best_failure_detector.joblib'
        if not path.exists():
            raise FileNotFoundError('Failure detector not trained. Run python scripts/train_failure_detector.py with both outcome classes in training.')
        self.bundle=joblib.load(path)
        validate_features(self.bundle['feature_columns'])

    def predict_many(self,rows,threshold=.5):
        threshold=validate_threshold(threshold)
        frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
        values=probabilities(self.bundle['pipeline'],frame)
        return [{'failure_probability':float(p),'predicted_failure':bool(p>=threshold),
                 'predicted_outcome':'FAILURE' if p>=threshold else 'SUCCESS','threshold':threshold} for p in values]

    def predict(self,feature_row,threshold=.5):
        return self.predict_many([dict(feature_row)],threshold)[0]


def predict_failure(feature_row,threshold=.5,model_dir='models/failure_detector'):
    return FailureDetector(model_dir).predict(feature_row,threshold)
