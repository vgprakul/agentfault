"""Independent localizer API. Input is assumed to be a faulty trajectory."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import feature_frame
from .ranking import rank_steps
from .utils import validate_features


def root_probabilities(pipeline, rows):
    index = np.flatnonzero(pipeline.classes_==1)
    if len(index)!=1:
        raise ValueError('Saved model must support positive class 1')
    probabilities = pipeline.predict_proba(feature_frame(rows))[:,index[0]]
    if not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any():
        raise ValueError('Invalid model probabilities')
    return probabilities


class RootCauseLocalizer:
    def __init__(self, model_dir='models/localizer'):
        self.bundle = joblib.load(Path(model_dir)/'best_localizer.joblib')
        validate_features(self.bundle['feature_columns'])

    def score_steps(self, rows):
        frame = rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
        return root_probabilities(self.bundle['pipeline'],frame)

    def predict(self, step_feature_rows):
        rows = step_feature_rows if isinstance(step_feature_rows,pd.DataFrame) else pd.DataFrame(step_feature_rows)
        if rows.empty or rows['trajectory_id'].nunique()!=1:
            raise ValueError('Provide exactly one nonempty trajectory')
        ranked = rank_steps(rows,self.score_steps(rows))
        first = ranked.iloc[0]
        return {'predicted_root_cause_step':int(first['step_index']),
            'root_cause_probability':float(first['root_cause_probability']),
            'ranked_steps':[{'step_index':int(row.step_index),'probability':float(row.root_cause_probability)}
                            for row in ranked.itertuples()]}


def predict_root_cause(step_feature_rows, model_dir='models/localizer'):
    return RootCauseLocalizer(model_dir).predict(step_feature_rows)
