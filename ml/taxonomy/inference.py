"""Reusable inference from trajectory feature rows; no LangGraph dependency."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .data import feature_frame
from .utils import validate_features


class TaxonomyPredictor:
    def __init__(self, model_dir='models/taxonomy'):
        # Load only trusted local joblib artifacts.
        self.bundle = joblib.load(Path(model_dir)/'best_model.joblib')
        validate_features(self.bundle['feature_columns'])

    def predict_many(self, rows):
        frame = rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
        features = feature_frame(frame)
        encoder = self.bundle['label_encoder']
        probabilities = self.bundle['pipeline'].predict_proba(features)
        classes = encoder.inverse_transform(self.bundle['pipeline'].classes_.astype(int))
        results = []
        for i, probabilities_row in enumerate(probabilities):
            if not np.isfinite(probabilities_row).all() or not np.isclose(probabilities_row.sum(),1,atol=1e-5):
                raise ValueError('Model returned invalid probabilities')
            index = int(probabilities_row.argmax())
            category = str(classes[index])
            result = {'predicted_category':category, 'category_confidence':float(probabilities_row[index]),
                      'category_probabilities':dict(zip(classes.tolist(),map(float,probabilities_row))),
                      'predicted_subtype':None,'subtype_confidence':None}
            if 'trajectory_id' in frame:
                result['trajectory_id'] = str(frame.iloc[i]['trajectory_id'])
            submodel = self.bundle['subtype_models'].get(category)
            if submodel is not None:
                subprob = submodel.predict_proba(features.iloc[[i]])[0]
                if not np.isfinite(subprob).all() or not np.isclose(subprob.sum(),1,atol=1e-5):
                    raise ValueError('Subtype model returned invalid probabilities')
                subindex = int(subprob.argmax())
                result['predicted_subtype'] = str(submodel.classes_[subindex])
                result['subtype_confidence'] = float(subprob[subindex])
            results.append(result)
        return results

    def predict(self, feature_row):
        return self.predict_many([dict(feature_row)])[0]


def predict_taxonomy(feature_row, model_dir='models/taxonomy'):
    return TaxonomyPredictor(model_dir).predict(feature_row)
