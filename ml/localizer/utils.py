"""Predictor contract, training-only imbalance weights, and artifact helpers."""
import hashlib
import json
from pathlib import Path

from feature_engineering.common import FORBIDDEN
from feature_engineering.step_features import FEATURE_COLUMNS, LABEL_COLUMNS, METADATA_COLUMNS

LEAKAGE_COLUMNS = set(FORBIDDEN) | set(LABEL_COLUMNS) | set(METADATA_COLUMNS) | {
    'base_task_id','fault_category','fault_subtype','injected_fault','split','source',
    'filename','source_filename','injection_operator','operator','injection_parameters',
    'predicted_category',  # Taxonomy context is deliberately not a v1 predictor.
}


def validate_features(columns=FEATURE_COLUMNS):
    forbidden = [c for c in columns if c.lower() in LEAKAGE_COLUMNS
                 or c.lower().startswith(('fault_', 'injected_', 'modified_', 'original_'))]
    if forbidden:
        raise ValueError(f'Forbidden leakage/noncausal columns: {forbidden}')
    if len(columns) != len(set(columns)) or set(columns) != set(FEATURE_COLUMNS):
        raise ValueError('Predictors must match the existing step FEATURE_COLUMNS')


def training_weights(training_rows):
    if not training_rows['split'].eq('train').all():
        raise ValueError('Imbalance weights may only use training rows')
    labels = training_rows['is_root_cause']
    if not labels.isin([0,1]).all():
        raise ValueError('Binary root labels required')
    positive, negative = int(labels.sum()), int((labels==0).sum())
    if not positive or not negative:
        raise ValueError('Training requires positive and negative steps')
    return {'positive_training_steps':positive, 'negative_training_steps':negative,
            'scale_pos_weight':negative/positive}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
