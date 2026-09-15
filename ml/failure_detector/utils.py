"""Explicit outcome labels, predictor contract and training-only weights."""
import math
from feature_engineering.common import FORBIDDEN
from feature_engineering.trajectory_features import FEATURE_COLUMNS, LABEL_COLUMNS, METADATA_COLUMNS
from ml.localizer.utils import save_json, file_hash

OUTCOME_MAPPING = {'SUCCESS':0,'PASS':0,'PASSED':0,'SUCCEEDED':0,'OK':0,
                   'FAILURE':1,'FAIL':1,'FAILED':1,'ERROR':1,'PARTIAL_SUCCESS':1}
CLASS_MAPPING = {0:'SUCCESS',1:'FAILURE'}
FORBIDDEN_COLUMNS = set(FORBIDDEN)|set(LABEL_COLUMNS)|set(METADATA_COLUMNS)|{
    'failure_label','fault_category','fault_subtype','split','base_task_id','filename',
    'source_filename','injection_parameters','injection_operator','operator','target','label'}


def outcome_to_label(outcome):
    """Return None for unrecognized/missing outcomes; never infer from injection."""
    return OUTCOME_MAPPING.get(str(outcome).strip().upper())


def validate_features(columns=FEATURE_COLUMNS):
    bad=[c for c in columns if c.lower() in FORBIDDEN_COLUMNS or c.lower().startswith(('fault_','injected_','original_','modified_','outcome_'))]
    if bad:
        raise ValueError(f'Forbidden predictive columns: {bad}')
    if len(columns)!=len(set(columns)) or set(columns)!=set(FEATURE_COLUMNS):
        raise ValueError('Features must match the existing trajectory FEATURE_COLUMNS')


def validate_threshold(value):
    value=float(value)
    if not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('Threshold must be finite and between 0 and 1')
    return value


def training_weights(rows):
    if not rows['split'].eq('train').all():
        raise ValueError('Class weights must use training rows only')
    labels=rows['failure_label']
    if not labels.isin([0,1]).all() or labels.nunique()!=2:
        raise ValueError('Both SUCCESS and FAILURE training examples are required')
    negative=int((labels==0).sum()); positive=int((labels==1).sum())
    return {'negative_training_samples':negative,'positive_training_samples':positive,'scale_pos_weight':negative/positive}
