"""Taxonomy mapping and shared artifact helpers."""
import hashlib
import json
from pathlib import Path

from injector.faults import FaultType
from feature_engineering.common import FORBIDDEN
from feature_engineering.trajectory_features import FEATURE_COLUMNS, LABEL_COLUMNS, METADATA_COLUMNS

# The repository has subtype enums but no category enum. Use enum members, not
# string-prefix inference, so new/renamed faults require an explicit decision.
CATEGORY_SUBTYPES = {
    'PLANNING': (FaultType.PLAN_MISSING_STEP, FaultType.PLAN_INCORRECT_SUCCESS_CRITERIA, FaultType.PLAN_INVALID_DEPENDENCY),
    'TOOL': (FaultType.TOOL_WRONG_TOOL, FaultType.TOOL_WRONG_ARGUMENT, FaultType.TOOL_UNNECESSARY_CALL, FaultType.TOOL_FAILED_RECOVERY),
    'KNOWLEDGE': (FaultType.KNOW_RETRIEVAL_FAILURE, FaultType.KNOW_CITATION_MISMATCH, FaultType.KNOW_CONTEXT_TRUNCATION),
    'MULTI_AGENT': (FaultType.MA_INCORRECT_HANDOFF, FaultType.MA_INFORMATION_LOSS, FaultType.MA_MISSING_RESPONSIBILITY, FaultType.MA_ROLE_OVERLAP),
    'CONTROL': (FaultType.CTRL_LOOP, FaultType.CTRL_PREMATURE_TERMINATION, FaultType.CTRL_EXCESSIVE_EXPLORATION),
    'SECURITY': (FaultType.SEC_PROMPT_INJECTION, FaultType.SEC_UNAUTHORIZED_ACTION, FaultType.SEC_CROSS_USER_DATA_LEAKAGE),
}
CATEGORIES = tuple(CATEGORY_SUBTYPES)
SUBTYPE_TO_CATEGORY = {subtype.value: category for category, subtypes in CATEGORY_SUBTYPES.items() for subtype in subtypes}
if set(SUBTYPE_TO_CATEGORY) != {s.value for s in FaultType}:
    raise RuntimeError('Category mapping must cover every existing FaultType exactly once')

LEAKAGE_COLUMNS = set(FORBIDDEN) | set(LABEL_COLUMNS) | set(METADATA_COLUMNS) | {
    'base_task_id', 'fault_category', 'fault_subtype', 'split', 'source_filename',
    'filename', 'injected_parameters', 'injection_operator', 'operator',
    'original_injected_value', 'modified_injected_value',
}


def validate_features(columns=FEATURE_COLUMNS):
    bad = [c for c in columns if c.lower() in LEAKAGE_COLUMNS
           or c.lower().startswith(('injected_', 'original_', 'modified_', 'fault_'))]
    if bad:
        raise ValueError(f'Suspicious label/provenance columns are forbidden: {bad}')
    if len(columns) != len(set(columns)) or set(columns) != set(FEATURE_COLUMNS):
        raise ValueError('Predictors must match the feature-engineering FEATURE_COLUMNS contract')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
