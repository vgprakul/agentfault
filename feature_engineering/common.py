"""Observable payload handling. Never read injection provenance as features."""
import json
from collections import Counter

FORBIDDEN = {'fault_type', 'fault_injected', 'origin_step', 'is_root_cause',
             'outcome', 'outcome.status', 'outcome_status', 'injection_params',
             'original_value', 'modified_value', 'injected_value', 'metadata'}
TEXT_KEYS = {'task', 'query', 'text', 'content', 'context', 'research', 'analysis',
             'draft', 'final_answer', 'answer', 'message', 'prompt', 'description',
             'success_criteria', 'responsibility', 'plan', 'summary'}
LOOP_REPEAT_THRESHOLD = 3


def safe_payload(value):
    if isinstance(value, dict):
        return {k: safe_payload(v) for k, v in value.items()
                if k.lower() not in FORBIDDEN and not k.lower().startswith(('original_', 'modified_', 'injected_'))}
    if isinstance(value, list):
        return [safe_payload(v) for v in value]
    return value


def text(value):
    if isinstance(value, str):
        value = value.strip()
        return value if value.lower() not in {'none', 'null', 'unknown'} else ''
    if isinstance(value, list):
        return ' '.join(filter(None, (text(v) for v in value)))
    if isinstance(value, dict):
        return ' '.join(filter(None, (text(v) for k, v in value.items() if k in TEXT_KEYS)))
    return ''


def canonical(value):
    return json.dumps(safe_payload(value), sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def payload_text(value):
    """Size actual payloads, including numeric tool results, without provenance."""
    if value is None:
        return ''
    return value if isinstance(value, str) else canonical(value)


def action(step):
    return (step.step_type.upper(), (step.tool_name or '').strip().lower(), canonical(step.input))


def status(step):
    raw = (step.status or '').upper()
    error = isinstance(step.output, dict) and bool(step.output.get('error'))
    if error or raw in {'ERROR', 'FAILED', 'FAIL', 'FAILURE', 'FAILED_RECOVERY'}:
        return 'ERROR' if error or raw == 'ERROR' else 'FAILED'
    return 'SUCCESS' if raw in {'SUCCESS', 'OK', 'SUCCEEDED'} else raw or 'UNKNOWN'


def error(step):
    return int(status(step) in {'ERROR', 'FAILED'})


def ratio(a, b):
    return a / b if b else 0.0


def mean(values):
    return sum(values) / len(values) if values else None


def repeats(keys):
    return sum(n - 1 for n in Counter(keys).values())


def run_max(keys):
    longest = run = 0
    previous = object()
    for key in keys:
        run = run + 1 if key == previous else 1
        previous = key
        longest = max(longest, run)
    return longest
