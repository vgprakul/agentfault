"""Read record JSON/JSONL and recorder event JSON/JSONL without changing schemas."""
import json
from dataclasses import fields
from pathlib import Path

from injector.schema import AgentConfig, AgentFaultRecord, Outcome, StepRecord
from injector.real_trajectory import events_to_record
from trajectory.events import TrajectoryEvent


def task_text(record):
    # Only the original, first-step task is usable; never search future steps.
    if record.steps and record.steps[0].step_index in {0, 1} and isinstance(record.steps[0].input, dict):
        return record.steps[0].input.get('task') or record.steps[0].input.get('query')
    return None


def validate(record):
    if not record.trajectory_id or not record.task_id:
        raise ValueError('Missing trajectory/task identifier')
    indices = [s.step_index for s in record.steps]
    if any(type(i) is not int or i < 0 for i in indices):
        raise ValueError('Step indices must be nonnegative integers')
    if indices != sorted(set(indices)):
        raise ValueError('Step indices must be unique and ordered')
    if record.num_steps != len(indices):
        raise ValueError('num_steps does not match steps')
    if type(record.fault_injected) is not bool:
        raise ValueError('fault_injected must be boolean')
    positives = [s.step_index for s in record.steps if s.is_root_cause]
    if not record.fault_injected and positives:
        raise ValueError('Clean trajectory has positive root labels')
    if record.fault_injected and record.origin_step is not None and indices:
        # Mirrors FaultInjector: a removed origin maps to the next surviving step.
        expected = next((i for i in indices if i >= record.origin_step), indices[-1])
        if positives and positives != [expected]:
            raise ValueError('Root labels disagree with injector origin semantics')
        if not positives:
            next(s for s in record.steps if s.step_index == expected).is_root_cause = True


def load(path):
    """Return (records, original task texts, diagnostics). Skip invalid records."""
    path = Path(path)
    paths = sorted(p for p in path.rglob('*') if p.suffix in {'.json', '.jsonl'}) if path.is_dir() else [path]
    records, texts, issues, seen = [], {}, [], set()
    allowed = {f.name for f in fields(AgentFaultRecord)}
    for file in paths:
        try:
            if file.suffix == '.jsonl':
                objects = []
                broken_lines = False
                for number, line in enumerate(file.read_text(encoding='utf-8-sig').splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        objects.append(json.loads(line))
                    except ValueError as exc:
                        broken_lines = True
                        issues.append(f'{file}:{number}: {exc}')
                        # A broken event stream cannot safely become a complete trajectory.
                        if objects and 'event_type' in objects[0]:
                            raise ValueError('Corrupt event stream') from exc
                if broken_lines and any('event_type' in o for o in objects):
                    raise ValueError('Corrupt event stream')
            else:
                obj = json.loads(file.read_text(encoding='utf-8-sig'))
                objects = obj if isinstance(obj, list) else [obj]
            event_groups, candidates = {}, []
            for obj in objects:
                if 'events' in obj:
                    candidates.append(('events', obj['events']))
                elif 'event_type' in obj:
                    event_groups.setdefault(obj['trajectory_id'], []).append(obj)
                else:
                    candidates.append(('record', obj))
            candidates.extend(('events', group) for group in event_groups.values())
            for kind, obj in candidates:
                try:
                    if kind == 'events':
                        events = [TrajectoryEvent(**e) for e in obj]
                        record = events_to_record(events)
                        start = next((e for e in events if e.event_type == 'trajectory_start'), None)
                        original = start.input if start else None
                        if isinstance(original, dict):
                            original = original.get('query') or original.get('task')
                        # Adapter defaults outcome to SUCCESS, which is not observed evidence.
                        end = next((e for e in reversed(events) if e.event_type == 'trajectory_end'), None)
                        record.outcome.status = end.status if end else 'UNKNOWN'
                        known = {'llm_call', 'tool_call', 'retrieval', 'handoff'}
                        for e in events:
                            if e.event_type not in known | {'trajectory_start', 'trajectory_end'}:
                                record.steps.append(StepRecord(e.step, e.event_type.upper(), e.agent or 'UNKNOWN', e.input, e.output, tool_name=e.tool, status=e.status, metadata=e.metadata))
                        record.steps.sort(key=lambda s: s.step_index)
                        record.num_steps = len(record.steps)
                    else:
                        values = {k: v for k, v in obj.items() if k in allowed}
                        values['agent_config'] = AgentConfig(**obj['agent_config'])
                        values['outcome'] = Outcome(**obj['outcome'])
                        values['steps'] = [StepRecord(**s) for s in obj['steps']]
                        if obj.get('base_task_id'):
                            values['task_id'] = obj['base_task_id']
                        record = AgentFaultRecord(**values)
                        original = obj.get('original_task') or task_text(record)
                    validate(record)
                    if record.trajectory_id in seen:
                        raise ValueError(f'Duplicate trajectory ID: {record.trajectory_id}')
                    seen.add(record.trajectory_id)
                    records.append(record)
                    texts[record.trajectory_id] = original
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    issues.append(f'{file}: {exc}')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            issues.append(f'{file}: {exc}')
    return records, texts, issues
