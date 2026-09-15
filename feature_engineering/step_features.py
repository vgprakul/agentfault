"""Each feature row uses only the prefix ending at the current step."""
from collections import Counter
from .common import action, status, error, ratio, payload_text

LABEL_COLUMNS = ['is_root_cause']
METADATA_COLUMNS = ['trajectory_id', 'task_id', 'step_index', 'normalized_step_position']
FEATURE_COLUMNS = ['step_type', 'agent_id', 'tool_name', 'status', 'has_error',
    'is_tool_step', 'is_retrieval_step', 'is_llm_step', 'is_handoff_step',
    'input_length_chars', 'output_length_chars', 'input_length_words', 'output_length_words',
    'cumulative_tool_calls', 'cumulative_retrievals', 'cumulative_errors', 'cumulative_retries',
    'cumulative_handoffs', 'cumulative_unique_tools', 'cumulative_unique_agents',
    'repeated_action_count_so_far', 'previous_step_same_type', 'previous_step_same_agent',
    'previous_step_same_tool', 'steps_since_last_successful_action',
    'task_step_similarity', 'semantic_drift', 'previous_step_similarity']


def extract(record, semantic):
    rows, tools, agents, actions = [], set(), set(), Counter()
    totals = Counter()
    previous, last_success = None, None
    for position, (s, sem) in enumerate(zip(record.steps, semantic)):
        kind = s.step_type.upper()
        flags = dict(tool_calls=int(kind == 'TOOL_CALL'), retrievals=int(kind == 'RETRIEVAL'),
                     errors=error(s), retries=int(kind == 'RETRY'), handoffs=int(kind == 'HANDOFF'))
        totals.update(flags)
        if s.tool_name and kind == 'TOOL_CALL':
            tools.add(s.tool_name.strip().lower())
        if s.agent_id:
            agents.add(s.agent_id)
        key = action(s)
        totals['repeated'] += int(actions[key] > 0)
        actions[key] += 1
        if status(s) == 'SUCCESS':
            last_success = position
        row = dict(trajectory_id=record.trajectory_id, task_id=record.task_id,
                   step_index=s.step_index, normalized_step_position=ratio(s.step_index, len(record.steps)),
                   step_type=kind, agent_id=s.agent_id or 'UNKNOWN', tool_name=s.tool_name or 'UNKNOWN',
                   status=status(s), has_error=error(s), is_tool_step=int(kind in {'TOOL_CALL','TOOL_RESPONSE','TOOL_RESULT'}),
                   is_retrieval_step=flags['retrievals'], is_llm_step=int(kind == 'LLM_CALL'), is_handoff_step=flags['handoffs'],
                   cumulative_unique_tools=len(tools), cumulative_unique_agents=len(agents),
                   repeated_action_count_so_far=totals['repeated'],
                   steps_since_last_successful_action=position-last_success if last_success is not None else None,
                   is_root_cause=int(s.is_root_cause) if record.fault_injected else 0)
        for side in ['input', 'output']:
            value = payload_text(getattr(s, side))
            row[f'{side}_length_chars'] = len(value)
            row[f'{side}_length_words'] = len(value.split())
        for name in flags:
            row[f'cumulative_{name}'] = totals[name]
        for name, attr in [('type','step_type'), ('agent','agent_id'), ('tool','tool_name')]:
            row[f'previous_step_same_{name}'] = int(previous is not None and bool(getattr(s, attr)) and getattr(s, attr) == getattr(previous, attr))
        row.update(sem)
        rows.append(row)
        previous = s
    return rows
