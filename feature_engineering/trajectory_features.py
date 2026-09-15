"""Interpretable counts and aggregates of observed execution."""
from collections import Counter
from .common import action, status, error, ratio, text, mean, repeats, run_max, LOOP_REPEAT_THRESHOLD


def tool_observations(ss):
    """Pair separate responses with the latest unmatched same-tool call.

    Inline results take precedence. This is the sequential recorder convention;
    concurrent calls need a future explicit call-ID adapter.
    """
    observations = {}
    pending = []
    for s in ss:
        if s.step_type.upper() == 'TOOL_CALL':
            observations[s.step_index] = s
            if s.output is None:
                pending.append(s)
        elif s.step_type.upper() in {'TOOL_RESULT', 'TOOL_RESPONSE'}:
            match = next((p for p in reversed(pending) if p.tool_name == s.tool_name),None)
            if match:
                observations[match.step_index] = s
                pending.remove(match)
    return observations

LABEL_COLUMNS = ['fault_injected', 'fault_type', 'outcome_status', 'origin_step']
METADATA_COLUMNS = ['trajectory_id', 'task_id', 'source']
FEATURE_COLUMNS = '''total_steps total_agents total_tool_calls total_tool_responses total_retrieval_steps total_llm_steps total_handoffs total_retries total_errors unique_tools_used unique_agents tool_calls_per_step retrievals_per_step retries_per_step errors_per_step handoffs_per_step tool_call_count successful_tool_call_count failed_tool_call_count tool_success_rate tool_error_count duplicate_tool_call_count repeated_same_arguments_count consecutive_tool_failures_max tool_result_missing_count unique_tool_count tool_diversity average_tool_latency_ms max_tool_latency_ms repeated_action_count repeated_tool_argument_count consecutive_identical_action_max retry_count loop_indicator premature_termination_indicator max_repeated_sequence_length steps_since_last_successful_action retrieval_count empty_retrieval_count retrieval_failure_count total_retrieved_text_length average_retrieved_text_length max_retrieved_text_length average_retrieval_confidence minimum_retrieval_confidence agent_count unique_agent_count agent_switch_count handoff_count dominant_agent dominant_agent_step_count sender_receiver_similarity handoff_information_loss_score total_input_chars total_output_chars total_input_words total_output_words average_step_input_length average_step_output_length max_step_output_length total_retrieved_chars average_retrieved_chars success_step_count failed_step_count error_step_count failure_step_ratio first_error_step_position avg_task_step_similarity min_task_step_similarity final_task_step_similarity avg_semantic_drift max_semantic_drift final_semantic_drift semantic_drift_slope avg_consecutive_step_similarity min_consecutive_step_similarity max_semantic_change'''.split()


def repeated_sequence(keys):
    """Longest adjacent repeated block, excluding a single occurrence."""
    return max((width for width in range(1, len(keys)//2+1)
                for i in range(len(keys)-2*width+1)
                if keys[i:i+width] == keys[i+width:i+2*width]), default=0)


def extract(record, steps, semantic):
    ss = record.steps
    n = len(ss)
    calls = [s for s in ss if s.step_type.upper() == 'TOOL_CALL']
    retrievals = [s for s in ss if s.step_type.upper() == 'RETRIEVAL']
    kinds = Counter(s.step_type.upper() for s in ss)
    agents = Counter(s.agent_id for s in ss if s.agent_id)
    tools = {s.tool_name.strip().lower() for s in calls if s.tool_name}
    keys, call_keys = [action(s) for s in ss], [action(s) for s in calls]
    states = [status(s) for s in ss]
    observations = tool_observations(ss)
    call_states = [status(observations[s.step_index]) for s in calls]
    known_calls = all(v in {'SUCCESS','FAILED','ERROR'} for v in call_states)
    known_states = all(v in {'SUCCESS','FAILED','ERROR'} for v in states)
    errors = sum(error(s) for s in ss)
    tool_errors = sum(error(observations[s.step_index]) for s in calls)
    success = call_states.count('SUCCESS') if known_calls else None
    failed = tool_errors if known_calls else None
    lengths = [len(text(s.retrieved_docs)) for s in retrievals if s.retrieved_docs is not None]
    all_docs = len(lengths) == len(retrievals)
    latency = [s.metadata['latency_ms'] for s in calls if isinstance(s.metadata.get('latency_ms'), (int,float)) and s.metadata['latency_ms'] >= 0]
    confidence = [d['score'] for s in retrievals if isinstance(s.retrieved_docs,list) for d in s.retrieved_docs
                  if isinstance(d,dict) and isinstance(d.get('score'),(int,float))]
    dominant = agents.most_common(1)
    result = dict(trajectory_id=record.trajectory_id, task_id=record.task_id, source=record.source or 'UNKNOWN',
                  fault_injected=int(record.fault_injected), fault_type=record.fault_type or 'NONE',
                  outcome_status=record.outcome.status or 'UNKNOWN', origin_step=record.origin_step,
                  total_steps=n, total_agents=len(agents), total_tool_calls=len(calls),
                  total_tool_responses=kinds['TOOL_RESPONSE']+kinds['TOOL_RESULT'],
                  total_retrieval_steps=len(retrievals), total_llm_steps=kinds['LLM_CALL'],
                  total_handoffs=kinds['HANDOFF'], total_retries=kinds['RETRY'], total_errors=errors,
                  unique_tools_used=len(tools), unique_agents=len(agents), tool_call_count=len(calls),
                  successful_tool_call_count=success, failed_tool_call_count=failed,
                  tool_success_rate=ratio(success,len(calls)) if success is not None else None,
                  tool_error_count=tool_errors, duplicate_tool_call_count=repeats(call_keys),
                  repeated_same_arguments_count=repeats(call_keys), unique_tool_count=len(tools),
                  tool_diversity=ratio(len(tools),len(calls)),
                  average_tool_latency_ms=mean(latency), max_tool_latency_ms=max(latency) if latency else None,
                  repeated_action_count=repeats(keys), repeated_tool_argument_count=repeats(call_keys),
                  consecutive_identical_action_max=run_max(keys), retry_count=kinds['RETRY'],
                  loop_indicator=int(max(Counter(keys).values(),default=0) >= LOOP_REPEAT_THRESHOLD),
                  premature_termination_indicator=None,
                  max_repeated_sequence_length=repeated_sequence(keys),
                  steps_since_last_successful_action=steps[-1]['steps_since_last_successful_action'] if steps else None,
                  retrieval_count=len(retrievals),
                  empty_retrieval_count=sum(s.retrieved_docs in ([], '', {}) for s in retrievals) if all_docs else None,
                  retrieval_failure_count=sum(error(s) for s in retrievals),
                  total_retrieved_text_length=sum(lengths) if all_docs else None,
                  average_retrieved_text_length=mean(lengths), max_retrieved_text_length=max(lengths) if lengths else None,
                  average_retrieval_confidence=mean(confidence), minimum_retrieval_confidence=min(confidence) if confidence else None,
                  agent_count=len(agents), unique_agent_count=len(agents),
                  agent_switch_count=sum(a.agent_id != b.agent_id for a,b in zip(ss,ss[1:]) if a.agent_id and b.agent_id),
                  handoff_count=kinds['HANDOFF'], dominant_agent=dominant[0][0] if dominant else 'UNKNOWN',
                  dominant_agent_step_count=dominant[0][1] if dominant else 0,
                  sender_receiver_similarity=None, handoff_information_loss_score=None,
                  success_step_count=states.count('SUCCESS') if known_states else None,
                  failed_step_count=sum(v in {'FAILED','ERROR'} for v in states) if known_states else None,
                  error_step_count=errors, failure_step_ratio=ratio(errors,n) if known_states else None,
                  first_error_step_position=next((ratio(s.step_index,n) for s in ss if error(s)),None))
    # Only observable tool-result presence; no assumption that a present result succeeded.
    result['tool_result_missing_count'] = sum(observations[s.step_index].output is None for s in calls)
    run = longest = 0
    for value in call_states:
        run = run+1 if value in {'FAILED','ERROR'} else 0
        longest = max(longest,run)
    result['consecutive_tool_failures_max'] = longest if known_calls else None
    for name, value in [('tool_calls',len(calls)),('retrievals',len(retrievals)),('retries',kinds['RETRY']),('errors',errors),('handoffs',kinds['HANDOFF'])]:
        result[f'{name}_per_step'] = ratio(value,n)
    for side in ['input','output']:
        for unit in ['chars','words']:
            result[f'total_{side}_{unit}'] = sum(s[f'{side}_length_{unit}'] for s in steps)
        result[f'average_step_{side}_length'] = mean([s[f'{side}_length_chars'] for s in steps])
    result['max_step_output_length'] = max((s['output_length_chars'] for s in steps),default=None)
    result['total_retrieved_chars'] = result['total_retrieved_text_length']
    result['average_retrieved_chars'] = result['average_retrieved_text_length']
    result.update(semantic)
    return result
