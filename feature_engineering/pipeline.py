"""Build raw feature tables, provenance dictionary, and shared split manifest."""
import csv
import json
import math
from pathlib import Path
from . import loader, semantic_features, step_features, trajectory_features, splitting
from .common import FORBIDDEN, text


def validate_columns():
    for module in [step_features, trajectory_features]:
        columns = module.FEATURE_COLUMNS + module.LABEL_COLUMNS + module.METADATA_COLUMNS
        if len(columns) != len(set(columns)):
            raise ValueError('Overlapping column roles')
        if set(module.FEATURE_COLUMNS) & FORBIDDEN:
            raise ValueError('Forbidden target leakage column')


def dictionary():
    result = {}
    for level, module in [('trajectory',trajectory_features),('step',step_features)]:
        for role, columns in [('feature',module.FEATURE_COLUMNS),('label',module.LABEL_COLUMNS),('metadata',module.METADATA_COLUMNS)]:
            for name in columns:
                categorical = name in {'trajectory_id','task_id','source','fault_type','outcome_status','step_type','agent_id','tool_name','status','dominant_agent'}
                entry = dict(name=name, level=level, role=role, type='string' if categorical else 'number',
                    description=name.replace('_',' ').capitalize(),
                    formula=f'See {module.__name__}.extract: {name}',
                    missing_condition='UNKNOWN for missing categories; null when required observation is absent; zero for observed empty counts')
                result[f'{level}.{name}'] = entry
    formulas = {
        'semantic_drift':'1 - cosine(original_task, current_meaningful_step)',
        'task_step_similarity':'cosine(original_task, current_meaningful_step)',
        'previous_step_similarity':'cosine(current_step, preceding_meaningful_step)',
        'normalized_step_position':'step_index / total_steps; retrospective metadata, excluded from predictors',
        'tool_diversity':'unique_tool_count / tool_call_count; 0 when denominator is 0',
        'loop_indicator':'int(any normalized (step_type, tool_name, input) occurs at least 3 times)',
        'semantic_drift_slope':'OLS slope of drift against recorded step_index, requiring at least two text steps',
        'max_repeated_sequence_length':'Longest adjacent repeated block of normalized actions',
        'is_root_cause':'Existing label; otherwise injector-compatible origin/next-surviving-step rule; 0 for clean',
    }
    formulas.update({
        'total_steps':'len(record.steps)',
        'total_agents':'count distinct nonempty step.agent_id',
        'total_tool_calls':'count step_type == TOOL_CALL',
        'total_tool_responses':'count step_type in {TOOL_RESULT, TOOL_RESPONSE}',
        'total_retrieval_steps':'count step_type == RETRIEVAL',
        'total_llm_steps':'count step_type == LLM_CALL',
        'total_handoffs':'count step_type == HANDOFF',
        'total_retries':'count explicit step_type == RETRY; repetitions are not assumed retries',
        'total_errors':'count observable failed/error status or nonempty output.error',
        'unique_tools_used':'count distinct normalized nonempty tool names among TOOL_CALL steps',
        'successful_tool_call_count':'count matched tool observations with SUCCESS status; null if any status unknown',
        'failed_tool_call_count':'count matched tool observations with FAILED/ERROR status; null if any status unknown',
        'tool_success_rate':'successful_tool_call_count / tool_call_count; 0 for no calls, null for unknown status',
        'tool_error_count':'count matched tool observations with observable failure/error',
        'duplicate_tool_call_count':'sum(max(0, count(normalized tool name, canonical input)-1))',
        'consecutive_tool_failures_max':'longest run of failed matched tool observations; null if any status unknown',
        'tool_result_missing_count':'count calls with neither inline output nor matched separate output',
        'average_tool_latency_ms':'mean observed nonnegative TOOL_CALL metadata.latency_ms',
        'max_tool_latency_ms':'max observed nonnegative TOOL_CALL metadata.latency_ms',
        'repeated_action_count':'sum(count(action)-1) over unique (step_type, normalized tool name, canonical input)',
        'consecutive_identical_action_max':'longest adjacent run of identical normalized actions',
        'steps_since_last_successful_action':'number of observed steps after most recent explicit successful step; null if none',
        'empty_retrieval_count':'count retrieved_docs equal to empty list/dict/string; null if documents unavailable',
        'retrieval_failure_count':'count RETRIEVAL steps with observable error/failure',
        'total_retrieved_text_length':'sum meaningful retrieved document text characters; null if any documents unavailable',
        'average_retrieved_text_length':'mean meaningful retrieved document text characters per observed retrieval',
        'max_retrieved_text_length':'max meaningful retrieved document text characters per observed retrieval',
        'average_retrieval_confidence':'mean numeric retrieved document score, when supplied',
        'minimum_retrieval_confidence':'min numeric retrieved document score, when supplied',
        'agent_switch_count':'count adjacent steps with different known agent IDs',
        'dominant_agent':'most frequent observed agent ID; ties resolved by first occurrence',
        'dominant_agent_step_count':'max count of observed steps per agent',
        'success_step_count':'count explicit SUCCESS statuses; null if any step status unknown',
        'failed_step_count':'count explicit FAILED/ERROR statuses; null if any step status unknown',
        'failure_step_ratio':'failed_step_count / total_steps; null if statuses incomplete',
        'first_error_step_position':'first observable error step_index / total_steps; null if no error observed',
        'has_error':'int(observable FAILED/ERROR status or nonempty output.error)',
        'cumulative_unique_tools':'count distinct known tool names in TOOL_CALL steps through current step',
        'cumulative_unique_agents':'count distinct known agent IDs through current step',
        'repeated_action_count_so_far':'sum repeated occurrences of normalized actions through current step',
        'premature_termination_indicator':'unavailable: no observed expected completion contract',
        'sender_receiver_similarity':'unavailable: no paired sent and received payloads',
        'handoff_information_loss_score':'unavailable: no paired sent and received payloads',
        'max_semantic_change':'1 - min consecutive meaningful-step cosine similarity',
    })
    aliases = {
        'unique_agents':'total_agents','agent_count':'total_agents','unique_agent_count':'total_agents',
        'tool_call_count':'total_tool_calls','unique_tool_count':'unique_tools_used',
        'repeated_same_arguments_count':'duplicate_tool_call_count','repeated_tool_argument_count':'duplicate_tool_call_count',
        'retry_count':'total_retries','retrieval_count':'total_retrieval_steps','handoff_count':'total_handoffs',
        'total_retrieved_chars':'total_retrieved_text_length','average_retrieved_chars':'average_retrieved_text_length',
        'error_step_count':'total_errors',
    }
    for entry in result.values():
        name = entry['name']
        if name in aliases:
            entry['formula'] = f'Alias of {aliases[name]}: {formulas.get(aliases[name], aliases[name])}'
        if name.endswith('_per_step'):
            entry['formula'] = f'count observed {name[:-9]} / total_steps; 0 for no steps'
        if name.startswith('cumulative_') and name not in formulas:
            entry['formula'] = f'Count observed {name[11:]} from first step through current step inclusive'
        if name.startswith('previous_step_same_'):
            entry['formula'] = '1 if current and previous step have the same nonmissing ' + name[19:] + ', else 0'
        if name.startswith('is_') and name.endswith('_step'):
            entry['formula'] = 'Indicator of normalized step type: ' + name[3:-5].upper()
        if name.startswith(('input_length_', 'output_length_')):
            entry['formula'] = 'Character/whitespace-word count of provenance-filtered input/output; strings verbatim, other payloads canonical JSON'
        if name.startswith(('total_input_', 'total_output_')):
            entry['formula'] = 'Sum corresponding per-step payload character/word counts'
        if name.startswith(('average_step_', 'max_step_')):
            entry['formula'] = 'Mean/max corresponding per-step payload character counts'
        if name.startswith(('avg_', 'min_', 'max_', 'final_')) and ('similarity' in name or 'drift' in name):
            entry['formula'] = 'Aggregate over nonmissing meaningful-step values: ' + name + '; final means last meaningful step'
        if entry['role'] != 'feature':
            entry['formula'] = 'Copied from existing schema (group task_id prefers base_task_id when supplied)'
        if name in {'step_type','agent_id','tool_name','status'}:
            entry['formula'] = 'Observed StepRecord field; status normalized from explicit status/output.error; missing category UNKNOWN'
        if name in formulas:
            entry['formula'] = formulas[name]
        if name in {'premature_termination_indicator','sender_receiver_similarity','handoff_information_loss_score'}:
            entry['missing_condition'] = 'Unavailable: no expected completion contract or paired sent/received payload instrumentation'
    return result


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build(input_path, output_dir='data/features', semantic=True, random_state=42):
    validate_columns()
    records, originals, issues = loader.load(input_path)
    if not records:
        raise ValueError('No valid trajectories. ' + '; '.join(issues))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    embeddings = semantic_features.Embeddings(output / '.cache', enabled=semantic)
    embeddings.prepare([text(originals[r.trajectory_id]) for r in records] +
                       [semantic_features.step_text(s) for r in records for s in r.steps])
    trajectories, steps = [], []
    for record in records:
        local_sem, aggregate = semantic_features.extract(record, originals[record.trajectory_id], embeddings)
        local = step_features.extract(record, local_sem)
        trajectories.append(trajectory_features.extract(record, local, aggregate))
        steps.extend(local)
    for row in trajectories + steps:
        for name, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'Nonfinite feature: {name}')
            if value is not None and ('similarity' in name or name == 'semantic_drift'):
                if not (-1 <= value <= (2 if name == 'semantic_drift' else 1)):
                    raise ValueError(f'Invalid semantic range: {name}')
    manifest = splitting.split(records,random_state)
    for name, rows, module in [('trajectory_features',trajectories,trajectory_features),('step_features',steps,step_features)]:
        write_csv(output / f'{name}.csv', rows, module.METADATA_COLUMNS + module.LABEL_COLUMNS + module.FEATURE_COLUMNS)
    write_csv(output/'split_manifest.csv',manifest,['trajectory_id','task_id','split'])
    (output/'feature_dictionary.json').write_text(json.dumps(dictionary(),indent=2),encoding='utf-8')
    (output/'build_report.json').write_text(json.dumps(dict(input=str(input_path), trajectories=len(records),steps=len(steps),
        semantic_model=semantic_features.MODEL if semantic else None, random_state=random_state,issues=issues),indent=2),encoding='utf-8')
    return trajectories, steps, manifest, issues
