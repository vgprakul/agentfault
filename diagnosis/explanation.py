"""Observed chronological links are not proof of causal propagation."""
from .utils import evidence


def propagation_chain(steps,index):
    downstream=[s for s in steps if s.step_index>=index]
    chain=[]
    for position,step in enumerate(downstream):
        node={'step_index':step.step_index,'step_type':step.step_type,'agent':step.agent_id,
              'relationship':'suspected_origin' if position==0 else 'temporal_order_only',
              'causal_effect_verified':False}
        if position:
            previous=downstream[position-1]
            node['from_step']=previous.step_index
            if previous.output is not None and previous.output==step.input:
                node['relationship']='exact_output_input_match'
            elif previous.step_type.upper()=='TOOL_CALL' and step.step_type.upper() in {'TOOL_RESULT','TOOL_RESPONSE'} and previous.tool_name and previous.tool_name==step.tool_name:
                node['relationship']='adjacent_same_tool_call_response'
        chain.append(node)
    return chain


def explain(diagnosis):
    status=diagnosis['diagnosis_status']
    if status=='NO_FAILURE_DETECTED': return 'Failure probability is below the configured threshold. Taxonomy and localization were not required.'
    if status=='UNDETERMINED': return 'Diagnosis is undetermined: '+ '; '.join(diagnosis['reasons'])
    selected=diagnosis['selected_hypothesis']
    observations='; '.join(selected['observations']) or 'No explicit error or repeated-action evidence was observed at this step.'
    chain=' -> '.join(f"Step {s['step_index']}" for s in diagnosis['propagation_chain'])
    return (f"Leading hypothesis: {selected['hypothesis']} Observed evidence: {observations} "
            f"Downstream chronology: {chain}. Chronological links do not establish causal effects. "
            f"Diagnosis confidence: {diagnosis['diagnosis_confidence']:.3f} (uncalibrated weighted score). "
            f"Counterfactual status: {diagnosis['counterfactual_status']}.")
