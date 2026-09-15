"""Observable-only evidence and input validation."""
import math
from injector.schema import StepRecord
from feature_engineering.common import safe_payload, FORBIDDEN


def probability(value):
    if value is None: return None
    if isinstance(value,bool): raise ValueError('Expected numeric confidence')
    value=float(value)
    if not math.isfinite(value) or not 0<=value<=1: raise ValueError('Confidence must be finite and in [0,1]')
    return value


def clean(value):
    if isinstance(value,dict):
        return {k:clean(v) for k,v in safe_payload(value).items()
                if k.lower() not in FORBIDDEN|{'fault_category','fault_subtype','failure_label','label','target','split','source_filename','filename','injection_parameters','operator'}}
    if isinstance(value,list): return [clean(v) for v in value]
    return value


def observable_steps(record):
    indices=[s.step_index for s in record.steps]
    if any(type(i) is not int or i<0 for i in indices) or indices!=sorted(set(indices)):
        raise ValueError('Trajectory steps must have ordered unique integer indices')
    # Intentionally no access to record.outcome, fault fields, origin or step root labels.
    return [StepRecord(s.step_index,s.step_type,s.agent_id,clean(s.input),clean(s.output),
                       tool_name=s.tool_name,retrieved_docs=clean(s.retrieved_docs),status=s.status) for s in record.steps]


def evidence(step):
    result={'step_index':step.step_index,'step_type':step.step_type,'agent':step.agent_id}
    for name,value in [('tool_name',step.tool_name),('input',step.input),('output',step.output),('status',step.status),('retrieved_docs',step.retrieved_docs)]:
        if value is not None: result[name]=value
    if isinstance(step.output,dict) and step.output.get('error'): result['error']=step.output['error']
    return result


def weighted(signals,weights):
    available={k:probability(v) for k,v in signals.items() if v is not None and weights.get(k,0)>0}
    denominator=sum(weights[k] for k in available)
    return sum(weights[k]*v for k,v in available.items())/denominator if denominator else None
