"""Central taxonomy-to-counterfactual templates, always phrased as hypotheses."""
from injector.faults import FaultType as F
from ml.taxonomy.utils import SUBTYPE_TO_CATEGORY, CATEGORIES
from feature_engineering.common import action, error
from .utils import evidence, probability

# (suspected decision/field, proposed change). These are proposals, not observed facts.
TEMPLATES={
 F.PLAN_MISSING_STEP:('plan','include the required task action'),
 F.PLAN_INCORRECT_SUCCESS_CRITERIA:('success criteria','check criteria against the original task'),
 F.PLAN_INVALID_DEPENDENCY:('dependency','satisfy prerequisites before dependent actions'),
 F.TOOL_WRONG_TOOL:('tool selection','choose a tool appropriate to the requested operation'),
 F.TOOL_WRONG_ARGUMENT:('tool arguments','verify arguments against the requested operation'),
 F.TOOL_UNNECESSARY_CALL:('tool invocation','omit an unnecessary call after checking task requirements'),
 F.TOOL_FAILED_RECOVERY:('recovery','handle the observed tool error before continuing'),
 F.KNOW_RETRIEVAL_FAILURE:('retrieval','obtain relevant supporting information'),
 F.KNOW_CITATION_MISMATCH:('citation','verify the citation against the supporting document'),
 F.KNOW_CONTEXT_TRUNCATION:('context','preserve task-relevant context'),
 F.MA_INCORRECT_HANDOFF:('handoff recipient','route the task to the responsible agent'),
 F.MA_INFORMATION_LOSS:('handoff context','preserve required information during handoff'),
 F.MA_MISSING_RESPONSIBILITY:('responsibility','assign the required action to an agent'),
 F.MA_ROLE_OVERLAP:('role assignment','resolve overlapping responsibilities'),
 F.CTRL_LOOP:('repeated action','stop repeated non-progressing actions and reassess the task'),
 F.CTRL_PREMATURE_TERMINATION:('termination','verify required work is complete before stopping'),
 F.CTRL_EXCESSIVE_EXPLORATION:('exploration','apply a task-relevant stopping criterion'),
 F.SEC_PROMPT_INJECTION:('instruction trust','validate instruction provenance before following it'),
 F.SEC_UNAUTHORIZED_ACTION:('authorization','check authorization before executing the action'),
 F.SEC_CROSS_USER_DATA_LEAKAGE:('data access','verify the task permits the accessed data'),
}
if set(TEMPLATES)!=set(F): raise RuntimeError('Missing taxonomy hypothesis template')


def generate(steps,taxonomy,localization,top_k=3):
    category=taxonomy.get('predicted_category'); subtype=taxonomy.get('predicted_subtype')
    if category is not None and category not in CATEGORIES: raise ValueError('Unknown predicted category')
    if subtype is not None and SUBTYPE_TO_CATEGORY.get(subtype)!=category: raise ValueError('Subtype/category mismatch')
    candidates=localization.get('ranked_steps',[])
    by_index={s.step_index:(i,s) for i,s in enumerate(steps)}
    seen=set(); ranked=[]
    for candidate in candidates:
        index=candidate['step_index']
        if type(index) is not int or index not in by_index or index in seen: raise ValueError('Invalid/duplicate localized step')
        seen.add(index)
        ranked.append({'step_index':index,'probability':probability(candidate.get('probability'))})
    ranked.sort(key=lambda r:(-(r['probability'] if r['probability'] is not None else -1),r['step_index']))
    result=[]
    for candidate in ranked[:top_k]:
        index=candidate['step_index']; position,step=by_index[index]
        field,change=TEMPLATES[F(subtype)] if subtype else ('task action',f'review this action for the predicted {category or "unknown"} category')
        repeated=sum(action(s)==action(step) for s in steps[:position+1])>=3
        observed=[]
        if error(step): observed.append('Observable error/failure status or error payload')
        if repeated: observed.append('This normalized action has occurred at least three times in the observed prefix')
        if step.step_type.upper()=='RETRIEVAL' and step.retrieved_docs==[]: observed.append('Retrieved document list is empty')
        compatible={'TOOL':{'TOOL_CALL','TOOL_RESULT','TOOL_RESPONSE','RETRY'},'KNOWLEDGE':{'RETRIEVAL'},
                    'MULTI_AGENT':{'HANDOFF'},'PLANNING':{'LLM_CALL'},'CONTROL':{'LLM_CALL','TOOL_CALL','RETRY'},'SECURITY':set()}
        structural=float(step.step_type.upper() in compatible.get(category,set()))
        if category=='TOOL' and not structural:
            field='task action'
            change='review whether this action supplied adequate context for downstream tool use'
        result.append({'step_index':index,'hypothesis_type':subtype or category,'suspected_field':field,
            'tool_name':step.tool_name,'proposed_change':change,
            'hypothesis':f'If the {field} decision at Step {index} had been different so as to {change}, the task might have succeeded; this requires verification.',
            'evidence':evidence(step),'surrounding_steps':[evidence(s) for s in steps[max(0,position-1):position+2] if s.step_index!=index],
            'observations':observed,'signals':{'localization':candidate['probability'],'taxonomy':probability(taxonomy.get('category_confidence')),
                'evidence':float(bool(observed)),'structural':structural},
            'limitations':['Taxonomy suggests a failure type; the payload alone may not establish correctness. No counterfactual has been executed.']})
    return result,ranked
