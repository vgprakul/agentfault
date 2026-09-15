"""Transparent scoring and small-margin earlier-step selection."""
from .utils import weighted


def rank_hypotheses(hypotheses,config):
    unique={}
    for hypothesis in hypotheses:
        item={**hypothesis,'score':weighted(hypothesis['signals'],config.weights)}
        key=(item['step_index'],item.get('tool_name'),item['suspected_field'])
        if key not in unique or (item['score'] or 0)>(unique[key]['score'] or 0): unique[key]=item
    ranked=sorted(unique.values(),key=lambda h:(-(h['score'] or 0),h['step_index']))[:config.top_k]
    if not ranked: return [],None
    eligible=[h for h in ranked if h['score'] is not None and h['score']>=config.hypothesis_threshold]
    if not eligible: return ranked,None
    best=eligible[0]['score']
    # Preserve a descending score table; identify the selected close-score candidate separately.
    selected=min((h for h in eligible if best-h['score']<=config.tie_tolerance),key=lambda h:h['step_index'])
    return ranked,selected
