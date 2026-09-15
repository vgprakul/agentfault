"""Configuration and replay contracts; the trajectory schema stays in injector."""
from dataclasses import dataclass, field
import math


@dataclass
class DiagnosisConfig:
    top_k: int = 3
    failure_threshold: float = .5
    category_threshold: float = .5
    localization_threshold: float = .5
    hypothesis_threshold: float = .5
    tie_tolerance: float = .01
    weights: dict = field(default_factory=lambda: {'localization':.5, 'taxonomy':.25, 'evidence':.15, 'structural':.1})

    def __post_init__(self):
        if type(self.top_k) is not int or self.top_k<1: raise ValueError('top_k must be a positive integer')
        for value in [self.failure_threshold,self.category_threshold,self.localization_threshold,self.hypothesis_threshold,self.tie_tolerance]:
            if not math.isfinite(value) or not 0<=value<=1: raise ValueError('Thresholds must be in [0,1]')
        if set(self.weights)!={'localization','taxonomy','evidence','structural'}:
            raise ValueError('Expected localization, taxonomy, evidence, structural weights')
        if any(not math.isfinite(v) or v<0 for v in self.weights.values()) or not sum(self.weights.values()):
            raise ValueError('Weights must be finite, nonnegative and have positive sum')


@dataclass(frozen=True)
class ReplayRequest:
    trajectory_id: str
    target_step: int
    hypothesis_type: str
    proposed_change: str


@dataclass(frozen=True)
class ReplayResult:
    trajectory_id: str
    target_step: int
    hypothesis_type: str
    original_score: float
    replay_score: float


def interpret_replay(request, result, minimum_delta=.1):
    """Interpret supplied, higher-is-better scores; never execute or simulate replay."""
    if (request.trajectory_id,request.target_step,request.hypothesis_type)!=(result.trajectory_id,result.target_step,result.hypothesis_type):
        raise ValueError('Replay result does not match selected request')
    if not all(math.isfinite(v) for v in [result.original_score,result.replay_score,minimum_delta]) or minimum_delta<=0:
        raise ValueError('Replay scores must be finite and minimum_delta positive')
    delta=result.replay_score-result.original_score
    return {'status':'SUPPORTED' if delta>=minimum_delta else 'REJECTED', 'causal_importance':delta,
            'original_score':result.original_score,'replay_score':result.replay_score,
            'note':'Supplied replay evidence; interpretation assumes comparable higher-is-better scores and a controlled replay.'}
