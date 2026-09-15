# Diagnosis orchestration

This package composes the existing FailureDetector, TaxonomyPredictor and
RootCauseLocalizer APIs. It does not train models or require an LLM. Trajectories
remain injector.schema.AgentFaultRecord / StepRecord. File loading reuses
feature_engineering.loader, including its existing root-label validation; those
labels do not enter diagnosis generation.

## Commands

```powershell
python scripts/diagnose_trajectory.py --trajectory-id AF-0005_fault_TOOL_WRONG_ARGUMENT_step_5 --show-ground-truth
python scripts/diagnose_trajectory.py --trajectory-id AF-0095_fault_CTRL_LOOP_step_7 --json
python scripts/diagnose_trajectory.py --trajectory-id traj_19bc05a7
```

Default input locations are dataset/agentfault-100.jsonl and data/trajectories;
feature caches are data/features and data/features/real. Repeat
`--trajectory-input` / `--feature-dir` to use other existing inputs. Cache IDs,
task IDs and step indices are checked. If features are absent, the CLI provides
the existing feature-building command instead of silently using a different
embedding/preprocessing configuration. Cache payload freshness is not verified.

Output is results/diagnosis/<trajectory_id>_diagnosis.json. `--json` prints only
JSON; `--show-ground-truth` adds evaluation-only comparison to stdout, never to
saved diagnosis or predictor inputs. IDs select records, never infer labels.
Payloads are evidence data, not instructions to execute.

## APIs

`diagnose_trajectory(id)` loads caches, calls the saved models and saves JSON.
`DiagnosisAgent.diagnose(record, trajectory_feature_row, step_feature_rows)`
supports injected predictors for testing. Predictor objects load lazily and can
be reused by retaining the agent. `assemble_diagnosis(record, failure_detection,
taxonomy, localization)` consumes already-computed predictions without loading
models. Returned objects are JSON-compatible dictionaries.

Failure probability below the configurable threshold (default .5) returns
NO_FAILURE_DETECTED before taxonomy/localization. Missing or low confidence
returns UNDETERMINED. Default category, localization and hypothesis thresholds
are .5. HYPOTHESIS_SELECTED means a tentative explanation, not proven causation.

## Ranking

Generate at most top_k hypotheses (default 3) from the highest localizer
probabilities. Templates centrally reference existing FaultType enum members.
Same step/tool/suspected-field hypotheses are deduplicated.

Score = weighted mean of available signals:

- .50 localization probability
- .25 category confidence
- .15 observed evidence indicator: observable error, empty retrieval, or at
  least three identical normalized actions through that step
- .10 structural compatibility: category/step-type match

No error evidence is proof of a particular subtype. Structural compatibility
does not assert an observed dependency. Weights are configurable through
DiagnosisConfig; missing signals are omitted and weights renormalized. Scores
are sorted descending. Among eligible candidates within .01 of the highest
score, the earlier step is selected; the sorted table itself is not reordered.

Diagnosis confidence = .40 selected-step localizer probability + .30 category
confidence + .20 subtype confidence + .10 hypothesis margin. Missing subtype
confidence renormalizes the other weights. Margin is max(0, selected score minus
best other score); no runner-up gives margin 0. This is an uncalibrated score.
The top-level predicted root step preserves the localizer output; the selected
hypothesis may choose a different step through deterministic ranking.

Propagation lists downstream steps in recorded order. Exact output/input matches
and adjacent same-tool call/response pairs receive specific relationship tags;
other links are explicitly temporal_order_only. None establish causal effects.

## Replay boundary

ReplayRequest describes a proposed test but does not execute it. Default status
is NOT_RUN. ReplayResult must match trajectory, step and hypothesis type.
interpret_replay computes replay_score - original_score assuming comparable
higher-is-better scores. Delta >= .1 supports the hypothesis; smaller or negative
delta rejects it. Supplying a rejecting result to assemble_diagnosis returns
UNDETERMINED. No replay executor, invented scores or calibration is included.

## Limitations

The current failure detector has no successful validation/test examples; source
differences and duplicate synthetic features can inflate confidence. The three
success recordings belong to one training task. Taxonomy/localizer errors carry
through unchanged. Generic task prompts often provide insufficient information
to verify tool argument correctness or the success counterfactual. Explanations
therefore say 'might have succeeded' and require verification. Complete cached
trajectories are used; this is not early prediction or causal proof.
