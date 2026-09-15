# Review 3 diagnosis demonstration

All values below came from the saved models and captured trajectories. No models
were retrained. Ground truth was displayed only after diagnosis generation.

| Example | Failure probability | Predicted subtype | Localizer step | Diagnosis confidence |
|---|---:|---|---:|---:|
| traj_19bc05a7 | 0.0043616880 | Not run | Not run | Unavailable |
| AF-0005_fault_TOOL_WRONG_ARGUMENT_step_5 | 0.9932629274 | TOOL_WRONG_ARGUMENT | 5 | 0.8577040987 |
| AF-0095_fault_CTRL_LOOP_step_7 | 0.9964236813 | CTRL_EXCESSIVE_EXPLORATION | 5 | 0.5967166133 |

The SUCCESS recording exits with NO_FAILURE_DETECTED. Taxonomy and localization
are skipped. It belongs to the detector's training data, not an independent test.

## TOOL example

Category confidence 0.8821721683; subtype confidence 0.8866666667; root confidence
0.9519469738. Ranked hypothesis scores: Step 5 = 0.796517; Step 2 = 0.447113;
Step 7 = 0.400502. Selected hypothesis: verifying tool arguments at Step 5 might
have allowed success. Observed Step 5 is a researcher TOOL_CALL to calculator,
input query "calculate nonsense", output result 42. The record has no explicit
error at that step. Correctness is a model-guided hypothesis, not established
from the generic task alone. Downstream chronology: 5 -> 6 -> 7 -> 8.

Evaluation-only truth: TOOL_WRONG_ARGUMENT at Step 5. Both predictions match.

## CONTROL example

Category confidence 0.74; subtype confidence 0.7533333333; root confidence
0.5378795862. Ranked hypothesis scores: Step 5 = 0.553940; Step 7 = 0.464959;
Step 2 = 0.411570. Selected hypothesis: applying a task-relevant exploration
stopping criterion at Step 5 might have allowed success. Downstream chronology:
5 -> 6 -> 7 -> 8 -> 9 -> 10.

Evaluation-only truth: CTRL_LOOP at Step 7. The category matches, but the subtype
and root step are incorrect. Orchestration preserves the predictions rather
than substituting injected labels.

## Confidence and boundaries

Hypothesis score uses .50 localization + .25 taxonomy + .15 observable evidence
+ .10 structural compatibility, renormalizing available weights.
Diagnosis confidence uses .40 root + .30 category + .20 subtype + .10 margin.
For TOOL: .40(0.95194697) + .30(0.88217217) + .20(0.88666667)
+ .10(0.34940325) = 0.85770410. These are uncalibrated scores.

All three results have counterfactual_status NOT_RUN. Replay requests are only
proposals. Chronological propagation is not verified causal propagation.
Current models inherit single-class detector evaluation, source confounding,
duplicate feature vectors and small synthetic training data limitations.

## Verification

59 relevant tests passed, including 12 new deterministic diagnosis tests. Checks
cover early exit, model integration, ranking, deduplication, confidence gates,
chronological order, replay interpretation and invariance to changed truth labels.
The success, TOOL and CONTROL CLIs were executed; JSON mode was also verified.
