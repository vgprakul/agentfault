# AgentFault feature engineering

Run from the existing repository root:

```powershell
python -m pip install -r requirements-features.txt
python scripts/build_features.py --input dataset/agentfault-100.jsonl
python scripts/build_features.py --input data/trajectories --output data/features/real
python scripts/compare_features.py
python -m unittest tests.test_feature_engineering -v
```

The first semantic run downloads `sentence-transformers/all-MiniLM-L6-v2`.
Embeddings and downloaded weights are cached under the output directory's `.cache`.
Subsequent runs reuse model-specific SHA256 text keys. `--no-semantic` explicitly
leaves semantic columns empty, and the build report records that choice. Model
download/import failures are surfaced rather than silently substituting scores.
MiniLM uses its native token truncation for long texts; this is a lightweight v1
semantic view, not full-document reasoning.

## Existing schemas and input

The loader constructs the existing `injector.schema.AgentFaultRecord` and
`StepRecord` dataclasses. Record JSON, record JSONL, event JSONL, and JSON event
wrappers are supported, recursively for directories. Event conversion reuses
`events_to_record`, preserves separate tool results and other observed step types,
and avoids treating its hardcoded SUCCESS outcome default as observed truth.
Broken records and duplicate trajectory IDs are reported and skipped. A corrupt
event stream is skipped as a whole. Root labels follow the injector's existing
next-surviving-step rule when the original step was removed.

Original task text comes from an explicit `original_task`, the original first
step's `task`/`query`, or the event stream's trajectory-start input. It is never
inferred from later steps, IDs, or filenames. Synthetic trajectories that remove
the initial task-bearing step have unavailable task similarity/drift.

## Leakage boundaries

Both dataset modules define `FEATURE_COLUMNS`, `LABEL_COLUMNS`, and
`METADATA_COLUMNS`. The pipeline validates disjoint roles and rejects forbidden
label columns in predictor lists. Injection parameters and step metadata never
enter text, action, or size extraction. Only numeric `metadata.latency_ms` is
explicitly allowlisted. Provenance keys are removed recursively from payloads.
Actual observed mutated tool inputs/outputs remain execution evidence.

Step predictors use only the current prefix. `normalized_step_position` is saved
as **metadata** because the requested denominator is the complete trajectory
length; it must not be used in online prediction. `step_index` is metadata too.
The three-repeat loop rule counts identical `(step_type, normalized tool name,
canonical input)` actions anywhere in the trajectory. Dictionaries are key-sorted.
Repeats are not assumed to be retries: retries count explicit RETRY steps.

Grouped splitting prefers supplied `base_task_id`, otherwise existing `task_id`.
The effective group key is exported as `task_id`; no task is inferred from fault
filenames. Seed 42 shuffles groups, allocating floor(70%) and floor(15%), with the
remainder to test. A nonempty dataset gets at least one training group. Tiny group
counts may have empty validation/test sets. The step table joins the same manifest
on `trajectory_id`. Existing source split labels are not reused.

## Observable information and limitations

- Status is explicit execution status or an observable `output.error`. Present
  tool results are not proof of success. Unknown status yields null success/failure
  counts and rates; observable error counts count only recorded evidence.
- Synthetic steps mostly omit status. No errors recorded means zero *observed*
  errors, not proof of successful execution.
- Tool calls use inline output or the next matched separate result for the same
  tool. Pairing follows the sequential recorder; concurrent calls would require
  explicit call IDs. Step features never borrow a later tool response.
- Text sizes measure filtered actual payloads (canonical JSON for structured
  values), including numeric outputs. Semantic extraction uses meaningful text
  fields and skips document IDs, numeric-only values, and metadata.
- Retrieval length comes from `retrieved_docs`; document `score` is used only
  when present. A missing collection is unknown; an empty collection is zero.
- Latency and retrieval confidence are unavailable in the supplied trajectories.
- Premature termination is unavailable without an expected completion contract.
  Outcome labels, injected truncation parameters, and shorter lengths cannot
  supply such a contract.
- Handoff steps exist, but paired sent/received payloads do not. Sender/receiver
  similarity and information loss therefore remain unavailable.
- `final_*` semantic aggregates refer to the last meaningful text step. If no
  original task exists, task similarity and drift remain unavailable; consecutive
  similarity may still be available. Cosines are clipped to [-1,1], drift to [0,2]
  by its definition. Drift slope uses original recorded step indices.
- The synthetic dataset contains 100 faulty trajectories, 100 distinct task IDs,
  and no clean examples. Repeated templates across distinct IDs are not reliable
  provenance for grouping. The three clean real recordings share one task group.
  Their comparison with synthetic faults is illustrative and not a matched trial.

## Preprocessing

`build_preprocessor(level='trajectory', standardize=False)` returns a scikit-learn
ColumnTransformer using only declared predictors: median numeric imputation,
optional standardization, and categorical one-hot encoding with unknown-category
handling. Read CSVs with pandas, join the split manifest, and **fit only on train**;
use transform on validation/test. Columns entirely missing during training are
retained by sklearn's explicit `keep_empty_features` fallback (zero at this encoded
stage). Raw CSV values remain null. No classifier is implemented.

## Files

`loader.py` validates existing data; `common.py` handles safe payloads;
`trajectory_features.py` and `step_features.py` define column contracts and feature
extraction; `semantic_features.py` isolates embeddings; `splitting.py` groups tasks;
`preprocessing.py` prepares train-fitted transforms; `pipeline.py` writes outputs.
Every output column is documented in `feature_dictionary.json` by level and role.
`build_report.json` records counts, model choice, seed, and skipped inputs.

The original `.gitignore` ignores `data/`, so generated outputs are present locally
but remain ignored by Git. The archive's legacy tests import a missing
`TrajectoryGenerator`; full test discovery reports those two pre-existing errors.
