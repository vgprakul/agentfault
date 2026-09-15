"""Deterministic task-group allocation; no trajectory-level randomization."""
import random


def split(records, random_state=42):
    groups = sorted({r.task_id for r in records})
    random.Random(random_state).shuffle(groups)
    n = len(groups)
    train = max(1, int(n * .70)) if n else 0
    validation = int(n * .15)
    assignment = {task: 'train' if i < train else 'validation' if i < train+validation else 'test'
                  for i, task in enumerate(groups)}
    rows = [dict(trajectory_id=r.trajectory_id, task_id=r.task_id, split=assignment[r.task_id]) for r in records]
    validate(rows)
    return rows


def validate(rows):
    seen = {}
    for row in rows:
        if row['task_id'] in seen and seen[row['task_id']] != row['split']:
            raise ValueError('Task appears in multiple splits')
        seen[row['task_id']] = row['split']
