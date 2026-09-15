"""Optional pretrained embeddings with persistent, model-specific text caching."""
import hashlib
import json
import math
from pathlib import Path
from .common import text, mean

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def cosine(a, b):
    denominator = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    if not denominator:
        return None
    return max(-1.0, min(1.0, sum(x*y for x, y in zip(a, b)) / denominator))


class Embeddings:
    def __init__(self, cache_dir, enabled=True):
        self.enabled = enabled
        self.directory = Path(cache_dir)
        self.path = self.directory / 'embeddings.json'
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.model = None

    def key(self, value):
        return hashlib.sha256((MODEL + '\0' + value).encode()).hexdigest()

    def prepare(self, values):
        if not self.enabled:
            return
        missing = sorted({v for v in values if v and self.key(v) not in self.cache})
        if missing:
            from sentence_transformers import SentenceTransformer
            self.directory.mkdir(parents=True, exist_ok=True)
            self.model = SentenceTransformer(MODEL, cache_folder=str(self.directory / 'models'))
            vectors = self.model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            self.cache.update({self.key(v): e.tolist() for v, e in zip(missing, vectors)})
            self.path.write_text(json.dumps(self.cache), encoding='utf-8')

    def get(self, value):
        return self.cache.get(self.key(value)) if self.enabled and value else None


def step_text(step):
    # Output is the current semantic state; input is a fallback only.
    return text(step.output) or text(step.input)


def extract(record, original, embeddings):
    rows, previous = [], None
    task = embeddings.get(text(original))
    for step in record.steps:
        vector = embeddings.get(step_text(step))
        similarity = cosine(task, vector) if task is not None and vector is not None else None
        consecutive = cosine(previous, vector) if previous is not None and vector is not None else None
        rows.append({'task_step_similarity': similarity,
                     'semantic_drift': 1 - similarity if similarity is not None else None,
                     'previous_step_similarity': consecutive})
        if vector is not None:
            previous = vector
    result = {}
    for column, prefix in [('task_step_similarity', 'task_step_similarity'), ('semantic_drift', 'semantic_drift')]:
        values = [r[column] for r in rows if r[column] is not None]
        for aggregate, fn in [('avg', mean), ('min' if column == 'task_step_similarity' else 'max', min if column == 'task_step_similarity' else max), ('final', lambda v: v[-1])]:
            result[f'{aggregate}_{prefix}'] = fn(values) if values else None
    points = [(s.step_index, r['semantic_drift']) for s, r in zip(record.steps, rows) if r['semantic_drift'] is not None]
    slope = None
    if len(points) >= 2:
        mx, my = mean([p[0] for p in points]), mean([p[1] for p in points])
        slope = sum((x-mx)*(y-my) for x,y in points) / sum((x-mx)**2 for x,y in points)
    result['semantic_drift_slope'] = slope
    consecutive = [r['previous_step_similarity'] for r in rows if r['previous_step_similarity'] is not None]
    result.update(avg_consecutive_step_similarity=mean(consecutive), min_consecutive_step_similarity=min(consecutive) if consecutive else None,
                  max_semantic_change=1-min(consecutive) if consecutive else None)
    return rows, result
