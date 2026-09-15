from pathlib import Path


KNOWLEDGE_BASE = Path(__file__).parent.parent / "data" / "knowledge.txt"


def retrieve(query: str) -> str:
    text = KNOWLEDGE_BASE.read_text()

    # Temporary dumb retrieval.
    # We'll replace this with proper retrieval later.
    return text
