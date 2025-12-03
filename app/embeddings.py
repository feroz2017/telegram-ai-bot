"""
Embedding function wrapper for ChromaDB.

Provides thread-safe access to sentence transformer models for generating
text embeddings used in semantic search.
"""

import threading
from typing import List

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingFunction:
    """Thread-safe embedding function wrapper for ChromaDB and manual use."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model_lock = threading.Lock()
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = SentenceTransformer(self._model_name)
        return self._model

    def __call__(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()
