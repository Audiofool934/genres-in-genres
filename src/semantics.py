"""
Semantic Mapping Module (cache-first, no MuQ-MuLan dependency).

Uses precomputed MuQ-MuLan text embeddings stored under data/cache/tags to map
audio embeddings to human-readable semantic tags.
"""
from typing import List, Tuple, Optional, Dict
import csv
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


class SemanticMapper:
    """
    Maintains a registry of semantic tags and their cached embeddings.

    Tag embeddings are expected to be precomputed offline and stored alongside
    the tag list under data/cache/tags (see initialize_tags for details).
    """

    def __init__(
        self,
        cache_dir: str,
        metadata_dir: Optional[str] = None,
        encoder=None,
    ):
        self.cache_dir = Path(cache_dir)
        self.metadata_dir = Path(metadata_dir) if metadata_dir else None
        self.encoder = encoder
        self.tags: List[str] = []
        self.tag_embeddings: Optional[np.ndarray] = None

    @property
    def has_embeddings(self) -> bool:
        return self.tag_embeddings is not None or self.encoder is not None

    def initialize_tags(self, max_tags: int = 2000, cache_filename: str = "tags_cache.pkl"):
        """
        Load tags and (optionally) their embeddings from cache. Falls back to
        metadata CSVs for tag list if cache is missing.

        Cache formats supported:
        - tags_cache.pkl with keys {"tags": [...], "embeddings": np.ndarray | list}
        - Separate files in cache_dir: tag_embeddings.npy / tag_embeddings.pkl
        """
        cache_path = self.cache_dir / cache_filename
        tags: List[str] = []
        tag_embeddings: Optional[np.ndarray] = None

        if cache_path.exists():
            with cache_path.open("rb") as f:
                data = pickle.load(f)
            tags = data.get("tags", []) or []
            tag_embeddings = data.get("embeddings") or data.get("tag_embeddings")

        # Optional separate embedding file
        if tag_embeddings is None:
            npy_path = self.cache_dir / "tag_embeddings.npy"
            pkl_path = self.cache_dir / "tag_embeddings.pkl"
            if npy_path.exists():
                tag_embeddings = np.load(npy_path)
            elif pkl_path.exists():
                with pkl_path.open("rb") as f:
                    tag_embeddings = pickle.load(f)

        # Fallback: load tag list from metadata if cache tags missing
        if not tags and self.metadata_dir:
            tags = self.load_tags_from_metadata(self.metadata_dir, max_tags=max_tags)

        if max_tags and tags:
            tags = tags[:max_tags]
            if tag_embeddings is not None and len(tag_embeddings) >= len(tags):
                tag_embeddings = tag_embeddings[: len(tags)]

        if tag_embeddings is not None:
            tag_embeddings = np.asarray(tag_embeddings, dtype=np.float32)
            if tag_embeddings.shape[0] < len(tags):
                print(
                    f"[SemanticMapper] Tag embeddings shorter than tag list "
                    f"({tag_embeddings.shape[0]} < {len(tags)}); disabling embeddings."
                )
                tag_embeddings = None
            else:
                tag_embeddings = self._l2_normalize(tag_embeddings)

        self.tags = tags
        self.tag_embeddings = tag_embeddings
        print(
            f"[SemanticMapper] Loaded {len(self.tags)} tags from cache "
            f"(embeddings={'yes' if self.tag_embeddings is not None else 'no'})."
        )

    @staticmethod
    def load_tags_from_metadata(metadata_dir: Path, max_tags: int = 2000) -> List[str]:
        """
        Load tags from local Music4All metadata CSVs under `metadata_dir`.

        Reads:
        - id_tags.csv   (column: tags, comma-separated)
        - id_genres.csv (column: genres, comma-separated)

        Returns top `max_tags` by frequency (descending).
        """
        md = Path(metadata_dir)
        tags_path = md / "id_tags.csv"
        genres_path = md / "id_genres.csv"

        counter: Counter[str] = Counter()

        def _consume_csv(path: Path, field: str) -> None:
            if not path.exists():
                return
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    raw = (row.get(field) or "").strip()
                    if not raw:
                        continue
                    for tok in raw.split(","):
                        t = tok.strip()
                        if t:
                            counter[t] += 1

        _consume_csv(tags_path, "tags")
        _consume_csv(genres_path, "genres")

        most_common = [t for t, _ in counter.most_common(max_tags)]
        print(f"[SemanticMapper] Loaded {len(most_common)} tags from {md} (top {max_tags}).")
        return most_common

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms

    def get_nearest_tags(self, embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Finds the top-k nearest tags for a given embedding vector."""
        if self.tag_embeddings is None or not self.tags:
            return []

        query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.tag_embeddings.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: query {query.shape[1]} vs tags {self.tag_embeddings.shape[1]}"
            )
        query = self._l2_normalize(query)
        sims = query @ self.tag_embeddings.T  # [1, N]
        sims = sims.squeeze(0)
        top_indices = np.argsort(-sims)[:k]
        return [(self.tags[idx], float(sims[idx])) for idx in top_indices]

    def get_tag_vectors(self, tags: List[str]) -> Optional[np.ndarray]:
        """
        Returns normalized embedding vectors for the requested tags.
        If embeddings are unavailable or any tag is missing, tries to encode them
        using the runtime encoder if available.
        """
        # Case 1: All tags in cache
        if self.tag_embeddings is not None:
            tag_to_idx: Dict[str, int] = {t: i for i, t in enumerate(self.tags)}
            missing = [t for t in tags if t not in tag_to_idx]
            
            if not missing:
                indices = [tag_to_idx[t] for t in tags]
                return self.tag_embeddings[indices]
        
        # Case 2: Missing tags but have Encoder
        if self.encoder:
            print(f"[SemanticMapper] Encoding {len(tags)} tags on-the-fly...")
            try:
                # Encode all requested tags fresh to ensure consistency
                vecs = self.encoder.encode(tags)
                return self._l2_normalize(vecs)
            except Exception as e:
                print(f"[SemanticMapper] Encoding failed: {e}")
                return None

        # Case 3: Missing tags and No Encoder
        if self.tag_embeddings is not None:
            # We already identified missing tags in Case 1 block, if we are here it implies missing > 0
            # Recalculate missing for clarity if we didn't store it, or just generic error
            tag_to_idx = {t: i for i, t in enumerate(self.tags)}
            missing = [t for t in tags if t not in tag_to_idx]
            print(f"[SemanticMapper] Missing tag embeddings for: {missing} (and no encoder available)")
            return None
            
        return None

    def batch_annotate(self, embeddings: List[np.ndarray], k: int = 1) -> List[List[str]]:
        """Annotates a batch of embeddings with their top-k tags."""
        return [[t[0] for t in self.get_nearest_tags(emb, k=k)] for emb in embeddings]
