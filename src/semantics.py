"""
Semantic Mapping Module.

Maps audio embeddings to human-readable semantic tags (genres, moods, instruments)
using the shared MuQ-MuLan embedding space. Tags are loaded from Music4All database.
"""
from typing import List, Tuple, Optional, Dict
import csv
import pickle
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

from pipeline_mulan import MuQMuLanEncoder

class SemanticMapper:
    """
    Maintains a registry of semantic tags and their embeddings.
    Tags are loaded from Music4All database instead of hardcoded lists.
    Allows querying for the nearest semantic tags for a given audio embedding.
    """
    
    def __init__(
        self, 
        encoder: MuQMuLanEncoder,
        metadata_dir: str,
        cache_dir: Optional[str] = None,
        device: str = "cpu"
    ):
        self.encoder = encoder
        self.device = device
        self.tags: List[str] = []
        self.tag_embeddings: Optional[torch.Tensor] = None
        self.metadata_dir = metadata_dir
        self.cache_dir = cache_dir

    @staticmethod
    def load_tags_from_cache(cache_path: str, max_tags: Optional[int] = None) -> Optional[List[str]]:
        """
        Load tags from cached pickle file.
        
        Args:
            cache_path: Path to tags_cache.pkl file
            max_tags: Optional limit on number of tags (if None, returns all cached tags)
            
        Returns:
            List of tags, or None if cache doesn't exist
        """
        cache_file = Path(cache_path)
        if not cache_file.exists():
            return None
        
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        
        tags = data.get("tags", [])
        if max_tags is not None and len(tags) > max_tags:
            tags = tags[:max_tags]
            
        print(f"[SemanticMapper] Loaded {len(tags)} tags from cache: {cache_path}")
        return tags

    @staticmethod
    def load_tags_from_metadata(metadata_dir: str, max_tags: int = 2000) -> List[str]:
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

    def initialize_tags(self, max_tags: int = 2000, cache_filename: str = "tags_cache.pkl"):
        """
        Load tags from cache (if available) or metadata CSVs, then encode them.
        
        Args:
            max_tags: Maximum number of tags to load
            cache_filename: Name of cache file to look for in cache_dir
        """
        tags = None
        
        # Try loading from cache first
        if self.cache_dir:
            cache_path = Path(self.cache_dir) / cache_filename
            tags = self.load_tags_from_cache(str(cache_path), max_tags=max_tags)
        
        # Fallback to loading from metadata CSVs
        if tags is None:
            tags = self.load_tags_from_metadata(self.metadata_dir, max_tags=max_tags)
        
        self.register_tags(tags)

    def register_tags(self, tags: List[str]):
        """Encodes a list of tags and adds/updates the registry."""
        new_tags = list(set(tags) - set(self.tags))
        all_tags = self.tags + new_tags
        
        print(f"[SemanticMapper] Encoding {len(all_tags)} tags...")
        with torch.no_grad():
            embeddings = self.encoder.encode_text(all_tags)
        
        self.tags = all_tags
        self.tag_embeddings = embeddings.to(self.device)
        print(f"[SemanticMapper] Registry updated: {len(self.tags)} tags.")

    def get_nearest_tags(self, embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Finds the top-k nearest tags for a given embedding vector."""
        query = torch.from_numpy(embedding).to(self.device).float()
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        query = F.normalize(query, p=2, dim=1)
        refs = F.normalize(self.tag_embeddings, p=2, dim=1)  # type: ignore
        
        sims = torch.mm(query, refs.t()).squeeze(0)
        topk_sims, topk_indices = torch.topk(sims, k)
        
        return [(self.tags[idx], score) for score, idx in zip(topk_sims.tolist(), topk_indices.tolist())]

    def batch_annotate(self, embeddings: List[np.ndarray], k: int = 1) -> List[List[str]]:
        """Annotates a batch of embeddings with their top-k tags."""
        return [[t[0] for t in self.get_nearest_tags(emb, k=k)] for emb in embeddings]
