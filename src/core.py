"""
Core Data Structures for Genres in Genres Project.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import datetime
import numpy as np

@dataclass
class Track:
    """Represents a single track in an artist's career."""
    file_path: str
    title: str
    album: str
    release_date: datetime.date
    # Optional metadata
    duration: Optional[float] = None
    original_genre_tags: List[str] = field(default_factory=list)

    @property
    def year(self) -> int:
        return self.release_date.year

@dataclass
class StyleEmbedding:
    """Wrapper for the 512-dim MuLan embedding."""
    vector: np.ndarray  # Shape: (512,)
    track_ref: Track

    def __post_init__(self):
        if self.vector.shape != (512,):
            raise ValueError(f"StyleEmbedding expects 512-dim vector, got {self.vector.shape}")

@dataclass
class ArtistCareer:
    """Represents an artist's entire discography, sorted by time."""
    artist_name: str
    tracks: List[Track] = field(default_factory=list)
    embeddings: List[StyleEmbedding] = field(default_factory=list)

    def add_track(self, track: Track):
        self.tracks.append(track)
        self.tracks.sort(key=lambda x: x.release_date)

    def add_embedding(self, embedding: StyleEmbedding):
        self.embeddings.append(embedding)
        # Ensure embeddings are aligned with tracks? 
        # For simplicity, we might store embedding in Track or keep separate parallel list.
        # But given we process careers, maybe binding them is better. 
        # Let's keep them somewhat decoupled but consistent.

    @property
    def timeline(self) -> List[datetime.date]:
        return [t.release_date for t in self.tracks]
