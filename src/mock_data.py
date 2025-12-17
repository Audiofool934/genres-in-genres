"""
Mock Data Generator.

Simulates realistic 512-dim MuLan embeddings representing an artist's career
with evolving styles (clusters) and temporal progression.
"""
import numpy as np
import datetime
from typing import List, Tuple
from .core import Track, ArtistCareer, StyleEmbedding

class MockDataGenerator:
    """Generates synthetic artist careers."""
    
    @staticmethod
    def generate_career(
        artist_name: str = "Synthetic Artist",
        num_albums: int = 5,
        tracks_per_album: int = 10,
        start_year: int = 2010
    ) -> ArtistCareer:
        """
        Generates a career with distinct 'eras' (albums) acting as clusters in style space.
        """
        career = ArtistCareer(artist_name=artist_name)
        
        # Random seed for reproducibility
        np.random.seed(42)
        
        current_date = datetime.date(start_year, 1, 1)
        
        # Base "Artist Identity" vector
        artist_base = np.random.randn(512)
        artist_base /= np.linalg.norm(artist_base)
        
        for i in range(num_albums):
            album_name = f"Album {i+1}"
            
            # Album "Style" Center: Drift from artist base
            # We add some random vector to create distinct clusters aka "Genres in Genres"
            genre_drift = np.random.randn(512) * 0.5 
            album_center = artist_base + genre_drift
            album_center /= np.linalg.norm(album_center)
            
            # Release date increment
            current_date += datetime.timedelta(days=365 + np.random.randint(-30, 30))
            
            for j in range(tracks_per_album):
                track_title = f"{album_name} - Track {j+1}"
                
                # Track variation within album
                track_noise = np.random.randn(512) * 0.1
                track_vec = album_center + track_noise
                track_vec /= np.linalg.norm(track_vec) 
                
                # Create Track
                track = Track(
                    file_path=f"/mock/path/{album_name}/{track_title}.wav",
                    title=track_title,
                    album=album_name,
                    release_date=current_date,
                    duration=180.0
                )
                
                # Create Embedding
                emb = StyleEmbedding(vector=track_vec.astype(np.float32), track_ref=track)
                
                career.add_track(track)
                career.embeddings.append(emb)
                
        return career

def generate_blobs(n_samples=100, centers=3, n_features=512):
    """
    Utility to generate raw blobs for testing clustering purely.
    """
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=42)
    # Normalize to simulate cosine space embeddings
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X, y
