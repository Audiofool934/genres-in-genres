"""
Metrics Module for Genres in Genres.

Defines the "Physics of Music Style" by quantifying subjective concepts.
"""
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from typing import List, Dict, Any, Tuple
from .core import ArtistCareer, Track

class MusicMetrics:
    """
    Stateless calculator for music style metrics.
    """

    @staticmethod
    def get_album_centroids(career: ArtistCareer) -> Tuple[List[str], np.ndarray]:
        """
        Returns (album_names_sorted, centroids_matrix).
        """
        # Group by album
        albums = {}
        # Pre-sort tracks by date/album
        sorted_tracks = sorted(career.tracks, key=lambda t: (t.release_date, t.album))
        
        # We need to map track object to embedding
        # Ideally career.embeddings is aligned or we have a map
        # Let's rebuild map
        # Use file_path as key for stable mapping
        track_to_vec = {e.track_ref.file_path: e.vector for e in career.embeddings}
        
        track_bins = {}
        first_dates = {}
        
        for t in sorted_tracks:
            track_key = t.file_path
            if track_key not in track_to_vec: continue
            
            if t.album not in track_bins:
                track_bins[t.album] = []
                first_dates[t.album] = t.release_date
            
            track_bins[t.album].append(track_to_vec[track_key])
            
        # Sort albums strictly by release date
        sorted_album_names = sorted(track_bins.keys(), key=lambda a: first_dates[a])
        
        centroids = []
        for album in sorted_album_names:
            vecs = np.stack(track_bins[album])
            centroids.append(np.mean(vecs, axis=0))
            
        return sorted_album_names, np.stack(centroids)

    @staticmethod
    def calculate_style_velocity(career: ArtistCareer) -> Dict[str, float]:
        """
        Calculates the distance between consecutive albums.
        Metric: Euclidean distance between centroids (or Cosine Distance).
        Interpretation: How much did the style change from the previous album?
        """
        names, centroids = MusicMetrics.get_album_centroids(career)
        if len(names) < 2:
            return {}
            
        velocities = {}
        # velocity[album_i] = dist(album_{i-1}, album_i)
        
        for i in range(1, len(names)):
            # Using Cosine Distance (0-2) as logical "Style Distance"
            dist = cosine(centroids[i-1], centroids[i])
            velocities[names[i]] = float(dist)
            
        return velocities

    @staticmethod
    def calculate_novelty(career: ArtistCareer) -> Dict[str, float]:
        """
        Calculates the distance of an album from the cumulative centroid of ALL past work.
        Metric: Cosine Distance(Current_Album, Mean(All_Past_Albums)).
        Interpretation: Is this album a departure from the artist's established canon?
        """
        names, centroids = MusicMetrics.get_album_centroids(career)
        if len(names) < 2:
            return {}
            
        novelty_scores = {}
        
        # Accumulate past
        # We need weighted mean? Or simple mean of centroids?
        # Simple mean of past centroids represents "Canonical Style"
        
        running_sum = centroids[0].copy()
        count = 1
        
        for i in range(1, len(names)):
            past_centroid = running_sum / count
            current_centroid = centroids[i]
            
            dist = cosine(past_centroid, current_centroid)
            novelty_scores[names[i]] = float(dist)
            
            # Update history
            running_sum += current_centroid
            count += 1
            
        return novelty_scores

    @staticmethod
    def calculate_cohesion(career: ArtistCareer) -> Dict[str, float]:
        """
        Calculates 1 - Intra-Album Variance.
        Interpretation: High Cohesion = Consistent Style. Low Cohesion = Eclectic.
        """
        # We can reuse the logic from StyleAnalyzer or reimplement clean here
        # Let's reimplement for independence
        # Use file_path as key for stable mapping
        track_to_vec = {e.track_ref.file_path: e.vector for e in career.embeddings}
        
        album_vecs = {}
        for t in career.tracks:
            track_key = t.file_path
            if track_key not in track_to_vec: continue
            if t.album not in album_vecs: album_vecs[t.album] = []
            album_vecs[t.album].append(track_to_vec[track_key])
            
        results = {}
        for album, vecs in album_vecs.items():
            if not vecs: continue
            mat = np.stack(vecs)
            centroid = np.mean(mat, axis=0)
            # Variance = Mean Cosine Distance from Centroid
            dists = [cosine(v, centroid) for v in vecs]
            variance = np.mean(dists)
            
            # Cohesion = 1 - Variance (assuming var is small, usually < 0.5)
            # Or just normalize. Let's return raw variance for now, invert in UI?
            # User asked for Cohesion Index. 
            results[album] = 1.0 - float(variance)
            
        return results

