"""
Analysis Logic for Genres in Genres.

Implements algorithms for style trajectory, diversity analysis, and sub-genre clustering.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine
from typing import List, Dict, Any, Tuple, Optional
from .core import ArtistCareer, StyleEmbedding, Track
from .semantics import SemanticMapper
from .metrics import MusicMetrics

class StyleAnalyzer:
    """
    Analyzes an ArtistCareer to find patterns in style evolution.
    """
    
    def __init__(self, career: ArtistCareer, semantic_mapper: Optional[SemanticMapper] = None):
        self.career = career
        self.mapper = semantic_mapper
        
        # Cache for matrix form
        self._X = np.stack([e.vector for e in career.embeddings]) if career.embeddings else None

    @property
    def X(self) -> np.ndarray:
        if self._X is None:
            self._X = np.stack([e.vector for e in self.career.embeddings])
        return self._X

    def compute_intra_variance(self, by_album: bool = True) -> Dict[str, float]:
        """
        Computes the diversity (variance) of style within albums or the whole career.
        Deprecated: Use MusicMetrics.calculate_cohesion (inverse)
        """
        if self.X is None:
            raise ValueError("No embeddings available for variance computation.")

        results = {}
        
        if by_album:
            cohesion = MusicMetrics.calculate_cohesion(self.career)
            # variance approx 1 - cohesion
            for alb, score in cohesion.items():
                results[alb] = 1.0 - score
        else:
            centroid = np.mean(self.X, axis=0)
            dists = [cosine(v, centroid) for v in self.X]
            results["total"] = float(np.mean(dists))
            
        return results

    def reduce_dimension(self, method: str = "pca", n_components: int = 2) -> np.ndarray:
        """
        Projects embeddings to 2D/3D for visualization.
        """
        if self.X is None:
            raise ValueError("No embeddings available for dimensionality reduction.")
            
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            n_samples = len(self.X)
            if n_samples <= 2:
                raise ValueError("TSNE requires at least three samples.")
            perplexity = min(30, max(2, n_samples - 1))
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        elif method == "umap":
            try:
                import umap
            except ImportError as exc:
                raise ImportError(
                    "UMAP is unavailable. Install it with `pip install umap-learn`."
                ) from exc
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        result = reducer.fit_transform(self.X)
        return np.asarray(result)

    def find_optimal_k(self, k_range: Tuple[int, int] = (2, 10), max_samples: int = 1000) -> int:
        """
        Automatically finds the optimal number of clusters using Silhouette Score.
        
        Silhouette Score is a widely-used method in both academia and industry for
        determining optimal cluster numbers. It measures how similar an object is to
        its own cluster (cohesion) compared to other clusters (separation).
        
        Score range: [-1, 1]
        - Higher scores indicate better-defined clusters
        - Score near 1: Well-separated clusters
        - Score near 0: Overlapping clusters
        - Score near -1: Samples assigned to wrong clusters
        
        Alternative methods commonly used:
        - Elbow Method: Visual inspection of within-cluster sum of squares (WCSS)
        - Gap Statistic: Statistical test for optimal K (Tibshirani et al., 2001)
        - Calinski-Harabasz Index: Ratio of between-cluster to within-cluster variance
        
        Args:
            k_range: (min_k, max_k) range to search
            max_samples: Maximum samples to use for silhouette calculation (for performance)
            
        Returns:
            Optimal K value
        """
        if self.X is None or len(self.X) < 2:
            raise ValueError("Cannot estimate optimal K without at least two embeddings.")
            
        min_k, max_k = k_range
        n_samples = len(self.X)
        
        # Limit search range based on data size
        max_k = min(max_k, n_samples - 1)
        min_k = max(2, min_k)
        
        if min_k >= max_k:
            return min_k
        
        # For large datasets, sample for faster computation
        if n_samples > max_samples:
            indices = np.random.choice(n_samples, max_samples, replace=False)
            X_sample = self.X[indices]
        else:
            X_sample = self.X
        
        best_k = min_k
        best_score = -1.0
        
        print(f"[StyleAnalyzer] Finding optimal K in range [{min_k}, {max_k}]...")
        
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_sample)
            
            # Silhouette score requires at least 2 clusters and 2 samples per cluster
            if len(set(labels)) < 2:
                continue
                
            score = silhouette_score(X_sample, labels)
            print(f"  K={k}: Silhouette Score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"[StyleAnalyzer] Optimal K = {best_k} (Silhouette Score = {best_score:.4f})")
        return best_k

    def cluster_songs(self, n_clusters: int = 3) -> Dict[int, List[Track]]:
        """
        Identifies sub-genres using K-Means clustering.
        Returns a mapping from Cluster ID to list of Tracks.
        """
        if self.X is None:
            raise ValueError("No embeddings available for clustering.")
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.X)
        
        clusters = {i: [] for i in range(n_clusters)}
        # Use embeddings index to get corresponding tracks
        for idx, label in enumerate(labels):
            clusters[label].append(self.career.embeddings[idx].track_ref)
            
        # Optional: Name clusters using SemanticMapper
        if self.mapper:
            self._name_clusters(kmeans.cluster_centers_, clusters)
            
        return clusters

    def _name_clusters(self, centers: np.ndarray, clusters: Dict[int, List[Track]]):
        """
        Annotates clusters with semantic tags based on their centroids.
        """
        # This is a bit "in-place" enhancement on the return dict or just printing?
        # Ideally we return a rich structure. For now, let's print or store in a property.
        print("\n[StyleAnalyzer] Cluster Semantics:")
        for i, center in enumerate(centers):
            tags = self.mapper.get_nearest_tags(center, k=3)
            tag_str = ", ".join([f"{t} ({s:.2f})" for t, s in tags])
            print(f"  Cluster {i} ({len(clusters[i])} tracks): {tag_str}")

    def generate_career_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive "Data Insight Report" for the artist.
        """
        if not self.career.embeddings:
            raise ValueError("No embeddings available for report generation.")
            
        # 1. Calc Metrics
        velocity = MusicMetrics.calculate_style_velocity(self.career)
        novelty = MusicMetrics.calculate_novelty(self.career)
        cohesion = MusicMetrics.calculate_cohesion(self.career)
        
        # 2. Identify Peaks
        max_velocity_album = max(velocity.items(), key=lambda x: x[1])[0] if velocity else "N/A"
        max_novelty_album = max(novelty.items(), key=lambda x: x[1])[0] if novelty else "N/A"
        
        # 3. Narrative Generation
        narrative = []
        narrative.append(f"### 📊 Career Resume: {self.career.artist_name}")
        
        # Introduction
        total_albums = len(set(t.album for t in self.career.tracks))
        narrative.append(f"Analyzed **{len(self.career.tracks)} tracks** across **{total_albums} eras**.")
        
        # Innovation Insight
        if max_novelty_album != "N/A":
             narrative.append(f"**Most Innovative Era**: The album *{max_novelty_album}* represents the sharpest departure from your previous discography (Novelty Score: {novelty[max_novelty_album]:.3f}).")
        
        # Cohesion Insight
        # Find lowest cohesion (Experimental) vs highest (Focused)
        if cohesion:
            min_coh = min(cohesion.items(), key=lambda x: x[1])[0]
            max_coh = max(cohesion.items(), key=lambda x: x[1])[0]
            narrative.append(f"**Most Eclectic Album**: *{min_coh}* shows the widest stylistic variety.")
            narrative.append(f"**Most Consistent Album**: *{max_coh}* finds you locking into a singular groove.")
            
        return {
            "velocity": velocity,
            "novelty": novelty,
            "cohesion": cohesion,
            "markdown": "\n\n".join(narrative)
        }

    def detect_career_stages(self) -> List[Tuple[int, int]]:
        """
        Detects major shifts in style using change point detection on the time-ordered series.
        Simple implementation: Peak finding in moving window dists.
        """
        # Placeholder for advanced segmentation
        return []
