"""
Visualization Module for Genres in Genres.

Provides plotting functions for career trajectories and style distributions.
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import seaborn as sns
import numpy as np
import datetime
from scipy.spatial import ConvexHull
from typing import Dict, List, Optional
from .core import ArtistCareer, Track
from .analysis import StyleAnalyzer

# Configure matplotlib to support Chinese characters
def _setup_chinese_font():
    """Setup Chinese font for matplotlib if available."""
    # Try common Chinese fonts
    chinese_fonts = [
        'SimHei',  # Windows
        'WenQuanYi Micro Hei',  # Linux
        'WenQuanYi Zen Hei',  # Linux
        'Noto Sans CJK SC',  # Linux/Cross-platform
        'Source Han Sans CN',  # Adobe
        'Microsoft YaHei',  # Windows
        'STHeiti',  # macOS
        'PingFang SC',  # macOS
    ]
    
    # Get available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Find first available Chinese font
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
            return font_name
    
    # If no Chinese font found, at least disable the warning
    plt.rcParams['axes.unicode_minus'] = False
    return None

# Initialize font on module import
_setup_chinese_font()

class GenreTrajectoryVisualizer:
    """Plots the artist's career path in latent style space."""
    
    @staticmethod
    def plot_2d_trajectory(
        analyzer: StyleAnalyzer, 
        method: str = "pca",
        show_clusters: bool = True,
        clusters: Optional[Dict[int, List[Track]]] = None,
        cluster_labels: Optional[Dict[int, str]] = None,
        show_album_contours: bool = False,
    ) -> Figure:
        """
        Plots a 2D scatter plot of the career trajectory, connecting chronological points.
        """
        X_2d = analyzer.reduce_dimension(method=method, n_components=2)

        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by Album and compute centroids
        # Use embeddings, not all tracks, to match X_2d indices
        albums = {}
        album_order = []  # Preserve album order by release date
        for i, emb in enumerate(analyzer.career.embeddings):
            album = emb.track_ref.album
            if album not in albums:
                albums[album] = []
                album_order.append(album)
            albums[album].append(X_2d[i])
        
        # Sort albums by release date (use first track's date from each album)
        def get_album_date(album_name):
            for emb in analyzer.career.embeddings:
                if emb.track_ref.album == album_name:
                    return emb.track_ref.release_date
            return None
        
        album_order.sort(key=lambda a: get_album_date(a) or datetime.date.max)
        
        # Compute album centroids
        album_centroids = []
        for album in album_order:
            if album in albums:
                pts = np.array(albums[album])
                center = np.mean(pts, axis=0)
                album_centroids.append(center)
        
        # Draw album contours (convex hulls) - Layer 1 (bottom) - Optional
        # Use a colormap to assign different colors to different albums
        n_albums = len(album_order)
        album_colors = {}
        if show_album_contours and n_albums > 0:
            # Use a colormap that provides distinct colors
            colors = plt.cm.Set3(np.linspace(0, 1, n_albums))
            # Alternative: use tab20 for more distinct colors
            if n_albums > 12:
                colors = plt.cm.tab20(np.linspace(0, 1, n_albums))
            
            for idx, album in enumerate(album_order):
                album_colors[album] = colors[idx]
                if album not in albums or len(albums[album]) < 3:
                    # Need at least 3 points for a convex hull
                    continue
                
                pts = np.array(albums[album])
                
                # Compute convex hull
                hull = ConvexHull(pts)
                # Get hull vertices
                hull_points = pts[hull.vertices]
                
                # Draw filled polygon (album contour) - low zorder so it's behind points
                polygon = Polygon(hull_points, closed=True, 
                                facecolor=colors[idx], 
                                alpha=0.15,  # Very transparent fill
                                edgecolor=colors[idx], 
                                linewidth=1.5, 
                                linestyle='-',
                                zorder=1)  # Bottom layer
                ax.add_patch(polygon)
        
        # Color by Time (Year) - only use tracks that have embeddings
        years = [emb.track_ref.year for emb in analyzer.career.embeddings]
        
        # Scatter: Plot all individual tracks - Layer 2
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=years, cmap='viridis', s=50, alpha=0.8, edgecolors='w', zorder=2)
        plt.colorbar(scatter, label='Year')
        
        # Connect trajectory: Only connect album centroids (not individual tracks)
        # This shows the evolution of style across albums, not individual songs - Layer 3
        if len(album_centroids) > 1:
            centroids_arr = np.array(album_centroids)
            ax.plot(centroids_arr[:, 0], centroids_arr[:, 1], 
                   c='gray', alpha=0.5, linestyle='--', linewidth=2, 
                   marker='o', markersize=8, markerfacecolor='white', 
                   markeredgecolor='gray', markeredgewidth=1.5,
                   label='Album Trajectory', zorder=3)  # Above points
             
        for album, points in albums.items():
            pts = np.array(points)
            center = np.mean(pts, axis=0)
            # Album labels: smaller font, italic, light blue background, very transparent
            ax.text(
                center[0],
                center[1],
                album,
                fontsize=7,
                fontweight='normal',
                style='italic',
                color='#2c3e50',  # Dark gray-blue text
                alpha=0.7,
                bbox=dict(
                    facecolor='#e8f4f8',  # Light blue background
                    alpha=0.3,  # Very transparent
                    edgecolor='#3498db',  # Blue border
                    linewidth=0.8,
                    boxstyle='round,pad=0.3'
                ),
                ha='center',
                va='center',
            )

        # Annotate clusters (centroids) - Cluster/Genre labels
        if clusters and show_clusters:
            # Build index lookup: embedding index by track reference
            # Map track object id to embedding index
            track_to_emb_idx = {id(emb.track_ref): idx for idx, emb in enumerate(analyzer.career.embeddings)}
            for cid, track_list in clusters.items():
                # Only include tracks that have embeddings
                pts = [X_2d[track_to_emb_idx[id(t)]] for t in track_list if id(t) in track_to_emb_idx]
                if not pts:
                    continue
                pts_arr = np.vstack(pts)
                center = np.mean(pts_arr, axis=0)
                label = cluster_labels.get(cid, f"Cluster {cid}") if cluster_labels else f"Cluster {cid}"
                
                # Cluster labels: slightly offset to avoid overlap with album labels
                # Use different style: bold, orange/red tint, positioned slightly offset
                offset_x = 0.02 * (np.max(X_2d[:, 0]) - np.min(X_2d[:, 0]))  # 2% of x-range
                offset_y = 0.02 * (np.max(X_2d[:, 1]) - np.min(X_2d[:, 1]))  # 2% of y-range
                
                # Alternate offset direction for visual variety
                if cid % 2 == 0:
                    label_x = center[0] + offset_x
                    label_y = center[1] + offset_y
                else:
                    label_x = center[0] - offset_x
                    label_y = center[1] - offset_y
                
                # Cluster labels: bold, orange-red tint, very transparent
                ax.text(
                    label_x,
                    label_y,
                    label,
                    fontsize=7.5,
                    fontweight='bold',
                    color='#c0392b',  # Dark red text
                    alpha=0.75,
                    bbox=dict(
                        facecolor='#fdeaa7',  # Light yellow-orange background
                        alpha=0.25,  # Very transparent
                        edgecolor='#e67e22',  # Orange border
                        linewidth=1.0,
                        boxstyle='round,pad=0.4'
                    ),
                    ha='center',
                    va='center',
                )

        ax.set_title(f"Style Trajectory: {analyzer.career.artist_name} ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        sns.despine()
        
        return fig

class StyleDistributionPlotter:
    """Plots the density of styles."""
    
    @staticmethod
    def plot_density(analyzer: StyleAnalyzer, generic_anchors: Optional[Dict[str, np.ndarray]] = None):
        """
        Placeholder for density plots relative to genres.
        """
        # Future work: KDE plot of similarity scores to generic anchors
        pass

class CareerStoryteller:
    """
    Generates narrative-driven visualizations: Streamgraphs, Radars, and Consistency charts.
    """
    
    @staticmethod
    def plot_streamgraph(analyzer: StyleAnalyzer, clusters: Dict[int, List[Track]], cluster_labels: Optional[Dict[int, str]] = None) -> Figure:
        """
        Plots a streamgraph (stacked area) showing the evolution of cluster influence over time (Albums).
        """
        # Only use tracks that have embeddings
        embeddings = analyzer.career.embeddings
            
        # Group by Album and count cluster occurrences
        # 1. Identify all unique albums and sort them by date
        unique_albums = []
        seen = set()
        for emb in embeddings:
            album = emb.track_ref.album
            if album not in seen:
                unique_albums.append(album)
                seen.add(album)
        
        # 2. Build frequency matrix: [n_clusters, n_albums]
        # We need a track -> cluster map
        track_to_cluster = {}
        sorted_cids = sorted(clusters.keys())
        for cid in sorted_cids:
            for t in clusters[cid]:
                track_to_cluster[id(t)] = cid
                
        data = np.zeros((len(sorted_cids), len(unique_albums)))
        
        for j, album in enumerate(unique_albums):
            album_embeddings = [e for e in embeddings if e.track_ref.album == album]
            total_tracks = len(album_embeddings)
            if total_tracks == 0: continue
            
            for emb in album_embeddings:
                cid = track_to_cluster.get(id(emb.track_ref), -1)
                if cid != -1:
                    # Map cid to row index
                    row_idx = sorted_cids.index(cid)
                    data[row_idx, j] += 1
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Stackplot
        # Colors
        cmap = plt.get_cmap('Pastel1')
        colors = [cmap(i % cmap.N) for i in range(len(sorted_cids))]
        
        labels = [cluster_labels.get(cid, f"Cluster {cid}") if cluster_labels else f"Cluster {cid}" for cid in sorted_cids]
        
        ax.stackplot(unique_albums, data, labels=labels, colors=colors, alpha=0.8)
        
        ax.set_title("The River of Style: Sub-Genre Evolution")
        ax.set_ylabel("Track Count / Intensity")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Rotate x labels if many albums
        if len(unique_albums) > 5:
            plt.xticks(rotation=45, ha='right')
            
        sns.despine()
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_cluster_composition(analyzer: StyleAnalyzer, clusters: Dict[int, List[Track]], cluster_labels: Optional[Dict[int, str]] = None) -> Figure:
        """
        Plots a Stacked Bar Chart showing the composition of each cluster by Album.
        X-Axis: Clusters
        Y-Axis: Count of Tracks
        Segments: Albums
        """
        # 1. Identify all unique albums (sorted by date)
        # We want consistent colors for albums across clusters
        embeddings = analyzer.career.embeddings
        unique_albums = []
        seen = set()
        for emb in embeddings:
            album = emb.track_ref.album
            if album not in seen:
                unique_albums.append(album)
                seen.add(album)
                
        # 2. Build Count Matrix: [n_clusters, n_albums]
        sorted_cids = sorted(clusters.keys())
        data = np.zeros((len(sorted_cids), len(unique_albums)))
        
        for i, cid in enumerate(sorted_cids):
            cluster_tracks = clusters[cid]
            for t in cluster_tracks:
                alb_idx = unique_albums.index(t.album)
                data[i, alb_idx] += 1
                    
        # 3. Plot Stacked Bar
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Colors: We need a distinct color for each album
        # Use a qualitative colormap
        cmap = plt.get_cmap('tab20') # Good for many categories
        colors = [cmap(i % cmap.N) for i in range(len(unique_albums))]
        
        bottom = np.zeros(len(sorted_cids))
        
        for j, album in enumerate(unique_albums):
            counts = data[:, j]
            # Plot only if this album has counts > 0 total (optimization)
            if np.sum(counts) > 0:
                ax.bar(
                    range(len(sorted_cids)), 
                    counts, 
                    bottom=bottom, 
                    label=album,
                    color=colors[j],
                    alpha=0.9,
                    edgecolor='white',
                    width=0.6
                )
                bottom += counts
                
        # Labels
        x_labels = [cluster_labels.get(cid, f"C{cid}") if cluster_labels else f"C{cid}" for cid in sorted_cids]
        # Wrap labels if too long
        x_labels_wrapped = [l.replace(': ', ':\n') if len(l) > 15 else l for l in x_labels]
        
        ax.set_xticks(range(len(sorted_cids)))
        ax.set_xticklabels(x_labels_wrapped)
        ax.set_title("Cluster Composition by Album")
        ax.set_ylabel("Number of Tracks")
        ax.set_xlabel("Sub-Genre (Cluster)")
        
        # Legend outside
        ax.legend(title="Album", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        sns.despine()
        plt.tight_layout()
        
        return fig

    @staticmethod
    def plot_radar(
        analyzer: StyleAnalyzer, 
        album_names: List[str], 
        tags: List[str] = ["Happy", "Sad", "Energy", "Acoustic", "Electronic"]
    ) -> Figure:
        """
        Compares specific albums on semantic dimensions using a radar chart.
        """
        # 1. Get embeddings for tags
        # Ensure tags are registered/encoded
        if not set(tags).issubset(set(analyzer.mapper.tags)):
            analyzer.mapper.register_tags(tags)
        
        # Get tag embeddings [K, 512]
        import torch
        import torch.nn.functional as F
        
        device = analyzer.mapper.device
        with torch.no_grad():
             tag_vecs = analyzer.mapper.encoder.encode_text(tags) # [K, 512]
             tag_vecs = tag_vecs.to(device)  # Ensure tag_vecs is on the same device
             tag_vecs = F.normalize(tag_vecs, p=2, dim=1)
        
        # 2. Setup Radar
        # Number of variables
        N = len(tags)
        # Angles
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1] # Close the loop
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Draw one line per album
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        
        for i, album in enumerate(album_names):
            # Get track embeddings for album - only use tracks with embeddings
            album_embeddings = [e for e in analyzer.career.embeddings if e.track_ref.album == album]
            if not album_embeddings: continue
            
            vecs = np.stack([e.vector for e in album_embeddings])
            centroid = np.mean(vecs, axis=0)
            
            # Score against tags
            c_tensor = torch.from_numpy(centroid).unsqueeze(0).to(device) # [1, 512]
            c_tensor = F.normalize(c_tensor, p=2, dim=1)
            
            # Sim: [1, K]
            sims = torch.mm(c_tensor, tag_vecs.t()).squeeze(0).cpu().numpy()
            
            values = sims.tolist()
            values += values[:1] # Close loop
            
            color = colors[i % len(colors)]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=album, color=color)
            ax.fill(angles, values, color=color, alpha=0.25)
            
        plt.xticks(angles[:-1], tags)
        ax.set_title("Semantic Radar: Album Comparison")
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        return fig

    @staticmethod
    def plot_consistency(analyzer: StyleAnalyzer) -> Figure:
        """
        Plots the Intra-Album Variance (Cohesion Meter).
        """
        variances = analyzer.compute_intra_variance(by_album=True)
            
        albums = list(variances.keys())
        values = list(variances.values())
        
        # Sort by time
        # Get album order from embeddings
        seen = []
        ordered_albums = []
        for emb in analyzer.career.embeddings:
            album = emb.track_ref.album
            if album not in seen:
                seen.append(album)
                ordered_albums.append(album)
                
        # Filter variances by ordered list to maintain partial order
        final_albums = [a for a in ordered_albums if a in variances]
        final_values = [variances[a] for a in final_albums]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Line + Dots
        ax.plot(final_albums, final_values, marker='o', linestyle='-', color='#2c3e50', linewidth=2)
        
        # Fill area to show "Base"
        ax.fill_between(final_albums, 0, final_values, color='#95a5a6', alpha=0.3)
        
        # Annotate Peak (Most Experimental)
        if final_values:
            max_idx = int(np.argmax(final_values))
            ax.annotate(
                'Most Experimental\n(Highest Variance)', 
                xy=(float(max_idx), float(final_values[max_idx])), 
                xytext=(float(max_idx), float(final_values[max_idx] + 0.05 * max(final_values))),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='center'
            )
        
        ax.set_title("The Cohesion Meter: Album Consistency vs. Experimentation")
        ax.set_ylabel("Style Variance (Mean Cosine Dist)")
        ax.set_ylim(bottom=0)
        
        if len(final_albums) > 5:
            plt.xticks(rotation=45, ha='right')
            
        sns.despine()
        plt.tight_layout()
        
        return fig

# Placeholder for future visualization methods
def plot_advanced_metrics(analyzer: StyleAnalyzer):
    """
    Reserved for future 3D interactive plots or network graphs.
    """
    pass
