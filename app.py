"""
Gradio App for 'Genres in Genres' Demo.

Allows users to:
1. Visualize Mock Data (Simulation Mode).
2. Analyze Pre-processed Music Libraries (Library Mode).
"""
import gradio as gr
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Get current directory (genres_in_genres/)
current_dir = os.path.dirname(os.path.abspath(__file__))

from src.core import ArtistCareer
from src.mock_data import MockDataGenerator
from src.analysis import StyleAnalyzer
from src.semantics import SemanticMapper
from src.visualization import GenreTrajectoryVisualizer, CareerStoryteller
from src.library_manager import LibraryManager

# Global constants
DATA_DIR = os.path.join(current_dir, "data/music")
CACHE_ROOT = os.path.join(current_dir, "data/cache")
CACHE_TAGS_DIR = os.path.join(CACHE_ROOT, "tags")
METADATA_DIR = os.path.join(current_dir, "data/metadata")

LIBRARY_MANAGER = LibraryManager(DATA_DIR, CACHE_ROOT)

# Initialize semantic mapper from cache (no MuQ-MuLan dependency at runtime)
SEMANTIC_MAPPER = None
try:
    _mapper = SemanticMapper(cache_dir=CACHE_TAGS_DIR, metadata_dir=METADATA_DIR)
    _mapper.initialize_tags(max_tags=2000)
    if _mapper.has_embeddings:
        SEMANTIC_MAPPER = _mapper
    else:
        print("[App] Tag embeddings not found; semantic features disabled.")
except Exception as exc:
    print(f"[App] Semantic mapper unavailable: {exc}")
    SEMANTIC_MAPPER = None

# Helper for session-safe analyzer retrieval (simplified for demo)
# In a production app, this would be a cache indexed by session ID
_ANALYZER_CACHE = {}

def get_analyzer(session_id="default"):
    return _ANALYZER_CACHE.get(session_id)

def set_analyzer(analyzer, session_id="default"):
    _ANALYZER_CACHE[session_id] = analyzer

# --- Analysis Logic (Modified to return State) ---

def get_cached_artists():
    return LIBRARY_MANAGER.list_cached_artists()

def run_mock_analysis(artist_name, num_albums, method="pca"):
    """
    Returns (StateDict, [Plots...])
    """
    career = MockDataGenerator.generate_career(
        artist_name=artist_name, 
        num_albums=int(num_albums), 
        start_year=2010
    )
    return process_analysis(career, method, n_clusters=3)

def run_library_analysis(artist_name, method="pca", n_clusters=3, selected_albums=None, radar_albums=None, radar_tags=None, auto_k=False, show_album_contours=False):
    if not artist_name:
        return None, *([None]*6)
        
    # Load original full career
    full_career = LIBRARY_MANAGER.load_from_cache(artist_name)
    if not full_career:
        return None, *([None]*6)

    # Filter if specific albums selected
    if selected_albums and len(selected_albums) > 0:
        # Create a shallow copy or new instance to avoid modifying cache
        # We need to filter tracks and embeddings
        filtered_tracks = [t for t in full_career.tracks if t.album in selected_albums]
        filtered_embeddings = [e for e in full_career.embeddings if e.track_ref.album in selected_albums]
        
        if not filtered_tracks:
             return None, *([None]*6) # Should not happen if UI is consistent

        # Create new Career object for this analysis
        career = ArtistCareer(artist_name=full_career.artist_name)
        career.tracks = filtered_tracks
        career.embeddings = filtered_embeddings
    else:
        career = full_career
        
    return process_analysis(career, method, n_clusters, radar_albums=radar_albums, radar_tags=radar_tags, auto_k=auto_k, show_album_contours=show_album_contours)

def process_analysis(career, method="pca", n_clusters=3, radar_albums=None, radar_tags=None, auto_k=False, show_album_contours=False):
    """
    Common analysis pipeline.
    Returns:
        state: Dict containing all analysis data needed for interactivity.
        figures: Tuple of plot objects.
    """
    global CURRENT_ANALYZER
    mapper = SEMANTIC_MAPPER if SEMANTIC_MAPPER and SEMANTIC_MAPPER.has_embeddings else None
    analyzer = StyleAnalyzer(career, mapper)
    set_analyzer(analyzer)  # Store for dynamic radar updates

    # 1. Clustering
    if auto_k:
        # Automatically find optimal K
        n_tracks = len(career.embeddings)
        # Set reasonable range: min 2, max based on data size (but cap at 10)
        max_k = min(10, max(2, n_tracks // 10))  # Roughly 1 cluster per 10 tracks, but cap at 10
        k = analyzer.find_optimal_k(k_range=(2, max_k))
    else:
        k = int(n_clusters)
    
    clusters = analyzer.cluster_songs(n_clusters=k)
    
    # Label Sets
    plot_labels = {}      # For visualizations (C0, C1...)
    explorer_labels = {}  # For accordion titles (C0: Tag1, Tag2...)
    
    # 2. Tagging Logic
    for cid, tracks in clusters.items():
        # Plot Label: Clean
        plot_labels[cid] = f"C{cid}"
        
        # Explorer Label: Rich (Top-5)
        vecs = [t_emb.vector for t_emb in career.embeddings if t_emb.track_ref in tracks]
        if vecs and mapper:
            centroid = np.mean(vecs, axis=0)
            tags = mapper.get_nearest_tags(centroid, k=5)
            if tags:
                tag_str = ", ".join([f"{tag_name} ({score:.2f})" for tag_name, score in tags])
                explorer_labels[cid] = f"C{cid}: {tag_str}"
                continue
        explorer_labels[cid] = f"C{cid}"

    # 3. Figures (Use plot_labels)
    fig_traj = GenreTrajectoryVisualizer.plot_2d_trajectory(
        analyzer, method=method, clusters=clusters, cluster_labels=plot_labels, show_album_contours=show_album_contours
    )
    fig_stream = CareerStoryteller.plot_streamgraph(analyzer, clusters, cluster_labels=plot_labels)
    # Consistency doesn't technically use cluster labels but good consistency passing generic valid analyzer
    fig_consistency = CareerStoryteller.plot_consistency(analyzer)
    fig_composition = CareerStoryteller.plot_cluster_composition(analyzer, clusters, cluster_labels=plot_labels)
    
    # Radar - Use custom albums and tags if provided
    albums = sorted(list(set(t.album for t in career.tracks)), key=lambda x: [t for t in career.tracks if t.album == x][0].release_date)
    fig_radar = None
    
    # Default radar albums: first and last
    if radar_albums is None or len(radar_albums) == 0:
        if len(albums) >= 2:
            radar_albums = [albums[0], albums[-1]]
        elif len(albums) == 1:
            radar_albums = [albums[0]]
        else:
            radar_albums = []
    
    # Default radar tags
    if radar_tags is None or len(radar_tags) == 0:
        radar_tags = ["Happy", "Sad", "Energetic", "Calm", "Dark", "Bright", "Romantic", "Aggressive"]
    
    if len(radar_albums) > 0 and len(radar_tags) > 0:
        fig_radar = CareerStoryteller.plot_radar(analyzer, radar_albums, tags=radar_tags)
    else:
        fig_radar = plt.figure()

    # 4. State Object for UI
    # We serialize what we need for the render function
    # Pass explorer_labels here
    state_data = {
        "artist_name": career.artist_name,
        "total_tracks": len(career.tracks),
        "timeline": f"{career.timeline[0].year} - {career.timeline[-1].year}" if career.timeline else "N/A",
        "clusters": clusters, # {cid: [Track]}
        "cluster_labels": explorer_labels, # {cid: str} <- Use Rich Labels for Explorer
        "available_albums": albums,  # For radar album selector
    }
    
    # 5. Generate Report
    report = analyzer.generate_career_report()
    report_md = report.get("markdown", "No data available.")

    return state_data, fig_traj, fig_stream, fig_radar, fig_consistency, fig_composition, report_md

# --- UI Renderer ---

def play_track(evt: gr.SelectData, track_map):
    """
    Event handler for DataFrame selection.
    track_map: List[str] identifying file path for each row index.
    """
    if evt.index is None:
        return None
    row_idx = evt.index[0]
    if row_idx < len(track_map):
        return track_map[row_idx] # Absolute path
    return None

# --- UI Layout ---
with gr.Blocks(title="Genres in Genres") as demo:
    gr.Markdown("# 🎵 Genres in Genres Analysis")
    
    # Analysis State
    analysis_state = gr.State(None)
    
    # Audio Player (Global/Pinned)
    with gr.Row():
        audio_player = gr.Audio(label="Now Playing", type="filepath", autoplay=True)
    
    with gr.Tabs():
        # TAB 1: MOCK
        with gr.TabItem("Simulate"):
            with gr.Row():
                mock_name = gr.Textbox(label="Artist Name", value="The Synthetic Band")
                mock_albums = gr.Slider(2, 10, value=4, step=1, label="Eras")
                mock_method = gr.Dropdown(["pca", "tsne", "umap"], value="pca", label="Method")
                mock_btn = gr.Button("Simulate", variant="primary")

        # TAB 2: LIBRARY
        with gr.TabItem("Library"):
            refresh_btn = gr.Button("Refresh Library List")
            artist_dropdown = gr.Dropdown(label="Select Artist", choices=get_cached_artists())
            
            # Album Multi-Select
            album_selector = gr.Dropdown(
                label="Select Albums (Analyze specific era)", 
                choices=[], 
                multiselect=True,
                info="Select specific albums to analyze. Default: All"
            )
            
            with gr.Row():
                lib_method = gr.Dropdown(["pca", "tsne", "umap"], value="pca", label="Method")
                # Auto K selection
                auto_k_checkbox = gr.Checkbox(
                    value=False, 
                    label="Auto-select K (Optimal Clusters)", 
                    info="Automatically find optimal number of clusters using Silhouette Score"
                )
                # Album contours option
                show_contours_checkbox = gr.Checkbox(
                    value=False,
                    label="Show Album Contours",
                    info="Display convex hull boundaries for each album (may be cluttered with many albums)"
                )
            
            with gr.Row():
                # Slider default 1-10, will be updated dynamically
                # Only visible when auto_k is False
                lib_clusters = gr.Slider(
                    1, 10, value=3, step=1, 
                    label="Clusters K (Sub-genres)", 
                    visible=True,
                    info="Number of sub-genre clusters (ignored if Auto-select K is enabled)"
                )
            
            analyze_btn = gr.Button("Analyze Library Data", variant="primary")
            
            # --- CALLBACKS ---
            def update_artist_list():
                return gr.update(choices=get_cached_artists())
                
            def update_albums_for_artist(artist_name):
                """Loads artist and returns list of albums."""
                if not artist_name:
                    return gr.update(choices=[], value=[])
                
                career = LIBRARY_MANAGER.load_from_cache(artist_name)
                if not career:
                    return gr.update(choices=[], value=[])
                    
                seen = set()
                unique_albums = []
                for t in career.tracks:
                    if t.album not in seen:
                        unique_albums.append(t.album)
                        seen.add(t.album)
                
                return gr.update(choices=unique_albums, value=unique_albums, visible=True)

            def update_cluster_max(selected_albums):
                """Updates max K based on number of selected albums."""
                if not selected_albums:
                    return gr.update(maximum=10, value=3)
                
                n = len(selected_albums)
                # K can be up to N (1 cluster per album is max meaningful granularity usually, though technically can be more, user asked for 1 to N)
                # Ensure min is 1
                new_max = max(1, n)
                # Ensure current value is valid
                return gr.update(maximum=new_max, value=min(3, new_max))
            
            def toggle_cluster_slider(auto_k):
                """Show/hide cluster slider based on auto_k checkbox."""
                return gr.update(visible=not auto_k)
        
        # TAB 3: INSIGHTS
        with gr.TabItem("Insight Report"):
            gr.Markdown("### 🧠 AI Musicologist Report")
            insight_markdown = gr.Markdown("Run analysis to generate insights.")
            
    # --- SHARED OUTPUT AREA ---
    with gr.Row():
        traj_plot = gr.Plot(label="Trajectory")
    
    with gr.Row():
        stream_plot = gr.Plot(label="Streamgraph")
        comp_plot = gr.Plot(label="Cluster Composition")
        
    with gr.Row():
        with gr.Column():
            radar_plot = gr.Plot(label="Radar")
            # Dynamic radar customization (after analysis)
            with gr.Accordion("🎯 Customize Radar Chart (After Analysis)", open=False):
                radar_albums_dynamic = gr.Dropdown(
                    label="Select Albums for Radar",
                    choices=[],
                    multiselect=True,
                    info="Select 2-4 albums to compare"
                )
                radar_tags_dynamic = gr.Dropdown(
                    label="Select Semantic Features",
                    choices=["Happy", "Sad", "Energetic", "Calm", "Dark", "Bright", "Romantic", "Aggressive", "Acoustic", "Electronic", "Rock", "Pop", "Jazz", "Classical", "Hip Hop", "Folk"],
                    multiselect=True,
                    value=["Happy", "Sad", "Energetic", "Calm", "Acoustic", "Electronic"],
                    info="Select 3-8 semantic features"
                )
                update_radar_dynamic_btn = gr.Button("Update Radar Chart", variant="secondary")
                
                def update_radar_dynamic(selected_albums, selected_tags):
                    """Update radar chart dynamically."""
                    analyzer = get_analyzer()
                    if not analyzer:
                        return plt.figure()
                    if not selected_albums or len(selected_albums) == 0:
                        return plt.figure()
                    if not selected_tags or len(selected_tags) == 0:
                        return plt.figure()
                    return CareerStoryteller.plot_radar(analyzer, selected_albums, tags=selected_tags)
                
                def update_radar_albums_from_state(state):
                    """Update radar album choices from analysis state."""
                    if not state:
                        return gr.update(choices=[], value=[])
                    available = state.get('available_albums', [])
                    default = available[:2] if len(available) >= 2 else available
                    return gr.update(choices=available, value=default)
                
                update_radar_dynamic_btn.click(
                    update_radar_dynamic,
                    inputs=[radar_albums_dynamic, radar_tags_dynamic],
                    outputs=radar_plot
                )
                
                # Auto-update album choices when analysis state changes
                analysis_state.change(
                    update_radar_albums_from_state,
                    inputs=[analysis_state],
                    outputs=[radar_albums_dynamic]
                )
        const_plot = gr.Plot(label="Consistency")

    # --- DYNAMIC EXPLORER ---
    @gr.render(inputs=analysis_state)
    def render_explorer(state):
        if not state:
            return gr.Markdown("Run analysis to see results.")
            
        gr.Markdown(f"### 🔍 Cluster Explorer: {state['artist_name']}")
        gr.Markdown(f"**Timeline**: {state['timeline']} | **Tracks**: {state['total_tracks']}")
        
        
        # Iterate clusters
        # Ensure keys are sorted ints
        cids = sorted([k for k in state['clusters'].keys() if isinstance(k, int)])
        
        for cid in cids:
            # Use explorer labels (rich) if available, else standard
            label = state['cluster_labels'].get(cid, f"Cluster {cid}")
            tracks = state['clusters'][cid]
            
            with gr.Accordion(f"{label} ({len(tracks)} tracks)", open=False):
                # Build DF data
                sorted_tracks = sorted(tracks, key=lambda t: (t.year, t.album))
                
                rows = []
                file_paths = []
                for t in sorted_tracks:
                    rows.append([t.title, t.album, t.year])
                    
                    # Fix path: Construct relative path from metadata
                    parts = t.file_path.split(os.sep)
                    if len(parts) >= 3:
                        candidate = os.path.join(DATA_DIR, parts[-3], parts[-2], parts[-1])
                        if os.path.exists(candidate):
                            file_paths.append(candidate)
                            continue
                    file_paths.append(t.file_path)

                df = gr.DataFrame(
                    headers=["Title", "Album", "Year"],
                    value=rows,
                    interactive=False,
                    wrap=True
                )
                
                df.select(
                    fn=play_track,
                    inputs=[gr.State(file_paths)],
                    outputs=audio_player
                )

    # --- EVENT WIRING ---
    outputs_list = [analysis_state, traj_plot, stream_plot, radar_plot, const_plot, comp_plot, insight_markdown]

    mock_btn.click(
        run_mock_analysis,
        inputs=[mock_name, mock_albums, mock_method],
        outputs=outputs_list
    )
    
    # Library Listeners
    refresh_btn.click(update_artist_list, outputs=artist_dropdown)
    
    artist_dropdown.change(
        update_albums_for_artist, 
        inputs=[artist_dropdown], 
        outputs=[album_selector]
    )
    
    album_selector.change(
        update_cluster_max,
        inputs=[album_selector],
        outputs=[lib_clusters]
    )
    
    # Toggle cluster slider visibility based on auto_k checkbox
    auto_k_checkbox.change(
        toggle_cluster_slider,
        inputs=[auto_k_checkbox],
        outputs=[lib_clusters]
    )
    
    analyze_btn.click(
        run_library_analysis, 
        inputs=[artist_dropdown, lib_method, lib_clusters, album_selector, gr.State(None), gr.State(None), auto_k_checkbox, show_contours_checkbox], 
        outputs=outputs_list
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Genres in Genres Analysis App")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server name (default: 0.0.0.0)")
    parser.add_argument("--server_port", type=int, default=7860, help="Server port (default: 7860)")
    args = parser.parse_args()
    
    # Allow serving from Data Dir for audio
    launch_kwargs = {
        "server_name": args.server_name,
        "server_port": args.server_port,
        "share": args.share,
        "allowed_paths": [DATA_DIR]
    }
    
    demo.launch(**launch_kwargs)
