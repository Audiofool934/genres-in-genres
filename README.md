# Genres in Genres: Style Evolution Analysis

This project enables fine-grained analysis of an artist's musical career by leveraging the **MuQ-MuLan** joint audio-text embedding space. It systematically extracts features from your music library, analyzes stylistic trajectories, and identifies "sub-genres" using unsupervised learning.

## 🌟 Key Features

- **Professional Workflow**: Separation of Feature Extraction (batch script) and Interactive Analysis (Gradio App).
- **Library Management**: Automatic scanning and caching of large music collections.
- **Smart Clustering**: Identifies sub-genres using K-Means. Includes **Auto-K selection** using Silhouette Score.
- **Rich Visualizations**:
    - **Style Trajectory**: 2D projection (PCA/t-SNE/UMAP) of career evolution with optional **Album Contours**.
    - **Streamgraph**: Visualization of sub-genre intensity over time.
    - **Semantic Radar**: Comparison of different eras/albums across semantic dimensions (e.g., Happy, Sad, Electronic).
    - **Consistency Meter**: Analysis of stylistic variance within albums.
- **Semantic Tagging**: Tags audio clusters with human-readable genres/moods using Music4All database tags.
- **Mock Data Support**: Built-in simulators for rapid testing and development.

## 📂 Directory Structure

```
genres-in-genres/
├── app.py                  # Main Interactive Application (Gradio)
├── run_demo.sh             # Setup and Startup Script
├── scripts/
│   └── preprocess.py       # Batch Feature Extraction Script
├── src/                    # Source Code Package
│   ├── core.py             # Data Structures (ArtistCareer, Track)
│   ├── library_manager.py  # File Scanning & Cache Management
│   ├── extractor.py        # MuQ-MuLan Wrapper (Placeholder)
│   ├── analysis.py         # Style Evolution & Clustering Logic
│   ├── semantics.py        # Semantic Mapper
│   └── visualization.py    # Plotting Utilities
└── data/
    ├── music/              # Put your mp3/wav files here
    ├── cache/              # Stores processed embeddings (.pkl)
    └── metadata/           # Music4All metadata CSV files
```

## 🚀 Quick Start

### 1. Setup Environment
Ensure you have Python 3.8+ installed. The following script will create a virtual environment and install all dependencies (including `umap-learn`, `gradio`, `scikit-learn`, etc.):
```bash
# This will install dependencies from requirements.txt automatically
./run_demo.sh
```

### 2. Prepare Data (Real Mode)
Organize your music files as: `data/music/{Artist}/{Year}-{Album}/*.mp3`.
Then run the pre-processing script to extract embeddings:
```bash
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
# Ensure you have your extractor setup or use existing caches
python3 scripts/preprocess.py --device cuda
```

### 3. Run Analysis
Launch the interactive dashboard:
```bash
./run_demo.sh
```
1.  Navigate to the **Library** tab.
2.  Select an artist from the dropdown (must be pre-processed).
3.  Adjust parameters like **Dimensionality Reduction Method** (PCA, t-SNE, or **UMAP**) and **Clustering**.
4.  Click **Analyze Library Data** to generate the report and plots.

## 🛠 Requirements
All dependencies are listed in `requirements.txt`. Key libraries include:
- `torch` & `torchaudio`
- `transformers`
- `gradio` (UI)
- `scikit-learn` & `umap-learn` (Analysis)
- `matplotlib` & `seaborn` (Visualization)
