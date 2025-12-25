# Genres in Genres

Style evolution analysis for music collections using MuQ-MuLan embeddings.

## Features

- Automatic artist library scanning and caching
- Sub-genre identification using K-Means clustering (with Auto-K)
- 2D trajectory visualization (PCA/t-SNE/UMAP)
- Semantic radar charts for album comparison
- Streamgraph for temporal style distribution

## Installation

```bash
./run_demo.sh
```

This creates a virtual environment and installs dependencies.

## Usage

### 1. Preprocess Audio (Optional)

If you have your own music files:

```bash
# Organize files as: data/music/{Artist}/{Year}-{Album}/*.mp3
source venv/bin/activate
pip install muq
python scripts/preprocess.py --device cuda  # or mps/cpu
```

### 2. Run Dashboard

```bash
./run_demo.sh
```

Open the URL in browser, go to **Library** tab, select an artist, click **Analyze**.

## Directory Structure

```
genres-in-genres/
├── app.py                  # Gradio application
├── run_demo.sh             # Setup script
├── scripts/
│   ├── preprocess.py       # Audio feature extraction
│   ├── prepare_artist.py   # Library preparation
│   ├── cache_tags.py       # Tag caching
│   └── verify_semantics.py # Model verification
├── src/
│   ├── core.py             # Data structures
│   ├── library_manager.py  # Cache management
│   ├── muq.py              # MuQ-MuLan wrapper
│   ├── analysis.py         # Clustering logic
│   ├── semantics.py        # Semantic mapper
│   ├── metrics.py          # Style metrics
│   ├── mock_data.py        # Test data
│   └── visualization.py    # Plotting
└── data/
    ├── music/              # Input audio files
    ├── cache/              # Cached embeddings
    └── metadata/           # Music4All tags
```

## Requirements

- Python 3.8+
- torch, torchaudio
- muq (MuQ-MuLan)
- gradio
- scikit-learn, umap-learn
- matplotlib, seaborn
