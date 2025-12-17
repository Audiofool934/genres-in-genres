# Genres in Genres: Style Evolution Analysis

This project enables fine-grained analysis of an artist's musical career by leveraging the **MuQ-MuLan** joint audio-text embedding space. It systematically extracts features from your music library, analyzes stylistic trajectories, and identifies "sub-genres" using unsupervised learning.

## Key Features

- **Professional Workflow**: Separation of Feature Extraction (batch script) and Interactive Analysis (Gradio App).
- **Library Management**: Automatic scanning and caching of large music collections.
- **Semantic Tagging**: Tags audio clusters with human-readable genres/moods using MuLan's shared embedding space. Tags are dynamically loaded from Music4All database instead of hardcoded lists.
- **Mock Data Support**: Built-in simulators for rapid testing and development.

## Directory Structure
The project follows a clean, modular architecture:

```
genres_in_genres/
├── app.py                  # Main Interactive Application (Gradio)
├── run_demo.sh             # Setup and Startup Script
├── scripts/
│   └── preprocess.py       # Batch Feature Extraction Script
├── src/                    # Source Code Package
│   ├── core.py             # Data Structures (ArtistCareer, Track)
│   ├── library_manager.py  # File Scanning & Cache Management
│   ├── extractor.py        # MuQ-MuLan Wrapper
│   ├── analysis.py         # Style Evolution & Clustering Logic
│   ├── semantics.py        # Semantic Mapper (loads tags from Music4All)
│   └── visualization.py    # Plotting Utilities
└── data/
    ├── music/              # Put your mp3/wav files here
    ├── cache/              # Stores processed embeddings (.pkl)
    └── metadata/           # Music4All metadata CSV files
        ├── id_tags.csv     # Tags for semantic mapping
        ├── id_genres.csv   # Genre labels
        ├── id_information.csv
        ├── id_metadata.csv
        └── id_lang.csv
```

## Quick Start
### 1. Setup Environment
```bash
./run_demo.sh
```

### 2. Prepare Data (real mode)
Organize your music files as: `data/music/{Artist}/{Year}-{Album}/*.mp3`.
Then run the pre-processing script:
```bash
# Requires GPU and 'muq' installed
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 docs/education/data_science/genres_in_genres/scripts/preprocess.py --device cuda
```

### 3. Run Analysis
Launch the app to visualize the results:
```bash
./run_demo.sh
```
Select "Library Analysis" tab and choose your artist.
